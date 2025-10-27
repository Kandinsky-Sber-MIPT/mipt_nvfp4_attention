import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial


import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

# Добавляем путь к wan модулю
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


from wan.distributed.fsdp import shard_model
from wan.modules.model import WanModel
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae import WanVAE
from wan.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.modules.model import parallelize_seq_T2V
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
from wan.utils.utils import cache_video
from torch.distributed.device_mesh import init_device_mesh

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.compress import compress, is_real_quantized
from modelopt.torch.quantization.config import CompressConfig


def nvfp4_mpt(model):
    # Select quantization config
    _default_disabled_quantizer_cfg = {
    "nn.BatchNorm1d": {"*": {"enable": False}},
    "nn.BatchNorm2d": {"*": {"enable": False}},
    "nn.BatchNorm3d": {"*": {"enable": False}},
    "nn.LeakyReLU": {"*": {"enable": False}},
    "*lm_head*": {"enable": False},
    "*proj_out.*": {"enable": False},
    "*block_sparse_moe.gate*": {"enable": False},
    "*router*": {"enable": False},
    "*mlp.gate.*": {"enable": False},
    "*mlp.shared_expert_gate.*": {"enable": False},
    "*output_layer*": {"enable": False},
    "output.*": {"enable": False},
    "default": {"enable": False},
    }

    config = {
        "quant_cfg": {
            **_default_disabled_quantizer_cfg,

            # Включаем только для всех nn.Linear (weights-only, NVFP4)
            "nn.Linear": {
                "*weight_quantizer": {
                    "num_bits": (2, 1),
                    "block_sizes": {
                        -1: 32,
                        "type": "dynamic",
                        "scale_bits": (4, 3),
                    },
                    "enable": True,
                    "pass_through_bwd": False
                },
                "*input_quantizer": {"enable": False},
            }

        },
        "algorithm": "max",
        }
   
    # Quantize the model and perform calibration (PTQ)
    model = mtq.quantize(model, config)

    mtq.print_quant_summary(model)

    ccfg = CompressConfig()
    # сжать всё, что поддерживается:
    ccfg.compress = {"default": True}
    ccfg.quant_gemm = False

    compress(model, ccfg)      


    print("Real-quantized?", is_real_quantized(model))
    for n, m in model.named_modules():
        if "Linear" in type(m).__name__:
            print(n, type(m).__name__)

    return model

class WanT2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        sparse_algo=None
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            sparse_algo (`str`, *optional*, defaults to None):
                Sparse attention algorithm.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.sparse_algo = sparse_algo

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)


        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        
        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir, sparse_algo=sparse_algo)

        self.model = nvfp4_mpt(self.model)

        self.model.eval().requires_grad_(False)

        '''
        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1
        '''
        try:
            world_size = int(os.environ["WORLD_SIZE"])
        except:
            world_size = 1
        if world_size > 1:
            device_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("tensor_parallel",))
            self.model = parallelize_seq_T2V(self.model, device_mesh["tensor_parallel"])
        self.sp_size = 1
        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            sparse_algo (`str`, *optional*, defaults to "nablaT"):
                Sparse attention algorithm.

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        min_crop = 16 if self.sparse_algo else 1
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1]//min_crop * min_crop,
                        size[0] // self.vae_stride[2]//min_crop * min_crop)

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            def set_sparse_attention(t):
                if not self.sparse_algo:
                    return False
                if not self.sparse_algo.startswith("sta"):
                    return True
                if t < 12:
                    return False
                return True

            for i, t in enumerate(tqdm(timesteps)):
                arg_c = {'context': context, 'seq_len': seq_len,
                         'sparse_attention': set_sparse_attention(i)}
                arg_null = {'context': context_null, 'seq_len': seq_len,
                            'sparse_attention': set_sparse_attention(i)}
                
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                self.model.to(self.device)
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None


def main():
    """
    Основная функция для инициализации и использования WanT2V модели
    """

    # Параметры задачи
    task = "t2v-14B"  # Выбираем задачу
    checkpoint_dir = "/home/shared/Wan2.1-NABLA/Wan2.1-T2V-14B"  # Путь к чекпоинтам
    device_id = 0  # ID GPU устройства
    
    # Загружаем конфигурацию
    cfg = WAN_CONFIGS[task]
    
    # Создаем пайплайн
    wan_t2v = WanT2V(
        config=cfg,
        checkpoint_dir=checkpoint_dir,
        device_id=device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        sparse_algo="nabla-0.9_sta-11-3-3",  
    )
    
    
    # Параметры генерации
    input_prompt = "A breathtaking scene unfolds as several massive wooly mammoths slowly traverse a serene, snowy meadow, their thick, shaggy fur gently swaying in the crisp winter breeze. Snow-laden trees line the landscape, standing tall against the backdrop of towering, snow-capped mountains in the distance. The mid-afternoon sun casts a warm golden light across the scene, illuminating wispy clouds that drift lazily across the sky. The camera is positioned low to the ground, capturing the grandeur of the mammoths up close, with a shallow depth of field that accentuates their massive forms. As they move forward, the mammoths occasionally stomp their feet, sending small puffs of snow into the air, adding a dynamic and lifelike energy to the scene."  
    size = (832, 480)  
    frame_num = 81 
    shift = 5.0
    sample_solver = 'unipc'
    sampling_steps = 50
    guide_scale = 5.0
    n_prompt = ""
    seed = 42
    offload_model = False  
    
    # Генерируем видео
    print(f"Генерируем видео с промптом: {input_prompt}")
    print(f"Разрешение: {size}, Фреймов: {frame_num}")
    
    video = wan_t2v.generate(
        input_prompt=input_prompt,
        size=size,
        frame_num=frame_num,
        shift=shift,
        sample_solver=sample_solver,
        sampling_steps=sampling_steps,
        guide_scale=guide_scale,
        n_prompt=n_prompt,
        seed=seed,
        offload_model=offload_model
    )
    
    if video is not None:
        
        # Сохраняем видео
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"wan_video_{timestamp}.mp4"

        
        try:
            saved_path = cache_video(
                tensor=video[None],  # Добавляем batch dimension
                save_file=output_file,
                fps=24,  # FPS для видео
                nrow=1,  # Количество кадров в ряду
                normalize=True,
                value_range=(-1, 1)
            )

            print(f"Полный путь к файлу: {os.path.abspath(saved_path)}")
        except Exception as e:
            print(f"Ошибка сохранения видео: {e}")
    else:
        print("Ошибка генерации видео")




if __name__ == "__main__":
    main()
