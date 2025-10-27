# nvfp4

Генератор тексто-в-видео на базе Wan 2.1 с квантованием **NVFP4**. Создаёт короткие видео по текстовому описанию.

## Требования

- Python 3.8+
- PyTorch 2.0+ с CUDA
- GPU (≥24GB VRAM рекомендуется)
- Установленный модуль `wan`
- modelopt для квантования

## Структура файла

- `nvfp4_mpt()` — применение NVFP4-квантования
- `WanT2V` — пайплайн генерации (из оригинального `text2video.py`)
- `main()` — запуск через `main()`

## Использование

### Запуск из командной строки

#### Настройка до запуска

```python
    # Параметры задачи
    task = "t2v-14B"  # Выбираем задачу
    checkpoint_dir = "/home/shared/Wan2.1-NABLA/Wan2.1-T2V-14B"  # Путь к чекпоинтам
    device_id = 0  # ID GPU устройства


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
```

```bash
# Из директории с моделью
python nvfp4.py
```


# Изменения в FA3

Комментарий разработчика по коду: `Баг состоял в размерностях и паддинге рага.`

Ошибка при запуске baseline:
```
RuntimeError: This flash attention build does not support varlen.
```
Текущая версия FA3 у нас: 
```bash
FA3 (flash_attn_interface): YES
FA2 (flash_attn):          YES
flash_attn version: 2.8.3
CUDA: 12.8 PyTorch: 2.8.0+cu128
GPU: NVIDIA H100 PCIe CC: (9, 0)
```
