import os
from typing import List
from pathlib import Path


class Settings:
    """Конфигурация приложения"""

    # Пути к папкам (создаем их принудительно)
    INPUT_DIR: str = os.getenv('INPUT_DIR', '/app/data/input')
    OUTPUT_DIR: str = os.getenv('OUTPUT_DIR', '/app/data/output')
    PROCESSED_DIR: str = os.getenv('PROCESSED_DIR', '/app/data/processed')

    # Список поддерживаемых языков
    SUPPORTED_LANGUAGES: List[str] = [
        'en', 'ru', 'fr', 'de', 'es', 'it', 'pt', 'nl', 'pl', 'uk'
    ]

    # Настройки OCR
    TESSERACT_LANGUAGES: str = "eng+rus+fra+deu+spa+ita+por+nld+pol+ukr"
    EASYOCR_LANGUAGES: List[str] = ['en', 'ru', 'fr', 'de', 'es', 'it', 'pt']

    # Пороги
    MIN_CONFIDENCE: float = float(os.getenv('MIN_CONFIDENCE', '0.5'))
    MIN_TEXT_LENGTH: int = 3

    # Обработка изображений
    MAX_IMAGE_SIZE: int = 4096
    ROTATION_ANGLES: List[int] = [0, 90, 180, 270]
    AUTO_DESKEW: bool = True

    # Настройки производительности
    MAX_WORKERS: int = 4
    CACHE_SIZE: int = 100


# Создаем папки при инициализации
for dir_path in [Settings.INPUT_DIR, Settings.OUTPUT_DIR, Settings.PROCESSED_DIR]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

settings = Settings()