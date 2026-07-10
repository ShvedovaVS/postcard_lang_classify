import os
from typing import List
from pathlib import Path


class Settings:
    """Конфигурация приложения"""

    # Пути к папкам
    INPUT_DIR: str = os.getenv('INPUT_DIR', '/app/data/input')
    OUTPUT_DIR: str = os.getenv('OUTPUT_DIR', '/app/data/output')
    PROCESSED_DIR: str = os.getenv('PROCESSED_DIR', '/app/data/processed')

    # 👇 ВЫБОР OCR ДВИЖКА
    OCR_ENGINE: str = os.getenv('OCR_ENGINE', 'tesseract')

    # Языки для OCR
    OCR_LANGUAGES: List[str] = ['en', 'fr', 'de', 'es', 'it', 'pt', 'nl']

    # Настройки для Tesseract
    TESSERACT_LANGUAGES: str = "eng+fra+deu+spa+ita+por+nld"

    # Поддерживаемые языки для классификатора
    SUPPORTED_LANGUAGES: List[str] = [
        'en', 'fr', 'de', 'es', 'it', 'pt', 'nl'
    ]

    # Пороги
    MIN_CONFIDENCE: float = float(os.getenv('MIN_CONFIDENCE', '0.1'))
    MIN_TEXT_LENGTH: int = 2

    # Обработка изображений
    MAX_IMAGE_SIZE: int = 4096
    ROTATION_ANGLES: List[int] = [0, 90, 180, 270]
    AUTO_DESKEW: bool = True

    # Настройки производительности
    MAX_WORKERS: int = 2
    CACHE_SIZE: int = 100


settings = Settings()