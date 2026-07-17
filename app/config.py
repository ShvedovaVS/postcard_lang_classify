import os
from typing import List, Optional
from pathlib import Path


class Settings:
    """Конфигурация приложения"""

    # Пути к папкам
    INPUT_DIR: str = os.getenv('INPUT_DIR', '/app/data/input')
    OUTPUT_DIR: str = os.getenv('OUTPUT_DIR', '/app/data/output')
    PROCESSED_DIR: str = os.getenv('PROCESSED_DIR', '/app/data/processed')

    # Доступные: tesseract, easyocr, paddleocr, kosmos, trocr
    OCR_ENGINE: str = os.getenv('OCR_ENGINE', 'tesseract')

    # Языки для OCR
    OCR_LANGUAGES: List[str] = ['en', 'fr', 'de', 'es', 'it', 'pt', 'nl']

    # Настройки для Tesseract
    TESSERACT_LANGUAGES: str = "eng+fra+deu+spa+ita+por+nld"

    # Поддерживаемые языки для классификатора
    SUPPORTED_LANGUAGES: List[str] = [
        'en', 'fr', 'de', 'es', 'it', 'pt', 'nl'
    ]

    # НАСТРОЙКИ ПРЕДОБРАБОТКИ
    # Включить/выключить улучшение качества изображения
    ENHANCE_IMAGE: bool = os.getenv('ENHANCE_IMAGE', 'true').lower() == 'true'

    # Включить/выключить автоматическое выравнивание
    AUTO_DESKEW: bool = os.getenv('AUTO_DESKEW', 'true').lower() == 'true'

    # Включить/выключить проверку поворотов
    TRY_ROTATIONS: bool = os.getenv('TRY_ROTATIONS', 'true').lower() == 'true'

    # Углы поворота для проверки
    ROTATION_ANGLES: List[int] = [0, 90, 180, 270]

    # Пороги
    MIN_CONFIDENCE: float = float(os.getenv('MIN_CONFIDENCE', '0.1'))
    MIN_TEXT_LENGTH: int = 2

    # Обработка изображений
    MAX_IMAGE_SIZE: int = 4096

    # Настройки производительности
    MAX_WORKERS: int = 2
    CACHE_SIZE: int = 100


settings = Settings()