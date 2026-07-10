"""
OCR Wrappers - фабрика для создания OCR движков
"""

from typing import List, Optional
from app.config import settings

# Импортируем доступные OCR движки
from app.ocr.tesseract_wrapper import TesseractOCR
from app.ocr.easyocr_wrapper import EasyOCRWrapper

# PaddleOCR опционально
try:
    from app.ocr.paddleocr_wrapper import PaddleOCRWrapper
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    PaddleOCRWrapper = None


def get_ocr_engine(engine_name: Optional[str] = None, languages: Optional[List[str]] = None):
    """
    Фабрика для создания OCR движка

    Args:
        engine_name: 'tesseract', 'easyocr', 'paddleocr'
        languages: список языков

    Returns:
        OCR движок
    """
    engine_name = engine_name or settings.OCR_ENGINE
    languages = languages or settings.OCR_LANGUAGES

    print(f"🔧 Создание OCR движка: {engine_name}")

    if engine_name == 'tesseract':
        return TesseractOCR()
    elif engine_name == 'easyocr':
        return EasyOCRWrapper(languages)
    elif engine_name == 'paddleocr':
        if PADDLE_AVAILABLE and PaddleOCRWrapper is not None:
            return PaddleOCRWrapper(languages)
        else:
            print("⚠️ PaddleOCR not available, falling back to Tesseract")
            return TesseractOCR()
    else:
        print(f"⚠️ Unknown engine: {engine_name}, using Tesseract")
        return TesseractOCR()


__all__ = ['get_ocr_engine', 'TesseractOCR', 'EasyOCRWrapper']