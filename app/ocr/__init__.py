"""
OCR Wrappers - фабрика с ленивой загрузкой
"""

from typing import List, Optional
from app.config import settings

# Храним загруженные движки в кеше
_loaded_engines = {}


def get_ocr_engine(engine_name: Optional[str] = None, languages: Optional[List[str]] = None):
    """
    Фабрика для создания OCR движка с ленивой загрузкой
    """
    engine_name = engine_name or settings.OCR_ENGINE
    languages = languages or settings.OCR_LANGUAGES

    print(f"🔧 Создание OCR движка: {engine_name}")

    # Проверяем кеш
    cache_key = f"{engine_name}_{'_'.join(languages) if languages else ''}"
    if cache_key in _loaded_engines:
        print(f"✅ Используем закешированный движок: {engine_name}")
        return _loaded_engines[cache_key]

    # Ленивая загрузка
    if engine_name == 'tesseract':
        from app.ocr.tesseract_wrapper import TesseractOCR
        engine = TesseractOCR()
    elif engine_name == 'easyocr':
        try:
            from app.ocr.easyocr_wrapper import EasyOCRWrapper
            engine = EasyOCRWrapper(languages)
        except ImportError as e:
            print(f"❌ EasyOCR не установлен. Установите: pip install easyocr")
            print(f"   Ошибка: {e}")
            print("⚠️ Переключаюсь на Tesseract")
            from app.ocr.tesseract_wrapper import TesseractOCR
            engine = TesseractOCR()
    elif engine_name == 'paddleocr':
        try:
            from app.ocr.paddleocr_wrapper import PaddleOCRWrapper
            engine = PaddleOCRWrapper(languages)
        except ImportError as e:
            print(f"❌ PaddleOCR не установлен. Установите: pip install paddlepaddle paddleocr")
            print(f"   Ошибка: {e}")
            print("⚠️ Переключаюсь на Tesseract")
            from app.ocr.tesseract_wrapper import TesseractOCR
            engine = TesseractOCR()
    elif engine_name == 'kosmos':
        try:
            from app.ocr.kosmos_wrapper import KosmosOCR
            engine = KosmosOCR()
        except ImportError as e:
            print(f"❌ KOSMOS не установлен. Установите: pip install torch transformers accelerate")
            print(f"   Ошибка: {e}")
            print("⚠️ Переключаюсь на Tesseract")
            from app.ocr.tesseract_wrapper import TesseractOCR
            engine = TesseractOCR()
    elif engine_name == 'ocrspace':
        try:
            from app.ocr.ocrspace_wrapper import OCRSpaceWrapper
            engine = OCRSpaceWrapper()
        except ImportError as e:
            print(f"❌ OCRSpace не доступен: {e}")
            print("⚠️ Переключаюсь на Tesseract")
            from app.ocr.tesseract_wrapper import TesseractOCR
            engine = TesseractOCR()
    else:
        print(f"⚠️ Неизвестный движок: {engine_name}, использую Tesseract")
        from app.ocr.tesseract_wrapper import TesseractOCR
        engine = TesseractOCR()

    # Сохраняем в кеш
    _loaded_engines[cache_key] = engine
    return engine


def clear_engine_cache():
    """Очистка кеша движков (полезно при переключении)"""
    global _loaded_engines
    _loaded_engines = {}
    print("🧹 Кеш движков очищен")


__all__ = ['get_ocr_engine', 'clear_engine_cache']