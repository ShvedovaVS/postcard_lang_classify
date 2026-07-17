"""
OCR Wrappers - фабрика с ленивой загрузкой
"""

from typing import List, Optional
from app.config import settings
import os

_loaded_engines = {}


def get_ocr_engine(engine_name: Optional[str] = None, languages: Optional[List[str]] = None):
    """
    Фабрика для создания OCR движка с ленивой загрузкой
    """
    engine_name = engine_name or settings.OCR_ENGINE
    languages = languages or settings.OCR_LANGUAGES

    cache_key = f"{engine_name}_{'_'.join(languages) if languages else ''}"

    if cache_key in _loaded_engines:
        print(f"✅ Используем закешированный движок: {engine_name}")
        return _loaded_engines[cache_key]

    print(f"🔧 Загрузка OCR движка: {engine_name}")

    if engine_name == 'tesseract':
        from app.ocr.tesseract_wrapper import TesseractOCR
        engine = TesseractOCR()

    elif engine_name == 'easyocr':
        try:
            from app.ocr.easyocr_wrapper import EasyOCRWrapper
            engine = EasyOCRWrapper(languages)
        except ImportError as e:
            print(f"❌ EasyOCR не установлен: {e}")
            print("⚠️ Переключаюсь на Tesseract")
            from app.ocr.tesseract_wrapper import TesseractOCR
            engine = TesseractOCR()

    elif engine_name == 'paddleocr':
        try:
            from app.ocr.paddleocr_wrapper import PaddleOCRWrapper
            engine = PaddleOCRWrapper(languages)
        except ImportError as e:
            print(f"❌ PaddleOCR не установлен: {e}")
            print("⚠️ Переключаюсь на Tesseract")
            from app.ocr.tesseract_wrapper import TesseractOCR
            engine = TesseractOCR()

    elif engine_name == 'kosmos':
        try:
            from app.ocr.kosmos_wrapper import KosmosOCR
            engine = KosmosOCR()
            if engine.model is None:
                print("⚠️ KOSMOS не загрузился, переключаюсь на Tesseract")
                from app.ocr.tesseract_wrapper import TesseractOCR
                engine = TesseractOCR()
        except ImportError as e:
            print(f"❌ KOSMOS не установлен: {e}")
            print("⚠️ Переключаюсь на Tesseract")
            from app.ocr.tesseract_wrapper import TesseractOCR
            engine = TesseractOCR()
        except Exception as e:
            print(f"❌ Ошибка загрузки KOSMOS: {e}")
            print("⚠️ Переключаюсь на Tesseract")
            from app.ocr.tesseract_wrapper import TesseractOCR
            engine = TesseractOCR()

    elif engine_name == 'trocr':
        try:
            from app.ocr.trocr_wrapper import TrOCRWrapper
            model_name = os.getenv('TROCR_MODEL', 'microsoft/trocr-base-printed')
            engine = TrOCRWrapper(model_name=model_name)
            if engine.model is None:
                print("⚠️ TrOCR не загрузился, переключаюсь на Tesseract")
                from app.ocr.tesseract_wrapper import TesseractOCR
                engine = TesseractOCR()
        except ImportError as e:
            print(f"❌ TrOCR не установлен: {e}")
            print("⚠️ Переключаюсь на Tesseract")
            from app.ocr.tesseract_wrapper import TesseractOCR
            engine = TesseractOCR()
        except Exception as e:
            print(f"❌ Ошибка загрузки TrOCR: {e}")
            print("⚠️ Переключаюсь на Tesseract")
            from app.ocr.tesseract_wrapper import TesseractOCR
            engine = TesseractOCR()

    else:
        print(f"⚠️ Неизвестный движок: {engine_name}, использую Tesseract")
        from app.ocr.tesseract_wrapper import TesseractOCR
        engine = TesseractOCR()

    _loaded_engines[cache_key] = engine
    return engine


def get_available_engines():
    """Возвращает список доступных движков"""
    engines = {'tesseract': True}

    try:
        import easyocr
        engines['easyocr'] = True
    except:
        engines['easyocr'] = False

    try:
        import paddleocr
        engines['paddleocr'] = True
    except:
        engines['paddleocr'] = False

    try:
        import transformers
        engines['kosmos'] = True
    except:
        engines['kosmos'] = False

    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        engines['trocr'] = True
    except:
        engines['trocr'] = False

    return engines


__all__ = ['get_ocr_engine', 'get_available_engines']