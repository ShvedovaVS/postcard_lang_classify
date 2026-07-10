#!/usr/bin/env python3
"""
Проверка доступных OCR движков
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def check_engine(engine_name: str):
    """Проверяет доступность движка"""
    try:
        if engine_name == 'tesseract':
            import pytesseract
            return True, f"Tesseract {pytesseract.__version__}"
        elif engine_name == 'easyocr':
            import easyocr
            return True, "EasyOCR доступен"
        elif engine_name == 'paddleocr':
            import paddleocr
            return True, "PaddleOCR доступен"
        elif engine_name == 'kosmos':
            import transformers
            return True, "KOSMOS доступен"
        else:
            return False, "Неизвестный движок"
    except ImportError as e:
        return False, f"Не установлен: {e}"


def main():
    print("=" * 60)
    print("ПРОВЕРКА ДОСТУПНЫХ OCR ДВИЖКОВ")
    print("=" * 60)

    engines = ['tesseract', 'easyocr', 'paddleocr', 'kosmos']

    for engine in engines:
        available, info = check_engine(engine)
        status = "✅" if available else "❌"
        print(f"{status} {engine:>10}: {info}")

    print("=" * 60)
    print("\nДля переключения движка установите переменную OCR_ENGINE")
    print("Пример: OCR_ENGINE=easyocr docker-compose up")


if __name__ == "__main__":
    main()