#!/bin/bash
set -e

echo "=========================================="
echo "  🚀 ЗАПУСК OCR ПРИЛОЖЕНИЯ"
echo "=========================================="
echo "OCR_ENGINE: ${OCR_ENGINE:-tesseract}"
echo "=========================================="

case "${OCR_ENGINE:-tesseract}" in
  "easyocr")
    echo "📦 Устанавливаю EasyOCR..."
    pip install --quiet easyocr==1.7.1 || echo "⚠️ EasyOCR не установлен"
    ;;

  "paddleocr")
    echo "📦 Устанавливаю PaddleOCR..."
    pip install --quiet paddlepaddle==2.6.2 paddleocr==2.7.0.3 || echo "⚠️ PaddleOCR не установлен"
    ;;

  "kosmos")
    echo "📦 Устанавливаю KOSMOS (это может занять несколько минут)..."
    pip install --quiet torch>=2.0.0 transformers>=4.30.0 accelerate>=0.20.0 || {
      echo "❌ Ошибка установки KOSMOS"
      export OCR_ENGINE=tesseract
    }
    # Проверяем установку
    python -c "import transformers; from transformers import Kosmos2_5ForConditionalGeneration, AutoProcessor; print('✅ KOSMOS готов')" 2>/dev/null || {
      echo "❌ KOSMOS не работает, переключаюсь на Tesseract"
      export OCR_ENGINE=tesseract
    }
    ;;

  "trocr")
    echo "📦 Устанавливаю TrOCR (это может занять несколько минут)..."
    # 👇 ПРИНУДИТЕЛЬНАЯ УСТАНОВКА С ПРОВЕРКОЙ
    echo "  Устанавливаю sentencepiece..."
    pip install --quiet sentencepiece>=0.1.99 tiktoken>=0.5.0 torch>=2.0.0 transformers>=4.30.0 pillow>=9.0.0 || {
      echo "❌ Ошибка установки TrOCR"
      export OCR_ENGINE=tesseract
    }
    # Проверяем установку
    python -c "import transformers; from transformers import TrOCRProcessor, VisionEncoderDecoderModel; print('✅ TrOCR готов')" 2>/dev/null || {
      echo "❌ TrOCR не работает, переключаюсь на Tesseract"
      export OCR_ENGINE=tesseract
    }
    ;;

  "tesseract"|"")
    echo "✅ Использую Tesseract (уже установлен)"
    ;;

  *)
    echo "⚠️ Неизвестный движок: $OCR_ENGINE"
    export OCR_ENGINE=tesseract
    ;;
esac

echo "=========================================="
echo "✅ Запуск с движком: ${OCR_ENGINE}"
echo "=========================================="

exec python -m app.main "$@"