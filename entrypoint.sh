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
    pip install --quiet easyocr==1.7.1
    python -c "import easyocr; print('✅ EasyOCR OK')"
    ;;
  "paddleocr")
    echo "📦 Устанавливаю PaddleOCR..."
    pip install --quiet paddlepaddle==2.6.2 paddleocr==2.7.0.3
    python -c "import paddleocr; print('✅ PaddleOCR OK')"
    ;;
  "kosmos")
    echo "📦 Устанавливаю KOSMOS (это может занять несколько минут)..."
    pip install --quiet torch>=2.0.0 transformers>=4.30.0 accelerate>=0.20.0
    python -c "import transformers; import torch; import accelerate; print('✅ KOSMOS OK')"
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