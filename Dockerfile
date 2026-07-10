FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# Базовые системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    tesseract-ocr-spa \
    tesseract-ocr-ita \
    tesseract-ocr-por \
    tesseract-ocr-nld \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    wget \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Базовые Python зависимости (всегда нужны)
COPY requirements-base.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-base.txt

# Устанавливаем только базовый Tesseract (всегда)
RUN pip install --no-cache-dir pytesseract

## Дополнительные зависимости для EasyOCR (опционально)
#COPY requirements-easyocr.txt .
#RUN pip install --no-cache-dir -r requirements-easyocr.txt || echo "⚠️ EasyOCR не установлен"
#
## Дополнительные зависимости для PaddleOCR (опционально)
#COPY requirements-paddleocr.txt .
#RUN pip install --no-cache-dir -r requirements-paddleocr.txt || echo "⚠️ PaddleOCR не установлен"

# Дополнительные зависимости для KOSMOS (опционально)
COPY requirements-kosmos.txt .
RUN pip install --no-cache-dir -r requirements-kosmos.txt || echo "⚠️ KOSMOS не установлен"

COPY app/ ./app/

RUN mkdir -p /app/data/{input,output,processed}

CMD ["python", "-m", "app.main"]