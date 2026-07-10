FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# Устанавливаем правильные пакеты для Debian slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    tesseract-ocr-spa \
    tesseract-ocr-ita \
    tesseract-ocr-por \
    tesseract-ocr-nld \
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

# Сначала копируем только requirements.txt
COPY requirements.txt .

# Устанавливаем зависимости (этот слой кешируется, если requirements.txt не менялся)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Проверяем установку
RUN python -c "import easyocr, cv2, numpy, PIL, langdetect; print('✅ All dependencies installed')"

# Копируем код (этот слой пересобирается при изменении кода)
COPY app/ ./app/

# Создаем папки
RUN mkdir -p /app/data/{input,output,processed}

CMD ["python", "-m", "app.main"]