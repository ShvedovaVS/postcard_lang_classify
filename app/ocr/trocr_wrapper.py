"""
TrOCR wrapper - Transformer-based OCR
"""

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import cv2
import numpy as np
import re
from app.ocr.base import BaseOCR


class TrOCRWrapper(BaseOCR):
    def __init__(self, model_name="microsoft/trocr-base-printed", device=None):
        super().__init__()
        self.name = "trocr"
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.dtype = None
        self._load_model()

        if self.model is None:
            print("❌ TrOCR не загружен! Используйте другой движок.")

    def _load_model(self):
        """Загружает модель TrOCR"""
        print(f"🔄 Загрузка TrOCR ({self.model_name}) на {self.device}...")
        try:
            print("  📥 Загрузка процессора...")
            self.processor = TrOCRProcessor.from_pretrained(self.model_name)

            print("  📥 Загрузка модели...")
            self.model = VisionEncoderDecoderModel.from_pretrained(
                self.model_name,
                low_cpu_mem_usage=True
            )

            if self.device == "cuda":
                print(f"  📥 Перемещение на GPU...")
                self.model.to(self.device)
                self.model.eval()

            # Определяем тип данных модели
            for param in self.model.parameters():
                self.dtype = param.dtype
                break

            if self.dtype is None:
                self.dtype = torch.float32

            print(f"✅ TrOCR загружен на {self.device} с типом {self.dtype}")

        except ImportError as e:
            print(f"❌ Ошибка импорта: {e}")
            print("   Установите: pip install torch transformers pillow")
            self.model = None
            self.processor = None
        except Exception as e:
            print(f"❌ Ошибка загрузки TrOCR: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.processor = None

    def recognize(self, img: np.ndarray) -> dict:
        """Распознавание текста через TrOCR"""
        if self.model is None:
            return {
                "text": "",
                "confidence": 0,
                "engine": self.name,
                "error": "Модель не загружена"
            }

        if self.processor is None:
            return {
                "text": "",
                "confidence": 0,
                "engine": self.name,
                "error": "Процессор не загружен"
            }

        try:
            # Конвертируем numpy в PIL Image
            if hasattr(img, 'shape'):
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(img_rgb)
                else:
                    image = Image.fromarray(img)
            else:
                image = img

            # Подготовка входных данных
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values

            # Перемещаем на устройство и приводим к правильному типу
            pixel_values = pixel_values.to(self.device)
            if self.dtype is not None:
                pixel_values = pixel_values.to(self.dtype)

            # Генерируем текст
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True
                )

            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Очистка текста
            cleaned_text = self._post_process(generated_text)

            return {
                "text": cleaned_text,
                "confidence": 0.9,
                "engine": self.name,
                "word_count": len(cleaned_text.split()),
                "model": self.model_name
            }

        except Exception as e:
            print(f"❌ TrOCR ошибка при распознавании: {e}")
            import traceback
            traceback.print_exc()
            return {
                "text": "",
                "confidence": 0,
                "engine": self.name,
                "error": str(e)
            }

    def _post_process(self, text: str) -> str:
        """Пост-обработка текста"""
        if not text:
            return ""

        # Убираем лишние пробелы
        text = re.sub(r'\s+', ' ', text).strip()

        return text


# Для тестирования
if __name__ == "__main__":
    print("Тестирование TrOCR...")
    ocr = TrOCRWrapper()
    if ocr.model is not None:
        print("✅ TrOCR готов к работе")
    else:
        print("❌ TrOCR не загружен")