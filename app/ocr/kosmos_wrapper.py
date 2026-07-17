"""
KOSMOS-2.5 OCR wrapper
"""

import torch
from PIL import Image
from transformers import AutoProcessor, Kosmos2_5ForConditionalGeneration
import cv2
import numpy as np
import re
from app.ocr.base import BaseOCR


class KosmosOCR(BaseOCR):
    def __init__(self):
        super().__init__()
        self.name = "kosmos"
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = None  # Будет определено после загрузки модели
        self._load_model()

        if self.model is None:
            print("❌ KOSMOS не загружен! Используйте другой движок.")

    def _load_model(self):
        """Загружает модель KOSMOS-2.5"""
        print(f"🔄 Загрузка KOSMOS-2.5 на {self.device}... Это может занять время")
        try:
            repo = "microsoft/kosmos-2.5"

            print("  📥 Загрузка процессора...")
            self.processor = AutoProcessor.from_pretrained(repo)

            print("  📥 Загрузка модели...")
            # Загружаем без явного указания типа
            self.model = Kosmos2_5ForConditionalGeneration.from_pretrained(
                repo,
                low_cpu_mem_usage=True
            )

            if self.device == "cuda":
                print(f"  📥 Перемещение на GPU...")
                self.model.to(self.device)
                self.model.eval()

            # Определяем тип данных модели из первого параметра
            for param in self.model.parameters():
                self.dtype = param.dtype
                break

            if self.dtype is None:
                self.dtype = torch.float32

            print(f"✅ KOSMOS-2.5 загружен на {self.device} с типом {self.dtype}")

        except ImportError as e:
            print(f"❌ Ошибка импорта: {e}")
            print("   Установите: pip install torch transformers accelerate")
            self.model = None
            self.processor = None
        except Exception as e:
            print(f"❌ Ошибка загрузки KOSMOS-2.5: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.processor = None

    def recognize(self, img: np.ndarray) -> dict:
        """Распознавание текста через KOSMOS-2.5"""
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

            # Используем промпт для OCR
            prompt = "<ocr>"
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")

            # 👇 ПРИВОДИМ ВСЕ ТЕНЗОРЫ К ПРАВИЛЬНЫМ ТИПАМ И УСТРОЙСТВУ
            cleaned_inputs = {}

            for key, tensor in inputs.items():
                # Перемещаем на устройство
                tensor = tensor.to(self.device)

                # Определяем тип для каждого ключа
                if key in ['pixel_values', 'flattened_patches']:
                    # Эти тензоры должны быть того же типа, что и модель
                    tensor = tensor.to(self.dtype)
                elif key in ['input_ids', 'attention_mask', 'width', 'height', 'image_embeds_position_mask']:
                    # Эти тензоры должны быть целочисленными
                    tensor = tensor.long()
                else:
                    # Для остальных - пытаемся определить тип
                    if hasattr(tensor, 'dtype') and tensor.dtype.is_floating_point:
                        tensor = tensor.to(self.dtype)
                    else:
                        tensor = tensor.long()

                cleaned_inputs[key] = tensor

            # Проверка типов для отладки
            print(f"🔍 Типы тензоров после приведения:")
            for key, tensor in cleaned_inputs.items():
                print(f"  {key}: dtype={tensor.dtype}, device={tensor.device}")

            # Генерируем
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **cleaned_inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )

            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Пост-обработка
            cleaned_text = self._post_process(generated_text, prompt)

            return {
                "text": cleaned_text,
                "confidence": 0.9,
                "engine": self.name,
                "word_count": len(cleaned_text.split())
            }

        except Exception as e:
            print(f"❌ KOSMOS ошибка при распознавании: {e}")
            import traceback
            traceback.print_exc()
            return {
                "text": "",
                "confidence": 0,
                "engine": self.name,
                "error": str(e)
            }

    def _post_process(self, text: str, prompt: str) -> str:
        """Пост-обработка текста"""
        # Убираем промпт
        result = text.replace(prompt, "")

        # Убираем все теги <...>
        result = re.sub(r'<[^>]+>', '', result)

        # Убираем лишние пробелы
        result = re.sub(r'\s+', ' ', result).strip()

        return result


# Для тестирования
if __name__ == "__main__":
    print("Тестирование KOSMOS...")
    ocr = KosmosOCR()
    if ocr.model is not None:
        print("✅ KOSMOS готов к работе")
    else:
        print("❌ KOSMOS не загружен")