import torch
from PIL import Image
from transformers import AutoProcessor, Kosmos2_5ForConditionalGeneration
import io
import re
from app.ocr.base import BaseOCR


class KosmosOCR(BaseOCR):
    def __init__(self):
        super().__init__()
        self.name = "kosmos"
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        """Загружает модель KOSMOS-2.5"""
        print(f"🔄 Загрузка KOSMOS-2.5 на {self.device}... Это может занять время")
        try:
            repo = "microsoft/kosmos-2.5"
            self.processor = AutoProcessor.from_pretrained(repo)
            self.model = Kosmos2_5ForConditionalGeneration.from_pretrained(
                repo,
                device_map=self.device if self.device == "cuda" else None,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
            )
            if self.device == "cuda":
                self.model.to(self.device)
            print(f"✅ KOSMOS-2.5 загружен")
        except Exception as e:
            print(f"❌ Ошибка загрузки KOSMOS-2.5: {e}")
            self.model = None

    def recognize(self, img):
        """Распознавание текста через KOSMOS-2.5"""
        if self.model is None:
            return {"text": "", "confidence": 0, "engine": self.name, "error": "Не загружен"}

        try:
            # Конвертируем numpy в PIL Image
            if hasattr(img, 'shape'):
                image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                image = img

            # Используем промпт для OCR
            prompt = "<ocr>"
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            height, width = inputs.pop("height"), inputs.pop("width")
            raw_width, raw_height = image.size
            scale_height = raw_height / height
            scale_width = raw_width / width

            inputs = {k: v.to(self.device) if v is not None else None for k, v in inputs.items()}
            inputs["flattened_patches"] = inputs["flattened_patches"].to(
                torch.bfloat16 if self.device == "cuda" else torch.float32)

            generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Пост-обработка для извлечения текста
            cleaned_text = self._post_process(generated_text, prompt, scale_height, scale_width)

            return {
                "text": cleaned_text,
                "confidence": 0.9,  # KOSMOS не дает confidence, ставим высокий
                "engine": self.name,
                "word_count": len(cleaned_text.split())
            }

        except Exception as e:
            print(f"❌ KOSMOS ошибка: {e}")
            return {"text": "", "confidence": 0, "engine": self.name, "error": str(e)}

    def _post_process(self, text, prompt, scale_h, scale_w):
        """Извлекает текст из ответа модели и убирает координаты"""
        # Убираем промпт из ответа
        y = text.replace(prompt, "")

        # Находим все блоки с координатами
        pattern = r"<bbox><x_\d+><y_\d+><x_\d+><y_\d+></bbox>"
        bboxs_raw = re.findall(pattern, y)
        lines = re.split(pattern, y)[1:]

        # Собираем только текст, пропуская координаты
        result_parts = []
        for i, line in enumerate(lines):
            if line.strip():
                result_parts.append(line.strip())

        return " ".join(result_parts)