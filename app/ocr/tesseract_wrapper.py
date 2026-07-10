import pytesseract
import cv2
import numpy as np
from typing import Dict, List, Tuple
from app.config import settings


class TesseractOCR:
    def __init__(self):
        self.languages = settings.TESSERACT_LANGUAGES

    def recognize(self, img: np.ndarray) -> Dict[str, any]:
        """Распознавание текста через Tesseract"""
        try:
            # Попытка с разными конфигурациями
            configs = [
                f'--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ',
                f'--psm 3 -l {self.languages}',
                f'--psm 4 -l {self.languages}',
            ]

            best_result = None
            best_confidence = 0

            for config in configs:
                data = pytesseract.image_to_data(
                    img,
                    config=config,
                    output_type=pytesseract.Output.DICT
                )

                # Извлечение текста и уверенности
                text_parts = []
                confidences = []

                for i, conf in enumerate(data['conf']):
                    if conf > 0 and data['text'][i].strip():
                        text_parts.append(data['text'][i].strip())
                        confidences.append(conf)

                text = ' '.join(text_parts)
                avg_confidence = np.mean(confidences) if confidences else 0

                if avg_confidence > best_confidence:
                    best_confidence = avg_confidence
                    best_result = {
                        'text': text,
                        'confidence': avg_confidence / 100,
                        'engine': 'tesseract',
                        'language': self.languages
                    }

            return best_result or {'text': '', 'confidence': 0, 'engine': 'tesseract'}

        except Exception as e:
            print(f"Tesseract error: {e}")
            return {'text': '', 'confidence': 0, 'engine': 'tesseract'}

    def get_supported_languages(self) -> List[str]:
        """Получение списка поддерживаемых языков"""
        return settings.SUPPORTED_LANGUAGES