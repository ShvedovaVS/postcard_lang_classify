import pytesseract
import cv2
import numpy as np
from typing import Dict, List
from app.config import settings


class TesseractOCR:
    def __init__(self):
        self.name = "tesseract"
        self.languages = settings.TESSERACT_LANGUAGES
        print(f"✅ Tesseract initialized with languages: {self.languages}")

    def recognize(self, img: np.ndarray) -> Dict[str, any]:
        """Распознавание текста через Tesseract"""
        try:
            # Конвертируем в RGB если нужно
            if len(img.shape) == 3 and img.shape[2] == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img

            # Пробуем разные конфигурации
            configs = [
                f'--psm 6 -l {self.languages}',
                f'--psm 3 -l {self.languages}',
                f'--psm 4 -l {self.languages}',
            ]

            best_result = None
            best_confidence = 0

            for config in configs:
                try:
                    data = pytesseract.image_to_data(
                        img_rgb,
                        config=config,
                        output_type=pytesseract.Output.DICT
                    )

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
                            'engine': self.name,
                            'word_count': len(text_parts)
                        }

                except Exception as e:
                    print(f"Tesseract config error ({config}): {e}")
                    continue

            if best_result:
                return best_result

            # Fallback: просто получить текст
            text = pytesseract.image_to_string(img_rgb, config=f'-l {self.languages}')
            if text.strip():
                return {
                    'text': text.strip(),
                    'confidence': 0.5,
                    'engine': self.name,
                    'word_count': len(text.split())
                }

            return {'text': '', 'confidence': 0, 'engine': self.name}

        except Exception as e:
            print(f"❌ Tesseract error: {e}")
            return {'text': '', 'confidence': 0, 'engine': self.name, 'error': str(e)}