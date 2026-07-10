import easyocr
import cv2
import numpy as np
from typing import Dict, List
import re
from app.config import settings


class EasyOCRWrapper:
    def __init__(self, languages: List[str] = None):
        self.name = "easyocr"
        self.languages = languages or settings.OCR_LANGUAGES

        print(f"🔄 Загрузка EasyOCR с языками: {self.languages}")

        try:
            self.reader = easyocr.Reader(
                self.languages,
                gpu=False,
                model_storage_directory='./models',
                download_enabled=True,
                verbose=False,
                recog_network='standard'
            )
            print(f"✅ EasyOCR загружен")
        except Exception as e:
            print(f"❌ Ошибка загрузки EasyOCR: {e}")
            self.reader = None

    def recognize(self, img: np.ndarray) -> Dict[str, any]:
        if self.reader is None:
            return {'text': '', 'confidence': 0, 'engine': self.name, 'error': 'Not loaded'}

        try:
            # Пробуем несколько вариантов обработки
            variants = []

            # Вариант 1: Оригинал
            if len(img.shape) == 3 and img.shape[2] == 3:
                variants.append(('original', cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            else:
                variants.append(('original', img))

            # Вариант 2: Увеличение контраста
            if len(img.shape) == 3:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                enhanced = cv2.merge((l, a, b))
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
                variants.append(('enhanced', enhanced))

            # Вариант 3: Оттенки серого
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                variants.append(('gray', gray_rgb))

            # Пробуем разные параметры
            configs = [
                {'paragraph': True, 'width_ths': 0.7, 'height_ths': 0.7,
                 'text_threshold': 0.5, 'low_text': 0.3, 'link_threshold': 0.4},
                {'paragraph': True, 'width_ths': 0.5, 'height_ths': 0.5,
                 'text_threshold': 0.3, 'low_text': 0.2, 'link_threshold': 0.3},
                {'paragraph': True, 'width_ths': 1.0, 'height_ths': 1.0,
                 'text_threshold': 0.6, 'low_text': 0.4, 'link_threshold': 0.5},
            ]

            best_result = None
            best_confidence = 0

            for variant_name, variant_img in variants:
                for config in configs:
                    try:
                        results = self.reader.readtext(variant_img, **config)

                        if results:
                            all_texts = []
                            all_confidences = []

                            for result in results:
                                if len(result) >= 3:
                                    text = result[1]
                                    confidence = result[2]
                                    if len(text) > 1:
                                        all_texts.append(text)
                                        all_confidences.append(confidence)

                            if all_texts:
                                full_text = ' '.join(all_texts)
                                full_text = re.sub(r'\s+', ' ', full_text).strip()
                                avg_conf = np.mean(all_confidences) if all_confidences else 0

                                if avg_conf > best_confidence and len(full_text) > 5:
                                    best_confidence = avg_conf
                                    best_result = {
                                        'text': full_text,
                                        'confidence': avg_conf,
                                        'engine': self.name,
                                        'word_count': len(all_texts),
                                        'variant': variant_name
                                    }
                    except Exception as e:
                        continue

            if best_result:
                return best_result

            # Если ничего не найдено, пробуем простой вариант
            try:
                results = self.reader.readtext(img, paragraph=True)
                if results:
                    all_texts = [r[1] for r in results if len(r) >= 3 and len(r[1]) > 1]
                    if all_texts:
                        return {
                            'text': ' '.join(all_texts),
                            'confidence': 0.5,
                            'engine': self.name,
                            'word_count': len(all_texts)
                        }
            except:
                pass

            return {'text': '', 'confidence': 0, 'engine': self.name}

        except Exception as e:
            print(f"❌ EasyOCR ошибка: {e}")
            return {'text': '', 'confidence': 0, 'engine': self.name, 'error': str(e)}