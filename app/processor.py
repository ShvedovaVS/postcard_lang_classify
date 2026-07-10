"""
Основной обработчик изображений - использует выбранный OCR движок
"""

import cv2
import numpy as np
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import re

from app.ocr import get_ocr_engine
from app.preprocessing.image_processor import ImageProcessor
from app.language.classifier import LanguageClassifier
from app.config import settings


class BatchProcessor:
    def __init__(self, ocr_engine: str = None):
        self.image_processor = ImageProcessor()
        self.ocr = get_ocr_engine(ocr_engine)
        self.language_classifier = LanguageClassifier()
        self.cache = {}
        print(f"✅ Используется OCR: {self.ocr.name}")

    def process_image(
        self,
        image_bytes: bytes,
        filename: str = "",
        try_rotations: bool = True
    ) -> Dict[str, any]:
        try:
            cache_key = hash(image_bytes)
            if cache_key in self.cache:
                return self.cache[cache_key].copy()

            img = self.image_processor.preprocess_image(image_bytes)
            enhanced_img = self.image_processor.enhance_image(img)

            if settings.AUTO_DESKEW:
                enhanced_img = self.image_processor.deskew(enhanced_img)

            best_result = None
            best_score = 0

            angles = settings.ROTATION_ANGLES if try_rotations else [0]

            for angle in angles:
                if angle != 0:
                    rotated_img = self.image_processor.rotate_image(enhanced_img, angle)
                else:
                    rotated_img = enhanced_img

                result = self.ocr.recognize(rotated_img)

                if result.get('text') and result.get('confidence', 0) > settings.MIN_CONFIDENCE:
                    cleaned_text = self._clean_text(result['text'])

                    if not cleaned_text or len(cleaned_text) < settings.MIN_TEXT_LENGTH:
                        continue

                    lang_scores = self.language_classifier.classify(cleaned_text)
                    best_lang, lang_conf = self.language_classifier.get_best_language(
                        cleaned_text
                    )

                    score = result['confidence'] * 0.6 + lang_conf * 0.4

                    result.update({
                        'text': cleaned_text,
                        'raw_text': result['text'],
                        'language': best_lang,
                        'language_confidence': lang_conf,
                        'all_languages': lang_scores,
                        'rotation': angle,
                        'total_score': score,
                        'word_count': len(cleaned_text.split())
                    })

                    if score > best_score:
                        best_score = score
                        best_result = result

            if not best_result:
                return {
                    'text': '',
                    'language': 'unknown',
                    'confidence': 0,
                    'total_score': 0,
                    'engine': self.ocr.name,
                    'error': 'Текст не обнаружен'
                }

            self.cache[cache_key] = best_result.copy()
            return best_result

        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return {
                'text': '',
                'language': 'unknown',
                'confidence': 0,
                'total_score': 0,
                'engine': self.ocr.name,
                'error': str(e)
            }

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def clear_cache(self):
        self.cache.clear()