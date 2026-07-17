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
    def __init__(
        self,
        ocr_engine: str = None,
        enhance_image: Optional[bool] = None,
        auto_deskew: Optional[bool] = None,
        try_rotations: Optional[bool] = None,
        rotation_angles: Optional[List[int]] = None,
        min_confidence: Optional[float] = None,
        min_text_length: Optional[int] = None
    ):
        """
        Инициализация обработчика

        Args:
            ocr_engine: Имя OCR движка ('tesseract', 'easyocr', 'paddleocr', 'kosmos', 'trocr')
            enhance_image: Улучшать ли качество изображения (True/False)
            auto_deskew: Выравнивать ли текст автоматически (True/False)
            try_rotations: Проверять ли разные углы поворота (True/False)
            rotation_angles: Список углов для проверки (по умолчанию [0, 90, 180, 270])
            min_confidence: Минимальная уверенность для принятия результата
            min_text_length: Минимальная длина текста для принятия результата
        """
        self.image_processor = ImageProcessor()
        self.ocr = get_ocr_engine(ocr_engine)
        self.language_classifier = LanguageClassifier()
        self.cache = {}

        # 👇 НАСТРОЙКИ С ВОЗМОЖНОСТЬЮ ПЕРЕОПРЕДЕЛЕНИЯ
        self.enhance_image = enhance_image if enhance_image is not None else settings.ENHANCE_IMAGE
        self.auto_deskew = auto_deskew if auto_deskew is not None else settings.AUTO_DESKEW
        self.try_rotations = try_rotations if try_rotations is not None else settings.TRY_ROTATIONS
        self.rotation_angles = rotation_angles if rotation_angles is not None else settings.ROTATION_ANGLES
        self.min_confidence = min_confidence if min_confidence is not None else settings.MIN_CONFIDENCE
        self.min_text_length = min_text_length if min_text_length is not None else settings.MIN_TEXT_LENGTH

        print(f"✅ Используется OCR: {self.ocr.name}")
        print(f"   Настройки:")
        print(f"   - Улучшение изображения: {self.enhance_image}")
        print(f"   - Автовыравнивание: {self.auto_deskew}")
        print(f"   - Проверка поворотов: {self.try_rotations}")
        print(f"   - Углы поворота: {self.rotation_angles}")
        print(f"   - Мин. уверенность: {self.min_confidence}")
        print(f"   - Мин. длина текста: {self.min_text_length}")

    def process_image(
        self,
        image_bytes: bytes,
        filename: str = "",
        try_rotations: Optional[bool] = None,
        enhance_image: Optional[bool] = None,
        auto_deskew: Optional[bool] = None
    ) -> Dict[str, any]:
        """
        Обработка одного изображения

        Args:
            image_bytes: байты изображения
            filename: имя файла для логирования
            try_rotations: переопределить проверку поворотов для этого изображения
            enhance_image: переопределить улучшение качества для этого изображения
            auto_deskew: переопределить автовыравнивание для этого изображения
        """
        try:
            cache_key = hash(image_bytes)
            if cache_key in self.cache:
                return self.cache[cache_key].copy()

            # 1. Загрузка изображения
            img = self.image_processor.preprocess_image(image_bytes)

            # 2. Улучшение качества (если включено)
            if enhance_image if enhance_image is not None else self.enhance_image:
                img = self.image_processor.enhance_image(img)

            # 3. Автовыравнивание (если включено)
            if auto_deskew if auto_deskew is not None else self.auto_deskew:
                img = self.image_processor.deskew(img)

            # 4. Определяем углы для проверки
            use_rotations = try_rotations if try_rotations is not None else self.try_rotations
            angles = self.rotation_angles if use_rotations else [0]

            best_result = None
            best_score = 0

            for angle in angles:
                if angle != 0:
                    rotated_img = self.image_processor.rotate_image(img, angle)
                else:
                    rotated_img = img

                result = self.ocr.recognize(rotated_img)

                if result.get('text') and result.get('confidence', 0) > self.min_confidence:
                    cleaned_text = self._clean_text(result['text'])

                    if not cleaned_text or len(cleaned_text) < self.min_text_length:
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
                        'word_count': len(cleaned_text.split()),
                        # Добавляем информацию о настройках
                        'settings_used': {
                            'enhance_image': self.enhance_image,
                            'auto_deskew': self.auto_deskew,
                            'try_rotations': self.try_rotations
                        }
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

    def get_settings(self) -> Dict:
        """Возвращает текущие настройки обработчика"""
        return {
            'ocr_engine': self.ocr.name,
            'enhance_image': self.enhance_image,
            'auto_deskew': self.auto_deskew,
            'try_rotations': self.try_rotations,
            'rotation_angles': self.rotation_angles,
            'min_confidence': self.min_confidence,
            'min_text_length': self.min_text_length
        }