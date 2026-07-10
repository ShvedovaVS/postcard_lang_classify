"""
Основной обработчик изображений - использует Tesseract
"""

import cv2
import numpy as np
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

from app.ocr.tesseract_wrapper import TesseractOCR
from app.preprocessing.image_processor import ImageProcessor
from app.language.classifier import LanguageClassifier
from app.config import settings


class BatchProcessor:
    """Обработчик для пакетной обработки (только Tesseract)"""

    def __init__(self):
        self.image_processor = ImageProcessor()
        self.tesseract = TesseractOCR()
        self.language_classifier = LanguageClassifier()
        self.cache = {}

        print("✅ BatchProcessor initialized with Tesseract only")
        print(f"   Supported languages: {settings.SUPPORTED_LANGUAGES}")
        print(f"   Rotation angles: {settings.ROTATION_ANGLES}")

    def process_image(
        self,
        image_bytes: bytes,
        filename: str = "",
        try_rotations: bool = True
    ) -> Dict[str, any]:
        """
        Обработка одного изображения

        Args:
            image_bytes: байты изображения
            filename: имя файла для логирования
            try_rotations: пробовать ли разные углы поворота

        Returns:
            Dict с результатами
        """
        try:
            # Проверка кеша
            cache_key = hash(image_bytes)
            if cache_key in self.cache:
                result = self.cache[cache_key].copy()
                result['from_cache'] = True
                return result

            # 1. Загрузка и предобработка
            img = self.image_processor.preprocess_image(image_bytes)
            enhanced_img = self.image_processor.enhance_image(img)

            # 2. Автоматическое выравнивание
            if settings.AUTO_DESKEW:
                enhanced_img = self.image_processor.deskew(enhanced_img)

            # 3. Распознавание текста с разными поворотами
            best_result = None
            best_score = 0

            angles = settings.ROTATION_ANGLES if try_rotations else [0]

            for angle in angles:
                # Поворачиваем изображение
                if angle != 0:
                    rotated_img = self.image_processor.rotate_image(enhanced_img, angle)
                else:
                    rotated_img = enhanced_img

                # Распознаем текст через Tesseract
                result = self.tesseract.recognize(rotated_img)

                # Проверяем, есть ли текст
                if result['text'] and result['confidence'] > settings.MIN_CONFIDENCE:
                    # Определяем язык
                    if result['text'] == 'MondialCollection':
                        continue
                    lang_scores = self.language_classifier.classify(result['text'])
                    best_lang, lang_conf = self.language_classifier.get_best_language(
                        result['text']
                    )

                    # Общая оценка
                    score = result['confidence'] * 0.6 + lang_conf * 0.4

                    # Добавляем информацию

                    if score > best_score:
                        result.update({
                            'language': best_lang,
                            'language_confidence': lang_conf,
                            'all_languages': lang_scores,
                            'rotation': angle,
                            'total_score': score
                        })
                        best_score = score
                        best_result = result

            # Если не удалось распознать текст
            if not best_result:
                return {
                    'text': '',
                    'language': 'unknown',
                    'confidence': 0,
                    'total_score': 0,
                    'engine': 'tesseract',
                    'error': 'Текст не обнаружен'
                }

            # Сохраняем в кеш
            self.cache[cache_key] = best_result.copy()
            return best_result

        except Exception as e:
            return {
                'text': '',
                'language': 'unknown',
                'confidence': 0,
                'total_score': 0,
                'engine': 'tesseract',
                'error': f'Ошибка обработки: {str(e)}'
            }

    def process_batch(
        self,
        images: List[bytes],
        max_workers: int = 4
    ) -> List[Dict[str, any]]:
        """
        Пакетная обработка изображений

        Args:
            images: список байтов изображений
            max_workers: количество параллельных потоков

        Returns:
            Список результатов
        """
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.process_image, img)
                for img in images
            ]
            for future in futures:
                results.append(future.result())
        return results

    def clear_cache(self):
        """Очистка кеша"""
        self.cache.clear()

    def get_stats(self) -> Dict:
        """Получение статистики"""
        return {
            'cache_size': len(self.cache),
            'tesseract_loaded': True,
            'supported_languages': settings.SUPPORTED_LANGUAGES,
            'rotation_angles': settings.ROTATION_ANGLES
        }