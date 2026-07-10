"""
Базовый класс для всех OCR движков
"""

from typing import Dict, List
import numpy as np

class BaseOCR:
    """Базовый класс для OCR движков"""

    def __init__(self):
        self.name = "base"
        self.languages = []

    def recognize(self, img: np.ndarray) -> Dict[str, any]:
        """
        Распознавание текста

        Returns:
            Dict с полями:
            - text: распознанный текст
            - confidence: уверенность (0-1)
            - engine: имя движка
            - word_count: количество слов
        """
        raise NotImplementedError

    def get_supported_languages(self) -> List[str]:
        return self.languages