from langdetect import detect, detect_langs, DetectorFactory
import re
from typing import Dict, List, Tuple
from collections import Counter
import numpy as np

# Для воспроизводимости результатов
DetectorFactory.seed = 0


class LanguageClassifier:
    def __init__(self):
        # Общие слова для разных языков (можно расширить)
        self.language_keywords = {
            'en': ['the', 'and', 'for', 'with', 'you', 'that', 'this', 'from'],
            'ru': ['и', 'в', 'на', 'с', 'по', 'что', 'как', 'это', 'для'],
            'fr': ['le', 'la', 'les', 'et', 'pour', 'avec', 'vous', 'que', 'dans'],
            'de': ['der', 'die', 'das', 'und', 'für', 'mit', 'den', 'von', 'zu'],
            'es': ['el', 'la', 'los', 'las', 'y', 'para', 'con', 'que', 'en'],
            'it': ['il', 'la', 'le', 'e', 'per', 'con', 'che', 'nel', 'del'],
            'pt': ['o', 'a', 'os', 'as', 'e', 'para', 'com', 'que', 'em'],
            'nl': ['de', 'het', 'een', 'en', 'voor', 'met', 'die', 'van', 'te'],
            'pl': ['i', 'w', 'na', 'z', 'do', 'po', 'za', 'nie', 'się'],
            'uk': ['і', 'в', 'на', 'з', 'до', 'по', 'за', 'не', 'що']
        }

    def classify(self, text: str) -> Dict[str, float]:
        """Определение языка текста"""
        if not text or len(text.strip()) < 3:
            return {'unknown': 1.0}

        # Очистка текста
        clean_text = self._clean_text(text)

        results = {}

        # 1. Использование langdetect
        try:
            detected_langs = detect_langs(clean_text)
            for lang in detected_langs:
                results[lang.lang] = lang.prob
        except:
            pass

        # 2. Дополнительная проверка по ключевым словам
        keyword_scores = self._keyword_analysis(clean_text)

        # Объединение результатов
        final_scores = {}

        # Если есть результаты от langdetect
        if results:
            # Нормализация оценок
            for lang, prob in results.items():
                if lang in keyword_scores:
                    # Комбинируем оценки
                    final_scores[lang] = prob * 0.7 + keyword_scores[lang] * 0.3
                else:
                    final_scores[lang] = prob
        else:
            # Только по ключевым словам
            for lang, score in keyword_scores.items():
                if score > 0:
                    final_scores[lang] = score

        # Сортировка по убыванию уверенности
        sorted_scores = dict(sorted(
            final_scores.items(),
            key=lambda x: x[1],
            reverse=True
        ))

        return sorted_scores

    def _clean_text(self, text: str) -> str:
        """Очистка текста"""
        # Удаление специальных символов
        text = re.sub(r'[^\w\s]', ' ', text)
        # Удаление лишних пробелов
        text = ' '.join(text.split())
        return text.lower()

    def _keyword_analysis(self, text: str) -> Dict[str, float]:
        """Анализ ключевых слов"""
        words = text.split()
        word_count = len(words)

        if word_count == 0:
            return {}

        scores = {}
        for lang, keywords in self.language_keywords.items():
            matches = sum(1 for word in words if word in keywords)
            if matches > 0:
                scores[lang] = matches / len(keywords)

        return scores

    def get_best_language(self, text: str) -> Tuple[str, float]:
        """Получение наиболее вероятного языка"""
        scores = self.classify(text)
        if scores:
            best_lang = max(scores.items(), key=lambda x: x[1])
            return best_lang
        return 'unknown', 0.0