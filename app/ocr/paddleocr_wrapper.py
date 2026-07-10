"""
PaddleOCR wrapper
"""

import cv2
import numpy as np
from typing import Dict, List
import re

try:
    from paddleocr import PaddleOCR

    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    print("⚠️ PaddleOCR not installed. Install with: pip install paddlepaddle paddleocr")

from app.ocr.base import BaseOCR


class PaddleOCRWrapper(BaseOCR):
    """PaddleOCR implementation"""

    def __init__(self, languages: List[str] = None):
        super().__init__()
        self.name = "paddleocr"

        if not PADDLE_AVAILABLE:
            print("❌ PaddleOCR not available")
            self.reader = None
            return

        # PaddleOCR поддерживает многие языки
        self.languages = languages or ['en', 'fr', 'de', 'es', 'it', 'pt', 'nl']

        print("🔄 Загрузка PaddleOCR...")

        try:
            # PaddleOCR использует свои обозначения языков
            # 'en' - английский, 'fr' - французский, 'de' - немецкий и т.д.
            self.reader = PaddleOCR(
                use_angle_cls=True,  # Определение угла текста
                lang='en',  # Основной язык (можно менять)
                show_log=False,
                use_gpu=False,
                enable_mkldnn=True,  # Ускорение на CPU
                det_db_thresh=0.3,
                det_db_box_thresh=0.5,
                det_db_unclip_ratio=1.6,
                max_batch_size=10,
                use_dilation=True,
                det_db_score_mode='slow'
            )
            print(f"✅ PaddleOCR загружен")
        except Exception as e:
            print(f"❌ Ошибка загрузки PaddleOCR: {e}")
            self.reader = None

    def recognize(self, img: np.ndarray) -> Dict[str, any]:
        """Распознавание текста через PaddleOCR"""
        if self.reader is None:
            return {'text': '', 'confidence': 0, 'engine': self.name, 'error': 'Not loaded'}

        try:
            # PaddleOCR ожидает RGB или BGR
            if len(img.shape) == 3 and img.shape[2] == 3:
                # Уже цветное
                pass
            else:
                # Конвертируем в цветное если нужно
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # Распознаем
            result = self.reader.ocr(img, cls=True)

            if not result or not result[0]:
                return {'text': '', 'confidence': 0, 'engine': self.name}

            all_texts = []
            all_confidences = []
            all_boxes = []

            for line in result[0]:
                # Формат: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], (text, confidence)]
                if len(line) >= 2:
                    bbox = line[0]
                    text, confidence = line[1]

                    all_texts.append(text)
                    all_confidences.append(confidence)
                    all_boxes.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox
                    })

            if not all_texts:
                return {'text': '', 'confidence': 0, 'engine': self.name}

            full_text = ' '.join(all_texts)
            full_text = re.sub(r'\s+', ' ', full_text).strip()

            avg_confidence = np.mean(all_confidences) if all_confidences else 0

            return {
                'text': full_text,
                'confidence': avg_confidence,
                'engine': self.name,
                'word_count': len(all_texts),
                'segments': all_boxes
            }

        except Exception as e:
            print(f"❌ PaddleOCR ошибка: {e}")
            return {'text': '', 'confidence': 0, 'engine': self.name, 'error': str(e)}