import cv2
import numpy as np
from PIL import Image
import io
from typing import Tuple, Optional


class ImageProcessor:
    @staticmethod
    def preprocess_image(image_bytes: bytes) -> np.ndarray:
        """Подготовка изображения к OCR"""
        # Конвертация байтов в изображение
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Не удалось загрузить изображение")

        # Проверяем количество каналов
        if len(img.shape) == 2:
            # Если изображение уже черно-белое, конвертируем в цветное
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            # Если есть альфа-канал, убираем его
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Изменение размера если слишком большое
        h, w = img.shape[:2]
        max_size = 4096
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            img = cv2.resize(img, new_size)

        return img

    @staticmethod
    def enhance_image(img: np.ndarray) -> np.ndarray:
        """Улучшение качества изображения"""
        # Проверяем, что изображение цветное (3 канала)
        if len(img.shape) == 2:
            # Конвертируем в BGR если灰度
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            # Убираем альфа-канал
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Конвертация в оттенки серого (теперь это безопасно)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Применение адаптивной пороговой обработки
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Удаление шума
        denoised = cv2.fastNlMeansDenoising(thresh)

        # Увеличение контраста
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Возвращаем изображение в цветном формате для Tesseract
        # Tesseract лучше работает с цветными изображениями
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
        """Поворот изображения"""
        if angle == 0:
            return img

        # Проверяем количество каналов
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Вычисление нового размера
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        rotated = cv2.warpAffine(img, M, (new_w, new_h))
        return rotated

    @staticmethod
    def deskew(img: np.ndarray) -> np.ndarray:
        """Автоматическое выравнивание текста"""
        # Проверяем количество каналов
        if len(img.shape) == 2:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        if lines is not None:
            angles = []
            for line in lines:
                rho, theta = line[0]
                angle = theta * 180 / np.pi - 90
                angles.append(angle)

            if angles:
                median_angle = np.median(angles)
                if abs(median_angle) > 0.5:
                    rotated = ImageProcessor.rotate_image(img, -median_angle)
                    return rotated

        return img