import cv2
import numpy as np


class ImageProcessor:
    @staticmethod
    def _ensure_3channel(img: np.ndarray) -> np.ndarray:
        """Гарантирует, что изображение имеет 3 канала (BGR)"""
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 3:
            return img
        elif img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def preprocess_image(image_bytes: bytes) -> np.ndarray:
        """Подготовка изображения к OCR"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError("Не удалось загрузить изображение")

        img = ImageProcessor._ensure_3channel(img)

        # Изменение размера если слишком большое
        h, w = img.shape[:2]
        max_size = 4096
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            img = cv2.resize(img, new_size)

        return img

    @staticmethod
    def _upscale_if_needed(img: np.ndarray, min_size: int = 800) -> np.ndarray:
        """Увеличивает изображение если оно слишком маленькое"""
        h, w = img.shape[:2]

        if min(h, w) < min_size:
            scale = min_size / min(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        return img

    @staticmethod
    def enhance_image(img: np.ndarray) -> np.ndarray:
        """Улучшение качества изображения"""
        img = ImageProcessor._ensure_3channel(img)

        # Увеличиваем если нужно
        img = ImageProcessor._upscale_if_needed(img)

        # Конвертация в оттенки серого
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Улучшение контраста
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 2. Удаление шума
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

        # 3. Улучшение резкости
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)

        # 4. Адаптивная пороговая обработка
        thresh = cv2.adaptiveThreshold(
            sharpened, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 2
        )

        # Возвращаем как 3-канальное изображение
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
        """Поворот изображения"""
        if angle == 0:
            return img

        img = ImageProcessor._ensure_3channel(img)

        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

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
        img = ImageProcessor._ensure_3channel(img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

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