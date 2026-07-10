import cv2
import numpy as np
from PIL import Image
import pytesseract
import easyocr
import fasttext
import os
from pathlib import Path
from tqdm import tqdm
import json

print("🚀 Инициализация моделей...")
reader = easyocr.Reader(['ru', 'en', 'de', 'fr', 'es', 'zh_sim', 'ja', 'ko'],
                        gpu=False, verbose=False)

fasttext_model = fasttext.load_model('lid.176.ftz')


def preprocess_image(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.fastNlMeansDenoising(img)
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(binary, -1, kernel)
    return sharpened


def ocr_easyocr(image_path):
    img = preprocess_image(image_path)
    if img is None:
        return ""
    temp_path = f"temp_{os.getpid()}.jpg"
    cv2.imwrite(temp_path, img)
    result = reader.readtext(temp_path, detail=0, paragraph=True)
    text = ' '.join(result)
    if os.path.exists(temp_path):
        os.remove(temp_path)
    return text.strip()


def ocr_tesseract(image_path):
    img = preprocess_image(image_path)
    if img is None:
        return ""
    pil_img = Image.fromarray(img)
    text = pytesseract.image_to_string(pil_img, lang='rus+eng+deu+fra+spa', config='--psm 6')
    return text.strip()


def detect_language(text):
    if not text or len(text.strip()) < 15:
        return "unknown", 0.0
    pred = fasttext_model.predict(text.replace('\n', ' ').strip())
    lang_code = pred[0][0].replace('__label__', '')
    confidence = float(pred[1][0])
    return lang_code, confidence


def process_single_image(image_path):
    print(f"\n📸 {image_path.name}")
    text_easy = ocr_easyocr(image_path)
    lang, conf = detect_language(text_easy)

    print(f"EasyOCR: {text_easy[:120]}..." if len(text_easy) > 120 else f"EasyOCR: {text_easy}")
    print(f"→ Язык: {lang.upper()} (уверенность: {conf:.3f})")

    return {
        "file": image_path.name,
        "text": text_easy,
        "language": lang,
        "confidence": conf,
        "engine": "easyocr"
    }


def batch_process(folder_path="postcards", output_json="results.json"):
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Папка {folder_path} не найдена!")
        return

    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_files.extend(folder.glob(ext))

    print(f"Найдено {len(image_files)} изображений.")

    results = []
    for img_path in tqdm(image_files):
        result = process_single_image(img_path)
        results.append(result)

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Готово! Результаты в файле: {output_json}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        batch_process(sys.argv[1])
    else:
        batch_process()