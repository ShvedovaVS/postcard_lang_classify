#!/usr/bin/env python3
"""
Postcard Language Detector - Console Version
Обрабатывает все изображения из папки input и сохраняет результаты
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import shutil

from app.processor import BatchProcessor
from app.config import settings

# Создаем папку для логов
log_dir = Path('/app/data/output')
log_dir.mkdir(parents=True, exist_ok=True)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / 'processing.log')
    ]
)
logger = logging.getLogger(__name__)


class PostcardProcessor:
    """Основной класс обработки открыток"""

    def __init__(self):
        self.processor = BatchProcessor()
        self.input_dir = Path(settings.INPUT_DIR)
        self.output_dir = Path(settings.OUTPUT_DIR)
        self.processed_dir = Path(settings.PROCESSED_DIR)

        # Создаем папки если их нет
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("ИНИЦИАЛИЗАЦИЯ ОБРАБОТЧИКА")
        logger.info("=" * 60)
        logger.info(f"Входная папка: {self.input_dir}")
        logger.info(f"Выходная папка: {self.output_dir}")
        logger.info(f"Папка обработанных: {self.processed_dir}")
        logger.info(f"Поддерживаемые языки: {settings.SUPPORTED_LANGUAGES}")
        logger.info("=" * 60)

        # Проверяем содержимое папки input
        self._debug_input_folder()

    def _debug_input_folder(self):
        """Диагностика содержимого папки input"""
        logger.info("ПРОВЕРКА ПАПКИ INPUT")
        logger.info("-" * 40)

        if not self.input_dir.exists():
            logger.error(f"❌ Папка {self.input_dir} не существует!")
            return

        # Все файлы в папке
        all_files = list(self.input_dir.glob('*'))
        logger.info(f"Всего файлов в папке: {len(all_files)}")

        if not all_files:
            logger.warning("⚠️ Папка пуста!")
            logger.info(f"Поместите изображения в: {self.input_dir.absolute()}")
            return

        # Проверяем изображения
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
                      '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF', '.TIF'}

        images = []
        other_files = []

        for f in all_files:
            if f.is_file():
                if f.suffix in extensions:
                    images.append(f)
                else:
                    other_files.append(f)

        logger.info(f"Найдено изображений: {len(images)}")
        logger.info(f"Других файлов: {len(other_files)}")

        if images:
            logger.info("📷 Изображения для обработки:")
            for f in images[:10]:
                size_kb = f.stat().st_size / 1024
                logger.info(f"  - {f.name} ({size_kb:.1f} KB)")
            if len(images) > 10:
                logger.info(f"  ... и еще {len(images) - 10} файлов")
        else:
            logger.warning("⚠️ Изображения не найдены!")
            if other_files:
                logger.info("Файлы в папке (не изображения):")
                for f in other_files[:5]:
                    logger.info(f"  - {f.name} (расширение: {f.suffix})")
            logger.info(f"Допустимые расширения: {', '.join(extensions)}")

        logger.info("-" * 40)

    def get_images(self) -> List[Path]:
        """Получение списка изображений для обработки"""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
                      '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF', '.TIF'}
        images = []

        for ext in extensions:
            images.extend(self.input_dir.glob(f'*{ext}'))

        # Удаляем дубликаты и сортируем
        return sorted(set(images), key=lambda x: x.name)

    def process_single_image(self, image_path: Path) -> Dict:
        """Обработка одного изображения"""
        logger.info(f"🔄 Обработка: {image_path.name}")

        try:
            start_time = time.time()

            # Читаем изображение
            with open(image_path, 'rb') as f:
                image_bytes = f.read()

            # Обрабатываем
            result = self.processor.process_image(image_bytes, image_path.name)

            # Добавляем метаданные
            result['file_name'] = image_path.name
            result['file_size'] = image_path.stat().st_size
            result['processing_time'] = time.time() - start_time
            result['processed_at'] = datetime.now().isoformat()

            # Логируем результат
            lang = result.get('language', 'unknown')
            confidence = result.get('total_score', 0)
            text_preview = result.get('text', '')[:50]

            if 'error' in result:
                logger.warning(f"  ⚠️ Ошибка: {result['error']}")
            else:
                logger.info(f"  ✅ Язык: {lang} (уверенность: {confidence:.2f})")
                if text_preview:
                    logger.info(f"  📝 Текст: {text_preview}...")

            return result

        except Exception as e:
            logger.error(f"  ❌ Ошибка при обработке {image_path.name}: {e}")
            return {
                'file_name': image_path.name,
                'error': str(e),
                'language': 'unknown',
                'text': '',
                'confidence': 0,
                'total_score': 0
            }

    def process_batch(self, max_images: Optional[int] = None):
        """Обработка всех изображений в папке"""
        images = self.get_images()

        if not images:
            logger.warning("⚠️ Нет изображений для обработки")
            logger.info(f"Поместите изображения в папку: {self.input_dir.absolute()}")
            logger.info("Поддерживаемые форматы: JPG, JPEG, PNG, BMP, TIFF")
            return

        if max_images:
            images = images[:max_images]
            logger.info(f"Ограничение: обработаем только {max_images} изображений")

        logger.info("=" * 60)
        logger.info(f"🚀 НАЧАЛО ОБРАБОТКИ {len(images)} ИЗОБРАЖЕНИЙ")
        logger.info("=" * 60)

        # Результаты для сохранения
        all_results = []
        summary = {
            'total': len(images),
            'processed': 0,
            'failed': 0,
            'start_time': datetime.now().isoformat(),
            'languages': {},
            'engines': {}
        }

        # Обрабатываем каждое изображение
        for idx, image_path in enumerate(images, 1):
            logger.info(f"\n[{idx}/{len(images)}] {image_path.name}")

            result = self.process_single_image(image_path)
            all_results.append(result)

            # Статистика по языкам
            lang = result.get('language', 'unknown')
            summary['languages'][lang] = summary['languages'].get(lang, 0) + 1

            # Статистика по движкам
            engine = result.get('engine', 'unknown')
            summary['engines'][engine] = summary['engines'].get(engine, 0) + 1

            if 'error' in result:
                summary['failed'] += 1
            else:
                summary['processed'] += 1

            # Сохраняем промежуточные результаты каждые 5 изображений
            if idx % 5 == 0:
                self.save_results(all_results, summary)
                logger.info(f"💾 Промежуточные результаты сохранены ({idx}/{len(images)})")

        # Финальное сохранение
        summary['end_time'] = datetime.now().isoformat()
        summary['duration'] = (
                datetime.fromisoformat(summary['end_time']) -
                datetime.fromisoformat(summary['start_time'])
        ).total_seconds()

        self.save_results(all_results, summary)
        self.move_processed_files(images)
        self.print_summary(summary)

        # Генерируем отчет
        self.generate_report(all_results, summary)

        logger.info("\n✅ ОБРАБОТКА ЗАВЕРШЕНА!")
        logger.info(f"📁 Результаты сохранены в: {self.output_dir}")

    def save_results(self, results: List[Dict], summary: Dict):
        """Сохранение результатов в JSON файл"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.output_dir / f'results_{timestamp}.json'
        summary_file = self.output_dir / f'summary_{timestamp}.json'

        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            logger.debug(f"Результаты сохранены: {results_file}")
            logger.debug(f"Сводка сохранена: {summary_file}")
        except Exception as e:
            logger.error(f"Ошибка сохранения результатов: {e}")

    def move_processed_files(self, processed_images: List[Path]):
        """Перемещение обработанных файлов в папку processed"""
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        moved_count = 0
        for image_path in processed_images:
            if image_path.exists():
                dest_path = self.processed_dir / image_path.name
                try:
                    # Если файл уже существует, добавляем суффикс
                    if dest_path.exists():
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        dest_path = self.processed_dir / f"{image_path.stem}_{timestamp}{image_path.suffix}"

                    shutil.move(str(image_path), str(dest_path))
                    moved_count += 1
                except Exception as e:
                    logger.error(f"Не удалось переместить {image_path.name}: {e}")

        if moved_count > 0:
            logger.info(f"📦 Перемещено {moved_count} файлов в {self.processed_dir}")

    def print_summary(self, summary: Dict):
        """Вывод сводки в консоль"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 СВОДКА ОБРАБОТКИ")
        logger.info("=" * 60)
        logger.info(f"Всего изображений:    {summary['total']}")
        logger.info(f"Успешно обработано:   {summary['processed']}")
        logger.info(f"С ошибками:           {summary['failed']}")
        logger.info(f"Время выполнения:     {summary.get('duration', 0):.2f} секунд")

        logger.info("\n📝 Распределение по языкам:")
        if summary['languages']:
            for lang, count in sorted(
                    summary['languages'].items(),
                    key=lambda x: x[1],
                    reverse=True
            ):
                percentage = (count / summary['total']) * 100 if summary['total'] > 0 else 0
                bar = '█' * int(percentage / 2) + '░' * (50 - int(percentage / 2))
                logger.info(f"  {lang:>10} : {count:>3} ({percentage:>5.1f}%) {bar}")
        else:
            logger.info("  Нет данных")

        logger.info("\n🔧 Использованные OCR движки:")
        if summary.get('engines'):
            for engine, count in summary['engines'].items():
                percentage = (count / summary['total']) * 100 if summary['total'] > 0 else 0
                logger.info(f"  {engine:>10} : {count:>3} ({percentage:>5.1f}%)")
        else:
            logger.info("  Нет данных")

        logger.info("=" * 60)

    def generate_report(self, results: List[Dict], summary: Dict):
        """Генерация CSV отчета"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.output_dir / f'report_{timestamp}.csv'

        try:
            import csv

            with open(report_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'File Name',
                    'Language',
                    'Confidence',
                    'Text',
                    'OCR Engine',
                    'Processing Time (s)',
                    'File Size (bytes)',
                    'Rotation',
                    'Error'
                ])

                for result in results:
                    writer.writerow([
                        result.get('file_name', ''),
                        result.get('language', 'unknown'),
                        f"{result.get('total_score', 0):.3f}",
                        result.get('text', '')[:200],
                        result.get('engine', ''),
                        f"{result.get('processing_time', 0):.2f}",
                        result.get('file_size', 0),
                        result.get('rotation', 0),
                        result.get('error', '')
                    ])

            logger.info(f"📄 CSV отчет сохранен: {report_file}")
        except Exception as e:
            logger.error(f"Ошибка создания CSV отчета: {e}")


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(
        description='Определение языка на открытках',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python -m app.main                          # Обработать все изображения
  python -m app.main --max-images 10          # Обработать только 10 изображений
  python -m app.main --input-dir /custom/path # Указать другую папку с изображениями
        """
    )
    parser.add_argument(
        '--max-images',
        type=int,
        help='Максимальное количество изображений для обработки'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        help='Путь к папке с изображениями (переопределяет INPUT_DIR)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Подробный вывод (DEBUG уровень)'
    )

    args = parser.parse_args()

    # Устанавливаем уровень логирования
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Включен подробный режим")

    # Если указан входной каталог, обновляем настройки
    if args.input_dir:
        settings.INPUT_DIR = args.input_dir
        os.environ['INPUT_DIR'] = args.input_dir
        logger.info(f"Используется кастомная папка: {args.input_dir}")

    processor = PostcardProcessor()

    processor.process_batch(args.max_images)


if __name__ == "__main__":
    main()