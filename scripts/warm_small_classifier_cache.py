"""Параллельная подготовка того же кэша миниатюр для нового Small baseline.

Не меняет изображения/разметку; готовые PNG публикуются атомарно.
Запускать только при остановленном основном процессе: Windows может блокировать
файл при одновременной записи/чтении даже с атомарной публикацией.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path
import tempfile

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]


def prepare_one(row, root, cache):
    path = cache / (row["sha256"] + ".png")
    if path.exists():
        return False
    source = (root / row["file_path"]).resolve()
    if not source.is_relative_to(root.resolve()):
        raise ValueError("Источник вне проекта")
    stat = source.stat()
    if stat.st_mtime_ns != int(row["snapshot_mtime_ns"]) or stat.st_size != int(row["snapshot_file_size"]):
        raise ValueError(f"Изображение изменилось: {row['image_id']}")
    with Image.open(source) as original:
        original.seek(0)
        original.draft("RGB", (640, 640))
        image = ImageOps.exif_transpose(original).convert("RGB")
        image.thumbnail((320, 320), Image.Resampling.LANCZOS)
    with tempfile.NamedTemporaryFile(dir=cache, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        image.save(temporary, format="PNG")
        try:
            temporary.replace(path)
        except PermissionError:
            # Основной загрузчик мог уже начать запись того же изображения.
            if not path.exists():
                raise
    finally:
        temporary.unlink(missing_ok=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "models/mobilenetv3_small_classifier")
    parser.add_argument("--split", choices=["train", "val", "test", "negative_val", "negative_test"], required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("--workers: от 1 до 8")
    cache = ROOT / ".runtime/small_classifier_cache_v1"
    cache.mkdir(parents=True, exist_ok=True)
    with (args.artifacts / "splits" / f"{args.split}.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    created = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for i, result in enumerate(executor.map(lambda row: prepare_one(row, ROOT, cache), rows), 1):
            created += int(result)
            if i % 50 == 0 or i == len(rows):
                print(f"{args.split}: {i}/{len(rows)}; new={created}", flush=True)


if __name__ == "__main__":
    main()
