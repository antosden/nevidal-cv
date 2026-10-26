"""Загрузка Commons-категорий из CSV в каталог city/district/poi."""
import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
API = "https://commons.wikimedia.org/w/api.php"
DEFAULT_CONFIG = ROOT / "config/wikimedia_pois.csv"
DEFAULT_OUTPUT = ROOT / "data/raw/wikimedia"
HEADERS = {
    "User-Agent": "NevidalDatasetResearch/0.1 (contact: alisa.m.shilkova@gmail.com)"
}
MAX_PER_POI = 50
MIME_EXTENSIONS = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
REQUIRED_COLUMNS = ("city", "district", "poi", "category")
PROGRESS_COLUMNS = ("status", "downloaded_count", "last_error", "progress_key")
STATUSES = {"pending", "in_progress", "done", "error"}
METADATA_COLUMNS = [
    "city", "district", "poi", "category", "file", "source_url", "commons_title",
    "width", "height", "license", "artist", "license_url",
]
SLUG = re.compile(r"[a-z0-9][a-z0-9_-]*\Z")
WINDOWS_RESERVED = {"con", "prn", "aux", "nul"} | {
    f"{prefix}{i}" for prefix in ("com", "lpt") for i in range(1, 10)
}


@dataclass(frozen=True)
class PoiConfig:
    city: str
    district: str
    poi: str
    category: str
    status: str = "pending"
    downloaded_count: int = 0
    progress_key: str = ""

    def folder(self, output):
        return Path(output) / self.city / self.district / self.poi


def load_pois(path):
    """Проверяем весь CSV до создания папок и любых сетевых запросов."""
    configs = []
    seen = set()
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)):
            raise ValueError("Повторяющиеся заголовки CSV")
        missing = set(REQUIRED_COLUMNS) - set(columns)
        if missing:
            raise ValueError(f"В CSV отсутствуют колонки: {', '.join(sorted(missing))}")
        for row in reader:
            if None in row:
                raise ValueError(f"Строка {reader.line_num}: лишние значения CSV")
            values = {key: (row.get(key) or "").strip() for key in REQUIRED_COLUMNS}
            for key in ("city", "district", "poi"):
                value = values[key]
                if not SLUG.fullmatch(value) or value in WINDOWS_RESERVED:
                    raise ValueError(
                        f"Строка {reader.line_num}: {key} должен быть безопасным slug "
                        "из строчных латинских букв, цифр, _ и -"
                    )
            category = values["category"]
            if category.startswith("Category:"):
                category = category[len("Category:"):].strip()
            if not category:
                raise ValueError(f"Строка {reader.line_num}: пустая category")
            identity = (values["city"], values["district"], values["poi"])
            if identity in seen:
                raise ValueError(f"Строка {reader.line_num}: повтор city/district/poi {identity}")
            seen.add(identity)
            status = (row.get("status") or "pending").strip()
            if status not in STATUSES:
                raise ValueError(f"Строка {reader.line_num}: неизвестный status {status!r}")
            try:
                count = int(row.get("downloaded_count") or 0)
                if count < 0:
                    raise ValueError
            except ValueError:
                raise ValueError(f"Строка {reader.line_num}: некорректный downloaded_count") from None
            configs.append(PoiConfig(*identity, category, status, count, row.get("progress_key") or ""))
    if not configs:
        raise ValueError("В CSV нет POI")
    return configs


def create_session():
    retry_strategy = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    session = requests.Session()
    session.headers.update(HEADERS)
    session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
    return session


def get_files(category, session):
    params = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": f"Category:{category}",
        "gcmtype": "file",
        "gcmlimit": 50,
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "format": "json",
        "formatversion": 2,
    }
    while True:
        response = session.get(API, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise RuntimeError(f"Commons API: {data['error'].get('code', 'unknown_error')}")
        for page in data.get("query", {}).get("pages", []):
            info = page.get("imageinfo", [])
            if info:
                yield page, info[0]
        if "continue" not in data:
            break
        params.update(data["continue"])
        time.sleep(0.2)


def metadata_value(meta, key):
    return meta.get(key, {}).get("value", "")


def atomic_csv(path, columns, rows):
    """Замена CSV только после полной записи; старый файл переживает прерывание."""
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8", dir=path.parent,
                                         suffix=".csv.tmp", delete=False) as stream:
            temp_path = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        temp_path.replace(path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def save_progress(config_path, config, status, count, key, error=""):
    if config_path is None:
        return
    path = Path(config_path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames)
        rows = list(reader)
    columns.extend(col for col in PROGRESS_COLUMNS if col not in columns)
    identity = (config.city, config.district, config.poi)
    matches = [row for row in rows if tuple((row.get(k) or "").strip()
               for k in ("city", "district", "poi")) == identity]
    if len(matches) != 1:
        raise ValueError(f"CSV изменился во время загрузки: {identity}")
    category = matches[0]["category"].strip().removeprefix("Category:").strip()
    if category != config.category:
        raise ValueError(f"Категория изменена во время загрузки: {identity}")
    matches[0].update(status=status, downloaded_count=count,
                      last_error=error[:500], progress_key=key)
    atomic_csv(path, columns, rows)


def save_metadata(output, new_rows):
    """Сохраняем также прежние записи, авторов, лицензии и дополнительные поля."""
    path = output / "metadata.csv"
    rows = {}
    columns = METADATA_COLUMNS.copy()
    if path.exists():
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            columns.extend(key for key in (reader.fieldnames or []) if key not in columns)
            for row in reader:
                rows[row["file"]] = row
    for row in new_rows:
        rows[row["file"]] = {**rows.get(row["file"], {}), **row}
    atomic_csv(path, columns, rows.values())


def completed_files(output, config):
    """Учитываем только непустые файлы с сохранёнными метаданными нужной категории."""
    path = output / "metadata.csv"
    if not path.exists():
        return set()
    completed = set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if any(row.get(k) != getattr(config, k) for k in REQUIRED_COLUMNS):
                continue
            image_path = Path(row["file"])
            if image_path.resolve().parent == config.folder(output).resolve() and image_path.is_file() and image_path.stat().st_size > 0:
                completed.add(image_path.resolve())
    return completed


def download_pois(configs, output, session, max_per_poi=MAX_PER_POI, config_path=None):
    if max_per_poi < 1:
        raise ValueError("max_per_poi должен быть положительным")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    downloaded = 0
    for config in configs:
        folder = config.folder(output)
        key = hashlib.sha256(json.dumps([config.city, config.district, config.poi,
            config.category, str(output), max_per_poi], ensure_ascii=False).encode()).hexdigest()
        completed = completed_files(output, config)
        count = len(completed)
        if config.status == "done" and config.progress_key == key and count == config.downloaded_count:
            print(f"{config.poi}: already done ({count} images), skipping")
            continue
        save_progress(config_path, config, "in_progress", count, key)
        errors = []
        try:
            folder.mkdir(parents=True, exist_ok=True)
            files = iter(get_files(config.category, session)) if count < max_per_poi else iter(())
            while count < max_per_poi:
                try:
                    page, info = next(files)
                except StopIteration:
                    break
                extension = MIME_EXTENSIONS.get(info.get("mime"))
                if extension is None:
                    continue
                url = info["url"]
                path = folder / f"{int(page['pageid'])}{extension}"
                if path in completed:
                    continue
                part_path = None
                try:
                    # Восстанавливаем metadata, если прошлый процесс упал после rename файла.
                    if not path.is_file() or path.stat().st_size == 0:
                        with session.get(url, stream=True, timeout=60) as response:
                            response.raise_for_status()
                            with tempfile.NamedTemporaryFile(dir=folder, suffix=".part", delete=False) as stream:
                                part_path = Path(stream.name)
                                for chunk in response.iter_content(1024 * 1024):
                                    if chunk:
                                        stream.write(chunk)
                        if part_path.stat().st_size == 0:
                            raise OSError("Downloaded file is empty")
                        part_path.replace(path)
                        downloaded += 1
                except (requests.RequestException, OSError) as exc:
                    errors.append(str(exc))
                    save_progress(config_path, config, "error", count, key, str(exc))
                    print(f"Download failed: {url}: {exc}")
                    continue
                finally:
                    if part_path is not None:
                        part_path.unlink(missing_ok=True)
                meta = info.get("extmetadata", {})
                row = {
                    "city": config.city, "district": config.district,
                    "poi": config.poi, "category": config.category,
                    "file": str(path), "source_url": url, "commons_title": page["title"],
                    "width": info.get("width"), "height": info.get("height"),
                    "license": metadata_value(meta, "LicenseShortName"),
                    "artist": metadata_value(meta, "Artist"),
                    "license_url": metadata_value(meta, "LicenseUrl"),
                }
                # Сначала фиксируем metadata, потом обновляем счётчик в конфигурации.
                save_metadata(output, [row])
                rows.append(row)
                completed.add(path)
                count = len(completed)
                save_progress(config_path, config, "in_progress", count, key,
                              errors[-1] if errors else "")
                time.sleep(0.2)
            failed = bool(errors) and count < max_per_poi
            save_progress(config_path, config, "error" if failed else "done", count, key,
                          errors[-1] if failed else "")
        except KeyboardInterrupt:
            save_progress(config_path, config, "in_progress", count, key, "Interrupted")
            raise
        except Exception as exc:
            save_progress(config_path, config, "error", count, key, str(exc))
            raise
        print(f"{config.city}/{config.district}/{config.poi}: {count} saved images")
    print(f"Downloaded this run: {downloaded} images")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="CSV: city,district,poi,category")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Корень загрузки")
    parser.add_argument("--max-per-poi", type=int, default=MAX_PER_POI)
    args = parser.parse_args()
    try:
        configs = load_pois(args.config)
        if args.max_per_poi < 1:
            raise ValueError("--max-per-poi должен быть положительным")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    with create_session() as session:
        download_pois(configs, args.output, session, args.max_per_poi, config_path=args.config)


if __name__ == "__main__":
    main()
