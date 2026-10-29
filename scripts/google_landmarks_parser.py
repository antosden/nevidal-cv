"""Выборочная загрузка наших POI из GLDv2 train по CSV, без TAR-архивов."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from urllib.parse import unquote, urlsplit

from PIL import Image
import requests

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.wikimedia_parser import ROOT, SLUG, WINDOWS_RESERVED, atomic_csv, create_session

BASE = "https://s3.amazonaws.com/google-landmark/metadata/"
SCHEMAS = {
    "train_label_to_category.csv": {"landmark_id", "category"},
    "train_clean.csv": {"landmark_id", "images"},
    "train.csv": {"id", "url", "landmark_id"},
    "train_attribution.csv": {"id", "url", "author", "license", "title"},
}
PROGRESS = ["status", "downloaded_count", "last_error", "progress_key"]
METADATA = ["image_id", "gld_image_id", "landmark_id", "city", "district", "poi",
            "poi_id", "category", "gld_category", "mapping_method", "file", "file_path",
            "source", "source_url", "attribution_url", "author", "licence", "title",
            "width", "height", "sha256", "label_status"]
IMAGE_ID = re.compile(r"[0-9a-f]{16}\Z")


def read_csv(path, required):
    # Поля attribution иногда содержат большие HTML-строки.
    csv.field_size_limit(16 * 1024 * 1024)
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)) or not set(required) <= set(columns):
            raise ValueError(f"Некорректные колонки: {path}; нужны {sorted(required)}")
        for row in reader:
            if None in row:
                raise ValueError(f"Лишние значения CSV: {path}:{reader.line_num}")
            yield row


def category_key(value):
    value = value.strip()
    if value.startswith(("https://", "http://")):
        parsed = urlsplit(value)
        if parsed.hostname != "commons.wikimedia.org" or not parsed.path.startswith("/wiki/Category:"):
            raise ValueError(f"Ожидалась ссылка Commons Category: {value}")
        value = parsed.path[len("/wiki/"):]
    return unquote(value).removeprefix("Category:").replace("_", " ").strip()


def load_config(path):
    rows = list(read_csv(path, {"city", "district", "poi", "category"}))
    seen = set()
    for row in rows:
        for key in ("city", "district", "poi"):
            row[key] = row[key].strip()
            if not SLUG.fullmatch(row[key]) or row[key] in WINDOWS_RESERVED:
                raise ValueError(f"Небезопасный {key}: {row[key]!r}")
        identity = tuple(row[k] for k in ("city", "district", "poi"))
        if identity in seen:
            raise ValueError(f"Повтор POI: {identity}")
        seen.add(identity)
        ids = row.get("landmark_ids", "").split()
        if any(not re.fullmatch(r"0|[1-9][0-9]*", item) for item in ids):
            raise ValueError("landmark_ids: целые неотрицательные ID через пробел")
        if row["category"].strip():
            category_key(row["category"])
    if not rows:
        raise ValueError("Пустой конфиг")
    return rows


def stream_download(session, url, path, max_bytes, *, resume=False, attempts=1):
    """Докачка метаданных только при подтверждённой неизменности HTTP-ресурса."""
    if urlsplit(url).scheme not in {"https", "http"}:
        raise ValueError("Допустимы только HTTP(S) ссылки")
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    state_path = part.with_suffix(part.suffix + ".json")
    try:
        for attempt in range(1, attempts + 1):
            state = {}
            if resume and state_path.exists():
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                except (ValueError, OSError):
                    pass
            offset = part.stat().st_size if resume and part.exists() else 0
            if state.get("url") != url or not state.get("validator"):
                offset = 0
            headers = {"Accept-Encoding": "identity"}
            if offset:
                headers.update(Range=f"bytes={offset}-", **{"If-Range": state["validator"]})
            try:
                with session.get(url, stream=True, timeout=(15, 180), headers=headers) as response:
                    if response.status_code == 416 and offset:
                        # Не принимаем неподтверждённый остаток за готовый CSV.
                        part.unlink(missing_ok=True)
                        state_path.unlink(missing_ok=True)
                        raise requests.ConnectionError("Сервер отклонил Range; начнём файл заново")
                    response.raise_for_status()
                    response_headers = response.headers
                    validator = response_headers.get("ETag", "")
                    if validator.startswith("W/"):
                        validator = ""
                    validator = validator or response_headers.get("Last-Modified", "")
                    total = None
                    if response.status_code == 206:
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response_headers.get("Content-Range", ""))
                        if not match or int(match[1]) != offset or not offset or validator != state.get("validator"):
                            part.unlink(missing_ok=True)
                            state_path.unlink(missing_ok=True)
                            raise requests.ConnectionError("Некорректный Content-Range/версия ресурса; повторяем сначала")
                        total = int(match[3])
                        if int(match[2]) < offset or int(match[2]) >= total:
                            raise ValueError("Некорректная граница Content-Range")
                    elif response.status_code == 200:
                        # Сервер вправе проигнорировать Range или вернуть новую версию.
                        offset = 0
                        length = response_headers.get("Content-Length")
                        total = int(length) if length is not None else None
                    else:
                        raise ValueError(f"Неожиданный HTTP-статус {response.status_code}")
                    if response_headers.get("Content-Encoding", "identity") not in ("", "identity"):
                        raise ValueError("Для безопасной докачки нужен ответ без Content-Encoding")
                    if total is not None and total > max_bytes:
                        raise ValueError(f"Превышен лимит файла {max_bytes} байт")
                    size = offset
                    last_print = time.monotonic()
                    with part.open("ab" if offset else "wb") as stream:
                        if resume:
                            state_path.write_text(json.dumps(dict(url=url, validator=validator)), encoding="utf-8")
                        for chunk in response.iter_content(64 * 1024):
                            size += len(chunk)
                            if size > max_bytes:
                                raise ValueError(f"Превышен лимит файла {max_bytes} байт")
                            stream.write(chunk)
                            if resume and time.monotonic() - last_print >= 5:
                                expected = f" / {total / 1024**2:.1f}" if total is not None else ""
                                print(f"{path.name}: {size / 1024**2:.1f}{expected} MiB", flush=True)
                                last_print = time.monotonic()
                    if not size:
                        raise requests.ConnectionError("Пустой ответ")
                    if total is not None and size != total:
                        raise requests.ConnectionError(f"Неполный ответ: {size} из {total} байт")
                part.replace(path)
                state_path.unlink(missing_ok=True)
                return
            except requests.RequestException as exc:
                if attempt == attempts:
                    raise
                print(f"{path.name}: обрыв загрузки ({exc}); повтор {attempt + 1}/{attempts}", flush=True)
                time.sleep(min(2 ** (attempt - 1), 8))
    finally:
        # Остаток метаданных переживает исчерпание попыток и Ctrl+C.
        if not resume:
            part.unlink(missing_ok=True)
            state_path.unlink(missing_ok=True)


def metadata_file(cache, name, session, offline=False):
    path = cache / name
    if not path.exists():
        if offline:
            raise FileNotFoundError(f"Нет локальных метаданных: {path}")
        print(f"Метаданные: {name}", flush=True)
        stream_download(session, BASE + name, path, 2 * 1024**3, resume=True, attempts=5)
    # Проверка схемы выполняется также для кэша.
    next(read_csv(path, SCHEMAS[name]), None)
    return path


def resolve_pois(configs, category_path):
    categories = {r["landmark_id"]: r["category"] for r in
                  read_csv(category_path, SCHEMAS["train_label_to_category.csv"])}
    by_category = {}
    for landmark, category in categories.items():
        by_category.setdefault(category_key(category), []).append(landmark)
    owners = {}
    resolved = []
    for config in configs:
        override = config.get("landmark_ids", "").split()
        ids = override or by_category.get(category_key(config["category"]), [])
        for landmark in ids:
            if landmark not in categories:
                raise ValueError(f"Неизвестный train landmark_id: {landmark}")
            if landmark in owners:
                raise ValueError(f"landmark_id {landmark} назначен нескольким POI")
            owners[landmark] = config["poi"]
        resolved.append((config, {i: categories[i] for i in ids},
                         "explicit_id" if override else "exact_category"))
    return resolved


def collect_candidates(clean_path, train_path, attribution_path, wanted):
    print("Чтение train_clean.csv: отбор ID наших POI", flush=True)
    clean = {}
    for row in read_csv(clean_path, SCHEMAS["train_clean.csv"]):
        if row["landmark_id"] in wanted:
            for image_id in row["images"].split():
                if not IMAGE_ID.fullmatch(image_id):
                    raise ValueError(f"Некорректный GLD image ID: {image_id}")
                if image_id in clean and clean[image_id] != row["landmark_id"]:
                    raise ValueError(f"Конфликт clean-разметки {image_id}")
                clean[image_id] = row["landmark_id"]
    selected = {}
    print(f"Чтение train.csv: поиск {len(clean)} очищенных изображений", flush=True)
    for row in read_csv(train_path, SCHEMAS["train.csv"]):
        if row["id"] in clean:
            if row["landmark_id"] != clean[row["id"]]:
                raise ValueError(f"Конфликт train/clean {row['id']}")
            selected[row["id"]] = row
    print(f"Чтение train_attribution.csv: атрибуция {len(selected)} изображений", flush=True)
    for row in read_csv(attribution_path, SCHEMAS["train_attribution.csv"]):
        if row["id"] in selected:
            selected[row["id"]]["attribution"] = row
    return selected


def fingerprint(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def valid_saved(row, folder):
    path = folder / (row["gld_image_id"] + ".jpg")
    if path.is_file() and fingerprint(path) == row.get("sha256"):
        return True
    return False


def export_pois(configs, resolved, candidates, output, config_path, session, limit):
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "metadata.csv"
    previous = list(read_csv(metadata_path, METADATA)) if metadata_path.exists() else []
    metadata = {(r["poi_id"], r["gld_image_id"]): r for r in previous}
    errors_path = output / "download_errors.csv"
    error_columns = ["poi_id", "gld_image_id", "error"]
    errors = {(r["poi_id"], r["gld_image_id"]): r for r in
              read_csv(errors_path, error_columns)} if errors_path.exists() else {}
    columns = list(dict.fromkeys([*configs[0], *PROGRESS]))
    failures = False
    for config, landmarks, method in resolved:
        poi_id = "/".join(config[k] for k in ("city", "district", "poi"))
        folder = output.joinpath(*poi_id.split("/"))
        choices = sorted((r for r in candidates.values() if r["landmark_id"] in landmarks),
                         key=lambda r: r["id"])
        key = hashlib.sha256(json.dumps([poi_id, landmarks, str(output), limit,
                    [(r["id"], r["url"]) for r in choices]], sort_keys=True).encode()).hexdigest()
        completed = {image_id for (poi, image_id), row in metadata.items()
                     if poi == poi_id and image_id in candidates
                     and row["landmark_id"] in landmarks and valid_saved(row, folder)}

        def progress(status, error=""):
            config.update(status=status, downloaded_count=str(len(completed)),
                          last_error=error[:1000], progress_key=key)
            atomic_csv(config_path, columns, configs)

        if not landmarks:
            progress("not_found", "Нет точного соответствия; укажите проверенные category или landmark_ids")
            print(f"{poi_id}: not_found", flush=True)
            continue
        progress("in_progress")
        current_errors = []
        try:
            for candidate in choices:
                image_id = candidate["id"]
                if len(completed) >= limit:
                    break
                if image_id in completed:
                    continue
                destination = folder / (image_id + ".jpg")
                try:
                    attribution = candidate.get("attribution")
                    if not attribution:
                        raise ValueError("Нет записи attribution; загрузка отложена")
                    stream_download(session, candidate["url"], destination, 64 * 1024**2)
                    with Image.open(destination) as image:
                        image.verify()
                    with Image.open(destination) as image:
                        width, height = image.size
                        if image.format != "JPEG":
                            raise ValueError("Ожидался JPEG GLDv2 train")
                        image.load()
                    row = dict(image_id="gld_" + image_id, gld_image_id=image_id,
                               landmark_id=candidate["landmark_id"],
                               **{k: config[k] for k in ("city", "district", "poi", "category")},
                               poi_id=poi_id, gld_category=landmarks[candidate["landmark_id"]],
                               mapping_method=method, file=destination.relative_to(output).as_posix(),
                               file_path=destination.relative_to(output).as_posix(),
                               source="google_landmarks_v2", source_url=candidate["url"],
                               attribution_url=attribution["url"], author=attribution["author"] or "unknown",
                               licence=attribution["license"] or "unknown", title=attribution["title"],
                               width=width, height=height, sha256=fingerprint(destination),
                               label_status="needs_review")
                    metadata[(poi_id, image_id)] = row
                    atomic_csv(metadata_path, METADATA, metadata.values())
                    completed.add(image_id)
                    errors.pop((poi_id, image_id), None)
                    atomic_csv(errors_path, error_columns, errors.values())
                    progress("in_progress")
                    print(f"{poi_id}: {len(completed)}/{limit}", flush=True)
                except (requests.RequestException, OSError, ValueError, Image.DecompressionBombError) as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    errors[(poi_id, image_id)] = dict(poi_id=poi_id, gld_image_id=image_id, error=message)
                    atomic_csv(errors_path, error_columns, errors.values())
                    current_errors.append(message)
            if current_errors:
                failures = True
            progress("error" if current_errors else "done", "; ".join(current_errors))
        except BaseException:
            progress("error", "Прервано; повторите команду для продолжения")
            raise
    return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/google_landmarks_pois.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw/google_landmarks_v2")
    parser.add_argument("--metadata-dir", type=Path, default=ROOT / ".runtime/google_landmarks_v2/metadata")
    parser.add_argument("--max-per-poi", type=int, default=50)
    parser.add_argument("--offline", action="store_true", help="Метаданные уже скачаны; фото требуют сети")
    parser.add_argument("--plan-only", action="store_true", help="Только соответствия категорий, без фото")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/google_landmarks_mapping.csv")
    args = parser.parse_args(argv)
    if args.max_per_poi < 1:
        parser.error("--max-per-poi должен быть положительным")
    configs = load_config(args.config)
    with create_session() as session:
        category_path = metadata_file(args.metadata_dir, "train_label_to_category.csv", session, args.offline)
        resolved = resolve_pois(configs, category_path)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        atomic_csv(args.report, ["poi_id", "category", "landmark_ids", "gld_categories", "mapping_method", "status"],
                   [dict(poi_id="/".join(c[k] for k in ("city", "district", "poi")),
                         category=c["category"], landmark_ids=" ".join(ids),
                         gld_categories=" | ".join(ids.values()), mapping_method=method,
                         status="matched" if ids else "not_found") for c, ids, method in resolved])
        print(f"Соответствия: {sum(bool(ids) for _, ids, _ in resolved)}/{len(configs)}; {args.report}")
        if args.plan_only:
            return 0
        wanted = {i for _, ids, _ in resolved for i in ids}
        if not wanted:
            # Остальные большие таблицы не нужны, если ни один POI не найден.
            return export_pois(configs, resolved, {}, args.output.resolve(), args.config, session, args.max_per_poi)
        paths = [metadata_file(args.metadata_dir, name, session, args.offline)
                 for name in ("train_clean.csv", "train.csv", "train_attribution.csv")]
        candidates = collect_candidates(*paths, wanted)
        return export_pois(configs, resolved, candidates, args.output.resolve(), args.config, session, args.max_per_poi)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (requests.RequestException, OSError, ValueError) as exc:
        print(f"Ошибка: {exc}\nПовторите команду. Готовые файлы и остатки метаданных сохранены.", file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        print("\nОстановлено. Повторите команду для продолжения.", file=sys.stderr)
        raise SystemExit(130)
