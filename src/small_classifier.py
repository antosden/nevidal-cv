"""Новый supervised baseline: MobileNetV3-Small, выбор эпохи по val Macro-F1.

Без зависимости от pandas: текущий Windows runtime не загружает его DLL.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

WEIGHTS_SHA256 = "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def safe_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Путь вне проекта: {relative}")
    return path


def validate_splits(frames, mapping):
    for name, rows in frames.items():
        if not rows or any(r["label_status"] != "verified" for r in rows):
            raise ValueError(f"{name}: нет проверенной разметки")
        if len({r["image_id"] for r in rows}) != len(rows):
            raise ValueError(f"{name}: повтор image_id")
        if not name.startswith("negative"):
            if {r["poi_id"] for r in rows} != set(mapping):
                raise ValueError(f"{name}: не все классы представлены")
            if any(int(r["class_index"]) != mapping[r["poi_id"]] for r in rows):
                raise ValueError(f"{name}: неверный class_index")
    names = list(frames)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for field in ("image_id", "sha256", "duplicate_group", "split_group", "series_id", "commons_file_id"):
                first = {r.get(field, "") for r in frames[a]} - {""}
                second = {r.get(field, "") for r in frames[b]} - {""}
                if first & second:
                    raise ValueError(f"Утечка {field}: {a}/{b}")


def prepare_snapshot(root, output):
    """Не используем устаревшие processed-копии удалённых пользователем raw-файлов."""
    annotations = root / "data/annotations"
    mapping = json.loads((annotations / "class_mapping.json").read_text(encoding="utf-8"))
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("Нужен фиксированный непрерывный class_mapping")
    manual_config = read_csv(root / "config/manual_pois.csv")
    manual_folders = {r["folder"]: r for r in manual_config}
    frames = {s: read_csv(annotations / f"{s}.csv") for s in ("train", "val", "test", "negative_val", "negative_test")}
    review_path = root / "config/small_classifier_content_reviews.csv"
    reviews = {r["image_id"]: r for r in read_csv(review_path)} if review_path.exists() else {}
    label_review_path = root / "config/small_classifier_label_reviews.csv"
    label_reviews = {r["image_id"]: r for r in read_csv(label_review_path)} if label_review_path.exists() else {}
    applied_labels = []
    known = {r["file_path"] for rows in frames.values() for r in rows}
    extra = {}
    for path in sorted((root / "data/raw/manual").rglob("*")):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".mpo", ".webp"}:
            if path.relative_to(root).as_posix() not in known:
                extra.setdefault(digest(path), []).append(path)
    changes, excluded, edited = [], [], []
    for name, rows in frames.items():
        kept = []
        for original in rows:
            row = original.copy()
            correction = label_reviews.get(row["image_id"])
            if correction:
                if (correction["sha256"] != row["sha256"] or correction["label_status"] != "verified" or
                        correction["poi_id"] not in mapping):
                    raise ValueError(f"Несогласованная правка метки: {row['image_id']}")
                new_poi = correction["poi_id"]
                city, district, slug = new_poi.split("/")
                row.update(poi_id=new_poi, class_index=str(mapping[new_poi]), city=city, district=district,
                           poi_slug=slug, reviewer=correction["reviewer"], review_note=correction["review_note"])
                applied_labels.append(correction)
            source = safe_path(root, row["file_path"])
            if not source.is_file():
                moved = extra.get(row["sha256"], []) if row.get("source") == "manual" else []
                if len(moved) == 1:
                    source = moved[0]
                    folder = source.relative_to(root / "data/raw/manual").parts[0]
                    config = manual_folders.get(folder)
                    if not config:
                        raise ValueError(f"Нет manual mapping для {folder}")
                    new_poi = "/".join(config[k] for k in ("city", "district", "poi"))
                    if new_poi not in mapping:
                        raise ValueError(f"Новый класс требует пересборки EDA: {new_poi}")
                    changes.append(dict(image_id=row["image_id"], old_path=row["file_path"],
                                        new_path=source.relative_to(root).as_posix(), old_poi=row["poi_id"],
                                        new_poi=new_poi, reason="exact_sha256_manual_move", split=name))
                    row.update(file_path=source.relative_to(root).as_posix(), poi_id=new_poi,
                               class_index=str(mapping[new_poi]), author=config["author"], licence=config["licence"],
                               city=config["city"], district=config["district"], poi_slug=config["poi"])
                else:
                    excluded.append(dict(image_id=row["image_id"], split=name, file_path=row["file_path"],
                                         reason="raw_missing"))
                    continue
            current_sha = digest(source)
            if current_sha != row["sha256"]:
                review = reviews.get(row["image_id"], {})
                if (review.get("sha256"), review.get("poi_id"), review.get("label_status")) != (current_sha, row["poi_id"], "verified"):
                    raise ValueError(f"Изменилось содержимое {row['image_id']}; нужна повторная EDA/разметка")
                with Image.open(source) as original:
                    image = ImageOps.exif_transpose(original).convert("RGB")
                    gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
                    basis = np.cos(np.pi * (2 * np.arange(32) + 1)[None, :] * np.arange(32)[:, None] / 64) * np.sqrt(2 / 32)
                    basis[0] /= np.sqrt(2)
                    low = (basis @ gray @ basis.T)[:8, :8].ravel()[1:]
                    value = 0
                    for bit in low > np.median(low):
                        value = (value << 1) | int(bit)
                    phash = f"{value:016x}"
                    row.update(width=str(image.width), height=str(image.height), aspect_ratio=str(image.width / image.height))
                edited.append(dict(image_id=row["image_id"], old_sha256=row["sha256"], sha256=current_sha,
                                   poi_id=row["poi_id"], split=name, reviewer=review["reviewer"], note=review["note"]))
                row.update(sha256=current_sha, perceptual_hash=phash, file_size=str(source.stat().st_size))
            row["processed_path"] = ""
            row["snapshot_mtime_ns"] = str(source.stat().st_mtime_ns)
            row["snapshot_file_size"] = str(source.stat().st_size)
            kept.append(row)
        frames[name] = kept
    validate_splits(frames, mapping)
    # После обрезки проверяем также новые pHash-соседства, сохраняя старые группы.
    edited_ids = {r["image_id"] for r in edited}
    flat = [(name, row) for name, rows in frames.items() for row in rows]
    for name, row in flat:
        if row["image_id"] not in edited_ids:
            continue
        for other_name, other in flat:
            if name != other_name and other.get("perceptual_hash"):
                distance = (int(row["perceptual_hash"], 16) ^ int(other["perceptual_hash"], 16)).bit_count()
                if distance <= 6:
                    raise ValueError(f"Новый near-duplicate после правки: {row['image_id']}/{other['image_id']}, distance={distance}")
    snapshot = output / "splits"
    snapshot.mkdir(parents=True, exist_ok=True)
    for name, rows in frames.items():
        write_csv(snapshot / f"{name}.csv", rows)
    write_json(output / "class_mapping.json", mapping)
    write_csv(output / "data_changes.csv", changes, ["image_id", "old_path", "new_path", "old_poi", "new_poi", "reason", "split"])
    write_csv(output / "data_exclusions.csv", excluded, ["image_id", "split", "file_path", "reason"])
    write_csv(output / "content_reviews.csv", edited,
              ["image_id", "old_sha256", "sha256", "poi_id", "split", "reviewer", "note"])
    if applied_labels:
        write_csv(output / "label_reviews.csv", applied_labels)
    audit = {"counts": {k: len(v) for k, v in frames.items()}, "relocated": changes, "excluded": excluded,
             "content_reviews": edited, "label_reviews": applied_labels,
             "sources": {name: {source: sum(r["source"] == source for r in rows)
                                for source in sorted({r["source"] for r in rows})} for name, rows in frames.items()},
             "input_csv_sha256": {s: digest(annotations / f"{s}.csv") for s in frames},
             "split_snapshot_sha256": {s: digest(snapshot / f"{s}.csv") for s in frames},
             "group_leakage_checks": "passed", "raw_sha256_checks": "passed",
             "global_annotations_modified": False}
    write_json(output / "data_audit.json", audit)
    return frames, mapping, audit


def classification_metrics(truth, prediction, classes):
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (np.asarray(truth, dtype=int), np.asarray(prediction, dtype=int)), 1)
    tp = np.diag(matrix)
    fp = matrix.sum(0) - tp
    fn = matrix.sum(1) - tp
    tn = matrix.sum() - tp - fp - fn
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(classes), where=(2 * tp + fp + fn) > 0)
    result = {"accuracy": float(tp.sum() / max(1, matrix.sum())), "macro_f1": float(f1.mean()),
              "images": int(matrix.sum()), "wrong_predictions": int(fp.sum()), "false_positive_count": int(fp.sum()),
              "macro_one_vs_rest_fpr": float(np.divide(fp, fp + tn, out=np.zeros(classes), where=fp + tn > 0).mean())}
    return result, matrix, [dict(class_index=i, support=int(matrix[i].sum()), tp=int(tp[i]), fp=int(fp[i]),
                               fn=int(fn[i]), tn=int(tn[i]), f1=float(f1[i]),
                               fpr=float(fp[i] / (fp[i] + tn[i])) if fp[i] + tn[i] else 0.) for i in range(classes)]


def rejection_threshold(negative_scores, target_fpr):
    scores = np.asarray(negative_scores, dtype=np.float64)
    if not len(scores) or not 0 <= target_fpr < 1:
        raise ValueError("Нужны negatives и target_fpr в [0,1)")
    allowed = int(math.floor(target_fpr * len(scores)))
    # Точное правило >=: исключаем граничные совпадения при одинаковых score.
    threshold = float(np.nextafter(np.sort(scores)[::-1][allowed], np.inf))
    assert int((scores >= threshold).sum()) <= allowed
    return threshold


def letterbox(image):
    return ImageOps.pad(image, (224, 224), method=Image.Resampling.BILINEAR, color=(124, 116, 104))


def transform(training):
    ops = [transforms.Lambda(letterbox)]
    if training:
        ops += [transforms.RandomAffine(5, translate=(.03, .03), scale=(.95, 1.05), fill=(124, 116, 104)),
                transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.1)]
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])


class Images(torch.utils.data.Dataset):
    def __init__(self, root, rows, cache, training=False):
        self.rows, self.images, self.transform = rows, [], transform(training)
        cache.mkdir(parents=True, exist_ok=True)
        for row in rows:
            source = safe_path(root, row["file_path"])
            if (source.stat().st_size != int(row["snapshot_file_size"]) or
                    source.stat().st_mtime_ns != int(row["snapshot_mtime_ns"])):
                raise ValueError(f"Фото изменилось во время запуска: {row['image_id']}")
            path = cache / (row["sha256"] + ".png")
            if not path.exists():
                with Image.open(source) as original:
                    original.seek(0)
                    original.draft("RGB", (640, 640))
                    image = ImageOps.exif_transpose(original).convert("RGB")
                    image.thumbnail((320, 320), Image.Resampling.LANCZOS)
                    image.save(path, format="PNG")
            with Image.open(path) as image:
                self.images.append(image.convert("RGB").copy())

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.transform(self.images[i]), int(self.rows[i].get("class_index", -1) or -1)


def predict(model, dataset, batch_size):
    model.eval()
    scores = []
    with torch.inference_mode():
        for images, _ in torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            scores.append(model(images).softmax(dim=1).numpy())
    return np.concatenate(scores).astype(np.float64)


def run(root, output, weights, config, prepare_only=False):
    if config["seed"] != 42:
        raise ValueError("Для baseline требуется SEED=42")
    if not prepare_only and (not weights.exists() or digest(weights) != WEIGHTS_SHA256):
        raise ValueError("Нужен проверенный mobilenet_v3_small-047dcff4.pth (ImageNet V1)")
    if output.exists() and any(p.name != "config.json" for p in output.iterdir()):
        raise FileExistsError(f"Нужен новый каталог запуска: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    report = root / "reports" / output.name
    if not prepare_only:
        report.mkdir(parents=True, exist_ok=True)
        write_json(report / "run_status.json", dict(status="preparing_data", training_started=False, metrics_available=False))
    frames, mapping, audit = prepare_snapshot(root, output)
    print(json.dumps(audit["counts"]), flush=True)
    if prepare_only:
        return audit
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    model = mobilenet_v3_small(weights=None)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(mapping))
    model.features.requires_grad_(False)
    blocks = list(model.features.children())[-config["unfreeze_last_blocks"]:]
    for block in blocks:
        block.requires_grad_(True)
    for module in model.features.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.requires_grad_(False)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.features.parameters() if p.requires_grad], "lr": config["encoder_lr"]},
        {"params": model.classifier.parameters(), "lr": config["head_lr"]}], weight_decay=config["weight_decay"])
    cache = root / ".runtime/small_classifier_cache_v1"
    print("Подготовка train/val; test пока не используется для оценки", flush=True)
    train = Images(root, frames["train"], cache, True)
    val = Images(root, frames["val"], cache)
    labels = np.array([int(r["class_index"]) for r in frames["train"]])
    counts = np.bincount(labels, minlength=len(mapping))
    class_weights = 1 / np.sqrt(counts.astype(np.float64))
    loss_function = nn.CrossEntropyLoss(weight=torch.tensor(class_weights / class_weights.mean(), dtype=torch.float32),
                                        label_smoothing=.05)
    loader = torch.utils.data.DataLoader(train, batch_size=config["batch_size"], shuffle=True, num_workers=0,
                                         generator=torch.Generator().manual_seed(config["seed"]))
    truth = [int(r["class_index"]) for r in frames["val"]]
    history, best, stale = [], (-1., -1.), 0
    write_json(report / "run_status.json", dict(status="training", training_started=True, metrics_available=False))
    for epoch in range(1, config["epochs"] + 1):
        started = time.monotonic()
        model.train()
        for module in model.features.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        for block in blocks:
            block.requires_grad_(epoch > config["warmup_epochs"])
        for module in model.features.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.requires_grad_(False)
        total_loss = 0.
        for images, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images), target)
            if not torch.isfinite(loss):
                raise ValueError("Неконечный loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total_loss += float(loss.detach()) * len(target)
        scores = predict(model, val, config["batch_size"])
        metrics, _, _ = classification_metrics(truth, scores.argmax(1), len(mapping))
        record = dict(epoch=epoch, loss=total_loss / len(train), seconds=time.monotonic() - started, **metrics)
        history.append(record)
        write_csv(output / "training_history.csv", history)
        print(json.dumps(record), flush=True)
        key = (metrics["macro_f1"], metrics["accuracy"])
        if key > best:
            best, stale = key, 0
            torch.save(dict(state_dict=model.state_dict(), epoch=epoch), output / "best_checkpoint.pt")
        else:
            stale += 1
        if epoch > config["warmup_epochs"] and stale >= config["patience"]:
            break
    checkpoint = torch.load(output / "best_checkpoint.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    # Классификатор фиксируется до калибровки отказа и открытия test.
    write_json(output / "selection.json", dict(best_epoch=checkpoint["epoch"], objective="val_macro_f1_then_accuracy",
                                                test_used_for_selection=False))
    negative_val = Images(root, frames["negative_val"], cache)
    negative_val_scores = predict(model, negative_val, config["batch_size"])
    threshold = rejection_threshold(negative_val_scores.max(1), config["target_external_fpr"])
    write_json(output / "rejection_policy.json", dict(score="max_softmax", threshold=threshold,
               comparison=">=", calibration_split="negative_val", target_external_fpr=config["target_external_fpr"],
               probability_calibrated=False))
    return finish_evaluation(root, output, model, frames, mapping, audit, config, checkpoint, threshold, negative_val_scores)


def resume_evaluation(root, output):
    """Возобновить только оценку, сохраняя выбранную эпоху и уже заданный порог."""
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    mapping = json.loads((output / "class_mapping.json").read_text(encoding="utf-8"))
    audit = json.loads((output / "data_audit.json").read_text(encoding="utf-8"))
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    policy = json.loads((output / "rejection_policy.json").read_text(encoding="utf-8"))
    frames = {}
    for name, expected in audit["split_snapshot_sha256"].items():
        path = output / "splits" / f"{name}.csv"
        if digest(path) != expected:
            raise ValueError(f"Изменился снимок разметки: {name}")
        frames[name] = read_csv(path)
    validate_splits(frames, mapping)
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(mapping))
    checkpoint = torch.load(output / "best_checkpoint.pt", map_location="cpu", weights_only=True)
    if checkpoint["epoch"] != selection["best_epoch"] or policy["calibration_split"] != "negative_val":
        raise ValueError("Несогласованные checkpoint, selection или policy")
    model.load_state_dict(checkpoint["state_dict"])
    return finish_evaluation(root, output, model, frames, mapping, audit, config, checkpoint, policy["threshold"])


def finish_evaluation(root, output, model, frames, mapping, audit, config, checkpoint, threshold, negative_val_scores=None):
    cache = root / ".runtime/small_classifier_cache_v1"
    report = root / "reports" / output.name
    results = {}
    names = sorted(mapping, key=mapping.get)
    print("Эпоха и порог зафиксированы; оценка test", flush=True)
    for split in ("val", "test", "negative_val", "negative_test"):
        if split == "negative_val" and negative_val_scores is not None:
            scores = negative_val_scores
        else:
            scores = predict(model, Images(root, frames[split], cache), config["batch_size"])
        predicted, confidence = scores.argmax(1), scores.max(1)
        accepted = confidence >= threshold
        rows = frames[split]
        np.save(output / f"{split}_probabilities.npy", scores, allow_pickle=False)
        write_csv(output / f"{split}_predictions.csv", [dict(image_id=r["image_id"], file_path=r["file_path"],
                    true_poi=r["poi_id"], predicted_poi=names[int(p)], score=float(s), accepted=bool(a),
                    source=r["source"], author=r["author"], licence=r["licence"])
                    for r, p, s, a in zip(rows, predicted, confidence, accepted)])
        if split.startswith("negative"):
            results[split] = dict(images=len(rows), false_positives=int(accepted.sum()),
                                 false_positive_rate=float(accepted.mean()), definition="any_known_POI_accepted_on_external_negative")
        else:
            truth = np.array([int(r["class_index"]) for r in rows])
            summary, matrix, per_poi = classification_metrics(truth, predicted, len(mapping))
            summary.update(accepted_images=int(accepted.sum()),
                           correct_accept_rate=float(((truth == predicted) & accepted).mean()),
                           wrong_accept_count=int(((truth != predicted) & accepted).sum()))
            results[split] = summary
            write_csv(output / f"{split}_per_poi.csv", [dict(poi_id=names[r["class_index"]], **r) for r in per_poi])
            write_csv(output / f"{split}_confusion.csv", [dict(true_poi=names[i], **dict(zip(names, map(int, line))))
                                                         for i, line in enumerate(matrix)])
    write_json(output / "metrics.json", results)
    torch.save(model.state_dict(), output / "model.pt")
    import platform
    import PIL
    import torchvision
    metadata = dict(architecture="mobilenet_v3_small", task="supervised_POI_classification_with_optional_rejection",
                    seed=config["seed"], best_epoch=checkpoint["epoch"], device="cpu", weights_sha256=WEIGHTS_SHA256,
                    preprocessing="EXIF RGB JPEG draft640 thumbnail320 letterbox224 ImageNet normalization",
                    selection="validation Macro-F1 then Accuracy", torch_version=torch.__version__,
                    train_sources=audit["sources"]["train"], class_count=len(mapping),
                    versions=dict(python=platform.python_version(), torch=torch.__version__, torchvision=torchvision.__version__,
                                  numpy=np.__version__, Pillow=PIL.__version__),
                    weights_url="https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
                    source_sha256={"src/small_classifier.py": digest(root / "src/small_classifier.py"),
                                   "scripts/train_small_classifier.py": digest(root / "scripts/train_small_classifier.py")},
                    training_source_snapshot="source/training_small_classifier.py" if (output / "source/training_small_classifier.py").exists() else None,
                    negative_images_used_for_gradients=False,
                    limitations=["Existing test is diagnostic, not a new blind field test.",
                                 "External negatives may contain target POIs in background; audit apparent false positives.",
                                 "Max-softmax is a score, not a calibrated probability."])
    metadata["artifacts_sha256"] = {p.relative_to(output).as_posix(): digest(p) for p in output.rglob("*")
                                  if p.is_file() and p != output / "metadata.json"}
    write_json(output / "metadata.json", metadata)
    report.mkdir(parents=True, exist_ok=True)
    text = ["# MobileNetV3-Small: новый baseline", "",
            f"Выбрана эпоха {checkpoint['epoch']} по val Macro-F1; веса обучались только на train.",
            f"Train: {audit['counts']['train']}; источники: {audit['sources']['train']}.",
            "", "| Выборка | Accuracy | Macro-F1 | FP | External FPR |",
            "|---|---:|---:|---:|---:|"]
    for name, metric in results.items():
        acc = f"{metric['accuracy']:.2%}" if "accuracy" in metric else "—"
        f1 = f"{metric['macro_f1']:.2%}" if "macro_f1" in metric else "—"
        fpr = f"{metric['false_positive_rate']:.2%}" if "false_positive_rate" in metric else "—"
        text.append(f"| {name} | {acc} | {f1} | {metric.get('false_positives', metric.get('false_positive_count'))} | {fpr} |")
    text += ["", "Accuracy/Macro-F1 рассчитаны до отказа на всех положительных изображениях.",
             "FP положительных классов — неверное назначение POI; подробные one-vs-rest FP/FPR в per_poi.csv.",
             "External FPR — доля внешних negatives, принятых за любой известный POI после порога.",
             f"Порог {threshold:.6f} выбран только на negative_val с целевым FPR {config['target_external_fpr']:.1%}.",
             "", "Наблюдение → интерпретация → действие:",
             "- Несбалансированные классы и малые test-подвыборки → метрики отдельных POI нестабильны → собрать новые независимые ракурсы.",
             "- Старый test использовался в прошлых экспериментах → оценка диагностическая → собрать закрытый полевой test.",
             "- На negatives могут присутствовать целевые объекты → FP требует визуальной проверки → просмотреть принятые негативы.",
             "- Softmax может быть уверенным на чужих объектах → порог не гарантирует отказ на новых типах → расширять независимые negatives."]
    (report / "summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    write_json(report / "metrics.json", results)
    write_json(report / "run_status.json", dict(status="complete", training_started=True, metrics_available=True,
                                                best_epoch=checkpoint["epoch"]))
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    return results
