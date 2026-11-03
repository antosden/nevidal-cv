"""Подготовка и проверка мобильного выпуска без изменения обученной модели."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small

from src.small_classifier import read_csv, write_csv, write_json, digest, validate_splits, classification_metrics

SPLITS = ("val", "negative_val", "test", "negative_test")
MODEL_SHA = "b03050356a085ed2e5c8190a7ed312624e74efd380b10069419b992616050fb1"
PUBLIC_FILES = {"model.tflite", "class_mapping.json", "poi_2gis_mapping.json", "preprocessing.json", "decision_policy.json", "conformance.json", "manifest.json"}


def synthetic_inputs():
    """Контрольные входы без фотографий: каналы RGB и геометрический узор."""
    images = np.zeros((5, 224, 224, 3), dtype=np.uint8)
    for i in range(3):
        images[i, :, :, i] = 255
    images[3] = (124, 116, 104)
    y, x = np.indices((224, 224))
    images[4, :, :, 0] = x % 256
    images[4, :, :, 1] = y % 256
    images[4, :, :, 2] = ((x // 16 + y // 16) % 2) * 255
    return images


class MobileModel(nn.Module):
    """Публичный вход: NHWC float32 RGB 0..255. Выход: logits, без softmax."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, rgb):
        image = rgb.permute(0, 3, 1, 2).contiguous() / 255.0
        return self.model((image - self.mean) / self.std)


def load_model(directory):
    mapping = json.loads((directory / "class_mapping.json").read_text(encoding="utf-8"))
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("Некорректный class_mapping")
    if digest(directory / "model.pt") != MODEL_SHA:
        raise ValueError("Изменился исходный checkpoint; нужен отдельный выпуск и аудит")
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(mapping))
    model.load_state_dict(torch.load(directory / "model.pt", map_location="cpu", weights_only=True))
    return MobileModel(model).eval(), mapping


def calibration_rows(rows, limit=300):
    """SEED=42: чередуем классы, берём не более одного кадра группы."""
    if limit < len({r["poi_id"] for r in rows}):
        raise ValueError("Лимит меньше числа классов")
    buckets = {}
    for row in rows:
        buckets.setdefault(row["poi_id"], []).append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda r: hashlib.sha256(("42:" + r["image_id"]).encode()).hexdigest())
    result, seen = [], set()
    while len(result) < limit and any(buckets.values()):
        for poi in sorted(buckets):
            while buckets[poi]:
                row = buckets[poi].pop()
                group = row.get("split_group") or row["sha256"]
                if group not in seen:
                    seen.add(group)
                    result.append(row)
                    break
            if len(result) == limit:
                break
    if {r["poi_id"] for r in result} != set(buckets):
        raise ValueError("Калибровка не покрыла все классы")
    return result


def prepare(root, output, limit=300):
    """Используем зафиксированный кеш обучения; raw остаётся неизменным."""
    source = root / "models/mobilenetv3_small_classifier"
    if output.exists():
        raise FileExistsError(f"Нужна новая рабочая директория: {output}")
    torch.set_num_threads(4)
    model, mapping = load_model(source)
    frames = {s: read_csv(source / "splits" / f"{s}.csv") for s in ("train",) + SPLITS}
    validate_splits(frames, mapping)
    output.mkdir(parents=True)
    for name in ("model.pt", "class_mapping.json", "rejection_policy.json"):
        shutil.copyfile(source / name, output / name)
    shutil.copyfile(root / "config/poi_2gis_mapping.json", output / "poi_2gis_mapping.json")
    frames["calibration"] = calibration_rows(frames["train"], limit)
    report = {"seed": 42, "model_sha256": MODEL_SHA, "calibration_split": "train", "splits": {},
              "limitations": ["Test is a reused diagnostic benchmark, not a blind field test.",
                              "Cross-platform image decoding/resizing requires device parity checks."]}
    for split in ("calibration",) + SPLITS:
        rows = frames[split]
        images = np.lib.format.open_memmap(output / f"{split}_rgb.npy", mode="w+", dtype=np.uint8,
                                          shape=(len(rows), 224, 224, 3))
        for i, row in enumerate(rows):
            cache = root / ".runtime/small_classifier_cache_v1" / (row["sha256"] + ".png")
            with Image.open(cache) as original:
                images[i] = np.asarray(ImageOps.pad(original.convert("RGB"), (224, 224),
                    method=Image.Resampling.BILINEAR, color=(124, 116, 104)))
        images.flush()
        write_csv(output / f"{split}.csv", rows)
        if split != "calibration":
            logits = []
            with torch.inference_mode():
                for start in range(0, len(images), 16):
                    logits.append(model(torch.from_numpy(images[start:start + 16].astype(np.float32))).numpy())
            logits = np.concatenate(logits)
            np.save(output / f"{split}_reference_logits.npy", logits)
            probabilities = torch.from_numpy(logits).softmax(1).numpy()
            archived = np.load(source / f"{split}_probabilities.npy")
            error = float(np.max(np.abs(probabilities - archived)))
            if error > 2e-6 or not np.array_equal(probabilities.argmax(1), archived.argmax(1)):
                raise ValueError(f"{split}: нарушена воспроизводимость PyTorch, error={error}")
            report["splits"][split] = {"images": len(rows), "archive_probability_max_abs_error": error}
        else:
            report["splits"][split] = {"images": len(rows), "classes": len({r['poi_id'] for r in rows}),
                                      "selection": "train-only, class round-robin, one per split_group"}
        print(f"Prepared {split}: {len(rows)}", flush=True)
    write_json(output / "preparation.json", report)
    # Хешируем и входные массивы, и разметку, и контрольные ответы.
    write_json(output / "input_checksums.json", {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()})
    return report


def softmax(logits):
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Некорректные logits")
    values = np.exp(values - values.max(1, keepdims=True))
    return values / values.sum(1, keepdims=True)


def calibrate(scores, fpr=.01):
    """Порог между score, чтобы >= было переносимо в double/Dart."""
    scores = np.asarray(scores, dtype=np.float64)
    if not len(scores) or not np.isfinite(scores).all() or not 0 <= fpr < 1:
        raise ValueError("Некорректные negatives/FPR")
    allowed = math.floor(len(scores) * fpr)
    ordered = np.sort(scores)[::-1]
    excluded = ordered[allowed]
    higher = scores[scores > excluded]
    threshold = float((excluded + higher.min()) / 2) if len(higher) else float(np.nextafter(excluded, np.inf))
    assert int((scores >= threshold).sum()) <= allowed
    return threshold


def metrics(work, split, logits, threshold, classes):
    scores = softmax(logits)
    accepted = scores.max(1) >= threshold
    result = {"images": len(scores), "accepted": int(accepted.sum())}
    if split.startswith("negative"):
        result["external_fpr"] = float(accepted.mean())
    else:
        rows = read_csv(work / f"{split}.csv")
        truth = np.array([int(r["class_index"]) for r in rows])
        predicted = scores.argmax(1)
        result.update(classification_metrics(truth, predicted, classes)[0])
        result["true_accept_rate"] = float(((truth == predicted) & accepted).mean())
        result["wrong_accepts"] = int(((truth != predicted) & accepted).sum())
    return result


def quality_gate(reference, candidate, max_drop=.01):
    return all(candidate[k] >= reference[k] - max_drop for k in ("accuracy", "macro_f1"))


def target_matches(logits, target_poi_id, mapping, policy):
    """Сравнение с целью сохраняется даже при отключённом пороге уверенности."""
    values = np.asarray(logits, dtype=np.float64)
    if target_poi_id not in mapping or values.shape != (len(mapping),) or not np.isfinite(values).all():
        return False
    if int(values.argmax()) != mapping[target_poi_id]:
        return False
    mode = policy.get('mode', 'threshold')
    if mode == 'argmax_only' and policy.get('confidence_rejection_enabled') is False:
        return True
    if mode != 'threshold':
        raise ValueError('Unsupported decision policy')
    return bool(softmax(values[None]).max() >= policy['threshold'])


def verify_inputs(work):
    checksums = json.loads((work / "input_checksums.json").read_text(encoding="utf-8"))
    for name, expected in checksums.items():
        if Path(name).name != name or digest(work / name) != expected:
            raise ValueError(f"Повреждён вход: {name}")


def package(work, destination, model_path, version, variant, policy, tensor_contract, evaluation, conformance):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}", version):
        raise ValueError("Некорректная версия выпуска")
    if destination.exists():
        raise FileExistsError(f"Выпуск уже существует: {destination}")
    # FlatBuffer identifier, нельзя выдать checkpoint за TFLite.
    with model_path.open("rb") as stream:
        if stream.read(8)[4:8] != b"TFL3":
            raise ValueError("Файл не является TFLite FlatBuffer")
    destination.mkdir(parents=True)
    shutil.copyfile(model_path, destination / "model.tflite")
    for name in ("class_mapping.json", "poi_2gis_mapping.json"):
        shutil.copyfile(work / name, destination / name)
    write_json(destination / "decision_policy.json", policy)
    write_json(destination / "conformance.json", conformance)
    write_json(destination / "preprocessing.json", {
        "version": "rgb-letterbox224-v1", "input": tensor_contract["input"], "output": tensor_contract["output"],
        "color": "RGB", "layout": "NHWC", "pixel_range": [0, 255],
        "pixel_range_semantics": "logical RGB before optional I/O quantization",
        "steps": ["first frame; apply EXIF orientation; convert RGB",
                  "reference decoder: Pillow JPEG draft RGB 640x640",
                  "thumbnail within 320x320 LANCZOS, no upscale",
                  "ImageOps.pad to 224x224 BILINEAR, centered, fill RGB(124,116,104)",
                  "float32 I/O: cast RGB to float32; integer I/O: quantize RGB with input scale/zero_point",
                  "DO NOT apply ImageNet normalization in app (embedded in model)"],
        "io_quantization": {"integer_input": "q=clip(round_half_away_from_zero(rgb/scale+zero_point), dtype_min, dtype_max)",
                            "integer_output": "logit=(q-zero_point)*scale; dequantize BEFORE softmax",
                            "float32_io": "no quantization transform"},
        "embedded_normalization": {"divide": 255, "mean": [.485, .456, .406], "std": [.229, .224, .225]},
        "output_semantics": "23 logits after optional dequantization; stable float64 softmax exactly once in app",
        "device_parity_status": "pending; decoder and resampling may differ from Pillow"})
    manifest = {"schema_version": 1, "release_version": version, "variant": variant,
                "model_version": version,
                "preprocessing_version": "rgb-letterbox224-v1",
                "policy_version": policy.get("version"),
                "poi_mapping_version": json.loads((work / "poi_2gis_mapping.json").read_text(encoding="utf-8")).get("mapping_version"),
                "source_checkpoint_sha256": MODEL_SHA, "class_count": 23,
                "status": "integration_candidate", "production_ready": False,
                "runtime_requirements": tensor_contract.get("runtime", {"backend": "not_recorded"}),
                "blockers": ["Android device numerical/performance checks pending", "2GIS SDK bindings pending; main entrance scope mismatch", "Strict rejection lowers positive acceptance"] +
                            (["XNNPACK failed on Linux; evaluated with builtin CPU kernels. Android backend compatibility pending."]
                             if tensor_contract.get("runtime", {}).get("backend") == "builtin_without_default_delegates" else []),
                "evaluation": evaluation,
                "files": {p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
                          for p in sorted(destination.iterdir())}}
    write_json(destination / "manifest.json", manifest)
    verify_release(destination)


def verify_release(directory):
    if {p.name for p in directory.iterdir()} != PUBLIC_FILES:
        raise ValueError("В публичном пакете лишние или отсутствующие файлы")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if set(manifest["files"]) != PUBLIC_FILES - {"manifest.json"}:
        raise ValueError("Неверный список файлов манифеста")
    for name, info in manifest["files"].items():
        path = directory / name
        if path.is_symlink() or digest(path) != info["sha256"] or path.stat().st_size != info["bytes"]:
            raise ValueError(f"Проверка хеша не пройдена: {name}")
    mapping = json.loads((directory / "class_mapping.json").read_text(encoding="utf-8"))
    catalog = json.loads((directory / "poi_2gis_mapping.json").read_text(encoding="utf-8"))
    if sorted(mapping.values()) != list(range(manifest["class_count"])):
        raise ValueError("Неверный порядок классов")
    if {e["cv_poi_id"]: e["class_index"] for e in catalog["entries"]} != mapping:
        raise ValueError("Каталог 2GIS расходится с классами модели")
    return manifest
