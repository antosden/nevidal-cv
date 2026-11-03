"""Независимая проверка артефактов и пересчёт метрик из сохранённых score."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def check(directory):
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    for relative, expected in metadata["artifacts_sha256"].items():
        path = (directory / relative).resolve()
        assert path.is_relative_to(directory.resolve())
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, relative
    mapping = json.loads((directory / "class_mapping.json").read_text(encoding="utf-8"))
    names = sorted(mapping, key=mapping.get)
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    policy = json.loads((directory / "rejection_policy.json").read_text(encoding="utf-8"))
    selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
    history = read_csv(directory / "training_history.csv")
    best = max(history, key=lambda row: (float(row["macro_f1"]), float(row["accuracy"])))
    assert int(best["epoch"]) == metadata["best_epoch"] == selection["best_epoch"]
    assert selection["test_used_for_selection"] is False
    frames = {s: read_csv(directory / "splits" / f"{s}.csv")
              for s in ("train", "val", "test", "negative_val", "negative_test")}
    splits = list(frames)
    for i, first in enumerate(splits):
        for second in splits[i + 1:]:
            for column in ("image_id", "sha256", "split_group", "duplicate_group", "series_id", "commons_file_id"):
                assert not (({r.get(column, "") for r in frames[first]} - {""}) &
                            ({r.get(column, "") for r in frames[second]} - {""})), (first, second, column)
    for split in ("val", "test", "negative_val", "negative_test"):
        rows = frames[split]
        assert all(r["label_status"] == "verified" for r in rows)
        scores = np.load(directory / f"{split}_probabilities.npy", allow_pickle=False)
        saved = read_csv(directory / f"{split}_predictions.csv")
        assert scores.shape == (len(rows), len(mapping))
        assert np.isfinite(scores).all() and np.allclose(scores.sum(1), 1., atol=1e-5)
        predictions = scores.argmax(1)
        accepted = scores.max(1) >= policy["threshold"]
        for i, (row, prediction) in enumerate(zip(rows, saved)):
            assert row["image_id"] == prediction["image_id"]
            assert prediction["predicted_poi"] == names[int(predictions[i])]
            assert (prediction["accepted"] == "True") == bool(accepted[i])
            for key in ("source", "author", "licence"):
                assert row[key] == prediction[key]
        if split.startswith("negative"):
            assert metrics[split]["false_positives"] == int(accepted.sum())
            assert np.isclose(metrics[split]["false_positive_rate"], accepted.mean())
        else:
            truth = np.array([mapping[r["poi_id"]] for r in rows])
            matrix = np.zeros((len(mapping), len(mapping)), dtype=int)
            np.add.at(matrix, (truth, predictions), 1)
            tp = matrix.diagonal()
            denominator = matrix.sum(0) + matrix.sum(1)
            f1 = np.divide(2 * tp, denominator, out=np.zeros(len(mapping)), where=denominator > 0).mean()
            assert np.isclose(f1, metrics[split]["macro_f1"])
            assert np.isclose((truth == predictions).mean(), metrics[split]["accuracy"])
    assert metrics["negative_val"]["false_positive_rate"] <= policy["target_external_fpr"]
    return dict(status="passed", artifact_hashes=len(metadata["artifacts_sha256"]),
                checks=["artifact_sha256", "group_leakage", "verified_labels", "epoch_selection", "prediction_alignment",
                        "source_author_licence", "accuracy_recomputed", "macro_f1_recomputed", "external_fpr_recomputed"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "models/mobilenetv3_small_classifier")
    args = parser.parse_args()
    result = check(args.artifacts)
    report = ROOT / "reports" / args.artifacts.name
    report.mkdir(parents=True, exist_ok=True)
    (report / "integrity_checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
