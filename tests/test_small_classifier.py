"""Проверки смысла метрик и защиты от утечек нового baseline."""
import unittest
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
from src.small_classifier import (classification_metrics, rejection_threshold, validate_splits,
                                  write_csv, write_json, prepare_snapshot, digest, Images)
from scripts.warm_small_classifier_cache import prepare_one


class SmallClassifierTests(unittest.TestCase):
    def test_metrics_count_false_positive_for_wrong_class(self):
        metrics, matrix, rows = classification_metrics([0, 0, 1, 1, 2], [0, 1, 1, 2, 2], 3)
        self.assertAlmostEqual(metrics["accuracy"], .6)
        self.assertEqual(metrics["false_positive_count"], 2)
        self.assertEqual(rows[0]["fp"], 0)
        self.assertEqual(rows[1]["fp"], 1)
        self.assertEqual(rows[2]["fp"], 1)
        self.assertAlmostEqual(metrics["macro_f1"], (2/3 + .5 + 2/3) / 3)
        self.assertEqual(int(matrix.sum()), 5)

    def test_rejection_ties_do_not_exceed_fpr_budget(self):
        scores = np.array([.9, .9, .8, .7], dtype=np.float32)
        threshold = rejection_threshold(scores, .25)
        self.assertEqual(int((scores.astype(np.float64) >= threshold).sum()), 0)
        self.assertGreater(threshold, float(scores.max()))

    def test_zero_fpr_threshold_accepts_no_calibration_negatives(self):
        scores = np.array([.2, .3, .4], dtype=np.float64)
        self.assertFalse((scores >= rejection_threshold(scores, 0.)).any())

    def test_cross_split_group_leakage_rejected(self):
        def row(image_id, group):
            return dict(image_id=image_id, label_status="verified", poi_id="a", class_index=0,
                        sha256=image_id, split_group=group)
        with self.assertRaisesRegex(ValueError, "split_group"):
            validate_splits({"train": [row("one", "same")], "test": [row("two", "same")]}, {"a": 0})

    def test_unverified_labels_rejected(self):
        with self.assertRaisesRegex(ValueError, "разметки"):
            validate_splits({"train": [dict(image_id="x", label_status="needs_review")]}, {"a": 0})

    def test_snapshot_tracks_move_and_deleted_raw_without_using_processed_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ann = root / "data/annotations"
            ann.mkdir(parents=True)
            (root / "config").mkdir()
            mapping = {"moscow/one/a": 0, "moscow/one/b": 1}
            write_json(ann / "class_mapping.json", mapping)
            write_csv(root / "config/manual_pois.csv", [dict(folder=x, city="moscow", district="one", poi=x,
                       author="Author", licence="Public Domain") for x in ("a", "b")])
            frames = {}
            count = 0
            for name in ("train", "val", "test", "negative_val", "negative_test"):
                rows = []
                labels = ["a", "a", "b", "b"] if name == "train" else ["a", "b"] if not name.startswith("negative") else ["outside"]
                for label in labels:
                    count += 1
                    path = root / "data/raw/manual" / label / f"{count}.jpg"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (20, 30), (count * 15, count * 7, count * 4)).save(path)
                    poi = "moscow/one/" + label
                    rows.append(dict(image_id=str(count), file_path=path.relative_to(root).as_posix(),
                         poi_id=poi, class_index=mapping.get(poi, -1), sha256=digest(path),
                         source="manual", author="Author", licence="Public Domain", label_status="verified",
                         split_group=str(count), split=name))
                frames[name] = rows
                write_csv(ann / f"{name}.csv", rows)
            first = frames["train"][0]
            source = root / first["file_path"]
            moved = root / "data/raw/manual/b/moved.jpg"
            source.rename(moved)
            deleted = root / frames["train"][2]["file_path"]
            processed = root / "old_processed.jpg"
            processed.write_bytes(deleted.read_bytes())
            deleted.unlink()
            reviewed = frames["train"][3]
            write_csv(root / "config/small_classifier_label_reviews.csv", [dict(
                image_id=reviewed["image_id"], sha256=reviewed["sha256"], poi_id="moscow/one/a",
                label_status="verified", reviewer="test", review_note="Confirmed label correction")])
            output = root / "experiment"
            output.mkdir()
            actual, _, audit = prepare_snapshot(root, output)
            self.assertEqual(len(actual["train"]), 3)
            self.assertEqual(actual["train"][0]["poi_id"], "moscow/one/b")
            self.assertEqual(actual["train"][0]["split"], "train")
            self.assertEqual(len(audit["relocated"]), 1)
            self.assertEqual(len(audit["excluded"]), 1)
            self.assertEqual(actual["train"][-1]["poi_id"], "moscow/one/a")
            self.assertEqual(len(audit["label_reviews"]), 1)
            self.assertTrue(processed.exists())
            self.assertEqual(frames["train"][0]["poi_id"], "moscow/one/a")

    def test_parallel_cache_matches_training_preprocessing_with_exif(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "image.jpg"
            image = Image.new("RGB", (640, 480), (20, 40, 70))
            exif = Image.Exif()
            exif[274] = 6
            image.save(source, exif=exif)
            row = dict(image_id="one", file_path="image.jpg", sha256=digest(source),
                       snapshot_mtime_ns=str(source.stat().st_mtime_ns), snapshot_file_size=str(source.stat().st_size))
            Images(root, [row], root / "sequential")
            parallel = root / "parallel"
            parallel.mkdir()
            self.assertTrue(prepare_one(row, root, parallel))
            self.assertEqual(digest(root / "sequential" / (row["sha256"] + ".png")),
                             digest(parallel / (row["sha256"] + ".png")))


if __name__ == "__main__":
    unittest.main()
