"""Контракт GLDv2, продолжение загрузки и атрибуция без сетевых запросов."""
import csv
import hashlib
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import requests
from scripts import google_landmarks_parser as gld


class GoogleLandmarksTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = dict(city="moscow", district="arbat", poi="house", category="A house")
        self.config_path = self.write("config.csv", [self.config])
        self.image_id = "0123456789abcdef"
        self.candidate = dict(id=self.image_id, landmark_id="42", url="https://example.org/image.jpg",
                              attribution=dict(url="https://example.org/page", author="Artist", license="CC-BY-4.0", title="House"))
        self.output = self.root / "images"
        self.resolved = [(self.config, {"42": "https://commons.wikimedia.org/wiki/Category:A_house"}, "exact_category")]

    def write(self, name, rows):
        path = self.root / name
        gld.atomic_csv(path, list(rows[0]), rows)
        return path

    def session(self, content=None):
        if content is None:
            data = BytesIO()
            Image.new("RGB", (20, 30)).save(data, "JPEG")
            content = data.getvalue()
        response = Mock()
        response.status_code = 200
        response.headers = {"Content-Length": str(len(content)), "ETag": '"v1"'}
        response.iter_content.return_value = [content]
        session = Mock()
        session.get.return_value.__enter__ = Mock(return_value=response)
        session.get.return_value.__exit__ = Mock(return_value=False)
        return session

    def export(self, session):
        return gld.export_pois([self.config], self.resolved, {self.image_id: self.candidate},
                               self.output, self.config_path, session, 50)

    def test_exact_category_no_fuzzy_and_explicit_override(self):
        path = self.write("categories.csv", [dict(landmark_id="42", category="https://commons.wikimedia.org/wiki/Category:A_house")])
        self.assertEqual(list(gld.resolve_pois([self.config], path)[0][1]), ["42"])
        self.config["category"] = "Other house"
        self.assertEqual(gld.resolve_pois([self.config], path)[0][1], {})
        self.config["landmark_ids"] = "42"
        self.assertEqual(gld.resolve_pois([self.config], path)[0][2], "explicit_id")
        with self.assertRaises(ValueError):
            gld.resolve_pois([self.config, dict(self.config, poi="other")], path)

    def test_reject_unsafe_config_and_unknown_ids(self):
        path = self.write("bad.csv", [dict(self.config, district="../escape")])
        with self.assertRaises(ValueError):
            gld.load_config(path)
        path = self.write("categories.csv", [dict(landmark_id="1", category="Category:Other")])
        with self.assertRaises(ValueError):
            gld.resolve_pois([dict(self.config, landmark_ids="999")], path)

    def test_clean_join_keeps_only_target_and_preserves_attribution(self):
        clean = self.write("clean.csv", [dict(landmark_id="42", images=self.image_id)])
        train = self.write("train.csv", [dict(id=self.image_id, url=self.candidate["url"], landmark_id="42"),
                                         dict(id="1111111111111111", url="https://other", landmark_id="7")])
        attr = self.write("attr.csv", [dict(id=self.image_id, **self.candidate["attribution"])])
        result = gld.collect_candidates(clean, train, attr, {"42"})
        self.assertEqual(list(result), [self.image_id])
        self.assertEqual(result[self.image_id]["attribution"]["author"], "Artist")

    def test_resume_and_repair_changed_file(self):
        session = self.session()
        self.assertEqual(self.export(session), 0)
        rows = list(gld.read_csv(self.output / "metadata.csv", gld.METADATA))
        self.assertEqual(rows[0]["licence"], "CC-BY-4.0")
        self.assertEqual(rows[0]["label_status"], "needs_review")
        self.assertEqual(rows[0]["author"], "Artist")
        self.assertEqual(rows[0]["source"], "google_landmarks_v2")
        session.reset_mock()
        self.assertEqual(self.export(session), 0)
        session.get.assert_not_called()
        image = self.output / rows[0]["file"]
        image.write_bytes(b"broken")
        self.assertEqual(self.export(session), 0)
        session.get.assert_called_once()
        self.assertEqual(hashlib.sha256(image.read_bytes()).hexdigest(), rows[0]["sha256"])

    def test_failed_download_retries_and_cleans_part(self):
        session = self.session()
        session.get.side_effect = requests.ConnectionError("offline")
        self.assertEqual(self.export(session), 1)
        self.assertEqual(list(gld.read_csv(self.config_path, []))[0]["status"], "error")
        self.assertEqual(self.export(self.session()), 0)
        self.assertFalse(list(self.output.rglob("*.part")))
        self.assertEqual(list(gld.read_csv(self.output / "download_errors.csv", [])), [])

    def test_invalid_image_and_missing_attribution_not_accepted(self):
        self.assertEqual(self.export(self.session(b"not an image")), 1)
        self.assertFalse((self.output / "metadata.csv").exists())
        self.candidate.pop("attribution")
        session = self.session()
        self.assertEqual(self.export(session), 1)
        session.get.assert_not_called()

    def test_plan_only_uses_local_category_and_changes_no_config(self):
        self.write("train_label_to_category.csv", [dict(landmark_id="42", category="Category:A_house")])
        before = self.config_path.read_bytes()
        self.assertEqual(gld.main(["--config", str(self.config_path), "--metadata-dir", str(self.root),
                                  "--report", str(self.root / "report.csv"), "--plan-only", "--offline"]), 0)
        self.assertEqual(self.config_path.read_bytes(), before)
        self.assertFalse(self.output.exists())

    def response(self, chunks, status, headers):
        response = Mock(status_code=status, headers=headers)
        response.iter_content.side_effect = lambda *args: iter(chunks)
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        return response

    @patch.object(gld.time, "sleep")
    def test_timeout_midstream_resumes_exact_bytes(self, sleep):
        first = self.response([], 200, {"Content-Length": "6", "ETag": '"v1"'})
        def interrupted(*args):
            yield b"abc"
            raise requests.ConnectionError("read timeout")
        first.iter_content.side_effect = interrupted
        second = self.response([b"def"], 206, {"Content-Range": "bytes 3-5/6", "ETag": '"v1"'})
        session = Mock()
        session.get.side_effect = [first, second]
        path = self.root / "metadata.csv"
        gld.stream_download(session, "https://example.org/table", path, 100, resume=True, attempts=2)
        self.assertEqual(path.read_bytes(), b"abcdef")
        self.assertEqual(session.get.call_args.kwargs["headers"]["Range"], "bytes=3-")
        self.assertFalse(list(self.root.glob("*.part*")))

    def test_exhausted_attempts_persist_partial_for_next_run(self):
        first = self.response([b"abc"], 200, {"Content-Length": "6", "ETag": '"v1"'})
        session = Mock()
        session.get.return_value = first
        path = self.root / "metadata.csv"
        with self.assertRaises(requests.ConnectionError):
            gld.stream_download(session, "https://example.org/table", path, 100, resume=True)
        self.assertFalse(path.exists())
        self.assertEqual(path.with_suffix(".csv.part").read_bytes(), b"abc")
        session.get.return_value = self.response([b"def"], 206, {"Content-Range": "bytes 3-5/6", "ETag": '"v1"'})
        gld.stream_download(session, "https://example.org/table", path, 100, resume=True)
        self.assertEqual(path.read_bytes(), b"abcdef")

    def test_server_ignoring_range_replaces_partial(self):
        path = self.root / "metadata.csv"
        partial = path.with_suffix(".csv.part")
        partial.write_bytes(b"old")
        partial.with_suffix(".part.json").write_text('{"url":"https://example.org/table","validator":"old"}')
        session = Mock()
        session.get.return_value = self.response([b"new data"], 200, {"Content-Length": "8", "ETag": '"new"'})
        gld.stream_download(session, "https://example.org/table", path, 100, resume=True)
        self.assertEqual(path.read_bytes(), b"new data")

    def test_wrong_range_never_promotes_corrupted_file(self):
        path = self.root / "metadata.csv"
        partial = path.with_suffix(".csv.part")
        partial.write_bytes(b"abc")
        partial.with_suffix(".part.json").write_text('{"url":"https://example.org/table","validator":"v1"}')
        session = Mock()
        session.get.return_value = self.response([b"def"], 206, {"Content-Range": "bytes 2-4/6", "ETag": "v1"})
        with self.assertRaises(requests.ConnectionError):
            gld.stream_download(session, "https://example.org/table", path, 100, resume=True)
        self.assertFalse(path.exists())
        self.assertFalse(partial.exists())


if __name__ == "__main__":
    unittest.main()
