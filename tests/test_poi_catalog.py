"""Проверки ID без сети, ML и численного округления внешних идентификаторов."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from src.poi_catalog import PoiCatalog, PoiMappingError

ROOT = Path(__file__).resolve().parents[1]


class PoiCatalogTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "config/poi_2gis_mapping.json").read_text(encoding="utf-8"))
        self.mapping = {r["cv_poi_id"]: r["class_index"] for r in self.data["entries"]}
        self.catalog = PoiCatalog(self.data, self.mapping)

    def test_all_provided_ids_roundtrip_with_explicit_test_override(self):
        mapped = [r for r in self.data["entries"] if r["binding_status"] == "mapped"]
        self.assertEqual(len(mapped), 22)
        for row in mapped:
            cv = self.catalog.target_for_dgis(row["dgis_poi_id"], allow_unverified=True)
            self.assertEqual(cv, row["cv_poi_id"])
            self.assertEqual(self.catalog.dgis_for_cv(cv, allow_unverified=True), row["dgis_poi_id"])

    def test_pending_sdk_verification_is_not_silently_accepted(self):
        with self.assertRaises(PoiMappingError) as error:
            self.catalog.target_for_dgis("70030076661708120")
        self.assertEqual(error.exception.code, "dgis_mapping_unverified")

    def test_whole_park_cannot_become_main_entrance(self):
        with self.assertRaises(PoiMappingError) as error:
            self.catalog.target_for_dgis("4504127908473840", allow_unverified=True)
        self.assertEqual(error.exception.code, "dgis_mapping_blocked")
        row = self.catalog.decode_class_index(7)
        self.assertEqual(row["cv_poi_id"], "moscow/ostankinskiy/main_entrance")
        self.assertIsNone(row["dgis_poi_id"])

    def test_large_id_remains_exact_string_and_numbers_are_rejected(self):
        self.assertEqual(self.catalog.resolve_dgis("70030076661708120", allow_unverified=True)["class_index"], 14)
        for invalid in (70030076661708120, 7.003007666170812e16, '"70030076661708120"', "70030076661708120 "):
            with self.assertRaises(PoiMappingError):
                self.catalog.resolve_dgis(invalid, allow_unverified=True)

    def test_duplicate_external_id_and_wrong_class_are_rejected(self):
        data = deepcopy(self.data)
        data["entries"][1]["dgis_poi_id"] = data["entries"][0]["dgis_poi_id"]
        with self.assertRaisesRegex(ValueError, "Неоднозначный"):
            PoiCatalog(data, self.mapping)
        data = deepcopy(self.data)
        data["entries"][0]["class_index"] = 1
        with self.assertRaisesRegex(ValueError, "Несовпадение"):
            PoiCatalog(data, self.mapping)

    def test_verified_binding_works_without_override(self):
        data = deepcopy(self.data)
        data["entries"][0]["sdk_validation_status"] = "verified"
        catalog = PoiCatalog(data, self.mapping)
        self.assertEqual(catalog.target_for_dgis(data["entries"][0]["dgis_poi_id"]), data["entries"][0]["cv_poi_id"])

    def test_unknown_ids_and_missing_classes_fail_explicitly(self):
        with self.assertRaises(PoiMappingError):
            self.catalog.resolve_dgis("123")
        with self.assertRaises(PoiMappingError):
            self.catalog.decode_class_index(True)
        data = deepcopy(self.data)
        data["entries"].pop()
        with self.assertRaisesRegex(ValueError, "покрывает"):
            PoiCatalog(data, self.mapping)


if __name__ == "__main__":
    unittest.main()
