"""Проверка связей 2GIS с выходами существующего CV-checkpoint без ML-зависимостей."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.poi_catalog import PoiCatalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "config/poi_2gis_mapping.json")
    parser.add_argument("--model-mapping", type=Path, default=ROOT / "models/mobilenetv3_small_classifier/class_mapping.json")
    parser.add_argument("--require-sdk-verified", action="store_true")
    args = parser.parse_args()
    catalog = PoiCatalog.from_files(args.catalog, args.model_mapping)
    data = json.loads(args.catalog.read_text(encoding="utf-8"))
    rows = data["entries"]
    mapped = [r for r in rows if r["binding_status"] == "mapped"]
    for row in mapped:
        assert catalog.target_for_dgis(row["dgis_poi_id"], allow_unverified=True) == row["cv_poi_id"]
        assert catalog.decode_class_index(row["class_index"])["cv_poi_id"] == row["cv_poi_id"]
    result = dict(mapping_version=catalog.version, classes=len(rows), mapped=len(mapped),
                  blocked=[r["cv_poi_id"] for r in rows if r["binding_status"] != "mapped"],
                  sdk_verified=sum(r["sdk_validation_status"] == "verified" for r in mapped),
                  model_indices_match=True, roundtrip_checks="passed", sdk_calls_performed=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.require_sdk_verified and any(r["binding_status"] != "mapped" or r["sdk_validation_status"] != "verified" for r in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
