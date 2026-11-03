"""Проверка содержимого и контрольных сумм готового публичного пакета."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.mobile_export import verify_release

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    manifest = verify_release(args.directory.resolve())
    print(json.dumps({"hash_checks": "passed", "version": manifest["release_version"],
                      "status": manifest["status"], "production_ready": manifest["production_ready"]}, indent=2))
