"""Подготовка закрытых входов для локальной Linux-конвертации."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.mobile_export import prepare

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime/mobile_export_v1")
    parser.add_argument("--calibration-limit", type=int, default=300)
    args = parser.parse_args()
    print(json.dumps(prepare(ROOT, args.output.resolve(), args.calibration_limit), indent=2))
