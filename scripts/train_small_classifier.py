"""Запуск нового MobileNetV3-Small baseline с выбором по Macro-F1."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/mobilenetv3_small_classifier.json")
    parser.add_argument("--output", type=Path, default=ROOT / "models/mobilenetv3_small_classifier")
    parser.add_argument("--weights", type=Path, default=ROOT / ".runtime/torch/checkpoints/mobilenet_v3_small-047dcff4.pth")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true", help="Продолжить оценку сохранённых checkpoint/selection/policy без обучения")
    args = parser.parse_args()
    from src.small_classifier import run, resume_evaluation
    if args.evaluate_only:
        resume_evaluation(ROOT, args.output)
    else:
        run(ROOT, args.output, args.weights, json.loads(args.config.read_text(encoding="utf-8")), args.prepare_only)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        print(f"Запуск остановлен: {exc}", file=sys.stderr)
        raise SystemExit(2)
