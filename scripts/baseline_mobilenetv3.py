"""CLI нового baseline и локальной верификации одного POI, без API."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extra-site', type=Path, help='Дополнительный site-packages после пакетов текущего Python')
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', help='Эмбеддинги, галерея, val-пороги и test-оценка')
    run.add_argument('--config', type=Path, default=ROOT / 'config/baseline_mobilenetv3.json')
    run.add_argument('--output', type=Path, default=ROOT / 'models/baseline_mobilenetv3')
    run.add_argument('--weights-file', type=Path, default=ROOT / '.runtime/torch/checkpoints/mobilenet_v3_small-047dcff4.pth')
    run.add_argument('--download-weights', action='store_true', help='При отсутствии весов скачать исходный ImageNet checkpoint (~10 MB)')
    verify = commands.add_parser('verify', help='Проверить фото относительно одного полного poi_id')
    verify.add_argument('--artifacts', type=Path, default=ROOT / 'models/baseline_mobilenetv3')
    verify.add_argument('--image', type=Path, required=True)
    verify.add_argument('--target-poi-id', required=True)
    args = parser.parse_args()
    if args.extra_site:
        sys.path.append(str(args.extra_site.resolve()))
    sys.path.insert(0, str(ROOT))
    cache = ROOT/'.runtime/matplotlib'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache))
    from src.poi_baseline import run_baseline, POIVerifier
    if args.command == 'run':
        if not args.weights_file.exists() and args.download_weights:
            import torch
            from torchvision.models import MobileNet_V3_Small_Weights
            args.weights_file.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(MobileNet_V3_Small_Weights.IMAGENET1K_V1.url,
                                          str(args.weights_file), hash_prefix='047dcff4', progress=True)
        if not args.weights_file.exists():
            parser.error('Нет исходных весов. Передайте --weights-file или --download-weights при доступной сети.')
        metadata, policy, metrics = run_baseline(ROOT, args.output, args.weights_file, args.config)
        print(json.dumps({'gallery_size':metadata['gallery_size'], 'policy':policy, 'metrics':metrics}, ensure_ascii=False, indent=2))
    else:
        metadata=json.loads((args.artifacts/'metadata.json').read_text(encoding='utf-8'))
        if metadata.get('schema_version')==4:
            from src.poi_geometry import GeometricPOIVerifier
            verifier=GeometricPOIVerifier(args.artifacts)
        elif metadata.get('schema_version')==3:
            from src.poi_metric import CalibratedPOIVerifier
            verifier=CalibratedPOIVerifier(args.artifacts)
        elif metadata.get('schema_version')==2:
            from src.poi_finetune import FinetunedPOIVerifier
            verifier=FinetunedPOIVerifier(args.artifacts)
        else:
            verifier=POIVerifier(args.artifacts)
        result = verifier.verify(args.image, args.target_poi_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
