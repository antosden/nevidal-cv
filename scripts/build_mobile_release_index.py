"""Собрать индекс для статического хостинга после проверки обоих вариантов."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.mobile_export import verify_release
from src.small_classifier import digest, write_json


def build_index(result, selected_variants=None):
    status = json.loads((result / 'status.json').read_text(encoding='utf-8'))
    if selected_variants is None and status['status'] != 'validated_integration_candidates':
        raise ValueError('Конвертация/сравнение ещё не завершены успешно')
    selected = tuple(selected_variants) if selected_variants is not None else ('fp32', 'int8')
    if not selected or len(set(selected)) != len(selected) or set(selected) - {'fp32', 'int8'}:
        raise ValueError('Неверный список вариантов')
    report = json.loads((result / 'comparison.json').read_text(encoding='utf-8'))
    version = report['release_version']
    public = result / 'public'
    variants = {}
    for variant in selected:
        if variant not in status['completed_variants'] or variant not in report['variants']:
            raise ValueError(f'{variant}: нет завершённой проверки')
        directory = public / version / variant
        manifest = verify_release(directory)
        gate = json.loads((result / f'{variant}_validation_gate.json').read_text(encoding='utf-8'))
        if not manifest['evaluation']['validation_gate_passed'] or not gate['passed']:
            raise ValueError('Не пройдена проверка качества')
        if manifest['evaluation'] != report['variants'][variant]:
            raise ValueError('Метрики манифеста и отчёта расходятся')
        variants[variant] = {'manifest_path': f'{version}/{variant}/manifest.json',
                             'manifest_sha256': digest(directory / 'manifest.json'),
                             'model_bytes': manifest['files']['model.tflite']['bytes']}
    index = {'schema_version': 1, 'release_version': version, 'variants': variants,
             'excluded_variants': sorted({'fp32', 'int8'} - set(selected)),
             'status': 'integration_candidates', 'automatic_activation': False,
             'note': 'Device benchmark and 2GIS validation required before activation.'}
    write_json(public / 'release_index.json', index)
    return index


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result', type=Path)
    parser.add_argument('--variants', nargs='+', choices=['fp32', 'int8'], help='Явно выбрать только завершённые и прошедшие проверки варианты')
    args = parser.parse_args()
    print(json.dumps(build_index(args.result.resolve(), args.variants), indent=2))
