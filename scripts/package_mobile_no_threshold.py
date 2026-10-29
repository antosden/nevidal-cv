"""Отдельный FP32-выпуск без отказа по score; исходный строгий пакет сохраняется."""
from pathlib import Path
import argparse
import copy
import json
import shutil
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.mobile_export import SPLITS, verify_release, metrics
from src.small_classifier import digest, write_json


def build(result, work, output):
    old_report = json.loads((result / 'comparison.json').read_text(encoding='utf-8'))
    source = result / 'public' / old_report['release_version'] / 'fp32'
    original = verify_release(source)
    if not original['evaluation']['validation_gate_passed']:
        raise ValueError('FP32 did not pass export validation')
    if output.exists():
        raise FileExistsError(output)
    version = 'poi-small-2026-10-30-argmax-v2'
    destination = output / 'public' / version / 'fp32'
    shutil.copytree(source, destination)
    policy = {
        'version': version + '-policy-v2', 'mode': 'argmax_only',
        'confidence_rejection_enabled': False, 'score': 'max_softmax',
        'threshold': 0.0, 'comparison': '>=',
        'threshold_note': 'Compatibility only; do not use score as an acceptance condition in argmax_only mode.',
        'arithmetic': 'float64 stable softmax', 'probability_calibrated': False,
        'target_external_fpr': None, 'calibration_split': None,
        'target_verification': 'supported target AND argmax == target class_index; otherwise reject',
        'limitations': ['Every valid image receives one of the 23 known POI labels.',
                        'No confidence-based rejection for unrelated photos. Unknown targets and invalid input still rejected.']}
    write_json(destination / 'decision_policy.json', policy)
    manifest = copy.deepcopy(original)
    evaluation = copy.deepcopy(original['evaluation'])
    for split in SPLITS:
        evaluation[split] = metrics(work, split, np.load(result / f'fp32_{split}_logits.npy'), 0., original['class_count'])
        for field in ('accuracy', 'macro_f1'):
            if field in evaluation[split] and evaluation[split][field] != original['evaluation'][split][field]:
                raise ValueError('Classifier metrics unexpectedly changed')
    evaluation['decision_policy_mode'] = 'argmax_only'
    evaluation['external_fpr_definition'] = 'fraction of unrelated images accepted as ANY known label; not target-specific FPR'
    manifest.update(release_version=version, policy_version=policy['version'], evaluation=evaluation,
                    derived_from_release=original['release_version'])
    manifest['blockers'] = [b for b in original['blockers'] if b != 'Strict rejection lowers positive acceptance']
    manifest['blockers'].append('Confidence rejection disabled by user; unrelated photos always receive a known label.')
    manifest['files'] = {p.name: {'sha256': digest(p), 'bytes': p.stat().st_size}
                         for p in sorted(destination.iterdir()) if p.name != 'manifest.json'}
    write_json(destination / 'manifest.json', manifest)
    verify_release(destination)
    if digest(destination / 'model.tflite') != original['files']['model.tflite']['sha256']:
        raise ValueError('Model weights changed')
    write_json(output / 'public/release_index.json', {
        'schema_version': 1, 'release_version': version, 'status': 'integration_candidates',
        'automatic_activation': False, 'decision_policy_mode': 'argmax_only', 'excluded_variants': ['int8'],
        'variants': {'fp32': {'manifest_path': f'{version}/fp32/manifest.json',
                             'manifest_sha256': digest(destination / 'manifest.json'),
                             'model_bytes': manifest['files']['model.tflite']['bytes']}}})
    write_json(output / 'comparison.json', {'release_version': version, 'variants': {'fp32': evaluation}})
    archive = output / (version + '.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted((output / 'public').rglob('*')):
            if path.is_file():
                zipped.write(path, path.relative_to(output / 'public').as_posix())
    archive.with_suffix('.zip.sha256').write_text(digest(archive) + '  ' + archive.name + '\n', encoding='ascii')
    print(json.dumps({'archive': str(archive), 'model_unchanged': True, 'test': evaluation['test'],
                      'negative_test': evaluation['negative_test']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, default=ROOT / '.runtime/mobile_export_result_v3')
    parser.add_argument('--work', type=Path, default=ROOT / '.runtime/mobile_export_v1')
    parser.add_argument('--output', type=Path, default=ROOT / '.runtime/mobile_argmax_v2')
    args = parser.parse_args()
    build(args.result.resolve(), args.work.resolve(), args.output.resolve())
