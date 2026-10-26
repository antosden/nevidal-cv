"""Воспроизводимая верификация POI: frozen MobileNetV3 → галерея → отказ.

Галерея использует только train; пороги выбираются только на val.
Cosine similarity не является вероятностью правильного ответа.
"""
from __future__ import annotations

import hashlib
import json
import platform
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

SEED = 42
WEIGHTS_SHA256 = '047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f'
PREPROCESSING_VERSION = 'torchvision-small-v1-rgb-exif-resize256-center224-v1'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Нужна конечная матрица эмбеддингов [N,D].')
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError('Нулевой эмбеддинг недопустим.')
    return values / norms


def safe_project_path(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Путь вне проекта: {relative_path}')
    return path


def load_splits(root: Path):
    """Проверить разметку, mapping и все известные связи утечек до вычислений."""
    annotations = root / 'data/annotations'
    mapping = json.loads((annotations / 'class_mapping.json').read_text(encoding='utf-8'))
    if len(set(mapping.values())) != len(mapping) or any(type(v) is not int for v in mapping.values()):
        raise ValueError('Невалидный class_mapping.')
    splits = {}
    required = {'image_id', 'file_path', 'poi_id', 'label_status', 'sha256',
                'duplicate_group', 'split_group', 'source', 'author', 'licence', 'class_index'}
    for name in ('train', 'val', 'test'):
        frame = pd.read_csv(annotations / f'{name}.csv', keep_default_na=False)
        if not required.issubset(frame.columns) or frame.empty:
            raise ValueError(f'{name}: пустая выборка или неполная схема.')
        if frame.image_id.duplicated().any() or not frame.label_status.eq('verified').all():
            raise ValueError(f'{name}: повторяющиеся ID или неподтверждённые метки.')
        if not frame.poi_id.isin(mapping).all():
            raise ValueError(f'{name}: POI отсутствует в mapping.')
        if not np.array_equal(frame.class_index.to_numpy(), frame.poi_id.map(mapping).to_numpy()):
            raise ValueError(f'{name}: class_index не соответствует mapping.')
        if 'split' in frame and not frame.split.eq(name).all():
            raise ValueError(f'{name}: неверное поле split.')
        if set(frame.poi_id) != set(mapping):
            raise ValueError(f'{name}: отсутствуют классы из mapping.')
        splits[name] = frame.sort_values('image_id').reset_index(drop=True)
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        for field in ('image_id', 'sha256', 'duplicate_group', 'split_group', 'series_id', 'commons_file_id'):
            if field not in splits[a] or field not in splits[b]:
                continue
            if (set(splits[a][field].astype(str)) - {''}) & (set(splits[b][field].astype(str)) - {''}):
                raise ValueError(f'Утечка {field}: {a}/{b}')
    return splits, mapping


def configure_torch(seed: int = SEED, threads: int = 4):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    # Эталонный запуск CPU: меньше зависимостей воспроизводимости от CUDA/cuDNN.
    return torch


def create_encoder(weights_path: Path):
    """Загрузить только проверенные исходные ImageNet-веса, без fallback на random."""
    import torch
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    if not weights_path.is_file() or sha256_file(weights_path) != WEIGHTS_SHA256:
        raise ValueError('Нет исходных MobileNetV3-Small V1 весов или SHA-256 не совпадает.')
    model = mobilenet_v3_small(weights=None)
    model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True), strict=True)
    model.classifier = torch.nn.Identity()
    model.eval().requires_grad_(False)
    return model, MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()


def load_rgb(path: Path) -> Image.Image:
    # MPO: явно первый кадр, как в EDA. Тот же путь декодирования при запросах.
    with Image.open(path) as image:
        image.seek(0)
        return ImageOps.exif_transpose(image).convert('RGB')


def extract_embeddings(root: Path, frame: pd.DataFrame, model, transform, batch_size: int) -> np.ndarray:
    import torch
    from tqdm.auto import tqdm
    result = []
    for start in tqdm(range(0, len(frame), batch_size), desc='MobileNetV3 embeddings'):
        batch = []
        for row in frame.iloc[start:start + batch_size].itertuples():
            relative = row.processed_path if getattr(row, 'processed_path', '') else row.file_path
            path = safe_project_path(root, relative)
            if sha256_file(path) != row.sha256:
                raise ValueError(f'Фото изменилось после EDA: {row.image_id}')
            batch.append(transform(load_rgb(path)))
        with torch.inference_mode():
            result.append(model(torch.stack(batch)).numpy())
    return l2_normalize(np.concatenate(result))


def select_gallery(train: pd.DataFrame, embeddings: np.ndarray, max_per_poi: int):
    """Детерминированный farthest-first: разнообразные ракурсы, ≤1 кадра группы/POI."""
    if max_per_poi < 1 or len(train) != len(embeddings):
        raise ValueError('Неверный размер галереи/эмбеддингов.')
    vectors = l2_normalize(embeddings)
    selected = []
    for poi in sorted(train.poi_id.unique()):
        candidates = train[train.poi_id.eq(poi)].sort_values('image_id')
        # Не выдаём копии одного оригинала за независимые эталоны.
        candidates = candidates.drop_duplicates('split_group')
        indices = candidates.index.to_numpy()
        local = vectors[indices]
        centre = l2_normalize(local.mean(axis=0, keepdims=True))[0]
        chosen = [int(np.argmax(local @ centre))]
        nearest = local @ local[chosen[0]]
        while len(chosen) < min(max_per_poi, len(local)):
            nearest[chosen] = np.inf
            next_index = int(np.argmin(nearest))
            chosen.append(next_index)
            nearest = np.minimum(nearest, local @ local[next_index])
        selected.extend(indices[chosen].tolist())
    gallery = train.loc[selected].copy().reset_index(drop=True)
    gallery['gallery_index'] = np.arange(len(gallery))
    return gallery, vectors[selected]


def poi_scores(queries: np.ndarray, gallery: np.ndarray, labels, poi_ids: list[str]):
    queries, gallery = l2_normalize(queries), l2_normalize(gallery)
    similarities = np.clip(queries @ gallery.T, -1., 1.)
    labels = np.asarray(labels)
    scores, neighbours = [], []
    for poi in poi_ids:
        positions = np.flatnonzero(labels == poi)
        if not len(positions):
            raise ValueError(f'Нет эталонов для {poi}')
        nearest = np.argmax(similarities[:, positions], axis=1)
        neighbours.append(positions[nearest])
        scores.append(similarities[np.arange(len(queries)), positions[nearest]])
    # Пороги сохраняются JSON float64, сравнение выполняем в той же точности.
    return np.stack(scores, axis=1).astype(np.float64), np.stack(neighbours, axis=1)


def score_margins(scores: np.ndarray):
    if scores.ndim != 2 or scores.shape[1] < 2 or not np.isfinite(scores).all():
        raise ValueError('Нужны конечные scores для ≥2 POI.')
    competitors = np.repeat(scores[:, None, :], scores.shape[1], axis=1)
    diagonal = np.arange(scores.shape[1])
    competitors[:, diagonal, diagonal] = -np.inf
    return scores - competitors.max(axis=2)


def acceptance_matrix(scores: np.ndarray, policy: dict):
    scores = np.asarray(scores, dtype=np.float64)
    margins = score_margins(scores)
    if policy.get('threshold_type')=='per_poi':
        score_threshold=np.asarray(policy['score_thresholds'],dtype=np.float64)
        margin_threshold=np.asarray(policy['margin_thresholds'],dtype=np.float64)
        if score_threshold.shape!=(scores.shape[1],) or margin_threshold.shape!=(scores.shape[1],):
            raise ValueError('Число per-POI порогов не соответствует scores.')
        return (scores>=score_threshold)&(margins>=margin_threshold)&(margins>0)
    # При равенстве двух лучших POI отказываем; косинус не probability.
    return ((scores >= policy['score_threshold']) & (margins >= policy['margin_threshold'])
            & (margins > 0))


def calibrate_policy(scores: np.ndarray, truths: np.ndarray, config: dict, negative_scores: np.ndarray | None = None):
    """TPR максимум при эмпирическом known-wrong FAR ≤ заданного значения на val.

    Каждое val-фото образует одну positive-пару и C-1 wrong-target пар.
    Test не используется. При передаче negative_scores разрешён только negative-val.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) != len(truths) or not len(scores):
        raise ValueError('Нет validation-запросов.')
    target_far = float(config['target_known_wrong_far'])
    target_hard_far = float(config['target_hard_wrong_far'])
    external_target = float(config.get('target_external_far', .01))
    if not 0 <= target_far < 1 or not 0 <= target_hard_far < 1 or not 0 <= external_target < 1:
        raise ValueError('FAR должен быть в [0,1).')
    margins = score_margins(scores)
    external = negative_scores is not None
    if external:
        negative_scores = np.asarray(negative_scores, dtype=np.float64)
        if not len(negative_scores) or negative_scores.shape[1] != scores.shape[1]:
            raise ValueError('Нужен непустой negative-val с тем же порядком POI.')
        negative_margins = score_margins(negative_scores)
        negative_best = negative_scores.argmax(axis=1)
        negative_top = negative_scores[np.arange(len(negative_scores)),negative_best]
        negative_gap = negative_margins[np.arange(len(negative_scores)),negative_best]
    positive = np.zeros_like(scores, dtype=bool)
    positive[np.arange(len(scores)), truths] = True
    calibration_scores = np.r_[scores.ravel(),negative_scores.ravel()] if external else scores.ravel()
    score_grid = np.unique(np.r_[np.quantile(calibration_scores, np.linspace(0, 1, config['score_grid_size'])),
                                 np.nextafter(float(calibration_scores.max()), np.inf)])
    if 'minimum_score_threshold' in config:
        floor=float(config['minimum_score_threshold'])
        score_grid=np.unique(np.r_[score_grid[score_grid>=floor],floor])
    positive_margins = margins[margins >= 0]
    if external:
        positive_margins = np.r_[positive_margins,negative_gap]
    margin_grid = np.unique(np.r_[0., np.quantile(positive_margins, np.linspace(0, 1, config['margin_grid_size']))])
    best_key = None
    rows = []
    for threshold in score_grid:
        for margin in margin_grid:
            accepted = (scores >= threshold) & (margins >= margin) & (margins > 0)
            tp = int((accepted & positive).sum())
            fp = int((accepted & ~positive).sum())
            far, tpr = fp / int((~positive).sum()), tp / len(scores)
            # При margin>0 принимается максимум одна цель. Ошибочное принятие
            # совпадает с принятием наиболее похожего неверного POI этого фото.
            hard_far = fp / len(scores)
            external_fp = int(((negative_top >= threshold)&(negative_gap >= margin)&(negative_gap > 0)).sum()) if external else 0
            external_far = external_fp/len(negative_scores) if external else 0.
            rows.append({'score_threshold':float(threshold), 'margin_threshold':float(margin),
                         'true_accept_rate':tpr, 'known_wrong_far':far,
                         'hard_wrong_far':hard_far, 'tp':tp, 'fp':fp})
            if external:
                rows[-1].update(external_any_poi_far=external_far,external_false_accepts=external_fp)
            key = (tp, -(fp+external_fp), -float(threshold), -float(margin))
            if far <= target_far and hard_far <= target_hard_far and external_far <= external_target and (best_key is None or key > best_key):
                best_key = key
                best = rows[-1].copy()
    policy = {**best, 'calibration_split':'val', 'target_known_wrong_far':target_far,
              'target_hard_wrong_far':target_hard_far,
              'negative_scope':'other_known_poi_targets_only', 'external_unknown_calibrated':False,
              'score_type':'max_gallery_cosine_not_probability', 'tie_rule':'reject',
              'decision_rule':'score >= score_threshold AND margin >= margin_threshold AND margin > 0'}
    if external:
        policy.update(negative_scope='other_known_poi_and_external_unknown',external_unknown_calibrated=True,
                      external_calibration_split='negative_val',target_external_any_poi_far=external_target)
    return policy, pd.DataFrame(rows)


def evaluate(scores: np.ndarray, frame: pd.DataFrame, poi_ids: list[str], policy: dict):
    truths = np.array([poi_ids.index(poi) for poi in frame.poi_id])
    accepted = acceptance_matrix(scores, policy)
    predicted = scores.argmax(axis=1)
    genuine = accepted[np.arange(len(scores)), truths]
    positive = np.zeros_like(accepted)
    positive[np.arange(len(scores)), truths] = True
    tp, fp = int(genuine.sum()), int((accepted & ~positive).sum())
    wrong_n = len(scores) * (len(poi_ids) - 1)
    per_poi = []
    for index, poi in enumerate(poi_ids):
        own = truths == index
        other = ~own
        per_poi.append({'poi_id':poi, 'query_count':int(own.sum()),
                        'top1_accuracy':float((predicted[own] == index).mean()),
                        'true_accept_rate':float(genuine[own].mean()),
                        'known_wrong_far':float(accepted[other, index].mean()),
                        'true_accepts':int(genuine[own].sum()),
                        'wrong_accepts':int(accepted[other, index].sum())})
    metrics = {'query_count':len(frame), 'positive_pairs':len(frame), 'known_wrong_pairs':wrong_n,
               'top1_accuracy':float((predicted == truths).mean()),
               'macro_top1_accuracy':float(np.mean([p['top1_accuracy'] for p in per_poi])),
               'true_accept_rate':tp / len(frame), 'false_reject_rate':1 - tp / len(frame),
               'known_wrong_far':fp / wrong_n, 'true_accepts':tp, 'wrong_accepts':fp,
               'hard_wrong_far':fp / len(frame),
               'accepted_pair_precision':tp / (tp + fp) if tp + fp else None,
               'macro_true_accept_rate':float(np.mean([p['true_accept_rate'] for p in per_poi])),
               'open_set_far':None, 'external_unknown_tested':False}
    rows = []
    margins = score_margins(scores)
    for n, row in enumerate(frame.itertuples()):
        for target_index, target in enumerate(poi_ids):
            yes = bool(accepted[n, target_index])
            reason = ('poi_verified' if yes else 'poi_not_matched' if margins[n, target_index] <= 0
                      else 'poi_below_threshold' if scores[n, target_index] < (policy['score_thresholds'][target_index] if policy.get('threshold_type')=='per_poi' else policy['score_threshold'])
                      else 'poi_ambiguous')
            rows.append({'image_id':row.image_id, 'source':row.source, 'true_poi_id':row.poi_id,
                         'target_poi_id':target, 'genuine':target == row.poi_id, 'accepted':yes,
                         'score':float(scores[n, target_index]), 'margin':float(margins[n, target_index]),
                         'predicted_poi_id':poi_ids[predicted[n]], 'reason':reason})
    confusion = pd.crosstab(pd.Series([poi_ids[i] for i in truths], name='true_poi_id'),
                            pd.Series([poi_ids[i] for i in predicted], name='predicted_poi_id'))
    return metrics, pd.DataFrame(per_poi), pd.DataFrame(rows), confusion.reindex(index=poi_ids, columns=poi_ids, fill_value=0)


def validate_config(config: dict):
    if config['seed'] != SEED or config['architecture'] != 'mobilenet_v3_small' or config['weights'] != 'IMAGENET1K_V1':
        raise ValueError('Baseline фиксирует SEED=42, MobileNetV3-Small / IMAGENET1K_V1.')
    for field in ('batch_size', 'threads', 'gallery_max_per_poi', 'score_grid_size', 'margin_grid_size'):
        if type(config[field]) is not int or config[field] < 1:
            raise ValueError(f'Неверный параметр {field}')


def run_baseline(root: Path, output: Path, weights_path: Path, config_path: Path):
    """Полный запуск без обучения: извлечение → галерея → val-пороги → test."""
    root, output = root.resolve(), output.resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    validate_config(config)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Артефакты уже существуют: {output}. Укажите новую папку --output.')
    splits, mapping = load_splits(root)
    torch = configure_torch(config['seed'], config['threads'])
    import torchvision
    model, transform = create_encoder(weights_path)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'splits').mkdir()
    json_write(output / 'config.json', config)
    json_write(output / 'class_mapping.json', mapping)
    shutil.copyfile(weights_path, output / 'imagenet_weights.pth')
    vectors = {}
    split_hashes = {}
    for name, frame in splits.items():
        source = root / f'data/annotations/{name}.csv'
        shutil.copyfile(source, output / 'splits' / source.name)
        split_hashes[name] = sha256_file(source)
        frame[['image_id', 'poi_id', 'source', 'author', 'licence', 'sha256']].to_csv(output / f'{name}_embedding_index.csv', index=False)
        vectors[name] = extract_embeddings(root, frame, model, transform, config['batch_size'])
        np.save(output / f'{name}_embeddings.npy', vectors[name], allow_pickle=False)
    gallery, gallery_vectors = select_gallery(splits['train'], vectors['train'], config['gallery_max_per_poi'])
    gallery.to_csv(output / 'gallery.csv', index=False)
    np.save(output / 'gallery_embeddings.npy', gallery_vectors, allow_pickle=False)
    poi_ids = sorted(mapping, key=mapping.get)
    val_scores, _ = poi_scores(vectors['val'], gallery_vectors, gallery.poi_id, poi_ids)
    truths = np.array([poi_ids.index(poi) for poi in splits['val'].poi_id])
    policy, grid = calibrate_policy(val_scores, truths, config)
    json_write(output / 'rejection_policy.json', policy)
    grid.to_csv(output / 'val_threshold_search.csv', index=False)
    metrics = {}
    for name in ('val', 'test'):
        scores, nearest = poi_scores(vectors[name], gallery_vectors, gallery.poi_id, poi_ids)
        summary, per_poi, pairs, confusion = evaluate(scores, splits[name], poi_ids, policy)
        # Пары позволяют проверить решение и ближайший эталон для любой цели.
        pairs['reference_image_id'] = gallery.image_id.to_numpy()[nearest.flatten()]
        per_poi.to_csv(output / f'{name}_per_poi.csv', index=False)
        pairs.to_csv(output / f'{name}_pairs.csv', index=False)
        confusion.to_csv(output / f'{name}_confusion.csv')
        metrics[name] = summary
        source_metrics = {}
        for source in sorted(splits[name].source.unique()):
            mask = splits[name].source.eq(source).to_numpy()
            subset = pairs[pairs.image_id.isin(splits[name].loc[mask, 'image_id'])]
            positive = subset[subset.genuine]
            negative = subset[~subset.genuine]
            source_metrics[source] = {'queries':int(mask.sum()),
                                      'true_accept_rate':float(positive.accepted.mean()),
                                      'known_wrong_far':float(negative.accepted.mean()),
                                      'hard_wrong_far':float(negative.accepted.sum()/mask.sum())}
        metrics[name]['by_source'] = source_metrics
    json_write(output / 'metrics.json', metrics)
    # Версия галереи зависит от реально выпущенных эталонов и их векторов.
    reference_version = sha256_file(output / 'gallery.csv')[:12] + '-' + sha256_file(output / 'gallery_embeddings.npy')[:12]
    metadata = {'schema_version':1, 'seed':SEED, 'architecture':config['architecture'],
                'weights':config['weights'], 'weights_sha256':WEIGHTS_SHA256,
                'weights_url':'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth',
                'embedding_layer':'features -> avgpool -> flatten -> L2', 'embedding_dim':int(gallery_vectors.shape[1]),
                'preprocessing_version':PREPROCESSING_VERSION, 'preprocessing':str(transform),
                'reference_version':reference_version, 'threshold_version':sha256_file(output / 'rejection_policy.json')[:16],
                'poi_ids':poi_ids, 'gallery_size':len(gallery), 'gallery_source':'train_only',
                'score':'max_cosine_per_poi', 'device':'cpu', 'training_performed':False,
                'split_sha256':split_hashes, 'class_mapping_sha256':sha256_file(output / 'class_mapping.json'),
                'versions':{'python':platform.python_version(), 'numpy':np.__version__, 'pandas':pd.__version__,
                            'torch':torch.__version__, 'torchvision':torchvision.__version__},
                'limitations':['No external unknown negatives; open-set FAR unmeasured.',
                               'Val/test small for several POIs; same-source split is not field validation.',
                               'ImageNet features are not fine-tuned for landmark identity.']}
    metadata['artifact_sha256'] = {p.relative_to(output).as_posix():sha256_file(p)
                                    for p in sorted(output.rglob('*')) if p.is_file()}
    json_write(output / 'metadata.json', metadata)
    write_baseline_report(root, output, metadata, policy, metrics)
    return metadata, policy, metrics


def write_baseline_report(root: Path, output: Path, metadata: dict, policy: dict, metrics: dict):
    report = root / 'reports/baseline_mobilenetv3'
    report.mkdir(parents=True, exist_ok=True)
    val, test = metrics['val'], metrics['test']
    # Снимок фактически доступных пакетов, включая приоритет двух site-packages.
    from importlib import metadata as package_metadata
    packages = {}
    for package in package_metadata.distributions():
        name = package.metadata.get('Name')
        if name:
            packages.setdefault(name.lower().replace('_', '-'), f'{name}=={package.version}')
    (report / 'environment_freeze.txt').write_text('\n'.join(packages[name] for name in sorted(packages)) + '\n', encoding='utf-8')
    lines = ['# Baseline верификации POI: MobileNetV3-Small', '',
             'Обучение весов не выполнялось. Frozen ImageNet1K V1 → avgpool 576D → L2 → train-галерея → cosine + margin.',
             f'Артефакты: `{output.relative_to(root).as_posix() if output.is_relative_to(root) else output.name}`.',
             f'POI: {len(metadata["poi_ids"])}; эталонов: {metadata["gallery_size"]}; SEED=42; CPU.', '',
             '## Проверка целевого POI',
             'score(target) = максимальный cosine с эталонами target. margin = score(target) − max score(других POI).',
             f'Подтверждаем при score ≥ {policy["score_threshold"]:.6f}, margin ≥ {policy["margin_threshold"]:.6f} и margin > 0. При равенстве лидеров — отказ.',
             'Score не является вероятностью; confidence не выдумывается. Неизвестный галерее target возвращает отказ.', '',
             '## Результаты',
             '| Выборка | Фото | Top-1 | Macro Top-1 | True accept | Known-wrong FAR | Hard-wrong FAR |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, m in [('val (калибровка)', val), ('test (после фиксации порогов)', test)]:
        lines.append(f'| {name} | {m["query_count"]} | {m["top1_accuracy"]:.2%} | {m["macro_top1_accuracy"]:.2%} | {m["true_accept_rate"]:.2%} | {m["known_wrong_far"]:.2%} | {m["hard_wrong_far"]:.2%} |')
    lines += ['', '## Наблюдение → интерпретация → действие',
              f'- Test: {test["true_accepts"]}/{test["positive_pairs"]} верных подтверждений и {test["wrong_accepts"]}/{test["known_wrong_pairs"]} ошибочных подтверждений чужой цели → измерена верификация среди известных POI → изучить test_pairs.csv и test_per_poi.csv.',
              f'- Пороги выбраны только на val: known-wrong FAR ≤{policy["target_known_wrong_far"]:.1%}, hard-wrong FAR ≤{policy["target_hard_wrong_far"]:.1%} → это настройка baseline, не гарантия на новых данных → test не использовать для повторного подбора.',
              '- Hard-wrong FAR использует для каждого фото наиболее похожий неверный POI; known-wrong FAR усредняет все C−1 неверных целей → множество лёгких пар может скрывать ошибки → оценивать оба показателя.',
              f'- На test hard-wrong FAR={test["hard_wrong_far"]:.2%}, целевой val-предел={policy["target_hard_wrong_far"]:.1%} → ограничение на val не гарантирует качество новых фото → пороги текущего выпуска не менять; собирать независимую проверку и оценивать новый выпуск отдельно.',
              '- Нет подтверждённого внешнего negative-набора → отказ на произвольных улицах, интерьерах и других достопримечательностях не измерен → собрать независимые unknown-val и unknown-test, затем выпустить новую версию порогов.',
              '- В малых классах val/test содержат единичные кадры → оценки нестабильны; wrong-target пары одного фото зависимы → добрать независимые полевые серии; не трактовать число пар как число независимых фото.',
              '- Пороговые и top-1 метрики отвечают разным вопросам → правильный лидер ещё может быть отклонён → оценивать true accept и FAR вместе.',
              '- Разделение по источникам сохранено в metrics.json → возможен сдвиг Wikimedia/manual → дополнительно провести полевой тест с мобильных устройств.', '',
              '## Воспроизводимость и ограничения',
              'Галерея: детерминированный farthest-first из train, максимум 20 эталонов на POI и один кадр каждой split_group внутри POI (точный лимит в config.json).',
              'CSV разбиения, атрибуция source/author/licence, веса, эмбеддинги, версии библиотек и SHA-256 артефактов сохранены рядом с metadata.json.',
              'Проверены отсутствие пересечений ID/SHA/групп/серий/Commons-ID и хэши всех используемых фотографий. Исходники не изменены.',
              'Побитовая воспроизводимость ожидается в зафиксированной CPU-среде; между версиями PyTorch/оборудованием возможны численные отличия.',
              'Baseline пригоден для локальных экспериментов. Готовность к production-отказу на unknown не подтверждена.', '',
              '[TorchVision: MobileNetV3-Small и преобразования весов](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html).']
    (report / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    json_write(report / 'metrics.json', metrics)
    json_write(report / 'integrity_checks.json', {'status':'passed', 'gallery_train_only':True,
               'calibration_val_only':True, 'test_not_used_for_thresholds':True,
               'all_image_sha256_checked':True, 'split_overlap_checks':'passed'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    confusion = pd.read_csv(output / 'test_confusion.csv', index_col=0)
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.imshow(confusion.to_numpy(), cmap='Blues')
    ticks = [p.split('/')[-1] for p in metadata['poi_ids']]
    ax.set_xticks(range(len(ticks)), ticks, rotation=90, fontsize=8)
    ax.set_yticks(range(len(ticks)), ticks, fontsize=8)
    ax.set(xlabel='Predicted POI (без отказа)', ylabel='True POI', title='MobileNetV3: test top-1 confusion')
    fig.tight_layout(); fig.savefig(report / 'test_confusion.png', dpi=140); plt.close(fig)
    pairs = pd.read_csv(output / 'test_pairs.csv')
    fig, ax = plt.subplots(figsize=(8, 4))
    for genuine, title in [(False, 'wrong target (known POI)'), (True, 'correct target')]:
        ax.hist(pairs.loc[pairs.genuine.eq(genuine), 'score'], bins=30, density=True, alpha=.5, label=title)
    ax.axvline(policy['score_threshold'], color='black', linestyle='--', label='score threshold')
    ax.set(xlabel='Max gallery cosine (не probability)', ylabel='Density'); ax.legend()
    fig.tight_layout(); fig.savefig(report / 'test_score_distribution.png', dpi=140); plt.close(fig)


class POIVerifier:
    """Локальная проверка одной цели; загрузка артефактов без сети."""
    def __init__(self, artifact_dir: Path):
        directory = Path(artifact_dir).resolve()
        self.metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
        if self.metadata['schema_version'] != 1 or self.metadata['preprocessing_version'] != PREPROCESSING_VERSION:
            raise ValueError('Несовместимая версия артефактов.')
        # Проверяем всё, что требуется для вывода, включая manifest галереи.
        for name in ('imagenet_weights.pth', 'gallery.csv', 'gallery_embeddings.npy', 'rejection_policy.json', 'class_mapping.json', 'config.json'):
            if sha256_file(directory / name) != self.metadata['artifact_sha256'][name]:
                raise ValueError(f'Артефакт изменён: {name}')
        self.policy = json.loads((directory / 'rejection_policy.json').read_text(encoding='utf-8'))
        config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
        configure_torch(SEED, config['threads'])
        self.model, self.transform = create_encoder(directory / 'imagenet_weights.pth')
        self.gallery = pd.read_csv(directory / 'gallery.csv', keep_default_na=False)
        self.vectors = np.load(directory / 'gallery_embeddings.npy', allow_pickle=False)
        self.poi_ids = self.metadata['poi_ids']
        if len(self.gallery) != len(self.vectors) or set(self.gallery.poi_id) != set(self.poi_ids):
            raise ValueError('Галерея не согласована с metadata.')

    def verify(self, image_path: Path, target_poi_id: str):
        version = {'reference_version':self.metadata['reference_version'],
                   'threshold_version':self.metadata['threshold_version'],
                   'preprocessing_version':PREPROCESSING_VERSION}
        if target_poi_id not in self.poi_ids:
            return {'accepted':False, 'reason':'target_poi_not_supported', 'target_poi_id':target_poi_id,
                    'score':None, 'margin':None, **version}
        import torch
        with torch.inference_mode():
            vector = self.model(self.transform(load_rgb(Path(image_path))).unsqueeze(0)).numpy()
        scores, nearest = poi_scores(vector, self.vectors, self.gallery.poi_id, self.poi_ids)
        target = self.poi_ids.index(target_poi_id)
        margin = float(score_margins(scores)[0, target])
        score = float(scores[0, target])
        accepted = bool(acceptance_matrix(scores, self.policy)[0, target])
        reason = ('poi_verified' if accepted else 'poi_not_matched' if margin <= 0
                  else 'poi_below_threshold' if score < self.policy['score_threshold'] else 'poi_ambiguous')
        reference = self.gallery.iloc[nearest[0, target]]
        return {'accepted':accepted, 'reason':reason, 'target_poi_id':target_poi_id,
                'predicted_poi_id':self.poi_ids[int(scores[0].argmax())],
                'score':score, 'margin':margin, 'score_is_probability':False,
                'reference_image_id':reference.image_id, 'reference_source':reference.source,
                'reference_author':reference.author, 'reference_licence':reference.licence, **version}
