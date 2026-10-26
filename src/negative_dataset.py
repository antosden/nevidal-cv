"""Аудит проверенных external-negatives и разделение по независимым POI.

Исходники и положительные выборки не изменяются. POI negatives не становятся
классами распознавания и не попадают в галерею эталонов.
"""
from __future__ import annotations

import html
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm.auto import tqdm

from .dataset_eda import fingerprints, UnionFind, IMAGE_SUFFIXES
from .poi_baseline import sha256_file, json_write, safe_project_path


def bit_counts(values):
    """NumPy 2.x fast path и совместимость с заявленной NumPy 1.26."""
    values=np.asarray(values,dtype=np.uint64)
    if hasattr(np,'bitwise_count'):
        return np.bitwise_count(values)
    octets=np.ascontiguousarray(values).view(np.uint8).reshape(values.shape+(8,))
    return np.unpackbits(octets,axis=-1).sum(axis=-1)


def negative_inventory(root: Path, confirm_reviewed: bool):
    """Сопоставить физические файлы с metadata, сохранить также удалённые строки."""
    raw = root/'data/raw/wikimedia_negatives'
    metadata = pd.read_csv(raw/'metadata.csv', dtype=str, keep_default_na=False)
    config = pd.read_csv(root/'config/wikimedia_negatives.csv', dtype=str, keep_default_na=False)
    catalog = config.set_index(['city','district','poi'])
    if not catalog.index.is_unique:
        raise ValueError('Повторяющиеся POI negatives в конфиге.')
    index = {}
    for row in metadata.to_dict('records'):
        key = (row['city'], row['district'], row['poi'], PurePosixPath(row['file'].replace('\\','/')).name)
        if key in index:
            raise ValueError(f'Неоднозначные metadata: {key}')
        index[key] = row
    actual = {p.relative_to(raw).parts: p for p in raw.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
    if any(len(parts) != 4 for parts in actual):
        raise ValueError('Ожидается negative-каталог city/district/poi/file.')
    keys = sorted(set(index)|set(actual))
    rows, hashes = [], {}
    for key in tqdm(keys, desc='Аудит negatives'):
        city, district, slug, filename = key
        relative = (raw/Path(*key)).relative_to(root).as_posix()
        path = safe_project_path(root, relative)
        uid = 'neg_'+hashlib.sha256(relative.encode()).hexdigest()[:16]
        info = index.get(key, {})
        if (city,district,slug) not in catalog.index:
            raise ValueError(f'POI отсутствует в negative-конфиге: {key}')
        cfg = catalog.loc[(city,district,slug)]
        exists = key in actual
        attributed = bool(info and 'wikimedia.org/' in info.get('source_url',''))
        status = 'verified' if confirm_reviewed and attributed else 'needs_review'
        record = dict(image_id=uid, file_path=relative, city=city, district=district,
            source_poi_id=f'{city}/{district}/{slug}', poi_id='', dataset_role='external_negative',
            source='wikimedia_commons' if attributed else 'unknown',
            author=html.unescape(re.sub('<[^>]+>', '', info.get('artist',''))).strip() or 'unknown',
            author_raw=info.get('artist',''), licence=info.get('license','') or 'unknown',
            license_url=info.get('license_url',''), source_url=info.get('source_url',''),
            commons_title=info.get('commons_title',''), commons_file_id=Path(filename).stem if info else '',
            category=info.get('category',cfg.category), negative_type=cfg.negative_type,
            confusable_with=cfg.confusable_with, label_status=status,
            reviewer='user' if confirm_reviewed else '',
            review_note='Проверка negatives подтверждена пользователем.' if confirm_reviewed else '',
            rejection_reason='', file_exists=exists, readable=False, sha256='',
            perceptual_hash='', width=0, height=0, duplicate_group='', split_group='', split='', active=False)
        if not exists:
            record.update(label_status='rejected', rejection_reason='removed_by_user')
        else:
            record['sha256'] = sha256_file(path)
            try:
                with Image.open(path) as original:
                    original.seek(0)
                    image = ImageOps.exif_transpose(original).convert('RGB')
                    image.load()
                    record.update(width=image.width, height=image.height, readable=True)
                    hashes[uid] = fingerprints(image)
                    record['perceptual_hash'] = hashes[uid][0]
            except Exception as exc:
                record.update(label_status='rejected', rejection_reason='unreadable', read_error=str(exc))
        rows.append(record)
    return pd.DataFrame(rows).fillna(''), hashes


def duplicate_links(frame: pd.DataFrame, hashes: dict):
    """SHA / Commons-ID / DCT pHash и кропы; векторизация сравнения кандидатов."""
    frame = frame.reset_index(drop=True)
    ids = frame.image_id.to_numpy()
    uf = UnionFind(ids)
    pairs = []
    for field in ['sha256','commons_file_id']:
        for value, group in frame[frame[field].ne('')].groupby(field):
            members = group.image_id.tolist()
            for other in members[1:]:
                uf.union(members[0],other)
                pairs.append(dict(image_id_a=members[0], image_id_b=other, kind=field, distance=0))
    valid = frame.image_id.isin(hashes).to_numpy()
    # Отсутствующие хэши исключаются маской, а не сравниваются с нулём.
    values = np.array([[int(v,16) for v in hashes[uid]] if uid in hashes else [0]*11 for uid in ids], dtype=np.uint64)
    for i in tqdm(range(len(frame)), desc='Cross-source duplicates'):
        if not valid[i]:
            continue
        positions = np.flatnonzero(valid & (np.arange(len(frame)) > i))
        if not len(positions):
            continue
        full = bit_counts(values[positions,0] ^ values[i,0])
        crop = np.minimum(bit_counts(values[positions,1:] ^ values[i,0]).min(axis=1),
                          bit_counts(values[positions,0,None] ^ values[i,1:]).min(axis=1))
        for pos in np.flatnonzero((full <= 6)|(crop <= 4)):
            j = positions[pos]
            if uf.find(ids[i]) == uf.find(ids[j]):
                continue
            uf.union(ids[i],ids[j])
            pairs.append(dict(image_id_a=ids[i],image_id_b=ids[j],
                              kind='phash_candidate' if full[pos]<=6 else 'crop_candidate',
                              distance=int(full[pos] if full[pos]<=6 else crop[pos])))
    return uf, pd.DataFrame(pairs,columns=['image_id_a','image_id_b','kind','distance'])


def split_negative_pois(frame: pd.DataFrame, seed=42):
    """50/50 по изображениям, насколько позволяет разделение целых POI/групп."""
    uf = UnionFind(frame.image_id)
    for field in ['source_poi_id','duplicate_group']:
        for _, group in frame.groupby(field):
            members = group.image_id.tolist()
            for other in members[1:]:
                uf.union(members[0],other)
    result = frame.copy()
    result['split_group'] = ['neg_group_'+uf.find(uid) for uid in result.image_id]
    active = result[result.active]
    groups = sorted(active.split_group.unique())
    if len(groups)<2:
        raise ValueError('Недостаточно независимых POI-компонент для negative val/test.')
    matrix = pd.crosstab(active.split_group,active.negative_type).reindex(groups,fill_value=0).to_numpy()
    sizes = matrix.sum(axis=1)
    target = matrix.sum(axis=0)/2
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(10000):
        mask = rng.integers(0,2,size=len(groups)).astype(bool)
        if not mask.any() or mask.all():
            continue
        score = (abs(sizes[mask].sum()-len(active)/2)/len(active)
                 + .15*np.mean(np.abs(matrix[mask].sum(axis=0)-target)/np.maximum(target,1)))
        if best is None or score < best[0]:
            best = score,mask.copy()
    assignment = {group:'val' if best[1][i] else 'test' for i,group in enumerate(groups)}
    result.loc[result.active,'split'] = result.loc[result.active,'split_group'].map(assignment)
    val,test = (result[result.split.eq(name)] for name in ['val','test'])
    for field in ['image_id','sha256','commons_file_id','duplicate_group','split_group','source_poi_id']:
        assert not (set(val[field])-{''}) & (set(test[field])-{''}), field
    assert result.loc[~result.active,'split'].eq('').all()
    return result


def prepare_negatives(root: Path, confirm_reviewed: bool):
    annotations = root/'data/annotations'
    report = root/'reports/negatives'
    report.mkdir(parents=True,exist_ok=True)
    manifest_path = annotations/'negative_manifest.csv'
    if manifest_path.exists():
        raise FileExistsError('Выпуск negatives уже существует. Не перезаписывайте зафиксированные val/test.')
    frame, hashes = negative_inventory(root,confirm_reviewed)
    frame.to_csv(report/'inventory_cache.csv',index=False)
    json_write(report/'image_fingerprints.json',hashes)
    positive = pd.read_csv(annotations/'dataset_manifest.csv',keep_default_na=False)
    # Все существующие положительные фото, включая два конфликта вне splits.
    positive = positive[positive.file_exists.eq(True)].copy()
    positive_hashes = json.loads((root/'reports/image_fingerprints.json').read_text(encoding='utf-8'))
    combined = pd.concat([positive[['image_id','sha256','commons_file_id']],
                          frame[['image_id','sha256','commons_file_id']]],ignore_index=True).fillna('')
    uf,pairs = duplicate_links(combined,{**positive_hashes,**hashes})
    # Сохраняем ранее известные near/series-связи положительного датасета.
    for _,group in positive.groupby('split_group'):
        members=group.image_id.tolist()
        for other in members[1:]: uf.union(members[0],other)
    positive_components = {uf.find(uid) for uid in positive.image_id}
    frame['duplicate_group'] = ['dup_'+uf.find(uid) for uid in frame.image_id]
    overlap = frame.image_id.map(lambda uid:uf.find(uid) in positive_components)&frame.file_exists&frame.readable
    frame.loc[overlap,'label_status']='needs_review'
    frame.loc[overlap,'rejection_reason']='positive_overlap_candidate'
    edits_path = annotations/'negative_review_edits.csv'
    review_columns=['image_id','file_path','sha256','source_poi_id','source','author','licence',
                    'label_status','rejection_reason','reviewer','review_note']
    if edits_path.exists():
        raise FileExistsError('negative_review_edits.csv уже существует; решения нельзя перезаписать.')
    frame[review_columns].to_csv(edits_path,index=False)
    frame['active'] = frame.file_exists&frame.readable&frame.label_status.eq('verified')&~overlap
    # Одна побайтовая копия; near-duplicates группируются, но не удаляются.
    candidates=frame[frame.active].sort_values('image_id')
    duplicate_ids=candidates[candidates.sha256.duplicated()].image_id
    frame.loc[frame.image_id.isin(duplicate_ids),'active']=False
    frame['exclusion_reason']=frame.rejection_reason
    frame.loc[frame.image_id.isin(duplicate_ids),'exclusion_reason']='exact_duplicate'
    frame.loc[~frame.active&frame.exclusion_reason.eq(''),'exclusion_reason']='label_not_verified'
    frame=split_negative_pois(frame)
    # Явный audit-файл конфликтов содержит связи, а не автоматические новые метки.
    pairs['a_is_negative']=pairs.image_id_a.isin(frame.image_id)
    pairs['b_is_negative']=pairs.image_id_b.isin(frame.image_id)
    pairs=pairs[pairs.a_is_negative|pairs.b_is_negative]
    pairs.to_csv(annotations/'negative_duplicate_pairs.csv',index=False)
    frame.to_csv(manifest_path,index=False)
    frame[frame.label_status.isin(['needs_review','uncertain'])].to_csv(annotations/'negative_review_queue.csv',index=False)
    for name in ['val','test']:
        frame[frame.split.eq(name)].sort_values('image_id').to_csv(annotations/f'negative_{name}.csv',index=False)
    summary={'audit_records':len(frame),'existing_images':int(frame.file_exists.sum()),
             'removed_by_user':int(frame.rejection_reason.eq('removed_by_user').sum()),
             'unreadable':int((frame.file_exists&~frame.readable).sum()),
             'positive_overlap_candidates':int(overlap.sum()),'exact_duplicate_excluded':len(duplicate_ids),
             'released_images':int(frame.active.sum()),
             'split_counts':{s:int(frame.split.eq(s).sum()) for s in ['val','test']},
             'split_poi_counts':{s:int(frame.loc[frame.split.eq(s),'source_poi_id'].nunique()) for s in ['val','test']},
             'unknown_authors':int((frame.file_exists&frame.author.eq('unknown')).sum()),
             'unknown_licences':int((frame.file_exists&frame.licence.eq('unknown')).sum()),
             'user_review_confirmed':bool(confirm_reviewed),'seed':42,'split_method':'disjoint_POI_and_duplicate_components',
             'cross_positive_overlap_excluded':True,'raw_images_modified':False}
    json_write(report/'dataset_summary.json',summary)
    frame.groupby(['source_poi_id','negative_type','split','exclusion_reason'],dropna=False).size().rename('images').to_csv(report/'distribution.csv')
    json_write(report/'source_snapshot.json',{
        'metadata_sha256':sha256_file(root/'data/raw/wikimedia_negatives/metadata.csv'),
        'config_sha256':sha256_file(root/'config/wikimedia_negatives.csv'),
        'positive_manifest_sha256':sha256_file(annotations/'dataset_manifest.csv'),
        'images':dict(zip(frame.loc[frame.file_exists,'file_path'],frame.loc[frame.file_exists,'sha256']))})
    write_negative_report(root)
    return frame,summary


def write_negative_report(root: Path):
    """Сводка и пары конфликтов для просмотра; без изменения решений и splits."""
    report=root/'reports/negatives'
    summary=json.loads((report/'dataset_summary.json').read_text())
    manifest=pd.read_csv(root/'data/annotations/negative_manifest.csv',keep_default_na=False)
    pairs=pd.read_csv(root/'data/annotations/negative_duplicate_pairs.csv',keep_default_na=False)
    positive=pd.read_csv(root/'data/annotations/dataset_manifest.csv',keep_default_na=False)
    active=manifest[manifest.active]
    distribution=pd.crosstab(active.negative_type,active.split)
    lines=['# Проверенный negative-набор','',
           f'Физических файлов: {summary["existing_images"]}; удалённых пользователем: {summary["removed_by_user"]}; нечитаемых: {summary["unreadable"]}.',
           f'Выпущено {summary["released_images"]}: val={summary["split_counts"]["val"]}, test={summary["split_counts"]["test"]}.',
           '','## Наблюдение → интерпретация → действие',
           f'- Просмотр оставшихся файлов подтверждён пользователем → семантическая проверка зафиксирована → подтверждённые и технически допустимые фото имеют verified.',
           f'- Связи с положительным набором у {summary["positive_overlap_candidates"]} фото → потенциальное противоречие цели и негатива → эти фото исключены; см. negative_review_queue.csv и лист пар.',
           '- Часть снимков похожа или происходит из серий → случайное разделение отдельных фото завысило бы качество → группы копий и целые посторонние POI не пересекают val/test.',
           f'- POI в val: {summary["split_poi_counts"]["val"]}, в test: {summary["split_poi_counts"]["test"]} → проверяется перенос на другие посторонние объекты → не добавлять test-объекты при настройке отказа.',
           '- Некоторые типы представлены одним POI → покрытие типов в двух splits не может совпасть при разделении по объектам → добрать другие башни и колёса обозрения для второго split.',
           f'- Автор unknown у {summary["unknown_authors"]} существующих фото; licence unknown у {summary["unknown_licences"]} → часть атрибуции не заполнена источником → сохранить исходные поля и уточнить авторов по Commons.',
           '- Набор состоит преимущественно из достопримечательностей Commons → оценка не описывает любые телефонные фото → дополнительно собрать улицы, парки и реальные ошибочные запросы.',
           '','## Распределение типов','```',distribution.to_string(),'```','',
           'Исходные файлы, metadata и конфиг прогресса не изменялись. Источник, авторы, лицензии и URL сохранены. Отдельного класса unknown нет.']
    (report/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    cross=pairs[pairs.a_is_negative.ne(pairs.b_is_negative)].drop_duplicates(['image_id_a','image_id_b']).head(12)
    if cross.empty:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .poi_baseline import load_rgb
    index=pd.concat([positive,manifest],ignore_index=True).set_index('image_id')
    fig,axes=plt.subplots(len(cross),2,figsize=(12,len(cross)*4),squeeze=False)
    for n,pair in enumerate(cross.itertuples()):
        for ax,uid in zip(axes[n],[pair.image_id_a,pair.image_id_b]):
            row=index.loc[uid]
            image=load_rgb(root/row.file_path);image.thumbnail((640,480))
            ax.imshow(image);ax.axis('off')
            label=row.poi_id if row.poi_id else row.source_poi_id
            ax.set_title(f'{uid}\n{label}\n{pair.kind}',fontsize=9)
    fig.tight_layout();fig.savefig(report/'positive_overlap_examples.jpg',dpi=120);plt.close(fig)
