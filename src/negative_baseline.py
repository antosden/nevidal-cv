"""Новый выпуск отказа с external negative-val, без переобучения backbone."""
from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd

from .poi_baseline import (POIVerifier, sha256_file, json_write, load_splits, extract_embeddings,
                           poi_scores, score_margins, acceptance_matrix, calibrate_policy, evaluate)


def evaluate_external(scores, frame, poi_ids, policy):
    accepted = acceptance_matrix(scores,policy)
    any_accept = accepted.any(axis=1)
    best = scores.argmax(axis=1)
    margins = score_margins(scores)
    queries=frame.copy()
    queries['predicted_poi_id']=[poi_ids[i] for i in best]
    queries['score']=scores[np.arange(len(scores)),best]
    queries['margin']=margins[np.arange(len(scores)),best]
    queries['accepted_any_poi']=any_accept
    summary={'images':len(frame),'false_accepts_any_poi':int(any_accept.sum()),
             'any_poi_far':float(any_accept.mean()),'rejection_rate':float(1-any_accept.mean()),
             'all_target_pairs':scores.size,'false_accepted_pairs':int(accepted.sum()),
             'uniform_target_pair_far':float(accepted.mean()),
             'poi_identity_disjoint_between_negative_val_test':True}
    if not len(frame): raise ValueError('Пустая external-выборка.')
    details=[]
    for field in ['negative_type','source_poi_id']:
        for key,group in queries.groupby(field):
            details.append({'group_by':field,'group':key,'images':len(group),
                            'false_accepts':int(group.accepted_any_poi.sum()),
                            'any_poi_far':float(group.accepted_any_poi.mean())})
    return summary,queries,pd.DataFrame(details)


def recalibrate_with_negatives(root:Path,base:Path,output:Path):
    if output.exists(): raise FileExistsError(f'Не перезаписываем выпуск {output}')
    base_meta=json.loads((base/'metadata.json').read_text(encoding='utf-8'))
    for name,digest in base_meta['artifact_sha256'].items():
        if sha256_file(base/name)!=digest: raise ValueError(f'Исходный baseline изменён: {name}')
    splits,mapping=load_splits(root)
    for name,digest in base_meta['split_sha256'].items():
        if sha256_file(root/f'data/annotations/{name}.csv')!=digest:
            raise ValueError('Положительный датасет изменился после baseline.')
    negatives={name:pd.read_csv(root/f'data/annotations/negative_{name}.csv',keep_default_na=False)
               for name in ['val','test']}
    for name,frame in negatives.items():
        if frame.empty or not frame.label_status.eq('verified').all() or not frame.dataset_role.eq('external_negative').all():
            raise ValueError(f'Невалидные negative-{name}')
        for positive in splits.values():
            for field in ['image_id','sha256','commons_file_id']:
                if (set(frame[field].astype(str))-{''})&(set(positive[field].astype(str))-{''}): raise ValueError(f'Positive overlap: {field}')
    for field in ['image_id','sha256','commons_file_id','source_poi_id','duplicate_group','split_group']:
        if (set(negatives['val'][field].astype(str))-{''})&(set(negatives['test'][field].astype(str))-{''}): raise ValueError(f'Negative leakage: {field}')
    verifier=POIVerifier(base)
    # Только исходные эталоны и эмбеддинги; negative-фото не становятся эталонами.
    # Metadata публикуется последним: незавершённый выпуск не должен загружаться
    # как будто он уже содержит новую политику.
    shutil.copytree(base,output,ignore=shutil.ignore_patterns('metadata.json'))
    config=json.loads((base/'config.json').read_text())
    config.update(json.loads((root/'config/baseline_mobilenetv3_negatives.json').read_text()))
    json_write(output/'config.json',config)
    gallery=verifier.gallery
    gallery_values=verifier.vectors
    poi_ids=verifier.poi_ids
    negative_vectors={}
    for name,frame in negatives.items():
        frame.to_csv(output/f'splits/negative_{name}.csv',index=False)
        frame[['image_id','file_path','source_poi_id','source','author','licence','sha256']].to_csv(output/f'negative_{name}_embedding_index.csv',index=False)
        negative_vectors[name]=extract_embeddings(root,frame,verifier.model,verifier.transform,config['batch_size'])
        np.save(output/f'negative_{name}_embeddings.npy',negative_vectors[name],allow_pickle=False)
    neg_val_scores,_=poi_scores(negative_vectors['val'],gallery_values,gallery.poi_id,poi_ids)
    pos_val_vectors=np.load(base/'val_embeddings.npy',allow_pickle=False)
    pos_val_scores,_=poi_scores(pos_val_vectors,gallery_values,gallery.poi_id,poi_ids)
    truths=np.array([poi_ids.index(poi) for poi in splits['val'].poi_id])
    # Единственное место подбора политики: positive-val + negative-val.
    policy,grid=calibrate_policy(pos_val_scores,truths,config,negative_scores=neg_val_scores)
    json_write(output/'rejection_policy.json',policy)
    grid.to_csv(output/'val_threshold_search.csv',index=False)
    metrics={}
    for name in ['val','test']:
        vectors=np.load(base/f'{name}_embeddings.npy',allow_pickle=False)
        scores,nearest=poi_scores(vectors,gallery_values,gallery.poi_id,poi_ids)
        summary,per_poi,pairs,confusion=evaluate(scores,splits[name],poi_ids,policy)
        # External FAR хранится отдельно: он не смешивается с known-wrong FAR.
        pairs['reference_image_id']=gallery.image_id.to_numpy()[nearest.flatten()]
        pairs.to_csv(output/f'{name}_pairs.csv',index=False)
        per_poi.to_csv(output/f'{name}_per_poi.csv',index=False)
        confusion.to_csv(output/f'{name}_confusion.csv')
        metrics[name]=summary
        negative_scores,_=poi_scores(negative_vectors[name],gallery_values,gallery.poi_id,poi_ids)
        external,queries,groups=evaluate_external(negative_scores,negatives[name],poi_ids,policy)
        metrics[name]['external_unknown_tested']=True
        metrics[name]['open_set_far']=external['any_poi_far']
        metrics[name]['open_set_far_definition']='any_poi_per_external_image'
        queries.to_csv(output/f'negative_{name}_predictions.csv',index=False)
        groups.to_csv(output/f'negative_{name}_groups.csv',index=False)
        metrics[f'negative_{name}']=external
    json_write(output/'metrics.json',metrics)
    metadata=base_meta.copy()
    metadata['threshold_version']=sha256_file(output/'rejection_policy.json')[:16]
    metadata['parent_baseline_metadata_sha256']=sha256_file(base/'metadata.json')
    metadata['calibration']='positive_val_and_external_negative_val_only'
    metadata['deployment_status']='experimental_not_production_validated'
    metadata['negative_split_sha256']={name:sha256_file(root/f'data/annotations/negative_{name}.csv') for name in ['val','test']}
    metadata['limitations']=['Wikimedia landmark negatives only; no independent mobile street dataset.',
        'Small positive validation/test sets; existing positive test is diagnostic, not a new blind holdout.',
        'Thresholds calibrated on val do not guarantee test FAR.']
    metadata['artifact_sha256']={p.relative_to(output).as_posix():sha256_file(p)
                                  for p in sorted(output.rglob('*')) if p.is_file() and p.name!='metadata.json'}
    json_write(output/'metadata.json',metadata)
    report=root/'reports/baseline_mobilenetv3_negatives'
    report.mkdir(parents=True,exist_ok=True)
    json_write(report/'metrics.json',metrics)
    dataset=json.loads((root/'reports/negatives/dataset_summary.json').read_text())
    lines=['# MobileNetV3: отказ с проверенными external-negatives','',
           'Это экспериментальная политика отказа; автоматически вместо прежней версии не включается.',
           f'Backbone, {len(gallery)} train-эталона и положительные splits сохранены; веса не дообучались. Предыдущий выпуск не изменён.',
           f'Новый выпуск: `{output.relative_to(root).as_posix()}`; reference_version={metadata["reference_version"]}.',
           f'Пороги: score ≥ {policy["score_threshold"]:.6f}, margin ≥ {policy["margin_threshold"]:.6f}, margin > 0.',
           'Подбор: только positive-val и negative-val; test в выборе порогов не участвует.','',
           '| Выборка | Фото | True accept | Known hard-wrong FAR | External any-POI FAR |',
           '|---|---:|---:|---:|---:|']
    for name in ['val','test']:
        pos,neg=metrics[name],metrics[f'negative_{name}']
        lines.append(f'| positive-{name} | {pos["query_count"]} | {pos["true_accept_rate"]:.2%} | {pos["hard_wrong_far"]:.2%} | — |')
        lines.append(f'| negative-{name} | {neg["images"]} | — | — | {neg["any_poi_far"]:.2%} ({neg["false_accepts_any_poi"]}/{neg["images"]}) |')
    lines+=['','## Наблюдение → интерпретация → действие',
            f'- Из {dataset["existing_images"]} оставшихся negatives выпущено {dataset["released_images"]}; связей-кандидатов с положительным набором: {dataset["positive_overlap_candidates"]} фото → подтверждение пользователя не устраняет технические пересечения → спорные кадры исключены; проверить negative_review_queue.csv.',
            '- Negative val/test разделены по source POI и графу дубликатов → один посторонний объект не используется одновременно для настройки и оценки → сохранять выпущенные CSV неизменными.',
            '- External any-POI FAR считает фото ошибочным при подтверждении хотя бы одной из 23 целей → это строже средней ошибки по всем целям → использовать его для основной оценки отказа, не маскировать ошибку множеством лёгких пар.',
            '- Добавлены external-negatives → теперь измерен отказ на посторонних достопримечательностях Commons → не переносить результат на любую мобильную съёмку; дополнить полевыми negatives.',
            '- Более строгий отказ может снизить true accept → настройка порогов не улучшает различимость POI → для 85% нужны новые положительные ракурсы и дообучение/сравнение эмбеддингов.',
            '- Positive-test уже использовался для диагностики прежней версии → это сравнение на фиксированном диагностическом наборе → окончательное качество новой модели подтвердить на новом закрытом полевом test.',
            '- Небольшие и зависимые внутри POI выборки → редкие ошибки оценены неточно → расширять число независимых POI и серий; не объявлять FAR гарантированным.',
            '- source/author/licence сохранены в manifest, snapshots и prediction-таблицах → атрибуция доступна для каждого фото → сохранять её при дальнейших экспортах.']
    lines+=['',f'Цель true accept ≥85%: {"достигнута на текущем диагностическом test" if metrics["test"]["true_accept_rate"]>=.85 else "не достигнута"}; подтверждено {metrics["test"]["true_accepts"]}/{metrics["test"]["positive_pairs"]} ({metrics["test"]["true_accept_rate"]:.2%}). Для улучшения нужны независимые положительные фото и обучение/сравнение эмбеддингов; ослабление порогов по test не выполнялось.']
    (report/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_negative_figures(root,output,report)
    return policy,metrics


def write_negative_figures(root,output,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .poi_baseline import load_rgb
    frame=pd.read_csv(output/'negative_test_predictions.csv',keep_default_na=False)
    errors=frame[frame.accepted_any_poi].sort_values('score',ascending=False).head(12)
    if not errors.empty:
        rows=int(np.ceil(len(errors)/3))
        fig,axes=plt.subplots(rows,3,figsize=(13,rows*3.5),squeeze=False)
        for ax in axes.ravel():ax.axis('off')
        for ax,record in zip(axes.ravel(),errors.itertuples()):
            image=load_rgb(root/record.file_path);image.thumbnail((640,480))
            ax.imshow(image)
            ax.set_title(f'{record.image_id}\n{record.source_poi_id.split("/")[-1]} -> {record.predicted_poi_id.split("/")[-1]}\nscore={record.score:.3f}; margin={record.margin:.3f}',fontsize=8)
        fig.tight_layout();fig.savefig(report/'negative_test_false_accepts.jpg',dpi=120);plt.close(fig)
    pos=pd.read_csv(output/'test_pairs.csv')
    fig,ax=plt.subplots(figsize=(8,4))
    ax.hist(pos.loc[pos.genuine,'score'],bins=25,density=True,alpha=.5,label='Positive: correct target')
    ax.hist(frame.score,bins=25,density=True,alpha=.5,label='External negative: best target')
    policy=json.loads((output/'rejection_policy.json').read_text())
    ax.axvline(policy['score_threshold'],color='black',linestyle='--',label='score threshold')
    ax.set(xlabel='Max gallery cosine (not probability)',ylabel='Density');ax.legend()
    fig.tight_layout();fig.savefig(report/'test_score_distribution.png',dpi=140);plt.close(fig)
