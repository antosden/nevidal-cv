"""MobileNetV3/Ridge retrieval → SIFT/FLANN → RANSAC → глобальный отказ.

Геометрические эталоны только из train. Настройка политики только по validation.
Исходные изображения и существующие разбиения не изменяются.
"""
from pathlib import Path
import hashlib
import json
import time
import shutil

import cv2
import numpy as np
import pandas as pd
from PIL import Image,ImageOps
from tqdm.auto import tqdm

from .poi_baseline import (sha256_file,json_write,configure_torch,poi_scores,calibrate_policy,
                          acceptance_matrix,evaluate,score_margins)
from .poi_metric import ridge_scores
from .poi_finetune import classification_metrics
from .negative_baseline import evaluate_external

GEOMETRY_VERSION='sift960-ratio072-ransac-hybrid-v1'


def configure_geometry(seed=42):
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);cv2.setRNGSeed(seed)


def geometry_gray(path,side):
    with Image.open(path) as source:
        source.seek(0);source.draft('RGB',(side*2,side*2))
        image=ImageOps.exif_transpose(source).convert('L')
        image.thumbnail((side,side),Image.Resampling.LANCZOS)
        return np.asarray(image).copy()


def extract_local(path,config):
    gray=geometry_gray(path,config['image_long_side'])
    sift=cv2.SIFT_create(nfeatures=config['sift_features'],contrastThreshold=config['sift_contrast_threshold'])
    keypoints,descriptors=sift.detectAndCompute(gray,None)
    points=np.array([point.pt for point in keypoints],dtype=np.float32).reshape(-1,2)
    if descriptors is None:descriptors=np.empty((0,128),dtype=np.float32)
    return {'points':points,'descriptors':descriptors,'shape':np.array(gray.shape,dtype=np.int32)}


def local_cache(root,frame,config,cache):
    cache.mkdir(parents=True,exist_ok=True)
    result={};index=[]
    fingerprint=hashlib.sha256(json.dumps({key:config[key] for key in ['image_long_side','sift_features','sift_contrast_threshold']},sort_keys=True).encode()).hexdigest()[:12]
    for row in tqdm(frame.itertuples(),total=len(frame),desc='SIFT features'):
        path=cache/f'{row.sha256}_{fingerprint}.npz'
        provenance=path.with_suffix('.json')
        if not path.exists():
            source=root/row.file_path
            if sha256_file(source)!=row.sha256:raise ValueError(f'Изменён исходник {row.image_id}')
            values=extract_local(source,config)
            temporary=path.with_suffix('.npz.tmp')
            with temporary.open('wb') as stream:np.savez_compressed(stream,**values)
            temporary.replace(path)
            json_write(provenance,{'source_sha256':row.sha256,'cache_sha256':sha256_file(path),'opencv':cv2.__version__})
        expected=json.loads(provenance.read_text())
        if expected['source_sha256']!=row.sha256 or expected['cache_sha256']!=sha256_file(path) or expected['opencv']!=cv2.__version__:
            raise ValueError('Кэш локальных признаков изменён или несовместим.')
        saved=np.load(path,allow_pickle=False)
        result[row.image_id]={key:saved[key] for key in ['points','descriptors','shape']}
        index.append({'image_id':row.image_id,'file_path':row.file_path,'source_sha256':row.sha256,
                      'feature_file':path.relative_to(root).as_posix(),'feature_sha256':sha256_file(path)})
    return result,pd.DataFrame(index)


def matcher_for(reference,seed=42):
    if len(reference['descriptors'])<2:return None
    cv2.setRNGSeed(seed)
    matcher=cv2.FlannBasedMatcher({'algorithm':1,'trees':4},{'checks':48})
    matcher.add([reference['descriptors']]);matcher.train()
    return matcher


def pair_evidence(query,reference,config,seed=42,matcher=None):
    empty={'matches':0,'inliers':0,'inlier_ratio':0.,'query_coverage':0.,'reference_coverage':0.,'geometric_score':0.}
    if len(query['descriptors'])<config['minimum_matches'] or len(reference['descriptors'])<config['minimum_matches']:return empty
    cv2.setRNGSeed(seed)
    if matcher is None:matcher=matcher_for(reference,seed)
    pairs=matcher.knnMatch(query['descriptors'],k=2)
    good=[pair[0] for pair in pairs if len(pair)==2 and pair[0].distance<config['ratio_test']*pair[1].distance]
    # Разным query-деталям не разрешаем голосовать за один и тот же ref keypoint.
    unique={}
    for match in sorted(good,key=lambda item:item.distance):unique.setdefault(match.trainIdx,match)
    good=list(unique.values())
    if len(good)<config['minimum_matches']:return {**empty,'matches':len(good)}
    src=np.array([query['points'][match.queryIdx] for match in good],dtype=np.float32)
    dst=np.array([reference['points'][match.trainIdx] for match in good],dtype=np.float32)
    homography,mask=cv2.findHomography(src,dst,cv2.RANSAC,config['ransac_reprojection_threshold'],
                                      maxIters=config['ransac_max_iterations'],confidence=.995)
    if homography is None or mask is None or not np.isfinite(homography).all():return {**empty,'matches':len(good)}
    valid=mask.ravel().astype(bool);count=int(valid.sum())
    if count<4:return {**empty,'matches':len(good),'inliers':count}
    q_coverage=float(cv2.contourArea(cv2.convexHull(src[valid]))/np.prod(query['shape']))
    r_coverage=float(cv2.contourArea(cv2.convexHull(dst[valid]))/np.prod(reference['shape']))
    ratio=count/len(good)
    result={'matches':len(good),'inliers':count,'inlier_ratio':ratio,'query_coverage':q_coverage,'reference_coverage':r_coverage,'geometric_score':0.}
    if count>=config['minimum_inliers'] and min(q_coverage,r_coverage)>=config['minimum_coverage']:
        result['geometric_score']=float(np.log1p(count)*ratio*np.sqrt(min(q_coverage,r_coverage)))
    return result


def retrieve_and_match(frame,raw,local,gallery,gallery_vectors,reference_local,parameters,pois,config):
    coarse=ridge_scores(raw,parameters)
    normalized=raw/np.linalg.norm(raw,axis=1,keepdims=True)
    similarity=normalized@gallery_vectors.T
    labels=gallery.poi_id.to_numpy()
    matchers={}
    scores=np.zeros((len(frame),len(pois),config['references_per_poi']),dtype=np.float64)
    references=np.full(scores.shape,-1,dtype=np.int32)
    evidence=[];elapsed=[]
    for n,row in enumerate(tqdm(frame.itertuples(),total=len(frame),desc='Geometric reranking')):
        started=time.monotonic()
        candidates=np.argsort(-coarse[n],kind='stable')[:config['candidate_pois']]
        for target in candidates:
            positions=np.flatnonzero(labels==pois[target])
            selected=positions[np.argsort(-similarity[n,positions],kind='stable')[:config['references_per_poi']]]
            for slot,position in enumerate(selected):
                reference_id=gallery.iloc[position].image_id
                if reference_id not in matchers:matchers[reference_id]=matcher_for(reference_local[reference_id],config['seed'])
                # Seed не зависит от ID/пути query: batch и одиночный вывод
                # одного и того же фото используют идентичную геометрию.
                seed=int(hashlib.sha256(reference_id.encode()).hexdigest()[:8],16)%2147483647
                values=pair_evidence(local[row.image_id],reference_local[reference_id],config,seed,matchers[reference_id])
                scores[n,target,slot]=values['geometric_score'];references[n,target,slot]=position
                evidence.append({'image_id':row.image_id,'target_poi_id':pois[target],'reference_image_id':reference_id,**values})
        elapsed.append(time.monotonic()-started)
    return scores,references,coarse,pd.DataFrame(evidence),elapsed


def aggregate_scores(local_scores,coarse,mode,fusion='geometry_only'):
    ordered=np.sort(local_scores,axis=2)
    if mode=='best':scores=ordered[:,:,-1]
    elif mode=='mean_two':scores=ordered[:,:,-2:].mean(axis=2)
    else:raise ValueError('Неизвестная агрегация.')
    if fusion=='hybrid_ridge':return scores+coarse
    if fusion!='geometry_only':raise ValueError('Неизвестная fusion.')
    # При отсутствии локальной поддержки ранжирование откатывается к coarse.
    # Но score floor >0.01 запрещает подтверждение без геометрии.
    return scores+1e-6*coarse


def run_geometric_validation(root,output):
    if (output/'selection.json').exists() or (output/'metadata.json').exists():raise FileExistsError('Validation уже зафиксирован; используйте finalize.')
    config=json.loads((root/'config/mobilenetv3_geometric.json').read_text())
    configure_torch(42,4);configure_geometry(42)
    base=root/'models/baseline_mobilenetv3_negatives';classifier=root/'models/mobilenetv3_calibrated'
    parent=json.loads((base/'metadata.json').read_text());classifier_meta=json.loads((classifier/'metadata.json').read_text())
    for directory,metadata in [(base,parent),(classifier,classifier_meta)]:
        for name,digest in metadata['artifact_sha256'].items():
            if sha256_file(directory/name)!=digest:raise ValueError(f'Изменён источник {name}')
    gallery=pd.read_csv(base/'gallery.csv',keep_default_na=False)
    train_ids=set(pd.read_csv(base/'train_embedding_index.csv').image_id)
    assert set(gallery.image_id)<=train_ids
    gallery_vectors=np.load(base/'gallery_embeddings.npy',allow_pickle=False)
    stored=np.load(classifier/'ridge_parameters.npz',allow_pickle=False)
    parameters={key:stored[key] for key in stored.files};pois=parent['poi_ids']
    if (output/'config.json').exists() and json.loads((output/'config.json').read_text())!=config:
        raise ValueError('Нельзя продолжить partial-run с другой конфигурацией.')
    output.mkdir(parents=True,exist_ok=True);(output/'splits').mkdir(exist_ok=True)
    json_write(output/'config.json',config)
    json_write(output/'input_snapshot.json',{'parent_sha256':sha256_file(base/'metadata.json'),
             'classifier_sha256':sha256_file(classifier/'metadata.json'),
             'split_sha256':{name:sha256_file(root/f'data/annotations/{name}.csv') for name in ['train','val','test','negative_val','negative_test']}})
    cache=root/'.runtime/geometric_features'
    ref_local,ref_index=local_cache(root,gallery,config,cache)
    ref_index.to_csv(output/'reference_feature_index.csv',index=False)
    gallery.to_csv(output/'gallery.csv',index=False);np.save(output/'gallery_embeddings.npy',gallery_vectors,allow_pickle=False)
    np.savez(output/'ridge_parameters.npz',**parameters)
    queries={};frames={};vectors={};indices=[ref_index.assign(split='train_gallery')]
    for name in ['val','negative_val']:
        frames[name]=pd.read_csv(base/f'splits/{name}.csv',keep_default_na=False).sort_values('image_id').reset_index(drop=True)
        vectors[name]=np.load(base/f'{name}_embeddings.npy',allow_pickle=False)
        frames[name].to_csv(output/f'splits/{name}.csv',index=False)
        query_local,index=local_cache(root,frames[name],config,cache);indices.append(index.assign(split=name))
        queries[name]=retrieve_and_match(frames[name],vectors[name],query_local,gallery,gallery_vectors,ref_local,parameters,pois,config)
        np.savez(output/f'{name}_matching.npz',local_scores=queries[name][0],references=queries[name][1],coarse=queries[name][2])
        queries[name][3].to_csv(output/f'{name}_evidence.csv',index=False)
    pd.concat(indices).to_csv(output/'feature_index.csv',index=False)
    return select_geometry_policy(root,output)


def select_geometry_policy(root,output):
    """Выбор из сохранённых validation matching; ни одной test-строки на входе."""
    if (output/'metadata.json').exists():raise FileExistsError('Нельзя менять завершённый выпуск.')
    config=json.loads((root/'config/mobilenetv3_geometric.json').read_text())
    old=json.loads((output/'config.json').read_text())
    for key in old:
        if key not in ['score_fusions'] and old[key]!=config[key]:raise ValueError('Настройки matching изменились; нужен новый запуск.')
    json_write(output/'config.json',config)
    data={name:np.load(output/f'{name}_matching.npz',allow_pickle=False) for name in ['val','negative_val']}
    frames=pd.read_csv(output/'splits/val.csv',keep_default_na=False)
    pois=json.loads((root/'models/baseline_mobilenetv3_negatives/metadata.json').read_text())['poi_ids']
    truth=np.array([pois.index(p) for p in frames.poi_id]);candidates=[];selected=None
    for mode in config['score_aggregations']:
        for fusion in config['score_fusions']:
            scores=aggregate_scores(data['val']['local_scores'],data['val']['coarse'],mode,fusion)
            negatives=aggregate_scores(data['negative_val']['local_scores'],data['negative_val']['coarse'],mode,fusion)
            policy,grid=calibrate_policy(scores,truth,config,negatives)
            policy['score_type']='sift_ransac_consistency_not_probability' if fusion=='geometry_only' else 'sift_ransac_plus_ridge_response_not_probability'
            quality={**classification_metrics(scores,frames,pois),'aggregation':mode,'score_fusion':fusion,
                     'true_accept_rate':policy['true_accept_rate'],'external_any_poi_far':policy['external_any_poi_far']}
            candidates.append(quality);key=(quality['true_accept_rate'],quality['macro_f1'],quality['accuracy'])
            if selected is None or key>selected[0]:selected=(key,mode,fusion,policy,grid)
            print(json.dumps(quality),flush=True)
    _,mode,fusion,policy,grid=selected
    json_write(output/'rejection_policy.json',policy)
    pd.DataFrame(candidates).to_csv(output/'validation_candidates.csv',index=False)
    grid.to_csv(output/'val_threshold_search.csv',index=False)
    json_write(output/'selection.json',{'aggregation':mode,'score_fusion':fusion,'selection_split':'val_and_negative_val','test_used_for_selection':False})
    return mode,fusion,policy


def finalize_geometric(root,output):
    if (output/'metadata.json').exists():raise FileExistsError('Завершённый выпуск не перезаписывается.')
    base=root/'models/baseline_mobilenetv3_negatives';classifier=root/'models/mobilenetv3_calibrated'
    config=json.loads((output/'config.json').read_text());selection=json.loads((output/'selection.json').read_text())
    snapshot=json.loads((output/'input_snapshot.json').read_text())
    assert sha256_file(base/'metadata.json')==snapshot['parent_sha256']
    assert sha256_file(classifier/'metadata.json')==snapshot['classifier_sha256']
    for name,digest in snapshot['split_sha256'].items():assert sha256_file(root/f'data/annotations/{name}.csv')==digest,name
    policy=json.loads((output/'rejection_policy.json').read_text());mode=selection['aggregation'];fusion=selection['score_fusion']
    configure_torch(42,4);configure_geometry(42)
    parent=json.loads((base/'metadata.json').read_text());pois=parent['poi_ids']
    gallery=pd.read_csv(output/'gallery.csv',keep_default_na=False)
    gallery_vectors=np.load(output/'gallery_embeddings.npy',allow_pickle=False)
    stored=np.load(output/'ridge_parameters.npz',allow_pickle=False);parameters={k:stored[k] for k in stored.files}
    cache=root/'.runtime/geometric_features'
    ref_local,_=local_cache(root,gallery,config,cache)
    indices=[pd.read_csv(output/'feature_index.csv',keep_default_na=False)]
    frames={};queries={};timings={}
    for name in ['val','negative_val']:
        frames[name]=pd.read_csv(output/f'splits/{name}.csv',keep_default_na=False)
        data=np.load(output/f'{name}_matching.npz',allow_pickle=False)
        queries[name]=[data['local_scores'],data['references'],data['coarse']]
    for name in ['test','negative_test']:
        frame=pd.read_csv(base/f'splits/{name}.csv',keep_default_na=False).sort_values('image_id').reset_index(drop=True)
        frames[name]=frame;frame.to_csv(output/f'splits/{name}.csv',index=False)
        local,index=local_cache(root,frame,config,cache);indices.append(index.assign(split=name))
        raw=np.load(base/f'{name}_embeddings.npy',allow_pickle=False)
        result=retrieve_and_match(frame,raw,local,gallery,gallery_vectors,ref_local,parameters,pois,config)
        queries[name]=result[:3];timings[name]=result[4]
        np.savez(output/f'{name}_matching.npz',local_scores=result[0],references=result[1],coarse=result[2])
        result[3].to_csv(output/f'{name}_evidence.csv',index=False)
    pd.concat(indices).to_csv(output/'feature_index.csv',index=False)
    metrics={}
    for name in ['val','test']:
        scores=aggregate_scores(queries[name][0],queries[name][2],mode,fusion)
        np.save(output/f'{name}_scores.npy',scores,allow_pickle=False)
        summary,per_poi,pairs,confusion=evaluate(scores,frames[name],pois,policy)
        summary.update(classification_metrics(scores,frames[name],pois))
        predicted=scores.argmax(1)
        predicted_values=queries[name][0][np.arange(len(scores)),predicted]
        slots=predicted_values.argmax(1)
        positions=queries[name][1][np.arange(len(scores)),predicted,slots]
        query_predictions=frames[name].copy()
        query_predictions['predicted_poi_id']=[pois[i] for i in predicted]
        query_predictions['accepted']=acceptance_matrix(scores,policy).any(1)
        query_predictions['reference_image_id']=[gallery.iloc[p].image_id if p>=0 else '' for p in positions]
        query_predictions.to_csv(output/f'{name}_predictions.csv',index=False)
        pairs.to_csv(output/f'{name}_pairs.csv',index=False);per_poi.to_csv(output/f'{name}_per_poi.csv',index=False)
        confusion.to_csv(output/f'{name}_confusion.csv')
        ns=aggregate_scores(queries[f'negative_{name}'][0],queries[f'negative_{name}'][2],mode,fusion)
        np.save(output/f'negative_{name}_scores.npy',ns,allow_pickle=False)
        external,neg_queries,groups=evaluate_external(ns,frames[f'negative_{name}'],pois,policy)
        neg_queries.to_csv(output/f'negative_{name}_predictions.csv',index=False)
        groups.to_csv(output/f'negative_{name}_groups.csv',index=False)
        summary.update(external_unknown_tested=True,open_set_far=external['any_poi_far'],open_set_far_definition='any_poi_per_external_image')
        metrics[name]=summary;metrics[f'negative_{name}']=external
    json_write(output/'metrics.json',metrics)
    # Самодостаточный выпуск: дескрипторы эталонов и ImageNet-веса локальны.
    (output/'references').mkdir()
    ref_index=pd.read_csv(output/'reference_feature_index.csv',keep_default_na=False)
    for row in ref_index.itertuples():shutil.copyfile(root/row.feature_file,output/f'references/{row.image_id}.npz')
    for name in ['imagenet_weights.pth','class_mapping.json']:shutil.copyfile(base/name,output/name)
    metadata={'schema_version':4,'model_type':'mobilenetv3_sift_geometric','poi_ids':pois,
        'geometry_version':GEOMETRY_VERSION,'preprocessing_version':parent['preprocessing_version'],
        'reference_version':sha256_file(output/'reference_feature_index.csv')[:16],
        'threshold_version':sha256_file(output/'rejection_policy.json')[:16],
        'aggregation':mode,'score_fusion':fusion,'score_type':policy['score_type'],'embedding_dim':576,'training_performed':False,
        'local_references_source':'positive_train_only','calibration':'positive_val_and_negative_val_only',
        'seed':42,'parent_metadata_sha256':snapshot['parent_sha256'],'classifier_metadata_sha256':snapshot['classifier_sha256'],
        'input_split_sha256':snapshot['split_sha256'],'opencv':cv2.__version__,
        'deployment_status':'not_promoted_far_exceeded' if metrics['negative_test']['any_poi_far']>.01 else 'experimental_not_production_validated',
        'matching_only_latency_seconds':{name:{'mean':float(np.mean(values)),'p95':float(np.quantile(values,.95))} for name,values in timings.items()},
        'limitations':['Homography assumes a locally planar view; view/lighting changes and moving water can fail.',
                       'Positive test was used diagnostically before; new blind field test needed.',
                       'Global false accept rate not guaranteed per POI or on new negative types.']}
    metadata['artifact_sha256']={p.relative_to(output).as_posix():sha256_file(p) for p in sorted(output.rglob('*')) if p.is_file()}
    json_write(output/'metadata.json',metadata)
    write_geometry_report(root,output,metadata,metrics)
    return metadata,metrics


def write_geometry_report(root,output,metadata,metrics):
    report=root/'reports/mobilenetv3_geometric';report.mkdir(parents=True,exist_ok=True)
    old=json.loads((root/'models/baseline_mobilenetv3_negatives/metrics.json').read_text())
    ridge=json.loads((root/'models/mobilenetv3_calibrated/metrics.json').read_text())
    lines=['# MobileNetV3 + локальная геометрическая проверка','',
        'Ridge на MobileNetV3-эмбеддингах предлагает 5 POI. Для каждого сравниваются 3 ближайших train-эталона.',
        'SIFT → FLANN ratio test → уникальные ref keypoints → RANSAC homography. Геометрическая ветка требует ≥8 inliers и покрытия ≥2% каждого кадра.',
        f'Агрегация эталонов: {metadata["aggregation"]}; fusion={metadata["score_fusion"]}. Вариант и глобальные score/margin пороги выбраны только по positive-val/negative-val.',
        ('Score — геометрическая согласованность; без локальных совпадений фиксированный floor запрещает подтверждение.' if metadata['score_fusion']=='geometry_only' else 'Score = геометрическая согласованность + Ridge-отклик. При отсутствии геометрии возможен только проход строгих глобальных порогов Ridge-fallback.'),
        'Score не является вероятностью. Все четыре кандидата и их validation-метрики сохранены.',
        '','| Test / модель | Accuracy | Macro-F1 | True accept | External any-POI FAR |',
        '|---|---:|---:|---:|---:|',
        f'| Frozen + строгие пороги | {old["test"]["top1_accuracy"]:.2%} | 73.26% | {old["test"]["true_accept_rate"]:.2%} | {old["negative_test"]["any_poi_far"]:.2%} |',
        f'| Ridge + per-POI пороги | {ridge["test"]["accuracy"]:.2%} | {ridge["test"]["macro_f1"]:.2%} | {ridge["test"]["true_accept_rate"]:.2%} | {ridge["negative_test"]["any_poi_far"]:.2%} |',
        f'| Геометрическая проверка | {metrics["test"]["accuracy"]:.2%} | {metrics["test"]["macro_f1"]:.2%} | {metrics["test"]["true_accept_rate"]:.2%} | {metrics["negative_test"]["any_poi_far"]:.2%} |','',
        '## Наблюдение → интерпретация → действие',
        f'- Подтверждено {metrics["test"]["true_accepts"]}/{metrics["test"]["positive_pairs"]} правильных фото → true accept {metrics["test"]["true_accept_rate"]:.2%} → цель 85% {"достигнута на диагностическом test" if metrics["test"]["true_accept_rate"]>=.85 else "не достигнута"}.',
        f'- External ложных подтверждений {metrics["negative_test"]["false_accepts_any_poi"]}/{metrics["negative_test"]["images"]} → FAR {metrics["negative_test"]["any_poi_far"]:.2%} → не менять пороги по test; оценить на новом закрытом полевом наборе.',
        '- Геометрия проверяет локальные детали конкретного экземпляра; hybrid допускает также Ridge-fallback → часть решений может не иметь геометрической поддержки → отдельно анализировать такие подтверждения по evidence.csv.',
        '- При отсутствии геометрии coarse ranking остаётся для диагностики → accuracy без отказа и true accept отвечают разным вопросам → не выдавать ranking за подтверждение.',
        '- Homography приближает одну плоскость; статуи, разные стороны здания и движущаяся вода могут не совпасть → отказ не доказывает неверный POI → добрать разные ракурсы train-эталонов.',
        '- Стоимость вычислений выше исходного MobileNet → исследовательский CPU-baseline → измерить полный latency на целевом оборудовании перед применением.',
        '- Атрибуция source/author/licence сохраняется в галерее и snapshots → происхождение доступно → сохранять эти поля при экспорте.',
        '','[OpenCV: matching](https://docs.opencv.org/4.10.0/dc/dc3/tutorial_py_matcher.html).',
        '[OpenCV: homography](https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html).']
    (report/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');json_write(report/'metrics.json',metrics)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    values=[('Frozen + strict',old),('Ridge + per-POI',ridge),('Geometry',metrics)]
    for label,value in values:
        x=100*value['negative_test']['any_poi_far'];y=100*value['test']['true_accept_rate']
        ax.scatter(x,y,s=55);ax.annotate(label,(x,y),xytext=(6,5),textcoords='offset points')
    ax.axvline(1,color='red',linestyle='--',label='FAR target 1%')
    ax.axhline(85,color='green',linestyle='--',label='TAR target 85%')
    ax.set(xlabel='External any-POI FAR, %',ylabel='True accept, %',ylim=(0,100),xlim=(0,max(6,100*metrics['negative_test']['any_poi_far']+2)))
    ax.legend(loc='lower right');fig.tight_layout();fig.savefig(report/'test_tradeoff.png',dpi=140);plt.close(fig)


class GeometricPOIVerifier:
    def __init__(self,directory):
        from .poi_baseline import create_encoder
        self.directory=Path(directory);self.metadata=json.loads((self.directory/'metadata.json').read_text())
        if self.metadata['schema_version']!=4 or self.metadata['geometry_version']!=GEOMETRY_VERSION:raise ValueError('Неверная схема geometry.')
        if self.metadata['opencv']!=cv2.__version__:raise ValueError('Версия OpenCV отличается от проверенной; используйте зафиксированную среду.')
        for name,digest in self.metadata['artifact_sha256'].items():
            if sha256_file(self.directory/name)!=digest:raise ValueError(f'Изменён артефакт {name}')
        configure_torch(42,4);configure_geometry(42)
        self.config=json.loads((self.directory/'config.json').read_text());self.policy=json.loads((self.directory/'rejection_policy.json').read_text())
        self.encoder,self.transform=create_encoder(self.directory/'imagenet_weights.pth')
        self.gallery=pd.read_csv(self.directory/'gallery.csv',keep_default_na=False)
        self.vectors=np.load(self.directory/'gallery_embeddings.npy',allow_pickle=False)
        stored=np.load(self.directory/'ridge_parameters.npz',allow_pickle=False);self.parameters={k:stored[k] for k in stored.files}
        self.references={}
        for uid in self.gallery.image_id:
            data=np.load(self.directory/f'references/{uid}.npz',allow_pickle=False)
            self.references[uid]={k:data[k] for k in data.files}
        self.poi_ids=self.metadata['poi_ids']

    def verify(self,image_path,target_poi_id):
        import torch
        from .poi_baseline import load_rgb,l2_normalize
        version={k:self.metadata[k] for k in ['reference_version','threshold_version','preprocessing_version','geometry_version']}
        if target_poi_id not in self.poi_ids:return {'accepted':False,'reason':'target_poi_not_supported','score':None,'margin':None,'target_poi_id':target_poi_id,**version}
        with torch.inference_mode():raw=l2_normalize(self.encoder(self.transform(load_rgb(Path(image_path))).unsqueeze(0)).numpy())
        local=extract_local(Path(image_path),self.config)
        uid='query_'+sha256_file(Path(image_path))[:16]
        result=retrieve_and_match(pd.DataFrame([{'image_id':uid}]),raw,{uid:local},self.gallery,self.vectors,self.references,self.parameters,self.poi_ids,self.config)
        scores=aggregate_scores(result[0],result[2],self.metadata['aggregation'],self.metadata['score_fusion'])
        target=self.poi_ids.index(target_poi_id);score=float(scores[0,target]);margin=float(score_margins(scores)[0,target])
        accepted=bool(acceptance_matrix(scores,self.policy)[0,target])
        reference_index=int(result[1][0,target,int(result[0][0,target].argmax())])
        reference=self.gallery.iloc[reference_index] if reference_index>=0 else None
        return {'accepted':accepted,'reason':'poi_verified' if accepted else 'poi_not_matched' if margin<=0 else 'poi_below_threshold' if score<self.policy['score_threshold'] else 'poi_ambiguous',
                'target_poi_id':target_poi_id,'predicted_poi_id':self.poi_ids[int(scores[0].argmax())],
                'score':score,'margin':margin,'score_type':self.metadata['score_type'],'score_is_probability':False,
                'verification_method':'geometry_supported' if result[0][0,target].max()>0 else 'strict_ridge_fallback',
                'reference_image_id':reference.image_id if reference is not None else None,
                'reference_source':reference.source if reference is not None else None,
                'reference_author':reference.author if reference is not None else None,
                'reference_licence':reference.licence if reference is not None else None,**version}
