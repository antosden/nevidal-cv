"""Train-only Ridge над MobileNet-эмбеддингами и val-only per-POI отказ.

Это отдельный эксперимент: score является регрессионным откликом, не probability.
Глобальные бюджеты ошибок распределяются между POI динамическим программированием.
"""
from pathlib import Path
import json
import shutil
import math
import numpy as np
import pandas as pd
import torch

from .poi_baseline import (configure_torch,sha256_file,json_write,score_margins,create_encoder,load_rgb,
                          acceptance_matrix,poi_scores,POIVerifier,evaluate,l2_normalize)
from .poi_finetune import classification_metrics
from .negative_baseline import evaluate_external


def fit_ridge(train,labels,classes,alpha):
    x=torch.as_tensor(train,dtype=torch.float64)
    labels=torch.as_tensor(labels,dtype=torch.long)
    y=torch.nn.functional.one_hot(labels,classes).double()
    weights=1./torch.bincount(labels,minlength=classes)[labels].double()
    weights/=weights.mean()
    mean=(x*weights[:,None]).sum(0)/weights.sum()
    offset=(y*weights[:,None]).sum(0)/weights.sum()
    centered=x-mean
    covariance=centered.T@(centered*weights[:,None])
    coefficient=torch.linalg.solve(covariance+alpha*torch.eye(x.shape[1],dtype=torch.float64),
                                   centered.T@((y-offset)*weights[:,None]))
    return {'mean':mean.numpy(),'coefficient':coefficient.numpy(),'offset':offset.numpy()}


def ridge_scores(vectors,parameters):
    return (np.asarray(vectors,dtype=np.float64)-parameters['mean'])@parameters['coefficient']+parameters['offset']


def calibrate_per_poi(scores,truths,external,config):
    scores=np.asarray(scores,dtype=np.float64);external=np.asarray(external,dtype=np.float64)
    if not len(scores) or not len(external) or scores.shape[1]!=external.shape[1]:raise ValueError('Неполные validation scores.')
    if any(not 0<=config[key]<1 for key in ['target_external_far','target_hard_wrong_far','target_known_wrong_far']):
        raise ValueError('FAR должен быть в [0,1).')
    margin=score_margins(scores);neg_margin=score_margins(external)
    best=scores.argmax(1);neg_best=external.argmax(1)
    ext_budget=math.floor(config['target_external_far']*len(external)+1e-10)
    known_budget=min(math.floor(config['target_hard_wrong_far']*len(scores)+1e-10),
                     math.floor(config['target_known_wrong_far']*len(scores)*(scores.shape[1]-1)+1e-10))
    all_options=[]
    for index in range(scores.shape[1]):
        own=(truths==index)&(best==index)&(margin[:,index]>0)
        wrong=(truths!=index)&(best==index)&(margin[:,index]>0)
        neg=(neg_best==index)&(neg_margin[:,index]>0)
        reject=float(np.nextafter(max(scores[:,index].max(),external[:,index].max()),np.inf))
        thresholds=np.unique(np.r_[scores[own,index],reject])
        margins=np.unique(np.r_[0.,margin[own,index]])
        options={}
        for threshold in thresholds:
            for gap in margins:
                tp=int((own&(scores[:,index]>=threshold)&(margin[:,index]>=gap)).sum())
                fp=int((wrong&(scores[:,index]>=threshold)&(margin[:,index]>=gap)).sum())
                efp=int((neg&(external[:,index]>=threshold)&(neg_margin[:,index]>=gap)).sum())
                if efp>ext_budget or fp>known_budget:continue
                option=(tp,float(threshold),float(gap))
                if (efp,fp) not in options or option>options[efp,fp]:options[efp,fp]=option
        all_options.append(options)
    states={(0,0):(0,[])}
    for options in all_options:
        next_states={}
        for (ext_count,known_count),(tp,chosen) in states.items():
            for (extra_ext,extra_known),(extra_tp,threshold,gap) in options.items():
                key=(ext_count+extra_ext,known_count+extra_known)
                if key[0]>ext_budget or key[1]>known_budget:continue
                candidate=(tp+extra_tp,chosen+[(threshold,gap)])
                if key not in next_states or candidate[0]>next_states[key][0]:next_states[key]=candidate
        states=next_states
    cost,selected=max(states.items(),key=lambda item:(item[1][0],-sum(item[0]),-item[0][0]))
    policy={'threshold_type':'per_poi','score_thresholds':[p[0] for p in selected[1]],
        'margin_thresholds':[p[1] for p in selected[1]],'calibration_split':'val_and_negative_val',
        'true_accept_rate':selected[0]/len(scores),'external_any_poi_far':cost[0]/len(external),
        'hard_wrong_far':cost[1]/len(scores),'external_false_accepts':cost[0],'known_wrong_accepts':cost[1],
        'target_external_any_poi_far':config['target_external_far'],'target_hard_wrong_far':config['target_hard_wrong_far'],
        'external_unknown_calibrated':True,'score_is_probability':False,
        'decision_rule':'unique top POI AND score >= its threshold AND margin >= its threshold',
        'limitation':'Per-POI thresholds fitted to small validation; no out-of-sample FAR guarantee.'}
    accepted=acceptance_matrix(scores,policy);negative=acceptance_matrix(external,policy)
    assert int(accepted[np.arange(len(scores)),truths].sum())==selected[0]
    assert int(negative.any(axis=1).sum())==cost[0]
    return policy


def val_experiment(root,output,alphas=(.01,.1,1.)):
    base=root/'models/baseline_mobilenetv3_negatives'
    meta=json.loads((base/'metadata.json').read_text())
    for name,digest in meta['artifact_sha256'].items():
        if sha256_file(base/name)!=digest:raise ValueError(f'Изменён baseline {name}')
    configure_torch(42,4)
    config=json.loads((base/'config.json').read_text())
    experiment_config=json.loads((root/'config/mobilenetv3_calibrated.json').read_text())
    config.update(experiment_config)
    alphas=experiment_config['ridge_alphas']
    frames={s:pd.read_csv(base/f'splits/{s}.csv',keep_default_na=False).sort_values('image_id').reset_index(drop=True)
            for s in ['train','val','negative_val']}
    vectors={s:np.load(base/f'{s}_embeddings.npy',allow_pickle=False) for s in frames}
    pois=meta['poi_ids'];truth=np.array([pois.index(p) for p in frames['val'].poi_id])
    labels=np.array([pois.index(p) for p in frames['train'].poi_id])
    gallery=pd.read_csv(base/'gallery.csv',keep_default_na=False);g=np.load(base/'gallery_embeddings.npy',allow_pickle=False)
    results=[];best=None
    for alpha in [None,*alphas]:
        if alpha is None:
            parameters=None
            scores,_=poi_scores(vectors['val'],g,gallery.poi_id,pois)
            negatives,_=poi_scores(vectors['negative_val'],g,gallery.poi_id,pois)
        else:
            parameters=fit_ridge(vectors['train'],labels,len(pois),alpha)
            scores=ridge_scores(vectors['val'],parameters);negatives=ridge_scores(vectors['negative_val'],parameters)
        policy=calibrate_per_poi(scores,truth,negatives,config)
        quality=classification_metrics(scores,frames['val'],pois)
        result={'scorer':'gallery_cosine' if alpha is None else 'balanced_ridge','ridge_alpha':alpha,
                **quality,'true_accept_rate':policy['true_accept_rate'],'external_any_poi_far':policy['external_any_poi_far'],
                'hard_wrong_far':policy['hard_wrong_far']}
        results.append(result);print(json.dumps(result),flush=True)
        key=(result['true_accept_rate'],result['macro_f1'],result['accuracy'])
        if best is None or key>best[0]:best=(key,result,parameters,policy)
    output.mkdir(parents=True,exist_ok=False)
    json_write(output/'config.json',config)
    pd.DataFrame(results).to_csv(output/'validation_candidates.csv',index=False)
    json_write(output/'selection.json',{'selected':best[1],'selection_split':'val_and_negative_val','test_used':False})
    return best,frames,vectors,meta,config


def export_calibrated(root,output):
    best,frames,vectors,parent,config=val_experiment(root,output)
    _,chosen,parameters,policy=best
    base=root/'models/baseline_mobilenetv3_negatives'
    # Выбор зафиксирован до чтения test-эмбеддингов и расчёта test-метрик.
    json_write(output/'rejection_policy.json',policy)
    if parameters is not None:np.savez(output/'ridge_parameters.npz',**parameters)
    (output/'splits').mkdir()
    for name in ['imagenet_weights.pth','gallery.csv','gallery_embeddings.npy','class_mapping.json']:
        shutil.copyfile(base/name,output/name)
    for name in ['train','val','test','negative_val','negative_test']:
        shutil.copyfile(base/f'splits/{name}.csv',output/f'splits/{name}.csv')
    pois=parent['poi_ids'];gallery=pd.read_csv(output/'gallery.csv',keep_default_na=False)
    g=np.load(output/'gallery_embeddings.npy',allow_pickle=False)
    json_write(output/'thresholds_by_poi.json',{poi:{'score_threshold':policy['score_thresholds'][i],
                    'margin_threshold':policy['margin_thresholds'][i]} for i,poi in enumerate(pois)})
    metrics={}
    print('Validation selection and thresholds frozen; evaluating test.',flush=True)
    for name in ['val','test']:
        frame=pd.read_csv(output/f'splits/{name}.csv',keep_default_na=False).sort_values('image_id').reset_index(drop=True)
        raw=np.load(base/f'{name}_embeddings.npy',allow_pickle=False)
        gallery_scores,nearest=poi_scores(raw,g,gallery.poi_id,pois)
        scores=ridge_scores(raw,parameters) if parameters is not None else gallery_scores
        summary,per_poi,pairs,confusion=evaluate(scores,frame,pois,policy)
        summary.update(classification_metrics(scores,frame,pois))
        pairs['reference_image_id']=gallery.image_id.to_numpy()[nearest.flatten()]
        per_poi.to_csv(output/f'{name}_per_poi.csv',index=False);pairs.to_csv(output/f'{name}_pairs.csv',index=False)
        confusion.to_csv(output/f'{name}_confusion.csv')
        negative=pd.read_csv(output/f'splits/negative_{name}.csv',keep_default_na=False).sort_values('image_id').reset_index(drop=True)
        raw_negative=np.load(base/f'negative_{name}_embeddings.npy',allow_pickle=False)
        ns=ridge_scores(raw_negative,parameters) if parameters is not None else poi_scores(raw_negative,g,gallery.poi_id,pois)[0]
        external,queries,groups=evaluate_external(ns,negative,pois,policy)
        summary.update(external_unknown_tested=True,open_set_far=external['any_poi_far'],
                       open_set_far_definition='any_poi_per_external_image')
        queries.to_csv(output/f'negative_{name}_predictions.csv',index=False)
        groups.to_csv(output/f'negative_{name}_groups.csv',index=False)
        metrics[name]=summary;metrics[f'negative_{name}']=external
    json_write(output/'metrics.json',metrics)
    metadata={'schema_version':3,'model_type':'mobilenetv3_calibrated','architecture':'mobilenet_v3_small',
        'poi_ids':pois,'scorer':chosen['scorer'],'ridge_alpha':chosen['ridge_alpha'],
        'backbone_training_performed':False,'linear_head_training_performed':parameters is not None,
        'training_split':'positive_train_only','calibration_split':'positive_val_and_negative_val',
        'test_used_for_selection':False,'seed':42,'preprocessing_version':parent['preprocessing_version'],
        'reference_version':parent['reference_version'],'threshold_version':sha256_file(output/'rejection_policy.json')[:16],
        'parent_metadata_sha256':sha256_file(base/'metadata.json'),
        'input_split_sha256':{name:sha256_file(root/f'data/annotations/{name}.csv') for name in ['train','val','test','negative_val','negative_test']},
        'score_type':'balanced_ridge_response_not_probability' if parameters is not None else 'gallery_cosine_not_probability',
        'deployment_status':'not_promoted_far_exceeded' if metrics['negative_test']['any_poi_far']>config['target_external_far'] else 'experimental_not_production_validated',
        'versions':parent['versions'],
        'limitations':['Per-POI thresholds have few validation positives; FAR is not guaranteed on new data.',
                       'Positive-test is diagnostic; a new blind field test is needed.',
                       'External negative types are incompletely represented in test.']}
    metadata['artifact_sha256']={p.relative_to(output).as_posix():sha256_file(p) for p in sorted(output.rglob('*')) if p.is_file()}
    json_write(output/'metadata.json',metadata)
    write_metric_report(root,output,metadata,metrics)
    return metadata,metrics


def write_metric_report(root,output,metadata,metrics):
    report=root/'reports/mobilenetv3_calibrated';report.mkdir(parents=True,exist_ok=True)
    old=json.loads((root/'models/baseline_mobilenetv3_negatives/metrics.json').read_text())
    current=metrics['test'];external=metrics['negative_test']
    lines=['# MobileNetV3: улучшение классификатора и политики отказа','',
        f'Выбран scorer={metadata["scorer"]}, ridge_alpha={metadata["ridge_alpha"]}. Backbone ImageNet заморожен; обучаемый линейный слой использует только positive-train.',
        'Сбалансированная Ridge-регрессия к one-hot меткам, с intercept. Отклик не является вероятностью и не обязан лежать в [0,1].',
        'Отдельные score/margin-пороги каждого POI выбраны совместно динамическим программированием на positive-val и negative-val.',
        'Общие validation-бюджеты: external any-POI FAR ≤1%, hard-wrong FAR ≤5%, known-wrong FAR ≤1%. Цель — максимум true accept, затем macro-F1 и accuracy.',
        'Принимается только единственный лидирующий POI после прохождения его двух порогов. Галерея сохранена как контроль и источник ближайших эталонов; при scorer=Ridge решение принимает обученный слой.',
        '','| Test | Accuracy | Macro-F1 | True accept | External any-POI FAR |',
        '|---|---:|---:|---:|---:|',
        f'| Frozen Small + global thresholds | {old["test"]["top1_accuracy"]:.2%} | 73.26% | {old["test"]["true_accept_rate"]:.2%} | {old["negative_test"]["any_poi_far"]:.2%} |',
        f'| Выбранный вариант | {current["accuracy"]:.2%} | {current["macro_f1"]:.2%} | {current["true_accept_rate"]:.2%} | {external["any_poi_far"]:.2%} |','',
        '## Наблюдение → интерпретация → действие',
        f'- Верных подтверждений {current["true_accepts"]}/{current["positive_pairs"]} → true accept {current["true_accept_rate"]:.2%} → цель 85% {"достигнута на диагностическом test" if current["true_accept_rate"]>=.85 else "не достигнута"}.',
        f'- External false accepts {external["false_accepts_any_poi"]}/{external["images"]} → FAR {external["any_poi_far"]:.2%} → validation-ограничение не гарантирует test FAR; не корректировать пороги по этому test.',
        f'- Требование FAR ≤1% на test {"выполнено в этом измерении" if external["any_poi_far"]<=.01 else "НЕ выполнено"} → кандидат {"требует дальнейшей независимой проверки" if external["any_poi_far"]<=.01 else "не рекомендуется как замена строгой политики"} → предыдущая версия остаётся доступной.',
        '- Per-POI пороги дают больше гибкости, но на некоторые классы приходится 1–2 val-фото → высокий риск переучёта калибровки → считать выпуск экспериментальным и добрать независимые положительные/отрицательные кадры по классам.',
        '- CNN fine-tuning с новым preprocessing не улучшил validation true accept → этот вариант не заменяет baseline → история отдельного эксперимента сохранена в models/mobilenetv3_finetuned.',
        '- Гиперпараметры и пороги выбраны только на val/negative-val → test не участвовал в выборе текущего выпуска → итоговую проверку провести также на новом закрытом полевом наборе.',
        '- Атрибуция source/author/licence сохранена в snapshots и галерее → происхождение каждого изображения доступно → сохранять поля при экспорте.',
        '','## Проверенные варианты (validation)','```',pd.read_csv(output/'validation_candidates.csv').to_string(index=False),'```',
        '','Исходники, разбиение и предыдущие модели не изменялись. Автоматического переключения используемой модели нет.']
    (report/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');json_write(report/'metrics.json',metrics)


class CalibratedPOIVerifier:
    def __init__(self,directory):
        directory=Path(directory);self.metadata=json.loads((directory/'metadata.json').read_text())
        if self.metadata['schema_version']!=3:raise ValueError('Неверная схема.')
        for name,digest in self.metadata['artifact_sha256'].items():
            if sha256_file(directory/name)!=digest:raise ValueError(f'Изменён артефакт {name}')
        configure_torch(42,4)
        self.encoder,self.transform=create_encoder(directory/'imagenet_weights.pth')
        self.policy=json.loads((directory/'rejection_policy.json').read_text())
        self.gallery=pd.read_csv(directory/'gallery.csv',keep_default_na=False)
        self.values=np.load(directory/'gallery_embeddings.npy',allow_pickle=False)
        self.poi_ids=self.metadata['poi_ids']
        self.parameters=None
        if self.metadata['scorer']=='balanced_ridge':
            data=np.load(directory/'ridge_parameters.npz',allow_pickle=False)
            self.parameters={key:data[key] for key in ['mean','coefficient','offset']}

    def verify(self,image_path,target_poi_id):
        version={key:self.metadata[key] for key in ['reference_version','threshold_version','preprocessing_version']}
        if target_poi_id not in self.poi_ids:return {'accepted':False,'reason':'target_poi_not_supported','score':None,'margin':None,'target_poi_id':target_poi_id,**version}
        with torch.inference_mode():raw=self.encoder(self.transform(load_rgb(Path(image_path))).unsqueeze(0)).numpy()
        raw=l2_normalize(raw)
        gs,nearest=poi_scores(raw,self.values,self.gallery.poi_id,self.poi_ids)
        scores=ridge_scores(raw,self.parameters) if self.parameters is not None else gs
        index=self.poi_ids.index(target_poi_id);margin=float(score_margins(scores)[0,index]);score=float(scores[0,index])
        accepted=bool(acceptance_matrix(scores,self.policy)[0,index])
        reason='poi_verified' if accepted else 'poi_not_matched' if margin<=0 else 'poi_below_threshold' if score<self.policy['score_thresholds'][index] else 'poi_ambiguous'
        reference=self.gallery.iloc[nearest[0,index]]
        return {'accepted':accepted,'reason':reason,'target_poi_id':target_poi_id,'predicted_poi_id':self.poi_ids[int(scores[0].argmax())],
                'score':score,'margin':margin,'score_type':self.metadata['score_type'],'score_is_probability':False,
                'reference_image_id':reference.image_id,'reference_source':reference.source,'reference_author':reference.author,
                'reference_licence':reference.licence,**version}
