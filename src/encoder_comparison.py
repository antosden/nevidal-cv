"""Сравнение frozen encoder и train-галерей; выбор только по validation."""
from pathlib import Path
import json
import time
import platform
import hashlib
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from .poi_baseline import (configure_torch,create_encoder,load_rgb,l2_normalize,sha256_file,
    json_write,load_splits,select_gallery,poi_scores,calibrate_policy,evaluate,acceptance_matrix)
from .poi_finetune import classification_metrics
from .negative_baseline import evaluate_external


from .frozen_encoders import (DinoAttention,DinoScale,DinoMlp,DinoBlock,DinoPatch,DinoSmall,ENCODERS,load_encoder)


def gallery_variant(frame,vectors,size,method):
    limit=len(frame) if size=='all' else int(size)
    if method=='diverse':return select_gallery(frame,vectors,limit)
    selected=[]
    for _,group in frame.groupby('poi_id',sort=True):
        unique=group.sort_values('image_id').drop_duplicates('split_group')
        selected.extend(unique.sample(min(limit,len(unique)),random_state=42).index.tolist())
    result=frame.loc[selected].copy().reset_index(drop=True);result['gallery_index']=np.arange(len(result))
    return result,l2_normalize(vectors[selected])


def embeddings(root,frame,model,transform,output,name,encoder,config):
    path=output/f'{name}_embeddings.npy';index=output/f'{name}_embedding_index.csv'
    digest_path=output/f'{name}_embedding_cache.json'
    if path.exists():
        if not digest_path.exists():raise ValueError('Незавершённый кэш embeddings: удалите только этот кэш для повторения')
        saved=json.loads(digest_path.read_text())
        if sha256_file(path)!=saved['array_sha256'] or sha256_file(index)!=saved['index_sha256']:raise ValueError('Изменён кэш embeddings')
        if not index.exists() or pd.read_csv(index,keep_default_na=False).image_id.tolist()!=frame.image_id.tolist():raise ValueError('Неверный индекс кэша embeddings')
        value=np.load(path,allow_pickle=False)
        if value.shape!=(len(frame),ENCODERS[encoder]['dimension']):raise ValueError('Неверная форма embeddings')
        return value
    # Переиспользуем только совпадающие source SHA и точно тот же encoder/preprocessing.
    cached={}
    if encoder=='mobilenetv3_small':
        parent=root/'models/baseline_mobilenetv3_negatives'
        meta=json.loads((parent/'metadata.json').read_text())
        for split in ['train','val','test','negative_val','negative_test']:
            if sha256_file(parent/f'splits/{split}.csv')!=meta['artifact_sha256'][f'splits/{split}.csv']:raise ValueError('Изменён прежний snapshot')
            old_frame=pd.read_csv(parent/f'splits/{split}.csv',keep_default_na=False).sort_values('image_id')
            old_file=parent/f'{split}_embeddings.npy'
            if sha256_file(old_file)!=meta['artifact_sha256'][old_file.name]:raise ValueError('Изменены прежние embeddings')
            old_vectors=np.load(old_file,allow_pickle=False)
            cached.update({(row.image_id,row.sha256):old_vectors[i] for i,row in enumerate(old_frame.itertuples())})
    result=np.zeros((len(frame),ENCODERS[encoder]['dimension']),dtype=np.float32);pending=[]
    completed=np.zeros(len(frame),dtype=bool)
    signature=hashlib.sha256(json.dumps({'weights':json.loads((output/'weights.json').read_text()),
        'images':frame[['image_id','sha256']].values.tolist()},sort_keys=True).encode()).hexdigest()
    partial=output/f'{name}_partial.npz'
    if partial.exists():
        with np.load(partial,allow_pickle=False) as stored:
            if stored['signature'].item()!=signature:raise ValueError('Partial embeddings относятся к другому входу')
            result=stored['embeddings'];completed=stored['completed']
        if result.shape!=(len(frame),ENCODERS[encoder]['dimension']) or completed.shape!=(len(frame),):raise ValueError('Неверная форма partial embeddings')
        if not np.isfinite(result[completed]).all():raise ValueError('Повреждён partial embeddings')
    for index_number,row in enumerate(frame.itertuples()):
        if completed[index_number]:continue
        if (row.image_id,row.sha256) in cached:
            result[index_number]=cached[(row.image_id,row.sha256)];completed[index_number]=True
        else:pending.append(index_number)
    started=time.perf_counter()
    for start in tqdm(range(0,len(pending),config['batch_size']),desc=f'{encoder}/{name}'):
        positions=pending[start:start+config['batch_size']]
        batch=[transform(load_rgb(root/frame.iloc[i].file_path)) for i in positions]
        with torch.inference_mode():result[positions]=l2_normalize(model(torch.stack(batch)).numpy())
        completed[positions]=True
        temporary=output/f'{name}_partial.tmp.npz'
        np.savez(temporary,embeddings=result,completed=completed,signature=np.array(signature))
        temporary.replace(partial)
    result=l2_normalize(result)
    frame[['image_id','file_path','sha256','source','author','licence']].to_csv(index,index=False)
    np.save(path,result,allow_pickle=False)
    json_write(digest_path,{'array_sha256':sha256_file(path),'index_sha256':sha256_file(index)})
    if partial.exists():partial.unlink()
    json_write(output/f'{name}_extraction.json',{'cached_images':len(frame)-len(pending),'extracted_images':len(pending),
        'wall_seconds':time.perf_counter()-started,'includes_decode':True})
    return result


def quality(scores,frame,pois,policy,negative_scores,negative):
    summary,per_poi,pairs,confusion=evaluate(scores,frame,pois,policy)
    summary.update(classification_metrics(scores,frame,pois))
    external,queries,groups=evaluate_external(negative_scores,negative,pois,policy)
    summary.update(external_any_poi_far=external['any_poi_far'],external_unknown_tested=True,
                   open_set_far=external['any_poi_far'],open_set_far_definition='any_poi_per_external_image')
    return summary,external,per_poi,pairs,confusion,queries,groups


def run_encoder(root,output,name,frames,mapping,config,stage='all'):
    directory=output/name;directory.mkdir(exist_ok=True)
    if (directory/'metadata.json').exists():
        meta=json.loads((directory/'metadata.json').read_text())
        for file,digest in meta['artifact_sha256'].items():
            if sha256_file(directory/file)!=digest:raise ValueError('Изменён завершённый выпуск')
        return 'complete'
    if stage=='test' and not (directory/'selection.json').exists():raise ValueError('Сначала зафиксируйте validation-выбор')
    weights=root/'.runtime/torch/checkpoints'/ENCODERS[name]['file']
    if not weights.exists():return 'missing_weights'
    checksum=sha256_file(weights);lock=directory/'weights.json'
    if lock.exists() and json.loads(lock.read_text())['sha256']!=checksum:raise ValueError('Веса изменены при продолжении')
    model,transform=load_encoder(name,weights)
    json_write(lock,{'file':weights.relative_to(root).as_posix(),'sha256':checksum,'official_url':ENCODERS[name]['url'],
        'transform':str(transform),'torch':torch.__version__})
    pois=sorted(mapping,key=mapping.get)
    values={s:embeddings(root,frames[s],model,transform,directory,s,name,config) for s in ['train','val','negative_val']}
    selection_path=directory/'selection.json'
    if not selection_path.exists():
        candidates=[];truth=np.array([pois.index(p) for p in frames['val'].poi_id])
        for size in config['gallery_sizes']:
            for method in config['gallery_methods']:
                if size=='all' and method!='diverse':continue
                key=f'{method}_{size}';folder=directory/key;folder.mkdir(exist_ok=True)
                gallery,vectors=gallery_variant(frames['train'],values['train'],size,method)
                scores,_=poi_scores(values['val'],vectors,gallery.poi_id,pois)
                neg,_=poi_scores(values['negative_val'],vectors,gallery.poi_id,pois)
                policy,grid=calibrate_policy(scores,truth,config,neg)
                summary,external,*_=quality(scores,frames['val'],pois,policy,neg,frames['negative_val'])
                gallery.to_csv(folder/'gallery.csv',index=False);np.save(folder/'gallery_embeddings.npy',vectors,allow_pickle=False)
                json_write(folder/'policy.json',policy);json_write(folder/'validation_metrics.json',summary)
                grid.to_csv(folder/'threshold_search.csv',index=False)
                candidates.append(dict(encoder=name,variant=key,gallery_images=len(gallery),**summary))
        pd.DataFrame(candidates).to_csv(directory/'validation_candidates.csv',index=False)
        best=max(candidates,key=lambda c:(c['true_accept_rate'],-c['external_any_poi_far'],c['macro_f1'],c['accuracy'],-c['gallery_images']))
        json_write(selection_path,{'variant':best['variant'],'selection_split':'val_and_negative_val','test_used_for_selection':False,
            'key':['max_true_accept','min_external_far','max_macro_f1','max_accuracy','min_gallery_size']})
    selected=json.loads(selection_path.read_text());folder=directory/selected['variant']
    if stage=='validation':return 'validation_complete'
    gallery=pd.read_csv(folder/'gallery.csv',keep_default_na=False);vectors=np.load(folder/'gallery_embeddings.npy',allow_pickle=False)
    policy=json.loads((folder/'policy.json').read_text())
    # Test-фото обрабатываются только после фиксации выбора галереи и порогов.
    values.update({s:embeddings(root,frames[s],model,transform,directory,s,name,config) for s in ['test','negative_test']})
    metrics={}
    for split in ['val','test']:
        scores,_=poi_scores(values[split],vectors,gallery.poi_id,pois)
        neg,_=poi_scores(values[f'negative_{split}'],vectors,gallery.poi_id,pois)
        summary,external,per_poi,pairs,confusion,queries,groups=quality(scores,frames[split],pois,policy,neg,frames[f'negative_{split}'])
        metrics[split]=summary;metrics[f'negative_{split}']=external
        per_poi.to_csv(directory/f'{split}_per_poi.csv',index=False);pairs.to_csv(directory/f'{split}_pairs.csv',index=False)
        confusion.to_csv(directory/f'{split}_confusion.csv');queries.to_csv(directory/f'negative_{split}_predictions.csv',index=False)
        groups.to_csv(directory/f'negative_{split}_groups.csv',index=False)
        np.save(directory/f'{split}_scores.npy',scores,allow_pickle=False);np.save(directory/f'negative_{split}_scores.npy',neg,allow_pickle=False)
    json_write(directory/'metrics.json',metrics)
    metadata={'encoder':name,'dimension':ENCODERS[name]['dimension'],'weights_sha256':checksum,'training_performed':False,
        'torch':torch.__version__,'python':platform.python_version(),'input_resolution':224,'poi_ids':pois,
        'deployment_status':'experimental_not_promoted','dataset_snapshot_sha256':sha256_file(output/'input_snapshot.json')}
    metadata['artifact_sha256']={p.relative_to(directory).as_posix():sha256_file(p) for p in sorted(directory.rglob('*')) if p.is_file()}
    json_write(directory/'metadata.json',metadata)
    return 'complete'


def compare(root,output,config,only_encoder=None,stage='all',reuse_input_audit=False):
    if config['seed']!=42:raise ValueError('Протокол требует SEED=42')
    configure_torch(42,4);splits,mapping=load_splits(root)
    frames={**splits,**{f'negative_{s}':pd.read_csv(root/f'data/annotations/negative_{s}.csv',keep_default_na=False).sort_values('image_id').reset_index(drop=True) for s in ['val','test']}}
    snapshot={name:sha256_file(root/f'data/annotations/{name}.csv') for name in frames}
    snapshot['class_mapping']=sha256_file(root/'data/annotations/class_mapping.json');snapshot['config']=config
    snapshot_path=output/'input_snapshot.json'
    if snapshot_path.exists() and json.loads(snapshot_path.read_text())!=snapshot:raise ValueError('Датасет/конфигурация изменились: нужен новый output')
    if reuse_input_audit and not snapshot_path.exists():raise ValueError('Нет ранее завершённого входного аудита')
    cutoff=snapshot_path.stat().st_mtime_ns if reuse_input_audit else None
    audited=0;reused=0
    for name,frame in frames.items():
        if not frame.label_status.eq('verified').all():raise ValueError('Непроверенные метки')
        for row in tqdm(frame.itertuples(),total=len(frame),desc=f'Input SHA/{name}'):
            stat=(root/row.file_path).stat()
            size=getattr(row,'file_size',stat.st_size)
            if cutoff is not None and stat.st_mtime_ns<=cutoff and stat.st_size==int(size):
                reused+=1;continue
            if sha256_file(root/row.file_path)!=row.sha256:raise ValueError(f'Изменён raw {row.image_id}')
            audited+=1
    for n in ['negative_val','negative_test']:
        for positive in splits.values():
            for field in ['image_id','sha256','commons_file_id']:
                if (set(frames[n][field].astype(str))-{''})&(set(positive[field].astype(str))-{''}):raise ValueError('Positive/negative overlap')
    for field in ['image_id','sha256','commons_file_id','duplicate_group','split_group','source_poi_id']:
        if (set(frames['negative_val'][field].astype(str))-{''})&(set(frames['negative_test'][field].astype(str))-{''}):raise ValueError('Negative val/test leakage')
    output.mkdir(parents=True,exist_ok=True)
    if not snapshot_path.exists():json_write(snapshot_path,snapshot)
    json_write(output/'last_input_audit.json',{'sha256_read_now':audited,'previous_audit_reused_by_mtime_and_size':reused,
        'reuse_input_audit':reuse_input_audit,'note':'Reuse is a filesystem metadata cache, not a fresh cryptographic verification.'})
    (output/'splits').mkdir(exist_ok=True)
    for name,frame in frames.items():frame.to_csv(output/f'splits/{name}.csv',index=False)
    json_write(output/'class_mapping.json',mapping)
    status={}
    for name in config['encoders']:
        if only_encoder is not None and name!=only_encoder:
            status[name]='complete' if (output/name/'metadata.json').exists() else 'validation_complete' if (output/name/'selection.json').exists() else 'pending'
            continue
        print('Encoder:',name,flush=True)
        try:status[name]=run_encoder(root,output,name,frames,mapping,config,stage)
        except (OSError,ValueError,RuntimeError) as error:
            status[name]='failed: '+str(error);json_write(output/'status.json',status)
            print(name,status[name],flush=True)
    json_write(output/'status.json',status);write_report(root,output,status)
    return status


def write_report(root,output,status):
    report=root/'reports/encoder_comparison';report.mkdir(parents=True,exist_ok=True)
    rows=[];candidates=[];source_rows=[]
    for name,state in status.items():
        if state!='complete':continue
        directory=output/name;metrics=json.loads((directory/'metrics.json').read_text());selection=json.loads((directory/'selection.json').read_text())
        candidates.append(pd.read_csv(directory/'validation_candidates.csv'))
        for split in ['val','test']:rows.append(dict(encoder=name,split=split,variant=selection['variant'],**metrics[split]))
        pois=json.loads((directory/'metadata.json').read_text())['poi_ids']
        policy=json.loads((directory/selection['variant']/'policy.json').read_text())
        for split in ['val','test']:
            frame=pd.read_csv(output/f'splits/{split}.csv',keep_default_na=False)
            scores=np.load(directory/f'{split}_scores.npy',allow_pickle=False)
            accepted=acceptance_matrix(scores,policy);truth=np.array([pois.index(p) for p in frame.poi_id])
            for source,group in frame.groupby('source'):
                positions=group.index.to_numpy()
                source_rows.append(dict(encoder=name,split=split,source=source,images=len(group),poi_count=group.poi_id.nunique(),
                    accuracy=float((scores[positions].argmax(1)==truth[positions]).mean()),
                    true_accept_rate=float(accepted[positions,truth[positions]].mean())))
    table=pd.DataFrame(rows);table.to_csv(report/'comparison.csv',index=False)
    pd.DataFrame(source_rows).to_csv(report/'by_source.csv',index=False)
    if candidates:pd.concat(candidates).to_csv(report/'validation_candidates.csv',index=False)
    lines=['# Сравнение frozen encoder и галерей','',
        'Все encoder используют один обогащённый датасет и train-only галереи. Выбор размера/способа галереи и глобальных порогов — только по val/negative-val.',
        'Размеры галерей: 5, 10, 20 и все независимые train-группы на POI; diverse/farthest-first и random (SEED=42). Score=max cosine, без Ridge и геометрии.',
        'Каждый encoder использует свой документированный preprocessing, crop 224. DINOv2 CLS 384D — eval-only реализация совместимой архитектуры, загрузка официальных весов strict=True; 224 px выбран для CPU, это не максимум качества DINOv2.',
        'Изменения данных не сравниваются напрямую с прежними версиями: честное сравнение encoder — внутри этого выпуска.',
        'Текущий test диагностический: он уже использовался в предыдущих экспериментах. Нужен новый закрытый полевой test. Перебор галерей может давать оптимистичную val-оценку.',
        '', '| Encoder | Статус |','|---|---|']
    lines += [f'| {name} | {state} |' for name,state in status.items()]
    lines += ['', '| Encoder | Split | Gallery | Accuracy | Macro-F1 | True accept | External FAR |','|---|---|---|---:|---:|---:|---:|']
    for row in rows:lines.append(f'| {row["encoder"]} | {row["split"]} | {row["variant"]} | {row["accuracy"]:.2%} | {row["macro_f1"]:.2%} | {row["true_accept_rate"]:.2%} | {row["external_any_poi_far"]:.2%} |')
    if len(candidates)==len(ENCODERS):
        chosen=table[table.split.eq('val')].sort_values(['true_accept_rate','external_any_poi_far','macro_f1','accuracy'],ascending=[False,True,False,False]).iloc[0]
        json_write(output/'selection.json',{'encoder':chosen.encoder,'variant':chosen.variant,'selection_split':'val_and_negative_val','test_used_for_selection':False})
        lines += ['',f'Предпочтительный encoder по validation: {chosen.encoder}, {chosen.variant}. Автоматической замены основной модели нет.']
        tested=table[(table.encoder.eq(chosen.encoder))&table.split.eq('test')].iloc[0]
        lines += [f'На диагностическом test выбранного варианта: true accept {tested.true_accept_rate:.2%}, external FAR {tested.external_any_poi_far:.2%}. Цель 85% true accept {"достигнута только на этом диагностическом наборе" if tested.true_accept_rate>=.85 else "не достигнута"}.',
            'Ограничение external FAR ≤1% на test '+('выполнено эмпирически; для окончательной оценки нужен новый закрытый полевой набор.' if tested.external_any_poi_far<=.01 else 'не выполнено; вариант остаётся экспериментальным.')]
    else:lines += ['','Сравнение трёх encoder НЕ ЗАВЕРШЕНО: отсутствующие результаты не оцениваются и не заменяются случайными весами.']
    lines += ['','## Наблюдение → интерпретация → действие','',
        '- Малые классы и зависимые ракурсы → метрики нестабильны → смотреть per-POI таблицы и добирать независимые мобильные кадры.',
        '- После полного добавления manual меняются размер и баланс классов → общий прирост не доказывает улучшение на мобильных фото → смотреть by_source.csv; состав POI у источников различается, поэтому это диагностика, а не причинное сравнение источников.',
        '- Калибровка ограничивает эмпирический external FAR на val → это не гарантия FAR на новых объектах → оценивать отдельный test и не подстраивать по нему пороги.',
        '- Source/author/licence сохранены в snapshots и каждой галерее → происхождение прослеживается → сохранять атрибуцию при последующем использовании.']
    for name,state in status.items():
        if state!='complete':continue
        group=table[table.encoder.eq(name)];test=group[group.split.eq('test')].iloc[0]
        if test.external_any_poi_far>.01:
            lines.append(f'- {name}: test FAR {test.external_any_poi_far:.2%} >1% → validation-порог не удержал ограничение на других негативных POI → не менять его по test, сохранить вариант экспериментальным.')
        all_candidates=pd.read_csv(output/name/'validation_candidates.csv')
        highest=all_candidates.sort_values('accuracy',ascending=False).iloc[0]
        selected=group[group.split.eq('val')].iloc[0]
        if highest.variant!=selected.variant:
            lines.append(f'- {name}: максимальная val accuracy {highest.accuracy:.2%} у {highest.variant}, true accept {highest.true_accept_rate:.2%}; выбран {selected.variant} с true accept {selected.true_accept_rate:.2%} → ранжирование и подтверждение цели дают разный выбор галереи → оценивать true accept совместно с FAR.')
    (report/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
