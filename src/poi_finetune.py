"""Дообучение MobileNetV3 на train POI и воспроизводимый эксперимент отказа.

Негативы не используются в градиентах. Эпоха, галерея и пороги выбираются по val.
Test открывается для оценки только после фиксации выбранной эпохи и политики.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from tqdm.auto import tqdm

from .poi_baseline import (configure_torch, create_encoder, load_splits, safe_project_path,
    sha256_file, json_write, l2_normalize, select_gallery, poi_scores, calibrate_policy,
    acceptance_matrix, score_margins, evaluate, WEIGHTS_SHA256)
from .negative_baseline import evaluate_external

PREPROCESSING_VERSION = 'mobilenetv3-ft-exif-draft640-thumb320-letterbox224-v1'
FILL = (124,116,104)


def decode_thumbnail(path: Path) -> Image.Image:
    """Тот же путь декодирования в кэше и при одиночном выводе."""
    with Image.open(path) as original:
        original.seek(0)
        # JPEG draft уменьшает затраты декодирования больших оригиналов Commons.
        original.draft('RGB',(640,640))
        image=ImageOps.exif_transpose(original).convert('RGB')
        image.thumbnail((320,320),Image.Resampling.LANCZOS)
        return image.copy()


def letterbox(image: Image.Image, size=224):
    fitted=ImageOps.contain(image,(size,size),Image.Resampling.BILINEAR)
    canvas=Image.new('RGB',(size,size),FILL)
    canvas.paste(fitted,((size-fitted.width)//2,(size-fitted.height)//2))
    return canvas


def image_transform(training=False):
    pipeline=[transforms.Lambda(letterbox)]
    if training:
        pipeline += [transforms.RandomAffine(5,translate=(.03,.03),scale=(.95,1.05),fill=FILL),
                     transforms.RandomHorizontalFlip(.3),
                     transforms.ColorJitter(brightness=.15,contrast=.15,saturation=.1,hue=.02)]
    pipeline += [transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]
    return transforms.Compose(pipeline)


def cache_images(root:Path,frame:pd.DataFrame,cache:Path):
    cache.mkdir(parents=True,exist_ok=True)
    images=[];records=[]
    for row in tqdm(frame.itertuples(),total=len(frame),desc='Подготовка изображений'):
        relative=getattr(row,'processed_path','') or row.file_path
        source=safe_project_path(root,relative)
        dest=cache/f'{row.sha256}.png'
        if not dest.exists():
            if sha256_file(source)!=row.sha256:raise ValueError(f'Изменён исходник {row.image_id}')
            image=decode_thumbnail(source)
            temporary=dest.with_suffix('.png.tmp')
            image.save(temporary,format='PNG');temporary.replace(dest)
            json_write(dest.with_suffix('.json'),{'source_sha256':row.sha256,'cache_sha256':sha256_file(dest),
                                                'preprocessing_version':PREPROCESSING_VERSION})
        provenance=dest.with_suffix('.json')
        if not provenance.exists():raise ValueError(f'Нет атрибуции кэша: {dest}')
        cached=json.loads(provenance.read_text())
        if cached!={'source_sha256':row.sha256,'cache_sha256':sha256_file(dest),'preprocessing_version':PREPROCESSING_VERSION}:
            raise ValueError(f'Кэш изменён: {dest}')
        with Image.open(dest) as image:images.append(image.convert('RGB').copy())
        records.append({'image_id':row.image_id,'file_path':row.file_path,'source_sha256':row.sha256,
                        'cache_file':dest.relative_to(root).as_posix(),'cache_sha256':sha256_file(dest)})
    return images,pd.DataFrame(records)


def balanced_batches(frame,classes_per_batch,images_per_class,steps,seed):
    """P×K: классы равновероятны, внутри POI разные независимые группы."""
    rng=np.random.default_rng(seed)
    labels=sorted(frame.class_index.unique())
    if classes_per_batch>len(labels):raise ValueError('P больше числа классов.')
    per_class={label:frame[frame.class_index.eq(label)].drop_duplicates('split_group').index.to_numpy() for label in labels}
    if min(map(len,per_class.values()))<images_per_class:raise ValueError('Меньше K независимых train-групп в классе.')
    for _ in range(steps):
        selected=rng.choice(labels,size=classes_per_batch,replace=False)
        yield np.concatenate([rng.choice(per_class[label],size=images_per_class,replace=False) for label in selected])


def supervised_contrastive_loss(features,labels,temperature=.1):
    """SupCon для разных изображений одного POI; self-пары исключены."""
    normalized=F.normalize(features,dim=1)
    similarities=normalized@normalized.T/temperature
    diagonal=torch.eye(len(labels),device=features.device,dtype=torch.bool)
    positives=labels[:,None].eq(labels[None,:])&~diagonal
    similarities=similarities.masked_fill(diagonal,-torch.inf)
    log_probability=similarities-torch.logsumexp(similarities,dim=1,keepdim=True)
    counts=positives.sum(dim=1)
    valid=counts>0
    if not valid.any():return features.sum()*0.
    selected=log_probability.masked_fill(~positives,0.).sum(dim=1)
    return -(selected[valid]/counts[valid]).mean()


class CosineHead(nn.Module):
    def __init__(self,dimension,classes,scale=16.):
        super().__init__()
        self.weight=nn.Parameter(torch.empty(classes,dimension))
        nn.init.normal_(self.weight,std=.02)
        self.scale=scale

    def forward(self,features):
        return self.scale*F.linear(F.normalize(features,dim=1),F.normalize(self.weight,dim=1))


def freeze_for_finetune(encoder,last_blocks):
    encoder.requires_grad_(False)
    for block in list(encoder.features.children())[-last_blocks:]:block.requires_grad_(True)
    # Статистики BN и affine-параметры не переоцениваем на небольшом датасете.
    for module in encoder.modules():
        if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval().requires_grad_(False)
    encoder.eval()


def embed_images(encoder,images,batch_size=24):
    transform=image_transform(False)
    encoder.eval();result=[]
    with torch.inference_mode():
        for start in range(0,len(images),batch_size):
            batch=torch.stack([transform(image) for image in images[start:start+batch_size]])
            result.append(encoder(batch).numpy())
    return l2_normalize(np.concatenate(result))


def classification_metrics(scores,frame,poi_ids):
    truths=np.array([poi_ids.index(poi) for poi in frame.poi_id])
    predicted=np.argmax(scores,axis=1)
    f1=[]
    for index in range(len(poi_ids)):
        tp=int(((truths==index)&(predicted==index)).sum())
        fp=int(((truths!=index)&(predicted==index)).sum())
        fn=int(((truths==index)&(predicted!=index)).sum())
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return {'accuracy':float((truths==predicted).mean()),'macro_f1':float(np.mean(f1))}


def fit_embedding_transform(train,mode,shrinkage=.05):
    """Параметры метрики определяются только из train; никаких val/test в fit."""
    train=np.asarray(train,dtype=np.float64)
    if train.ndim!=2 or len(train)<2 or not np.isfinite(train).all():raise ValueError('Нужна конечная train-матрица из ≥2 строк.')
    dimension=train.shape[1]
    mean=np.zeros(dimension) if mode=='identity' else train.mean(axis=0)
    matrix=np.eye(dimension)
    if mode=='pca128_whiten':
        centered=torch.as_tensor(train-mean,dtype=torch.float64)
        covariance=centered.T@centered/max(1,len(train)-1)
        eigenvalues,eigenvectors=torch.linalg.eigh(covariance)
        dimensions=min(128,len(train)-1,dimension)
        values=eigenvalues[-dimensions:].clamp_min(0)
        vectors=eigenvectors[:,-dimensions:]
        # Фиксируем знак собственных векторов для повторяемого экспорта.
        peaks=torch.argmax(vectors.abs(),dim=0)
        signs=torch.sign(vectors[peaks,torch.arange(dimensions)])
        vectors=vectors*signs
        regularizer=max(float(eigenvalues[-1])*shrinkage,1e-9)
        matrix=(vectors/torch.sqrt(values+regularizer)[None,:]).numpy()
    elif mode not in ['identity','center']:
        raise ValueError(f'Неизвестное преобразование эмбеддингов: {mode}')
    return {'mode':mode,'mean':mean,'matrix':matrix}


def transform_embeddings(raw,transform):
    centered=np.asarray(raw,dtype=np.float64)-transform['mean']
    values=centered if transform.get('mode') in ['identity','center'] else centered@transform['matrix']
    return l2_normalize(values)


def validation_state(encoder,images,frames,poi_ids,config):
    raw={name:embed_images(encoder,images[name]) for name in ['train','val','negative_val']}
    best=None
    for mode in config['embedding_transforms']:
        transform=fit_embedding_transform(raw['train'],mode,config['pca_shrinkage'])
        vectors={name:transform_embeddings(values,transform) for name,values in raw.items()}
        gallery,values=select_gallery(frames['train'],vectors['train'],config['gallery_max_per_poi'])
        scores,_=poi_scores(vectors['val'],values,gallery.poi_id,poi_ids)
        negatives,_=poi_scores(vectors['negative_val'],values,gallery.poi_id,poi_ids)
        truths=np.array([poi_ids.index(poi) for poi in frames['val'].poi_id])
        policy,grid=calibrate_policy(scores,truths,config,negatives)
        quality=classification_metrics(scores,frames['val'],poi_ids)
        quality.update(true_accept_rate=policy['true_accept_rate'],external_any_poi_far=policy['external_any_poi_far'],
                       known_hard_wrong_far=policy['hard_wrong_far'],embedding_transform=mode)
        key=(quality['true_accept_rate'],quality['macro_f1'],quality['accuracy'])
        if best is None or key>best[0]:best=(key,quality,policy,grid,vectors,gallery,values,transform)
    return best[1:]


def load_training_frames(root,base):
    splits,mapping=load_splits(root)
    metadata=json.loads((base/'metadata.json').read_text())
    for name,digest in metadata['artifact_sha256'].items():
        if sha256_file(base/name)!=digest:raise ValueError(f'Исходный baseline изменён: {name}')
    for name,digest in metadata['split_sha256'].items():
        if sha256_file(root/f'data/annotations/{name}.csv')!=digest:raise ValueError('Positive splits изменились.')
    for name,digest in metadata['negative_split_sha256'].items():
        if sha256_file(root/f'data/annotations/negative_{name}.csv')!=digest:raise ValueError('Negative splits изменились.')
    frames={**splits,**{f'negative_{name}':pd.read_csv(root/f'data/annotations/negative_{name}.csv',keep_default_na=False) for name in ['val','test']}}
    for frame in frames.values():
        if frame.empty or not frame.label_status.eq('verified').all():raise ValueError('Нужны подтверждённые непустые выборки.')
    for a,b in [('train','negative_val'),('train','negative_test'),('val','negative_val'),('test','negative_test'),('negative_val','negative_test')]:
        for field in ['image_id','sha256','commons_file_id']:
            if (set(frames[a][field].astype(str))-{''})&(set(frames[b][field].astype(str))-{''}):raise ValueError(f'Утечка {a}/{b}: {field}')
    return frames,mapping,metadata


def train_model(root,output,config_path):
    if output.exists():raise FileExistsError('Нужен новый каталог эксперимента; существующий не перезаписывается.')
    config=json.loads(config_path.read_text())
    if config['seed']!=42:raise ValueError('SEED должен быть 42.')
    configure_torch(config['seed'],config['threads'])
    base=root/'models/baseline_mobilenetv3_negatives'
    frames,mapping,parent=load_training_frames(root,base)
    poi_ids=sorted(mapping,key=mapping.get)
    encoder,_=create_encoder(base/'imagenet_weights.pth')
    freeze_for_finetune(encoder,config['unfreeze_last_blocks'])
    head=CosineHead(576,len(mapping),config['cosine_logit_scale'])
    optimizer=torch.optim.AdamW([
        {'params':[p for p in encoder.parameters() if p.requires_grad],'lr':config['encoder_learning_rate']},
        {'params':head.parameters(),'lr':config['head_learning_rate']}],weight_decay=config['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['max_epochs'])
    output.mkdir(parents=True);(output/'splits').mkdir()
    json_write(output/'config.json',config);json_write(output/'class_mapping.json',mapping)
    for name,frame in frames.items():frame.to_csv(output/f'splits/{name}.csv',index=False)
    cache=root/'.runtime/finetune_cache'/PREPROCESSING_VERSION
    images={};cache_indices=[]
    # Ни содержимое positive-test, ни negative-test не открываются до выбора эпохи.
    for name in ['train','val','negative_val']:
        images[name],index=cache_images(root,frames[name],cache)
        cache_indices.append(index.assign(split=name))
    pd.concat(cache_indices).to_csv(output/'training_cache_index.csv',index=False)
    transform=image_transform(True)
    labels=torch.tensor(frames['train'].class_index.to_numpy(),dtype=torch.long)
    quality,policy,grid,_,_,_,_=validation_state(encoder,images,frames,poi_ids,config)
    history=[{'epoch':0,'loss':None,**quality}]
    # Frozen encoder с новым preprocessing — обязательная точка сравнения.
    best_key=(quality['true_accept_rate'],quality['macro_f1'],quality['accuracy'])
    best_epoch=0;stale=0
    torch.save({'encoder':encoder.state_dict(),'head':head.state_dict(),'epoch':0},output/'best_checkpoint.pt')
    print(json.dumps(history[-1]),flush=True)
    for epoch in range(1,config['max_epochs']+1):
        started=time.monotonic();losses=[]
        encoder.eval();head.train()
        for indices in balanced_batches(frames['train'],config['classes_per_batch'],config['images_per_class'],
                                         config['steps_per_epoch'],config['seed']+epoch):
            batch=torch.stack([transform(images['train'][i]) for i in indices])
            target=labels[indices]
            optimizer.zero_grad(set_to_none=True)
            features=encoder(batch)
            loss=F.cross_entropy(head(features),target,label_smoothing=.05)
            loss=loss+config['contrastive_weight']*supervised_contrastive_loss(features,target,config['contrastive_temperature'])
            if not torch.isfinite(loss):raise ValueError('Неконечная функция потерь.')
            loss.backward();torch.nn.utils.clip_grad_norm_([p for p in encoder.parameters() if p.requires_grad]+list(head.parameters()),5.)
            optimizer.step();losses.append(float(loss.detach()))
        scheduler.step()
        quality,policy,grid,_,_,_,_=validation_state(encoder,images,frames,poi_ids,config)
        history.append({'epoch':epoch,'loss':float(np.mean(losses)),'elapsed_seconds':time.monotonic()-started,**quality})
        key=(quality['true_accept_rate'],quality['macro_f1'],quality['accuracy'])
        if key>best_key:
            best_key=key;best_epoch=epoch;stale=0
            torch.save({'encoder':encoder.state_dict(),'head':head.state_dict(),'epoch':epoch},output/'best_checkpoint.pt')
        else:stale+=1
        pd.DataFrame(history).to_csv(output/'training_history.csv',index=False)
        print(json.dumps(history[-1]),flush=True)
        if stale>=config['patience']:break
    checkpoint=torch.load(output/'best_checkpoint.pt',map_location='cpu',weights_only=True)
    encoder.load_state_dict(checkpoint['encoder']);encoder.eval().requires_grad_(False)
    quality,policy,grid,vectors,gallery,gallery_values,embedding_transform=validation_state(encoder,images,frames,poi_ids,config)
    torch.save(encoder.state_dict(),output/'encoder.pt')
    json_write(output/'rejection_policy.json',policy)
    grid.to_csv(output/'val_threshold_search.csv',index=False)
    gallery.to_csv(output/'gallery.csv',index=False)
    np.save(output/'gallery_embeddings.npy',gallery_values,allow_pickle=False)
    np.savez(output/'embedding_transform.npz',mean=embedding_transform['mean'],matrix=embedding_transform['matrix'])
    json_write(output/'selection.json',{'best_epoch':best_epoch,'selection_split':'val_and_negative_val',
                'test_used_for_selection':False,'selected_validation_metrics':quality,
                'selection_objective':config['selection'],'weights_source_sha256':WEIGHTS_SHA256})
    print('Selected epoch and thresholds frozen; opening test sets.',flush=True)
    for name in ['test','negative_test']:
        images[name],index=cache_images(root,frames[name],cache)
        cache_indices.append(index.assign(split=name))
        vectors[name]=transform_embeddings(embed_images(encoder,images[name]),embedding_transform)
    pd.concat(cache_indices).to_csv(output/'cache_index.csv',index=False)
    metrics={}
    for name in ['train','val','test','negative_val','negative_test']:
        np.save(output/f'{name}_embeddings.npy',vectors[name],allow_pickle=False)
        frames[name][['image_id','file_path','poi_id','source','author','licence','sha256']].to_csv(output/f'{name}_embedding_index.csv',index=False)
    for name in ['val','test']:
        scores,nearest=poi_scores(vectors[name],gallery_values,gallery.poi_id,poi_ids)
        summary,per_poi,pairs,confusion=evaluate(scores,frames[name],poi_ids,policy)
        summary.update(classification_metrics(scores,frames[name],poi_ids))
        pairs['reference_image_id']=gallery.image_id.to_numpy()[nearest.flatten()]
        pairs.to_csv(output/f'{name}_pairs.csv',index=False)
        per_poi.to_csv(output/f'{name}_per_poi.csv',index=False)
        confusion.to_csv(output/f'{name}_confusion.csv')
        neg_scores,_=poi_scores(vectors[f'negative_{name}'],gallery_values,gallery.poi_id,poi_ids)
        external,queries,groups=evaluate_external(neg_scores,frames[f'negative_{name}'],poi_ids,policy)
        queries.to_csv(output/f'negative_{name}_predictions.csv',index=False)
        groups.to_csv(output/f'negative_{name}_groups.csv',index=False)
        metrics[name]=summary;metrics[f'negative_{name}']=external
    json_write(output/'metrics.json',metrics)
    import torchvision,platform,PIL
    metadata={'schema_version':2,'model_type':'finetuned_mobilenet_v3_small','architecture':'mobilenet_v3_small',
        'preprocessing_version':PREPROCESSING_VERSION,'embedding_dim':int(gallery_values.shape[1]),'poi_ids':poi_ids,'seed':42,
        'embedding_transform':embedding_transform['mode'],'embedding_transform_fit':'positive_train_only',
        'best_epoch':best_epoch,'training_performed':best_epoch>0,'device':'cpu',
        'negative_images_used_for_gradients':False,'batchnorm_frozen':True,
        'parent_metadata_sha256':sha256_file(base/'metadata.json'),
        'reference_version':sha256_file(output/'gallery.csv')[:12]+'-'+sha256_file(output/'gallery_embeddings.npy')[:12],
        'threshold_version':sha256_file(output/'rejection_policy.json')[:16],
        'versions':{'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,
                    'numpy':np.__version__,'pandas':pd.__version__,'Pillow':PIL.__version__},
        'input_split_sha256':{name:sha256_file(root/f'data/annotations/{name}.csv') for name in ['train','val','test','negative_val','negative_test']},
        'deployment_status':'experimental_not_production_validated',
        'limitations':['Positive-test is an existing diagnostic holdout, not a new blind field test.',
                       'External negative test omits some rare types; per-POI samples are small.']}
    metadata['artifact_sha256']={p.relative_to(output).as_posix():sha256_file(p) for p in sorted(output.rglob('*')) if p.is_file()}
    json_write(output/'metadata.json',metadata)
    write_report(root,output,metadata,metrics,history)
    return metadata,metrics


def write_report(root,output,metadata,metrics,history):
    report=root/'reports/mobilenetv3_finetuned';report.mkdir(parents=True,exist_ok=True)
    old=json.loads((root/'models/baseline_mobilenetv3_negatives/metrics.json').read_text())
    test=metrics['test'];negative=metrics['negative_test']
    lines=['# Дообучение MobileNetV3-Small для POI','',
        f'Выбранная эпоха: {metadata["best_epoch"]}; выбор только по positive-val и negative-val. Веса обучались только на positive-train.',
        'Сбалансированные P×K батчи, cosine classification + SupCon; последние блоки backbone; BatchNorm заморожен.',
        f'Preprocessing: EXIF → JPEG draft 640 → thumbnail 320 → letterbox 224 → ImageNet normalization. Метрика: {metadata["embedding_transform"]}, fit только positive-train; галерея только train.',
        '','| Модель / test | Accuracy без отказа | Macro-F1 без отказа | True accept | External any-POI FAR |',
        '|---|---:|---:|---:|---:|',
        f'| Frozen Small + строгий отказ | {old["test"]["top1_accuracy"]:.2%} | 73.26% | {old["test"]["true_accept_rate"]:.2%} | {old["negative_test"]["any_poi_far"]:.2%} |',
        f'| Выбранный эксперимент | {test["accuracy"]:.2%} | {test["macro_f1"]:.2%} | {test["true_accept_rate"]:.2%} | {negative["any_poi_far"]:.2%} |','',
        '## Наблюдение → интерпретация → действие',
        f'- Верных подтверждений {test["true_accepts"]}/{test["positive_pairs"]} → текущий true accept={test["true_accept_rate"]:.2%} → цель 85% {"достигнута на диагностическом test" if test["true_accept_rate"]>=.85 else "пока не достигнута"}.',
        f'- Внешних ошибочных подтверждений {negative["false_accepts_any_poi"]}/{negative["images"]} → отказ измерен на Commons landmarks → не переносить результат на произвольную мобильную съёмку.',
        '- Выбор эпохи и порогов выполнялся на одном validation → его оценки оптимистичны → ориентироваться на отложенную оценку, а окончательное качество подтверждать новым закрытым полевым test.',
        '- Малые классы содержат единичные val/test фото → оценки нестабильны и возможен переучёт → добрать независимые ракурсы, особенно малые классы.',
        '- Условия FAR заданы для validation → test может превысить ограничение → не подбирать пороги повторно по test.',
        '- Preprocessing и веса изменились одновременно → прирост общего эксперимента не доказывает вклад одного изменения → epoch 0 в training_history.csv фиксирует frozen-encoder контроль с новым preprocessing.',
        '- Источники, авторы и лицензии сохранены в snapshots splits и gallery.csv → происхождение доступно для каждого эталона → сохранять атрибуцию при дальнейшем экспорте.','',
        'Исходники, положительные/отрицательные splits и прежние версии моделей не изменялись. Новый выпуск экспериментальный.',
        '[PyTorch: transfer learning](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).',
        '[SupCon](https://arxiv.org/abs/2004.11362).']
    (report/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    json_write(report/'metrics.json',metrics)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame=pd.DataFrame(history)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name in ['accuracy','macro_f1','true_accept_rate']:
        axes[0].plot(frame.epoch,frame[name],marker='.',label=name)
    axes[0].axvline(metadata['best_epoch'],linestyle='--',color='black');axes[0].legend();axes[0].set(xlabel='Epoch',ylabel='Validation metric')
    axes[1].plot(frame.epoch,frame.external_any_poi_far,marker='.',label='external any-POI FAR')
    axes[1].axhline(.01,linestyle='--',color='red');axes[1].legend();axes[1].set(xlabel='Epoch',ylabel='Validation FAR')
    fig.tight_layout();fig.savefig(report/'training_curves.png',dpi=140);plt.close(fig)


class FinetunedPOIVerifier:
    def __init__(self,directory:Path):
        directory=Path(directory)
        self.metadata=json.loads((directory/'metadata.json').read_text())
        if self.metadata['schema_version']!=2 or self.metadata['preprocessing_version']!=PREPROCESSING_VERSION:
            raise ValueError('Несовместимые артефакты.')
        for name in ['encoder.pt','embedding_transform.npz','gallery.csv','gallery_embeddings.npy','config.json','rejection_policy.json','class_mapping.json']:
            if sha256_file(directory/name)!=self.metadata['artifact_sha256'][name]:raise ValueError(f'Изменён артефакт {name}')
        configure_torch(42,json.loads((directory/'config.json').read_text())['threads'])
        from torchvision.models import mobilenet_v3_small
        self.encoder=mobilenet_v3_small(weights=None);self.encoder.classifier=nn.Identity()
        self.encoder.load_state_dict(torch.load(directory/'encoder.pt',map_location='cpu',weights_only=True))
        self.encoder.eval().requires_grad_(False)
        self.transform=image_transform(False)
        self.gallery=pd.read_csv(directory/'gallery.csv',keep_default_na=False)
        self.vectors=np.load(directory/'gallery_embeddings.npy',allow_pickle=False)
        self.policy=json.loads((directory/'rejection_policy.json').read_text())
        projection=np.load(directory/'embedding_transform.npz',allow_pickle=False)
        self.embedding_transform={'mode':self.metadata['embedding_transform'],'mean':projection['mean'],'matrix':projection['matrix']}
        self.poi_ids=self.metadata['poi_ids']

    def verify(self,image_path:Path,target_poi_id:str):
        version={key:self.metadata[key] for key in ['reference_version','threshold_version','preprocessing_version']}
        if target_poi_id not in self.poi_ids:
            return {'accepted':False,'reason':'target_poi_not_supported','target_poi_id':target_poi_id,'score':None,'margin':None,**version}
        with torch.inference_mode():vector=self.encoder(self.transform(decode_thumbnail(Path(image_path))).unsqueeze(0)).numpy()
        vector=transform_embeddings(l2_normalize(vector),self.embedding_transform)
        scores,nearest=poi_scores(vector,self.vectors,self.gallery.poi_id,self.poi_ids)
        index=self.poi_ids.index(target_poi_id);margin=float(score_margins(scores)[0,index]);score=float(scores[0,index])
        accepted=bool(acceptance_matrix(scores,self.policy)[0,index])
        reason='poi_verified' if accepted else 'poi_not_matched' if margin<=0 else 'poi_below_threshold' if score<self.policy['score_threshold'] else 'poi_ambiguous'
        ref=self.gallery.iloc[nearest[0,index]]
        return {'accepted':accepted,'reason':reason,'target_poi_id':target_poi_id,'predicted_poi_id':self.poi_ids[int(scores[0].argmax())],
                'score':score,'margin':margin,'score_is_probability':False,'reference_image_id':ref.image_id,
                'reference_source':ref.source,'reference_author':ref.author,'reference_licence':ref.licence,**version}
