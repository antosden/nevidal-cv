"""Новые Small/Large × О/Р; старые checkpoint не загружаются."""
from pathlib import Path
import json
import random
import time
import platform

import numpy as np
from PIL import Image,ImageOps
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small,mobilenet_v3_large

from src.viewpoint_audit import read,write,dump,digest,folders
from src.small_classifier import classification_metrics

ARCH={'small':mobilenet_v3_small,'large':mobilenet_v3_large}


def verify(root):
    data,report,models,runtime=folders(root)
    for name,sha in json.loads((data/'manifests_lock.json').read_text(encoding='utf-8')).items():
        if digest(data/name)!=sha:raise ValueError(f'Изменился манифест {name}')
    for name,sha in json.loads((data/'previous_work_sha256.json').read_text(encoding='utf-8')).items():
        if digest(root/name)!=sha:raise ValueError(f'Изменилась прежняя работа: {name}')
    for name,sha in json.loads((data/'cache_sha256.json').read_text(encoding='utf-8')).items():
        if digest(root/name)!=sha:raise ValueError(f'Изменился кеш изображения: {name}')


class Images(torch.utils.data.Dataset):
    def __init__(self,root,rows,training=False,seed=42):
        self.rows=rows;self.seed=seed;self.epoch=0;self.training=training
        self.images=[];runtime=folders(root)[3]
        for row in rows:
            with Image.open(runtime/'cache'/(row['sha256']+'.png')) as image:
                self.images.append(ImageOps.pad(image.convert('RGB'),(224,224),method=Image.Resampling.BILINEAR,color=(124,116,104)))
        self.color=transforms.ColorJitter(brightness=.15,contrast=.15,saturation=.1)
        self.tensor=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        image=self.images[index]
        if self.training:
            stable=int(self.rows[index]['sha256'][:12],16)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed((stable+self.seed*1000003+self.epoch*10007)%(2**63-1))
                image=self.color(image)
        return self.tensor(image),int(self.rows[index]['class_index'])


def balanced_batches(rows,batch_size,seed):
    """Ровно len(rows) выборок/эпоху, class-balanced; последний singleton объединён."""
    labels=[int(r['class_index']) for r in rows]
    counts=np.bincount(labels)
    sampler=torch.utils.data.WeightedRandomSampler([1./counts[y] for y in labels],num_samples=len(rows),replacement=True,
        generator=torch.Generator().manual_seed(seed))
    indices=list(sampler);batches=[indices[i:i+batch_size] for i in range(0,len(indices),batch_size)]
    if len(batches)>1 and len(batches[-1])==1:batches[-2]+=batches.pop()
    return batches


def predict(model,dataset,batch_size):
    model.eval();outputs=[]
    with torch.inference_mode():
        for images,_ in torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0):outputs.append(model(images).numpy())
    return np.concatenate(outputs)


def fit(root,architecture,mode,seed):
    root=Path(root);verify(root);data,report,models,runtime=folders(root)
    config=json.loads((data/'final_config.json').read_text(encoding='utf-8'))
    if architecture not in ARCH or mode not in config['modes'] or seed not in config['seeds']:raise ValueError('Неверная конфигурация')
    if (data/'test_opened.json').exists():raise ValueError('Test уже открыт: нельзя дообучать этот run_id')
    train_rows=read(data/'manifests'/f'train_{mode}_seed{seed}.csv');val_rows=read(data/'manifests/val.csv')
    mapping=json.loads((data/'class_mapping.json').read_text())
    pretrained=root/'.runtime/torch/checkpoints'/config['weights'][architecture]
    pretrained_sha=digest(pretrained)
    if not pretrained_sha.startswith(pretrained.stem.split('-')[-1]):raise ValueError('Не совпадает официальный SHA-префикс исходных весов')
    run=models/'runs'/f'{architecture}_{mode}_seed{seed}'
    existed=run.exists();run.mkdir(parents=True,exist_ok=True)
    signature=dict(architecture=architecture,mode=mode,seed=seed,config_sha256=digest(data/'final_config.json'),
                   manifests_sha256=digest(data/'manifests_lock.json'),train_sha256=digest(data/'manifests'/f'train_{mode}_seed{seed}.csv'),
                   val_sha256=digest(data/'manifests/val.csv'),pretrained_sha256=pretrained_sha,
                   pipeline_sha256=digest(Path(__file__)),torch=torch.__version__,python=platform.python_version(),
                   train_images=len(train_rows),steps_per_epoch=len(balanced_batches(train_rows,config['batch_size'],seed)),
                   sampler='class-balanced replacement; merge final singleton',selection='max val macro-F1, ties retain earlier epoch',test_accessed=False)
    if existed:
        if json.loads((run/'signature.json').read_text())!=signature:raise ValueError('Несовместимое возобновление')
        if json.loads((run/'status.json').read_text())['status']=='complete':return run
    else:
        dump(run/'signature.json',signature);dump(run/'status.json',dict(status='initializing',epoch=0))
    torch.set_num_threads(config['threads']);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    torch.use_deterministic_algorithms(True)
    model=ARCH[architecture](weights=None)
    model.load_state_dict(torch.load(pretrained,map_location='cpu',weights_only=True))
    model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,len(mapping));model.requires_grad_(True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    criterion=nn.CrossEntropyLoss();training=Images(root,train_rows,True,seed);validation=Images(root,val_rows)
    truth=np.array([int(r['class_index']) for r in val_rows]);best=-1.;start=1;history=[]
    if (run/'last.pt').exists():
        last=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(last['model']);optimizer.load_state_dict(last['optimizer']);start=last['epoch']+1
        history=read(run/'history.csv')[:last['epoch']];best=json.loads((run/'selection.json').read_text())['validation']['macro_f1']
    for epoch in range(start,config['epochs']+1):
        training.epoch=epoch;torch.manual_seed(seed*100003+epoch);model.train();started=time.monotonic();total=0
        batches=balanced_batches(train_rows,config['batch_size'],seed*100003+epoch)
        for images,labels in torch.utils.data.DataLoader(training,batch_sampler=batches,num_workers=0):
            optimizer.zero_grad(set_to_none=True);loss=criterion(model(images),labels)
            if not torch.isfinite(loss):raise ValueError('Неконечный loss')
            loss.backward();optimizer.step();total+=loss.item()*len(labels)
        logits=predict(model,validation,config['batch_size']);metrics,_,_=classification_metrics(truth,logits.argmax(1),len(mapping))
        history.append(dict(epoch=epoch,train_loss=total/len(training),seconds=time.monotonic()-started,**metrics))
        write(run/'history.csv',history)
        if metrics['macro_f1']>best:
            best=metrics['macro_f1'];torch.save(model.state_dict(),run/'model.pt');np.save(run/'val_logits.npy',logits)
            dump(run/'selection.json',dict(epoch=epoch,validation=metrics))
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch),run/'last.tmp')
        (run/'last.tmp').replace(run/'last.pt')
        dump(run/'status.json',dict(status='training',epoch=epoch))
        print(json.dumps(dict(architecture=architecture,mode=mode,seed=seed,**history[-1])),flush=True)
    dump(run/'status.json',dict(status='complete',epoch=config['epochs'],test_accessed=False))
    return run


def pilot_review(root):
    root=Path(root);data,report,models,_=folders(root);rows=[]
    for architecture in ARCH:
        for mode in ['O','R']:
            run=models/'runs'/f'{architecture}_{mode}_seed42'
            history=read(run/'history.csv');selection=json.loads((run/'selection.json').read_text())
            tail=[float(r['macro_f1']) for r in history[-5:]]
            rows.append(dict(architecture=architecture,mode=mode,epochs=len(history),best_epoch=selection['epoch'],
                val_macro_f1=selection['validation']['macro_f1'],last_loss=float(history[-1]['train_loss']),
                tail_f1_gain=tail[-1]-tail[0],training_seconds=sum(float(r['seconds']) for r in history),
                convergence_warning=(tail[-1]-tail[0]>.03 and selection['epoch']>=len(history)-1)))
    write(report/'pilot_review.csv',rows)
    dump(report/'pilot_status.json',dict(completed=4,recipe_review_required=any(r['convergence_warning'] for r in rows),
         criterion='warning if final-five-epoch val F1 gain >3pp and best at end; inspect development only',test_accessed=False))
    return rows
