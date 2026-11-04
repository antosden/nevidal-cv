"""Отдельный запуск H1: новые модели, манифесты и результаты."""
from pathlib import Path
import argparse,csv,json,hashlib,itertools,random,time,platform,sys,os
from collections import Counter,defaultdict
import numpy as np
from PIL import Image,ImageOps

RUN='h1_20261104_v1'
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/annotations'/RUN
REPORT=ROOT/'reports'/RUN
MODELS=ROOT/'models'/RUN
CACHE=ROOT/'.runtime'/RUN/'cache'
CONFIGS=('small_O','large_O','small_R','large_R')

def read(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if not rows:p.write_text('',encoding='utf8');return
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def dump(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metric(y,p,k,weights=None):
 w=np.ones(len(y)) if weights is None else weights
 cm=np.bincount(np.asarray(y)*k+np.asarray(p),weights=w,minlength=k*k).reshape(k,k)
 den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(k),where=den>0)
 return dict(macro_f1=float(f.mean()),accuracy=float(np.trace(cm)/max(1,cm.sum()))),cm,f
def contrasts(y,p,k,weights=None):
 scores={key:np.array([metric(y,a,k,weights)[0]['macro_f1'] for a in pp]) for key,pp in p.items()}
 go=scores['large_O']-scores['small_O'];gr=scores['large_R']-scores['small_R']
 return dict(F={key:float(a.mean()) for key,a in scores.items()},G_O=float(go.mean()),G_R=float(gr.mean()),C=float((go-gr).mean()),
  small_gain=float((scores['small_R']-scores['small_O']).mean()),large_gain=float((scores['large_R']-scores['large_O']).mean()),
  per_seed=[dict(seed=s,G_O=float(a),G_R=float(b),C=float(a-b)) for s,a,b in zip([42,43,44],go,gr)])
def verify():
 for name,h in load(DATA/'lock.json').items():assert sha(DATA/name)==h,name
 for r in read(DATA/'inventory.csv'):
  assert sha(ROOT/r['file_path'])==r['sha256'],r['file_path']
 for r in load(DATA/'cache_lock.json'):assert sha(CACHE/r['name'])==r['sha256']

def prepare(bundle):
 assert not MODELS.exists(),'Run already exists; do not overwrite a training run'
 bundle=load(bundle);config=bundle['config'];mapping={g:i for i,g in enumerate(sorted({r['poi_id'] for r in bundle['train']}))}
 for d in [DATA,REPORT,MODELS,CACHE,REPORT/'figures',MODELS/'runs']:d.mkdir(parents=True,exist_ok=True)
 for split in ['train','val','test']:
  for r in bundle[split]:
   r['file_path']=Path(r['file_path']).relative_to(ROOT).as_posix();r['class_index']=mapping[r['poi_id']]
   r['split']=split
 rows=bundle['train']+bundle['val']+bundle['test']
 assert all(set(r['poi_id'] for r in bundle[s])==set(mapping) for s in ['train','val','test'])
 for a,b in [('train','val'),('train','test'),('val','test')]:
  assert not ({r['sha256'] for r in bundle[a]}&{r['sha256'] for r in bundle[b]})
  assert not ({r['split_group'] for r in bundle[a]}&{r['split_group'] for r in bundle[b]})
  visits=lambda s:{r['outing_id'] for r in bundle[s] if r.get('outing_id')}
  assert not (visits(a)&visits(b))
 write(DATA/'inventory.csv',rows);write(DATA/'manifests/val.csv',bundle['val']);write(DATA/'manifests/test.csv',bundle['test'])
 dump(DATA/'class_mapping.json',mapping);dump(DATA/'config.json',config)
 coverage=[];balances=[]
 for poi in mapping:
  pool=[r for r in bundle['train'] if r['poi_id']==poi];groups={v:[r for r in pool if r['viewpoint']==v] for v in ['main','left','right']}
  for n in [24,21,18,15,12,9,6]:
   if len(groups['main'])<n or min(len(groups['left']),len(groups['right']))<n//3:continue
   found={}
   for seed in [42,43,44]:
    rng=random.Random(seed+mapping[poi]*100003);best=None
    for attempt in range(10000):
     o=rng.sample(groups['main'],n);r=sum([rng.sample(groups[v],n//3) for v in ['main','left','right']],[])
     tv={}
     for attr in ['illumination','scale','occlusion']:
      ca=Counter(x.get(attr,'unknown') for x in o);cb=Counter(x.get(attr,'unknown') for x in r)
      tv[attr]=sum(abs(ca[t]-cb[t]) for t in ca.keys()|cb.keys())/(2*n)
     cost=max(tv.values())
     if best is None or cost<best[0]:best=(cost,o,r,tv)
     if cost<=.20:break
    found[seed]=best
   if len(found)==3:break
  assert len(found)==3,f'No balanced comparison for {poi}'
  for seed,(cost,o,r,tv) in found.items():
   for mode,selected in [('O',o),('R',r)]:
    path=DATA/'manifests'/f'train_{mode}_seed{seed}.csv';prev=read(path) if path.exists() else [];write(path,prev+selected)
    coverage.append(dict(poi_id=poi,seed=seed,mode=mode,N=n,**{v:sum(x['viewpoint']==v for x in selected) for v in groups}))
   balances.append(dict(poi_id=poi,seed=seed,N=n,within_TV_020=cost<=.20,**{f'TV_{k}':v for k,v in tv.items()}))
 write(REPORT/'train_coverage.csv',coverage);write(REPORT/'balance.csv',balances)
 for s in [42,43,44]:
  a=read(DATA/'manifests'/f'train_O_seed{s}.csv');b=read(DATA/'manifests'/f'train_R_seed{s}.csv');assert Counter(r['poi_id'] for r in a)==Counter(r['poi_id'] for r in b)
 cache_lock=[]
 for r in rows:
  with Image.open(ROOT/r['file_path']) as im:
   im.draft('RGB',(512,512));im=ImageOps.exif_transpose(im).convert('RGB');im=ImageOps.pad(im,(224,224),method=Image.Resampling.BILINEAR,color=(124,116,104));im.save(CACHE/(r['sha256']+'.png'))
  cache_lock.append(dict(name=r['sha256']+'.png',sha256=sha(CACHE/(r['sha256']+'.png'))))
 dump(DATA/'cache_lock.json',cache_lock)
 test=bundle['test'];unknown=sum(not r.get('session_id') and not r.get('outing_id') for r in test)
 sessions={g:len({r.get('outing_id') or r.get('session_id') for r in test if r['poi_id']==g and (r.get('outing_id') or r.get('session_id'))}) for g in mapping}
 dump(DATA/'bootstrap_plan.json',dict(requested_replicates=5000,seed=2026,unknown_session_images=unknown,sessions_per_poi=sessions,
  feasible=unknown==0 and min(sessions.values())>=2,scheme='paired clusters: shared outings jointly, sessions within fixed POI'))
 dump(DATA/'lock.json',{str(p.relative_to(DATA)):sha(p) for p in sorted(DATA.rglob('*')) if p.is_file() and p.name!='lock.json'})
 dump(REPORT/'preflight.json',dict(classes=len(mapping),train_pool=len(bundle['train']),val=len(bundle['val']),test=len(bundle['test']),
  train_labels='human_confirmed',attributes_used=['viewpoint','illumination','scale','occlusion'],split_hashes_disjoint=True,known_visits_disjoint=True,
  novel_closed_test=False,annotation_decisions=bundle.get('decisions',{}),old_models_loaded=False))
 verify();print(load(REPORT/'preflight.json'),flush=True)

def torch_setup():
 import torch,torchvision
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
 return torch,torchvision
def make_model(arch,k,pretrained=False):
 torch,tv=torch_setup();fn=getattr(tv.models,'mobilenet_v3_'+arch);m=fn(weights=None)
 if pretrained:
  f=ROOT/'.runtime/torch/checkpoints'/({'small':'mobilenet_v3_small-047dcff4.pth','large':'mobilenet_v3_large-8738ca79.pth'}[arch]);assert sha(f).startswith(f.stem.split('-')[-1]);m.load_state_dict(torch.load(f,map_location='cpu',weights_only=True))
 m.classifier[-1]=torch.nn.Linear(m.classifier[-1].in_features,k);return m
def datasets(rows,training=False,seed=42):
 import torch
 from torchvision import transforms
 class Images(torch.utils.data.Dataset):
  def __init__(self):
   self.epoch=0;self.images=[]
   for r in rows:
    with Image.open(CACHE/(r['sha256']+'.png')) as im:self.images.append(im.convert('RGB').copy())
   self.color=transforms.ColorJitter(brightness=.15,contrast=.15,saturation=.1)
   self.tensor=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
  def __len__(self):return len(rows)
  def __getitem__(self,i):
   im=self.images[i]
   if training:
    with torch.random.fork_rng(devices=[]):
     torch.manual_seed((int(rows[i]['sha256'][:12],16)+seed*1000003+self.epoch*10007)%(2**63-1));im=self.color(im)
   return self.tensor(im),int(rows[i]['class_index'])
 return Images()
def predict(model,ds,batch=32):
 import torch
 model.eval();logits=[]
 with torch.inference_mode():
  for x,_ in torch.utils.data.DataLoader(ds,batch_size=batch,shuffle=False,num_workers=0):logits.append(model(x).numpy())
 return np.concatenate(logits)
def batches(rows,seed):
 import torch
 labels=[int(r['class_index']) for r in rows];counts=np.bincount(labels)
 sampler=torch.utils.data.WeightedRandomSampler([1/counts[y] for y in labels],len(rows),replacement=True,generator=torch.Generator().manual_seed(seed))
 ix=list(sampler);bb=[ix[i:i+32] for i in range(0,len(ix),32)]
 if len(bb)>1 and len(bb[-1])==1:bb[-2]+=bb.pop()
 return bb
def fit(arch,mode,seed):
 torch,tv=torch_setup();config=load(DATA/'config.json');mapping=load(DATA/'class_mapping.json');run=MODELS/'runs'/f'{arch}_{mode}_seed{seed}'
 signature=dict(arch=arch,mode=mode,seed=seed,config_sha=sha(DATA/'config.json'),manifest_sha=sha(DATA/'lock.json'),code_sha=sha(__file__))
 if (run/'status.json').exists() and load(run/'status.json')['status']=='complete':
  assert load(run/'signature.json')==signature;return
 assert not (DATA/'test_opened.json').exists(),'Test already opened; new training prohibited'
 assert not run.exists(),'Partial run requires explicit recovery, not overwrite'
 run.mkdir(parents=True);dump(run/'signature.json',signature)
 torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
 model=make_model(arch,len(mapping),True);model.requires_grad_(True)
 train=read(DATA/'manifests'/f'train_{mode}_seed{seed}.csv');val=read(DATA/'manifests/val.csv');ds=datasets(train,True,seed);vd=datasets(val);truth=np.array([int(r['class_index']) for r in val])
 optim=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=.0001);criterion=torch.nn.CrossEntropyLoss();history=[];best=-1
 for epoch in range(1,config['epochs']+1):
  torch.manual_seed(seed*100003+epoch);model.train();ds.epoch=epoch;start=time.perf_counter();loss_sum=0;steps=batches(train,seed*100003+epoch)
  for x,y in torch.utils.data.DataLoader(ds,batch_sampler=steps,num_workers=0):
   optim.zero_grad(set_to_none=True);loss=criterion(model(x),y);assert torch.isfinite(loss);loss.backward();optim.step();loss_sum+=loss.item()*len(y)
  logits=predict(model,vd);scores,_,_=metric(truth,logits.argmax(1),len(mapping));history.append(dict(epoch=epoch,train_loss=loss_sum/len(train),seconds=time.perf_counter()-start,**scores));write(run/'history.csv',history)
  if scores['macro_f1']>best:
   best=scores['macro_f1'];torch.save(model.state_dict(),run/'model.pt');np.save(run/'val_logits.npy',logits);dump(run/'selection.json',dict(epoch=epoch,**scores))
  torch.save(dict(model=model.state_dict(),optimizer=optim.state_dict(),epoch=epoch),run/'last.pt')
  dump(run/'status.json',dict(status='training',epoch=epoch));print(json.dumps(dict(run=run.name,**history[-1])),flush=True)
 dump(run/'status.json',dict(status='complete',epochs=config['epochs'],old_model_loaded=False));del model,optim
def pilot():
 verify();torch,tv=torch_setup();dump(REPORT/'environment.json',dict(python=sys.version,torch=torch.__version__,torchvision=tv.__version__,platform=platform.platform(),cpu=platform.processor(),threads=4,device='CPU',dtype='float32'))
 for arch in ['small','large']:
  for mode in ['O','R']:fit(arch,mode,42)
 review=[]
 for key in CONFIGS:
  run=MODELS/'runs'/f'{key}_seed42';hist=read(run/'history.csv');sel=load(run/'selection.json');tail=[float(h['macro_f1']) for h in hist[-5:]]
  review.append(dict(config=key,epochs=len(hist),best_epoch=sel['epoch'],best_val_f1=sel['macro_f1'],last_loss=float(hist[-1]['train_loss']),tail_gain=tail[-1]-tail[0],warning=tail[-1]-tail[0]>.03 and sel['epoch']>=len(hist)-1))
 write(REPORT/'pilot_review.csv',review);print('PILOT_REVIEW',review,flush=True)
def main_runs():
 verify();assert load(REPORT/'pilot_decision.json')['continue_unchanged']
 for seed in [43,44]:
  for arch in ['small','large']:
   for mode in ['O','R']:fit(arch,mode,seed)

def bootstrap(y,p,rows,k):
 plan=load(DATA/'bootstrap_plan.json');point=contrasts(y,p,k);res=dict(point=point,plan=plan,performed_replicates=0,C_ci95=None,G_R_ci95=None,decision='Неопределённый результат: история съёмок неполная.',hypothesis_supported=False)
 if not plan['feasible']:return res
 outings=defaultdict(list);sessions=defaultdict(lambda:defaultdict(list))
 for i,r in enumerate(rows):
  if r['outing_id']:outings[r['outing_id']].append(i)
  else:sessions[r['poi_id']][r['session_id']].append(i)
 blocks=[np.array(a) for a in outings.values()];strata=[[np.array(a) for a in ss.values()] for ss in sessions.values()];rng=np.random.default_rng(2026);values=[];attempts=0
 while len(values)<5000:
  attempts+=1;assert attempts<500000;w=np.zeros(len(rows))
  if blocks:
   for j in rng.integers(len(blocks),size=len(blocks)):w[blocks[j]]+=1
  for group in strata:
   for j in rng.integers(len(group),size=len(group)):w[group[j]]+=1
  if min(np.bincount(y,weights=w,minlength=k))<=0:continue
  v=contrasts(y,p,k,w);values.append((v['C'],v['G_R']))
 cis=np.quantile(values,[.025,.975],axis=0);support=cis[0,0]>=.03 and cis[1,1]<=.03
 res.update(performed_replicates=5000,C_ci95=cis[:,0].tolist(),G_R_ci95=cis[:,1].tolist(),decision='Гипотеза поддерживается на внутреннем тесте.' if support else ('Один из порогов не достигнут.' if cis[1,0]<.03 or cis[0,1]>.03 else 'Результат неопределённый.'),hypothesis_supported=bool(support));return res

def evaluate():
 import torch
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 verify();mapping=load(DATA/'class_mapping.json');k=len(mapping);test=read(DATA/'manifests/test.csv');truth=np.array([int(r['class_index']) for r in test]);ds=datasets(test)
 for key in CONFIGS:
  for seed in [42,43,44]:assert load(MODELS/'runs'/f'{key}_seed{seed}'/'status.json')['status']=='complete'
 assert not (DATA/'test_opened.json').exists(),'Evaluation already opened'
 dump(DATA/'test_opened.json',dict(opened_after_12_runs=True,test_sha256=sha(DATA/'manifests/test.csv'),status='exploratory_internal'))
 predictions={key:[] for key in CONFIGS};all_metrics=[];records=[];classes=[];slices=[];cms=defaultdict(list)
 for key in CONFIGS:
  arch,mode=key.split('_')
  for seed in [42,43,44]:
   run=MODELS/'runs'/f'{key}_seed{seed}';model=make_model(arch,k);model.load_state_dict(torch.load(run/'model.pt',weights_only=True,map_location='cpu'))
   logits=predict(model,ds);np.save(run/'test_logits.npy',logits);pred=logits.argmax(1);probs=torch.softmax(torch.from_numpy(logits),1).numpy();scores,cm,f1=metric(truth,pred,k);predictions[key].append(pred);cms[key].append(cm)
   all_metrics.append(dict(config=key,seed=seed,best_epoch=load(run/'selection.json')['epoch'],**scores));print('TEST',all_metrics[-1],flush=True)
   for poi,i in mapping.items():classes.append(dict(config=key,seed=seed,poi_id=poi,support=int(cm[i].sum()),f1=float(f1[i]),recall=float(cm[i,i]/max(1,cm[i].sum()))))
   for v in ['main','left','right','undetermined']:
    ii=[i for i,r in enumerate(test) if r['viewpoint']==v]
    if ii:slices.append(dict(config=key,seed=seed,viewpoint=v,n=len(ii),**metric(truth[ii],pred[ii],k)[0]))
   current=[dict(image_id=r['image_id'],poi_id=r['poi_id'],viewpoint=r['viewpoint'],session_id=r['session_id'],outing_id=r['outing_id'],true_class=int(truth[i]),predicted_class=int(pred[i]),config=key,seed=seed,probabilities=json.dumps(probs[i].tolist())) for i,r in enumerate(test)]
   write(run/'test_predictions.csv',current);records+=current;write(run/'confusion.csv',[dict(poi_id=poi,**{str(j):int(cm[i,j]) for j in range(k)}) for poi,i in mapping.items()]);del model
 inference=bootstrap(truth,predictions,test,k);dump(REPORT/'inference.json',inference);write(REPORT/'per_seed_contrasts.csv',inference['point']['per_seed'])
 summary=[dict(config=key,macro_f1_mean=float(np.mean([r['macro_f1'] for r in all_metrics if r['config']==key])),macro_f1_SD=float(np.std([r['macro_f1'] for r in all_metrics if r['config']==key],ddof=1)),accuracy_mean=float(np.mean([r['accuracy'] for r in all_metrics if r['config']==key]))) for key in CONFIGS]
 write(REPORT/'test_metrics.csv',all_metrics);write(REPORT/'test_summary.csv',summary);write(REPORT/'per_poi_metrics.csv',classes);write(REPORT/'viewpoint_slices.csv',slices);write(REPORT/'test_predictions.csv',records)
 fig,axs=plt.subplots(3,4,figsize=(16,10))
 for row,seed in enumerate([42,43,44]):
  for col,key in enumerate(CONFIGS):
   h=read(MODELS/'runs'/f'{key}_seed{seed}'/'history.csv');ax=axs[row,col];ax.plot([int(a['epoch']) for a in h],[float(a['macro_f1']) for a in h]);ax.set_ylim(0,1);ax.set_title(f'{key}, seed {seed}');ax.set_xlabel('Epoch');ax2=ax.twinx();ax2.plot([int(a['epoch']) for a in h],[float(a['train_loss']) for a in h],color='gray',alpha=.5);ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(REPORT/'figures/learning_curves.png',dpi=140);plt.close(fig)
 fig,axs=plt.subplots(1,4,figsize=(18,4))
 for ax,key in zip(axs,CONFIGS):
  cm=np.mean(cms[key],0);ax.imshow(cm,cmap='Blues');ax.set_title(key);ax.set_xlabel('Predicted');ax.set_ylabel('True');ax.set_xticks(range(k));ax.set_yticks(range(k))
  for i in range(k):
   for j in range(k):ax.text(j,i,f'{cm[i,j]:.1f}',ha='center',va='center',fontsize=7)
 fig.tight_layout();fig.savefig(REPORT/'figures/confusion_matrices.png',dpi=140);plt.close(fig)
 loo=[]
 for excluded,i in mapping.items():
  ii=truth!=i;v=contrasts(truth[ii],{key:np.asarray(a)[:,ii] for key,a in predictions.items()},k)
  loo.append(dict(excluded_poi=excluded,C=v['C'],G_R=v['G_R'],mapping_classes=k))
 write(REPORT/'leave_one_poi_out.csv',loo);dump(REPORT/'status.json',dict(completed_runs=12,stage='evaluated',study='exploratory_internal',test_images=len(test)))
 resources()

def resources():
 import torch
 k=len(load(DATA/'class_mapping.json'));test=read(DATA/'manifests/test.csv');ds=datasets(test);sample=ds[0][0].unsqueeze(0);models={};results=[];parity=[]
 for arch in ['small','large']:
  run=MODELS/'runs'/f'{arch}_R_seed42';m=make_model(arch,k);m.load_state_dict(torch.load(run/'model.pt',weights_only=True,map_location='cpu'));m.eval()
  with torch.inference_mode():export=torch.jit.trace(m,sample,check_trace=True);export=torch.jit.freeze(export)
  path=MODELS/f'{arch}_R_seed42_fp32.torchscript.pt';torch.jit.save(export,str(path));loaded=torch.jit.load(str(path));loaded.eval();models[arch]=loaded
  native=predict(m,ds);exported=predict(loaded,ds);maxdiff=float(np.max(abs(native-exported)));assert np.allclose(native,exported,atol=1e-4,rtol=1e-4);assert np.array_equal(native.argmax(1),exported.argmax(1))
  parity.append(dict(architecture=arch,max_logit_difference=maxdiff,predictions_equal=True,file_bytes=path.stat().st_size,parameters=sum(a.numel() for a in m.parameters()),dtype='float32',format='TorchScript'))
 with torch.inference_mode():
  for m in models.values():
   for _ in range(20):m(sample)
  for repeat in range(3):
   for arch in (['small','large'] if repeat%2==0 else ['large','small']):
    m=models[arch];times=[];full=[]
    for j in range(100):
     t=time.perf_counter();m(sample);times.append((time.perf_counter()-t)*1000)
     r=test[j%len(test)];t=time.perf_counter()
     with Image.open(ROOT/r['file_path']) as im:
      im.draft('RGB',(512,512));im=ImageOps.pad(ImageOps.exif_transpose(im).convert('RGB'),(224,224),method=Image.Resampling.BILINEAR,color=(124,116,104))
     a=np.asarray(im,dtype=np.float32)/255;x=torch.from_numpy(((a-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)).transpose(2,0,1).copy()).unsqueeze(0);m(x);full.append((time.perf_counter()-t)*1000)
    results.append(dict(architecture=arch,repeat=repeat+1,forward_p50_ms=float(np.quantile(times,.5)),forward_p95_ms=float(np.quantile(times,.95)),full_p50_ms=float(np.quantile(full,.5)),full_p95_ms=float(np.quantile(full,.95)),warmups=20,measurements=100,threads=4,device='Windows CPU',dtype='float32',runtime='TorchScript',scope='file decode + RGB/pad/normalize + model; excludes camera and network'))
 write(REPORT/'export_parity.csv',parity);write(REPORT/'resources.csv',results);dump(REPORT/'mobile_status.json',dict(phone_measurements=False,product_quality_threshold=None,product_p95_budget=None,release_recommendation=False))

def notebook():
 import nbformat as nb
 from nbclient import NotebookClient
 from jupyter_client import KernelManager
 def md(s):return nb.v4.new_markdown_cell(s)
 def code(s):return nb.v4.new_code_cell(s)
 book=nb.v4.new_notebook();book.metadata.kernelspec=dict(display_name='Python 3',language='python',name='python3')
 cells=[md('# H1. Ракурсы и качество MobileNetV3\n\nПроверяем, помогает ли разнообразие ракурсов сократить разрыв между Small и Large. Объём обучения в двух режимах одинаковый. Фото train просмотрены и подтверждены вручную.'),
 code("from pathlib import Path\nimport sys, json\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT = Path.cwd()\nif not (ROOT / 'scripts').exists(): ROOT = ROOT.parent\nsys.path.insert(0, str(ROOT / 'scripts'))\nimport h1_research as h1\nh1.verify()\nDATA, REPORT, MODELS = h1.DATA, h1.REPORT, h1.MODELS\nload = h1.load\n"),
 md('## Данные\n\nВ train входят только фото из h1_experiment. Validation и test собраны отдельно. Разметку ракурсов train сохраняем по папкам.'),
 code("display(pd.DataFrame([load(REPORT/'preflight.json')])[['classes','train_pool','val','test']])\ndisplay(pd.read_csv(REPORT/'train_coverage.csv'))\ndisplay(pd.read_csv(REPORT/'balance.csv'))"),
 md('## Гипотеза\n\nО — основной вид. Р — три ракурса поровну. Считаем средний test Macro-F1 трёх моделей, без ансамбля.\n\n`GО = F(Large, О) − F(Small, О)`\n\n`GР = F(Large, Р) − F(Small, Р)`\n\n`C = GО − GР`\n\nНужны оба условия: нижняя 95% граница C ≥ 0,03 и верхняя граница GР ≤ 0,03.'),
 md('## Обучение\n\nНовые Small и Large начинаем с ImageNet V1. Обучаем все слои и BatchNorm: RGB 224 × 224, FP32, AdamW, lr 0,0001, weight decay 0,0001, batch 32, 20 эпох. Используем балансировку классов и только цветовые аугментации. Лучшую эпоху выбираем по validation Macro-F1, при равенстве — раннюю.'),
 code("display(pd.DataFrame([load(DATA/'config.json')]))\ndisplay(pd.read_csv(REPORT/'pilot_review.csv'))\nprint(load(REPORT/'pilot_decision.json')['reason'])")]
 for seed in [42,43,44]:
  cells.append(md(f'### Seed {seed}'))
  for key in CONFIGS:
   cells.append(code(f"run = MODELS/'runs/{key}_seed{seed}'\nprint('{key}, seed {seed}')\ndisplay(pd.DataFrame([load(run/'selection.json')]))\nhistory = pd.read_csv(run/'history.csv')\nax = history.plot(x='epoch', y=['macro_f1'], ylim=(0,1), title='{key}, seed {seed}')\nax.set_ylabel('Validation Macro-F1')\ndisplay(history[['epoch','train_loss','macro_f1','accuracy']].tail())\nimport matplotlib.pyplot as plt\nplt.show()"))
 cells += [md('## Результаты\n\nВсе 12 моделей проверяем на одном test после завершения обучения. SD считаем по трём seed с ddof=1.'),code("display(pd.read_csv(REPORT/'test_summary.csv'))\ndisplay(pd.read_csv(REPORT/'test_metrics.csv'))\ninference = load(REPORT/'inference.json')\ndisplay(pd.DataFrame([inference['point']]).drop(columns=['F','per_seed']))\ndisplay(pd.read_csv(REPORT/'per_seed_contrasts.csv'))"),
 md('### Ошибки по объектам и ракурсам'),code("display(Image(filename=str(REPORT/'figures/confusion_matrices.png')))\ndisplay(pd.read_csv(REPORT/'per_poi_metrics.csv'))\ndisplay(pd.read_csv(REPORT/'viewpoint_slices.csv'))\ndisplay(pd.read_csv(REPORT/'leave_one_poi_out.csv'))"),
 md('### Неопределённость\n\nПлан: 5000 парных кластерных повторов, seed 2026, 95% интервалы. Все модели получают одни и те же пересэмплированные сеансы. Если история сеансов неполная, интервалы не заменяем пересэмплированием отдельных фото.'),code("display(pd.DataFrame([inference['plan']]))\nprint(inference['decision'])\nprint('C CI95:', inference['C_ci95'], 'GР CI95:', inference['G_R_ci95'])\nprint('Выполнено повторов:', inference['performed_replicates'])"),
 md('## Размер и скорость\n\nСравниваем Small Р и Large Р с seed 42 в одном FP32 runtime. Проверяем экспорт, затем делаем 20 прогревов и три серии по 100 замеров. Замеры выполнены на CPU компьютера.'),code("display(pd.read_csv(REPORT/'export_parity.csv'))\ndisplay(pd.read_csv(REPORT/'resources.csv'))"),
 md('## Вывод\n\nРезультат относится к этим объектам и этому разбиению. Внутренний test не заменяет отдельные новые съёмки. Для выбора модели приложения ещё нужны требования к качеству и p95 и замеры на телефоне.'),code("p = inference['point']\nprint(f\"GО = {p['G_O']*100:.2f} п.п.; GР = {p['G_R']*100:.2f} п.п.; C = {p['C']*100:.2f} п.п.\")\nprint(inference['decision'])"),
 md('## Повторный запуск\n\nИз корня проекта: `python scripts/h1_research.py pilot`, затем после проверки пилота `main`, `evaluate` и `notebook`. После открытия test настройки и данные этого запуска не меняем. Для другого эксперимента нужен отдельный run_id.')]
 book.cells=cells;path=ROOT/'notebooks/H1_mobilenetv3_viewpoint_research.ipynb';assert not path.exists(),'Notebook already exists; do not overwrite'
 nb.write(book,path);manager=KernelManager(kernel_name='python3');manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'];manager.kernel_spec.env={**os.environ,'PYTHONIOENCODING':'utf-8','MPLCONFIGDIR':str(ROOT/'.runtime'/RUN/'matplotlib'),'IPYTHONDIR':str(ROOT/'.runtime'/RUN/'ipython')}
 try:NotebookClient(book,km=manager,timeout=240,resources={'metadata':{'path':str(ROOT)}}).execute()
 finally:
  if manager.has_kernel:manager.shutdown_kernel(now=True)
 assert not [o for c in book.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='error'];nb.validate(book);nb.write(book,path);dump(REPORT/'notebook_status.json',dict(executed=True,errors=0,path=str(path)))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','pilot','main','evaluate','notebook']);parser.add_argument('--bundle');a=parser.parse_args()
 if a.command=='prepare':prepare(a.bundle)
 elif a.command=='pilot':pilot()
 elif a.command=='main':main_runs()
 elif a.command=='evaluate':evaluate()
 elif a.command=='notebook':notebook()
