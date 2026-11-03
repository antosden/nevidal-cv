"""Итоговая парная оценка: без подбора каталога или порога по test."""
from collections import defaultdict
from pathlib import Path
import json
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.viewpoint_audit import read,write,dump,digest,folders
from src.viewpoint_training import ARCH,Images,predict,verify
from src.viewpoint_statistics import clustered_bootstrap,bootstrap_plan,CONFIGS
from src.small_classifier import classification_metrics


def resource_benchmark(root,config,mapping):
    """Заранее заданная пара Р/42: CPU, FP32, 20 warmups + 3×100."""
    _,report,models,_=folders(root);results=[]
    torch.set_num_threads(config['threads'])
    for architecture in ['small','large']:
        path=models/'runs'/f'{architecture}_R_seed42'/'model.pt'
        model=ARCH[architecture](weights=None)
        model.classifier[-1]=torch.nn.Linear(model.classifier[-1].in_features,len(mapping))
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));model.eval()
        image=torch.zeros(1,3,224,224,dtype=torch.float32)
        with torch.inference_mode():
            for _ in range(20):model(image)
            for repeat in range(3):
                times=[]
                for _ in range(100):
                    started=time.perf_counter();model(image);times.append((time.perf_counter()-started)*1000)
                results.append(dict(architecture=architecture,mode='R',seed=42,repeat=repeat+1,
                    p50_ms=float(np.quantile(times,.5)),p95_ms=float(np.quantile(times,.95)),
                    model_bytes=path.stat().st_size,parameters=sum(p.numel() for p in model.parameters()),
                    device='Windows CPU; not Android',dtype='float32',batch_size=1,
                    threads=config['threads'],warmups=20,measurements=100,
                    scope='torch forward only; excludes JPEG/preprocessing; no mobile export measured'))
        del model
    write(report/'resource_benchmark.csv',results)


def plot_learning(root):
    _,report,models,_=folders(root)
    fig,axes=plt.subplots(3,4,figsize=(16,10))
    for row,seed in enumerate([42,43,44]):
        for col,key in enumerate(CONFIGS):
            history=read(models/'runs'/f'{key}_seed{seed}'/'history.csv')
            ax=axes[row,col]
            if history:
                ax.plot([int(r['epoch']) for r in history],[float(r['macro_f1']) for r in history],label='val Macro-F1')
                ax2=ax.twinx();ax2.plot([int(r['epoch']) for r in history],[float(r['train_loss']) for r in history],color='gray',alpha=.45,label='train CE')
                ax2.set_ylim(bottom=0);ax.set_ylim(0,1)
            ax.set_title(f'{key}, seed {seed}');ax.set_xlabel('epoch');ax.grid(alpha=.2)
    fig.suptitle('Blue: validation Macro-F1; gray: train cross-entropy')
    fig.tight_layout();fig.savefig(report/'figures/learning_curves.png',dpi=130);plt.close(fig)


def evaluate(root):
    root=Path(root);verify(root);data,report,models,_=folders(root)
    config=json.loads((data/'final_config.json').read_text(encoding='utf-8'))
    from importlib.metadata import version
    environment=json.loads((report/'environment.json').read_text(encoding='utf-8'))
    environment['dependencies']={name:version(name) for name in ['numpy','Pillow','matplotlib','nbformat','nbclient']}
    dump(report/'environment.json',environment)
    mapping=json.loads((data/'class_mapping.json').read_text(encoding='utf-8'));k=len(mapping)
    # Ни одной оценки test до завершения всех двенадцати совместимых обучений.
    for key in CONFIGS:
        for seed in config['seeds']:
            run=models/'runs'/f'{key}_seed{seed}'
            assert json.loads((run/'status.json').read_text())['status']=='complete'
            signature=json.loads((run/'signature.json').read_text())
            assert signature['config_sha256']==digest(data/'final_config.json')
            assert signature['manifests_sha256']==digest(data/'manifests_lock.json')
    test=read(data/'manifests/test.csv')
    assert bootstrap_plan(test)==json.loads((data/'bootstrap_plan_before_predictions.json').read_text(encoding='utf-8'))
    dump(data/'test_opened.json',dict(status='opened_after_all_12',manifest_sha256=digest(data/'manifests/test.csv'),
         study_status='exploratory_internal',independent=False))
    torch.set_num_threads(config['threads']);truth=np.array([int(r['class_index']) for r in test]);dataset=Images(root,test)
    metrics=[];per_poi=[];slices=[];all_predictions=[];predictions={key:[] for key in CONFIGS}
    matrices=defaultdict(list)
    for key in CONFIGS:
        architecture,mode=key.split('_')
        for seed in config['seeds']:
            run=models/'runs'/f'{key}_seed{seed}'
            model=ARCH[architecture](weights=None);model.classifier[-1]=torch.nn.Linear(model.classifier[-1].in_features,k)
            model.load_state_dict(torch.load(run/'model.pt',map_location='cpu',weights_only=True));model.eval()
            logits=predict(model,dataset,config['batch_size']);np.save(run/'test_logits.npy',logits)
            probability=torch.softmax(torch.from_numpy(logits),dim=1).numpy();prediction=logits.argmax(1)
            predictions[key].append(prediction)
            result,matrix,classes=classification_metrics(truth,prediction,k);matrices[key].append(matrix)
            write(run/'test_confusion.csv',[dict(true_poi=poi,**{str(j):int(v) for j,v in enumerate(matrix[i])}) for poi,i in mapping.items()])
            selected=json.loads((run/'selection.json').read_text())
            metrics.append(dict(config=key,seed=seed,best_epoch=selected['epoch'],**result))
            for item in classes:
                poi=list(mapping)[item['class_index']]
                per_poi.append(dict(config=key,seed=seed,poi_id=poi,recall=item['tp']/max(1,item['support']),**item))
            for view in ['main','left','right','undetermined']:
                indices=[i for i,r in enumerate(test) if r['viewpoint']==view]
                if indices:
                    score,_,_=classification_metrics(truth[indices],prediction[indices],k)
                    slices.append(dict(config=key,seed=seed,viewpoint=view,mapping_classes=k,**score))
            records=[dict(image_id=r['image_id'],poi_id=r['poi_id'],true_class=int(truth[i]),predicted_class=int(prediction[i]),
                config=key,seed=seed,viewpoint=r['viewpoint'],probabilities=json.dumps(probability[i].tolist()),
                session_id=r['session_id'],outing_id=r['outing_id']) for i,r in enumerate(test)]
            write(run/'test_predictions.csv',records);all_predictions.extend(records)
            print(json.dumps(metrics[-1]),flush=True)
            del model
    summary=[]
    for key in CONFIGS:
        bucket=[r for r in metrics if r['config']==key]
        summary.append(dict(config=key,runs=len(bucket),**{f'{metric}_{stat}':float(func([r[metric] for r in bucket]))
            for metric in ['macro_f1','accuracy'] for stat,func in [('mean',np.mean),('sample_SD',lambda a:np.std(a,ddof=1))]}))
    inference=clustered_bootstrap(truth,predictions,test,k,config['bootstrap_replicates'],config['bootstrap_seed'])
    for seed,item in zip(config['seeds'],inference['point']['per_seed']):item['seed']=seed
    write(report/'per_seed_contrasts.csv',inference['point']['per_seed'])
    write(report/'test_metrics.csv',metrics);write(report/'test_summary.csv',summary)
    write(report/'per_poi_metrics.csv',per_poi);write(report/'viewpoint_slices.csv',slices)
    write(report/'test_predictions.csv',all_predictions);dump(report/'inference.json',inference)
    fig,axes=plt.subplots(1,4,figsize=(19,4))
    for ax,key in zip(axes,CONFIGS):
        matrix=np.mean(matrices[key],axis=0)
        write(report/f'confusion_{key}.csv',[dict(true_poi=poi,**{str(j):float(v) for j,v in enumerate(matrix[i])}) for poi,i in mapping.items()])
        ax.imshow(matrix,cmap='Blues');ax.set_title(key+' (mean counts)');ax.set_xticks(range(k));ax.set_yticks(range(k));ax.set_xlabel('predicted index');ax.set_ylabel('true index')
        for i in range(k):
            for j in range(k):ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',fontsize=7)
    fig.tight_layout();fig.savefig(report/'figures/confusion_matrices.png',dpi=130);plt.close(fig)
    plot_learning(root);resource_benchmark(root,config,mapping)
    verify(root)
    dump(report/'experiment_status.json',dict(completed_runs=12,planned_runs=12,study_status='exploratory_internal',
        independent_test=False,hypothesis_confirmed=False,previous_work_sha256_verified=True,stage='complete'))
    selection=json.loads((report/'selection_status.json').read_text(encoding='utf-8'))
    selection.update(stage='evaluated_all_12',study_status='exploratory_internal')
    dump(report/'selection_status.json',selection)
    dump(report/'research_code_sha256.json',{p.relative_to(root).as_posix():digest(p) for p in
        list((root/'src').glob('viewpoint_*.py'))+[root/'src/small_classifier.py',root/'scripts/run_viewpoint_research.py',root/'scripts/check_viewpoint_integrity.py',root/'scripts/execute_viewpoint_notebook.py']})
    from src.viewpoint_reporting import write_report
    write_report(root)
