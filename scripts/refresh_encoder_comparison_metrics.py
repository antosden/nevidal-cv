"""Пересобрать отчётные метрики из frozen scores без изменения выбора/порогов.

Служит также миграцией отчётной схемы: evaluate() по умолчанию описывает только
known-POI пары, поэтому external-флаги задаются после отдельной оценки negatives.
"""
from pathlib import Path
import argparse
import json
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--extra-site',type=Path)
args=parser.parse_args()
if args.extra_site:sys.path.append(str(args.extra_site.resolve()))
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.encoder_comparison import quality,write_report
from src.poi_baseline import sha256_file,json_write,poi_scores
output=ROOT/'models/encoder_comparison_manual_all'
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
status=load(output/'status.json')
for encoder,state in status.items():
    if state!='complete':continue
    directory=output/encoder;meta=load(directory/'metadata.json')
    for name,digest in meta['artifact_sha256'].items():assert sha256_file(directory/name)==digest,name
    if meta.get('evaluation_schema_version')==2:continue
    frames={name:pd.read_csv(output/f'splits/{name}.csv',keep_default_na=False) for name in ['val','test','negative_val','negative_test']}
    values={name:np.load(directory/f'{name}_embeddings.npy',allow_pickle=False) for name in frames}
    old=load(directory/'metrics.json');selection=load(directory/'selection.json');pois=meta['poi_ids']
    backup=ROOT/'.runtime/comparison_reporting_backup'/encoder;backup.mkdir(parents=True,exist_ok=True)
    for name in ['metadata.json','metrics.json','validation_candidates.csv']:
        if not (backup/name).exists():shutil.copyfile(directory/name,backup/name)
    rows=[];new={}
    for row in pd.read_csv(directory/'validation_candidates.csv').itertuples():
        folder=directory/row.variant
        gallery=pd.read_csv(folder/'gallery.csv',keep_default_na=False);vectors=np.load(folder/'gallery_embeddings.npy',allow_pickle=False)
        policy=load(folder/'policy.json')
        scores,_=poi_scores(values['val'],vectors,gallery.poi_id,pois)
        neg,_=poi_scores(values['negative_val'],vectors,gallery.poi_id,pois)
        summary,*_=quality(scores,frames['val'],pois,policy,neg,frames['negative_val'])
        previous=load(folder/'validation_metrics.json')
        for key in ['accuracy','macro_f1','true_accept_rate','external_any_poi_far']:assert summary[key]==previous[key],key
        json_write(folder/'validation_metrics.json',summary)
        rows.append(dict(encoder=encoder,variant=row.variant,gallery_images=len(gallery),**summary))
    for name in ['val','test']:
        scores=np.load(directory/f'{name}_scores.npy',allow_pickle=False)
        neg=np.load(directory/f'negative_{name}_scores.npy',allow_pickle=False)
        summary,external,*_=quality(scores,frames[name],pois,load(directory/selection['variant']/'policy.json'),neg,frames[f'negative_{name}'])
        for key in ['accuracy','macro_f1','true_accept_rate','external_any_poi_far']:assert summary[key]==old[name][key],key
        new[name]=summary;new[f'negative_{name}']=external
    json_write(directory/'metrics.json',new)
    pd.DataFrame(rows).to_csv(directory/'validation_candidates.csv',index=False)
    meta['evaluation_schema_version']=2
    meta['artifact_sha256']={p.relative_to(directory).as_posix():sha256_file(p) for p in sorted(directory.rglob('*')) if p.is_file() and p.name!='metadata.json'}
    json_write(directory/'metadata.json',meta)
write_report(ROOT,output,status)
print('Reporting metadata refreshed; numeric metrics, selections and thresholds unchanged.')
