"""Независимо повторить выбор галерей, пороги и метрики из эмбеддингов."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--extra-site',type=Path)
args=parser.parse_args()
if args.extra_site:sys.path.append(str(args.extra_site.resolve()))
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.encoder_comparison import gallery_variant,quality,load_encoder
from src.poi_baseline import sha256_file,poi_scores,calibrate_policy,json_write,configure_torch,load_rgb,l2_normalize
torch=configure_torch(42,4)
output=ROOT/'models/encoder_comparison_manual_all'
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
snapshot=load(output/'input_snapshot.json');status=load(output/'status.json');config=snapshot['config']
frames={name:pd.read_csv(output/f'splits/{name}.csv',keep_default_na=False) for name in ['train','val','test','negative_val','negative_test']}
for name,frame in frames.items():assert sha256_file(ROOT/f'data/annotations/{name}.csv')==snapshot[name],name
mapping=load(output/'class_mapping.json');pois=sorted(mapping,key=mapping.get)
checked=[];maximum_embedding_difference=0.
for encoder,state in status.items():
    if state!='complete':continue
    directory=output/encoder;meta=load(directory/'metadata.json')
    for name,digest in meta['artifact_sha256'].items():assert sha256_file(directory/name)==digest,name
    assert sha256_file(ROOT/load(directory/'weights.json')['file'])==meta['weights_sha256']
    values={name:np.load(directory/f'{name}_embeddings.npy',allow_pickle=False) for name in frames}
    candidates=[]
    for row in pd.read_csv(directory/'validation_candidates.csv').itertuples():
        method,size=row.variant.split('_');size=size if size=='all' else int(size)
        gallery,vectors=gallery_variant(frames['train'],values['train'],size,method)
        saved=pd.read_csv(directory/row.variant/'gallery.csv',keep_default_na=False)
        assert gallery.image_id.tolist()==saved.image_id.tolist()
        assert set(gallery.image_id)<=set(frames['train'].image_id)
        scores,_=poi_scores(values['val'],vectors,gallery.poi_id,pois)
        neg,_=poi_scores(values['negative_val'],vectors,gallery.poi_id,pois)
        policy,_=calibrate_policy(scores,np.array([pois.index(p) for p in frames['val'].poi_id]),config,neg)
        assert policy==load(directory/row.variant/'policy.json')
        summary,*_=quality(scores,frames['val'],pois,policy,neg,frames['negative_val'])
        assert summary==load(directory/row.variant/'validation_metrics.json')
        candidates.append((summary['true_accept_rate'],-summary['external_any_poi_far'],summary['macro_f1'],summary['accuracy'],-len(gallery),row.variant))
    selection=load(directory/'selection.json')
    assert selection['variant']==max(candidates,key=lambda c:c[:-1])[-1]
    assert not selection['test_used_for_selection']
    metrics=load(directory/'metrics.json');folder=directory/selection['variant']
    gallery=pd.read_csv(folder/'gallery.csv',keep_default_na=False);vectors=np.load(folder/'gallery_embeddings.npy',allow_pickle=False)
    for name in ['val','test']:
        scores,_=poi_scores(values[name],vectors,gallery.poi_id,pois)
        neg,_=poi_scores(values[f'negative_{name}'],vectors,gallery.poi_id,pois)
        summary,external,*_=quality(scores,frames[name],pois,load(folder/'policy.json'),neg,frames[f'negative_{name}'])
        assert summary==metrics[name];assert external==metrics[f'negative_{name}']
    # Свежая обработка raw проверяет согласованность одиночного вывода и батчей.
    model,transform=load_encoder(encoder,ROOT/load(directory/'weights.json')['file'])
    example=frames['val'].iloc[0]
    with torch.inference_mode():fresh=l2_normalize(model(transform(load_rgb(ROOT/example.file_path)).unsqueeze(0)).numpy())
    difference=float(np.abs(fresh[0]-values['val'][0]).max())
    maximum_embedding_difference=max(maximum_embedding_difference,difference)
    np.testing.assert_allclose(fresh[0],values['val'][0],atol=1e-5,rtol=1e-4)
    checked.append(encoder)
result={'status':'passed_for_completed_encoders','checked_encoders':checked,'all_three_complete':len(checked)==3,
    'input_splits_unchanged':True,'gallery_train_only':True,'selection_and_metrics_reproduced':True,
    'fresh_single_embedding_checks':len(checked),'maximum_embedding_difference':maximum_embedding_difference}
json_write(ROOT/'reports/encoder_comparison/integrity_checks.json',result);print(json.dumps(result,indent=2))
