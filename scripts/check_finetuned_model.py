"""Проверка checkpoint, выборки, галереи, порогов, метрик и одиночного вывода."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--extra-site',type=Path)
args=parser.parse_args()
if args.extra_site:sys.path.append(str(args.extra_site.resolve()))
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
import nbformat
from PIL import Image
from src.poi_baseline import sha256_file,select_gallery,poi_scores,calibrate_policy,json_write,acceptance_matrix,evaluate
from src.poi_finetune import FinetunedPOIVerifier,embed_images,classification_metrics,transform_embeddings,fit_embedding_transform
from src.negative_baseline import evaluate_external

out=ROOT/'models/mobilenetv3_finetuned'
load=lambda path:json.loads(path.read_text(encoding='utf-8'))
metadata=load(out/'metadata.json');policy=load(out/'rejection_policy.json');config=load(out/'config.json');metrics=load(out/'metrics.json')
for name,digest in metadata['artifact_sha256'].items():assert sha256_file(out/name)==digest,name
for name,digest in metadata['input_split_sha256'].items():assert sha256_file(ROOT/f'data/annotations/{name}.csv')==digest,name
parent=ROOT/'models/baseline_mobilenetv3_negatives'
assert sha256_file(parent/'metadata.json')==metadata['parent_metadata_sha256']
parent_meta=load(parent/'metadata.json')
for name,digest in parent_meta['artifact_sha256'].items():assert sha256_file(parent/name)==digest,name
assert load(out/'class_mapping.json')==load(ROOT/'data/annotations/class_mapping.json')
history=pd.read_csv(out/'training_history.csv',keep_default_na=False)
best=history.sort_values(['true_accept_rate','macro_f1','accuracy','epoch'],ascending=[False,False,False,True]).iloc[0]
assert int(best.epoch)==metadata['best_epoch']
selection=load(out/'selection.json');assert not selection['test_used_for_selection']
checkpoint=torch.load(out/'best_checkpoint.pt',map_location='cpu',weights_only=True)
state=torch.load(out/'encoder.pt',map_location='cpu',weights_only=True)
assert checkpoint['epoch']==metadata['best_epoch']
for key,tensor in state.items():torch.testing.assert_close(tensor,checkpoint['encoder'][key],rtol=0,atol=0)
frames={name:pd.read_csv(out/f'splits/{name}.csv',keep_default_na=False) for name in metadata['input_split_sha256']}
vectors={name:np.load(out/f'{name}_embeddings.npy',allow_pickle=False) for name in frames}
projection=np.load(out/'embedding_transform.npz',allow_pickle=False)
gallery,gallery_values=select_gallery(frames['train'],vectors['train'],config['gallery_max_per_poi'])
pd.testing.assert_frame_equal(gallery,pd.read_csv(out/'gallery.csv',keep_default_na=False))
np.testing.assert_array_equal(gallery_values,np.load(out/'gallery_embeddings.npy',allow_pickle=False))
pois=metadata['poi_ids']
val_scores,_=poi_scores(vectors['val'],gallery_values,gallery.poi_id,pois)
neg_val_scores,_=poi_scores(vectors['negative_val'],gallery_values,gallery.poi_id,pois)
truth=np.array([pois.index(poi) for poi in frames['val'].poi_id])
new_policy,_=calibrate_policy(val_scores,truth,config,neg_val_scores)
assert new_policy==policy
for name in ['val','test']:
    scores,_=poi_scores(vectors[name],gallery_values,gallery.poi_id,pois)
    computed,*_=evaluate(scores,frames[name],pois,policy)
    computed.update(classification_metrics(scores,frames[name],pois))
    assert computed==metrics[name],name
    neg_scores,_=poi_scores(vectors[f'negative_{name}'],gallery_values,gallery.poi_id,pois)
    computed,*_=evaluate_external(neg_scores,frames[f'negative_{name}'],pois,policy)
    assert computed==metrics[f'negative_{name}']
cache_index=pd.read_csv(out/'cache_index.csv',keep_default_na=False)
for row in cache_index.itertuples():assert sha256_file(ROOT/row.cache_file)==row.cache_sha256
verifier=FinetunedPOIVerifier(out)
fresh_checks=0;max_error=0.
for name in ['test','negative_test']:
    sample=frames[name].iloc[:3]
    images=[]
    for row in sample.itertuples():
        cached=cache_index[cache_index.image_id.eq(row.image_id)].iloc[0]
        with Image.open(ROOT/cached.cache_file) as image:images.append(image.convert('RGB'))
    fresh=transform_embeddings(embed_images(verifier.encoder,images,3),verifier.embedding_transform)
    max_error=max(max_error,float(np.max(np.abs(fresh-vectors[name][:3]))))
    np.testing.assert_allclose(fresh,vectors[name][:3],atol=1e-6,rtol=1e-5)
    original_scores,_=poi_scores(vectors[name][:3],gallery_values,gallery.poi_id,pois)
    for n,row in enumerate(sample.itertuples()):
        target=row.poi_id if name=='test' else pois[int(original_scores[n].argmax())]
        result=verifier.verify(ROOT/row.file_path,target)
        index=pois.index(target)
        assert abs(result['score']-original_scores[n,index])<1e-6
        assert result['accepted']==bool(acceptance_matrix(original_scores,policy)[n,index])
        fresh_checks+=1
book=nbformat.read(ROOT/'notebooks/04_mobilenetv3_finetuning.ipynb',as_version=4)
nbformat.validate(book)
cells=[c for c in book.cells if c.cell_type=='code']
assert all(c.execution_count is not None for c in cells)
assert not any(o.output_type=='error' for c in cells for o in c.outputs)
checks={'status':'passed','parent_baseline_unchanged':True,'input_splits_unchanged':True,
        'selected_epoch_matches_validation_history':True,'checkpoint_weights_match_export':True,
        'train_gallery_reproduced':True,'val_policy_reproduced':True,'val_test_metrics_reproduced':True,
        'cache_image_hashes_checked':len(cache_index),'raw_sources_hashed_at_cache_creation':True,
        'fresh_embedding_and_raw_inference_checks':fresh_checks,'max_embedding_error':max_error,
        'executed_notebook_cells':len(cells)}
json_write(ROOT/'reports/mobilenetv3_finetuned/integrity_checks.json',checks)
print(json.dumps(checks,ensure_ascii=False,indent=2))
