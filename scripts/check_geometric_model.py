"""Проверить выбор validation-политики, метрики и совпадение batch/single."""
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
import nbformat
from src.poi_baseline import sha256_file,calibrate_policy,evaluate,json_write,acceptance_matrix
from src.poi_geometry import GeometricPOIVerifier,aggregate_scores
from src.poi_finetune import classification_metrics
from src.negative_baseline import evaluate_external

out=ROOT/'models/mobilenetv3_geometric'
load=lambda path:json.loads(path.read_text(encoding='utf-8'))
meta=load(out/'metadata.json');config=load(out/'config.json');policy=load(out/'rejection_policy.json')
selection=load(out/'selection.json');metrics=load(out/'metrics.json')
for name,digest in meta['artifact_sha256'].items():assert sha256_file(out/name)==digest,name
for name,digest in meta['input_split_sha256'].items():assert sha256_file(ROOT/f'data/annotations/{name}.csv')==digest,name
for directory,key in [('baseline_mobilenetv3_negatives','parent_metadata_sha256'),('mobilenetv3_calibrated','classifier_metadata_sha256')]:
    parent=ROOT/'models'/directory;assert sha256_file(parent/'metadata.json')==meta[key]
    parent_meta=load(parent/'metadata.json')
    for name,digest in parent_meta['artifact_sha256'].items():assert sha256_file(parent/name)==digest,name
train_ids=set(pd.read_csv(ROOT/'data/annotations/train.csv').image_id)
gallery=pd.read_csv(out/'gallery.csv',keep_default_na=False);assert set(gallery.image_id)<=train_ids
assert not selection['test_used_for_selection']
pois=meta['poi_ids'];mode=selection['aggregation'];fusion=selection['score_fusion']
frames={};scores={};matching={}
for name in ['val','test','negative_val','negative_test']:
    frames[name]=pd.read_csv(out/f'splits/{name}.csv',keep_default_na=False)
    matching[name]=np.load(out/f'{name}_matching.npz',allow_pickle=False)
    scores[name]=aggregate_scores(matching[name]['local_scores'],matching[name]['coarse'],mode,fusion)
    np.testing.assert_array_equal(scores[name],np.load(out/f'{name}_scores.npy',allow_pickle=False))
truth=np.array([pois.index(p) for p in frames['val'].poi_id])
new_policy,_=calibrate_policy(scores['val'],truth,config,scores['negative_val'])
new_policy['score_type']='sift_ransac_consistency_not_probability' if fusion=='geometry_only' else 'sift_ransac_plus_ridge_response_not_probability'
assert new_policy==policy
candidates=[]
for aggregation in config['score_aggregations']:
    for score_fusion in config['score_fusions']:
        vs=aggregate_scores(matching['val']['local_scores'],matching['val']['coarse'],aggregation,score_fusion)
        ns=aggregate_scores(matching['negative_val']['local_scores'],matching['negative_val']['coarse'],aggregation,score_fusion)
        p,_=calibrate_policy(vs,truth,config,ns)
        quality=classification_metrics(vs,frames['val'],pois)
        candidates.append(((p['true_accept_rate'],quality['macro_f1'],quality['accuracy']),aggregation,score_fusion))
winner=max(candidates,key=lambda row:row[0]);assert (winner[1],winner[2])==(mode,fusion)
for name in ['val','test']:
    computed,*_=evaluate(scores[name],frames[name],pois,policy)
    computed.update(classification_metrics(scores[name],frames[name],pois))
    external,*_=evaluate_external(scores[f'negative_{name}'],frames[f'negative_{name}'],pois,policy)
    computed.update(external_unknown_tested=True,open_set_far=external['any_poi_far'],open_set_far_definition='any_poi_per_external_image')
    assert computed==metrics[name]
    assert external==metrics[f'negative_{name}']
verifier=GeometricPOIVerifier(out)
checks=0;maximum_error=0.
for name in ['test','negative_test']:
    # Проверяем и поддержанный, и неподдержанный геометрией запрос, если есть.
    supported=matching[name]['local_scores'].max(axis=(1,2))>0
    indices=[]
    for exists in [True,False]:
        positions=np.flatnonzero(supported==exists)
        if len(positions):indices.append(int(positions[0]))
    for index in indices:
        row=frames[name].iloc[index]
        target=row.poi_id if name=='test' else pois[int(scores[name][index].argmax())]
        result=verifier.verify(ROOT/row.file_path,target)
        target_index=pois.index(target)
        error=abs(result['score']-scores[name][index,target_index]);maximum_error=max(maximum_error,error)
        assert error<1e-6,(name,index,error)
        assert result['accepted']==bool(acceptance_matrix(scores[name],policy)[index,target_index])
        checks+=1
assert verifier.verify(Path('not-read.jpg'),'unsupported')['reason']=='target_poi_not_supported'
book=nbformat.read(ROOT/'notebooks/06_mobilenetv3_geometric_verification.ipynb',as_version=4)
nbformat.validate(book)
cells=[c for c in book.cells if c.cell_type=='code']
assert all(c.execution_count is not None for c in cells)
assert not any(o.output_type=='error' for c in cells for o in c.outputs)
result={'status':'passed','old_models_unchanged':True,'input_splits_unchanged':True,
        'gallery_train_only':True,'validation_selection_reproduced':True,'validation_policy_reproduced':True,
        'all_metrics_reproduced':True,'raw_single_inference_checks':checks,'maximum_score_difference':maximum_error,
        'executed_notebook_cells':len(cells)}
json_write(ROOT/'reports/mobilenetv3_geometric/integrity_checks.json',result)
print(json.dumps(result,ensure_ascii=False,indent=2))
