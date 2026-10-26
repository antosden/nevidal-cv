"""Независимая проверка train-only Ridge, val-порогов и test-предсказаний."""
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
from src.poi_metric import fit_ridge,ridge_scores,calibrate_per_poi,CalibratedPOIVerifier
from src.poi_finetune import classification_metrics
from src.poi_baseline import configure_torch,sha256_file,json_write,poi_scores,acceptance_matrix,evaluate
from src.negative_baseline import evaluate_external

configure_torch(42,4)
out=ROOT/'models/mobilenetv3_calibrated';parent=ROOT/'models/baseline_mobilenetv3_negatives'
load=lambda path:json.loads(path.read_text(encoding='utf-8'))
meta,policy,config,metrics=map(load,[out/'metadata.json',out/'rejection_policy.json',out/'config.json',out/'metrics.json'])
for name,digest in meta['artifact_sha256'].items():assert sha256_file(out/name)==digest,name
for name,digest in meta['input_split_sha256'].items():assert sha256_file(ROOT/f'data/annotations/{name}.csv')==digest,name
parent_meta=load(parent/'metadata.json');assert sha256_file(parent/'metadata.json')==meta['parent_metadata_sha256']
for name,digest in parent_meta['artifact_sha256'].items():assert sha256_file(parent/name)==digest,name
frames={name:pd.read_csv(out/f'splits/{name}.csv',keep_default_na=False).sort_values('image_id').reset_index(drop=True) for name in ['train','val','test','negative_val','negative_test']}
vectors={name:np.load(parent/f'{name}_embeddings.npy',allow_pickle=False) for name in frames}
pois=meta['poi_ids'];labels=np.array([pois.index(p) for p in frames['train'].poi_id])
parameters=fit_ridge(vectors['train'],labels,len(pois),meta['ridge_alpha'])
stored=np.load(out/'ridge_parameters.npz',allow_pickle=False)
for key in parameters:np.testing.assert_array_equal(parameters[key],stored[key])
truth=np.array([pois.index(p) for p in frames['val'].poi_id])
vs=ridge_scores(vectors['val'],parameters);ns=ridge_scores(vectors['negative_val'],parameters)
assert calibrate_per_poi(vs,truth,ns,config)==policy
for name in ['val','test']:
    scores=ridge_scores(vectors[name],parameters)
    summary,*_=evaluate(scores,frames[name],pois,policy)
    summary.update(classification_metrics(scores,frames[name],pois))
    ext_scores=ridge_scores(vectors[f'negative_{name}'],parameters)
    external,*_=evaluate_external(ext_scores,frames[f'negative_{name}'],pois,policy)
    summary.update(external_unknown_tested=True,open_set_far=external['any_poi_far'],open_set_far_definition='any_poi_per_external_image')
    assert summary==metrics[name]
    assert external==metrics[f'negative_{name}']
verifier=CalibratedPOIVerifier(out)
fresh=0
for name in ['test','negative_test']:
    scores=ridge_scores(vectors[name][:3],parameters)
    for n,row in enumerate(frames[name].iloc[:3].itertuples()):
        target=row.poi_id if name=='test' else pois[int(scores[n].argmax())]
        result=verifier.verify(ROOT/row.file_path,target)
        index=pois.index(target)
        assert abs(result['score']-scores[n,index])<1e-6
        assert result['accepted']==bool(acceptance_matrix(scores,policy)[n,index])
        fresh+=1
assert verifier.verify(Path('not-read.jpg'),'unsupported')['reason']=='target_poi_not_supported'
book=nbformat.read(ROOT/'notebooks/05_mobilenetv3_classifier_and_calibration.ipynb',as_version=4)
nbformat.validate(book)
cells=[c for c in book.cells if c.cell_type=='code']
assert all(c.execution_count is not None for c in cells)
assert not any(o.output_type=='error' for c in cells for o in c.outputs)
checks={'status':'passed','parent_baseline_unchanged':True,'input_splits_unchanged':True,
        'ridge_fit_reproduced_exactly':True,'validation_policy_reproduced_exactly':True,
        'validation_and_test_metrics_reproduced':True,'fresh_raw_inference_checks':fresh,
        'executed_notebook_cells':len(cells),'production_promotion':False,
        'promotion_reason':'External test FAR exceeds 1% target.'}
json_write(ROOT/'reports/mobilenetv3_calibrated/integrity_checks.json',checks)
print(json.dumps(checks,ensure_ascii=False,indent=2))
