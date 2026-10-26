"""Проверить выпуск baseline повторными вычислениями, без настройки по test."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--extra-site',type=Path)
parser.add_argument('--artifacts',type=Path,default=ROOT/'models/baseline_mobilenetv3')
args=parser.parse_args()
if args.extra_site: sys.path.append(str(args.extra_site.resolve()))
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import nbformat
from src.poi_baseline import (sha256_file,load_splits,select_gallery,poi_scores,calibrate_policy,
                              evaluate,POIVerifier,extract_embeddings,acceptance_matrix,json_write)

out=args.artifacts.resolve()
load=lambda name:json.loads((out/name).read_text(encoding='utf-8'))
metadata,policy,config,metrics=map(load,['metadata.json','rejection_policy.json','config.json','metrics.json'])
splits,mapping=load_splits(ROOT)
for name,digest in metadata['artifact_sha256'].items():
    assert sha256_file(out/name)==digest, name
for name,digest in metadata['split_sha256'].items():
    assert sha256_file(ROOT/f'data/annotations/{name}.csv')==digest, name
assert sha256_file(ROOT/'data/annotations/class_mapping.json')==metadata['class_mapping_sha256']
assert json.loads((ROOT/'config/baseline_mobilenetv3.json').read_text())==config
vectors={name:np.load(out/f'{name}_embeddings.npy',allow_pickle=False) for name in splits}
gallery,values=select_gallery(splits['train'],vectors['train'],config['gallery_max_per_poi'])
saved_gallery=pd.read_csv(out/'gallery.csv',keep_default_na=False)
pd.testing.assert_frame_equal(gallery,saved_gallery)
np.testing.assert_array_equal(values,np.load(out/'gallery_embeddings.npy',allow_pickle=False))
poi_ids=metadata['poi_ids']
val_scores,_=poi_scores(vectors['val'],values,gallery.poi_id,poi_ids)
truths=np.array([poi_ids.index(p) for p in splits['val'].poi_id])
new_policy,_=calibrate_policy(val_scores,truths,config)
assert policy==new_policy, 'Изменились пороги'
for name in ['val','test']:
    scores,_=poi_scores(vectors[name],values,gallery.poi_id,poi_ids)
    recalculated,*_=evaluate(scores,splits[name],poi_ids,policy)
    for key,value in recalculated.items(): assert metrics[name][key]==value,(name,key)

verifier=POIVerifier(out)
fresh_checks=0
max_error=0.
for name in ['val','test']:
    sample=splits[name].iloc[:3].copy()
    fresh=extract_embeddings(ROOT,sample,verifier.model,verifier.transform,3)
    error=float(np.max(np.abs(fresh-vectors[name][:3])))
    max_error=max(max_error,error)
    np.testing.assert_allclose(fresh,vectors[name][:3],atol=1e-6,rtol=1e-5)
    fresh_scores,_=poi_scores(fresh,values,gallery.poi_id,poi_ids)
    original_scores,_=poi_scores(vectors[name][:3],values,gallery.poi_id,poi_ids)
    np.testing.assert_array_equal(acceptance_matrix(fresh_scores,policy),acceptance_matrix(original_scores,policy))
    for index,row in sample.iterrows():
        result=verifier.verify(ROOT/(row.processed_path or row.file_path),row.poi_id)
        own=poi_ids.index(row.poi_id)
        assert result['accepted']==bool(acceptance_matrix(original_scores,policy)[index,own])
        assert abs(result['score']-original_scores[index,own])<1e-6
        fresh_checks+=1
assert verifier.verify(Path('not-read.jpg'),'unsupported')['reason']=='target_poi_not_supported'
book=nbformat.read(ROOT/'notebooks/02_poi_verification_baseline.ipynb',as_version=4)
nbformat.validate(book)
cells=[c for c in book.cells if c.cell_type=='code']
assert all(c.execution_count is not None for c in cells)
assert not any(o.output_type=='error' for c in cells for o in c.outputs)
result={'status':'passed','gallery_reproduced_exactly':True,'val_policy_reproduced_exactly':True,
        'val_test_metrics_reproduced_exactly':True,'artifact_hashes_checked':len(metadata['artifact_sha256']),
        'fresh_embedding_and_inference_checks':fresh_checks,'max_embedding_absolute_error':max_error,
        'notebook_executed_code_cells':len(cells)}
json_write(ROOT/'reports/baseline_mobilenetv3/reproducibility_checks.json',result)
print(json.dumps(result,ensure_ascii=False,indent=2))
