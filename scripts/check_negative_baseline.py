"""Независимо проверить negatives, артефакты, val-политику и сохранность baseline."""
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
from src.poi_baseline import (sha256_file,poi_scores,calibrate_policy,load_splits,POIVerifier,
                              extract_embeddings,acceptance_matrix,json_write,evaluate)
from src.negative_dataset import split_negative_pois
from src.dataset_eda import IMAGE_SUFFIXES
from src.negative_baseline import evaluate_external

out=ROOT/'models/baseline_mobilenetv3_negatives'
base=ROOT/'models/baseline_mobilenetv3'
load=lambda path:json.loads(path.read_text(encoding='utf-8'))
meta,policy,config,metrics=map(load,[out/'metadata.json',out/'rejection_policy.json',out/'config.json',out/'metrics.json'])
old=load(base/'metadata.json')
assert sha256_file(base/'metadata.json')==meta['parent_baseline_metadata_sha256']
for folder,metadata in [(out,meta),(base,old)]:
    for name,digest in metadata['artifact_sha256'].items(): assert sha256_file(folder/name)==digest,name
assert meta['reference_version']==old['reference_version']
for name in ['imagenet_weights.pth','gallery.csv','gallery_embeddings.npy','class_mapping.json']:
    assert sha256_file(out/name)==sha256_file(base/name),name
positive,mapping=load_splits(ROOT)
for name,digest in old['split_sha256'].items():
    assert sha256_file(ROOT/f'data/annotations/{name}.csv')==digest
negative={s:pd.read_csv(ROOT/f'data/annotations/negative_{s}.csv',keep_default_na=False) for s in ['val','test']}
manifest=pd.read_csv(ROOT/'data/annotations/negative_manifest.csv',keep_default_na=False)
repeated=split_negative_pois(manifest.assign(split=''))
assert repeated.set_index('image_id').split.to_dict()==manifest.set_index('image_id').split.to_dict()
assert len(manifest[manifest.active])==sum(map(len,negative.values()))
assert manifest.loc[~manifest.active,'split'].eq('').all()
for frame in negative.values():
    assert frame.label_status.eq('verified').all() and frame.poi_id.eq('').all()
    assert {'source','author','licence'}<=set(frame.columns)
    assert not set(frame.source_poi_id)&set(mapping)
for field in ['source_poi_id','duplicate_group','split_group','image_id','sha256','commons_file_id']:
    assert not (set(negative['val'][field].astype(str))-{''})&(set(negative['test'][field].astype(str))-{''}),field
snapshot=load(ROOT/'reports/negatives/source_snapshot.json')
actual={p.relative_to(ROOT).as_posix() for p in (ROOT/'data/raw/wikimedia_negatives').rglob('*')
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
assert actual==set(snapshot['images']), 'Изменился состав raw negatives'
assert sha256_file(ROOT/'data/raw/wikimedia_negatives/metadata.csv')==snapshot['metadata_sha256']
assert sha256_file(ROOT/'config/wikimedia_negatives.csv')==snapshot['config_sha256']
assert sha256_file(ROOT/'data/annotations/dataset_manifest.csv')==snapshot['positive_manifest_sha256']
from tqdm.auto import tqdm
for path,digest in tqdm(snapshot['images'].items(),desc='Проверка неизменности raw negatives'):
    assert sha256_file(ROOT/path)==digest,path
vectors={s:np.load(out/f'negative_{s}_embeddings.npy',allow_pickle=False) for s in ['val','test']}
gallery=pd.read_csv(out/'gallery.csv',keep_default_na=False)
values=np.load(out/'gallery_embeddings.npy',allow_pickle=False)
pois=meta['poi_ids']
pos_val=np.load(out/'val_embeddings.npy',allow_pickle=False)
positive_scores,_=poi_scores(pos_val,values,gallery.poi_id,pois)
negative_scores,_=poi_scores(vectors['val'],values,gallery.poi_id,pois)
new_policy,_=calibrate_policy(positive_scores,np.array([pois.index(p) for p in positive['val'].poi_id]),config,negative_scores)
assert new_policy==policy
for name in ['val','test']:
    scores,_=poi_scores(vectors[name],values,gallery.poi_id,pois)
    stats,_,_=evaluate_external(scores,negative[name],pois,policy)
    assert stats==metrics[f'negative_{name}']
    pos=np.load(out/f'{name}_embeddings.npy',allow_pickle=False)
    pos_scores,_=poi_scores(pos,values,gallery.poi_id,pois)
    stats,*_=evaluate(pos_scores,positive[name],pois,policy)
    for key,value in stats.items():
        if key not in ['external_unknown_tested','open_set_far']:assert value==metrics[name][key]
verifier=POIVerifier(out)
sample=negative['test'].iloc[:3]
fresh=extract_embeddings(ROOT,sample,verifier.model,verifier.transform,3)
np.testing.assert_allclose(fresh,vectors['test'][:3],atol=1e-6,rtol=1e-5)
scores,_=poi_scores(fresh,values,gallery.poi_id,pois)
original_scores,_=poi_scores(vectors['test'][:3],values,gallery.poi_id,pois)
np.testing.assert_array_equal(acceptance_matrix(scores,policy),acceptance_matrix(original_scores,policy))
book=nbformat.read(ROOT/'notebooks/03_negative_dataset_and_rejection.ipynb',as_version=4)
nbformat.validate(book)
cells=[c for c in book.cells if c.cell_type=='code']
assert all(c.execution_count is not None for c in cells)
assert not any(o.output_type=='error' for c in cells for o in c.outputs)
checks={'status':'passed','raw_images_checked':len(snapshot['images']),
        'positive_baseline_unchanged':True,'positive_splits_unchanged':True,'gallery_unchanged':True,
        'poi_and_duplicate_groups_disjoint':True,'negative_split_reproduced':True,
        'val_only_thresholds_reproduced':True,'val_test_metrics_reproduced':True,
        'fresh_embedding_checks':len(sample),'max_embedding_error':float(np.max(np.abs(fresh-vectors['test'][:3]))),
        'executed_notebook_cells':len(cells)}
json_write(ROOT/'reports/baseline_mobilenetv3_negatives/integrity_checks.json',checks)
print(json.dumps(checks,ensure_ascii=False,indent=2))
