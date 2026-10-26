"""Изолировать положительные группы, связанные с фиксированными external negatives."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--extra-site',type=Path)
args=parser.parse_args()
if args.extra_site:sys.path.append(str(args.extra_site.resolve()))
sys.path.insert(0,str(ROOT))
import pandas as pd
from src.negative_dataset import duplicate_links
from src.dataset_eda import split_components,selected_present,apply_reviews
positive=pd.read_csv(ROOT/'data/annotations/initial_manifest.csv',keep_default_na=False)
positive=apply_reviews(positive,ROOT,json.loads((ROOT/'data/annotations/class_mapping.json').read_text()))
positive=positive[selected_present(positive)].copy()
negative=pd.concat([pd.read_csv(ROOT/f'data/annotations/negative_{name}.csv',keep_default_na=False) for name in ['val','test']],ignore_index=True)
hashes=json.loads((ROOT/'reports/image_fingerprints.json').read_text())
hashes.update(json.loads((ROOT/'reports/negatives/image_fingerprints.json').read_text()))
combined=pd.concat([positive[['image_id','sha256','commons_file_id']],negative[['image_id','sha256','commons_file_id']]],ignore_index=True)
uf,pairs=duplicate_links(combined,hashes)
positive['component']=split_components(positive)
for _,group in positive.groupby('component'):
    for uid in group.image_id:uf.union(group.image_id.iloc[0],uid)
negative_roots={uf.find(uid) for uid in negative.image_id}
blocked=positive[positive.image_id.map(lambda uid:uf.find(uid) in negative_roots)].copy()
blocked[['image_id','file_path','poi_id','sha256']].to_csv(ROOT/'data/annotations/external_overlap_quarantine.csv',index=False)
pairs.to_csv(ROOT/'reports/enriched_cross_source_duplicate_pairs.csv',index=False)
print(json.dumps({'quarantined_positive_images':len(blocked),'negative_splits_changed':False}))
