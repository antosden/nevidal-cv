"""Проверить полноту manual, атрибуцию и сохранение прежнего разбиения."""
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
from src.dataset_eda import IMAGE_SUFFIXES,selected_present
from src.poi_baseline import sha256_file,json_write,load_splits
frame=pd.read_csv(ROOT/'data/annotations/dataset_manifest.csv',keep_default_na=False)
present=frame[selected_present(frame)]
manual=present[present.source.eq('manual')]
files={p.relative_to(ROOT).as_posix() for p in (ROOT/'data/raw/manual').rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
assert set(manual.file_path)==files
assert manual.author.eq('Shilkova A.M.').all()
assert manual.licence.eq('Public Domain').all()
assert manual.label_status.eq('verified').all()
anchors=pd.read_csv(ROOT/'data/annotations/split_anchors.csv',keep_default_na=False)
anchors=anchors[anchors.split.ne('')]
linked=frame.merge(anchors[['image_id','split','sha256']],on='image_id',suffixes=('','_old'))
assert linked.sha256.eq(linked.sha256_old).all()
released=linked[linked.split.ne('')]
assert released.split.eq(released.split_old).all()
splits,mapping=load_splits(ROOT)
old=json.loads((ROOT/'models/baseline_mobilenetv3_negatives/metadata.json').read_text())
for split in ['negative_val','negative_test']:
    assert sha256_file(ROOT/f'data/annotations/{split}.csv')==old['artifact_sha256'][f'splits/{split}.csv']
stats={'available_images':len(present),'available_manual':len(manual),'poi_count':len(mapping),
       'released_images':sum(map(len,splits.values())),'splits':{k:len(v) for k,v in splits.items()},
       'previous_released_images':len(anchors),'previous_images_now_excluded':int(linked.split.eq('').sum()),
       'previous_released_split_changes':0,'manual_files_missing_from_manifest':0,
       'manual_attribution_valid':True,'negative_splits_unchanged':True,
       'exclusions':frame.loc[frame.exclusion_reason.ne(''),'exclusion_reason'].value_counts().to_dict()}
json_write(ROOT/'reports/manual_enrichment_checks.json',stats)
print(json.dumps(stats,ensure_ascii=False,indent=2))
