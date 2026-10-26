"""Экспортировать проверенную галерею и веса без raw-фото и без переобучения."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--extra-site',type=Path)
parser.add_argument('--encoder',choices=['mobilenetv3_small','mobilenetv3_large','dinov2_small'],default='dinov2_small')
parser.add_argument('--output',type=Path,default=ROOT/'models/poi_service_dinov2')
args=parser.parse_args()
if args.extra_site:sys.path.append(str(args.extra_site.resolve()))
sys.path.insert(0,str(ROOT))
import pandas as pd
from src.poi_baseline import sha256_file,json_write
from src.poi_inference import BUNDLE_FILES
experiment=ROOT/'models/encoder_comparison_manual_all';source=experiment/args.encoder
load=lambda path:json.loads(path.read_text(encoding='utf-8'))
meta=load(source/'metadata.json')
for name,digest in meta['artifact_sha256'].items():
    if sha256_file(source/name)!=digest:raise ValueError(f'Изменён эксперимент: {name}')
selection=load(source/'selection.json')
if selection['test_used_for_selection']:raise ValueError('Невалидный протокол выбора')
weights=load(source/'weights.json');weight_path=ROOT/weights['file']
if sha256_file(weight_path)!=weights['sha256']:raise ValueError('Веса изменены')
out=args.output.resolve()
if not out.is_relative_to(ROOT.resolve()):raise ValueError('Экспорт должен оставаться в проекте')
if out.exists():raise FileExistsError('Существующий bundle не перезаписывается; задайте новый --output')
out.mkdir(parents=True)
gallery=source/selection['variant']
for original,destination in [(weight_path,'weights.pth'),(gallery/'gallery_embeddings.npy','gallery_embeddings.npy'),
    (gallery/'policy.json','rejection_policy.json'),(experiment/'class_mapping.json','class_mapping.json')]:
    shutil.copyfile(original,out/destination)
frame=pd.read_csv(gallery/'gallery.csv',keep_default_na=False)
columns=['image_id','poi_id','class_index','source','author','licence','source_url','license_url']
frame[[c for c in columns if c in frame]].to_csv(out/'gallery.csv',index=False)
json_write(out/'provenance.json',{'experiment':source.relative_to(ROOT).as_posix(),
    'experiment_metadata_sha256':sha256_file(source/'metadata.json'),'selection':selection,
    'weights':weights,'metrics':load(source/'metrics.json'),'training_performed':False})
digests={name:sha256_file(out/name) for name in sorted(BUNDLE_FILES)}
version=hashlib.sha256(json.dumps(digests,sort_keys=True).encode()).hexdigest()[:20]
json_write(out/'metadata.json',{'schema_version':1,'bundle_type':'poi_encoder_gallery','encoder':args.encoder,
    'model_version':version,'artifact_sha256':digests,'validated_for_production':False,
    'preprocessing':'original experiment EXIF/RGB + encoder transform + L2','gallery_images':len(frame)})
print(json.dumps({'bundle':out.relative_to(ROOT).as_posix(),'model_version':version,'gallery_images':len(frame)},ensure_ascii=False,indent=2))
