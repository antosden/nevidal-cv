"""Дообучение MobileNetV3-Small для POI, локально и без скачивания весов."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--extra-site',type=Path)
parser.add_argument('--config',type=Path,default=ROOT/'config/mobilenetv3_finetune.json')
parser.add_argument('--output',type=Path,default=ROOT/'models/mobilenetv3_finetuned')
args=parser.parse_args()
if args.extra_site:sys.path.append(str(args.extra_site.resolve()))
sys.path.insert(0,str(ROOT))
cache=ROOT/'.runtime/matplotlib';cache.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(cache))
from src.poi_finetune import train_model
metadata,metrics=train_model(ROOT,args.output,args.config)
print(json.dumps({'best_epoch':metadata['best_epoch'],'metrics':metrics},ensure_ascii=False,indent=2))
