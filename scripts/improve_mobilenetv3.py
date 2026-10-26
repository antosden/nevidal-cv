"""Выбрать классификатор и per-POI отказ только по validation; затем оценить test."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--extra-site',type=Path)
parser.add_argument('--output',type=Path,default=ROOT/'models/mobilenetv3_calibrated')
args=parser.parse_args()
if args.extra_site:sys.path.append(str(args.extra_site.resolve()))
sys.path.insert(0,str(ROOT))
from src.poi_metric import export_calibrated
meta,metrics=export_calibrated(ROOT,args.output)
print(json.dumps({'scorer':meta['scorer'],'metrics':metrics},indent=2))
