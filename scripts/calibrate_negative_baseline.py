"""Пересобрать политику отказа на проверенных external-negatives, offline."""
from pathlib import Path
import argparse
import os
import sys
import json

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--extra-site',type=Path)
parser.add_argument('--base',type=Path,default=ROOT/'models/baseline_mobilenetv3')
parser.add_argument('--output',type=Path,default=ROOT/'models/baseline_mobilenetv3_negatives')
args=parser.parse_args()
if args.extra_site:sys.path.append(str(args.extra_site.resolve()))
sys.path.insert(0,str(ROOT))
cache=ROOT/'.runtime/matplotlib'
cache.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(cache))
from src.negative_baseline import recalibrate_with_negatives
policy,metrics=recalibrate_with_negatives(ROOT,args.base,args.output)
print(json.dumps({'policy':policy,'metrics':metrics},ensure_ascii=False,indent=2))
