"""Геометрическая проверка POI: frozen MobileNetV3 → SIFT → RANSAC."""
from pathlib import Path
import argparse
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--extra-site',type=Path)
parser.add_argument('--output',type=Path,default=ROOT/'models/mobilenetv3_geometric')
parser.add_argument('--stage',choices=['validate','finalize','all'],default='all')
args=parser.parse_args()
if args.extra_site:sys.path.append(str(args.extra_site.resolve()))
sys.path.insert(0,str(ROOT))
cache=ROOT/'.runtime/matplotlib';cache.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(cache))
from src.poi_geometry import run_geometric_validation,finalize_geometric
if args.stage in ['validate','all']:
    run_geometric_validation(ROOT,args.output)
    print('Geometric validation complete; selection frozen.')
if args.stage in ['finalize','all']:
    metadata,metrics=finalize_geometric(ROOT,args.output)
    import json
    print(json.dumps(metrics,indent=2))
