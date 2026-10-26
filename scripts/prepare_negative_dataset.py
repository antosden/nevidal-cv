"""Подготовить подтверждённые пользователем negatives, не меняя исходники."""
import argparse
from pathlib import Path
import sys
import json

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--extra-site',type=Path)
parser.add_argument('--confirm-reviewed',action='store_true',required=True,
                    help='Фиксировать явное подтверждение просмотра оставшихся negative-файлов')
args=parser.parse_args()
if args.extra_site:sys.path.append(str(args.extra_site.resolve()))
sys.path.insert(0,str(ROOT))
from src.negative_dataset import prepare_negatives
frame,summary=prepare_negatives(ROOT,args.confirm_reviewed)
print(json.dumps(summary,ensure_ascii=False,indent=2))
