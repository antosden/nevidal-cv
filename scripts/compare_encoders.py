"""Сравнить три encoder на одном выпуске датасета; поддерживает продолжение."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--extra-site',type=Path)
parser.add_argument('--output',type=Path,default=ROOT/'models/encoder_comparison_manual_all')
parser.add_argument('--debug-stacks',action='store_true',help='Диагностировать длительные импорты/ожидания')
parser.add_argument('--only-encoder',choices=['mobilenetv3_small','mobilenetv3_large','dinov2_small'])
parser.add_argument('--stage',choices=['all','validation','test'],default='all')
parser.add_argument('--reuse-input-audit',action='store_true',help='Использовать прошлый полный SHA-аудит при неизменных splits, размере/mtime файлов')
args=parser.parse_args()
if args.debug_stacks:
    import faulthandler
    faulthandler.dump_traceback_later(90,repeat=True)
if args.extra_site:sys.path.append(str(args.extra_site.resolve()))
sys.path.insert(0,str(ROOT))
print('Loading comparison modules...',flush=True)
from src.encoder_comparison import compare
print('Verifying dataset and starting comparison...',flush=True)
status=compare(ROOT,args.output,json.loads((ROOT/'config/encoder_comparison.json').read_text()),args.only_encoder,args.stage,args.reuse_input_audit)
print(json.dumps(status,ensure_ascii=False,indent=2))
if args.debug_stacks:faulthandler.cancel_dump_traceback_later()
