"""Отдельный CLI нового исследования ракурсов."""
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['audit','reconcile','finalize','pilot','main','evaluate']);args=parser.parse_args()
    if args.command=='audit':
        from src.viewpoint_audit import audit
        audit(ROOT)
    elif args.command=='reconcile':
        from src.viewpoint_audit import reconcile_history
        reconcile_history(ROOT)
    elif args.command=='finalize':
        from src.viewpoint_finalize import finalize
        print(finalize(ROOT))
    elif args.command in ('pilot','main'):
        from src.viewpoint_training import fit,pilot_review
        from src.viewpoint_audit import folders
        import json
        data,report,_,_=folders(ROOT)
        if args.command=='main':
            approval=json.loads((report/'pilot_decision.json').read_text(encoding='utf-8'))
            if not approval['continue_unchanged']:raise ValueError('Пилот требует изменения рецепта')
        for seed in ([42] if args.command=='pilot' else [43,44]):
            for architecture in ['small','large']:
                for mode in ['O','R']:fit(ROOT,architecture,mode,seed)
        if args.command=='pilot':print(pilot_review(ROOT))
    elif args.command=='evaluate':
        from src.viewpoint_evaluation import evaluate
        evaluate(ROOT)
