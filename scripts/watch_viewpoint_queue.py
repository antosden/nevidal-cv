"""Читаемая очередь 12 обучений, без запуска или изменения моделей."""
from pathlib import Path
import argparse
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.viewpoint_audit import read,folders


def snapshot():
    data,report,models,_=folders(ROOT)
    config=json.loads((data/'final_config.json').read_text(encoding='utf-8'))
    done=0;lines=[]
    for seed in config['seeds']:
        for arch in config['architectures']:
            for mode in config['modes']:
                name=f'{arch}_{mode}_seed{seed}';run=models/'runs'/name
                try:
                    status=json.loads((run/'status.json').read_text(encoding='utf-8'))
                    epoch=status['epoch'];state=status['status']
                except (FileNotFoundError,json.JSONDecodeError):state='queued';epoch=0
                done+=state=='complete'
                history=read(run/'history.csv')
                score=f"val F1={float(history[-1]['macro_f1']):.3f}" if history else ''
                lines.append(f'{name:23} {state:12} {epoch:2}/{config["epochs"]} {score}')
    return f'Completed {done}/12 | test unopened: {not (data/"test_opened.json").exists()}\n'+'\n'.join(lines),done


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    while True:
        message,done=snapshot();print('\n'+message,flush=True)
        if not args.watch or done==12:break
        time.sleep(15)
