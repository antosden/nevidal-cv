"""Согласовать ручную очистку Wikimedia и добавление manual, не меняя raw-файлы."""
from pathlib import Path
import argparse
import datetime
import json
import shutil
import sys
import uuid
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def backup(root):
    folder=root/'_archive'/('dataset_update_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True)
    candidates=[root/'data/annotations',root/'reports',root/'notebooks/01_dataset_eda.ipynb']
    with zipfile.ZipFile(folder/'previous_dataset.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for candidate in candidates:
            files=sorted(p for p in candidate.rglob('*') if p.is_file()) if candidate.is_dir() else [candidate] if candidate.exists() else []
            for path in files:
                if not path.resolve().is_relative_to(root):raise ValueError('Ресурс вне workspace')
                archive.write(path,path.relative_to(root).as_posix())
    # Архивируем прежние visual derivatives, чтобы glob не показывал удалённые кадры.
    figures=root/'reports/figures'
    destination=folder/'figures'
    if figures.exists():
        if not figures.resolve().is_relative_to(root) or not destination.resolve().is_relative_to(root):
            raise ValueError('Недопустимый путь переноса')
        shutil.move(str(figures),str(destination))
    return folder


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-wikimedia',action='store_true')
    parser.add_argument('--confirm-manual',action='store_true')
    parser.add_argument('--extra-site',type=Path)
    args=parser.parse_args()
    if args.extra_site:sys.path.append(str(args.extra_site.resolve()))
    from src.dataset_eda import prepare_paths,inventory,find_duplicates,reconcile_reviews
    root=prepare_paths(ROOT)
    archive=backup(root)
    frame,hashes,mapping=inventory(root)
    frame,pairs=find_duplicates(frame,hashes,root)
    edits=reconcile_reviews(frame,root,args.confirm_wikimedia,args.confirm_manual)
    result={'audit_records':len(frame),'current_images':int(frame.file_exists.sum()),
            'statuses':edits.label_status.value_counts().to_dict(),'poi_count':len(mapping),
            'backup':archive.relative_to(root).as_posix()}
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
