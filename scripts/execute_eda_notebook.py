"""Выполнить EDA через Jupyter из текущей Python-среды."""
import os
from pathlib import Path
import sys
import argparse
import json

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--extra-site',type=Path,help='Добавить вторичные site-packages после пакетов текущего Python')
parser.add_argument('--reuse-inventory',action='store_true',help='Использовать уже рассчитанные текущие метрики и хэши')
args=parser.parse_args()
extra=[str(args.extra_site.resolve())] if args.extra_site else []
sys.path.extend(extra)
for key,folder in [('JUPYTER_RUNTIME_DIR','jupyter'),('IPYTHONDIR','ipython'),('MPLCONFIGDIR','matplotlib')]:
    path=ROOT/'.runtime'/folder;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
os.environ['MPLBACKEND']='Agg'
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

path=ROOT/'notebooks/01_dataset_eda.ipynb'
book=nbformat.read(path,as_version=4);nbformat.validate(book)
original_sources={cell.id:cell.source for cell in book.cells}
if args.reuse_inventory:
    for cell in book.cells:
        if cell.cell_type!='code':continue
        cell.source=cell.source.replace('initial, hashes, mapping=inventory(ROOT)',
                                       'initial, hashes, mapping=load_inventory(ROOT)')
        cell.source=cell.source.replace('initial, duplicate_pairs=find_duplicates(initial,hashes,ROOT)',
            "duplicate_pairs=pd.read_csv(ROOT/'data/annotations/duplicate_pairs.csv',keep_default_na=False)")
manager=KernelManager(kernel_name='python3')
if extra:
    bootstrap='import sys,runpy;sys.path.extend('+repr(extra)+');runpy.run_module("ipykernel_launcher",run_name="__main__")'
    manager.kernel_spec.argv=[sys.executable,'-c',bootstrap,'-f','{connection_file}']
else:
    manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
client=NotebookClient(book,km=manager,timeout=1800,resources={'metadata':{'path':str(ROOT)}})
def progress(cell,cell_index,**kwargs):
    if cell.cell_type=='code':print(f'Cell {cell_index+1}',flush=True)
client.on_cell_start=progress
client.execute()
for cell in book.cells:cell.source=original_sources[cell.id]
import pandas,numpy,cv2,matplotlib
book.metadata['execution_environment']={'method':'Jupyter kernel via nbclient','python':sys.version.split()[0],
    'pandas':pandas.__version__,'numpy':numpy.__version__,'opencv':cv2.__version__,
    'matplotlib':matplotlib.__version__,'secondary_site_packages':bool(extra),
    'reused_current_inventory':args.reuse_inventory}
(ROOT/'reports/eda_environment.json').write_text(
    json.dumps(dict(book.metadata['execution_environment']),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
nbformat.validate(book);nbformat.write(book,path)
print('Executed notebook saved.',flush=True)
