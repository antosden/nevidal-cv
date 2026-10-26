"""Сжать встроенные листы без изменения figure-файлов и обновить итог через Jupyter."""
import argparse
import base64
import io
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--extra-site',type=Path)
args=parser.parse_args()
extra=[str(args.extra_site.resolve())] if args.extra_site else []
sys.path.extend(extra)
for key,folder in [('JUPYTER_RUNTIME_DIR','jupyter'),('IPYTHONDIR','ipython'),('MPLCONFIGDIR','matplotlib')]:
    path=ROOT/'.runtime'/folder;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
os.environ['MPLBACKEND']='Agg'
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from PIL import Image

path=ROOT/'notebooks/01_dataset_eda.ipynb'
book=nbformat.read(path,as_version=4)
setup=next(c for c in book.cells if c.cell_type=='code')
setup.source=setup.source.replace('from IPython.display import display, Markdown\n',
                                'from IPython.display import display, Markdown, Image as DisplayImage\n')
for cell in book.cells:
    if cell.cell_type!='code':continue
    if 'display(Image.open(path))' in cell.source:
        cell.source=cell.source.replace('display(Image.open(path))','display(DisplayImage(filename=str(path)))')
        # Изменение только представления inline-контактов, не фото/метрик/figure-файлов.
        for output in cell.outputs:
            data=output.get('data',{})
            if 'image/png' not in data:continue
            with Image.open(io.BytesIO(base64.b64decode(data.pop('image/png')))) as image:
                buffer=io.BytesIO();image.convert('RGB').save(buffer,format='JPEG',quality=86)
            data['image/jpeg']=base64.b64encode(buffer.getvalue()).decode('ascii')
    cell.source=cell.source.replace('display(Image.open(ROOT/\'reports/figures\'/name))',
                                  'display(DisplayImage(filename=str(ROOT/\'reports/figures\'/name)))')
last=next(c for c in reversed(book.cells) if c.cell_type=='code')
prelude=nbformat.v4.new_code_cell('''manifest=pd.read_csv(ROOT/'data/annotations/dataset_manifest.csv',keep_default_na=False)
duplicate_pairs=pd.read_csv(ROOT/'data/annotations/duplicate_pairs.csv',keep_default_na=False)
ready=bool(json.loads((ROOT/'reports/eda_statistics.json').read_text(encoding='utf-8'))['training_ready'])
''')
partial=nbformat.v4.new_notebook(cells=[setup,prelude,last],metadata=book.metadata)
manager=KernelManager(kernel_name='python3')
if extra:
    bootstrap='import sys,runpy;sys.path.extend('+repr(extra)+');runpy.run_module("ipykernel_launcher",run_name="__main__")'
    manager.kernel_spec.argv=[sys.executable,'-c',bootstrap,'-f','{connection_file}']
else:manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
NotebookClient(partial,km=manager,timeout=300,resources={'metadata':{'path':str(ROOT)}}).execute()
book.metadata['inline_contact_sheets']='JPEG previews; standalone figures unchanged'
nbformat.validate(book);nbformat.write(book,path)
(ROOT/'reports/eda_environment.json').write_text(
    json.dumps(dict(book.metadata['execution_environment']),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Notebook finalized:',round(path.stat().st_size/1024**2,1),'MB')
