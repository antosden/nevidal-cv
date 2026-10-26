"""Выполнить baseline-ноутбук в настоящем Jupyter-ядре."""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--extra-site', type=Path)
parser.add_argument('--notebook', type=Path, default=ROOT/'notebooks/02_poi_verification_baseline.ipynb')
args = parser.parse_args()
extra = [str(args.extra_site.resolve())] if args.extra_site else []
sys.path.extend(extra)
for key, folder in [('JUPYTER_RUNTIME_DIR','jupyter'), ('IPYTHONDIR','ipython'), ('MPLCONFIGDIR','matplotlib')]:
    path = ROOT/'.runtime'/folder
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
os.environ['MPLBACKEND'] = 'Agg'
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

path = args.notebook.resolve()
book = nbformat.read(path, as_version=4)
manager = KernelManager(kernel_name='python3')
bootstrap = 'import sys,runpy;sys.path.extend('+repr(extra)+');runpy.run_module("ipykernel_launcher",run_name="__main__")'
manager.kernel_spec.argv = [sys.executable, '-c', bootstrap, '-f', '{connection_file}']
NotebookClient(book, km=manager, timeout=1800, resources={'metadata':{'path':str(ROOT)}}).execute()
nbformat.validate(book)
nbformat.write(book, path)
print('Executed:', path)
