"""Сформировать и выполнить только новый notebook10 по готовым артефактам."""
from pathlib import Path
import os
import sys
import json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['MPLCONFIGDIR']=str(ROOT/'.runtime/matplotlib')
os.environ['IPYTHONDIR']=str(ROOT/'.runtime/ipython')
from src.viewpoint_audit import folders,dump
from src.viewpoint_reporting import create_notebook


if __name__=='__main__':
    _,report,_,_=folders(ROOT)
    assert json.loads((report/'experiment_status.json').read_text())['completed_runs']==12
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    path=create_notebook(ROOT);book=nbformat.read(path,as_version=4)
    manager=KernelManager(kernel_name='python3')
    manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    manager.kernel_spec.env={**os.environ,'PYTHONIOENCODING':'utf-8'}
    try:
        NotebookClient(book,km=manager,timeout=180,resources={'metadata':{'path':str(ROOT)}}).execute()
    finally:
        if manager.has_kernel:manager.shutdown_kernel(now=True)
    for cell in book.cells:
        if 'local-photo-preview' in cell.metadata.get('tags',[]):cell.outputs=[]
    assert not any(output.output_type=='error' for cell in book.cells if cell.cell_type=='code' for output in cell.outputs)
    nbformat.validate(book);nbformat.write(book,path)
    dump(report/'notebook_status.json',dict(executed=True,errors=0,training_started=False,photo_bytes_embedded=False,
        code_cells=sum(cell.cell_type=='code' for cell in book.cells),path=path.relative_to(ROOT).as_posix()))
    print('Notebook10 executed without errors')
