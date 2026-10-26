"""Скачать только официальные encoder-веса в проект (без загрузки фотографий)."""
from pathlib import Path
import hashlib
import json
import urllib.request
ROOT=Path(__file__).resolve().parents[1]
destination=ROOT/'.runtime/torch/checkpoints';destination.mkdir(parents=True,exist_ok=True)
urls={
    'mobilenet_v3_large-5c1a4163.pth':'https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth',
    'dinov2_vits14_pretrain.pth':'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth',
}
records=[]
for name,url in urls.items():
    path=destination/name
    if not path.exists():
        temporary=path.with_suffix('.pth.part')
        print('Downloading',url,flush=True)
        try:
            with urllib.request.urlopen(url,timeout=60) as response,temporary.open('wb') as stream:
                size=0
                while block:=response.read(1024*1024):
                    size+=len(block)
                    if size>150*1024*1024:raise ValueError('Превышен ожидаемый размер checkpoint')
                    stream.write(block)
            temporary.replace(path)
        except Exception:
            if temporary.exists():temporary.unlink()
            raise
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if name.startswith('mobilenet') and not digest.startswith('5c1a4163'):raise ValueError('Неверный SHA MobileNetV3-Large')
    records.append({'file':name,'url':url,'sha256':digest})
    print(name,digest,flush=True)
(destination/'download_provenance.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
