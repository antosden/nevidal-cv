"""Проверить реальный HTTP inference и совпадение с frozen экспериментом."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--extra-site',type=Path)
args=parser.parse_args()
if args.extra_site:sys.path.append(str(args.extra_site.resolve()))
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from src.poi_baseline import json_write,acceptance_matrix
from service.app import create_app,Settings
experiment=ROOT/'models/encoder_comparison_manual_all';model=experiment/'dinov2_small'
selection=json.loads((model/'selection.json').read_text());policy=json.loads((model/selection['variant']/'policy.json').read_text())
mapping=json.loads((experiment/'class_mapping.json').read_text());pois=sorted(mapping,key=mapping.get)
checked=[]
with TestClient(create_app(Settings(ROOT/'models/poi_service_dinov2'))) as client:
    health=client.get('/health/ready');assert health.status_code==200
    assert len(client.get('/pois').json()['pois'])==23
    for split in ['test','negative_test']:
        frame=pd.read_csv(experiment/f'splits/{split}.csv',keep_default_na=False)
        scores=np.load(model/f'{split}_scores.npy',allow_pickle=False)
        index=next(i for i,row in enumerate(frame.itertuples()) if (ROOT/row.file_path).stat().st_size<10*1024*1024 and row.width*row.height<=40_000_000)
        row=frame.iloc[index]
        response=client.post('/predict?top_k=3',files={'file':('photo.jpg',(ROOT/row.file_path).read_bytes(),'image/jpeg')})
        assert response.status_code==200,response.text
        result=response.json();best=int(scores[index].argmax())
        assert result['predicted_poi_id']==pois[best]
        error=abs(result['score']-float(scores[index,best]));assert error<1e-5,error
        assert result['accepted']==bool(acceptance_matrix(scores,policy)[index,best])
        assert result['poi_id']==(pois[best] if result['accepted'] else None)
        checked.append({'split':split,'image_id':row.image_id,'predicted_poi_id':result['predicted_poi_id'],
            'accepted':result['accepted'],'maximum_score_difference':error})
    assert client.post('/predict',files={'file':('bad.jpg',b'bad')}).status_code==422
report={'status':'passed','model_version':health.json()['model_version'],'http_checks':checked,'docker_container_tested':False}
json_write(ROOT/'reports/poi_service/smoke_checks.json',report)
print(json.dumps(report,ensure_ascii=False,indent=2))
