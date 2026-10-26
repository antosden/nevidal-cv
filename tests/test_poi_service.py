"""Контракт HTTP, лимиты загрузки и отказ на неподходящих входах."""
from io import BytesIO
from pathlib import Path
import asyncio
import json
import tempfile
import time
import unittest
import httpx
from fastapi.testclient import TestClient
from PIL import Image
from service.app import Settings,create_app
from src.poi_inference import decode_photo,POIPredictor,BUNDLE_FILES


def photo_bytes(format='PNG',size=(8,8),color='red'):
    content=BytesIO();Image.new('RGB',size,color).save(content,format=format);return content.getvalue()


class FakePredictor:
    encoder_name='synthetic';model_version='test-v1'
    poi_ids=['moscow/d/a','moscow/d/b'];mapping={p:i for i,p in enumerate(poi_ids)}
    def __init__(self,max_pixels=1000,delay=0):self.max_pixels=max_pixels;self.delay=delay;self.calls=0
    def predict_bytes(self,content,top_k):
        self.calls+=1;image=decode_photo(content,self.max_pixels)
        accepted=image.getpixel((0,0))[0]>100;image.close()
        time.sleep(self.delay)
        return {'poi_id':self.poi_ids[0] if accepted else None,'predicted_poi_id':self.poi_ids[0],
            'accepted':accepted,'reason':'poi_verified' if accepted else 'below_score_threshold',
            'score':.9 if accepted else .2,'margin':.3,'score_threshold':.6,'margin_threshold':.1,
            'score_type':'cosine_similarity','score_is_probability':False,
            'candidates':[{'poi_id':p,'score':.9-i*.3} for i,p in enumerate(self.poi_ids[:top_k])],
            'reference':{'image_id':'ref','source':'manual','author':'Shilkova A.M.','licence':'Public Domain'},
            'encoder':self.encoder_name,'model_version':self.model_version}


class ServiceTests(unittest.TestCase):
    def client(self,limit=1024,max_pixels=1000):
        fake=FakePredictor(max_pixels)
        application=create_app(Settings(Path('unused'),max_upload_bytes=limit),lambda:fake)
        return TestClient(application),fake

    def test_upload_returns_label_and_attribution_and_reuses_model(self):
        client,fake=self.client()
        with client:
            self.assertEqual(client.get('/health/ready').json()['poi_count'],2)
            self.assertEqual(len(client.get('/pois').json()['pois']),2)
            result=client.post('/predict?top_k=1',files={'file':('../../photo.exe',photo_bytes(),'application/octet-stream')})
            self.assertEqual(result.status_code,200);body=result.json()
            self.assertEqual(body['poi_id'],'moscow/d/a');self.assertTrue(body['accepted'])
            self.assertFalse(body['score_is_probability']);self.assertEqual(len(body['candidates']),1)
            self.assertEqual(body['reference']['author'],'Shilkova A.M.')
            self.assertNotIn('file_path',body['reference'])
            client.post('/predict',files={'file':('photo.png',photo_bytes())})
            self.assertEqual(fake.calls,2)

    def test_refusal_preserves_suggestion_but_has_null_label(self):
        client,_=self.client()
        with client:
            result=client.post('/predict',files={'file':('photo.png',photo_bytes(color='black'))}).json()
            self.assertIsNone(result['poi_id']);self.assertFalse(result['accepted'])
            self.assertEqual(result['predicted_poi_id'],'moscow/d/a')

    def test_bad_inputs_and_pixel_limit(self):
        client,_=self.client(max_pixels=100)
        with client:
            for content,status in [(b'',422),(b'not an image',422),(photo_bytes('GIF'),415),(photo_bytes(size=(20,20)),413)]:
                result=client.post('/predict',files={'file':('x',content)})
                self.assertEqual(result.status_code,status,result.text)
            self.assertEqual(client.post('/predict').status_code,422)
            self.assertEqual(client.post('/predict?top_k=11',files={'file':('x',photo_bytes())}).status_code,422)

    def test_file_and_total_body_limits(self):
        client,fake=self.client(limit=32)
        with client:
            self.assertEqual(client.post('/predict',files={'file':('x',photo_bytes())}).status_code,413)
            self.assertEqual(client.post('/predict',content=b'x'*70000).status_code,413)
            # Без Content-Length ограничивается и фактически полученное тело.
            result=client.post('/predict',content=iter([b'x'*35000,b'y'*35000]),headers={'content-type':'application/octet-stream'})
            self.assertEqual(result.status_code,413)
            self.assertEqual(fake.calls,0)

    def test_bad_bundle_fails_startup_instead_of_random_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for file in BUNDLE_FILES:(root/file).write_bytes(b'bad')
            (root/'metadata.json').write_text(json.dumps({'schema_version':1,'bundle_type':'poi_encoder_gallery',
                'artifact_sha256':{file:'0'*64 for file in BUNDLE_FILES}}))
            with self.assertRaisesRegex(ValueError,'SHA'):POIPredictor(root)

    def test_exif_orientation_matches_training_decode(self):
        stream=BytesIO();exif=Image.Exif();exif[274]=6
        Image.new('RGB',(8,4),'red').save(stream,format='JPEG',exif=exif)
        image=decode_photo(stream.getvalue(),100)
        self.assertEqual(image.size,(4,8));image.close()


class ConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_busy_request_is_bounded_and_gate_recovers(self):
        fake=FakePredictor(delay=.3)
        application=create_app(Settings(Path('unused'),max_inflight=1),lambda:fake)
        async with application.router.lifespan_context(application):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application),base_url='http://local') as client:
                responses=await asyncio.gather(*[client.post('/predict',files={'file':('x.png',photo_bytes())}) for _ in range(2)])
                self.assertEqual(sorted(r.status_code for r in responses),[200,503])
                self.assertEqual((await client.get('/health/ready')).status_code,200)
                self.assertEqual((await client.post('/predict',files={'file':('x.png',photo_bytes())})).status_code,200)
