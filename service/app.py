"""FastAPI: загрузка фото → POI с откалиброванным отказом."""
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
import asyncio
import os
from fastapi import FastAPI,File,UploadFile,Query,HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse,HTMLResponse
from src.poi_inference import POIPredictor,ImageInputError


@dataclass(frozen=True)
class Settings:
    model_dir:Path
    max_upload_bytes:int=10*1024*1024
    max_pixels:int=40_000_000
    cpu_threads:int=4
    max_inflight:int=2

    @classmethod
    def from_env(cls):
        value=cls(Path(os.getenv('MODEL_DIR',str(Path(__file__).resolve().parents[1]/'models/poi_service_dinov2'))),
            int(os.getenv('MAX_UPLOAD_MB','10'))*1024*1024,int(os.getenv('MAX_IMAGE_PIXELS','40000000')),
            int(os.getenv('CPU_THREADS','4')),int(os.getenv('MAX_INFLIGHT_REQUESTS','2')))
        if min(value.max_upload_bytes,value.max_pixels,value.cpu_threads,value.max_inflight)<=0:
            raise ValueError('Лимиты должны быть положительными')
        return value


class Candidate(BaseModel):
    poi_id:str
    score:float


class Reference(BaseModel):
    image_id:str
    source:str
    author:str
    licence:str
    source_url:str=''
    license_url:str=''


class Prediction(BaseModel):
    poi_id:str|None
    predicted_poi_id:str
    accepted:bool
    reason:str
    score:float
    margin:float
    score_threshold:float
    margin_threshold:float
    score_type:str
    score_is_probability:bool
    candidates:list[Candidate]
    reference:Reference
    encoder:str
    model_version:str


class BoundedUpload:
    """Лимит всего HTTP-body до multipart parsing, включая chunked-запросы."""
    def __init__(self,app,max_bytes,max_inflight):
        self.app=app;self.max_bytes=max_bytes;self.gate=asyncio.Semaphore(max_inflight)

    async def __call__(self,scope,receive,send):
        if scope['type']!='http' or scope['method']!='POST':return await self.app(scope,receive,send)
        async def error(status,code):
            await JSONResponse({'detail':{'code':code}},status_code=status)(scope,receive,send)
        headers=dict(scope.get('headers',[]))
        try:length=int(headers.get(b'content-length',b'0'))
        except ValueError:return await error(400,'invalid_content_length')
        if length<0:return await error(400,'invalid_content_length')
        if length>self.max_bytes:return await error(413,'request_too_large')
        try:await asyncio.wait_for(self.gate.acquire(),timeout=.1)
        except TimeoutError:return await error(503,'service_busy')
        try:
            body=bytearray()
            while True:
                message=await receive()
                if message['type']=='http.disconnect':return
                chunk=message.get('body',b'')
                if len(body)+len(chunk)>self.max_bytes:return await error(413,'request_too_large')
                body.extend(chunk)
                if not message.get('more_body',False):break
            delivered=False
            async def replay():
                nonlocal delivered
                if not delivered:
                    delivered=True
                    return {'type':'http.request','body':bytes(body),'more_body':False}
                return await receive()
            await self.app(scope,replay,send)
        finally:self.gate.release()


def create_app(settings=None,predictor_factory=None):
    settings=settings or Settings.from_env()
    factory=predictor_factory or (lambda:POIPredictor(settings.model_dir,settings.cpu_threads,settings.max_pixels))
    @asynccontextmanager
    async def lifespan(app):
        # Ошибка bundle останавливает startup; random fallback и загрузок сети нет.
        app.state.predictor=await run_in_threadpool(factory)
        try:yield
        finally:app.state.predictor=None
    application=FastAPI(title='Невидаль: POI recognition',version='1.0.0',lifespan=lifespan,
        description='[Открыть форму загрузки фотографии](/).\n\nПредсказание конкретного POI. poi_id=null при отказе; predicted_poi_id содержит предположение. Score — cosine similarity, не вероятность.')
    application.add_middleware(BoundedUpload,max_bytes=settings.max_upload_bytes+64*1024,max_inflight=settings.max_inflight)

    @application.get('/',response_class=HTMLResponse,include_in_schema=False)
    async def upload_page():
        page=(Path(__file__).with_name('index.html')).read_text(encoding='utf-8')
        # Числовой лимит берётся из настроек того же API.
        return HTMLResponse(page.replace('__MAX_UPLOAD_BYTES__',str(settings.max_upload_bytes)),
            headers={'Cache-Control':'no-store','X-Content-Type-Options':'nosniff','Referrer-Policy':'no-referrer'})

    @application.get('/health/live')
    async def live():return {'status':'alive'}

    @application.get('/health')
    @application.get('/health/ready')
    async def ready():
        predictor=getattr(application.state,'predictor',None)
        if predictor is None:raise HTTPException(503,detail='model_not_ready')
        return {'status':'ready','encoder':predictor.encoder_name,'model_version':predictor.model_version,'poi_count':len(predictor.poi_ids)}

    @application.get('/pois')
    async def pois():
        predictor=application.state.predictor
        return {'model_version':predictor.model_version,'pois':[{'poi_id':p,'class_index':predictor.mapping[p]} for p in predictor.poi_ids]}

    @application.post('/predict',response_model=Prediction)
    async def predict(file:UploadFile=File(...),top_k:int=Query(3,ge=1,le=10)):
        try:
            content=await file.read(settings.max_upload_bytes+1)
            if len(content)>settings.max_upload_bytes:raise HTTPException(413,detail={'code':'file_too_large'})
            try:
                return await run_in_threadpool(application.state.predictor.predict_bytes,content,top_k)
            except ImageInputError as error:
                raise HTTPException(error.status_code,detail={'code':error.code,'message':str(error)}) from error
        finally:await file.close()
    return application


app=create_app()
