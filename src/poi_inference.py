"""Offline inference по самодостаточному выпуску encoder + gallery + policy."""
from io import BytesIO
from pathlib import Path
import json
import threading
import warnings
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
from .frozen_encoders import ENCODERS,load_encoder
from .poi_baseline import configure_torch,sha256_file,l2_normalize,poi_scores,score_margins,acceptance_matrix

BUNDLE_FILES={'weights.pth','gallery.csv','gallery_embeddings.npy','class_mapping.json','rejection_policy.json','provenance.json'}


class ImageInputError(ValueError):
    def __init__(self,code,message,status_code=422):
        super().__init__(message);self.code=code;self.status_code=status_code


def decode_photo(content,max_pixels):
    if not content:raise ImageInputError('empty_image','Файл пуст.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as image:
                if image.format not in {'JPEG','MPO','PNG','WEBP'}:
                    raise ImageInputError('unsupported_format','Поддерживаются JPEG, PNG, WebP и первый кадр MPO.',415)
                if image.width*image.height>max_pixels:
                    raise ImageInputError('image_too_large','Превышен лимит числа пикселей.',413)
                if image.format=='WEBP' and getattr(image,'is_animated',False):
                    raise ImageInputError('animated_image','Анимированный WebP не поддерживается.',415)
                image.seek(0)
                # Тот же EXIF/RGB preprocessing, что в проверенном эксперименте.
                return ImageOps.exif_transpose(image).convert('RGB')
    except (Image.DecompressionBombError,Image.DecompressionBombWarning) as error:
        raise ImageInputError('image_too_large','Превышен безопасный размер изображения.',413) from error
    except ImageInputError:raise
    except (UnidentifiedImageError,OSError,SyntaxError,ValueError,EOFError) as error:
        raise ImageInputError('invalid_image','Файл не является читаемой фотографией.') from error


class POIPredictor:
    def __init__(self,directory,threads=4,max_pixels=40_000_000):
        directory=Path(directory)
        metadata=json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
        if metadata.get('schema_version')!=1 or metadata.get('bundle_type')!='poi_encoder_gallery':
            raise ValueError('Неподдерживаемый inference bundle')
        if set(metadata['artifact_sha256'])!=BUNDLE_FILES:raise ValueError('Неполный состав bundle')
        for name,digest in metadata['artifact_sha256'].items():
            if sha256_file(directory/name)!=digest:raise ValueError(f'Неверный SHA артефакта {name}')
        self.encoder_name=metadata['encoder']
        if self.encoder_name not in ENCODERS:raise ValueError('Неизвестный encoder')
        self.mapping=json.loads((directory/'class_mapping.json').read_text(encoding='utf-8'))
        if not self.mapping or any(type(v) is not int for v in self.mapping.values()) or len(set(self.mapping.values()))!=len(self.mapping):
            raise ValueError('Неверный class_mapping')
        self.poi_ids=sorted(self.mapping,key=self.mapping.get)
        self.gallery=pd.read_csv(directory/'gallery.csv',keep_default_na=False)
        required={'image_id','poi_id','source','author','licence','class_index'}
        if not required<=set(self.gallery) or not self.gallery.image_id.is_unique:raise ValueError('Неверная gallery.csv')
        if set(self.gallery.poi_id)!=set(self.poi_ids):raise ValueError('Галерея не покрывает mapping')
        if not self.gallery.class_index.eq(self.gallery.poi_id.map(self.mapping)).all():raise ValueError('Индексы классов не совпадают')
        self.vectors=np.load(directory/'gallery_embeddings.npy',allow_pickle=False)
        if self.vectors.shape!=(len(self.gallery),ENCODERS[self.encoder_name]['dimension']):raise ValueError('Размеры галереи не совпадают')
        if not np.isfinite(self.vectors).all() or not np.allclose(np.linalg.norm(self.vectors,axis=1),1,atol=1e-4):
            raise ValueError('Некорректные embeddings галереи')
        self.policy=json.loads((directory/'rejection_policy.json').read_text(encoding='utf-8'))
        if self.policy.get('threshold_type','global')!='global':raise ValueError('Нужна глобальная policy')
        if any(not isinstance(self.policy.get(k),(int,float)) or not np.isfinite(self.policy[k]) for k in ['score_threshold','margin_threshold']):
            raise ValueError('Неверные пороги')
        if not -1<=self.policy['score_threshold']<=1 or not 0<=self.policy['margin_threshold']<=2:
            raise ValueError('Пороги вне диапазонов cosine/margin')
        self.torch=configure_torch(42,threads)
        self.model,self.transform=load_encoder(self.encoder_name,directory/'weights.pth')
        self.model_version=metadata['model_version'];self.max_pixels=max_pixels
        self.lock=threading.Lock()

    def predict_bytes(self,content,top_k=3):
        if not 1<=top_k<=10:raise ValueError('top_k должен быть в [1,10]')
        # Один inference одновременно ограничивает CPU/RAM и сохраняет стабильный eval.
        with self.lock:
            image=decode_photo(content,self.max_pixels)
            try:
                tensor=self.transform(image).unsqueeze(0)
            finally:image.close()
            with self.torch.inference_mode():embedding=l2_normalize(self.model(tensor).numpy())
            scores,neighbours=poi_scores(embedding,self.vectors,self.gallery.poi_id,self.poi_ids)
            order=np.argsort(-scores[0],kind='stable');best=int(order[0])
            score=float(scores[0,best]);margin=float(score_margins(scores)[0,best])
            accepted=bool(acceptance_matrix(scores,self.policy)[0,best])
            reference=self.gallery.iloc[int(neighbours[0,best])]
            reason='poi_verified' if accepted else 'tie' if margin<=0 else 'below_score_threshold' if score<self.policy['score_threshold'] else 'ambiguous_poi'
            return {'poi_id':self.poi_ids[best] if accepted else None,'predicted_poi_id':self.poi_ids[best],
                'accepted':accepted,'reason':reason,'score':score,'margin':margin,
                'score_threshold':self.policy['score_threshold'],'margin_threshold':self.policy['margin_threshold'],
                'score_type':'cosine_similarity','score_is_probability':False,
                'candidates':[{'poi_id':self.poi_ids[int(i)],'score':float(scores[0,i])} for i in order[:top_k]],
                'reference':{key:str(reference.get(key,'')) for key in ['image_id','source','author','licence','source_url','license_url']},
                'encoder':self.encoder_name,'model_version':self.model_version}
