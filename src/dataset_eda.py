"""EDA Wikimedia-изображений: метаданные, проверка разметки и групповые splits."""
from pathlib import Path, PurePosixPath
import hashlib
import html
from html.parser import HTMLParser
import json
import math
import shutil
import textwrap
import warnings
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont
from tqdm.auto import tqdm

SEED = 42
IMAGE_SUFFIXES = {'.jpg','.jpeg','.png','.webp','.tif','.tiff','.bmp','.gif'}
STATUSES = {'verified','needs_review','uncertain','rejected'}
REVIEW_COLUMNS = ['image_id','file_path','sha256','source','author','licence','poi_id','label_status','rejection_reason',
                  'series_id','reviewer','review_note']
PAIR_COLUMNS = ['image_id_a','image_id_b','kind','phash_distance','crop_distance','cross_poi']


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def prepare_paths(root):
    root=Path(root).resolve()
    for folder in ['data/annotations','data/processed','reports/figures']:
        (root/folder).mkdir(parents=True,exist_ok=True)
    return root


class PlainText(HTMLParser):
    def __init__(self):super().__init__();self.parts=[]
    def handle_data(self,data):self.parts.append(data)


def plain_author(value):
    parser=PlainText();parser.feed(str(value))
    return ' '.join(html.unescape(' '.join(parser.parts)).split())


def read_rgb(path):
    with warnings.catch_warnings():
        warnings.simplefilter('error',Image.DecompressionBombWarning)
        with Image.open(path) as image:
            image.load()
            return ImageOps.exif_transpose(image).convert('RGB')


def perceptual_hash(image):
    # DCT 32x32, 8x8 низких частот. DC исключён: используются 63 AC-бита.
    gray=np.asarray(image.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=np.float64)
    low=cv2.dct(gray)[:8,:8].ravel()[1:]
    bits=low>np.median(low)
    value=0
    for bit in bits:value=(value<<1)|int(bit)
    return f'{value:016x}'


def fingerprints(image):
    result=[perceptual_hash(image)]
    # Для кроп-хэшей ограничиваем рабочую копию: не создаём десять больших растров Commons.
    image=image.copy();image.thumbnail((1024,1024),Image.Resampling.LANCZOS)
    w,h=image.size
    for fraction in [.8,.6]:
        cw=max(1,int(w*fraction));ch=max(1,int(h*fraction))
        for x,y in [(0,0),(w-cw,0),(0,h-ch),(w-cw,h-ch),((w-cw)//2,(h-ch)//2)]:
            result.append(perceptual_hash(image.crop((x,y,x+cw,y+ch))))
    return result


def technical_metrics(image):
    sample=image.copy();sample.thumbnail((1024,1024),Image.Resampling.LANCZOS)
    gray=cv2.cvtColor(np.asarray(sample),cv2.COLOR_RGB2GRAY)
    lap=cv2.Laplacian(gray,cv2.CV_64F,ksize=1)[1:-1,1:-1]
    return {'brightness':float(gray.mean()),'sharpness':float(lap.var()) if lap.size else np.nan,
            'dark_fraction':float((gray<10).mean()),'bright_fraction':float((gray>245).mean())}


def metadata_index(root):
    path=root/'data/raw/wikimedia/metadata.csv'
    metadata=pd.read_csv(path,dtype=str,keep_default_na=False)
    required={'city','district','poi','file','source_url','artist','license','license_url','commons_title'}
    if not required<=set(metadata.columns):raise ValueError('В metadata.csv отсутствуют обязательные поля')
    metadata['filename']=metadata.file.map(lambda value:PurePosixPath(value.replace('\\','/')).name)
    index={}
    for key,group in metadata.groupby(['city','district','poi','filename'],sort=False):
        index[key]=group.to_dict('records')
    return metadata,index


def update_class_mapping(root,poi_ids):
    path=root/'data/annotations/class_mapping.json'
    mapping=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    if any(not isinstance(v,int) or isinstance(v,bool) or v<0 for v in mapping.values()):
        raise ValueError('Некорректные индексы классов')
    if len(set(mapping.values()))!=len(mapping):raise ValueError('Повтор class_index')
    next_index=max(mapping.values(),default=-1)+1
    for poi in sorted(set(poi_ids)-set(mapping)):
        mapping[poi]=next_index;next_index+=1
    path.write_text(json.dumps(mapping,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return mapping


def selected_present(frame):
    mask=pd.Series(True,index=frame.index)
    for col in ['file_exists','selected_for_dataset']:
        if col in frame:mask &= frame[col].astype(bool)
    return mask


def discover_images(root):
    """Wikimedia и manual согласно воспроизводимой политике отбора."""
    root=Path(root);raw=root/'data/raw/wikimedia'
    config=pd.read_csv(root/'config/wikimedia_pois.csv',dtype=str,keep_default_na=False)
    registered={f'{r.city}/{r.district}/{r.poi}':r for r in config.itertuples()}
    descriptors=[];counts={}
    for path in sorted(p for p in raw.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
        if not path.resolve().is_relative_to(raw.resolve()):raise ValueError('Изображение вне Wikimedia')
        rel=path.relative_to(raw)
        if len(rel.parts)!=4:raise ValueError(f'Ожидается city/district/poi/file: {rel}')
        city,district,slug,_=rel.parts;poi=f'{city}/{district}/{slug}'
        counts[poi]=counts.get(poi,0)+1
        descriptors.append(dict(path=path,city=city,district=district,slug=slug,source_kind='wikimedia'))
    plans=[];manual_root=root/'data/raw/manual';manual_config=root/'config/manual_pois.csv'
    policy_path=root/'config/dataset_sources.json'
    manual_policy=json.loads(policy_path.read_text(encoding='utf-8')).get('manual_selection','below_30') if policy_path.exists() else 'below_30'
    if manual_policy not in {'all','below_30'}:raise ValueError('Неверная manual_selection')
    available_pois=set()
    if manual_root.is_dir() and manual_config.exists():
        manual=pd.read_csv(manual_config,dtype=str,keep_default_na=False)
        if not {'folder','city','district','poi','author','licence'}<=set(manual):
            raise ValueError('Неверные колонки manual_pois.csv')
        if not manual.folder.is_unique:raise ValueError('Повтор папки manual')
        unknown={p.name for p in manual_root.iterdir() if p.is_dir()}-set(manual.folder)
        if unknown:raise ValueError(f'Нет mapping для manual-папок: {sorted(unknown)}')
        for row in manual.itertuples():
            folder=manual_root/row.folder
            if not folder.resolve().is_relative_to(manual_root.resolve()):raise ValueError('Неверная manual-папка')
            parts=[row.city,row.district,row.poi]
            if any(not p or '/' in p or '\\' in p or p in {'.','..'} for p in parts):raise ValueError('Неверный manual POI')
            poi='/'.join(parts);available_pois.add(poi)
            files=sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
            new=poi not in registered and poi not in counts
            include=manual_policy=='all' or new or counts.get(poi,0)<30
            reason='all_manual' if manual_policy=='all' else 'new_poi' if new else 'wikimedia_below_30' if include else 'wikimedia_at_least_30'
            plans.append(dict(poi_id=poi,manual_folder=row.folder,wikimedia_images=counts.get(poi,0),
                              available_manual=len(files),selected_manual=len(files) if include else 0,selection_reason=reason))
            if include:
                for path in files:
                    if not path.resolve().is_relative_to(manual_root.resolve()):raise ValueError('Manual-ссылка вне raw')
                    descriptors.append(dict(path=path,city=row.city,district=row.district,slug=row.poi,
                                            source_kind='manual',author='Shilkova A.M.',licence=row.licence or 'unknown'))
    for poi in sorted(set(registered)|set(counts)):
        if counts.get(poi,0)<30 and poi not in available_pois:
            plans.append(dict(poi_id=poi,manual_folder='',wikimedia_images=counts.get(poi,0),available_manual=0,
                              selected_manual=0,selection_reason='no_manual_folder'))
    return descriptors,registered,pd.DataFrame(plans)


def inventory(root):
    root=prepare_paths(root)
    raw=root/'data/raw/wikimedia'
    if not raw.is_dir():raise FileNotFoundError(raw)
    metadata,index=metadata_index(root)
    descriptors,config_by_id,enrichment=discover_images(root)
    enrichment.to_csv(root/'reports/manual_enrichment_plan.csv',index=False)
    previous_path=root/'data/annotations/initial_manifest.csv'
    previous=pd.read_csv(previous_path,keep_default_na=False) if previous_path.exists() else pd.DataFrame()
    rows=[];hashes={}
    thumb_dir=root/'reports/figures/thumbnails';thumb_dir.mkdir(parents=True,exist_ok=True)
    for descriptor in tqdm(descriptors,desc='Wikimedia + manual: инвентаризация'):
        path=descriptor['path'];city=descriptor['city'];district=descriptor['district'];slug=descriptor['slug']
        filename=path.name;is_manual=descriptor['source_kind']=='manual'
        poi=f'{city}/{district}/{slug}'
        file_path=path.relative_to(root).as_posix()
        image_id=('manual_' if is_manual else 'wm_')+hashlib.sha256(file_path.encode('utf-8')).hexdigest()[:16]
        matches=[] if is_manual else index.get((city,district,slug,filename),[])
        meta=matches[0] if matches else {}
        url=meta.get('source_url','')
        host=(urlparse(url).hostname or '').lower()
        source='manual' if is_manual else 'wikimedia_commons' if matches and (host=='wikimedia.org' or host.endswith('.wikimedia.org')) else 'unknown'
        author_raw=descriptor['author'] if is_manual else meta.get('artist','')
        licence=descriptor['licence'] if is_manual else meta.get('license','')
        row=dict(image_id=image_id,file_path=file_path,city=city,district=district,poi_slug=slug,
                 initial_poi_id=poi,poi_id=poi,label_status='needs_review',source=source,
                 source_url=url,commons_title=meta.get('commons_title',''),
                 commons_file_id=path.stem if matches and path.stem.isdigit() else '',
                 category=meta.get('category',''),author=plain_author(author_raw) or 'unknown',author_raw=author_raw,
                 licence=licence,license=licence,license_url=meta.get('license_url',''),
                 file_exists=True,selected_for_dataset=True,
                 metadata_original_file=meta.get('file',''),metadata_match_count=len(matches),
                 width=np.nan,height=np.nan,aspect_ratio=np.nan,file_size=path.stat().st_size,
                 sha256=digest(path),perceptual_hash='',readable=False,rejection_reason='',
                 duplicate_group='',split='',series_id='',reviewer='',review_note='',
                 brightness=np.nan,sharpness=np.nan,dark_fraction=np.nan,bright_fraction=np.nan,
                 format=path.suffix.lower(),quality_flags='',capture_datetime='',camera_model='',read_error='')
        flags=[]
        if not matches and not is_manual:flags.append('missing_metadata')
        if len(matches)>1:flags.append('multiple_metadata_rows')
        if not author_raw:flags.append('missing_author')
        if not licence or licence=='unknown':flags.append('missing_licence')
        if poi not in config_by_id and not is_manual:flags.append('folder_not_in_config')
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error',Image.DecompressionBombWarning)
                with Image.open(path) as original:
                    row['format']=original.format
                    exif=original.getexif()
                    row['capture_datetime']=str(exif.get(36867,exif.get(306,'')))
                    row['camera_model']=str(exif.get(272,''))
            image=read_rgb(path);w,h=image.size
            row.update(width=w,height=h,aspect_ratio=w/h,readable=True,**technical_metrics(image))
            hashes[image_id]=fingerprints(image)
            row['perceptual_hash']=hashes[image_id][0]
            thumbnail=image.copy();thumbnail.thumbnail((512,512),Image.Resampling.LANCZOS)
            thumbnail.save(thumb_dir/f'{image_id}.jpg',quality=85)
            if min(w,h)<224:flags.append('small_side_lt224')
            if row['brightness']<40:flags.append('dark_mean_lt40')
            if row['brightness']>215:flags.append('bright_mean_gt215')
            if w/h<.5 or w/h>2:flags.append('unusual_aspect')
        except (OSError,ValueError,SyntaxError,Image.DecompressionBombError,Image.DecompressionBombWarning) as exc:
            row.update(label_status='rejected',rejection_reason='unreadable',read_error=str(exc))
        row['quality_flags']=';'.join(flags);rows.append(row)
    current_ids={r['image_id'] for r in rows}
    # Сохраняем исторические строки удалённых пользователем файлов для аудита.
    if not previous.empty:
        for old in previous[~previous.image_id.isin(current_ids)].to_dict('records'):
            exists=(root/old['file_path']).is_file()
            old.update(file_exists=exists,selected_for_dataset=False,
                       label_status='rejected',rejection_reason='not_selected_by_enrichment_rule' if exists else 'removed_by_user')
            if not exists:old['readable']=False
            rows.append(old)
    if not rows:raise ValueError('Не найдены изображения Wikimedia')
    frame=pd.DataFrame(rows)
    if not frame.image_id.is_unique:raise ValueError('Коллизия image_id')
    mapping=update_class_mapping(root,set(config_by_id)|set(frame.poi_id))
    catalog=[]
    for poi in sorted(set(config_by_id)|set(frame.poi_id)):
        city,district,slug=poi.split('/')
        catalog.append(dict(poi_id=poi,city=city,district=district,poi_slug=slug,
                            class_index=mapping[poi],image_count=int((frame.poi_id.eq(poi)&selected_present(frame)).sum()),
                            category=getattr(config_by_id.get(poi),'category','')))
    pd.DataFrame(catalog).to_csv(root/'data/annotations/poi_catalog.csv',index=False)
    frame.to_csv(root/'data/annotations/initial_manifest.csv',index=False)
    # Фиксируем raw-файлы и таблицы происхождения для последующей проверки неизменности.
    snapshot={r.file_path:r.sha256 for r in frame[selected_present(frame)].itertuples()}
    metadata_paths=[raw/'metadata.csv',root/'config/wikimedia_pois.csv',root/'config/manual_pois.csv']
    for path in [p for p in metadata_paths if p.exists()]:
        snapshot[path.relative_to(root).as_posix()]=digest(path)
    (root/'reports/raw_integrity_snapshot.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding='utf-8')
    (root/'reports/image_fingerprints.json').write_text(json.dumps(hashes,indent=1),encoding='utf-8')
    return frame,hashes,mapping


def load_inventory(root):
    frame=pd.read_csv(root/'data/annotations/initial_manifest.csv',keep_default_na=False,
                      dtype={'commons_file_id':str,'capture_datetime':str,'camera_model':str})
    for column in ['width','height','aspect_ratio','file_size','brightness','sharpness','dark_fraction','bright_fraction']:
        frame[column]=pd.to_numeric(frame[column],errors='coerce')
    frame['author']=frame.author.replace('','unknown')
    hashes=json.loads((root/'reports/image_fingerprints.json').read_text(encoding='utf-8'))
    mapping=json.loads((root/'data/annotations/class_mapping.json').read_text(encoding='utf-8'))
    return frame,hashes,mapping


class UnionFind:
    def __init__(self,ids):self.parents={i:i for i in ids}
    def find(self,i):
        while self.parents[i]!=i:
            self.parents[i]=self.parents[self.parents[i]];i=self.parents[i]
        return i
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        self.parents[max(a,b)]=min(a,b)


def find_duplicates(frame,hashes,root,phash_threshold=6,crop_threshold=4):
    uf=UnionFind(frame.image_id);pairs=[]
    records=list(frame.to_dict('records'))
    for idx,a in enumerate(tqdm(records,desc='Точные и потенциальные дубликаты')):
        for b in records[idx+1:]:
            kind='';distance=None;crop_distance=None
            if a['sha256']==b['sha256']:kind='sha256_exact'
            elif a['commons_file_id'] and a['commons_file_id']==b['commons_file_id']:kind='commons_same_file'
            elif a['readable'] and b['readable'] and a['image_id'] in hashes and b['image_id'] in hashes:
                ah,bh=hashes[a['image_id']],hashes[b['image_id']]
                distance=(int(ah[0],16)^int(bh[0],16)).bit_count()
                if distance<=phash_threshold:kind='phash_candidate'
                else:
                    crop_distance=min([(int(ah[0],16)^int(x,16)).bit_count() for x in bh[1:]]+
                                      [(int(bh[0],16)^int(x,16)).bit_count() for x in ah[1:]])
                    if crop_distance<=crop_threshold:kind='crop_candidate'
            if kind:
                uf.union(a['image_id'],b['image_id'])
                pairs.append(dict(image_id_a=a['image_id'],image_id_b=b['image_id'],kind=kind,
                                  phash_distance=distance,crop_distance=crop_distance,cross_poi=a['poi_id']!=b['poi_id']))
    result=frame.copy()
    result['duplicate_group']=['dup_'+uf.find(i) for i in result.image_id]
    pair_frame=pd.DataFrame(pairs,columns=PAIR_COLUMNS)
    pair_frame.to_csv(root/'data/annotations/duplicate_pairs.csv',index=False)
    grouped=result.groupby('duplicate_group').agg(count=('image_id','size'),
        image_ids=('image_id',lambda x:';'.join(x)),poi_ids=('poi_id',lambda x:';'.join(sorted(set(x)))))
    grouped[grouped['count']>1].to_csv(root/'data/annotations/duplicate_groups.csv')
    result.to_csv(root/'data/annotations/initial_manifest.csv',index=False)
    return result,pair_frame


def apply_reviews(frame,root,mapping):
    path=root/'data/annotations/review_edits.csv'
    if not path.exists():frame[REVIEW_COLUMNS].to_csv(path,index=False,encoding='utf-8-sig')
    edits=pd.read_csv(path,dtype=str,keep_default_na=False)
    if not set(REVIEW_COLUMNS)<=set(edits.columns):raise ValueError('Отсутствуют колонки review_edits.csv')
    if not edits.image_id.is_unique or set(edits.image_id)!=set(frame.image_id):
        raise ValueError('Состав файлов изменился: согласуйте review_edits.csv с initial_manifest.csv')
    if not set(edits.label_status)<=STATUSES:raise ValueError('Некорректный label_status')
    if not set(edits.poi_id)<=set(mapping):raise ValueError('Неизвестный poi_id')
    verified=edits.label_status.eq('verified')
    if not edits.loc[verified,'reviewer'].str.strip().ne('').all():raise ValueError('verified требует reviewer')
    if not edits.loc[edits.label_status.eq('rejected'),'rejection_reason'].str.strip().ne('').all():
        raise ValueError('rejected требует rejection_reason')
    if edits.loc[verified,'rejection_reason'].str.strip().ne('').any():raise ValueError('У verified не должно быть rejection_reason')
    indexed=edits.set_index('image_id')
    current=frame.set_index('image_id')
    for col in ['file_path','sha256']:
        if not current[col].sort_index().equals(indexed[col].sort_index()):
            raise ValueError(f'{col} не совпадает: источник изменился или CSV испорчен')
    result=frame.copy()
    for col in ['poi_id','label_status','rejection_reason','series_id','reviewer','review_note']:
        result[col]=result.image_id.map(indexed[col])
    unreadable=~result.readable & selected_present(result)
    result.loc[unreadable,['label_status','rejection_reason']]=['rejected','unreadable']
    excluded=~selected_present(result)
    result.loc[excluded,'label_status']='rejected'
    result.loc[excluded,'rejection_reason']=frame.loc[excluded,'rejection_reason']
    result[REVIEW_COLUMNS+['source_url','license_url','author_raw']].to_csv(root/'data/annotations/annotations.csv',index=False)
    return result


def reconcile_reviews(frame,root,confirm_wikimedia=False,confirm_manual=False):
    """Явное согласование CSV после пользовательского удаления/пополнения источников."""
    path=root/'data/annotations/review_edits.csv'
    old=pd.read_csv(path,dtype=str,keep_default_na=False).set_index('image_id') if path.exists() else pd.DataFrame()
    output=frame[REVIEW_COLUMNS].copy()
    editable=['poi_id','label_status','rejection_reason','series_id','reviewer','review_note']
    for idx,row in output.iterrows():
        if row.image_id in old.index:
            if str(old.loc[row.image_id,'sha256'])!=row.sha256:
                raise ValueError(f'Исходник изменён: {row.image_id}; нужна повторная проверка')
            for col in editable:output.at[idx,col]=old.loc[row.image_id,col]
    present=selected_present(frame)
    output.loc[~present,'label_status']='rejected'
    output.loc[~present,'rejection_reason']=frame.loc[~present,'rejection_reason']
    output.loc[~present,'reviewer']='user'
    output.loc[~present,'review_note']='Удалён пользователем как нерелевантный или исключён правилом отбора источников.'
    for source,confirmed in [('wikimedia_commons',confirm_wikimedia),('manual',confirm_manual)]:
        mask=present & output.source.eq(source) & output.label_status.isin(['needs_review','uncertain'])
        if confirmed:
            output.loc[mask,['label_status','rejection_reason','reviewer']]=['verified','','user']
            output.loc[mask,'review_note']='Проверка подтверждена пользователем для этого источника.'
    output.to_csv(path,index=False,encoding='utf-8-sig')
    return output


def split_components(frame):
    uf=UnionFind(frame.image_id)
    for _,group in frame.groupby('duplicate_group'):
        for image_id in group.image_id:uf.union(group.image_id.iloc[0],image_id)
    for _,group in frame[frame.series_id.ne('')].groupby('series_id'):
        for image_id in group.image_id:uf.union(group.image_id.iloc[0],image_id)
    return ['grp_'+uf.find(i) for i in frame.image_id]


def allocate_groups(active,classes,seed=SEED,trials=200):
    groups=pd.crosstab(active.split_group,active.poi_id).reindex(columns=classes,fill_value=0)
    values=groups.to_numpy()
    if not len(values):return None
    target=np.array([.70,.15,.15])[:,None]*values.sum(axis=0)
    rng=np.random.default_rng(seed);best=None;best_score=float('inf')
    for _ in range(trials):
        totals=np.zeros_like(target);assignment=np.zeros(len(values),dtype=int)
        for idx in rng.permutation(len(values)):
            costs=[]
            for split in range(3):
                candidate=totals.copy();candidate[split]+=values[idx]
                costs.append(np.sum((candidate-target)**2/np.maximum(target,1)))
            split=int(np.argmin(costs));assignment[idx]=split;totals[split]+=values[idx]
        score=float(np.sum((totals-target)**2/np.maximum(target,1)))
        if (totals>0).all() and score<best_score:best=assignment.copy();best_score=score
    if best is None:return None
    return dict(zip(groups.index,np.array(['train','val','test'])[best]))


def build_dataset(frame,root,mapping):
    frame=frame.copy();frame['split']='';frame['processed_path']='';frame['active']=False
    frame['exclusion_reason']='';frame['duplicate_of']=''
    frame['class_index']=frame.poi_id.map(mapping).astype('Int64')
    frame['split_group']=split_components(frame)
    eligible=frame.readable & frame.label_status.eq('verified') & selected_present(frame)
    quarantine_path=root/'data/annotations/external_overlap_quarantine.csv'
    quarantine_ids=set(pd.read_csv(quarantine_path,keep_default_na=False).image_id) if quarantine_path.exists() else set()
    quarantine_mask=frame.image_id.isin(quarantine_ids)
    eligible.loc[quarantine_mask]=False
    frame.loc[~eligible,'exclusion_reason']='label_'+frame.loc[~eligible,'label_status']
    frame.loc[~frame.readable,'exclusion_reason']='unreadable'
    excluded=~selected_present(frame)
    if excluded.any():frame.loc[excluded,'exclusion_reason']=frame.loc[excluded,'rejection_reason']
    frame.loc[quarantine_mask,'exclusion_reason']='external_negative_overlap_candidate'
    for _,group in frame[eligible].groupby('sha256'):
        if group.poi_id.nunique()>1:
            eligible.loc[group.index]=False
            frame.loc[group.index,'exclusion_reason']='exact_label_conflict'
        else:
            ordered=group.assign(pixels=group.width*group.height).sort_values(['pixels','image_id'],ascending=[False,True])
            extras=ordered.index[1:]
            eligible.loc[extras]=False
            frame.loc[extras,'exclusion_reason']='exact_duplicate'
            frame.loc[extras,'duplicate_of']=ordered.image_id.iloc[0]
    frame.loc[eligible,'active']=True
    classes=sorted(mapping,key=mapping.get)
    preliminary=frame[frame.readable & selected_present(frame)].groupby('initial_poi_id').split_group.nunique().reindex(classes,fill_value=0)
    verified=frame[eligible].groupby('poi_id').split_group.nunique().reindex(classes,fill_value=0)
    feasibility=pd.DataFrame({'poi_id':classes,'preliminary_independent_groups':preliminary.to_numpy(),
                              'verified_independent_groups':verified.to_numpy(),
                              'has_three_verified_groups':verified.ge(3).to_numpy()})
    feasibility.to_csv(root/'data/annotations/split_feasibility.csv',index=False)
    anchors_path=root/'data/annotations/split_anchors.csv'
    if anchors_path.exists():
        anchors=pd.read_csv(anchors_path,keep_default_na=False)
        anchors=anchors[anchors.split.ne('')].copy()
        if not anchors.image_id.is_unique or not anchors.split.isin(['train','val','test']).all():raise ValueError('Неверные split anchors')
        linked=frame[['image_id','sha256','split_group']].merge(anchors[['image_id','sha256','split']],on='image_id',suffixes=('','_anchor'))
        if not linked.sha256.eq(linked.sha256_anchor).all():raise ValueError('Изменён исходник split anchor')
        conflicts=set(linked.groupby('split_group').split.nunique().loc[lambda s:s>1].index)
        conflict_mask=frame.split_group.isin(conflicts)&eligible
        frame.loc[conflict_mask,['active','exclusion_reason']]=[False,'cross_split_anchor_conflict']
        eligible.loc[conflict_mask]=False
        locked=linked[~linked.split_group.isin(conflicts)].drop_duplicates('split_group').set_index('split_group').split.to_dict()
        active=frame[eligible]
        groups=pd.crosstab(active.split_group,active.poi_id).reindex(columns=classes,fill_value=0)
        names=['train','val','test'];totals=np.zeros((3,len(classes)));allocation={}
        for group,split in locked.items():
            if group in groups.index:allocation[group]=split;totals[names.index(split)]+=groups.loc[group].to_numpy()
        target=np.array([.70,.15,.15])[:,None]*groups.sum().to_numpy()
        rng=np.random.default_rng(SEED)
        for group in rng.permutation(sorted(set(groups.index)-set(allocation))):
            values=groups.loc[group].to_numpy();costs=[]
            for slot in range(3):
                candidate=totals.copy();candidate[slot]+=values
                costs.append(np.sum((candidate-target)**2/np.maximum(target,1)))
            slot=int(np.argmin(costs));allocation[group]=names[slot];totals[slot]+=values
        if not (totals>0).all():allocation=None
    else:
        allocation=allocate_groups(frame[eligible],classes) if verified.ge(3).all() else None
    ready=allocation is not None
    if ready:frame.loc[eligible,'split']=frame.loc[eligible,'split_group'].map(allocation)
    for idx,row in tqdm(frame[frame.split.ne('')].iterrows(),desc='Подготовленные копии',total=int(frame.split.ne('').sum())):
        destination=Path('data/processed/images')/row['split']/row['poi_id']/(row.image_id+Path(row.file_path).suffix.lower())
        path=root/destination
        if not path.resolve().is_relative_to((root/'data/processed').resolve()):raise ValueError('Недопустимый путь POI')
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists() or digest(path)!=row.sha256:shutil.copy2(root/row.file_path,path)
        if digest(path)!=row.sha256:raise AssertionError('Неверный хэш копии')
        frame.at[idx,'processed_path']=destination.as_posix()
    released=frame[frame.split.ne('')]
    for column in ['image_id','sha256','duplicate_group','split_group']:
        assert released.groupby(column).split.nunique().le(1).all(),f'Утечка {column}'
    assert released.sha256.is_unique and released.label_status.eq('verified').all() and released.readable.all()
    if ready:
        for split in ['train','val','test']:assert set(released[released.split.eq(split)].poi_id)==set(classes)
    for split in ['train','val','test']:
        frame[frame.split.eq(split)].to_csv(root/f'data/annotations/{split}.csv',index=False)
    frame.to_csv(root/'data/annotations/dataset_manifest.csv',index=False)
    released.to_csv(root/'data/processed/dataset.csv',index=False)
    frame[frame.exclusion_reason.eq('exact_label_conflict')].to_csv(root/'data/annotations/review_conflicts.csv',index=False)
    sizes=frame.groupby('duplicate_group').size()
    queue=frame[frame.label_status.isin(['needs_review','uncertain'])|frame.exclusion_reason.eq('exact_label_conflict')].copy()
    queue['review_priority']=(queue.quality_flags.ne('').astype(int)+queue.duplicate_group.map(sizes).gt(1).astype(int)*2)
    queue.sort_values(['review_priority','poi_id','image_id'],ascending=[False,True,True]).to_csv(
        root/'data/annotations/review_queue.csv',index=False,encoding='utf-8-sig')
    return frame,ready


def contact_sheet(frame,root,path,title='',columns=3):
    import matplotlib
    font_path=Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'
    font=ImageFont.truetype(str(font_path),13)
    small=ImageFont.truetype(str(font_path),11)
    width,height=420,300
    canvas=Image.new('RGB',(columns*width,45+max(1,math.ceil(len(frame)/columns))*height),'white')
    draw=ImageDraw.Draw(canvas);draw.text((12,12),title,font=font,fill='black')
    for n,row in enumerate(frame.itertuples()):
        x,y=(n%columns)*width,45+(n//columns)*height
        try:
            thumb=root/f'reports/figures/thumbnails/{row.image_id}.jpg'
            image=read_rgb(thumb if thumb.exists() else root/row.file_path)
            image.thumbnail((width-16,220),Image.Resampling.LANCZOS)
            canvas.paste(image,(x+(width-image.width)//2,y))
        except (OSError,ValueError,Image.DecompressionBombError,Image.DecompressionBombWarning):
            draw.text((x+10,y+50),'UNREADABLE',font=font,fill='red')
        draw.text((x+8,y+225),row.image_id,font=font,fill='black')
        draw.text((x+8,y+244),'\n'.join(textwrap.wrap(row.poi_id,width=48)),font=font,fill='black')
        if hasattr(row,'label_status'):
            draw.text((x+8,y+282),row.label_status,font=small,fill='gray')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path,quality=88)
    return path


def visual_eda(frame,pairs,root):
    frame=frame[selected_present(frame)].copy()
    current_ids=set(frame.image_id)
    pairs=pairs[pairs.image_id_a.isin(current_ids)&pairs.image_id_b.isin(current_ids)]
    paths=[];out=root/'reports/figures'
    for poi,group in frame.groupby('poi_id',sort=True):
        slug=poi.replace('/','__')
        for page,start in enumerate(range(0,len(group),24),1):
            paths.append(contact_sheet(group.iloc[start:start+24],root,out/f'all_{slug}_{page:02}.jpg',f'{poi} | page {page}'))
        readable=group[group.readable]
        if readable.empty:continue
        selection=pd.concat([
            readable.sample(min(6,len(readable)),random_state=SEED),
            readable.nsmallest(2,'brightness'),readable.nlargest(2,'brightness'),
            readable.nsmallest(2,'sharpness'),
            readable.assign(aspect_deviation=np.abs(np.log(readable.aspect_ratio))).nlargest(2,'aspect_deviation')
        ]).drop_duplicates('image_id')
        paths.append(contact_sheet(selection,root,out/f'quality_{slug}.jpg',f'{poi} | random + brightness + sharpness + aspect'))
    indexed=frame.set_index('image_id',drop=False)
    for page,start in enumerate(range(0,len(pairs),10),1):
        ids=[]
        for pair in pairs.iloc[start:start+10].itertuples():ids.extend([pair.image_id_a,pair.image_id_b])
        paths.append(contact_sheet(indexed.loc[ids],root,out/f'duplicate_pairs_{page:03}.jpg',
                                   f'Duplicate candidates | page {page} | one pair per row',columns=2))
    pd.DataFrame({'figure_path':[p.relative_to(root).as_posix() for p in paths]}).to_csv(root/'reports/contact_sheets.csv',index=False)
    return paths


def quality_plots(frame,root):
    import matplotlib.pyplot as plt
    frame=frame[selected_present(frame)].copy()
    columns=['width','height','aspect_ratio','file_size','brightness','sharpness']
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for col,ax in zip(columns,axes.ravel()):
        ax.hist(frame[col].dropna(),bins=30,color='steelblue',edgecolor='white');ax.set_title(col)
    fig.tight_layout();fig.savefig(root/'reports/figures/technical_histograms.png',dpi=120);plt.close(fig)
    groups=list(frame.groupby('poi_id',sort=True))
    fig,axes=plt.subplots(1,3,figsize=(18,max(9,len(groups)*.5)))
    for col,ax in zip(['brightness','sharpness','aspect_ratio'],axes):
        values=[g[col].dropna().to_numpy() for _,g in groups]
        ax.boxplot(values,vert=False,tick_labels=[p for p,_ in groups]);ax.set_title(col)
        ax.tick_params(axis='y',labelsize=6)
        if col=='sharpness':ax.set_xscale('symlog',linthresh=1)
    fig.tight_layout();fig.savefig(root/'reports/figures/technical_boxplots.png',dpi=140);plt.close(fig)
    counts=frame.groupby('poi_id').size().sort_values()
    fig,ax=plt.subplots(figsize=(12,8));counts.plot.barh(ax=ax,color='steelblue')
    ax.set_xlabel('Images');fig.tight_layout();fig.savefig(root/'reports/figures/class_counts.png',dpi=120);plt.close(fig)


def show_image(frame,root,image_id):
    from IPython.display import display
    rows=frame[frame.image_id.eq(image_id)]
    if len(rows)!=1:raise KeyError(image_id)
    row=rows.iloc[0]
    print(f'{image_id} | {row.poi_id} | {row.file_path}')
    print(f'Автор: {row.author}; лицензия: {row.licence}; источник: {row.source_url}')
    display(read_rgb(root/row.file_path))


def verify_integrity(root,frame):
    root=Path(root)
    snapshot=json.loads((root/'reports/raw_integrity_snapshot.json').read_text(encoding='utf-8'))
    descriptors,_,_=discover_images(root)
    current={d['path'].relative_to(root).as_posix() for d in descriptors}
    expected={key for key in snapshot if Path(key).suffix.lower() in IMAGE_SUFFIXES}
    assert current==expected,'Состав raw изменился во время EDA'
    for path,expected_hash in tqdm(snapshot.items(),desc='SHA-256 исходников'):
        assert digest(root/path)==expected_hash,f'Исходник изменён: {path}'
    released=frame[frame.split.ne('')]
    for a,b in [('train','val'),('train','test'),('val','test')]:
        for key in ['image_id','sha256','duplicate_group','split_group']:
            assert set(released.loc[released.split.eq(a),key]).isdisjoint(set(released.loc[released.split.eq(b),key]))
    assert released.label_status.eq('verified').all() and released.sha256.is_unique
    assert frame.loc[frame.label_status.isin(['needs_review','uncertain']),'split'].eq('').all()
    for row in released.itertuples():assert digest(root/row.processed_path)==row.sha256
    return {'raw_files_checked':len(snapshot),'released_images_checked':len(released),'status':'passed'}


def make_report(frame,pairs,ready,root):
    released=frame[frame.split.ne('')];active=frame[frame.active]
    present=frame[selected_present(frame)]
    counts=present.groupby('initial_poi_id').size()
    exact_sizes=present.groupby('sha256').size()
    current_ids=set(present.image_id)
    pairs=pairs[pairs.image_id_a.isin(current_ids)&pairs.image_id_b.isin(current_ids)]
    potential=pairs[pairs.kind.isin(['phash_candidate','crop_candidate'])]
    stats={'total_images':len(present),'audit_records':len(frame),'removed_or_unselected':len(frame)-len(present),
           'cities':int(present.city.nunique()),'districts':sorted(present.district.unique()),
           'poi_count':int(frame.initial_poi_id.nunique()),'readable':int(frame.readable.sum()),
           'verified':int(frame.label_status.eq('verified').sum()),'needs_review':int(frame.label_status.eq('needs_review').sum()),
           'uncertain':int(frame.label_status.eq('uncertain').sum()),'rejected':int(frame.label_status.eq('rejected').sum()),
           'active_verified_after_cleaning':len(active),'released_images':len(released),
           'exact_duplicate_groups':int(exact_sizes.gt(1).sum()),'exact_duplicate_excess':int((exact_sizes-1).clip(lower=0).sum()),
           'potential_duplicate_pairs':len(potential),'commons_same_file_pairs':int(pairs.kind.eq('commons_same_file').sum()),
           'all_duplicate_pairs':len(pairs),'cross_poi_duplicate_pairs':int(pairs.cross_poi.sum()),
           'exact_label_conflict_images':int(frame.exclusion_reason.eq('exact_label_conflict').sum()),
           'raw_imbalance_max_min':float(counts.max()/counts.min()),
           'sources':{str(k):int(v) for k,v in present.source.value_counts().items()},
           'formats':{str(k):int(v) for k,v in present['format'].value_counts().items()},
           'split_counts':{s:int(frame.split.eq(s).sum()) for s in ['train','val','test']},
           'training_ready':bool(ready),'review_complete':not (
               frame.label_status.isin(['needs_review','uncertain']).any() or frame.exclusion_reason.eq('exact_label_conflict').any())}
    (root/'reports/eda_statistics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    table=[]
    for poi in sorted(set(frame.initial_poi_id)|set(frame.poi_id)):
        table.append(dict(poi_id=poi,original=int(present.initial_poi_id.eq(poi).sum()),
                          removed_or_unselected=int((frame.initial_poi_id.eq(poi)&~selected_present(frame)).sum()),
                          verified=int((frame.poi_id.eq(poi)&frame.label_status.eq('verified')).sum()),
                          active_after_cleaning=int((frame.poi_id.eq(poi)&frame.active).sum()),
                          train=int((frame.poi_id.eq(poi)&frame.split.eq('train')).sum()),
                          val=int((frame.poi_id.eq(poi)&frame.split.eq('val')).sum()),
                          test=int((frame.poi_id.eq(poi)&frame.split.eq('test')).sum())))
    pd.DataFrame(table).to_csv(root/'reports/class_distribution.csv',index=False)
    lines=['# EDA Wikimedia: «Невидаль»','',
      f'Готовность к первому обучению: **{"ДА, технически для выпущенных CSV" if ready else "НЕТ"}**.',
      f'В текущем наборе: {len(present)} изображений, {stats["poi_count"]} POI, {len(stats["districts"])} района: {", ".join(stats["districts"])}. В аудите {len(frame)} записей; удалённых/невыбранных: {len(frame)-len(present)}.',
      f'Читаемых: {stats["readable"]}; подтверждено: {stats["verified"]}; после очистки подтверждённых: {len(active)}; выпущено: {len(released)}.',
      f'needs_review: {stats["needs_review"]}; uncertain: {stats["uncertain"]}; rejected: {stats["rejected"]}.',
      '', '## Количество по POI','```',pd.DataFrame(table).to_string(index=False),'```',
      '', '## Причины исключения из активного датасета']
    for reason,n in frame.loc[~frame.active,'exclusion_reason'].value_counts().items():lines.append(f'- {reason}: {n}')
    lines+=['','Причины ручного/технического отклонения:']
    for reason,n in frame.loc[frame.label_status.eq('rejected'),'rejection_reason'].value_counts().items():lines.append(f'- {reason}: {n}')
    if not frame.label_status.eq('rejected').any():lines.append('- Отклонённых изображений пока нет.')
    lines+=['','## Наблюдение → интерпретация → рекомендуемое действие',
      (f'- {stats["needs_review"]} needs_review и {stats["uncertain"]} uncertain → часть меток не подтверждена → проверить review_edits.csv; эти кадры не выпускаются в splits.'
       if stats['needs_review'] or stats['uncertain'] else
       '- Метки всех текущих фотографий подтверждены пользователем → ручная проверка источников завершена; отдельные межклассовые конфликты проверяются дополнительно → сохранять решения в review_edits.csv.'),
      f'- Точных групп {stats["exact_duplicate_groups"]}, лишних точных копий {stats["exact_duplicate_excess"]}, потенциальных near-пар {len(potential)} → риск утечек и переоценки качества → проверить duplicate_pairs.csv; граф групп сохраняется до разделения.',
      f'- Межклассовых связей дубликатов {stats["cross_poi_duplicate_pairs"]} → возможны ошибки сортировки или совместное присутствие объектов → проверять такие пары в первую очередь.',
      f'- Подтверждённых строк с конфликтом меток точной копии: {stats["exact_label_conflict_images"]} → одна фотография не может подтверждать две разные single-label метки → конфликтующие строки исключены из splits; исправить review_conflicts.csv через review_edits.csv.',
      f'- Дисбаланс исходных классов max/min={stats["raw_imbalance_max_min"]:.2f} → малые классы и тесты будут нестабильны → добрать независимые ракурсы; сравнивать будущие модели по macro-метрикам.',
      f'- Малых кадров {int(present.quality_flags.str.contains("small_side",regex=False).sum())}, тёмных {int(present.quality_flags.str.contains("dark_mean",regex=False).sum())}, светлых {int(present.quality_flags.str.contains("bright_mean",regex=False).sum())} → это эвристические подозрения, а не доказанный брак → проверить тематические подборки; автоматического удаления по резкости нет.',
      f'- Неизвестных источников {int(present.source.eq("unknown").sum())}, без известного автора {int(present.author.isin(["","unknown"]).sum())}, без licence {int(present.licence.eq("").sum())} → сведения о происхождении неполны → сверить metadata.csv или manual_pois.csv, сохранив исходную атрибуцию.',
      f'- EXIF-время есть у {int(present.capture_datetime.ne("").sum())} кадров → оно само по себе не доказывает общую съёмку → внести известные глобальные series_id; они объединяют группы до split.',
      '- pHash и фиксированные кропы не обнаруживают все преобразования → сильное кадрирование/рамки могут остаться → просмотреть листы и вручную объединить пропущенные копии через series_id.',
      f'- Файлов формата MPO: {stats["formats"].get("MPO",0)} → расширение JPG может скрывать контейнер нескольких кадров; текущая проверка декодирует первый кадр → при ручном просмотре проверить пригодность первого кадра и позднее явно зафиксировать такое декодирование в обучающем загрузчике.',
      '- Предыдущие замечания могут относиться к удалённым кадрам → состав после ручной очистки изменился → пользоваться актуальными contact_sheets.csv и review_conflicts.csv; удалённые изображения не восстанавливаются.',
      '- Внешние фотографии Commons не гарантируют покрытие мобильной съёмки → возможен сдвиг домена → позднее собрать отдельный независимый полевой test.',
      '- Отбор manual фиксируется в config/dataset_sources.json → доступные кадры и причины отбора доступны в manual_enrichment_plan.csv → сохранять подтверждения пользователя в review_edits.csv; при split_anchors.csv старые назначения сохраняются, конфликтующие группы изолируются.',
      (f'- Manual licence=unknown у {int((present.source.eq("manual")&present.licence.eq("unknown")).sum())} кадров → лицензия не сообщена → заполнить config/manual_pois.csv.'
       if (present.source.eq('manual')&present.licence.eq('unknown')).any() else
       '- Для manual сохранены автор Shilkova A.M. и лицензия Public Domain по подтверждению пользователя → происхождение зафиксировано → сохранять эти поля при дальнейшем экспорте.'),
      '- Набора негативов нет → отказ на посторонних объектах не оценивается → собрать отдельный набор; не превращать целевые POI в unknown.',
      '- Метаданные лицензий и авторов сохранены → для использования остаются условия конкретных файлов → учитывать ссылки, атрибуцию и ограничения; наличие файла не означает автоматического разрешения всех способов использования.',
      '', '## Train/val/test',
      'Цель: 70/15/15, SEED=42. Граф включает exact/near/Commons-ID и ручные серии, в том числе связи через исключённые изображения.',
      'Выпуск требует хотя бы трёх подтверждённых независимых групп каждого класса и найденного распределения с покрытием всех классов. Это технический минимум, а не достаточная выборка для надёжного тестирования.',
      f'Распределение: {json.dumps(stats["split_counts"],ensure_ascii=False)}.',
      'Если покрытия нет, train/val/test содержат заголовки без строк. Процедура не подменяет непроверенные метки.',
      '', '## Блокирующие проблемы',
      ('Для текущих выпущенных CSV технических блокеров нет. Оставшиеся непроверенные кадры и конфликты исключены.' if ready else 'Ручная проверка не завершена или недостаточно подтверждённых независимых групп для всех классов; см. review_queue.csv и split_feasibility.csv.'),
      '', '## Методика',
      'Яркость и дисперсия Laplacian рассчитаны на копии с длинной стороной ≤1024 px. Резкость зависит от текстуры и масштаба.',
      'pHash: DCT 32×32, 63 AC-бита; Hamming ≤6. Для десяти 80%/60% кропов рабочая копия уменьшена до ≤1024 px; Hamming ≤4. Порог даёт кандидатов, не подтверждение копии.',
      'Thumbnails в reports/figures/thumbnails используются только для визуализаций. Выпущенные processed-файлы — побайтовые копии исходников; аугментации и обучение не выполнялись.']
    (root/'reports/eda_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return stats
