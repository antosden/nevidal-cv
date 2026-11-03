"""История разбиений и визуальный аудит нового исследования ракурсов."""
from pathlib import Path
import csv
import hashlib
import json
from collections import Counter,defaultdict

import numpy as np
from PIL import Image,ImageOps,ImageDraw,ImageFont

RUN_ID='viewpoint_20261102_v1'


def read(path):
    if not path.exists():return []
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def write(path,rows,fields=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


class Graph:
    def __init__(self):self.parent={}
    def find(self,key):
        self.parent.setdefault(key,key)
        while self.parent[key]!=key:
            self.parent[key]=self.parent[self.parent[key]];key=self.parent[key]
        return key
    def join(self,a,b):self.parent[self.find(a)]=self.find(b)


def folders(root):
    return root/'data/annotations'/RUN_ID,root/'reports'/RUN_ID,root/'models'/RUN_ID,root/'.runtime'/RUN_ID


def reconcile_history(root):
    """Включить связи отчётов, в том числе через отсутствующие/исключённые файлы."""
    root=Path(root);data,report,models,runtime=folders(root)
    rows=read(data/'inventory.csv');history=read(data/'history.csv')
    graph=Graph();graph.parent=json.loads((data/'history_components.json').read_text(encoding='utf-8'))
    files=[root/'data/annotations/duplicate_pairs.csv',root/'data/annotations/duplicate_groups.csv',
           root/'reports/research_vdnh_restart_2026_11_02/duplicate_candidates.csv']
    files+=list((root/'.runtime/research_reset_quarantine_2026_11_02/reports').glob('*/duplicate_candidates.csv'))
    files+=list((root/'.runtime/research_reset_quarantine_2026_11_02/reports').glob('*/duplicate_pairs.csv'))
    edges=[]
    for path in files:
        for row in read(path):
            a=row.get('image_id_a') or row.get('image_a');b=row.get('image_id_b') or row.get('image_b')
            ids=row.get('image_ids','').split(';') if row.get('image_ids') else []
            if a and b:ids=[a,b]
            for image_id in ids[1:]:
                graph.join('id:'+ids[0],'id:'+image_id)
                edges.append(dict(image_a=ids[0],image_b=image_id,source_table=path.relative_to(root).as_posix()))
    roles=defaultdict(set)
    for item in history:
        if item['historical_role'] in ('val','test'):roles[graph.find(item['node'])].add(item['historical_role'])
    rejected={graph.find('id:'+r['image_id']) for path in [root/'data/annotations/review_edits.csv',root/'data/annotations/dataset_manifest.csv'] for r in read(path) if r.get('label_status')=='rejected'}
    members=defaultdict(list)
    for row in rows:members[graph.find('id:'+row['image_id'])].append(row['image_id'])
    for row in rows:
        component=graph.find('id:'+row['image_id']);flags=roles[component]
        row['duplicate_group']='component_'+min(members[component])
        if component in rejected:row.update(split='excluded',selection_decision='reject_history_component')
        elif row['readable']!='True':row.update(split='excluded',selection_decision='unreadable')
        else:row['split']='test' if 'test' in flags else ('val' if 'val' in flags else 'train')
    # Предыдущий аудит этого нового run остаётся как история корректировки.
    if not (data/'inventory_initial.csv').exists():
        (data/'inventory_initial.csv').write_bytes((data/'inventory.csv').read_bytes())
    write(data/'inventory.csv',rows);write(data/'historical_duplicate_links.csv',edges)
    dump(data/'history_components.json',{key:graph.find(key) for key in graph.parent})
    target=report/'figures/train_final';target.mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    for poi in sorted({r['poi_id'] for r in rows}):
        bucket=[r for r in rows if r['poi_id']==poi and r['split']=='train']
        write(target/f'{poi.split("/")[-1]}_index.csv',bucket,fields=list(rows[0]))
        for start in range(0,len(bucket),40):
            sheet=Image.new('RGB',(1350,1760),'white');draw=ImageDraw.Draw(sheet)
            for local,row in enumerate(bucket[start:start+40]):
                with Image.open(runtime/'cache'/(row['sha256']+'.png')) as image:thumb=ImageOps.contain(image,(260,175))
                x,y=local%5*270,local//5*220;sheet.paste(thumb,(x+(270-thumb.width)//2,y))
                draw.text((x+2,y+176),f"{start+local+1}: {row['image_id']}",font=font,fill='black')
                draw.text((x+2,y+194),poi.split('/')[-1]+' '+row['source'][:8],font=font,fill='black')
            sheet.save(target/f'{poi.split("/")[-1]}_{start//40+1:02d}.jpg',quality=92)
    coverage=[dict(poi_id=poi,**{s:sum(r['poi_id']==poi and r['split']==s for r in rows) for s in ['train','val','test','excluded']}) for poi in sorted({r['poi_id'] for r in rows})]
    write(report/'reconciled_coverage.csv',coverage)
    result=dict(added_historical_links=len(edges),splits=dict(Counter(r['split'] for r in rows)),stage='ready_for_visual_train_review')
    dump(report/'history_reconciliation.json',result);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result


def audit(root):
    root=Path(root);data,report,models,runtime=folders(root)
    if (data/'inventory.csv').exists():raise FileExistsError('Аудит уже создан, не перезаписываем')
    for folder in (data,report,models,runtime/'cache',report/'figures/train',report/'figures/heldout'):folder.mkdir(parents=True,exist_ok=True)
    # Предыдущие результаты и production фиксируются до нового вычисления.
    protected=[]
    for folder in (root/'models/mobilenetv3_small_classifier',root/'models/research_vdnh_restart_2026_11_02',root/'reports/research_vdnh_restart_2026_11_02',root/'.runtime/mobile_argmax_v2/public'):
        protected.extend(p for p in folder.rglob('*') if p.is_file())
    protected.extend([root/'notebooks/09_mobilenetv3_similarity_research.ipynb',root/'config/research_vdnh.json'])
    dump(data/'previous_work_sha256.json',{p.relative_to(root).as_posix():digest(p) for p in protected if p.exists()})
    catalogue=read(root/'data/annotations/poi_catalog.csv');pois={r['poi_id'] for r in catalogue}
    graph=Graph();history=[];role_nodes=[];rejected=set();metadata={};path_id={};sha_id={}
    history_files=list((root/'data/annotations').glob('*.csv'))
    history_files+=list((root/'models/mobilenetv3_small_classifier/splits').glob('*.csv'))
    history_files+=list((root/'models/research_vdnh_restart_2026_11_02/splits').glob('*.csv'))
    history_files+=list((root/'.runtime/research_reset_quarantine_2026_11_02/models').glob('*/splits/*.csv'))
    history_files+=list((root/'.runtime/research_reset_quarantine_2026_11_02/models').glob('*/splits/historical_test_diagnostic_only.csv'))
    history_files+=list((root/'config').glob('small_classifier_*reviews.csv'))
    history_files+=list((root/'data/annotations/research_vdnh').glob('*inventory.csv'))
    history_files+=list((root/'data/annotations/research_vdnh').glob('*proposals.csv'))
    history_files+=list((root/'data/annotations/research_vdnh').glob('*decisions.csv'))
    for path in sorted(set(history_files)):
        snapshot=path.parent.as_posix()
        for row in read(path):
            image_id=row.get('image_id','');file_path=row.get('file_path','');sha=row.get('sha256','')
            if not image_id and not file_path and not sha:continue
            node='id:'+image_id if image_id else ('path:'+file_path if file_path else 'sha:'+sha)
            graph.find(node)
            if file_path:
                graph.join(node,'path:'+file_path);path_id.setdefault(file_path,image_id)
                if row.get('poi_id') in pois:metadata.setdefault(file_path,{}).update({k:v for k,v in row.items() if v})
            if sha:graph.join(node,'sha:'+sha);sha_id.setdefault(sha,image_id)
            if row.get('label_status')=='rejected':rejected.add(node)
            for key in ('duplicate_group','split_group','series_id'):
                if row.get(key):graph.join(node,f'history:{snapshot}:{key}:{row[key]}')
            role=row.get('split','')
            if 'test' in path.stem:role='test'
            elif path.stem.endswith('val'):role='val'
            elif path.stem=='train':role='train'
            if role in ('val','test'):role_nodes.append((node,role))
            history.append(dict(node=node,image_id=image_id,file_path=file_path,sha256=sha,historical_role=role,table=path.relative_to(root).as_posix()))
    # Современный immutable snapshot authoritative для точных существующих меток/происхождения.
    for path in [root/'models/mobilenetv3_small_classifier/splits'/f'{s}.csv' for s in ('train','val','test')]+[root/'models/research_vdnh_restart_2026_11_02/dataset_manifest.csv']:
        for row in read(path):
            if row.get('poi_id') in pois:metadata.setdefault(row['file_path'],{}).update({k:v for k,v in row.items() if v})
    paths={}
    for poi in pois:
        for path in (root/'data/raw/wikimedia'/poi).rglob('*'):
            if path.suffix.lower() in ('.jpg','.jpeg','.png','.webp','.mpo') and path.is_file():paths[path.relative_to(root).as_posix()]=poi
    for item in read(root/'config/manual_pois.csv'):
        poi='/'.join((item['city'],item['district'],item['poi']))
        for path in (root/'data/raw/manual'/item['folder']).rglob('*'):
            if path.suffix.lower() in ('.jpg','.jpeg','.png','.webp','.mpo') and path.is_file():paths[path.relative_to(root).as_posix()]=poi
    for row in read(root/'data/annotations/research_vdnh/source_inventory.csv'):
        if row['poi_id'] in pois:paths[row['file_path']]=row['poi_id']
    # Снимки Кыргызстана вне авторитетного каталога учитываются отдельно, не создают новый класс.
    uncatalogued=[r for r in read(root/'data/annotations/research_vdnh/walk_label_proposals.csv') if r['poi_id'] not in pois]
    dump(report/'uncatalogued_walk.json',dict(files=len(uncatalogued),pois=sorted({r['poi_id'] for r in uncatalogued}),used=False))
    dct=np.cos(np.pi*np.arange(32)[:,None]*(2*np.arange(32)[None,:]+1)/64)
    def phash(image):
        low=(dct@np.asarray(image.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=float)@dct.T)[:8,:8]
        return int(''.join('1' if b else '0' for b in (low>np.median(low)).flat),2)
    rows=[];hashes=[]
    for number,(name,initial_poi) in enumerate(sorted(paths.items()),1):
        path=root/name;old=metadata.get(name,{})
        image_id=path_id.get(name) or 'view_'+hashlib.sha256(name.encode()).hexdigest()[:16]
        node='id:'+image_id;graph.join(node,'path:'+name)
        row=dict(image_id=image_id,file_path=name,poi_id=old.get('poi_id') or initial_poi,
                 source=old.get('source') or ('wikimedia_commons' if '/wikimedia/' in name else 'unknown'),
                 author=old.get('author') or 'unknown',licence=old.get('licence') or old.get('license') or 'unknown',
                 source_url=old.get('source_url',''),sha256='',width=0,height=0,readable=False,
                 previous_label_status=old.get('label_status','needs_review'),viewpoint='undetermined',
                 visual_reason='',confidence='unreviewed',review_method='not_reviewed',reviewer='',
                 session_id=old.get('series_id',''),outing_id='walk_2026_11_01' if 'vdnh_experimenty/' in name else '',
                 scale='undetermined',illumination='undetermined',occlusion='undetermined',duplicate_group='',
                 split='',selection_decision='',capture_datetime=old.get('capture_datetime',''),error='')
        variants=[]
        try:
            row['sha256']=digest(path);graph.join(node,'sha:'+row['sha256'])
            with Image.open(path) as original:
                exif=original.getexif()
                try:details=exif.get_ifd(34665)
                except Exception:details={}
                row['capture_datetime']=str(details.get(36867) or exif.get(306) or row['capture_datetime'])
                image=ImageOps.exif_transpose(original).convert('RGB');row['width'],row['height']=image.size
                image=ImageOps.contain(image,(480,480),Image.Resampling.LANCZOS)
                w,h=image.size;boxes=[(0,0,w,h),(.1*w,.1*h,.9*w,.9*h),(0,0,.8*w,.8*h),(.2*w,.2*h,w,h)]
                variants=[phash(image.crop(tuple(map(int,box)))) for box in boxes]
                image.save(runtime/'cache'/(row['sha256']+'.png'));row['readable']=True
            if row['session_id']:graph.join(node,'session:'+row['poi_id']+':'+row['session_id'])
            elif row['capture_datetime'] and row['author'] not in ('unknown',''):
                row['session_id']='exif_'+hashlib.sha256((row['poi_id']+'|'+row['author']+'|'+row['capture_datetime'][:10]).encode()).hexdigest()[:16]
                graph.join(node,'session:'+row['session_id'])
            if row['outing_id']:graph.join(node,'outing:'+row['outing_id'])
        except (OSError,ValueError,SyntaxError) as error:row['error']=str(error)
        rows.append(row);hashes.append(variants)
        if number%200==0:print(f'Audit {number}/{len(paths)}',flush=True)
    edges=[]
    for i,a in enumerate(rows):
        for j in range(i):
            b=rows[j]
            if not hashes[i] or not hashes[j]:continue
            exact=a['sha256']==b['sha256'];full=(hashes[i][0]^hashes[j][0]).bit_count()
            crop=min((x^y).bit_count() for x in hashes[i] for y in hashes[j]) if a['poi_id']==b['poi_id'] else 65
            if exact or full<=6 or crop<=4:
                graph.join('id:'+a['image_id'],'id:'+b['image_id'])
                edges.append(dict(image_a=a['image_id'],image_b=b['image_id'],exact=exact,full_phash=full,crop_phash=crop))
    forbidden=defaultdict(set)
    for node,role in role_nodes:forbidden[graph.find(node)].add(role)
    rejected_components={graph.find(node) for node in rejected}
    comp_members=defaultdict(list)
    for row in rows:comp_members[graph.find('id:'+row['image_id'])].append(row['image_id'])
    for row in rows:
        component=graph.find('id:'+row['image_id']);roles=forbidden[component]
        row['duplicate_group']='component_'+min(comp_members[component])
        row['split']='test' if 'test' in roles else ('val' if 'val' in roles else 'train')
        row['selection_decision']='reject_history_component' if component in rejected_components else ('unreadable' if not row['readable'] else 'pending_visual_review')
        if row['selection_decision']!='pending_visual_review':row['split']='excluded'
    write(data/'inventory.csv',rows);write(data/'history.csv',history);write(data/'duplicate_candidates.csv',edges)
    dump(data/'history_components.json',{key:graph.find(key) for key in graph.parent})
    coverage=[]
    for poi in sorted(pois):
        bucket=[r for r in rows if r['poi_id']==poi]
        coverage.append(dict(poi_id=poi,raw=len(bucket),**{s:sum(r['split']==s for r in bucket) for s in ['train','val','test','excluded']}))
    write(report/'initial_coverage.csv',coverage)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    for split,subdir in [('train','train'),('val','heldout'),('test','heldout')]:
        for poi in sorted(pois):
            bucket=[r for r in rows if r['poi_id']==poi and r['split']==split]
            write(report/'figures'/subdir/f'{poi.split("/")[-1]}_{split}_index.csv',bucket)
            for start in range(0,len(bucket),40):
                sheet=Image.new('RGB',(5*270,8*220),'white');draw=ImageDraw.Draw(sheet)
                for local,row in enumerate(bucket[start:start+40]):
                    with Image.open(runtime/'cache'/(row['sha256']+'.png')) as image:thumb=ImageOps.contain(image,(260,175))
                    x,y=local%5*270,local//5*220;sheet.paste(thumb,(x+(270-thumb.width)//2,y))
                    draw.text((x+2,y+176),f"{start+local+1}: {row['image_id']}",font=font,fill='black')
                    draw.text((x+2,y+194),poi.split('/')[-1]+' '+row['source'][:8],font=font,fill='black')
                sheet.save(report/'figures'/subdir/f'{poi.split("/")[-1]}_{split}_{start//40+1:02d}.jpg',quality=90)
    summary=dict(run_id=RUN_ID,full_catalogue=len(pois),raw_files=len(rows),splits=dict(Counter(r['split'] for r in rows)),
                 historical_rows=len(history),duplicate_candidate_pairs=len(edges),independent_new_test=False,
                 known_walk_outing='Entire known outing held out conservatively; not moved into train',
                 stage='visual_annotation_pending',second_human_review=False)
    dump(report/'audit.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return summary
