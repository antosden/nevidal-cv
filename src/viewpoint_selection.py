"""Управляемые О/Р из явно просмотренных ракурсов, без ответов моделей."""
from collections import Counter,defaultdict
from pathlib import Path
import json
import random

from src.viewpoint_audit import read,write,dump,digest,Graph,folders,RUN_ID

VIEWPOINTS=('main','left','right')
ATTRIBUTES=('source','illumination','scale','occlusion')


def annotate(root):
    root=Path(root);data,report,_,_=folders(root)
    rows=read(data/'inventory.csv')
    review=json.loads((data/'visual_review_decisions.json').read_text(encoding='utf-8'))
    cards=[];annotations=[]
    for poi in sorted({r['poi_id'] for r in rows}):
        slug=poi.split('/')[-1];decision=review.get(slug)
        train=read(report/'figures/train_final'/f'{slug}_index.csv')
        lookup={r['image_id']:i+1 for i,r in enumerate(train)}
        if decision:
            cards.append(dict(poi_id=poi,main=decision['main_definition'],left=decision['left_definition'],right=decision['right_definition'],
                              undetermined=decision.get('undetermined_rule','Задний/неузнаваемый фасад, интерьер, общий фрагмент или недостоверная сторона.'),
                              examples={v:[train[i-1]['image_id'] for i in decision.get(v,[])[:3]] for v in VIEWPOINTS},
                              reviewer='codex_visual_review',independent_second_human=False))
        for original in [r for r in rows if r['poi_id']==poi]:
            row=dict(original)
            row.update(viewpoint='undetermined',visual_reason='Ракурс не установлен; не используется для управляемого train.',
                       confidence='undetermined',review_method='not_reviewed',reviewer='',target_label_status='needs_review')
            index=lookup.get(row['image_id'])
            if decision and index:
                memberships=[v for v in VIEWPOINTS if index in decision.get(v,[])]
                if len(memberships)>1:raise ValueError(f'Противоречивый ракурс: {row["image_id"]}')
                if decision.get('reviewed_all'):
                    row.update(review_method='contact_sheet_visual_review',reviewer='codex_visual_review',
                               visual_reason='Просмотрен контактный лист: нельзя уверенно установить один из трёх заданных видов.',target_label_status='verified')
                if memberships:
                    v=memberships[0]
                    row.update(viewpoint=v,visual_reason=decision[v+'_definition'],confidence='high',
                               review_method='contact_sheet_visual_review',reviewer='codex_visual_review',target_label_status='verified',
                               scale='whole',illumination='day',occlusion='low')
                    for field,buckets in decision.get('attributes',{}).items():
                        for value,indices in buckets.items():
                            if index in indices:row[field]=value
                    if index in decision.get('originals_checked',[]):row['review_method']='contact_sheet_and_original_visual_review'
                if index in decision.get('target_rejected',[]):
                    row.update(target_label_status='rejected',viewpoint='undetermined',visual_reason='На просмотренном кадре другой POI либо целевой объект отсутствует.')
                if index in decision.get('target_uncertain',[]):
                    row.update(target_label_status='uncertain',viewpoint='undetermined',visual_reason='Идентичность целевого POI по кадру неоднозначна.')
            annotations.append(row)
    # Разметка heldout — отдельный просмотр; её никогда не используем для train.
    heldout=review.get('_heldout',{})
    for row in annotations:
        found=heldout.get(row['image_id'])
        if found:
            row.update(found,reviewer='codex_visual_review',review_method='contact_sheet_visual_review')
    heldout_file=data/'heldout_visual_review_decisions.json'
    if heldout_file.exists():
        batches=json.loads(heldout_file.read_text(encoding='utf-8'))
        by_id={r['image_id']:r for r in annotations}
        for slug,decision in batches.items():
            indexed=read(report/'figures/heldout_selected'/f'{slug}_index.csv')
            card=review[slug]
            for index,item in enumerate(indexed,1):
                row=by_id[item['image_id']]
                if not decision.get('reviewed_all'):continue
                row.update(target_label_status='verified',reviewer='codex_visual_review',review_method='contact_sheet_visual_review',
                           visual_reason='Целевой POI визуально присутствует; направление по карточке не установлено.',confidence='undetermined')
                for v in VIEWPOINTS:
                    if index in decision.get(v,[]):
                        row.update(viewpoint=v,visual_reason=card[v+'_definition'],confidence='high',scale='whole',illumination='day',occlusion='low')
                for field,buckets in decision.get('attributes',{}).items():
                    for value,indices in buckets.items():
                        if index in indices:row[field]=value
                if index in decision.get('target_rejected',[]):
                    row.update(target_label_status='rejected',viewpoint='undetermined',visual_reason='Другой POI либо целевой объект отсутствует.')
                if index in decision.get('target_uncertain',[]):
                    row.update(target_label_status='uncertain',viewpoint='undetermined',visual_reason='Идентичность POI по кадру не установлена уверенно, без использования модели.')
                if index in decision.get('originals_checked',[]):row['review_method']='contact_sheet_and_original_visual_review'
    write(data/'viewpoint_labels.csv',annotations);dump(data/'orientation_cards.json',cards)
    coverage=[]
    for poi in sorted({r['poi_id'] for r in rows}):
        bucket=[r for r in annotations if r['poi_id']==poi and r['split']=='train']
        coverage.append(dict(poi_id=poi,**{v:sum(r['viewpoint']==v for r in bucket) for v in (*VIEWPOINTS,'undetermined')},
                             unreviewed=sum(r['review_method']=='not_reviewed' for r in bucket)))
    write(report/'viewpoint_coverage.csv',coverage)
    return annotations


def strata_candidates(rows):
    result=defaultdict(lambda:defaultdict(list))
    for row in rows:
        if row['viewpoint'] in VIEWPOINTS and row['target_label_status']=='verified':
            result[tuple(row[a] for a in ATTRIBUTES)][row['viewpoint']].append(row)
    return result


def paired_selection(rows,n,seed):
    """Точные источники, TV<=0.20 для остальных условий; правило до предсказаний."""
    if n%3 or n<6:raise ValueError('N должен быть кратен трём и >=6')
    rng=random.Random(seed);strata=strata_candidates(rows)
    capacities={key:min(len(bucket['main'])//3,len(bucket['left']),len(bucket['right'])) for key,bucket in strata.items()}
    if sum(capacities.values())>=n//3:
        slots=[key for key,count in capacities.items() for _ in range(count)]
        rng.shuffle(slots);allocation=Counter(slots[:n//3]);mono=[];diverse=[]
        for key,k in sorted(allocation.items()):
            bucket=strata[key]
            chosen={v:rng.sample(sorted(bucket[v],key=lambda r:r['image_id']),3*k if v=='main' else k) for v in VIEWPOINTS}
            mono.extend(chosen['main']);diverse.extend(rng.sample(chosen['main'],k)+chosen['left']+chosen['right'])
        return {'O':sorted(mono,key=lambda r:r['image_id']),'R':sorted(diverse,key=lambda r:r['image_id'])}
    pools={v:sorted([r for r in rows if r['viewpoint']==v],key=lambda r:r['image_id']) for v in VIEWPOINTS}
    if len(pools['main'])<n or any(len(pools[v])<n//3 for v in ['left','right']):return None
    def tv(a,b,field):
        x,y=Counter(r[field] for r in a),Counter(r[field] for r in b)
        return sum(abs(x[k]-y[k]) for k in set(x)|set(y))/(2*n)
    for attempt in range(4000):
        picked={v:rng.sample(pools[v],n//3) for v in VIEWPOINTS}
        diverse=picked['main']+picked['left']+picked['right'];mono=list(picked['main'])
        needed=Counter(r['source'] for r in picked['left']+picked['right'])
        used={r['image_id'] for r in mono};valid=True
        for source,count in needed.items():
            available=[r for r in pools['main'] if r['source']==source and r['image_id'] not in used]
            if len(available)<count:valid=False;break
            extra=rng.sample(available,count);mono.extend(extra);used.update(r['image_id'] for r in extra)
        if not valid:continue
        if any(tv(mono,diverse,field)>.20+1e-9 for field in ATTRIBUTES[1:]):continue
        sessions=lambda items:len({r['session_id'] for r in items if r['session_id']})
        if abs(sessions(mono)-sessions(diverse))>1:continue
        if abs(sum(not r['session_id'] for r in mono)-sum(not r['session_id'] for r in diverse))/n>.20+1e-9:continue
        assert Counter(r['source'] for r in mono)==Counter(r['source'] for r in diverse)
        return {'O':sorted(mono,key=lambda r:r['image_id']),'R':sorted(diverse,key=lambda r:r['image_id'])}
    return None


def build_manifests(root):
    root=Path(root);data,report,models,_=folders(root)
    config=json.loads((root/'config/research_mobilenetv3_viewpoint.json').read_text(encoding='utf-8'))
    if (data/'manifests_lock.json').exists():raise FileExistsError('Манифесты уже зафиксированы')
    rows=read(data/'viewpoint_labels.csv')
    # Семьи копий без смешения с широкими сеансами; графические связи не исчезают через исключения.
    graph=Graph();by_sha={}
    for row in rows:
        graph.find(row['image_id'])
        if row['sha256'] in by_sha:graph.join(row['image_id'],by_sha[row['sha256']])
        by_sha[row['sha256']]=row['image_id']
    for edge in read(data/'duplicate_candidates.csv')+read(data/'historical_duplicate_links.csv'):
        graph.join(edge['image_a'],edge['image_b'])
    for link in read(data/'visual_duplicate_links.csv'):
        graph.join(link['image_a'],link['image_b'])
    families=defaultdict(list)
    for row in rows:
        if row['split']=='train' and row['viewpoint'] in VIEWPOINTS and row['target_label_status']=='verified':
            families[(row['poi_id'],graph.find(row['image_id']))].append(row)
    candidates=[];conflicts=[]
    for key,bucket in families.items():
        if len({r['viewpoint'] for r in bucket})>1:
            conflicts.extend(dict(image_id=r['image_id'],reason='near_family_has_multiple_viewpoints') for r in bucket)
            continue
        row=max(bucket,key=lambda r:(r['source'] not in ('unknown','maps_unknown_service'),int(r['width'])*int(r['height']),r['image_id']))
        candidates.append(dict(row,near_duplicate_family=key[1]))
    selections={seed:{'O':[],'R':[]} for seed in config['seeds']};included=[];excluded=[]
    for poi in sorted({r['poi_id'] for r in rows}):
        bucket=[r for r in candidates if r['poi_id']==poi];found=None
        for n in config['candidate_N']:
            proposal={seed:paired_selection(bucket,n,seed) for seed in config['seeds']}
            if all(proposal.values()):found=(n,proposal);break
        if found:
            n,proposal=found;included.append(dict(poi_id=poi,N=n))
            for seed in config['seeds']:
                for mode in ['O','R']:selections[seed][mode].extend(proposal[seed][mode])
        else:
            counts=Counter(r['viewpoint'] for r in bucket)
            excluded.append(dict(poi_id=poi,main=counts['main'],left=counts['left'],right=counts['right'],
                                 reason='Не хватает минимум 6 основных + 2 левых + 2 правых разных семей при точном балансе источников и TV<=0.20 по свету/масштабу/перекрытиям, либо ракурсы не установлены.'))
    mapping={row['poi_id']:i for i,row in enumerate(included)}
    if len(mapping)<2:raise ValueError('Менее двух POI с выполнимым контролируемым сравнением')
    selected_frames={}
    for seed in config['seeds']:
        for mode in ['O','R']:
            bucket=[dict(r,class_index=mapping[r['poi_id']],mode=mode,seed=seed) for r in selections[seed][mode]]
            write(data/'manifests'/f'train_{mode}_seed{seed}.csv',bucket)
            selected_frames[(mode,seed)]=bucket
    for split in ['val','test']:
        bucket=[dict(r,class_index=mapping[r['poi_id']]) for r in rows if r['poi_id'] in mapping and r['split']==split and r['target_label_status']=='verified']
        # Один экземпляр полного SHA; сложные корректные ракурсы остаются, даже undetermined.
        seen=set();unique=[]
        for row in sorted(bucket,key=lambda r:r['image_id']):
            if row['sha256'] not in seen:unique.append(row);seen.add(row['sha256'])
        if {r['poi_id'] for r in unique}!=set(mapping):raise ValueError(f'{split}: не все POI визуально подтверждены')
        write(data/'manifests'/f'{split}.csv',unique);selected_frames[(split,0)]=unique
    for (a,seed_a),first in selected_frames.items():
        for (b,seed_b),second in selected_frames.items():
            if a in ('O','R') and b in ('O','R'):continue
            if a==b:continue
            for field in ['image_id','sha256','duplicate_group','session_id','outing_id']:
                assert not ({r[field] for r in first if r.get(field)} & {r[field] for r in second if r.get(field)}),(a,b,field)
    balance=[]
    for seed in config['seeds']:
        for poi in mapping:
            for mode in ['O','R']:
                bucket=[r for r in selections[seed][mode] if r['poi_id']==poi]
                balance.append(dict(poi_id=poi,seed=seed,mode=mode,images=len(bucket),
                    known_sessions=len({r['session_id'] for r in bucket if r['session_id']}),unknown_session_images=sum(not r['session_id'] for r in bucket),
                    **{a:json.dumps(dict(Counter(r[a] for r in bucket)),ensure_ascii=False) for a in (*ATTRIBUTES,'viewpoint')}))
    write(report/'selection_balance.csv',balance);write(report/'included_pois.csv',included);write(report/'excluded_pois.csv',excluded)
    write(report/'viewpoint_family_conflicts.csv',conflicts,fields=['image_id','reason'])
    dump(data/'class_mapping.json',mapping)
    dump(data/'selection_rule.json',dict(strata=ATTRIBUTES,exact_sources=True,max_TV_other_attributes=.20,
        rule='Prefer exact joint strata; otherwise bounded matching, 4000 deterministic attempts; R main included in O',
        known_session_count_difference_max=1,unknown_session_fraction_difference_max=.20,
        duplicate_family_one_image=True,independence_not_assumed_for_unknown_sessions=True,
        limitation='Residual framing/light/occlusion differences are reported; not asserted to be a pure causal viewpoint effect'))
    dump(data/'manifests_lock.json',{p.relative_to(data).as_posix():digest(p) for p in list((data/'manifests').glob('*.csv'))+[data/'class_mapping.json',data/'orientation_cards.json',data/'viewpoint_labels.csv',data/'selection_rule.json']})
    dump(report/'selection_status.json',dict(K=len(mapping),included=included,excluded=excluded,
         independent_second_human_review=False,stage='exploratory_internal_pending_pilot',
         train_images_per_configuration=sum(r['N'] for r in included),val=len(selected_frames[('val',0)]),test=len(selected_frames[('test',0)])))
    return included
