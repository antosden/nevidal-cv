"""Фиксация данных и схемы анализа до первого нового обучения."""
from collections import Counter
from pathlib import Path
import json
import platform

from src.viewpoint_audit import read, write, dump, digest, folders, Graph
from src.viewpoint_statistics import bootstrap_plan


def finalize(root):
    root=Path(root);data,report,models,runtime=folders(root)
    if list((models/'runs').glob('*/signature.json')):
        raise ValueError('Нельзя менять фиксацию после начала обучения')
    config=json.loads((root/'config/research_mobilenetv3_viewpoint.json').read_text(encoding='utf-8'))
    config.update(balance_rule='exact source distribution; TV<=0.20 separately for light/scale/occlusion',
                  annotation_policy='Codex visual review; no independent second human',
                  study_status='exploratory_internal',test_independent=False)
    dump(data/'final_config.json',config)
    test=read(data/'manifests/test.csv')
    dump(data/'bootstrap_plan_before_predictions.json',bootstrap_plan(test))
    manifests=list((data/'manifests').glob('*.csv'))
    frames={p.stem:read(p) for p in manifests}
    used={r['image_id'] for rows in frames.values() for r in rows}
    labels=read(data/'viewpoint_labels.csv')
    # Проверяем исторические запреты через весь граф, включая отсутствующие файлы.
    graph=Graph();graph.parent=json.loads((data/'history_components.json').read_text(encoding='utf-8'))
    forbidden={graph.find(r['node']) for r in read(data/'history.csv') if r['historical_role'] in ('val','test')}
    for name,rows in frames.items():
        if name.startswith('train'):
            assert all(graph.find('id:'+r['image_id']) not in forbidden for r in rows),name
    decisions=[]
    for row in labels:
        membership=[name for name,rows in frames.items() if any(x['image_id']==row['image_id'] for x in rows)]
        decisions.append(dict(row,selected_manifests=';'.join(membership),
            research_selection='selected' if membership else (
                'target_'+row['target_label_status'] if row['target_label_status']!='verified' else
                'undetermined_viewpoint' if row['viewpoint']=='undetermined' and row['split']=='train' else
                'outside_matched_selection_or_catalogue')))
    write(data/'selection_decisions.csv',decisions)
    cache={}
    for row in labels:
        if row['image_id'] not in used:continue
        assert digest(root/row['file_path'])==row['sha256'],row['image_id']
        path=runtime/'cache'/(row['sha256']+'.png')
        cache[path.relative_to(root).as_posix()]=digest(path)
    dump(data/'cache_sha256.json',cache)
    coverage=[]
    for name,rows in frames.items():
        for poi in sorted({r['poi_id'] for r in rows}):
            group=[r for r in rows if r['poi_id']==poi]
            coverage.append(dict(manifest=name,poi_id=poi,images=len(group),
                **dict(Counter(r['viewpoint'] for r in group))))
    write(report/'manifest_viewpoint_coverage.csv',coverage)
    old=json.loads((data/'manifests_lock.json').read_text(encoding='utf-8'))
    extra=['final_config.json','bootstrap_plan_before_predictions.json','cache_sha256.json',
           'selection_decisions.csv','visual_review_decisions.json','heldout_visual_review_decisions.json',
           'source_design.json','history.csv','history_components.json','visual_duplicate_links.csv']
    for name in extra:
        if (data/name).exists():old[name]=digest(data/name)
    dump(data/'manifests_lock.json',old)
    import torch, torchvision
    dump(report/'environment.json',dict(python=platform.python_version(),torch=torch.__version__,
        torchvision=torchvision.__version__,device='CPU',platform=platform.platform(),
        weights={a:dict(file=name,sha256=digest(root/'.runtime/torch/checkpoints'/name)) for a,name in config['weights'].items()}))
    dump(report/'integrity_before_training.json',dict(historical_val_test_not_in_train=True,
         manifests_frozen=True,selected_raw_files_verified=True,cache_files=len(cache),
         model_predictions_used_for_selection=False,previous_work_changed=False))
    return dict(train=len(frames['train_O_seed42']),val=len(frames['val']),test=len(test),cache_files=len(cache))
