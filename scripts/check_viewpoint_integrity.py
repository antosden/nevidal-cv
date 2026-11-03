"""Проверка реальных manifests без обращения к ответам моделей."""
from pathlib import Path
import json
import sys
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.viewpoint_audit import read,dump,folders,digest,Graph


def main():
    data,report,models,_=folders(ROOT)
    config=json.loads((data/'final_config.json').read_text(encoding='utf-8'))
    included=read(report/'included_pois.csv');expected={r['poi_id']:int(r['N']) for r in included}
    mapping=json.loads((data/'class_mapping.json').read_text(encoding='utf-8'))
    assert set(mapping)==set(expected)
    heldout={name:read(data/'manifests'/f'{name}.csv') for name in ['val','test']}
    for seed in config['seeds']:
        modes={mode:read(data/'manifests'/f'train_{mode}_seed{seed}.csv') for mode in ['O','R']}
        for mode,rows in modes.items():
            assert Counter(r['poi_id'] for r in rows)==Counter(expected)
            assert len({r['near_duplicate_family'] for r in rows})==len(rows)
            assert len({r['image_id'] for r in rows})==len(rows)
            assert all(r['target_label_status']=='verified' for r in rows)
            assert all(int(r['class_index'])==mapping[r['poi_id']] for r in rows)
            for split,other in heldout.items():
                for field in ['image_id','sha256','duplicate_group','session_id','outing_id']:
                    assert not ({r[field] for r in rows if r[field]} & {r[field] for r in other if r[field]}),(seed,mode,split,field)
        for poi,n in expected.items():
            mono=[r for r in modes['O'] if r['poi_id']==poi]
            diverse=[r for r in modes['R'] if r['poi_id']==poi]
            assert Counter(r['viewpoint'] for r in mono)=={'main':n}
            assert Counter(r['viewpoint'] for r in diverse)=={v:n//3 for v in ['main','left','right']}
            assert Counter(r['source'] for r in mono)==Counter(r['source'] for r in diverse)
            for field in ['illumination','scale','occlusion']:
                a,b=Counter(r[field] for r in mono),Counter(r[field] for r in diverse)
                assert sum(abs(a[k]-b[k]) for k in set(a)|set(b))/(2*n)<=.2+1e-9
    for split,rows in heldout.items():
        assert set(r['poi_id'] for r in rows)==set(mapping)
        assert all(r['target_label_status']=='verified' for r in rows)
    for field in ['image_id','sha256','duplicate_group','session_id','outing_id']:
        assert not ({r[field] for r in heldout['val'] if r[field]} & {r[field] for r in heldout['test'] if r[field]})
    for name,sha in json.loads((data/'manifests_lock.json').read_text(encoding='utf-8')).items():assert digest(data/name)==sha,name
    dump(report/'manifest_assertions.json',dict(passed=True,classes=len(mapping),seeds=3,
         equal_per_poi_O_R=True,exact_sources=True,bounded_attributes_TV=True,
         controlled_viewpoint_ratios=True,all_labels_verified=True,no_known_split_intersections=True,
         independent_unknown_sessions_not_assumed=True,all_locked_hashes_match=True))
    print('All manifest assertions passed')


if __name__=='__main__':main()
