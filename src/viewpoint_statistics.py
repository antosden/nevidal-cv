"""Подписанные контрасты четырёх конфигураций и кластерная неопределённость."""
from collections import defaultdict
import numpy as np

CONFIGS=('small_O','large_O','small_R','large_R')


def macro_f1(truth,prediction,classes,weights=None):
    truth=np.asarray(truth,dtype=int);prediction=np.asarray(prediction,dtype=int)
    weights=np.ones(len(truth)) if weights is None else np.asarray(weights,dtype=float)
    matrix=np.bincount(truth*classes+prediction,weights=weights,minlength=classes*classes).reshape(classes,classes)
    denominator=matrix.sum(0)+matrix.sum(1)
    return float(np.divide(2*np.diag(matrix),denominator,out=np.zeros(classes),where=denominator>0).mean())


def contrasts(truth,predictions,classes,weights=None):
    arrays={key:np.asarray(predictions[key]) for key in CONFIGS}
    shape=arrays[CONFIGS[0]].shape
    if len(shape)!=2 or shape[1]!=len(truth) or any(value.shape!=shape for value in arrays.values()):
        raise ValueError('Нужны четыре парные матрицы seed × image')
    scores={key:np.array([macro_f1(truth,row,classes,weights) for row in value]) for key,value in arrays.items()}
    go=scores['large_O']-scores['small_O'];gr=scores['large_R']-scores['small_R'];c=go-gr
    small_gain=scores['small_R']-scores['small_O'];large_gain=scores['large_R']-scores['large_O']
    return dict(F={key:float(value.mean()) for key,value in scores.items()},
                G_O=float(go.mean()),G_R=float(gr.mean()),C=float(c.mean()),
                small_gain=float(small_gain.mean()),large_gain=float(large_gain.mean()),
                per_seed=[dict(G_O=float(a),G_R=float(b),C=float(d),small_gain=float(s),large_gain=float(l)) for a,b,d,s,l in zip(go,gr,c,small_gain,large_gain)],
                sample_SD={key:float(np.std(value,ddof=1)) if len(value)>1 else 0. for key,value in scores.items()})


def bootstrap_plan(rows):
    unknown=[r['image_id'] for r in rows if not r.get('outing_id') and not r.get('session_id')]
    outings=defaultdict(list);sessions=defaultdict(lambda:defaultdict(list))
    for i,row in enumerate(rows):
        if row.get('outing_id'):outings[row['outing_id']].append(i)
        elif row.get('session_id'):sessions[row['poi_id']][row['session_id']].append(i)
    per_poi={poi:len({('outing:'+r['outing_id']) if r.get('outing_id') else ('session:'+r['session_id']) for r in rows if r['poi_id']==poi and (r.get('outing_id') or r.get('session_id'))}) for poi in sorted({r['poi_id'] for r in rows})}
    reasons=[]
    if unknown:reasons.append(f'{len(unknown)} test-фото без известного сеанса/выхода; нельзя считать отдельными независимыми наблюдениями')
    if any(n<2 for n in per_poi.values()):reasons.append('У части POI меньше двух установленных независимых сеансов')
    if len(outings)==1:reasons.append('Только один общий известный выход: его межвыходную вариативность оценить нельзя')
    return dict(feasible=not reasons,reasons=reasons,unknown_session_images=len(unknown),known_outings=len(outings),
                sessions_per_poi=per_poi,scheme='shared outing blocks jointly across POI; other sessions within fixed POI; reject draws missing a class',
                unknown_images_are_not_independent_sessions=True)


def clustered_bootstrap(truth,predictions,rows,classes,replicates=5000,seed=2026):
    point=contrasts(truth,predictions,classes);plan=bootstrap_plan(rows)
    result=dict(point=point,bootstrap_plan=plan,requested_replicates=replicates,performed_replicates=0,
                confidence=.95,seed=seed,C_ci95=None,G_R_ci95=None,
                conclusion='inconclusive_missing_session_history',hypothesis_confirmed=False,
                study_status='exploratory_internal',conditional_on_fitted_seed_pairs=True)
    if not plan['feasible']:return result
    outings=defaultdict(list);sessions=defaultdict(lambda:defaultdict(list))
    for i,row in enumerate(rows):
        if row.get('outing_id'):outings[row['outing_id']].append(i)
        else:sessions[row['poi_id']][row['session_id']].append(i)
    outing_blocks=[np.array(v) for v in outings.values()]
    strata=[[np.array(v) for v in bucket.values()] for bucket in sessions.values()]
    rng=np.random.default_rng(seed);c_values=[];gr_values=[];attempts=0
    while len(c_values)<replicates:
        attempts+=1
        if attempts>replicates*100:raise ValueError('Невозможно сохранить покрытие классов при кластерном bootstrap')
        weights=np.zeros(len(rows))
        if outing_blocks:
            for index in rng.integers(0,len(outing_blocks),len(outing_blocks)):weights[outing_blocks[index]]+=1
        for blocks in strata:
            for index in rng.integers(0,len(blocks),len(blocks)):weights[blocks[index]]+=1
        if any(weights[np.asarray(truth)==label].sum()==0 for label in range(classes)):continue
        current=contrasts(truth,predictions,classes,weights)
        c_values.append(current['C']);gr_values.append(current['G_R'])
    c_interval=np.quantile(c_values,[.025,.975]).tolist();gr_interval=np.quantile(gr_values,[.025,.975]).tolist()
    decision='supported_internal_only' if c_interval[0]>=.03 and gr_interval[1]<=.03 else (
        'threshold_not_reached' if c_interval[1]<.03 or gr_interval[0]>.03 else 'inconclusive')
    result.update(C_ci95=c_interval,G_R_ci95=gr_interval,performed_replicates=replicates,conclusion=decision)
    # Поисковый внутренний test не превращается в независимое подтверждение.
    return result
