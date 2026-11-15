"""Independent goal score reconstruction, paired root-level uncertainty."""
import argparse,json,re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--split',choices=['control_test','control_shift'],required=True);a=p.parse_args()
freeze=json.loads((ROOT/'results/control-freeze.json').read_text());roots=freeze['roots'][a.split]
families=np.array([['counter','toggle','flow'].index(json.loads((ROOT/'data/panel-control-v1'/a.split/r/'inputs.json').read_text())['family']) for r in roots])
rng=np.random.default_rng(20261113)
samples=np.stack([np.concatenate([rng.choice(np.flatnonzero(families==f),sum(families==f),replace=True) for f in range(3)]) for _ in range(10000)])
def interval(values):
    # A percentile bootstrap is degenerate when every root is 0 or 1. Use a
    # conservative extreme-event bound with n independent roots, never n*seeds.
    # AM-GM bounds the all-success event by mean(p_i)**n, even across families.
    if np.all(values==1):return [float(.025**(1/len(values))),1.]
    if np.all(values==0):return [0.,float(1-.025**(1/len(values)))]
    return np.quantile(values[samples].mean(1),[.025,.975]).tolist()
data={};metrics={}
for path in sorted((ROOT/'results/control-goals').glob('*-'+a.split)):
    if not (path/'complete.json').exists():continue
    records=[json.loads(x) for x in (path/'trials.jsonl').read_text().splitlines()]
    assert [r['root_id'] for r in records]==roots,path
    values=[];errors=[];id_ok=[];binding_ok=[]
    for r in records:
        audit=json.loads((path/r['root_id']/'audit.json').read_text());assert len(audit)==8
        lab=json.loads((ROOT/'data/panel-control-v1'/a.split/r['root_id']/'labels.json').read_text())
        assert [x['state'] for x in audit[:4]]==lab['context_states']
        actions=[t for d in r['decisions'] for t in d['executed_packet']]
        assert len(actions)==4 and actions==[x['event']['command']['target'] for x in audit[4:]]
        assert all(x['event']['command']['dt']==1 for x in audit[4:])
        assert [x['event']['before'] for x in audit[4:]]==[x['state'] for x in audit[3:-1]]
        true_goal=lab['goal_state']['values'];final=audit[-1]['state']['values']
        assert r['true_goal_by_slot']==true_goal and r['final_true_values']==final
        ok=final==true_goal;err=sum(abs(x-y) for x,y in zip(final,true_goal))/3
        assert ok==r['success'] and abs(err-r['absolute_goal_error']/3)<1e-12
        values.append(ok);errors.append(err)
        id_ok.append(r['ids']==audit[-1]['state']['ids'])
        binding_ok.append(r['decisions'][0]['goal_by_slot']==true_goal)
    values=np.asarray(values,dtype=float);data[path.name]=values
    metrics[path.name]={'roots':len(roots),'success':float(values.mean()),'success_ci95':interval(values),
        'by_family':{f:float(values[families==i].mean()) for i,f in enumerate(['counter','toggle','flow'])},
        'mean_absolute_goal_error':float(np.mean(errors)),'all_identifiers_exact':float(np.mean(id_ok)),
        'initial_goal_binding_exact':float(np.mean(binding_ok))}
groups={}
assert len(metrics)==24,('Wait for all21prespecified and3exploratory conditions',len(metrics))
for name,values in data.items():
    group=re.sub(r'-seed(?:17|29|43)-','-seedmean-',name)
    groups.setdefault(group,[]).append((name,values))
means={};mean_arrays={}
for group,items in groups.items():
    assert len(items)==(3 if '-seedmean-' in group else 1),(group,'incomplete seed set')
    values=np.mean([v for _,v in items],0);mean_arrays[group]=values
    means[group]={'seeds':len(items),'success':float(values.mean()),'success_ci95':interval(values),
        'seed_range':[float(min(v.mean() for _,v in items)),float(max(v.mean() for _,v in items))],
        'by_family':{f:float(values[families==i].mean()) for i,f in enumerate(['counter','toggle','flow'])}}
pairs={}
names=list(mean_arrays)
for i,first in enumerate(names):
    for second in names[i+1:]:
        diff=mean_arrays[first]-mean_arrays[second]
        pairs[first+' MINUS '+second]={'success_difference':float(diff.mean()),'ci95':np.quantile(diff[samples].mean(1),[.025,.975]).tolist()}
out={'split':a.split,'roots':len(roots),'unit':'root; ten roots per family; 10000 paired stratified bootstrap samples; seeds averaged within root',
    'boundary_uncertainty':'All0/all1 root outcomes use a conservative two-sided extreme-event bound, alpha/2=.025 and n=30 independent roots (not90 model-root records). Other intervals are paired stratified percentile root bootstrap. Pairwise differences remain bootstrap descriptive intervals; identical observations do not prove population equivalence.',
    'runs':metrics,'seed_means':means,'paired_differences':pairs}
target=ROOT/'results/analysis'/('control-'+a.split+'.json');target.write_text(json.dumps(out,indent=2))
print(json.dumps({'split':a.split,'runs':len(metrics),'means':means}),flush=True)
