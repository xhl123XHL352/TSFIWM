"""Within-root input-equivalence upper bounds, independent of any predictor.

The analysis grants each root/object its own arbitrary prediction function.
It is therefore an optimistic empirical ceiling, not an executable baseline.
Within a root, omitted other-slot action identities can still force two
causally distinct branches to have identical local inputs and different labels.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def pareto(options):
    unique=set(options)
    return [(c,u) for c,u in unique if not any(c2>=c and u2>=u and (c2>c or u2>u) for c2,u2 in unique)]


def ceiling(groups):
    fixed_c=fixed_u=den_c=den_u=total=correct_upper=0
    ambiguous=[]
    for records in groups.values():
        den_c+=sum(c for _,c in records);den_u+=sum(not c for _,c in records)
        total+=len(records)
        counts=Counter(v for v,_ in records);correct_upper+=max(counts.values())
        options=pareto([(sum(y==v and ch for y,ch in records),sum(y==v and not ch for y,ch in records)) for v in counts])
        if len(options)==1:
            fixed_c+=options[0][0];fixed_u+=options[0][1]
        else:
            ambiguous.append(options)
    # Exact integer Pareto dynamic program for maximizing C*U, not independent
    # changed/unchanged maxima that might be mutually unattainable.
    dp=np.array([0],dtype=np.int64)
    for options in ambiguous:
        result=np.full(len(dp)+max(c for c,_ in options),-1,dtype=np.int64)
        for c,u in options:
            result[c:c+len(dp)]=np.maximum(result[c:c+len(dp)],np.where(dp>=0,dp+u,-1))
        dp=result
    valid=dp>=0
    products=(fixed_c+np.arange(len(dp)))[valid]*(fixed_u+dp[valid])
    return {'cge_upper':float(np.sqrt(products.max()/(den_c*den_u))),
        'field_exact_upper':correct_upper/total,'fields':total,'changed_fields':den_c,'unchanged_fields':den_u,
        'equivalence_groups':len(groups),'mixed_tradeoff_groups':len(ambiguous)}


def analyze(split,mode):
    family_groups=[defaultdict(list) for _ in range(3)]
    root_possible={};witness=None;contradiction_roots=set()
    for folder in sorted((ROOT/'data'/'panel-v3'/split).iterdir()):
        inp=json.loads((folder/'inputs.json').read_text());lab=json.loads((folder/'labels.json').read_text())
        family=['counter','toggle','flow'].index(inp['family']);root=inp['root_id']
        root_possible[root]=True
        within=defaultdict(list)
        for branch,states in zip(inp['branches'],lab['future_states']):
            prefix=[]
            for t,(action,state) in enumerate(zip(branch['actions'],states)):
                prefix.append((action['target'],action['dt']))
                previous=lab['context_states'][-1] if t==0 else states[t-1]
                for obj in range(3):
                    if mode=='local_incomplete':
                        external=tuple((target==obj,target<0,dt) for target,dt in prefix)
                    elif mode=='global_target_blind':
                        external=tuple(dt for _,dt in prefix)
                    else:
                        external=tuple(prefix)
                    key=(root,obj,external)
                    value=int(state['values'][obj]);changed=value!=previous['values'][obj]
                    family_groups[family][key].append((value,changed))
                    within[key].append({'branch':branch['branch'],'step':t,'object':obj,
                        'value':value,'previous':previous['values'][obj],'pressed':lab['branch_events'][branch['branch']][t]['pressed'],
                        'actions':[a['target'] for a in branch['actions'][:t+1]],
                        'durations':[a['dt'] for a in branch['actions'][:t+1]],
                        'image':str((folder/branch['future_images'][t]).relative_to(ROOT))})
        for key,records in within.items():
            if len(set(r['value'] for r in records))>1:
                contradiction_roots.add(root);root_possible[root]=False
                if witness is None and family==0 and mode=='local_incomplete':
                    for first in records:
                        for second in records:
                            if first['value']!=second['value'] and first['previous']==second['previous'] and first['pressed']!=second['pressed']:
                                witness={'root_id':root,'family':inp['family'],'object':key[1],
                                    'context_own_values':[s['values'][key[1]] for s in lab['context_states']],
                                    'identical_local_action_prefix':list(key[2]),'pair':[first,second]}
                                break
                        if witness:
                            break
    family=[ceiling(g) for g in family_groups]
    return {'mode':mode,'split':split,'family':dict(zip(['counter','toggle','flow'],family)),
        'cge_upper':float(np.mean([r['cge_upper'] for r in family])),
        'roots':len(root_possible),'roots_with_forced_contradiction':len(contradiction_roots),
        'all_branches_root_success_upper':float(np.mean(list(root_possible.values()))),
        'interpretation':'Optimistic within-root empirical input-equivalence ceiling; root/object identity is additionally granted. Independent outputs per equivalence class are allowed. No test-label oracle is used by an evaluated predictor.',
        'witness':witness}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='dev');a=p.parse_args()
    if a.split!='dev':
        assert (ROOT/'results'/'evaluation-freeze.json').exists()
    out=ROOT/'results'/'analysis';out.mkdir(exist_ok=True)
    result={mode:analyze(a.split,mode) for mode in ['local_incomplete','global_target_blind','complete_actions']}
    (out/f'information-ceiling-{a.split}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:{'cge_upper':v['cge_upper'],'contradiction_roots':v['roots_with_forced_contradiction'],
        'root_success_upper':v['all_branches_root_success_upper'],'witness':v['witness']} for k,v in result.items()}),flush=True)
