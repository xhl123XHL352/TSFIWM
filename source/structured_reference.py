"""Public numerical rules plus an optional execution table learned from text.

The table's labels are inferred from observed training-number transitions,
never from button contact or hidden simulator parameters.
"""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def key(history,target):
    return ','.join(str(x) for x in ([-2,-2]+history)[-2:]+[int(target)])


def fit_table():
    from train_state import load_data
    counts={}
    for row in load_data('train'):
        values=np.asarray(row['context']+row['truth'])
        actions=row['context_actions']+row['actions']
        modes=np.zeros(3,dtype=int);history=[]
        for t in range(1,len(values)):
            target=actions[t];delta=values[t]-values[t-1]
            if target>=0:
                pressed=(delta[target]!=0) if row['family']!=2 else (int(delta[target]>0)!=modes[target])
                c=counts.setdefault(key(history,target),[0,0]);c[int(pressed)]+=1
                history.append(target)
            if row['family']==2:
                modes=(delta>0).astype(int)
    record={'counts':counts,'label_source':'Numeric differences only; counter/toggle change or flow gate-rate flip',
        'features':'Last two non-wait requested slots and current requested slot; no numbers, future contacts or physical state',
        'training_roots':600,'prediction':'Majority outcome per key; unseen keys assume success',
        'source':'panel-v3 train, same trajectories as neural references'}
    out=ROOT/'results'/'control-development';out.mkdir(exist_ok=True)
    (out/'execution-table.json').write_text(json.dumps(record,indent=2))
    print(json.dumps({'execution_keys':len(counts),'observed_transitions':sum(sum(v) for v in counts.values())}),flush=True)


def forecast(row,use_table=False):
    values=np.asarray(row['context'],dtype=float)
    family=row['family'];gains=np.full(3,2.);modes=np.zeros(3,dtype=int)
    observations=[[] for _ in range(3)]
    for t in range(1,len(values)):
        delta=values[t]-values[t-1];dt=row['context_dt'][t];target=row['context_actions'][t]
        if family in [0,1]:
            if target>=0 and abs(delta[target])>.5:
                observations[target].append(abs(delta[target]))
                if family==1:
                    modes[target]=int(delta[target]>0)
        else:
            for i in range(3):
                if delta[i]>.5:
                    observations[i].append(delta[i]/dt)
            modes=(delta>.5).astype(int)
    for i in range(3):
        if observations[i]:
            gains[i]=np.median(observations[i])
    counts=json.loads((ROOT/'results'/'control-development'/'execution-table.json').read_text())['counts'] if use_table else None
    history=[x for x in row['context_actions'] if x>=0]
    current=values[-1].copy();outputs=[]
    for target,dt in zip(row['actions'],row['durations']):
        pressed=target>=0
        if pressed and counts is not None:
            n=counts.get(key(history,target),[0,1]);pressed=n[1]>=n[0]
        if target>=0:
            history.append(int(target))
        if pressed:
            if family==0:
                current[target]+=gains[target]
            elif family==1:
                modes[target]=1-modes[target]
                current[target]+=gains[target]*(1 if modes[target] else -1)
            else:
                modes[target]=1-modes[target]
        if family==2:
            current+=modes*gains*dt
        current=np.clip(current,0,999);outputs.append(current.copy())
    return np.asarray(outputs)


if __name__=='__main__':
    fit_table()
