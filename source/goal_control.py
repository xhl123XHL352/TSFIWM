"""Fresh exact-text goal planning with four actions and delayed feedback.

Goal entries are part of the task instruction, in a shuffled order. The
withheld reference plan, future pixels, contacts and modes never enter a
controller. Intermediate images are delivered only after a whole packet.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from panel_control import Reader,distance,receive,torch,ROOT
from state_models import StatePredictor
from structured_reference import forecast


def associate(ids,entries):
    permutation=min(itertools.permutations(range(3)),
        key=lambda perm:sum(distance(ids[i],entries[perm[i]]['id']) for i in range(3)))
    return [entries[j]['value'] for j in permutation],list(permutation)


def plan(model,row,goal,horizon,history_mode):
    if history_mode=='window4':
        row={**row,**{k:row[k][-4:] for k in ['context','context_masks','context_actions','context_dt']}}
    plans=np.asarray(list(itertools.product([-1,0,1,2],repeat=horizon)))
    n=len(plans)
    if isinstance(model,str):
        if model=='copy':
            predictions=np.repeat(np.repeat(np.asarray(row['context'][-1])[None,None],n,0),horizon,1)
        else:
            predictions=np.stack([forecast({**row,'actions':p,'durations':[1]*horizon},use_table=model=='rules_table') for p in plans])
    else:
        tensors={k:torch.as_tensor(np.repeat(np.asarray(row[k])[None],n,0),
            dtype=torch.long if k in ['context_actions','family'] else torch.float32)
            for k in ['context','context_masks','context_actions','context_dt','family']}
        tensors.update(actions=torch.as_tensor(plans),durations=torch.ones((n,horizon)))
        with torch.inference_mode():
            predictions=model(**tensors)[0].numpy()
    rounded=np.clip(np.rint(predictions),0,999)
    endpoint=np.abs(rounded[:,-1]-np.asarray(goal)).sum(1)
    cumulative=np.abs(rounded-np.asarray(goal)[None,None]).sum((1,2))
    presses=(plans>=0).sum(1)
    choice=int(np.lexsort((np.arange(n),presses,cumulative,endpoint))[0])
    return plans[choice].tolist(),predictions[choice].tolist(),[float(endpoint[choice]),float(cumulative[choice]),int(presses[choice])]


def update(trial,readout,path,action):
    r=readout[path]
    trial['context_masks'].append([v is not None for v in r['values']])
    trial['context'].append([v if v is not None else (trial['context'][-1][i] if trial['context'] else 0) for i,v in enumerate(r['values'])])
    trial['context_actions'].append(action);trial['context_dt'].append(int(action!=-2))
    if trial['identity']=='wide':
        for i in range(3):
            if r['ids'][i] is not None and r['confidence'][i]>trial['confidence'][i]:
                trial['ids'][i],trial['confidence'][i]=r['ids'][i],r['confidence'][i]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--split',choices=['control_dev','control_test','control_shift'],required=True)
    p.add_argument('--methods',nargs='+',default=['copy','rules','rules_table','global_abs','global_relative','local_fullaction_relative'])
    p.add_argument('--seeds',type=int,nargs='+',default=[17,29,43])
    p.add_argument('--training',choices=['original','window_aug','history_aug'],default='original')
    p.add_argument('--history',choices=['full','window4'],default='full')
    p.add_argument('--feedback',choices=['two_step','open_loop'],default='two_step')
    p.add_argument('--identity',choices=['wide','near'],default='wide')
    p.add_argument('--limit',type=int)
    a=p.parse_args()
    frozen=None
    if a.split!='control_dev':
        assert (ROOT/'results'/'control-freeze.json').exists(),'Freeze fresh goal-control evaluation'
        frozen=json.loads((ROOT/'results'/'control-freeze.json').read_text())
        assert a.limit is None,'Independent evaluation uses every frozen root'
        for path,digest in frozen['sources'].items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
        assert hashlib.sha256((ROOT/'models/parseq/parseq-bb5792a6.pt').read_bytes()).hexdigest()==frozen['reader_sha256']
        assert hashlib.sha256((ROOT/'results/control-development/execution-table.json').read_bytes()).hexdigest()==frozen['execution_table_sha256']
    folders=sorted((ROOT/'data'/'panel-control-v1'/a.split).iterdir())
    if a.limit:
        folders=folders[:a.limit]
    assert all((f/'complete.json').exists() for f in folders)
    if frozen:
        assert [f.name for f in folders]==frozen['roots'][a.split]
    near={}
    if a.identity=='near':
        near={r['root_id']:r['ids'] for line in (ROOT/'results'/'perception'/f'qwen-initial-views-{a.split}.jsonl').read_text().splitlines() if (r:=json.loads(line))}
        assert all(f.name in near for f in folders)
    reader=Reader()
    for method in a.methods:
        for seed in ([0] if method in ['copy','rules','rules_table'] else a.seeds):
            name=f'{method}-seed{seed}-{a.training}-{a.history}-{a.feedback}-{a.identity}-{a.split}'
            out=ROOT/'results'/'control-goals'/name;out.mkdir(parents=True,exist_ok=True)
            if (out/'complete.json').exists():
                assert json.loads((out/'complete.json').read_text())['roots']==len(folders),'Existing run has a different root count'
                print(json.dumps({'preserved_complete':name}),flush=True);continue
            assert not (out/'trials.jsonl').exists(),'Inspect partial goal-control run'
            (out/'protocol.json').write_text(json.dumps({'args':vars(a),'root_ids':[f.name for f in folders],
                'freeze_sha256':hashlib.sha256((ROOT/'results/control-freeze.json').read_bytes()).hexdigest() if frozen else None},indent=2))
            checkpoint_path=None
            if method in ['copy','rules','rules_table']:
                model=method
            else:
                group=({'window_aug':'state-contextaug1000','history_aug':'state-historyaug1000'}.get(a.training)
                    or ('state-controls1000' if method=='local_fullaction_relative' else 'state-budget1000'))
                checkpoint_path=ROOT/'results'/group/f'{method}-seed-{seed}'/'best.pt'
                assert (checkpoint_path.parent/'summary.json').exists(),'Wait for completed training'
                if a.split!='control_dev':
                    frozen=json.loads((ROOT/'results'/'control-freeze.json').read_text())
                    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()==frozen['checkpoints'][str(checkpoint_path.relative_to(ROOT))]
                with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
                    ck=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
                model=StatePredictor(method).eval();model.load_state_dict(ck['model'],strict=True)
            start=time.monotonic()
            with (out/'trials.jsonl').open('w') as log,ThreadPoolExecutor(max_workers=12) as pool:
                for offset in range(0,len(folders),12):
                    subset=folders[offset:offset+12];workers=[];trials=[]
                    try:
                        for folder in subset:
                            worker=subprocess.Popen([sys.executable,str(ROOT/'source'/'panel_control_worker.py'),
                                '--split',a.split,'--root-id',folder.name,'--out',str((out/folder.name).relative_to(ROOT))],
                                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=(out/(folder.name+'.stderr.log')).open('w'),
                                text=True,bufsize=1,cwd=ROOT)
                            workers.append(worker)
                        initial=list(pool.map(receive,workers))
                        readout=reader.read([path for item in initial for path in item['context_images']])
                        for folder,item in zip(subset,initial):
                            inp=json.loads((folder/'inputs.json').read_text())
                            trial={'root_id':folder.name,'family':['counter','toggle','flow'].index(inp['family']),
                                'goal_entries':inp['goal_entries'],'identity':a.identity,'ids':[None]*3,'confidence':[-1.]*3,
                                'context':[],'context_masks':[],'context_actions':[],'context_dt':[],'decisions':[]}
                            for t,path in enumerate(item['context_images']):
                                update(trial,readout,path,-2 if t==0 else t-1)
                            trial['context_actions'][0]=-1
                            if a.identity=='near':
                                trial['ids']=near[folder.name]
                            trials.append(trial)
                        used=0;packet=4 if a.feedback=='open_loop' else 2
                        while used<4:
                            for i,trial in enumerate(trials):
                                goal,assignment=associate(trial['ids'],trial['goal_entries'])
                                selected,predicted,cost=plan(model,trial,goal,4-used,a.history)
                                actions=selected[:packet]
                                trial['decisions'].append({'actions_used':used,'goal_by_slot':goal,'goal_assignment':assignment,
                                    'plan':selected,'predicted_states':predicted,'lexicographic_cost':cost,'executed_packet':actions})
                                workers[i].stdin.write(json.dumps({'command':'step','targets':actions})+'\n');workers[i].stdin.flush()
                            responses=list(pool.map(receive,workers))
                            readout=reader.read([path for item in responses for path in item['images']])
                            for trial,response in zip(trials,responses):
                                for path,action in zip(response['images'],response['requested_actions']):
                                    update(trial,readout,path,action['target'])
                                trial['decisions'][-1]['delivered_images']=response['images']
                                trial['decisions'][-1]['delivered_readouts']=[readout[path] for path in response['images']]
                            used+=packet
                        for worker in workers:
                            worker.stdin.write(json.dumps({'command':'close'})+'\n');worker.stdin.flush()
                        for worker in workers:
                            assert worker.wait(timeout=30)==0
                        for trial in trials:
                            audit=json.loads((out/trial['root_id']/'audit.json').read_text())
                            state=audit[-1]['state'];desired={e['id']:e['value'] for e in trial['goal_entries']}
                            goal=[desired[identity] for identity in state['ids']]
                            trial['success']=state['values']==goal
                            trial['final_true_values']=state['values'];trial['true_goal_by_slot']=goal
                            trial['absolute_goal_error']=sum(abs(x-y) for x,y in zip(state['values'],goal))
                            log.write(json.dumps(trial)+'\n');log.flush()
                    finally:
                        for worker in workers:
                            if worker.poll() is None:
                                worker.terminate();worker.wait(timeout=10)
                    print(json.dumps({'run':name,'roots_completed':min(offset+12,len(folders))}),flush=True)
            records=[json.loads(s) for s in (out/'trials.jsonl').read_text().splitlines()]
            summary={'run':name,'config':vars(a),'roots':len(records),'success':float(np.mean([r['success'] for r in records])),
                'by_family':{f:float(np.mean([r['success'] for r in records if r['family']==i])) for i,f in enumerate(['counter','toggle','flow'])},
                'mae_goal':float(np.mean([r['absolute_goal_error']/3 for r in records])),
                'cpu_wall_elapsed_s':time.monotonic()-start,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'checkpoint':str(checkpoint_path.relative_to(ROOT)) if checkpoint_path else None,
                'input':'Past images, requested commands, family and shuffled desired ID/value entries; no reference plan or future contacts'}
            (out/'complete.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
