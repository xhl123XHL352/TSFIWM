"""CPU image-feedback MPC on actual Panda contact transitions.

Counter instruments only. Every trial replays calibration, then executes new
actions selected from images. Recorded open-loop branches are never executed
as substitutes for feedback control.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from PIL import Image
import torch
from calibrated_ocr import crop
from state_models import StatePredictor
from rule_baseline import forecast
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'cache'/'repos'/'parseq'))
from strhub.models.utils import create_model
torch.set_num_threads(4)


def distance(a,b):
    if a is None:
        return 100
    row=list(range(len(b)+1))
    for i,c in enumerate(a):
        nxt=[i+1]
        for j,d in enumerate(b):
            nxt.append(min(row[j+1]+1,nxt[-1]+1,row[j]+(c!=d)))
        row=nxt
    return row[-1]


def receive(process):
    while True:
        line=process.stdout.readline()
        if not line:
            raise RuntimeError('Our control worker ended without a response')
        try:
            return json.loads(line)
        except ValueError:
            continue


class Reader:
    def __init__(self):
        self.model=create_model('parseq',pretrained=False).eval()
        state=torch.load(ROOT/'models'/'parseq'/'parseq-bb5792a6.pt',map_location='cpu',weights_only=True)
        self.model.model.load_state_dict(state,strict=True)

    def read(self,paths):
        tasks=[(path,panel,field) for path in paths for panel in range(3) for field in ['id','value']]
        tensors=[]
        for path,panel,field in tasks:
            image=Image.open(ROOT/path).convert('RGB')
            word=crop(image,panel,field).resize((128,32),Image.Resampling.BICUBIC)
            tensors.append(np.asarray(word,dtype=np.float32).transpose(2,0,1)/127.5-1)
        with torch.inference_mode():
            probs=self.model(torch.as_tensor(np.stack(tensors))).softmax(-1)
        labels,confidence=self.model.tokenizer.decode(probs)
        out={p:{'ids':[None]*3,'values':[None]*3,'confidence':[0.0]*3} for p in paths}
        for (path,panel,field),text,conf in zip(tasks,labels,confidence):
            if field=='id':
                if len(text)==5 and text.isalnum():
                    out[path]['ids'][panel]=text
                    out[path]['confidence'][panel]=float(conf.prod())
            elif text.isdigit() and 0<=int(text)<=999:
                out[path]['values'][panel]=int(text)
        return out


def choose(model,row,goal):
    row={**row,**{k:row[k][-4:] for k in ['context','context_masks','context_actions','context_dt']}}
    plans=np.asarray(list(itertools.product([-1,0,1,2],repeat=4)))
    if model=='copy':
        predictions=np.repeat(np.asarray(row['context'][-1])[None,None],len(plans),axis=0)
    elif model=='rules':
        predictions=np.stack([forecast({**row,'actions':p,'durations':[1]*4}) for p in plans])
    else:
        n=len(plans)
        inputs={k:torch.as_tensor(np.repeat(np.asarray(row[k])[None],n,axis=0),
            dtype=torch.long if k in ['context_actions','family'] else torch.float32)
            for k in ['context','context_masks','context_actions','context_dt','family']}
        inputs.update(actions=torch.as_tensor(plans),durations=torch.ones((n,4)))
        with torch.inference_mode():
            predictions=model(**inputs)[0].numpy()
    trajectory=np.clip(np.rint(predictions),0,999)
    costs=np.abs(trajectory-np.asarray(goal)[None,None]).sum((1,2))+.05*(plans>=0).sum(1)
    index=int(np.argmin(costs))
    return int(plans[index,0]),plans[index].tolist(),float(costs[index])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--split',choices=['dev','test','test_shift'],required=True)
    p.add_argument('--limit',type=int)
    p.add_argument('--methods',nargs='+',default=['copy','rules','global_abs','global_relative','local_fullaction_relative'])
    p.add_argument('--seeds',nargs='+',type=int,default=[17,29,43])
    p.add_argument('--identity',choices=['wide','near'],default='wide')
    a=p.parse_args()
    if a.split!='dev' and not (ROOT/'results'/'control-freeze.json').exists():
        raise RuntimeError('Freeze feedback-control protocol before test trials')
    cfg=json.loads((ROOT/'source'/'panel-config.json').read_text())
    folders=[f for f in sorted((ROOT/'data'/'panel-v3'/a.split).iterdir())
        if json.loads((f/'inputs.json').read_text())['family']=='counter']
    if a.limit:
        folders=folders[:a.limit]
    near={}
    if a.identity=='near':
        near={r['root_id']:r['ids'] for line in (ROOT/'results'/'perception'/f'qwen-initial-views-{a.split}.jsonl').read_text().splitlines()
            if (r:=json.loads(line))}
    reader=Reader()
    started=time.monotonic()
    for method in a.methods:
        for seed in ([0] if method in ['copy','rules'] else a.seeds):
            name=f'{method}-seed{seed}-{a.identity}-{a.split}'
            destination=ROOT/'results'/'control-window4'/name
            destination.mkdir(parents=True,exist_ok=True)
            if (destination/'complete.json').exists():
                print(json.dumps({'preserved_complete':name}),flush=True)
                continue
            if (destination/'trials.jsonl').exists():
                raise RuntimeError('Partial control run requires inspection: '+name)
            if method in ['copy','rules']:
                model=method
            else:
                group='state-controls1000' if method=='local_fullaction_relative' else 'state-budget1000'
                path=ROOT/'results'/group/f'{method}-seed-{seed}'/'best.pt'
                frozen=json.loads((ROOT/'results'/'evaluation-freeze.json').read_text())
                assert hashlib.sha256(path.read_bytes()).hexdigest()==frozen['checkpoints'][str(path.relative_to(ROOT))]
                with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
                    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
                model=StatePredictor(method).eval();model.load_state_dict(checkpoint['model'],strict=True)
            with (destination/'trials.jsonl').open('w') as log,ThreadPoolExecutor(max_workers=12) as pool:
                for offset in range(0,len(folders),12):
                    group=folders[offset:offset+12]
                    workers=[]
                    trials=[]
                    try:
                        for folder in group:
                            rel=str((destination/folder.name).relative_to(ROOT))
                            worker=subprocess.Popen([sys.executable,str(ROOT/'source'/'panel_control_worker.py'),
                                '--split',a.split,'--root-id',folder.name,'--out',rel],
                                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=(destination/(folder.name+'.stderr.log')).open('w'),
                                text=True,bufsize=1,cwd=ROOT)
                            workers.append(worker)
                        initial=list(pool.map(receive,workers))
                        observations=reader.read([path for r in initial for path in r['context_images']])
                        for folder,item in zip(group,initial):
                            context=[];masks=[];identities=[None]*3;conf=[-1.]*3
                            for path in item['context_images']:
                                r=observations[path]
                                mask=[v is not None for v in r['values']]
                                values=[v if v is not None else (context[-1][i] if context else 0) for i,v in enumerate(r['values'])]
                                context.append(values);masks.append(mask)
                                for i in range(3):
                                    if r['confidence'][i]>conf[i] and r['ids'][i] is not None:
                                        identities[i],conf[i]=r['ids'][i],r['confidence'][i]
                            if a.identity=='near':
                                identities=near[folder.name]
                            # A task setter specifies a target identifier and desired display.
                            # This reads context only, never a recorded future branch.
                            setter=json.loads((folder/'labels.json').read_text())['context_states'][-1]
                            target=(int(folder.name.split('-')[-1])//3)%3
                            trials.append({'root_id':folder.name,'context':context,'context_masks':masks,
                                'context_actions':[-1,0,1,2],'context_dt':[0,1,1,1],'family':0,
                                'ids':identities,'confidence':conf,'initial_observed':context[-1].copy(),
                                'target_identifier':setter['ids'][target], 'desired_value':setter['values'][target]+6,
                                'decisions':[],'finished':False})
                        for step in range(12):
                            selected=[]
                            for i,trial in enumerate(trials):
                                if trial['finished']:
                                    continue
                                slot=min(range(3),key=lambda j:distance(trial['ids'][j],trial['target_identifier']))
                                goal=trial['initial_observed'].copy();goal[slot]=trial['desired_value']
                                if trial['context'][-1]==goal:
                                    trial['finished']=True
                                    continue
                                target,plan,cost=choose(model,trial,goal)
                                trial['decisions'].append({'step':step,'matched_slot':slot,'goal':goal,
                                    'requested_target':target,'four_step_plan':plan,'predicted_cost':cost})
                                workers[i].stdin.write(json.dumps({'command':'step','target':target})+'\n');workers[i].stdin.flush()
                                selected.append(i)
                            if not selected:
                                break
                            responses=list(pool.map(receive,[workers[i] for i in selected]))
                            readings=reader.read([r['image'] for r in responses])
                            for i,response in zip(selected,responses):
                                trial=trials[i];r=readings[response['image']]
                                trial['context_masks'].append([v is not None for v in r['values']])
                                trial['context'].append([v if v is not None else trial['context'][-1][j] for j,v in enumerate(r['values'])])
                                trial['context_actions'].append(response['requested_action']['target']);trial['context_dt'].append(1)
                                trial['decisions'][-1]['observation']=response['image']
                                trial['decisions'][-1]['reading']=r
                                if a.identity=='wide':
                                    for j in range(3):
                                        if r['ids'][j] is not None and r['confidence'][j]>trial['confidence'][j]:
                                            trial['ids'][j],trial['confidence'][j]=r['ids'][j],r['confidence'][j]
                        for worker in workers:
                            worker.stdin.write(json.dumps({'command':'close'})+'\n');worker.stdin.flush()
                        for worker in workers:
                            assert worker.wait(timeout=30)==0
                        # Audit states are evaluated only after all actions in the batch.
                        for trial in trials:
                            audit=json.loads((destination/trial['root_id']/'audit.json').read_text())
                            true_initial=audit[3]['state'];true_final=audit[-1]['state']
                            slot=true_initial['ids'].index(trial['target_identifier'])
                            goal=true_initial['values'].copy();goal[slot]=trial['desired_value']
                            trial['success']=true_final['values']==goal
                            trial['final_true_values']=true_final['values'];trial['true_goal']=goal
                            trial['steps']=len(trial['decisions'])
                            log.write(json.dumps(trial)+'\n');log.flush()
                    finally:
                        for worker in workers:
                            if worker.poll() is None:
                                worker.terminate();worker.wait(timeout=10)
                    print(json.dumps({'method':name,'roots_completed':min(offset+12,len(folders)),'roots':len(folders)}),flush=True)
            records=[json.loads(x) for x in (destination/'trials.jsonl').read_text().splitlines()]
            summary={'method':name,'roots':len(records),'success':sum(r['success'] for r in records)/len(records),
                'mean_steps':float(np.mean([r['steps'] for r in records])),'cpu_wall_elapsed_s':time.monotonic()-started,
                'scope':'Counter instruments; image feedback, fuzzy ID association, four-step cumulative L1 MPC +.05 per press, at most12 actions, desired reading +6; all non-target values preserved',
                'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
            (destination/'complete.json').write_text(json.dumps(summary,indent=2))
            print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
