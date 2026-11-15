"""Fresh reachable goal tasks, disjoint from every Panel forecasting root.

All three instrument families; four-action deadline; goal entries shuffled.
A withheld reference trajectory establishes reachability without entering
controller inputs. No model outcome is used to select a task.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pybullet as p
from PIL import Image
from panel_env import ROOT,PanelEnv


def config():
    base=json.loads((ROOT/'source'/'panel-config.json').read_text())
    base['version']='panel-control-v1'
    base['splits']={
        'control_dev':{'roots':30,'seed_start':900000,'value_min':15,'value_max':70,'styles':[0,1]},
        'control_test':{'roots':30,'seed_start':1000000,'value_min':15,'value_max':70,'styles':[0,1]},
        'control_shift':{'roots':30,'seed_start':1100000,'value_min':200,'value_max':900,'styles':[2]}}
    base['task']='Exact three-identifier/value goal in four primitives; reference plan withheld; two-action feedback packets'
    return base


def generate(split,index):
    cfg=config();s=cfg['splits'][split];seed=s['seed_start']+index
    root_id=f'{split}-{seed}'
    out=ROOT/'data'/cfg['version']/split/root_id
    if (out/'complete.json').exists():
        return json.loads((out/'complete.json').read_text())
    assert not out.exists(),'Partial control root requires inspection'
    out.mkdir(parents=True)
    env=PanelEnv(seed=seed,family=cfg['families'][index%3],style=s['styles'][index%len(s['styles'])])
    env._texture_prefix+='-goalset'
    env.values=np.random.default_rng(seed+cfg['seed']).integers(s['value_min'],s['value_max'],3).astype(float)
    env._refresh()
    states=[];context=[]
    try:
        states.append(env.state());env.image().save(out/'context-0.png')
        context.append({'image':'context-0.png','preceding_action':None})
        near=ROOT/'data'/'panel-initial-views-v1'/split/root_id
        near.mkdir(parents=True,exist_ok=True)
        for i,x in enumerate(env.xs):
            view=p.computeViewMatrix([0,.13,2.0],[x,.13,.672],[0,1,0])
            projection=p.computeProjectionMatrixFOV(15,1,.05,4)
            result=p.getCameraImage(384,384,view,projection,renderer=p.ER_TINY_RENDERER,
                shadow=1,physicsClientId=env.client)
            Image.fromarray(np.asarray(result[2],np.uint8)[...,:3]).save(near/f'panel-{i}.png')
        (near/'complete.json').write_text(json.dumps({'root_id':root_id,'images':[f'panel-{i}.png' for i in range(3)],
            'observations':3,'time':'initial, before calibration','camera_origin':[0,.13,2.0],
            'vertical_fov_degrees':15,'pixels_per_image':[384,384]},indent=2))
        for i,target in enumerate(cfg['calibration_targets']):
            event=env.step(target,dt=1)
            name=f'context-{i+1}.png';env.image().save(out/name)
            states.append(env.state());context.append({'image':name,'preceding_action':event['command']})
        saved=p.saveState(physicsClientId=env.client)
        snapshot=(env.values.copy(),env.modes.copy(),env.tick)
        rng=np.random.default_rng(seed+7000000)
        for attempt in range(64):
            p.restoreState(saved,physicsClientId=env.client)
            env.values,env.modes,env.tick=snapshot[0].copy(),snapshot[1].copy(),snapshot[2]
            plan=rng.choice([-1,0,1,2],4).tolist();events=[]
            for action in plan:
                events.append(env.step(action,dt=1))
            goal=env.state()
            if goal['values']!=states[-1]['values']:
                break
        else:
            raise RuntimeError('No changing reachable goal sampled within bound')
        entries=[{'id':identity,'value':value} for identity,value in zip(goal['ids'],goal['values'])]
        rng.shuffle(entries)
        inputs={'root_id':root_id,'family':env.family,'context':context,'goal_entries':entries,
            'deadline_actions':4,'feedback_packet_actions':2}
        labels={'context_states':states,'goal_state':goal,'reference_actions_for_audit_only':plan,
            'reference_events_for_audit_only':events,'resample_attempts':attempt,
            'hidden_gains_for_audit_only':env.gains.tolist()}
        (out/'inputs.json').write_text(json.dumps(inputs,indent=2))
        (out/'labels.json').write_text(json.dumps(labels,indent=2))
        record={'root_id':root_id,'family':env.family,'ids':env.ids,'goal_resamples':attempt,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'inputs_sha256':hashlib.sha256((out/'inputs.json').read_bytes()).hexdigest(),
            'labels_sha256':hashlib.sha256((out/'labels.json').read_bytes()).hexdigest()}
        (out/'complete.json').write_text(json.dumps(record,indent=2))
        return record
    finally:
        env.close()


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--split');q.add_argument('--worker-index',type=int)
    a=q.parse_args()
    if a.worker_index is not None:
        print(json.dumps(generate(a.split,a.worker_index)),flush=True)
    else:
        cfg=config();path=ROOT/'source'/'control-config.json'
        if path.exists():
            assert json.loads(path.read_text())==cfg
        path.write_text(json.dumps(cfg,indent=2))
        jobs=[(s,i) for s in cfg['splits'] for i in range(cfg['splits'][s]['roots'])]
        def worker(item):
            result=subprocess.run([sys.executable,str(Path(__file__)),'--split',item[0],
                '--worker-index',str(item[1])],capture_output=True,text=True,cwd=ROOT,timeout=180)
            if result.returncode:
                raise RuntimeError(result.stdout+result.stderr)
            return json.loads(result.stdout.strip().splitlines()[-1])
        records=[]
        with ThreadPoolExecutor(max_workers=12) as pool:
            for future in as_completed([pool.submit(worker,j) for j in jobs]):
                records.append(future.result())
                if len(records)%15==0:
                    print(json.dumps({'completed_roots':len(records),'total':len(jobs)}),flush=True)
        (ROOT/'data'/cfg['version']/'manifest.json').write_text(json.dumps({'records':records,'config':cfg},indent=2))
