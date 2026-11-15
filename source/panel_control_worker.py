"""Task-owned physical simulator process with image-only controller responses."""
import argparse
import json
import sys
import numpy as np
from panel_env import ROOT,PanelEnv
p=argparse.ArgumentParser()
p.add_argument('--split',required=True)
p.add_argument('--root-id',required=True)
p.add_argument('--out',required=True)
a=p.parse_args()
out=(ROOT/a.out).resolve()
assert ROOT in out.parents
out.mkdir(parents=True,exist_ok=True)
control_goals=a.split.startswith('control_')
config=json.loads((ROOT/'source'/('control-config.json' if control_goals else 'panel-config.json')).read_text())
spec=config['splits'][a.split]
seed=int(a.root_id.split('-')[-1])
index=seed-spec['seed_start']
family=config['families'][index%3]
env=PanelEnv(seed=seed,family=family,style=spec['styles'][index%len(spec['styles'])])
env._texture_prefix+='-control-'+str(out.parent.name)
env.values=np.random.default_rng(seed+config['seed']).integers(spec['value_min'],spec['value_max'],3).astype(float)
env._refresh()
audit=[]
try:
    images=[]
    image=out/'context-0.png';env.image().save(image);images.append(str(image.relative_to(ROOT)))
    audit.append({'state':env.state(),'event':None})
    for t,target in enumerate(config['calibration_targets']):
        event=env.step(target,dt=1)
        image=out/f'context-{t+1}.png';env.image().save(image);images.append(str(image.relative_to(ROOT)))
        audit.append({'state':env.state(),'event':event})
    original=json.loads((ROOT/'data'/('panel-control-v1' if control_goals else 'panel-v3')/a.split/a.root_id/'labels.json').read_text())['context_states']
    assert [x['state'] for x in audit]==original,'Physical calibration replay differs from dataset'
    print(json.dumps({'context_images':images,'root_id':a.root_id}),flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        if request['command']=='close':
            break
        assert request['command']=='step'
        targets=request.get('targets',[request.get('target')])
        assert 1<=len(targets)<=4 and all(t in [-1,0,1,2] for t in targets)
        paths=[];commands=[]
        for target in targets:
            event=env.step(target,dt=1)
            image=out/f'control-{len(audit)-4}.png';env.image().save(image)
            paths.append(str(image.relative_to(ROOT)));commands.append({'target':target,'dt':1})
            audit.append({'state':env.state(),'event':event})
        # Contact and true text state never enter the controller response.
        print(json.dumps({'image':paths[-1],'requested_action':commands[-1],
            'images':paths,'requested_actions':commands}),flush=True)
finally:
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    env.close()
