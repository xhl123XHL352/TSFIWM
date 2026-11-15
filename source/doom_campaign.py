"""One bounded native-WM evaluation batch; cache full unchanged public models."""
import argparse
import contextlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from gpu_accounting import ROOT,GpuLease

p=argparse.ArgumentParser()
p.add_argument('--subset',choices=['development','test_seen','test_unseen'],required=True)
p.add_argument('--backbone',choices=['unet','pixart'],required=True)
p.add_argument('--gpu-uuid',required=True)
p.add_argument('--start',type=int,default=0)
p.add_argument('--stop',type=int)
a=p.parse_args()
config=json.loads((ROOT/'source'/'doom-campaign-config.json').read_text())
if a.subset!='development' and not (ROOT/'results'/'doom-evaluation-freeze.json').exists():
    raise RuntimeError('Independent Doom campaign not frozen')
items=config[a.subset][a.start:a.stop]
with GpuLease(a.gpu_uuid,f'doom-campaign-{a.backbone}-{a.subset}-{a.start}',max_memory_gib=8):
    from doom_evaluate import evaluate
    for item in items:
        for horizon,counts in config['horizons'].items():
            horizon=int(horizon)
            n=sum(counts.values())
            name=f"{a.subset.replace('_','-')}-{a.backbone}-ep{item['episode']:05}-h{horizon}-s{config['generation_seed']}"
            output=ROOT/'results'/'doomshift'/name
            if (output/'complete.json').exists():
                print(json.dumps({'preserved_complete':name}),flush=True)
                continue
            output.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,str(ROOT/'source'/'doom_prepare_windows.py'),
                '--episode',str(item['episode']),'--horizon',str(horizon),
                '--raw-segment',item['raw_segment'],'--latent-segment',item['latent_segment'],
                '--counts',str(counts['health']),str(counts['ammo']),str(counts['unchanged'])]
            subprocess.run(command,check=True,cwd=ROOT,timeout=180)
            args=SimpleNamespace(episode=item['episode'],horizon=horizon,backbone=a.backbone,
                raw_segment=item['raw_segment'],latent_segment=item['latent_segment'],
                windows=0,seed=config['generation_seed'],name=name,tuned_decoder=True,audit_only=False,
                windows_file=str(ROOT/'results'/'doomshift'/'windows'/f"ep-{item['episode']:05}-h{horizon}-n{n}.json"))
            with (output/'execution.log').open('w') as log,contextlib.redirect_stdout(log):
                evaluate(args)
            subprocess.run([sys.executable,str(ROOT/'source'/'doom_score.py'),'--name',name],
                check=True,cwd=ROOT,stdout=subprocess.DEVNULL,timeout=180)
            print(json.dumps({'completed':name,**json.loads((output/'complete.json').read_text())}),flush=True)
