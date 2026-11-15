"""Frozen-font, visibility-aligned development window selection for Doom HUDs."""
import argparse
import io
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'cache'/'repos'/'doomshift'))
from doom_data import TicWindowDataset
from doom_text_audit import read,FIELDS
import pyarrow.parquet as pq

p=argparse.ArgumentParser()
p.add_argument('--episode',type=int,default=6000)
p.add_argument('--horizon',type=int,default=1)
p.add_argument('--latent-segment',default='val')
p.add_argument('--raw-segment',default='arenas')
p.add_argument('--seed',type=int,default=1911)
p.add_argument('--counts',type=int,nargs=3,default=[12,12,8])
a=p.parse_args()
if not 6000<=a.episode<6100 and not (ROOT/'results'/'doom-evaluation-freeze.json').exists():
    raise RuntimeError('Freeze episode list, selection and label semantics before independent windows')
ds=TicWindowDataset(ROOT/'data'/'doomshift'/'latents'/'sd15'/a.latent_segment,
    episode_ids=[a.episode],context_frames=32,horizon=a.horizon,with_horizon=True,action_history=32)
meta=np.load(ROOT/'data'/'doomshift'/'latents'/'sd15'/a.latent_segment/f'ep_{a.episode:05}_meta.npz')
raw=pq.read_table(ROOT/'data'/'doomshift'/'raw'/a.raw_segment/f'ep_{a.episode:05}.parquet',columns=['frame']).to_pydict()['frame']
templates=np.load(ROOT/'results'/'doomshift'/'reader'/'train-0-templates.npy')
# Display is one rendered tic behind game variables in the development audit.
# Selection uses aligned values, and every scored raw frame is independently read.
strata={'health':[],'ammo':[],'unchanged':[]}
for idx in range(len(ds)):
    _,s=ds.locate(idx)
    before=s+30;target=s+30+a.horizon
    key='health' if meta['health'][target]!=meta['health'][before] else 'ammo' if meta['ammo'][target]!=meta['ammo'][before] else 'unchanged'
    strata[key].append(idx)
rng=np.random.default_rng(a.seed+a.episode+a.horizon)
selected=sorted(int(v) for key,n in zip(['health','ammo','unchanged'],a.counts)
    for v in rng.choice(strata[key],size=min(n,len(strata[key])),replace=False))
rows=[]
for idx in selected:
    _,s=ds.locate(idx)
    targets={}
    for role,t in [('previous',s+31),('target',s+31+a.horizon)]:
        values,strings,d=read(Image.open(io.BytesIO(raw[t])),templates)
        expected={f:int(meta[f][t-1]) for f in FIELDS}
        targets[role]={'visible':values,'aligned_gamevars':expected,
            'match':values==expected,'reader_distance':d,'row':t}
    rows.append({'index':idx,**targets})
out=ROOT/'results'/'doomshift'/'windows'
out.mkdir(parents=True,exist_ok=True)
path=out/f'ep-{a.episode:05}-h{a.horizon}-n{sum(a.counts)}.json'
record={'episode':a.episode,'horizon':a.horizon,'indices':selected,'raw_segment':a.raw_segment,
    'latent_segment':a.latent_segment,'sample_seed':a.seed,
    'selection':dict(zip(['health changes','ammo-only changes','unchanged'],a.counts)),
    'available_strata':{k:len(v) for k,v in strata.items()},'windows':rows,'dataset_summary':ds.summary}
if path.exists() and json.loads(path.read_text())!=record:
    raise RuntimeError('Window selection differs from preserved protocol')
path.write_text(json.dumps(record,indent=2))
print(json.dumps({'windows_file':str(path.relative_to(ROOT)),'windows':len(selected),
    'available_strata':record['available_strata'],'label_agreement':sum(r[k]['match'] for r in rows for k in ['previous','target'])/(2*len(rows))}),flush=True)
