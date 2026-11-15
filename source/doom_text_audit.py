"""Train-only native HUD glyph templates, then inspect display/state semantics.

Templates use stable intervals of released training episode 0. Target-game
variables are never used by the reader. Templates are a fixed-font control,
not a general scene-text recognizer or a method contribution.
"""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
FIELDS=['ammo','health']


def regions(image):
    a=np.asarray(image.convert('RGB'),dtype=float)
    return [a[210:228,x+13*j:x+13*(j+1)] for x in [5,51] for j in range(3)]


def representation(patches):
    a=np.stack(patches).astype(np.float32)
    v=np.maximum(a[:,:,:,0]-np.maximum(a[:,:,:,1],a[:,:,:,2]),0)
    return (v/np.maximum(v.max((1,2),keepdims=True),1)).reshape(len(a),-1)


def build_templates():
    import pyarrow.parquet as pq
    raw=pq.read_table(ROOT/'data'/'doomshift'/'raw'/'arenas'/'ep_00000.parquet',
                      columns=FIELDS+['frame','deaths']).to_pydict()
    rng=np.random.default_rng(1961)
    order=rng.permutation(np.arange(4,len(raw['frame'])-4))
    pools={i:[] for i in range(11)}
    for t in order:
        if any(len(set(raw[f][t-4:t+5]))!=1 for f in FIELDS+['deaths']):
            continue
        values=[raw[f][t] for f in FIELDS]
        if any(v<0 or v>999 for v in values):
            continue
        targets=[10 if c==' ' else int(c) for v in values for c in f'{v:3d}']
        if all(len(pools[c])>=200 for c in set(targets)):
            continue
        features=representation(regions(Image.open(io.BytesIO(raw['frame'][t]))))
        for c,v in zip(targets,features):
            if len(pools[c])<200:
                pools[c].append(v)
        if all(len(v)>=200 for v in pools.values()):
            break
    assert all(pools.values()), {k:len(v) for k,v in pools.items()}
    templates=np.stack([np.median(pools[c],axis=0) for c in range(11)])
    out=ROOT/'results'/'doomshift'/'reader'
    out.mkdir(parents=True,exist_ok=True)
    np.save(out/'train-0-templates.npy',templates)
    manifest={'train_episode':0,'stable_radius':4,'examples_per_class':{k:len(v) for k,v in pools.items()},
        'boxes':'ammo x5+13*j, health x51+13*j, y210:228; 13x18 pixels per slot',
        'features':'positive R-max(G,B), per-slot maximum normalization',
        'classifier':'nearest median template, squared L2; class 10 is blank',
        'purpose':'fixed-font audit; requires stable-frame validation before use as transcription labels'}
    (out/'templates.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest),flush=True)
    return templates


def read(image, templates):
    v=representation(regions(image))
    d=((v[:,None]-templates[None])**2).mean(-1)
    indices=d.argmin(-1)
    chars=[' ' if i==10 else str(i) for i in indices]
    strings=[''.join(chars[k:k+3]).strip() for k in [0,3]]
    values=[int(s) if s.isdigit() else None for s in strings]
    return dict(zip(FIELDS,values)),dict(zip(FIELDS,strings)),float(d.min(-1).mean())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--build',action='store_true')
    p.add_argument('--evaluate',default='dev-native-16')
    args=p.parse_args()
    folder=ROOT/'results'/'doomshift'/'reader'
    templates=build_templates() if args.build else np.load(folder/'train-0-templates.npy')
    import pyarrow.parquet as pq
    raw=pq.read_table(ROOT/'data'/'doomshift'/'raw'/'arenas'/'ep_06000.parquet',columns=FIELDS+['frame','deaths']).to_pydict()
    scores={f:[] for f in FIELDS}
    stable={f:[] for f in FIELDS}
    discrepancies=[]
    readouts=[]
    for t in range(len(raw['frame'])):
        values,strings,d=read(Image.open(io.BytesIO(raw['frame'][t])),templates)
        readouts.append([values[f] if values[f] is not None else -10000 for f in FIELDS])
        for field in FIELDS:
            same=values[field]==raw[field][t]
            scores[field].append(same)
            if t>=4 and t<len(raw['frame'])-4 and len(set(raw[field][t-4:t+5]))==1:
                stable[field].append(same)
            if not same and len(discrepancies)<30:
                discrepancies.append({'row':t,'field':field,'visible_template':values[field],
                    'gamevar':raw[field][t],'distance':d,'gamevar_nearby':raw[field][max(0,t-3):t+4]})
    readouts=np.asarray(readouts,dtype=np.int32)
    np.save(folder/'dev-raw-readouts.npy',readouts)
    lag_matches={}
    for lag in range(-3,4):
        valid=np.arange(4,len(readouts)-4)
        lag_matches[str(lag)]={f:float(np.mean(readouts[valid,k]==np.asarray(raw[f])[valid+lag])) for k,f in enumerate(FIELDS)}
    audit={'episode':6000,'rows':len(raw['frame']), 'gamevar_relative_row_lag_matches':lag_matches,
        'field_match_gamevars':{f:float(np.mean(v)) for f,v in scores.items()},
        'stable_match_gamevars':{f:float(np.mean(v)) for f,v in stable.items()},
        'stable_counts':{f:len(v) for f,v in stable.items()},'discrepancy_examples':discrepancies}
    (folder/'dev-display-gamevar-audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps({k:v for k,v in audit.items() if k!='discrepancy_examples'}),flush=True)
    rows=[]
    evaluation=ROOT/'results'/'doomshift'/args.evaluate
    for line in (evaluation/'windows.jsonl').read_text().splitlines():
        r=json.loads(line)
        path=ROOT/r['images']
        r['readout']={}
        for condition in ['target','previous','predicted','reconstructed','copy-decoded']:
            values,strings,d=read(Image.open(path/f'{condition}.png'),templates)
            r['readout'][condition]={'values':values,'strings':strings,'distance':d}
        rows.append(r)
    (evaluation/'template-readout.json').write_text(json.dumps(rows,indent=2))
    for condition in ['predicted','reconstructed','copy-decoded','previous']:
        print(json.dumps({'condition':condition,'n':len(rows),
            'exact_vs_visible_target':{f:float(np.mean([r['readout'][condition]['values'][f]==r['readout']['target']['values'][f] for r in rows])) for f in FIELDS},
            'exact_vs_gamevars':{f:float(np.mean([r['readout'][condition]['values'][f]==r['target'][f] for r in rows])) for f in FIELDS}}),flush=True)


if __name__=='__main__':
    main()
