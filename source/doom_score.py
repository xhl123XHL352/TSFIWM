"""Independent text-state readout scores; does not import a world model."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from doom_text_audit import read,FIELDS
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--name',required=True)
a=p.parse_args()
out=ROOT/'results'/'doomshift'/a.name
assert (out/'complete.json').exists(),'Wait for every specified window'
protocol=json.loads((out/'protocol.json').read_text())
templates=np.load(ROOT/'results'/'doomshift'/'reader'/'train-0-templates.npy')
records=[json.loads(x) for x in (out/'windows.jsonl').read_text().splitlines()]
assert [r['index'] for r in records]==protocol['indices']
conditions=['previous','predicted','reconstructed','copy-decoded']
if (ROOT/records[0]['images']/'predicted-tuned.png').exists():
    conditions+=['predicted-tuned','reconstructed-tuned','copy-decoded-tuned']
for r in records:
    r['readout']={}
    for condition in ['target']+conditions:
        values,strings,d=read(Image.open(ROOT/r['images']/f'{condition}.png'),templates)
        r['readout'][condition]={'values':values,'strings':strings,'template_distance':d}
    reference=np.asarray(Image.open(ROOT/r['images']/'target.png').convert('RGB'),dtype=float)/255
    r['pixel_metrics']={}
    for condition in conditions:
        image=np.asarray(Image.open(ROOT/r['images']/f'{condition}.png').convert('RGB'),dtype=float)/255
        mse=float(np.mean((image[208:]-reference[208:])**2))
        r['pixel_metrics'][condition]={'hud_mse':mse,'hud_psnr':float(-10*np.log10(max(mse,1e-10)))}
scores={}
for condition in conditions:
    byfield={}
    for field in FIELDS:
        truth=[r['readout']['target']['values'][field] for r in records]
        pred=[r['readout'][condition]['values'][field] for r in records]
        previous=[r['readout']['previous']['values'][field] for r in records]
        correct=np.asarray([x==y for x,y in zip(pred,truth)])
        changed=np.asarray([x!=y for x,y in zip(previous,truth)])
        changed_acc=float(correct[changed].mean()) if changed.any() else None
        kept_acc=float(correct[~changed].mean()) if (~changed).any() else None
        byfield[field]={'n_changed':int(changed.sum()),'n_unchanged':int((~changed).sum()),
            'changed_exact':changed_acc,'unchanged_exact':kept_acc,
            'cge':float(np.sqrt(changed_acc*kept_acc)) if changed_acc is not None and kept_acc is not None else None,
            'exact':float(correct.mean())}
    scores[condition]={'fields':byfield,'hud_psnr':float(np.mean([r['pixel_metrics'][condition]['hud_psnr'] for r in records]))}
(out/'template-readout.json').write_text(json.dumps(records,indent=2))
result={'windows':len(records),'metric':'Exact numeric recovery by frozen train0 native-font templates, relative to independently read raw HUD; reader-dependent application metric',
    'unit':'one episode; intervals require multi-episode evaluation','scores':scores}
(out/'text-metrics.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
