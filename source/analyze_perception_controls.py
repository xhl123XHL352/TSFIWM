"""Root-paired audit of native camera views versus same-prompt wide crops."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
folders=sorted((ROOT/'data/panel-v3/dev').iterdir());roots=[p.name for p in folders]
truth={p.name:json.loads((p/'labels.json').read_text())['context_states'][0] for p in folders}
rng=np.random.default_rng(20261113);samples=rng.integers(0,len(roots),(10000,len(roots)))
arrays={};results={}
for condition,stem in [('native_camera','qwen-initial-views'),('rectified_wide','qwen-initial-crops')]:
    records=[json.loads(s) for s in (ROOT/'results/perception'/f'{stem}-dev.jsonl').read_text().splitlines()]
    byroot={r['root_id']:r for r in records};assert len(records)==len(byroot)==len(roots) and set(byroot)==set(roots)
    ids=np.array([[a==b for a,b in zip(byroot[r]['ids'],truth[r]['ids'])] for r in roots])
    values=np.array([[a==b for a,b in zip(byroot[r]['values'],truth[r]['values'])] for r in roots])
    vectors={'identifier_fields':ids.mean(1),'all_identifiers':ids.all(1).astype(float),'numeric_fields':values.mean(1)}
    arrays[condition]=vectors
    results[condition]={k:{'exact':float(v.mean()),'ci95':np.quantile(v[samples].mean(1),[.025,.975]).tolist()} for k,v in vectors.items()}
differences={}
for metric in arrays['native_camera']:
    d=arrays['native_camera'][metric]-arrays['rectified_wide'][metric]
    differences[metric]={'native_minus_crops':float(d.mean()),'ci95':np.quantile(d[samples].mean(1),[.025,.975]).tolist()}
out={'roots':len(roots),'split':'development','unit':'root, paired bootstrap10000',
    'control':'Same frozen Qwen3.5-9B, identical three-image prompt and 384x384 image dimensions. Crop condition adds no observation. Native-camera condition adds calibrated views and changes the available visual evidence. Native pixel dimensions do not equate encoder FLOPs.',
    'conditions':results,'paired_differences':differences}
(ROOT/'results/analysis/perception-controls.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
