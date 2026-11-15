"""Audit every selected raw endpoint before accepting held-out label claims.

No generated prediction or model-correctness score is inspected. All windows
are retained; raw-visible and aligned simulator labels remain separate.
"""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
config=json.loads((ROOT/'source/doom-campaign-config.json').read_text())
out=ROOT/'results/analysis/doom-label-exceptions';out.mkdir(exist_ok=True)
failures=[];counts={};unique={}
for subset in ['development','test_seen','test_unseen']:
 n=0;matches=0;windows_count=0
 for item in config[subset]:
  for h in [1,16]:
   count=sum(config['horizons'][str(h)].values())
   path=ROOT/'results/doomshift/windows'/f"ep-{item['episode']:05}-h{h}-n{count}.json"
   manifest=json.loads(path.read_text());assert len(manifest['windows'])==count
   for w in manifest['windows']:
    windows_count+=1
    for role in ['previous','target']:
     endpoint=w[role];n+=1;matches+=bool(endpoint['match'])
     unique[(item['episode'],endpoint['row'])]=endpoint
     if endpoint['match']:continue
     name=f"{subset.replace('_','-')}-unet-ep{item['episode']:05}-h{h}-s{config['generation_seed']}"
     folder=ROOT/'results/doomshift'/name
     records=json.loads((folder/'template-readout.json').read_text())
     r=next(r for r in records if r['index']==w['index'])
     raw_path=ROOT/r['images']/(role+'.png')
     target=out/f"ep{item['episode']:05}-h{h}-i{w['index']}-{role}.png"
     if target.exists():assert target.read_bytes()==raw_path.read_bytes()
     else:shutil.copyfile(raw_path,target)
     meta=np.load(ROOT/'data/doomshift/latents/sd15'/manifest['latent_segment']/f"ep_{item['episode']:05}_meta.npz")
     t=endpoint['row'];nearby={field:[int(x) for x in meta[field][max(0,t-4):t+4]] for field in ['ammo','health']}
     failures.append({'subset':subset,'episode':item['episode'],'map':item['map'],'horizon':h,'index':w['index'],
      'role':role,**endpoint,'nearby_rows':list(range(max(0,t-4),min(len(meta['ammo']),t+4))),
      'nearby_gamevars':nearby,'image':target.relative_to(ROOT).as_posix(),
      'image_sha256':hashlib.sha256(raw_path.read_bytes()).hexdigest(),
      'raw_font_strings':r['readout'][role]['strings']})
 counts[subset]={'windows':windows_count,'raw_endpoint_occurrences':n,'aligned_matches':matches}
result={'scope':'Raw endpoints only; no generated image quality inspected; no windows removed or resampled.',
 'counts':counts,'unique_raw_frames':len(unique),'exceptions':failures}
(ROOT/'results/analysis/doom-raw-label-audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
