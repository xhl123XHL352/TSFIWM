"""Paired development-only prompt/format analysis, using saved responses."""
import hashlib,json
from pathlib import Path
import numpy as np
from analyze_results import truth_data,evaluate
ROOT=Path(__file__).resolve().parents[1]
truth,previous,families,identities=truth_data('dev')
roots=sorted(families);rng=np.random.default_rng(11321)
draws=[rng.choice([i for i,r in enumerate(roots) if families[r]==f],
 size=(10000,sum(families[r]==f for r in roots)),replace=True) for f in range(3)]
names=['internvl-pixels-dev','qwen-annotation-dev','qwen-pixels-batched8-dev',
       'qwen-pixels-json768-dev','qwen-annotation-json768-dev']
conditions={};samples={};sources={}
for name in names:
 path=ROOT/'results/vlm'/(name+'.jsonl')
 rows=[json.loads(s) for s in path.read_text().splitlines()]
 assert len(rows)==480 and len({(r['root_id'],r['branch']) for r in rows})==480
 result,bs,_=evaluate(path,truth,previous,families,identities,draws)
 assert result['roots']==120
 conditions[name]={'cge':result['balanced_accuracy'],'ci95':result['balanced_ci95'],
  'invalid_format':sum(not r['parse_valid'] for r in rows),'branches':480,
  'mean_response_characters':float(np.mean([len(r['raw']) for r in rows])),
  'numerical_exact':result['all_values_accuracy']}
 samples[name]=bs;sources[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
pairs={}
for left,right in [('qwen-pixels-json768-dev','qwen-annotation-json768-dev'),
 ('qwen-annotation-json768-dev','qwen-annotation-dev')]:
 diff=samples[left]-samples[right]
 pairs[left+' MINUS '+right]={'cge_difference':conditions[left]['cge']-conditions[right]['cge'],
  'ci95':np.quantile(diff,[.025,.975]).tolist()}
out={'split':'development only','roots':120,'unit':'family-stratified paired root bootstrap; same root draws across conditions',
 'conditions':conditions,'paired_differences':pairs,'sha256':sources,
 'interpretation':'Prompt and token ceiling change jointly in the JSON control. Images and transcripts expose different information. No numerical prompt selection, architecture-only comparison, or independent-test superiority claim.'}
(ROOT/'results/analysis/vlm-format-controls.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!='sha256'},indent=2),flush=True)
