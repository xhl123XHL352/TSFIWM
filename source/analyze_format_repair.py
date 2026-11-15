"""Exploratory syntax-only repair, preserving primary forecast API scores.

After inspecting six malformed development responses, insert one missing outer
array pair only when the entire response has exactly three IDs and four flat
three-number rows. No numbers or strings are corrected or inferred.
"""
from datetime import datetime,timezone
import hashlib,json,re
from pathlib import Path
import numpy as np
from analyze_results import truth_data,evaluate
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/vlm-format-repaired';OUT.mkdir(exist_ok=True)
protocol={'created_utc':datetime.now(timezone.utc).isoformat(),
 'status':'Post-hoc development diagnostic, specified after observing primary scores and six malformed transcript responses. Not a primary API metric or confirmatory prompt experiment.',
 'rule':'Only insert a missing outer pair of array brackets around exactly four comma-separated, flat three-number arrays, inside an otherwise complete object with exactly ids and values in that order. No token, number or identifier correction. No recovery from truncated analysis.',
 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
receipt=OUT/'protocol.json'
if receipt.exists():assert json.loads(receipt.read_text())['source_sha256']==protocol['source_sha256']
else:receipt.write_text(json.dumps(protocol,indent=2))

def repair(raw):
 match=re.fullmatch(r'\s*\{\s*"ids"\s*:\s*(\[[^\]]*\])\s*,\s*"values"\s*:\s*(.*?)\s*\}\s*',raw,re.S)
 if not match:return None
 body=match[2]
 if not re.fullmatch(r'\s*(?:\[[^\[\]{}]*\]\s*,\s*){3}\[[^\[\]{}]*\]\s*',body,re.S):return None
 try:
  identities=json.loads(match[1]);values=json.loads('['+body+']')
  if len(identities)!=3 or not all(x is None or isinstance(x,str) for x in identities):return None
  if len(values)!=4 or not all(len(v)==3 for v in values):return None
  numeric=[[float(x) for x in row] for row in values]
  if not all(0<=x<=999 for row in numeric for x in row):return None
  return {'values':numeric,'ids':identities,'parse_valid':True}
 except (TypeError,ValueError):return None

paths=[];counts={}
for stem in ['qwen-pixels-json768-dev','qwen-annotation-json768-dev']:
 original=ROOT/'results/vlm'/(stem+'.jsonl');rows=[json.loads(s) for s in original.read_text().splitlines()]
 assert len(rows)==480
 repaired=0
 for row in rows:
  if row['parse_valid']:continue
  updated=repair(row['raw'])
  if updated:row.update(updated);row['syntax_repaired']=True;repaired+=1
 target=OUT/(stem+'.jsonl')
 payload=''.join(json.dumps(r)+'\n' for r in rows)
 if target.exists():assert target.read_text()==payload,'Preserve different repair outputs'
 else:target.write_text(payload)
 paths.append((stem,original,target));counts[stem]=repaired

# Truth is first loaded after the fixed syntax-only export has been saved.
truth,previous,families,identities=truth_data('dev');roots=sorted(families)
rng=np.random.default_rng(11321)
draws=[rng.choice([i for i,r in enumerate(roots) if families[r]==f],
 size=(10000,sum(families[r]==f for r in roots)),replace=True) for f in range(3)]
results={}
for stem,original,path in paths:
 primary,pbs,_=evaluate(original,truth,previous,families,identities,draws)
 value,bs,_=evaluate(path,truth,previous,families,identities,draws)
 results[stem]={'repaired_responses':counts[stem],'unrepaired_invalid_fraction':value['parse_failure_rate'],
  'primary_cge':primary['balanced_accuracy'],'repaired_cge':value['balanced_accuracy'],
  'repaired_ci95':value['balanced_ci95'],'paired_change_ci95':np.quantile(bs-pbs,[.025,.975]).tolist(),
  'raw_sha256':hashlib.sha256(original.read_bytes()).hexdigest()}
output={'protocol':protocol,'split':'development only','roots':120,'results':results,
 'interpretation':'Lenient syntax sensitivity only. Primary raw API scores and files are unchanged. Inserting brackets does not supply unknown physical execution outcomes.'}
(ROOT/'results/analysis/vlm-syntax-repair.json').write_text(json.dumps(output,indent=2))
print(json.dumps(output,indent=2),flush=True)
