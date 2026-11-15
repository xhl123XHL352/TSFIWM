"""Model-independent manifest/state-chain and retained numerical-array audits."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
freeze=json.loads((ROOT/'results/evaluation-freeze.json').read_text());parity=[]
for rel in freeze['checkpoints']:
 model=(ROOT/rel).parent;group=model.parent.name
 native=model/'dev-predictions.jsonl';export=ROOT/'results/panel-annotation'/(group+'--'+model.name+'-dev.jsonl')
 left=[json.loads(x) for x in native.read_text().splitlines()];right=[json.loads(x) for x in export.read_text().splitlines()]
 assert [(r['root_id'],r['branch']) for r in left]==[(r['root_id'],r['branch']) for r in right]
 x=np.asarray([r['prediction'] for r in left],dtype=np.float32);y=np.asarray([r['prediction'] for r in right],dtype=np.float32)
 assert x.tobytes()==y.tobytes(),rel
 parity.append({'checkpoint':rel,'records':len(left),'prediction_array_sha256':hashlib.sha256(x.tobytes()).hexdigest()})
sets={};n=0;calibration=0;families={}
for split in ['train','dev','test','test_shift']:
 ids=set();family_counts={f:0 for f in ['counter','toggle','flow']}
 for folder in sorted((ROOT/'data/panel-v3'/split).iterdir()):
  inp=json.loads((folder/'inputs.json').read_text());lab=json.loads((folder/'labels.json').read_text());done=json.loads((folder/'complete.json').read_text())
  assert done['inputs_sha256']==sha(folder/'inputs.json') and done['labels_sha256']==sha(folder/'labels.json')
  assert done['signature']=='6f024a1ad91124c691e9425302bb39daac830b42e4e96fcd7a227b64d7430356'
  family=inp['family'];family_counts[family]+=1;gain=lab['hidden_parameters_for_audit_only']['gains']
  states=lab['context_states'];assert len(states)==4 and states[0]['modes']==[0,0,0]
  assert len(set(states[0]['ids']))==3;ids.update(states[0]['ids'])
  def check(event,before,after):
   global n
   assert event['before']==before and event['after']==after
   v=before['values'].copy();m=before['modes'].copy();target=event['command']['target'];dt=event['command']['dt']
   assert target in [-1,0,1,2] and dt in [1,2,3]
   if event['pressed']:
    assert target>=0 and event['contact_count']>0 and event['min_depression_m']<-.003
    if family=='counter':v[target]+=gain[target]
    else:
     m[target]=1-m[target]
     if family=='toggle':v[target]+=gain[target]*(1 if m[target] else -1)
   if family=='flow':v=[x+mode*g*dt for x,mode,g in zip(v,m,gain)]
   assert after['values']==[min(999,max(0,x)) for x in v]
   assert after['modes']==m and after['tick']==before['tick']+dt and after['ids']==before['ids'];n+=1
  for i,event in enumerate(lab['calibration_events']):
   check(event,states[i],states[i+1]);assert event['pressed'];calibration+=1
  assert len(lab['branch_events'])==4
  for b,(events,future,branch) in enumerate(zip(lab['branch_events'],lab['future_states'],inp['branches'])):
   assert len(events)==len(future)==4 and branch['branch']==b
   chain=[states[-1]]+future
   for i,event in enumerate(events):
    assert event['command']==branch['actions'][i];check(event,chain[i],chain[i+1])
   assert branch['actions'][0]['target']==[-1,0,1,2][b]
   assert branch['actions'][1:]==inp['branches'][0]['actions'][1:]
  sets[split]=ids
 families[split]=family_counts
for i,s in enumerate(sets):
 for t in list(sets)[i+1:]:assert not sets[s]&sets[t],(s,t,'identity leakage')
reruns=[]
for old in (ROOT/'results/interrupted-control-runs').glob('*/*/trials.jsonl'):
 current=ROOT/'results/control-goals'/old.parent.name/'trials.jsonl'
 if not (current.parent/'complete.json').exists():continue
 before=[json.loads(x) for x in old.read_text().splitlines()];after={r['root_id']:r for x in current.read_text().splitlines() if (r:=json.loads(x))}
 fields=['root_id','ids','true_goal_by_slot','final_true_values','success','decisions']
 for row in before:
  assert all(row[k]==after[row['root_id']][k] for k in fields),('Interrupted/rerun trajectory changed',row['root_id'])
 reruns.append({'interrupted':str(old.relative_to(ROOT)),'rerun':str(current.relative_to(ROOT)),'retained_roots_compared':len(before),'comparison':'ids, goals, final physical values, success, all decisions and executed action packets'})
out={'development_prediction_array_parity':parity,'matched_checkpoints':len(parity),'audited_original_transitions':n,'successful_calibration_presses':calibration,'family_roots':families,'cross_split_identifier_intersections':0,'interrupted_rerun_checks':reruns}
(ROOT/'results/analysis/saved-evidence-validation.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='development_prediction_array_parity'},indent=2),flush=True)
