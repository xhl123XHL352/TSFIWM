"""Register verified raw timing exceptions; do not change any scored label.

Seven raw PNGs were visually checked by the AI assistant before viewing their
generated-image correctness scores. This is not independent human annotation.
"""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
audit=ROOT/'results/analysis/doom-raw-label-audit.json'
data=json.loads(audit.read_text())
expected={(1303,'previous'):95,(1527,'previous'):82,(1878,'target'):51,
 (2537,'previous'):85,(3497,'previous'):56,(4354,'previous'):59,(4514,'previous'):20}
assert len(data['exceptions'])==len(expected)==7
entries=[]
for r in data['exceptions']:
 assert (r['episode'],r['map'],r['horizon'])==(37,16,1)
 assert r['visible']['health']==expected[(r['index'],r['role'])]
 assert r['reader_distance']==0 and r['visible']['ammo']==r['aligned_gamevars']['ammo']
 assert r['visible']['health']!=r['aligned_gamevars']['health']
 offset=r['nearby_rows'].index(r['row'])
 assert all(r['visible'][f]==r['nearby_gamevars'][f][offset] for f in ['ammo','health'])
 assert hashlib.sha256((ROOT/r['image']).read_bytes()).hexdigest()==r['image_sha256']
 entries.append({k:r[k] for k in ['episode','map','horizon','index','role','row','visible','aligned_gamevars','image','image_sha256']})
out={'registered_utc':datetime.now(timezone.utc).isoformat(),
 'kind':'Post-freeze raw-label alignment audit amendment; fixed-font readouts, selected windows, model predictions and all images unchanged.',
 'evidence':'All seven fixed-reader glyph matches have zero distance and equal same-tic game variables. The AI assistant visually checked each retained raw PNG; no independent human annotation is claimed. Generated correctness scores were not used to accept or reject windows.',
 'scoring':'Retain every window and original fixed-font visible label; permit only these exact non-null alignment exceptions. Unlisted or changed mismatches remain fatal.',
 'raw_audit_sha256':hashlib.sha256(audit.read_bytes()).hexdigest(),'entries':entries}
path=ROOT/'results/analysis/doom-visible-label-verification.json'
if path.exists():
 previous=json.loads(path.read_text());assert all(previous[k]==out[k] for k in out if k!='registered_utc')
else:path.write_text(json.dumps(out,indent=2))
print(json.dumps({'registered_timing_exceptions':len(entries),'label_changes':0,'excluded_windows':0,'registry':path.relative_to(ROOT).as_posix()}),flush=True)
