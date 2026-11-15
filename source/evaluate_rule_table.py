"""Post-hoc forecast application of the already frozen control reference.

No fitting or hyperparameter selection. Context-only annotation inputs use the
same loader as the original forecast references. Future labels are scored by a
separate evaluator, never passed to this reference.
"""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import numpy as np
from evaluate_panel import inputs
from structured_reference import forecast
from rule_baseline import forecast as original_rules
ROOT=Path(__file__).resolve().parents[1]
freeze=json.loads((ROOT/'results/control-freeze.json').read_text())
table=ROOT/'results/control-development/execution-table.json'
assert hashlib.sha256(table.read_bytes()).hexdigest()==freeze['execution_table_sha256']
source=ROOT/'source/structured_reference.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()==freeze['sources']['source/structured_reference.py']
out=ROOT/'results/posthoc-rule-table';out.mkdir(exist_ok=True)
protocol={'registered_utc':datetime.now(timezone.utc).isoformat(),
 'status':'Post-hoc application to already evaluated independent forecast splits; not a prespecified primary forecast condition.',
 'reference':'Unmodified structured_reference.forecast(use_table=True), previously frozen for independent physical control.',
 'table_sha256':hashlib.sha256(table.read_bytes()).hexdigest(),
 'reference_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'information':'Observed four-frame numerical context, public family, requested action history and future requested sequence. No future contact, gain/mode label or future image.',
 'prior':'Known numerical equations plus training-derived execution table; stronger mechanism prior than GRUs.'}
receipt=out/'protocol.json'
if receipt.exists():assert all(json.loads(receipt.read_text())[k]==v for k,v in protocol.items() if k!='registered_utc')
else:receipt.write_text(json.dumps(protocol,indent=2))
for split in ['test','test_shift']:
 rows=inputs(split,'annotation');records=[]
 for row in rows:
  assert np.array_equal(forecast(row,False),original_rules(row)),'Known-rule adaptation must preserve original forecasts'
  pred=forecast(row,True)
  records.append({'root_id':row['root_id'],'branch':row['branch'],'family':row['family'],
   'ids':row['ids'],'prediction':pred.tolist()})
 assert len(records)==720
 path=out/('rules-table-'+split+'.jsonl');data=''.join(json.dumps(r)+'\n' for r in records)
 if path.exists():assert path.read_text()==data,'Preserve different post-hoc predictions'
 else:path.write_text(data)
 print(json.dumps({'split':split,'branches':len(records),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}),flush=True)
