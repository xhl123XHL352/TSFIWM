"""Freeze model identity, selection, and independent Panel evaluation before scoring."""
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'results'/'evaluation-freeze.json'
if out.exists():
    raise RuntimeError('Protocol already frozen; inspect rather than silently refreeze')
checkpoints={}
for group in ['state-budget100','state-budget1000','state-controls1000']:
    for path in sorted((ROOT/'results'/group).glob('*/best.pt')):
        assert (path.parent/'summary.json').exists(),'Wait for all training runs to finish'
        checkpoints[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
assert len(checkpoints)==42,len(checkpoints)
sources=['state_models.py','train_state.py','evaluate_panel.py','analyze_results.py',
    'calibrated_ocr.py','read_initial_views.py','panel_initial_views.py','vlm_forecast.py',
    'rule_baseline.py','panel_data.py','panel_env.py','panel-config.json']
source_hashes={s:hashlib.sha256((ROOT/'source'/s).read_bytes()).hexdigest() for s in sources}
# Identity overlap is a dataset audit, not a numerical test outcome.
seen=set()
for split in ['train','dev','test','test_shift']:
    current=set()
    for path in (ROOT/'data'/'panel-v3'/split).glob('*/complete.json'):
        current.update(json.loads(path.read_text())['ids'])
    assert not seen.intersection(current),'Cross-split identifier collision requires separate audit'
    seen.update(current)
record={'frozen_utc':datetime.now(timezone.utc).isoformat(),'dataset':'panel-v3',
    'test_roots':180,'test_shift_roots':180,'seeds':[17,29,43],
    'checkpoints':checkpoints,'source_sha256':source_hashes,
    'selection':'Every variant and seed retained; best development CGE checkpoint at budget100 or1000; no test model/hyperparameter selection',
    'primary_comparisons':['same-budget absolute vs residual vs centered inputs',
        'local relative vs local relative with complete addressed-action history',
        'global relative with vs without target commands',
        'actual PARSeq numeric inputs vs context annotations'],
    'primary_metric':'CGE: mean over families sqrt(changed exact accuracy * unchanged exact accuracy)',
    'additional_metrics':['numeric exact accuracy','MAE','all48-values root success',
        'three-identifier exact accuracy','joint numeric-and-identity root success'],
    'statistics':'10000 paired root bootstraps, stratified by family; seeds also reported individually and as mean/range',
    'reader':'Frozen PARSeq base, common calibrated regions; IDs: most confident valid5-alnum context read; unknown numbers forward-fill with observation mask0',
    'near_view_condition':'Frozen Qwen3.5 initial3-view identity read, separately report observation cost; no future input',
    'procedural_note':'PARSeq test pixel-only readout was precomputed after reader/prompt/regions selection but before this final all-checkpoint hash freeze. No test recognition scores or prediction outcomes were inspected before freezing; this precomputation did not alter any reader or model selection rule.',
    'direct_vlm':'Frozen Qwen3.5 and InternVL3.5, greedy, public qualitative rules, four context images or explicit annotation condition; no future contacts'}
out.write_text(json.dumps(record,indent=2))
print(json.dumps({'frozen':str(out.relative_to(ROOT)),'checkpoints':len(checkpoints),'cross_split_id_overlap':0}),flush=True)
