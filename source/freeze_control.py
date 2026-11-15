"""Freeze fresh goal scenes, sources and checkpoint identities before trials."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'results/control-freeze.json'
assert not path.exists(),'Preserve original control freeze'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sources=['goal_control.py','control_campaign.py','panel_control.py','panel_control_worker.py','structured_reference.py',
    'state_models.py','panel_env.py','calibrated_ocr.py','read_initial_views.py','control-config.json']
checkpoints={}
for group in ['state-budget1000','state-controls1000','state-contextaug1000','state-historyaug1000']:
    for p in sorted((ROOT/'results'/group).glob('*/best.pt')):
        assert (p.parent/'summary.json').exists()
        checkpoints[str(p.relative_to(ROOT))]=sha(p)
roots={};data={}
for split in ['control_dev','control_test','control_shift']:
    folders=sorted((ROOT/'data/panel-control-v1'/split).iterdir());assert len(folders)==30
    roots[split]=[p.name for p in folders]
    for folder in folders:
        for f in folder.glob('*'):
            if f.is_file():data[str(f.relative_to(ROOT))]=sha(f)
assert not any((ROOT/'results/control-goals').glob('*control_test/trials.jsonl'))
assert not any((ROOT/'results/control-goals').glob('*control_shift/trials.jsonl'))
record={'utc':datetime.now(timezone.utc).isoformat(),'roots':roots,'sources':{'source/'+p:sha(ROOT/'source'/p) for p in sources},
    'checkpoints':checkpoints,'data':data,
    'reader_sha256':sha(ROOT/'models/parseq/parseq-bb5792a6.pt'),
    'execution_table_sha256':sha(ROOT/'results/control-development/execution-table.json'),
    'protocol':'Four-action deadline; shuffled identifier/value goals; exhaustive requested-plan enumeration; endpoint L1 then cumulative L1 then presses then lexicographic tie order; dt1; no future contact or reference plans; two-step packets versus four-step open loop.',
    'planned_comparisons':'Original, rolling-window augmentation, full-history augmentation: local complete-action centered predictor, seeds17,29,43, two-step feedback. Full-history augmented predictor additionally open-loop. Rules and rules+train-derived execution table, both feedback conditions; copy. Near-view identity for full-history augmented and rules+table, two-step only.',
    'test_use':'Original forecast test set was previously evaluated; all control test scenes are new and have never been used for controller selection. Nine control-dev scenes informed the feedback-history diagnostic. Test contains30roots, shift30roots, ten per family.',
    'decision_scope':'Physical Panda primitives in a controlled simulator, not hardware validation; public rules references have stronger mechanism priors than learned references.'}
path.write_text(json.dumps(record,indent=2));print(json.dumps({'frozen':str(path),'checkpoints':len(checkpoints),'data_files':len(data)}),flush=True)
