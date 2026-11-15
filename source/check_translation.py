"""CPU counterfactual check of trained numeric representation, not visual OCR.

Uses already evaluated ordinary test contexts, shifts all three observed
readings by +400, and holds actions/masks/family fixed. No model is selected or
fitted here. Floats and half-integer rounding ties remain implementation limits.
"""
import json,os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
from evaluate_panel import inputs
from state_models import StatePredictor
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
rows=inputs('test','annotation');keys=['context','context_masks','context_actions','context_dt','actions','durations','family']
tensors={k:torch.as_tensor(np.asarray([r[k] for r in rows]),dtype=torch.long if k in ['context_actions','actions','family'] else torch.float32) for k in keys}
shifted={**tensors,'context':tensors['context']+400}
out={}
freeze=json.loads((ROOT/'results/evaluation-freeze.json').read_text())
for variant in ['global_abs','global_delta','global_relative','local_fullaction_relative']:
    for seed in [17,29,43]:
        group='state-controls1000' if variant.startswith('local_full') else 'state-budget1000'
        p=ROOT/'results'/group/f'{variant}-seed-{seed}'/'best.pt'
        import hashlib
        assert hashlib.sha256(p.read_bytes()).hexdigest()==freeze['checkpoints'][str(p.relative_to(ROOT))]
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            ck=torch.load(p,map_location='cpu',weights_only=True)
        model=StatePredictor(variant).eval();model.load_state_dict(ck['model'],strict=True)
        with torch.inference_mode():
            first=model(**tensors)[0].numpy();second=model(**shifted)[0].numpy()
        error=second-first-400
        out[f'{variant}-seed{seed}']={'max_absolute_equivariance_error':float(np.abs(error).max()),
            'mean_absolute_equivariance_error':float(np.abs(error).mean()),
            'rounded_equivariance_fraction':float(np.mean(np.rint(first)+400==np.rint(second))),
            'any_shifted_prediction_outside_display_range':bool(((second<0)|(second>999)).any())}
record={'shift':400,'split':'already evaluated ordinary test','branches':len(rows),'floatingpoint':'FP32 CPU',
    'scope':'Structured complete numeric observations, no new visual condition, no model selection','results':out}
(ROOT/'results/analysis/translation-check.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
