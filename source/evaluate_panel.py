"""Frozen checkpoints with annotated or actual localized-OCR observations.

The model inputs select only context labels in the annotation condition.
Future labels are used only by the separate evaluator, never by predictors.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from gpu_accounting import ROOT, GpuLease


def inputs(split, condition):
    cfg=json.loads((ROOT/'source'/'panel-config.json').read_text())
    folders=sorted((ROOT/'data'/'panel-v3'/split).iterdir())
    assert len(folders)==cfg['splits'][split]['roots']
    ocr={}
    if condition=='parseq':
        path=ROOT/'results'/'perception'/f'parseq-calibrated-{split}.jsonl'
        for line in path.read_text().splitlines():
            r=json.loads(line)
            key=(r['root_id'],r['time'],r['panel'],r['field'])
            assert key not in ocr
            ocr[key]=r
        assert len(ocr)==len(folders)*24
    rows=[]
    for folder in folders:
        assert (folder/'complete.json').exists()
        inp=json.loads((folder/'inputs.json').read_text())
        masks=np.ones((4,3),np.float32)
        if condition=='annotation':
            # Explicit context selection: never pass future_states or hidden parameters.
            states=json.loads((folder/'labels.json').read_text())['context_states']
            context=[s['values'] for s in states]
            identities=states[0]['ids']
        else:
            context=np.zeros((4,3),np.float32)
            identities=[]
            for panel in range(3):
                candidates=[]
                for t in range(4):
                    r=ocr[(folder.name,t,panel,'id')]
                    text=r['text']
                    if len(text)==5 and text.isalnum():
                        candidates.append((r['token_probability_product'],text))
                    s=ocr[(folder.name,t,panel,'value')]['text']
                    if s.isdigit() and 0<=int(s)<=999:
                        context[t,panel]=int(s)
                    else:
                        context[t,panel]=context[t-1,panel] if t else 0
                        masks[t,panel]=0
                identities.append(max(candidates,key=lambda v:v[0])[1] if candidates else None)
        for branch in inp['branches']:
            rows.append({'root_id':folder.name,'branch':branch['branch'],
                'family':['counter','toggle','flow'].index(inp['family']),
                'context':np.asarray(context).tolist(),'context_masks':masks.tolist(),
                'context_actions':[-1]+[r['preceding_action']['target'] for r in inp['context'][1:]],
                'context_dt':[0]+[r['preceding_action']['dt'] for r in inp['context'][1:]],
                'actions':[r['target'] for r in branch['actions']],
                'durations':[r['dt'] for r in branch['actions']], 'ids':identities})
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--split',required=True,choices=['dev','test','test_shift'])
    p.add_argument('--condition',required=True,choices=['annotation','parseq'])
    p.add_argument('--gpu-uuid',required=True)
    a=p.parse_args()
    freeze=ROOT/'results'/'evaluation-freeze.json'
    if a.split!='dev' and not freeze.exists():
        raise RuntimeError('Freeze independent evaluation before test inference')
    manifest=json.loads(freeze.read_text()) if freeze.exists() else None
    rows=inputs(a.split,a.condition)
    destination=ROOT/'results'/f'panel-{a.condition}'
    destination.mkdir(parents=True,exist_ok=True)
    def write(name, values):
        path=destination/f'{name}-{a.split}.jsonl'
        data=''.join(json.dumps({'root_id':row['root_id'],'branch':row['branch'],
                    'family':row['family'],'ids':row['ids'],'prediction':pred.tolist()})+'\n'
                    for row,pred in zip(rows,values))
        if path.exists():
            if path.read_text()!=data:
                raise RuntimeError('Existing predictions differ; preserve and inspect: '+str(path))
            return
        path.write_text(data)
    from rule_baseline import forecast
    write('copy',np.repeat(np.asarray([r['context'][-1] for r in rows])[:,None],4,axis=1))
    write('known-rules',np.stack([forecast(r) for r in rows]))
    with GpuLease(a.gpu_uuid,'panel-evaluate-'+a.condition+'-'+a.split,max_memory_gib=2):
        import torch
        from state_models import StatePredictor
        torch.set_num_threads(4)
        keys=['context','context_masks','context_actions','context_dt','actions','durations','family']
        tensors={k:torch.as_tensor(np.asarray([r[k] for r in rows]),device='cuda',
            dtype=torch.long if k in ['context_actions','actions','family'] else torch.float32) for k in keys}
        checkpoints=[]
        for group in ['state-budget100','state-budget1000','state-controls1000']:
            checkpoints.extend(sorted((ROOT/'results'/group).glob('*/best.pt')))
        if a.split!='dev':
            assert len(checkpoints)==42
        with torch.inference_mode():
            for path in checkpoints:
                rel=str(path.relative_to(ROOT))
                if manifest and hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['checkpoints'][rel]:
                    raise RuntimeError('Checkpoint differs from freeze: '+rel)
                with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
                    ck=torch.load(path,map_location='cpu',weights_only=True)
                model=StatePredictor(ck['config']['variant']).cuda().eval()
                model.load_state_dict(ck['model'],strict=True)
                pred=model(**tensors)[0].cpu().numpy()
                name=path.parents[1].name+'--'+path.parent.name
                write(name,pred)
                print(json.dumps({'written':name,'split':a.split,'condition':a.condition,
                    'checkpoint_epoch':ck['epoch']}),flush=True)


if __name__=='__main__':
    main()
