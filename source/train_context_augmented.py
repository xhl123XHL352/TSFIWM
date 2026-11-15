"""Matched-step context augmentation from existing training trajectories only.

Each batch samples an observed prefix of 0..3 future frames. Its last four
observations become context; only the remaining 4-prefix targets enter loss.
No extra simulator trajectories or test observations are used in training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from gpu_accounting import ROOT,GpuLease
from train_state import load_data,pack,predict,score,dump_predictions


def examples(tensors,prefix,history_mode='window4'):
    import torch
    s=tensors
    start=-4 if history_mode=='window4' else 0
    context=torch.cat([s['context'],s['truth'][:,:prefix]],1)[:,start:]
    return {'context':context,
        'context_masks':torch.ones_like(context),
        'context_actions':torch.cat([s['context_actions'],s['actions'][:,:prefix]],1)[:,start:],
        'context_dt':torch.cat([s['context_dt'],s['durations'][:,:prefix]],1)[:,start:],
        'actions':s['actions'][:,prefix:],'durations':s['durations'][:,prefix:],
        'family':s['family'],'truth':s['truth'][:,prefix:]}


def rows_for(rows,prefix,history_mode='window4'):
    start=-4 if history_mode=='window4' else 0
    return [{**r,'context':(r['context']+r['truth'][:prefix])[start:],
        'context_actions':(r['context_actions']+r['actions'][:prefix])[start:],
        'context_dt':(r['context_dt']+r['durations'][:prefix])[start:],
        'actions':r['actions'][prefix:],'durations':r['durations'][prefix:],
        'truth':r['truth'][prefix:]} for r in rows]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--epochs',type=int,default=1000)
    p.add_argument('--seeds',type=int,nargs='+',default=[17,29,43])
    p.add_argument('--variants',nargs='+',default=['global_abs','global_relative','local_fullaction_relative'])
    p.add_argument('--history-mode',choices=['window4','full'],default='window4')
    a=p.parse_args()
    with GpuLease(a.gpu_uuid,'context-augmentation-training',max_memory_gib=2):
        import torch
        from state_models import StatePredictor
        torch.set_num_threads(4)
        train_rows,dev_rows=load_data('train'),load_data('dev')
        train,dev=pack(train_rows,'cuda'),pack(dev_rows,'cuda')
        tr=[examples(train,i,a.history_mode) for i in range(4)]
        dv=[examples(dev,i,a.history_mode) for i in range(4)]
        dr=[rows_for(dev_rows,i,a.history_mode) for i in range(4)]
        group=ROOT/'results'/('state-contextaug1000' if a.history_mode=='window4' else 'state-historyaug1000');group.mkdir(exist_ok=True)
        code_hash=hashlib.sha256(Path(__file__).read_bytes()+Path(__file__).with_name('state_models.py').read_bytes()).hexdigest()
        for variant in a.variants:
            for seed in a.seeds:
                out=group/f'{variant}-seed-{seed}'
                if (out/'summary.json').exists():
                    prior=json.loads((out/'summary.json').read_text())
                    assert prior['config']['source_hash']==code_hash and prior['config']['epochs']==a.epochs
                    print(json.dumps({'preserved_complete':str(out.relative_to(ROOT))}),flush=True)
                    continue
                out.mkdir(exist_ok=True)
                assert not (out/'training.jsonl').exists(),'Partial training requires inspection'
                torch.manual_seed(seed);np.random.seed(seed)
                model=StatePredictor(variant).cuda()
                optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
                config={'variant':variant,'seed':seed,'epochs':a.epochs,'batch_size':128,
                    'lr':3e-4,'weight_decay':1e-4,'clip_grad':1,'source_hash':code_hash,
                    'context':('Last4 states' if a.history_mode=='window4' else 'Entire observed history')+' after a uniform observed prefix0..3 from original train branch',
                    'history_mode':a.history_mode,
                    'selection':'Mean development CGE across4 prefix conditions; no control-test feedback',
                    'loss':'Mean remaining-horizon squared numeric error /100',
                    'training_data':'Same600 panel-v3 training roots; no extra observations or trajectories',
                    'torch':str(torch.__version__),'parameters':sum(p.numel() for p in model.parameters())}
                (out/'config.json').write_text(json.dumps(config,indent=2))
                best=-1;started=time.monotonic()
                with (out/'training.jsonl').open('w') as log:
                    for epoch in range(1,a.epochs+1):
                        model.train()
                        for ids in torch.randperm(len(train_rows),device='cuda').split(128):
                            prefix=int(torch.randint(0,4,(1,)).item())
                            optimizer.zero_grad(set_to_none=True)
                            pred,_=predict(model,tr[prefix],ids)
                            loss=((pred-tr[prefix]['truth'][ids])/10).square().mean()
                            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
                        if epoch%5==0 or epoch==1:
                            model.eval();metrics=[];predictions=[]
                            with torch.inference_mode():
                                for prefix in range(4):
                                    values=predict(model,dv[prefix])[0].cpu().numpy()
                                    metrics.append(score(dr[prefix],values));predictions.append(values)
                            mean=float(np.mean([m['balanced_accuracy'] for m in metrics]))
                            record={'epoch':epoch,'mean_prefix_cge':mean,
                                'prefix_cge':[m['balanced_accuracy'] for m in metrics]}
                            log.write(json.dumps(record)+'\n');log.flush()
                            if mean>best:
                                best=mean;selected_epoch=epoch;selected_metrics=metrics
                                torch.save({'model':model.state_dict(),'config':config,'epoch':epoch},out/'best.pt')
                                for prefix in range(4):
                                    dump_predictions(out/f'dev-prefix{prefix}.jsonl',dr[prefix],predictions[prefix])
                            if epoch%100==0 or epoch==1:
                                print(json.dumps({'variant':variant,'seed':seed,**record}),flush=True)
                summary={'config':config,'selected_epoch':selected_epoch,'mean_prefix_cge':best,
                    'prefix_metrics':selected_metrics,'elapsed_s':time.monotonic()-started}
                (out/'summary.json').write_text(json.dumps(summary,indent=2))
                print(json.dumps({'finished':str(out.relative_to(ROOT)),'mean_prefix_cge':best}),flush=True)


if __name__=='__main__':
    main()
