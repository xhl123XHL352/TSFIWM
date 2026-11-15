"""Multi-episode held-out readout analysis, no model or prediction imports.

Paired episode resampling retains all windows, all decoder conditions and both
backbones together. Seen maps are strata. Unseen sampling is across the 13
released held-out maps (one selected episode per map); it cannot estimate
within-map uncertainty. No individual video frame is treated as a new scene.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
FIELDS=['ammo','health']
CONDITIONS=['previous','predicted','predicted-tuned','reconstructed','reconstructed-tuned','copy-decoded','copy-decoded-tuned']
p=argparse.ArgumentParser();p.add_argument('--subset',choices=['development','test_seen','test_unseen'],required=True);a=p.parse_args()
config=json.loads((ROOT/'source/doom-campaign-config.json').read_text())
registry_path=ROOT/'results/analysis/doom-visible-label-verification.json'
registry=json.loads(registry_path.read_text()) if registry_path.exists() else {'entries':[]}
timing_exceptions={(r['episode'],r['horizon'],r['index'],r['role']):r for r in registry['entries']}

def verify_alignment(endpoint,episode,horizon,index,role):
    if endpoint['match']:return
    key=(episode,horizon,index,role)
    assert key in timing_exceptions,(key,'Unregistered raw-label timing mismatch')
    r=timing_exceptions[key]
    assert all(endpoint[k]==r[k] for k in ['row','visible','aligned_gamevars'])
    assert hashlib.sha256((ROOT/r['image']).read_bytes()).hexdigest()==r['image_sha256']
items=config[a.subset];episodes=[x['episode'] for x in items];maps=np.array([x['map'] for x in items]);n=len(items)
if a.subset!='development':assert (ROOT/'results/doom-evaluation-freeze.json').exists()
rng=np.random.default_rng(20261113)
if a.subset=='test_seen':
    indices=np.stack([np.concatenate([rng.choice(np.flatnonzero(maps==m),sum(maps==m),replace=True) for m in sorted(set(maps))]) for _ in range(10000)])
else:indices=rng.integers(0,n,(10000,n))


def score(counts):
    # Counts: [..., condition, field, (correct_changed,changed,correct_kept,kept)]
    ch=counts[...,0]/counts[...,1];ke=counts[...,2]/counts[...,3]
    return np.sqrt(ch*ke).mean(-1)


result={'subset':a.subset,'episodes':episodes,'maps':maps.tolist(),'bootstrap_replicates':10000,
    'unit':'episode cluster; maps2-5 stratified for seen; one episode per map for unseen; all architectures, decoders and windows paired',
    'generation_seed':config['generation_seed'],'notes':'Event-stratified evaluation, not deployment-frequency estimates; one released checkpoint per architecture; one generation seed; raw visible labels, not same-tic game variables.',
    'horizons':{}}
result['raw_label_alignment']={'verified_timing_exceptions':len([r for r in registry['entries'] if r['episode'] in episodes]),
    'label_changes':0,'excluded_windows':0,'semantics':'Original frozen fixed-font visible labels; registered same-tic health exceptions to the developmental lag-one check are retained.',
    'registry_sha256':hashlib.sha256(registry_path.read_bytes()).hexdigest() if registry_path.exists() else None}
for horizon in [1,16]:
    bymodel={};arrays={};bootstrap={};psnr_boot={}
    expected_n=sum(config['horizons'][str(horizon)].values())
    shared_windows={};total_raw_checked=0
    for backbone in ['unet','pixart']:
        counts=np.zeros((n,len(CONDITIONS),2,4),dtype=np.int64)
        psnr=np.zeros((n,len(CONDITIONS)))
        secondary=np.zeros_like(counts);secondary_raw=np.zeros((n,2),dtype=np.int64);has_secondary=True
        for e,item in enumerate(items):
            name=f"{a.subset.replace('_','-')}-{backbone}-ep{item['episode']:05}-h{horizon}-s{config['generation_seed']}"
            folder=ROOT/'results/doomshift'/name;assert (folder/'complete.json').exists(),name
            records=json.loads((folder/'template-readout.json').read_text());assert len(records)==expected_n
            ids=[r['index'] for r in records];assert len(ids)==len(set(ids))
            window_manifest=json.loads((ROOT/'results/doomshift/windows'/f"ep-{item['episode']:05}-h{horizon}-n{expected_n}.json").read_text())
            aligned={r['index']:r for r in window_manifest['windows']}
            assert set(aligned)==set(ids)
            if item['episode'] in shared_windows:assert ids==shared_windows[item['episode']]
            else:shared_windows[item['episode']]=ids
            other=json.loads((folder/'parseq-readout.json').read_text()) if (folder/'parseq-readout.json').exists() else None
            if other:assert [r['index'] for r in other]==ids
            else:has_secondary=False
            for w,r in enumerate(records):
                assert r['episode']==item['episode'] and r['map']==item['map']
                for f,field in enumerate(FIELDS):
                    truth=r['readout']['target']['values'][field];previous=r['readout']['previous']['values'][field]
                    assert truth is not None and previous is not None,(name,r['index'],field,'null raw transcription')
                    audit=aligned[r['index']]
                    verify_alignment(audit['target'],item['episode'],horizon,r['index'],'target')
                    verify_alignment(audit['previous'],item['episode'],horizon,r['index'],'previous')
                    assert truth==audit['target']['visible'][field] and previous==audit['previous']['visible'][field],(name,r['index'],field,'visible label audit mismatch')
                    total_raw_checked+=1
                    changed=truth!=previous
                    if other:secondary_raw[e,f]+=other[w]['parseq']['target'][field]['value']==truth
                    for c,condition in enumerate(CONDITIONS):
                        correct=r['readout'][condition]['values'][field]==truth
                        counts[e,c,f]+=np.array([correct and changed,changed,correct and not changed,not changed],dtype=int)
                        if other:
                            ok=other[w]['parseq'][condition][field]['value']==truth
                            secondary[e,c,f]+=np.array([ok and changed,changed,ok and not changed,not changed],dtype=int)
                for c,condition in enumerate(CONDITIONS):psnr[e,c]+=r['pixel_metrics'][condition]['hud_psnr']/expected_n
        aggregate=counts.sum(0);bs=score(counts[indices].sum(1));value=score(aggregate)
        arrays[backbone]=counts;bootstrap[backbone]=bs;psnr_boot[backbone]=psnr[indices].mean(1)
        conditions={}
        for c,condition in enumerate(CONDITIONS):
            fields={field:{'n_changed':int(aggregate[c,f,1]),'n_unchanged':int(aggregate[c,f,3]),
                'changed_exact':float(aggregate[c,f,0]/aggregate[c,f,1]),'unchanged_exact':float(aggregate[c,f,2]/aggregate[c,f,3]),
                'cge':float(np.sqrt(aggregate[c,f,0]/aggregate[c,f,1]*aggregate[c,f,2]/aggregate[c,f,3]))} for f,field in enumerate(FIELDS)}
            conditions[condition]={'cge':float(value[c]),'cge_ci95':np.quantile(bs[:,c],[.025,.975]).tolist(),
                'fields':fields,'hud_psnr':float(psnr[:,c].mean()),'hud_psnr_ci95':np.quantile(psnr_boot[backbone][:,c],[.025,.975]).tolist()}
        gains={}
        for left,right in [('predicted-tuned','predicted'),('reconstructed-tuned','reconstructed'),('predicted-tuned','reconstructed-tuned'),('predicted-tuned','previous')]:
            l,r=CONDITIONS.index(left),CONDITIONS.index(right)
            gains[left+' MINUS '+right]={'cge_difference':float(value[l]-value[r]),'ci95':np.quantile(bs[:,l]-bs[:,r],[.025,.975]).tolist()}
        entry={'windows':n*expected_n,'conditions':conditions,'paired_differences':gains,
            'per_episode':{str(episode):dict(zip(CONDITIONS,score(counts[e]).tolist())) for e,episode in enumerate(episodes)}}
        if has_secondary:
            sec=score(secondary.sum(0));secbs=score(secondary[indices].sum(1))
            agreement=secondary_raw.sum(0)/(n*expected_n)
            entry['secondary_parseq']={'raw_target_agreement':dict(zip(FIELDS,agreement.tolist())),
                'interpretation':'Generic-reader sensitivity. Low raw-target agreement disqualifies attributing its errors solely to generation.',
                'conditions':{condition:{'cge':float(sec[c]),'ci95':np.quantile(secbs[:,c],[.025,.975]).tolist()} for c,condition in enumerate(CONDITIONS)}}
        bymodel[backbone]=entry
    paired={}
    v1=score(arrays['unet'].sum(0));v2=score(arrays['pixart'].sum(0))
    for c,condition in enumerate(CONDITIONS):
        paired[condition]={'unet_minus_pixart':float(v1[c]-v2[c]),'ci95':np.quantile(bootstrap['unet'][:,c]-bootstrap['pixart'][:,c],[.025,.975]).tolist()}
    result['horizons'][str(horizon)]={'backbones':bymodel,'paired_backbone_differences':paired,'raw_field_checks':total_raw_checked}
target=ROOT/'results/analysis'/('doom-'+a.subset+'.json');target.write_text(json.dumps(result,indent=2))
print(json.dumps({'saved':str(target),'episodes':n,'summary':{h:{b:{c:v['cge'] for c,v in row['conditions'].items()} for b,row in r['backbones'].items()} for h,r in result['horizons'].items()}}),flush=True)
