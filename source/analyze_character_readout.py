"""String-level CER from saved native-model OCR outputs; no model inference."""
import argparse,hashlib,json,math,re
from pathlib import Path
from zipfile import ZipFile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--archive',type=Path)
p.add_argument('--records',type=Path,default=ROOT/'cache/results/hud-character-records.jsonl')
p.add_argument('--output',type=Path,default=ROOT/'cache/results/hud-character-metrics.json')
p.add_argument('--verify-primary',action='store_true')
a=p.parse_args()
sources=[('previous','Raw previous','unet','previous'),
 ('gt-stock','GT stock','unet','reconstructed'),('gt-tuned','GT tuned','unet','reconstructed-tuned'),
 ('unet-stock','U-Net stock','unet','predicted'),('unet-tuned','U-Net tuned','unet','predicted-tuned'),
 ('pixart-stock','PixArt stock','pixart','predicted'),('pixart-tuned','PixArt tuned','pixart','predicted-tuned')]

def edit_distance(x,y):
    row=list(range(len(y)+1))
    for i,c in enumerate(x,1):
        new=[i]
        for j,d in enumerate(y,1):new.append(min(new[-1]+1,row[j]+1,row[j-1]+(c!=d)))
        row=new
    return row[-1]

provenance={}
if a.archive:
    rows=[]
    pattern=re.compile(r'results/doomshift/test-(seen|unseen)-(unet|pixart)-ep(\d+)-h(1|16)-s1911/parseq-readout.json$')
    members={}
    with ZipFile(a.archive) as z:
        for name in z.namelist():
            match=pattern.fullmatch(name)
            if match:
                subset,model,episode,h=match.groups();members[subset,model,int(episode),int(h)]=name
        assert len(members)==84,len(members)
        for (subset,model,episode,h),name in sorted(members.items()):
            raw=z.read(name);provenance[name]=hashlib.sha256(raw).hexdigest();data=json.loads(raw)
            assert len(data)==(32 if h==1 else 16),(name,len(data))
            original=json.loads(z.read(name.replace('parseq-readout','template-readout')))
            assert [(r['index'],r['readout']) for r in data]==[(r['index'],r['readout']) for r in original]
            for record in data:
                assert record['episode']==episode
                for source,label,owner,condition in sources:
                    if owner!=model:continue
                    for field in ['ammo','health']:
                        target=record['readout']['target']['strings'][field]
                        fixed=record['readout'][condition]['strings'][field]
                        generic=record['parseq'][condition][field]['text']
                        assert isinstance(target,str) and target and isinstance(fixed,str) and isinstance(generic,str)
                        rows.append({'subset':subset,'episode':episode,'map':record['map'],'horizon':h,
                            'index':record['index'],'field':field,'source':source,
                            'target':target,'previous':record['readout']['previous']['strings'][field],
                            'target_value':record['readout']['target']['values'][field],
                            'previous_value':record['readout']['previous']['values'][field],
                            'fixed':fixed,'parseq':generic,
                            'fixed_value':record['readout'][condition]['values'][field],
                            'parseq_value':record['parseq'][condition][field]['value']})
    assert len(rows)==14112,len(rows)
    a.records.parent.mkdir(parents=True,exist_ok=True)
    a.records.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
else:
    rows=[json.loads(line) for line in a.records.read_text(encoding='utf-8').splitlines()]
assert len({(r['subset'],r['episode'],r['horizon'],r['index'],r['field'],r['source']) for r in rows})==len(rows)
out={'metric':'Sum of Levenshtein distances divided by sum of target string lengths',
 'target':'Calibrated raw visible-HUD strings; both readers use the same targets',
 'normalization':'None; raw OCR strings, including non-digits, insertions and leading zeros are retained',
 'resampling':'10000 paired episode draws; seen maps are strata; unseen episodes resampled across maps',
 'bootstrap_seed':413,'records':len(rows),'inputs':provenance,
 'records_sha256':hashlib.sha256(a.records.read_bytes()).hexdigest(),'results':{},'decoder_differences':{}}
for subset,n_episodes in [('seen',8),('unseen',13)]:
    for h in [1,16]:
        selected=[r for r in rows if r['subset']==subset and r['horizon']==h]
        episodes=sorted({r['episode'] for r in selected});assert len(episodes)==n_episodes
        index={ep:i for i,ep in enumerate(episodes)}
        rng=np.random.default_rng(413+h+(100 if subset=='unseen' else 0))
        if subset=='seen':
            maps={ep:next(r['map'] for r in selected if r['episode']==ep) for ep in episodes}
            draws=[]
            for mp in sorted(set(maps.values())):
                slots=np.array([index[ep] for ep in episodes if maps[ep]==mp]);assert len(slots)==2
                draws.append(rng.choice(slots,size=(10000,len(slots))))
            draws=np.concatenate(draws,axis=1)
        else:draws=rng.integers(0,len(episodes),size=(10000,len(episodes)))
        boot={}
        for source,label,owner,condition in sources:
            group=[r for r in selected if r['source']==source]
            assert len(group)==n_episodes*(32 if h==1 else 16)*2
            expected_len=sum(len(r['target']) for r in group);assert expected_len>0
            for reader in ['fixed','parseq']:
                errors=np.zeros(n_episodes,dtype=int);chars=np.zeros(n_episodes,dtype=int)
                for r in group:
                    i=index[r['episode']];errors[i]+=edit_distance(r[reader],r['target']);chars[i]+=len(r['target'])
                point=float(errors.sum()/chars.sum());sample=errors[draws].sum(1)/chars[draws].sum(1)
                changed=[];kept=[]
                for field in ['ammo','health']:
                    field_rows=[r for r in group if r['field']==field]
                    c=[r for r in field_rows if r['target_value']!=r['previous_value']]
                    k=[r for r in field_rows if r['target_value']==r['previous_value']]
                    changed.append(sum(r[reader+'_value']==r['target_value'] for r in c)/len(c))
                    kept.append(sum(r[reader+'_value']==r['target_value'] for r in k)/len(k))
                cge=float(np.mean(np.sqrt(np.array(changed)*np.array(kept))))
                if a.verify_primary:
                    ds=json.loads((ROOT/'cache/results'/('doom-test_'+subset+'.json')).read_text())
                    b=ds['horizons'][str(h)]['backbones'][owner]
                    ref=b['conditions'][condition]['cge'] if reader=='fixed' else b['secondary_parseq']['conditions'][condition]['cge']
                    assert abs(cge-ref)<1e-12,(subset,h,source,reader,cge,ref)
                key=f'{subset}/{h}/{source}/{reader}'
                out['results'][key]={'cer':point,'ci95':np.quantile(sample,[.025,.975]).tolist(),
                    'edits':int(errors.sum()),'target_characters':int(chars.sum()),'cge_crosscheck':cge,
                    'episodes':n_episodes,'field_strings':len(group)}
                boot[source,reader]=sample
        for model in ['unet','pixart']:
            for reader in ['fixed','parseq']:
                stock=f'{subset}/{h}/{model}-stock/{reader}';tuned=f'{subset}/{h}/{model}-tuned/{reader}'
                delta=boot[model+'-tuned',reader]-boot[model+'-stock',reader]
                out['decoder_differences'][f'{subset}/{h}/{model}/{reader}']={
                    'tuned_minus_stock':out['results'][tuned]['cer']-out['results'][stock]['cer'],
                    'ci95':np.quantile(delta,[.025,.975]).tolist()}
a.output.write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({'records':len(rows),'conditions':len(out['results']),
 'unseen_one_tic':{k:v for k,v in out['decoder_differences'].items() if k.startswith('unseen/1/')},
 'primary_cge_crosscheck':a.verify_primary,'output':str(a.output)},indent=2))
