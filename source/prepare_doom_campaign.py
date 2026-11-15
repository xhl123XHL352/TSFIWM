"""Local primary-release split inspection; no experimental outcome is inspected."""
import hashlib
import json
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'cache'/'dataset-audit'/'doomshift'/'official-splits'
folder.mkdir(parents=True,exist_ok=True)
episodes=[]
for map_id in [1]+list(range(6,18)):
    name=f'split_adapt_arenas13_map{map_id:02}_seed0.json'
    path=folder/name
    if not path.exists():
        url=f'https://huggingface.co/datasets/RohanNaga/doom-dense-arnold/resolve/cab7cb4b6a6d3c5d92b191ed65357344ee536324/arenas13/splits/{name}'
        r=requests.get(url,timeout=(15,60));r.raise_for_status();path.write_bytes(r.content)
    d=json.loads(path.read_text())
    print(json.dumps({'map':map_id,'keys':list(d)}),flush=True)
    held=d.get('held_out')
    assert isinstance(held,list) and len(held)==8,(name,type(held))
    ep=int(held[0])
    episodes.append({'episode':ep,'map':map_id,'raw_segment':'arenas13','latent_segment':'arenas13',
        'split_source':name,'split_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
out=ROOT/'source'/'doom-campaign-config.json'
config={'data_revision':'cab7cb4b6a6d3c5d92b191ed65357344ee536324',
    'latent_revision':'1835a6003e3cebbec12ccf4e73188f1e683b1cb0',
    'development':[{'episode':i,'map':i%4+2,'raw_segment':'arenas','latent_segment':'val'} for i in range(6000,6004)],
    'test_seen':[{'episode':i,'map':i%4+2,'raw_segment':'arenas','latent_segment':'test'} for i in range(7000,7008)],
    'test_unseen':episodes,'backbones':['unet','pixart'],
    'decoder_conditions':['stock','official-train-only-tuned'],
    'horizons':{'1':{'health':12,'ammo':12,'unchanged':8},'16':{'health':6,'ammo':6,'unchanged':4}},
    'generation_seed':1911,'statistics':'pair predictions by episode/window; cluster episode bootstrap stratified by seen map; unseen bootstrap across13 heldout maps, one episode per map; disclose within-map uncertainty unmeasured',
    'main_metric':'Per-field changed and unchanged visible-HUD exactness and their geometric mean; equal fields; no game-variable/display conflation',
    'limitations':'One released checkpoint per architecture, two architectures, one generation seed; stochastic health transitions and partial observability prevent deterministic-oracle interpretation; fixed-font readout metric'}
if out.exists() and json.loads(out.read_text())!=config:
    raise RuntimeError('Preserve existing campaign config and inspect any change')
out.write_text(json.dumps(config,indent=2))
print(json.dumps({'config':str(out),'seen_episodes':8,'unseen_maps':len(episodes),'unseen_episodes':[v['episode'] for v in episodes]}),flush=True)
