"""Freeze external WM identity, visible-text semantics and held-out episode list."""
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'results'/'doom-evaluation-freeze.json'
assert not path.exists(),'Inspect preserved protocol instead of refreezing'
config=json.loads((ROOT/'source'/'doom-campaign-config.json').read_text())
provenance=json.loads((ROOT/'models'/'doomshift'/'decoders'/'vae_decoder_sd1x_mse_lpips'/'provenance.json').read_text())
assert max(provenance['train_episodes'][0]['ids'])<2000
assets={}
for subset in ['development','test_seen','test_unseen']:
    for r in config[subset]:
        files=[ROOT/'data'/'doomshift'/'raw'/r['raw_segment']/f"ep_{r['episode']:05}.parquet"]
        files+=[ROOT/'data'/'doomshift'/'latents'/'sd15'/r['latent_segment']/f"ep_{r['episode']:05}_{suffix}" for suffix in ['latents.npy','meta.npz']]
        for p in files:
            assert p.exists() and p.stat().st_size>0
            h=hashlib.sha256()
            with p.open('rb') as f:
                while chunk:=f.read(8*1024**2):
                    h.update(chunk)
            assets[str(p.relative_to(ROOT))]={'sha256':h.hexdigest(),'size':p.stat().st_size}
sources=['doom_evaluate.py','doom_campaign.py','doom_prepare_windows.py','doom_score.py',
    'doom_text_audit.py','doom-campaign-config.json']
record={'utc':datetime.now(timezone.utc).isoformat(),'config':config,
    'source_hashes':{n:hashlib.sha256((ROOT/'source'/n).read_bytes()).hexdigest() for n in sources},
    'assets':assets,'decoder_train_ids_max':max(provenance['train_episodes'][0]['ids']),
    'decoder_selection_dev_ids':'6000:6100; no target held-out episode used',
    'label_semantics':'Raw visible HUD; fixed train0 font reader; one-tic display lag established on6000, raw target and previous text audited independently for every sampled window',
    'pretrained_ckpts':{'unet':'0728fb73086c00da7bb2b40eb843fbde7836b0dfae8e38adbc9b7300ef8fd9cb',
        'pixart':'46b1139b50ea5860fe03563fe9aad1b5b3fe802bfb518ffcb7c71f4b6d704b0a'},
    'reader_template_sha256':hashlib.sha256((ROOT/'results'/'doomshift'/'reader'/'train-0-templates.npy').read_bytes()).hexdigest()}
path.write_text(json.dumps(record,indent=2))
print(json.dumps({'frozen':str(path.relative_to(ROOT)),'assets':len(assets),'seen_episodes':8,'unseen_maps':13}),flush=True)
