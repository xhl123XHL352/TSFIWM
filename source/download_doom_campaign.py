"""Download only protocol-selected pinned episodes, not the full public corpus."""
import argparse
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='60'
from huggingface_hub import hf_hub_download
p=argparse.ArgumentParser()
p.add_argument('--subsets',nargs='+',default=['development','test_seen','test_unseen'])
a=p.parse_args()
config=json.loads((ROOT/'source'/'doom-campaign-config.json').read_text())
for subset in a.subsets:
    for item in config[subset]:
        ep=item['episode']
        filename=f"{item['raw_segment']}/ep_{ep:05}.parquet"
        hf_hub_download('RohanNaga/doom-dense-arnold',filename,repo_type='dataset',
            revision=config['data_revision'],local_dir=ROOT/'data'/'doomshift'/'raw',token=False)
        for suffix in ['latents.npy','meta.npz']:
            filename=f"sd15/{item['latent_segment']}/ep_{ep:05}_{suffix}"
            hf_hub_download('RohanNaga/doom-dense-arnold-latents',filename,repo_type='dataset',
                revision=config['latent_revision'],local_dir=ROOT/'data'/'doomshift'/'latents',token=False)
        print(json.dumps({'downloaded':subset,'episode':ep,'map':item['map']}),flush=True)
