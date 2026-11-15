"""Pinned public assets; no original authors' absolute paths are executed."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='60'
from huggingface_hub import hf_hub_download,snapshot_download

hf_hub_download('CompVis/stable-diffusion-v1-4','unet/config.json',
    revision='133a221b8aa7292a167afc5127cb63fb5005638b',
    local_dir=ROOT/'models'/'doom-base'/'sd14',token=False)
snapshot_download('stabilityai/sd-vae-ft-mse',revision='31f26fdeee1355a5c34592e401dd41e45d25a493',
    local_dir=ROOT/'models'/'doom-base'/'stock-vae',token=False,max_workers=2,
    allow_patterns=['config.json','diffusion_pytorch_model.safetensors'])
for filename in ['sd15/arenas/canonical_controls.json','sd15/arenas/encode_meta_00.json',
                 'sd15/val/ep_06000_latents.npy','sd15/val/ep_06000_meta.npz']:
    if (ROOT/'data'/'doomshift'/'latents'/filename).exists():
        continue
    hf_hub_download('RohanNaga/doom-dense-arnold-latents',filename,repo_type='dataset',
        revision='1835a6003e3cebbec12ccf4e73188f1e683b1cb0',
        local_dir=ROOT/'data'/'doomshift'/'latents',token=False)
hf_hub_download('RohanNaga/doom-dense-arnold','arenas/ep_06000.parquet',repo_type='dataset',
    revision='cab7cb4b6a6d3c5d92b191ed65357344ee536324',
    local_dir=ROOT/'data'/'doomshift'/'raw',token=False)
print('Pinned VAE, architecture config, and development episode 6000 available',flush=True)
