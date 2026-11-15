"""Download the pinned public comparison model into this private task only."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'
from huggingface_hub import snapshot_download

path = snapshot_download('OpenGVLab/InternVL3_5-8B-HF',
    revision='741a7d03020411e666c6109218ab71e08151ef86',
    local_dir=ROOT/'models'/'internvl3.5-8b', max_workers=4,
    allow_patterns=['*.json', '*.jinja', '*.safetensors', 'merges.txt', 'vocab.json', 'README.md'])
print(path, flush=True)
