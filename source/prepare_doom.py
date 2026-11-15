"""Retrieve a pinned public release and audit a training episode, privately."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile
import requests

ROOT = Path(__file__).resolve().parents[1]
CODE_REV = '7099c11e49b3a1acbbb5b06880aec8804379b4ef'
DATA_REV = 'cab7cb4b6a6d3c5d92b191ed65357344ee536324'
MODEL_REV = '28e76afcf8e1a8cdc98fb256719bf1767f61a9e2'


def verify(path, expected):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while data := f.read(8*1024*1024):
            h.update(data)
    if h.hexdigest() != expected:
        raise RuntimeError('Public LFS hash mismatch: '+str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action='store_true')
    args = parser.parse_args()
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'
    from huggingface_hub import hf_hub_download, snapshot_download
    cache = ROOT/'cache'/'repos'/'doomshift'
    if not (cache/'SOURCE_REVISION').exists():
        r = requests.get('https://codeload.github.com/intelligent-control-lab/DoomShift/zip/'+CODE_REV, timeout=(15,90))
        r.raise_for_status()
        cache.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            for member in z.infolist():
                parts = Path(member.filename).parts[1:]
                if not parts or member.is_dir():
                    continue
                target = cache.joinpath(*parts).resolve()
                if cache.resolve() not in target.parents:
                    raise RuntimeError('Archive member escapes private source cache')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(member))
        (cache/'SOURCE_REVISION').write_text(CODE_REV)
    if (cache/'SOURCE_REVISION').read_text().strip() != CODE_REV:
        raise RuntimeError('Different source revision already present')
    raw = ROOT/'data'/'doomshift'/'raw'
    episode = hf_hub_download('RohanNaga/doom-dense-arnold', 'arenas/ep_00000.parquet',
              repo_type='dataset', revision=DATA_REV, local_dir=raw, token=False)
    verify(episode, 'ef1c14c3477a226b2f35b22c682754992dbcb4d24fa5ac8a8114db5ef48b401d')
    import pyarrow.parquet as pq
    from PIL import Image
    parquet = pq.ParquetFile(episode)
    columns = ['tic','action','buttons','health','ammo','frame']
    batch = next(parquet.iter_batches(batch_size=128, columns=columns)).to_pydict()
    audit = ROOT/'data'/'doomshift'/'audit'
    audit.mkdir(parents=True, exist_ok=True)
    for index in [0, 64, 127]:
        Image.open(io.BytesIO(batch['frame'][index])).save(audit/f'frame-{index}.png')
    summary = {'source_revision': CODE_REV, 'data_revision': DATA_REV,
        'episode_rows': parquet.metadata.num_rows, 'columns': parquet.schema.names,
        'metadata': {k.decode(errors='replace'):v.decode(errors='replace') for k,v in (parquet.schema_arrow.metadata or {}).items()},
        'sample_rows': [{k:batch[k][i] for k in columns if k != 'frame'} for i in [0,64,127]],
        'note': 'Training episode 0 inspected for release audit; no held-out episode inspected'}
    (audit/'release-audit.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    if args.model:
        path = hf_hub_download('RohanNaga/doomdit-nexttic', '040-unet-nexttic/snap_0200000.pt',
            revision=MODEL_REV, local_dir=ROOT/'models'/'doomshift', token=False)
        verify(path, '0728fb73086c00da7bb2b40eb843fbde7836b0dfae8e38adbc9b7300ef8fd9cb')
        snapshot_download('RohanNaga/doomdit-nexttic', revision=MODEL_REV,
            local_dir=ROOT/'models'/'doomshift', token=False, max_workers=3,
            allow_patterns=['040-unet-nexttic/config.json', '040-unet-nexttic/train_episodes.json',
               'decoders/vae_decoder_sd1x_mse_lpips/vae/*',
               'decoders/vae_decoder_sd1x_mse_lpips/provenance.json'])
        print(json.dumps({'world_model_downloaded': str(path)}), flush=True)


if __name__ == '__main__':
    main()
