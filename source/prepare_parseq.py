"""Unpack pinned OCR source and retrieve the official, hash-checked weights."""
from pathlib import Path
import hashlib
import os
import socket
import zipfile

ROOT = Path(__file__).resolve().parents[1]
archive = ROOT/'cache'/'parseq-source.zip'
if hashlib.sha256(archive.read_bytes()).hexdigest() != '5dfa5dd3583dd63911f8b2df53368ae35f3528a40c495c50b07a75c741dfeb09':
    raise RuntimeError('Unexpected source archive')
destination = ROOT/'cache'/'repos'/'parseq'
with zipfile.ZipFile(archive) as z:
    for member in z.infolist():
        if member.is_dir():
            continue
        target = destination.joinpath(*Path(member.filename).parts[1:]).resolve()
        if destination.resolve() not in target.parents:
            raise RuntimeError('Archive path escapes private cache')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(z.read(member))
(destination/'SOURCE_REVISION').write_text('1902db043c029a7e03a3818c616c06600af574be')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
socket.setdefaulttimeout(45)
weights = ROOT/'models'/'parseq'/'parseq-bb5792a6.pt'
weights.parent.mkdir(parents=True, exist_ok=True)
if not weights.exists():
    torch.hub.download_url_to_file('https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt',
                                  weights, hash_prefix='bb5792a6')
if not hashlib.sha256(weights.read_bytes()).hexdigest().startswith('bb5792a6'):
    raise RuntimeError('OCR weight hash mismatch')
print(str(weights), flush=True)
