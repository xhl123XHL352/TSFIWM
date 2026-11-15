"""Cache small pinned primary-source repositories over working raw HTTPS."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import requests
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPOS = {
 'doomshift': ('intelligent-control-lab/DoomShift', '7099c11e49b3a1acbbb5b06880aec8804379b4ef'),
 'parseq': ('baudm/parseq', '1902db043c029a7e03a3818c616c06600af574be'),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', choices=REPOS)
    args = parser.parse_args()
    repo, rev = REPOS[args.name]
    if args.name == 'doomshift' and (ROOT/'cache'/'dataset-audit'/'doomshift'/'tree.json').is_file():
        meta = json.loads((ROOT/'cache'/'dataset-audit'/'doomshift'/'tree.json').read_text())
    else:
        r = requests.get(f'https://api.github.com/repos/{repo}/git/trees/{rev}?recursive=1', timeout=30)
        r.raise_for_status()
        meta = r.json()
    paths = [r['path'] for r in meta['tree'] if r['type'] == 'blob']
    destination = ROOT/'cache'/'repos'/args.name
    destination.mkdir(parents=True, exist_ok=True)
    def get(path):
        target = (destination/path).resolve()
        if destination.resolve() not in target.parents:
            raise RuntimeError('Repository path escapes cache')
        if not target.exists():
            r = requests.get(f'https://raw.githubusercontent.com/{repo}/{rev}/{path}', timeout=(15,45))
            r.raise_for_status()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(r.content)
        return {'path':path,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(get, paths))
    manifest = {'repository':repo,'revision':rev,'files':records}
    (destination/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    archive = ROOT/'cache'/f'{args.name}-source.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            z.write(destination/path, 'source/'+path)
    print(json.dumps({'archive':str(archive),'files':len(paths),'bytes':archive.stat().st_size,
                      'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}),flush=True)


if __name__ == '__main__':
    main()
