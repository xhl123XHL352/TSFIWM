"""Additional calibrated initial views; original trajectories stay immutable.

Three 384x384 sensor views use fewer native pixels than one 768x640 image.
They cost three camera observations. No future state is observed or queried.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
import pybullet as p
from panel_env import PanelEnv, ROOT


def generate(split, index):
    cfg = json.loads((ROOT/'source'/'panel-config.json').read_text())
    spec = cfg['splits'][split]
    seed = spec['seed_start']+index
    out = ROOT/'data'/'panel-initial-views-v1'/split/f'{split}-{seed}'
    if (out/'complete.json').exists():
        return json.loads((out/'complete.json').read_text())
    out.mkdir(parents=True, exist_ok=True)
    family = cfg['families'][index % 3]
    env = PanelEnv(seed=seed, family=family, width=384, height=384,
                   style=spec['styles'][index % len(spec['styles'])])
    try:
        rng = np.random.default_rng(seed+cfg['seed'])
        env.values = rng.integers(spec['value_min'], spec['value_max'], 3).astype(float)
        env._refresh()
        # Compare only the initial state for replay provenance. Not a predictor input.
        lab = json.loads((ROOT/'data'/'panel-v3'/split/f'{split}-{seed}'/'labels.json').read_text())
        if env.state() != lab['context_states'][0]:
            raise RuntimeError('Initial replay differs from the fixed trajectory')
        names = []
        for i, x in enumerate(env.xs):
            view = p.computeViewMatrix([0, .13, 2.0], [x, .13, .672], [0, 1, 0])
            projection = p.computeProjectionMatrixFOV(15, 1, .05, 4)
            result = p.getCameraImage(384, 384, view, projection,
                                     renderer=p.ER_TINY_RENDERER, shadow=1, physicsClientId=env.client)
            name = f'panel-{i}.png'
            Image.fromarray(np.asarray(result[2], np.uint8)[..., :3]).save(out/name)
            names.append(name)
        record = {'root_id': f'{split}-{seed}', 'images': names, 'time': 'initial, before calibration',
                  'camera_origin': [0, .13, 2.0], 'vertical_fov_degrees': 15,
                  'pixels_per_image': [384, 384], 'observations': 3,
                  'source_hash': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (out/'complete.json').write_text(json.dumps(record, indent=2))
        return record
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='dev')
    parser.add_argument('--worker-index', type=int)
    args = parser.parse_args()
    if args.worker_index is not None:
        print(json.dumps(generate(args.split, args.worker_index)), flush=True)
        return
    cfg = json.loads((ROOT/'source'/'panel-config.json').read_text())
    count = cfg['splits'][args.split]['roots']
    def worker(index):
        completed = subprocess.run([sys.executable, str(Path(__file__)), '--split', args.split,
            '--worker-index', str(index)], cwd=ROOT, capture_output=True, text=True, timeout=120)
        if completed.returncode:
            raise RuntimeError(completed.stderr+completed.stdout)
        return json.loads(completed.stdout.strip().splitlines()[-1])
    with ThreadPoolExecutor(max_workers=12) as pool:
        tasks = [pool.submit(worker, i) for i in range(count)]
        for i, future in enumerate(as_completed(tasks), 1):
            future.result()
            if i % 24 == 0 or i == count:
                print(json.dumps({'initial_view_roots': i, 'total': count}), flush=True)


if __name__ == '__main__':
    main()
