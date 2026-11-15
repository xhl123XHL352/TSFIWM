"""Two bounded frozen-reader conditions for fresh independent goal scenes."""
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--gpu-uuid',required=True);a=p.parse_args()
for split in ['control_test','control_shift']:
    base=[sys.executable,'source/read_initial_views.py','--split',split]
    subprocess.run(base+['--gpu-uuid',a.gpu_uuid,'--memory-cap-gib','30'],cwd=ROOT,check=True,timeout=3600)
    subprocess.run(base+['--score'],cwd=ROOT,check=True,timeout=120)
