"""Finite CPU-only independent goal-control campaign."""
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--split',choices=['control_test','control_shift'],required=True)
p.add_argument('--near-only',action='store_true');a=p.parse_args()
base=[sys.executable,str(ROOT/'source/goal_control.py'),'--split',a.split]
calls=[]
if not a.near_only:
    for training,history in [('original','full'),('window_aug','window4'),('history_aug','full')]:
        calls.append(['--methods','local_fullaction_relative','--training',training,'--history',history])
    calls.append(['--methods','local_fullaction_relative','--training','history_aug','--history','full','--feedback','open_loop'])
    calls.append(['--methods','rules','rules_table','copy'])
    calls.append(['--methods','rules','rules_table','--feedback','open_loop'])
else:
    calls.append(['--methods','local_fullaction_relative','--training','history_aug','--history','full','--identity','near'])
    calls.append(['--methods','rules_table','--identity','near'])
for call in calls:
    subprocess.run(base+call,cwd=ROOT,check=True,timeout=7200)
print({'campaign_complete':a.split,'near_only':a.near_only},flush=True)
