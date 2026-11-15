"""Generate paired intervention rollouts, grouped by independent initial root."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pybullet as p
from panel_env import PanelEnv, ROOT, assets


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def generate(job):
    config, split, index, signature = job
    spec = config["splits"][split]
    seed = spec["seed_start"]+index
    root_id = f"{split}-{seed}"
    out = ROOT / "data" / config["version"] / split / root_id
    finished = out / "complete.json"
    if finished.exists():
        meta = json.loads(finished.read_text(encoding="utf-8"))
        if meta["signature"] != signature:
            raise RuntimeError("Existing data has another generator/config: " + str(out))
        return meta
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("Partial root requires inspection, not silent overwrite: " + str(out))
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed+config["seed"])
    family = config["families"][index % len(config["families"])]
    env = PanelEnv(seed=seed, family=family, style=spec["styles"][index % len(spec["styles"])],
                   width=config["image_width"], height=config["image_height"])
    env.values = rng.integers(spec["value_min"], spec["value_max"], 3).astype(float)
    env._refresh()
    context, context_states, events = [], [], []
    try:
        env.image().save(out / "context-0.png")
        context.append({"image": "context-0.png", "preceding_action": None})
        context_states.append(env.state())
        for i, target in enumerate(config["calibration_targets"]):
            event = env.step(target, dt=1)
            events.append(event)
            image = f"context-{i+1}.png"
            env.image().save(out / image)
            context.append({"image": image, "preceding_action": event["command"]})
            context_states.append(env.state())
        state_id = p.saveState(physicsClientId=env.client)
        snapshot = (env.values.copy(), env.modes.copy(), env.tick)
        continuation = rng.choice([-1, 0, 1, 2], config["future_steps"]-1).tolist()
        durations = rng.choice(config["dt_values"], config["future_steps"]).tolist()
        hidden = int(rng.integers(0, 3))
        branches, future_states, branch_events = [], [], []
        for b, first in enumerate(config["branch_first_actions"]):
            p.restoreState(state_id, physicsClientId=env.client)
            env.values, env.modes, env.tick = snapshot[0].copy(), snapshot[1].copy(), snapshot[2]
            env.hide(-1)
            env._refresh()
            actions, images, states, measured = [], [], [], []
            for t, target in enumerate([first]+continuation):
                env.hide(hidden if t in (1, 2) else -1)
                event = env.step(int(target), dt=int(durations[t]))
                measured.append(event)
                image = f"branch-{b}-step-{t}.png"
                env.image().save(out / image)
                actions.append(event["command"])
                images.append(image)
                states.append(env.state())
            branches.append({"branch": b, "actions": actions, "future_images": images})
            future_states.append(states)
            branch_events.append(measured)
        inputs = {"root_id": root_id, "split": split, "family": family,
                  "context": context, "branches": branches,
                  "action_interface": "Panda closed-loop press target slot, or wait; dt in abstract ticks"}
        labels = {"root_id": root_id, "context_states": context_states,
                  "future_states": future_states, "calibration_events": events,
                  "branch_events": branch_events,
                  "hidden_parameters_for_audit_only": {"gains": env.gains.tolist()}}
        dump(out / "inputs.json", inputs)
        dump(out / "labels.json", labels)
        all_events = events + [e for branch in branch_events for e in branch]
        presses = [e for e in all_events if e["command"]["target"] >= 0]
        meta = {"root_id": root_id, "family": family, "signature": signature,
                "ids": env.ids, "requested_presses": len(presses),
                "successful_presses": sum(e["pressed"] for e in presses),
                "images": len(context)+sum(len(b["future_images"]) for b in branches),
                "inputs_sha256": digest(out / "inputs.json"),
                "labels_sha256": digest(out / "labels.json")}
        dump(finished, meta)
        return meta
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "source" / "panel-config.json")
    parser.add_argument("--splits", nargs="+", default=["train", "dev"])
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--worker-split")
    parser.add_argument("--worker-index", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.audit:
        config["version"] = "panel-render-audit-v3"
        config["splits"]["train"]["roots"] = 1
        args.splits = ["train"]
    signature = hashlib.sha256((digest(args.config)+digest(__file__)+
              digest(Path(__file__).with_name("panel_env.py"))).encode()).hexdigest()
    if args.worker_split is not None:
        print(json.dumps(generate((config, args.worker_split, args.worker_index, signature))), flush=True)
        return
    assets()
    out = ROOT / "data" / config["version"]
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(config, split, i, signature) for split in args.splits for i in range(config["splits"][split]["roots"])]
    started = time.monotonic()
    records = []
    def external_worker(job):
        # Separate interpreters avoid shared PyBullet state. Ordinary pipes and
        # private files avoid multiprocessing's named /dev/shm semaphores.
        command = [sys.executable, str(Path(__file__)), "--config", str(args.config),
                   "--worker-split", job[1], "--worker-index", str(job[2])]
        if args.audit:
            command.append("--audit")
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180)
        if completed.returncode:
            raise RuntimeError(completed.stderr+completed.stdout)
        return json.loads(completed.stdout.strip().splitlines()[-1])
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        futures = [pool.submit(external_worker, job) for job in jobs]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records) % 24 == 0 or len(records) == len(jobs):
                print(json.dumps({"completed_roots": len(records), "total_roots": len(jobs),
                                  "elapsed_s": round(time.monotonic()-started, 1)}), flush=True)
    identities = [s for r in records for s in r["ids"]]
    collisions = len(identities)-len(set(identities))
    result = {"signature": signature, "splits": args.splits, "roots": len(records),
              "images": sum(r["images"] for r in records),
              "successful_presses": sum(r["successful_presses"] for r in records),
              "requested_presses": sum(r["requested_presses"] for r in records),
              "identity_collisions": collisions,
              "elapsed_s": time.monotonic()-started,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    dump(out / ("manifest-"+"-".join(args.splits)+".json"), {**result, "records": sorted(records, key=lambda r:r["root_id"])})
    print(json.dumps(result), flush=True)
    if collisions:
        raise RuntimeError("Inspect identity collisions before evaluation; do not silently ignore")


if __name__ == "__main__":
    main()
