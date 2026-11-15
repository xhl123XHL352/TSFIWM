"""Portable export: admit explicitly configured devices; retain allocated process time."""
from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone
import uuid

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = set(os.environ.get("TEXTSTATE_GPU_DEVICES", "").split(",")) - {""}


class GpuLease:
    def __init__(self, device, label, max_memory_gib=40):
        if device not in ALLOWED:
            raise ValueError("GPU not authorized")
        self.device, self.label = device, label
        if not 0 < max_memory_gib <= 40:
            raise ValueError('Task memory cap must stay within 40 GiB')
        self.max_memory_gib = max_memory_gib
        self.run_id = uuid.uuid4().hex
        self.folder = ROOT / "results"
        self.folder.mkdir(exist_ok=True)
        self.ledger = self.folder / "compute.jsonl"

    def append(self, record):
        with self.ledger.open("a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(json.dumps(record)+"\n")
            f.flush()
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def __enter__(self):
        rows = [json.loads(line) for line in self.ledger.read_text().splitlines()] if self.ledger.exists() else []
        closed = {r["run_id"] for r in rows if r["event"] == "end"}
        total = sum(r.get("elapsed_s", 0) for r in rows if r["event"] == "end")
        now = time.time()
        total += sum(now-r["unix_time"] for r in rows if r["event"] == "start" and r["run_id"] not in closed)
        active_limit = sum(r.get('max_memory_gib', 40) for r in rows if r['event'] == 'start'
                           and r['run_id'] not in closed and r['gpu'] == self.device)
        if active_limit+self.max_memory_gib > 40:
            raise RuntimeError('Combined task GPU memory caps would exceed 40 GiB on this device')
        if total >= 24*3600:
            raise RuntimeError("Initial 24 allocated GPU-hour envelope exhausted")
        free = float(subprocess.check_output(["nvidia-smi", "--id="+self.device,
             "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
        if free < (self.max_memory_gib+8)*1024:
            raise RuntimeError("Insufficient free GPU memory for the requested private limit + 8 GiB reserve")
        os.environ["CUDA_VISIBLE_DEVICES"] = self.device
        self.start = time.monotonic()
        self.append({"event": "start", "run_id": self.run_id, "label": self.label,
                     "gpu": self.device, "pid": os.getpid(), "unix_time": now,
                     "max_memory_gib": self.max_memory_gib,
                     "utc": datetime.now(timezone.utc).isoformat()})
        try:
            import torch
            torch.cuda.set_per_process_memory_fraction(self.max_memory_gib*1024**3/torch.cuda.get_device_properties(0).total_memory)
        except BaseException as error:
            self.__exit__(type(error), error, None)
            raise
        return self

    def __exit__(self, kind, value, traceback):
        self.append({"event": "end", "run_id": self.run_id, "label": self.label,
                     "gpu": self.device, "elapsed_s": time.monotonic()-self.start,
                     "status": "failed" if kind else "completed",
                     "error": str(value) if kind else None,
                     "utc": datetime.now(timezone.utc).isoformat()})
