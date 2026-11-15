"""Train and evaluate specified state predictors; retains per-root predictions."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from gpu_accounting import GpuLease, ROOT


def load_data(split, dataset="panel-v3"):
    config = json.loads((ROOT/"source"/"panel-config.json").read_text())
    if any((ROOT/"data"/dataset/name).exists() for name in ["INVALID.json", "SUPERSEDED.json"]):
        raise RuntimeError("Dataset is marked invalid")
    folders = sorted((ROOT/"data"/dataset/split).iterdir())
    if len(folders) != config["splits"][split]["roots"]:
        raise RuntimeError("Split generation not complete")
    rows = []
    for folder in folders:
        if not (folder / "complete.json").exists():
            raise RuntimeError("Incomplete data root: " + str(folder))
        inp = json.loads((folder / "inputs.json").read_text())
        lab = json.loads((folder / "labels.json").read_text())
        ctx = lab["context_states"]
        for branch, future in zip(inp["branches"], lab["future_states"]):
            rows.append({"root_id": inp["root_id"], "branch": branch["branch"],
                "family": ["counter", "toggle", "flow"].index(inp["family"]),
                "context": [s["values"] for s in ctx],
                "context_masks": np.ones((len(ctx), 3), dtype=np.float32),
                "context_actions": [-1]+[c["preceding_action"]["target"] for c in inp["context"][1:]],
                "context_dt": [0]+[c["preceding_action"]["dt"] for c in inp["context"][1:]],
                "actions": [a["target"] for a in branch["actions"]],
                "durations": [a["dt"] for a in branch["actions"]],
                "truth": [s["values"] for s in future]})
    return rows


def pack(rows, device):
    import torch
    out = {}
    for k in ["context", "context_masks", "context_actions", "context_dt", "actions", "durations", "family", "truth"]:
        dtype = torch.long if k in ("context_actions", "actions", "family") else torch.float32
        out[k] = torch.as_tensor(np.array([r[k] for r in rows]), dtype=dtype, device=device)
    return out


def predict(model, tensors, ids=None):
    keys = ["context", "context_masks", "context_actions", "context_dt", "actions", "durations", "family"]
    return model(**{k: (tensors[k][ids] if ids is not None else tensors[k]) for k in keys})


def score(rows, values):
    truth = np.asarray([r["truth"] for r in rows])
    prior = np.concatenate([np.asarray([r["context"][-1] for r in rows])[:, None], truth[:, :-1]], axis=1)
    changed = truth != prior
    rounded = np.clip(np.rint(values), 0, 999).astype(int)
    correct = rounded == truth
    result, balanced = {}, []
    for f, name in enumerate(["counter", "toggle", "flow"]):
        mask = np.asarray([r["family"] == f for r in rows])
        c, g = correct[mask], changed[mask]
        ch = float(c[g].mean()) if g.any() else None
        un = float(c[~g].mean()) if (~g).any() else None
        joint = float(np.sqrt(ch*un)) if ch is not None and un is not None else float(c.mean())
        balanced.append(joint)
        result[name] = {"changed_accuracy": ch, "unchanged_accuracy": un,
                        "balanced_accuracy": joint, "all_values_accuracy": float(c.mean()),
                        "mae": float(np.abs(values[mask]-truth[mask]).mean())}
    root_scores = {}
    for i, row in enumerate(rows):
        root_scores.setdefault(row["root_id"], []).append(float(correct[i].all()))
    return {"balanced_accuracy": float(np.mean(balanced)), "by_family": result,
            "all_values_accuracy": float(correct.mean()),
            "all_steps_all_branches_root_success": float(np.mean([all(s) for s in root_scores.values()])),
            "mae": float(np.abs(values-truth).mean()), "roots": len(root_scores), "branches": len(rows)}


def dump_predictions(path, rows, values):
    with path.open("w", encoding="utf-8") as f:
        for row, pred in zip(rows, values):
            f.write(json.dumps({"root_id": row["root_id"], "branch": row["branch"],
                     "family": row["family"], "prediction": pred.tolist(), "truth": row["truth"]})+"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--round", required=True)
    parser.add_argument("--output-group", default="state")
    args = parser.parse_args()
    if not args.output_group.replace('-', '').replace('_', '').isalnum():
        raise ValueError('Output group must be a plain directory name')
    with GpuLease(args.gpu_uuid, "state-training-"+args.round, max_memory_gib=2):
        import torch
        from state_models import StatePredictor
        torch.set_num_threads(4)
        train_rows, dev_rows = load_data("train"), load_data("dev")
        train, dev = pack(train_rows, "cuda"), pack(dev_rows, "cuda")
        source_hash = hashlib.sha256(Path(__file__).read_bytes()+Path(__file__).with_name("state_models.py").read_bytes()).hexdigest()
        destination = ROOT / "results" / args.output_group
        destination.mkdir(parents=True, exist_ok=True)
        copy_values = np.repeat(np.asarray([r["context"][-1] for r in dev_rows])[:, None], 4, axis=1)
        (destination / "copy-dev.json").write_text(json.dumps(score(dev_rows, copy_values), indent=2))
        dump_predictions(destination / "copy-dev.jsonl", dev_rows, copy_values)
        from rule_baseline import forecast
        rule_values = np.array([forecast(row) for row in dev_rows])
        (destination / "known-rules-dev.json").write_text(json.dumps(score(dev_rows, rule_values), indent=2))
        dump_predictions(destination / "known-rules-dev.jsonl", dev_rows, rule_values)
        for variant in args.variants:
            for seed in args.seeds:
                output = destination / f"{variant}-seed-{seed}"
                if (output / "summary.json").exists():
                    old = json.loads((output / "summary.json").read_text())
                    if old["config"]["source_hash"] != source_hash or old["config"]["epochs"] != args.epochs:
                        raise RuntimeError("Existing run differs; preserve evidence and inspect: " + str(output))
                    print(json.dumps({"reused": str(output), "balanced_accuracy": old["dev"]["balanced_accuracy"]}), flush=True)
                    continue
                output.mkdir(exist_ok=True)
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = StatePredictor(variant).cuda()
                optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
                config = {"variant": variant, "seed": seed, "epochs": args.epochs, "batch_size": 128,
                          "hidden": 128, "layers": 2, "lr": 3e-4, "weight_decay": 1e-4,
                          "loss": "open-loop four-step squared numeric error / 100; optional gate BCE weight .05",
                          "parameters": sum(p.numel() for p in model.parameters()),
                          "source_hash": source_hash, "round": args.round, "torch": torch.__version__,
                          "dataset": "panel-v3", "output_group": args.output_group,
                          "train_roots": len(train_rows)//4, "dev_roots": len(dev_rows)//4,
                          "input": "annotated context values + commands, durations, and public family; no hidden gains/modes"}
                (output / "config.json").write_text(json.dumps(config, indent=2))
                best_key = (-1, -float("inf"))
                started = time.monotonic()
                with (output / "training.jsonl").open("w") as log:
                    for epoch in range(1, args.epochs+1):
                        model.train()
                        order = torch.randperm(len(train_rows), device="cuda")
                        total = 0
                        for ids in order.split(128):
                            optim.zero_grad(set_to_none=True)
                            pred, gate = predict(model, train, ids)
                            loss = ((pred-train["truth"][ids])/10).square().mean()
                            if not torch.isfinite(pred).all():
                                raise RuntimeError('Nonfinite model predictions')
                            if gate is not None:
                                truth = train["truth"][ids]
                                previous = torch.cat([train["context"][ids, -1, None], truth[:, :-1]], dim=1)
                                changed = (truth != previous).float()
                                loss = loss+.05*torch.nn.functional.binary_cross_entropy_with_logits(gate, changed)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                            optim.step()
                            total += float(loss.detach())*len(ids)
                        if epoch % 5 == 0 or epoch == 1:
                            model.eval()
                            with torch.inference_mode():
                                values = predict(model, dev)[0].cpu().numpy()
                            metrics = score(dev_rows, values)
                            record = {"epoch": epoch, "train_loss": total/len(train_rows), **metrics}
                            log.write(json.dumps(record)+"\n")
                            log.flush()
                            key = (metrics["balanced_accuracy"], -metrics["mae"])
                            if key > best_key:
                                best_key = key
                                torch.save({"model": model.state_dict(), "config": config, "epoch": epoch}, output / "best.pt")
                                dump_predictions(output / "dev-predictions.jsonl", dev_rows, values)
                                best_metrics, best_epoch = metrics, epoch
                            if epoch % 25 == 0 or epoch == 1:
                                print(json.dumps({"variant": variant, "seed": seed, "epoch": epoch,
                                      "balanced_accuracy": metrics["balanced_accuracy"], "mae": metrics["mae"]}), flush=True)
                summary = {"config": config, "selected_epoch": best_epoch,
                           "elapsed_s": time.monotonic()-started, "dev": best_metrics}
                (output / "summary.json").write_text(json.dumps(summary, indent=2))
                print(json.dumps({"finished": str(output), **summary}), flush=True)


if __name__ == "__main__":
    main()
