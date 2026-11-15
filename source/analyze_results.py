"""Independent root-grouped evaluator; never imports the candidate models."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def truth_data(split):
    truth, previous, families, identities = {}, {}, {}, {}
    for folder in sorted((ROOT/'data'/'panel-v3'/split).iterdir()):
        inp = json.loads((folder/'inputs.json').read_text())
        lab = json.loads((folder/'labels.json').read_text())
        root = inp['root_id']
        families[root] = ['counter', 'toggle', 'flow'].index(inp['family'])
        identities[root] = lab['context_states'][0]['ids']
        for b, states in enumerate(lab['future_states']):
            key = (root, b)
            truth[key] = np.asarray([s['values'] for s in states])
            previous[key] = np.concatenate([np.asarray(lab['context_states'][-1]['values'])[None], truth[key][:-1]])
    return truth, previous, families, identities


def evaluate(path, truth, previous, families, identities, draws):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    prediction = {(r['root_id'], r['branch']): r for r in records}
    if len(prediction) != len(records) or set(prediction) != set(truth):
        return None
    roots = sorted(families)
    keys = [(root, b) for root in roots for b in range(4)]
    pred = np.array([prediction[k].get('prediction', prediction[k].get('values')) for k in keys], dtype=float)
    target = np.array([truth[k] for k in keys])
    changed = target != np.array([previous[k] for k in keys])
    valid = np.array([prediction[k].get('parse_valid', True) for k in keys])
    correct = (np.clip(np.rint(pred), 0, 999) == target) & valid[:, None, None]
    shape = (len(roots), 4, 4, 3)
    correct, changed = correct.reshape(shape), changed.reshape(shape)
    num_c = (correct & changed).sum((1, 2, 3))
    den_c = changed.sum((1, 2, 3))
    num_u = (correct & ~changed).sum((1, 2, 3))
    den_u = (~changed).sum((1, 2, 3))
    balanced, bootstrap, by_family = [], [], {}
    for f, name in enumerate(['counter', 'toggle', 'flow']):
        ids = np.array([i for i, root in enumerate(roots) if families[root] == f])
        c = num_c[ids].sum()/den_c[ids].sum()
        u = num_u[ids].sum()/den_u[ids].sum()
        metric = np.sqrt(c*u)
        balanced.append(metric)
        sampled = draws[f]
        c_sample = num_c[sampled].sum(1)/den_c[sampled].sum(1)
        u_sample = num_u[sampled].sum(1)/den_u[sampled].sum(1)
        bootstrap.append(np.sqrt(c_sample*u_sample))
        by_family[name] = {'changed_accuracy': float(c), 'unchanged_accuracy': float(u),
                           'balanced_accuracy': float(metric)}
    samples = np.mean(bootstrap, axis=0)
    root_flags=correct.all((1,2,3))
    root_samples=root_flags[np.concatenate(draws,axis=1)].mean(1)
    result = {'roots': len(roots), 'branches': len(keys), 'balanced_accuracy': float(np.mean(balanced)),
              'balanced_ci95': np.quantile(samples, [.025, .975]).tolist(),
              'all_values_accuracy': float(correct.mean()),
              'root_all_steps_all_branches_success': float(correct.all((1, 2, 3)).mean()),
              'root_success_ci95':np.quantile(root_samples,[.025,.975]).tolist(),
              'parse_failure_rate': float((~valid).mean()),
              'mae_valid_format': float(np.abs(pred[valid]-target[valid]).mean()) if valid.any() else None,
              'by_family': by_family, 'source': str(path.relative_to(ROOT))}
    if any('ids' in r for r in records):
        id_accuracy = [prediction[k].get('ids') == identities[k[0]] for k in keys]
        result['all_three_identifiers_exact'] = float(np.mean(id_accuracy))
        identity_by_branch = np.asarray(id_accuracy).reshape(len(roots),4)
        joint = correct.all((1,2,3)) & identity_by_branch.all(1)
        result['root_all_values_and_identifiers_success'] = float(joint.mean())
        result['state_step_all_fields_exact'] = float((correct.all(3) & identity_by_branch[:,:,None]).mean())
    return result, samples, root_samples


def perception(split):
    files = list((ROOT/'results'/'perception').glob('qwen-reader-context-*.jsonl')) if (ROOT/'results'/'perception').exists() else []
    observations = {}
    for file in files:
        for line in file.read_text().splitlines():
            row = json.loads(line)
            observations[row['image']] = row
    all_ids, all_values, first_ids, memory_ids, count = [], [], [], [], 0
    for folder in sorted((ROOT/'data'/'panel-v3'/split).iterdir()):
        inp = json.loads((folder/'inputs.json').read_text())
        lab = json.loads((folder/'labels.json').read_text())
        paths = [str((folder/c['image']).relative_to(ROOT)) for c in inp['context']]
        if any(path not in observations for path in paths):
            return None
        rows = [observations[path] for path in paths]
        truth_ids = lab['context_states'][0]['ids']
        first_ids.extend([rows[0]['ids'][i] == truth_ids[i] for i in range(3)])
        for i in range(3):
            first = next((r['ids'][i] for r in rows if isinstance(r['ids'][i], str)
                and len(r['ids'][i]) == 5 and r['ids'][i].isalnum()), None)
            memory_ids.append(first == truth_ids[i])
        for row, state in zip(rows, lab['context_states']):
            all_ids.extend([row['ids'][i] == state['ids'][i] for i in range(3)])
            all_values.extend([row['values'][i] == state['values'][i] for i in range(3)])
            count += 1
    return {'images': count, 'identifier_field_recovery': float(np.mean(all_ids)),
            'numeric_field_recovery': float(np.mean(all_values)),
            'initial_image_identifier_recovery': float(np.mean(first_ids)),
            'first_valid_identity_memory_recovery': float(np.mean(memory_ids)),
            'interpretation': 'State recovery from available observations, including robot occlusion; not conditional OCR accuracy on fully visible text'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='dev')
    parser.add_argument('--round', required=True)
    parser.add_argument('--bootstrap', type=int, default=10000)
    parser.add_argument('--groups', nargs='+', default=['state'])
    parser.add_argument('--summary-only', action='store_true')
    args = parser.parse_args()
    if args.split != 'dev' and not (ROOT/'results'/'evaluation-freeze.json').exists():
        raise RuntimeError('Independent evaluation not frozen')
    truth, previous, families, identities = truth_data(args.split)
    roots = sorted(families)
    rng = np.random.default_rng(11321)
    draws = [rng.choice([i for i, r in enumerate(roots) if families[r] == f],
        size=(args.bootstrap, sum(families[r] == f for r in roots)), replace=True) for f in range(3)]
    paths = []
    for state in [ROOT/'results'/group for group in args.groups]:
        if not state.is_dir():
            continue
        paths += list(state.glob(f'*-{args.split}.jsonl'))
        paths += [folder/f'{args.split}-predictions.jsonl' for folder in state.iterdir()
                  if folder.is_dir() and (folder/'summary.json').exists() and (folder/f'{args.split}-predictions.jsonl').exists()]
    vlm = ROOT/'results'/'vlm'
    if vlm.exists():
        paths += list(vlm.glob(f'*-{args.split}.jsonl'))
    results, pending, sampled, sampled_roots = {}, [], {}, {}
    for path in sorted(paths):
        value = evaluate(path, truth, previous, families, identities, draws)
        key = str(path.relative_to(ROOT/'results'))
        if value is None:
            pending.append(key)
        else:
            results[key], sampled[key], sampled_roots[key] = value
    for key in sampled:
        group = key.split('/')[0]
        reference = f'{group}/global_abs-seed-17/{args.split}-predictions.jsonl'
        if reference in sampled:
            diff = sampled[key]-sampled[reference]
            results[key]['paired_difference_vs_matched_global_abs_seed17_ci95'] = np.quantile(diff, [.025, .975]).tolist()
            results[key]['paired_reference'] = reference
    seed_sets = {}
    for key in sampled:
        match = re.search(r'(.*)-seed-(17|29|43)(?:/|-)'+re.escape(args.split)+r'(?:-predictions)?\.jsonl$',key)
        if match:
            seed_sets.setdefault(match[1],[]).append(key)
    seed_summary = {}
    for method,keys in seed_sets.items():
        if len(keys)!=3:
            continue
        points=[results[k]['balanced_accuracy'] for k in keys]
        seed_summary[method]={'seeds':3,'mean_cge':float(np.mean(points)),
            'seed_min_max':[min(points),max(points)],
            'mean_seed_root_bootstrap_ci95':np.quantile(np.mean([sampled[k] for k in keys],axis=0),[.025,.975]).tolist(),
            'mean_numeric_root_success':float(np.mean([results[k]['root_all_steps_all_branches_success'] for k in keys])),
            'numeric_root_success_ci95':np.quantile(np.mean([sampled_roots[k] for k in keys],axis=0),[.025,.975]).tolist()}
    output = {'split': args.split, 'round': args.round, 'bootstrap_unit': 'initial root, stratified by family',
              'bootstrap_samples': args.bootstrap, 'results': results, 'incomplete_outputs': pending,
              'seed_summary':seed_summary,'perception': perception(args.split)}
    destination = ROOT/'results'/'analysis'
    destination.mkdir(parents=True, exist_ok=True)
    (destination/f'round-{args.round}-{args.split}.json').write_text(json.dumps(output, indent=2))
    if args.summary_only:
        print(json.dumps({'split': args.split, 'round': args.round,
             'scores': {k: {'balanced_accuracy': v['balanced_accuracy'],
                           'ci95': v['balanced_ci95']} for k,v in results.items()},
             'perception': output['perception'], 'incomplete': pending}), flush=True)
    else:
        print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
