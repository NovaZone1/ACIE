"""Run a frozen method set once on exported cross-domain test pools."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import yaml

from acie.data import Bundle, group_key, holdout, resolve_split
from acie.engine import predict, train
from acie.io import digest_file, digest_object, read_json, read_jsonl, write_json
from acie.metrics import bootstrap


def make_fixed_split(bundle: Bundle, source: str, target: str, seed: int,
                     val_fraction: float) -> dict:
    train_pool = np.asarray([i for i, row in enumerate(bundle.meta)
                             if row['dataset'] == source and row['pool'] == 'train'], dtype=int)
    test_pool = np.asarray([i for i, row in enumerate(bundle.meta)
                            if row['dataset'] == target and row['pool'] == 'test'], dtype=int)
    if source == target or not len(train_pool) or not len(test_pool):
        raise ValueError('Each direction needs distinct, non-empty source-train and target-test pools')
    train_indices, val_indices = holdout(train_pool, bundle, val_fraction, seed)
    split = {
        'schema': 'acie.split.v1',
        'source': source,
        'target': target,
        'seed': seed,
        'protocol': 'fixed_exported_test_pool_source_group_dev',
        'train': [bundle.meta[i]['sample_id'] for i in train_indices],
        'val': [bundle.meta[i]['sample_id'] for i in val_indices],
        'test': [bundle.meta[i]['sample_id'] for i in test_pool],
    }
    resolve_split(bundle, split)
    return split


def ensemble(run_root: Path, method: str, seeds: list[int]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    tables = []
    for seed in seeds:
        rows = read_jsonl(run_root / method / f'seed{seed}' / 'test_predictions.jsonl')
        tables.append({row['sample_id']: row for row in rows})
    ids = sorted(tables[0])
    if any(set(table) != set(ids) for table in tables):
        raise ValueError('Prediction IDs differ across seeds')
    if any(len(table) != len(ids) for table in tables):
        raise ValueError('Duplicate prediction IDs')
    labels = np.asarray([tables[0][sample_id]['label'] for sample_id in ids])
    scores = np.mean([[table[sample_id]['score'] for sample_id in ids] for table in tables], axis=0)
    meta = [tables[0][sample_id] for sample_id in ids]
    return labels, scores, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--domains', nargs=2, required=True)
    parser.add_argument('--method', action='append', required=True,
                        help='Frozen method as NAME=CONFIG; repeat for every method')
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--confirm-frozen', action='store_true')
    args = parser.parse_args()
    if not args.confirm_frozen:
        raise SystemExit('Review and freeze every config, then pass --confirm-frozen')
    methods = {}
    for item in args.method:
        if '=' not in item:
            raise ValueError('--method must use NAME=CONFIG')
        name, path = item.split('=', 1)
        if not name or name in methods:
            raise ValueError('Method names must be non-empty and unique')
        methods[name] = Path(path)
    required = {'geometry', 'dual_no_pair', 'weighted_no_pair', 'full_selected'}
    if set(methods) != required:
        raise ValueError(f'Frozen method set must be exactly {sorted(required)}')
    configs = {name: yaml.safe_load(path.read_text()) for name, path in methods.items()}
    if configs['full_selected'].get('pair_lambda') != 0.03:
        raise ValueError('The preregistered full model must use pair_lambda=0.03')
    if configs['weighted_no_pair'].get('pair_lambda') != 0.0:
        raise ValueError('The weighted paired-loss ablation must use pair_lambda=0')

    bundle = Bundle.load(args.data)
    first, second = args.domains
    directions = [(first, second), (second, first)]
    splits = {}
    split_hashes = {}
    for source, target in directions:
        direction = f'{source}_to_{target}'
        split_path = args.out / direction / 'split.json'
        if split_path.is_file():
            split = read_json(split_path)
        else:
            split = make_fixed_split(bundle, source, target, args.split_seed, args.val_fraction)
            write_json(split_path, split)
        resolve_split(bundle, split)
        splits[direction] = split
        split_hashes[direction] = digest_file(split_path)
    lock_payload = {
        'schema': 'acie.frozen-cross-domain.v1',
        'data_sha256': digest_file(args.data),
        'domains': args.domains,
        'methods': {name: {'path': str(path), 'sha256': digest_file(path), 'config': configs[name]}
                    for name, path in methods.items()},
        'seeds': args.seeds,
        'split_seed': args.split_seed,
        'val_fraction': args.val_fraction,
        'split_sha256': split_hashes,
        'selection': 'all hyperparameters frozen before target test scoring',
    }
    lock_payload['lock_digest'] = digest_object(lock_payload)
    lock_path = args.out / 'FROZEN_PROTOCOL.json'
    args.out.mkdir(parents=True, exist_ok=True)
    if lock_path.is_file():
        existing = read_json(lock_path)
        if existing != lock_payload:
            raise ValueError('Frozen protocol changed after test evaluation started')
    else:
        write_json(lock_path, lock_payload)

    rows = read_json(args.out / 'runs.json') if (args.out / 'runs.json').is_file() else []
    for source, target in directions:
        direction = f'{source}_to_{target}'
        direction_root = args.out / direction
        split = splits[direction]
        indices = resolve_split(bundle, split)
        for method, base_config in configs.items():
            for seed in args.seeds:
                run = direction_root / method / f'seed{seed}'
                run.mkdir(parents=True, exist_ok=True)
                config = copy.deepcopy(base_config)
                config['seed'] = seed
                report_path = run / 'model' / 'training_report.json'
                if report_path.is_file() and (run / 'model' / 'best.pt').is_file():
                    training = read_json(report_path)
                else:
                    training = train(bundle, split, config, run / 'model',
                                     resume=(run / 'model' / 'run.json').is_file())
                metrics_path = run / 'test_predictions.metrics.json'
                if metrics_path.is_file():
                    result = read_json(metrics_path)
                else:
                    result = predict(bundle, run / 'model' / 'best.pt',
                                     run / 'test_predictions.jsonl', 'test', repeats=args.bootstrap)
                rows = [row for row in rows if (row['source'], row['target'], row['method'], row['seed']) !=
                        (source, target, method, seed)]
                rows.append({
                    'source': source,
                    'target': target,
                    'method': method,
                    'seed': seed,
                    'n_train': len(indices['train']),
                    'n_val': len(indices['val']),
                    'n_test': len(indices['test']),
                    'test_positives': int(bundle.y[indices['test']].sum()),
                    'source_val_AP': training['source_val']['AP'],
                    **result['metrics'],
                    'protocol': split['protocol'],
                })
                write_json(args.out / 'runs.json', rows)
                print(f"{direction} {method} seed={seed} test_AP={result['metrics']['AP']:.4f}", flush=True)

        summaries = []
        direction_rows = [row for row in rows if row['source'] == source and row['target'] == target]
        for method in methods:
            subset = [row for row in direction_rows if row['method'] == method]
            if len(subset) == len(args.seeds):
                summaries.append({
                    'method': method,
                    'AP_mean': float(np.mean([row['AP'] for row in subset])),
                    'AP_std': float(np.std([row['AP'] for row in subset], ddof=1)) if len(subset) > 1 else 0.0,
                    'AUROC_mean': float(np.mean([row['AUROC'] for row in subset])),
                })
        ensembles = {}
        for method in methods:
            labels, scores, meta = ensemble(direction_root, method, args.seeds)
            ensembles[method] = (labels, scores, meta)
        comparisons = {}
        for method, baseline in [('dual_no_pair', 'geometry'),
                                 ('full_selected', 'dual_no_pair'),
                                 ('full_selected', 'weighted_no_pair')]:
            labels, scores, meta = ensembles[method]
            other = ensembles[baseline][1]
            comparisons[f'{method}_minus_{baseline}'] = bootstrap(
                labels, scores, [group_key(row) for row in meta], args.bootstrap, other=other)
        write_json(direction_root / 'summary.json', {
            'source': source,
            'target': target,
            'protocol': split['protocol'],
            'n_train': len(indices['train']),
            'n_val': len(indices['val']),
            'n_test': len(indices['test']),
            'test_positives': int(bundle.y[indices['test']].sum()),
            'methods': summaries,
            'ensemble_comparisons': comparisons,
            'frozen_protocol_digest': lock_payload['lock_digest'],
        })


if __name__ == '__main__':
    main()
