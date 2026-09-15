"""Compare date-context pairing policies on source validation only."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import yaml

from acie.data import Bundle
from acie.engine import predict, train
from acie.io import read_json, write_json


def summarize(rows: list[dict], domains: list[str], policies: list[str], seeds: list[int]) -> list[dict]:
    summaries = []
    for domain in domains:
        baseline = {row['seed']: row for row in rows
                    if row['domain'] == domain and row['context_policy'] == 'any'}
        for policy in policies:
            subset = sorted([row for row in rows
                             if row['domain'] == domain and row['context_policy'] == policy],
                            key=lambda row: row['seed'])
            if len(subset) != len(seeds) or len(baseline) != len(seeds):
                continue
            delta = np.asarray([row['val_AP'] - baseline[row['seed']]['val_AP'] for row in subset])
            summaries.append({
                'domain': domain,
                'context_policy': policy,
                'n_runs': len(subset),
                'val_AP_mean': float(np.mean([row['val_AP'] for row in subset])),
                'val_AP_std': float(np.std([row['val_AP'] for row in subset], ddof=1)),
                'delta_from_any_mean': float(delta.mean()),
                'delta_from_any_by_seed': delta.tolist(),
                'positive_seeds_vs_any': int((delta > 0).sum()),
                'train_pair_coverage_mean': float(np.mean([row['train_pair_coverage'] for row in subset])),
                'train_same_date_fraction_mean': None if policy == 'any' else
                    float(np.mean([row['train_same_date_fraction'] for row in subset])),
            })
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--splits-root', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/full_source_selected.yaml'))
    parser.add_argument('--domains', nargs='+', required=True)
    parser.add_argument('--policies', nargs='+', default=['same_group', 'different_group'])
    parser.add_argument('--quantile', type=float, default=0.75)
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument('--bootstrap', type=int, default=300)
    args = parser.parse_args()
    allowed = {'same_group', 'different_group'}
    if not set(args.policies) <= allowed:
        raise ValueError(f'Policies must be drawn from {sorted(allowed)}')

    bundle = Bundle.load(args.data)
    base_config = yaml.safe_load(args.config.read_text())
    baseline_trials = read_json(args.baseline_root / 'trials.json')
    rows = []
    for row in baseline_trials:
        if row['domain'] in args.domains and row['quantile'] == args.quantile and row['seed'] in args.seeds:
            matching = read_json(args.baseline_root / row['domain'] / f"quantile{args.quantile:g}" /
                                 f"seed{row['seed']}" / 'model' / 'matching.json')
            rows.append({
                'domain': row['domain'],
                'context_policy': 'any',
                'seed': row['seed'],
                'val_AP': row['val_AP'],
                'val_AUROC': row['val_AUROC'],
                'train_pair_coverage': row['train_pair_coverage'],
                'train_same_date_fraction': None,
                'matching_radius': matching['config']['radius'],
                'test_evaluated': False,
                'new_training': False,
            })
    if len(rows) != len(args.domains) * len(args.seeds):
        raise ValueError('Incomplete unrestricted baseline rows')

    for domain in args.domains:
        split = read_json(args.splits_root / domain / 'split.json')
        for policy in args.policies:
            for seed in args.seeds:
                baseline = next(row for row in rows if row['domain'] == domain and
                                row['context_policy'] == 'any' and row['seed'] == seed)
                run = args.out / domain / policy / f'seed{seed}'
                run.mkdir(parents=True, exist_ok=True)
                config = copy.deepcopy(base_config)
                config['seed'] = seed
                config['matching'] = copy.deepcopy(config['matching'])
                config['matching'].update(
                    context_policy=policy,
                    radius=baseline['matching_radius'],
                    radius_quantile=args.quantile,
                )
                report_path = run / 'model' / 'training_report.json'
                if report_path.is_file() and (run / 'model' / 'best.pt').is_file():
                    report = read_json(report_path)
                else:
                    report = train(bundle, split, config, run / 'model',
                                   resume=(run / 'model' / 'run.json').is_file())
                metric_path = run / 'val_predictions.metrics.json'
                if metric_path.is_file():
                    validation = read_json(metric_path)
                else:
                    validation = predict(bundle, run / 'model' / 'best.pt',
                                         run / 'val_predictions.jsonl', 'val', repeats=args.bootstrap)
                matching = read_json(run / 'model' / 'matching.json')
                same_date = 1.0 if policy == 'same_group' else 0.0
                rows = [row for row in rows if (row['domain'], row['context_policy'], row['seed']) !=
                        (domain, policy, seed)]
                rows.append({
                    'domain': domain,
                    'context_policy': policy,
                    'seed': seed,
                    'val_AP': report['source_val']['AP'],
                    'val_AUROC': report['source_val']['AUROC'],
                    'train_pair_coverage': matching['positive_coverage'],
                    'train_n_pairs': matching['n_pairs'],
                    'train_same_date_fraction': same_date if matching['n_pairs'] else None,
                    'matching_radius': matching['config']['radius'],
                    'val_pair_coverage': validation['pair_diagnostic']['positive_coverage'],
                    'test_evaluated': False,
                    'new_training': True,
                })
                summaries = summarize(rows, args.domains, ['any'] + args.policies, args.seeds)
                joint = []
                for candidate in ['any'] + args.policies:
                    available = [row for row in summaries if row['context_policy'] == candidate]
                    if len(available) == len(args.domains):
                        joint.append({
                            'context_policy': candidate,
                            'mean_domain_delta_from_any': float(np.mean([
                                row['delta_from_any_mean'] for row in available])),
                        })
                write_json(args.out / 'trials.json', rows)
                write_json(args.out / 'source_validation_summary.json', {
                    'protocol': 'source validation only',
                    'test_evaluated': False,
                    'pair_lambda': base_config['pair_lambda'],
                    'radius_quantile': args.quantile,
                    'fixed_unrestricted_radius': True,
                    'seeds': args.seeds,
                    'summaries': summaries,
                    'joint_selection': max(joint, key=lambda row: row['mean_domain_delta_from_any']) if joint else None,
                })
                print(f"{domain} policy={policy} seed={seed} "
                      f"val_AP={report['source_val']['AP']:.4f} "
                      f"coverage={matching['positive_coverage']:.3f}", flush=True)


if __name__ == '__main__':
    main()
