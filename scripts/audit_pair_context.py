"""Audit date/recording context in training pairs without scoring test samples."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency

from acie.data import Bundle, group_key, resolve_split, track_key
from acie.io import read_json, read_jsonl, write_json


def recording_key(meta: dict) -> str:
    return f"{meta['dataset']}::{meta['recording']}"


def association(meta: list[dict], labels: np.ndarray, key_fn) -> dict:
    counts: dict[str, Counter] = defaultdict(Counter)
    for row, label in zip(meta, labels):
        counts[key_fn(row)][int(label)] += 1
    table = np.asarray([[value[0], value[1]] for value in counts.values()], dtype=float)
    chi2, p_value, _, _ = chi2_contingency(table, correction=False)
    denom = min(table.shape[0] - 1, table.shape[1] - 1)
    cramers_v = float(np.sqrt(chi2 / (table.sum() * denom))) if denom > 0 else 0.0
    units = []
    for name, value in sorted(counts.items()):
        total = value[0] + value[1]
        units.append({
            'context': name,
            'n': total,
            'positives': value[1],
            'prevalence': value[1] / total,
        })
    prevalence = [row['prevalence'] for row in units]
    return {
        'n_contexts': len(units),
        'cramers_v': cramers_v,
        'chi_square_p_value': float(p_value),
        'prevalence_min': min(prevalence),
        'prevalence_max': max(prevalence),
        'contexts_without_positives': sum(row['positives'] == 0 for row in units),
        'units': units,
    }


def expected_same_context(meta: list[dict], labels: np.ndarray, key_fn) -> float:
    positive = Counter(key_fn(row) for row, label in zip(meta, labels) if label == 1)
    negative = Counter(key_fn(row) for row, label in zip(meta, labels) if label == 0)
    n_positive = sum(positive.values())
    n_negative = sum(negative.values())
    return float(sum((positive[key] / n_positive) * (negative[key] / n_negative)
                     for key in set(positive) | set(negative)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--splits-root', type=Path, required=True)
    parser.add_argument('--pairs-root', type=Path, required=True)
    parser.add_argument('--domains', nargs='+', required=True)
    parser.add_argument('--quantile', type=float, default=0.75)
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    bundle = Bundle.load(args.data)
    sample = {row['sample_id']: (row, int(label)) for row, label in zip(bundle.meta, bundle.y)}
    report = {
        'schema': 'acie.pair-context-audit.v1',
        'protocol': 'training pairs only; test not scored',
        'test_evaluated': False,
        'quantile': args.quantile,
        'seeds': args.seeds,
        'domains': {},
    }
    for domain in args.domains:
        split = read_json(args.splits_root / domain / 'split.json')
        train_indices = resolve_split(bundle, split)['train']
        train_meta = [bundle.meta[i] for i in train_indices]
        train_labels = bundle.y[train_indices]
        train_ids = {row['sample_id'] for row in train_meta}
        available_fields = sorted(set.intersection(*(set(row) for row in train_meta)))
        key_fns = {'date_group': group_key, 'recording': recording_key}
        domain_report = {
            'n_train': len(train_indices),
            'n_positive': int(train_labels.sum()),
            'available_metadata_fields': available_fields,
            'explicit_scene_field_available': 'scene' in available_fields,
            'label_context_association': {
                name: association(train_meta, train_labels, key_fn)
                for name, key_fn in key_fns.items()
            },
            'expected_same_context_under_independent_label_sampling': {
                name: expected_same_context(train_meta, train_labels, key_fn)
                for name, key_fn in key_fns.items()
            },
            'runs': [],
        }
        positive_by_group = Counter(group_key(row) for row, label in zip(train_meta, train_labels) if label == 1)
        matched_by_group: dict[str, set[str]] = defaultdict(set)
        for seed in args.seeds:
            pair_file = (args.pairs_root / domain / f'quantile{args.quantile:g}' /
                         f'seed{seed}' / 'model' / 'pairs.train.jsonl')
            pairs = read_jsonl(pair_file)
            if not pairs:
                raise ValueError(f'No pairs in {pair_file}')
            endpoint_ids = {row[key] for row in pairs for key in ['positive', 'negative']}
            outside_train = sorted(endpoint_ids - train_ids)
            same = Counter()
            tracks_equal = 0
            distances = []
            weights = []
            for pair in pairs:
                positive, positive_label = sample[pair['positive']]
                negative, negative_label = sample[pair['negative']]
                if positive_label != 1 or negative_label != 0:
                    raise ValueError('Pair endpoint labels are invalid')
                same['date_group'] += group_key(positive) == group_key(negative)
                same['recording'] += recording_key(positive) == recording_key(negative)
                tracks_equal += track_key(positive) == track_key(negative)
                matched_by_group[group_key(positive)].add(positive['sample_id'])
                distances.append(float(pair['distance']))
                weights.append(float(pair['weight']))
            run = {
                'seed': seed,
                'n_pairs': len(pairs),
                'n_unique_positives': len({row['positive'] for row in pairs}),
                'positive_coverage': len({row['positive'] for row in pairs}) / int(train_labels.sum()),
                'same_date_group_fraction': same['date_group'] / len(pairs),
                'same_recording_fraction': same['recording'] / len(pairs),
                'same_track_pairs': tracks_equal,
                'outside_train_endpoints': outside_train,
                'mean_distance': float(np.mean(distances)),
                'mean_weight': float(np.mean(weights)),
            }
            domain_report['runs'].append(run)
        for field in ['positive_coverage', 'same_date_group_fraction', 'same_recording_fraction',
                      'mean_distance', 'mean_weight']:
            values = [run[field] for run in domain_report['runs']]
            domain_report[f'{field}_mean'] = float(np.mean(values))
            domain_report[f'{field}_std'] = float(np.std(values, ddof=1))
        domain_report['same_date_enrichment_over_independent'] = (
            domain_report['same_date_group_fraction_mean'] /
            domain_report['expected_same_context_under_independent_label_sampling']['date_group'])
        domain_report['same_recording_enrichment_over_independent'] = (
            domain_report['same_recording_fraction_mean'] /
            domain_report['expected_same_context_under_independent_label_sampling']['recording'])
        domain_report['positive_coverage_by_date_group_union_across_seeds'] = [
            {
                'context': key,
                'positives': count,
                'matched_unique_positives': len(matched_by_group[key]),
                'coverage': len(matched_by_group[key]) / count,
            }
            for key, count in sorted(positive_by_group.items())
        ]
        if any(run['outside_train_endpoints'] or run['same_track_pairs'] for run in domain_report['runs']):
            raise ValueError(f'Pair leakage or same-track pair detected in {domain}')
        report['domains'][domain] = domain_report
    write_json(args.out, report)
    print(f'Wrote training-only context audit to {args.out}')


if __name__ == '__main__':
    main()
