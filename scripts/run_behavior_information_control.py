#!/usr/bin/env python3
"""Run the frozen, capacity-matched permuted-behavior control."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from acie.data import Bundle, group_key, resolve_split
from acie.engine import predict, train
from acie.io import digest_file, read_json, read_jsonl, write_json, write_jsonl


DIRECTIONS = ["HUI360_to_SSUP-A", "SSUP-A_to_HUI360"]


def load_control_bundle(bundle: Bundle, split: dict, mapping_path: Path) -> Bundle:
    ids = {row["sample_id"]: index for index, row in enumerate(bundle.meta)}
    allowed = {part: set(indices.tolist()) for part, indices in resolve_split(bundle, split).items()}
    q = bundle.q.copy()
    rows = read_jsonl(mapping_path)
    seen = set()
    for row in rows:
        recipient = ids[row["recipient_sample_id"]]
        donor = ids[row["donor_sample_id"]]
        part = row["part"]
        if recipient in seen or recipient == donor:
            raise ValueError("Mapping has duplicate recipient or fixed point")
        if recipient not in allowed[part] or donor not in allowed[part]:
            raise ValueError("Mapping crosses a frozen split")
        if group_key(bundle.meta[recipient]) != row["group_id"] or group_key(bundle.meta[donor]) != row["group_id"]:
            raise ValueError("Mapping crosses a date group")
        q[recipient] = bundle.q[donor]
        seen.add(recipient)
    expected = set().union(*allowed.values())
    if seen != expected:
        raise ValueError("Mapping does not cover the exact frozen split")
    provenance = {
        "schema": "acie.behavior-information-control-bundle.v1",
        "parent": bundle.provenance,
        "mapping_sha256": digest_file(mapping_path),
        "transformation": "label-blind within-split within-date behavior derangement",
        "geometry_unchanged": True,
        "labels_unchanged": True,
        "synthetic": False,
    }
    return Bundle(bundle.a, q, bundle.y, bundle.meta, provenance)


def prediction_table(path: Path) -> dict[str, dict]:
    rows = read_jsonl(path)
    table = {row["sample_id"]: row for row in rows}
    if len(table) != len(rows):
        raise ValueError(f"Duplicate prediction IDs: {path}")
    return table


def ensemble(paths: list[Path]) -> tuple[list[str], np.ndarray, np.ndarray, list[dict], np.ndarray]:
    tables = [prediction_table(path) for path in paths]
    ids = sorted(tables[0])
    if any(sorted(table) != ids for table in tables):
        raise ValueError("Prediction IDs differ")
    labels = np.asarray([int(tables[0][sample_id]["label"]) for sample_id in ids], dtype=int)
    if any(any(int(table[sample_id]["label"]) != labels[i] for i, sample_id in enumerate(ids)) for table in tables[1:]):
        raise ValueError("Prediction labels differ")
    matrix = np.asarray([[float(table[sample_id]["score"]) for sample_id in ids] for table in tables])
    return ids, labels, matrix.mean(axis=0), [tables[0][sample_id] for sample_id in ids], matrix


def bootstrap_difference(labels: np.ndarray, first: np.ndarray, second: np.ndarray,
                         groups: list[str], repeats: int, seed: int) -> dict:
    group_array = np.asarray(groups)
    unique = sorted(set(groups))
    indices = {group: np.flatnonzero(group_array == group) for group in unique}
    rng = np.random.default_rng(seed)
    values = []
    attempts = 0
    while len(values) < repeats and attempts < repeats * 20:
        attempts += 1
        sampled = rng.choice(unique, size=len(unique), replace=True)
        chosen = np.concatenate([indices[group] for group in sampled])
        if np.unique(labels[chosen]).size < 2:
            continue
        values.append(float(average_precision_score(labels[chosen], first[chosen]) - average_precision_score(labels[chosen], second[chosen])))
    if len(values) != repeats:
        raise RuntimeError(f"Only {len(values)} valid bootstrap replicates")
    return {
        "metric": "AP difference",
        "point": float(average_precision_score(labels, first) - average_precision_score(labels, second)),
        "groups": len(unique),
        "replicates": repeats,
        "interval_95pct": [float(value) for value in np.percentile(values, [2.5, 97.5])],
        "bootstrap_seed": seed,
        "warning": "Few groups: percentile interval may be unstable" if len(unique) < 10 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    root = args.root.resolve()
    args.data = args.data.resolve()
    args.frozen_root = args.frozen_root.resolve()
    args.protocol = args.protocol.resolve()
    args.out = args.out.resolve()
    protocol = read_json(args.protocol)
    if protocol.get("schema") != "acie.behavior-information-control-protocol.v1":
        raise ValueError("Unexpected protocol")
    if args.seeds != protocol["seeds"] or digest_file(args.data) != protocol["data_sha256"]:
        raise ValueError("Seeds or data differ from frozen protocol")
    config_path = root / protocol["config"]
    if digest_file(config_path) != protocol["config_sha256"]:
        raise ValueError("Frozen model config changed")
    base_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if base_config.get("kind") != "dual_no_pair" or base_config.get("pair_lambda") != 0.0:
        raise ValueError("Control must use the unweighted dual_no_pair configuration")

    bundle = Bundle.load(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    for direction in DIRECTIONS:
        spec = protocol["directions"][direction]
        split_path = root / spec["split"]
        mapping_path = root / spec["mapping"]
        if digest_file(split_path) != spec["split_sha256"] or digest_file(mapping_path) != spec["mapping_sha256"]:
            raise ValueError(f"Frozen input changed for {direction}")
        split = read_json(split_path)
        control_bundle = load_control_bundle(bundle, split, mapping_path)
        direction_root = args.out / direction / "permuted_behavior"
        for seed in args.seeds:
            run = direction_root / f"seed{seed}"
            model_dir = run / "model"
            predictions = run / "test_predictions.jsonl"
            config = dict(base_config)
            config["seed"] = seed
            started = time.time()
            report_path = model_dir / "training_report.json"
            if report_path.is_file() and (model_dir / "best.pt").is_file():
                training = read_json(report_path)
            else:
                training = train(
                    control_bundle, split, config, model_dir,
                    resume=(model_dir / "run.json").is_file(),
                )
            metrics_path = predictions.with_suffix(".metrics.json")
            if metrics_path.is_file():
                result = read_json(metrics_path)
            else:
                result = predict(control_bundle, model_dir / "best.pt", predictions, "test", repeats=args.bootstrap)
            write_json(run / "run_provenance.json", {
                "schema": "acie.behavior-information-control-run.v1",
                "direction": direction,
                "seed": seed,
                "model": "dual_no_pair",
                "control": "within-split within-date permuted behavior",
                "mapping_sha256": spec["mapping_sha256"],
                "checkpoint_selection": "source validation AP only",
                "target_labels_used_for_selection": False,
                "source_val_AP": training["source_val"]["AP"],
                "target_AP": result["metrics"]["AP"],
                "target_AUROC": result["metrics"]["AUROC"],
                "wall_seconds_this_invocation": time.time() - started,
            })
            print(f"DONE {direction} seed={seed} AP={result['metrics']['AP']:.6f} AUROC={result['metrics']['AUROC']:.6f}", flush=True)

    summary = {
        "schema": "acie.behavior-information-control-summary.v1",
        "protocol_digest": protocol["protocol_digest"],
        "seeds": args.seeds,
        "directions": {},
    }
    supported = True
    for direction_index, direction in enumerate(DIRECTIONS):
        control_root = args.out / direction / "permuted_behavior"
        control_paths = [control_root / f"seed{seed}" / "test_predictions.jsonl" for seed in args.seeds]
        real_paths = [args.frozen_root / direction / "dual_no_pair" / f"seed{seed}" / "test_predictions.jsonl" for seed in args.seeds]
        geometry_paths = [args.frozen_root / direction / "geometry" / f"seed{seed}" / "test_predictions.jsonl" for seed in args.seeds]
        ids, labels, control_scores, metadata, control_matrix = ensemble(control_paths)
        real_ids, real_labels, real_scores, _, _ = ensemble(real_paths)
        geometry_ids, geometry_labels, geometry_scores, _, _ = ensemble(geometry_paths)
        if ids != real_ids or ids != geometry_ids or not np.array_equal(labels, real_labels) or not np.array_equal(labels, geometry_labels):
            raise ValueError("Control and frozen comparison samples differ")

        ensemble_path = control_root / "five_seed_ensemble_predictions.jsonl"
        write_jsonl(ensemble_path, [
            {
                "sample_id": sample_id,
                "dataset": meta["dataset"],
                "recording": meta["recording"],
                "group_id": meta.get("group_id"),
                "track_id": meta["track_id"],
                "label": int(label),
                "score": float(score),
                "seed_scores": {str(seed): float(value) for seed, value in zip(args.seeds, seed_scores, strict=True)},
            }
            for sample_id, meta, label, score, seed_scores in zip(ids, metadata, labels, control_scores, control_matrix.T, strict=True)
        ])
        groups = [group_key(row) for row in metadata]
        comparisons = {
            "real_dual_minus_permuted_behavior": bootstrap_difference(labels, real_scores, control_scores, groups, args.bootstrap, 20260915 + direction_index * 10),
            "permuted_behavior_minus_geometry": bootstrap_difference(labels, control_scores, geometry_scores, groups, args.bootstrap, 20260916 + direction_index * 10),
            "real_dual_minus_geometry": bootstrap_difference(labels, real_scores, geometry_scores, groups, args.bootstrap, 20260917 + direction_index * 10),
        }
        primary = comparisons["real_dual_minus_permuted_behavior"]
        direction_supported = primary["point"] > 0 and primary["interval_95pct"][0] > 0
        supported &= direction_supported
        per_seed = []
        for seed, scores in zip(args.seeds, control_matrix, strict=True):
            per_seed.append({
                "seed": seed,
                "AP": float(average_precision_score(labels, scores)),
                "AUROC": float(roc_auc_score(labels, scores)),
            })
        summary["directions"][direction] = {
            "n_test": len(ids),
            "test_positives": int(labels.sum()),
            "per_seed": per_seed,
            "single_seed": {
                "AP_mean": float(np.mean([row["AP"] for row in per_seed])),
                "AP_std": float(np.std([row["AP"] for row in per_seed], ddof=1)),
            },
            "ensemble": {
                "AP": float(average_precision_score(labels, control_scores)),
                "AUROC": float(roc_auc_score(labels, control_scores)),
                "predictions": str(ensemble_path.relative_to(root)),
                "predictions_sha256": digest_file(ensemble_path),
            },
            "comparisons": comparisons,
            "behavior_association_supported": direction_supported,
        }
    summary["behavior_association_supported_both_directions"] = bool(supported)
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
