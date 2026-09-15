"""Validate frozen cross-domain runs and write a compact evidence record."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from acie.data import Bundle, group_key, resolve_split
from acie.io import digest_file, read_json, read_jsonl, write_json
from acie.metrics import binary_metrics


def ensemble(run_root: Path, method: str, seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
    tables = []
    for seed in seeds:
        rows = read_jsonl(run_root / method / f"seed{seed}" / "test_predictions.jsonl")
        table = {row["sample_id"]: row for row in rows}
        if len(table) != len(rows):
            raise ValueError(f"Duplicate prediction IDs: {run_root}/{method}/seed{seed}")
        tables.append(table)
    ids = sorted(tables[0])
    if any(set(table) != set(ids) for table in tables):
        raise ValueError(f"Prediction IDs differ across seeds: {run_root}/{method}")
    labels = np.asarray([tables[0][sample_id]["label"] for sample_id in ids], dtype=int)
    scores = np.mean([[table[sample_id]["score"] for sample_id in ids] for table in tables], axis=0)
    return labels, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--preregistered", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    bundle = Bundle.load(args.data)
    prereg = read_json(args.preregistered)
    lock = read_json(args.runs / "FROZEN_PROTOCOL.json")
    seeds = prereg["seeds"]
    methods = list(prereg["methods"])
    if digest_file(args.data) != lock["data_sha256"]:
        raise ValueError("Frozen data hash mismatch")
    if lock["seeds"] != seeds or set(lock["methods"]) != set(methods):
        raise ValueError("Frozen seed or method set mismatch")
    for method, spec in prereg["methods"].items():
        if digest_file(spec["config"]) != spec["sha256"]:
            raise ValueError(f"Preregistered config changed: {method}")
        if lock["methods"][method]["sha256"] != spec["sha256"]:
            raise ValueError(f"Lock/preregistration mismatch: {method}")

    recorded_runs = read_json(args.runs / "runs.json")
    expected_keys = {
        (direction.split("_to_")[0], direction.split("_to_")[1], method, seed)
        for direction in prereg["directions"] for method in methods for seed in seeds
    }
    actual_keys = {(r["source"], r["target"], r["method"], r["seed"]) for r in recorded_runs}
    if len(recorded_runs) != len(actual_keys) or actual_keys != expected_keys:
        raise ValueError("Run matrix is incomplete or contains duplicate/unplanned runs")

    directions = {}
    behavior_supported = True
    pairing_positive = True
    id_to_index = {row["sample_id"]: i for i, row in enumerate(bundle.meta)}
    for direction in prereg["directions"]:
        source, target = direction.split("_to_")
        root = args.runs / direction
        split_path = root / "split.json"
        split = read_json(split_path)
        indices = resolve_split(bundle, split)
        if digest_file(split_path) != lock["split_sha256"][direction]:
            raise ValueError(f"Frozen split hash mismatch: {direction}")
        if any(bundle.meta[i]["pool"] != "train" for part in ("train", "val") for i in indices[part]):
            raise ValueError(f"Non-train row entered source development: {direction}")
        if any(bundle.meta[i]["pool"] != "test" for i in indices["test"]):
            raise ValueError(f"Non-test row entered target evaluation: {direction}")
        expected_ids = set(split["test"])
        for method in methods:
            for seed in seeds:
                run = root / method / f"seed{seed}"
                required = [run / "model/best.pt", run / "model/training_report.json",
                            run / "test_predictions.jsonl", run / "test_predictions.metrics.json"]
                if not all(path.is_file() for path in required):
                    raise ValueError(f"Missing run artifact: {direction}/{method}/seed{seed}")
                rows = read_jsonl(run / "test_predictions.jsonl")
                ids = [row["sample_id"] for row in rows]
                if len(ids) != len(set(ids)) or set(ids) != expected_ids:
                    raise ValueError(f"Prediction IDs mismatch: {direction}/{method}/seed{seed}")
                if any(int(row["label"]) != int(bundle.y[id_to_index[row["sample_id"]]]) for row in rows):
                    raise ValueError(f"Prediction labels mismatch: {direction}/{method}/seed{seed}")
                if not np.isfinite([row["score"] for row in rows]).all():
                    raise ValueError(f"Non-finite prediction: {direction}/{method}/seed{seed}")

        ensemble_metrics = {}
        ensemble_scores = {}
        for method in methods:
            labels, scores = ensemble(root, method, seeds)
            raw_metrics = binary_metrics(labels, scores)
            ensemble_metrics[method] = {
                key: raw_metrics[key] for key in ("n", "positives", "prevalence", "AP", "AUROC")
            }
            ensemble_scores[method] = scores
        comparisons = {
            "dual_no_pair_minus_geometry": ensemble_metrics["dual_no_pair"]["AP"] - ensemble_metrics["geometry"]["AP"],
            "full_selected_minus_dual_no_pair": ensemble_metrics["full_selected"]["AP"] - ensemble_metrics["dual_no_pair"]["AP"],
            "full_selected_minus_weighted_no_pair": ensemble_metrics["full_selected"]["AP"] - ensemble_metrics["weighted_no_pair"]["AP"],
        }
        summary = read_json(root / "summary.json")
        intervals = {name: value["interval"] for name, value in summary["ensemble_comparisons"].items()}
        if summary["frozen_protocol_digest"] != lock["lock_digest"]:
            raise ValueError(f"Summary/lock mismatch: {direction}")
        behavior_supported &= comparisons["dual_no_pair_minus_geometry"] > 0 and intervals["dual_no_pair_minus_geometry"][0] > 0
        pairing_positive &= comparisons["full_selected_minus_weighted_no_pair"] > 0
        directions[direction] = {
            "source": source,
            "target": target,
            "counts": {part: len(indices[part]) for part in ("train", "val", "test")},
            "positives": {part: int(bundle.y[indices[part]].sum()) for part in ("train", "val", "test")},
            "groups": {part: len({group_key(bundle.meta[i]) for i in indices[part]}) for part in ("train", "val", "test")},
            "ensemble_metrics": ensemble_metrics,
            "ensemble_AP_differences": comparisons,
            "bootstrap_95pct_intervals": intervals,
        }

    write_json(args.out, {
        "schema": "acie.frozen-cross-domain-validation.v1",
        "date": "2026-09-14",
        "status": "passed",
        "runs_expected": len(expected_keys),
        "runs_validated": len(actual_keys),
        "data_sha256": lock["data_sha256"],
        "frozen_protocol_digest": lock["lock_digest"],
        "preregistered_sha256": digest_file(args.preregistered),
        "checks": {
            "config_hashes_match_preregistration": True,
            "split_hashes_match_frozen_lock": True,
            "train_val_test_group_disjoint": True,
            "source_development_uses_train_pool_only": True,
            "target_evaluation_uses_test_pool_only": True,
            "prediction_ids_and_labels_match_frozen_test": True,
            "all_scores_finite": True,
        },
        "preregistered_pairing_support_rule_passed": bool(pairing_positive),
        "dual_behavior_residual_supported_both_directions": bool(behavior_supported),
        "directions": directions,
    })


if __name__ == "__main__":
    main()
