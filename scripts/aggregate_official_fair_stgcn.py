#!/usr/bin/env python3
"""Aggregate five official ST-GCN seeds and compare them with MLP and ACIE ensembles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


DIRECTIONS = ["HUI360_to_SSUP-A", "SSUP-A_to_HUI360"]
METHODS = ["geometry", "dual_no_pair", "weighted_no_pair", "full_selected"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aligned_ensemble(paths: list[Path]) -> tuple[list[str], np.ndarray, np.ndarray, list[dict], np.ndarray]:
    tables = [{row["sample_id"]: row for row in read_jsonl(path)} for path in paths]
    ids = sorted(tables[0])
    if any(len(table) != len(ids) or sorted(table) != ids for table in tables):
        raise ValueError("Prediction IDs differ across seeds or contain duplicates")
    labels = np.asarray([int(tables[0][sample_id]["label"]) for sample_id in ids], dtype=int)
    if any(any(int(table[sample_id]["label"]) != labels[i] for i, sample_id in enumerate(ids)) for table in tables[1:]):
        raise ValueError("Labels differ across seed predictions")
    matrix = np.asarray([[float(table[sample_id]["score"]) for sample_id in ids] for table in tables])
    return ids, labels, matrix.mean(axis=0), [tables[0][sample_id] for sample_id in ids], matrix


def bootstrap_difference(labels: np.ndarray, first: np.ndarray, second: np.ndarray,
                         groups: list[str], repeats: int, seed: int) -> dict:
    group_array = np.asarray(groups)
    unique = sorted(set(groups))
    group_indices = {group: np.flatnonzero(group_array == group) for group in unique}
    rng = np.random.default_rng(seed)
    values = []
    attempts = 0
    while len(values) < repeats and attempts < repeats * 20:
        attempts += 1
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled])
        if np.unique(labels[indices]).size < 2:
            continue
        values.append(average_precision_score(labels[indices], first[indices]) - average_precision_score(labels[indices], second[indices]))
    if len(values) != repeats:
        raise RuntimeError(f"Only {len(values)} valid bootstrap replicates")
    return {
        "metric": "AP difference",
        "point": float(average_precision_score(labels, first) - average_precision_score(labels, second)),
        "groups": len(unique), "replicates": repeats,
        "interval_95pct": [float(x) for x in np.percentile(values, [2.5, 97.5])],
        "bootstrap_seed": seed,
        "warning": "Few groups: percentile interval may be unstable",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    root = args.root.resolve()
    official_root = root / "outputs" / "official_baselines" / "fair_v1"
    summary = {"schema": "acie.official-fair-stgcn-five-seed-summary.v1", "model": "stgcn", "seeds": args.seeds, "aggregation": "arithmetic mean of per-seed probabilities", "directions": {}}

    for direction_index, direction in enumerate(DIRECTIONS):
        paths = [official_root / direction / "stgcn" / f"seed{seed}" / "predictions.jsonl" for seed in args.seeds]
        ids, labels, scores, metadata, matrix = aligned_ensemble(paths)
        per_seed = []
        for seed, row_scores in zip(args.seeds, matrix, strict=True):
            metrics = json.loads((official_root / direction / "stgcn" / f"seed{seed}" / "metrics.json").read_text(encoding="utf-8"))
            calculated = {"AP": float(average_precision_score(labels, row_scores)), "AUROC": float(roc_auc_score(labels, row_scores))}
            if metrics.get("model") != "stgcn" or metrics.get("seed") != seed or not np.allclose([metrics["AP"], metrics["AUROC"]], [calculated["AP"], calculated["AUROC"]], rtol=0, atol=1e-12):
                raise ValueError(f"Metric or identity mismatch for {direction} seed {seed}")
            per_seed.append({"seed": seed, **calculated, "checkpoint_sha256": metrics["checkpoint_sha256"], "predictions_sha256": metrics["predictions_sha256"]})

        ensemble_path = official_root / direction / "stgcn" / "five_seed_ensemble_predictions.jsonl"
        ensemble_rows = []
        for sample_id, label, score, meta, seed_scores in zip(ids, labels, scores, metadata, matrix.T, strict=True):
            ensemble_rows.append({"sample_id": sample_id, "dataset": meta["dataset"], "recording": meta["recording"], "group_id": meta["group_id"], "track_id": meta["track_id"], "label": int(label), "score": float(score), "seed_scores": {str(seed): float(value) for seed, value in zip(args.seeds, seed_scores, strict=True)}})
        ensemble_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ensemble_rows), encoding="utf-8")

        comparisons = {}
        comparison_sets = {}
        for method in METHODS:
            comparison_sets[method] = [root / "outputs" / "frozen_cross_domain" / direction / method / f"seed{seed}" / "test_predictions.jsonl" for seed in args.seeds]
        comparison_sets["official_mlp"] = [official_root / direction / "mlp" / f"seed{seed}" / "predictions.jsonl" for seed in args.seeds]
        comparison_sets["official_lstm"] = [official_root / direction / "lstm" / f"seed{seed}" / "predictions.jsonl" for seed in args.seeds]
        for comparison_index, (name, comparison_paths) in enumerate(comparison_sets.items()):
            other_ids, other_labels, other_scores, _, _ = aligned_ensemble(comparison_paths)
            if other_ids != ids or not np.array_equal(other_labels, labels):
                raise ValueError(f"ST-GCN/{name} sample mismatch for {direction}")
            comparisons[f"official_stgcn_minus_{name}"] = bootstrap_difference(labels, scores, other_scores, [row["group_id"] for row in metadata], args.bootstrap, 20261114 + direction_index * 10 + comparison_index)

        aps = [row["AP"] for row in per_seed]
        aucs = [row["AUROC"] for row in per_seed]
        summary["directions"][direction] = {
            "n_test": len(ids), "test_positives": int(labels.sum()), "per_seed": per_seed,
            "single_seed": {"AP_mean": float(np.mean(aps)), "AP_std": float(np.std(aps, ddof=1)), "AUROC_mean": float(np.mean(aucs)), "AUROC_std": float(np.std(aucs, ddof=1))},
            "ensemble": {"AP": float(average_precision_score(labels, scores)), "AUROC": float(roc_auc_score(labels, scores)), "predictions": str(ensemble_path.relative_to(root)), "predictions_sha256": sha256(ensemble_path)},
            "comparisons": comparisons,
        }

    output = official_root / "five_seed_stgcn_summary.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
