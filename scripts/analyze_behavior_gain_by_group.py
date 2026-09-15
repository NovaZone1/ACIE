#!/usr/bin/env python3
"""Decompose frozen behavior gains by target date and leave-one-date-out influence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from acie.io import read_jsonl, write_json


DIRECTIONS = ["HUI360_to_SSUP-A", "SSUP-A_to_HUI360"]
SEEDS = [11, 22, 33, 44, 55]


def ensemble(paths: list[Path]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    tables = []
    for path in paths:
        rows = read_jsonl(path)
        table = {row["sample_id"]: row for row in rows}
        if len(table) != len(rows):
            raise ValueError(f"Duplicate sample IDs: {path}")
        tables.append(table)
    ids = sorted(tables[0])
    if any(sorted(table) != ids for table in tables):
        raise ValueError("Prediction IDs differ across seeds")
    labels = np.asarray([int(tables[0][sample_id]["label"]) for sample_id in ids], dtype=int)
    scores = np.asarray([[float(table[sample_id]["score"]) for sample_id in ids] for table in tables]).mean(axis=0)
    evidence = np.asarray([[float(table[sample_id].get("evidence", 0.0)) for sample_id in ids] for table in tables]).mean(axis=0)
    return ids, labels, scores, evidence, [tables[0][sample_id] for sample_id in ids]


def ap(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(average_precision_score(labels, scores))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    payload = {
        "schema": "acie.behavior-gain-group-analysis.v1",
        "status": "descriptive_post_primary_result_analysis",
        "seeds": SEEDS,
        "directions": {},
    }

    for direction in DIRECTIONS:
        real_paths = [root / "outputs" / "frozen_cross_domain" / direction / "dual_no_pair" / f"seed{seed}" / "test_predictions.jsonl" for seed in SEEDS]
        geometry_paths = [root / "outputs" / "frozen_cross_domain" / direction / "geometry" / f"seed{seed}" / "test_predictions.jsonl" for seed in SEEDS]
        control_paths = [root / "outputs" / "behavior_information_control" / direction / "permuted_behavior" / f"seed{seed}" / "test_predictions.jsonl" for seed in SEEDS]
        ids, labels, real, evidence, metadata = ensemble(real_paths)
        geometry_ids, geometry_labels, geometry, _, _ = ensemble(geometry_paths)
        control_ids, control_labels, control, _, _ = ensemble(control_paths)
        if ids != geometry_ids or ids != control_ids or not np.array_equal(labels, geometry_labels) or not np.array_equal(labels, control_labels):
            raise ValueError("Comparison prediction samples differ")
        groups = np.asarray([str(row["group_id"]) for row in metadata])
        unique = sorted(set(groups))
        rows = []
        leave_one_out = []
        for group in unique:
            inside = groups == group
            outside = ~inside
            if np.unique(labels[inside]).size < 2 or np.unique(labels[outside]).size < 2:
                raise ValueError(f"Both classes required for group analysis: {direction}/{group}")
            rows.append({
                "group_id": group,
                "samples": int(inside.sum()),
                "positives": int(labels[inside].sum()),
                "AP": {
                    "geometry": ap(labels[inside], geometry[inside]),
                    "permuted_behavior": ap(labels[inside], control[inside]),
                    "real_dual": ap(labels[inside], real[inside]),
                },
                "differences": {
                    "real_dual_minus_permuted_behavior": ap(labels[inside], real[inside]) - ap(labels[inside], control[inside]),
                    "real_dual_minus_geometry": ap(labels[inside], real[inside]) - ap(labels[inside], geometry[inside]),
                },
            })
            leave_one_out.append({
                "excluded_group": group,
                "remaining_samples": int(outside.sum()),
                "real_dual_minus_permuted_behavior_AP": ap(labels[outside], real[outside]) - ap(labels[outside], control[outside]),
                "real_dual_minus_geometry_AP": ap(labels[outside], real[outside]) - ap(labels[outside], geometry[outside]),
            })
        primary_loo = [row["real_dual_minus_permuted_behavior_AP"] for row in leave_one_out]
        payload["directions"][direction] = {
            "groups": len(unique),
            "per_group": rows,
            "positive_per_group_real_minus_permuted": sum(row["differences"]["real_dual_minus_permuted_behavior"] > 0 for row in rows),
            "leave_one_group_out": leave_one_out,
            "leave_one_group_out_real_minus_permuted_range": [float(min(primary_loo)), float(max(primary_loo))],
            "all_leave_one_group_out_real_minus_permuted_positive": all(value > 0 for value in primary_loo),
            "real_behavior_residual_only": {
                "AP": ap(labels, evidence),
                "AUROC": float(roc_auc_score(labels, evidence)),
                "mean_positive_logit": float(evidence[labels == 1].mean()),
                "mean_negative_logit": float(evidence[labels == 0].mean()),
            },
        }
    write_json(args.out, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
