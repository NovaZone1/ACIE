#!/usr/bin/env python3
"""Summarize R5 behavior correspondence, confidence, and class-weight diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from acie.io import read_jsonl, write_json
from acie.vv_horizon import paired_date_seed_bootstrap

DIRECTIONS = ("HUI360_to_SSUP-A", "SSUP-A_to_HUI360")
SEEDS = (11, 22, 33, 44, 55)
PERMUTATIONS = (20260916, 20260917, 20260918)


def matrix(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loaded = [read_jsonl(path) for path in paths]
    reference_ids = [row["sample_id"] for row in loaded[0]]
    if any([row["sample_id"] for row in rows] != reference_ids for rows in loaded[1:]):
        raise ValueError("R5 prediction IDs differ")
    y = np.asarray([row["label"] for row in loaded[0]], dtype=int)
    if any(not np.array_equal(y, [row["label"] for row in rows]) for rows in loaded[1:]):
        raise ValueError("R5 labels differ")
    scores = np.asarray([[row["score"] for row in rows] for rows in loaded], dtype=float)
    groups = np.asarray([str(row.get("group_id", row["recording"])) for row in loaded[0]])
    return y, scores, groups


def compare(y: np.ndarray, groups: np.ndarray, a: np.ndarray, b: np.ndarray,
            name: str, repeats: int, seed: int) -> dict:
    differences = [float(average_precision_score(y, a[i]) - average_precision_score(y, b[i]))
                   for i in range(5)]
    return {
        "comparison": name, "seed_AP_differences": differences,
        "mean_AP_difference": float(np.mean(differences)),
        "positive_seeds": int(np.sum(np.asarray(differences) > 0)),
        "bootstrap": paired_date_seed_bootstrap(y, groups, a, b, repeats, seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root_default = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument("--output", type=Path, default=Path("outputs/vv_followup_v1/r5"))
    parser.add_argument("--r1", type=Path, default=Path("outputs/vv_followup_v1/r1"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    root = args.root.resolve()
    resolve = lambda value: value if value.is_absolute() else root / value
    output, r1 = map(resolve, (args.output, args.r1))
    summary = {"schema": "acie.vv-r5-summary.v1", "bootstrap_repeats": args.bootstrap,
               "directions": {}}
    flat = []
    for direction_index, direction in enumerate(DIRECTIONS):
        real_paths = [r1 / direction / "F" / "fixed" / str(seed) / "test_predictions.jsonl" for seed in SEEDS]
        y, real_f, groups = matrix(real_paths)
        _, real_jw, _ = matrix([r1 / direction / "Jw" / "fixed" / str(seed) / "test_predictions.jsonl" for seed in SEEDS])
        result = {"r5a": [], "r5b": None, "r5c": []}
        for permutation_index, permutation_seed in enumerate(PERMUTATIONS):
            _, permuted, _ = matrix([
                output / "r5a" / direction / f"perm_{permutation_seed}" / "F" / str(seed) / "test_predictions.jsonl"
                for seed in SEEDS])
            item = compare(y, groups, real_f, permuted, f"F_real-F_perm_{permutation_seed}",
                           args.bootstrap, 20260916 + direction_index * 100 + permutation_index)
            item["permutation_seed"] = permutation_seed
            result["r5a"].append(item)
        r5a_values = [item["mean_AP_difference"] for item in result["r5a"]]
        result["r5a_range"] = [float(min(r5a_values)), float(max(r5a_values))]
        _, confidence, _ = matrix([
            output / "r5b" / direction / "F_conf" / str(seed) / "test_predictions.jsonl" for seed in SEEDS])
        result["r5b"] = compare(y, groups, real_f, confidence, "F_real-F_conf", args.bootstrap,
                                20261016 + direction_index * 100)
        _, weighted_f, _ = matrix([
            output / "r5c" / direction / "F_weighted" / str(seed) / "test_predictions.jsonl" for seed in SEEDS])
        _, weighted_jw, _ = matrix([
            output / "r5c" / direction / "Jw_weighted" / str(seed) / "test_predictions.jsonl" for seed in SEEDS])
        comparisons = (
            (real_f, real_jw, "F-Jw_unweighted"),
            (weighted_f, weighted_jw, "F-Jw_weighted"),
            (weighted_f, real_f, "F_weighted-F_unweighted"),
            (weighted_jw, real_jw, "Jw_weighted-Jw_unweighted"),
        )
        for comparison_index, (first, second, name) in enumerate(comparisons):
            result["r5c"].append(compare(y, groups, first, second, name, args.bootstrap,
                                         20261116 + direction_index * 100 + comparison_index))
        summary["directions"][direction] = result
        for family in (result["r5a"], [result["r5b"]], result["r5c"]):
            for item in family:
                interval = item["bootstrap"]["joint_date_and_seed"]["interval_95"]
                flat.append({"direction": direction, "comparison": item["comparison"],
                             "mean_AP_difference": item["mean_AP_difference"],
                             "positive_seeds": item["positive_seeds"],
                             "interval_95_low": interval[0], "interval_95_high": interval[1]})
    write_json(output / "r5_summary.json", summary)
    with (output / "r5_differences.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader(); writer.writerows(flat)
    lines = ["# R5 behavior diagnostics", "",
             "| Direction | Comparison | Mean AP difference | Positive seeds | 95% interval |",
             "|---|---|---:|---:|---:|"]
    for row in flat:
        lines.append(f"| {row['direction']} | {row['comparison']} | {row['mean_AP_difference']:.6f} | {row['positive_seeds']}/5 | [{row['interval_95_low']:.6f}, {row['interval_95_high']:.6f}] |")
    (output / "R5_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "comparisons": len(flat),
                      "summary": str(output / "r5_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
