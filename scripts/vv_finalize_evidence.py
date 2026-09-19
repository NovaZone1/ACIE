#!/usr/bin/env python3
"""Generate the low-cost method-paper evidence package from locked predictions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

from acie.data import Bundle, resolve_split
from acie.io import read_json, read_jsonl, write_json, write_jsonl
from acie.metrics import binary_metrics, choose_threshold
from acie.vv_provenance import validate_locked_checkpoint


DIRECTIONS = {
    "HUI360_to_SSUP-A": "outputs/frozen_cross_domain/HUI360_to_SSUP-A/split.json",
    "SSUP-A_to_HUI360": "outputs/frozen_cross_domain/SSUP-A_to_HUI360/split.json",
}
SEEDS = (11, 22, 33, 44, 55)
NEURAL = ("G", "Q", "C64", "Cm", "Jc", "Jw", "JwD", "F")
PERMUTATIONS = (20260916, 20260917, 20260918)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def prediction_dir(base: Path, direction: str, model: str, seed: int) -> Path:
    return base / direction / model / "fixed" / str(seed)


def load_locked_matrix(base: Path, lock_path: Path, direction: str, model: str,
                       seeds: tuple[int, ...], expected_ids: list[str] | None = None,
                       expected_y: np.ndarray | None = None) -> tuple[list[str], np.ndarray, np.ndarray, list[list[dict]]]:
    loaded = []
    for seed in seeds:
        run = prediction_dir(base, direction, model, seed)
        checkpoint = run / ("tree.joblib" if model in ("histgb", "random_forest") else "best.pt")
        entry = validate_locked_checkpoint(lock_path, checkpoint, direction=direction,
                                           model_id=model, seed=seed)
        rows = read_jsonl(run / "test_predictions.jsonl")
        if not rows or any(row.get("checkpoint_hash") != entry["sha256"] for row in rows):
            raise ValueError(f"Prediction/checkpoint mismatch: {direction}/{model}/{seed}")
        loaded.append(rows)
    ids = [row["sample_id"] for row in loaded[0]]
    y = np.asarray([row["label"] for row in loaded[0]], dtype=int)
    if any([row["sample_id"] for row in rows] != ids for rows in loaded):
        raise ValueError(f"Seed prediction coverage differs: {direction}/{model}")
    if expected_ids is not None and ids != expected_ids:
        raise ValueError(f"Model prediction coverage differs: {direction}/{model}")
    if expected_y is not None and not np.array_equal(y, expected_y):
        raise ValueError(f"Model labels differ: {direction}/{model}")
    return ids, y, np.asarray([[row["score"] for row in rows] for rows in loaded]), loaded


def load_unlocked_matrix(paths: list[Path], expected_ids: list[str], expected_y: np.ndarray) -> np.ndarray:
    loaded = [read_jsonl(path) for path in paths]
    if any([row["sample_id"] for row in rows] != expected_ids for rows in loaded):
        raise ValueError(f"Diagnostic prediction coverage differs: {paths[0].parent}")
    if any(not np.array_equal(np.asarray([row["label"] for row in rows], dtype=int), expected_y)
           for rows in loaded):
        raise ValueError(f"Diagnostic labels differ: {paths[0].parent}")
    return np.asarray([[row["score"] for row in rows] for rows in loaded])


def seed_differences(y: np.ndarray, first: np.ndarray, second: np.ndarray,
                     indices: np.ndarray | None = None) -> np.ndarray:
    indices = np.arange(len(y)) if indices is None else indices
    if len(np.unique(y[indices])) < 2:
        return np.asarray([])
    count = max(len(first), len(second))
    if len(first) not in (1, count) or len(second) not in (1, count):
        raise ValueError("Incompatible seed structures")
    return np.asarray([
        average_precision_score(y[indices], first[0 if len(first) == 1 else i, indices]) -
        average_precision_score(y[indices], second[0 if len(second) == 1 else i, indices])
        for i in range(count)
    ], dtype=float)


class GroupedAveragePrecision:
    """Exact weighted AP from pre-aggregated score-tie blocks and date groups."""

    def __init__(self, y: np.ndarray, groups: np.ndarray, scores: np.ndarray):
        self.unique_groups = np.unique(groups)
        group_index = np.searchsorted(self.unique_groups, groups)
        self.group_positives = np.bincount(group_index, weights=y,
                                           minlength=len(self.unique_groups))
        self.group_totals = np.bincount(group_index, minlength=len(self.unique_groups))
        self.blocks = []
        self.cache: list[dict[tuple[int, ...], float]] = []
        for values in scores:
            _, block_index = np.unique(-values, return_inverse=True)
            block_count = int(block_index.max()) + 1
            totals = np.zeros((block_count, len(self.unique_groups)), dtype=float)
            positives = np.zeros_like(totals)
            np.add.at(totals, (block_index, group_index), 1.0)
            np.add.at(positives, (block_index, group_index), y.astype(float))
            self.blocks.append((totals, positives))
            self.cache.append({})

    def valid(self, counts: tuple[int, ...]) -> bool:
        weights = np.asarray(counts, dtype=float)
        positives = float(self.group_positives @ weights)
        total = float(self.group_totals @ weights)
        return positives > 0 and total > positives

    def ap(self, seed_index: int, counts: tuple[int, ...]) -> float:
        cached = self.cache[seed_index]
        if counts in cached:
            return cached[counts]
        weights = np.asarray(counts, dtype=float)
        totals, positives = self.blocks[seed_index]
        block_totals = totals @ weights
        block_positives = positives @ weights
        cumulative_total = np.cumsum(block_totals)
        cumulative_positive = np.cumsum(block_positives)
        total_positive = cumulative_positive[-1]
        precision = np.divide(cumulative_positive, cumulative_total,
                              out=np.zeros_like(cumulative_positive), where=cumulative_total > 0)
        value = float(np.sum(precision * block_positives) / total_positive)
        cached[counts] = value
        return value


def paired_bootstrap(y: np.ndarray, groups: np.ndarray, first: np.ndarray, second: np.ndarray,
                     repeats: int, seed: int) -> dict[str, Any]:
    unique = np.unique(groups)
    first_ap = GroupedAveragePrecision(y, groups, first)
    second_ap = GroupedAveragePrecision(y, groups, second)
    rng = np.random.default_rng(seed)
    joint, date_only = [], []
    invalid = 0
    for _ in range(repeats):
        drawn = rng.integers(0, len(unique), len(unique))
        counts = tuple(int(value) for value in np.bincount(drawn, minlength=len(unique)))
        if not first_ap.valid(counts):
            invalid += 1
            continue
        count = max(len(first), len(second))
        values = np.asarray([
            first_ap.ap(0 if len(first) == 1 else index, counts) -
            second_ap.ap(0 if len(second) == 1 else index, counts)
            for index in range(count)
        ])
        date_only.append(float(values.mean()))
        sampled = values if len(values) == 1 else values[rng.integers(0, len(values), len(values))]
        joint.append(float(sampled.mean()))
    describe = lambda values: {
        "valid": len(values),
        "interval_95": np.quantile(values, [0.025, 0.975]).tolist() if values else None,
        "interval_98_75": np.quantile(values, [0.00625, 0.99375]).tolist() if values else None,
    }
    return {
        "groups": len(unique), "replicates_requested": repeats,
        "invalid_no_both_classes": invalid, "bootstrap_seed": seed,
        "seed_structure": {"first": len(first), "second": len(second),
                           "deterministic_models_are_not_duplicated_as_independent_runs": True},
        "joint_date_and_seed": describe(joint), "date_only_fixed_seeds": describe(date_only),
    }


def comparison(direction: str, experiment: str, name: str, y: np.ndarray, groups: np.ndarray,
               first: np.ndarray, second: np.ndarray, repeats: int, seed: int,
               family: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    differences = seed_differences(y, first, second)
    boot = paired_bootstrap(y, groups, first, second, repeats, seed)
    result = {
        "experiment": experiment, "direction": direction, "family": family,
        "comparison": name, "seed_AP_differences": differences.tolist(),
        "mean_AP_difference": float(differences.mean()),
        "positive_seed_differences": int(np.sum(differences > 0)),
        "bootstrap": boot,
    }
    date_rows = []
    for group in np.unique(groups):
        within = np.flatnonzero(groups == group)
        keep = np.flatnonzero(groups != group)
        within_values = seed_differences(y, first, second, within)
        leave_values = seed_differences(y, first, second, keep)
        date_rows.append({
            "experiment": experiment, "direction": direction, "family": family,
            "comparison": name, "date_group": str(group), "n": len(within),
            "positives": int(y[within].sum()),
            "within_date_AP_difference": (float(within_values.mean()) if len(within_values) else None),
            "leave_one_date_out_AP_difference": (float(leave_values.mean()) if len(leave_values) else None),
        })
    return result, date_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root_default = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument("--data", type=Path, default=Path("data/official_cross_domain.npz"))
    parser.add_argument("--r1", type=Path, default=Path("outputs/vv_followup_v1/r1"))
    parser.add_argument("--r2", type=Path, default=Path("outputs/vv_followup_v1/r2_final"))
    parser.add_argument("--selection", type=Path, default=Path("outputs/vv_followup_v1/r2_cv"))
    parser.add_argument("--r5", type=Path, default=Path("outputs/vv_followup_v1/r5"))
    parser.add_argument("--out", type=Path, default=Path("outputs/vv_followup_v2/evidence"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    root = args.root.resolve()
    resolve = lambda value: value if value.is_absolute() else root / value
    data_path, r1, r2, selection, r5, out = map(resolve,
                                                (args.data, args.r1, args.r2, args.selection, args.r5, args.out))
    bundle = Bundle.load(data_path)
    out.mkdir(parents=True, exist_ok=True)
    comparisons, date_rows, metric_rows, bbox_rows, bbox_predictions = [], [], [], [], []
    comparison_index = 0
    for experiment, base in (("R1_fixed", r1), ("R2_source_selected", r2)):
        lock_path = base / "TARGET_SCORING_LOCK.json"
        for direction, split_relative in DIRECTIONS.items():
            scores: dict[str, np.ndarray] = {}
            loaded_rows: dict[str, list[list[dict]]] = {}
            ids = None
            y = None
            groups = None
            model_seeds = {**{model: SEEDS for model in NEURAL},
                           "histgb": (11,), "random_forest": SEEDS}
            for model, seeds in model_seeds.items():
                current_ids, current_y, matrix, rows = load_locked_matrix(
                    base, lock_path, direction, model, seeds, ids, y)
                if ids is None:
                    ids, y = current_ids, current_y
                    groups = np.asarray([str(row.get("group_id", row["recording"])) for row in rows[0]])
                scores[model], loaded_rows[model] = matrix, rows
                for seed_value, seed_rows, seed_scores in zip(seeds, rows, matrix):
                    threshold_values = {float(row["source_threshold"]) for row in seed_rows}
                    if len(threshold_values) != 1:
                        raise ValueError(f"Threshold mismatch: {experiment}/{direction}/{model}/{seed_value}")
                    metrics = binary_metrics(y, seed_scores, threshold_values.pop())
                    metric_rows.append({"experiment": experiment, "direction": direction,
                                        "model": model, "seed": seed_value, **metrics})
            assert ids is not None and y is not None and groups is not None
            selected = read_json(selection / direction / "MODEL_SELECTION_LOCK.json")
            pairs = []
            if experiment == "R1_fixed":
                pairs.extend((name, first, second, "R1_auxiliary") for name, first, second in (
                    ("F-G", "F", "G"), ("F-Cm", "F", "Cm"), ("Jw-Jc", "Jw", "Jc"),
                    ("JwD-Jw", "JwD", "Jw"), ("F-Q", "F", "Q")))
            else:
                pairs.append(("F-G", "F", "G", "behavior_vs_geometry"))
                for method in ("F", selected["J_star"], selected["C_star"]):
                    for tree in ("histgb", "random_forest"):
                        pairs.append((f"{method}-{tree}", method, tree, "strong_tree_baseline"))
            for name, first_name, second_name, family in pairs:
                seed_value = 20260919 + comparison_index
                item, per_date = comparison(direction, experiment, name, y, groups,
                                            scores[first_name], scores[second_name],
                                            args.bootstrap, seed_value, family)
                comparisons.append(item)
                date_rows.extend(per_date)
                comparison_index += 1

            if experiment == "R2_source_selected":
                split = read_json(root / split_relative)
                indices = resolve_split(bundle, split)
                source_val, target = indices["val"], indices["test"]
                if [bundle.meta[int(i)]["sample_id"] for i in target] != ids:
                    raise ValueError(f"Area baseline target IDs differ: {direction}")
                val_area = bundle.a[source_val, 5].astype(float)
                plus_ap = average_precision_score(bundle.y[source_val], val_area)
                minus_ap = average_precision_score(bundle.y[source_val], -val_area)
                sign = 1.0 if plus_ap >= minus_ap else -1.0
                val_score, target_score = sign * val_area, sign * bundle.a[target, 5].astype(float)
                threshold = choose_threshold(bundle.y[source_val], val_score)
                target_metrics = binary_metrics(bundle.y[target], target_score, threshold)
                bbox_rows.append({
                    "direction": direction, "feature": "last_log_box_area", "direction_sign": sign,
                    "selection_uses_target": False, "source_val_AP_plus": float(plus_ap),
                    "source_val_AP_minus": float(minus_ap), "source_threshold": threshold,
                    **{f"target_{key}": value for key, value in target_metrics.items()},
                })
                for index, score in zip(target, target_score):
                    bbox_predictions.append({**bundle.meta[int(index)], "label": int(bundle.y[int(index)]),
                                             "score": float(score), "source_threshold": threshold,
                                             "direction_sign": sign})

    for direction in DIRECTIONS:
        real_ids, real_y, real, real_rows = load_locked_matrix(
            r1, r1 / "TARGET_SCORING_LOCK.json", direction, "F", SEEDS)
        groups = np.asarray([str(row.get("group_id", row["recording"])) for row in real_rows[0]])
        diagnostics: list[tuple[str, np.ndarray]] = []
        for permutation_seed in PERMUTATIONS:
            paths = [r5 / "r5a" / direction / f"perm_{permutation_seed}" / "F" / str(seed) /
                     "test_predictions.jsonl" for seed in SEEDS]
            diagnostics.append((f"F_real-F_perm_{permutation_seed}",
                                load_unlocked_matrix(paths, real_ids, real_y)))
        confidence_paths = [r5 / "r5b" / direction / "F_conf" / str(seed) /
                            "test_predictions.jsonl" for seed in SEEDS]
        diagnostics.append(("F_real-F_conf", load_unlocked_matrix(confidence_paths, real_ids, real_y)))
        velocity_paths = [r5 / "r5b_velocity" / direction / "F_no_velocity" / str(seed) /
                          "test_predictions.jsonl" for seed in SEEDS]
        diagnostics.append(("F_real-F_no_velocity",
                            load_unlocked_matrix(velocity_paths, real_ids, real_y)))
        for name, control in diagnostics:
            seed_value = 20260919 + comparison_index
            item, per_date = comparison(direction, "R5_fixed_input_controls", name, real_y, groups,
                                        real, control, args.bootstrap, seed_value,
                                        "input_content_and_velocity")
            comparisons.append(item)
            date_rows.extend(per_date)
            comparison_index += 1

    flat_comparisons = []
    for item in comparisons:
        joint = item["bootstrap"]["joint_date_and_seed"]
        date_only = item["bootstrap"]["date_only_fixed_seeds"]
        flat_comparisons.append({
            "experiment": item["experiment"], "direction": item["direction"],
            "family": item["family"], "comparison": item["comparison"],
            "mean_AP_difference": item["mean_AP_difference"],
            "positive_seed_differences": item["positive_seed_differences"],
            "joint_95_low": joint["interval_95"][0], "joint_95_high": joint["interval_95"][1],
            "joint_98_75_low": joint["interval_98_75"][0],
            "joint_98_75_high": joint["interval_98_75"][1],
            "date_only_95_low": date_only["interval_95"][0],
            "date_only_95_high": date_only["interval_95"][1],
            "bootstrap_seed": item["bootstrap"]["bootstrap_seed"],
        })
    payload = {
        "schema": "acie.vv-method-evidence.v1", "bootstrap_repeats": args.bootstrap,
        "target_results_previously_observed": True,
        "comparisons": comparisons, "operating_points": metric_rows,
        "box_area_baseline": bbox_rows,
    }
    write_json(out / "evidence_summary.json", payload)
    write_csv(out / "comparisons.csv", flat_comparisons)
    write_csv(out / "target_date_sensitivity.csv", date_rows)
    write_csv(out / "operating_points.csv", metric_rows)
    write_csv(out / "box_area_baseline.csv", bbox_rows)
    write_jsonl(out / "box_area_predictions.jsonl", bbox_predictions)
    print(json.dumps({"status": "complete", "comparisons": len(comparisons),
                      "date_rows": len(date_rows), "operating_points": len(metric_rows),
                      "box_area_directions": len(bbox_rows), "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
