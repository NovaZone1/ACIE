#!/usr/bin/env python3
"""Run the registered R5d geometry-support and local-order diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors

from acie.data import Bundle, resolve_split, track_key
from acie.features import SourceScaler
from acie.io import read_json, read_jsonl, write_json, write_jsonl
from acie.matching import MatchConfig, accuracy, fit_radius, match

DIRECTIONS = {
    "HUI360_to_SSUP-A": "outputs/frozen_cross_domain/HUI360_to_SSUP-A/split.json",
    "SSUP-A_to_HUI360": "outputs/frozen_cross_domain/SSUP-A_to_HUI360/split.json",
}
NEURAL = ("G", "Q", "C64", "Cm", "Jc", "Jw", "JwD", "F")
SEEDS = (11, 22, 33, 44, 55)


def prediction_matrix(base: Path, direction: str, model: str, seeds: tuple[int, ...],
                      expected_ids: list[str]) -> tuple[np.ndarray, list[list[dict]]]:
    loaded = [read_jsonl(base / direction / model / "fixed" / str(seed) / "test_predictions.jsonl")
              for seed in seeds]
    if any([row["sample_id"] for row in rows] != expected_ids for rows in loaded):
        raise ValueError(f"R5d prediction coverage differs: {direction}/{model}")
    return np.asarray([[row["score"] for row in rows] for rows in loaded]), loaded


def additive_component_matrices(rows_by_seed: list[list[dict]], model: str) -> dict[str, np.ndarray]:
    """Return g, r, and z for additive models; reject incomplete component records."""
    result = {}
    for name, field in (("g", "geometry_logit"), ("r", "evidence_logit"), ("g_plus_r", "logit")):
        if any(row.get(field) is None for rows in rows_by_seed for row in rows):
            raise ValueError(f"Missing {field} for additive diagnostic: {model}")
        result[name] = np.asarray([[float(row[field]) for row in rows] for rows in rows_by_seed])
    if not np.allclose(result["g"] + result["r"], result["g_plus_r"], rtol=0, atol=1e-5):
        raise ValueError(f"Additive decomposition mismatch: {model}")
    return result


def pair_outcomes(scores: np.ndarray, pairs) -> np.ndarray:
    difference = scores[:, pairs.positive] - scores[:, pairs.negative]
    return (difference > 0).astype(float) + 0.5 * (difference == 0)


def paired_date_bootstrap_pair_accuracy(a: np.ndarray, b: np.ndarray, pair_groups: np.ndarray,
                                        repeats: int, seed: int) -> dict:
    unique = np.unique(pair_groups)
    indices = {group: np.flatnonzero(pair_groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    joint, date_only = [], []
    for _ in range(repeats):
        drawn = rng.integers(0, len(unique), len(unique))
        counts = np.bincount(drawn, minlength=len(unique))
        per_seed = []
        for seed_index in range(a.shape[0]):
            first = sum(count * float(a[seed_index, indices[group]].sum())
                        for group, count in zip(unique, counts))
            second = sum(count * float(b[seed_index, indices[group]].sum())
                         for group, count in zip(unique, counts))
            denominator = sum(count * len(indices[group]) for group, count in zip(unique, counts))
            per_seed.append(first / denominator - second / denominator)
        per_seed = np.asarray(per_seed)
        date_only.append(float(per_seed.mean()))
        seed_draw = rng.integers(0, a.shape[0], a.shape[0])
        joint.append(float(per_seed[seed_draw].mean()))
    return {"groups": len(unique), "replicates_requested": repeats,
            "joint_interval_95": np.quantile(joint, [0.025, 0.975]).tolist(),
            "date_only_interval_95": np.quantile(date_only, [0.025, 0.975]).tolist()}


def nearest_support(a: np.ndarray, meta: list[dict], train_ix: np.ndarray,
                    test_ix: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    train = a[train_ix]
    count = min(len(train), 129)
    distances, neighbors = NearestNeighbors(n_neighbors=count).fit(train).kneighbors(train)
    source = []
    for position, (row_distances, row_neighbors) in enumerate(zip(distances, neighbors)):
        own_track = track_key(meta[int(train_ix[position])])
        legal = [distance for distance, neighbor in zip(row_distances, row_neighbors)
                 if track_key(meta[int(train_ix[int(neighbor)])]) != own_track]
        if not legal:
            raise ValueError("No different-track source neighbor for R5d support threshold")
        source.append(float(legal[0] / np.sqrt(a.shape[1])))
    threshold = float(np.quantile(source, 0.9))
    target_distance = NearestNeighbors(n_neighbors=1).fit(train).kneighbors(a[test_ix])[0][:, 0] / np.sqrt(a.shape[1])
    return np.asarray(source), threshold, target_distance


def subgroup_rows(direction: str, y: np.ndarray, score: dict[str, np.ndarray],
                  diagnostic: str, bins: list[tuple[str, np.ndarray]],
                  opponents: tuple[str, ...]) -> list[dict]:
    rows = []
    for group_name, mask in bins:
        indices = np.flatnonzero(mask)
        positives = int(y[indices].sum())
        for opponent in opponents:
            differences = None
            if len(indices) and len(np.unique(y[indices])) == 2:
                differences = [float(average_precision_score(y[indices], score["F"][i, indices]) -
                                     average_precision_score(y[indices], score[opponent][i, indices]))
                               for i in range(5)]
            rows.append({"direction": direction, "diagnostic": diagnostic,
                         "group": group_name, "comparison": f"F-{opponent}",
                         "n": len(indices), "positives": positives,
                         "mean_AP_difference": None if differences is None else float(np.mean(differences)),
                         "positive_seeds": None if differences is None else int(np.sum(np.asarray(differences) > 0)),
                         "inference_strength": "descriptive_only" if positives < 10 else "standard_exploratory"})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root_default = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument("--data", type=Path, default=Path("data/official_cross_domain.npz"))
    parser.add_argument("--r2", type=Path, default=Path("outputs/vv_followup_v1/r2_final"))
    parser.add_argument("--selection", type=Path, default=Path("outputs/vv_followup_v1/r2_cv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/vv_followup_v1/r5/r5d_support"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    root = args.root.resolve()
    resolve = lambda value: value if value.is_absolute() else root / value
    data_path, r2, selection, out = map(resolve, (args.data, args.r2, args.selection, args.out))
    bundle = Bundle.load(data_path)
    results, method_rows, comparison_rows, subgroups = {}, [], [], []
    component_rows, pair_records = [], []
    for direction_index, (direction, split_relative) in enumerate(DIRECTIONS.items()):
        split = read_json(root / split_relative)
        indices = resolve_split(bundle, split)
        train_ix, test_ix = indices["train"], indices["test"]
        checkpoint = torch.load(r2 / direction / "G" / "fixed" / "11" / "best.pt",
                                map_location="cpu", weights_only=True)
        scaler = SourceScaler.from_dict(checkpoint["scaler"])
        standardized_a, _ = scaler.transform(bundle.a, bundle.q)
        train_meta = [bundle.meta[i] for i in train_ix]
        test_meta = [bundle.meta[i] for i in test_ix]
        global_config = fit_radius(standardized_a[train_ix], bundle.y[train_ix], train_meta, MatchConfig())
        global_pairs = match(standardized_a[test_ix], bundle.y[test_ix], test_meta,
                             global_config, seed=20260916)
        same_config = fit_radius(standardized_a[train_ix], bundle.y[train_ix], train_meta,
                                 MatchConfig(context_policy="same_group"))
        same_pairs = match(standardized_a[test_ix], bundle.y[test_ix], test_meta,
                           same_config, seed=20260916)
        for pair_name, pairs in (("global", global_pairs), ("same_date", same_pairs)):
            for pair_index, (positive, negative, weight, distance) in enumerate(
                    zip(pairs.positive, pairs.negative, pairs.weight, pairs.distance)):
                positive_meta, negative_meta = test_meta[int(positive)], test_meta[int(negative)]
                pair_records.append({
                    "direction": direction, "pair_set": pair_name, "pair_index": pair_index,
                    "positive_sample_id": positive_meta["sample_id"],
                    "negative_sample_id": negative_meta["sample_id"],
                    "positive_group": str(positive_meta.get("group_id", positive_meta["recording"])),
                    "negative_group": str(negative_meta.get("group_id", negative_meta["recording"])),
                    "weight": float(weight), "geometry_distance": float(distance),
                })
        expected_ids = [bundle.meta[i]["sample_id"] for i in test_ix]
        scores, prediction_rows = {}, {}
        model_seeds = {**{model: SEEDS for model in NEURAL}, "histgb": (11,), "random_forest": SEEDS}
        for model, seeds in model_seeds.items():
            scores[model], prediction_rows[model] = prediction_matrix(r2, direction, model, seeds, expected_ids)
            for pair_name, pairs in (("global", global_pairs), ("same_date", same_pairs)):
                outcomes = pair_outcomes(scores[model], pairs)
                method_rows.append({"direction": direction, "pair_set": pair_name, "model": model,
                                    "seeds": len(seeds), "pair_count": len(pairs),
                                    "pair_accuracy_mean": float(outcomes.mean(1).mean()),
                                    "pair_accuracy_sample_SD": (float(outcomes.mean(1).std(ddof=1))
                                                                if len(seeds) > 1 else None)})
            if model in ("F", "Jw", "JwD", "Jc"):
                components = additive_component_matrices(prediction_rows[model], model)
                for pair_name, pairs in (("global", global_pairs), ("same_date", same_pairs)):
                    for component, values in components.items():
                        outcomes = pair_outcomes(values, pairs)
                        per_seed = outcomes.mean(1)
                        weighted = np.asarray([
                            np.average(seed_outcomes, weights=pairs.weight)
                            for seed_outcomes in outcomes
                        ]) if len(pairs) else np.asarray([])
                        component_rows.append({
                            "direction": direction, "pair_set": pair_name, "model": model,
                            "component": component, "seeds": len(seeds), "pair_count": len(pairs),
                            "pair_accuracy_mean": float(per_seed.mean()) if len(per_seed) else None,
                            "pair_accuracy_sample_SD": (float(per_seed.std(ddof=1))
                                                        if len(per_seed) > 1 else None),
                            "weighted_pair_accuracy_mean": (float(weighted.mean())
                                                            if len(weighted) else None),
                        })
        selected = read_json(selection / direction / "MODEL_SELECTION_LOCK.json")
        opponents = ("G", selected["J_star"], selected["C_star"])
        pair_groups = np.asarray([str(test_meta[i].get("group_id", test_meta[i]["recording"]))
                                  for i in same_pairs.positive])
        for opponent_index, opponent in enumerate(opponents):
            for pair_name, pairs in (("global", global_pairs), ("same_date", same_pairs)):
                f_outcomes, other_outcomes = pair_outcomes(scores["F"], pairs), pair_outcomes(scores[opponent], pairs)
                row = {"direction": direction, "pair_set": pair_name, "comparison": f"F-{opponent}",
                       "mean_pair_accuracy_difference": float((f_outcomes.mean(1) - other_outcomes.mean(1)).mean()),
                       "positive_seeds": int(np.sum((f_outcomes.mean(1) - other_outcomes.mean(1)) > 0))}
                if pair_name == "same_date":
                    row["bootstrap"] = paired_date_bootstrap_pair_accuracy(
                        f_outcomes, other_outcomes, pair_groups, args.bootstrap,
                        20260916 + direction_index * 100 + opponent_index)
                comparison_rows.append(row)
        negative_tracks = Counter(track_key(test_meta[int(i)]) for i in global_pairs.negative)
        area_train = bundle.a[train_ix, 5]
        area_edges = np.quantile(area_train, [1 / 3, 2 / 3])
        area_target = bundle.a[test_ix, 5]
        valid = bundle.q.reshape(len(bundle.y), bundle.q.shape[1], 17, 7)[..., 5].mean((1, 2))
        valid_edges = np.quantile(valid[train_ix], [1 / 3, 2 / 3])
        source_nn, support_threshold, target_nn = nearest_support(standardized_a, bundle.meta, train_ix, test_ix)
        definitions = (
            ("last_log_box_area", area_target, [
                ("low", area_target <= area_edges[0]),
                ("middle", (area_target > area_edges[0]) & (area_target <= area_edges[1])),
                ("high", area_target > area_edges[1])]),
            ("mean_valid_joint_ratio", valid[test_ix], [
                ("low", valid[test_ix] <= valid_edges[0]),
                ("middle", (valid[test_ix] > valid_edges[0]) & (valid[test_ix] <= valid_edges[1])),
                ("high", valid[test_ix] > valid_edges[1])]),
            ("source_geometry_support", target_nn, [
                ("inside_source_90pct", target_nn <= support_threshold),
                ("outside_source_90pct", target_nn > support_threshold)]),
        )
        for diagnostic, values, bins in definitions:
            generated = subgroup_rows(direction, bundle.y[test_ix], scores, diagnostic, bins, opponents)
            subgroups.extend(generated)
        results[direction] = {
            "J_star": selected["J_star"], "C_star": selected["C_star"],
            "global_pairs": {**global_pairs.audit,
                             "negative_track_reuse": dict(sorted(negative_tracks.items())),
                             "negative_track_reuse_max": max(negative_tracks.values(), default=0)},
            "same_date_pairs": same_pairs.audit,
            "subgroup_thresholds": {"last_log_box_area_source_terciles": area_edges.tolist(),
                                    "mean_valid_joint_ratio_source_terciles": valid_edges.tolist(),
                                    "source_leave_track_out_nn_distance_90pct": support_threshold,
                                    "source_nn_distance_summary": {"min": float(source_nn.min()),
                                                                   "median": float(np.median(source_nn)),
                                                                   "max": float(source_nn.max())}},
        }
    out.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "acie.vv-r5d-support-diagnostic.v1", "directions": results,
               "method_pair_accuracy": method_rows, "comparisons": comparison_rows,
               "additive_component_pair_accuracy": component_rows,
               "pair_records_file": "r5d_pairs.jsonl", "subgroups": subgroups,
               "target_labels_used_for_offline_diagnostic": True}
    write_json(out / "r5d_summary.json", payload)
    write_jsonl(out / "r5d_pairs.jsonl", pair_records)
    for name, rows in (("r5d_method_pair_accuracy.csv", method_rows),
                       ("r5d_comparisons.csv", [{**row, "bootstrap": json.dumps(row.get("bootstrap"))} for row in comparison_rows]),
                       ("r5d_additive_components.csv", component_rows),
                       ("r5d_subgroups.csv", subgroups)):
        with (out / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"status": "complete", "methods": len(method_rows),
                      "comparisons": len(comparison_rows), "components": len(component_rows),
                      "pairs": len(pair_records), "subgroups": len(subgroups)}, indent=2))


if __name__ == "__main__":
    main()
