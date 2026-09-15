"""Validate the frozen revised-scope random-pair and tree result matrix."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from acie.data import Bundle, resolve_split
from acie.io import digest_file, read_json, read_jsonl, write_json
from acie.metrics import binary_metrics

METHODS = ["random_pair", "histgb", "random_forest"]


def close(first, second, tolerance=1e-12):
    return first is None and second is None or (
        first is not None and second is not None and abs(float(first) - float(second)) <= tolerance
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    bundle = Bundle.load(args.data)
    lock = read_json(args.runs / "FROZEN_PROTOCOL.json")
    parent_lock = read_json(args.parent / "FROZEN_PROTOCOL.json")
    if lock["data_sha256"] != digest_file(args.data):
        raise ValueError("Data digest mismatch")
    if lock["parent_protocol_digest"] != parent_lock["lock_digest"]:
        raise ValueError("Parent protocol mismatch")
    declared = lock["declared_protocol"]
    seeds = [int(seed) for seed in declared["seeds"]]
    directions = declared["directions"]
    rows = read_json(args.runs / "runs.json")
    keys = {(row["direction"], row["method"], row["seed"]) for row in rows}
    expected = {(direction, method, seed) for direction in directions for method in METHODS for seed in seeds}
    if len(keys) != len(rows) or keys != expected:
        raise ValueError(f"Run matrix mismatch: missing={expected-keys}, extra={keys-expected}")
    by = {(row["direction"], row["method"], row["seed"]): row for row in rows}

    checked = 0
    identity_controls_checked = 0
    changed = []
    splits = {}
    for direction in directions:
        parent_split = read_json(args.parent / direction / "split.json")
        split = read_json(args.runs / direction / "split.json")
        if split != parent_split or digest_file(args.parent / direction / "split.json") != lock["split_sha256"][direction]:
            raise ValueError(f"Split changed for {direction}")
        ix = resolve_split(bundle, split)
        expected_ids = {bundle.meta[i]["sample_id"] for i in ix["test"]}
        expected_labels = {bundle.meta[i]["sample_id"]: int(bundle.y[i]) for i in ix["test"]}
        splits[direction] = {
            "n_train": len(ix["train"]),
            "n_val": len(ix["val"]),
            "n_test": len(ix["test"]),
            "test_positives": int(bundle.y[ix["test"]].sum()),
        }
        for method in METHODS:
            for seed in seeds:
                run = args.runs / direction / method / f"seed{seed}"
                predictions = read_jsonl(run / "predictions.jsonl")
                table = {row["sample_id"]: row for row in predictions}
                if len(table) != len(predictions) or set(table) != expected_ids:
                    raise ValueError(f"Prediction/test ID mismatch: {run}")
                if any(row["label"] != expected_labels[sample_id] or not math.isfinite(row["score"])
                       for sample_id, row in table.items()):
                    raise ValueError(f"Invalid label or score: {run}")
                labels = [table[sample_id]["label"] for sample_id in sorted(table)]
                scores = [table[sample_id]["score"] for sample_id in sorted(table)]
                recomputed = binary_metrics(labels, scores, by[direction, method, seed]["threshold"])
                for metric in ["AP", "AUROC", "recall", "false_positive_rate", "precision"]:
                    if not close(recomputed[metric], by[direction, method, seed][metric]):
                        raise ValueError(f"Metric mismatch for {metric}: {run}")
                if method == "random_pair":
                    state = read_json(run / "model" / "run.json")
                    training = read_json(run / "model" / "training_report.json")
                    matching = read_json(run / "model" / "matching.json")
                    if state.get("status") != "complete" or training["spec"]["kind"] != "random_pair":
                        raise ValueError(f"Incomplete random-pair model: {run}")
                    if matching.get("mode") != "random_count_weight_reuse_matched":
                        raise ValueError(f"Wrong random matching mode: {run}")
                    fraction = matching.get("changed_negative_fraction")
                    if fraction is None or not 0 <= fraction <= 1:
                        raise ValueError(f"Invalid changed-negative fraction: {run}")
                    changed.append(float(fraction))
                    parent_model = args.parent / direction / "full_selected" / f"seed{seed}" / "model"
                    reference_pairs = read_jsonl(parent_model / "pairs.train.jsonl")
                    random_pairs = read_jsonl(run / "model" / "pairs.train.jsonl")
                    if (len(reference_pairs) != len(random_pairs)
                            or [row["positive"] for row in reference_pairs] != [row["positive"] for row in random_pairs]
                            or [row["weight"] for row in reference_pairs] != [row["weight"] for row in random_pairs]
                            or sorted(row["negative"] for row in reference_pairs) != sorted(row["negative"] for row in random_pairs)):
                        raise ValueError(f"Random-pair support/count/weight invariance failed: {run}")
                    reference_checkpoint = torch.load(parent_model / "best.pt", map_location="cpu", weights_only=True)
                    random_checkpoint = torch.load(run / "model" / "best.pt", map_location="cpu", weights_only=True)
                    geometry_keys = [key for key in reference_checkpoint["state"] if key.startswith("geometry.")]
                    if (not geometry_keys or any(not torch.equal(reference_checkpoint["state"][key],
                                                                  random_checkpoint["state"][key])
                                                 for key in geometry_keys)):
                        raise ValueError(f"Frozen geometry branch changed: {run}")
                    identity_controls_checked += 1
                else:
                    metrics = read_json(run / "metrics.json")
                    if not (run / "tree.joblib").is_file() or metrics["feature_dim"] <= 0:
                        raise ValueError(f"Missing tree artifact: {run}")
                    candidates = metrics["candidate_source_val_AP"]
                    selected = str(metrics["selected_leaf_or_depth"])
                    if not close(candidates[selected], max(candidates.values())):
                        raise ValueError(f"Tree selection did not maximize source validation AP: {run}")
                checked += 1

    summary = read_json(args.runs / "summary.json")
    if summary["protocol_digest"] != lock["lock_digest"] or set(summary["directions"]) != set(directions):
        raise ValueError("Summary/protocol mismatch")
    result = {
        "schema": "acie.revised-scope-controls-validation.v1",
        "status": "passed",
        "runs_checked": checked,
        "random_pair_identity_controls_checked": identity_controls_checked,
        "random_pair_invariants": "positive order, weights, negative multiset, and frozen geometry state exactly match full_selected",
        "directions": directions,
        "methods": METHODS,
        "seeds": seeds,
        "splits": splits,
        "random_pair_changed_negative_fraction": {
            "min": min(changed), "max": max(changed), "mean": sum(changed) / len(changed)
        },
        "protocol_digest": lock["lock_digest"],
    }
    write_json(args.out, result)
    print(f"Validated {checked} complete revised-scope runs")


if __name__ == "__main__":
    main()
