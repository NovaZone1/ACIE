"""Run frozen post-hoc random-pair and strong tree controls on both directions."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import yaml

from acie.baselines import run_tree
from acie.data import Bundle, group_key, resolve_split
from acie.engine import predict, train
from acie.io import digest_file, digest_object, read_json, read_jsonl, write_json
from acie.metrics import binary_metrics, bootstrap

METHODS = ["random_pair", "histgb", "random_forest"]


def prediction_path(root: Path, direction: str, method: str, seed: int) -> Path:
    return root / direction / method / f"seed{seed}" / "predictions.jsonl"


def ensemble(root: Path, direction: str, method: str, seeds: list[int]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    tables = []
    for seed in seeds:
        rows = read_jsonl(prediction_path(root, direction, method, seed))
        table = {row["sample_id"]: row for row in rows}
        if len(table) != len(rows):
            raise ValueError(f"Duplicate prediction IDs for {direction}/{method}/seed{seed}")
        tables.append(table)
    ids = sorted(tables[0])
    if any(set(table) != set(ids) for table in tables):
        raise ValueError(f"Prediction IDs differ across seeds for {direction}/{method}")
    labels = np.asarray([tables[0][sample_id]["label"] for sample_id in ids], dtype=int)
    scores = np.mean([[table[sample_id]["score"] for sample_id in ids] for table in tables], axis=0)
    meta = [tables[0][sample_id] for sample_id in ids]
    return labels, scores, meta


def frozen_ensemble(parent: Path, direction: str, method: str, seeds: list[int]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    tables = []
    for seed in seeds:
        rows = read_jsonl(parent / direction / method / f"seed{seed}" / "test_predictions.jsonl")
        tables.append({row["sample_id"]: row for row in rows})
    ids = sorted(tables[0])
    if any(set(table) != set(ids) for table in tables):
        raise ValueError(f"Frozen prediction IDs differ for {direction}/{method}")
    labels = np.asarray([tables[0][sample_id]["label"] for sample_id in ids], dtype=int)
    scores = np.mean([[table[sample_id]["score"] for sample_id in ids] for table in tables], axis=0)
    return labels, scores, [tables[0][sample_id] for sample_id in ids]


def describe(values: list[float]) -> dict:
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def build_summary(out: Path, parent: Path, rows: list[dict], seeds: list[int], repeats: int,
                  lock_digest: str) -> dict:
    result = {
        "schema": "acie.revised-scope-controls.summary.v1",
        "protocol_digest": lock_digest,
        "seeds": seeds,
        "directions": {},
    }
    directions = sorted({row["direction"] for row in rows})
    for direction in directions:
        payload = {"methods": {}, "comparisons": {}}
        by = {(row["method"], row["seed"]): row for row in rows if row["direction"] == direction}
        ensembles = {}
        for method in METHODS:
            subset = [by[method, seed] for seed in seeds]
            labels, scores, meta = ensemble(out, direction, method, seeds)
            ensembles[method] = (labels, scores, meta)
            method_payload = {
                "AP": describe([row["AP"] for row in subset]),
                "AUROC": describe([row["AUROC"] for row in subset]),
                "source_val_AP": describe([row["source_val_AP"] for row in subset]),
                "training_seconds": describe([row["training_seconds"] for row in subset]),
                "prediction_seconds": describe([row["prediction_seconds"] for row in subset]),
                "ensemble": binary_metrics(labels, scores),
                "ensemble_bootstrap": bootstrap(labels, scores, [group_key(row) for row in meta], repeats),
            }
            if method == "random_pair":
                method_payload["changed_negative_fraction"] = describe(
                    [row["changed_negative_fraction"] for row in subset]
                )
                method_payload["n_train_pairs"] = describe([float(row["n_train_pairs"]) for row in subset])
                method_payload["parameters"] = sorted({row["parameters"] for row in subset})
            else:
                method_payload["selected_leaf_or_depth"] = [row["selected_leaf_or_depth"] for row in subset]
                method_payload["feature_dim"] = sorted({row["feature_dim"] for row in subset})
                method_payload["serialized_bytes"] = describe([float(row["serialized_bytes"]) for row in subset])
            payload["methods"][method] = method_payload

        frozen = {}
        for method in ["geometry", "dual_no_pair", "weighted_no_pair", "full_selected"]:
            frozen[method] = frozen_ensemble(parent, direction, method, seeds)
        labels = ensembles["random_pair"][0]
        groups = [group_key(row) for row in ensembles["random_pair"][2]]
        comparisons = [
            ("full_selected_minus_random_pair", frozen["full_selected"][1], ensembles["random_pair"][1]),
            ("random_pair_minus_weighted_no_pair", ensembles["random_pair"][1], frozen["weighted_no_pair"][1]),
            ("histgb_minus_dual_no_pair", ensembles["histgb"][1], frozen["dual_no_pair"][1]),
            ("random_forest_minus_dual_no_pair", ensembles["random_forest"][1], frozen["dual_no_pair"][1]),
        ]
        for name, first, second in comparisons:
            payload["comparisons"][name] = {
                "AP_difference": float(binary_metrics(labels, first)["AP"] - binary_metrics(labels, second)["AP"]),
                "bootstrap": bootstrap(labels, first, groups, repeats, other=second),
            }
        result["directions"][direction] = payload
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--random-config", required=True, type=Path)
    parser.add_argument("--full-config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--confirm-frozen", action="store_true")
    args = parser.parse_args()
    if not args.confirm_frozen:
        raise SystemExit("Review the post-hoc protocol, then pass --confirm-frozen")

    declared = read_json(args.protocol)
    seeds = [int(seed) for seed in declared["seeds"]]
    directions = list(declared["directions"])
    if seeds != [11, 22, 33, 44, 55] or set(directions) != {"HUI360_to_SSUP-A", "SSUP-A_to_HUI360"}:
        raise ValueError("Unexpected protocol seed or direction set")
    random_config = yaml.safe_load(args.random_config.read_text())
    full_config = yaml.safe_load(args.full_config.read_text())
    if random_config.get("kind") != "random_pair" or full_config.get("kind") != "full":
        raise ValueError("Unexpected control kinds")
    random_comparable = copy.deepcopy(random_config); random_comparable["kind"] = "full"
    if random_comparable != full_config:
        raise ValueError("Random-pair config must differ from full_selected only by kind")

    parent_protocol = read_json(args.parent / "FROZEN_PROTOCOL.json")
    split_hashes = {direction: digest_file(args.parent / direction / "split.json") for direction in directions}
    lock = {
        "schema": "acie.revised-scope-controls.lock.v1",
        "declared_protocol": declared,
        "declared_protocol_sha256": digest_file(args.protocol),
        "data_sha256": digest_file(args.data),
        "parent_protocol_digest": parent_protocol["lock_digest"],
        "parent_protocol_sha256": digest_file(args.parent / "FROZEN_PROTOCOL.json"),
        "split_sha256": split_hashes,
        "random_config": random_config,
        "random_config_sha256": digest_file(args.random_config),
        "full_config_sha256": digest_file(args.full_config),
        "implementation_sha256": {
            "engine": digest_file(Path("src/acie/engine.py")),
            "models": digest_file(Path("src/acie/models.py")),
            "baselines": digest_file(Path("src/acie/baselines.py")),
        },
        "tree_specification": {
            "methods": ["histgb", "random_forest"],
            "candidates": [7, 15, 31],
            "feature_input": "source-scaled a plus q_last, q_mean, q_std",
        },
        "bootstrap": args.bootstrap,
    }
    lock["lock_digest"] = digest_object(lock)
    args.out.mkdir(parents=True, exist_ok=True)
    lock_path = args.out / "FROZEN_PROTOCOL.json"
    if lock_path.is_file():
        if read_json(lock_path) != lock:
            raise ValueError("Revised-scope protocol changed after scoring began")
    else:
        write_json(lock_path, lock)

    bundle = Bundle.load(args.data)
    rows = read_json(args.out / "runs.json") if (args.out / "runs.json").is_file() else []
    for direction in directions:
        split = read_json(args.parent / direction / "split.json")
        resolve_split(bundle, split)
        write_json(args.out / direction / "split.json", split)
        for method in METHODS:
            for seed in seeds:
                run = args.out / direction / method / f"seed{seed}"
                run.mkdir(parents=True, exist_ok=True)
                if method == "random_pair":
                    config = copy.deepcopy(random_config); config["seed"] = seed
                    report_path = run / "model" / "training_report.json"
                    if report_path.is_file() and (run / "model" / "best.pt").is_file():
                        training = read_json(report_path)
                    else:
                        training = train(bundle, split, config, run / "model",
                                         resume=(run / "model" / "run.json").is_file())
                    metrics_path = run / "predictions.metrics.json"
                    result = read_json(metrics_path) if metrics_path.is_file() else predict(
                        bundle, run / "model" / "best.pt", run / "predictions.jsonl",
                        repeats=args.bootstrap,
                    )
                    matching = read_json(run / "model" / "matching.json")
                    extra = {
                        "training_seconds": training["training_seconds"],
                        "prediction_seconds": result["prediction_seconds"],
                        "parameters": training["parameters"],
                        "n_train_pairs": matching["n_pairs"],
                        "changed_negative_fraction": matching["changed_negative_fraction"],
                    }
                else:
                    metrics_path = run / "metrics.json"
                    result = read_json(metrics_path) if metrics_path.is_file() else run_tree(
                        bundle, split, run, method, seed, args.bootstrap
                    )
                    extra = {
                        "training_seconds": result["fit_selection_seconds"],
                        "prediction_seconds": result["prediction_seconds"],
                        "selected_leaf_or_depth": result["selected_leaf_or_depth"],
                        "feature_dim": result["feature_dim"],
                        "serialized_bytes": result["serialized_bytes"],
                    }
                row = {
                    "direction": direction,
                    "method": method,
                    "seed": seed,
                    "source_val_AP": result["source_val_AP"] if method != "random_pair" else training["source_val"]["AP"],
                    **result["metrics"],
                    **extra,
                }
                rows = [old for old in rows if (old["direction"], old["method"], old["seed"]) !=
                        (direction, method, seed)]
                rows.append(row)
                write_json(args.out / "runs.json", rows)
                print(f"{direction} {method} seed={seed} AP={row['AP']:.4f}", flush=True)

    expected = {(direction, method, seed) for direction in directions for method in METHODS for seed in seeds}
    observed = {(row["direction"], row["method"], row["seed"]) for row in rows}
    if observed != expected:
        raise ValueError(f"Incomplete run matrix: missing={expected-observed}, extra={observed-expected}")
    summary = build_summary(args.out, args.parent, rows, seeds, args.bootstrap, lock["lock_digest"])
    write_json(args.out / "summary.json", summary)
    print(f"Wrote complete 30-run summary to {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
