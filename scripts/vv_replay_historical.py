"""Replay historical frozen checkpoints and compare every target prediction."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from acie.data import Bundle
from acie.engine import predict
from acie.io import digest_file, digest_object, read_json, read_jsonl, write_json
from acie.metrics import binary_metrics


def aligned(first: list[dict], second: list[dict], name: str) -> tuple[list[str], dict, dict]:
    aa = {row["sample_id"]: row for row in first}
    bb = {row["sample_id"]: row for row in second}
    if len(aa) != len(first) or len(bb) != len(second) or set(aa) != set(bb):
        raise ValueError(f"Prediction ID mismatch: {name}")
    ids = sorted(aa)
    if any(int(aa[sample_id]["label"]) != int(bb[sample_id]["label"]) for sample_id in ids):
        raise ValueError(f"Prediction label mismatch: {name}")
    return ids, aa, bb


def max_difference(ids: list[str], aa: dict, bb: dict, field: str) -> float | None:
    if any(field not in aa[sample_id] or field not in bb[sample_id] for sample_id in ids):
        return None
    return float(max(abs(float(aa[sample_id][field]) - float(bb[sample_id][field])) for sample_id in ids))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-probability-difference", type=float, default=1e-6)
    parser.add_argument("--max-ap-difference", type=float, default=1e-6)
    parser.add_argument("--bootstrap", type=int, default=10)
    parser.add_argument("--directions", nargs="*")
    parser.add_argument("--methods", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int)
    args = parser.parse_args()

    bundle = Bundle.load(args.data)
    historical_lock = read_json(args.runs / "FROZEN_PROTOCOL.json")
    directions = args.directions or sorted(path.name for path in args.runs.iterdir()
                                           if path.is_dir() and "_to_" in path.name)
    methods = args.methods or sorted(historical_lock["methods"])
    seeds = args.seeds or [int(seed) for seed in historical_lock["seeds"]]
    protocol = {
        "schema": "acie.vv-historical-replay.v1",
        "data_sha256": digest_file(args.data),
        "historical_lock_sha256": digest_file(args.runs / "FROZEN_PROTOCOL.json"),
        "historical_lock_digest": historical_lock["lock_digest"],
        "source_sha256": {
            name: digest_file(Path("src/acie") / name)
            for name in ["data.py", "engine.py", "features.py", "io.py", "matching.py", "metrics.py", "models.py"]
        },
        "directions": directions,
        "methods": methods,
        "seeds": seeds,
        "max_probability_difference": args.max_probability_difference,
        "max_ap_difference": args.max_ap_difference,
        "bootstrap": args.bootstrap,
        "selection": "checkpoint replay only; no training or target adaptation",
    }
    protocol["protocol_digest"] = digest_object(protocol)
    args.out.mkdir(parents=True, exist_ok=True)
    protocol_path = args.out / "FROZEN_PROTOCOL.json"
    if protocol_path.is_file():
        if read_json(protocol_path) != protocol:
            raise ValueError("Replay protocol changed after output creation")
    else:
        write_json(protocol_path, protocol)

    results = []
    for direction in directions:
        for method in methods:
            for seed in seeds:
                name = f"{direction}/{method}/seed{seed}"
                historical_run = args.runs / direction / method / f"seed{seed}"
                checkpoint = historical_run / "model" / "best.pt"
                historical_prediction = historical_run / "test_predictions.jsonl"
                if not checkpoint.is_file() or not historical_prediction.is_file():
                    raise FileNotFoundError(name)
                replay_run = args.out / direction / method / f"seed{seed}"
                replay_run.mkdir(parents=True, exist_ok=True)
                replay_prediction = replay_run / "test_predictions.jsonl"
                if replay_prediction.is_file() and replay_prediction.with_suffix(".metrics.json").is_file():
                    replay_metrics = read_json(replay_prediction.with_suffix(".metrics.json"))
                else:
                    replay_metrics = predict(bundle, checkpoint, replay_prediction,
                                             split_name="test", device="cpu", repeats=args.bootstrap)
                old_rows = read_jsonl(historical_prediction)
                new_rows = read_jsonl(replay_prediction)
                ids, old, new = aligned(old_rows, new_rows, name)
                labels = np.asarray([old[sample_id]["label"] for sample_id in ids], dtype=int)
                old_scores = np.asarray([old[sample_id]["score"] for sample_id in ids], dtype=float)
                new_scores = np.asarray([new[sample_id]["score"] for sample_id in ids], dtype=float)
                old_ap = binary_metrics(labels, old_scores)["AP"]
                new_ap = binary_metrics(labels, new_scores)["AP"]
                item = {
                    "direction": direction,
                    "method": method,
                    "seed": seed,
                    "n": len(ids),
                    "checkpoint_sha256": digest_file(checkpoint),
                    "historical_prediction_sha256": digest_file(historical_prediction),
                    "replay_prediction_sha256": digest_file(replay_prediction),
                    "probability_max_abs_difference": float(np.max(np.abs(old_scores - new_scores))),
                    "logit_max_abs_difference": max_difference(ids, old, new, "logit"),
                    "geometry_logit_max_abs_difference": max_difference(ids, old, new, "geometry_logit"),
                    "evidence_max_abs_difference": max_difference(ids, old, new, "evidence"),
                    "historical_AP_recomputed": old_ap,
                    "replay_AP_recomputed": new_ap,
                    "AP_abs_difference": abs(old_ap - new_ap),
                    "prediction_seconds": replay_metrics["prediction_seconds"],
                }
                item["passed"] = (
                    item["probability_max_abs_difference"] <= args.max_probability_difference
                    and item["AP_abs_difference"] <= args.max_ap_difference
                )
                results.append(item)
                write_json(args.out / "runs.json", results)
                print(f"{name} prob_diff={item['probability_max_abs_difference']:.3g} "
                      f"AP_diff={item['AP_abs_difference']:.3g}", flush=True)

    expected = len(directions) * len(methods) * len(seeds)
    if len(results) != expected or not all(item["passed"] for item in results):
        raise ValueError("Historical replay failed or is incomplete")
    summary = {
        "schema": "acie.vv-historical-replay-summary.v1",
        "status": "passed",
        "runs_expected": expected,
        "runs_replayed": len(results),
        "protocol_digest": protocol["protocol_digest"],
        "maximum_probability_abs_difference": max(item["probability_max_abs_difference"] for item in results),
        "maximum_logit_abs_difference": max(item["logit_max_abs_difference"] or 0.0 for item in results),
        "maximum_AP_abs_difference": max(item["AP_abs_difference"] for item in results),
        "all_ids_and_labels_match": True,
        "training_performed": False,
    }
    write_json(args.out / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
