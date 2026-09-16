#!/usr/bin/env python3
"""Validate all locked R5 target predictions and the final summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from acie.data import Bundle, resolve_split
from acie.io import digest_file, read_json, read_jsonl, write_json
from acie.vv_diagnostics import confidence_only_behavior, permute_behavior_within_partition_date

DIRECTIONS = {
    "HUI360_to_SSUP-A": "outputs/frozen_cross_domain/HUI360_to_SSUP-A/split.json",
    "SSUP-A_to_HUI360": "outputs/frozen_cross_domain/SSUP-A_to_HUI360/split.json",
}
SEEDS = (11, 22, 33, 44, 55)
PERMUTATIONS = (20260916, 20260917, 20260918)


def check_prediction(bundle: Bundle, checkpoint: Path, prediction: Path,
                     errors: list[str], label: str) -> None:
    if not prediction.is_file() or not prediction.with_suffix(".metrics.json").is_file():
        errors.append(f"{label}: prediction or metrics missing")
        return
    import torch
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    indices = resolve_split(bundle, payload["split"])["test"]
    rows = read_jsonl(prediction)
    expected_ids = [bundle.meta[i]["sample_id"] for i in indices]
    if [row["sample_id"] for row in rows] != expected_ids:
        errors.append(f"{label}: ID coverage/order mismatch")
    if any(int(row["label"]) != int(bundle.y[i]) for row, i in zip(rows, indices)):
        errors.append(f"{label}: label mismatch")
    checkpoint_hash = digest_file(checkpoint)
    if any(row["checkpoint_hash"] != checkpoint_hash for row in rows):
        errors.append(f"{label}: checkpoint hash mismatch")
    metrics = read_json(prediction.with_suffix(".metrics.json"))
    if any(metrics.get(key) is not False for key in ("weights_updated", "scaler_updated", "threshold_updated")):
        errors.append(f"{label}: mutable scoring flag")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root_default = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument("--data", type=Path, default=Path("data/official_cross_domain.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/vv_followup_v1/r5"))
    args = parser.parse_args()
    root = args.root.resolve()
    resolve = lambda value: value if value.is_absolute() else root / value
    data_path, output = map(resolve, (args.data, args.output))
    lock = read_json(output / "TARGET_SCORING_LOCK.json")
    original = Bundle.load(data_path)
    confidence = confidence_only_behavior(original)
    errors, observed = [], 0
    if lock.get("status") != "locked" or lock.get("checkpoint_count") != 60:
        errors.append("invalid target scoring lock")
    for direction, split_relative in DIRECTIONS.items():
        split = read_json(root / split_relative)
        for permutation_seed in PERMUTATIONS:
            permuted, audit = permute_behavior_within_partition_date(original, split, permutation_seed)
            saved = read_json(output / "r5a" / direction / f"perm_{permutation_seed}" / "PERMUTATION_MAP.json")
            if saved != audit:
                errors.append(f"r5a/{direction}/{permutation_seed}: mapping changed")
            for seed in SEEDS:
                run_dir = output / "r5a" / direction / f"perm_{permutation_seed}" / "F" / str(seed)
                check_prediction(permuted, run_dir / "best.pt", run_dir / "test_predictions.jsonl",
                                 errors, f"r5a/{direction}/{permutation_seed}/{seed}")
                observed += 1
        for seed in SEEDS:
            run_dir = output / "r5b" / direction / "F_conf" / str(seed)
            check_prediction(confidence, run_dir / "best.pt", run_dir / "test_predictions.jsonl",
                             errors, f"r5b/{direction}/{seed}")
            observed += 1
        for model_id in ("F", "Jw"):
            for seed in SEEDS:
                alias = f"{model_id}_weighted"
                run_dir = output / "r5c" / direction / alias / str(seed)
                check_prediction(original, run_dir / "best.pt", run_dir / "test_predictions.jsonl",
                                 errors, f"r5c/{direction}/{alias}/{seed}")
                observed += 1
    if observed != 60:
        errors.append(f"expected 60 slots, observed {observed}")
    summary = output / "r5_summary.json"
    if not summary.is_file() or read_json(summary).get("schema") != "acie.vv-r5-summary.v1":
        errors.append("missing or invalid R5 summary")
    report = {"schema": "acie.vv-r5-score-validation.v1",
              "status": "passed" if not errors else "failed",
              "prediction_slots": observed, "errors": errors}
    write_json(output / "r5_validation.json", report)
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
