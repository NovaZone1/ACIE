#!/usr/bin/env python3
"""Score the complete locked R5 diagnostic matrix on target data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from acie.data import Bundle
from acie.io import read_json
from acie.vv_diagnostics import confidence_only_behavior, permute_behavior_within_partition_date
from acie.vv_engine import score_locked_target

DIRECTIONS = {
    "HUI360_to_SSUP-A": "outputs/frozen_cross_domain/HUI360_to_SSUP-A/split.json",
    "SSUP-A_to_HUI360": "outputs/frozen_cross_domain/SSUP-A_to_HUI360/split.json",
}
SEEDS = (11, 22, 33, 44, 55)
PERMUTATIONS = (20260916, 20260917, 20260918)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root_default = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument("--data", type=Path, default=Path("data/official_cross_domain.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/vv_followup_v1/r5"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    root = args.root.resolve()
    resolve = lambda value: value if value.is_absolute() else root / value
    data_path, output = map(resolve, (args.data, args.output))
    lock = read_json(output / "TARGET_SCORING_LOCK.json")
    if lock.get("status") != "locked" or lock.get("checkpoint_count") != 60:
        raise ValueError("R5 target scoring lock is absent or incomplete")
    original = Bundle.load(data_path)
    confidence = confidence_only_behavior(original)
    completed = 0
    for direction, split_relative in DIRECTIONS.items():
        split = read_json(root / split_relative)
        for permutation_seed in PERMUTATIONS:
            permuted, audit = permute_behavior_within_partition_date(original, split, permutation_seed)
            saved = read_json(output / "r5a" / direction / f"perm_{permutation_seed}" / "PERMUTATION_MAP.json")
            if saved != audit:
                raise ValueError("R5a permutation mapping changed after target lock")
            for seed in SEEDS:
                run_dir = output / "r5a" / direction / f"perm_{permutation_seed}" / "F" / str(seed)
                result = score_locked_target(permuted, run_dir / "best.pt", run_dir / "test_predictions.jsonl", args.device)
                completed += 1
                print(json.dumps({"slot": completed, "experiment": "r5a", "direction": direction,
                                  "permutation": permutation_seed, "seed": seed,
                                  "AP": result["metrics"]["AP"]}), flush=True)
        for seed in SEEDS:
            run_dir = output / "r5b" / direction / "F_conf" / str(seed)
            result = score_locked_target(confidence, run_dir / "best.pt", run_dir / "test_predictions.jsonl", args.device)
            completed += 1
            print(json.dumps({"slot": completed, "experiment": "r5b", "direction": direction,
                              "seed": seed, "AP": result["metrics"]["AP"]}), flush=True)
        for model_id in ("F", "Jw"):
            for seed in SEEDS:
                run_dir = output / "r5c" / direction / f"{model_id}_weighted" / str(seed)
                result = score_locked_target(original, run_dir / "best.pt", run_dir / "test_predictions.jsonl", args.device)
                completed += 1
                print(json.dumps({"slot": completed, "experiment": "r5c", "direction": direction,
                                  "model": f"{model_id}_weighted", "seed": seed,
                                  "AP": result["metrics"]["AP"]}), flush=True)
    if completed != 60:
        raise AssertionError(f"Expected 60 R5 predictions, observed {completed}")


if __name__ == "__main__":
    main()
