#!/usr/bin/env python3
"""Train the registered R5a/R5b/R5c source-only diagnostic matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from acie.data import Bundle, resolve_split
from acie.io import digest_file, read_json, write_json
from acie.vv_diagnostics import confidence_only_behavior, permute_behavior_within_partition_date
from acie.vv_engine import train_source_model

DIRECTIONS = {
    "HUI360_to_SSUP-A": "outputs/frozen_cross_domain/HUI360_to_SSUP-A/split.json",
    "SSUP-A_to_HUI360": "outputs/frozen_cross_domain/SSUP-A_to_HUI360/split.json",
}
SEEDS = (11, 22, 33, 44, 55)
PERMUTATIONS = (20260916, 20260917, 20260918)


def train(bundle: Bundle, split: dict, model_id: str, seed: int, out: Path,
          geometry: Path, args: argparse.Namespace, experiment: str,
          positive_class_weight: float | None = None) -> None:
    config = {"device": args.device, "num_threads": args.num_threads}
    result = train_source_model(
        bundle, split, model_id, seed, out, config, geometry_checkpoint=geometry,
        resume=args.resume, positive_class_weight=positive_class_weight,
        context={"experiment": experiment, "split_id": "fixed",
                 "data_paths": [str(args.data_resolved)],
                 "data_hashes": [digest_file(args.data_resolved)]},
    )
    print(json.dumps({"experiment": experiment, "model": model_id, "seed": seed,
                      "status": result["status"], "out": str(out)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root_default = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument("--data", type=Path, default=Path("data/official_cross_domain.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/vv_followup_v1/r5"))
    parser.add_argument("--r1", type=Path, default=Path("outputs/vv_followup_v1/r1"))
    parser.add_argument("--modes", nargs="+", choices=("r5a", "r5b", "r5c"), default=["r5a", "r5b", "r5c"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    resolve = lambda value: value if value.is_absolute() else root / value
    args.data_resolved, output, r1 = map(resolve, (args.data, args.output, args.r1))
    slots = (30 if "r5a" in args.modes else 0) + (10 if "r5b" in args.modes else 0) + (20 if "r5c" in args.modes else 0)
    if args.dry_run:
        print(json.dumps({"training_slots": slots, "r5a": 30, "r5b": 10, "r5c": 20,
                          "modes": args.modes, "target_scoring": False}, indent=2))
        return
    original = Bundle.load(args.data_resolved)
    confidence = confidence_only_behavior(original) if "r5b" in args.modes else None
    completed = 0
    for direction, split_relative in DIRECTIONS.items():
        split = read_json(root / split_relative)
        if "r5a" in args.modes:
            for permutation_seed in PERMUTATIONS:
                permuted, audit = permute_behavior_within_partition_date(original, split, permutation_seed)
                audit_path = output / "r5a" / direction / f"perm_{permutation_seed}" / "PERMUTATION_MAP.json"
                if audit_path.exists() and read_json(audit_path) != audit:
                    raise ValueError(f"Permutation mapping changed: {audit_path}")
                write_json(audit_path, audit)
                for seed in SEEDS:
                    geometry = r1 / direction / "G" / "fixed" / str(seed) / "best.pt"
                    out = output / "r5a" / direction / f"perm_{permutation_seed}" / "F" / str(seed)
                    train(permuted, split, "F", seed, out, geometry, args, "r5a")
                    completed += 1
        if "r5b" in args.modes:
            assert confidence is not None
            for seed in SEEDS:
                geometry = r1 / direction / "G" / "fixed" / str(seed) / "best.pt"
                out = output / "r5b" / direction / "F_conf" / str(seed)
                train(confidence, split, "F", seed, out, geometry, args, "r5b")
                completed += 1
        if "r5c" in args.modes:
            indices = resolve_split(original, split)["train"]
            positives = int(original.y[indices].sum())
            class_weight = float((len(indices) - positives) / positives)
            for model_id in ("F", "Jw"):
                for seed in SEEDS:
                    geometry = r1 / direction / "G" / "fixed" / str(seed) / "best.pt"
                    out = output / "r5c" / direction / f"{model_id}_weighted" / str(seed)
                    train(original, split, model_id, seed, out, geometry, args, "r5c", class_weight)
                    completed += 1
    if completed != slots:
        raise AssertionError(f"Expected {slots} R5 slots, observed {completed}")


if __name__ == "__main__":
    main()
