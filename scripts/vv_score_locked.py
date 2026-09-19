#!/usr/bin/env python3
"""Generate target predictions only after a VV model/configuration lock exists."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from acie.data import Bundle
from acie.vv_engine import score_locked_target
from acie.vv_models import MODEL_IDS
from acie.vv_provenance import validate_locked_checkpoint


DIRECTIONS = ("HUI360_to_SSUP-A", "SSUP-A_to_HUI360")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data", type=Path, default=Path("data/official_cross_domain.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/vv_followup_v1"))
    parser.add_argument("--experiment", default="r1")
    parser.add_argument("--directions", nargs="+", choices=DIRECTIONS, default=list(DIRECTIONS))
    parser.add_argument("--models", nargs="+", choices=MODEL_IDS, default=list(MODEL_IDS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    data_path = args.data if args.data.is_absolute() else root / args.data
    output = args.output if args.output.is_absolute() else root / args.output
    lock_path = (args.lock if args.lock and args.lock.is_absolute() else
                 root / args.lock if args.lock else output / args.experiment / "TARGET_SCORING_LOCK.json")
    plans = []
    for direction in args.directions:
        for model_id in args.models:
            for seed in args.seeds:
                run = output / args.experiment / direction / model_id / "fixed" / str(seed)
                plans.append({"direction": direction, "model_id": model_id, "seed": seed,
                              "checkpoint": str(run / "best.pt"),
                              "prediction": str(run / "test_predictions.jsonl")})
    if args.dry_run:
        print(json.dumps({"schema": "acie.vv-score-dry-run.v1", "training": False,
                          "slots": len(plans), "plans": plans}, indent=2))
        return
    missing = [plan["checkpoint"] for plan in plans if not Path(plan["checkpoint"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Refusing partial target scoring; {len(missing)} checkpoints are missing. First: {missing[0]}")
    bundle = Bundle.load(data_path)
    for plan in plans:
        validate_locked_checkpoint(lock_path, plan["checkpoint"], direction=plan["direction"],
                                   model_id=plan["model_id"], seed=plan["seed"])
        result = score_locked_target(bundle, plan["checkpoint"], plan["prediction"], args.device)
        print(json.dumps({**plan, "AP": result["metrics"]["AP"], "status": "complete"}))


if __name__ == "__main__":
    main()
