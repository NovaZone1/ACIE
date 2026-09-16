#!/usr/bin/env python3
"""Train the VV R1/final source-only model matrix without target scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from acie.data import Bundle
from acie.io import digest_file, digest_object, read_json
from acie.vv_engine import DEFAULT_CONFIG, train_source_model
from acie.vv_models import MODEL_IDS


DIRECTIONS = {
    "HUI360_to_SSUP-A": "outputs/frozen_cross_domain/HUI360_to_SSUP-A/split.json",
    "SSUP-A_to_HUI360": "outputs/frozen_cross_domain/SSUP-A_to_HUI360/split.json",
}
SEEDS = (11, 22, 33, 44, 55)


def source_snapshot(root: Path) -> tuple[str, dict[str, str]]:
    files = [
        root / "src/acie/data.py", root / "src/acie/features.py", root / "src/acie/io.py",
        root / "src/acie/metrics.py", root / "src/acie/vv_models.py", root / "src/acie/vv_engine.py",
        Path(__file__).resolve(),
    ]
    hashes = {str(path.relative_to(root)): digest_file(path) for path in files}
    return digest_object(hashes), hashes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data", type=Path, default=Path("data/official_cross_domain.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/vv_followup_v1"))
    parser.add_argument("--experiment", default="r1")
    parser.add_argument("--directions", nargs="+", choices=tuple(DIRECTIONS), default=list(DIRECTIONS))
    parser.add_argument("--models", nargs="+", choices=MODEL_IDS, default=list(MODEL_IDS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--geometry-epochs", type=int, default=DEFAULT_CONFIG["geometry_epochs"])
    parser.add_argument("--stage2-epochs", type=int, default=DEFAULT_CONFIG["stage2_epochs"])
    parser.add_argument("--single-stage-epochs", type=int, default=DEFAULT_CONFIG["single_stage_epochs"])
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG["patience"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    data_path = args.data if args.data.is_absolute() else root / args.data
    output = args.output if args.output.is_absolute() else root / args.output
    requested = list(dict.fromkeys(args.models))
    dependency_g = any(model in ("F", "Jw", "JwD") for model in requested)
    planned_models = list(requested)
    if dependency_g and "G" not in planned_models:
        planned_models.insert(0, "G")
    # Public G must be produced before dependent models.
    planned_models.sort(key=lambda value: (value != "G", MODEL_IDS.index(value)))
    plans = []
    for direction in args.directions:
        for seed in args.seeds:
            for model_id in planned_models:
                implicit = model_id == "G" and model_id not in requested
                plans.append({
                    "direction": direction, "model_id": model_id, "seed": seed,
                    "implicit_shared_geometry_dependency": implicit,
                    "out": str(output / args.experiment / direction / model_id / "fixed" / str(seed)),
                })
    if args.dry_run:
        print(json.dumps({
            "schema": "acie.vv-train-dry-run.v1",
            "target_scoring": False,
            "directions": args.directions,
            "requested_models": requested,
            "seeds": args.seeds,
            "requested_slots": len(args.directions) * len(requested) * len(args.seeds),
            "actual_training_slots_including_dependencies": len(plans),
            "plans": plans,
        }, indent=2))
        return

    bundle = Bundle.load(data_path)
    snapshot_hash, snapshot_files = source_snapshot(root)
    config = {
        "lr": args.lr, "weight_decay": args.weight_decay, "device": args.device,
        "num_threads": args.num_threads, "geometry_epochs": args.geometry_epochs,
        "stage2_epochs": args.stage2_epochs, "single_stage_epochs": args.single_stage_epochs,
        "patience": args.patience,
    }
    for direction in args.directions:
        split_path = root / DIRECTIONS[direction]
        split = read_json(split_path)
        for seed in args.seeds:
            geometry_path = output / args.experiment / direction / "G" / "fixed" / str(seed) / "best.pt"
            for model_id in planned_models:
                out = output / args.experiment / direction / model_id / "fixed" / str(seed)
                context = {
                    "experiment": args.experiment,
                    "split_id": "fixed",
                    "data_paths": [str(data_path.resolve()), str(split_path.resolve())],
                    "data_hashes": [digest_file(data_path), digest_file(split_path)],
                    "source_snapshot_hash": snapshot_hash,
                    "source_snapshot_files": snapshot_files,
                }
                checkpoint = geometry_path if model_id in ("F", "Jw", "JwD") else None
                result = train_source_model(bundle, split, model_id, seed, out, config,
                                            geometry_checkpoint=checkpoint, resume=args.resume,
                                            context=context)
                print(json.dumps({"direction": direction, "model": model_id, "seed": seed,
                                  "status": result["status"], "out": str(out)}))


if __name__ == "__main__":
    main()
