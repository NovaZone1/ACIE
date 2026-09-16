#!/usr/bin/env python3
"""Run the triggered R5b2 explicit-velocity ablation as a separate locked study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from acie.data import Bundle, resolve_split
from acie.io import digest_file, read_json, read_jsonl, write_json
from acie.vv_diagnostics import zero_explicit_velocity_behavior
from acie.vv_engine import _tensor_state_digest, score_locked_target, train_source_model
from acie.vv_horizon import paired_date_seed_bootstrap

DIRECTIONS = {
    "HUI360_to_SSUP-A": "outputs/frozen_cross_domain/HUI360_to_SSUP-A/split.json",
    "SSUP-A_to_HUI360": "outputs/frozen_cross_domain/SSUP-A_to_HUI360/split.json",
}
SEEDS = (11, 22, 33, 44, 55)


def paths(root: Path, args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    resolve = lambda value: value if value.is_absolute() else root / value
    return tuple(map(resolve, (args.data, args.output, args.r1, args.r5)))  # type: ignore[return-value]


def train_stage(root: Path, args: argparse.Namespace) -> None:
    data_path, output, r1, _ = paths(root, args)
    bundle = zero_explicit_velocity_behavior(Bundle.load(data_path))
    for direction, split_relative in DIRECTIONS.items():
        split = read_json(root / split_relative)
        for seed in SEEDS:
            geometry = r1 / direction / "G" / "fixed" / str(seed) / "best.pt"
            out = output / direction / "F_no_velocity" / str(seed)
            result = train_source_model(
                bundle, split, "F", seed, out,
                {"device": args.device, "num_threads": args.num_threads},
                geometry_checkpoint=geometry, resume=args.resume,
                context={"experiment": "r5b_velocity", "split_id": "fixed",
                         "data_paths": [str(data_path)], "data_hashes": [digest_file(data_path)]})
            print(json.dumps({"direction": direction, "seed": seed,
                              "status": result["status"]}), flush=True)


def validate_stage(root: Path, args: argparse.Namespace) -> None:
    _, output, r1, _ = paths(root, args)
    errors, checkpoints = [], []
    for direction in DIRECTIONS:
        for seed in SEEDS:
            run_dir = output / direction / "F_no_velocity" / str(seed)
            required = ("run.json", "best.pt", "val_predictions.jsonl", "metrics.json",
                        "timing.json", "initial_state.json", "geometry_reference.json")
            missing = [name for name in required if not (run_dir / name).is_file()]
            if missing:
                errors.append(f"{direction}/{seed}: missing {missing}")
                continue
            if (run_dir / "test_predictions.jsonl").exists():
                errors.append(f"{direction}/{seed}: target prediction exists before lock")
            run = read_json(run_dir / "run.json")
            checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
            geometry_path = r1 / direction / "G" / "fixed" / str(seed) / "best.pt"
            geometry = torch.load(geometry_path, map_location="cpu", weights_only=True)
            reference = read_json(run_dir / "geometry_reference.json")
            if run.get("status") != "complete" or run.get("target_scoring_performed") is not False:
                errors.append(f"{direction}/{seed}: invalid run status")
            if reference["sha256"] != digest_file(geometry_path):
                errors.append(f"{direction}/{seed}: shared G hash mismatch")
            if _tensor_state_digest(checkpoint["geometry_state"]) != _tensor_state_digest(geometry["geometry_state"]):
                errors.append(f"{direction}/{seed}: frozen G changed")
            checkpoints.append({"direction": direction, "seed": seed,
                                "path": str((run_dir / "best.pt").resolve()),
                                "sha256": digest_file(run_dir / "best.pt")})
    report = {"schema": "acie.vv-r5b2-source-validation.v1",
              "status": "passed" if not errors and len(checkpoints) == 10 else "failed",
              "observed_slots": len(checkpoints), "errors": errors}
    write_json(output / "source_validation.json", report)
    if report["status"] != "passed":
        print(json.dumps(report, indent=2)); raise SystemExit(1)
    write_json(output / "TARGET_SCORING_LOCK.json",
               {"schema": "acie.vv-r5b2-target-lock.v1", "status": "locked",
                "checkpoint_count": 10, "checkpoints": checkpoints,
                "selection": "triggered by registered F_conf result; no target model selection"})
    print(json.dumps(report, indent=2))


def score_stage(root: Path, args: argparse.Namespace) -> None:
    data_path, output, _, _ = paths(root, args)
    lock = read_json(output / "TARGET_SCORING_LOCK.json")
    if lock.get("status") != "locked" or lock.get("checkpoint_count") != 10:
        raise ValueError("R5b2 target lock missing")
    bundle = zero_explicit_velocity_behavior(Bundle.load(data_path))
    for direction in DIRECTIONS:
        for seed in SEEDS:
            run_dir = output / direction / "F_no_velocity" / str(seed)
            result = score_locked_target(bundle, run_dir / "best.pt",
                                         run_dir / "test_predictions.jsonl", args.device)
            print(json.dumps({"direction": direction, "seed": seed,
                              "AP": result["metrics"]["AP"]}), flush=True)


def load_matrix(paths_: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loaded = [read_jsonl(path) for path in paths_]
    ids = [row["sample_id"] for row in loaded[0]]
    if any([row["sample_id"] for row in rows] != ids for rows in loaded[1:]):
        raise ValueError("R5b2 prediction coverage differs")
    return (np.asarray([row["label"] for row in loaded[0]], dtype=int),
            np.asarray([[row["score"] for row in rows] for rows in loaded]),
            np.asarray([str(row.get("group_id", row["recording"])) for row in loaded[0]]))


def comparison(y: np.ndarray, groups: np.ndarray, a: np.ndarray, b: np.ndarray,
               name: str, seed: int, repeats: int) -> dict:
    values = [float(average_precision_score(y, a[i]) - average_precision_score(y, b[i]))
              for i in range(5)]
    return {"comparison": name, "seed_AP_differences": values,
            "mean_AP_difference": float(np.mean(values)),
            "positive_seeds": int(np.sum(np.asarray(values) > 0)),
            "bootstrap": paired_date_seed_bootstrap(y, groups, a, b, repeats, seed)}


def summarize_stage(root: Path, args: argparse.Namespace) -> None:
    _, output, r1, r5 = paths(root, args)
    result = {"schema": "acie.vv-r5b2-velocity-summary.v1", "directions": {}}
    for direction_index, direction in enumerate(DIRECTIONS):
        y, real, groups = load_matrix([r1 / direction / "F" / "fixed" / str(seed) /
                                        "test_predictions.jsonl" for seed in SEEDS])
        _, no_velocity, _ = load_matrix([output / direction / "F_no_velocity" / str(seed) /
                                          "test_predictions.jsonl" for seed in SEEDS])
        _, confidence, _ = load_matrix([r5 / "r5b" / direction / "F_conf" / str(seed) /
                                         "test_predictions.jsonl" for seed in SEEDS])
        result["directions"][direction] = [
            comparison(y, groups, real, no_velocity, "F_real-F_no_velocity",
                       20261216 + direction_index * 100, args.bootstrap),
            comparison(y, groups, no_velocity, confidence, "F_no_velocity-F_conf",
                       20261217 + direction_index * 100, args.bootstrap),
        ]
    write_json(output / "r5b2_summary.json", result)
    print(json.dumps({"status": "complete", "summary": str(output / "r5b2_summary.json")}, indent=2))


def final_validate_stage(root: Path, args: argparse.Namespace) -> None:
    _, output, _, _ = paths(root, args)
    errors, count = [], 0
    lock = read_json(output / "TARGET_SCORING_LOCK.json")
    hashes = {(row["direction"], int(row["seed"])): row["sha256"] for row in lock["checkpoints"]}
    for direction in DIRECTIONS:
        for seed in SEEDS:
            run_dir = output / direction / "F_no_velocity" / str(seed)
            rows = read_jsonl(run_dir / "test_predictions.jsonl")
            if not rows or any(row["checkpoint_hash"] != hashes[(direction, seed)] for row in rows):
                errors.append(f"{direction}/{seed}: missing predictions or checkpoint mismatch")
            metrics = read_json(run_dir / "test_predictions.metrics.json")
            if any(metrics.get(key) is not False for key in ("weights_updated", "scaler_updated", "threshold_updated")):
                errors.append(f"{direction}/{seed}: mutable scoring flag")
            count += 1
    if not (output / "r5b2_summary.json").is_file():
        errors.append("summary missing")
    report = {"schema": "acie.vv-r5b2-final-validation.v1",
              "status": "passed" if not errors and count == 10 else "failed",
              "prediction_slots": count, "errors": errors}
    write_json(output / "r5b2_validation.json", report)
    print(json.dumps(report, indent=2))
    if report["status"] != "passed": raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root_default = Path(__file__).resolve().parents[1]
    parser.add_argument("stage", choices=("train", "validate", "score", "summarize", "final-validate"))
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument("--data", type=Path, default=Path("data/official_cross_domain.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/vv_followup_v1/r5/r5b_velocity"))
    parser.add_argument("--r1", type=Path, default=Path("outputs/vv_followup_v1/r1"))
    parser.add_argument("--r5", type=Path, default=Path("outputs/vv_followup_v1/r5"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); root = args.root.resolve()
    {"train": train_stage, "validate": validate_stage, "score": score_stage,
     "summarize": summarize_stage, "final-validate": final_validate_stage}[args.stage](root, args)


if __name__ == "__main__":
    main()
