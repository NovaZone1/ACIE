#!/usr/bin/env python3
"""Validate all R5 source-only runs before any target scoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from acie.data import Bundle, resolve_split
from acie.io import digest_file, read_json, write_json
from acie.vv_diagnostics import permute_behavior_within_partition_date
from acie.vv_engine import _tensor_state_digest

DIRECTIONS = {
    "HUI360_to_SSUP-A": "outputs/frozen_cross_domain/HUI360_to_SSUP-A/split.json",
    "SSUP-A_to_HUI360": "outputs/frozen_cross_domain/SSUP-A_to_HUI360/split.json",
}
SEEDS = (11, 22, 33, 44, 55)
PERMUTATIONS = (20260916, 20260917, 20260918)
REQUIRED = ("run.json", "best.pt", "val_predictions.jsonl", "metrics.json", "timing.json",
            "initial_state.json", "geometry_reference.json")


def validate_run(run_dir: Path, model_id: str, seed: int, geometry: Path,
                 errors: list[str], label: str, positive_class_weight: float | None = None) -> dict | None:
    missing = [name for name in REQUIRED if not (run_dir / name).is_file()]
    if missing:
        errors.append(f"{label}: missing {missing}")
        return None
    if (run_dir / "test_predictions.jsonl").exists():
        errors.append(f"{label}: target prediction exists before lock")
    run = read_json(run_dir / "run.json")
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    reference = read_json(run_dir / "geometry_reference.json")
    g = torch.load(geometry, map_location="cpu", weights_only=True)
    if run.get("status") != "complete" or run.get("target_scoring_performed") is not False:
        errors.append(f"{label}: invalid run status")
    if checkpoint.get("schema") != "acie.vv.checkpoint.v1" or checkpoint["model_spec"]["model_id"] != model_id:
        errors.append(f"{label}: checkpoint identity mismatch")
    if int(checkpoint["master_seed"]) != seed:
        errors.append(f"{label}: seed mismatch")
    if reference["sha256"] != digest_file(geometry) or reference["state_sha256"] != _tensor_state_digest(g["geometry_state"]):
        errors.append(f"{label}: shared G mismatch")
    if model_id == "F" and _tensor_state_digest(checkpoint["geometry_state"]) != _tensor_state_digest(g["geometry_state"]):
        errors.append(f"{label}: frozen G changed")
    observed_weight = checkpoint.get("positive_class_weight")
    if observed_weight != positive_class_weight or run.get("positive_class_weight") != positive_class_weight:
        errors.append(f"{label}: positive class weight mismatch")
    return read_json(run_dir / "initial_state.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root_default = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument("--data", type=Path, default=Path("data/official_cross_domain.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/vv_followup_v1/r5"))
    parser.add_argument("--r1", type=Path, default=Path("outputs/vv_followup_v1/r1"))
    args = parser.parse_args()
    root = args.root.resolve()
    resolve = lambda value: value if value.is_absolute() else root / value
    data_path, output, r1 = map(resolve, (args.data, args.output, args.r1))
    bundle = Bundle.load(data_path)
    errors, checkpoints = [], []
    for direction, split_relative in DIRECTIONS.items():
        split = read_json(root / split_relative)
        for permutation_seed in PERMUTATIONS:
            _, expected_audit = permute_behavior_within_partition_date(bundle, split, permutation_seed)
            audit_path = output / "r5a" / direction / f"perm_{permutation_seed}" / "PERMUTATION_MAP.json"
            if not audit_path.is_file() or read_json(audit_path) != expected_audit:
                errors.append(f"r5a/{direction}/{permutation_seed}: permutation audit mismatch")
            for seed in SEEDS:
                geometry = r1 / direction / "G" / "fixed" / str(seed) / "best.pt"
                run_dir = output / "r5a" / direction / f"perm_{permutation_seed}" / "F" / str(seed)
                state = validate_run(run_dir, "F", seed, geometry, errors,
                                     f"r5a/{direction}/{permutation_seed}/{seed}")
                real_state = read_json(r1 / direction / "F" / "fixed" / str(seed) / "initial_state.json")
                if state and any(state[key] != real_state[key] for key in ("behavior_state_sha256", "first_batch_ids_sha256")):
                    errors.append(f"r5a/{direction}/{permutation_seed}/{seed}: initialization or batches differ")
                if (run_dir / "best.pt").is_file():
                    checkpoints.append({"experiment": "r5a", "direction": direction,
                                        "permutation_seed": permutation_seed, "model_id": "F", "seed": seed,
                                        "path": str((run_dir / "best.pt").resolve()),
                                        "sha256": digest_file(run_dir / "best.pt")})
        for seed in SEEDS:
            geometry = r1 / direction / "G" / "fixed" / str(seed) / "best.pt"
            run_dir = output / "r5b" / direction / "F_conf" / str(seed)
            state = validate_run(run_dir, "F", seed, geometry, errors, f"r5b/{direction}/{seed}")
            real_state = read_json(r1 / direction / "F" / "fixed" / str(seed) / "initial_state.json")
            if state and any(state[key] != real_state[key] for key in ("behavior_state_sha256", "first_batch_ids_sha256")):
                errors.append(f"r5b/{direction}/{seed}: initialization or batches differ")
            if (run_dir / "best.pt").is_file():
                checkpoints.append({"experiment": "r5b", "direction": direction, "model_id": "F_conf",
                                    "seed": seed, "path": str((run_dir / "best.pt").resolve()),
                                    "sha256": digest_file(run_dir / "best.pt")})
        train_ix = resolve_split(bundle, split)["train"]
        positives = int(bundle.y[train_ix].sum())
        weight = float((len(train_ix) - positives) / positives)
        for seed in SEEDS:
            states = {}
            geometry = r1 / direction / "G" / "fixed" / str(seed) / "best.pt"
            for model_id in ("F", "Jw"):
                alias = f"{model_id}_weighted"
                run_dir = output / "r5c" / direction / alias / str(seed)
                states[model_id] = validate_run(run_dir, model_id, seed, geometry, errors,
                                                f"r5c/{direction}/{alias}/{seed}", weight)
                if (run_dir / "best.pt").is_file():
                    checkpoints.append({"experiment": "r5c", "direction": direction, "model_id": alias,
                                        "seed": seed, "path": str((run_dir / "best.pt").resolve()),
                                        "sha256": digest_file(run_dir / "best.pt"),
                                        "positive_class_weight": weight})
            if all(states.values()):
                for key in ("behavior_state_sha256", "first_batch_ids_sha256", "first_batch_initial_logits_sha256"):
                    if states["F"][key] != states["Jw"][key]:
                        errors.append(f"r5c/{direction}/{seed}: F/Jw differ for {key}")
    report = {"schema": "acie.vv-r5-source-validation.v1",
              "status": "passed" if not errors else "failed", "expected_slots": 60,
              "observed_slots": len(checkpoints), "errors": errors}
    write_json(output / "source_validation.json", report)
    if errors or len(checkpoints) != 60:
        print(json.dumps(report, indent=2))
        raise SystemExit(1)
    lock = {"schema": "acie.vv-r5-target-scoring-lock.v1", "status": "locked",
            "checkpoint_count": len(checkpoints), "checkpoints": checkpoints,
            "selection": "registered R5 transformations and R1 fixed configurations; no target selection"}
    write_json(output / "TARGET_SCORING_LOCK.json", lock)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
