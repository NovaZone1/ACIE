#!/usr/bin/env python3
"""Freeze the R4 horizon bundles and source-selected R2 checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from acie.data import Bundle
from acie.io import digest_file, digest_object, read_json, write_json

DIRECTIONS = {"HUI360_to_SSUP-A": "SSUP-A", "SSUP-A_to_HUI360": "HUI360"}
CUTOFFS = (16, 24, 32, 40, 48)
SEEDS = (11, 22, 33, 44, 55)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--data", type=Path, default=Path("data/horizon_evaluation"))
    parser.add_argument("--models", type=Path, default=Path("outputs/vv_followup_v1/r2_final"))
    parser.add_argument("--selection", type=Path, default=Path("outputs/vv_followup_v1/r2_cv"))
    parser.add_argument("--out", type=Path, default=Path("protocols/vv_followup_v1/R4_PROTOCOL_LOCK.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    resolve = lambda value: value if value.is_absolute() else root / value
    data, models, selection, out = map(resolve, (args.data, args.models, args.selection, args.out))

    directions = {}
    for direction, target in DIRECTIONS.items():
        selected = read_json(selection / direction / "MODEL_SELECTION_LOCK.json")
        model_ids = ["G", "F", selected["J_star"], selected["C_star"], "histgb"]
        checkpoints = {}
        for model_id in model_ids:
            seeds = (11,) if model_id == "histgb" else SEEDS
            checkpoints[model_id] = {}
            for seed in seeds:
                name = "tree.joblib" if model_id == "histgb" else "best.pt"
                path = models / direction / model_id / "fixed" / str(seed) / name
                if not path.is_file():
                    raise FileNotFoundError(path)
                checkpoints[model_id][str(seed)] = {
                    "path": str(path.relative_to(root)), "sha256": digest_file(path)
                }
        bundles = []
        for cutoff in CUTOFFS:
            path = data / target / f"cutoff_frames_{cutoff:03d}" / "bundle.npz"
            bundle = Bundle.load(path)
            bundles.append({
                "cutoff_frames": cutoff, "cutoff_seconds": cutoff / 15.0,
                "path": str(path.relative_to(root)), "sha256": digest_file(path),
                "n": len(bundle.y), "positives": int(bundle.y.sum()),
                "sample_id_digest": digest_object(sorted(meta["sample_id"] for meta in bundle.meta)),
            })
        directions[direction] = {
            "target": target, "J_star": selected["J_star"], "C_star": selected["C_star"],
            "models": model_ids, "checkpoints": checkpoints, "bundles": bundles,
        }
    protocol = {
        "schema": "acie.vv-r4-protocol-lock.v1", "status": "locked_before_r4_scoring",
        "selection": "R2 source-only selected identities and fixed final checkpoints",
        "target_labels_used_for_selection": False, "fps": 15.0,
        "cutoff_frames": list(CUTOFFS), "seeds": list(SEEDS),
        "primary_set": "tracks present at all five horizons within each target",
        "thresholds": {
            "f1": "each seed source-val maximum-F1 threshold frozen in checkpoint",
            "fpr_budgets": [0.01, 0.05],
            "fpr_rule": "on source val maximize recall subject to FPR budget; ties choose higher threshold; include all-negative threshold",
        },
        "comparisons": ["F-G", "F-J_star", "F-C_star"], "directions": directions,
        "training_in_r4": False,
    }
    protocol["protocol_digest"] = digest_object(protocol)
    if out.exists() and read_json(out) != protocol:
        raise ValueError(f"Existing R4 protocol differs: {out}")
    write_json(out, protocol)
    print(json.dumps({"status": "locked", "path": str(out), "digest": protocol["protocol_digest"],
                      "neural_predictions": 200, "tree_predictions": 10}, indent=2))


if __name__ == "__main__":
    main()
