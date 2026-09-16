#!/usr/bin/env python3
"""Validate and lock the restored VV data, splits, model specs, and R1 protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from acie.data import Bundle, group_key, resolve_split, track_key
from acie.io import digest_file, digest_object, read_json, write_json
from acie.vv_engine import DEFAULT_CONFIG
from acie.vv_models import EXPECTED_PARAMETERS, MODEL_IDS, build_model, effective_parameter_count


DIRECTIONS = {
    "HUI360_to_SSUP-A": "outputs/frozen_cross_domain/HUI360_to_SSUP-A/split.json",
    "SSUP-A_to_HUI360": "outputs/frozen_cross_domain/SSUP-A_to_HUI360/split.json",
}
EXPECTED = {
    "HUI360_to_SSUP-A": {
        "train": (1149, 164, 12), "val": (268, 49, 3), "test": (4875, 148, 5),
    },
    "SSUP-A_to_HUI360": {
        "train": (4896, 105, 4), "val": (1202, 30, 1), "test": (407, 68, 9),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data", type=Path, default=Path("data/official_cross_domain.npz"))
    parser.add_argument("--protocol", type=Path,
                        default=Path("protocols/vv_followup_v1/protocol_lock.json"))
    parser.add_argument("--config", type=Path,
                        default=Path("configs/vv_followup_v1/r1_fixed.json"))
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    data_path = args.data if args.data.is_absolute() else root / args.data
    bundle = Bundle.load(data_path)
    if bundle.a.shape[1:] != (32,) or bundle.q.shape[1:] != (32, 119):
        raise ValueError(f"Unexpected VV input shape: a={bundle.a.shape}, q={bundle.q.shape}")
    direction_checks = {}
    for direction, relative in DIRECTIONS.items():
        split_path = root / relative
        split = read_json(split_path)
        indices = resolve_split(bundle, split)
        partitions = {}
        for name, ix in indices.items():
            observed = (len(ix), int(bundle.y[ix].sum()), len({group_key(bundle.meta[i]) for i in ix}))
            if observed != EXPECTED[direction][name]:
                raise AssertionError(f"{direction}/{name}: expected {EXPECTED[direction][name]}, got {observed}")
            partitions[name] = {"n": observed[0], "positives": observed[1], "date_groups": observed[2],
                                "sample_ids_sha256": digest_object([bundle.meta[i]["sample_id"] for i in ix])}
        direction_checks[direction] = {
            "split_path": str(split_path.resolve()), "split_sha256": digest_file(split_path),
            "split_object_sha256": digest_object(split), "partitions": partitions,
        }

    parameter_counts = {}
    model_specs = {}
    for model_id in MODEL_IDS:
        model = build_model(model_id)
        count = effective_parameter_count(model)
        if count != EXPECTED_PARAMETERS[model_id]:
            raise AssertionError(f"{model_id}: expected {EXPECTED_PARAMETERS[model_id]}, got {count}")
        parameter_counts[model_id] = count
        model_specs[model_id] = model.spec

    source_files = [
        "src/acie/data.py", "src/acie/features.py", "src/acie/io.py", "src/acie/metrics.py",
        "src/acie/vv_models.py", "src/acie/vv_engine.py", "src/acie/vv_trees.py",
        "scripts/vv_prepare_data.py", "scripts/vv_train_final.py", "scripts/vv_score_locked.py",
        "scripts/vv_train_trees.py", "scripts/vv_score_trees_locked.py",
    ]
    source_hashes = {path: digest_file(root / path) for path in source_files}
    fixed_config = dict(DEFAULT_CONFIG)
    fixed_config.update({
        "loss": "BCEWithLogitsLoss", "class_weighted": False, "pair_lambda": 0.0,
        "epoch_selection": "strict source validation AP improvement; earlier epoch wins ties",
        "threshold_selection": "source validation maximum F1",
    })
    lock = {
        "schema": "acie.vv-followup-protocol.v1",
        "status": "locked_for_followup",
        "locked_at": "2026-09-16",
        "training_started_at_lock_creation": False,
        "study_note": "Follow-up designed after historical target results were already known; not a new blind test.",
        "feature_contract": {"a_dim": 32, "q_steps": 32, "q_dim": 119,
                             "implementation": "acie.features.extract + SourceScaler"},
        "data": {"path": str(data_path.resolve()), "npz_sha256": digest_file(data_path),
                 "json_sha256": digest_file(data_path.with_suffix(".json")),
                 "n": len(bundle.y), "positives": int(bundle.y.sum())},
        "directions": direction_checks,
        "models": model_specs,
        "effective_parameters": parameter_counts,
        "r1": {
            "models": list(MODEL_IDS), "seeds": [11, 22, 33, 44, 55],
            "neural_slots": 80,
            "primary_comparisons": [
                "HUI360_to_SSUP-A:F-Jw", "HUI360_to_SSUP-A:F-C64",
                "SSUP-A_to_HUI360:F-Jw", "SSUP-A_to_HUI360:F-C64",
            ],
            "config": fixed_config,
        },
        "statistics": {"primary_metric": "average_precision_score",
                       "single_model_summary": "five-seed arithmetic mean and sample SD (ddof=1)",
                       "paired_two_level_bootstrap_repeats": 10000,
                       "bootstrap_seed": 20260916,
                       "intervals": [0.95, 0.9875]},
        "source_snapshot": {"files": source_hashes, "sha256": digest_object(source_hashes)},
        "target_access": "No target prediction during training; separate vv_score_locked.py after full matrix lock.",
    }
    report = {"status": "passed", "data_shape": {"a": list(bundle.a.shape), "q": list(bundle.q.shape)},
              "directions": direction_checks, "parameter_counts": parameter_counts,
              "protocol_sha256": digest_object(lock)}
    if not args.check_only:
        protocol_path = args.protocol if args.protocol.is_absolute() else root / args.protocol
        config_path = args.config if args.config.is_absolute() else root / args.config
        write_json(protocol_path, lock)
        write_json(config_path, fixed_config)
        report["protocol_path"] = str(protocol_path)
        report["config_path"] = str(config_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
