#!/usr/bin/env python3
"""Validate the frozen R4 protocol, 210 predictions, and base-horizon replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from acie.data import Bundle
from acie.io import digest_file, read_json, read_jsonl, write_json
from vv_score_r4 import prediction_path, verify_protocol


def rows_by_id(path: Path) -> tuple[list[dict], dict[str, dict]]:
    rows = read_jsonl(path)
    table = {row["sample_id"]: row for row in rows}
    if len(table) != len(rows):
        raise ValueError(f"Duplicate prediction IDs: {path}")
    return rows, table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root_default = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument("--protocol", type=Path, default=Path("protocols/vv_followup_v1/R4_PROTOCOL_LOCK.json"))
    parser.add_argument("--out", type=Path, default=Path("outputs/vv_followup_v1/r4"))
    parser.add_argument("--r2", type=Path, default=Path("outputs/vv_followup_v1/r2_final"))
    args = parser.parse_args()
    root = args.root.resolve()
    resolve = lambda value: value if value.is_absolute() else root / value
    protocol_path, out, r2 = map(resolve, (args.protocol, args.out, args.r2))
    protocol = read_json(protocol_path)
    verify_protocol(protocol)
    errors, base_replay = [], []
    prediction_files = 0
    for direction, spec in protocol["directions"].items():
        horizon_sets = []
        for bundle_spec in spec["bundles"]:
            cutoff = int(bundle_spec["cutoff_frames"])
            bundle_path = root / bundle_spec["path"]
            if digest_file(bundle_path) != bundle_spec["sha256"]:
                errors.append(f"bundle hash mismatch {direction}/{cutoff}")
                continue
            bundle = Bundle.load(bundle_path)
            expected_ids = [meta["sample_id"] for meta in bundle.meta]
            expected_labels = {meta["sample_id"]: int(label) for meta, label in zip(bundle.meta, bundle.y)}
            horizon_sets.append(set(expected_ids))
            if len(expected_ids) != bundle_spec["n"] or int(bundle.y.sum()) != bundle_spec["positives"]:
                errors.append(f"bundle counts mismatch {direction}/{cutoff}")
            for model in spec["models"]:
                for seed_text, checkpoint_spec in spec["checkpoints"][model].items():
                    seed = int(seed_text)
                    checkpoint = root / checkpoint_spec["path"]
                    if digest_file(checkpoint) != checkpoint_spec["sha256"]:
                        errors.append(f"checkpoint hash mismatch {direction}/{model}/{seed}")
                    path = prediction_path(out, direction, model, seed, cutoff)
                    if not path.is_file() or not path.with_suffix(".metrics.json").is_file():
                        errors.append(f"missing prediction or metrics {path}")
                        continue
                    prediction_files += 1
                    rows, current = rows_by_id(path)
                    if [row["sample_id"] for row in rows] != expected_ids:
                        errors.append(f"prediction order/coverage mismatch {direction}/{model}/{seed}/{cutoff}")
                    if any(int(row["label"]) != expected_labels[row["sample_id"]] for row in rows):
                        errors.append(f"label mismatch {direction}/{model}/{seed}/{cutoff}")
                    if any(row["checkpoint_hash"] != checkpoint_spec["sha256"] for row in rows):
                        errors.append(f"prediction checkpoint mismatch {direction}/{model}/{seed}/{cutoff}")
                    metrics = read_json(path.with_suffix(".metrics.json"))
                    if metrics.get("weights_updated") is not False or metrics.get("scaler_updated") is not False or metrics.get("threshold_updated") is not False:
                        errors.append(f"mutable R4 scoring flags {direction}/{model}/{seed}/{cutoff}")
                    if metrics.get("bundle_sha256") != bundle_spec["sha256"]:
                        errors.append(f"prediction bundle hash mismatch {direction}/{model}/{seed}/{cutoff}")
                    if cutoff == 16:
                        old_path = r2 / direction / model / "fixed" / str(seed) / "test_predictions.jsonl"
                        _, old = rows_by_id(old_path)
                        if set(current) != set(old):
                            errors.append(f"base IDs differ from R2 {direction}/{model}/{seed}")
                            continue
                        fields = ("score",) if model == "histgb" else ("score", "logit", "geometry_logit", "evidence_logit")
                        maxima = {}
                        for field in fields:
                            values = [abs(float(current[sample_id][field]) - float(old[sample_id][field]))
                                      for sample_id in current if current[sample_id][field] is not None]
                            maxima[field] = max(values) if values else 0.0
                        tolerance = {"score": 1e-7, "logit": 5e-6, "geometry_logit": 5e-6, "evidence_logit": 5e-6}
                        if any(maxima[field] > tolerance[field] for field in fields):
                            errors.append(f"base prediction drift {direction}/{model}/{seed}: {maxima}")
                        base_replay.append({"direction": direction, "model": model, "seed": seed, "max_abs": maxima})
        if horizon_sets and not all(horizon_sets[i + 1].issubset(horizon_sets[i]) for i in range(len(horizon_sets) - 1)):
            errors.append(f"horizon coverage is not nested: {direction}")
    if prediction_files != 210:
        errors.append(f"expected 210 prediction files, observed {prediction_files}")
    summary_path = out / "r4_summary.json"
    if not summary_path.is_file():
        errors.append("missing r4_summary.json")
    else:
        summary = read_json(summary_path)
        if summary.get("protocol_digest") != protocol["protocol_digest"]:
            errors.append("summary protocol digest mismatch")
        for direction, result in summary.get("directions", {}).items():
            if len(result.get("horizons", [])) != 5:
                errors.append(f"summary horizon count mismatch {direction}")
            for horizon in result.get("horizons", []):
                methods = horizon.get("sets", {}).get("common_tracks", {}).get("methods", {})
                if set(methods) != set(protocol["directions"][direction]["models"]):
                    errors.append(f"summary method set mismatch {direction}/{horizon.get('cutoff_frames')}")
    report = {
        "schema": "acie.vv-r4-validation.v1", "status": "passed" if not errors else "failed",
        "prediction_files": prediction_files, "expected_prediction_files": 210,
        "base_replay_checks": len(base_replay), "base_replay": base_replay, "errors": errors,
    }
    write_json(out / "r4_validation.json", report)
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
