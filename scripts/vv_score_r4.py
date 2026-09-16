#!/usr/bin/env python3
"""Run inference-only R4 scoring from the frozen protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from acie.data import Bundle
import numpy as np

from acie.io import digest_file, digest_object, read_json, read_jsonl, write_json, write_jsonl
from acie.metrics import binary_metrics
from acie.vv_horizon import score_horizon_neural, score_horizon_tree


def verify_protocol(protocol: dict) -> None:
    frozen = dict(protocol)
    expected = frozen.pop("protocol_digest")
    if protocol.get("schema") != "acie.vv-r4-protocol-lock.v1" or digest_object(frozen) != expected:
        raise ValueError("Invalid R4 protocol lock")


def prediction_path(base: Path, direction: str, model: str, seed: int, cutoff: int) -> Path:
    return base / direction / model / "fixed" / str(seed) / f"cutoff_frames_{cutoff:03d}.jsonl"


def reuse_r2_base(bundle: Bundle, checkpoint: Path, checkpoint_sha256: str,
                  bundle_sha256: str, out: Path, model: str) -> None:
    """Reuse the locked R2 prediction at the registered 16-frame base point."""
    original = checkpoint.parent / "test_predictions.jsonl"
    rows = read_jsonl(original)
    by_id = {row["sample_id"]: row for row in rows}
    expected = [meta["sample_id"] for meta in bundle.meta]
    if len(by_id) != len(rows) or set(by_id) != set(expected):
        raise ValueError(f"R2 base prediction coverage mismatch: {original}")
    ordered = [by_id[sample_id] for sample_id in expected]
    if any(int(row["label"]) != int(label) for row, label in zip(ordered, bundle.y)):
        raise ValueError(f"R2 base prediction labels mismatch: {original}")
    if any(row["checkpoint_hash"] != checkpoint_sha256 for row in ordered):
        raise ValueError(f"R2 base prediction checkpoint mismatch: {original}")
    write_jsonl(out, ordered)
    scores = np.asarray([row["score"] for row in ordered], dtype=float)
    threshold = float(next(iter({row["source_threshold"] for row in ordered})))
    write_json(out.with_suffix(".metrics.json"), {
        "schema": "acie.vv-r4-score.v1", "kind": "tree" if model == "histgb" else "neural",
        "model_id": model, "seed": int(ordered[0]["seed"]),
        "checkpoint_hash": checkpoint_sha256, "bundle_sha256": bundle_sha256,
        "weights_updated": False, "scaler_updated": False, "threshold_updated": False,
        "metrics": binary_metrics(bundle.y, scores, threshold), "inference_seconds": 0.0,
        "reused_from": str(original),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--protocol", type=Path, default=Path("protocols/vv_followup_v1/R4_PROTOCOL_LOCK.json"))
    parser.add_argument("--out", type=Path, default=Path("outputs/vv_followup_v1/r4"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    root = args.root.resolve()
    protocol_path = args.protocol if args.protocol.is_absolute() else root / args.protocol
    out = args.out if args.out.is_absolute() else root / args.out
    protocol = read_json(protocol_path)
    verify_protocol(protocol)
    completed = 0
    for direction, spec in protocol["directions"].items():
        for bundle_spec in spec["bundles"]:
            bundle_path = root / bundle_spec["path"]
            if digest_file(bundle_path) != bundle_spec["sha256"]:
                raise ValueError(f"R4 bundle changed: {bundle_path}")
            bundle = Bundle.load(bundle_path)
            expected_ids = [meta["sample_id"] for meta in bundle.meta]
            for model in spec["models"]:
                for seed_text, checkpoint_spec in spec["checkpoints"][model].items():
                    seed = int(seed_text)
                    checkpoint = root / checkpoint_spec["path"]
                    if digest_file(checkpoint) != checkpoint_spec["sha256"]:
                        raise ValueError(f"R2 checkpoint changed: {checkpoint}")
                    path = prediction_path(out, direction, model, seed, bundle_spec["cutoff_frames"])
                    if bundle_spec["cutoff_frames"] == 16:
                        reuse_r2_base(bundle, checkpoint, checkpoint_spec["sha256"],
                                      bundle_spec["sha256"], path, model)
                    elif path.exists():
                        rows = read_jsonl(path)
                        if [row["sample_id"] for row in rows] != expected_ids:
                            raise ValueError(f"Existing R4 coverage differs: {path}")
                        if any(row["checkpoint_hash"] != checkpoint_spec["sha256"] for row in rows):
                            raise ValueError(f"Existing R4 checkpoint hash differs: {path}")
                    elif model == "histgb":
                        score_horizon_tree(bundle, checkpoint, path, bundle_spec["sha256"])
                    else:
                        score_horizon_neural(bundle, checkpoint, path, args.device, bundle_spec["sha256"])
                    completed += 1
                    print(json.dumps({"status": "complete", "slot": completed, "direction": direction,
                                      "model": model, "seed": seed,
                                      "cutoff": bundle_spec["cutoff_frames"]}), flush=True)
    if completed != 210:
        raise AssertionError(f"Expected 210 R4 predictions, observed {completed}")


if __name__ == "__main__":
    main()
