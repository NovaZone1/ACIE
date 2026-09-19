#!/usr/bin/env python3
"""Re-run the original R2 tree CV with complete per-candidate artifacts."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from acie.data import Bundle, resolve_split
from acie.features import SourceScaler
from acie.io import (digest_file, digest_object, environment, read_json, write_json,
                     write_jsonl)
from acie.metrics import binary_metrics, choose_threshold
from acie.vv_provenance import (IDENTITY_VERSION, source_snapshot, training_content_digest,
                                validate_complete_run)
from acie.vv_trees import _build, candidate_grid, tree_features


DIRECTIONS = ("HUI360_to_SSUP-A", "SSUP-A_to_HUI360")
FOLDS = ("A", "B", "C")
KINDS = ("histgb", "random_forest")


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def prepare(data: Path, fold_root: Path, original: Path, out: Path) -> None:
    snapshot = source_snapshot([Path(__file__)])
    protocol = {
        "schema": "acie.vv-r2-tree-cv-archive-protocol.v1", "created_at": "2026-09-19",
        "purpose": "complete-artifact re-run of the original R2 source-only tree CV",
        "selection_uses_target": False, "feature_set": "all", "feature_dim": 389,
        "data": {"path": str(data.resolve()), "npz_sha256": digest_file(data),
                 "metadata_sha256": digest_file(data.with_suffix(".json"))},
        "folds": {
            direction: {fold: {"path": str((fold_root / direction / f"{fold}.json").resolve()),
                               "sha256": digest_file(fold_root / direction / f"{fold}.json")}
                        for fold in FOLDS}
            for direction in DIRECTIONS
        },
        "grids": {kind: candidate_grid(kind, "r2") for kind in KINDS},
        "seeds": {"histgb": [11], "random_forest": [11, 22]},
        "selection_rule": "original R2 rule: maximum mean source-CV AP; within 1e-6 choose earlier candidate",
        "original_selection_locks": {
            direction: {"path": str((original / direction / "TREE_SELECTION_LOCK.json").resolve()),
                        "sha256": digest_file(original / direction / "TREE_SELECTION_LOCK.json")}
            for direction in DIRECTIONS
        },
        "source_snapshot": snapshot,
    }
    protocol["protocol_digest"] = digest_object(protocol)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "PROTOCOL_LOCK.json", protocol)
    print(json.dumps({"status": "locked", "protocol_digest": protocol["protocol_digest"]}, indent=2))


def load_protocol(out: Path) -> dict[str, Any]:
    protocol = read_json(out / "PROTOCOL_LOCK.json")
    frozen = dict(protocol)
    expected = frozen.pop("protocol_digest", None)
    if protocol.get("schema") != "acie.vv-r2-tree-cv-archive-protocol.v1" or digest_object(frozen) != expected:
        raise ValueError("Invalid R2 tree CV archive protocol")
    return protocol


def fit_candidate(bundle: Bundle, split: dict[str, Any], out: Path, direction: str,
                  kind: str, fold: str, candidate: int, params: dict[str, Any], seed: int,
                  protocol: dict[str, Any]) -> dict[str, Any]:
    indices = resolve_split(bundle, split)
    snapshot = source_snapshot([Path(__file__)])
    identity = {
        "identity_version": IDENTITY_VERSION, "protocol_digest": protocol["protocol_digest"],
        "direction": direction, "kind": kind, "fold": fold, "candidate": candidate,
        "params": params, "seed": seed, "feature_set": "all",
        "split_hash": digest_object(split),
        "training_content_hash": training_content_digest(bundle, indices),
        "source_snapshot_hash": snapshot["sha256"],
    }
    run_id = digest_object(identity)
    if (out / "run.json").is_file():
        prior = read_json(out / "run.json")
        if prior.get("run_id") != run_id:
            raise ValueError(f"Archive run identity differs: {out}")
        if prior.get("status") == "complete":
            validate_complete_run(out, prior, tree=True)
            return prior
        raise FileExistsError(f"Incomplete archive run exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    scaler = SourceScaler.fit(bundle.a[indices["train"]], bundle.q[indices["train"]])
    features = tree_features(scaler, bundle.a, bundle.q, "all")
    write_json(out / "scaler.json", scaler.to_dict())
    write_json(out / "split.json", split)
    run = {
        "schema": "acie.vv-tree-run.v2", "identity_version": IDENTITY_VERSION,
        "run_id": run_id, "protocol_digest": protocol["protocol_digest"],
        "experiment": "r2_tree_cv_complete_artifact_rerun", "direction": direction,
        "kind": kind, "model_id": kind, "fold": fold, "candidate": candidate,
        "seed": seed, "feature_set": "all", "feature_dim": 389, "params": params,
        "training_content_hash": identity["training_content_hash"],
        "source_snapshot_hash": snapshot["sha256"], "source_snapshot": snapshot,
        "target_scoring_performed": False, "status": "started", "environment": environment(),
    }
    write_json(out / "run.json", run)
    started = time.perf_counter()
    model = _build(kind, params, seed)
    model.fit(features[indices["train"]], bundle.y[indices["train"]])
    score = model.predict_proba(features[indices["val"]])[:, 1]
    threshold = choose_threshold(bundle.y[indices["val"]], score)
    payload = {"schema": "acie.vv-tree-checkpoint.v1", "kind": kind, "seed": seed,
               "protocol": "r2_tree_cv_complete_artifact_rerun", "model": model,
               "scaler": scaler.to_dict(), "split": split, "threshold": threshold,
               "selected_params": params, "run_id": run_id,
               "feature_set": "all", "feature_dim": 389,
               "protocol_digest": protocol["protocol_digest"]}
    joblib.dump(payload, out / "tree.joblib")
    checkpoint_hash = digest_file(out / "tree.joblib")
    metrics = binary_metrics(bundle.y[indices["val"]], score, threshold)
    rows = [{**bundle.meta[int(i)], "label": int(bundle.y[int(i)]), "score": float(value),
             "model_id": kind, "seed": seed, "split_id": fold,
             "checkpoint_hash": checkpoint_hash, "source_threshold": threshold,
             "geometry_logit": None, "evidence_logit": None}
            for i, value in zip(indices["val"], score)]
    write_jsonl(out / "val_predictions.jsonl", rows)
    write_json(out / "candidate_results.json", [{"candidate": candidate, "params": params,
                                                   "source_val_AP": metrics["AP"]}])
    write_json(out / "metrics.json", {"source_val": metrics, "target": None})
    write_json(out / "timing.json", {"fit_seconds": time.perf_counter() - started,
                                      "serialized_bytes": (out / "tree.joblib").stat().st_size})
    run.update(status="complete", checkpoint_hash=checkpoint_hash, source_val=metrics)
    write_json(out / "run.json", run)
    return run


def run(data: Path, fold_root: Path, out: Path, protocol: dict[str, Any]) -> None:
    bundle = Bundle.load(data)
    completed = 0
    for direction in DIRECTIONS:
        for kind in KINDS:
            for fold in FOLDS:
                split = read_json(fold_root / direction / f"{fold}.json")
                for candidate, params in enumerate(protocol["grids"][kind]):
                    for seed in protocol["seeds"][kind]:
                        run_dir = out / "runs" / direction / kind / fold / f"candidate_{candidate}" / str(seed)
                        fit_candidate(bundle, split, run_dir, direction, kind, fold, candidate,
                                      params, seed, protocol)
                        completed += 1
                        print(json.dumps({"slot": completed, "direction": direction, "kind": kind,
                                          "fold": fold, "candidate": candidate, "seed": seed}), flush=True)
    if completed != 108:
        raise AssertionError(f"Expected 108 archive runs, observed {completed}")


def validate(original: Path, out: Path, protocol: dict[str, Any]) -> None:
    errors, observed, comparisons = [], 0, {}
    for direction in DIRECTIONS:
        selected, tables = {}, {}
        for kind in KINDS:
            candidate_rows = []
            for candidate, params in enumerate(protocol["grids"][kind]):
                values = []
                for fold in FOLDS:
                    for seed in protocol["seeds"][kind]:
                        run_dir = out / "runs" / direction / kind / fold / f"candidate_{candidate}" / str(seed)
                        try:
                            run_record = read_json(run_dir / "run.json")
                            validate_complete_run(run_dir, run_record, tree=True)
                            values.append(read_json(run_dir / "metrics.json")["source_val"]["AP"])
                            observed += 1
                        except Exception as exc:
                            errors.append(f"{direction}/{kind}/{fold}/{candidate}/{seed}: {exc}")
                candidate_rows.append({"order": candidate, "params": params,
                                       "fold_seed_AP": values,
                                       "mean_source_CV_AP": float(np.mean(values)) if values else None})
            valid = [row for row in candidate_rows if row["mean_source_CV_AP"] is not None]
            best = max(row["mean_source_CV_AP"] for row in valid)
            pick = next(row for row in valid if best - row["mean_source_CV_AP"] <= 1e-6)
            tables[kind], selected[kind] = candidate_rows, pick
        original_lock = read_json(original / direction / "TREE_SELECTION_LOCK.json")
        same = all(selected[kind]["order"] == original_lock["models"][kind]["order"] and
                   abs(selected[kind]["mean_source_CV_AP"] -
                       original_lock["models"][kind]["mean_source_CV_AP"]) <= 1e-12
                   for kind in KINDS)
        if not same:
            errors.append(f"recomputed selection differs from original lock: {direction}")
        comparisons[direction] = {"selection_matches_original": same,
                                  "recomputed_models": selected, "candidate_tables": tables,
                                  "original_lock_sha256": digest_file(original / direction / "TREE_SELECTION_LOCK.json")}
    report = {"schema": "acie.vv-r2-tree-cv-archive-validation.v1",
              "status": "passed" if not errors and observed == 108 else "failed",
              "expected_runs": 108, "observed_runs": observed,
              "selection_comparison": comparisons, "errors": errors}
    write_json(out / "validation.json", report)
    print(json.dumps({"status": report["status"], "observed_runs": observed,
                      "errors": errors}, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root_default = Path(__file__).resolve().parents[1]
    parser.add_argument("phase", choices=("prepare", "run", "validate"))
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument("--data", type=Path, default=Path("data/official_cross_domain.npz"))
    parser.add_argument("--fold-root", type=Path, default=Path("protocols/vv_followup_v1/r2_folds"))
    parser.add_argument("--original", type=Path, default=Path("outputs/vv_followup_v1/r2_tree_cv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/vv_followup_v2/r2_tree_cv_archive"))
    args = parser.parse_args()
    root = args.root.resolve()
    data, fold_root, original, out = map(lambda value: resolve(root, value),
                                         (args.data, args.fold_root, args.original, args.out))
    if args.phase == "prepare":
        prepare(data, fold_root, original, out)
        return
    protocol = load_protocol(out)
    if args.phase == "run": run(data, fold_root, out, protocol)
    else: validate(original, out, protocol)


if __name__ == "__main__":
    main()
