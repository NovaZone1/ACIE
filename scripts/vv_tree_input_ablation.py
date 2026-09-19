#!/usr/bin/env python3
"""Locked HistGB geometry/pose/all input ablation for the method-paper follow-up."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import average_precision_score

from acie.data import Bundle, resolve_split
from acie.features import SourceScaler
from acie.io import (digest_file, digest_object, environment, read_json, read_jsonl,
                     write_json, write_jsonl)
from acie.metrics import binary_metrics, choose_threshold
from acie.vv_horizon import paired_date_seed_bootstrap
from acie.vv_provenance import (IDENTITY_VERSION, source_snapshot, training_content_digest,
                                validate_complete_run, validate_locked_checkpoint)
from acie.vv_trees import _build, candidate_grid, score_locked_tree, tree_features


DIRECTIONS = {
    "HUI360_to_SSUP-A": "outputs/frozen_cross_domain/HUI360_to_SSUP-A/split.json",
    "SSUP-A_to_HUI360": "outputs/frozen_cross_domain/SSUP-A_to_HUI360/split.json",
}
FEATURE_SETS = ("geometry", "pose", "all")
FOLDS = ("A", "B", "C")


def resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = read_json(path)
    frozen = dict(protocol)
    expected = frozen.pop("protocol_digest", None)
    if protocol.get("schema") != "acie.vv-tree-input-ablation-protocol.v1" or digest_object(frozen) != expected:
        raise ValueError("Invalid tree input-ablation protocol")
    return protocol


def prepare(root: Path, data: Path, fold_root: Path, out: Path, kind: str) -> None:
    snapshot = source_snapshot([Path(__file__)])
    cv_seeds = [11] if kind == "histgb" else [11, 22]
    final_seeds = [11] if kind == "histgb" else [11, 22, 33, 44, 55]
    protocol = {
        "schema": "acie.vv-tree-input-ablation-protocol.v1",
        "created_at": "2026-09-19",
        "target_results_previously_observed": True,
        "selection_uses_target": False,
        "model": ("HistGradientBoostingClassifier" if kind == "histgb" else
                  "RandomForestClassifier"),
        "kind": kind, "cv_seeds": cv_seeds, "final_seeds": final_seeds,
        "feature_sets": {
            "geometry": {"definition": "standardized 32-D geometry a", "dimension": 32},
            "pose": {"definition": "standardized q last/mean/std", "dimension": 357},
            "all": {"definition": "geometry plus pose last/mean/std", "dimension": 389},
        },
        "candidate_grid": candidate_grid(kind, "r2"),
        "folds": list(FOLDS),
        "selection_rule": "maximum mean source-CV AP; ties within 1e-6 choose earlier registered candidate",
        "data": {"path": str(data.resolve()), "npz_sha256": digest_file(data),
                 "metadata_sha256": digest_file(data.with_suffix(".json"))},
        "fold_files": {
            direction: {fold: {"path": str((fold_root / direction / f"{fold}.json").resolve()),
                               "sha256": digest_file(fold_root / direction / f"{fold}.json")}
                        for fold in FOLDS}
            for direction in DIRECTIONS
        },
        "fixed_split_files": {
            direction: {"path": str((root / relative).resolve()),
                        "sha256": digest_file(root / relative)}
            for direction, relative in DIRECTIONS.items()
        },
        "source_snapshot": snapshot,
        "output_root": str(out.resolve()),
    }
    protocol["protocol_digest"] = digest_object(protocol)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "PROTOCOL_LOCK.json", protocol)
    print(json.dumps({"status": "locked", "protocol_digest": protocol["protocol_digest"]}, indent=2))


def fit_one(bundle: Bundle, split: dict[str, Any], out: Path, feature_set: str,
            params: dict[str, Any], protocol_digest: str, role: str,
            direction: str, fold: str | None, candidate: int | None,
            kind: str, seed: int) -> dict[str, Any]:
    indices = resolve_split(bundle, split)
    snapshot = source_snapshot([Path(__file__)])
    identity = {
        "identity_version": IDENTITY_VERSION, "protocol_digest": protocol_digest,
        "role": role, "direction": direction, "fold": fold, "candidate": candidate,
        "kind": kind, "feature_set": feature_set, "seed": seed,
        "params": params, "split_hash": digest_object(split),
        "training_content_hash": training_content_digest(bundle, indices),
        "source_snapshot_hash": snapshot["sha256"],
    }
    run_id = digest_object(identity)
    if (out / "run.json").is_file():
        prior = read_json(out / "run.json")
        if prior.get("run_id") != run_id:
            raise ValueError(f"Existing tree ablation run has different identity: {out}")
        if prior.get("status") == "complete":
            validate_complete_run(out, prior, tree=True)
            return prior
        raise FileExistsError(f"Incomplete tree ablation run exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    scaler = SourceScaler.fit(bundle.a[indices["train"]], bundle.q[indices["train"]])
    features = tree_features(scaler, bundle.a, bundle.q, feature_set)
    write_json(out / "scaler.json", scaler.to_dict())
    write_json(out / "split.json", split)
    run = {
        "schema": "acie.vv-tree-run.v2", "identity_version": IDENTITY_VERSION,
        "run_id": run_id, "protocol_digest": protocol_digest, "role": role,
        "direction": direction, "fold": fold, "candidate": candidate,
        "model_id": kind, "feature_set": feature_set,
        "feature_dim": int(features.shape[1]), "seed": seed, "selected_params": params,
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
    payload = {
        "schema": "acie.vv-tree-checkpoint.v1", "kind": kind, "seed": seed,
        "protocol": "method_paper_tree_input_ablation_v1", "model": model,
        "scaler": scaler.to_dict(), "split": split, "threshold": threshold,
        "selected_params": params, "run_id": run_id,
        "feature_set": feature_set, "feature_dim": int(features.shape[1]),
        "protocol_digest": protocol_digest,
    }
    joblib.dump(payload, out / "tree.joblib")
    checkpoint_hash = digest_file(out / "tree.joblib")
    rows = [{**bundle.meta[int(i)], "label": int(bundle.y[int(i)]), "score": float(value),
             "model_id": kind, "feature_set": feature_set, "seed": seed,
             "split_id": split.get("split_id", fold or "fixed"),
             "checkpoint_hash": checkpoint_hash, "source_threshold": threshold,
             "geometry_logit": None, "evidence_logit": None}
            for i, value in zip(indices["val"], score)]
    metrics = binary_metrics(bundle.y[indices["val"]], score, threshold)
    write_jsonl(out / "val_predictions.jsonl", rows)
    write_json(out / "candidate_results.json", [{"candidate": candidate, "params": params,
                                                   "source_val_AP": metrics["AP"]}])
    write_json(out / "metrics.json", {"source_val": metrics, "target": None})
    write_json(out / "timing.json", {"fit_seconds": time.perf_counter() - started,
                                      "serialized_bytes": (out / "tree.joblib").stat().st_size})
    run.update(status="complete", checkpoint_hash=checkpoint_hash, source_val=metrics,
               target_scoring_performed=False)
    write_json(out / "run.json", run)
    return run


def run_cv(root: Path, data: Path, fold_root: Path, out: Path, protocol: dict[str, Any]) -> None:
    bundle = Bundle.load(data)
    kind = protocol["kind"]
    completed = 0
    for direction in DIRECTIONS:
        for feature_set in FEATURE_SETS:
            for fold in FOLDS:
                split = read_json(fold_root / direction / f"{fold}.json")
                for order, params in enumerate(protocol["candidate_grid"]):
                    for seed in protocol["cv_seeds"]:
                        run_dir = out / "cv" / direction / feature_set / fold / f"candidate_{order}" / str(seed)
                        fit_one(bundle, split, run_dir, feature_set, params,
                                protocol["protocol_digest"], "source_cv", direction, fold, order,
                                kind, seed)
                        completed += 1
                        print(json.dumps({"slot": completed, "direction": direction,
                                          "feature_set": feature_set, "fold": fold,
                                          "candidate": order, "seed": seed}), flush=True)
    expected = 2 * len(FEATURE_SETS) * len(FOLDS) * len(protocol["candidate_grid"]) * len(protocol["cv_seeds"])
    if completed != expected:
        raise AssertionError(f"Expected {expected} CV runs, observed {completed}")


def select(out: Path, protocol: dict[str, Any]) -> None:
    for direction in DIRECTIONS:
        models, tables = {}, {}
        for feature_set in FEATURE_SETS:
            rows = []
            for order, params in enumerate(protocol["candidate_grid"]):
                fold_ap = [read_json(out / "cv" / direction / feature_set / fold /
                                     f"candidate_{order}" / str(seed) / "metrics.json")["source_val"]["AP"]
                           for fold in FOLDS for seed in protocol["cv_seeds"]]
                rows.append({"order": order, "params": params, "fold_AP": fold_ap,
                             "mean_source_CV_AP": float(np.mean(fold_ap))})
            best = max(row["mean_source_CV_AP"] for row in rows)
            chosen = next(row for row in rows if best - row["mean_source_CV_AP"] <= 1e-6)
            tables[feature_set], models[feature_set] = rows, chosen
        lock = {
            "schema": "acie.vv-tree-input-ablation-selection.v1", "status": "locked",
            "direction": direction, "protocol_digest": protocol["protocol_digest"],
            "selection_uses_target": False, "models": models, "candidate_tables": tables,
        }
        lock["lock_digest"] = digest_object(lock)
        write_json(out / "selection" / direction / "SELECTION_LOCK.json", lock)
        print(json.dumps({"direction": direction, "selected": models}, indent=2))


def fit_final(root: Path, data: Path, out: Path, protocol: dict[str, Any]) -> None:
    bundle = Bundle.load(data)
    kind = protocol["kind"]
    for direction, split_relative in DIRECTIONS.items():
        selection = read_json(out / "selection" / direction / "SELECTION_LOCK.json")
        split = read_json(root / split_relative)
        for feature_set in FEATURE_SETS:
            params = selection["models"][feature_set]["params"]
            for seed in protocol["final_seeds"]:
                run_dir = out / "final" / direction / feature_set / "fixed" / str(seed)
                fit_one(bundle, split, run_dir, feature_set, params,
                        protocol["protocol_digest"], "source_final", direction, None,
                        selection["models"][feature_set]["order"], kind, seed)


def validate_final(out: Path, protocol: dict[str, Any]) -> None:
    errors, checkpoints = [], []
    for direction in DIRECTIONS:
        selection = read_json(out / "selection" / direction / "SELECTION_LOCK.json")
        for feature_set in FEATURE_SETS:
            for seed in protocol["final_seeds"]:
                run_dir = out / "final" / direction / feature_set / "fixed" / str(seed)
                try:
                    run = read_json(run_dir / "run.json")
                    validate_complete_run(run_dir, run, tree=True)
                    if run.get("protocol_digest") != protocol["protocol_digest"]:
                        raise ValueError("protocol digest mismatch")
                    if run.get("selected_params") != selection["models"][feature_set]["params"]:
                        raise ValueError("selected parameters mismatch")
                    checkpoints.append({"direction": direction, "model_id": protocol["kind"],
                                        "feature_set": feature_set, "seed": seed,
                                        "path": str((run_dir / "tree.joblib").resolve()),
                                        "sha256": digest_file(run_dir / "tree.joblib")})
                except Exception as exc:
                    errors.append(f"{direction}/{feature_set}/{seed}: {type(exc).__name__}: {exc}")
    expected = 2 * len(FEATURE_SETS) * len(protocol["final_seeds"])
    report = {"schema": "acie.vv-tree-input-ablation-validation.v1",
              "status": "passed" if not errors and len(checkpoints) == expected else "failed",
              "expected_slots": expected, "observed_slots": len(checkpoints), "errors": errors}
    write_json(out / "source_validation.json", report)
    if report["status"] != "passed":
        raise SystemExit(json.dumps(report, indent=2))
    lock = {"schema": "acie.vv-tree-input-ablation-target-lock.v1", "status": "locked",
            "protocol_digest": protocol["protocol_digest"], "checkpoint_count": len(checkpoints),
            "checkpoints": checkpoints,
            "selection": "feature-specific source CV only; target results were not used for selection"}
    lock["lock_digest"] = digest_object(lock)
    write_json(out / "TARGET_SCORING_LOCK.json", lock)
    print(json.dumps(report, indent=2))


def score(data: Path, out: Path, protocol: dict[str, Any]) -> None:
    bundle = Bundle.load(data)
    lock = out / "TARGET_SCORING_LOCK.json"
    completed = 0
    for direction in DIRECTIONS:
        for feature_set in FEATURE_SETS:
            for seed in protocol["final_seeds"]:
                run_dir = out / "final" / direction / feature_set / "fixed" / str(seed)
                checkpoint = run_dir / "tree.joblib"
                validate_locked_checkpoint(lock, checkpoint, direction=direction, model_id=protocol["kind"],
                                           feature_set=feature_set, seed=seed)
                result = score_locked_tree(bundle, checkpoint, run_dir / "test_predictions.jsonl")
                completed += 1
                print(json.dumps({"slot": completed, "direction": direction,
                                  "feature_set": feature_set, "seed": seed,
                                  "AP": result["metrics"]["AP"]}), flush=True)


def summarize(out: Path, repeats: int, protocol: dict[str, Any]) -> None:
    summary, metric_rows, difference_rows = {"schema": "acie.vv-tree-input-ablation-summary.v1",
                                             "bootstrap_repeats": repeats, "directions": {}}, [], []
    for direction_index, direction in enumerate(DIRECTIONS):
        matrices, labels, groups = {}, None, None
        direction_result = {"methods": {}, "comparisons": []}
        for feature_set in FEATURE_SETS:
            per_seed, values_by_seed = [], []
            reference_ids = None
            for seed in protocol["final_seeds"]:
                run_dir = out / "final" / direction / feature_set / "fixed" / str(seed)
                rows = read_jsonl(run_dir / "test_predictions.jsonl")
                ids = [row["sample_id"] for row in rows]
                if reference_ids is None:
                    reference_ids = ids
                    labels = np.asarray([row["label"] for row in rows], dtype=int)
                    groups = np.asarray([str(row.get("group_id", row["recording"])) for row in rows])
                elif ids != reference_ids:
                    raise ValueError(f"Target coverage differs: {direction}/{feature_set}/{seed}")
                values = np.asarray([row["score"] for row in rows], dtype=float)
                threshold = float(next(iter({row["source_threshold"] for row in rows})))
                metrics = binary_metrics(labels, values, threshold)
                per_seed.append({"seed": seed, **metrics})
                metric_rows.append({"direction": direction, "feature_set": feature_set,
                                    "seed": seed, **metrics})
                values_by_seed.append(values)
            matrices[feature_set] = np.asarray(values_by_seed)
            direction_result["methods"][feature_set] = {
                "seeds": list(protocol["final_seeds"]), "per_seed": per_seed,
                "AP_mean": float(np.mean([row["AP"] for row in per_seed])),
                "AP_sample_SD": (float(np.std([row["AP"] for row in per_seed], ddof=1))
                                 if len(per_seed) > 1 else None),
                "AUROC_mean": float(np.mean([row["AUROC"] for row in per_seed])),
            }
        for comparison_index, (first, second) in enumerate((("all", "geometry"),
                                                            ("pose", "geometry"),
                                                            ("all", "pose"))):
            difference = float(average_precision_score(labels, matrices[first][0]) -
                               average_precision_score(labels, matrices[second][0]))
            bootstrap_seed = 20260919 + direction_index * 10 + comparison_index
            boot = paired_date_seed_bootstrap(labels, groups, matrices[first], matrices[second],
                                              repeats, bootstrap_seed)
            item = {"comparison": f"{first}-{second}", "AP_difference": difference,
                    "bootstrap_seed": bootstrap_seed, "bootstrap": boot}
            direction_result["comparisons"].append(item)
            interval = boot["joint_date_and_seed"]["interval_95"]
            difference_rows.append({"direction": direction, "comparison": item["comparison"],
                                    "AP_difference": difference, "interval_95_low": interval[0],
                                    "interval_95_high": interval[1],
                                    "bootstrap_seed": bootstrap_seed})
        summary["directions"][direction] = direction_result
    write_json(out / "summary.json", summary)
    for path, rows in ((out / "metrics.csv", metric_rows), (out / "comparisons.csv", difference_rows)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"status": "complete", "metrics": len(metric_rows),
                      "comparisons": len(difference_rows)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root_default = Path(__file__).resolve().parents[1]
    parser.add_argument("phase", choices=("prepare", "cv", "select", "final", "validate", "score", "summarize"))
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument("--data", type=Path, default=Path("data/official_cross_domain.npz"))
    parser.add_argument("--fold-root", type=Path, default=Path("protocols/vv_followup_v1/r2_folds"))
    parser.add_argument("--out", type=Path, default=Path("outputs/vv_followup_v2/tree_input_ablation"))
    parser.add_argument("--kind", choices=("histgb", "random_forest"), default="histgb")
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    root = args.root.resolve()
    data, fold_root, out = (resolve(root, args.data), resolve(root, args.fold_root), resolve(root, args.out))
    if args.phase == "prepare":
        prepare(root, data, fold_root, out, args.kind)
        return
    protocol = load_protocol(out / "PROTOCOL_LOCK.json")
    if args.phase == "cv": run_cv(root, data, fold_root, out, protocol)
    elif args.phase == "select": select(out, protocol)
    elif args.phase == "final": fit_final(root, data, out, protocol)
    elif args.phase == "validate": validate_final(out, protocol)
    elif args.phase == "score": score(data, out, protocol)
    else: summarize(out, args.bootstrap, protocol)


if __name__ == "__main__":
    main()
