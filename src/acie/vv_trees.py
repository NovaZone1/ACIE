"""Source-only tree references with target scoring in a separate entry point."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score

from .data import Bundle, resolve_split
from .features import SourceScaler
from .io import digest_file, digest_object, environment, write_json, write_jsonl
from .metrics import binary_metrics, choose_threshold


def tree_features(scaler: SourceScaler, a: np.ndarray, q: np.ndarray) -> np.ndarray:
    aa, qq = scaler.transform(a, q)
    return np.concatenate([aa, qq[:, -1], qq.mean(1), qq.std(1)], axis=1).astype(np.float32)


def candidate_grid(kind: str, protocol: str = "r1") -> list[dict[str, Any]]:
    if protocol == "r1":
        if kind == "histgb":
            return [{"max_leaf_nodes": value, "l2_regularization": 1.0} for value in (7, 15, 31)]
        if kind == "random_forest":
            return [{"max_depth": value, "min_samples_leaf": 1} for value in (7, 15, 31)]
    elif protocol == "r2":
        if kind == "histgb":
            return [{"max_leaf_nodes": leaves, "l2_regularization": l2}
                    for leaves in (7, 15, 31) for l2 in (0.1, 1.0)]
        if kind == "random_forest":
            return [{"max_depth": depth, "min_samples_leaf": leaf}
                    for depth in (7, 15, 31) for leaf in (1, 5)]
    raise ValueError(f"Unknown tree/protocol: {kind}/{protocol}")


def _build(kind: str, params: dict[str, Any], seed: int):
    if kind == "histgb":
        return HistGradientBoostingClassifier(max_iter=150, early_stopping=False,
                                              random_state=seed, **params)
    if kind == "random_forest":
        return RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=2,
                                      random_state=seed, **params)
    raise ValueError(f"Unknown tree baseline: {kind}")


def train_source_tree(bundle: Bundle, split: dict[str, Any], out: str | Path,
                      kind: str = "histgb", seed: int = 11, protocol: str = "r1",
                      context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fit/select on source train/val. No target prediction is produced."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "tree.joblib").exists():
        prior = json.loads((out / "run.json").read_text())
        if prior.get("status") == "complete":
            return prior
        raise FileExistsError(f"Incomplete tree run exists at {out}")
    context = context or {}
    indices = resolve_split(bundle, split)
    train_ix, val_ix = indices["train"], indices["val"]
    scaler = SourceScaler.fit(bundle.a[train_ix], bundle.q[train_ix])
    write_json(out / "scaler.json", scaler.to_dict())
    x = tree_features(scaler, bundle.a, bundle.q)
    if x.shape[1] != 389:
        raise AssertionError(f"Expected 389 tree features, got {x.shape[1]}")
    grid = candidate_grid(kind, protocol)
    run_id = digest_object({"kind": kind, "seed": seed, "protocol": protocol,
                            "split": split, "grid": grid, "provenance": bundle.provenance})
    run = {
        "schema": "acie.vv-tree-run.v1", "run_id": run_id,
        "experiment": context.get("experiment", protocol), "source": split["source"],
        "target": split["target"], "model_id": kind, "seed": seed,
        "stage": "source_training_and_selection", "split_hash": digest_object(split),
        "feature_dim": 389, "candidate_count": len(grid), "grid": grid,
        "target_scoring_performed": False, "status": "started", "environment": environment(),
    }
    write_json(out / "run.json", run)
    write_json(out / "split.json", split)
    started = time.perf_counter()
    candidates = []
    for order, params in enumerate(grid):
        model = _build(kind, params, seed)
        model.fit(x[train_ix], bundle.y[train_ix])
        score = model.predict_proba(x[val_ix])[:, 1]
        ap = float(average_precision_score(bundle.y[val_ix], score))
        candidates.append({"order": order, "params": params, "source_val_AP": ap,
                           "model": model, "score": score})
    # Stable order resolves exact ties in favor of the earlier registered candidate.
    selected = max(candidates, key=lambda row: (row["source_val_AP"], -row["order"]))
    threshold = choose_threshold(bundle.y[val_ix], selected["score"])
    payload = {
        "schema": "acie.vv-tree-checkpoint.v1", "kind": kind, "seed": seed,
        "protocol": protocol, "model": selected["model"], "scaler": scaler.to_dict(),
        "split": split, "threshold": threshold, "selected_params": selected["params"],
        "run_id": run_id,
    }
    joblib.dump(payload, out / "tree.joblib")
    checkpoint_hash = digest_file(out / "tree.joblib")
    rows = [{**bundle.meta[i], "label": int(bundle.y[i]), "score": float(score),
             "model_id": kind, "seed": seed, "split_id": "fixed",
             "checkpoint_hash": checkpoint_hash, "source_threshold": threshold,
             "geometry_logit": None, "evidence_logit": None}
            for i, score in zip(val_ix, selected["score"])]
    write_jsonl(out / "val_predictions.jsonl", rows)
    write_json(out / "candidate_results.json", [
        {key: value for key, value in row.items() if key not in ("model", "score")}
        for row in candidates
    ])
    metrics = binary_metrics(bundle.y[val_ix], selected["score"], threshold)
    write_json(out / "metrics.json", {"source_val": metrics, "target": None})
    write_json(out / "timing.json", {"fit_selection_seconds": time.perf_counter() - started,
                                      "serialized_bytes": (out / "tree.joblib").stat().st_size})
    run.update(status="complete", selected_params=selected["params"], source_val=metrics,
               checkpoint_hash=checkpoint_hash, target_scoring_performed=False)
    write_json(out / "run.json", run)
    return run


def score_locked_tree(bundle: Bundle, checkpoint: str | Path, out: str | Path) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    payload = joblib.load(checkpoint)
    if payload.get("schema") != "acie.vv-tree-checkpoint.v1":
        raise ValueError("Unsupported VV tree checkpoint")
    indices = resolve_split(bundle, payload["split"])["test"]
    scaler = SourceScaler.from_dict(payload["scaler"])
    x = tree_features(scaler, bundle.a, bundle.q)
    started = time.perf_counter()
    score = payload["model"].predict_proba(x[indices])[:, 1]
    seconds = time.perf_counter() - started
    checkpoint_hash = digest_file(checkpoint)
    rows = [{**bundle.meta[i], "label": int(bundle.y[i]), "score": float(value),
             "model_id": payload["kind"], "seed": int(payload["seed"]), "split_id": "fixed",
             "checkpoint_hash": checkpoint_hash, "source_threshold": float(payload["threshold"]),
             "geometry_logit": None, "evidence_logit": None}
            for i, value in zip(indices, score)]
    out = Path(out)
    write_jsonl(out, rows)
    result = {"schema": "acie.vv-tree-locked-score.v1", "kind": payload["kind"],
              "seed": int(payload["seed"]), "checkpoint_hash": checkpoint_hash,
              "weights_updated": False, "scaler_updated": False, "threshold_updated": False,
              "metrics": binary_metrics(bundle.y[indices], score, payload["threshold"]),
              "inference_seconds": seconds}
    write_json(out.with_suffix(".metrics.json"), result)
    return result
