"""Inference and operating-point helpers for the locked VV horizon study."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from .data import Bundle
from .features import SourceScaler
from .io import digest_file, environment, write_json, write_jsonl
from .metrics import binary_metrics
from .vv_engine import _predict, _sigmoid
from .vv_models import build_model
from .vv_trees import tree_features


def fpr_budget_threshold(y: np.ndarray, scores: np.ndarray, budget: float) -> dict[str, Any]:
    """Choose the highest-recall source threshold under an FPR budget.

    The all-negative threshold is included. Ties in recall use the higher
    threshold, matching the registered R4 rule.
    """
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(y) == 0 or len(y) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid source validation predictions")
    if not 0 <= budget <= 1 or len(np.unique(y)) != 2:
        raise ValueError("FPR budgets require both classes and a budget in [0, 1]")
    candidates = np.concatenate(([np.nextafter(scores.max(), np.inf)], np.unique(scores)))
    feasible = []
    for threshold in candidates:
        metrics = binary_metrics(y, scores, float(threshold))
        if float(metrics["false_positive_rate"]) <= budget:
            feasible.append((float(metrics["recall"]), float(threshold), metrics))
    _, threshold, metrics = max(feasible, key=lambda item: (item[0], item[1]))
    return {"budget": float(budget), "threshold": threshold, "source_val": metrics}


def paired_date_seed_bootstrap(y: np.ndarray, groups: np.ndarray, a: np.ndarray,
                               b: np.ndarray, repeats: int, seed: int) -> dict[str, Any]:
    """Paired date/seed bootstrap for a mean per-seed AP difference."""
    y, groups = np.asarray(y, dtype=int), np.asarray(groups)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != len(y):
        raise ValueError("Prediction matrices do not align")
    unique = np.unique(groups)
    group_codes = np.searchsorted(unique, groups)
    group_positives = np.bincount(group_codes, weights=y, minlength=len(unique))
    group_negatives = np.bincount(group_codes, weights=1 - y, minlength=len(unique))
    rng = np.random.default_rng(seed)
    cache: dict[tuple[str, int, tuple[int, ...]], float] = {}
    joint, date_only = [], []
    invalid = 0

    def prepare(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        order = np.argsort(scores, kind="mergesort")[::-1]
        sorted_scores = scores[order]
        ends = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
        return y[order], group_codes[order], ends, sorted_scores

    prepared = {
        (which, seed_index): prepare(scores)
        for which, matrix in (("a", a), ("b", b))
        for seed_index, scores in enumerate(matrix)
    }

    def weighted_ap(item: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                    counts: tuple[int, ...]) -> float:
        sorted_y, sorted_groups, ends, _ = item
        weights = np.asarray(counts, dtype=float)[sorted_groups]
        true = np.cumsum(weights * sorted_y)[ends]
        false = np.cumsum(weights * (1 - sorted_y))[ends]
        total_positive = true[-1]
        precision = np.divide(true, true + false, out=np.ones_like(true), where=(true + false) != 0)
        increments = np.diff(np.r_[0.0, true]) / total_positive
        return float(np.sum(increments * precision))

    def ap(which: str, seed_index: int, counts: tuple[int, ...]) -> float:
        key = (which, seed_index, counts)
        if key not in cache:
            cache[key] = weighted_ap(prepared[(which, seed_index)], counts)
        return cache[key]

    for _ in range(repeats):
        drawn = rng.integers(0, len(unique), len(unique))
        counts = tuple(int(value) for value in np.bincount(drawn, minlength=len(unique)))
        count_array = np.asarray(counts)
        if np.dot(count_array, group_positives) == 0 or np.dot(count_array, group_negatives) == 0:
            invalid += 1
            continue
        differences = np.asarray([ap("a", i, counts) - ap("b", i, counts)
                                  for i in range(a.shape[0])])
        date_only.append(float(differences.mean()))
        sampled_seeds = rng.integers(0, a.shape[0], a.shape[0])
        joint.append(float(differences[sampled_seeds].mean()))

    def describe(values: list[float]) -> dict[str, Any]:
        return {
            "valid": len(values),
            "interval_95": (np.quantile(values, [0.025, 0.975]).tolist() if values else None),
        }

    return {
        "groups": len(unique), "replicates_requested": repeats,
        "invalid_no_both_classes": invalid,
        "joint_date_and_seed": describe(joint),
        "date_only_fixed_seeds": describe(date_only),
        "warning": "fewer than 90% valid replicates" if len(joint) < 0.9 * repeats else None,
    }


def _all_indices(bundle: Bundle) -> np.ndarray:
    return np.arange(len(bundle.y), dtype=int)


def score_horizon_neural(bundle: Bundle, checkpoint: str | Path, out: str | Path,
                         device: str = "cpu", bundle_sha256: str | None = None) -> dict[str, Any]:
    """Score every row of one horizon bundle without changing fitted state."""
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("schema") != "acie.vv.checkpoint.v1":
        raise ValueError("Unsupported VV checkpoint")
    torch.set_num_threads(int(payload["config"].get("num_threads", 2)))
    spec = payload["model_spec"]
    model = build_model(spec["model_id"], spec["a_dim"], spec["q_dim"], spec["dropout"])
    model.load_state_dict(payload["state"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    scaler = SourceScaler.from_dict(payload["scaler"])
    a, q = scaler.transform(bundle.a, bundle.q)
    started = time.perf_counter()
    logit, geometry, evidence = _predict(model, a, q, _all_indices(bundle), torch.device(device))
    seconds = time.perf_counter() - started
    score = _sigmoid(logit)
    checkpoint_hash = digest_file(checkpoint)
    rows = []
    for i, meta in enumerate(bundle.meta):
        rows.append({
            **meta, "label": int(bundle.y[i]), "score": float(score[i]),
            "logit": float(logit[i]), "model_id": spec["model_id"],
            "seed": int(payload["master_seed"]), "split_id": "r4_horizon",
            "checkpoint_hash": checkpoint_hash,
            "source_threshold": float(payload["threshold"]),
            "geometry_logit": None if np.isnan(geometry[i]) else float(geometry[i]),
            "evidence_logit": None if np.isnan(evidence[i]) else float(evidence[i]),
        })
    out = Path(out)
    write_jsonl(out, rows)
    result = {
        "schema": "acie.vv-r4-score.v1", "kind": "neural",
        "model_id": spec["model_id"], "seed": int(payload["master_seed"]),
        "checkpoint_hash": checkpoint_hash, "bundle_sha256": bundle_sha256,
        "weights_updated": False, "scaler_updated": False, "threshold_updated": False,
        "metrics": binary_metrics(bundle.y, score, payload["threshold"]),
        "inference_seconds": seconds, "environment": environment(),
    }
    write_json(out.with_suffix(".metrics.json"), result)
    return result


def score_horizon_tree(bundle: Bundle, checkpoint: str | Path, out: str | Path,
                       bundle_sha256: str | None = None) -> dict[str, Any]:
    """Score every row of one horizon bundle with a locked tree."""
    checkpoint = Path(checkpoint)
    payload = joblib.load(checkpoint)
    if payload.get("schema") != "acie.vv-tree-checkpoint.v1":
        raise ValueError("Unsupported VV tree checkpoint")
    scaler = SourceScaler.from_dict(payload["scaler"])
    x = tree_features(scaler, bundle.a, bundle.q)
    started = time.perf_counter()
    score = payload["model"].predict_proba(x)[:, 1]
    seconds = time.perf_counter() - started
    checkpoint_hash = digest_file(checkpoint)
    rows = [{
        **meta, "label": int(bundle.y[i]), "score": float(score[i]),
        "model_id": payload["kind"], "seed": int(payload["seed"]),
        "split_id": "r4_horizon", "checkpoint_hash": checkpoint_hash,
        "source_threshold": float(payload["threshold"]),
        "geometry_logit": None, "evidence_logit": None,
    } for i, meta in enumerate(bundle.meta)]
    out = Path(out)
    write_jsonl(out, rows)
    result = {
        "schema": "acie.vv-r4-score.v1", "kind": "tree",
        "model_id": payload["kind"], "seed": int(payload["seed"]),
        "checkpoint_hash": checkpoint_hash, "bundle_sha256": bundle_sha256,
        "weights_updated": False, "scaler_updated": False, "threshold_updated": False,
        "metrics": binary_metrics(bundle.y, score, payload["threshold"]),
        "inference_seconds": seconds,
    }
    write_json(out.with_suffix(".metrics.json"), result)
    return result
