"""Source-only training and locked target scoring for the VV follow-up."""
from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F

from .data import Bundle, resolve_split
from .features import SourceScaler
from .io import digest_file, digest_object, environment, write_json, write_jsonl
from .metrics import binary_metrics, choose_threshold
from .vv_models import (
    ADDITIVE_IDS,
    EXPECTED_PARAMETERS,
    build_model,
    configure_training_mode,
    effective_parameter_count,
    validate_parameter_count,
)
from .vv_provenance import (IDENTITY_VERSION, source_snapshot, training_content_digest,
                            validate_complete_run)


DEFAULT_CONFIG = {
    "lr": 0.001,
    "weight_decay": 0.0005,
    "batch_size": 64,
    "dropout": 0.1,
    "patience": 10,
    "gradient_clip": 5.0,
    "geometry_epochs": 30,
    "stage2_epochs": 50,
    "single_stage_epochs": 80,
    "device": "cpu",
    "num_threads": 2,
}


def derived_seed(master_seed: int, stream: str) -> int:
    raw = hashlib.sha256(f"{int(master_seed)}::{stream}".encode()).digest()
    return int.from_bytes(raw[:8], "big") % (2**31 - 1)


def _tensor_state_digest(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(str(array.dtype).encode())
        h.update(str(array.shape).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def _array_digest(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def _sigmoid(logit: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -60, 60)))


def _predict(model: torch.nn.Module, a: np.ndarray, q: np.ndarray, indices: np.ndarray,
             device: torch.device, batch_size: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    parts: list[list[np.ndarray]] = [[], [], []]
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            ix = indices[start:start + batch_size]
            output = model(torch.from_numpy(a[ix]).to(device), torch.from_numpy(q[ix]).to(device))
            for bucket, value in zip(parts, output):
                bucket.append(value.detach().cpu().numpy())
    return tuple(np.concatenate(bucket).astype(np.float32) for bucket in parts)  # type: ignore[return-value]


def _safe_metrics(y: np.ndarray, logits: np.ndarray) -> dict[str, float | int | None]:
    score = _sigmoid(logits)
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "AP": float(average_precision_score(y, score)),
        "AUROC": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
    }


def _set_runtime(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _write_yaml(path: Path, value: Any) -> None:
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(yaml.safe_dump(value, sort_keys=True, allow_unicode=True), encoding="utf-8")
    temp.replace(path)


def _validate_config(config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if config:
        unknown = set(config) - set(cfg)
        if unknown:
            raise ValueError(f"Unknown VV configuration keys: {sorted(unknown)}")
        cfg.update(config)
    positive = ("lr", "batch_size", "patience", "gradient_clip", "geometry_epochs",
                "stage2_epochs", "single_stage_epochs", "num_threads")
    if any(float(cfg[key]) <= 0 for key in positive) or float(cfg["weight_decay"]) < 0:
        raise ValueError("Invalid VV training configuration")
    return cfg


def _capture_rng(batch_rng: np.random.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy_batch": batch_rng.bit_generator.state,
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict[str, Any], batch_rng: np.random.Generator) -> None:
    random.setstate(state["python"])
    batch_rng.bit_generator.state = state["numpy_batch"]
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _fit(
    model: torch.nn.Module,
    model_id: str,
    a: np.ndarray,
    q: np.ndarray,
    y: np.ndarray,
    train_ix: np.ndarray,
    val_ix: np.ndarray,
    cfg: dict[str, Any],
    master_seed: int,
    epochs: int,
    out: Path,
    resume: bool,
    positive_class_weight: float | None = None,
) -> tuple[list[dict[str, Any]], int, int, int]:
    device = torch.device(cfg["device"])
    model.to(device)
    configure_training_mode(model, model_id)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if positive_class_weight is not None and positive_class_weight <= 0:
        raise ValueError("positive_class_weight must be positive")
    pos_weight = (None if positive_class_weight is None else
                  torch.tensor(float(positive_class_weight), dtype=torch.float32, device=device))
    batch_seed = derived_seed(master_seed, "classification_batches")
    dropout_seed = derived_seed(master_seed, "classification_dropout")
    batch_rng = np.random.default_rng(batch_seed)
    torch.manual_seed(dropout_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(dropout_seed)

    last_path = out / "last.pt"
    start_epoch = 1
    best_ap = -float("inf")
    best_epoch = None
    best_state = None
    patience_counter = 0
    history: list[dict[str, Any]] = []
    optimizer_steps = 0
    processed_examples = 0
    if resume and last_path.exists():
        last = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(last["state"])
        optimizer.load_state_dict(last["optimizer"])
        start_epoch = int(last["epoch"]) + 1
        best_ap = float(last["best_ap"])
        best_epoch = last["best_epoch"]
        best_state = last["best_state"]
        patience_counter = int(last["patience_counter"])
        history = last["history"]
        optimizer_steps = int(last["optimizer_steps"])
        processed_examples = int(last["processed_examples"])
        _restore_rng(last["rng"], batch_rng)
    elif last_path.exists():
        raise FileExistsError(f"{last_path} already exists; use --resume or a new directory")
    else:
        initial_logits = _predict(model, a, q, val_ix, device)[0]
        history.append({"epoch": 0, "train_loss": None, "BCE": None, "pair": None,
                        "source_val_AP": _safe_metrics(y[val_ix], initial_logits)["AP"],
                        "source_val_AUROC": _safe_metrics(y[val_ix], initial_logits)["AUROC"],
                        "best_epoch": None, "patience_counter": 0,
                        "actual_optimizer_steps": 0, "processed_examples": 0,
                        "cumulative_seconds": 0.0})
        write_jsonl(out / "history.jsonl", history)

    training_start = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        if patience_counter >= cfg["patience"]:
            break
        configure_training_mode(model, model_id)
        order = batch_rng.permutation(len(train_ix))
        losses: list[float] = []
        for start in range(0, len(order), cfg["batch_size"]):
            local = order[start:start + cfg["batch_size"]]
            ix = train_ix[local]
            aa = torch.from_numpy(a[ix]).to(device)
            qq = torch.from_numpy(q[ix]).to(device)
            target = torch.from_numpy(y[ix].astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(aa, qq)[0]
            loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{model_id} produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, cfg["gradient_clip"])
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            optimizer_steps += 1
            processed_examples += len(ix)

        val_logits = _predict(model, a, q, val_ix, device)[0]
        metrics = _safe_metrics(y[val_ix], val_logits)
        improved = float(metrics["AP"]) > best_ap
        if improved:
            best_ap = float(metrics["AP"])
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "BCE": float(np.mean(losses)),
            "pair": None,
            "source_val_AP": metrics["AP"],
            "source_val_AUROC": metrics["AUROC"],
            "best_epoch": best_epoch,
            "patience_counter": patience_counter,
            "actual_optimizer_steps": optimizer_steps,
            "processed_examples": processed_examples,
            "cumulative_seconds": time.perf_counter() - training_start,
            "positive_class_weight": positive_class_weight,
        })
        write_jsonl(out / "history.jsonl", history)
        payload = {
            "epoch": epoch,
            "state": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_ap": best_ap,
            "best_epoch": best_epoch,
            "best_state": best_state,
            "patience_counter": patience_counter,
            "history": history,
            "optimizer_steps": optimizer_steps,
            "processed_examples": processed_examples,
            "rng": _capture_rng(batch_rng),
        }
        temp = last_path.with_suffix(".tmp")
        torch.save(payload, temp)
        temp.replace(last_path)

    if best_state is None:
        raise RuntimeError("No trained epoch produced a selectable checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return history, int(best_epoch), optimizer_steps, processed_examples


def train_source_model(
    bundle: Bundle,
    split: dict[str, Any],
    model_id: str,
    master_seed: int,
    out: str | Path,
    config: dict[str, Any] | None = None,
    geometry_checkpoint: str | Path | None = None,
    resume: bool = False,
    context: dict[str, Any] | None = None,
    positive_class_weight: float | None = None,
) -> dict[str, Any]:
    """Train using source train/val only. This function never scores target test."""
    cfg = _validate_config(config)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    indices = resolve_split(bundle, split)
    train_ix, val_ix = indices["train"], indices["val"]
    if model_id in ("F", "Jw", "JwD") and geometry_checkpoint is None:
        raise ValueError(f"{model_id} requires a shared geometry checkpoint")
    if model_id not in EXPECTED_PARAMETERS:
        raise ValueError(f"Unknown VV model ID: {model_id}")
    if cfg["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    split_hash = digest_object(split)
    context = dict(context or {})
    code_snapshot = source_snapshot()
    training_hash = training_content_digest(bundle, indices)
    geometry_checkpoint_hash = (digest_file(geometry_checkpoint)
                                if geometry_checkpoint is not None else None)
    context_identity = {
        key: context.get(key) for key in (
            "experiment", "split_id", "fold_id", "feature_version",
            "data_hashes", "source_snapshot_hash", "selection_lock_hash",
        ) if context.get(key) is not None
    }
    run_identity = {
        "identity_version": IDENTITY_VERSION,
        "model_id": model_id,
        "seed": master_seed,
        "config": cfg,
        "split_hash": split_hash,
        "training_content_hash": training_hash,
        "code_snapshot_hash": code_snapshot["sha256"],
        "geometry_checkpoint_hash": geometry_checkpoint_hash,
        "context": context_identity,
    }
    if positive_class_weight is not None:
        run_identity["positive_class_weight"] = float(positive_class_weight)
    run_key = digest_object(run_identity)
    run_path = out / "run.json"
    if run_path.exists():
        prior = json.loads(run_path.read_text())
        if prior.get("run_id") != run_key:
            raise ValueError("Existing run directory has a different run identity")
        if prior.get("status") == "complete":
            validate_complete_run(out, prior)
            return prior
        if not resume:
            raise FileExistsError("Incomplete run exists; use --resume or a new directory")

    _write_yaml(out / "config.yaml", cfg)
    write_json(out / "split.json", split)
    scaler = SourceScaler.fit(bundle.a[train_ix], bundle.q[train_ix])
    write_json(out / "scaler.json", scaler.to_dict())
    scaler_hash = digest_object(scaler.to_dict())
    a, q = scaler.transform(bundle.a, bundle.q)

    init_stream = "behavior_init" if model_id in ("F", "Jw", "JwD") else "model_init"
    init_seed = derived_seed(master_seed, init_stream)
    _set_runtime(init_seed, cfg["num_threads"])
    model = build_model(model_id, a.shape[1], q.shape[2], cfg["dropout"])
    validate_parameter_count(model, model_id)

    geometry_reference = None
    geometry_before = None
    if geometry_checkpoint is not None:
        geometry_checkpoint = Path(geometry_checkpoint)
        geometry_payload = torch.load(geometry_checkpoint, map_location="cpu", weights_only=True)
        if geometry_payload.get("schema") != "acie.vv.checkpoint.v1" or geometry_payload["model_spec"]["model_id"] != "G":
            raise ValueError("Geometry reference is not a VV G checkpoint")
        if geometry_payload["split_hash"] != split_hash or int(geometry_payload["master_seed"]) != int(master_seed):
            raise ValueError("Geometry reference split/seed mismatch")
        model.geometry.load_state_dict(geometry_payload["geometry_state"])
        geometry_before = {k: v.detach().cpu().clone() for k, v in model.geometry.state_dict().items()}
        geometry_reference = {
            "path": str(geometry_checkpoint.resolve()),
            "sha256": geometry_checkpoint_hash,
            "state_sha256": _tensor_state_digest(geometry_before),
            "split_hash": split_hash,
            "master_seed": master_seed,
        }
        write_json(out / "geometry_reference.json", geometry_reference)

    configure_training_mode(model, model_id)
    model.to(torch.device(cfg["device"]))
    behavior_state = ({k: v.detach().cpu() for k, v in model.behavior.state_dict().items()}
                      if hasattr(model, "behavior") and model_id in ADDITIVE_IDS else {})
    preview_rng = np.random.default_rng(derived_seed(master_seed, "classification_batches"))
    preview_local = preview_rng.permutation(len(train_ix))[:cfg["batch_size"]]
    preview_ix = train_ix[preview_local]
    initial_logits = _predict(model, a, q, preview_ix, torch.device(cfg["device"]))[0]
    initial_state = {
        "initialization_stream": init_stream,
        "initialization_seed": init_seed,
        "classification_batch_seed": derived_seed(master_seed, "classification_batches"),
        "classification_dropout_seed": derived_seed(master_seed, "classification_dropout"),
        "behavior_state_sha256": _tensor_state_digest(behavior_state) if behavior_state else None,
        "first_batch_ids_sha256": digest_object([bundle.meta[i]["sample_id"] for i in preview_ix]),
        "first_batch_initial_logits_sha256": _array_digest(initial_logits),
        "zero_increment_head": model_id in ("Q", "Jc", "Jw", "JwD", "F", "C64", "Cm"),
    }
    write_json(out / "initial_state.json", initial_state)

    epochs = (cfg["geometry_epochs"] if model_id == "G" else
              cfg["stage2_epochs"] if model_id in ("F", "Jw", "JwD") else
              cfg["single_stage_epochs"])
    started = time.time()
    run = {
        "schema": "acie.vv-run.v1",
        "identity_version": IDENTITY_VERSION,
        "run_id": run_key,
        "experiment": context.get("experiment", "unspecified"),
        "source": split["source"],
        "target": split["target"],
        "model_id": model_id,
        "effective_architecture": model.spec,
        "stage": "source_training",
        "split_id": context.get("split_id", "fixed"),
        "split_hash": split_hash,
        "fold_id": context.get("fold_id"),
        "master_seed": master_seed,
        "data_paths": context.get("data_paths", []),
        "data_hashes": context.get("data_hashes", []),
        "feature_version": context.get("feature_version", "acie.features.extract.v1"),
        "training_content_hash": training_hash,
        "source_snapshot_hash": code_snapshot["sha256"],
        "source_snapshot": code_snapshot,
        "declared_source_snapshot_hash": context.get("source_snapshot_hash"),
        "resolved_config": cfg,
        "positive_class_weight": positive_class_weight,
        "scaler_hash": scaler_hash,
        "geometry_checkpoint_hash": geometry_reference["sha256"] if geometry_reference else None,
        "behavior_init_hash": initial_state["behavior_state_sha256"],
        "selection_rule": "strict source-validation AP improvement; ties keep earlier epoch",
        "target_scoring_performed": False,
        "status": "started",
        "started_at": started,
        "finished_at": None,
        "failure_reason": None,
    }
    write_json(run_path, run)

    try:
        training_start = time.perf_counter()
        history, best_epoch, optimizer_steps, processed_examples = _fit(
            model, model_id, a, q, bundle.y, train_ix, val_ix, cfg,
            master_seed, epochs, out, resume, positive_class_weight,
        )
        training_seconds = time.perf_counter() - training_start
        if model_id == "F" and geometry_before is not None:
            geometry_after = model.geometry.state_dict()
            if any(not torch.equal(geometry_before[k], geometry_after[k].cpu()) for k in geometry_before):
                raise AssertionError("F geometry state changed during frozen training")

        val_logit, val_geometry, val_evidence = _predict(
            model, a, q, val_ix, torch.device(cfg["device"])
        )
        val_score = _sigmoid(val_logit)
        threshold = choose_threshold(bundle.y[val_ix], val_score)
        val_rows = []
        for position, i in enumerate(val_ix):
            row = {
                **bundle.meta[i], "label": int(bundle.y[i]), "score": float(val_score[position]),
                "logit": float(val_logit[position]), "model_id": model_id, "seed": master_seed,
                "split_id": context.get("split_id", "fixed"), "source_threshold": threshold,
                "geometry_logit": None if np.isnan(val_geometry[position]) else float(val_geometry[position]),
                "evidence_logit": None if np.isnan(val_evidence[position]) else float(val_evidence[position]),
            }
            val_rows.append(row)
        write_jsonl(out / "val_predictions.jsonl", val_rows)
        val_metrics = binary_metrics(bundle.y[val_ix], val_score, threshold)
        checkpoint = {
            "schema": "acie.vv.checkpoint.v1",
            "model_spec": model.spec,
            "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "geometry_state": ({k: v.detach().cpu() for k, v in model.geometry.state_dict().items()}
                               if hasattr(model, "geometry") and model_id in ("G", "F", "Jw", "JwD", "Jc") else None),
            "scaler": scaler.to_dict(),
            "split": split,
            "split_hash": split_hash,
            "master_seed": master_seed,
            "config": cfg,
            "threshold": threshold,
            "best_epoch": best_epoch,
            "source_val": val_metrics,
            "run_id": run_key,
            "positive_class_weight": positive_class_weight,
        }
        torch.save(checkpoint, out / "best.pt")
        checkpoint_hash = digest_file(out / "best.pt")
        for row in val_rows:
            row["checkpoint_hash"] = checkpoint_hash
        write_jsonl(out / "val_predictions.jsonl", val_rows)
        write_json(out / "metrics.json", {"source_val": val_metrics, "target": None})
        write_json(out / "timing.json", {
            "training_seconds": training_seconds,
            "best_epoch": best_epoch,
            "epochs_completed": len(history) - 1,
            "actual_optimizer_steps": optimizer_steps,
            "processed_examples": processed_examples,
            "effective_parameters": effective_parameter_count(model),
            "trainable_final_stage_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        })
        run.update(status="complete", finished_at=time.time(), checkpoint_hash=checkpoint_hash,
                   best_epoch=best_epoch, source_val=val_metrics,
                   effective_parameters=effective_parameter_count(model),
                   target_scoring_performed=False)
        write_json(run_path, run)
        return run
    except Exception as exc:
        run.update(status="failed", finished_at=time.time(), failure_reason=f"{type(exc).__name__}: {exc}")
        write_json(run_path, run)
        raise


def score_locked_target(bundle: Bundle, checkpoint: str | Path, out: str | Path,
                        device: str = "cpu", split_name: str = "test") -> dict[str, Any]:
    """Pure inference entry point; it never updates weights, thresholds, or scalers."""
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("schema") != "acie.vv.checkpoint.v1":
        raise ValueError("Unsupported VV checkpoint")
    model_id = payload["model_spec"]["model_id"]
    model = build_model(model_id, payload["model_spec"]["a_dim"], payload["model_spec"]["q_dim"],
                        payload["model_spec"]["dropout"])
    model.load_state_dict(payload["state"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    indices = resolve_split(bundle, payload["split"])[split_name]
    scaler = SourceScaler.from_dict(payload["scaler"])
    a, q = scaler.transform(bundle.a, bundle.q)
    started = time.perf_counter()
    logit, geometry, evidence = _predict(model, a, q, indices, torch.device(device))
    inference_seconds = time.perf_counter() - started
    score = _sigmoid(logit)
    checkpoint_hash = digest_file(checkpoint)
    rows = []
    for position, i in enumerate(indices):
        rows.append({
            **bundle.meta[i], "label": int(bundle.y[i]), "score": float(score[position]),
            "logit": float(logit[position]), "model_id": model_id,
            "seed": int(payload["master_seed"]),
            "split_id": payload["split"].get("split_id", "fixed"),
            "checkpoint_hash": checkpoint_hash, "source_threshold": float(payload["threshold"]),
            "geometry_logit": None if np.isnan(geometry[position]) else float(geometry[position]),
            "evidence_logit": None if np.isnan(evidence[position]) else float(evidence[position]),
        })
    out = Path(out)
    write_jsonl(out, rows)
    result = {
        "schema": "acie.vv-locked-score.v1",
        "model_id": model_id,
        "seed": int(payload["master_seed"]),
        "split": split_name,
        "checkpoint_hash": checkpoint_hash,
        "weights_updated": False,
        "scaler_updated": False,
        "threshold_updated": False,
        "metrics": binary_metrics(bundle.y[indices], score, payload["threshold"]),
        "inference_seconds": inference_seconds,
        "environment": environment(),
    }
    write_json(out.with_suffix(".metrics.json"), result)
    return result
