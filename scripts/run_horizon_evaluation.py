#!/usr/bin/env python3
"""Score frozen ACIE checkpoints on E4 horizon bundles and summarize results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from acie.data import Bundle, group_key
from acie.engine import _predict, _sigmoid
from acie.features import SourceScaler
from acie.io import digest_file, digest_object, read_json, read_jsonl, write_json, write_jsonl
from acie.metrics import binary_metrics, bootstrap
from acie.models import Predictor


def score_bundle(bundle: Bundle, checkpoint_path: Path, output: Path, device: str, bundle_sha256: str) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != "acie.checkpoint.v1":
        raise ValueError("Unsupported checkpoint")
    # Reproduce the CPU reduction order saved in the frozen training config.
    torch.set_num_threads(int(checkpoint["config"].get("num_threads", 2)))
    model = Predictor(**checkpoint["spec"])
    model.load_state_dict(checkpoint["state"])
    model.to(device)
    scaler = SourceScaler.from_dict(checkpoint["scaler"])
    a, q = scaler.transform(bundle.a, bundle.q)
    indices = np.arange(len(bundle.y), dtype=int)
    logit, geometry, evidence = _predict(model, a, q, indices, torch.device(device))
    scores = _sigmoid(logit)
    rows = []
    for i, meta in enumerate(bundle.meta):
        audit_meta = {key: meta[key] for key in [
            "sample_id", "dataset", "recording", "track_id", "pool", "panoramic",
            "group_id", "cutoff_seconds", "event_lead_seconds", "anchor_policy"
        ] if key in meta}
        audit_meta["horizon_shift_frames"] = int(meta["source_index"]["horizon_shift_frames"])
        rows.append({
            **audit_meta, "label": int(bundle.y[i]), "score": float(scores[i]),
            "logit": float(logit[i]), "geometry_logit": float(geometry[i]),
            "evidence": float(evidence[i]), "threshold": float(checkpoint["threshold"]),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, rows)
    result = {
        "metrics": binary_metrics(bundle.y, scores, checkpoint["threshold"]),
        "checkpoint_sha256": digest_file(checkpoint_path),
        "bundle_sha256": bundle_sha256,
        "predictions_sha256": digest_file(output),
        "source_threshold": float(checkpoint["threshold"]),
        "source_val": checkpoint["source_val"],
        "target_labels_used_for_selection": False,
    }
    write_json(output.with_suffix(".metrics.json"), result)
    return result


def table(path: Path) -> dict[str, dict]:
    rows = read_jsonl(path)
    result = {row["sample_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate sample IDs in {path}")
    return result


def subset_metrics(rows: dict[str, dict], ids: list[str]) -> dict:
    y = np.asarray([rows[sid]["label"] for sid in ids], dtype=int)
    s = np.asarray([rows[sid]["score"] for sid in ids], dtype=float)
    threshold = next(iter({float(rows[sid]["threshold"]) for sid in ids}))
    return binary_metrics(y, s, threshold)


def summarize_metric_dicts(items: list[dict]) -> dict:
    result = {}
    for key in ["AP", "AUROC", "recall", "false_positive_rate", "precision"]:
        values = [float(item[key]) for item in items if item[key] is not None]
        result[key + "_mean"] = float(np.mean(values))
        result[key + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return result


def ensemble(paths: list[Path], ids: list[str]) -> tuple[np.ndarray, np.ndarray, list[dict], np.ndarray]:
    tables = [table(path) for path in paths]
    if any(set(ids) - set(t) for t in tables):
        raise ValueError("Ensemble prediction coverage mismatch")
    labels = np.asarray([tables[0][sid]["label"] for sid in ids], dtype=int)
    if any(any(int(t[sid]["label"]) != int(labels[i]) for i, sid in enumerate(ids)) for t in tables[1:]):
        raise ValueError("Ensemble label mismatch")
    matrix = np.asarray([[t[sid]["score"] for sid in ids] for t in tables], dtype=float)
    return labels, matrix.mean(axis=0), [tables[0][sid] for sid in ids], matrix


def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "n": len(labels), "positives": int(labels.sum()),
        "AP": float(average_precision_score(labels, scores)),
        "AUROC": float(roc_auc_score(labels, scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    root = args.root.resolve()
    args.out = args.out.resolve()
    protocol = read_json(args.protocol.resolve())
    if protocol.get("schema") != "acie.horizon-evaluation-protocol.v1":
        raise ValueError("Unexpected horizon protocol")
    frozen = dict(protocol)
    digest = frozen.pop("protocol_digest")
    if digest_object(frozen) != digest:
        raise ValueError("Protocol digest mismatch")

    args.out.mkdir(parents=True, exist_ok=True)
    predictions = {}
    for direction, target in protocol["directions"].items():
        predictions[direction] = {}
        for bundle_spec in protocol["bundles"][target]:
            cutoff = int(bundle_spec["cutoff_frames"])
            bundle_path = root / bundle_spec["bundle"]
            if digest_file(bundle_path) != bundle_spec["bundle_sha256"]:
                raise ValueError(f"Bundle changed: {bundle_path}")
            bundle = Bundle.load(bundle_path)
            predictions[direction][cutoff] = {}
            for method in protocol["methods"]:
                predictions[direction][cutoff][method] = []
                for seed in protocol["seeds"]:
                    ck_spec = protocol["checkpoints"][direction][method][str(seed)]
                    checkpoint = root / ck_spec["path"]
                    if digest_file(checkpoint) != ck_spec["sha256"]:
                        raise ValueError(f"Checkpoint changed: {checkpoint}")
                    output = args.out / direction / method / f"seed{seed}" / f"cutoff_frames_{cutoff:03d}.jsonl"
                    if output.is_file():
                        rows = table(output)
                        if set(rows) != {meta["sample_id"] for meta in bundle.meta}:
                            raise ValueError(f"Existing prediction coverage mismatch: {output}")
                    else:
                        score_bundle(bundle, checkpoint, output, args.device, bundle_spec["bundle_sha256"])
                    predictions[direction][cutoff][method].append(output)
                    print(f"DONE {direction} {method} seed={seed} cutoff={cutoff}", flush=True)

    # The base horizon must reproduce every original score before later horizons are interpreted.
    baseline_checks = []
    for direction in protocol["directions"]:
        for method in protocol["methods"]:
            for seed in protocol["seeds"]:
                new_path = predictions[direction][protocol["base_cutoff_frames"]][method][protocol["seeds"].index(seed)]
                old_path = root / "outputs/frozen_cross_domain" / direction / method / f"seed{seed}/test_predictions.jsonl"
                new, old = table(new_path), table(old_path)
                if set(new) != set(old):
                    raise AssertionError("Base horizon IDs differ from frozen target predictions")
                fields = ["score", "logit", "geometry_logit", "evidence"]
                maxima = {field: max(abs(float(new[sid][field]) - float(old[sid][field])) for sid in new) for field in fields}
                tolerances = {"score": 1e-7, "logit": 5e-6, "geometry_logit": 5e-6, "evidence": 5e-6}
                if any(maxima[field] > tolerances[field] for field in fields):
                    raise AssertionError(f"Base prediction mismatch {direction}/{method}/seed{seed}: {maxima}")
                baseline_checks.append({"direction": direction, "method": method, "seed": seed, "max_abs": maxima})

    summary = {
        "schema": "acie.horizon-evaluation-summary.v1",
        "protocol_digest": protocol["protocol_digest"],
        "bootstrap_repeats": args.bootstrap,
        "directions": {},
        "baseline_reproduction": baseline_checks,
    }
    for direction_index, (direction, target) in enumerate(protocol["directions"].items()):
        cutoffs = sorted(predictions[direction])
        all_ids = {}
        for cutoff in cutoffs:
            first = table(predictions[direction][cutoff][protocol["methods"][0]][0])
            all_ids[cutoff] = sorted(first)
        common = sorted(set.intersection(*(set(ids) for ids in all_ids.values())))
        direction_result = {"target": target, "common_track_count": len(common), "common_positives": None, "horizons": []}
        for cutoff_index, cutoff in enumerate(cutoffs):
            item = {"cutoff_frames": cutoff, "cutoff_seconds": cutoff / protocol["fps"], "sets": {}}
            for set_name, ids in [("complete", all_ids[cutoff]), ("common_tracks", common)]:
                method_result = {}
                ensemble_values = {}
                for method in protocol["methods"]:
                    paths = predictions[direction][cutoff][method]
                    per_seed = [subset_metrics(table(path), ids) for path in paths]
                    labels, scores, meta, matrix = ensemble(paths, ids)
                    ensemble_values[method] = (labels, scores, meta)
                    method_result[method] = {
                        "per_seed": [{"seed": seed, **metrics} for seed, metrics in zip(protocol["seeds"], per_seed, strict=True)],
                        "seed_summary_at_source_frozen_thresholds": summarize_metric_dicts(per_seed),
                        "ensemble_ranking": ranking_metrics(labels, scores),
                    }
                    ensemble_path = args.out / direction / method / f"ensemble_{set_name}_cutoff_frames_{cutoff:03d}.jsonl"
                    write_jsonl(ensemble_path, [
                        {**meta_row, "label": int(label), "score": float(score),
                         "seed_scores": {str(seed): float(value) for seed, value in zip(protocol["seeds"], seed_scores, strict=True)}}
                        for meta_row, label, score, seed_scores in zip(meta, labels, scores, matrix.T, strict=True)
                    ])
                    method_result[method]["ensemble_predictions"] = str(ensemble_path.relative_to(root))
                    method_result[method]["ensemble_predictions_sha256"] = digest_file(ensemble_path)
                labels, dual_scores, meta = ensemble_values["dual_no_pair"]
                geometry_scores = ensemble_values["geometry"][1]
                comparison = bootstrap(
                    labels, dual_scores, [group_key(row) for row in meta],
                    args.bootstrap, seed=20260915 + direction_index * 100 + cutoff_index,
                    other=geometry_scores,
                )
                comparison["point"] = float(average_precision_score(labels, dual_scores) - average_precision_score(labels, geometry_scores))
                item["sets"][set_name] = {
                    "n": len(ids), "positives": int(labels.sum()),
                    "methods": method_result,
                    "dual_no_pair_minus_geometry": comparison,
                }
                if set_name == "common_tracks" and direction_result["common_positives"] is None:
                    direction_result["common_positives"] = int(labels.sum())
            direction_result["horizons"].append(item)
        summary["directions"][direction] = direction_result
    summary["summary_digest"] = digest_object(summary)
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
