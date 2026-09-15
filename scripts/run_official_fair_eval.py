"""Evaluate an official HUI360 checkpoint once on the frozen ACIE target split."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def threshold_metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(scores) >= threshold
    tp = int(np.sum(pred & (labels == 1)))
    fp = int(np.sum(pred & (labels == 0)))
    fn = int(np.sum(~pred & (labels == 1)))
    tn = int(np.sum(~pred & (labels == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold), "accuracy": (tp + tn) / len(labels),
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--direction", choices=["HUI360_to_SSUP-A", "SSUP-A_to_HUI360"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = args.root.resolve()
    upstream = root / "external" / "HUI360-Baselines"
    checkpoint_path = args.checkpoint.resolve()
    target_config_path = args.target_config.resolve()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("The pinned official inference code requires CUDA timing events")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "val_f1_used_adaptative_threshold" not in checkpoint:
        raise ValueError("Checkpoint lacks its source-validation threshold")
    source_threshold = float(checkpoint["val_f1_used_adaptative_threshold"])
    target_config = yaml.safe_load(target_config_path.read_text())
    # The pinned upstream loader adds these defaults at runtime. Normalize the
    # raw target YAML before comparing it with the checkpoint configuration.
    target_config.setdefault("mb_desired_return", "representation")
    target_config.setdefault("cutoffs_filtering", True)
    target_config["acie_seed"] = args.seed
    source_config = checkpoint["config"]
    allowed_target_changes = {
        "comment", "cross_eval_type", "experiment_name", "fix_index_per_track_list_val",
        "include_recordings_val", "test_tracks_filename", "val_tracks_filename",
    }
    changed = {
        key for key in set(source_config) | set(target_config)
        if source_config.get(key) != target_config.get(key)
    }
    unexpected = changed - allowed_target_changes
    if unexpected:
        raise ValueError(f"Target evaluation changes immutable config fields: {sorted(unexpected)}")

    eval_checkpoint = dict(checkpoint)
    eval_checkpoint["config"] = target_config
    eval_checkpoint_path = output / "checkpoint_target_metadata_only.pth"
    torch.save(eval_checkpoint, eval_checkpoint_path)

    sys.path.insert(0, str(upstream))
    import infer  # noqa: PLC0415

    infer_args = SimpleNamespace(
        preload_data=True, preload_only=False, verbose=False,
        hf_local_dir=str(args.data_root.resolve()), offline_mode=True,
        num_workers=args.num_workers,
    )
    started = time.time()
    labels, scores, track_ids = infer.main(infer_args, str(eval_checkpoint_path))
    elapsed = time.time() - started
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    target_bundle_name = "ssup_official_test_windows" if args.direction == "HUI360_to_SSUP-A" else "hui_official_test_windows"
    manifest_path = root / "data" / target_bundle_name / "manifest.jsonl"
    target_meta = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    split = json.loads((root / "outputs" / "frozen_cross_domain" / args.direction / "split.json").read_text())
    test_ids = set(split["test"])
    expected_by_id = {row["sample_id"]: row for row in target_meta if row["sample_id"] in test_ids}
    by_track = {row["track_id"]: row for row in expected_by_id.values()}
    if len(track_ids) != len(by_track) or set(track_ids) != set(by_track):
        raise ValueError(f"Target track mismatch: got {len(track_ids)}, expected {len(by_track)}")
    observed = {track: (int(label), float(score)) for track, label, score in zip(track_ids, labels, scores)}
    rows = []
    ordered_labels = []
    ordered_scores = []
    for sample_id in split["test"]:
        meta = expected_by_id[sample_id]
        label, score = observed[meta["track_id"]]
        if label != int(meta["label"]):
            raise ValueError(f"Label mismatch for {sample_id}")
        ordered_labels.append(label)
        ordered_scores.append(score)
        rows.append({
            "sample_id": sample_id, "dataset": meta["dataset"], "recording": meta["recording"],
            "group_id": meta["group_id"], "track_id": meta["track_id"], "label": label,
            "score": score, "threshold": source_threshold,
        })

    predictions_path = output / "predictions.jsonl"
    predictions_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    metrics = {
        "schema": "acie.official-fair-result.v1",
        "direction": args.direction,
        "model": target_config["force_model_type"],
        "seed": args.seed,
        "n_test": len(rows),
        "test_positives": int(sum(ordered_labels)),
        "AP": float(average_precision_score(ordered_labels, ordered_scores)),
        "AUROC": float(roc_auc_score(ordered_labels, ordered_scores)),
        "source_validation_threshold_metrics": threshold_metrics(ordered_labels, ordered_scores, source_threshold),
        "checkpoint_selection": "best source-validation AP",
        "source_validation_AP_at_checkpoint": float(checkpoint["val_ap"]),
        "source_validation_AUROC_at_checkpoint": float(checkpoint["val_auc"]),
        "source_validation_epoch_zero_based": int(checkpoint["epoch"]),
        "target_labels_used_for_selection": False,
        "evaluation_seconds": elapsed,
        "checkpoint_sha256": sha256(checkpoint_path),
        "target_config_sha256": sha256(target_config_path),
        "predictions_sha256": sha256(predictions_path),
        "metadata_patch": "only checkpoint config fields needed to construct the frozen target loader; weights unchanged",
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
