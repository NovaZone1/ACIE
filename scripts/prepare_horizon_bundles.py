#!/usr/bin/env python3
"""Export deterministic earlier prefixes from frozen official windows for E4.

Labels and anchor windows come from the frozen official manifests.  A horizon is
formed by shifting the complete 32-frame prefix earlier by an integer number of
frames.  No track is relabelled and no future field enters the model features.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from acie.data import Bundle
from acie.features import JOINTS, extract
from acie.io import digest_file, digest_object, read_jsonl, write_json

FPS = 15.0
BASE_CUTOFF_FRAMES = 16
HORIZON_SHIFTS = [0, 8, 16, 24, 32]
TARGETS = {
    "HUI360": "data/hui_official_test_windows",
    "SSUP-A": "data/ssup_official_test_windows",
}
SCORES = [f"vitpose_{joint}_score" for joint in JOINTS]
COLUMNS = [
    "unique_track_identifier", "image_index", "validity", "mask_size",
    "xmin", "xmax", "ymin", "ymax", "image_width", "image_height",
    "engagement", "time_to_first_interaction",
] + [f"vitpose_{joint}_{channel}" for joint in JOINTS for channel in ["x", "y", "score"]]


def recording_files(data_root: Path) -> dict[str, Path]:
    result = {}
    for path in sorted(data_root.glob("*.csv")):
        recording = str(pd.read_csv(path, usecols=["recording"], nrows=1)["recording"].iloc[0])
        if recording in result:
            raise ValueError(f"Duplicate recording {recording}")
        result[recording] = path
    return result


def official_filter(frame: pd.DataFrame) -> pd.DataFrame:
    """Mirror ssupaug_dataset_handling at pinned upstream commit a6bd61c."""
    keep = (frame["xmin"] <= frame["xmax"]) & (
        (frame["xmax"] - frame["xmin"]) <= frame["image_width"] / 2
    )
    return frame.loc[keep].copy()


def valid_prefix(frame: pd.DataFrame) -> tuple[bool, str | None]:
    if not np.all(frame["validity"].astype(str).to_numpy() == "valid"):
        return False, "invalid_flag"
    mask = frame["mask_size"].to_numpy(dtype=float)
    if not np.all((mask >= 1000) & (mask <= 1e7)):
        return False, "mask_filter"
    if not np.all((frame[SCORES].to_numpy(dtype=float) >= 0.5).sum(axis=1) >= 9):
        return False, "keypoint_filter"
    return True, None


def window_features(frame: pd.DataFrame, panoramic: bool) -> tuple[np.ndarray, np.ndarray]:
    width = frame["image_width"].to_numpy(dtype=np.float64)
    height = frame["image_height"].to_numpy(dtype=np.float64)
    if np.any(width <= 0) or np.any(height <= 0):
        raise ValueError("Invalid image dimensions")
    bbox = np.stack([
        frame["xmin"].to_numpy(dtype=np.float64) / width,
        frame["ymin"].to_numpy(dtype=np.float64) / height,
        frame["xmax"].to_numpy(dtype=np.float64) / width,
        frame["ymax"].to_numpy(dtype=np.float64) / height,
    ], axis=1)
    if panoramic:
        raw_width = bbox[:, 2] - bbox[:, 0]
        bbox[:, 0] %= 1
        bbox[:, 2] %= 1
        full = np.isclose(np.abs(raw_width), 1)
        bbox[full, 0] = 0
        bbox[full, 2] = 1
    keypoints = np.stack([
        np.stack([
            frame[f"vitpose_{joint}_x"].to_numpy(dtype=np.float64) / width,
            frame[f"vitpose_{joint}_y"].to_numpy(dtype=np.float64) / height,
            frame[f"vitpose_{joint}_score"].to_numpy(dtype=np.float64),
        ], axis=1)
        for joint in JOINTS
    ], axis=1)
    if panoramic:
        valid = keypoints[..., 0] >= 0
        keypoints[..., 0] = np.where(valid, keypoints[..., 0] % 1, keypoints[..., 0])
    mask_area = frame["mask_size"].to_numpy(dtype=np.float64) / (width * height)
    image_index = frame["image_index"].to_numpy(dtype=np.float64)
    times = (image_index - image_index[0]) / FPS
    return extract(keypoints, bbox, mask_area, times, panoramic=panoramic)


def locate_shifted(track: pd.DataFrame, base_indexes: list[int], shift: int) -> tuple[pd.DataFrame | None, str | None]:
    positions = np.flatnonzero(track["image_index"].to_numpy(dtype=int) == int(base_indexes[0]))
    if len(positions) != 1:
        return None, "base_start_not_unique"
    start = int(positions[0]) - shift
    if start < 0 or start + len(base_indexes) > len(track):
        return None, "insufficient_history"
    frame = track.iloc[start:start + len(base_indexes)].copy()
    wanted = np.asarray(base_indexes, dtype=int) - shift
    if not np.array_equal(frame["image_index"].to_numpy(dtype=int), wanted):
        return None, "nonconsecutive_or_mismatch"
    valid, reason = valid_prefix(frame)
    return (frame, None) if valid else (None, reason)


def export_target(root: Path, raw_root: Path, out: Path, domain: str, source_dir: Path,
                  files: dict[str, Path]) -> dict:
    rows = read_jsonl(source_dir / "manifest.jsonl")
    if any(row["dataset"] != domain or row["pool"] != "test" for row in rows):
        raise ValueError(f"Unexpected base manifest content for {domain}")
    by_recording = defaultdict(list)
    for row in rows:
        by_recording[row["recording"]].append(row)
    missing = sorted(set(by_recording) - set(files))
    if missing:
        raise FileNotFoundError(f"Missing raw recordings: {missing}")

    outputs = {shift: {"a": [], "q": [], "y": [], "meta": []} for shift in HORIZON_SHIFTS}
    failures = {shift: Counter() for shift in HORIZON_SHIFTS}
    positive_tti = {shift: [] for shift in HORIZON_SHIFTS}
    for rec_index, recording in enumerate(sorted(by_recording), 1):
        raw = pd.read_csv(files[recording], usecols=COLUMNS)
        raw = official_filter(raw)
        tracks = {str(key): value.reset_index(drop=True)
                  for key, value in raw.groupby("unique_track_identifier", sort=False)}
        for row in by_recording[recording]:
            track = tracks.get(str(row["track_id"]))
            if track is None:
                for shift in HORIZON_SHIFTS:
                    failures[shift]["track_missing"] += 1
                continue
            base = row["source_index"]["frame_indexes"]
            if len(base) != 32 or float(row["source_index"]["fps"]) != FPS:
                raise ValueError("E4 expects the frozen 32-frame, 15-fps base windows")
            for shift in HORIZON_SHIFTS:
                frame, reason = locate_shifted(track, base, shift)
                if reason:
                    failures[shift][reason] += 1
                    continue
                a, q = window_features(frame, bool(row["panoramic"]))
                image_indexes = frame["image_index"].to_numpy(dtype=int).tolist()
                actual_tti = float(frame["time_to_first_interaction"].iloc[-1])
                meta = {key: row[key] for key in [
                    "sample_id", "dataset", "recording", "track_id", "pool", "panoramic", "group_id"
                ]}
                meta.update({
                    "source_index": {
                        **row["source_index"],
                        "base_frame_indexes": base,
                        "frame_indexes": image_indexes,
                        "horizon_shift_frames": shift,
                    },
                    "synthetic": False,
                    "start_time": 0.0,
                    "end_time": float((image_indexes[-1] - image_indexes[0]) / FPS),
                    "cutoff_seconds": float((BASE_CUTOFF_FRAMES + shift) / FPS),
                    "anchor_policy": "interaction onset for positives; maximum-mask proxy for negatives",
                })
                if int(row["label"]) == 1 and np.isfinite(actual_tti):
                    meta["event_lead_seconds"] = actual_tti
                    positive_tti[shift].append(actual_tti)
                outputs[shift]["a"].append(a)
                outputs[shift]["q"].append(q)
                outputs[shift]["y"].append(int(row["label"]))
                outputs[shift]["meta"].append(meta)
        print(f"{domain}: recording {rec_index}/{len(by_recording)}", flush=True)

    source_bundle = Bundle.load(source_dir / "bundle.npz")
    coverage = {"domain": domain, "base_count": len(rows), "horizons": []}
    for shift in HORIZON_SHIFTS:
        item = outputs[shift]
        provenance = {
            "schema": "acie.horizon-bundle.v1",
            "domain": domain,
            "base_manifest": str((source_dir / "manifest.jsonl").relative_to(root)),
            "base_manifest_sha256": digest_file(source_dir / "manifest.jsonl"),
            "base_bundle_sha256": digest_file(source_dir / "bundle.npz"),
            "upstream_commit": "a6bd61c8539061242cbf756a18847c0900a8707c",
            "fps": FPS,
            "base_cutoff_frames": BASE_CUTOFF_FRAMES,
            "horizon_shift_frames": shift,
            "cutoff_seconds": (BASE_CUTOFF_FRAMES + shift) / FPS,
            "window_length_frames": 32,
            "selection": "same frozen track and label; entire prefix shifted earlier; official validity filters reapplied",
            "negative_alignment": "same maximum-mask proxy anchor as the frozen official base window",
            "future_fields_as_features": False,
            "synthetic": False,
        }
        bundle = Bundle(
            np.stack(item["a"]).astype(np.float32),
            np.stack(item["q"]).astype(np.float32),
            np.asarray(item["y"], dtype=np.int64),
            item["meta"], provenance,
        )
        bundle.validate()
        horizon_dir = out / domain / f"cutoff_frames_{BASE_CUTOFF_FRAMES + shift:03d}"
        horizon_dir.mkdir(parents=True, exist_ok=True)
        bundle.save(horizon_dir / "bundle.npz")
        if shift == 0:
            old = {meta["sample_id"]: i for i, meta in enumerate(source_bundle.meta)}
            order = np.asarray([old[meta["sample_id"]] for meta in bundle.meta])
            max_a = float(np.max(np.abs(bundle.a - source_bundle.a[order])))
            max_q = float(np.max(np.abs(bundle.q - source_bundle.q[order])))
            if max_a > 1e-6 or max_q > 1e-6 or not np.array_equal(bundle.y, source_bundle.y[order]):
                raise AssertionError(f"Base feature mismatch: a={max_a}, q={max_q}")
        else:
            max_a = max_q = None
        tti = np.asarray(positive_tti[shift], dtype=float)
        record = {
            "cutoff_frames": BASE_CUTOFF_FRAMES + shift,
            "shift_frames": shift,
            "cutoff_seconds": (BASE_CUTOFF_FRAMES + shift) / FPS,
            "n": len(bundle.y),
            "positives": int(bundle.y.sum()),
            "failures": dict(failures[shift]),
            "positive_event_lead_seconds": None if not len(tti) else {
                "min": float(tti.min()), "median": float(np.median(tti)), "max": float(tti.max())
            },
            "bundle": str((horizon_dir / "bundle.npz").relative_to(root)),
            "bundle_sha256": digest_file(horizon_dir / "bundle.npz"),
            "base_feature_max_abs_difference": {"a": max_a, "q": max_q},
        }
        coverage["horizons"].append(record)
    write_json(out / domain / "coverage.json", coverage)
    return coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    raw_root = args.raw_root.resolve()
    out = args.out.resolve()
    protocol_path = args.protocol.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)
    files = recording_files(raw_root)
    coverage = {}
    for domain, relative in TARGETS.items():
        coverage[domain] = export_target(root, raw_root, out, domain, root / relative, files)

    checkpoints = {}
    for direction in ["HUI360_to_SSUP-A", "SSUP-A_to_HUI360"]:
        checkpoints[direction] = {}
        for method in ["geometry", "dual_no_pair"]:
            checkpoints[direction][method] = {}
            for seed in [11, 22, 33, 44, 55]:
                path = root / "outputs/frozen_cross_domain" / direction / method / f"seed{seed}/model/best.pt"
                checkpoints[direction][method][str(seed)] = {
                    "path": str(path.relative_to(root)), "sha256": digest_file(path)
                }
    protocol = {
        "schema": "acie.horizon-evaluation-protocol.v1",
        "date": "2026-09-15",
        "status": "frozen_before_horizon_model_scoring",
        "classification": "post-primary-result E4 follow-up from the teacher-provided study plan",
        "fps": FPS,
        "window_length_frames": 32,
        "base_cutoff_frames": BASE_CUTOFF_FRAMES,
        "horizon_shifts_frames": HORIZON_SHIFTS,
        "cutoff_seconds": [(BASE_CUTOFF_FRAMES + shift) / FPS for shift in HORIZON_SHIFTS],
        "directions": {"HUI360_to_SSUP-A": "SSUP-A", "SSUP-A_to_HUI360": "HUI360"},
        "methods": ["geometry", "dual_no_pair"],
        "seeds": [11, 22, 33, 44, 55],
        "checkpoints": checkpoints,
        "bundles": {
            domain: [{key: value for key, value in row.items() if key in ["cutoff_frames", "shift_frames", "cutoff_seconds", "n", "positives", "bundle", "bundle_sha256"]}
                     for row in report["horizons"]]
            for domain, report in coverage.items()
        },
        "threshold_policy": "each seed reuses its source-development frozen threshold; no target or horizon retuning",
        "reporting": {
            "ranking": "five-seed ensemble AP and AUROC",
            "operating_point": "mean and sample standard deviation of per-seed recall and false-positive rate at each seed's frozen threshold",
            "sets": ["complete available set at each horizon", "common tracks available at all five horizons"],
            "uncertainty": "date-group clustered bootstrap for ensemble AP difference dual_no_pair minus geometry",
        },
        "labels": "unchanged from frozen official target manifests",
        "negative_alignment": "shift the official maximum-mask proxy aligned negative prefix by the same frame offset",
        "target_labels_used_for_selection": False,
        "horizon_choice_note": "chosen after label-free feasibility counts and before any horizon model scores; 4.27 s excluded because HUI360 coverage fell to 100 tracks/18 positives",
        "scope": "central capacity-matched E4 comparison; official architecture horizon extension is separate",
    }
    protocol["protocol_digest"] = digest_object(protocol)
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise ValueError("Existing frozen horizon protocol differs")
    else:
        write_json(protocol_path, protocol)
    write_json(out / "coverage_summary.json", coverage)
    print(json.dumps({"protocol_digest": protocol["protocol_digest"], "coverage": coverage}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
