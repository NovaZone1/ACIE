"""Verify that an official loader config reproduces frozen ACIE tracks/windows."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml


def digest_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--direction", choices=["HUI360_to_SSUP-A", "SSUP-A_to_HUI360"], required=True)
    parser.add_argument("--expected-part", choices=["train", "val", "test"], required=True)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    upstream = root / "external" / "HUI360-Baselines"
    sys.path.insert(0, str(upstream))
    from utils.loader_utils import load_hui_dataset  # noqa: PLC0415
    from utils.debug_utils import update_old_config_dict  # noqa: PLC0415

    source_bundle = "hui_source_windows" if args.direction == "HUI360_to_SSUP-A" else "ssup_source_windows"
    target_bundle = "ssup_official_test_windows" if args.direction == "HUI360_to_SSUP-A" else "hui_official_test_windows"
    bundle_name = target_bundle if args.expected_part == "test" else source_bundle
    manifest_path = root / "data" / bundle_name / "manifest.jsonl"
    metadata = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    config = update_old_config_dict(yaml.safe_load(args.config.read_text()))
    frozen = json.loads((root / "outputs" / "frozen_cross_domain" / args.direction / "split.json").read_text())
    wanted = set(frozen[args.expected_part])
    expected_by_track = {row["track_id"]: row for row in metadata if row["sample_id"] in wanted}
    tracks_name = config[f"{args.split}_tracks_filename"]
    tracks_path = upstream / "datasets" / "tracks_saved_identifiers" / tracks_name
    configured_tracks = [line.strip() for line in tracks_path.read_text().splitlines() if line.strip()]
    expected = [expected_by_track[track] for track in configured_tracks]
    if len(expected) != len(wanted):
        raise ValueError("Frozen samples are missing from the expected bundle")

    loader_args = SimpleNamespace(
        preload_data=True, preload_only=False, verbose=False,
        hf_local_dir=str(args.data_root.resolve()), offline_mode=True,
    )
    dataset = load_hui_dataset(loader_args, config, split=args.split, num_workers=args.num_workers)
    if len(dataset) != len(expected):
        raise ValueError(f"Sample count mismatch: {len(dataset)} != {len(expected)}")

    observed = []
    failures = []
    for index, row in enumerate(expected):
        _, label, metadata, *_ = dataset[index]
        track = metadata["unique_track_identifier"]
        frame_indexes = [int(v) for v in metadata["image_indexes"]]
        item = [row["sample_id"], track, int(metadata["index_choice"]), frame_indexes]
        observed.append(item)
        if track != row["track_id"]:
            failures.append({"index": index, "field": "track_id", "got": track, "expected": row["track_id"]})
        if int(label) != int(row["label"]):
            failures.append({"index": index, "field": "label", "got": int(label), "expected": int(row["label"])})
        if int(metadata["index_choice"]) != int(row["source_index"]["choice"]):
            failures.append({"index": index, "field": "choice"})
        if frame_indexes != [int(v) for v in row["source_index"]["frame_indexes"]]:
            failures.append({"index": index, "field": "frame_indexes"})
        if len(failures) >= 10:
            break

    expected_windows = [
        [r["sample_id"], r["track_id"], int(r["source_index"]["choice"]), [int(v) for v in r["source_index"]["frame_indexes"]]]
        for r in expected
    ]
    report = {
        "schema": "acie.official-loader-scope-check.v1",
        "direction": args.direction,
        "loader_split": args.split,
        "expected_part": args.expected_part,
        "samples": len(expected),
        "positives": sum(int(r["label"]) for r in expected),
        "exact_track_label_choice_and_frames": not failures and observed == expected_windows,
        "expected_windows_sha256": digest_json(expected_windows),
        "observed_windows_sha256": digest_json(observed) if len(observed) == len(expected) else None,
        "failures": failures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures or observed != expected_windows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
