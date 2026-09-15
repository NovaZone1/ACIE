"""Build auditable HUI360 official-baseline configs for the frozen ACIE split.

The generated fair configs select checkpoints on source-domain validation data.
Target-domain configs are evaluation-only.  Exact track lists and the upstream
window-choice indices are carried over from the frozen exported manifests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import yaml


UPSTREAM_COMMIT = "a6bd61c8539061242cbf756a18847c0900a8707c"
DATA_REVISION = "c7bcce944ff451682ee996412917d70720592fbd"
LOADER_REVISION = "main"
EPOCHS = {"mlp": 7}

DIRECTIONS = {
    "HUI360_to_SSUP-A": {
        "source": "HUI360",
        "target": "SSUP-A",
        "source_bundle": "data/hui_source_windows/bundle.json",
        "target_bundle": "data/ssup_official_test_windows/bundle.json",
        "source_provenance": "data/hui_source_windows/export_provenance.json",
    },
    "SSUP-A_to_HUI360": {
        "source": "SSUP-A",
        "target": "HUI360",
        "source_bundle": "data/ssup_source_windows/bundle.json",
        "target_bundle": "data/hui_official_test_windows/bundle.json",
        "source_provenance": "data/ssup_source_windows/export_provenance.json",
    },
}


def load_json(path: Path):
    return json.loads(path.read_text())


def load_bundle(path: Path):
    bundle = load_json(path)
    labels = np.load(path.with_suffix(".npz"), allow_pickle=False)["y"]
    if len(labels) != len(bundle["meta"]):
        raise ValueError(f"Bundle metadata/label mismatch: {path}")
    bundle["meta"] = [{**row, "label": int(label)} for row, label in zip(bundle["meta"], labels)]
    return bundle


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return digest_bytes(payload.encode())


def ordered_subset(meta, ids):
    wanted = set(ids)
    rows = [row for row in meta if row["sample_id"] in wanted]
    found = {row["sample_id"] for row in rows}
    if found != wanted:
        missing = sorted(wanted - found)
        raise ValueError(f"Frozen IDs absent from bundle: {missing[:5]}")
    if len(rows) != len(wanted):
        raise ValueError("Bundle has duplicate sample IDs")
    tracks = [row["track_id"] for row in rows]
    if len(tracks) != len(set(tracks)):
        raise ValueError("Official one-window protocol requires one selected sample per track")
    return rows


def official_loader_order(root: Path, rows):
    """Match HUIDataset's lexicographically sorted CSV order, stably per recording."""
    order = {}
    for index, path in enumerate(sorted((root / "data" / "hui_processed").glob("data-*-????-of-????.csv"))):
        match = re.fullmatch(r"data-(.+)-\d{4}-of-\d{4}\.csv", path.name)
        if match:
            order[match.group(1)] = index
    missing = sorted({row["recording"] for row in rows} - set(order))
    if missing:
        raise ValueError(f"Selected recordings lack verified CSV order entries: {missing}")
    return sorted(rows, key=lambda row: order[row["recording"]])


def ordered_unique(rows, key):
    return list(dict.fromkeys(row[key] for row in rows))


def write_tracks(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{row['track_id']}\n" for row in rows))


def yaml_bytes(config) -> bytes:
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True).encode()


def make_config(base, *, rows_train, rows_val, train_file, val_file, direction, role):
    cfg = dict(base)
    cfg.update(
        force_model_type="mlp",
        epochs=EPOCHS["mlp"],
        train_tracks_filename=train_file,
        val_tracks_filename=val_file,
        test_tracks_filename=val_file,
        include_recordings_train=ordered_unique(rows_train, "recording"),
        include_recordings_val=ordered_unique(rows_val, "recording"),
        fix_index_per_track_train=True,
        fix_index_per_track_val=True,
        fix_index_per_track_list_train=[int(r["source_index"]["choice"]) for r in rows_train],
        fix_index_per_track_list_val=[int(r["source_index"]["choice"]) for r in rows_val],
        # This upstream loader only accepts the symbolic value "main" in offline
        # mode.  PROTOCOL.json separately pins the verified content commit.
        hf_dataset_revision=LOADER_REVISION,
        cross_eval_type=f"acie_{role}_{direction.lower().replace('-', '_')}",
        experiment_name=f"acie_{role}_{direction.lower().replace('-', '_')}_mlp_seed42",
        comment=(
            "ACIE official baseline protocol v1; exact frozen tracks/windows; "
            + ("source-only checkpoint selection" if role == "fair" else "official-native target validation")
        ),
    )
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    out = (args.out or root / "official_baselines" / "protocol_v1").resolve()
    upstream = root / "external" / "HUI360-Baselines"
    tracks_dir = upstream / "datasets" / "tracks_saved_identifiers"
    upstream_head = __import__("subprocess").check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if upstream_head != UPSTREAM_COMMIT:
        raise ValueError(f"Unexpected upstream commit: {upstream_head}")

    protocol = {
        "schema": "acie.official-baseline-protocol.v1",
        "upstream_commit": UPSTREAM_COMMIT,
        "data_revision": DATA_REVISION,
        "offline_loader_revision_argument": LOADER_REVISION,
        "official_seed": 42,
        "primary_metrics": ["AP", "AUROC"],
        "checkpoint_selection": "maximum source-validation AP",
        "target_policy": "one evaluation after checkpoint selection; no target threshold tuning",
        "native_results_policy": "reported separately because target labels select checkpoints",
        "directions": {},
    }

    for direction, spec in DIRECTIONS.items():
        split = load_json(root / "outputs" / "frozen_cross_domain" / direction / "split.json")
        source = load_bundle(root / spec["source_bundle"])
        target = load_bundle(root / spec["target_bundle"])
        provenance = load_json(root / spec["source_provenance"])
        if provenance["upstream_commit"] != UPSTREAM_COMMIT:
            raise ValueError(f"Provenance commit mismatch for {direction}")
        base = provenance.get("resolved_upstream_config", provenance.get("upstream_config"))
        source_meta = source["meta"]
        target_meta = target["meta"]
        train_rows = official_loader_order(root, ordered_subset(source_meta, split["train"]))
        val_rows = official_loader_order(root, ordered_subset(source_meta, split["val"]))
        test_rows = official_loader_order(root, ordered_subset(target_meta, split["test"]))
        all_source_rows = official_loader_order(
            root, ordered_subset(source_meta, [r["sample_id"] for r in source_meta])
        )

        sets = [set(r["track_id"] for r in rows) for rows in (train_rows, val_rows, test_rows)]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise ValueError(f"Track leakage in {direction}")

        slug = direction.lower().replace("-", "_")
        names = {
            "train": f"acie_v1_{slug}_source_train.txt",
            "val": f"acie_v1_{slug}_source_val.txt",
            "test": f"acie_v1_{slug}_target_test.txt",
            "source_all": f"acie_v1_{slug}_source_all.txt",
        }
        for key, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows), ("source_all", all_source_rows)):
            write_tracks(tracks_dir / names[key], rows)

        fair = make_config(
            base, rows_train=train_rows, rows_val=val_rows,
            train_file=names["train"], val_file=names["val"], direction=direction, role="fair",
        )
        target_eval = make_config(
            base, rows_train=train_rows, rows_val=test_rows,
            train_file=names["train"], val_file=names["test"], direction=direction, role="fair_target_eval",
        )
        native = make_config(
            base, rows_train=all_source_rows, rows_val=test_rows,
            train_file=names["source_all"], val_file=names["test"], direction=direction, role="native_available_subset",
        )

        cfg_dir = out / direction
        cfg_dir.mkdir(parents=True, exist_ok=True)
        configs = {}
        for name, cfg in (("fair_train_mlp", fair), ("fair_target_eval_mlp", target_eval), ("native_available_subset_mlp", native)):
            path = cfg_dir / f"{name}.yaml"
            payload = yaml_bytes(cfg)
            path.write_bytes(payload)
            configs[name] = {"path": str(path.relative_to(root)), "sha256": digest_bytes(payload)}

        def describe(rows):
            return {
                "samples": len(rows),
                "positives": sum(int(r.get("label", 0)) for r in rows),
                "recordings": ordered_unique(rows, "recording"),
                "tracks_sha256": digest_json([r["track_id"] for r in rows]),
                "windows_sha256": digest_json([
                    [r["sample_id"], r["track_id"], r["source_index"]["choice"], r["source_index"]["frame_indexes"]]
                    for r in rows
                ]),
            }

        protocol["directions"][direction] = {
            "source": spec["source"],
            "target": spec["target"],
            "fair": {"train": describe(train_rows), "source_val": describe(val_rows), "target_test": describe(test_rows)},
            "native_available_subset": {"train": describe(all_source_rows), "target_val": describe(test_rows)},
            "configs": configs,
            "track_files": names,
        }

    out.mkdir(parents=True, exist_ok=True)
    path = out / "PROTOCOL.json"
    path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n")
    print(path)
    for direction, item in protocol["directions"].items():
        fair = item["fair"]
        print(direction, {k: v["samples"] for k, v in fair.items()})


if __name__ == "__main__":
    main()
