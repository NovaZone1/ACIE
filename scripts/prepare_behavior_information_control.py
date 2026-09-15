#!/usr/bin/env python3
"""Freeze label-blind within-group behavior permutations for the follow-up control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from acie.data import Bundle, group_key, resolve_split
from acie.io import digest_file, digest_object, read_json, write_json, write_jsonl


DIRECTIONS = ["HUI360_to_SSUP-A", "SSUP-A_to_HUI360"]


def stable_seed(base: int, *parts: str) -> int:
    payload = ":".join([str(base), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def sattolo(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a single-cycle permutation, hence no fixed points for n > 1."""
    result = values.copy()
    for index in range(len(result) - 1, 0, -1):
        other = int(rng.integers(0, index))
        result[index], result[other] = result[other], result[index]
    return result


def make_mapping(bundle: Bundle, split: dict, direction: str, base_seed: int) -> list[dict]:
    indices = resolve_split(bundle, split)
    rows = []
    for part in ("train", "val", "test"):
        by_group: dict[str, list[int]] = {}
        for index in indices[part]:
            by_group.setdefault(group_key(bundle.meta[index]), []).append(int(index))
        for group in sorted(by_group):
            recipients = np.asarray(
                sorted(by_group[group], key=lambda i: bundle.meta[i]["sample_id"]), dtype=int
            )
            if len(recipients) < 2:
                raise ValueError(f"Cannot derange singleton group {direction}/{part}/{group}")
            donors = sattolo(
                recipients,
                np.random.default_rng(stable_seed(base_seed, direction, part, group)),
            )
            if np.any(recipients == donors) or set(recipients) != set(donors):
                raise AssertionError("Invalid behavior derangement")
            for recipient, donor in zip(recipients, donors, strict=True):
                rows.append({
                    "direction": direction,
                    "part": part,
                    "group_id": group,
                    "recipient_sample_id": bundle.meta[recipient]["sample_id"],
                    "donor_sample_id": bundle.meta[donor]["sample_id"],
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mappings", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--permutation-seed", type=int, default=20260915)
    args = parser.parse_args()

    bundle = Bundle.load(args.data)
    args.mappings.mkdir(parents=True, exist_ok=True)
    direction_payload = {}
    for direction in DIRECTIONS:
        split_path = args.frozen_root / direction / "split.json"
        split = read_json(split_path)
        rows = make_mapping(bundle, split, direction, args.permutation_seed)
        mapping_path = args.mappings / f"{direction}.jsonl"
        expected = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        if mapping_path.exists() and mapping_path.read_text(encoding="utf-8") != expected:
            raise ValueError(f"Existing frozen mapping differs: {mapping_path}")
        write_jsonl(mapping_path, rows)
        direction_payload[direction] = {
            "split": str(split_path),
            "split_sha256": digest_file(split_path),
            "mapping": str(mapping_path),
            "mapping_sha256": digest_file(mapping_path),
            "mapped_samples": len(rows),
            "parts": {part: sum(row["part"] == part for row in rows) for part in ("train", "val", "test")},
        }

    protocol = {
        "schema": "acie.behavior-information-control-protocol.v1",
        "date": "2026-09-15",
        "status": "frozen_before_control_training_and_target_scoring",
        "evidence_class": "post-primary-result mechanism follow-up; not part of the original preregistration",
        "motivation": "Test whether the dual_no_pair gain requires the correct sample-to-behavior association rather than branch capacity.",
        "data": str(args.data),
        "data_sha256": digest_file(args.data),
        "config": str(args.config),
        "config_sha256": digest_file(args.config),
        "model": "dual_no_pair",
        "seeds": [11, 22, 33, 44, 55],
        "permutation_seed": args.permutation_seed,
        "transformation": {
            "unit": "entire T x 119 behavior tensor",
            "scope": "separately within each direction, split, and date group",
            "algorithm": "deterministic Sattolo single-cycle permutation",
            "fixed_points": 0,
            "label_used": False,
            "same_mapping_for_all_training_seeds": True,
            "geometry_features_changed": False,
            "labels_changed": False,
        },
        "primary_metric": "AP",
        "secondary_metric": "AUROC",
        "aggregation": "arithmetic mean of five seed probabilities",
        "uncertainty": "2000-replicate target date-group bootstrap",
        "primary_comparison": "real dual_no_pair ensemble AP minus permuted-behavior ensemble AP",
        "support_rule": "The behavior-association mechanism is supported only if the primary AP difference and its 95% group-bootstrap lower bound are positive in both directions.",
        "directions": direction_payload,
    }
    protocol["protocol_digest"] = digest_object(protocol)
    if args.out.exists():
        existing = read_json(args.out)
        if existing != protocol:
            raise ValueError(f"Existing frozen protocol differs: {args.out}")
    else:
        write_json(args.out, protocol)
    print(json.dumps({"protocol": str(args.out), "digest": protocol["protocol_digest"], "directions": direction_payload}, indent=2))


if __name__ == "__main__":
    main()
