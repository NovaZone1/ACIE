#!/usr/bin/env python3
"""Validate frozen E4 horizon bundles, predictions, and summary invariants."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from acie.data import Bundle
from acie.io import digest_file, digest_object, read_json, read_jsonl, write_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--outputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve(); outputs = a.outputs.resolve()
    protocol = read_json(a.protocol.resolve()); summary = read_json(outputs / "summary.json")
    checks = []

    def check(name: str, passed: bool, detail=None) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            raise AssertionError(f"{name}: {detail}")

    protocol_payload = dict(protocol); expected_protocol_digest = protocol_payload.pop("protocol_digest")
    check("protocol digest", digest_object(protocol_payload) == expected_protocol_digest)
    summary_payload = dict(summary); expected_summary_digest = summary_payload.pop("summary_digest")
    check("summary digest", digest_object(summary_payload) == expected_summary_digest)
    check("summary/protocol linkage", summary["protocol_digest"] == protocol["protocol_digest"])
    check("frozen target policy", protocol["target_labels_used_for_selection"] is False)
    check("methods and seeds", protocol["methods"] == ["geometry", "dual_no_pair"] and protocol["seeds"] == [11,22,33,44,55])

    old_bundle = Bundle.load(root / "data/official_cross_domain.npz")
    old_index = {m["sample_id"]: i for i, m in enumerate(old_bundle.meta)}
    prediction_files = 0
    expected_base_scores = {}
    for direction, target in protocol["directions"].items():
        horizon_sets = []
        for spec in protocol["bundles"][target]:
            bundle_path = root / spec["bundle"]
            check(f"bundle hash {target}/{spec['cutoff_frames']}", digest_file(bundle_path) == spec["bundle_sha256"])
            bundle = Bundle.load(bundle_path)
            ids = [m["sample_id"] for m in bundle.meta]
            horizon_sets.append(set(ids))
            check(f"bundle counts {target}/{spec['cutoff_frames']}", len(bundle.y) == spec["n"] and int(bundle.y.sum()) == spec["positives"])
            check(f"unique ids {target}/{spec['cutoff_frames']}", len(ids) == len(set(ids)))
            shift = spec["shift_frames"]
            check(f"metadata shift {target}/{spec['cutoff_frames']}", all(m["source_index"]["horizon_shift_frames"] == shift for m in bundle.meta))
            check(f"labels unchanged {target}/{spec['cutoff_frames']}", all(int(bundle.y[i]) == int(old_bundle.y[old_index[sid]]) for i, sid in enumerate(ids)))
            if shift == 0:
                order = np.asarray([old_index[sid] for sid in ids])
                check(f"base features exact {target}", np.array_equal(bundle.a, old_bundle.a[order]) and np.array_equal(bundle.q, old_bundle.q[order]))
            for method in protocol["methods"]:
                for seed in protocol["seeds"]:
                    ck_spec = protocol["checkpoints"][direction][method][str(seed)]
                    checkpoint_path = root / ck_spec["path"]
                    check(f"checkpoint hash {direction}/{method}/{seed}", digest_file(checkpoint_path) == ck_spec["sha256"])
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                    pred_path = outputs / direction / method / f"seed{seed}" / f"cutoff_frames_{spec['cutoff_frames']:03d}.jsonl"
                    rows = read_jsonl(pred_path); prediction_files += 1
                    table = {row["sample_id"]: row for row in rows}
                    check(f"prediction coverage {direction}/{method}/{seed}/{spec['cutoff_frames']}", len(table) == len(rows) and set(table) == set(ids))
                    check(f"threshold frozen {direction}/{method}/{seed}/{spec['cutoff_frames']}", all(float(row["threshold"]) == float(checkpoint["threshold"]) for row in rows))
                    check(f"prediction labels {direction}/{method}/{seed}/{spec['cutoff_frames']}", all(int(table[sid]["label"]) == int(old_bundle.y[old_index[sid]]) for sid in table))
                    if shift == 0:
                        old_path = root / "outputs/frozen_cross_domain" / direction / method / f"seed{seed}/test_predictions.jsonl"
                        old = {row["sample_id"]: row for row in read_jsonl(old_path)}
                        max_score = max(abs(float(table[sid]["score"]) - float(old[sid]["score"])) for sid in table)
                        check(f"base score reproduction {direction}/{method}/{seed}", max_score <= 1e-7, max_score)
                        expected_base_scores[(direction, method, seed)] = max_score
        common = set.intersection(*horizon_sets)
        result = summary["directions"][direction]
        check(f"common count {direction}", result["common_track_count"] == len(common))
        check(f"nested horizon coverage {direction}", all(horizon_sets[i+1].issubset(horizon_sets[i]) for i in range(len(horizon_sets)-1)))
        for horizon in result["horizons"]:
            check(f"positive AP point {direction}/{horizon['cutoff_frames']}", horizon["sets"]["complete"]["dual_no_pair_minus_geometry"]["point"] > 0)

    check("prediction file count", prediction_files == 100, prediction_files)
    report = {
        "schema": "acie.horizon-evaluation-validation.v1",
        "passed": sum(row["passed"] for row in checks),
        "total": len(checks),
        "all_passed": all(row["passed"] for row in checks),
        "prediction_files": prediction_files,
        "max_base_score_difference": max(expected_base_scores.values()),
        "checks": checks,
    }
    write_json(a.out.resolve(), report)
    print(f"PASS {report['passed']}/{report['total']} max_base_score_difference={report['max_base_score_difference']:.3g}")


if __name__ == "__main__":
    main()
