#!/usr/bin/env python3
"""Validate the frozen behavior-information permutation control."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from acie.data import Bundle, group_key, resolve_split
from acie.io import digest_file, digest_object, read_json, read_jsonl, write_json


DIRECTIONS = ["HUI360_to_SSUP-A", "SSUP-A_to_HUI360"]
SEEDS = [11, 22, 33, 44, 55]


def table(path: Path) -> tuple[list[dict], dict[str, dict]]:
    rows = read_jsonl(path)
    return rows, {row["sample_id"]: row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    protocol = read_json(args.protocol)
    summary = read_json(args.runs / "summary.json")
    bundle = Bundle.load(args.data)
    checks = []
    errors = []

    def check(name: str, condition: bool, detail: object) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})
        if not condition:
            errors.append(name)

    digest_payload = dict(protocol)
    recorded_digest = digest_payload.pop("protocol_digest", None)
    check(
        "protocol identity and digest",
        protocol.get("schema") == "acie.behavior-information-control-protocol.v1"
        and protocol.get("seeds") == SEEDS
        and digest_object(digest_payload) == recorded_digest
        and digest_file(args.data) == protocol.get("data_sha256"),
        {"digest": recorded_digest, "evidence_class": protocol.get("evidence_class")},
    )
    check(
        "summary identity",
        summary.get("schema") == "acie.behavior-information-control-summary.v1"
        and summary.get("protocol_digest") == recorded_digest
        and summary.get("seeds") == SEEDS,
        {key: summary.get(key) for key in ("schema", "protocol_digest", "seeds")},
    )

    total_predictions = 0
    all_supported = True
    for direction in DIRECTIONS:
        spec = protocol["directions"][direction]
        split = read_json(root / spec["split"])
        indices = resolve_split(bundle, split)
        mapping_path = root / spec["mapping"]
        mappings = read_jsonl(mapping_path)
        expected_ids = {part: {bundle.meta[i]["sample_id"] for i in ix} for part, ix in indices.items()}
        recipients = {part: set() for part in indices}
        donors_by_scope: dict[tuple[str, str], set[str]] = {}
        recipients_by_scope: dict[tuple[str, str], set[str]] = {}
        mapping_valid = True
        for row in mappings:
            part = row["part"]
            recipient = row["recipient_sample_id"]
            donor = row["donor_sample_id"]
            scope = (part, row["group_id"])
            mapping_valid &= recipient in expected_ids[part] and donor in expected_ids[part] and recipient != donor
            recipients[part].add(recipient)
            recipients_by_scope.setdefault(scope, set()).add(recipient)
            donors_by_scope.setdefault(scope, set()).add(donor)
        mapping_valid &= all(recipients[part] == expected_ids[part] for part in indices)
        mapping_valid &= all(recipients_by_scope[scope] == donors_by_scope.get(scope) for scope in recipients_by_scope)
        check(f"{direction}: frozen mapping hash", digest_file(mapping_path) == spec["mapping_sha256"], spec["mapping_sha256"])
        check(f"{direction}: mapping is exact within-split/group derangement", mapping_valid and len(mappings) == spec["mapped_samples"], {"rows": len(mappings), "parts": {part: len(ids) for part, ids in recipients.items()}})

        control_tables = []
        prediction_hashes = []
        for seed in SEEDS:
            run = args.runs / direction / "permuted_behavior" / f"seed{seed}"
            prediction_path = run / "test_predictions.jsonl"
            rows, predictions = table(prediction_path)
            control_tables.append(predictions)
            prediction_hashes.append(digest_file(prediction_path))
            total_predictions += len(rows)
            metrics = read_json(prediction_path.with_suffix(".metrics.json"))["metrics"]
            provenance = read_json(run / "run_provenance.json")
            training = read_json(run / "model" / "training_report.json")
            labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
            scores = np.asarray([float(row["score"]) for row in rows], dtype=float)
            expected_test = expected_ids["test"]
            check(f"{direction}/seed{seed}: complete artifacts", (run / "model" / "best.pt").is_file() and training.get("status") == "complete", {"source_val_AP": training.get("source_val", {}).get("AP")})
            check(f"{direction}/seed{seed}: identity and source-only selection", provenance.get("direction") == direction and provenance.get("seed") == seed and provenance.get("mapping_sha256") == spec["mapping_sha256"] and provenance.get("target_labels_used_for_selection") is False, provenance)
            check(f"{direction}/seed{seed}: exact target IDs", len(predictions) == len(rows) and set(predictions) == expected_test, {"rows": len(rows), "unique": len(predictions)})
            calculated = (float(average_precision_score(labels, scores)), float(roc_auc_score(labels, scores)))
            check(f"{direction}/seed{seed}: metrics", np.allclose(calculated, [metrics["AP"], metrics["AUROC"]], rtol=0, atol=1e-12) and np.isfinite(scores).all(), {"calculated": calculated, "recorded": [metrics["AP"], metrics["AUROC"]]})
            real_rows, real = table(args.frozen_root / direction / "dual_no_pair" / f"seed{seed}" / "test_predictions.jsonl")
            geometry_same = len(real_rows) == len(rows) and set(real) == set(predictions) and all(abs(float(real[sample_id]["geometry_logit"]) - float(predictions[sample_id]["geometry_logit"])) <= 1e-7 for sample_id in predictions)
            check(f"{direction}/seed{seed}: geometry branch invariant", geometry_same, {"rows": len(rows), "tolerance": 1e-7})

        ids = sorted(control_tables[0])
        aligned = all(sorted(control) == ids for control in control_tables)
        check(f"{direction}: seeds align and differ", aligned and len(set(prediction_hashes)) == len(SEEDS), prediction_hashes)
        labels = np.asarray([int(control_tables[0][sample_id]["label"]) for sample_id in ids], dtype=int)
        control_scores = np.asarray([[float(control[sample_id]["score"]) for sample_id in ids] for control in control_tables]).mean(axis=0)
        direction_summary = summary["directions"][direction]
        check(f"{direction}: ensemble metrics", np.allclose([average_precision_score(labels, control_scores), roc_auc_score(labels, control_scores)], [direction_summary["ensemble"]["AP"], direction_summary["ensemble"]["AUROC"]], rtol=0, atol=1e-12), direction_summary["ensemble"])
        ensemble_path = root / direction_summary["ensemble"]["predictions"]
        check(f"{direction}: ensemble artifact hash", digest_file(ensemble_path) == direction_summary["ensemble"]["predictions_sha256"], direction_summary["ensemble"]["predictions_sha256"])

        real_tables = [table(args.frozen_root / direction / "dual_no_pair" / f"seed{seed}" / "test_predictions.jsonl")[1] for seed in SEEDS]
        real_scores = np.asarray([[float(real[sample_id]["score"]) for sample_id in ids] for real in real_tables]).mean(axis=0)
        primary_point = float(average_precision_score(labels, real_scores) - average_precision_score(labels, control_scores))
        recorded = direction_summary["comparisons"]["real_dual_minus_permuted_behavior"]
        direction_supported = recorded["point"] > 0 and recorded["interval_95pct"][0] > 0
        all_supported &= direction_supported
        check(f"{direction}: primary comparison", math.isclose(primary_point, recorded["point"], rel_tol=0, abs_tol=1e-12) and direction_supported == direction_summary["behavior_association_supported"], {"point": primary_point, "interval": recorded["interval_95pct"], "supported": direction_supported})

    check("support conclusion matches rule", bool(all_supported) == summary.get("behavior_association_supported_both_directions"), {"calculated": bool(all_supported), "recorded": summary.get("behavior_association_supported_both_directions")})
    report = {
        "schema": "acie.behavior-information-control-validation.v1",
        "status": "passed" if not errors else "failed",
        "models": 10,
        "prediction_rows": total_predictions,
        "checks_passed": sum(row["passed"] for row in checks),
        "checks_total": len(checks),
        "errors": errors,
        "checks": checks,
    }
    write_json(args.out, report)
    print(json.dumps({key: report[key] for key in ("status", "models", "prediction_rows", "checks_passed", "checks_total", "errors")}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
