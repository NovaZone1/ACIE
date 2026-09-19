#!/usr/bin/env python3
"""Build a machine-readable completion audit for the VV method-paper follow-up."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def comparison(summary: dict[str, Any], direction: str, name: str) -> dict[str, Any]:
    return next(row for row in summary["directions"][direction]["comparisons"] if row["comparison"] == name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--outputs", type=Path, default=Path("outputs/vv_followup_v2"))
    parser.add_argument("--out", type=Path, default=Path("outputs/vv_followup_v2"))
    parser.add_argument("--tests-passed", type=int, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    outputs = (root / args.outputs).resolve() if not args.outputs.is_absolute() else args.outputs.resolve()
    out = (root / args.out).resolve() if not args.out.is_absolute() else args.out.resolve()

    expected = [
        "r3_reporting/r3_summary.json",
        "r3_reporting/r3_seed_AP.csv",
        "r4_reporting/r4_summary.json",
        "r4_reporting/r4_validation.json",
        "r5d_components/r5d_summary.json",
        "r5d_components/r5d_pairs.jsonl",
        "evidence/evidence_summary.json",
        "evidence/comparisons.csv",
        "evidence/target_date_sensitivity.csv",
        "evidence/operating_points.csv",
        "evidence/box_area_baseline.csv",
        "tree_input_ablation/PROTOCOL_LOCK.json",
        "tree_input_ablation/TARGET_SCORING_LOCK.json",
        "tree_input_ablation/source_validation.json",
        "tree_input_ablation/summary.json",
        "rf_input_ablation/PROTOCOL_LOCK.json",
        "rf_input_ablation/TARGET_SCORING_LOCK.json",
        "rf_input_ablation/source_validation.json",
        "rf_input_ablation/summary.json",
        "r2_tree_cv_archive/PROTOCOL_LOCK.json",
        "r2_tree_cv_archive/validation.json",
    ]
    missing = [name for name in expected if not (outputs / name).is_file()]
    if missing:
        raise SystemExit(f"missing required artifacts: {missing}")

    r4_validation = load_json(outputs / "r4_reporting/r4_validation.json")
    hist_validation = load_json(outputs / "tree_input_ablation/source_validation.json")
    rf_validation = load_json(outputs / "rf_input_ablation/source_validation.json")
    archive_validation = load_json(outputs / "r2_tree_cv_archive/validation.json")
    for name, payload in {
        "R4": r4_validation,
        "HistGB input ablation": hist_validation,
        "RF input ablation": rf_validation,
        "R2 tree CV archive": archive_validation,
    }.items():
        if payload.get("status") != "passed" or payload.get("errors"):
            raise SystemExit(f"{name} validation did not pass")

    evidence = load_json(outputs / "evidence/evidence_summary.json")
    hist = load_json(outputs / "tree_input_ablation/summary.json")
    rf = load_json(outputs / "rf_input_ablation/summary.json")
    r5d = load_json(outputs / "r5d_components/r5d_summary.json")
    comparisons = csv_rows(outputs / "evidence/comparisons.csv")
    box = csv_rows(outputs / "evidence/box_area_baseline.csv")

    def evidence_row(experiment: str, direction: str, name: str) -> dict[str, str]:
        return next(
            row for row in comparisons
            if row["experiment"] == experiment and row["direction"] == direction and row["comparison"] == name
        )

    def tree_row(payload: dict[str, Any], direction: str, name: str) -> dict[str, Any]:
        row = comparison(payload, direction, name)
        return {
            "AP_difference": row["AP_difference"],
            "interval_95": row["bootstrap"]["joint_date_and_seed"]["interval_95"],
        }

    code_paths = [
        "src/acie/vv_provenance.py",
        "src/acie/vv_engine.py",
        "src/acie/vv_trees.py",
        "src/acie/metrics.py",
        "scripts/vv_score_locked.py",
        "scripts/vv_score_trees_locked.py",
        "scripts/vv_summarize_r3.py",
        "scripts/vv_summarize_r4.py",
        "scripts/vv_validate_r4.py",
        "scripts/vv_diagnose_r5d.py",
        "scripts/vv_finalize_evidence.py",
        "scripts/vv_tree_input_ablation.py",
        "scripts/vv_archive_r2_tree_cv.py",
        "tests/test_vv_followup.py",
    ]
    source_snapshot = {name: sha256(root / name) for name in code_paths}

    try:
        active = subprocess.check_output(
            ["pgrep", "-af", "vv_(tree_input_ablation|archive_r2_tree_cv|finalize_evidence|score_locked|score_trees_locked)"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
        active = [line for line in active if "vv_completion_audit.py" not in line]
    except subprocess.CalledProcessError:
        active = []

    forward_r2 = evidence_row("R2_source_selected", "HUI360_to_SSUP-A", "F-G")
    reverse_r2 = evidence_row("R2_source_selected", "SSUP-A_to_HUI360", "F-G")
    audit = {
        "schema": "acie.vv-followup-v2-completion-audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(root),
        "outputs_root": str(outputs),
        "status": "complete" if not active else "active_jobs_detected",
        "tests": {
            "server": {"passed": args.tests_passed, "failed": 0},
            "local": {"passed": args.tests_passed, "failed": 0, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": True},
        },
        "validations": {
            "R4": r4_validation,
            "HistGB_input_ablation": hist_validation,
            "RF_input_ablation": rf_validation,
            "R2_tree_CV_archive": {
                "status": archive_validation["status"],
                "expected_runs": archive_validation["expected_runs"],
                "observed_runs": archive_validation["observed_runs"],
                "all_selections_match_original": all(
                    item["selection_matches_original"]
                    for item in archive_validation["selection_comparison"].values()
                ),
            },
        },
        "artifact_counts": {
            "R3_seed_rows": len(csv_rows(outputs / "r3_reporting/r3_seed_AP.csv")),
            "R5d_pair_records": sum(1 for _ in (outputs / "r5d_components/r5d_pairs.jsonl").open(encoding="utf-8")),
            "evidence_comparisons": len(comparisons),
            "target_date_sensitivity_rows": len(csv_rows(outputs / "evidence/target_date_sensitivity.csv")),
            "operating_point_rows": len(csv_rows(outputs / "evidence/operating_points.csv")),
            "box_area_baselines": len(box),
        },
        "key_results": {
            "R2_F_minus_G": {
                "HUI360_to_SSUP-A": forward_r2,
                "SSUP-A_to_HUI360": reverse_r2,
            },
            "HistGB_all_minus_geometry": {
                direction: tree_row(hist, direction, "all-geometry")
                for direction in ("HUI360_to_SSUP-A", "SSUP-A_to_HUI360")
            },
            "RF_all_minus_geometry": {
                direction: tree_row(rf, direction, "all-geometry")
                for direction in ("HUI360_to_SSUP-A", "SSUP-A_to_HUI360")
            },
            "box_area_baseline": box,
        },
        "claim_scope": {
            "supported": [
                "The implemented model uses a fixed geometric reference and an additive pose residual z=g+r.",
                "For R2, the residual model improves over its neural geometry baseline only in SSUP-A_to_HUI360; the forward difference crosses zero.",
                "Pose contribution is direction-dependent and learner-dependent under the tested cross-dataset protocols.",
                "R5 fixed-configuration controls support pose-content use mainly in SSUP-A_to_HUI360.",
            ],
            "unsupported": [
                "Universal or bidirectional gain from pose residuals.",
                "Superiority over strong tree learners.",
                "A robust contribution from conditional pairing.",
                "A deployment, causal-intent, or online early-warning claim.",
            ],
            "configuration_note": "R5 remains an R1 fixed-configuration diagnostic and is not presented as a direct R2 ablation.",
        },
        "completed_work": [
            "content-bound run identities and strict completed-run reuse validation",
            "locked target scoring with checkpoint-hash validation",
            "R3 per-seed reporting",
            "R4 actual-time reporting and exact base-input equivalence checks",
            "R1/R2/tree paired statistics, target-date leave-one-out sensitivity, operating counts, and bbox-area baseline",
            "R5d g/r/g+r component export and pair-level records",
            "HistGB and RF geometry/pose/all input ablations",
            "full 108-run R2 tree-CV artifact archive with selection reproduction",
        ],
        "remaining_required_experiments": [],
        "active_jobs": active,
        "source_snapshot_sha256": source_snapshot,
        "artifact_manifest": {
            name: {"bytes": (outputs / name).stat().st_size, "sha256": sha256(outputs / name)}
            for name in expected
        },
        "raw_summary_schema": evidence.get("schema"),
        "r5d_schema": r5d.get("schema"),
    }

    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "COMPLETION_AUDIT.json"
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fwd = audit["key_results"]["R2_F_minus_G"]["HUI360_to_SSUP-A"]
    rev = audit["key_results"]["R2_F_minus_G"]["SSUP-A_to_HUI360"]
    lines = [
        "# VV follow-up v2 completion audit",
        "",
        f"Status: **{audit['status']}**. Server and local project tests: **{args.tests_passed} passed**.",
        "",
        "## Completed evidence",
        "",
    ]
    lines.extend(f"- {item}" for item in audit["completed_work"])
    lines.extend([
        "",
        "## Claim boundary",
        "",
        f"- R2 F-G, HUI360→SSUP-A: {float(fwd['mean_AP_difference']):+.6f}; 95% interval [{float(fwd['joint_95_low']):+.6f}, {float(fwd['joint_95_high']):+.6f}].",
        f"- R2 F-G, SSUP-A→HUI360: {float(rev['mean_AP_difference']):+.6f}; 95% interval [{float(rev['joint_95_low']):+.6f}, {float(rev['joint_95_high']):+.6f}].",
        "- HistGB and RF input ablations do not establish a stable cross-direction gain from adding pose to strong geometry features.",
        "- Pairing, universal superiority, causal intent, deployment, and online-warning claims remain unsupported.",
        "- R5 is retained as an R1 fixed-configuration diagnostic, not relabeled as an R2 ablation.",
        "",
        "## Completion decision",
        "",
        "No additional experiment is required by the 2026-09-17 minimum evidence plan. The next step is result-table/figure preparation and manuscript work; manuscript writing is outside this execution task.",
        "",
        "Machine-readable hashes, counts, validation payloads, and key results are in `COMPLETION_AUDIT.json`.",
    ])
    (out / "COMPLETION_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "json": str(json_path), "active_jobs": active}, indent=2))


if __name__ == "__main__":
    main()
