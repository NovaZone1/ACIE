#!/usr/bin/env python3
"""Validate the two-direction, five-seed official MLP fair comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


SEEDS = [11, 22, 33, 44, 55]
METHODS = ["geometry", "dual_no_pair", "weighted_no_pair", "full_selected"]
DIRECTIONS = {
    "HUI360_to_SSUP-A": {
        "scope_files": ("hui_train_scope.json", "hui_val_scope.json", "ssup_target_scope.json"),
        "parts": ("train", "source_val", "target_test"),
    },
    "SSUP-A_to_HUI360": {
        "scope_files": ("ssup_train_scope.json", "ssup_val_scope.json", "hui_target_scope.json"),
        "parts": ("train", "source_val", "target_test"),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out or root / "validation" / "official_fair_mlp_validation.json"
    protocol = load_json(root / "official_baselines" / "protocol_v1" / "PROTOCOL.json")
    summary = load_json(root / "outputs" / "official_baselines" / "fair_v1" / "five_seed_summary.json")
    checks: list[dict] = []
    errors: list[str] = []

    def check(name: str, condition: bool, detail: object) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})
        if not condition:
            errors.append(name)

    check("summary schema and seeds", summary.get("schema") == "acie.official-fair-mlp-five-seed-summary.v1" and summary.get("seeds") == SEEDS, {"schema": summary.get("schema"), "seeds": summary.get("seeds")})
    total_predictions = 0
    for direction, mapping in DIRECTIONS.items():
        direction_root = root / "outputs" / "official_baselines" / "fair_v1" / direction / "mlp"
        target = protocol["directions"][direction]["fair"]["target_test"]
        tables = []
        prediction_hashes = []

        for seed in SEEDS:
            result_dir = direction_root / f"seed{seed}"
            metrics_path = result_dir / "metrics.json"
            predictions_path = result_dir / "predictions.jsonl"
            checkpoint_path = result_dir / "checkpoint_source_best_ap.pth"
            provenance_path = result_dir / "run_provenance.json"
            metrics = load_json(metrics_path)
            provenance = load_json(provenance_path)
            rows = load_jsonl(predictions_path)
            total_predictions += len(rows)
            ids = [row["sample_id"] for row in rows]
            labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
            scores = np.asarray([float(row["score"]) for row in rows], dtype=float)
            table = {row["sample_id"]: row for row in rows}
            tables.append(table)
            prediction_hashes.append(metrics.get("predictions_sha256"))

            check(f"{direction}/seed{seed}: identity", metrics.get("schema") == "acie.official-fair-result.v1" and metrics.get("direction") == direction and metrics.get("model") == "mlp" and metrics.get("seed") == seed, {key: metrics.get(key) for key in ("schema", "direction", "model", "seed")})
            check(f"{direction}/seed{seed}: source-only selection", metrics.get("target_labels_used_for_selection") is False and metrics.get("checkpoint_selection") == "best source-validation AP", {"target_labels_used_for_selection": metrics.get("target_labels_used_for_selection"), "checkpoint_selection": metrics.get("checkpoint_selection")})
            check(f"{direction}/seed{seed}: finite metrics", all(math.isfinite(float(metrics[key])) for key in ("AP", "AUROC")), {"AP": metrics.get("AP"), "AUROC": metrics.get("AUROC")})
            check(f"{direction}/seed{seed}: artifact hashes", sha256(checkpoint_path) == metrics.get("checkpoint_sha256") and sha256(predictions_path) == metrics.get("predictions_sha256"), {"checkpoint": metrics.get("checkpoint_sha256"), "predictions": metrics.get("predictions_sha256")})
            check(f"{direction}/seed{seed}: count and labels", len(rows) == metrics.get("n_test") == target["samples"] and int(labels.sum()) == metrics.get("test_positives") == target["positives"], {"rows": len(rows), "positives": int(labels.sum())})
            check(f"{direction}/seed{seed}: predictions valid", len(ids) == len(set(ids)) and set(labels.tolist()) <= {0, 1} and np.isfinite(scores).all() and ((0 <= scores) & (scores <= 1)).all(), {"unique_ids": len(set(ids)), "score_min": float(scores.min()), "score_max": float(scores.max())})
            check(f"{direction}/seed{seed}: seed provenance", provenance.get("schema") == "acie.official-seed-run.v1" and provenance.get("direction") == direction and provenance.get("seed") == seed and provenance.get("target_labels_used_for_selection") is False, {"seed": provenance.get("seed"), "seed_control": provenance.get("seed_control")})

        ids = sorted(tables[0])
        check(f"{direction}: IDs align across seeds", all(sorted(table) == ids and len(table) == len(ids) for table in tables), len(ids))
        check(f"{direction}: seeds change predictions", len(set(prediction_hashes)) == len(SEEDS), prediction_hashes)
        labels = np.asarray([int(tables[0][sample_id]["label"]) for sample_id in ids], dtype=int)
        score_matrix = np.asarray([[float(table[sample_id]["score"]) for sample_id in ids] for table in tables])
        ensemble_scores = score_matrix.mean(axis=0)
        direction_summary = summary["directions"][direction]
        calculated = {"AP": float(average_precision_score(labels, ensemble_scores)), "AUROC": float(roc_auc_score(labels, ensemble_scores))}
        check(f"{direction}: ensemble metrics", np.allclose([calculated["AP"], calculated["AUROC"]], [direction_summary["ensemble"]["AP"], direction_summary["ensemble"]["AUROC"]], rtol=0, atol=1e-12), calculated)
        ensemble_path = root / direction_summary["ensemble"]["predictions"]
        ensemble_rows = load_jsonl(ensemble_path)
        ensemble_table = {row["sample_id"]: row for row in ensemble_rows}
        ensemble_file_valid = len(ensemble_table) == len(ids) and sorted(ensemble_table) == ids and all(abs(float(ensemble_table[sample_id]["score"]) - ensemble_scores[i]) <= 1e-12 and int(ensemble_table[sample_id]["label"]) == labels[i] for i, sample_id in enumerate(ids))
        check(f"{direction}: ensemble predictions", ensemble_file_valid and sha256(ensemble_path) == direction_summary["ensemble"]["predictions_sha256"], {"rows": len(ensemble_rows), "sha256": sha256(ensemble_path)})

        for method in METHODS:
            acie_tables = []
            for seed in SEEDS:
                rows = load_jsonl(root / "outputs" / "frozen_cross_domain" / direction / method / f"seed{seed}" / "test_predictions.jsonl")
                acie_tables.append({row["sample_id"]: row for row in rows})
            valid_ids = all(sorted(table) == ids and len(table) == len(ids) for table in acie_tables)
            acie_scores = np.asarray([[float(table[sample_id]["score"]) for sample_id in ids] for table in acie_tables]).mean(axis=0)
            point = calculated["AP"] - float(average_precision_score(labels, acie_scores))
            recorded = direction_summary["comparisons"][f"official_mlp_minus_{method}"]["point"]
            check(f"{direction}: comparison with {method}", valid_ids and abs(point - recorded) <= 1e-12, {"point": point, "recorded": recorded})

        for scope_file, part in zip(mapping["scope_files"], mapping["parts"], strict=True):
            scope = load_json(root / "validation" / "official_server" / scope_file)
            expected = protocol["directions"][direction]["fair"][part]
            valid = scope.get("exact_track_label_choice_and_frames") is True and not scope.get("failures") and scope.get("samples") == expected["samples"] and scope.get("positives") == expected["positives"] and scope.get("observed_windows_sha256") == expected["windows_sha256"]
            check(f"{direction}: official loader scope {part}", valid, {"file": scope_file, "samples": scope.get("samples"), "positives": scope.get("positives"), "windows_sha256": scope.get("observed_windows_sha256")})

    report = {
        "schema": "acie.official-fair-five-seed-validation.v1",
        "status": "passed" if not errors else "failed",
        "protocol": "official_baselines/protocol_v1/PROTOCOL.json",
        "seeds": SEEDS,
        "directions": list(DIRECTIONS),
        "models": len(SEEDS) * len(DIRECTIONS),
        "prediction_rows": total_predictions,
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "errors": errors,
        "checks": checks,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "models", "prediction_rows", "checks_passed", "checks_total", "errors")}, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
