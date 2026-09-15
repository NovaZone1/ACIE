#!/usr/bin/env python3
"""Validate the two-direction, five-seed official LSTM fair comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score


SEEDS = [11, 22, 33, 44, 55]
COMPARISONS = ["geometry", "dual_no_pair", "weighted_no_pair", "full_selected", "official_mlp"]
DIRECTIONS = {
    "HUI360_to_SSUP-A": {"scope_files": ("hui_train_scope.json", "hui_val_scope.json", "ssup_target_scope.json"), "parts": ("train", "source_val", "target_test"), "epochs": 75},
    "SSUP-A_to_HUI360": {"scope_files": ("ssup_train_scope.json", "ssup_val_scope.json", "hui_target_scope.json"), "parts": ("train", "source_val", "target_test"), "epochs": 30},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prediction_table(path: Path) -> dict[str, dict]:
    rows = load_jsonl(path)
    table = {row["sample_id"]: row for row in rows}
    if len(table) != len(rows):
        raise ValueError(f"Duplicate sample IDs: {path}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out or root / "validation" / "official_fair_lstm_validation.json"
    protocol = load_json(root / "official_baselines" / "protocol_v1" / "PROTOCOL.json")
    summary = load_json(root / "outputs" / "official_baselines" / "fair_v1" / "five_seed_lstm_summary.json")
    checks = []
    errors = []

    def check(name: str, condition: bool, detail: object) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})
        if not condition:
            errors.append(name)

    check("summary identity", summary.get("schema") == "acie.official-fair-lstm-five-seed-summary.v1" and summary.get("model") == "lstm" and summary.get("seeds") == SEEDS, {key: summary.get(key) for key in ("schema", "model", "seeds")})
    total_predictions = 0
    for direction, spec in DIRECTIONS.items():
        model_root = root / "outputs" / "official_baselines" / "fair_v1" / direction / "lstm"
        target = protocol["directions"][direction]["fair"]["target_test"]
        train_config = yaml.safe_load((root / "official_baselines" / "protocol_v1" / direction / "fair_train_lstm.yaml").read_text(encoding="utf-8"))
        target_config = yaml.safe_load((root / "official_baselines" / "protocol_v1" / direction / "fair_target_eval_lstm.yaml").read_text(encoding="utf-8"))
        model_spec = protocol["model_specs"]["lstm"][direction]
        config_valid = all(config["force_model_type"] == "lstm" and config["epochs"] == spec["epochs"] and config["lstm_hidden_dim"] == 128 and config["lstm_num_layers"] == 3 and float(config["lstm_dropout"]) == 0.0 for config in (train_config, target_config)) and model_spec["epochs"] == spec["epochs"]
        check(f"{direction}: official LSTM config", config_valid, {"epochs": train_config.get("epochs"), "hidden_dim": train_config.get("lstm_hidden_dim"), "num_layers": train_config.get("lstm_num_layers"), "dropout": train_config.get("lstm_dropout")})

        tables = []
        prediction_hashes = []
        for seed in SEEDS:
            run = model_root / f"seed{seed}"
            metrics = load_json(run / "metrics.json")
            provenance = load_json(run / "run_provenance.json")
            predictions_path = run / "predictions.jsonl"
            rows = load_jsonl(predictions_path)
            table = {row["sample_id"]: row for row in rows}
            tables.append(table)
            total_predictions += len(rows)
            ids = [row["sample_id"] for row in rows]
            labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
            scores = np.asarray([float(row["score"]) for row in rows], dtype=float)
            prediction_hashes.append(metrics.get("predictions_sha256"))
            check(f"{direction}/seed{seed}: identity", metrics.get("schema") == "acie.official-fair-result.v1" and metrics.get("direction") == direction and metrics.get("model") == "lstm" and metrics.get("seed") == seed, {key: metrics.get(key) for key in ("schema", "direction", "model", "seed")})
            check(f"{direction}/seed{seed}: source-only selection", metrics.get("target_labels_used_for_selection") is False and metrics.get("checkpoint_selection") == "best source-validation AP", {"target_labels_used_for_selection": metrics.get("target_labels_used_for_selection"), "checkpoint_selection": metrics.get("checkpoint_selection")})
            check(f"{direction}/seed{seed}: finite metrics", all(math.isfinite(float(metrics[key])) for key in ("AP", "AUROC")), {"AP": metrics.get("AP"), "AUROC": metrics.get("AUROC")})
            check(f"{direction}/seed{seed}: artifact hashes", sha256(run / "checkpoint_source_best_ap.pth") == metrics.get("checkpoint_sha256") and sha256(predictions_path) == metrics.get("predictions_sha256"), {"checkpoint": metrics.get("checkpoint_sha256"), "predictions": metrics.get("predictions_sha256")})
            check(f"{direction}/seed{seed}: count and labels", len(rows) == metrics.get("n_test") == target["samples"] and int(labels.sum()) == metrics.get("test_positives") == target["positives"], {"rows": len(rows), "positives": int(labels.sum())})
            check(f"{direction}/seed{seed}: predictions valid", len(ids) == len(set(ids)) and set(labels.tolist()) <= {0, 1} and np.isfinite(scores).all() and ((0 <= scores) & (scores <= 1)).all(), {"unique_ids": len(set(ids)), "score_min": float(scores.min()), "score_max": float(scores.max())})
            check(f"{direction}/seed{seed}: provenance", provenance.get("schema") == "acie.official-seed-run.v1" and provenance.get("direction") == direction and provenance.get("model") == "lstm" and provenance.get("seed") == seed and provenance.get("target_labels_used_for_selection") is False, {key: provenance.get(key) for key in ("direction", "model", "seed", "seed_control")})

        ids = sorted(tables[0])
        aligned = all(len(table) == len(ids) and sorted(table) == ids for table in tables)
        check(f"{direction}: IDs align across seeds", aligned, len(ids))
        check(f"{direction}: seeds change predictions", len(set(prediction_hashes)) == len(SEEDS), prediction_hashes)
        labels = np.asarray([int(tables[0][sample_id]["label"]) for sample_id in ids], dtype=int)
        scores = np.asarray([[float(table[sample_id]["score"]) for sample_id in ids] for table in tables]).mean(axis=0)
        calculated = {"AP": float(average_precision_score(labels, scores)), "AUROC": float(roc_auc_score(labels, scores))}
        direction_summary = summary["directions"][direction]
        check(f"{direction}: ensemble metrics", np.allclose([calculated["AP"], calculated["AUROC"]], [direction_summary["ensemble"]["AP"], direction_summary["ensemble"]["AUROC"]], rtol=0, atol=1e-12), calculated)
        ensemble_path = root / direction_summary["ensemble"]["predictions"]
        ensemble_table = prediction_table(ensemble_path)
        ensemble_valid = sorted(ensemble_table) == ids and all(abs(float(ensemble_table[sample_id]["score"]) - scores[i]) <= 1e-12 and int(ensemble_table[sample_id]["label"]) == labels[i] for i, sample_id in enumerate(ids))
        check(f"{direction}: ensemble predictions", ensemble_valid and sha256(ensemble_path) == direction_summary["ensemble"]["predictions_sha256"], {"rows": len(ensemble_table), "sha256": sha256(ensemble_path)})

        for comparison in COMPARISONS:
            if comparison == "official_mlp":
                paths = [root / "outputs" / "official_baselines" / "fair_v1" / direction / "mlp" / f"seed{seed}" / "predictions.jsonl" for seed in SEEDS]
            else:
                paths = [root / "outputs" / "frozen_cross_domain" / direction / comparison / f"seed{seed}" / "test_predictions.jsonl" for seed in SEEDS]
            other_tables = [prediction_table(path) for path in paths]
            valid_ids = all(sorted(table) == ids for table in other_tables)
            other_scores = np.asarray([[float(table[sample_id]["score"]) for sample_id in ids] for table in other_tables]).mean(axis=0)
            point = calculated["AP"] - float(average_precision_score(labels, other_scores))
            recorded = direction_summary["comparisons"][f"official_lstm_minus_{comparison}"]["point"]
            check(f"{direction}: comparison with {comparison}", valid_ids and abs(point - recorded) <= 1e-12, {"point": point, "recorded": recorded})

        for scope_file, part in zip(spec["scope_files"], spec["parts"], strict=True):
            scope = load_json(root / "validation" / "official_server" / scope_file)
            expected = protocol["directions"][direction]["fair"][part]
            valid = scope.get("exact_track_label_choice_and_frames") is True and not scope.get("failures") and scope.get("samples") == expected["samples"] and scope.get("positives") == expected["positives"] and scope.get("observed_windows_sha256") == expected["windows_sha256"]
            check(f"{direction}: official loader scope {part}", valid, {"file": scope_file, "samples": scope.get("samples"), "positives": scope.get("positives"), "windows_sha256": scope.get("observed_windows_sha256")})

    report = {
        "schema": "acie.official-fair-lstm-five-seed-validation.v1",
        "status": "passed" if not errors else "failed",
        "protocol": "official_baselines/protocol_v1/PROTOCOL.json",
        "seeds": SEEDS, "directions": list(DIRECTIONS), "models": 10,
        "prediction_rows": total_predictions,
        "checks_passed": sum(item["passed"] for item in checks), "checks_total": len(checks),
        "errors": errors, "checks": checks,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "models", "prediction_rows", "checks_passed", "checks_total", "errors")}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
