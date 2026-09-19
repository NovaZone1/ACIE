#!/usr/bin/env python3
"""Summarize R4 horizon ranking, frozen operating points, and paired differences."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from acie.data import Bundle
from acie.io import read_json, read_jsonl, write_json
from acie.metrics import binary_metrics
from acie.vv_horizon import fpr_budget_threshold, paired_date_seed_bootstrap
from vv_score_r4 import prediction_path, verify_protocol

_BUDGET_CACHE: dict[str, dict[str, dict]] = {}


def table(path: Path) -> dict[str, dict]:
    rows = read_jsonl(path)
    result = {row["sample_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate sample IDs: {path}")
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_matrix(paths: list[Path], ids: list[str]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    tables = [table(path) for path in paths]
    if any(set(ids) - set(item) for item in tables):
        raise ValueError("Prediction coverage mismatch")
    labels = np.asarray([tables[0][sample_id]["label"] for sample_id in ids], dtype=int)
    scores = np.asarray([[item[sample_id]["score"] for sample_id in ids] for item in tables])
    return labels, scores, [tables[0][sample_id] for sample_id in ids]


def source_budget_points(root: Path, checkpoint_spec: dict) -> dict[str, dict]:
    checkpoint = root / checkpoint_spec["path"]
    key = str(checkpoint.resolve())
    if key in _BUDGET_CACHE:
        return _BUDGET_CACHE[key]
    rows = read_jsonl(checkpoint.parent / "val_predictions.jsonl")
    y = np.asarray([row["label"] for row in rows], dtype=int)
    scores = np.asarray([row["score"] for row in rows], dtype=float)
    result = {str(budget): fpr_budget_threshold(y, scores, budget) for budget in (0.01, 0.05)}
    _BUDGET_CACHE[key] = result
    return result


def summarize_method(root: Path, out: Path, direction: str, model: str, seeds: list[int],
                     cutoff: int, ids: list[str], checkpoint_specs: dict) -> tuple[dict, np.ndarray, np.ndarray, list[dict]]:
    paths = [prediction_path(out, direction, model, seed, cutoff) for seed in seeds]
    y, matrix, meta = score_matrix(paths, ids)
    per_seed = []
    for seed, scores, path in zip(seeds, matrix, paths):
        rows = table(path)
        threshold = float(next(iter({rows[sample_id]["source_threshold"] for sample_id in ids})))
        item = {"seed": seed, "source_f1": binary_metrics(y, scores, threshold), "source_fpr_budgets": {}}
        for budget, point in source_budget_points(root, checkpoint_specs[str(seed)]).items():
            item["source_fpr_budgets"][budget] = {
                "threshold": point["threshold"], "source_val": point["source_val"],
                "target": binary_metrics(y, scores, point["threshold"]),
            }
        per_seed.append(item)
    ap = np.asarray([row["source_f1"]["AP"] for row in per_seed], dtype=float)
    auroc = np.asarray([row["source_f1"]["AUROC"] for row in per_seed], dtype=float)
    result = {
        "seeds": seeds, "AP_mean": float(ap.mean()),
        "AP_sample_SD": float(ap.std(ddof=1)) if len(ap) > 1 else None,
        "AUROC_mean": float(auroc.mean()),
        "AUROC_sample_SD": float(auroc.std(ddof=1)) if len(auroc) > 1 else None,
        "per_seed": per_seed,
    }
    return result, y, matrix, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root_default = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument("--protocol", type=Path, default=Path("protocols/vv_followup_v1/R4_PROTOCOL_LOCK.json"))
    parser.add_argument("--out", type=Path, default=Path("outputs/vv_followup_v1/r4"))
    parser.add_argument("--result-out", type=Path,
                        help="Write revised summaries separately while reading predictions from --out")
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol_path = args.protocol if args.protocol.is_absolute() else root / args.protocol
    out = args.out if args.out.is_absolute() else root / args.out
    result_out = (args.result_out if args.result_out and args.result_out.is_absolute() else
                  root / args.result_out if args.result_out else out)
    protocol = read_json(protocol_path)
    verify_protocol(protocol)
    summary = {
        "schema": "acie.vv-r4-summary.v2", "protocol_digest": protocol["protocol_digest"],
        "bootstrap_repeats": args.bootstrap, "primary_set": "common_tracks", "directions": {},
    }
    metric_rows, difference_rows, date_rows = [], [], []
    for direction_index, (direction, spec) in enumerate(protocol["directions"].items()):
        bundles, ids_by_cutoff, meta_by_cutoff = {}, {}, {}
        for bundle_spec in spec["bundles"]:
            cutoff = int(bundle_spec["cutoff_frames"])
            bundle = Bundle.load(root / bundle_spec["path"])
            bundles[cutoff] = bundle
            ids_by_cutoff[cutoff] = [meta["sample_id"] for meta in bundle.meta]
            meta_by_cutoff[cutoff] = {meta["sample_id"]: meta for meta in bundle.meta}
        common = sorted(set.intersection(*(set(ids) for ids in ids_by_cutoff.values())))
        base_ids = set(ids_by_cutoff[min(ids_by_cutoff)])
        lost = sorted(base_ids - set(common))
        base_meta = meta_by_cutoff[min(ids_by_cutoff)]
        base_predictions = table(prediction_path(out, direction, "G", 11, min(ids_by_cutoff)))
        direction_result = {
            "target": spec["target"], "J_star": spec["J_star"], "C_star": spec["C_star"],
            "common_n": len(common),
            "common_positives": int(sum(int(base_predictions[sample_id]["label"]) for sample_id in common)),
            "lost_from_base_to_common": {
                "n": len(lost), "positives": int(sum(int(base_predictions[sample_id]["label"]) for sample_id in lost)),
                "by_date": {group: sum(base_meta[sample_id].get("group_id", base_meta[sample_id]["recording"]) == group for sample_id in lost)
                            for group in sorted({base_meta[sample_id].get("group_id", base_meta[sample_id]["recording"]) for sample_id in lost})},
            },
            "horizons": [],
        }
        for cutoff_index, cutoff in enumerate(sorted(ids_by_cutoff)):
            nominal_seconds = cutoff / protocol["fps"]
            positive_meta = [meta for meta, label in zip(bundles[cutoff].meta, bundles[cutoff].y)
                             if int(label) == 1]
            if any(meta.get("event_lead_seconds") is None or
                   not np.isfinite(float(meta["event_lead_seconds"])) for meta in positive_meta):
                raise ValueError(f"Missing/non-finite actual event lead time: {direction}/{cutoff}")
            positive_lead = [float(meta["event_lead_seconds"]) for meta in positive_meta]
            horizon = {"cutoff_frames": cutoff, "nominal_cutoff_seconds": nominal_seconds, "sets": {}}
            horizon["complete_coverage"] = {
                "n": len(ids_by_cutoff[cutoff]), "positives": int(bundles[cutoff].y.sum()),
                "positive_event_lead_seconds_actual": {"min": min(positive_lead), "median": float(np.median(positive_lead)), "max": max(positive_lead)},
                "negative_anchor_policy": bundles[cutoff].provenance.get("negative_alignment"),
            }
            matrices = {}
            for set_name, ids in (("complete", ids_by_cutoff[cutoff]), ("common_tracks", common)):
                methods = {}
                matrices[set_name] = {}
                labels = None
                metadata = None
                for model in spec["models"]:
                    seeds = [11] if model == "histgb" else protocol["seeds"]
                    result, y, matrix, meta = summarize_method(
                        root, out, direction, model, seeds, cutoff, ids, spec["checkpoints"][model])
                    methods[model] = result
                    matrices[set_name][model] = matrix
                    labels, metadata = y, meta
                    metric_rows.append({
                        "direction": direction, "cutoff_frames": cutoff, "set": set_name,
                        "model": model, "n": len(y), "positives": int(y.sum()),
                        "AP_mean": result["AP_mean"], "AP_sample_SD": result["AP_sample_SD"],
                        "AUROC_mean": result["AUROC_mean"], "AUROC_sample_SD": result["AUROC_sample_SD"],
                    })
                set_positive_lead = [float(meta_by_cutoff[cutoff][sample_id]["event_lead_seconds"])
                                     for sample_id in ids if int(base_predictions[sample_id]["label"]) == 1]
                horizon["sets"][set_name] = {
                    "n": len(labels), "positives": int(labels.sum()), "methods": methods,
                    "positive_event_lead_seconds_actual": {
                        "min": min(set_positive_lead), "median": float(np.median(set_positive_lead)),
                        "max": max(set_positive_lead),
                    } if set_positive_lead else None,
                }
                if set_name == "common_tracks":
                    groups = np.asarray([str(row.get("group_id", row["recording"])) for row in metadata])
                    for opponent in ("G", spec["J_star"], spec["C_star"]):
                        f_scores, other = matrices[set_name]["F"], matrices[set_name][opponent]
                        seed_differences = [float(average_precision_score(labels, f_scores[i]) -
                                                  average_precision_score(labels, other[i])) for i in range(5)]
                        bootstrap_seed = 20260916 + direction_index * 100 + cutoff_index
                        boot = paired_date_seed_bootstrap(
                            labels, groups, f_scores, other, args.bootstrap,
                            bootstrap_seed)
                        comparison = {
                            "comparison": f"F-{opponent}", "seed_AP_differences": seed_differences,
                            "mean_AP_difference": float(np.mean(seed_differences)),
                            "positive_seeds": int(np.sum(np.asarray(seed_differences) > 0)),
                            "bootstrap_seed": bootstrap_seed, "bootstrap": boot,
                        }
                        horizon["sets"][set_name].setdefault("comparisons", []).append(comparison)
                        interval = boot["joint_date_and_seed"]["interval_95"]
                        difference_rows.append({
                            "direction": direction, "cutoff_frames": cutoff,
                            "comparison": f"F-{opponent}", "mean_AP_difference": comparison["mean_AP_difference"],
                            "positive_seeds": comparison["positive_seeds"], "interval_95_low": interval[0],
                            "interval_95_high": interval[1],
                        })
                        for group in np.unique(groups):
                            within = np.flatnonzero(groups == group)
                            keep = np.flatnonzero(groups != group)
                            within_delta = None if len(np.unique(labels[within])) < 2 else float(np.mean([
                                average_precision_score(labels[within], f_scores[i, within]) -
                                average_precision_score(labels[within], other[i, within]) for i in range(5)]))
                            loo_delta = None if len(np.unique(labels[keep])) < 2 else float(np.mean([
                                average_precision_score(labels[keep], f_scores[i, keep]) -
                                average_precision_score(labels[keep], other[i, keep]) for i in range(5)]))
                            date_rows.append({
                                "direction": direction, "cutoff_frames": cutoff,
                                "comparison": f"F-{opponent}", "date_group": group,
                                "n": len(within), "positives": int(labels[within].sum()),
                                "within_date_AP_difference": within_delta,
                                "leave_one_date_out_AP_difference": loo_delta,
                            })
            direction_result["horizons"].append(horizon)
        summary["directions"][direction] = direction_result
    result_out.mkdir(parents=True, exist_ok=True)
    write_json(result_out / "r4_summary.json", summary)
    write_csv(result_out / "r4_metrics.csv", metric_rows)
    write_csv(result_out / "r4_common_differences.csv", difference_rows)
    write_csv(result_out / "r4_date_diagnostics.csv", date_rows)
    lines = ["# R4 horizon and operating-point results", "",
             "Primary comparisons use tracks available at all five horizons.", "",
             "| Direction | Frames | Comparison | Mean AP difference | Positive seeds | 95% interval |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in difference_rows:
        lines.append(f"| {row['direction']} | {row['cutoff_frames']} | {row['comparison']} | {row['mean_AP_difference']:.6f} | {row['positive_seeds']}/5 | [{row['interval_95_low']:.6f}, {row['interval_95_high']:.6f}] |")
    (result_out / "R4_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "metric_rows": len(metric_rows),
                      "comparisons": len(difference_rows), "summary": str(result_out / "r4_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
