#!/usr/bin/env python3
"""Measure R2 geometry-branch drift for F, Jw, and JwD without retraining."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from acie.io import read_jsonl, write_json

DIRECTIONS = ("HUI360_to_SSUP-A", "SSUP-A_to_HUI360")
MODELS = ("F", "Jw", "JwD")
SEEDS = (11, 22, 33, 44, 55)


def aligned(path: Path) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    rows = read_jsonl(path)
    return ([row["sample_id"] for row in rows], np.asarray([row["label"] for row in rows], dtype=int),
            {key: np.asarray([row[key] for row in rows], dtype=float)
             for key in ("score", "geometry_logit", "evidence_logit")})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root_default = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument("--r2", type=Path, default=Path("outputs/vv_followup_v1/r2_final"))
    parser.add_argument("--out", type=Path, default=Path("outputs/vv_followup_v1/r5/r5e_geometry_drift"))
    args = parser.parse_args()
    root = args.root.resolve()
    r2 = args.r2 if args.r2.is_absolute() else root / args.r2
    out = args.out if args.out.is_absolute() else root / args.out
    rows = []
    for direction in DIRECTIONS:
        for split_name, filename in (("source_val", "val_predictions.jsonl"),
                                     ("target", "test_predictions.jsonl")):
            for seed in SEEDS:
                g_path = r2 / direction / "G" / "fixed" / str(seed) / filename
                g_ids, g_y, g_values = aligned(g_path)
                for model in MODELS:
                    path = r2 / direction / model / "fixed" / str(seed) / filename
                    ids, y, values = aligned(path)
                    if ids != g_ids or not np.array_equal(y, g_y):
                        raise ValueError(f"R5e coverage mismatch: {direction}/{split_name}/{model}/{seed}")
                    start, end, evidence = g_values["geometry_logit"], values["geometry_logit"], values["evidence_logit"]
                    rows.append({
                        "direction": direction, "split": split_name, "model": model, "seed": seed,
                        "n": len(y), "positives": int(y.sum()),
                        "geometry_start_AP": float(average_precision_score(y, start)),
                        "geometry_end_AP": float(average_precision_score(y, end)),
                        "geometry_AP_change": float(average_precision_score(y, end) - average_precision_score(y, start)),
                        "geometry_logit_MSE": float(np.mean(np.square(end - start))),
                        "geometry_evidence_correlation": (float(np.corrcoef(end, evidence)[0, 1])
                                                          if np.std(end) > 0 and np.std(evidence) > 0 else None),
                        "total_AP": float(average_precision_score(y, values["score"])),
                    })
    aggregate = []
    for direction in DIRECTIONS:
        for split_name in ("source_val", "target"):
            for model in MODELS:
                subset = [row for row in rows if row["direction"] == direction and
                          row["split"] == split_name and row["model"] == model]
                aggregate.append({
                    "direction": direction, "split": split_name, "model": model,
                    **{key + "_mean": float(np.mean([row[key] for row in subset]))
                       for key in ("geometry_start_AP", "geometry_end_AP", "geometry_AP_change",
                                   "geometry_logit_MSE", "total_AP")},
                    "geometry_evidence_correlation_mean": float(np.mean([
                        row["geometry_evidence_correlation"] for row in subset
                        if row["geometry_evidence_correlation"] is not None])),
                })
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "r5e_summary.json", {"schema": "acie.vv-r5e-geometry-drift.v1",
                                           "per_seed": rows, "aggregate": aggregate})
    for name, data in (("r5e_per_seed.csv", rows), ("r5e_aggregate.csv", aggregate)):
        with (out / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    print(json.dumps({"status": "complete", "rows": len(rows), "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
