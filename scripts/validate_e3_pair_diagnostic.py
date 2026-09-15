"""Validate that the E3 table exactly summarizes the frozen metric files."""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, stdev

from acie.io import digest_file, digest_object, read_json, write_json

FIELDS = [
    "n_pairs", "n_positive", "positive_coverage", "mean_distance", "p95_distance",
    "unique_negative_tracks", "max_negative_track_uses_observed", "geometry_accuracy",
    "evidence_accuracy", "final_accuracy",
]


def close(first, second, tolerance=1e-12):
    return first is None and second is None or (
        first is not None and second is not None and abs(float(first) - float(second)) <= tolerance
    )


def expected_stats(values):
    clean = [float(value) for value in values if value is not None]
    return {
        "mean": mean(clean) if clean else None,
        "std": stdev(clean) if len(clean) > 1 else (0.0 if clean else None),
        "min": min(clean) if clean else None,
        "max": max(clean) if clean else None,
        "n": len(clean),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    payload = read_json(args.summary)
    protocol = read_json(args.runs / "FROZEN_PROTOCOL.json")
    if payload["frozen_protocol_digest"] != protocol["lock_digest"]:
        raise ValueError("Frozen protocol digest mismatch")
    if payload["frozen_protocol_sha256"] != digest_file(args.runs / "FROZEN_PROTOCOL.json"):
        raise ValueError("Frozen protocol file changed")
    seeds = payload["seeds"]
    methods = payload["methods"]
    expected_manifest = []
    checked = 0
    for direction, direction_payload in payload["directions"].items():
        if set(direction_payload["methods"]) != set(methods):
            raise ValueError(f"Method mismatch for {direction}")
        for method in methods:
            item = direction_payload["methods"][method]
            if [row["seed"] for row in item["runs"]] != seeds:
                raise ValueError(f"Seed mismatch for {direction}/{method}")
            diagnostics = []
            for row in item["runs"]:
                path = args.runs / direction / method / f"seed{row['seed']}" / "test_predictions.metrics.json"
                source = read_json(path)["pair_diagnostic"]
                expected_manifest.append({"path": str(path), "sha256": digest_file(path)})
                for field in FIELDS:
                    if not close(row[field], source[field]):
                        raise ValueError(f"Raw field mismatch: {path}:{field}")
                expected_delta = source["final_accuracy"] - source["geometry_accuracy"]
                if not close(row["final_minus_geometry"], expected_delta):
                    raise ValueError(f"Delta mismatch: {path}")
                diagnostics.append(source)
                checked += 1
            for field in FIELDS:
                expected = expected_stats([diagnostic[field] for diagnostic in diagnostics])
                observed = item["summary"][field]
                if expected["n"] != observed["n"] or any(
                    not close(expected[key], observed[key]) for key in ["mean", "std", "min", "max"]
                ):
                    raise ValueError(f"Summary mismatch: {direction}/{method}/{field}")
    if payload["input_manifest"] != expected_manifest or payload["input_manifest_digest"] != digest_object(expected_manifest):
        raise ValueError("Input manifest mismatch")
    result = {
        "schema": "acie.e3-pair-diagnostic-validation.v1",
        "status": "passed",
        "files_checked": checked,
        "directions": sorted(payload["directions"]),
        "methods": methods,
        "seeds": seeds,
        "input_manifest_digest": payload["input_manifest_digest"],
        "frozen_protocol_digest": payload["frozen_protocol_digest"],
        "retraining_performed": False
    }
    write_json(args.out, result)
    print(f"Validated {checked} E3 diagnostic inputs and aggregates")


if __name__ == "__main__":
    main()
