"""Registered behavior-input transformations for VV R5 diagnostics."""
from __future__ import annotations

from typing import Any

import numpy as np

from .data import Bundle, resolve_split
from .io import digest_object


def _derangement(indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if len(indices) < 2:
        raise ValueError("Within-date permutation requires at least two samples")
    for _ in range(10000):
        donor = rng.permutation(indices)
        if np.all(donor != indices):
            return donor
    raise RuntimeError("Failed to construct a derangement")


def permute_behavior_within_partition_date(bundle: Bundle, split: dict[str, Any],
                                           permutation_seed: int) -> tuple[Bundle, dict[str, Any]]:
    """Derange whole q tensors within each split partition and date group."""
    partitions = resolve_split(bundle, split)
    rng = np.random.default_rng(permutation_seed)
    q = bundle.q.copy()
    mappings = []
    covered: set[int] = set()
    for partition in ("train", "val", "test"):
        indices = partitions[partition]
        groups = sorted({str(bundle.meta[i].get("group_id", bundle.meta[i]["recording"])) for i in indices})
        for group in groups:
            recipients = np.asarray([i for i in indices
                                     if str(bundle.meta[i].get("group_id", bundle.meta[i]["recording"])) == group])
            donors = _derangement(recipients, rng)
            q[recipients] = bundle.q[donors]
            covered.update(int(i) for i in recipients)
            mappings.extend({
                "partition": partition, "date_group": group,
                "recipient": bundle.meta[int(recipient)]["sample_id"],
                "donor": bundle.meta[int(donor)]["sample_id"],
            } for recipient, donor in zip(recipients, donors))
    expected = set(np.concatenate(list(partitions.values())).tolist())
    if covered != expected or any(row["recipient"] == row["donor"] for row in mappings):
        raise AssertionError("Permutation coverage or no-fixed-point contract failed")
    mapping_digest = digest_object(mappings)
    audit = {
        "schema": "acie.vv-r5a-permutation.v1", "permutation_seed": permutation_seed,
        "mapping_digest": mapping_digest, "mapping_count": len(mappings),
        "fixed_points": 0, "partitions": {name: len(indices) for name, indices in partitions.items()},
        "mapping": mappings,
    }
    provenance = dict(bundle.provenance)
    provenance["r5_transform"] = {
        "kind": "within_partition_date_whole_q_derangement",
        "permutation_seed": permutation_seed, "mapping_digest": mapping_digest,
        "labels_used_for_mapping": False,
    }
    transformed = Bundle(bundle.a.copy(), q, bundle.y.copy(), [dict(meta) for meta in bundle.meta], provenance)
    transformed.validate()
    return transformed, audit


def confidence_only_behavior(bundle: Bundle) -> Bundle:
    """Keep confidence, valid, and velocity-valid; zero pose and velocity values."""
    q = bundle.q.copy().reshape(*bundle.q.shape[:2], 17, 7)
    q[..., [0, 1, 3, 4]] = 0.0
    q = q.reshape(bundle.q.shape)
    provenance = dict(bundle.provenance)
    provenance["r5_transform"] = {
        "kind": "confidence_valid_velocity_valid_only", "kept_channels": [2, 5, 6],
        "zeroed_channels": [0, 1, 3, 4],
    }
    transformed = Bundle(bundle.a.copy(), q.astype(np.float32), bundle.y.copy(),
                         [dict(meta) for meta in bundle.meta], provenance)
    transformed.validate()
    return transformed


def zero_explicit_velocity_behavior(bundle: Bundle) -> Bundle:
    """Zero explicit velocity values/validity while retaining pose coordinates."""
    q = bundle.q.copy().reshape(*bundle.q.shape[:2], 17, 7)
    q[..., [3, 4, 6]] = 0.0
    q = q.reshape(bundle.q.shape)
    provenance = dict(bundle.provenance)
    provenance["r5_transform"] = {
        "kind": "explicit_velocity_zeroed", "kept_channels": [0, 1, 2, 5],
        "zeroed_channels": [3, 4, 6],
        "note": "temporal convolutions may still infer motion from coordinate sequences",
    }
    transformed = Bundle(bundle.a.copy(), q.astype(np.float32), bundle.y.copy(),
                         [dict(meta) for meta in bundle.meta], provenance)
    transformed.validate()
    return transformed
