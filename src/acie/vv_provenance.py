"""Content-bound identities and target-lock validation for VV experiments."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .data import Bundle
from .io import digest_file, digest_object, read_json, read_jsonl


IDENTITY_VERSION = 2


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def training_content_digest(bundle: Bundle, partitions: dict[str, np.ndarray]) -> str:
    """Hash the actual source samples used for fitting and validation."""
    content: dict[str, Any] = {"schema": "acie.vv-training-content.v1", "partitions": {}}
    for name in ("train", "val"):
        indices = np.asarray(partitions[name], dtype=int)
        content["partitions"][name] = {
            "sample_ids": [bundle.meta[int(i)]["sample_id"] for i in indices],
            "metadata": [bundle.meta[int(i)] for i in indices],
            "a_sha256": array_digest(bundle.a[indices]),
            "q_sha256": array_digest(bundle.q[indices]),
            "y_sha256": array_digest(bundle.y[indices]),
        }
    content["provenance"] = bundle.provenance
    return digest_object(content)


def source_snapshot(extra_files: Iterable[str | Path] = ()) -> dict[str, Any]:
    """Return a reproducible hash of the code that defines VV runs."""
    package = Path(__file__).resolve().parent
    files = [
        package / "data.py",
        package / "features.py",
        package / "io.py",
        package / "vv_engine.py",
        package / "vv_models.py",
        package / "vv_provenance.py",
        package / "vv_trees.py",
    ]
    files.extend(Path(path).resolve() for path in extra_files)
    project_root = package.parents[1]
    manifest = {}
    for path in sorted(set(files)):
        if not path.is_file():
            continue
        try:
            key = str(path.relative_to(project_root))
        except ValueError:
            key = path.name
        manifest[key] = digest_file(path)
    return {
        "schema": "acie.vv-source-snapshot.v1",
        "files": manifest,
        "sha256": digest_object(manifest),
    }


def validate_complete_run(run_dir: str | Path, run: dict[str, Any], tree: bool = False) -> None:
    """Reject a stale complete marker when required artifacts or hashes differ."""
    run_dir = Path(run_dir)
    common = ["run.json", "split.json", "scaler.json", "val_predictions.jsonl", "metrics.json", "timing.json"]
    required = common + (["tree.joblib", "candidate_results.json"] if tree else
                         ["config.yaml", "initial_state.json", "history.jsonl", "best.pt"])
    if run.get("geometry_checkpoint_hash"):
        required.append("geometry_reference.json")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise ValueError(f"Complete run is missing required artifacts: {missing}")
    checkpoint = run_dir / ("tree.joblib" if tree else "best.pt")
    observed = digest_file(checkpoint)
    if observed != run.get("checkpoint_hash"):
        raise ValueError("Complete run checkpoint hash does not match run.json")
    rows = read_jsonl(run_dir / "val_predictions.jsonl")
    if not rows or any(row.get("checkpoint_hash") != observed for row in rows):
        raise ValueError("Complete run validation predictions do not match the checkpoint")


def validate_locked_checkpoint(lock_path: str | Path, checkpoint: str | Path,
                               **identity: Any) -> dict[str, Any]:
    """Require a checkpoint to appear unchanged in a frozen target-scoring lock."""
    lock_path, checkpoint = Path(lock_path), Path(checkpoint).resolve()
    lock = read_json(lock_path)
    if lock.get("status") != "locked" or not isinstance(lock.get("checkpoints"), list):
        raise ValueError(f"Target scoring lock is absent or invalid: {lock_path}")
    if lock.get("lock_digest"):
        frozen = dict(lock)
        expected = frozen.pop("lock_digest")
        if digest_object(frozen) != expected:
            raise ValueError(f"Target scoring lock digest mismatch: {lock_path}")
    matches = []
    for item in lock["checkpoints"]:
        try:
            same_path = Path(item["path"]).resolve() == checkpoint
        except (KeyError, TypeError):
            same_path = False
        same_identity = all(item.get(key) == value for key, value in identity.items())
        if same_path and same_identity:
            matches.append(item)
    if len(matches) != 1:
        raise ValueError(f"Checkpoint is not uniquely authorized by {lock_path}: {checkpoint}")
    observed = digest_file(checkpoint)
    if observed != matches[0].get("sha256"):
        raise ValueError(f"Locked checkpoint hash mismatch: {checkpoint}")
    return matches[0]
