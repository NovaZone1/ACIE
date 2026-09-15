"""Verify a checksum manifest relative to the repository root."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="CHECKSUMS_SHA256.txt")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = root / args.manifest
    errors = []
    missing = []
    checked = 0
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        path = root / name
        if not path.is_file():
            missing.append(name)
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            errors.append(name)
        checked += 1
    if errors:
        raise SystemExit("Checksum mismatch: " + ", ".join(errors))
    if missing and not args.allow_missing:
        raise SystemExit("Missing files: " + ", ".join(missing))
    suffix = f"; {len(missing)} unavailable files skipped" if missing else ""
    print(f"All {checked} available file checksums match{suffix}.")


if __name__ == "__main__":
    main()
