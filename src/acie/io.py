"""Strict JSON I/O, atomic output, hashing, and run provenance."""
from __future__ import annotations
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def read_jsonl(path: str | Path) -> list[dict]:
    result = []
    with Path(path).open(encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{n}: {exc}') from exc
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{n}: expected an object')
            result.append(row)
    return result


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w',encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(temp, path)


def digest_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def digest_object(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def inside(root: str | Path, relative: str) -> Path:
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Path escapes declared root: {relative}')
    return path


def environment() -> dict:
    import importlib.metadata
    names = ['numpy','torch','scikit-learn','PyYAML','pytest','transformers','huggingface-hub','Pillow']
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {'python': sys.version, 'platform': platform.platform(), 'packages': versions}
