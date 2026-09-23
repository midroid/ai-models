"""Sha256 lock for vendored evaluation files. Eval refuses to run on a mismatch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from laya_jp.pins import EXTERNAL, MANIFEST


class LockError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_hashes(root: Path = EXTERNAL) -> dict[str, str]:
    hashes = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        hashes[path.relative_to(root).as_posix()] = sha256_file(path)
    return hashes


def read_manifest(path: Path = MANIFEST) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_lock(path: Path = MANIFEST, root: Path = EXTERNAL) -> dict:
    if not path.is_file():
        raise LockError(f"missing manifest: {path}")
    manifest = read_manifest(path)
    expected = manifest.get("files")
    if not isinstance(expected, dict) or not expected:
        raise LockError("manifest has no files")
    actual = file_hashes(root)
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    changed = sorted(name for name in expected if name in actual and actual[name] != expected[name])
    if missing or extra or changed:
        raise LockError(
            "vendored eval files do not match data/manifest.json: "
            f"missing={missing} extra={extra} changed={changed}"
        )
    return manifest
