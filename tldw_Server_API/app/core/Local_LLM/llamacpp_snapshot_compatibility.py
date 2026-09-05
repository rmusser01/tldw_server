"""Content fingerprint helpers for managed llama.cpp snapshots."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .llamacpp_snapshot_models import Fingerprint

_CHUNK_SIZE = 1024 * 1024


class UnstableFingerprintError(RuntimeError):
    """Raised when a file changes while its identity is being calculated."""


def compare_fingerprints(saved: Fingerprint, current: Fingerprint | None) -> list[str]:
    """Return every mismatched identity field, failing closed when unknown."""
    if current is None:
        return ["compatibility_unknown"]
    return [name for name in type(saved).model_fields if getattr(saved, name) != getattr(current, name)]


def hash_file_stable(path: Path) -> str:
    """Hash a regular file and reject identity changes during the read."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not _is_regular(before.st_mode):
            raise ValueError("fingerprint source must be a regular file")
        digest = hashlib.sha256()
        while chunk := os.read(fd, _CHUNK_SIZE):
            digest.update(chunk)
        after = os.fstat(fd)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after:
            raise UnstableFingerprintError("fingerprint source changed while hashing")
        return digest.hexdigest()
    finally:
        os.close(fd)


def canonical_sha256(value: object) -> str:
    """Hash canonical JSON for effective options and adapter descriptors."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_fingerprint(
    *,
    model: Path,
    executable: Path,
    effective_options: object,
    adapters: object,
    projector: Path | None = None,
) -> Fingerprint:
    """Build a content-only fingerprint; aliases and paths are not identity."""
    return Fingerprint(
        model_sha256=hash_file_stable(model),
        executable_sha256=hash_file_stable(executable),
        projector_sha256=hash_file_stable(projector) if projector is not None else None,
        effective_options_sha256=canonical_sha256(effective_options),
        adapters_sha256=canonical_sha256(adapters),
    )


def _is_regular(mode: int) -> bool:
    import stat

    return stat.S_ISREG(mode)
