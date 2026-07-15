"""Canonical filesystem handling for playlist-ingest upload staging."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def run_file_staging_prefix(
    *,
    batch_id: str,
    idempotency_identity: str,
    submission_lease_token: str | None = None,
) -> str:
    """Return an opaque reservation prefix, optionally owned by one lease generation."""
    marker = hashlib.sha256(f"{batch_id}\0{idempotency_identity}".encode()).hexdigest()[:24]
    lease_marker = hashlib.sha256(submission_lease_token.encode()).hexdigest()[:12] if submission_lease_token else ""
    return f"media_ingest_job_{marker}_{lease_marker + '_' if lease_marker else ''}"


def validated_run_file_staging_dir(
    *,
    temp_dir: Any,
    batch_id: str,
    idempotency_identity: str,
) -> Path | None:
    """Resolve one reservation staging directory without accepting aliases."""
    if not isinstance(temp_dir, str) or not temp_dir:
        return None
    if (os.altsep and os.altsep in temp_dir) or os.path.normpath(temp_dir) != temp_dir:
        return None
    try:
        temp_root_path = Path(tempfile.gettempdir())
        lexical_candidate = Path(temp_dir)
        if not lexical_candidate.is_absolute() or str(lexical_candidate) != temp_dir:
            return None
        if lexical_candidate.parent != temp_root_path or lexical_candidate.is_symlink():
            return None
        temp_root = temp_root_path.resolve()
        candidate = lexical_candidate.resolve()
        prefix = run_file_staging_prefix(
            batch_id=batch_id,
            idempotency_identity=idempotency_identity,
        )
    except (OSError, RuntimeError):
        return None
    if candidate.parent != temp_root or not candidate.name.startswith(prefix):
        return None
    return candidate


def cleanup_exact_run_file_staging(
    *,
    temp_dir: Any,
    batch_id: str,
    idempotency_identity: str,
    authoritative_temp_dir: Any = None,
) -> str:
    """Classify cleanup of one exact persisted path without scanning the temp root."""
    candidate = validated_run_file_staging_dir(
        temp_dir=temp_dir,
        batch_id=batch_id,
        idempotency_identity=idempotency_identity,
    )
    if candidate is None:
        return "invalid"
    try:
        if isinstance(authoritative_temp_dir, str) and authoritative_temp_dir:
            authoritative = validated_run_file_staging_dir(
                temp_dir=authoritative_temp_dir,
                batch_id=batch_id,
                idempotency_identity=idempotency_identity,
            )
            if authoritative is None or candidate == authoritative:
                return "protected"
        if not candidate.exists():
            return "absent"
        shutil.rmtree(candidate)
        return "deleted" if not candidate.exists() else "failed"
    except (OSError, RuntimeError):
        return "failed"


__all__ = [
    "cleanup_exact_run_file_staging",
    "run_file_staging_prefix",
    "validated_run_file_staging_dir",
]
