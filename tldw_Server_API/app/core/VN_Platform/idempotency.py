"""Idempotency hashing helpers shared by VN platform endpoints."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import BinaryIO, Any


def canonical_payload_hash(payload: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 hash for JSON-like request payloads."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stream_sha256(stream: BinaryIO, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash bytes from the stream's current position without rewinding it."""
    digest = hashlib.sha256()
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def canonical_multipart_payload_hash(
    fields: Mapping[str, Any],
    *,
    file_sha256: str,
    filename: str | None = None,
    content_type: str | None = None,
) -> str:
    """Return a stable hash for multipart metadata plus the uploaded file digest."""
    return canonical_payload_hash(
        {
            "fields": dict(fields),
            "file": {
                "sha256": file_sha256,
                "filename": filename,
                "content_type": content_type,
            },
        }
    )
