"""Canonical hashing helpers for prompt-bearing assets."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import unicodedata
from typing import Any

from tldw_Server_API.app.core.Context_Integrity.models import CanonicalDigest


def _normalize_text(value: str) -> str:
    return unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return _normalize_text(value)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite floats are not supported in canonical JSON")
        return value
    if isinstance(value, Mapping):
        for key in value:
            if not isinstance(key, str):
                raise TypeError("Canonical JSON mapping keys must be strings")
        return {key: _normalize_json_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    raise TypeError(f"Unsupported canonical JSON value type: {type(value).__name__}")


def _stable_json(payload: Mapping[str, Any]) -> str:
    normalized = _normalize_json_value(payload)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def canonical_filesystem_digest(
    *,
    source_type: str,
    asset_id: str,
    files: Mapping[str, bytes],
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Hash raw file bytes plus deterministic identity metadata."""
    hasher = hashlib.sha256()
    identity = _stable_json(
        {
            "asset_id": asset_id,
            "source_type": source_type,
            "metadata": dict(metadata or {}),
        }
    ).encode("utf-8")
    hasher.update(len(identity).to_bytes(8, "big"))
    hasher.update(identity)
    for relative_path in sorted(files):
        path_bytes = relative_path.replace("\\", "/").encode("utf-8")
        content = files[relative_path]
        hasher.update(len(path_bytes).to_bytes(8, "big"))
        hasher.update(path_bytes)
        hasher.update(len(content).to_bytes(8, "big"))
        hasher.update(content)
    return "sha256:" + hasher.hexdigest()


def canonical_db_prompt_digest(record: Mapping[str, Any]) -> CanonicalDigest:
    """Hash a stable prompt-version JSON representation."""
    canonical_json = _stable_json(dict(record))
    return CanonicalDigest(
        digest=_sha256(canonical_json.encode("utf-8")),
        canonical_json=canonical_json,
    )
