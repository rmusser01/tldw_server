"""Checksum and canonical fingerprint helpers for persona visual packs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_VOLATILE_FINGERPRINT_KEYS = {
    "archive_hash",
    "archive_sha256",
    "canonical_payload_fingerprint",
    "download_url",
    "export_id",
    "exported_at",
    "final_archive_hash",
    "final_archive_sha256",
}

_VOLATILE_FINGERPRINT_PATHS = {
    ("manifest", key) for key in _VOLATILE_FINGERPRINT_KEYS
} | {
    ("export", key) for key in _VOLATILE_FINGERPRINT_KEYS
} | {
    ("job", key) for key in _VOLATILE_FINGERPRINT_KEYS
} | {
    ("download", key) for key in _VOLATILE_FINGERPRINT_KEYS
}

_LIST_SORT_KEYS = (
    "asset_role",
    "source_asset_id",
    "checksum",
    "path",
    "id",
)

_UNORDERED_LIST_PATHS = {
    ("assets",),
    ("checksums",),
    ("files",),
}


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest for raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_stream(stream: Any, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 hex digest for a readable binary stream."""
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(chunk_size), b""):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest for a file without loading it all at once."""
    with Path(path).open("rb") as file_obj:
        return sha256_stream(file_obj)


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize JSON-compatible payloads deterministically as UTF-8 bytes."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_payload_fingerprint(payload: Mapping[str, Any]) -> str:
    """Return a stable fingerprint that ignores volatile export metadata."""
    canonical_payload = _canonical_payload_value(payload)
    return sha256_bytes(canonical_json_bytes(canonical_payload))


def build_persona_visual_pack_fingerprint(
    pack: Mapping[str, Any],
    assets: Sequence[Mapping[str, Any]],
) -> str:
    """Fingerprint the reviewed pack payload and reachable immutable assets."""
    payload = {
        "renderer_type": pack.get("renderer_type"),
        "manifest_version": pack.get("manifest_version"),
        "manifest": pack.get("manifest"),
        "companion_behavior": pack.get("companion_behavior"),
        "provenance": pack.get("provenance"),
        "provenance_version": pack.get("provenance_version"),
        "converter_version": pack.get("converter_version"),
        "assets": [
            _fingerprint_asset(asset)
            for asset in sorted(assets, key=lambda row: str(row.get("id") or ""))
        ],
    }
    return canonical_payload_fingerprint(payload)


def _fingerprint_asset(asset: Mapping[str, Any]) -> dict[str, Any]:
    """Project immutable asset metadata into the reviewed fingerprint payload."""
    return {
        key: asset.get(key)
        for key in (
            "id",
            "asset_role",
            "mime_type",
            "byte_size",
            "checksum_sha256",
            "width",
            "height",
            "duration_ms",
            "provenance",
            "provenance_version",
            "converter_version",
        )
    }


def _canonical_payload_value(value: Any, path: tuple[str, ...] = ()) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _canonical_payload_value(item, (*path, str(key)))
            for key, item in value.items()
            if (*path, str(key)) not in _VOLATILE_FINGERPRINT_PATHS
        }
    if isinstance(value, list):
        items = [_canonical_payload_value(item, path) for item in value]
        if _should_sort_list(path, items):
            return sorted(items, key=_canonical_sort_key)
        return items
    if isinstance(value, tuple):
        items = [_canonical_payload_value(item, path) for item in value]
        if _should_sort_list(path, items):
            return sorted(items, key=_canonical_sort_key)
        return items
    return value


def _should_sort_list(path: tuple[str, ...], items: list[Any]) -> bool:
    if path not in _UNORDERED_LIST_PATHS:
        return False
    return all(isinstance(item, Mapping) for item in items)


def _canonical_sort_key(value: Any) -> bytes:
    if isinstance(value, Mapping):
        semantic_key = tuple(value.get(key) for key in _LIST_SORT_KEYS)
        return canonical_json_bytes({"key": semantic_key, "value": value})
    return canonical_json_bytes(value)
