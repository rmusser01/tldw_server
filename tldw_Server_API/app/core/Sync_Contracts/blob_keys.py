"""Sync blob storage keys and their validation, without filesystem access (moved from core.Sync.v2.blob_store)."""

from __future__ import annotations


class SyncBlobStoreError(ValueError):
    """Raised when blob storage input or integrity checks fail."""


def legacy_storage_key(payload_hash: str) -> str:
    """Return the exact historical global content-addressed storage key."""

    digest = _hash_digest(payload_hash)
    return f"blobs/sha256/{digest[:2]}/{digest}.blob"


def namespace_storage_key(
    storage_namespace_id: str,
    payload_hash: str,
) -> str:
    """Return a v2 path composed only from an opaque namespace and digest."""

    namespace = _storage_namespace_id(storage_namespace_id)
    digest = _hash_digest(payload_hash)
    return f"blobs/v2/{namespace}/{digest}.blob"


def _hash_digest(value: str) -> str:
    prefix = "sha256:"
    if not value.startswith(prefix) or len(value) != len(prefix) + 64:
        raise SyncBlobStoreError("payload_hash must be sha256:<64 hex chars>")
    digest = value[len(prefix) :]
    if any(character not in "0123456789abcdef" for character in digest):
        raise SyncBlobStoreError("payload_hash digest must be lowercase hex")
    return digest


def _storage_namespace_id(value: str) -> str:
    if len(value) != 32 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise SyncBlobStoreError(
            "storage_namespace_id must be 32 lowercase hexadecimal characters"
        )
    return value
