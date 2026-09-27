"""Typed storage values and sanitized Personal Context repository errors."""

from __future__ import annotations

from dataclasses import dataclass

from tldw_Server_API.app.core.exceptions import (
    ConcurrentProfileUpdateError,
    ProfileAlreadyExistsError,
    ProfileIntegrityError,
    ProfileKeyAlreadyExistsError,
    ProfileQuotaExceededError,
    ProfileSemanticKeyCollisionError,
    ProfileStorageLockedError,
    ProfileUnsupportedSchemaError,
)


@dataclass(frozen=True, slots=True)
class ProfileKeyMaterial:
    """Decrypted per-profile encryption and canonical-integrity keys."""

    encryption_key: bytes
    integrity_key: bytes
    key_version: int = 1
    integrity_key_version: int = 1


@dataclass(frozen=True, slots=True, repr=False)
class PreparedPersonalContextActivation:
    """Authenticated exact-head baseline; decrypted bytes never enter diagnostics."""

    profile_id: str
    device_id: str
    activation_id: str
    baseline_digest: str
    purge_generation: int
    publication_watermark: int
    baseline: bytes
    state: str
    sync_receipt_id: str | None
    home_server_cursor: int | None
    activation_epoch: str | None
    continuity_token: str | None


__all__ = [
    "ConcurrentProfileUpdateError",
    "ProfileAlreadyExistsError",
    "ProfileIntegrityError",
    "ProfileKeyAlreadyExistsError",
    "ProfileKeyMaterial",
    "PreparedPersonalContextActivation",
    "ProfileQuotaExceededError",
    "ProfileSemanticKeyCollisionError",
    "ProfileStorageLockedError",
    "ProfileUnsupportedSchemaError",
]
