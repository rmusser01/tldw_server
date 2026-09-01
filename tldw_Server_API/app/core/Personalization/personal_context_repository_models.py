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


__all__ = [
    "ConcurrentProfileUpdateError",
    "ProfileAlreadyExistsError",
    "ProfileIntegrityError",
    "ProfileKeyAlreadyExistsError",
    "ProfileKeyMaterial",
    "ProfileQuotaExceededError",
    "ProfileSemanticKeyCollisionError",
    "ProfileStorageLockedError",
    "ProfileUnsupportedSchemaError",
]
