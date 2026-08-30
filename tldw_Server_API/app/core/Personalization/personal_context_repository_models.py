"""Typed storage values and sanitized Personal Context repository errors."""

from __future__ import annotations

from dataclasses import dataclass


class ProfileStorageLockedError(RuntimeError):
    """Report unavailable or unauthenticated server profile key material."""


class ProfileIntegrityError(RuntimeError):
    """Report canonical or encrypted object authentication failure."""


class ProfileUnsupportedSchemaError(ProfileIntegrityError):
    """Report authenticated profile data from an unsupported newer schema."""


class ProfileAlreadyExistsError(RuntimeError):
    """Report an attempt to create a second profile in one user database."""


class ProfileKeyAlreadyExistsError(RuntimeError):
    """Report an attempt to replace existing wrapped profile keys."""


class ConcurrentProfileUpdateError(RuntimeError):
    """Report an optimistic object-head mismatch."""


class ProfileSemanticKeyCollisionError(RuntimeError):
    """Report an active same-scope canonical semantic-key collision."""


class ProfileQuotaExceededError(RuntimeError):
    """Report a bounded Personal Context operational quota violation."""


@dataclass(frozen=True, slots=True)
class ProfileKeyMaterial:
    """Decrypted per-profile encryption and canonical-integrity keys."""

    encryption_key: bytes
    integrity_key: bytes
    key_version: int = 1
    integrity_key_version: int = 1
