"""Versioned authenticated envelopes for server Personal Context objects."""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ALGORITHM = "aes-256-gcm-v1"
KEY_BYTES = 32
NONCE_BYTES = 12


class EnvelopeAuthenticationError(RuntimeError):
    """Report an authenticated-envelope failure without crypto details."""


@dataclass(frozen=True, slots=True)
class EncryptedEnvelope:
    """A payload encrypted by a fresh DEK wrapped by one profile key."""

    algorithm: str
    nonce: bytes
    wrapped_dek: bytes
    wrapped_dek_nonce: bytes
    ciphertext: bytes
    key_version: int


class EnvelopeCipher:
    """Encrypt each canonical object with an independently wrapped DEK."""

    def __init__(self, profile_key: bytes, *, key_version: int = 1) -> None:
        if len(profile_key) != KEY_BYTES:
            raise ValueError("profile key must be exactly 32 bytes")
        if key_version < 1:
            raise ValueError("key_version must be positive")
        self._profile_key = profile_key
        self._key_version = key_version

    def encrypt(self, plaintext: bytes, aad: bytes) -> EncryptedEnvelope:
        """Return a fresh envelope bound to the supplied associated data."""

        dek = secrets.token_bytes(KEY_BYTES)
        nonce = secrets.token_bytes(NONCE_BYTES)
        wrapped_dek_nonce = secrets.token_bytes(NONCE_BYTES)
        return EncryptedEnvelope(
            algorithm=ALGORITHM,
            nonce=nonce,
            wrapped_dek=AESGCM(self._profile_key).encrypt(
                wrapped_dek_nonce,
                dek,
                aad,
            ),
            wrapped_dek_nonce=wrapped_dek_nonce,
            ciphertext=AESGCM(dek).encrypt(nonce, plaintext, aad),
            key_version=self._key_version,
        )

    def decrypt(self, envelope: EncryptedEnvelope, aad: bytes) -> bytes:
        """Authenticate and decrypt one envelope using the exact same AAD."""

        if envelope.algorithm != ALGORITHM:
            raise EnvelopeAuthenticationError("envelope authentication failed")
        if envelope.key_version != self._key_version:
            raise EnvelopeAuthenticationError("envelope authentication failed")
        if len(envelope.nonce) != NONCE_BYTES or len(envelope.wrapped_dek_nonce) != NONCE_BYTES:
            raise EnvelopeAuthenticationError("envelope authentication failed")
        try:
            dek = AESGCM(self._profile_key).decrypt(
                envelope.wrapped_dek_nonce,
                envelope.wrapped_dek,
                aad,
            )
            return AESGCM(dek).decrypt(envelope.nonce, envelope.ciphertext, aad)
        except (InvalidTag, TypeError, ValueError):
            raise EnvelopeAuthenticationError("envelope authentication failed") from None

    def rewrap(
        self,
        envelope: EncryptedEnvelope,
        aad: bytes,
        new_profile_key: bytes,
        *,
        new_key_version: int,
    ) -> EncryptedEnvelope:
        """Authenticate an envelope and rewrap only its data-encryption key."""

        if len(new_profile_key) != KEY_BYTES or new_key_version < 1:
            raise ValueError("new profile key material is invalid")
        if envelope.algorithm != ALGORITHM or envelope.key_version != self._key_version:
            raise EnvelopeAuthenticationError("envelope authentication failed")
        if len(envelope.nonce) != NONCE_BYTES or len(envelope.wrapped_dek_nonce) != NONCE_BYTES:
            raise EnvelopeAuthenticationError("envelope authentication failed")
        try:
            dek = AESGCM(self._profile_key).decrypt(
                envelope.wrapped_dek_nonce,
                envelope.wrapped_dek,
                aad,
            )
            AESGCM(dek).decrypt(envelope.nonce, envelope.ciphertext, aad)
            wrapped_dek_nonce = secrets.token_bytes(NONCE_BYTES)
            wrapped_dek = AESGCM(new_profile_key).encrypt(
                wrapped_dek_nonce,
                dek,
                aad,
            )
        except (InvalidTag, TypeError, ValueError):
            raise EnvelopeAuthenticationError("envelope authentication failed") from None
        return EncryptedEnvelope(
            algorithm=envelope.algorithm,
            nonce=envelope.nonce,
            wrapped_dek=wrapped_dek,
            wrapped_dek_nonce=wrapped_dek_nonce,
            ciphertext=envelope.ciphertext,
            key_version=new_key_version,
        )


def wrap_key(master_key: bytes, key: bytes, aad: bytes) -> tuple[bytes, bytes]:
    """Wrap one 32-byte key with a random 96-bit AES-GCM nonce."""

    if len(master_key) != KEY_BYTES or len(key) != KEY_BYTES:
        raise ValueError("keys must be exactly 32 bytes")
    nonce = secrets.token_bytes(NONCE_BYTES)
    return nonce, AESGCM(master_key).encrypt(nonce, key, aad)


def unwrap_key(master_key: bytes, nonce: bytes, wrapped: bytes, aad: bytes) -> bytes:
    """Authenticate and unwrap one 32-byte key without exposing crypto errors."""

    if len(master_key) != KEY_BYTES or len(nonce) != NONCE_BYTES:
        raise EnvelopeAuthenticationError("key material is unavailable")
    try:
        key = AESGCM(master_key).decrypt(nonce, wrapped, aad)
    except (InvalidTag, TypeError, ValueError):
        raise EnvelopeAuthenticationError("key material is unavailable") from None
    if len(key) != KEY_BYTES:
        raise EnvelopeAuthenticationError("key material is unavailable")
    return key
