"""Explicit Personal Context plaintext and recovery-export helpers."""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Mapping
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

PLAINTEXT_EXPORT_CONFIRMATION = "EXPORT PLAINTEXT"
RECOVERY_EXPORT_CONFIRMATION = "EXPORT RECOVERY"
RECOVERY_EXPORT_ALGORITHM = "scrypt-aes-256-gcm"
_RECOVERY_AAD = b"tldw-personal-context-recovery-v1"


def require_confirmation(actual: str, expected: str) -> None:
    """Reject an export unless its exact destructive-style phrase is supplied."""

    if actual != expected:
        raise ValueError(f"confirmation must be exactly {expected!r}")


def encrypt_recovery_export(
    snapshot: Mapping[str, Any],
    *,
    passphrase: str,
) -> dict[str, str]:
    """Encrypt one deterministic JSON snapshot with a user passphrase."""

    if len(passphrase) < 12:
        raise ValueError("recovery export passphrase must contain at least 12 characters")
    salt = os.urandom(16)
    nonce = os.urandom(12)
    key = Scrypt(salt=salt, length=32, n=2**14, r=8, p=1).derive(passphrase.encode("utf-8"))
    plaintext = json.dumps(
        snapshot,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, _RECOVERY_AAD)

    def encode(value: bytes) -> str:
        return base64.b64encode(value).decode("ascii")

    return {
        "algorithm": RECOVERY_EXPORT_ALGORITHM,
        "salt": encode(salt),
        "nonce": encode(nonce),
        "ciphertext": encode(ciphertext),
    }
