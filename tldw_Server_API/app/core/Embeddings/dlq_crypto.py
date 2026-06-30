"""
DLQ payload optional encryption helpers.

Uses AES-GCM if `cryptography` is available and EMBEDDINGS_DLQ_ENCRYPTION_KEY is set.
Configured encryption fails closed when AES-GCM is unavailable.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.testing import env_flag_enabled

try:
    from cryptography.exceptions import InvalidTag  # type: ignore

    _DECRYPT_NONCRITICAL_EXCEPTIONS: tuple[type[Exception], ...] = (
        InvalidTag,
        TypeError,
        ValueError,
        binascii.Error,
    )
except ImportError:
    _DECRYPT_NONCRITICAL_EXCEPTIONS = (TypeError, ValueError, binascii.Error)

_SCRYPT_PARAMS = {
    "n": 2**14,
    "r": 8,
    "p": 1,
    "dklen": 32,
}


class DLQEncryptionError(RuntimeError):
    """Raised when configured DLQ encryption cannot produce an encrypted payload."""


def _derive_key_from_passphrase_legacy(passphrase: str) -> bytes:
    import hashlib
    return hashlib.sha256(passphrase.encode("utf-8")).digest()


def _derive_key_from_passphrase(passphrase: str, salt: bytes) -> tuple[bytes, str]:
    import hashlib
    try:
        key = hashlib.scrypt(
            passphrase.encode("utf-8"),
            salt=salt,
            n=_SCRYPT_PARAMS["n"],
            r=_SCRYPT_PARAMS["r"],
            p=_SCRYPT_PARAMS["p"],
            dklen=_SCRYPT_PARAMS["dklen"],
        )
        return key, "scrypt"
    except (ValueError, MemoryError) as exc:
        allow_weak = env_flag_enabled("EMBEDDINGS_DLQ_ALLOW_WEAK_KDF")
        logger.warning(
            'DLQ KDF scrypt failed; weak fallback {}. error={}',
            "enabled" if allow_weak else "disabled",
            f"{type(exc).__name__}: {exc}",
        )
        if not allow_weak:
            raise
        return _derive_key_from_passphrase_legacy(passphrase), "sha256"


def _aesgcm_encrypt(plaintext: bytes, key: bytes) -> dict[str, str]:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore
    except ImportError:
        raise DLQEncryptionError("DLQ encryption unavailable: AES-GCM support is not installed")
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, plaintext, associated_data=None)
    return {
        "alg": "AESGCM",
        "nonce": base64.b64encode(nonce).decode("utf-8"),
        "ct": base64.b64encode(ct).decode("utf-8"),
    }


def _aesgcm_decrypt(obj: dict[str, str], key: bytes) -> bytes | None:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore
    except ImportError:
        # Fallback: base64-only marker
        if obj.get("alg") == "none" and obj.get("b64"):
            try:
                return base64.b64decode(obj.get("b64") or "")
            except (TypeError, ValueError, binascii.Error):
                return None
        return None
    try:
        if obj.get("alg") != "AESGCM":
            if obj.get("alg") == "none" and obj.get("b64"):
                return base64.b64decode(obj.get("b64") or "")
            return None
        nonce = base64.b64decode(obj.get("nonce") or "")
        ct = base64.b64decode(obj.get("ct") or "")
        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, ct, associated_data=None)
    except _DECRYPT_NONCRITICAL_EXCEPTIONS:
        return None


def encrypt_payload_if_configured(payload_obj: dict[str, Any]) -> str | None:
    """Return JSON string of encrypted blob when configured; else None.

    When EMBEDDINGS_DLQ_ENCRYPTION_KEY is set, encrypts with AES-GCM.
    """
    key_str = os.getenv("EMBEDDINGS_DLQ_ENCRYPTION_KEY")
    if not key_str:
        return None
    try:
        salt = os.urandom(16)
        key, kdf_used = _derive_key_from_passphrase(key_str, salt)
        raw = json.dumps(payload_obj, default=str).encode("utf-8")
        obj = _aesgcm_encrypt(raw, key)
        if obj.get("alg") != "AESGCM":
            raise DLQEncryptionError("DLQ encryption unavailable: AES-GCM support is required")
        obj["kdf"] = kdf_used
        if kdf_used == "scrypt":
            obj["salt"] = base64.b64encode(salt).decode("utf-8")
            obj["kdf_params"] = _SCRYPT_PARAMS
        return json.dumps(obj)
    except (MemoryError, OSError, TypeError, ValueError) as exc:
        raise DLQEncryptionError("DLQ encryption failed") from exc


def decrypt_payload_if_present(enc_json: str | None) -> dict[str, Any] | None:
    """Attempt to decrypt an encrypted payload blob; returns dict or None."""
    if not enc_json:
        return None
    try:
        obj = json.loads(enc_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    key_str = os.getenv("EMBEDDINGS_DLQ_ENCRYPTION_KEY")
    if not key_str:
        # Without key, cannot decrypt; return None
        return None
    try:
        kdf = obj.get("kdf")
        if kdf == "scrypt":
            salt_b64 = obj.get("salt") or ""
            salt = base64.b64decode(salt_b64)
            stored_params = obj.get("kdf_params") or {}
            if not isinstance(stored_params, dict):
                stored_params = {}
            try:
                import hashlib
                key = hashlib.scrypt(
                    key_str.encode("utf-8"),
                    salt=salt,
                    n=int(stored_params.get("n", _SCRYPT_PARAMS["n"])),
                    r=int(stored_params.get("r", _SCRYPT_PARAMS["r"])),
                    p=int(stored_params.get("p", _SCRYPT_PARAMS["p"])),
                    dklen=int(stored_params.get("dklen", _SCRYPT_PARAMS["dklen"])),
                )
            except (ValueError, MemoryError, TypeError) as exc:
                logger.debug(
                    'DLQ decrypt scrypt failed, using legacy KDF: {}',
                    type(exc).__name__,
                )
                key = _derive_key_from_passphrase_legacy(key_str)
        else:
            key = _derive_key_from_passphrase_legacy(key_str)
        raw = _aesgcm_decrypt(obj, key)
        if raw is None:
            return None
        return json.loads(raw.decode("utf-8"))
    except (
        AttributeError,
        MemoryError,
        OSError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
        binascii.Error,
    ):
        return None
