from __future__ import annotations

"""
Lightweight AES-GCM helpers for field-level encryption of JSON blobs.

Uses PyCryptodome (pycryptodomex) which is already a dependency of this project.

Env:
  - WORKFLOWS_ARTIFACT_ENC_KEY: base64-encoded 32-byte key (AES-256)
"""

import base64
import binascii
import json
import os
from typing import Any

from loguru import logger

try:
    from Cryptodome.Cipher import AES
    from Cryptodome.Random import get_random_bytes
    _HAS_CRYPTO = True
except ImportError:
    _HAS_CRYPTO = False

_B64_DECODE_EXCEPTIONS = (binascii.Error, TypeError, ValueError)
_JSON_CRYPTO_DECRYPT_EXCEPTIONS = (
    json.JSONDecodeError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
)
_JSON_CRYPTO_WITH_B64_EXCEPTIONS = _B64_DECODE_EXCEPTIONS + _JSON_CRYPTO_DECRYPT_EXCEPTIONS


def _get_key_from_env_var(env_name: str) -> bytes | None:
    key_b64 = os.getenv(env_name, "").strip()
    if not key_b64:
        return None
    try:
        raw = base64.b64decode(key_b64, validate=True)
    except _B64_DECODE_EXCEPTIONS:
        logger.error(f"{env_name} is set but invalid; expected strict base64 encoding of a 32-byte AES-256 key")
        return None
    if len(raw) != 32:
        logger.error(f"{env_name} is set but invalid; expected strict base64 encoding of a 32-byte AES-256 key")
        return None
    return raw


def _get_key_from_env() -> bytes | None:
    return _get_key_from_env_var("WORKFLOWS_ARTIFACT_ENC_KEY")


def _get_secondary_key_from_env() -> bytes | None:
    """Optional fallback key for dual-read stage during key rotation."""
    return _get_key_from_env_var("JOBS_CRYPTO_SECONDARY_KEY")


def encrypt_json_blob(data: dict[str, Any]) -> dict[str, Any] | None:
    """Encrypt a JSON-serializable dict and return an envelope, or None if disabled/unsupported."""
    if not _HAS_CRYPTO:
        return None
    key = _get_key_from_env()
    if not key:
        return None
    try:
        pt = json.dumps(data, default=str).encode("utf-8")
        nonce = get_random_bytes(12)
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        ct, tag = cipher.encrypt_and_digest(pt)
        return {
            "_enc": "aesgcm:v1",
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ct": base64.b64encode(ct).decode("ascii"),
            "tag": base64.b64encode(tag).decode("ascii"),
        }
    except (TypeError, ValueError):
        return None


def decrypt_json_blob(envelope: dict[str, Any]) -> dict[str, Any] | None:
    """Attempt to decrypt an envelope back to dict; returns None on failure."""
    if not _HAS_CRYPTO:
        return None
    if not isinstance(envelope, dict) or envelope.get("_enc") != "aesgcm:v1":
        return None
    primary = _get_key_from_env()
    secondary = _get_secondary_key_from_env()
    if not primary and not secondary:
        return None
    try:
        nonce_b = base64.b64decode(envelope.get("nonce", ""))
        ct_b = base64.b64decode(envelope.get("ct", ""))
        tag_b = base64.b64decode(envelope.get("tag", ""))
    except _B64_DECODE_EXCEPTIONS:
        return None
    # Try primary key first
    for key in (primary, secondary):
        if not key:
            continue
        try:
            cipher = AES.new(key, AES.MODE_GCM, nonce=nonce_b)
            pt = cipher.decrypt_and_verify(ct_b, tag_b)
            return json.loads(pt.decode("utf-8"))
        except _JSON_CRYPTO_DECRYPT_EXCEPTIONS:
            continue
    return None


def _decode_key_b64(key_b64: str) -> bytes | None:
    try:
        raw = base64.b64decode(key_b64, validate=True)
    except _B64_DECODE_EXCEPTIONS:
        return None
    return raw if len(raw) == 32 else None


def encrypt_json_blob_with_key(data: dict[str, Any], key_b64: str) -> dict[str, Any] | None:
    """Encrypt using an explicit base64-encoded key (AES-GCM)."""
    if not _HAS_CRYPTO:
        return None
    key = _decode_key_b64(key_b64)
    if not key:
        return None
    try:
        pt = json.dumps(data, default=str).encode("utf-8")
        nonce = get_random_bytes(12)
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        ct, tag = cipher.encrypt_and_digest(pt)
        return {
            "_enc": "aesgcm:v1",
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ct": base64.b64encode(ct).decode("ascii"),
            "tag": base64.b64encode(tag).decode("ascii"),
        }
    except (TypeError, ValueError):
        return None


def decrypt_json_blob_with_key(envelope: dict[str, Any], key_b64: str) -> dict[str, Any] | None:
    """Decrypt using an explicit base64-encoded key; returns dict or None."""
    if not _HAS_CRYPTO:
        return None
    if not isinstance(envelope, dict) or envelope.get("_enc") != "aesgcm:v1":
        return None
    key = _decode_key_b64(key_b64)
    if not key:
        return None
    try:
        nonce = base64.b64decode(envelope.get("nonce", ""))
        ct = base64.b64decode(envelope.get("ct", ""))
        tag = base64.b64decode(envelope.get("tag", ""))
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        pt = cipher.decrypt_and_verify(ct, tag)
        return json.loads(pt.decode("utf-8"))
    except _JSON_CRYPTO_WITH_B64_EXCEPTIONS:
        return None
