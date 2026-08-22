"""Dedicated contextual encryption for canonical admin webhook data."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from tldw_Server_API.app.core.Security.crypto import (
    decrypt_json_blob_with_key,
    encrypt_json_blob_with_key,
)

from .config import WEBHOOK_KEYS_ENV, WEBHOOK_PRIMARY_KEY_ID_ENV

EVENT_BODY_MAX_BYTES = 65_536
MIGRATION_DOMAIN_SYSTEM_OPS_SUBTREE = "system_ops.webhook_subtree"
MIGRATION_DOMAIN_SYSTEM_OPS_RECORD = "system_ops.webhook_record"
MIGRATION_DOMAIN_DATABASE_TABLE = "legacy_database.webhook_table"
MIGRATION_DOMAIN_DATABASE_RECORD = "legacy_database.webhook_record"

_MIGRATION_FINGERPRINT_DOMAINS = frozenset(
    {
        MIGRATION_DOMAIN_SYSTEM_OPS_SUBTREE,
        MIGRATION_DOMAIN_SYSTEM_OPS_RECORD,
        MIGRATION_DOMAIN_DATABASE_TABLE,
        MIGRATION_DOMAIN_DATABASE_RECORD,
    }
)
_KEY_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_PURPOSE_PATTERN = re.compile(r"^[a-z][a-z0-9._-]{0,127}$")
_IDENTITY_KEY_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_MAX_KEY_CONFIG_BYTES = 16_384
_MIGRATION_FINGERPRINT_PREFIX = b"tldw-admin-webhook-migration-v1\x00"


class WebhookKeyErrorCode(str, Enum):
    """Closed, secret-free key-ring failure codes."""

    KEY_UNAVAILABLE = "admin_webhook_key_unavailable"
    CONFIGURATION_INVALID = "admin_webhook_key_configuration_invalid"
    ENCRYPTION_FAILED = "admin_webhook_encryption_failed"
    DECRYPTION_FAILED = "admin_webhook_decryption_failed"
    CONTEXT_MISMATCH = "admin_webhook_envelope_context_mismatch"
    UNKNOWN_KEY = "admin_webhook_key_unknown"
    EVENT_BODY_TOO_LARGE = "admin_webhook_event_body_too_large"
    FINGERPRINT_DOMAIN_INVALID = "admin_webhook_migration_fingerprint_domain_invalid"


class WebhookKeyLoadCode(str, Enum):
    """Sanitized runtime key-ring availability state."""

    AVAILABLE = "available"
    KEY_UNAVAILABLE = WebhookKeyErrorCode.KEY_UNAVAILABLE.value
    CONFIGURATION_INVALID = WebhookKeyErrorCode.CONFIGURATION_INVALID.value


class WebhookKeyError(Exception):
    """Expected key-ring failure that exposes only a stable code."""

    def __init__(self, code: WebhookKeyErrorCode) -> None:
        self.code = code
        super().__init__(code.value)


@dataclass(frozen=True)
class ProtectedValue:
    """Serialized AES-GCM envelope and its declared key identity."""

    ciphertext_json: str
    key_id: str


@dataclass(frozen=True)
class WebhookKeyRingLoadResult:
    """Non-throwing runtime key-ring load result."""

    ring: WebhookKeyRing | None
    code: WebhookKeyLoadCode

    def require_ring(self) -> WebhookKeyRing:
        """Return the concrete ring or raise its closed load failure."""
        if self.ring is not None:
            return self.ring
        raise WebhookKeyError(WebhookKeyErrorCode(self.code.value))


class _JSONObjectPairs(list[tuple[str, object]]):
    """Distinguish a JSON object from a top-level array before validation."""


def _decode_configured_key(value: str) -> bytes:
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise WebhookKeyError(WebhookKeyErrorCode.CONFIGURATION_INVALID) from exc
    if len(raw) != 32 or base64.b64encode(raw).decode("ascii") != value:
        raise WebhookKeyError(WebhookKeyErrorCode.CONFIGURATION_INVALID)
    return raw


def _validate_key_pairs(raw_pairs: _JSONObjectPairs) -> dict[str, str]:
    keys: dict[str, str] = {}
    for key_id, encoded_key in raw_pairs:
        if key_id in keys:
            raise WebhookKeyError(WebhookKeyErrorCode.CONFIGURATION_INVALID)
        if _KEY_ID_PATTERN.fullmatch(key_id) is None or not isinstance(encoded_key, str):
            raise WebhookKeyError(WebhookKeyErrorCode.CONFIGURATION_INVALID)
        _decode_configured_key(encoded_key)
        keys[key_id] = encoded_key
    return keys


def _normalize_identity(identity: Mapping[str, str | int]) -> dict[str, str | int]:
    normalized: dict[str, str | int] = {}
    for key in sorted(identity):
        value = identity[key]
        if _IDENTITY_KEY_PATTERN.fullmatch(key) is None:
            raise WebhookKeyError(WebhookKeyErrorCode.CONTEXT_MISMATCH)
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise WebhookKeyError(WebhookKeyErrorCode.CONTEXT_MISMATCH)
        if isinstance(value, str) and (not value or len(value) > 1_024):
            raise WebhookKeyError(WebhookKeyErrorCode.CONTEXT_MISMATCH)
        normalized[key] = value
    if not normalized:
        raise WebhookKeyError(WebhookKeyErrorCode.CONTEXT_MISMATCH)
    return normalized


def _validate_purpose(purpose: str) -> str:
    if _PURPOSE_PATTERN.fullmatch(purpose) is None:
        raise WebhookKeyError(WebhookKeyErrorCode.CONTEXT_MISMATCH)
    return purpose


def _strict_envelope(ciphertext_json: str) -> dict[str, object]:
    try:
        raw = json.loads(ciphertext_json, object_pairs_hook=_JSONObjectPairs)
    except (json.JSONDecodeError, RecursionError, TypeError, ValueError) as exc:
        raise WebhookKeyError(WebhookKeyErrorCode.DECRYPTION_FAILED) from exc
    if not isinstance(raw, _JSONObjectPairs):
        raise WebhookKeyError(WebhookKeyErrorCode.DECRYPTION_FAILED)
    envelope: dict[str, object] = {}
    for key, value in raw:
        if key in envelope or not isinstance(value, str):
            raise WebhookKeyError(WebhookKeyErrorCode.DECRYPTION_FAILED)
        envelope[key] = value
    if set(envelope) != {"_enc", "nonce", "ct", "tag"}:
        raise WebhookKeyError(WebhookKeyErrorCode.DECRYPTION_FAILED)
    if envelope["_enc"] != "aesgcm:v1":
        raise WebhookKeyError(WebhookKeyErrorCode.DECRYPTION_FAILED)
    decoded_fields: dict[str, bytes] = {}
    for field in ("nonce", "ct", "tag"):
        encoded = envelope[field]
        if not isinstance(encoded, str):
            raise WebhookKeyError(WebhookKeyErrorCode.DECRYPTION_FAILED)
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise WebhookKeyError(WebhookKeyErrorCode.DECRYPTION_FAILED) from exc
        if base64.b64encode(decoded).decode("ascii") != encoded:
            raise WebhookKeyError(WebhookKeyErrorCode.DECRYPTION_FAILED)
        decoded_fields[field] = decoded
    if (
        len(decoded_fields["nonce"]) != 12
        or not decoded_fields["ct"]
        or len(decoded_fields["tag"]) != 16
    ):
        raise WebhookKeyError(WebhookKeyErrorCode.DECRYPTION_FAILED)
    return envelope


class WebhookKeyRing:
    """Immutable dedicated key ring with contextual webhook envelopes."""

    def __init__(self, keys: Mapping[str, str], *, primary_id: str) -> None:
        self._keys = MappingProxyType(dict(keys))
        self.primary_id = primary_id

    def has_key(self, key_id: str) -> bool:
        """Return whether one validated operator key ID is configured."""
        return isinstance(key_id, str) and key_id in self._keys

    @classmethod
    def from_environment(cls, environ: Mapping[str, str]) -> WebhookKeyRing:
        """Strictly load only the dedicated webhook key environment."""
        raw_json = environ.get(WEBHOOK_KEYS_ENV, "").strip()
        primary_id = environ.get(WEBHOOK_PRIMARY_KEY_ID_ENV, "").strip()
        if not raw_json or raw_json == "{}" or not primary_id:
            raise WebhookKeyError(WebhookKeyErrorCode.KEY_UNAVAILABLE)
        if len(raw_json.encode("utf-8")) > _MAX_KEY_CONFIG_BYTES:
            raise WebhookKeyError(WebhookKeyErrorCode.CONFIGURATION_INVALID)
        try:
            raw_pairs = json.loads(raw_json, object_pairs_hook=_JSONObjectPairs)
        except (json.JSONDecodeError, RecursionError, TypeError, ValueError) as exc:
            raise WebhookKeyError(WebhookKeyErrorCode.CONFIGURATION_INVALID) from exc
        if not isinstance(raw_pairs, _JSONObjectPairs):
            raise WebhookKeyError(WebhookKeyErrorCode.CONFIGURATION_INVALID)
        keys = _validate_key_pairs(raw_pairs)
        if not keys:
            raise WebhookKeyError(WebhookKeyErrorCode.KEY_UNAVAILABLE)
        if _KEY_ID_PATTERN.fullmatch(primary_id) is None or primary_id not in keys:
            raise WebhookKeyError(WebhookKeyErrorCode.CONFIGURATION_INVALID)
        return cls(keys, primary_id=primary_id)

    def _encrypt_bytes_to_key(
        self,
        *,
        purpose: str,
        identity: Mapping[str, str | int],
        plaintext: bytes,
        key_id: str,
    ) -> ProtectedValue:
        if not isinstance(plaintext, bytes):
            raise WebhookKeyError(WebhookKeyErrorCode.ENCRYPTION_FAILED)
        encoded_key = self._keys.get(key_id)
        if encoded_key is None:
            raise WebhookKeyError(WebhookKeyErrorCode.UNKNOWN_KEY)
        payload = {
            "schema": 1,
            "purpose": _validate_purpose(purpose),
            "identity": _normalize_identity(identity),
            "value_b64": base64.b64encode(plaintext).decode("ascii"),
        }
        envelope = encrypt_json_blob_with_key(payload, encoded_key)
        if envelope is None:
            raise WebhookKeyError(WebhookKeyErrorCode.ENCRYPTION_FAILED)
        return ProtectedValue(
            ciphertext_json=json.dumps(
                envelope,
                sort_keys=True,
                separators=(",", ":"),
            ),
            key_id=key_id,
        )

    def encrypt_bytes(
        self,
        *,
        purpose: str,
        identity: Mapping[str, str | int],
        plaintext: bytes,
    ) -> ProtectedValue:
        """Encrypt bytes under the ordinary-write primary key."""
        return self._encrypt_bytes_to_key(
            purpose=purpose,
            identity=identity,
            plaintext=plaintext,
            key_id=self.primary_id,
        )

    def decrypt_bytes(
        self,
        *,
        purpose: str,
        identity: Mapping[str, str | int],
        protected: ProtectedValue,
    ) -> bytes:
        """Decrypt and verify exact purpose and stable row identity."""
        encoded_key = self._keys.get(protected.key_id)
        if encoded_key is None:
            raise WebhookKeyError(WebhookKeyErrorCode.UNKNOWN_KEY)
        envelope = _strict_envelope(protected.ciphertext_json)
        payload = decrypt_json_blob_with_key(envelope, encoded_key)
        if payload is None or set(payload) != {
            "schema",
            "purpose",
            "identity",
            "value_b64",
        }:
            raise WebhookKeyError(WebhookKeyErrorCode.DECRYPTION_FAILED)
        expected_purpose = _validate_purpose(purpose)
        expected_identity = _normalize_identity(identity)
        actual_purpose = payload.get("purpose")
        actual_identity = payload.get("identity")
        if (
            payload.get("schema") != 1
            or not isinstance(actual_purpose, str)
            or not hmac.compare_digest(actual_purpose, expected_purpose)
            or not isinstance(actual_identity, dict)
            or actual_identity != expected_identity
        ):
            raise WebhookKeyError(WebhookKeyErrorCode.CONTEXT_MISMATCH)
        encoded_value = payload.get("value_b64")
        if not isinstance(encoded_value, str):
            raise WebhookKeyError(WebhookKeyErrorCode.DECRYPTION_FAILED)
        try:
            value = base64.b64decode(encoded_value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise WebhookKeyError(WebhookKeyErrorCode.DECRYPTION_FAILED) from exc
        if base64.b64encode(value).decode("ascii") != encoded_value:
            raise WebhookKeyError(WebhookKeyErrorCode.DECRYPTION_FAILED)
        return value

    def encrypt_text(
        self,
        *,
        purpose: str,
        identity: Mapping[str, str | int],
        plaintext: str,
    ) -> ProtectedValue:
        """Encode UTF-8 text and encrypt it under the primary key."""
        if not isinstance(plaintext, str):
            raise WebhookKeyError(WebhookKeyErrorCode.ENCRYPTION_FAILED)
        return self.encrypt_bytes(
            purpose=purpose,
            identity=identity,
            plaintext=plaintext.encode("utf-8"),
        )

    def decrypt_text(
        self,
        *,
        purpose: str,
        identity: Mapping[str, str | int],
        protected: ProtectedValue,
    ) -> str:
        """Decrypt bytes and require valid UTF-8 text."""
        plaintext = self.decrypt_bytes(
            purpose=purpose,
            identity=identity,
            protected=protected,
        )
        try:
            return plaintext.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WebhookKeyError(WebhookKeyErrorCode.DECRYPTION_FAILED) from exc

    def can_decrypt(
        self,
        *,
        purpose: str,
        identity: Mapping[str, str | int],
        protected: ProtectedValue,
    ) -> bool:
        """Return whether a protected value decrypts in the expected context."""
        try:
            self.decrypt_bytes(
                purpose=purpose,
                identity=identity,
                protected=protected,
            )
        except WebhookKeyError:
            return False
        return True

    def reencrypt_to_key(
        self,
        protected: ProtectedValue,
        *,
        purpose: str,
        identity: Mapping[str, str | int],
        target_key_id: str,
    ) -> ProtectedValue:
        """Validate and re-encrypt a value to an explicit rotation target."""
        plaintext = self.decrypt_bytes(
            purpose=purpose,
            identity=identity,
            protected=protected,
        )
        return self._encrypt_bytes_to_key(
            purpose=purpose,
            identity=identity,
            plaintext=plaintext,
            key_id=target_key_id,
        )

    def encrypt_event_body(
        self,
        *,
        event_id: str,
        api_version: str,
        body: bytes,
    ) -> ProtectedValue:
        """Encrypt an exact bounded event body under event identity."""
        if not isinstance(body, bytes):
            raise WebhookKeyError(WebhookKeyErrorCode.ENCRYPTION_FAILED)
        if len(body) > EVENT_BODY_MAX_BYTES:
            raise WebhookKeyError(WebhookKeyErrorCode.EVENT_BODY_TOO_LARGE)
        return self.encrypt_bytes(
            purpose="event.body",
            identity={"event_id": event_id, "api_version": api_version},
            plaintext=body,
        )

    def decrypt_event_body(
        self,
        *,
        event_id: str,
        api_version: str,
        protected: ProtectedValue,
    ) -> bytes:
        """Decrypt an event body only for its exact event/version context."""
        body = self.decrypt_bytes(
            purpose="event.body",
            identity={"event_id": event_id, "api_version": api_version},
            protected=protected,
        )
        if len(body) > EVENT_BODY_MAX_BYTES:
            raise WebhookKeyError(WebhookKeyErrorCode.EVENT_BODY_TOO_LARGE)
        return body

    def fingerprint_migration_source(
        self,
        domain: str,
        canonical_bytes: bytes,
    ) -> tuple[str, str]:
        """Authenticate migration source bytes under one closed domain."""
        if domain not in _MIGRATION_FINGERPRINT_DOMAINS:
            raise WebhookKeyError(WebhookKeyErrorCode.FINGERPRINT_DOMAIN_INVALID)
        if not isinstance(canonical_bytes, bytes):
            raise WebhookKeyError(WebhookKeyErrorCode.FINGERPRINT_DOMAIN_INVALID)
        raw_key = _decode_configured_key(self._keys[self.primary_id])
        payload = (
            _MIGRATION_FINGERPRINT_PREFIX
            + domain.encode("ascii")
            + b"\x00"
            + canonical_bytes
        )
        digest = hmac.new(raw_key, payload, hashlib.sha256).hexdigest()
        return self.primary_id, f"hmac-sha256:{digest}"


def load_webhook_key_ring(
    environ: Mapping[str, str] | None = None,
) -> WebhookKeyRingLoadResult:
    """Load a key ring for runtime status without leaking invalid input."""
    source = os.environ if environ is None else environ
    try:
        ring = WebhookKeyRing.from_environment(source)
    except WebhookKeyError as exc:
        if exc.code is WebhookKeyErrorCode.KEY_UNAVAILABLE:
            code = WebhookKeyLoadCode.KEY_UNAVAILABLE
        else:
            code = WebhookKeyLoadCode.CONFIGURATION_INVALID
        return WebhookKeyRingLoadResult(ring=None, code=code)
    return WebhookKeyRingLoadResult(ring=ring, code=WebhookKeyLoadCode.AVAILABLE)
