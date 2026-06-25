"""Signed context integrity manifest helpers."""

from __future__ import annotations

import base64
import hashlib
import hmac
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tldw_Server_API.app.core.Context_Integrity.canonicalization import (
    _stable_json as _canonical_stable_json,
)

_SIGNATURE_ALGORITHM = "hmac-sha256"
_SUPPORTED_SCHEMA_VERSION = 1
_BASE64URL_SIGNATURE_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")
_REQUIRED_ENTRY_STRING_FIELDS = ("asset_id", "source_type", "digest", "display_name", "owner_scope")
_NON_EMPTY_ENTRY_STRING_FIELDS = frozenset(("asset_id", "source_type", "digest", "display_name"))
_REQUIRED_ENTRY_BOOL_FIELDS = ("executable", "required")


class ManifestSignatureError(ValueError):
    """Raised when a manifest signature cannot be verified."""


class ManifestRollbackError(ValueError):
    """Raised when a valid manifest is older than the anti-rollback anchor."""


@dataclass(frozen=True, slots=True)
class AntiRollbackAnchor:
    """Last accepted manifest identity from a non-DB trust anchor."""

    sequence: int
    manifest_digest: str


@dataclass(frozen=True, slots=True)
class VerifiedManifest:
    """Verified signed manifest payload."""

    sequence: int
    manifest_digest: str
    key_id: str
    entries: tuple[dict[str, Any], ...]


class HmacManifestSigner:
    """Test and deployment signer backed by an externally supplied secret."""

    def __init__(self, *, key_id: str, secret: bytes) -> None:
        if not key_id:
            raise ValueError("key_id is required")
        if not secret:
            raise ValueError("secret is required")
        self.key_id = key_id
        self._secret = secret

    def sign(self, payload: bytes) -> str:
        digest = hmac.new(self._secret, payload, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")

    def verify(self, payload: bytes, signature: object) -> bool:
        if not isinstance(signature, str):
            return False
        try:
            signature.encode("ascii")
        except UnicodeEncodeError:
            return False
        if not signature or any(char not in _BASE64URL_SIGNATURE_CHARS for char in signature):
            return False
        try:
            return hmac.compare_digest(self.sign(payload), signature)
        except TypeError:
            return False


def _stable_json(payload: Mapping[str, Any]) -> bytes:
    return _canonical_stable_json(payload).encode("utf-8")


def _manifest_digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _require_int_field(manifest: Mapping[str, Any], field_name: str) -> int:
    if field_name not in manifest:
        raise ManifestSignatureError(f"manifest {field_name} is required")
    value = manifest[field_name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestSignatureError(f"manifest {field_name} must be an integer")
    return value


def _require_entry_string(entry: Mapping[str, Any], field_name: str) -> None:
    if field_name not in entry:
        raise ManifestSignatureError(f"manifest entry {field_name} is required")
    value = entry[field_name]
    if not isinstance(value, str):
        raise ManifestSignatureError(f"manifest entry {field_name} must be a string")
    if field_name in _NON_EMPTY_ENTRY_STRING_FIELDS and not value:
        raise ManifestSignatureError(f"manifest entry {field_name} must not be empty")


def _require_entry_bool(entry: Mapping[str, Any], field_name: str) -> None:
    if field_name not in entry:
        raise ManifestSignatureError(f"manifest entry {field_name} is required")
    if not isinstance(entry[field_name], bool):
        raise ManifestSignatureError(f"manifest entry {field_name} must be a boolean")


def _validate_optional_entry_fields(entry: Mapping[str, Any]) -> None:
    if "path" in entry and entry["path"] is not None and not isinstance(entry["path"], str):
        raise ManifestSignatureError("manifest entry path must be a string or null")
    if "metadata" in entry:
        metadata = entry["metadata"]
        if not isinstance(metadata, Mapping):
            raise ManifestSignatureError("manifest entry metadata must be an object")
        for key in metadata:
            if not isinstance(key, str):
                raise ManifestSignatureError("manifest entry metadata keys must be strings")
        try:
            _stable_json({"metadata": metadata})
        except (TypeError, ValueError) as exc:
            raise ManifestSignatureError("manifest entry metadata is not canonicalizable") from exc


def _validate_entry_schema(entry: Mapping[str, Any]) -> dict[str, Any]:
    for field_name in _REQUIRED_ENTRY_STRING_FIELDS:
        _require_entry_string(entry, field_name)
    for field_name in _REQUIRED_ENTRY_BOOL_FIELDS:
        _require_entry_bool(entry, field_name)
    _validate_optional_entry_fields(entry)
    return dict(entry)


def _require_entries(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if "entries" not in manifest:
        raise ManifestSignatureError("manifest entries are required")
    entries = manifest["entries"]
    if not isinstance(entries, list):
        raise ManifestSignatureError("manifest entries must be a list")
    verified_entries: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ManifestSignatureError("manifest entries must be objects")
        for key in entry:
            if not isinstance(key, str):
                raise ManifestSignatureError("manifest entry keys must be strings")
        verified_entries.append(_validate_entry_schema(entry))
    asset_ids = [entry["asset_id"] for entry in verified_entries]
    if asset_ids != sorted(asset_ids):
        raise ManifestSignatureError("manifest entries must be sorted by asset_id")
    return tuple(verified_entries)


def create_signed_manifest(
    *,
    sequence: int,
    entries: list[dict[str, Any]],
    signer: HmacManifestSigner,
    schema_version: int = _SUPPORTED_SCHEMA_VERSION,
) -> dict[str, Any]:
    manifest_entries = [dict(entry) for entry in entries]
    manifest = {
        "schema_version": schema_version,
        "sequence": sequence,
        "entries": sorted(manifest_entries, key=lambda item: str(item["asset_id"])),
    }
    payload = _stable_json(manifest)
    return {
        "manifest": manifest,
        "signature": {
            "alg": _SIGNATURE_ALGORITHM,
            "key_id": signer.key_id,
            "value": signer.sign(payload),
        },
        "manifest_digest": _manifest_digest(payload),
    }


def verify_signed_manifest(
    signed_manifest: Mapping[str, Any],
    *,
    signer: HmacManifestSigner,
    anti_rollback_anchor: AntiRollbackAnchor | None = None,
) -> VerifiedManifest:
    if not isinstance(signed_manifest, Mapping):
        raise ManifestSignatureError("signed manifest is malformed")

    manifest = signed_manifest.get("manifest")
    signature = signed_manifest.get("signature")
    if not isinstance(manifest, dict) or not isinstance(signature, dict):
        raise ManifestSignatureError("signed manifest is malformed")
    if signature.get("alg") != _SIGNATURE_ALGORITHM:
        raise ManifestSignatureError("manifest signature algorithm mismatch")
    if signature.get("key_id") != signer.key_id:
        raise ManifestSignatureError("manifest key id mismatch")

    try:
        payload = _stable_json(manifest)
    except (TypeError, ValueError) as exc:
        raise ManifestSignatureError("manifest payload is not canonicalizable") from exc
    expected_digest = _manifest_digest(payload)
    if signed_manifest.get("manifest_digest") != expected_digest:
        raise ManifestSignatureError("manifest digest mismatch")

    signature_value = signature.get("value")
    if not isinstance(signature_value, str) or not signer.verify(payload, signature_value):
        raise ManifestSignatureError("manifest signature mismatch")

    schema_version = _require_int_field(manifest, "schema_version")
    if schema_version != _SUPPORTED_SCHEMA_VERSION:
        raise ManifestSignatureError("unsupported manifest schema version")
    sequence = _require_int_field(manifest, "sequence")
    entries = _require_entries(manifest)
    if anti_rollback_anchor and (
        sequence < anti_rollback_anchor.sequence
        or (sequence == anti_rollback_anchor.sequence and expected_digest != anti_rollback_anchor.manifest_digest)
    ):
        raise ManifestRollbackError("manifest rollback detected")

    return VerifiedManifest(
        sequence=sequence,
        manifest_digest=expected_digest,
        key_id=signer.key_id,
        entries=entries,
    )
