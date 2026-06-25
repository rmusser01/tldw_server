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

    def verify(self, payload: bytes, signature: str) -> bool:
        return hmac.compare_digest(self.sign(payload), signature)


def _stable_json(payload: Mapping[str, Any]) -> bytes:
    return _canonical_stable_json(payload).encode("utf-8")


def _manifest_digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def create_signed_manifest(
    *,
    sequence: int,
    entries: list[dict[str, Any]],
    signer: HmacManifestSigner,
    schema_version: int = 1,
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
            "alg": "hmac-sha256",
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
    manifest = signed_manifest.get("manifest")
    signature = signed_manifest.get("signature")
    if not isinstance(manifest, dict) or not isinstance(signature, dict):
        raise ManifestSignatureError("signed manifest is malformed")
    if signature.get("key_id") != signer.key_id:
        raise ManifestSignatureError("manifest key id mismatch")

    payload = _stable_json(manifest)
    expected_digest = _manifest_digest(payload)
    if signed_manifest.get("manifest_digest") != expected_digest:
        raise ManifestSignatureError("manifest digest mismatch")

    signature_value = signature.get("value")
    if not isinstance(signature_value, str) or not signer.verify(payload, signature_value):
        raise ManifestSignatureError("manifest signature mismatch")

    sequence = int(manifest.get("sequence") or 0)
    if anti_rollback_anchor and (
        sequence < anti_rollback_anchor.sequence
        or (sequence == anti_rollback_anchor.sequence and expected_digest != anti_rollback_anchor.manifest_digest)
    ):
        raise ManifestRollbackError("manifest rollback detected")

    entries = manifest.get("entries") or []
    if not isinstance(entries, list):
        raise ManifestSignatureError("manifest entries must be a list")
    return VerifiedManifest(
        sequence=sequence,
        manifest_digest=expected_digest,
        key_id=signer.key_id,
        entries=tuple(dict(item) for item in entries),
    )
