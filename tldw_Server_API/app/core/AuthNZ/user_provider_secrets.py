from __future__ import annotations

import json
from typing import Any

from tldw_Server_API.app.core.AuthNZ.exceptions import UserRegistrationException
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.LLM_Calls.provider_identity import canonical_provider_name
from tldw_Server_API.app.core.Security.crypto import (
    decrypt_json_blob_with_key,
    encrypt_json_blob_with_key,
)


def normalize_provider_name(provider: str) -> str:
    return (provider or "").strip().lower()


class ProviderCredentialAliasConflictError(ValueError, UserRegistrationException):
    """Raised when more than one legacy alias row exists for one provider."""


def fold_provider_credential_rows(
    rows: list[dict[str, Any]],
    *,
    identity_fields: tuple[str, ...] = (),
    include_revoked: bool = False,
) -> list[dict[str, Any]]:
    """Select one authoritative row per canonical provider identity."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        stored_provider = str(row.get("provider") or "")
        canonical = canonical_provider_name(stored_provider)
        key = tuple(row.get(field) for field in identity_fields) + (canonical,)
        grouped.setdefault(key, []).append(row)

    selected: list[dict[str, Any]] = []
    for key, candidates in grouped.items():
        canonical = str(key[-1])
        canonical_rows = [
            row
            for row in candidates
            if str(row.get("provider") or "").strip().lower() == canonical
        ]
        if len(canonical_rows) > 1:
            raise ProviderCredentialAliasConflictError("conflicting canonical provider credentials")
        if canonical_rows:
            row = canonical_rows[0]
        elif len(candidates) == 1:
            row = candidates[0]
        else:
            raise ProviderCredentialAliasConflictError("conflicting legacy provider credentials")
        if not include_revoked and row.get("revoked_at") is not None:
            continue
        materialized = dict(row)
        materialized["provider"] = canonical
        selected.append(materialized)
    return selected


def normalize_secret_owner_scope_type(scope_type: str) -> str:
    normalized = (scope_type or "").strip().lower()
    if normalized in {"user", "users"}:
        return "user"
    if normalized in {"org", "organization", "organizations", "orgs"}:
        return "org"
    if normalized in {"team", "teams"}:
        return "team"
    raise ValueError(f"Invalid secret owner scope type: {scope_type}")


def build_managed_secret_backend_ref(scope_type: str, scope_id: int, provider: str) -> str:
    scope_norm = normalize_secret_owner_scope_type(scope_type)
    return f"{scope_norm}:{int(scope_id)}:{normalize_provider_name(provider)}"


def key_hint_for_api_key(api_key: str) -> str:
    api_key = api_key or ""
    if len(api_key) <= 4:
        return api_key
    return api_key[-4:]


def build_secret_payload(
    api_key: str,
    credential_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"api_key": api_key}
    if credential_fields:
        payload["credential_fields"] = credential_fields
    return payload


def _get_byok_keys() -> tuple[str | None, str | None]:
    settings = get_settings()
    primary = getattr(settings, "BYOK_ENCRYPTION_KEY", None)
    secondary = getattr(settings, "BYOK_SECONDARY_ENCRYPTION_KEY", None)
    return primary, secondary


def encrypt_byok_payload(payload: dict[str, Any]) -> dict[str, Any]:
    primary, _secondary = _get_byok_keys()
    if not primary:
        raise ValueError("BYOK_ENCRYPTION_KEY is not configured")
    envelope = encrypt_json_blob_with_key(payload, primary)
    if not envelope:
        raise ValueError("Failed to encrypt BYOK payload")
    return envelope


def decrypt_byok_payload(envelope: dict[str, Any]) -> dict[str, Any]:
    primary, secondary = _get_byok_keys()
    if not primary and not secondary:
        raise ValueError("BYOK_ENCRYPTION_KEY is not configured")

    if primary:
        payload = decrypt_json_blob_with_key(envelope, primary)
        if payload is not None:
            return payload
    if secondary:
        payload = decrypt_json_blob_with_key(envelope, secondary)
        if payload is not None:
            return payload

    raise ValueError("Failed to decrypt BYOK payload")


def dumps_envelope(envelope: dict[str, Any]) -> str:
    return json.dumps(envelope)


def loads_envelope(encrypted_blob: str) -> dict[str, Any]:
    if not encrypted_blob:
        return {}
    return json.loads(encrypted_blob)
