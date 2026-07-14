from __future__ import annotations

import os
from typing import Any

from tldw_Server_API.app.core.AuthNZ.principal_model import (
    AuthContext,
    AuthPrincipal,
    is_single_user_principal,
)
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.LLM_Calls.provider_identity import canonical_provider_name
from tldw_Server_API.app.core.config import load_and_log_configs
from tldw_Server_API.app.core.custom_openai_providers import iter_custom_openai_provider_names
from tldw_Server_API.app.core.LLM_Calls.provider_metadata import get_byok_credential_policy

DEFAULT_BYOK_ALLOWED_PROVIDERS: set[str] = {
    "anthropic",
    "bedrock",
    "cohere",
    "custom-openai-api",
    "custom-openai-api-2",
    "deepseek",
    "elevenlabs",
    "google",
    "groq",
    "huggingface",
    "mistral",
    "moonshot",
    "openai",
    "openrouter",
    "novita",
    "poe",
    "qwen",
    "together",
    "voyage",
    "zai",
}
DEFAULT_BYOK_ALLOWED_PROVIDERS.update(iter_custom_openai_provider_names(start=3))
_PLATFORM_ADMIN_ROLES = frozenset({"admin", "owner", "super_admin"})
_ADMIN_CLAIM_PERMISSIONS = frozenset({"*", "system.configure"})


def _normalized_claim_values(values: list[Any] | tuple[Any, ...] | set[Any] | None) -> set[str]:
    return {
        str(value).strip().lower()
        for value in (values or [])
        if str(value).strip()
    }


def _principal_has_platform_admin_claims(principal: AuthPrincipal | None) -> bool:
    if not isinstance(principal, AuthPrincipal):
        return False
    roles = _normalized_claim_values(principal.roles)
    permissions = _normalized_claim_values(principal.permissions)
    if roles & _PLATFORM_ADMIN_ROLES:
        return True
    return bool(permissions & _ADMIN_CLAIM_PERMISSIONS)


def _legacy_user_has_platform_admin_claims(user: dict[str, Any] | None) -> bool:
    if not isinstance(user, dict):
        return False
    role = str(user.get("role") or "").strip().lower()
    roles = _normalized_claim_values(user.get("roles") or [])
    permissions = _normalized_claim_values(user.get("permissions") or [])
    if role in _PLATFORM_ADMIN_ROLES or roles & _PLATFORM_ADMIN_ROLES:
        return True
    return bool(permissions & _ADMIN_CLAIM_PERMISSIONS)


def resolve_byok_base_url_allowlist() -> set[str]:
    settings = get_settings()
    raw = getattr(settings, "BYOK_ALLOWED_BASE_URL_PROVIDERS", []) or []
    allowed = {canonical_provider_name(p) for p in raw if str(p).strip()}
    return allowed


def is_byok_enabled() -> bool:
    settings = get_settings()
    if not bool(settings.BYOK_ENABLED):
        return False
    # Self-disable when encryption key is not configured — returns 403
    # instead of 500 when endpoints are called without a key.
    return bool(settings.BYOK_ENCRYPTION_KEY)


def resolve_byok_allowlist() -> set[str]:
    settings = get_settings()
    raw = getattr(settings, "BYOK_ALLOWED_PROVIDERS", []) or []
    allowed = {canonical_provider_name(p) for p in raw if str(p).strip()}
    return allowed or set(DEFAULT_BYOK_ALLOWED_PROVIDERS)


def is_provider_allowlisted(provider: str) -> bool:
    provider_norm = canonical_provider_name(provider)
    return provider_norm in resolve_byok_allowlist()


def validate_credential_fields(
    provider: str,
    credential_fields: dict[str, Any] | None,
    *,
    allow_base_url: bool = False,
) -> dict[str, Any]:
    if credential_fields is None:
        return {}
    if not isinstance(credential_fields, dict):
        raise ValueError("credential_fields must be an object")

    provider_norm = canonical_provider_name(provider)
    allowed_keys, required_keys = get_byok_credential_policy(provider_norm)
    if allow_base_url and provider_norm in resolve_byok_base_url_allowlist():
        allowed_keys.add("base_url")
    cleaned: dict[str, Any] = {}
    for key, value in credential_fields.items():
        if key not in allowed_keys:
            raise ValueError(f"Unsupported credential field: {key}")
        if isinstance(value, str) and value.strip() == "":
            raise ValueError(f"Credential field '{key}' cannot be empty")
        cleaned[key] = value
    for required_key in required_keys:
        if required_key not in cleaned:
            raise ValueError(f"Credential field '{required_key}' is required")
    return cleaned


def is_trusted_base_url_principal(principal: AuthPrincipal | None) -> bool:
    if not isinstance(principal, AuthPrincipal):
        return False
    if _principal_has_platform_admin_claims(principal):
        return True
    if principal.kind == "service":
        return True
    return bool(is_single_user_principal(principal))


def is_trusted_base_url_request(
    request: Any = None,
    *,
    principal: AuthPrincipal | None = None,
    user: dict[str, Any] | None = None,
) -> bool:
    if principal is None and request is not None:
        try:
            ctx = getattr(getattr(request, "state", None), "auth", None)
        except Exception:
            ctx = None
        if isinstance(ctx, AuthContext):
            principal = ctx.principal

    if is_trusted_base_url_principal(principal):
        return True

    if _legacy_user_has_platform_admin_claims(user):
        return True

    return False


def derive_trusted_credential_scope(
    request: Any,
    current_user: Any,
) -> tuple[int | None, list[int], list[int], bool]:
    """Derive user and explicit active workspace IDs from authenticated state."""
    request_state = getattr(request, "state", None)
    auth_context = getattr(request_state, "auth", None)
    principal = getattr(auth_context, "principal", None)
    if principal is None and isinstance(current_user, AuthPrincipal):
        principal = current_user

    user_id_raw = getattr(principal, "user_id", None)
    if user_id_raw is None:
        user_id_raw = getattr(current_user, "id_int", None)
    if user_id_raw is None:
        user_id_raw = getattr(current_user, "id", None)
    try:
        user_id = int(user_id_raw) if user_id_raw is not None else None
    except (TypeError, ValueError):
        user_id = None

    def scope_ids(kind: str) -> list[int]:
        values = getattr(principal, f"{kind}_ids", None)
        if values is None:
            values = getattr(request_state, f"{kind}_ids", None)
        active = getattr(principal, f"active_{kind}_id", None)
        if active is None:
            active = getattr(request_state, f"active_{kind}_id", None)
        if active is None:
            return []
        try:
            members = {int(value) for value in (values or ()) if value is not None}
            active_id = int(active)
        except (TypeError, ValueError):
            from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionError

            raise ByokResolutionError("credential_scope_revoked", "credential_scope") from None
        if active_id not in members:
            from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionError

            raise ByokResolutionError("credential_scope_revoked", "credential_scope")
        return [active_id]

    return (
        user_id,
        scope_ids("team"),
        scope_ids("org"),
        is_trusted_base_url_principal(principal),
    )


def validate_base_url_override(base_url: Any) -> str:
    if not isinstance(base_url, str):
        raise ValueError("base_url must be a string")
    cleaned = base_url.strip()
    if not cleaned:
        raise ValueError("base_url cannot be empty")

    from tldw_Server_API.app.core.Security.egress import evaluate_url_policy

    result = evaluate_url_policy(cleaned)
    if not result.allowed:
        raise ValueError(result.reason or "URL blocked by security policy")
    return cleaned


def _provider_env_key(provider: str) -> str:
    normalized = provider.upper().replace(".", "_").replace("-", "_")
    if normalized.endswith("_API"):
        normalized = normalized[: -len("_API")]
    return f"{normalized}_API_KEY"


def resolve_server_default_key(provider: str) -> str | None:
    provider_norm = canonical_provider_name(provider)
    try:
        from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import get_llm_provider_override

        override = get_llm_provider_override(provider_norm)
        if override and override.api_key:
            return override.api_key
    except Exception as override_error:
        _ = override_error  # fallback to env/default key lookup
    env_key = _provider_env_key(provider_norm)
    env_val = os.getenv(env_key)
    if env_val is not None and env_val.strip():
        return env_val

    try:
        config = load_and_log_configs() or {}
    except Exception:
        config = {}

    section_key = f"{provider_norm}_api"
    section = config.get(section_key)
    if isinstance(section, dict):
        api_key = section.get("api_key")
        if api_key:
            return api_key

    return None
