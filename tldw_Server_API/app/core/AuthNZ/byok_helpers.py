from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from tldw_Server_API.app.core.AuthNZ.byok_config import (
    PROVIDER_APP_CONFIG_KEYS,
    PROVIDER_RUNTIME_ENV_CONFIG_KEYS,
    normalize_runtime_environment_config_value,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import (
    AuthContext,
    AuthPrincipal,
    is_single_user_principal,
)
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.config import load_and_log_configs
from tldw_Server_API.app.core.custom_openai_providers import (
    custom_openai_api_key_env_keys,
    custom_openai_endpoint_env_keys,
    custom_openai_provider_number,
    custom_openai_section_name,
    iter_custom_openai_provider_names,
)
from tldw_Server_API.app.core.LLM_Calls.provider_identity import canonical_provider_name
from tldw_Server_API.app.core.LLM_Calls.provider_metadata import get_byok_credential_policy

DEFAULT_BYOK_ALLOWED_PROVIDERS: set[str] = {
    "anthropic",
    "bedrock",
    "cohere",
    "custom-openai-api",
    "custom-openai-api-2",
    "deepseek",
    "elevenlabs",
    "fish_s2",
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
_CREDENTIAL_HEADER_FIELDS = frozenset({"org_id", "project_id"})
_MAX_CREDENTIAL_HEADER_VALUE_LENGTH = 512
_RUNTIME_CONFIG_PLACEHOLDERS = frozenset(
    {
        "CHANGE_ME",
        "CHANGEME",
        "REPLACE_ME",
        "REPLACEME",
        "YOUR_API_KEY",
        "YOUR_API_KEY_HERE",
        "YOUR_ENDPOINT",
        "YOUR_MODEL",
        "YOUR_VALUE",
        "API_KEY",
    }
)


def _usable_runtime_config_value(value: object) -> str | None:
    """Return a trimmed runtime value unless the whole value is a placeholder."""
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    if not trimmed or (trimmed.startswith("<") and trimmed.endswith(">")):
        return None
    placeholder = "_".join(
        part for part in trimmed.upper().replace("-", "_").split("_") if part
    )
    if placeholder.startswith("CHANGE_ME") or placeholder in _RUNTIME_CONFIG_PLACEHOLDERS:
        return None
    return trimmed


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


def get_byok_gateway_specs() -> Mapping[str, Any]:
    """Return locally normalized gateway specs without discovery or network I/O."""
    from tldw_Server_API.app.core.TTS.tts_config import get_tts_config_manager

    return get_tts_config_manager().get_gateway_specs()


def get_byok_gateway_spec(provider: str) -> Any | None:
    """Return an enabled configured gateway for one canonical provider ID."""
    from tldw_Server_API.app.core.TTS.gateway_config import canonicalize_gateway_id

    try:
        backend_id = canonicalize_gateway_id(canonical_provider_name(provider))
    except ValueError:
        return None
    spec = get_byok_gateway_specs().get(backend_id)
    return spec if spec is not None and bool(spec.enabled) else None


def resolve_byok_allowlist() -> set[str]:
    settings = get_settings()
    raw = getattr(settings, "BYOK_ALLOWED_PROVIDERS", []) or []
    allowed = {canonical_provider_name(p) for p in raw if str(p).strip()}
    resolved = {
        provider
        for provider in (allowed or set(DEFAULT_BYOK_ALLOWED_PROVIDERS))
        if not provider.startswith("gateway:")
    }
    resolved.update(
        backend_id
        for backend_id, spec in get_byok_gateway_specs().items()
        if bool(spec.enabled) and bool(spec.allow_user_api_key)
    )
    return resolved


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
    gateway_spec = (
        get_byok_gateway_spec(provider_norm)
        if provider_norm.startswith("gateway:")
        else None
    )
    is_named_gateway = provider_norm.startswith("gateway:") and gateway_spec is not None
    if is_named_gateway:
        allowed_keys, required_keys = set(), set()
    else:
        allowed_keys, required_keys = get_byok_credential_policy(provider_norm)
    if (
        not is_named_gateway
        and allow_base_url
        and provider_norm in resolve_byok_base_url_allowlist()
    ):
        allowed_keys.add("base_url")
    cleaned: dict[str, Any] = {}
    for key, value in credential_fields.items():
        if key not in allowed_keys:
            raise ValueError(f"Unsupported credential field: {key}")
        if key in _CREDENTIAL_HEADER_FIELDS:
            if not isinstance(value, str):
                raise ValueError(f"Credential field '{key}' must be a string")
            value = value.strip()
            if (
                not value
                or len(value) > _MAX_CREDENTIAL_HEADER_VALUE_LENGTH
                or not value.isascii()
                or any(ord(char) < 32 or ord(char) == 127 for char in value)
            ):
                raise ValueError(f"Credential field '{key}' is not a safe header value")
        elif isinstance(value, str) and value.strip() == "":
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

    return bool(_legacy_user_has_platform_admin_claims(user))
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


def load_server_config_snapshot() -> dict[str, Any]:
    """Load one server configuration generation for credential resolution."""
    environment_snapshot = dict(os.environ)
    captured_sections: dict[str, dict[str, Any]] = {}
    for provider, field_aliases in PROVIDER_RUNTIME_ENV_CONFIG_KEYS.items():
        section = PROVIDER_APP_CONFIG_KEYS.get(provider)
        if section is None:
            continue
        captured: dict[str, Any] = {}
        for field, aliases in field_aliases.items():
            raw_value = _first_nonempty_environment_value(
                aliases,
                environment_snapshot,
            )
            if raw_value is None:
                continue
            value = normalize_runtime_environment_config_value(
                provider,
                field,
                raw_value,
            )
            if value is not None:
                captured[field] = value
        if captured:
            captured_sections[section] = captured

    for provider in iter_custom_openai_provider_names():
        number = custom_openai_provider_number(provider)
        if number is None:
            continue
        captured: dict[str, str] = {}
        api_key = _first_nonempty_environment_value(
            custom_openai_api_key_env_keys(number),
            environment_snapshot,
        )
        endpoint = _first_nonempty_environment_value(
            custom_openai_endpoint_env_keys(number),
            environment_snapshot,
        )
        if api_key is not None:
            captured["api_key"] = api_key
        if endpoint is not None:
            captured["api_ip"] = endpoint
        if captured:
            captured_sections[custom_openai_section_name(number)] = captured

    try:
        snapshot = load_and_log_configs(environment=environment_snapshot) or {}
    except Exception:
        snapshot = {}
    result = dict(snapshot) if isinstance(snapshot, dict) else {}
    for section, captured in captured_sections.items():
        existing = result.get(section)
        merged = dict(existing) if isinstance(existing, dict) else {}
        # Environment precedence is captured before the loader starts so a
        # concurrent rotation cannot cross key and endpoint generations.
        merged.update(captured)
        result[section] = merged
    _remove_provider_placeholders(result)
    return result


def _remove_provider_placeholders(snapshot: dict[str, Any]) -> None:
    """Remove whole-value placeholders from provider sections in place."""
    provider_sections = set(PROVIDER_APP_CONFIG_KEYS.values())
    provider_sections.update(
        custom_openai_section_name(number)
        for number in range(1, 100)
    )
    provider_sections.add("api_keys")
    for section_name in provider_sections:
        section = snapshot.get(section_name)
        if not isinstance(section, dict):
            continue
        snapshot[section_name] = {
            key: value
            for key, value in section.items()
            if not isinstance(value, str)
            or _usable_runtime_config_value(value) is not None
        }


def _first_nonempty_environment_value(
    names: tuple[str, ...],
    environment: Mapping[str, str],
) -> str | None:
    """Return the first non-empty environment value from ordered aliases."""
    for name in names:
        value = _usable_runtime_config_value(environment.get(name))
        if value is not None:
            return value
    return None


def resolve_server_default_key_from_snapshot(
    provider: str,
    config_snapshot: dict[str, Any],
) -> str | None:
    """Resolve one server key without loading a second config generation."""
    provider_norm = canonical_provider_name(provider)
    custom_number = custom_openai_provider_number(provider_norm)
    section_key = PROVIDER_APP_CONFIG_KEYS.get(provider_norm)
    if section_key is None and custom_number is not None:
        section_key = custom_openai_section_name(custom_number)
    if section_key is None:
        section_key = f"{provider_norm.replace('.', '_').replace('-', '_')}_api"

    section = config_snapshot.get(section_key)
    if isinstance(section, dict):
        api_key = _usable_runtime_config_value(section.get("api_key"))
        if api_key is not None:
            return api_key

    legacy_keys = config_snapshot.get("api_keys")
    if isinstance(legacy_keys, dict):
        api_key = _usable_runtime_config_value(legacy_keys.get(provider_norm))
        if api_key is not None:
            return api_key
    return None


def resolve_server_default_key(
    provider: str,
    *,
    include_override: bool = True,
) -> str | None:
    """Resolve a server key, optionally including the structured override cache."""
    provider_norm = canonical_provider_name(provider)
    if include_override:
        from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
            get_override_server_fallback,
        )

        override = get_override_server_fallback(provider_norm)
        if override and override.api_key:
            return override.api_key
    return resolve_server_default_key_from_snapshot(
        provider_norm,
        load_server_config_snapshot(),
    )
