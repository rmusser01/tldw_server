from __future__ import annotations

import base64
import json
from typing import Any, Mapping
from urllib.parse import urlparse

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
    build_secret_payload,
    decrypt_byok_payload,
    encrypt_byok_payload,
    key_hint_for_api_key,
    loads_envelope,
)
from tldw_Server_API.app.core.Persona.dialogue_tree_context import redact_sensitive_payload
from tldw_Server_API.app.core.Security.egress import evaluate_url_policy


PERSONA_CONNECTION_STATUS_FIELD = "secret_configured"


class PersonaConnectionError(ValueError):
    """Base exception for persona connection helper failures."""


class PersonaConnectionConfigError(PersonaConnectionError):
    """Raised when a persona connection is misconfigured."""


class PersonaConnectionSecretError(PersonaConnectionError):
    """Raised when a stored persona connection secret is invalid."""


class PersonaConnectionTargetError(PersonaConnectionError):
    """Raised when a connection request target violates host or egress policy."""


def normalize_hostname(host: str) -> str:
    return str(host or "").strip().rstrip(".").lower()


def host_matches_allowlist(host: str, allowlist: list[str]) -> bool:
    normalized_host = normalize_hostname(host)
    normalized_allowlist = [
        normalize_hostname(item)
        for item in allowlist
        if normalize_hostname(item)
    ]
    if not normalized_allowlist:
        return False
    for allowed in normalized_allowlist:
        if normalized_host == allowed or normalized_host.endswith(f".{allowed}"):
            return True
    return False


def normalize_connection_base_url(base_url: str) -> tuple[str, list[str]]:
    parsed = urlparse(str(base_url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise PersonaConnectionConfigError(
            "Connection base_url must be a valid http(s) URL."
        )
    normalized_base_url = parsed.geturl().rstrip("/")
    return normalized_base_url, [normalize_hostname(parsed.hostname)]


def connection_content_from_row(row: dict[str, Any]) -> dict[str, Any]:
    raw_content = row.get("content")
    if isinstance(raw_content, str):
        try:
            content = json.loads(raw_content)
        except (TypeError, ValueError):
            content = {}
    elif isinstance(raw_content, dict):
        content = dict(raw_content)
    else:
        content = {}
    return content if isinstance(content, dict) else {}


def build_connection_memory_content(
    *,
    name: str,
    base_url: str,
    auth_type: str,
    headers_template: Mapping[str, Any] | None,
    timeout_ms: int,
    secret: str | None = None,
) -> dict[str, Any]:
    normalized_base_url, allowed_hosts = normalize_connection_base_url(base_url)
    content: dict[str, Any] = {
        "schema_version": 1,
        "name": str(name).strip(),
        "base_url": normalized_base_url,
        "auth_type": str(auth_type or "none").strip().lower(),
        "headers_template": {
            str(key): str(value)
            for key, value in dict(headers_template or {}).items()
        },
        "timeout_ms": int(timeout_ms),
        "allowed_hosts": allowed_hosts,
        PERSONA_CONNECTION_STATUS_FIELD: False,
        "key_hint": None,
    }

    raw_secret = str(secret or "").strip()
    if raw_secret:
        envelope = encrypt_byok_payload(build_secret_payload(raw_secret))
        content["secret_envelope"] = json.dumps(envelope, ensure_ascii=True)
        content[PERSONA_CONNECTION_STATUS_FIELD] = True
        content["key_hint"] = key_hint_for_api_key(raw_secret)
    return content


def build_updated_connection_memory_content(
    existing_content: dict[str, Any],
    *,
    name: str,
    base_url: str,
    auth_type: str,
    headers_template: Mapping[str, Any] | None,
    timeout_ms: int,
    secret: str | None,
    clear_secret: bool,
) -> dict[str, Any]:
    if clear_secret and secret is not None:
        raise PersonaConnectionConfigError(
            "Provide either secret or clear_secret, not both."
        )

    content = build_connection_memory_content(
        name=name,
        base_url=base_url,
        auth_type=auth_type,
        headers_template=headers_template,
        timeout_ms=timeout_ms,
        secret=secret,
    )

    if clear_secret:
        content.pop("secret_envelope", None)
        content[PERSONA_CONNECTION_STATUS_FIELD] = False
        content["key_hint"] = None
        return content

    if secret is None:
        existing_envelope = existing_content.get("secret_envelope")
        if isinstance(existing_envelope, dict) and existing_envelope:
            content["secret_envelope"] = json.dumps(existing_envelope, ensure_ascii=True)
            content[PERSONA_CONNECTION_STATUS_FIELD] = bool(
                existing_content.get(PERSONA_CONNECTION_STATUS_FIELD, False)
            )
            content["key_hint"] = existing_content.get("key_hint")
        elif str(existing_envelope or "").strip():
            content["secret_envelope"] = str(existing_envelope).strip()
            content[PERSONA_CONNECTION_STATUS_FIELD] = bool(
                existing_content.get(PERSONA_CONNECTION_STATUS_FIELD, False)
            )
            content["key_hint"] = existing_content.get("key_hint")
    return content


def get_connection_allowed_hosts(
    connection: Mapping[str, Any],
    *,
    fail_closed: bool = False,
) -> list[str]:
    stored_allowlist = [
        normalize_hostname(item)
        for item in list(connection.get("allowed_hosts") or [])
        if normalize_hostname(item)
    ]
    if stored_allowlist:
        return stored_allowlist

    base_url = str(connection.get("base_url") or "").strip()
    base_host = normalize_hostname(urlparse(base_url).hostname or "")
    if base_host:
        return [base_host]

    if fail_closed:
        raise PersonaConnectionConfigError(
            "Connection is missing a valid allowed host."
        )
    return []


class _SafeFormatDict(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def safe_template_context(*payloads: Mapping[str, Any]) -> dict[str, str]:
    context: dict[str, str] = {}
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        for key, value in payload.items():
            if value is None:
                continue
            context[str(key)] = str(value)
    return context


def render_template_value(value: str, context: dict[str, str]) -> str:
    rendered = str(value)
    for key, replacement in context.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", replacement)
    try:
        return rendered.format_map(_SafeFormatDict(context))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to render persona connection template '{}': {}",
            redact_sensitive_payload(value),
            exc,
        )
        return rendered


def render_nested_templates(value: Any, context: dict[str, str]) -> Any:
    if isinstance(value, str):
        return render_template_value(value, context)
    if isinstance(value, dict):
        return {
            str(key): render_nested_templates(item, context)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [render_nested_templates(item, context) for item in value]
    return value


def validate_connection_request_target(
    url: str,
    connection: Mapping[str, Any],
) -> list[str]:
    parsed_url = urlparse(url)
    final_host = normalize_hostname(parsed_url.hostname or "")
    allowlist = get_connection_allowed_hosts(connection, fail_closed=True)
    if not final_host or not host_matches_allowlist(final_host, allowlist):
        raise PersonaConnectionTargetError(
            f"Host '{final_host}' is not allowed for this connection."
        )

    policy = evaluate_url_policy(url, allowlist=allowlist)
    if not getattr(policy, "allowed", False):
        reason = str(getattr(policy, "reason", None) or "egress policy denied")
        raise PersonaConnectionTargetError(
            f"Egress policy denied connection request: {reason}"
        )
    return allowlist


def resolve_connection_secret(connection: Mapping[str, Any]) -> str | None:
    raw_envelope = connection.get("secret_envelope")
    if isinstance(raw_envelope, Mapping):
        envelope = dict(raw_envelope)
    else:
        encrypted_blob = str(raw_envelope or "").strip()
        if not encrypted_blob:
            return None
        try:
            envelope = loads_envelope(encrypted_blob)
        except (TypeError, ValueError) as exc:
            logger.warning("Stored persona connection secret could not be decrypted: {}", exc)
            raise PersonaConnectionSecretError(
                "Invalid stored secret for this connection."
            ) from exc

    if not envelope:
        return None
    try:
        payload = decrypt_byok_payload(envelope)
    except (TypeError, ValueError) as exc:
        logger.warning("Stored persona connection secret could not be decrypted: {}", exc)
        raise PersonaConnectionSecretError(
            "Invalid stored secret for this connection."
        ) from exc

    secret = str(payload.get("api_key") or "").strip()
    if not secret:
        raise PersonaConnectionSecretError(
            "Invalid stored secret for this connection."
        )
    return secret


def build_connection_headers(
    connection: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
    extra_headers: Mapping[str, Any] | None = None,
    auth_header_name: str | None = None,
) -> tuple[dict[str, str], str | None]:
    secret = resolve_connection_secret(connection)
    template_context = safe_template_context(
        payload,
        {
            "secret": secret or "",
            "connection_id": str(connection.get("id") or ""),
            "base_url": str(connection.get("base_url") or ""),
        },
    )
    headers = {
        str(key): render_template_value(str(value), template_context)
        for key, value in dict(connection.get("headers_template") or {}).items()
    }
    if extra_headers:
        headers.update(
            {
                str(key): render_template_value(str(value), template_context)
                for key, value in extra_headers.items()
            }
        )

    auth_type = str(connection.get("auth_type") or "none").strip().lower()
    existing_header_names = {key.lower() for key in headers}
    if secret:
        if auth_type == "bearer" and "authorization" not in existing_header_names:
            headers["Authorization"] = f"Bearer {secret}"
        elif auth_type == "api_key" and "x-api-key" not in existing_header_names:
            headers["X-API-Key"] = secret
        elif auth_type == "basic" and "authorization" not in existing_header_names:
            encoded = base64.b64encode(secret.encode("utf-8")).decode("ascii")
            headers["Authorization"] = f"Basic {encoded}"
        elif auth_type == "custom_header":
            header_name = str(auth_header_name or "X-API-Key").strip() or "X-API-Key"
            if header_name.lower() not in existing_header_names:
                headers[header_name] = secret

    return headers, secret


def redact_connection_headers(
    headers: Mapping[str, Any],
    secret: str | None,
) -> dict[str, str]:
    redacted: dict[str, str] = {}
    for key, value in headers.items():
        normalized_key = str(key).strip().lower()
        value_text = str(value)
        if (
            normalized_key in {"authorization", "proxy-authorization"}
            or "api-key" in normalized_key
            or "token" in normalized_key
            or "secret" in normalized_key
            or (secret and secret in value_text)
        ):
            redacted[str(key)] = "[redacted]"
        else:
            redacted[str(key)] = value_text
    return redacted
