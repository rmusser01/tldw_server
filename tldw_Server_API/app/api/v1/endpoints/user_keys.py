from __future__ import annotations

import base64
import copy
import hashlib
import json
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import (
    get_or_create_audit_service_for_user_id_optional,
)
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.schemas.user_keys import (
    OpenAICredentialSourceSwitchRequest,
    OpenAICredentialSourceSwitchResponse,
    OpenAIOAuthAuthorizeRequest,
    OpenAIOAuthAuthorizeResponse,
    OpenAIOAuthCallbackResponse,
    OpenAIOAuthRefreshResponse,
    OpenAIOAuthStatusResponse,
    ProviderKeyTestRequest,
    ProviderKeyTestResponse,
    UserProviderKeyResponse,
    UserProviderKeysResponse,
    UserProviderKeyStatusItem,
    UserProviderKeyUpsertRequest,
)
from tldw_Server_API.app.core.Audit.unified_audit_service import (
    AuditContext,
    AuditEventCategory,
    AuditEventType,
)
from tldw_Server_API.app.core.AuthNZ.byok_helpers import (
    is_byok_enabled,
    is_provider_allowlisted,
    is_trusted_base_url_request,
    resolve_byok_allowlist,
    resolve_server_default_key,
    validate_base_url_override,
    validate_credential_fields,
)
from tldw_Server_API.app.core.AuthNZ.byok_testing import test_provider_credentials
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.orgs_teams import list_memberships_for_user
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos.byok_oauth_state_repo import AuthnzByokOAuthStateRepo
from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import (
    AuthnzOrgProviderSecretsRepo,
)
from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import (
    AuthnzUserProviderSecretsRepo,
)
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
    build_secret_payload,
    decrypt_byok_payload,
    dumps_envelope,
    encrypt_byok_payload,
    key_hint_for_api_key,
    loads_envelope,
    normalize_provider_name,
)
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAPIError
from tldw_Server_API.app.core.http_client import RetryPolicy as _RetryPolicy
from tldw_Server_API.app.core.http_client import afetch as _http_afetch
from tldw_Server_API.app.core.Metrics import increment_counter, observe_histogram

router = APIRouter(prefix="/users", tags=["users"])

_OPENAI_PROVIDER = "openai"
_OPENAI_OAUTH_HINT = "oauth"
_OPENAI_SOURCE_API_KEY = "api_key"
_OPENAI_SOURCE_OAUTH = "oauth"
_OPENAI_CREDENTIAL_VERSION = 2
_OPENAI_DEFAULT_OAUTH_STATE_TTL_MINUTES = 10


async def _get_user_repo() -> AuthnzUserProviderSecretsRepo:
    pool = await get_db_pool()
    repo = AuthnzUserProviderSecretsRepo(pool)
    await repo.ensure_tables()
    return repo


async def _get_org_repo() -> AuthnzOrgProviderSecretsRepo:
    pool = await get_db_pool()
    repo = AuthnzOrgProviderSecretsRepo(pool)
    await repo.ensure_tables()
    return repo


async def _get_oauth_state_repo() -> AuthnzByokOAuthStateRepo:
    pool = await get_db_pool()
    repo = AuthnzByokOAuthStateRepo(pool)
    await repo.ensure_tables()
    return repo


def _require_byok_enabled() -> None:
    if not is_byok_enabled():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="BYOK is disabled in this deployment",
        )


def _coerce_nonempty_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    return trimmed if trimmed else None


def _openai_oauth_metric_labels(
    *,
    reason: str | None = None,
    outcome: str | None = None,
) -> dict[str, str]:
    labels = {"provider": _OPENAI_PROVIDER}
    reason_value = _coerce_nonempty_string(reason)
    if reason_value:
        labels["reason"] = reason_value
    outcome_value = _coerce_nonempty_string(outcome)
    if outcome_value:
        labels["outcome"] = outcome_value
    return labels


def _record_openai_oauth_counter(
    metric_name: str,
    *,
    labels: dict[str, str] | None = None,
    value: float = 1,
) -> None:
    try:
        increment_counter(metric_name, value=value, labels=labels or {})
    except Exception:
        logger.debug("OpenAI OAuth metric emission failed")


def _record_openai_oauth_histogram(
    metric_name: str,
    *,
    value: float,
    labels: dict[str, str] | None = None,
) -> None:
    try:
        observe_histogram(metric_name, value=value, labels=labels or {})
    except Exception:
        logger.debug("OpenAI OAuth histogram emission failed")


async def _emit_openai_oauth_audit_event(
    *,
    user_id: int,
    action: str,
    result: str = "success",
    error_message: str | None = None,
    request: Request | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    action_name = _coerce_nonempty_string(action)
    if not action_name:
        return

    try:
        audit_service = await get_or_create_audit_service_for_user_id_optional(user_id)
        correlation_id = None
        request_id = ""
        ip_address = None
        user_agent = None
        endpoint = None
        method = None

        if request is not None:
            correlation_id = request.headers.get("X-Correlation-ID") or getattr(request.state, "correlation_id", None)
            request_id = request.headers.get("X-Request-ID") or getattr(request.state, "request_id", None) or ""
            ip_address = request.client.host if request.client else None
            user_agent = request.headers.get("user-agent")
            endpoint = str(request.url.path)
            method = request.method

        context = AuditContext(
            user_id=str(user_id),
            correlation_id=correlation_id,
            request_id=request_id,
            ip_address=ip_address,
            user_agent=user_agent,
            endpoint=endpoint,
            method=method,
        )
        result_norm = _coerce_nonempty_string(result) or "success"
        event_type = AuditEventType.API_ERROR if result_norm in {"failure", "error"} else AuditEventType.DATA_UPDATE
        category = (
            AuditEventCategory.SECURITY if result_norm in {"failure", "error"} else AuditEventCategory.DATA_MODIFICATION
        )
        event_metadata: dict[str, Any] = {"provider": _OPENAI_PROVIDER}
        if metadata:
            event_metadata.update(metadata)

        await audit_service.log_event(
            event_type=event_type,
            category=category,
            context=context,
            resource_type="provider_oauth",
            resource_id=f"{_OPENAI_PROVIDER}:{user_id}",
            action=action_name,
            result=result_norm,
            error_message=error_message,
            metadata=event_metadata,
        )
    except Exception:
        logger.debug("OpenAI OAuth audit emission skipped")


def _parse_iso_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _principal_user_id(principal: AuthPrincipal) -> int:
    raw_id = principal.user_id
    try:
        user_id = int(raw_id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid user context") from exc
    if user_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid user context")
    return user_id


def _parse_metadata_value(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def _row_metadata(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    return _parse_metadata_value(row.get("metadata"))


def _extract_payload_from_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    encrypted_blob = row.get("encrypted_blob")
    if not encrypted_blob:
        return None
    try:
        payload = decrypt_byok_payload(loads_envelope(encrypted_blob))
    except Exception:
        logger.warning("Failed to decrypt BYOK payload for provider row")
        return None
    return payload if isinstance(payload, dict) else None


def _is_openai_v2_payload(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("credential_version") != _OPENAI_CREDENTIAL_VERSION:
        return False
    credentials = payload.get("credentials")
    return isinstance(credentials, dict)


def _openai_credentials_map(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not _is_openai_v2_payload(payload):
        return {}
    credentials = payload.get("credentials")
    if isinstance(credentials, dict):
        return credentials
    return {}


def _openai_source_payload(payload: dict[str, Any] | None, source: str) -> dict[str, Any]:
    credentials = _openai_credentials_map(payload)
    blob = credentials.get(source)
    if isinstance(blob, dict):
        return blob
    return {}


def _legacy_payload_api_key(payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    return _coerce_nonempty_string(payload.get("api_key"))


def _v2_payload_api_key(payload: dict[str, Any] | None) -> str | None:
    source_blob = _openai_source_payload(payload, _OPENAI_SOURCE_API_KEY)
    return _coerce_nonempty_string(source_blob.get("api_key"))


def _v2_payload_oauth_access_token(payload: dict[str, Any] | None) -> str | None:
    source_blob = _openai_source_payload(payload, _OPENAI_SOURCE_OAUTH)
    return _coerce_nonempty_string(source_blob.get("access_token"))


def _v2_payload_oauth_refresh_token(payload: dict[str, Any] | None) -> str | None:
    source_blob = _openai_source_payload(payload, _OPENAI_SOURCE_OAUTH)
    return _coerce_nonempty_string(source_blob.get("refresh_token"))


def _v2_source_available(
    payload: dict[str, Any] | None,
    source: str,
    *,
    require_access_for_oauth: bool = False,
) -> bool:
    if source == _OPENAI_SOURCE_API_KEY:
        return bool(_v2_payload_api_key(payload))
    if source == _OPENAI_SOURCE_OAUTH:
        access_token = _v2_payload_oauth_access_token(payload)
        refresh_token = _v2_payload_oauth_refresh_token(payload)
        if require_access_for_oauth:
            return bool(access_token)
        return bool(access_token or refresh_token)
    return False


def _payload_active_auth_source(payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None

    legacy_api_key = _legacy_payload_api_key(payload)
    if legacy_api_key and not _is_openai_v2_payload(payload):
        return _OPENAI_SOURCE_API_KEY

    if not _is_openai_v2_payload(payload):
        return None

    active_raw = _coerce_nonempty_string(payload.get("active_auth_source"))
    active = (active_raw or "").lower()
    if active in {_OPENAI_SOURCE_API_KEY, _OPENAI_SOURCE_OAUTH} and _v2_source_available(payload, active):
        return active

    if _v2_source_available(payload, _OPENAI_SOURCE_API_KEY):
        return _OPENAI_SOURCE_API_KEY
    if _v2_source_available(payload, _OPENAI_SOURCE_OAUTH):
        return _OPENAI_SOURCE_OAUTH
    return None


def _payload_runtime_api_key(payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None

    legacy_api_key = _legacy_payload_api_key(payload)
    if legacy_api_key:
        return legacy_api_key

    if not _is_openai_v2_payload(payload):
        return None

    active_source = _payload_active_auth_source(payload)
    if active_source == _OPENAI_SOURCE_OAUTH:
        access_token = _v2_payload_oauth_access_token(payload)
        if access_token:
            return access_token
    elif active_source == _OPENAI_SOURCE_API_KEY:
        api_key = _v2_payload_api_key(payload)
        if api_key:
            return api_key

    fallback_api_key = _v2_payload_api_key(payload)
    if fallback_api_key:
        return fallback_api_key
    return _v2_payload_oauth_access_token(payload)


def _payload_key_hint(payload: dict[str, Any] | None) -> str:
    active_source = _payload_active_auth_source(payload)
    if active_source == _OPENAI_SOURCE_OAUTH:
        return _OPENAI_OAUTH_HINT

    api_key = _v2_payload_api_key(payload)
    if api_key:
        return key_hint_for_api_key(api_key)

    legacy_api_key = _legacy_payload_api_key(payload)
    if legacy_api_key:
        return key_hint_for_api_key(legacy_api_key)

    if _v2_source_available(payload, _OPENAI_SOURCE_OAUTH):
        return _OPENAI_OAUTH_HINT
    return ""


def _openai_payload_credential_fields(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    raw = payload.get("credential_fields")
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def _coerce_openai_payload_v2(payload: dict[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "credential_version": _OPENAI_CREDENTIAL_VERSION,
        "credentials": {},
    }
    credentials: dict[str, Any] = {}

    if _is_openai_v2_payload(payload):
        existing_credentials = _openai_credentials_map(payload)
        api_blob = existing_credentials.get(_OPENAI_SOURCE_API_KEY)
        if isinstance(api_blob, dict):
            api_key = _coerce_nonempty_string(api_blob.get("api_key"))
            if api_key:
                api_payload = dict(api_blob)
                api_payload["api_key"] = api_key
                credentials[_OPENAI_SOURCE_API_KEY] = api_payload
        oauth_blob = existing_credentials.get(_OPENAI_SOURCE_OAUTH)
        if isinstance(oauth_blob, dict):
            oauth_payload: dict[str, Any] = {}
            for key in (
                "access_token",
                "refresh_token",
                "token_type",
                "scope",
                "subject",
                "issued_at",
                "expires_at",
            ):
                value = oauth_blob.get(key)
                if isinstance(value, datetime):
                    oauth_payload[key] = value.astimezone(timezone.utc).isoformat()
                    continue
                text_value = _coerce_nonempty_string(value)
                if text_value:
                    oauth_payload[key] = text_value
            if oauth_payload:
                credentials[_OPENAI_SOURCE_OAUTH] = oauth_payload

    legacy_api_key = _legacy_payload_api_key(payload)
    if legacy_api_key and _OPENAI_SOURCE_API_KEY not in credentials:
        credentials[_OPENAI_SOURCE_API_KEY] = {"api_key": legacy_api_key}

    result["credentials"] = credentials

    existing_credential_fields = _openai_payload_credential_fields(payload)
    if existing_credential_fields:
        result["credential_fields"] = existing_credential_fields

    active_source = _payload_active_auth_source(payload)
    if active_source in {_OPENAI_SOURCE_API_KEY, _OPENAI_SOURCE_OAUTH} and _v2_source_available(result, active_source):
        result["active_auth_source"] = active_source
    elif _v2_source_available(result, _OPENAI_SOURCE_API_KEY):
        result["active_auth_source"] = _OPENAI_SOURCE_API_KEY
    elif _v2_source_available(result, _OPENAI_SOURCE_OAUTH):
        result["active_auth_source"] = _OPENAI_SOURCE_OAUTH

    return result


def _normalize_openai_oauth_scopes(raw_scopes: Any) -> list[str]:
    if isinstance(raw_scopes, str):
        parts = [s.strip() for s in raw_scopes.replace(",", " ").split(" ") if s.strip()]
        return parts
    if isinstance(raw_scopes, (list, tuple)):
        scopes: list[str] = []
        for item in raw_scopes:
            text = _coerce_nonempty_string(item)
            if text:
                scopes.append(text)
        return scopes
    return []


def _normalize_openai_return_path_prefixes(raw_prefixes: Any) -> list[str]:
    normalized: list[str] = []
    if isinstance(raw_prefixes, str):
        raw_prefixes = [raw_prefixes]
    if not isinstance(raw_prefixes, (list, tuple)):
        raw_prefixes = ["/"]
    for prefix in raw_prefixes:
        text = _coerce_nonempty_string(prefix)
        if not text:
            continue
        if not text.startswith("/"):
            text = f"/{text}"
        normalized.append(text)
    if not normalized:
        return ["/"]
    return normalized


def _sanitize_return_path(raw_return_path: Any, *, allowed_prefixes: list[str]) -> str | None:
    return_path = _coerce_nonempty_string(raw_return_path)
    if return_path is None:
        return None
    if not return_path.startswith("/") or return_path.startswith("//"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="return_path must be an app-relative path",
        )
    if "://" in return_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="return_path must be an app-relative path",
        )
    for prefix in _normalize_openai_return_path_prefixes(allowed_prefixes):
        if prefix == "/":
            return return_path
        canonical_prefix = prefix.rstrip("/")
        if return_path == canonical_prefix or return_path.startswith(f"{canonical_prefix}/"):
            return return_path
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="return_path is not allowed",
    )


def _resolve_openai_redirect_uri(settings: Any) -> str:
    configured_redirect = _coerce_nonempty_string(getattr(settings, "OPENAI_OAUTH_REDIRECT_URI", None))
    if configured_redirect:
        return configured_redirect
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="OpenAI OAuth redirect URI is not configured",
    )


def _oauth_code_challenge_s256(code_verifier: str) -> str:
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


async def _close_http_response(response: Any) -> None:
    close_async = getattr(response, "aclose", None)
    if callable(close_async):
        await close_async()
        return
    close_sync = getattr(response, "close", None)
    if callable(close_sync):
        close_sync()


def _extract_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _require_openai_oauth_settings() -> Any:
    _require_byok_enabled()
    if not is_provider_allowlisted(_OPENAI_PROVIDER):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Provider not allowed for BYOK",
        )

    settings = get_settings()
    if not bool(getattr(settings, "OPENAI_OAUTH_ENABLED", False)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="OpenAI OAuth is disabled in this deployment",
        )

    missing: list[str] = []
    if not _coerce_nonempty_string(getattr(settings, "OPENAI_OAUTH_CLIENT_ID", None)):
        missing.append("OPENAI_OAUTH_CLIENT_ID")
    if not _coerce_nonempty_string(getattr(settings, "OPENAI_OAUTH_CLIENT_SECRET", None)):
        missing.append("OPENAI_OAUTH_CLIENT_SECRET")
    if not _coerce_nonempty_string(getattr(settings, "OPENAI_OAUTH_AUTH_URL", None)):
        missing.append("OPENAI_OAUTH_AUTH_URL")
    if not _coerce_nonempty_string(getattr(settings, "OPENAI_OAUTH_TOKEN_URL", None)):
        missing.append("OPENAI_OAUTH_TOKEN_URL")
    if not _coerce_nonempty_string(getattr(settings, "OPENAI_OAUTH_REDIRECT_URI", None)):
        missing.append("OPENAI_OAUTH_REDIRECT_URI")
    if missing:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="OpenAI OAuth is not fully configured (missing: {})".format(", ".join(missing)),
        )

    return settings


def _openai_oauth_state_ttl_minutes(settings: Any) -> int:
    raw_value = getattr(settings, "OPENAI_OAUTH_STATE_TTL_MINUTES", _OPENAI_DEFAULT_OAUTH_STATE_TTL_MINUTES)
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        parsed = _OPENAI_DEFAULT_OAUTH_STATE_TTL_MINUTES
    if parsed <= 0:
        return _OPENAI_DEFAULT_OAUTH_STATE_TTL_MINUTES
    return parsed


def _openai_oauth_max_outstanding_states(settings: Any) -> int:
    raw_value = getattr(settings, "OPENAI_OAUTH_MAX_OUTSTANDING_STATES", 10)
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        parsed = 10
    return max(1, parsed)


async def _openai_oauth_token_exchange(
    *,
    token_url: str,
    form_data: dict[str, Any],
) -> dict[str, Any]:
    response = await _http_afetch(
        method="POST",
        url=token_url,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data=form_data,
        timeout=30,
        retry=_RetryPolicy(attempts=1),
    )
    try:
        status_code = int(getattr(response, "status_code", 0))
        payload: dict[str, Any] | None = None
        try:
            maybe_payload = response.json()
            if isinstance(maybe_payload, dict):
                payload = dict(maybe_payload)
        except Exception:
            payload = None

        if status_code < 200 or status_code >= 300:
            detail = "OpenAI OAuth token exchange failed"
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=detail,
            )

        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="OpenAI OAuth token exchange returned invalid payload",
            )

        return payload
    finally:
        await _close_http_response(response)


async def _touch_user_last_used_if_match(
    repo: AuthnzUserProviderSecretsRepo,
    *,
    user_id: int,
    provider: str,
    api_key: str,
) -> None:
    row = await repo.fetch_secret_for_user(user_id, provider)
    payload = _extract_payload_from_row(row)
    if not payload:
        return
    if _payload_runtime_api_key(payload) != api_key:
        return
    await repo.touch_last_used(user_id, provider, datetime.now(timezone.utc))


def _user_row_openai_auth_source(user_row: dict[str, Any] | None) -> str | None:
    payload = _extract_payload_from_row(user_row)
    if not payload:
        return None
    source = _payload_active_auth_source(payload)
    if source in {_OPENAI_SOURCE_API_KEY, _OPENAI_SOURCE_OAUTH}:
        return source
    return None


@router.post(
    "/keys",
    response_model=UserProviderKeyResponse,
    status_code=status.HTTP_200_OK,
)
async def upsert_user_provider_key(
    payload: UserProviderKeyUpsertRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> UserProviderKeyResponse:
    _require_byok_enabled()
    provider_norm = normalize_provider_name(payload.provider)
    if not is_provider_allowlisted(provider_norm):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Provider not allowed for BYOK",
        )

    api_key = (payload.api_key or "").strip()
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="api_key is required",
        )

    allow_base_url = is_trusted_base_url_request(request, principal=principal)
    raw_fields = payload.credential_fields or {}
    if isinstance(raw_fields, dict) and "base_url" in raw_fields and not allow_base_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="base_url override requires admin or service principal",
        )

    try:
        credential_fields = validate_credential_fields(
            provider_norm,
            payload.credential_fields,
            allow_base_url=allow_base_url,
        )
        if "base_url" in credential_fields:
            credential_fields["base_url"] = validate_base_url_override(credential_fields["base_url"])
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid provider credential fields",
        ) from exc

    try:
        await test_provider_credentials(
            provider=provider_norm,
            api_key=api_key,
            credential_fields=credential_fields,
            model=None,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provider credential validation failed",
        ) from exc
    except ChatAPIError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Provider test call failed",
        ) from exc

    user_id = _principal_user_id(principal)
    repo = await _get_user_repo()
    now = datetime.now(timezone.utc)
    existing_row = await repo.fetch_secret_for_user(user_id, provider_norm)
    existing_payload = _extract_payload_from_row(existing_row)
    metadata_to_store = payload.metadata
    if metadata_to_store is None:
        metadata_to_store = _row_metadata(existing_row)

    if provider_norm == _OPENAI_PROVIDER:
        secret_payload = _coerce_openai_payload_v2(existing_payload)
        credentials = _openai_credentials_map(secret_payload)
        existing_api_blob = credentials.get(_OPENAI_SOURCE_API_KEY)
        api_blob = dict(existing_api_blob) if isinstance(existing_api_blob, dict) else {}
        api_blob["api_key"] = api_key
        api_blob["stored_at"] = now.isoformat()
        credentials[_OPENAI_SOURCE_API_KEY] = api_blob
        secret_payload["credentials"] = credentials
        if payload.credential_fields is not None:
            if credential_fields:
                secret_payload["credential_fields"] = credential_fields
            else:
                secret_payload.pop("credential_fields", None)
        secret_payload["active_auth_source"] = _OPENAI_SOURCE_API_KEY
        key_hint = key_hint_for_api_key(api_key)
    else:
        secret_payload = build_secret_payload(api_key, credential_fields or None)
        key_hint = key_hint_for_api_key(api_key)

    try:
        envelope = encrypt_byok_payload(secret_payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="BYOK encryption is not configured",
        ) from exc

    row = await repo.upsert_secret(
        user_id=user_id,
        provider=provider_norm,
        encrypted_blob=dumps_envelope(envelope),
        key_hint=key_hint,
        metadata=metadata_to_store,
        updated_at=now,
        created_by=user_id,
        updated_by=user_id,
    )
    return UserProviderKeyResponse(
        provider=provider_norm,
        key_hint=row.get("key_hint") or key_hint,
        updated_at=row.get("updated_at") or now,
    )


@router.get("/keys", response_model=UserProviderKeysResponse)
async def list_user_provider_keys(
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> UserProviderKeysResponse:
    _require_byok_enabled()
    user_id = _principal_user_id(principal)
    allowlist = resolve_byok_allowlist()

    user_repo = await _get_user_repo()
    org_repo = await _get_org_repo()

    user_rows = await user_repo.list_secrets_for_user(user_id)
    user_keys = {row.get("provider"): row for row in user_rows}
    openai_user_full_row: dict[str, Any] | None = None
    if _OPENAI_PROVIDER in user_keys:
        openai_user_full_row = await user_repo.fetch_secret_for_user(
            user_id,
            _OPENAI_PROVIDER,
        )

    memberships = await list_memberships_for_user(user_id)
    team_ids = sorted({m.get("team_id") for m in memberships if m.get("team_id") is not None})
    org_ids = sorted({m.get("org_id") for m in memberships if m.get("org_id") is not None})

    def _filter_scopes(ids: list[int], active_id: Any) -> list[int]:
        if not ids:
            return []
        if active_id is None:
            return ids if len(ids) == 1 else []
        try:
            active = int(active_id)
        except (TypeError, ValueError):
            return []
        return [active] if active in ids else []

    active_team_id = getattr(request.state, "active_team_id", None)
    active_org_id = getattr(request.state, "active_org_id", None)
    team_scope_ids = _filter_scopes(team_ids, active_team_id)
    org_scope_ids = _filter_scopes(org_ids, active_org_id)

    shared_keys: dict[str, dict[str, Any]] = {}
    shared_sources: dict[str, str] = {}
    for team_id in team_scope_ids:
        rows = await org_repo.list_secrets(scope_type="team", scope_id=int(team_id))
        for row in rows:
            provider = row.get("provider")
            if provider and provider not in shared_keys:
                shared_keys[provider] = row
                shared_sources[provider] = "team"
    for org_id in org_scope_ids:
        rows = await org_repo.list_secrets(scope_type="org", scope_id=int(org_id))
        for row in rows:
            provider = row.get("provider")
            if provider and provider not in shared_keys:
                shared_keys[provider] = row
                shared_sources[provider] = "org"

    providers = sorted(set(allowlist) | set(user_keys.keys()) | set(shared_keys.keys()))
    items: list[UserProviderKeyStatusItem] = []
    for provider in providers:
        allowed = provider in allowlist
        user_row = user_keys.get(provider)
        shared_row = shared_keys.get(provider)
        auth_source = _user_row_openai_auth_source(openai_user_full_row) if provider == _OPENAI_PROVIDER else None
        if not allowed and (user_row or shared_row):
            items.append(
                UserProviderKeyStatusItem(
                    provider=provider,
                    has_key=bool(user_row),
                    source="disabled",
                    key_hint=user_row.get("key_hint") if user_row else None,
                    auth_source=auth_source,
                    last_used_at=(user_row or shared_row or {}).get("last_used_at"),
                )
            )
            continue

        if user_row:
            items.append(
                UserProviderKeyStatusItem(
                    provider=provider,
                    has_key=True,
                    source="user",
                    key_hint=user_row.get("key_hint"),
                    auth_source=auth_source,
                    last_used_at=user_row.get("last_used_at"),
                )
            )
            continue

        if shared_row:
            items.append(
                UserProviderKeyStatusItem(
                    provider=provider,
                    has_key=False,
                    source=shared_sources.get(provider, "org"),
                    last_used_at=shared_row.get("last_used_at"),
                )
            )
            continue

        if resolve_server_default_key(provider):
            items.append(
                UserProviderKeyStatusItem(
                    provider=provider,
                    has_key=False,
                    source="server_default",
                )
            )
            continue

        items.append(
            UserProviderKeyStatusItem(
                provider=provider,
                has_key=False,
                source="none",
            )
        )

    return UserProviderKeysResponse(items=items)


@router.post("/keys/test", response_model=ProviderKeyTestResponse)
async def test_user_provider_key(
    payload: ProviderKeyTestRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> ProviderKeyTestResponse:
    _require_byok_enabled()
    provider_norm = normalize_provider_name(payload.provider)
    if not is_provider_allowlisted(provider_norm):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Provider not allowed for BYOK",
        )

    user_id = _principal_user_id(principal)
    repo = await _get_user_repo()
    row = await repo.fetch_secret_for_user(user_id, provider_norm)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Key not found")

    stored_payload = _extract_payload_from_row(row)
    if not stored_payload:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Key not found")

    api_key = _payload_runtime_api_key(stored_payload)
    if not api_key:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Key not found")

    allow_base_url = is_trusted_base_url_request(request, principal=principal)
    credential_fields_raw = stored_payload.get("credential_fields") or {}
    if isinstance(credential_fields_raw, dict) and "base_url" in credential_fields_raw and not allow_base_url:
        credential_fields_raw = dict(credential_fields_raw)
        credential_fields_raw.pop("base_url", None)

    try:
        credential_fields = validate_credential_fields(
            provider_norm,
            credential_fields_raw,
            allow_base_url=allow_base_url,
        )
        if "base_url" in credential_fields:
            credential_fields["base_url"] = validate_base_url_override(credential_fields["base_url"])
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid provider credential fields",
        ) from exc

    try:
        model_used = await test_provider_credentials(
            provider=provider_norm,
            api_key=api_key,
            credential_fields=credential_fields,
            model=payload.model,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provider credential validation failed",
        ) from exc
    except ChatAPIError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Provider test call failed",
        ) from exc

    await _touch_user_last_used_if_match(
        repo,
        user_id=user_id,
        provider=provider_norm,
        api_key=api_key,
    )

    return ProviderKeyTestResponse(provider=provider_norm, status="valid", model=model_used)


@router.post(
    "/keys/openai/oauth/authorize",
    response_model=OpenAIOAuthAuthorizeResponse,
    status_code=status.HTTP_200_OK,
)
async def authorize_openai_oauth(
    request: Request,
    payload: OpenAIOAuthAuthorizeRequest | None = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> OpenAIOAuthAuthorizeResponse:
    settings = _require_openai_oauth_settings()
    user_id = _principal_user_id(principal)
    _record_openai_oauth_counter(
        "byok_oauth_authorize_started_total",
        labels=_openai_oauth_metric_labels(),
    )
    auth_payload = payload or OpenAIOAuthAuthorizeRequest()

    try:
        credential_fields = validate_credential_fields(
            _OPENAI_PROVIDER,
            auth_payload.credential_fields,
            allow_base_url=False,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid OpenAI OAuth credential fields",
        ) from exc

    allowed_prefixes = _normalize_openai_return_path_prefixes(
        getattr(settings, "OPENAI_OAUTH_ALLOWED_RETURN_PATH_PREFIXES", ["/"])
    )
    return_path = _sanitize_return_path(
        auth_payload.return_path,
        allowed_prefixes=allowed_prefixes,
    )

    redirect_uri = _resolve_openai_redirect_uri(settings)
    auth_url_base = _coerce_nonempty_string(getattr(settings, "OPENAI_OAUTH_AUTH_URL", None))
    client_id = _coerce_nonempty_string(getattr(settings, "OPENAI_OAUTH_CLIENT_ID", None))
    if not auth_url_base or not client_id:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="OpenAI OAuth is not fully configured",
        )

    code_verifier = secrets.token_urlsafe(64)
    code_challenge = _oauth_code_challenge_s256(code_verifier)
    state = secrets.token_urlsafe(32)
    auth_session_id = secrets.token_urlsafe(24)

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=_openai_oauth_state_ttl_minutes(settings))

    state_blob: dict[str, Any] = {"pkce_verifier": code_verifier}
    if credential_fields:
        state_blob["credential_fields"] = copy.deepcopy(credential_fields)
    try:
        encrypted_state_blob = dumps_envelope(encrypt_byok_payload(state_blob))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="BYOK encryption is not configured",
        ) from exc

    state_repo = await _get_oauth_state_repo()
    await state_repo.enforce_outstanding_cap(
        user_id=user_id,
        provider=_OPENAI_PROVIDER,
        max_outstanding=_openai_oauth_max_outstanding_states(settings),
        now=now,
    )
    await state_repo.create_state(
        state=state,
        user_id=user_id,
        provider=_OPENAI_PROVIDER,
        auth_session_id=auth_session_id,
        redirect_uri=redirect_uri,
        pkce_verifier_encrypted=encrypted_state_blob,
        expires_at=expires_at,
        return_path=return_path,
        created_at=now,
    )

    scopes = _normalize_openai_oauth_scopes(getattr(settings, "OPENAI_OAUTH_SCOPES", []))
    query = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    if scopes:
        query["scope"] = " ".join(scopes)
    auth_url = f"{auth_url_base}?{urlencode(query)}"

    await _emit_openai_oauth_audit_event(
        user_id=user_id,
        action="provider_oauth_authorize_started",
        request=request,
        metadata={
            "auth_session_id": auth_session_id,
            "return_path": return_path,
            "credential_field_keys": sorted(credential_fields.keys()) if credential_fields else [],
        },
    )

    return OpenAIOAuthAuthorizeResponse(
        provider=_OPENAI_PROVIDER,
        auth_url=auth_url,
        auth_session_id=auth_session_id,
        expires_at=expires_at,
    )


@router.get(
    "/keys/openai/oauth/callback",
    response_model=OpenAIOAuthCallbackResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_303_SEE_OTHER: {
            "description": "Redirect to the stored OAuth return path when redirect=true.",
            "headers": {
                "Location": {
                    "description": "Stored OAuth return path.",
                    "schema": {"type": "string"},
                },
            },
        },
    },
)
async def callback_openai_oauth(
    request: Request,
    code: str,
    state: str,
    redirect: bool = False,
) -> OpenAIOAuthCallbackResponse:
    settings = _require_openai_oauth_settings()
    failure_reason: str | None = None
    user_id: int | None = None
    try:
        code_value = _coerce_nonempty_string(code)
        state_value = _coerce_nonempty_string(state)
        if not code_value or not state_value:
            failure_reason = "missing_callback_params"
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing OAuth callback parameters",
            )

        state_repo = await _get_oauth_state_repo()
        state_record = await state_repo.consume_state(
            state=state_value,
            provider=_OPENAI_PROVIDER,
        )
        if not state_record:
            failure_reason = "invalid_or_expired_state"
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid or expired OAuth state",
            )

        user_id_raw = state_record.get("user_id")
        try:
            user_id = int(user_id_raw)
        except (TypeError, ValueError) as exc:
            failure_reason = "invalid_state_user_context"
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="OAuth state user context is invalid",
            ) from exc

        redirect_uri = _coerce_nonempty_string(state_record.get("redirect_uri"))
        if not redirect_uri:
            failure_reason = "missing_state_redirect_metadata"
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="OAuth state is missing redirect metadata",
            )
        expected_redirect_uri = _resolve_openai_redirect_uri(settings)
        if redirect_uri != expected_redirect_uri:
            failure_reason = "state_redirect_mismatch"
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="OAuth state redirect URI mismatch",
            )

        encrypted_pkce_blob = _coerce_nonempty_string(state_record.get("pkce_verifier_encrypted"))
        if not encrypted_pkce_blob:
            failure_reason = "missing_state_pkce_verifier"
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="OAuth state is missing PKCE verifier",
            )

        try:
            state_secret = decrypt_byok_payload(loads_envelope(encrypted_pkce_blob))
        except Exception as exc:
            failure_reason = "state_verifier_decrypt_failed"
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="OAuth state verifier could not be decrypted",
            ) from exc

        if not isinstance(state_secret, dict):
            failure_reason = "invalid_state_verifier_payload"
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="OAuth state verifier is invalid",
            )

        code_verifier = _coerce_nonempty_string(state_secret.get("pkce_verifier"))
        if not code_verifier:
            failure_reason = "missing_state_code_verifier"
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="OAuth state verifier is invalid",
            )

        try:
            failure_reason = "invalid_state_credential_fields"
            state_credential_fields = validate_credential_fields(
                _OPENAI_PROVIDER,
                state_secret.get("credential_fields"),
                allow_base_url=False,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid OpenAI OAuth credential fields",
            ) from exc
        failure_reason = None

        token_url = _coerce_nonempty_string(getattr(settings, "OPENAI_OAUTH_TOKEN_URL", None))
        client_id = _coerce_nonempty_string(getattr(settings, "OPENAI_OAUTH_CLIENT_ID", None))
        client_secret = _coerce_nonempty_string(getattr(settings, "OPENAI_OAUTH_CLIENT_SECRET", None))
        if not token_url or not client_id or not client_secret:
            failure_reason = "oauth_configuration_incomplete"
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="OpenAI OAuth is not fully configured",
            )

        failure_reason = "token_exchange_failed"
        token_payload = await _openai_oauth_token_exchange(
            token_url=token_url,
            form_data={
                "grant_type": "authorization_code",
                "code": code_value,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
                "code_verifier": code_verifier,
            },
        )
        failure_reason = None

        access_token = _coerce_nonempty_string(token_payload.get("access_token"))
        if not access_token:
            failure_reason = "missing_access_token"
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="OpenAI OAuth token response is missing access_token",
            )

        refresh_token = _coerce_nonempty_string(token_payload.get("refresh_token"))
        token_type = _coerce_nonempty_string(token_payload.get("token_type")) or "Bearer"
        scope = _coerce_nonempty_string(token_payload.get("scope"))
        expires_in = _extract_positive_int(token_payload.get("expires_in"))
        subject = _coerce_nonempty_string(token_payload.get("sub"))
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=expires_in) if expires_in else None

        try:
            failure_reason = "provider_test_failed"
            await test_provider_credentials(
                provider=_OPENAI_PROVIDER,
                api_key=access_token,
                credential_fields=state_credential_fields,
                model=None,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Provider credential validation failed",
            ) from exc
        except ChatAPIError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Provider test call failed",
            ) from exc
        failure_reason = None

        user_repo = await _get_user_repo()
        existing_row = await user_repo.fetch_secret_for_user(user_id, _OPENAI_PROVIDER)
        existing_payload = _extract_payload_from_row(existing_row)
        merged_payload = _coerce_openai_payload_v2(existing_payload)
        credentials = _openai_credentials_map(merged_payload)

        oauth_payload = {
            "access_token": access_token,
            "token_type": token_type,
            "issued_at": now.isoformat(),
        }
        if refresh_token:
            oauth_payload["refresh_token"] = refresh_token
        if scope:
            oauth_payload["scope"] = scope
        if expires_at is not None:
            oauth_payload["expires_at"] = expires_at.isoformat()
        if subject:
            oauth_payload["subject"] = subject
        credentials[_OPENAI_SOURCE_OAUTH] = oauth_payload
        merged_payload["credentials"] = credentials
        merged_payload["active_auth_source"] = _OPENAI_SOURCE_OAUTH

        existing_credential_fields = _openai_payload_credential_fields(merged_payload)
        if state_credential_fields:
            merged_fields = dict(existing_credential_fields)
            merged_fields.update(state_credential_fields)
            merged_payload["credential_fields"] = merged_fields
        elif existing_credential_fields:
            merged_payload["credential_fields"] = existing_credential_fields

        metadata_to_store = _row_metadata(existing_row)
        try:
            failure_reason = "payload_encrypt_failed"
            envelope = encrypt_byok_payload(merged_payload)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="BYOK encryption is not configured",
            ) from exc
        failure_reason = None

        failure_reason = "payload_persist_failed"
        row = await user_repo.upsert_secret(
            user_id=user_id,
            provider=_OPENAI_PROVIDER,
            encrypted_blob=dumps_envelope(envelope),
            key_hint=_OPENAI_OAUTH_HINT,
            metadata=metadata_to_store,
            updated_at=now,
            created_by=user_id,
            updated_by=user_id,
        )
        failure_reason = None

        _record_openai_oauth_counter(
            "byok_oauth_callback_success_total",
            labels=_openai_oauth_metric_labels(),
        )
        await _emit_openai_oauth_audit_event(
            user_id=user_id,
            action="provider_oauth_connected",
            request=request,
            metadata={
                "scope": scope,
                "expires_at": expires_at.isoformat() if expires_at else None,
                "has_refresh_token": bool(refresh_token),
                "auth_session_id": _coerce_nonempty_string(state_record.get("auth_session_id")),
                "return_path": _coerce_nonempty_string(state_record.get("return_path")),
            },
        )
        callback_response = OpenAIOAuthCallbackResponse(
            provider=_OPENAI_PROVIDER,
            status="stored",
            auth_source=_OPENAI_SOURCE_OAUTH,
            key_hint=_OPENAI_OAUTH_HINT,
            updated_at=row.get("updated_at") or now,
            expires_at=expires_at,
        )
        callback_return_path = _coerce_nonempty_string(state_record.get("return_path"))
        if redirect and callback_return_path:
            return RedirectResponse(url=callback_return_path, status_code=status.HTTP_303_SEE_OTHER)
        return callback_response
    except HTTPException as exc:
        _record_openai_oauth_counter(
            "byok_oauth_callback_failure_total",
            labels=_openai_oauth_metric_labels(
                reason=failure_reason or f"http_{exc.status_code}",
            ),
        )
        raise
    except Exception:
        _record_openai_oauth_counter(
            "byok_oauth_callback_failure_total",
            labels=_openai_oauth_metric_labels(
                reason=failure_reason or "unexpected_error",
            ),
        )
        raise


@router.get(
    "/keys/openai/oauth/status",
    response_model=OpenAIOAuthStatusResponse,
)
async def openai_oauth_status(
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> OpenAIOAuthStatusResponse:
    _require_openai_oauth_settings()
    user_id = _principal_user_id(principal)
    user_repo = await _get_user_repo()
    row = await user_repo.fetch_secret_for_user(user_id, _OPENAI_PROVIDER)
    if not row:
        return OpenAIOAuthStatusResponse(
            provider=_OPENAI_PROVIDER,
            connected=False,
            auth_source="none",
        )

    payload = _extract_payload_from_row(row)
    if not payload:
        return OpenAIOAuthStatusResponse(
            provider=_OPENAI_PROVIDER,
            connected=False,
            auth_source="none",
            updated_at=row.get("updated_at"),
            last_used_at=row.get("last_used_at"),
        )

    oauth_payload = _openai_source_payload(payload, _OPENAI_SOURCE_OAUTH)
    connected = _v2_source_available(payload, _OPENAI_SOURCE_OAUTH)
    active_source = _payload_active_auth_source(payload)
    if active_source not in {_OPENAI_SOURCE_API_KEY, _OPENAI_SOURCE_OAUTH}:
        active_source = "none"

    return OpenAIOAuthStatusResponse(
        provider=_OPENAI_PROVIDER,
        connected=connected,
        auth_source=active_source,
        updated_at=row.get("updated_at"),
        last_used_at=row.get("last_used_at"),
        expires_at=_parse_iso_datetime(oauth_payload.get("expires_at")),
        scope=_coerce_nonempty_string(oauth_payload.get("scope")),
    )


@router.post(
    "/keys/openai/oauth/refresh",
    response_model=OpenAIOAuthRefreshResponse,
    status_code=status.HTTP_200_OK,
)
async def refresh_openai_oauth(
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> OpenAIOAuthRefreshResponse:
    refresh_started = time.perf_counter()
    refresh_outcome = "failure"
    user_id: int | None = None
    try:
        settings = _require_openai_oauth_settings()
        user_id = _principal_user_id(principal)
        user_repo = await _get_user_repo()
        row = await user_repo.fetch_secret_for_user(user_id, _OPENAI_PROVIDER)
        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Key not found")

        existing_payload = _extract_payload_from_row(row)
        if not existing_payload:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="OAuth credential not found")

        merged_payload = _coerce_openai_payload_v2(existing_payload)
        oauth_payload = _openai_source_payload(merged_payload, _OPENAI_SOURCE_OAUTH)
        refresh_token = _coerce_nonempty_string(oauth_payload.get("refresh_token"))
        if not refresh_token:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="OAuth credential not found")

        token_url = _coerce_nonempty_string(getattr(settings, "OPENAI_OAUTH_TOKEN_URL", None))
        client_id = _coerce_nonempty_string(getattr(settings, "OPENAI_OAUTH_CLIENT_ID", None))
        client_secret = _coerce_nonempty_string(getattr(settings, "OPENAI_OAUTH_CLIENT_SECRET", None))
        if not token_url or not client_id or not client_secret:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="OpenAI OAuth is not fully configured",
            )

        token_payload = await _openai_oauth_token_exchange(
            token_url=token_url,
            form_data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id,
                "client_secret": client_secret,
            },
        )

        access_token = _coerce_nonempty_string(token_payload.get("access_token"))
        if not access_token:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="OpenAI OAuth token response is missing access_token",
            )

        next_refresh_token = _coerce_nonempty_string(token_payload.get("refresh_token")) or refresh_token
        token_type = (
            _coerce_nonempty_string(token_payload.get("token_type")) or oauth_payload.get("token_type") or "Bearer"
        )
        scope = _coerce_nonempty_string(token_payload.get("scope")) or _coerce_nonempty_string(
            oauth_payload.get("scope")
        )
        expires_in = _extract_positive_int(token_payload.get("expires_in"))
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=expires_in) if expires_in else None

        updated_oauth_payload = dict(oauth_payload)
        updated_oauth_payload["access_token"] = access_token
        updated_oauth_payload["refresh_token"] = next_refresh_token
        updated_oauth_payload["token_type"] = token_type
        updated_oauth_payload["issued_at"] = now.isoformat()
        if scope:
            updated_oauth_payload["scope"] = scope
        if expires_at is not None:
            updated_oauth_payload["expires_at"] = expires_at.isoformat()
        else:
            updated_oauth_payload.pop("expires_at", None)

        credentials = _openai_credentials_map(merged_payload)
        credentials[_OPENAI_SOURCE_OAUTH] = updated_oauth_payload
        merged_payload["credentials"] = credentials

        active_source = _payload_active_auth_source(merged_payload)
        if active_source not in {_OPENAI_SOURCE_API_KEY, _OPENAI_SOURCE_OAUTH}:
            active_source = (
                _OPENAI_SOURCE_API_KEY
                if _v2_source_available(merged_payload, _OPENAI_SOURCE_API_KEY)
                else _OPENAI_SOURCE_OAUTH
            )
        merged_payload["active_auth_source"] = active_source

        metadata_to_store = _row_metadata(row)
        key_hint = _payload_key_hint(merged_payload) or row.get("key_hint") or _OPENAI_OAUTH_HINT
        try:
            envelope = encrypt_byok_payload(merged_payload)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="BYOK encryption is not configured",
            ) from exc

        updated_row = await user_repo.upsert_secret(
            user_id=user_id,
            provider=_OPENAI_PROVIDER,
            encrypted_blob=dumps_envelope(envelope),
            key_hint=key_hint,
            metadata=metadata_to_store,
            updated_at=now,
            created_by=user_id,
            updated_by=user_id,
        )
        refresh_outcome = "success"
        await _emit_openai_oauth_audit_event(
            user_id=user_id,
            action="provider_oauth_refreshed",
            request=request,
            metadata={
                "scope": scope,
                "expires_at": expires_at.isoformat() if expires_at else None,
                "auth_source": active_source,
            },
        )
        return OpenAIOAuthRefreshResponse(
            provider=_OPENAI_PROVIDER,
            status="refreshed",
            updated_at=updated_row.get("updated_at") or now,
            expires_at=expires_at,
        )
    except HTTPException as exc:
        if user_id is not None:
            await _emit_openai_oauth_audit_event(
                user_id=user_id,
                action="provider_oauth_refresh_failed",
                request=request,
                result="failure",
                error_message=_coerce_nonempty_string(exc.detail),
                metadata={"status_code": str(exc.status_code)},
            )
        raise
    except Exception as exc:
        if user_id is not None:
            await _emit_openai_oauth_audit_event(
                user_id=user_id,
                action="provider_oauth_refresh_failed",
                request=request,
                result="failure",
                error_message=type(exc).__name__,
            )
        raise
    finally:
        elapsed_ms = (time.perf_counter() - refresh_started) * 1000.0
        _record_openai_oauth_counter(
            "byok_oauth_refresh_total",
            labels=_openai_oauth_metric_labels(outcome=refresh_outcome),
        )
        _record_openai_oauth_histogram(
            "byok_oauth_refresh_latency_ms",
            value=elapsed_ms,
            labels=_openai_oauth_metric_labels(),
        )


@router.delete(
    "/keys/openai/oauth",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
async def disconnect_openai_oauth(
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> Response:
    _require_openai_oauth_settings()
    user_id = _principal_user_id(principal)
    user_repo = await _get_user_repo()
    row = await user_repo.fetch_secret_for_user(user_id, _OPENAI_PROVIDER)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="OAuth credential not found")

    existing_payload = _extract_payload_from_row(row)
    if not existing_payload:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="OAuth credential not found")

    merged_payload = _coerce_openai_payload_v2(existing_payload)
    credentials = _openai_credentials_map(merged_payload)
    if _OPENAI_SOURCE_OAUTH not in credentials:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="OAuth credential not found")

    credentials.pop(_OPENAI_SOURCE_OAUTH, None)
    merged_payload["credentials"] = credentials
    now = datetime.now(timezone.utc)

    if _v2_source_available(merged_payload, _OPENAI_SOURCE_API_KEY):
        merged_payload["active_auth_source"] = _OPENAI_SOURCE_API_KEY
        metadata_to_store = _row_metadata(row)
        key_hint = _payload_key_hint(merged_payload) or row.get("key_hint") or ""
        try:
            envelope = encrypt_byok_payload(merged_payload)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="BYOK encryption is not configured",
            ) from exc
        await user_repo.upsert_secret(
            user_id=user_id,
            provider=_OPENAI_PROVIDER,
            encrypted_blob=dumps_envelope(envelope),
            key_hint=key_hint,
            metadata=metadata_to_store,
            updated_at=now,
            created_by=user_id,
            updated_by=user_id,
        )
        await _emit_openai_oauth_audit_event(
            user_id=user_id,
            action="provider_oauth_disconnected",
            request=request,
            metadata={
                "fallback_auth_source": _OPENAI_SOURCE_API_KEY,
                "credential_removed": _OPENAI_SOURCE_OAUTH,
            },
        )
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    deleted = await user_repo.delete_secret(
        user_id,
        _OPENAI_PROVIDER,
        revoked_by=user_id,
    )
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="OAuth credential not found")
    await _emit_openai_oauth_audit_event(
        user_id=user_id,
        action="provider_oauth_disconnected",
        request=request,
        metadata={
            "fallback_auth_source": "none",
            "credential_removed": _OPENAI_SOURCE_OAUTH,
        },
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/keys/openai/source",
    response_model=OpenAICredentialSourceSwitchResponse,
    status_code=status.HTTP_200_OK,
)
async def switch_openai_credential_source(
    payload: OpenAICredentialSourceSwitchRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> OpenAICredentialSourceSwitchResponse:
    _require_openai_oauth_settings()
    user_id = _principal_user_id(principal)
    user_repo = await _get_user_repo()
    row = await user_repo.fetch_secret_for_user(user_id, _OPENAI_PROVIDER)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Key not found")

    existing_payload = _extract_payload_from_row(row)
    if not existing_payload:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Key not found")

    merged_payload = _coerce_openai_payload_v2(existing_payload)
    requested_source = payload.auth_source
    if requested_source == _OPENAI_SOURCE_OAUTH:
        available = _v2_source_available(
            merged_payload,
            _OPENAI_SOURCE_OAUTH,
            require_access_for_oauth=True,
        )
        if not available:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Requested auth source is unavailable",
            )
    elif requested_source == _OPENAI_SOURCE_API_KEY:
        available = _v2_source_available(merged_payload, _OPENAI_SOURCE_API_KEY)
        if not available:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Requested auth source is unavailable",
            )

    merged_payload["active_auth_source"] = requested_source
    key_hint = _payload_key_hint(merged_payload) or row.get("key_hint") or ""
    metadata_to_store = _row_metadata(row)
    now = datetime.now(timezone.utc)

    try:
        envelope = encrypt_byok_payload(merged_payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="BYOK encryption is not configured",
        ) from exc

    updated_row = await user_repo.upsert_secret(
        user_id=user_id,
        provider=_OPENAI_PROVIDER,
        encrypted_blob=dumps_envelope(envelope),
        key_hint=key_hint,
        metadata=metadata_to_store,
        updated_at=now,
        created_by=user_id,
        updated_by=user_id,
    )
    await _emit_openai_oauth_audit_event(
        user_id=user_id,
        action="provider_oauth_source_switched",
        request=request,
        metadata={"auth_source": requested_source},
    )
    return OpenAICredentialSourceSwitchResponse(
        provider=_OPENAI_PROVIDER,
        auth_source=requested_source,
        updated_at=updated_row.get("updated_at") or now,
    )


@router.delete(
    "/keys/{provider}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
async def delete_user_provider_key(
    provider: str,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> Response:
    _require_byok_enabled()
    user_id = _principal_user_id(principal)
    provider_norm = normalize_provider_name(provider)
    repo = await _get_user_repo()
    deleted = await repo.delete_secret(
        user_id,
        provider_norm,
        revoked_by=user_id,
    )
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Key not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
