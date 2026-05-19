from __future__ import annotations

import hashlib
import json
import os
import secrets
from typing import Any, Callable
from urllib.parse import urlencode, urlparse

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from loguru import logger
from starlette.status import HTTP_403_FORBIDDEN, HTTP_500_INTERNAL_SERVER_ERROR

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    RequirePermission,
    RequireRole,
    get_auth_principal,
    get_db_transaction,
    get_org_policy_from_principal,
)
from tldw_Server_API.app.api.v1.schemas.connectors import (
    AuthorizeURLResponse,
    ConnectorAccount,
    ConnectorBrowseResponse,
    ConnectorPolicy,
    ConnectorProvider,
    ConnectorSource,
    ConnectorSourceSyncSummary,
    ConnectorSourceCreateRequest,
    ConnectorSourcePatchRequest,
    ConnectorSourceSyncStatus,
    ConnectorSourceSyncTriggerResponse,
    ConnectorWebhookCallbackResponse,
    ImportJob,
    SyncOptions,
)
from tldw_Server_API.app.api.v1.utils.pagination import build_cursor_pagination_meta
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_CONFIGURE
from tldw_Server_API.app.core.External_Sources import (
    get_connector_by_name,
)
from tldw_Server_API.app.core.External_Sources.connectors_service import (
    FILE_SYNC_PROVIDERS,
    consume_oauth_state,
    count_connectors_jobs_today,
    create_account,
    create_import_job,
    create_oauth_state,
    create_source,
    delete_account,
    get_account_email,
    get_account_for_user,
    get_account_tokens,
    get_policy,
    get_source_binding_health,
    get_source_by_id,
    get_source_by_webhook_subscription,
    get_source_sync_state,
    list_accounts,
    list_sources,
    record_webhook_receipt,
    update_source,
    upsert_source_sync_state,
    upsert_policy,
)
from tldw_Server_API.app.core.External_Sources.policy import (
    evaluate_policy_constraints,
    get_default_policy_from_env,
)
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.http_client import RetryPolicy as _RetryPolicy
from tldw_Server_API.app.core.http_client import afetch as _http_afetch
from tldw_Server_API.app.core.Logging.log_context import ensure_request_id, ensure_traceparent, get_ps_logger

router = APIRouter(prefix="/connectors", tags=["connectors"])


def _extract_request_base(request: Request | None) -> str:
    if request is None:
        return ""
    try:
        return str(request.base_url).rstrip("/")
    except (AttributeError, TypeError, ValueError):
        logger.debug("Failed to resolve base_url from request")
        return ""


def _is_connector_test_mode() -> bool:
    return (
        os.getenv("TEST_MODE", "").strip().lower() == "true"
        or os.getenv("TESTING", "").strip().lower() == "true"
    )


def _is_local_callback_base(base_url: str) -> bool:
    try:
        host = str(urlparse(base_url).hostname or "").strip().lower()
    except (AttributeError, TypeError, ValueError):
        return False
    return host in {"localhost", "127.0.0.1", "testserver"} or host.endswith(".localhost")


def _validate_redirect_base(base_url: str, *, setting_name: str = "CONNECTOR_REDIRECT_BASE_URL") -> str:
    normalized = str(base_url or "").strip().rstrip("/")
    if not normalized:
        return ""
    try:
        parsed = urlparse(normalized)
    except (AttributeError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"{setting_name} must be a valid absolute URL",
        ) from exc
    scheme = str(parsed.scheme or "").strip().lower()
    host = str(parsed.hostname or "").strip().lower()
    if not scheme or not host:
        raise HTTPException(
            status_code=500,
            detail=f"{setting_name} must be a valid absolute URL",
        )
    if scheme == "https":
        return normalized
    if scheme == "http" and _is_local_callback_base(normalized):
        return normalized
    raise HTTPException(
        status_code=500,
        detail=f"{setting_name} must use https unless it targets localhost",
    )


def _resolve_redirect_base(request: Request | None, conn) -> str:
    """Resolve connector redirect base for OAuth flows.

    Non-test mode requires an explicit CONNECTOR_REDIRECT_BASE_URL so request host
    headers cannot influence OAuth callback construction. Test mode keeps the
    local request/fixture fallback for isolated integration tests.
    """
    configured_base = _validate_redirect_base(os.getenv("CONNECTOR_REDIRECT_BASE_URL") or "")
    if configured_base:
        return configured_base

    request_base = _extract_request_base(request)
    if not _is_connector_test_mode():
        if request_base:
            logger.warning(
                "Rejecting request-derived base_url for connector OAuth redirect base in non-test mode: {}",
                request_base,
            )
        raise HTTPException(
            status_code=500,
            detail="CONNECTOR_REDIRECT_BASE_URL must be configured before enabling OAuth connectors",
        )

    if request_base and _is_local_callback_base(request_base):
        return _validate_redirect_base(request_base)

    resolved = _validate_redirect_base(
        getattr(conn, "redirect_base", "") or "",
        setting_name="connector redirect base",
    )
    if request_base and resolved and request_base != resolved:
        logger.warning(
            "Ignoring untrusted request base_url for connector OAuth redirect base: {}",
            request_base,
        )
    if resolved:
        return resolved
    if request is not None:
        logger.warning(
            "Redirect base could not be resolved; OAuth redirect_uri may be invalid (expected only in tests)"
        )
    return ""


def _resolve_webhook_callback_base(request: Request | None, conn) -> str:
    configured_base = _validate_redirect_base(os.getenv("CONNECTOR_REDIRECT_BASE_URL") or "")
    if configured_base:
        return configured_base
    connector_base = _validate_redirect_base(
        getattr(conn, "redirect_base", "") or "",
        setting_name="connector redirect base",
    )
    if connector_base:
        return connector_base
    request_base = _extract_request_base(request)
    if request_base and (_is_connector_test_mode() or _is_local_callback_base(request_base)):
        return _validate_redirect_base(request_base)
    raise HTTPException(
        status_code=500,
        detail="CONNECTOR_REDIRECT_BASE_URL must be configured before enabling webhook subscriptions",
    )


def _get_user_id(principal: AuthPrincipal) -> int:
    user_id = principal.user_id
    if user_id is None:
        raise HTTPException(status_code=401, detail="User ID not found in principal")
    try:
        return int(user_id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=401, detail="Invalid user ID in principal") from exc


def _normalize_policy_role(role_name: str | None) -> str:
    normalized = str(role_name or "").strip().lower()
    if normalized == "user":
        return "member"
    return normalized


def _principal_has_role(principal: AuthPrincipal, role_name: str) -> bool:
    target = _normalize_policy_role(role_name)
    return any(_normalize_policy_role(str(role)) == target for role in principal.roles or [])


def _principal_role_for_policy(principal: AuthPrincipal) -> str:
    if _principal_has_role(principal, "admin"):
        return "admin"
    for role in principal.roles or []:
        role_text = _normalize_policy_role(str(role))
        if role_text:
            return role_text
    return "member"


def get_connectors_job_counter() -> Callable[[int], int]:
    """Dependency to supply the connectors job counter (overrideable in tests)."""
    return count_connectors_jobs_today


def _as_str_or_none(value: Any) -> str | None:
    return str(value) if value is not None else None


def _derive_sync_state(sync_state: dict[str, Any] | None, active_job: dict[str, Any] | None) -> str:
    status = str((active_job or {}).get("status") or "").strip().lower()
    if status == "processing":
        return "running"
    if status == "queued":
        return "queued"
    if sync_state and sync_state.get("needs_full_rescan"):
        return "needs_full_rescan"
    if sync_state and sync_state.get("last_sync_failed_at"):
        return "failed"
    if sync_state and sync_state.get("last_sync_succeeded_at"):
        return "succeeded"
    return "idle"


def _summarize_job(job: dict[str, Any] | None) -> dict[str, Any] | None:
    if not job:
        return None
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    return {
        "id": str(job.get("id") or job.get("job_id") or job.get("uuid")),
        "type": str(job.get("job_type") or "import"),
        "status": str(job.get("status") or "queued"),
        "progress_pct": int(job.get("progress_percent") or 0),
        "counts": {
            "processed": int((result or {}).get("processed") or 0),
            "skipped": int((result or {}).get("skipped") or 0),
            "failed": int((result or {}).get("failed") or 0),
        },
    }


def _load_active_job(sync_state: dict[str, Any] | None) -> dict[str, Any] | None:
    active_job_id = str((sync_state or {}).get("active_job_id") or "").strip() or None
    if not active_job_id:
        return None
    try:
        from tldw_Server_API.app.core.Jobs.manager import JobManager

        return JobManager().get_job(int(active_job_id))
    except Exception:
        logger.warning("Failed to load active connectors job")
        return None


def _build_source_sync_summary(
    sync_state: dict[str, Any] | None,
    binding_health: dict[str, int] | None = None,
) -> ConnectorSourceSyncSummary | None:
    if not sync_state:
        return None
    active_job = _load_active_job(sync_state)
    binding_health = binding_health or {}
    return ConnectorSourceSyncSummary(
        state=_derive_sync_state(sync_state, active_job),
        sync_mode=str(sync_state.get("sync_mode") or "manual"),
        last_sync_succeeded_at=_as_str_or_none(sync_state.get("last_sync_succeeded_at")),
        last_sync_failed_at=_as_str_or_none(sync_state.get("last_sync_failed_at")),
        last_error=sync_state.get("last_error"),
        webhook_status=sync_state.get("webhook_status"),
        needs_full_rescan=bool(sync_state.get("needs_full_rescan")),
        active_job_id=str(sync_state.get("active_job_id") or "").strip() or None,
        tracked_item_count=int(binding_health.get("tracked_item_count") or 0),
        degraded_item_count=int(binding_health.get("degraded_item_count") or 0),
        duplicate_count=int(binding_health.get("duplicate_count") or 0),
        metadata_only_count=int(binding_health.get("metadata_only_count") or 0),
    )


def _webhook_payload_hash(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _webhook_receipt_key(notification: dict[str, Any]) -> tuple[str | None, str | None]:
    subscription_id = str(notification.get("subscriptionId") or "").strip() or None
    if not subscription_id:
        return None, None
    payload_hash = _webhook_payload_hash(notification)
    return f"{subscription_id}:{payload_hash}", payload_hash


def _drive_webhook_receipt_key(request: Request) -> tuple[str | None, str | None, str | None]:
    channel_id = str(request.headers.get("X-Goog-Channel-Id") or "").strip() or None
    message_number = str(request.headers.get("X-Goog-Message-Number") or "").strip() or None
    resource_state = str(request.headers.get("X-Goog-Resource-State") or "").strip() or None
    resource_id = str(request.headers.get("X-Goog-Resource-Id") or "").strip() or None
    if not channel_id or not message_number or not resource_state:
        return channel_id, None, None
    receipt_key = f"{channel_id}:{message_number}:{resource_state}:{resource_id or ''}"
    payload_hash = hashlib.sha256(receipt_key.encode("utf-8")).hexdigest()
    return channel_id, receipt_key, payload_hash


def _webhook_response(
    *,
    provider: str,
    status: str,
    queued_jobs: int = 0,
    duplicate_notifications: int = 0,
    ignored_notifications: int = 0,
    source_ids: list[int] | None = None,
) -> JSONResponse:
    payload = ConnectorWebhookCallbackResponse(
        provider=provider,
        status=status,
        queued_jobs=queued_jobs,
        duplicate_notifications=duplicate_notifications,
        ignored_notifications=ignored_notifications,
        source_ids=source_ids or [],
    )
    return JSONResponse(status_code=202, content=payload.model_dump())


def _provider_webhook_secret(source: dict[str, Any] | None) -> str | None:
    metadata = dict((source or {}).get("webhook_metadata") or {})
    secret = str(metadata.get("clientState") or metadata.get("token") or "").strip()
    return secret or None


def _matches_webhook_secret(expected: str | None, received: str | None) -> bool:
    if not expected or not received:
        return False
    return secrets.compare_digest(expected, received)


async def _queue_source_job(
    *,
    source_id: int,
    request: Request,
    principal: AuthPrincipal,
    org_policy: dict[str, Any],
    count_jobs_fn: Callable[[int], int],
    job_type: str = "import",
) -> ImportJob:
    user_id = _get_user_id(principal)
    role = _principal_role_for_policy(principal)
    qpr = org_policy.get("quotas_per_role") or {}
    limits = qpr.get(role) or {}
    max_jobs = int(limits.get("max_jobs_per_day") or 0)
    if max_jobs > 0:
        try:
            today_count = count_jobs_fn(user_id)
        except Exception as exc:
            logger.error("Connectors quota check failed")
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Daily import quota check failed",
            ) from exc
        if today_count >= max_jobs:
            raise HTTPException(status_code=429, detail="Daily import quota reached for your role")
    rid = ensure_request_id(request) if request is not None else None
    tp = ensure_traceparent(request) if request is not None else ""
    job = await create_import_job(user_id, source_id, request_id=rid, job_type=job_type)
    get_ps_logger(request_id=rid, ps_component="endpoint", ps_job_kind="connectors", traceparent=tp).info(
        "Queued connectors job: type=%s job_id=%s source_id=%s", job_type, job.get("id"), source_id
    )
    return ImportJob(**job)


def _gmail_connector_enabled() -> bool:
    try:
        return bool(settings.get("EMAIL_GMAIL_CONNECTOR_ENABLED", False))
    except Exception:
        return False


def _manual_sync_job_type(source: dict[str, Any], sync_state: dict[str, Any] | None) -> str:
    provider = str(source.get("provider") or "").strip().lower()
    if provider not in FILE_SYNC_PROVIDERS:
        return "import"
    if bool((sync_state or {}).get("needs_full_rescan")):
        return "repair_rescan"
    return "incremental_sync"


def _ensure_connector_provider_enabled(provider: str) -> str:
    normalized = str(provider or "").strip().lower()
    if normalized not in {"drive", "notion", "gmail", "onedrive", "zotero"}:
        raise HTTPException(status_code=404, detail=f"Unknown connector provider: {provider}")
    if normalized == "gmail" and not _gmail_connector_enabled():
        raise HTTPException(status_code=404, detail="Connector provider 'gmail' is disabled.")
    return normalized


def _option_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _normalize_source_options_for_provider(provider: str, options: dict[str, Any] | None) -> dict[str, Any]:
    normalized_options = dict(options or {})
    if provider == "zotero":
        if _option_enabled(normalized_options.get("recursive")):
            raise HTTPException(
                status_code=422,
                detail="Zotero collection sync is flat in v1; link child collections separately.",
            )
        normalized_options["recursive"] = False
    return normalized_options


@router.get("/providers", response_model=list[ConnectorProvider])
async def list_providers() -> list[ConnectorProvider]:
    providers: list[ConnectorProvider] = [
        ConnectorProvider(name="drive", scopes_required=["drive.readonly"], auth_type="oauth2"),
        ConnectorProvider(name="onedrive", scopes_required=["Files.Read"], auth_type="oauth2"),
        ConnectorProvider(name="notion", scopes_required=[], auth_type="oauth2"),
        ConnectorProvider(name="zotero", scopes_required=["library_access=1", "write_access=0", "notes_access=0"], auth_type="oauth1"),
    ]
    if _gmail_connector_enabled():
        providers.append(
            ConnectorProvider(
                name="gmail",
                scopes_required=["https://www.googleapis.com/auth/gmail.readonly"],
                auth_type="oauth2",
            )
        )
    return providers


@router.post("/providers/{provider}/authorize", response_model=AuthorizeURLResponse)
async def start_authorize(
    provider: str,
    request: Request,
    state: str | None = None,
    scopes: str | None = None,
    db=Depends(get_db_transaction),
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> AuthorizeURLResponse:
    provider = _ensure_connector_provider_enabled(provider)
    conn = get_connector_by_name(provider)
    redirect_base = _resolve_redirect_base(request, conn)
    if redirect_base:
        conn.redirect_base = redirect_base
    state = state or secrets.token_urlsafe(32)
    user_id = _get_user_id(principal)
    scopes_list = [s for s in (scopes or "").split(",") if s]
    oauth_state_metadata: dict[str, Any] | None = None
    if provider == "zotero":
        callback_params = urlencode({"state": state})
        callback_url = f"{redirect_base}/api/v1/connectors/providers/{provider}/callback?{callback_params}" if redirect_base else ""
        temporary_credentials = await conn.request_temporary_credential(callback_url)
        oauth_token = str((temporary_credentials or {}).get("oauth_token") or "").strip()
        oauth_token_secret = str((temporary_credentials or {}).get("oauth_token_secret") or "").strip()
        if not oauth_token or not oauth_token_secret:
            raise HTTPException(status_code=502, detail="Zotero authorization handshake failed.")
        oauth_state_metadata = {
            "oauth_token": oauth_token,
            "oauth_token_secret": oauth_token_secret,
        }
        scopes_list = [*scopes_list, f"oauth_token={oauth_token}"]
    await create_oauth_state(db, user_id, provider, state, metadata=oauth_state_metadata)
    url = conn.authorize_url(state=state, scopes=scopes_list or None, redirect_path=f"/api/v1/connectors/providers/{provider}/callback")
    return AuthorizeURLResponse(auth_url=url, state=state)


@router.get("/providers/{provider}/callback", response_model=ConnectorAccount)
async def oauth_callback(
    provider: str,
    request: Request,
    code: str | None = None,
    oauth_token: str | None = None,
    oauth_verifier: str | None = None,
    state: str | None = None,
    db=Depends(get_db_transaction),
    principal: AuthPrincipal = Depends(get_auth_principal),
    org_policy: dict[str, Any] = Depends(get_org_policy_from_principal),
) -> ConnectorAccount:
    provider = _ensure_connector_provider_enabled(provider)
    conn = get_connector_by_name(provider)
    pol = org_policy
    user_id = _get_user_id(principal)
    if not state:
        raise HTTPException(status_code=400, detail="Missing OAuth state")
    default_ttl_minutes = 10
    raw_ttl_minutes = os.getenv("CONNECTOR_OAUTH_STATE_TTL_MINUTES")
    ttl_minutes = default_ttl_minutes
    if raw_ttl_minutes is not None:
        raw_ttl_minutes = raw_ttl_minutes.strip()
        if not raw_ttl_minutes:
            logger.warning(
                "CONNECTOR_OAUTH_STATE_TTL_MINUTES is empty; using default {}",
                default_ttl_minutes,
            )
        else:
            try:
                ttl_minutes = int(raw_ttl_minutes)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid CONNECTOR_OAUTH_STATE_TTL_MINUTES={!r}; using default {}",
                    raw_ttl_minutes,
                    default_ttl_minutes,
                )
                ttl_minutes = default_ttl_minutes
    if ttl_minutes <= 0:
        logger.warning(
            "CONNECTOR_OAUTH_STATE_TTL_MINUTES must be positive; using default {}",
            default_ttl_minutes,
        )
        ttl_minutes = default_ttl_minutes
    oauth_state = await consume_oauth_state(
        db,
        user_id=user_id,
        provider=provider,
        state=state,
        max_age_minutes=ttl_minutes,
    )
    if not oauth_state:
        raise HTTPException(status_code=403, detail="Invalid or expired OAuth state")

    # Enforce org-level account linking role based on org policy; single-user
    # callers pass via their role/admin claims rather than global mode checks.
    try:
        role = _principal_role_for_policy(principal)
        required = _normalize_policy_role(str(pol.get("account_linking_role", "admin")))
        # Admin bypass
        if role != "admin" and required and role != required:
            raise HTTPException(status_code=403, detail="Account linking not permitted for your role")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Connector callback policy enforcement failed")
        raise HTTPException(
            status_code=403,
            detail="Account linking denied: policy enforcement failed",
        ) from e

    # Exchange code with redirect derived from env base + this path
    base = _resolve_redirect_base(request, conn)
    redirect_uri = f"{base.rstrip('/')}/api/v1/connectors/providers/{provider}/callback" if base else ""
    if provider == "zotero":
        state_metadata = oauth_state if isinstance(oauth_state, dict) else {}
        stored_oauth_token = str(state_metadata.get("oauth_token") or "").strip()
        oauth_token_secret = str(state_metadata.get("oauth_token_secret") or "").strip()
        if not oauth_token or not oauth_verifier:
            raise HTTPException(status_code=400, detail="Missing Zotero OAuth callback parameters")
        if stored_oauth_token and stored_oauth_token != oauth_token:
            raise HTTPException(status_code=403, detail="Zotero OAuth callback token mismatch")
        if not oauth_token_secret:
            raise HTTPException(status_code=400, detail="Missing Zotero OAuth temporary credential")
        code = json.dumps(
            {
                "oauth_token": oauth_token,
                "oauth_token_secret": oauth_token_secret,
                "oauth_verifier": oauth_verifier,
            },
            sort_keys=True,
        )
    elif not code:
        raise HTTPException(status_code=400, detail="Missing OAuth code")
    tokens = await conn.exchange_code(code, redirect_uri)
    # Optional email/workspace fetch for policy enforcement
    acct_email: str | None = None
    notion_workspace_id: str | None = None
    if provider == 'drive' and (tokens.get('access_token')):
        try:
            # userinfo endpoint provides email when scopes include 'email'
            resp = await _http_afetch(
                method="GET",
                url="https://openidconnect.googleapis.com/v1/userinfo",
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
                timeout=15,
                retry=_RetryPolicy(attempts=1),
            )
            try:
                if int(getattr(resp, "status_code", 0)) == 200:
                    info = resp.json()
                    if isinstance(info, dict):
                        acct_email = info.get('email')
                        tokens['email'] = acct_email
            finally:
                close = getattr(resp, "aclose", None)
                if callable(close):
                    await close()
                else:
                    close = getattr(resp, "close", None)
                    if callable(close):
                        close()
        except Exception:
            logger.debug("Failed to fetch drive userinfo")
    elif provider == 'notion':
        notion_workspace_id = tokens.get('workspace_id')
    # Enforce additional org policy constraints at callback across modes using
    # the same org policy surface.
    try:
        ok, why = evaluate_policy_constraints(
            pol,
            provider=provider,
            remote_path=None,
            notion_workspace_id=notion_workspace_id,
            account_email=acct_email,
        )
    except Exception as e:
        logger.error("Connector callback constraint evaluation failed")
        raise HTTPException(
            status_code=500,
            detail="Account linking denied: policy evaluation failed",
        ) from e
    if not ok:
        raise HTTPException(status_code=403, detail=why or "Account not permitted by org policy")

    acct = await create_account(
        db,
        user_id=user_id,
        provider=provider,
        display_name=str(tokens.get("display_name") or tokens.get("workspace_name") or f"{provider.title()} Account"),
        email=acct_email or tokens.get("email"),
        tokens=tokens,
    )
    return ConnectorAccount(
        id=int(acct.get("id")),
        provider=provider, display_name=str(acct.get("display_name")),
        email=acct.get("email"), created_at=str(acct.get("created_at")), connected=True,
    )


@router.get("/accounts", response_model=list[ConnectorAccount])
async def get_accounts(
    db=Depends(get_db_transaction), principal: AuthPrincipal = Depends(get_auth_principal)
) -> list[ConnectorAccount]:
    user_id = _get_user_id(principal)
    rows = await list_accounts(db, user_id)
    return [ConnectorAccount(id=int(r["id"]), provider=r["provider"], display_name=r.get("display_name") or "", email=r.get("email"), created_at=str(r.get("created_at")), connected=True) for r in rows]


@router.delete("/accounts/{account_id}")
async def remove_account(
    account_id: int, db=Depends(get_db_transaction), principal: AuthPrincipal = Depends(get_auth_principal)
) -> dict[str, Any]:
    user_id = _get_user_id(principal)
    await delete_account(db, user_id, account_id)
    return {"ok": True}


@router.get("/providers/{provider}/sources/browse", response_model=ConnectorBrowseResponse)
async def browse_provider_sources(
    provider: str,
    account_id: int = Query(..., ge=1),
    parent_remote_id: str | None = None,
    page_size: int = Query(50, ge=1, le=200),
    cursor: str | None = None,
    db=Depends(get_db_transaction),
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> ConnectorBrowseResponse:
    provider = _ensure_connector_provider_enabled(provider)
    user_id = _get_user_id(principal)
    tokens = await get_account_tokens(db, user_id, account_id)
    acct = await get_account_for_user(db, user_id, account_id) or {}
    if not tokens:
        raise HTTPException(status_code=404, detail="Account not found")
    acct_provider = str(acct.get("provider") or "").lower()
    if acct_provider and acct_provider != provider:
        raise HTTPException(status_code=400, detail="Account provider mismatch")
    email = await get_account_email(db, user_id, account_id)
    conn = get_connector_by_name(provider)
    account_payload = {**acct, "tokens": tokens, "email": email}
    if provider == "zotero" and not str(account_payload.get("provider_user_id") or "").strip():
        provider_user_id = str(tokens.get("provider_user_id") or tokens.get("userID") or tokens.get("user_id") or "").strip()
        if provider_user_id:
            account_payload["provider_user_id"] = provider_user_id
    # For Drive, parent_remote_id None implies root
    try:
        if provider == "drive":
            items, next_cursor = await conn.list_files(account_payload, parent_remote_id or "root", page_size=page_size, cursor=cursor)
        elif provider == "onedrive":
            items, next_cursor = await conn.list_files(account_payload, parent_remote_id or "root", page_size=page_size, cursor=cursor)
        elif provider == "notion":
            # Notion: treat parent_remote_id as workspace hint; we search globally for now
            items, next_cursor = await conn.list_sources(account_payload, parent_remote_id=parent_remote_id, page_size=page_size, cursor=cursor)
        elif provider == "zotero":
            items, next_cursor = await conn.list_collections(account_payload, cursor=cursor, page_size=page_size)
        else:
            items, next_cursor = [], None
    except Exception as e:
        logger.error("Connector browse failed")
        raise HTTPException(status_code=502, detail="Browse failed") from e
    return ConnectorBrowseResponse(
        items=items,
        limit=page_size,
        cursor=cursor,
        next_cursor=next_cursor,
        has_more=bool(next_cursor),
        pagination=build_cursor_pagination_meta(
            limit=page_size,
            cursor=cursor,
            next_cursor=next_cursor,
        ),
    )


@router.post("/sources", response_model=ConnectorSource)
async def add_source(
    request: Request,
    payload: ConnectorSourceCreateRequest,
    db=Depends(get_db_transaction),
    principal: AuthPrincipal = Depends(get_auth_principal),
    org_policy: dict[str, Any] = Depends(get_org_policy_from_principal),
) -> ConnectorSource:
    # payload keys: account_id, provider, remote_id, type, path, options
    account_id = int(payload.account_id)
    provider = str(payload.provider)
    provider = _ensure_connector_provider_enabled(provider)
    remote_id = str(payload.remote_id)
    type_ = str(payload.type)
    path = payload.path
    options = _normalize_source_options_for_provider(provider, payload.options)

    user_id = _get_user_id(principal)
    acct = await get_account_for_user(db, user_id, account_id)
    if not acct:
        raise HTTPException(status_code=404, detail="Account not found")
    acct_provider = str(acct.get("provider") or "").lower()
    if acct_provider and acct_provider != provider.lower():
        raise HTTPException(status_code=400, detail="Account provider mismatch")

    # Enforce org policy on provider/path for all modes; single-user callers
    # rely on their admin/role claims rather than mode flags.
    try:
        ok, why = evaluate_policy_constraints(org_policy, provider=provider, remote_path=path)
    except Exception as e:
        logger.error("Connector source policy evaluation failed")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Source denied: policy evaluation failed",
        ) from e
    if not ok:
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail=why or "Source denied by org policy")

    row = await create_source(
        db,
        account_id=account_id,
        provider=provider,
        remote_id=remote_id,
        type_=type_,
        path=path,
        options=options,
        enabled=True,
    )
    if provider in {"onedrive", "drive"}:
        try:
            tokens = await get_account_tokens(db, user_id, account_id)
            if tokens:
                conn = get_connector_by_name(provider)
                callback_base = _resolve_webhook_callback_base(request, conn)
                callback_url = f"{callback_base}/api/v1/connectors/providers/{provider}/webhook"
                resource_id = str(remote_id or "root").strip("/") or "root"
                if provider == "onedrive":
                    resource = {
                        "resource": "me/drive/root" if resource_id == "root" else f"me/drive/items/{resource_id}",
                        "change_type": "updated",
                        "clientState": secrets.token_urlsafe(24),
                    }
                else:
                    resource = {
                        "pageToken": options.get("cursor"),
                        "clientState": secrets.token_urlsafe(24),
                    }
                subscription = await conn.subscribe_webhook(
                    {**acct, "tokens": tokens},
                    resource=resource,
                    callback_url=callback_url,
                )
                sync_updates: dict[str, Any] = {
                    "sync_mode": "hybrid",
                    "webhook_status": "pending",
                }
                if subscription:
                    sync_updates["webhook_status"] = "active"
                    sync_updates["webhook_subscription_id"] = subscription.subscription_id
                    sync_updates["webhook_expires_at"] = subscription.expires_at
                    sync_updates["webhook_metadata"] = subscription.metadata or {}
                    if provider == "drive":
                        page_token = str((subscription.metadata or {}).get("pageToken") or "").strip()
                        if page_token:
                            sync_updates["cursor"] = page_token
                            sync_updates["cursor_kind"] = "drive_start_page_token"
                await upsert_source_sync_state(
                    db,
                    source_id=int(row.get("id")),
                    **sync_updates,
                )
        except Exception as exc:
            logger.warning("Connector webhook provisioning failed")
            await upsert_source_sync_state(
                db,
                source_id=int(row.get("id")),
                sync_mode="hybrid",
                webhook_status="failed",
                last_error=str(exc),
            )
    return ConnectorSource(
        id=int(row.get("id")),
        account_id=int(row.get("account_id")),
        provider=row.get("provider"),
        remote_id=row.get("remote_id"),
        type=row.get("type"),
        path=row.get("path"),
        options=SyncOptions(**(row.get("options") or {})),
        enabled=bool(row.get("enabled", True)),
        last_synced_at=str(row.get("last_synced_at")) if row.get("last_synced_at") else None,
    )


@router.get("/sources", response_model=list[ConnectorSource])
async def get_sources(
    db=Depends(get_db_transaction), principal: AuthPrincipal = Depends(get_auth_principal)
) -> list[ConnectorSource]:
    user_id = _get_user_id(principal)
    rows = await list_sources(db, user_id)
    out: list[ConnectorSource] = []
    for r in rows:
        sync_state = await get_source_sync_state(db, source_id=int(r.get("id")))
        binding_health = await get_source_binding_health(db, source_id=int(r.get("id")))
        out.append(
            ConnectorSource(
                id=int(r.get("id")),
                account_id=int(r.get("account_id")),
                provider=r.get("provider"),
                remote_id=r.get("remote_id"),
                type=r.get("type"),
                path=r.get("path"),
                options=SyncOptions(**(r.get("options") or {})),
                enabled=bool(r.get("enabled", True)),
                last_synced_at=str(r.get("last_synced_at")) if r.get("last_synced_at") else None,
                sync=_build_source_sync_summary(sync_state, binding_health),
            )
        )
    return out


@router.patch("/sources/{source_id}", response_model=ConnectorSource)
async def patch_source(
    source_id: int,
    payload: ConnectorSourcePatchRequest,
    db=Depends(get_db_transaction),
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> ConnectorSource:
    enabled = payload.enabled
    options = payload.options
    user_id = _get_user_id(principal)
    source = await get_source_by_id(db, user_id, source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    provider = str(source.get("provider") or "").strip().lower()
    if options is not None:
        options = _normalize_source_options_for_provider(provider, options)
    row = await update_source(db, user_id, source_id, enabled=enabled, options=options)
    if not row:
        raise HTTPException(status_code=404, detail="Source not found")
    return ConnectorSource(
        id=int(row.get("id")),
        account_id=int(row.get("account_id")),
        provider=row.get("provider"),
        remote_id=row.get("remote_id"),
        type=row.get("type"),
        path=row.get("path"),
        options=SyncOptions(**(row.get("options") or {})),
        enabled=bool(row.get("enabled", True)),
        last_synced_at=str(row.get("last_synced_at")) if row.get("last_synced_at") else None,
    )


@router.post("/sources/{source_id}/import", response_model=ImportJob)
async def import_source(
    source_id: int,
    request: Request,
    db=Depends(get_db_transaction),
    principal: AuthPrincipal = Depends(get_auth_principal),
    org_policy: dict[str, Any] = Depends(get_org_policy_from_principal),
    count_jobs_fn: Callable[[int], int] = Depends(get_connectors_job_counter),
) -> ImportJob:
    return await _queue_source_job(
        source_id=source_id,
        request=request,
        principal=principal,
        org_policy=org_policy,
        count_jobs_fn=count_jobs_fn,
        job_type="import",
    )


@router.get("/sources/{source_id}/sync", response_model=ConnectorSourceSyncStatus)
async def get_source_sync_status(
    source_id: int,
    db=Depends(get_db_transaction),
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> ConnectorSourceSyncStatus:
    user_id = _get_user_id(principal)
    source = await get_source_by_id(db, user_id, source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")

    sync_state = await get_source_sync_state(db, source_id=source_id) or {}
    binding_health = await get_source_binding_health(db, source_id=source_id)
    active_job = _load_active_job(sync_state)
    active_job_id = str(sync_state.get("active_job_id") or "").strip() or None

    return ConnectorSourceSyncStatus(
        source_id=int(source_id),
        provider=str(source.get("provider")),
        enabled=bool(source.get("enabled", True)),
        state=_derive_sync_state(sync_state, active_job),
        sync_mode=str(sync_state.get("sync_mode") or "manual"),
        cursor=sync_state.get("cursor"),
        cursor_kind=sync_state.get("cursor_kind"),
        last_bootstrap_at=_as_str_or_none(sync_state.get("last_bootstrap_at")),
        last_sync_started_at=_as_str_or_none(sync_state.get("last_sync_started_at")),
        last_sync_succeeded_at=_as_str_or_none(sync_state.get("last_sync_succeeded_at")),
        last_sync_failed_at=_as_str_or_none(sync_state.get("last_sync_failed_at")),
        last_error=sync_state.get("last_error"),
        retry_backoff_count=int(sync_state.get("retry_backoff_count") or 0),
        webhook_status=sync_state.get("webhook_status"),
        webhook_expires_at=_as_str_or_none(sync_state.get("webhook_expires_at")),
        needs_full_rescan=bool(sync_state.get("needs_full_rescan")),
        active_job_id=active_job_id,
        active_job_started_at=_as_str_or_none(sync_state.get("active_job_started_at")),
        active_job=_summarize_job(active_job),
        tracked_item_count=int(binding_health.get("tracked_item_count") or 0),
        degraded_item_count=int(binding_health.get("degraded_item_count") or 0),
        duplicate_count=int(binding_health.get("duplicate_count") or 0),
        metadata_only_count=int(binding_health.get("metadata_only_count") or 0),
    )


@router.post("/sources/{source_id}/sync", response_model=ConnectorSourceSyncTriggerResponse)
async def trigger_source_sync(
    source_id: int,
    request: Request,
    db=Depends(get_db_transaction),
    principal: AuthPrincipal = Depends(get_auth_principal),
    org_policy: dict[str, Any] = Depends(get_org_policy_from_principal),
    count_jobs_fn: Callable[[int], int] = Depends(get_connectors_job_counter),
) -> ConnectorSourceSyncTriggerResponse:
    user_id = _get_user_id(principal)
    source = await get_source_by_id(db, user_id, source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    sync_state = await get_source_sync_state(db, source_id=source_id) or {}

    job = await _queue_source_job(
        source_id=source_id,
        request=request,
        principal=principal,
        org_policy=org_policy,
        count_jobs_fn=count_jobs_fn,
        job_type=_manual_sync_job_type(source, sync_state),
    )
    return ConnectorSourceSyncTriggerResponse(
        source_id=int(source_id),
        provider=str(source.get("provider")),
        status="queued",
        job=job,
    )


@router.api_route(
    "/providers/{provider}/webhook",
    methods=["GET", "POST"],
    response_model=ConnectorWebhookCallbackResponse,
    responses={
        200: {
            "description": "Connector webhook callback result or plaintext validation challenge.",
            "content": {"text/plain": {}},
        },
    },
)
async def provider_webhook_callback(
    provider: str,
    request: Request,
    validation_token: str | None = Query(None, alias="validationToken"),
    db=Depends(get_db_transaction),
) -> ConnectorWebhookCallbackResponse | PlainTextResponse:
    provider = _ensure_connector_provider_enabled(provider)
    if validation_token is not None:
        return PlainTextResponse(validation_token)

    if provider == "drive":
        request_id = ensure_request_id(request)
        channel_id, receipt_key, payload_hash = _drive_webhook_receipt_key(request)
        if not channel_id or not receipt_key:
            return _webhook_response(provider=provider, status="ignored", ignored_notifications=1)

        source = await get_source_by_webhook_subscription(
            db,
            provider=provider,
            subscription_id=channel_id,
        )
        if not source:
            return _webhook_response(provider=provider, status="ignored", ignored_notifications=1)
        expected_secret = _provider_webhook_secret(source)
        received_secret = str(request.headers.get("X-Goog-Channel-Token") or "").strip() or None
        if not _matches_webhook_secret(expected_secret, received_secret):
            logger.warning(
                "Rejected drive webhook with invalid secret for source_id={} channel_id={}",
                source.get("id"),
                channel_id,
            )
            return _webhook_response(provider=provider, status="ignored", ignored_notifications=1)

        source_id = int(source.get("id"))
        is_new = await record_webhook_receipt(
            db,
            provider=provider,
            receipt_key=receipt_key,
            source_id=source_id,
            payload_hash=payload_hash,
        )
        if not is_new:
            return _webhook_response(
                provider=provider,
                status="duplicate",
                duplicate_notifications=1,
            )

        await create_import_job(
            int(source.get("user_id")),
            source_id,
            request_id=request_id,
            job_type="incremental_sync",
        )
        return _webhook_response(
            provider=provider,
            status="queued",
            queued_jobs=1,
            source_ids=[source_id],
        )

    if provider != "onedrive":
        raise HTTPException(status_code=404, detail="Webhook callback not supported for this provider")

    try:
        payload = await request.json()
    except Exception:
        payload = {}
    notifications = payload.get("value") if isinstance(payload, dict) else []
    if not isinstance(notifications, list):
        notifications = []

    request_id = ensure_request_id(request)
    queued_source_ids: list[int] = []
    queued_jobs = 0
    duplicate_notifications = 0
    ignored_notifications = 0
    seen_source_ids: set[int] = set()

    for notification in notifications:
        if not isinstance(notification, dict):
            ignored_notifications += 1
            continue
        receipt_key, payload_hash = _webhook_receipt_key(notification)
        subscription_id = str(notification.get("subscriptionId") or "").strip()
        if not receipt_key or not subscription_id:
            ignored_notifications += 1
            continue
        source = await get_source_by_webhook_subscription(
            db,
            provider=provider,
            subscription_id=subscription_id,
        )
        if not source:
            ignored_notifications += 1
            continue
        expected_secret = _provider_webhook_secret(source)
        received_secret = str(notification.get("clientState") or "").strip() or None
        if not _matches_webhook_secret(expected_secret, received_secret):
            logger.warning(
                "Rejected onedrive webhook with invalid secret for source_id={} subscription_id={}",
                source.get("id"),
                subscription_id,
            )
            ignored_notifications += 1
            continue
        source_id = int(source.get("id"))
        is_new = await record_webhook_receipt(
            db,
            provider=provider,
            receipt_key=receipt_key,
            source_id=source_id,
            payload_hash=payload_hash,
        )
        if not is_new:
            duplicate_notifications += 1
            continue
        if source_id in seen_source_ids:
            continue
        seen_source_ids.add(source_id)
        await create_import_job(
            int(source.get("user_id")),
            source_id,
            request_id=request_id,
            job_type="incremental_sync",
        )
        queued_source_ids.append(source_id)
        queued_jobs += 1

    status = "queued" if queued_jobs else ("duplicate" if duplicate_notifications else "ignored")
    return _webhook_response(
        provider=provider,
        status=status,
        queued_jobs=queued_jobs,
        duplicate_notifications=duplicate_notifications,
        ignored_notifications=ignored_notifications,
        source_ids=queued_source_ids,
    )


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: int) -> dict[str, Any]:
    try:
        from tldw_Server_API.app.core.Jobs.manager import JobManager
        jm = JobManager()
        job = jm.get_job(int(job_id))
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        # Trim payload if large
        job.pop("payload", None)
        return job
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get connector job status") from e


# Admin: Org-level policy
@router.get(
    "/admin/policy",
    response_model=ConnectorPolicy,
    dependencies=[
        Depends(get_auth_principal),
        Depends(RequireRole("admin")),
        Depends(RequirePermission(SYSTEM_CONFIGURE)),
    ],
)
async def get_org_policy(
    org_id: int = Query(..., ge=1),
    db=Depends(get_db_transaction),
) -> ConnectorPolicy:
    # In single-user mode, mimic multi-user but no enforcement beyond exposure
    pol = await get_policy(db, org_id)
    if not pol:
        pol = get_default_policy_from_env(org_id)
    return ConnectorPolicy(
        org_id=org_id,
        enabled_providers=[p for p in pol.get("enabled_providers") or [] if p],
        allowed_export_formats=[f for f in pol.get("allowed_export_formats") or [] if f],
        allowed_file_types=[t for t in pol.get("allowed_file_types") or [] if t],
        max_file_size_mb=int(pol.get("max_file_size_mb") or 0),
        account_linking_role=str(pol.get("account_linking_role") or "admin"),
        allowed_account_domains=[d for d in pol.get("allowed_account_domains") or [] if d],
        allowed_remote_paths=[p for p in pol.get("allowed_remote_paths") or [] if p],
        denied_remote_paths=[p for p in pol.get("denied_remote_paths") or [] if p],
        allowed_notion_workspaces=[w for w in pol.get("allowed_notion_workspaces") or [] if w],
        denied_notion_workspaces=[w for w in pol.get("denied_notion_workspaces") or [] if w],
        quotas_per_role=dict(pol.get("quotas_per_role") or {}),
    )


@router.put(
    "/admin/policy",
    response_model=ConnectorPolicy,
    dependencies=[
        Depends(get_auth_principal),
        Depends(RequireRole("admin")),
        Depends(RequirePermission(SYSTEM_CONFIGURE)),
    ],
)
async def upsert_org_policy(
    policy: ConnectorPolicy,
    db=Depends(get_db_transaction),
) -> ConnectorPolicy:
    # Enforce: only meaningful in multi-user mode; but persist anyway for forward usage
    p = policy.model_dump()
    await upsert_policy(db, int(policy.org_id), p)
    return await get_org_policy(org_id=int(policy.org_id), db=db)
