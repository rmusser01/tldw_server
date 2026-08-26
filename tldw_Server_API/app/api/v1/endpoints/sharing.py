# sharing.py
# Description: API endpoints for workspace sharing, share tokens, and admin
"""
Sharing API Endpoints
---------------------

Provides REST API endpoints for sharing workspaces with teams/orgs,
creating share links (tokens), and admin management.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import sqlite3
import threading
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response, status
from loguru import logger
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    User,
    get_request_user,
    rbac_rate_limit,
    require_permissions,
)
from tldw_Server_API.app.api.v1.API_Deps.jobs_deps import try_get_job_manager
from tldw_Server_API.app.api.v1.schemas.shared_workspace_recipient_schemas import (
    SharedWorkspaceBootstrapResponse,
    SharedWorkspaceChatRequest,
    SharedWorkspaceChatResponse,
    SharedWorkspaceCloneOperationResponse,
    SharedWorkspaceCloneRequest,
    SharedWorkspaceErrorResponse,
    SharedWorkspaceMessage,
    SharedWorkspaceMessagePage,
    SharedWorkspaceSource,
    SharedWorkspaceSourcePage,
    SharedWorkspaceSourcePreview,
)
from tldw_Server_API.app.api.v1.utils.pagination import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.utils.shared_workspace_recipient_route import (
    SharedWorkspaceRecipientRoute,
    recipient_error_detail,
)
from tldw_Server_API.app.core.AuthNZ.byok_runtime import resolve_byok_credentials
from tldw_Server_API.app.core.Chat.chat_service import ChatConfigurationError
from tldw_Server_API.app.core.Chat.chat_target_resolution import (
    resolve_chat_provider_identity,
    resolve_chat_target,
)
from tldw_Server_API.app.core.config import load_and_log_configs
from tldw_Server_API.app.core.custom_openai_providers import (
    custom_openai_provider_number,
    custom_openai_section_name,
)
from tldw_Server_API.app.core.DB_Management.chacha.shared_workspace_chat_store import (
    SharedWorkspaceCursorInputError,
)
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    IdempotentOperationConflict,
    IdempotentOperationConflictReason,
    IdempotentOperationDisposition,
    IdempotentOperationUnavailableError,
)
from tldw_Server_API.app.core.LLM_Calls.provider_metadata import (
    provider_requires_api_key,
)
from tldw_Server_API.app.core.Sharing.shared_workspace_chat_service import (
    SharedWorkspaceChatService,
    SharedWorkspaceChatServiceError,
    SharedWorkspaceSourceChanged,
)
from tldw_Server_API.app.core.Sharing.shared_workspace_clone_operations import (
    CloneOperationNotFound,
    CloneOperationUnavailable,
    build_clone_admission_command,
    project_clone_operation,
)

from ..schemas.sharing_schemas import (
    AdminShareListResponse,
    AuditEventResponse,
    AuditLogResponse,
    CreateTokenRequest,
    PrototypeLinkExchangeRequest,
    PrototypeLinkExchangeResponse,
    PublicImportRequest,
    PublicSharePreview,
    ResourceType,
    SharedWithMeItem,
    SharedWithMeResponse,
    ShareListResponse,
    ShareResponse,
    ShareWorkspaceRequest,
    TokenListResponse,
    TokenResponse,
    UpdateConfigRequest,
    UpdateShareRequest,
    VerifyPasswordRequest,
    VerifyPasswordResponse,
)
from ..utils.prototype_error_contract import (
    PROTOTYPE_LINK_ERROR_RESPONSES,
    prototype_http_error,
)

router = APIRouter(prefix="/sharing", tags=["sharing"])
recipient_router = APIRouter(
    prefix="/shared-with-me/{share_id}",
    route_class=SharedWorkspaceRecipientRoute,
)

_RECIPIENT_PERMISSION_DETAIL = recipient_error_detail("sharing_permission_required")
_RECIPIENT_READ_RATE_DETAIL = recipient_error_detail("shared_workspace_rate_limited")
_RECIPIENT_CHAT_RATE_DETAIL = recipient_error_detail("shared_chat_rate_limited")
_RECIPIENT_CLONE_RATE_DETAIL = recipient_error_detail("shared_workspace_rate_limited")
_RECIPIENT_READ_DEPENDENCIES = [
    Depends(require_permissions("sharing.read", detail=_RECIPIENT_PERMISSION_DETAIL)),
    Depends(rbac_rate_limit("sharing.read", detail=_RECIPIENT_READ_RATE_DETAIL)),
]
_RECIPIENT_CHAT_DEPENDENCIES = [
    Depends(require_permissions("sharing.read", detail=_RECIPIENT_PERMISSION_DETAIL)),
    Depends(rbac_rate_limit("sharing.read", detail=_RECIPIENT_CHAT_RATE_DETAIL)),
]
_RECIPIENT_CLONE_DEPENDENCIES = [
    Depends(require_permissions("sharing.read", detail=_RECIPIENT_PERMISSION_DETAIL)),
    Depends(rbac_rate_limit("sharing.clone", detail=_RECIPIENT_CLONE_RATE_DETAIL)),
]


def _recipient_error_responses(*, unavailable_description: str) -> dict[int, dict[str, Any]]:
    descriptions = {
        401: "Authentication is required.",
        403: "The sharing.read permission is required.",
        404: "The shared workspace or source was not found.",
        422: "The recipient request is invalid.",
        429: "The recipient request is temporarily rate limited.",
        503: unavailable_description,
    }
    return {
        status_code: {
            "model": SharedWorkspaceErrorResponse,
            "description": description,
        }
        for status_code, description in descriptions.items()
    }


_RECIPIENT_READ_ERROR_RESPONSES = _recipient_error_responses(
    unavailable_description="Shared workspace data is temporarily unavailable."
)
_RECIPIENT_CHAT_ERROR_RESPONSES = _recipient_error_responses(
    unavailable_description="Shared workspace generation is not available."
)
_RECIPIENT_CHAT_ERROR_RESPONSES[409] = {
    "model": SharedWorkspaceErrorResponse,
    "description": "The shared chat receipt or source scope conflicts.",
}
_RECIPIENT_CLONE_ERROR_RESPONSES = _recipient_error_responses(
    unavailable_description="Workspace copy status is temporarily unavailable."
)
_RECIPIENT_CLONE_ERROR_RESPONSES[409] = {
    "model": SharedWorkspaceErrorResponse,
    "description": "The idempotency key or active copy conflicts with this request.",
}


# ── Lazy service construction ──


def _get_repo():
    """Lazily construct the SharedWorkspaceRepo from the AuthNZ DB pool."""
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.repos.shared_workspace_repo import SharedWorkspaceRepo

    async def _build():
        return SharedWorkspaceRepo(db_pool=await get_db_pool())

    return _build()


def _get_token_service():
    """Lazily construct the share-token service from the shared workspace repo."""
    from tldw_Server_API.app.core.Sharing.share_token_service import ShareTokenService

    async def _build():
        return ShareTokenService(await _maybe_await(_get_repo()))

    return _build()


def _get_prototype_repo():
    """Lazily construct the prototype workspace repo for share-token checks."""
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.repos.prototype_workspaces_repo import (
        PrototypeWorkspacesRepo,
    )

    async def _build():
        return PrototypeWorkspacesRepo(db_pool=await get_db_pool())

    return _build()


def _get_prototype_access_service():
    """Lazily construct the prototype private-link access service."""
    from tldw_Server_API.app.core.Prototype_Workspaces.access import PrototypeAccessService

    async def _build():
        return PrototypeAccessService(await _maybe_await(_get_prototype_repo()))

    return _build()


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


_cached_audit_service: ShareAuditService | None = None  # noqa: F821


def _get_audit_service():
    from tldw_Server_API.app.core.Sharing.share_audit_service import ShareAuditService

    global _cached_audit_service
    if _cached_audit_service is None:
        _cached_audit_service = ShareAuditService()
    return _cached_audit_service


def _safe_exception_type(exc: BaseException) -> str:
    exc_type = exc.__class__.__name__
    if exc_type and all(char.isalnum() or char == "_" for char in exc_type):
        return exc_type
    return "Exception"


def _is_duplicate_share_error(exc: BaseException) -> bool:
    if isinstance(exc, asyncpg.UniqueViolationError):
        return exc.constraint_name == "uq_shared_workspaces_scope"
    sqlite_scope_error = (
        "unique constraint failed: shared_workspaces.workspace_id, "
        "shared_workspaces.owner_user_id, shared_workspaces.share_scope_type, "
        "shared_workspaces.share_scope_id"
    )
    return (
        isinstance(exc, sqlite3.IntegrityError)
        and str(exc).strip().lower() == sqlite_scope_error
    )


async def _audit_log_best_effort(audit: Any, event_type: str, **kwargs: Any) -> None:
    try:
        await audit.log(event_type, **kwargs)
    except Exception as exc:
        logger.warning(
            f"Sharing audit log failed; event_type={event_type}; "
            f"exception_type={_safe_exception_type(exc)}"
        )


async def shutdown_sharing_audit_service() -> None:
    """Stop the cached ShareAuditService (and its writer) during app shutdown."""
    global _cached_audit_service
    if _cached_audit_service is not None:
        await _cached_audit_service.stop()
        _cached_audit_service = None


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _request_is_secure(request: Request) -> bool:
    """Mirror SecurityHeadersMiddleware HTTPS detection for cookie security."""
    forwarded_proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower()
    if forwarded_proto:
        return forwarded_proto == "https"
    return request.url.scheme == "https"


def _coerce_int(value: Any) -> int | None:
    """Return an integer value when possible without raising for malformed inputs."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


async def _get_owned_prototype_workspace(
    prototype_workspace_id: str,
    owner_user_id: Any,
    *,
    use_prototype_error_contract: bool = False,
) -> dict[str, Any]:
    """Return a prototype workspace only when the expected owner matches."""
    repo = await _maybe_await(_get_prototype_repo())
    workspace = await repo.get_workspace(prototype_workspace_id)
    expected_owner_id = _coerce_int(owner_user_id)
    actual_owner_id = _coerce_int(workspace.get("owner_user_id")) if workspace else None
    if not workspace or expected_owner_id is None or actual_owner_id != expected_owner_id:
        if use_prototype_error_contract:
            raise prototype_http_error(
                status_code=status.HTTP_404_NOT_FOUND,
                category="invalid_or_unavailable_link",
                message="Prototype link is unavailable",
                frontend_state="link_unavailable",
            )
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resource not found")
    return workspace


# ── IP-based rate limiter for public (unauthenticated) endpoints ──
# 10 requests per minute per IP, as specified in the design doc.

_PUBLIC_RATE_LIMIT_WINDOW = 60  # seconds
_PUBLIC_RATE_LIMIT_MAX = 10
_public_rate_lock = threading.Lock()
_public_rate_buckets: dict[str, list[float]] = defaultdict(list)


def _check_public_rate_limit(request: Request) -> None:
    """Raise 429 if IP exceeds 10 req/min on public endpoints."""
    ip = _client_ip(request)
    now = time.monotonic()
    with _public_rate_lock:
        bucket = _public_rate_buckets[ip]
        # Prune old entries
        cutoff = now - _PUBLIC_RATE_LIMIT_WINDOW
        _public_rate_buckets[ip] = [t for t in bucket if t > cutoff]
        bucket = _public_rate_buckets[ip]
        if len(bucket) >= _PUBLIC_RATE_LIMIT_MAX:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Try again later.",
            )
        bucket.append(now)


# ── Scope membership validation helper ──


async def _validate_user_has_share_access(share: dict, user: User) -> None:
    """Verify the user belongs to the team/org that a share targets."""
    if share["owner_user_id"] == user.id:
        return  # Owner always has access to their own shares

    scope_type = share.get("share_scope_type")
    scope_id = share.get("share_scope_id")

    team_ids = getattr(user, "team_ids", None) or []
    org_ids = getattr(user, "org_ids", None) or []

    if scope_type == "team" and scope_id in team_ids:
        return
    if scope_type == "org" and scope_id in org_ids:
        return

    raise HTTPException(status_code=403, detail="You do not have access to this share")


def _coerce_scope_ids(values: Any) -> set[int]:
    """Normalize claim/model scope IDs without trusting their input type."""
    scope_ids: set[int] = set()
    if values is None:
        return scope_ids
    if not isinstance(values, (list, tuple, set)):
        values = [values]
    for value in values:
        coerced = _coerce_int(value)
        if coerced is not None:
            scope_ids.add(coerced)
    return scope_ids


async def _validate_share_target_scope(body: ShareWorkspaceRequest, user: User) -> None:
    """Only create shares for scopes the current user can actually address."""
    scope_id = _coerce_int(body.share_scope_id)
    if scope_id is None:
        raise HTTPException(status_code=422, detail="Invalid share scope ID.")

    scope_type = body.share_scope_type.value
    if scope_type == "team":
        if scope_id not in _coerce_scope_ids(getattr(user, "team_ids", None)):
            raise HTTPException(
                status_code=403,
                detail="You can only share with teams you belong to.",
            )
        return

    if scope_type == "org":
        if scope_id not in _coerce_scope_ids(getattr(user, "org_ids", None)):
            raise HTTPException(
                status_code=403,
                detail="You can only share with organizations you belong to.",
            )
        return

    raise HTTPException(status_code=422, detail="Invalid share scope type.")


# ── Workspace ownership verification helper ──


async def _verify_workspace_ownership(workspace_id: str, user: User) -> None:
    """Verify the user owns the workspace before sharing it."""
    try:
        from ..API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user_id

        db = await get_chacha_db_for_user_id(user.id)
        ws = db.get_workspace(workspace_id)
        if ws is None:
            raise HTTPException(
                status_code=404,
                detail=f"Workspace '{workspace_id}' not found in your account",
            )
    except HTTPException:
        raise
    except Exception as exc:
        # In single-user mode, workspace validation may not be available
        from ....core.AuthNZ.settings import get_settings

        if get_settings().auth_mode == "single_user":
            logger.warning("Workspace ownership check skipped in single-user mode")
            return
        logger.error("Workspace ownership check failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not verify workspace ownership due to a database error.",
        ) from exc


# ── Recipient shared-workspace read plane ──


def _recipient_http_error(
    status_code: int,
    code: str,
    *,
    retry_after_ms: int | None = None,
    operation_id: str | None = None,
) -> HTTPException:
    detail = recipient_error_detail(code)
    if retry_after_ms is not None:
        detail["retry_after_ms"] = max(0, min(1_800_000, int(retry_after_ms)))
    if operation_id is not None:
        try:
            detail["operation_id"] = str(UUID(operation_id))
        except (TypeError, ValueError, AttributeError):
            pass
    return HTTPException(status_code=status_code, detail=detail)


async def get_shared_workspace_access_service():
    """Build the authoritative access service used by recipient routes."""
    from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
    from tldw_Server_API.app.core.Sharing.shared_workspace_access_service import (
        SharedWorkspaceAccessService,
    )

    from ..API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_owner

    repo = await _maybe_await(_get_repo())
    return SharedWorkspaceAccessService(
        repo,
        AuthnzUsersRepo(db_pool=repo.db_pool),
        get_chacha_db_for_owner,
    )


async def _resolve_recipient_access(
    service: Any,
    *,
    share_id: int,
    recipient_user_id: int,
):
    from tldw_Server_API.app.core.Sharing.shared_workspace_access_service import (
        SharedWorkspaceNotFound,
        SharedWorkspaceUnavailable,
    )

    try:
        return await service.resolve(
            share_id=share_id,
            recipient_user_id=recipient_user_id,
        )
    except SharedWorkspaceNotFound as exc:
        raise _recipient_http_error(404, "shared_workspace_not_found") from exc
    except SharedWorkspaceUnavailable as exc:
        raise _recipient_http_error(503, "shared_workspace_unavailable") from exc


async def _load_recipient_workspace_sources(context: Any) -> list[dict[str, Any]]:
    """Open the authorized owner's workspace DB only after access resolution."""
    from ..API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_owner

    try:
        owner_db = await get_chacha_db_for_owner(context.owner_user_id)
        sources = owner_db.list_workspace_sources(context.workspace_id)
        return [dict(source) for source in sources]
    except Exception as exc:
        raise _recipient_http_error(503, "shared_workspace_unavailable") from exc


def _recipient_partial_error(
    *,
    area: str,
    code: str,
    message: str,
    retryable: bool,
) -> dict[str, Any]:
    return {
        "area": area,
        "code": code,
        "message": message,
        "retryable": retryable,
    }


async def _project_recipient_source_status(
    context: Any,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    """Project source readiness from owner media and optional Jobs."""
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import managed_media_db_for_owner
    from tldw_Server_API.app.api.v1.API_Deps.jobs_deps import try_get_job_manager
    from tldw_Server_API.app.core.Workspaces.job_status import (
        list_recent_workspace_source_ingest_jobs,
    )
    from tldw_Server_API.app.core.Workspaces.status_projection import (
        build_source_status_projection,
    )

    partial_errors: list[dict[str, Any]] = []
    try:
        jobs = list_recent_workspace_source_ingest_jobs(
            try_get_job_manager(),
            owner_user_id=context.owner_user_id,
        )
    except Exception as exc:
        logger.bind(
            owner_user_id=context.owner_user_id,
            workspace_id=context.workspace_id,
        ).warning(
            "Recipient shared-workspace Jobs status is unavailable: {}",
            exc,
        )
        jobs = []
        partial_errors.append(
            _recipient_partial_error(
                area="source_status",
                code="jobs_status_unavailable",
                message="Live source progress is temporarily unavailable.",
                retryable=True,
            )
        )
    try:
        with managed_media_db_for_owner(context.owner_user_id) as media_db:
            projection = build_source_status_projection(
                workspace_id=context.workspace_id,
                sources=sources,
                media_db=media_db,
                jobs=jobs,
            )
    except Exception as exc:
        logger.bind(
            owner_user_id=context.owner_user_id,
            workspace_id=context.workspace_id,
        ).warning(
            "Recipient shared-workspace source readiness is unavailable: {}",
            exc,
        )
        projection = build_source_status_projection(
            workspace_id=context.workspace_id,
            sources=sources,
            media_db=None,
            jobs=jobs,
        )
        partial_errors.append(
            _recipient_partial_error(
                area="source_status",
                code="source_readiness_unavailable",
                message="Source readiness is temporarily unavailable.",
                retryable=True,
            )
        )
    projection["partial_errors"] = partial_errors[:8]
    return projection


async def _resolve_recipient_generation_default(context: Any) -> dict[str, Any]:
    """Resolve a disclosure-safe default using only the current share scope."""
    unavailable = {
        "provider": None,
        "model": None,
        "ready": False,
        "reason_code": "no_provider_configured",
    }
    try:
        target = resolve_chat_target(
            requested_provider=None,
            requested_model=None,
        )
        team_ids = (
            [context.share_scope_id] if context.share_scope_type == "team" else []
        )
        org_ids = (
            [context.share_scope_id] if context.share_scope_type == "org" else []
        )
        credentials = await resolve_byok_credentials(
            target.provider,
            user_id=context.recipient_user_id,
            request=None,
            team_ids=team_ids,
            org_ids=org_ids,
            trusted_base_url_override=False,
        )
        if provider_requires_api_key(target.provider) and not credentials.api_key:
            return unavailable
        return {
            "provider": target.provider,
            "model": target.model,
            "ready": True,
            "reason_code": None,
        }
    except Exception:  # noqa: BLE001 - bootstrap readiness never exposes internals.
        return unavailable


async def _load_recipient_chat_history(
    context: Any,
    *,
    before: str | None,
    limit: int,
) -> Any:
    """Read recipient-owned history without creating a thread."""
    from ..API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user_id

    recipient_db = await get_chacha_db_for_user_id(context.recipient_user_id)
    store = recipient_db.shared_workspace_chat_store
    page = await run_in_threadpool(
        store.list_messages,
        share_id=context.share_id,
        before=before,
        limit=limit,
    )
    thread = await run_in_threadpool(store.get_thread, share_id=context.share_id)
    return page, (thread.conversation_id if thread is not None else None)


@dataclass(frozen=True)
class _RecipientChatResources:
    chat_service: SharedWorkspaceChatService


async def _load_recipient_chat_store(context: Any) -> Any:
    """Open only recipient-owned persistence for thread and receipt handling."""
    from ..API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user_id

    recipient_db = await get_chacha_db_for_user_id(context.recipient_user_id)
    return recipient_db.shared_workspace_chat_store


@asynccontextmanager
async def _load_recipient_chat_resources(context: Any):
    """Open authorized owner data only for a claimed generation call."""
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import (
        get_media_db_path_for_rag,
        managed_media_db_for_owner,
    )

    from ..API_Deps.ChaCha_Notes_DB_Deps import (
        get_chacha_db_for_owner,
    )

    owner_db = await get_chacha_db_for_owner(context.owner_user_id)
    with managed_media_db_for_owner(context.owner_user_id) as media_db:
        yield _RecipientChatResources(
            chat_service=SharedWorkspaceChatService(
                owner_chacha_db=owner_db,
                owner_media_db=media_db,
                owner_media_db_path=get_media_db_path_for_rag(media_db),
                owner_user_id=context.owner_user_id,
                workspace_id=context.workspace_id,
            ),
        )


def _shared_chat_fingerprint(body: SharedWorkspaceChatRequest) -> str:
    payload = {
        "model": (body.model or "").strip().lower() or None,
        "provider": (body.provider or "").strip().lower() or None,
        "query": body.query.strip(),
        "source_ids": sorted(set(body.source_scope.source_ids)),
        "source_mode": body.source_scope.mode,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


_SHARED_CHAT_PROVIDER_CONFIG_SECTIONS = {
    "llama.cpp": "llama_api",
    "kobold": "kobold_api",
    "local-llm": "local_llm",
    "ooba": "ooba_api",
    "tabbyapi": "tabby_api",
}


def _shared_chat_lease_seconds(body: SharedWorkspaceChatRequest) -> int:
    """Bound a receipt lease around the selected provider's call timeout."""

    provider = resolve_chat_provider_identity(
        requested_provider=body.provider,
        requested_model=body.model,
    )
    custom_number = custom_openai_provider_number(provider)
    if custom_number is not None:
        section = custom_openai_section_name(custom_number)
    else:
        section = _SHARED_CHAT_PROVIDER_CONFIG_SECTIONS.get(
            provider,
            f"{provider.replace('.', '_').replace('-', '_')}_api",
        )

    timeout_seconds = 90.0
    try:
        config = load_and_log_configs()
        provider_config = config.get(section) if isinstance(config, dict) else None
        configured_timeout = (
            provider_config.get("api_timeout")
            if isinstance(provider_config, dict)
            else None
        )
        parsed_timeout = float(configured_timeout)
        if math.isfinite(parsed_timeout) and parsed_timeout > 0:
            timeout_seconds = parsed_timeout
    except (TypeError, ValueError):
        timeout_seconds = 90.0

    return max(300, min(1_800, math.ceil(timeout_seconds + 60)))


def _same_recipient_access(original: Any, current: Any) -> bool:
    fields = (
        "share_id",
        "workspace_id",
        "owner_user_id",
        "recipient_user_id",
        "share_scope_type",
        "share_scope_id",
    )
    return all(getattr(original, field, None) == getattr(current, field, None) for field in fields)


async def _reauthorize_recipient_chat(
    access_service: Any,
    context: Any,
) -> None:
    current = await _resolve_recipient_access(
        access_service,
        share_id=context.share_id,
        recipient_user_id=context.recipient_user_id,
    )
    if not _same_recipient_access(context, current):
        raise _recipient_http_error(404, "shared_workspace_not_found")


def _shared_chat_service_http_error(exc: SharedWorkspaceChatServiceError) -> HTTPException:
    status_by_code = {
        "invalid_shared_chat_request": 422,
        "source_subset_required": 409,
        "shared_source_changed": 409,
        "no_relevant_evidence": 409,
        "shared_chat_context_too_large": 422,
        "retrieval_unavailable": 503,
        "no_provider_configured": 503,
        "generation_failed": 503,
        "shared_workspace_unavailable": 503,
    }
    code = exc.code if exc.code in status_by_code else "shared_workspace_unavailable"
    return _recipient_http_error(status_by_code[code], code)


def _shared_chat_response(turn: Any, *, replayed: bool) -> SharedWorkspaceChatResponse:
    return SharedWorkspaceChatResponse(
        request_id=turn.request_id,
        conversation_id=turn.conversation_id,
        turn={
            "user_message": {
                "message_id": turn.user_message.message_id,
                "role": turn.user_message.role,
                "content": turn.user_message.content,
                "created_at": turn.user_message.created_at,
            },
            "assistant_message": {
                "message_id": turn.assistant_message.message_id,
                "role": turn.assistant_message.role,
                "content": turn.assistant_message.content,
                "created_at": turn.assistant_message.created_at,
            },
        },
        citations=list(turn.citations),
        generation={"provider": turn.provider, "model": turn.model},
        source_scope={
            "mode": turn.source_mode,
            "effective_source_count": turn.effective_source_count,
        },
        replay={"replayed": replayed},
    )


@dataclass(frozen=True)
class _SharedChatFailureTransition:
    completed_response: SharedWorkspaceChatResponse | None = None
    replacement_error: HTTPException | None = None


async def _mark_shared_chat_failure(
    store: Any,
    claim: Any,
    code: str,
) -> _SharedChatFailureTransition:
    method = store.mark_conflicted if code == "shared_source_changed" else store.mark_retryable
    try:
        transitioned = await run_in_threadpool(method, claim=claim, error_code=code)
    except Exception:
        logger.warning("Shared chat receipt failure transition failed")
        transitioned = False
    if transitioned:
        return _SharedChatFailureTransition()

    try:
        winner = await run_in_threadpool(
            store.reload_claim_state,
            claim=claim,
            now=datetime.now(timezone.utc),
        )
    except Exception:
        logger.warning("Shared chat receipt winner reload failed")
        winner = None

    if (
        winner is not None
        and winner.disposition == "replay"
        and winner.completed_turn is not None
    ):
        return _SharedChatFailureTransition(
            completed_response=_shared_chat_response(
                winner.completed_turn,
                replayed=True,
            )
        )
    if (
        winner is not None
        and winner.disposition == "in_progress"
        and winner.lease_epoch > claim.lease_epoch
    ):
        return _SharedChatFailureTransition(
            replacement_error=_recipient_http_error(
                409,
                "request_in_progress",
                retry_after_ms=winner.retry_after_ms,
            )
        )
    return _SharedChatFailureTransition(
        replacement_error=_recipient_http_error(
            503,
            "shared_workspace_unavailable",
        )
    )


async def _audit_shared_chat(
    audit: Any,
    *,
    context: Any,
    provider: str | None,
    model: str | None,
    source_count: int,
    outcome: str,
    replay: bool,
    started_at: float,
) -> None:
    if audit is None:
        return
    await _audit_log_best_effort(
        audit,
        "share.chat.completed" if outcome == "completed" else "share.chat.failed",
        resource_type="workspace",
        resource_id=context.workspace_id,
        owner_user_id=context.owner_user_id,
        actor_user_id=context.recipient_user_id,
        share_id=context.share_id,
        metadata={
            "effective_source_count": source_count,
            "provider": provider,
            "model": model,
            "outcome": outcome,
            "replay": replay,
            "timings_ms": {"total": max(0, int((time.monotonic() - started_at) * 1000))},
        },
    )


async def _orchestrate_shared_workspace_chat(
    *,
    share_id: int,
    body: SharedWorkspaceChatRequest,
    request: Request | Any,
    recipient_user_id: int,
    access_service: Any,
    store_loader: Any,
    resource_loader: Any,
    rate_limiter: Any,
    audit: Any,
) -> SharedWorkspaceChatResponse:
    """Coordinate authorization, fenced receipts, grounded generation, and commit."""
    started_at = time.monotonic()
    context = await _resolve_recipient_access(
        access_service,
        share_id=share_id,
        recipient_user_id=recipient_user_id,
    )
    try:
        store = await store_loader(context)
        thread = await run_in_threadpool(
            store.get_or_create_thread,
            share_id=share_id,
            owner_user_id=str(context.owner_user_id),
            workspace_id=context.workspace_id,
            workspace_name=str(context.workspace.get("name") or "Shared workspace"),
        )
        claim = await run_in_threadpool(
            store.claim_request,
            share_id=share_id,
            request_id=body.request_id,
            request_fingerprint=_shared_chat_fingerprint(body),
            conversation_id=thread.conversation_id,
            lease_seconds=_shared_chat_lease_seconds(body),
            now=datetime.now(timezone.utc),
        )
    except Exception as exc:
        raise _recipient_http_error(503, "shared_workspace_unavailable") from exc

    if claim.disposition == "replay" and claim.completed_turn is not None:
        response = _shared_chat_response(claim.completed_turn, replayed=True)
        await _audit_shared_chat(
            audit,
            context=context,
            provider=claim.completed_turn.provider,
            model=claim.completed_turn.model,
            source_count=claim.completed_turn.effective_source_count,
            outcome="completed",
            replay=True,
            started_at=started_at,
        )
        return response
    if claim.disposition == "in_progress":
        error = _recipient_http_error(
            409,
            "request_in_progress",
            retry_after_ms=claim.retry_after_ms,
        )
        await _audit_shared_chat(
            audit,
            context=context,
            provider=claim.provider,
            model=claim.model,
            source_count=len(claim.source_ids),
            outcome="request_in_progress",
            replay=False,
            started_at=started_at,
        )
        raise error
    if claim.disposition == "request_id_conflict":
        error = _recipient_http_error(409, "request_id_conflict")
        await _audit_shared_chat(
            audit,
            context=context,
            provider=None,
            model=None,
            source_count=0,
            outcome="request_id_conflict",
            replay=False,
            started_at=started_at,
        )
        raise error

    target = None
    snapshot = None
    try:
        if rate_limiter is not None:
            allowed, _private_reason = await rate_limiter.check_rate_limit(
                user_id=str(recipient_user_id),
                conversation_id=f"shared:{recipient_user_id}:{share_id}",
                estimated_tokens=len(body.query.strip().encode("utf-8")),
            )
            if not allowed:
                raise _recipient_http_error(429, "shared_chat_rate_limited")

        frozen_claim = bool(
            claim.provider
            and claim.model
            and claim.source_ids
            and claim.source_mode
        )
        if frozen_claim:
            target = resolve_chat_target(
                requested_provider=claim.provider,
                requested_model=claim.model,
            )
            if (target.provider, target.model) != (claim.provider, claim.model):
                raise ChatConfigurationError(
                    provider=claim.provider,
                    message="The frozen chat target is no longer available.",
                )
        else:
            target = resolve_chat_target(
                requested_provider=body.provider,
                requested_model=body.model,
            )

        async with resource_loader(context) as resources:
            if frozen_claim:
                snapshot = resources.chat_service.resolve_source_snapshot(
                    mode=claim.source_mode,
                    source_ids=claim.source_ids if claim.source_mode == "include" else (),
                    frozen_source_ids=claim.source_ids if claim.source_mode == "all" else None,
                )
                if snapshot.snapshot_hash != claim.source_snapshot_hash:
                    raise SharedWorkspaceSourceChanged()
            else:
                snapshot = resources.chat_service.resolve_source_snapshot(
                    mode=body.source_scope.mode,
                    source_ids=body.source_scope.source_ids,
                )
                frozen = await run_in_threadpool(
                    store.freeze_sources,
                    claim=claim,
                    source_mode=snapshot.mode,
                    source_ids=snapshot.source_ids,
                    snapshot_hash=snapshot.snapshot_hash,
                    provider=target.provider,
                    model=target.model,
                )
                if not frozen:
                    raise _recipient_http_error(503, "shared_workspace_unavailable")

            evidence = await resources.chat_service.retrieve_verified_evidence(
                query=body.query.strip(),
                snapshot=snapshot,
            )
            await _reauthorize_recipient_chat(access_service, context)
            resources.chat_service.revalidate_source_snapshot(snapshot=snapshot)
            generated = await resources.chat_service.generate_grounded_answer(
                query=body.query.strip(),
                evidence=evidence,
                target=target,
                recipient_user_id=recipient_user_id,
                share_scope_type=context.share_scope_type,
                share_scope_id=context.share_scope_id,
                request=request,
            )
            await _reauthorize_recipient_chat(access_service, context)
            resources.chat_service.revalidate_source_snapshot(snapshot=snapshot)
            turn = await run_in_threadpool(
                store.complete_turn,
                claim=claim,
                query=body.query.strip(),
                answer=generated.answer,
                citations=list(generated.citations),
                provider=target.provider,
                model=target.model,
                source_mode=snapshot.mode,
                effective_source_count=len(snapshot.source_ids),
            )
    except ChatConfigurationError:
        error = _recipient_http_error(503, "no_provider_configured")
    except SharedWorkspaceChatServiceError as exc:
        error = _shared_chat_service_http_error(exc)
    except HTTPException as exc:
        error = exc
    except Exception:
        error = _recipient_http_error(503, "shared_workspace_unavailable")
    else:
        response = _shared_chat_response(turn, replayed=False)
        await _audit_shared_chat(
            audit,
            context=context,
            provider=target.provider,
            model=target.model,
            source_count=len(snapshot.source_ids),
            outcome="completed",
            replay=False,
            started_at=started_at,
        )
        return response

    code = str(error.detail.get("code") or "shared_workspace_unavailable")
    transition = await _mark_shared_chat_failure(store, claim, code)
    if transition.completed_response is not None:
        completed = transition.completed_response
        await _audit_shared_chat(
            audit,
            context=context,
            provider=completed.generation.provider,
            model=completed.generation.model,
            source_count=completed.source_scope.effective_source_count,
            outcome="completed",
            replay=True,
            started_at=started_at,
        )
        return completed
    if transition.replacement_error is not None:
        error = transition.replacement_error
        code = str(error.detail.get("code") or "shared_workspace_unavailable")
    await _audit_shared_chat(
        audit,
        context=context,
        provider=target.provider if target is not None else claim.provider,
        model=target.model if target is not None else claim.model,
        source_count=len(snapshot.source_ids) if snapshot is not None else len(claim.source_ids),
        outcome=code,
        replay=False,
        started_at=started_at,
    )
    raise error


def _bounded_recipient_text(value: Any, limit: int, *, fallback: str = "") -> str:
    printable = "".join(
        character if character.isprintable() else " " for character in str(value or "")
    )
    normalized = " ".join(printable.split())
    return normalized[:limit].strip() or fallback


def _safe_reason_code(value: Any, *, fallback: str) -> str:
    candidate = str(value or "").strip()
    if candidate and len(candidate) <= 128 and all(
        character.isalnum() or character in "_.-" for character in candidate
    ):
        return candidate
    return fallback


def _safe_position(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _sanitize_recipient_source_origin(value: Any) -> tuple[str | None, str | None]:
    """Return only a normalized HTTP(S) origin or a bounded host label."""
    raw = str(value or "").strip()
    if not raw:
        return None, None
    try:
        parsed = urlsplit(raw)
        scheme = parsed.scheme.lower()
        host = (parsed.hostname or "").lower().rstrip(".")
    except ValueError:
        return None, None
    if not host:
        return None, None
    bounded_host = host[:255]
    try:
        port = parsed.port
    except ValueError:
        return None, bounded_host
    if (
        scheme not in {"http", "https"}
        or parsed.username is not None
        or parsed.password is not None
        or len(host) > 255
    ):
        return None, bounded_host
    origin_host = f"[{host}]" if ":" in host else host
    origin = f"{scheme}://{origin_host}{f':{port}' if port is not None else ''}"
    return origin[:2048], bounded_host


def _source_is_retrieval_ready(source_status: dict[str, Any]) -> bool:
    state = str(source_status.get("state") or "").strip().lower()
    readiness = source_status.get("readiness") or {}
    return state == "queryable" or (
        state == "partially_queryable"
        and bool(readiness.get("text_extracted"))
        and bool(readiness.get("fts_ready"))
        and bool(readiness.get("tool_accessible"))
    )


def _recipient_source_search_haystack(source: dict[str, Any]) -> str:
    source_id = str(source.get("id") or "")
    if len(source_id) > 512:
        source_id = ""
    origin_url, origin_host = _sanitize_recipient_source_origin(source.get("url"))
    return " ".join(
        (
            source_id,
            _bounded_recipient_text(source.get("title"), 512),
            _bounded_recipient_text(source.get("source_type"), 64, fallback="media"),
            origin_url or "",
            origin_host or "",
        )
    )


def _recipient_source_model(
    source: dict[str, Any],
    source_status: dict[str, Any],
) -> SharedWorkspaceSource:
    origin_url, origin_host = _sanitize_recipient_source_origin(source.get("url"))
    readiness = source_status.get("readiness") or {}
    return SharedWorkspaceSource(
        source_id=str(source.get("id") or ""),
        title=_bounded_recipient_text(source.get("title"), 512),
        source_type=_bounded_recipient_text(
            source.get("source_type"), 64, fallback="media"
        ),
        origin_url=origin_url,
        origin_host=origin_host,
        state=_bounded_recipient_text(
            source_status.get("state"), 64, fallback="unavailable"
        ),
        reason_code=_safe_reason_code(
            source_status.get("status_reason"), fallback="source_unavailable"
        ),
        citation_ready=bool(readiness.get("citation_ready")),
        retrieval_ready=_source_is_retrieval_ready(source_status),
        position=_safe_position(source.get("position")),
        added_at=source.get("added_at") or None,
    )


def _source_summary(value: Any, *, total: int) -> dict[str, int]:
    summary = value if isinstance(value, dict) else {}
    return {
        "total": total,
        "queryable": max(0, int(summary.get("queryable") or 0)),
        "processing": max(0, int(summary.get("processing") or 0)),
        "failed": max(0, int(summary.get("failed") or 0)),
    }


def _split_history_result(value: Any) -> tuple[Any, str | None]:
    if isinstance(value, tuple) and len(value) == 2:
        return value[0], value[1]
    return value, None


def _recipient_message_model(message: Any) -> SharedWorkspaceMessage:
    citations = []
    for citation in tuple(getattr(message, "citations", ()) or ())[:20]:
        citations.append(
            {
                "citation_id": str(citation.get("citation_id") or ""),
                "source_id": str(citation.get("source_id") or ""),
                "source_title": _bounded_recipient_text(
                    citation.get("source_title"), 512
                ),
                "locator": dict(citation.get("locator") or {}),
                "quote": str(citation.get("quote") or "")[:1000],
                "score": citation.get("score"),
            }
        )
    return SharedWorkspaceMessage(
        message_id=str(message.message_id),
        role=str(message.role),
        content=str(message.content)[:100_000],
        created_at=message.created_at,
        citations=citations,
    )


def _recipient_message_page(value: Any) -> SharedWorkspaceMessagePage:
    page, conversation_id = _split_history_result(value)
    return SharedWorkspaceMessagePage(
        conversation_id=conversation_id,
        messages=[_recipient_message_model(message) for message in page.messages],
        next_before=page.next_before,
    )


def _recipient_preview_text_projection(
    preview: dict[str, Any],
    *,
    max_chars: int,
    focus_chunk_index: int | None,
) -> dict[str, Any]:
    """Allocate one response-wide text budget, prioritizing an explicit focus."""
    if not 1 <= max_chars <= 12_000:
        raise ValueError("max_chars must be between 1 and 12000")

    main_text = str(preview.get("text_preview") or "")
    chunks = [
        dict(snippet)
        for snippet in list(preview.get("snippets") or [])[:10]
        if snippet.get("kind") == "chunk" and str(snippet.get("text") or "")
    ]
    focused = next(
        (
            snippet
            for snippet in chunks
            if focus_chunk_index is not None
            and snippet.get("chunk_index") == focus_chunk_index
        ),
        None,
    )
    remaining_chunks = [snippet for snippet in chunks if snippet is not focused]
    candidates: list[tuple[str, str, dict[str, Any] | None]] = []
    if focused is not None:
        candidates.append(("snippet", str(focused["text"]), focused))
    if main_text:
        candidates.append(("preview", main_text, None))
    candidates.extend(
        ("snippet", str(snippet["text"]), snippet)
        for snippet in remaining_chunks
    )

    remaining = max_chars
    emitted_by_text: dict[str, int] = {}
    primary_was_shortened = False
    bounded_preview: str | None = None
    bounded_snippets: list[dict[str, Any]] = []
    for kind, text, snippet in candidates:
        prior_emitted = emitted_by_text.get(text)
        if prior_emitted is not None:
            if kind == "preview" and prior_emitted < len(text):
                primary_was_shortened = True
            continue
        if remaining <= 0:
            emitted_by_text[text] = 0
            if kind == "preview":
                primary_was_shortened = True
            continue
        emitted = text[:remaining]
        emitted_by_text[text] = len(emitted)
        remaining -= len(emitted)
        if kind == "preview":
            bounded_preview = emitted
            primary_was_shortened = len(emitted) < len(text)
            continue
        bounded = dict(snippet or {})
        bounded["text"] = emitted
        if len(emitted) < len(text):
            start_char = bounded.get("start_char")
            bounded["end_char"] = (
                start_char + len(emitted) if isinstance(start_char, int) else None
            )
        bounded_snippets.append(bounded)

    return {
        "text_preview": bounded_preview,
        "snippets": bounded_snippets,
        "text_truncated": bool(preview.get("text_truncated"))
        or primary_was_shortened,
    }


async def _build_recipient_source_preview(
    context: Any,
    source: dict[str, Any],
    *,
    max_chars: int,
    chunk_limit: int,
    chunk_index: int | None,
) -> dict[str, Any]:
    """Project the local preview helper into a recipient-safe response."""
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import managed_media_db_for_owner
    from tldw_Server_API.app.api.v1.API_Deps.jobs_deps import try_get_job_manager
    from tldw_Server_API.app.core.Workspaces.job_status import (
        list_recent_workspace_source_ingest_jobs,
    )
    from tldw_Server_API.app.core.Workspaces.source_preview import (
        build_workspace_source_preview,
    )
    from tldw_Server_API.app.core.Workspaces.status_projection import (
        build_source_status_projection,
    )

    try:
        jobs = list_recent_workspace_source_ingest_jobs(
            try_get_job_manager(), owner_user_id=context.owner_user_id
        )
        with managed_media_db_for_owner(context.owner_user_id) as media_db:
            status_payload = build_source_status_projection(
                workspace_id=context.workspace_id,
                sources=[source],
                media_db=media_db,
                jobs=jobs,
            )
            source_status = (status_payload.get("sources") or [{}])[0]
            preview = build_workspace_source_preview(
                workspace_id=context.workspace_id,
                source=source,
                source_status=source_status,
                media_db=media_db,
                max_chars=max_chars,
                chunk_limit=chunk_limit,
                focus_chunk_index=chunk_index,
            )
    except Exception as exc:
        raise _recipient_http_error(503, "shared_workspace_unavailable") from exc

    origin_url, origin_host = _sanitize_recipient_source_origin(source.get("url"))
    text_projection = _recipient_preview_text_projection(
        preview,
        max_chars=max_chars,
        focus_chunk_index=chunk_index,
    )
    snippets = []
    for snippet in text_projection["snippets"]:
        text = str(snippet.get("text") or "")
        if not text:
            continue
        snippets.append(
            {
                "kind": "chunk" if snippet.get("kind") == "chunk" else "content_excerpt",
                "text": text,
                "start_char": snippet.get("start_char"),
                "end_char": snippet.get("end_char"),
                "chunk_index": snippet.get("chunk_index"),
            }
        )
    return {
        "source_id": str(source.get("id") or ""),
        "title": _bounded_recipient_text(source.get("title"), 512),
        "source_type": _bounded_recipient_text(
            source.get("source_type"), 64, fallback="media"
        ),
        "origin_url": origin_url,
        "origin_host": origin_host,
        "state": _bounded_recipient_text(
            preview.get("state"), 64, fallback="unavailable"
        ),
        "reason_code": _safe_reason_code(
            preview.get("status_reason"), fallback="source_unavailable"
        ),
        "content_available": bool(preview.get("content_available")),
        "preview_mode": _bounded_recipient_text(
            preview.get("preview_mode"), 64, fallback="empty"
        ),
        "unavailable_reason": (
            _safe_reason_code(
                preview.get("unavailable_reason"), fallback="content_unavailable"
            )
            if preview.get("unavailable_reason")
            else None
        ),
        "text_preview": text_projection["text_preview"],
        "text_total_chars": preview.get("text_total_chars"),
        "text_truncated": text_projection["text_truncated"],
        "snippets": snippets,
        "generated_at": preview.get("generated_at"),
    }


def _chatbook_ownership_exists_sync(
    *,
    normalized_id: str,
    user_id: Any,
    user_id_int: int | None,
    db: Any | None,
) -> bool:
    """Run sync chatbook ownership checks away from the event loop."""
    from pathlib import Path

    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

    if db is not None:
        try:
            from tldw_Server_API.app.core.Chatbooks.chatbook_models import ExportStatus
            from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService

            service = ChatbookService(user_id, db, user_id_int=user_id_int)
            export_job = service.get_export_job(normalized_id)
            if export_job and str(export_job.user_id) == str(user_id):
                job_status = getattr(export_job.status, "value", export_job.status)
                if job_status == ExportStatus.COMPLETED.value and export_job.output_path:
                    output_path = Path(export_job.output_path).resolve()
                    export_dir = Path(service.export_dir).resolve()
                    try:
                        output_path.relative_to(export_dir)
                    except ValueError:
                        return False
                    if output_path.is_file():
                        return True
        except Exception:
            logger.debug("Chatbook export job ownership check skipped")

    candidate_names = {Path(normalized_id).name}
    if not normalized_id.endswith(".chatbook"):
        candidate_names.add(f"{Path(normalized_id).name}.chatbook")

    for root in (
        DatabasePaths.get_user_chatbooks_exports_dir(user_id),
        DatabasePaths.get_user_chatbooks_imports_dir(user_id),
    ):
        root_path = root.resolve()
        for name in candidate_names:
            candidate = (root_path / name).resolve()
            try:
                candidate.relative_to(root_path)
            except ValueError:
                continue
            if candidate.is_file():
                return True

    return False


async def _verify_chatbook_ownership(chatbook_id: str, user: User) -> None:
    """Verify a chatbook token target resolves to a file owned by this user."""
    normalized_id = str(chatbook_id or "").strip()
    if not normalized_id:
        raise HTTPException(status_code=404, detail="Resource not found")

    user_id_int = user.id_int if hasattr(user, "id_int") else None
    storage_user_id = user_id_int if user_id_int is not None else user.id
    try:
        db_lookup_user_id = int(storage_user_id)
    except (TypeError, ValueError):
        db_lookup_user_id = None

    db = None
    if db_lookup_user_id is not None:
        try:
            from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user_id

            db = await get_chacha_db_for_user_id(db_lookup_user_id)
        except Exception:
            logger.debug("Chatbook export job ownership check skipped")

    owned = await run_in_threadpool(
        _chatbook_ownership_exists_sync,
        normalized_id=normalized_id,
        user_id=storage_user_id,
        user_id_int=user_id_int,
        db=db,
    )
    if owned:
        return

    raise HTTPException(status_code=404, detail="Resource not found")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Workspace Sharing CRUD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.post(
    "/workspaces/{workspace_id}/share",
    response_model=ShareResponse,
    dependencies=[Depends(rbac_rate_limit("sharing.create"))],
    summary="Share a workspace with a team or org",
)
async def share_workspace(
    workspace_id: str,
    body: ShareWorkspaceRequest,
    request: Request,
    user: User = Depends(get_request_user),
):
    # [CRITICAL FIX #3] Verify the user owns this workspace
    await _verify_workspace_ownership(workspace_id, user)
    await _validate_share_target_scope(body, user)

    repo = await _maybe_await(_get_repo())
    audit = _get_audit_service()
    try:
        share = await repo.create_share(
            workspace_id=workspace_id,
            owner_user_id=user.id,
            share_scope_type=body.share_scope_type.value,
            share_scope_id=body.share_scope_id,
            access_level=body.access_level.value,
            allow_clone=body.allow_clone,
            created_by=user.id,
        )
    except Exception as exc:
        if _is_duplicate_share_error(exc):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This workspace is already shared with the specified scope.",
            ) from exc
        logger.error("Failed to create share")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while creating the share.",
        ) from exc

    await _audit_log_best_effort(
        audit,
        "share.created",
        resource_type="workspace",
        resource_id=workspace_id,
        owner_user_id=user.id,
        actor_user_id=user.id,
        share_id=share.get("id"),
        metadata={"scope_type": body.share_scope_type.value, "scope_id": body.share_scope_id},
        ip_address=_client_ip(request),
    )
    return ShareResponse(**share)


@router.get(
    "/workspaces/{workspace_id}/shares",
    response_model=ShareListResponse,
    dependencies=[Depends(rbac_rate_limit("sharing.read"))],
    summary="List shares for a workspace",
)
async def list_workspace_shares(
    workspace_id: str,
    include_revoked: bool = Query(False),
    user: User = Depends(get_request_user),
):
    repo = await _maybe_await(_get_repo())
    shares = await repo.list_shares_for_workspace(workspace_id, user.id, include_revoked=include_revoked)
    return ShareListResponse(shares=[ShareResponse(**s) for s in shares], total=len(shares))


@router.patch(
    "/shares/{share_id}",
    response_model=ShareResponse,
    dependencies=[Depends(rbac_rate_limit("sharing.update"))],
    summary="Update a share's access level or clone permission",
)
async def update_share(
    share_id: int,
    body: UpdateShareRequest,
    request: Request,
    user: User = Depends(get_request_user),
):
    repo = await _maybe_await(_get_repo())
    audit = _get_audit_service()

    existing = await repo.get_share(share_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Share not found")
    if existing["owner_user_id"] != user.id:
        raise HTTPException(status_code=403, detail="Not the workspace owner")

    updated = await repo.update_share(
        share_id,
        access_level=body.access_level.value if body.access_level else None,
        allow_clone=body.allow_clone,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Share not found")

    await _audit_log_best_effort(
        audit,
        "share.updated",
        resource_type="workspace",
        resource_id=existing["workspace_id"],
        owner_user_id=user.id,
        actor_user_id=user.id,
        share_id=share_id,
        ip_address=_client_ip(request),
    )
    return ShareResponse(**updated)


@router.delete(
    "/shares/{share_id}",
    dependencies=[Depends(rbac_rate_limit("sharing.delete"))],
    summary="Revoke a share",
)
async def revoke_share(
    share_id: int,
    request: Request,
    user: User = Depends(get_request_user),
):
    repo = await _maybe_await(_get_repo())
    audit = _get_audit_service()

    existing = await repo.get_share(share_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Share not found")
    if existing["owner_user_id"] != user.id:
        raise HTTPException(status_code=403, detail="Not the workspace owner")

    await repo.revoke_share(share_id)

    await _audit_log_best_effort(
        audit,
        "share.revoked",
        resource_type="workspace",
        resource_id=existing["workspace_id"],
        owner_user_id=user.id,
        actor_user_id=user.id,
        share_id=share_id,
        ip_address=_client_ip(request),
    )
    return {"detail": "Share revoked"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shared With Me
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.get(
    "/shared-with-me",
    response_model=SharedWithMeResponse,
    dependencies=[Depends(rbac_rate_limit("sharing.read"))],
    summary="List workspaces shared with the current user",
)
async def shared_with_me(
    user: User = Depends(get_request_user),
):
    repo = await _maybe_await(_get_repo())

    shares = await repo.list_active_shares_for_user(user.id)
    items = [
        SharedWithMeItem(
            share_id=share["id"],
            workspace_id=share["workspace_id"],
            owner_user_id=share["owner_user_id"],
            access_level=share["access_level"],
            allow_clone=share["allow_clone"],
            shared_at=share.get("created_at"),
        )
        for share in shares
    ]

    # Batch-populate workspace names from each owner's ChaChaNotes DB
    if items:
        try:
            from ..API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_owner

            owner_ids = {item.owner_user_id for item in items}
            owner_dbs: dict[int, Any] = {}
            for oid in owner_ids:
                try:
                    owner_dbs[oid] = await get_chacha_db_for_owner(oid)
                except Exception:
                    logger.debug("Skipping shared workspace name preload")

            for item in items:
                db = owner_dbs.get(item.owner_user_id)
                if db:
                    try:
                        ws = db.get_workspace(item.workspace_id)
                        if ws:
                            item.workspace_name = ws.get("name")
                    except Exception:
                        logger.debug("Failed to resolve shared workspace name")
        except Exception:
            logger.debug("Shared workspace name population skipped")

    return SharedWithMeResponse(items=items, total=len(items))


@recipient_router.get(
    "/workspace",
    response_model=SharedWorkspaceBootstrapResponse,
    responses=_RECIPIENT_READ_ERROR_RESPONSES,
    dependencies=_RECIPIENT_READ_DEPENDENCIES,
    summary="Read shared workspace metadata",
)
async def get_shared_workspace(
    share_id: int,
    user: User = Depends(get_request_user),
    service: Any = Depends(get_shared_workspace_access_service),
) -> SharedWorkspaceBootstrapResponse:
    context = await _resolve_recipient_access(
        service,
        share_id=share_id,
        recipient_user_id=user.id,
    )
    sources = await _load_recipient_workspace_sources(context)
    projection = await _project_recipient_source_status(context, sources)
    status_by_id = {
        str(item.get("id") or ""): item
        for item in projection.get("sources") or []
        if isinstance(item, dict)
    }
    ordered_sources = sorted(
        sources,
        key=lambda source: (
            _safe_position(source.get("position")),
            str(source.get("id") or ""),
        ),
    )
    try:
        source_models = [
            _recipient_source_model(
                source,
                status_by_id.get(str(source.get("id") or ""), {}),
            )
            for source in ordered_sources
        ]
    except ValidationError as exc:
        raise _recipient_http_error(503, "shared_workspace_unavailable") from exc

    partial_errors = list(projection.get("partial_errors") or [])[:8]
    try:
        generation_default = await _resolve_recipient_generation_default(context)
    except Exception:
        generation_default = {
            "provider": None,
            "model": None,
            "ready": False,
            "reason_code": "no_provider_configured",
        }
        partial_errors.append(
            _recipient_partial_error(
                area="generation",
                code="generation_default_unavailable",
                message="The default generation target is temporarily unavailable.",
                retryable=True,
            )
        )
    try:
        history = await _load_recipient_chat_history(
            context,
            before=None,
            limit=30,
        )
        conversation = _recipient_message_page(history)
    except Exception:
        conversation = SharedWorkspaceMessagePage(
            conversation_id=None,
            messages=[],
            next_before=None,
        )
        partial_errors.append(
            _recipient_partial_error(
                area="history",
                code="history_unavailable",
                message="Shared chat history is temporarily unavailable.",
                retryable=True,
            )
        )

    actions = {
        str(name): dict(value)
        for name, value in context.policy_actions.items()
        if isinstance(value, dict)
    }
    has_retrieval_source = any(source.retrieval_ready for source in source_models)
    if not has_retrieval_source:
        actions["ask_grounded_questions"] = {
            "allowed": False,
            "reason_code": "no_queryable_sources",
        }
    elif not bool(generation_default.get("ready")):
        actions["ask_grounded_questions"] = {
            "allowed": False,
            "reason_code": _safe_reason_code(
                generation_default.get("reason_code"),
                fallback="no_provider_configured",
            ),
        }

    try:
        return SharedWorkspaceBootstrapResponse(
            generated_at=datetime.now(timezone.utc),
            share={
                "share_id": context.share_id,
                "access_level": _bounded_recipient_text(
                    context.access_level, 64, fallback="view_chat"
                ),
                "allow_clone": context.allow_clone,
                "owner_display_name": _bounded_recipient_text(
                    context.owner_display_name,
                    128,
                    fallback="Workspace owner",
                ),
                "shared_at": context.shared_at,
            },
            workspace={
                "workspace_id": context.workspace_id,
                "name": _bounded_recipient_text(context.workspace.get("name"), 512),
                "description": _bounded_recipient_text(
                    context.workspace.get("description"), 2_000
                ),
            },
            allowed_actions=actions,
            generation_default=generation_default,
            source_summary=_source_summary(
                projection.get("summary"), total=len(source_models)
            ),
            sources={
                "items": source_models[:50],
                "pagination": {
                    "offset": 0,
                    "limit": 50,
                    "total": len(source_models),
                    "has_more": len(source_models) > 50,
                },
            },
            conversation=conversation,
            partial_errors=partial_errors[:8],
        )
    except ValidationError as exc:
        raise _recipient_http_error(503, "shared_workspace_unavailable") from exc


def _clone_operation_response_status(
    operation: SharedWorkspaceCloneOperationResponse,
) -> int:
    return 200 if operation.status in {"succeeded", "failed"} else 202


def _project_clone_job(
    job: dict[str, Any],
    *,
    share_id: int,
    recipient_user_id: int,
) -> SharedWorkspaceCloneOperationResponse:
    try:
        return project_clone_operation(
            job,
            share_id=share_id,
            recipient_user_id=recipient_user_id,
        )
    except CloneOperationNotFound as exc:
        raise _recipient_http_error(404, "shared_workspace_not_found") from exc
    except CloneOperationUnavailable as exc:
        raise _recipient_http_error(503, "clone_operation_unavailable") from exc


def _clone_conflict_http_error(exc: IdempotentOperationConflict) -> HTTPException:
    code = (
        "idempotency_key_reused"
        if exc.reason is IdempotentOperationConflictReason.KEY_REUSED
        else "clone_already_in_progress"
    )
    return _recipient_http_error(
        409,
        code,
        operation_id=exc.job_uuid,
    )


@recipient_router.post(
    "/clone",
    response_model=SharedWorkspaceCloneOperationResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        **_RECIPIENT_CLONE_ERROR_RESPONSES,
        200: {
            "model": SharedWorkspaceCloneOperationResponse,
            "description": "An existing clone operation is terminal.",
        },
    },
    dependencies=_RECIPIENT_CLONE_DEPENDENCIES,
    summary="Copy a shared workspace into the current user's account",
)
async def clone_shared_workspace(
    share_id: int,
    body: SharedWorkspaceCloneRequest,
    request: Request,
    response: Response,
    idempotency_key: str = Header(alias="Idempotency-Key"),
    user: User = Depends(get_request_user),
    service: Any = Depends(get_shared_workspace_access_service),
    job_manager: Any = Depends(try_get_job_manager),
) -> SharedWorkspaceCloneOperationResponse:
    if job_manager is None:
        raise _recipient_http_error(503, "clone_operation_unavailable")
    try:
        command = build_clone_admission_command(
            share_id=share_id,
            recipient_user_id=user.id,
            requested_name=body.name,
            idempotency_key=idempotency_key,
        )
    except ValueError as exc:
        raise _recipient_http_error(422, "invalid_shared_workspace_request") from exc

    try:
        replay = job_manager.replay_idempotent_operation(command)
    except IdempotentOperationConflict as exc:
        raise _clone_conflict_http_error(exc) from exc
    except Exception as exc:
        raise _recipient_http_error(503, "clone_operation_unavailable") from exc
    if replay is not None:
        operation = _project_clone_job(
            replay.job,
            share_id=share_id,
            recipient_user_id=user.id,
        )
        response.status_code = _clone_operation_response_status(operation)
        return operation

    context = await _resolve_recipient_access(
        service,
        share_id=share_id,
        recipient_user_id=user.id,
    )
    if not context.allow_clone:
        raise _recipient_http_error(403, "clone_not_allowed")
    try:
        admission = job_manager.admit_idempotent_operation(command)
    except IdempotentOperationConflict as exc:
        raise _clone_conflict_http_error(exc) from exc
    except (IdempotentOperationUnavailableError, ValueError) as exc:
        raise _recipient_http_error(503, "clone_operation_unavailable") from exc
    except Exception as exc:
        raise _recipient_http_error(503, "clone_operation_unavailable") from exc

    operation = _project_clone_job(
        admission.job,
        share_id=share_id,
        recipient_user_id=user.id,
    )
    if admission.disposition is IdempotentOperationDisposition.CREATED:
        await _audit_log_best_effort(
            _get_audit_service(),
            "share.clone_requested",
            resource_type="workspace",
            resource_id=context.workspace_id,
            owner_user_id=context.owner_user_id,
            actor_user_id=user.id,
            share_id=share_id,
            metadata={"operation_id": operation.operation_id},
            ip_address=_client_ip(request),
        )
    response.status_code = _clone_operation_response_status(operation)
    return operation


@recipient_router.get(
    "/clone/{operation_id}",
    response_model=SharedWorkspaceCloneOperationResponse,
    responses=_RECIPIENT_CLONE_ERROR_RESPONSES,
    dependencies=_RECIPIENT_READ_DEPENDENCIES,
    summary="Read a shared workspace copy operation",
)
async def get_shared_workspace_clone_operation(
    share_id: int,
    operation_id: str,
    user: User = Depends(get_request_user),
    job_manager: Any = Depends(try_get_job_manager),
) -> SharedWorkspaceCloneOperationResponse:
    if job_manager is None:
        raise _recipient_http_error(503, "clone_operation_unavailable")
    try:
        normalized_operation_id = str(UUID(operation_id))
    except (TypeError, ValueError, AttributeError) as exc:
        raise _recipient_http_error(404, "shared_workspace_not_found") from exc
    try:
        job = job_manager.get_job_or_archived_by_uuid(
            normalized_operation_id,
            domain="sharing",
            owner_user_id=str(user.id),
        )
    except Exception as exc:
        raise _recipient_http_error(503, "clone_operation_unavailable") from exc
    if job is None:
        raise _recipient_http_error(404, "shared_workspace_not_found")
    return _project_clone_job(
        job,
        share_id=share_id,
        recipient_user_id=user.id,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Recipient Shared-Workspace Read Endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@recipient_router.get(
    "/sources",
    response_model=SharedWorkspaceSourcePage,
    responses=_RECIPIENT_READ_ERROR_RESPONSES,
    dependencies=_RECIPIENT_READ_DEPENDENCIES,
    summary="List sources of a shared workspace",
)
async def list_shared_workspace_sources(
    share_id: int,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    q: str | None = Query(default=None, min_length=1, max_length=512),
    state: str | None = Query(default=None, min_length=1, max_length=64),
    user: User = Depends(get_request_user),
    service: Any = Depends(get_shared_workspace_access_service),
) -> SharedWorkspaceSourcePage:
    context = await _resolve_recipient_access(
        service,
        share_id=share_id,
        recipient_user_id=user.id,
    )
    sources = await _load_recipient_workspace_sources(context)
    if q is not None:
        needle = q.casefold()
        sources = [
            source
            for source in sources
            if needle in _recipient_source_search_haystack(source).casefold()
        ]
    sources.sort(
        key=lambda source: (
            _safe_position(source.get("position")),
            str(source.get("id") or ""),
        )
    )
    projection = await _project_recipient_source_status(context, sources)
    status_by_id = {
        str(item.get("id") or ""): item
        for item in projection.get("sources") or []
        if isinstance(item, dict)
    }
    try:
        projected = [
            _recipient_source_model(
                source,
                status_by_id.get(str(source.get("id") or ""), {}),
            )
            for source in sources
        ]
        if state is not None:
            requested_state = state.casefold()
            projected = [
                source for source in projected if source.state.casefold() == requested_state
            ]
        page = projected[offset : offset + limit]
        return SharedWorkspaceSourcePage(
            items=page,
            pagination={
                "offset": offset,
                "limit": limit,
                "total": len(projected),
                "has_more": offset + len(page) < len(projected),
            },
            summary=_source_summary(projection.get("summary"), total=len(sources)),
            partial_errors=list(projection.get("partial_errors") or [])[:8],
        )
    except ValidationError as exc:
        raise _recipient_http_error(503, "shared_workspace_unavailable") from exc


@recipient_router.get(
    "/sources/{source_id}/preview",
    response_model=SharedWorkspaceSourcePreview,
    responses=_RECIPIENT_READ_ERROR_RESPONSES,
    dependencies=_RECIPIENT_READ_DEPENDENCIES,
    summary="Preview a source in a shared workspace",
)
async def preview_shared_workspace_source(
    share_id: int,
    source_id: str,
    max_chars: int = Query(default=3_000, ge=1, le=12_000),
    chunk_limit: int = Query(default=3, ge=0, le=10),
    chunk_index: int | None = Query(default=None, ge=0),
    user: User = Depends(get_request_user),
    service: Any = Depends(get_shared_workspace_access_service),
) -> SharedWorkspaceSourcePreview:
    context = await _resolve_recipient_access(
        service,
        share_id=share_id,
        recipient_user_id=user.id,
    )
    sources = await _load_recipient_workspace_sources(context)
    source = next(
        (candidate for candidate in sources if str(candidate.get("id") or "") == source_id),
        None,
    )
    if source is None:
        raise _recipient_http_error(404, "shared_workspace_not_found")
    payload = await _build_recipient_source_preview(
        context,
        source,
        max_chars=max_chars,
        chunk_limit=chunk_limit,
        chunk_index=chunk_index,
    )
    try:
        return SharedWorkspaceSourcePreview(**payload)
    except ValidationError as exc:
        raise _recipient_http_error(503, "shared_workspace_unavailable") from exc


@recipient_router.get(
    "/chat/messages",
    response_model=SharedWorkspaceMessagePage,
    responses=_RECIPIENT_READ_ERROR_RESPONSES,
    dependencies=_RECIPIENT_READ_DEPENDENCIES,
    summary="Read recipient-owned shared chat history",
)
async def get_shared_workspace_messages(
    share_id: int,
    before: str | None = Query(default=None, min_length=1, max_length=2_048),
    limit: int = Query(default=30, ge=1, le=100),
    user: User = Depends(get_request_user),
    service: Any = Depends(get_shared_workspace_access_service),
) -> SharedWorkspaceMessagePage:
    context = await _resolve_recipient_access(
        service,
        share_id=share_id,
        recipient_user_id=user.id,
    )
    try:
        history = await _load_recipient_chat_history(
            context,
            before=before,
            limit=limit,
        )
        return _recipient_message_page(history)
    except SharedWorkspaceCursorInputError as exc:
        raise _recipient_http_error(422, "invalid_shared_workspace_request") from exc
    except (HTTPException, ValidationError) as exc:
        if isinstance(exc, HTTPException):
            raise
        raise _recipient_http_error(503, "shared_workspace_unavailable") from exc
    except Exception as exc:
        raise _recipient_http_error(503, "shared_workspace_unavailable") from exc


@recipient_router.post(
    "/chat",
    response_model=SharedWorkspaceChatResponse,
    responses=_RECIPIENT_CHAT_ERROR_RESPONSES,
    dependencies=_RECIPIENT_CHAT_DEPENDENCIES,
    summary="Ask a grounded question about a shared workspace",
)
async def create_shared_workspace_chat_turn(
    share_id: int,
    body: SharedWorkspaceChatRequest,
    request: Request,
    user: User = Depends(get_request_user),
    service: Any = Depends(get_shared_workspace_access_service),
) -> SharedWorkspaceChatResponse:
    from tldw_Server_API.app.core.Chat.rate_limiter import get_rate_limiter

    return await _orchestrate_shared_workspace_chat(
        share_id=share_id,
        body=body,
        request=request,
        recipient_user_id=user.id,
        access_service=service,
        store_loader=_load_recipient_chat_store,
        resource_loader=_load_recipient_chat_resources,
        rate_limiter=get_rate_limiter(),
        audit=_get_audit_service(),
    )


router.include_router(recipient_router)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Share Tokens
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.post(
    "/tokens",
    response_model=TokenResponse,
    dependencies=[Depends(rbac_rate_limit("sharing.token.create"))],
    summary="Create a share token (link)",
)
async def create_token(
    body: CreateTokenRequest,
    request: Request,
    user: User = Depends(get_request_user),
):
    svc = await _maybe_await(_get_token_service())
    audit = _get_audit_service()
    if body.resource_type == ResourceType.WORKSPACE:
        await _verify_workspace_ownership(body.resource_id, user)
    elif body.resource_type == ResourceType.CHATBOOK:
        await _verify_chatbook_ownership(body.resource_id, user)
    elif body.resource_type == ResourceType.PROTOTYPE_WORKSPACE:
        await _get_owned_prototype_workspace(
            prototype_workspace_id=body.resource_id,
            owner_user_id=user.id,
        )

    result = await svc.generate_token(
        resource_type=body.resource_type.value,
        resource_id=body.resource_id,
        owner_user_id=user.id,
        access_level=body.access_level.value,
        allow_clone=body.allow_clone,
        password=body.password,
        max_uses=body.max_uses,
        expires_at=body.expires_at,
    )

    await _audit_log_best_effort(
        audit,
        "token.created",
        resource_type=body.resource_type.value,
        resource_id=body.resource_id,
        owner_user_id=user.id,
        actor_user_id=user.id,
        token_id=result.get("id"),
        ip_address=_client_ip(request),
    )
    return TokenResponse(**result)


@router.get(
    "/tokens",
    response_model=TokenListResponse,
    dependencies=[Depends(rbac_rate_limit("sharing.token.read"))],
    summary="List my share tokens",
)
async def list_tokens(
    user: User = Depends(get_request_user),
):
    svc = await _maybe_await(_get_token_service())
    tokens = await svc.list_tokens(user.id)
    return TokenListResponse(tokens=[TokenResponse(**t) for t in tokens], total=len(tokens))


@router.delete(
    "/tokens/{token_id}",
    dependencies=[Depends(rbac_rate_limit("sharing.token.delete"))],
    summary="Revoke a share token",
)
async def revoke_token(
    token_id: int,
    request: Request,
    user: User = Depends(get_request_user),
):
    repo = await _maybe_await(_get_repo())
    audit = _get_audit_service()

    token = await repo.get_token(token_id)
    if not token:
        raise HTTPException(status_code=404, detail="Token not found")
    if token["owner_user_id"] != user.id:
        raise HTTPException(status_code=403, detail="Not the token owner")

    svc = await _maybe_await(_get_token_service())
    await svc.revoke_token(token_id)

    await _audit_log_best_effort(
        audit,
        "token.revoked",
        resource_type=token["resource_type"],
        resource_id=token["resource_id"],
        owner_user_id=user.id,
        actor_user_id=user.id,
        token_id=token_id,
        ip_address=_client_ip(request),
    )
    return {"detail": "Token revoked"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Public Token Access (rate limited by IP, no auth required for preview/verify)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.get(
    "/public/{token}",
    response_model=PublicSharePreview,
    summary="Preview a shared resource (public, rate limited)",
)
async def public_preview(
    token: str,
    request: Request,
):
    # [CRITICAL FIX #1] Rate limit public endpoints (10 req/min per IP)
    _check_public_rate_limit(request)

    svc = await _maybe_await(_get_token_service())
    validated = await svc.validate_token(token)
    # Return identical 404 for not-found / expired / revoked to prevent enumeration
    if not validated:
        raise HTTPException(status_code=404, detail="Resource not found")

    return PublicSharePreview(
        resource_type=validated["resource_type"],
        is_password_protected=validated.get("is_password_protected", False),
        access_level=validated["access_level"],
    )


@router.post(
    "/public/{token}/verify",
    response_model=VerifyPasswordResponse,
    summary="Verify password for a protected share link",
)
async def public_verify_password(
    token: str,
    body: VerifyPasswordRequest,
    request: Request,
):
    # [CRITICAL FIX #1] Rate limit public endpoints (10 req/min per IP)
    _check_public_rate_limit(request)

    svc = await _maybe_await(_get_token_service())
    audit = _get_audit_service()

    validated = await svc.validate_token(token)
    if not validated:
        raise HTTPException(status_code=404, detail="Resource not found")

    ok = await svc.verify_password(validated, body.password)
    event = "token.password_verified" if ok else "token.password_failed"
    await _audit_log_best_effort(
        audit,
        event,
        resource_type=validated["resource_type"],
        resource_id=validated["resource_id"],
        owner_user_id=validated["owner_user_id"],
        token_id=validated["id"],
        ip_address=_client_ip(request),
        user_agent=request.headers.get("user-agent"),
    )

    if not ok:
        raise HTTPException(status_code=403, detail="Invalid password")

    return VerifyPasswordResponse(verified=True)


@router.post(
    "/public/{token}/prototype-session",
    response_model=PrototypeLinkExchangeResponse,
    summary="Exchange a prototype private link for an external collaborator session",
    responses=PROTOTYPE_LINK_ERROR_RESPONSES,
)
async def public_prototype_session_exchange(
    token: str,
    body: PrototypeLinkExchangeRequest,
    request: Request,
    response: Response,
):
    _check_public_rate_limit(request)

    from tldw_Server_API.app.core.Prototype_Workspaces.access import (
        PROTOTYPE_SHARED_ACTOR_COOKIE,
        PrototypeAccessError,
    )

    svc = await _maybe_await(_get_token_service())
    audit = _get_audit_service()
    validated = await svc.validate_token(token, allow_exhausted=True)
    if not validated:
        raise prototype_http_error(
            status_code=status.HTTP_404_NOT_FOUND,
            category="invalid_or_unavailable_link",
            message="Prototype link is unavailable",
            frontend_state="link_unavailable",
        )

    resource_type = str(validated.get("resource_type") or "").strip().lower()
    prototype_workspace_id = str(validated.get("resource_id") or "").strip()
    if resource_type != ResourceType.PROTOTYPE_WORKSPACE.value or not prototype_workspace_id:
        raise prototype_http_error(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            category="invalid_request",
            message="Share token is not a prototype workspace link",
            frontend_state="invalid_request",
        )
    await _get_owned_prototype_workspace(
        prototype_workspace_id=prototype_workspace_id,
        owner_user_id=validated.get("owner_user_id"),
        use_prototype_error_contract=True,
    )

    resume_cookie_value = request.cookies.get(PROTOTYPE_SHARED_ACTOR_COOKIE)
    access_service = await _maybe_await(_get_prototype_access_service())
    can_resume_without_password = await access_service.can_resume_external_collaborator(
        prototype_workspace_id=prototype_workspace_id,
        share_link_id=int(validated["id"]),
        resume_cookie_value=resume_cookie_value,
    )
    if validated.get("is_password_protected"):
        if body.password:
            password_ok = await svc.verify_password(validated, body.password)
            await _audit_log_best_effort(
                audit,
                "token.password_verified" if password_ok else "token.password_failed",
                resource_type=validated["resource_type"],
                resource_id=validated["resource_id"],
                owner_user_id=validated["owner_user_id"],
                token_id=validated["id"],
                ip_address=_client_ip(request),
                user_agent=request.headers.get("user-agent"),
            )
            if not password_ok:
                raise prototype_http_error(
                    status_code=status.HTTP_403_FORBIDDEN,
                    category="invalid_password",
                    message="Prototype link password is invalid",
                    frontend_state="password_rejected",
                    retryable=True,
                )
        else:
            if not can_resume_without_password:
                raise prototype_http_error(
                    status_code=status.HTTP_403_FORBIDDEN,
                    category="password_required",
                    message="Prototype link password is required",
                    frontend_state="password_required",
                    retryable=True,
                )

    if not can_resume_without_password and not str(body.display_name or "").strip():
        raise prototype_http_error(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            category="invalid_request",
            message="display_name is required for first-time sessions",
            frontend_state="invalid_request",
        )

    claimed_new_use = False
    if not can_resume_without_password:
        claimed_new_use = await svc.claim_token_use(validated["id"])
        if not claimed_new_use:
            raise prototype_http_error(
                status_code=status.HTTP_404_NOT_FOUND,
                category="invalid_or_unavailable_link",
                message="Prototype link is unavailable",
                frontend_state="link_unavailable",
            )

    claim_released = False
    provisioning_succeeded = False
    try:
        access_context = await access_service.exchange_external_collaborator(
            prototype_workspace_id=prototype_workspace_id,
            share_link_id=int(validated["id"]),
            display_name=body.display_name,
            resume_cookie_value=resume_cookie_value,
            allow_create=claimed_new_use,
            expires_at=validated.get("expires_at"),
        )
        provisioning_succeeded = bool(access_context.shared_actor_id or access_context.session_token)
    except PrototypeAccessError as exc:
        if claimed_new_use:
            await svc.release_token_use(validated["id"])
            claim_released = True
        if exc.code == "workspace_not_found":
            raise prototype_http_error(
                status_code=status.HTTP_404_NOT_FOUND,
                category="invalid_or_unavailable_link",
                message="Prototype link is unavailable",
                frontend_state="link_unavailable",
            ) from exc
        if exc.code == "workspace_archived":
            raise prototype_http_error(
                status_code=status.HTTP_403_FORBIDDEN,
                category="workspace_unavailable",
                message="Prototype workspace is archived",
                frontend_state="workspace_unavailable",
            ) from exc
        if exc.code == "resume_required":
            raise prototype_http_error(
                status_code=status.HTTP_404_NOT_FOUND,
                category="invalid_or_unavailable_link",
                message="Prototype link is unavailable",
                frontend_state="link_unavailable",
            ) from exc
        raise
    except Exception:
        if claimed_new_use and not claim_released:
            await svc.release_token_use(validated["id"])
            claim_released = True
        raise
    if access_context.is_resume and claimed_new_use:
        await svc.release_token_use(validated["id"])
        claim_released = True
    try:
        await _audit_log_best_effort(
            audit,
            "token.prototype_session_exchanged",
            resource_type=validated["resource_type"],
            resource_id=validated["resource_id"],
            owner_user_id=validated["owner_user_id"],
            token_id=validated["id"],
            ip_address=_client_ip(request),
            user_agent=request.headers.get("user-agent"),
            metadata={
                "shared_actor_id": access_context.shared_actor_id,
                "actor_type": access_context.actor_type,
                "runtime_policy_profile": access_context.runtime_policy_profile,
                "is_resume": access_context.is_resume,
                "resumed_without_password": can_resume_without_password,
                "claimed_new_use": claimed_new_use,
            },
        )

        response.set_cookie(
            key=PROTOTYPE_SHARED_ACTOR_COOKIE,
            value=access_context.resume_cookie_value,
            max_age=7 * 24 * 60 * 60,
            httponly=True,
            samesite="lax",
            secure=_request_is_secure(request),
        )
        return PrototypeLinkExchangeResponse(
            shared_actor_id=access_context.shared_actor_id,
            actor_type="external_collaborator",
            session_token=access_context.session_token,
            runtime_policy_profile=access_context.runtime_policy_profile,
        )
    except Exception:
        if claimed_new_use and not claim_released and not provisioning_succeeded:
            await svc.release_token_use(validated["id"])
        raise


@router.post(
    "/public/{token}/import",
    dependencies=[Depends(rbac_rate_limit("sharing.read"))],
    summary="Import resource from share token (requires auth)",
)
async def public_import(
    token: str,
    request: Request,
    body: PublicImportRequest | None = None,
    user: User = Depends(get_request_user),
):
    svc = await _maybe_await(_get_token_service())
    audit = _get_audit_service()

    validated = await svc.validate_token(token, allow_exhausted=True)
    if not validated or validated.get("is_use_exhausted"):
        raise HTTPException(status_code=404, detail="Resource not found")

    if validated.get("resource_type") == ResourceType.PROTOTYPE_WORKSPACE.value:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Prototype workspace links must be exchanged via /prototype-session",
        )

    if validated.get("is_password_protected"):
        if body is None or not body.password:
            raise HTTPException(
                status_code=403,
                detail="Password verification required.",
            )
        ok = await svc.verify_password(validated, body.password)
        await _audit_log_best_effort(
            audit,
            "token.password_verified" if ok else "token.password_failed",
            resource_type=validated["resource_type"],
            resource_id=validated["resource_id"],
            owner_user_id=validated["owner_user_id"],
            token_id=validated["id"],
            ip_address=_client_ip(request),
            user_agent=request.headers.get("user-agent"),
        )
        if not ok:
            raise HTTPException(status_code=403, detail="Invalid password")

    claimed = await svc.claim_token_use(validated["id"])
    if not claimed:
        raise HTTPException(
            status_code=404,
            detail="Resource not found",
        )

    await _audit_log_best_effort(
        audit,
        "token.used",
        resource_type=validated["resource_type"],
        resource_id=validated["resource_id"],
        owner_user_id=validated["owner_user_id"],
        actor_user_id=user.id,
        token_id=validated["id"],
        ip_address=_client_ip(request),
    )

    return {
        "resource_type": validated["resource_type"],
        "resource_id": validated["resource_id"],
        "access_level": validated["access_level"],
        "owner_user_id": validated["owner_user_id"],
        "message": "Resource access granted. Use the resource_id to interact.",
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Admin
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.get(
    "/admin/shares",
    response_model=AdminShareListResponse,
    dependencies=[Depends(rbac_rate_limit("sharing.admin"))],
    summary="List all shares (admin)",
)
async def admin_list_shares(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    include_revoked: bool = Query(False),
    user: User = Depends(get_request_user),
):
    repo = await _maybe_await(_get_repo())
    shares = await repo.list_all_shares(limit=limit, offset=offset, include_revoked=include_revoked)
    total = await repo.count_all_shares(include_revoked=include_revoked)
    pagination = build_offset_pagination_meta(
        limit=limit,
        offset=offset,
        total=total,
        count=len(shares),
    )
    return AdminShareListResponse(
        shares=[ShareResponse(**s) for s in shares],
        total=total,
        offset=offset,
        limit=limit,
        pagination=pagination,
    )


@router.patch(
    "/admin/config",
    dependencies=[Depends(rbac_rate_limit("sharing.admin"))],
    summary="Update sharing configuration",
)
async def admin_update_config(
    body: UpdateConfigRequest,
    user: User = Depends(get_request_user),
):
    repo = await _maybe_await(_get_repo())
    for key, value in body.config.items():
        await repo.set_config(
            key,
            value,
            scope_type=body.scope_type,
            scope_id=body.scope_id,
            updated_by=user.id,
        )
    return {"detail": "Config updated"}


@router.get(
    "/admin/audit",
    response_model=AuditLogResponse,
    dependencies=[Depends(rbac_rate_limit("sharing.admin"))],
    summary="Query sharing audit log",
)
async def admin_audit_log(
    owner_user_id: int | None = Query(None),
    resource_type: str | None = Query(None),
    resource_id: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_request_user),
):
    audit = _get_audit_service()
    events = await audit.query(
        owner_user_id=owner_user_id,
        resource_type=resource_type,
        resource_id=resource_id,
        limit=limit,
        offset=offset,
    )
    total = await audit.count(
        owner_user_id=owner_user_id,
        resource_type=resource_type,
        resource_id=resource_id,
    )
    pagination = build_offset_pagination_meta(
        limit=limit,
        offset=offset,
        total=total,
        count=len(events),
    )
    return AuditLogResponse(
        events=[AuditEventResponse(**e) for e in events],
        total=total,
        offset=offset,
        limit=limit,
        pagination=pagination,
    )
