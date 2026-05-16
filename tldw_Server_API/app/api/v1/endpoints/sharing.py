# sharing.py
# Description: API endpoints for workspace sharing, share tokens, and admin
"""
Sharing API Endpoints
---------------------

Provides REST API endpoints for sharing workspaces with teams/orgs,
creating share links (tokens), and admin management.
"""
from __future__ import annotations

import inspect
import threading
import time
from collections import defaultdict
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, Response, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user, rbac_rate_limit
from tldw_Server_API.app.api.v1.utils.pagination import build_offset_pagination_meta

from ..schemas.prototype_workspace_schemas import (
    PrototypeErrorCategory,
    PrototypeErrorResponse,
    PrototypeFrontendState,
    prototype_error_detail,
)
from ..schemas.sharing_schemas import (
    AdminShareListResponse,
    AuditEventResponse,
    AuditLogResponse,
    CloneWorkspaceRequest,
    CloneWorkspaceResponse,
    CreateTokenRequest,
    PrototypeLinkExchangeRequest,
    PrototypeLinkExchangeResponse,
    PublicSharePreview,
    ResourceType,
    SharedChatRequest,
    SharedMediaResponse,
    SharedWithMeItem,
    SharedWithMeResponse,
    SharedWorkspaceSourceResponse,
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

router = APIRouter(prefix="/sharing", tags=["sharing"])
_SHARED_CHAT_ERROR_MESSAGE = "Chat request failed"
_SHARED_CHAT_ERRORS_MESSAGE = "One or more internal pipeline errors were suppressed."
_PROTOTYPE_LINK_ERROR_RESPONSES: dict[int, dict[str, Any]] = {
    status.HTTP_403_FORBIDDEN: {
        "model": PrototypeErrorResponse,
        "description": "Prototype link requires credentials or the workspace/session is unavailable.",
    },
    status.HTTP_404_NOT_FOUND: {
        "model": PrototypeErrorResponse,
        "description": "Prototype link is invalid, unavailable, exhausted, or cannot be resumed.",
    },
    status.HTTP_422_UNPROCESSABLE_ENTITY: {
        "model": PrototypeErrorResponse,
        "description": "Prototype link exchange request is semantically invalid.",
    },
}


def _prototype_http_error(
    *,
    status_code: int,
    category: PrototypeErrorCategory,
    message: str,
    frontend_state: PrototypeFrontendState,
    retryable: bool = False,
) -> HTTPException:
    """Build a prototype-link HTTPException with stable machine-readable detail."""
    return HTTPException(
        status_code=status_code,
        detail=prototype_error_detail(
            category=category,
            message=message,
            frontend_state=frontend_state,
            retryable=retryable,
        ),
    )


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
            raise _prototype_http_error(
                status_code=status.HTTP_404_NOT_FOUND,
                category="invalid_or_unavailable_link",
                message="Prototype link is unavailable",
                frontend_state="link_unavailable",
            )
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resource not found")
    return workspace


def _sanitize_shared_chat_result(value: Any) -> Any:
    """Redact nested internal error details from shared chat data."""
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"exception", "traceback", "stack", "stack_trace"} and item:
                sanitized[key] = _SHARED_CHAT_ERROR_MESSAGE
                continue
            if key in {"error", "errors"}:
                continue
            sanitized[key] = _sanitize_shared_chat_result(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_shared_chat_result(item) for item in value]
    return value


def _build_shared_chat_response(result: Any) -> dict[str, Any]:
    """Construct a safe shared-chat response without echoing raw pipeline errors."""
    if hasattr(result, "model_dump"):
        payload = result.model_dump()
    elif isinstance(result, dict):
        payload = dict(result)
    else:
        return {"result": _sanitize_shared_chat_result(result)}

    response: dict[str, Any] = {}
    for key in (
        "query",
        "documents",
        "expanded_queries",
        "metadata",
        "timings",
        "citations",
        "academic_citations",
        "chunk_citations",
        "generated_answer",
        "cache_hit",
        "security_report",
        "total_time",
        "claims",
        "factuality",
    ):
        if key in payload:
            response[key] = _sanitize_shared_chat_result(payload[key])

    if payload.get("error"):
        response["error"] = _SHARED_CHAT_ERROR_MESSAGE
    if payload.get("errors"):
        response["errors"] = [_SHARED_CHAT_ERRORS_MESSAGE]

    return response


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
        if "UNIQUE constraint" in str(exc):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This workspace is already shared with the specified scope.",
            ) from exc
        logger.error("Failed to create share")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while creating the share.",
        ) from exc

    await audit.log(
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
    shares = await repo.list_shares_for_workspace(
        workspace_id, user.id, include_revoked=include_revoked
    )
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

    await audit.log(
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

    await audit.log(
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

    # Gather shares from all teams/orgs the user belongs to
    items: list[SharedWithMeItem] = []
    team_ids = getattr(user, "team_ids", None) or []
    org_ids = getattr(user, "org_ids", None) or []

    for tid in team_ids:
        shares = await repo.list_shares_for_scope("team", tid)
        for s in shares:
            if s["owner_user_id"] != user.id:
                items.append(SharedWithMeItem(
                    share_id=s["id"],
                    workspace_id=s["workspace_id"],
                    owner_user_id=s["owner_user_id"],
                    access_level=s["access_level"],
                    allow_clone=s["allow_clone"],
                    shared_at=s.get("created_at"),
                ))

    for oid in org_ids:
        shares = await repo.list_shares_for_scope("org", oid)
        for s in shares:
            if s["owner_user_id"] != user.id:
                # Deduplicate by share_id
                if not any(i.share_id == s["id"] for i in items):
                    items.append(SharedWithMeItem(
                        share_id=s["id"],
                        workspace_id=s["workspace_id"],
                        owner_user_id=s["owner_user_id"],
                        access_level=s["access_level"],
                        allow_clone=s["allow_clone"],
                        shared_at=s.get("created_at"),
                    ))

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


@router.get(
    "/shared-with-me/{share_id}/workspace",
    dependencies=[Depends(rbac_rate_limit("sharing.read"))],
    summary="Read shared workspace metadata",
)
async def get_shared_workspace(
    share_id: int,
    user: User = Depends(get_request_user),
):
    repo = await _maybe_await(_get_repo())
    share = await repo.get_share(share_id)
    if not share or share.get("is_revoked"):
        raise HTTPException(status_code=404, detail="Share not found or revoked")

    # [CRITICAL FIX #2] Validate the user belongs to the share's scope
    await _validate_user_has_share_access(share, user)

    return {"share": ShareResponse(**share)}


@router.post(
    "/shared-with-me/{share_id}/clone",
    response_model=CloneWorkspaceResponse,
    dependencies=[Depends(rbac_rate_limit("sharing.clone"))],
    summary="Clone a shared workspace into your own account",
)
async def clone_shared_workspace(
    share_id: int,
    body: CloneWorkspaceRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_request_user),
):
    repo = await _maybe_await(_get_repo())
    audit = _get_audit_service()

    share = await repo.get_share(share_id)
    if not share or share.get("is_revoked"):
        raise HTTPException(status_code=404, detail="Share not found or revoked")

    # [CRITICAL FIX #2] Validate the user belongs to the share's scope
    await _validate_user_has_share_access(share, user)

    if not share.get("allow_clone"):
        raise HTTPException(status_code=403, detail="Cloning is not allowed for this share")

    # Generate a job ID for async clone tracking
    import uuid as _uuid
    job_id = str(_uuid.uuid4())

    await audit.log(
        "share.cloned",
        resource_type="workspace",
        resource_id=share["workspace_id"],
        owner_user_id=share["owner_user_id"],
        actor_user_id=user.id,
        share_id=share_id,
        metadata={"job_id": job_id, "new_name": body.new_name},
        ip_address=_client_ip(request),
    )

    background_tasks.add_task(
        _run_clone_task,
        share=share,
        user_id=user.id,
        new_name=body.new_name,
        job_id=job_id,
    )

    return CloneWorkspaceResponse(
        job_id=job_id,
        status="pending",
        message="Clone job created. Use the job_id to track progress.",
    )


def _run_clone_task(
    share: dict[str, Any],
    user_id: int,
    new_name: str | None,
    job_id: str,
) -> None:
    """Background task that performs the actual workspace clone."""
    import asyncio

    async def _do_clone() -> None:
        from ....core.Sharing.clone_service import CloneService
        from ..API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_owner, get_chacha_db_for_user_id
        from ..API_Deps.DB_Deps import managed_media_db_for_owner

        owner_id = share["owner_user_id"]
        workspace_id = share["workspace_id"]

        try:
            src_chacha = await get_chacha_db_for_owner(owner_id)
            tgt_chacha = await get_chacha_db_for_user_id(user_id)
            with managed_media_db_for_owner(owner_id) as src_media, managed_media_db_for_owner(user_id) as tgt_media:
                svc = CloneService(
                    source_chacha_db=src_chacha,
                    source_media_db=src_media,
                    target_chacha_db=tgt_chacha,
                    target_media_db=tgt_media,
                )
                result = svc.clone_workspace(workspace_id, new_name=new_name)
            logger.info(f"Clone job {job_id} completed: {result.get('workspace_id')}")
        except Exception:
            logger.error("Clone job failed")

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(_do_clone())
        else:
            loop.run_until_complete(_do_clone())
    except RuntimeError:
        asyncio.run(_do_clone())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shared-With-Me Proxy Endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.get(
    "/shared-with-me/{share_id}/sources",
    response_model=list[SharedWorkspaceSourceResponse],
    dependencies=[Depends(rbac_rate_limit("sharing.read"))],
    summary="List sources of a shared workspace",
)
async def list_shared_workspace_sources(
    share_id: int,
    user: User = Depends(get_request_user),
):
    repo = await _maybe_await(_get_repo())
    share = await repo.get_share(share_id)
    if not share or share.get("is_revoked"):
        raise HTTPException(status_code=404, detail="Share not found or revoked")

    await _validate_user_has_share_access(share, user)

    from ..API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_owner
    db = await get_chacha_db_for_owner(share["owner_user_id"])
    sources = db.list_workspace_sources(share["workspace_id"])
    return [
        SharedWorkspaceSourceResponse(
            id=s["id"],
            workspace_id=s["workspace_id"],
            media_id=s.get("media_id"),
            title=s.get("title", ""),
            source_type=s.get("source_type", "media"),
            url=s.get("url"),
            position=s.get("position", 0),
            added_at=str(s.get("added_at", "")),
        )
        for s in sources
    ]


@router.get(
    "/shared-with-me/{share_id}/media/{media_id}",
    response_model=SharedMediaResponse,
    dependencies=[Depends(rbac_rate_limit("sharing.read"))],
    summary="Read a media item from a shared workspace",
)
async def get_shared_workspace_media(
    share_id: int,
    media_id: int,
    user: User = Depends(get_request_user),
):
    repo = await _maybe_await(_get_repo())
    share = await repo.get_share(share_id)
    if not share or share.get("is_revoked"):
        raise HTTPException(status_code=404, detail="Share not found or revoked")

    await _validate_user_has_share_access(share, user)

    # Verify media_id is a source in this workspace
    from ..API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_owner
    chacha_db = await get_chacha_db_for_owner(share["owner_user_id"])
    sources = chacha_db.list_workspace_sources(share["workspace_id"])
    source_media_ids = {s.get("media_id") for s in sources}
    if media_id not in source_media_ids:
        raise HTTPException(
            status_code=404,
            detail="Media item not found in this shared workspace",
        )

    from ..API_Deps.DB_Deps import managed_media_db_for_owner

    with managed_media_db_for_owner(share["owner_user_id"]) as media_db:
        media = media_db.get_media_by_id(media_id)
    if not media:
        raise HTTPException(status_code=404, detail="Media item not found")

    return SharedMediaResponse(
        id=media["id"],
        title=media.get("title", ""),
        url=media.get("url"),
        media_type=media.get("type"),
        content=media.get("content"),
        author=media.get("author"),
        ingestion_date=media.get("ingestion_date"),
    )


@router.post(
    "/shared-with-me/{share_id}/chat",
    response_model=dict[str, Any],
    dependencies=[Depends(rbac_rate_limit("sharing.read"))],
    summary="Chat with a shared workspace's sources via RAG",
)
async def chat_with_shared_workspace(
    share_id: int,
    body: SharedChatRequest,
    request: Request,
    user: User = Depends(get_request_user),
):
    repo = await _maybe_await(_get_repo())
    audit = _get_audit_service()

    share = await repo.get_share(share_id)
    if not share or share.get("is_revoked"):
        raise HTTPException(status_code=404, detail="Share not found or revoked")

    await _validate_user_has_share_access(share, user)

    from ..API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_owner
    from ..API_Deps.DB_Deps import get_media_db_path_for_rag, managed_media_db_for_owner

    owner_chacha = await get_chacha_db_for_owner(share["owner_user_id"])

    # Build RAG pipeline kwargs using the owner's databases but the accessor's namespace
    try:
        from ....core.RAG.rag_service.unified_pipeline import unified_rag_pipeline

        with managed_media_db_for_owner(share["owner_user_id"]) as owner_media:
            owner_media_db_path = get_media_db_path_for_rag(owner_media)
            result = await unified_rag_pipeline(
                query=body.query,
                media_db_path=owner_media_db_path,
                notes_db_path=owner_chacha.db_path if hasattr(owner_chacha, "db_path") else None,
                api_name=body.api_name,
                model=body.model,
                system_message=body.system_message,
                index_namespace=f"user_{share['owner_user_id']}_media_embeddings",
                media_db=owner_media,
                chacha_db=owner_chacha,
            )

        await audit.log(
            "share.chat",
            resource_type="workspace",
            resource_id=share["workspace_id"],
            owner_user_id=share["owner_user_id"],
            actor_user_id=user.id,
            share_id=share_id,
            ip_address=_client_ip(request),
        )

        return _build_shared_chat_response(result)
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="RAG pipeline not available",
        ) from None
    except Exception:
        logger.error("Shared workspace chat failed")
        raise HTTPException(status_code=500, detail="Chat request failed") from None


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
    if body.resource_type == ResourceType.PROTOTYPE_WORKSPACE:
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

    await audit.log(
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

    await audit.log(
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
    await audit.log(
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
    responses=_PROTOTYPE_LINK_ERROR_RESPONSES,
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
        raise _prototype_http_error(
            status_code=status.HTTP_404_NOT_FOUND,
            category="invalid_or_unavailable_link",
            message="Prototype link is unavailable",
            frontend_state="link_unavailable",
        )

    resource_type = str(validated.get("resource_type") or "").strip().lower()
    prototype_workspace_id = str(validated.get("resource_id") or "").strip()
    if resource_type != ResourceType.PROTOTYPE_WORKSPACE.value or not prototype_workspace_id:
        raise _prototype_http_error(
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
            await audit.log(
                "token.password_verified" if password_ok else "token.password_failed",
                resource_type=validated["resource_type"],
                resource_id=validated["resource_id"],
                owner_user_id=validated["owner_user_id"],
                token_id=validated["id"],
                ip_address=_client_ip(request),
                user_agent=request.headers.get("user-agent"),
            )
            if not password_ok:
                raise _prototype_http_error(
                    status_code=status.HTTP_403_FORBIDDEN,
                    category="invalid_password",
                    message="Prototype link password is invalid",
                    frontend_state="password_rejected",
                    retryable=True,
                )
        else:
            if not can_resume_without_password:
                raise _prototype_http_error(
                    status_code=status.HTTP_403_FORBIDDEN,
                    category="password_required",
                    message="Prototype link password is required",
                    frontend_state="password_required",
                    retryable=True,
                )

    if not can_resume_without_password and not str(body.display_name or "").strip():
        raise _prototype_http_error(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            category="invalid_request",
            message="display_name is required for first-time sessions",
            frontend_state="invalid_request",
        )

    claimed_new_use = False
    if not can_resume_without_password:
        claimed_new_use = await svc.claim_token_use(validated["id"])
        if not claimed_new_use:
            raise _prototype_http_error(
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
            raise _prototype_http_error(
                status_code=status.HTTP_404_NOT_FOUND,
                category="invalid_or_unavailable_link",
                message="Prototype link is unavailable",
                frontend_state="link_unavailable",
            ) from exc
        if exc.code == "workspace_archived":
            raise _prototype_http_error(
                status_code=status.HTTP_403_FORBIDDEN,
                category="workspace_unavailable",
                message="Prototype workspace is archived",
                frontend_state="workspace_unavailable",
            ) from exc
        if exc.code == "resume_required":
            raise _prototype_http_error(
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
        await audit.log(
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
    user: User = Depends(get_request_user),
):
    svc = await _maybe_await(_get_token_service())
    audit = _get_audit_service()

    validated = await svc.validate_token(token)
    if not validated:
        raise HTTPException(status_code=404, detail="Resource not found")

    if validated.get("resource_type") == ResourceType.PROTOTYPE_WORKSPACE.value:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Prototype workspace links must be exchanged via /prototype-session",
        )

    # [CRITICAL FIX #4] Block import on password-protected tokens without verification
    if validated.get("is_password_protected"):
        raise HTTPException(
            status_code=403,
            detail="Password verification required. Call /verify first.",
        )

    # Increment use count
    await svc.use_token(validated["id"])

    await audit.log(
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
            key, value,
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
