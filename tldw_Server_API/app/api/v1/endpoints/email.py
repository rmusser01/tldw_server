from __future__ import annotations

from math import ceil
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    get_auth_principal,
    get_db_transaction,
)
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    DatabaseError,
    InputError,
)
from tldw_Server_API.app.core.External_Sources.connectors_service import (
    create_import_job,
    get_source_by_id,
    list_sources as list_connector_sources,
)
from tldw_Server_API.app.core.Logging.log_context import ensure_request_id
from tldw_Server_API.app.core.config import settings

router = APIRouter(tags=["Email"])


def _ensure_email_operator_search_enabled() -> None:
    if bool(settings.get("EMAIL_OPERATOR_SEARCH_ENABLED", True)):
        return
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Email operator search APIs are disabled.",
    )


def _ensure_email_source_sync_enabled() -> None:
    if bool(settings.get("EMAIL_GMAIL_CONNECTOR_ENABLED", False)):
        return
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Email source sync APIs are disabled.",
    )


def _get_user_id(principal: AuthPrincipal) -> int:
    user_id = principal.user_id
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User ID not found in principal.",
        )
    try:
        return int(user_id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user ID in principal.",
        ) from exc


def _as_str_or_none(value: Any) -> str | None:
    return None if value is None else str(value)


def _derive_sync_state(sync_state: dict[str, Any] | None) -> str:
    if not sync_state:
        return "never_synced"
    error_state = str(sync_state.get("error_state") or "").strip()
    retry_backoff_count = int(sync_state.get("retry_backoff_count") or 0)
    if error_state:
        return "retrying" if retry_backoff_count > 0 else "failed"
    if sync_state.get("last_success_at"):
        return "healthy"
    if sync_state.get("last_run_at"):
        return "running"
    return "never_synced"


@router.get(
    "/sources",
    status_code=status.HTTP_200_OK,
    summary="List email sources with sync status",
)
async def list_email_sources(
    db=Depends(get_db_transaction),
    principal: AuthPrincipal = Depends(get_auth_principal),
    media_db: Any = Depends(get_media_db_for_user),
) -> dict[str, Any]:
    _ensure_email_source_sync_enabled()
    user_id = _get_user_id(principal)
    tenant_id = str(user_id)

    try:
        rows = await list_connector_sources(db, user_id)
    except Exception as exc:
        logger.error("Failed to list email sources")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list email sources.",
        ) from exc

    items: list[dict[str, Any]] = []
    for row in rows:
        provider = str(row.get("provider") or "").strip().lower()
        if provider != "gmail":
            continue
        source_id = int(row.get("id"))
        sync_state: dict[str, Any] | None = None
        if hasattr(media_db, "get_email_sync_state"):
            try:
                sync_state = media_db.get_email_sync_state(
                    provider=provider,
                    source_key=str(source_id),
                    tenant_id=tenant_id,
                )
            except (DatabaseError, InputError):
                logger.warning("Failed to fetch email sync state")

        sync_payload = {
            "state": _derive_sync_state(sync_state),
            "cursor": (sync_state or {}).get("cursor"),
            "last_run_at": _as_str_or_none((sync_state or {}).get("last_run_at")),
            "last_success_at": _as_str_or_none((sync_state or {}).get("last_success_at")),
            "error_state": (sync_state or {}).get("error_state"),
            "retry_backoff_count": int((sync_state or {}).get("retry_backoff_count") or 0),
            "updated_at": _as_str_or_none((sync_state or {}).get("updated_at")),
        }

        items.append(
            {
                "id": source_id,
                "account_id": int(row.get("account_id")),
                "provider": provider,
                "remote_id": row.get("remote_id"),
                "type": row.get("type"),
                "path": row.get("path"),
                "options": row.get("options") or {},
                "enabled": bool(row.get("enabled", True)),
                "last_synced_at": _as_str_or_none(row.get("last_synced_at")),
                "sync": sync_payload,
            }
        )

    return {"items": items, "total": len(items)}


@router.post(
    "/sources/{source_id}/sync",
    status_code=status.HTTP_200_OK,
    summary="Trigger email source sync",
)
async def trigger_email_source_sync(
    request: Request,
    source_id: int = Path(..., ge=1),
    db=Depends(get_db_transaction),
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> dict[str, Any]:
    _ensure_email_source_sync_enabled()
    user_id = _get_user_id(principal)

    source = await get_source_by_id(db, user_id, source_id)
    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Email source not found.",
        )
    provider = str(source.get("provider") or "").strip().lower()
    if provider != "gmail":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only Gmail sources can be synced via this endpoint.",
        )

    rid = ensure_request_id(request) if request is not None else None
    try:
        job = await create_import_job(user_id, source_id, request_id=rid)
    except Exception as exc:
        logger.error("Failed to queue email sync job")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to queue email sync job.",
        ) from exc

    return {
        "source_id": int(source_id),
        "provider": provider,
        "status": "queued",
        "job": job,
    }


@router.get(
    "/search",
    status_code=status.HTTP_200_OK,
    summary="Search normalized email messages",
)
async def search_email_messages(
    q: str | None = Query(
        default=None,
        description=(
            "Operator query (from:, to:, cc:, bcc:, subject:, label:, has:attachment, "
            "before:, after:, older_than:, newer_than:) and free text."
        ),
    ),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Any = Depends(get_media_db_for_user),
) -> dict[str, Any]:
    """Run Stage-1 email operator search."""

    _ensure_email_operator_search_enabled()

    try:
        rows, total = db.search_email_messages(
            query=q,
            include_deleted=False,
            limit=limit,
            offset=offset,
        )
        total_pages = ceil(total / limit) if limit > 0 and total > 0 else 0
        pagination = build_offset_pagination_meta(
            limit=limit,
            offset=offset,
            total=total,
            count=len(rows),
        ).model_dump(mode="json")
        pagination["total_pages"] = int(total_pages)
        return {
            "items": rows,
            "pagination": pagination,
            "has_more": pagination["has_more"],
            "next_offset": pagination["next_offset"],
        }
    except (InputError, DatabaseError) as exc:
        if isinstance(exc, DatabaseError):
            logger.error("Database error during email search")
        raise map_db_error_to_http(
            exc,
            default_detail="A database error occurred during email search.",
        ) from exc


@router.get(
    "/messages/{email_message_id}",
    status_code=status.HTTP_200_OK,
    summary="Get normalized email message detail",
)
async def get_email_message_detail(
    email_message_id: int = Path(..., ge=1, description="Normalized email message id"),
    db: Any = Depends(get_media_db_for_user),
) -> dict[str, Any]:
    """Fetch participants, labels, attachments, and source/media links for one message."""

    _ensure_email_operator_search_enabled()

    try:
        detail = db.get_email_message_detail(
            email_message_id=email_message_id,
            include_deleted=False,
        )
        if detail is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Email message not found.",
            )
        return detail
    except (InputError, DatabaseError) as exc:
        if isinstance(exc, DatabaseError):
            logger.error("Database error during email detail lookup")
        raise map_db_error_to_http(
            exc,
            default_detail="A database error occurred while fetching email message detail.",
        ) from exc
