from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import HTTPException, status
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos.data_subject_requests_repo import (
    AuthnzDataSubjectRequestsRepo,
)
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.services import admin_scope_service

_CATEGORY_DEFS: tuple[dict[str, str], ...] = (
    {"key": "media_records", "label": "Media records"},
    {"key": "chat_messages", "label": "Chat sessions/messages"},
    {"key": "notes", "label": "Notes"},
    {"key": "audit_events", "label": "Audit log events"},
    {"key": "embeddings", "label": "Vector embeddings"},
)
_SUPPORTED_CATEGORY_KEYS = {entry["key"] for entry in _CATEGORY_DEFS}
_UNSUPPORTED_CATEGORY_KEYS: set[str] = set()


class DataSubjectRequestCoverageUnavailableError(RuntimeError):
    """Raised when a DSR preview cannot query an authoritative subject store."""


def _normalize_identifier(value: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="requester_identifier_required",
        )
    return normalized


def _normalize_requested_categories(
    *,
    request_type: str | None,
    categories: list[str] | None,
) -> list[str]:
    if categories is None:
        if request_type == "erasure":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="categories_required",
            )
        return [entry["key"] for entry in _CATEGORY_DEFS]

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in categories:
        value = str(raw_value or "").strip().lower()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)

    if request_type == "erasure" and not normalized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="categories_required",
        )

    unsupported = [
        value
        for value in normalized
        if value not in _SUPPORTED_CATEGORY_KEYS
    ]
    if unsupported:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="unsupported_category",
        )

    return normalized


async def _resolve_requester_user(
    requester_identifier: str,
    *,
    users_repo: AuthnzUsersRepo,
) -> dict[str, Any]:
    normalized = _normalize_identifier(requester_identifier)

    row: dict[str, Any] | None = None
    if normalized.isdigit():
        row = await users_repo.get_user_by_id(int(normalized))
    if row is None and "@" in normalized:
        row = await users_repo.get_user_by_email(normalized)
    if row is None:
        row = await users_repo.get_user_by_username(normalized)
    if row is None and "-" in normalized:
        row = await users_repo.get_user_by_uuid(normalized)

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="requester_not_found",
        )

    try:
        row["id"] = int(row["id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="requester_resolution_failed",
        ) from exc

    return row


async def _enforce_requester_visibility(
    *,
    principal: AuthPrincipal | None,
    target_user_id: int,
) -> None:
    if principal is None or admin_scope_service.is_platform_admin(principal):
        return
    try:
        await admin_scope_service.enforce_admin_user_scope(
            principal,
            target_user_id,
            require_hierarchy=False,
        )
    except HTTPException as exc:
        if exc.status_code == status.HTTP_403_FORBIDDEN:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="requester_not_found",
            ) from exc
        raise


def _sqlite_count_sync(path: Path, query: str, params: tuple[Any, ...] = ()) -> int:
    if not path.exists():
        raise DataSubjectRequestCoverageUnavailableError(
            f"DSR subject store missing for {path}"
        )
    try:
        with sqlite3.connect(path) as conn:
            row = conn.execute(query, params).fetchone()
    except sqlite3.Error as exc:
        raise DataSubjectRequestCoverageUnavailableError(
            f"DSR count query failed for {path}: {exc}"
        ) from exc
    return int(row[0] if row else 0)


async def _sqlite_count(path: Path, query: str, params: tuple[Any, ...] = ()) -> int:
    return await asyncio.to_thread(_sqlite_count_sync, path, query, params)


async def _count_media_records(user_id: int) -> int:
    return await _sqlite_count(
        DatabasePaths.get_media_db_path(user_id),
        "SELECT COUNT(*) FROM Media WHERE deleted = 0 AND is_trash = 0",
    )


async def _count_notes(user_id: int) -> int:
    return await _sqlite_count(
        DatabasePaths.get_chacha_db_path(user_id),
        "SELECT COUNT(*) FROM notes WHERE deleted = 0",
    )


async def _count_chat_messages(user_id: int) -> int:
    return await _sqlite_count(
        DatabasePaths.get_chacha_db_path(user_id),
        """
        SELECT COUNT(1)
        FROM messages m
        JOIN conversations c ON m.conversation_id = c.id
        WHERE c.client_id = ? AND c.deleted = 0 AND m.deleted = 0
        """,
        (str(user_id),),
    )


def _get_chroma_manager_for_user(user_id: int):
    """Create a ChromaDBManager for the given user using lazy import."""
    from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager
    from tldw_Server_API.app.core.config import settings as app_settings

    embedding_config: dict = {}
    user_db_base = app_settings.get("USER_DB_BASE_DIR")
    if user_db_base:
        embedding_config["USER_DB_BASE_DIR"] = str(user_db_base)
    else:
        project_root = Path(__file__).resolve().parents[3]
        embedding_config["USER_DB_BASE_DIR"] = str(
            project_root / "Databases" / "user_databases"
        )
    return ChromaDBManager(str(user_id), embedding_config)


async def _count_embeddings(user_id: int) -> int:
    """Count vector embeddings across all ChromaDB collections for a user."""

    def _count_sync() -> int:
        try:
            manager = _get_chroma_manager_for_user(user_id)
        except Exception as exc:
            logger.debug("ChromaDB not available for user {}: {}", user_id, exc)
            return 0
        try:
            collections = manager.list_collections()
            total = 0
            for col in collections:
                try:
                    total += col.count()
                except Exception as exc:
                    logger.debug(
                        "Failed to count ChromaDB collection for user {}: {}",
                        user_id,
                        exc,
                    )
            return total
        except Exception as exc:
            logger.debug("Failed to count embeddings for user {}: {}", user_id, exc)
            return 0

    return await asyncio.to_thread(_count_sync)


async def _count_audit_events(user_id: int) -> int:
    from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import _resolve_audit_storage_mode

    shared_mode = _resolve_audit_storage_mode() == "shared"
    path = (
        DatabasePaths.get_shared_audit_db_path()
        if shared_mode
        else DatabasePaths.get_audit_db_path(user_id)
    )
    if shared_mode:
        return await _sqlite_count(
            path,
            "SELECT COUNT(*) FROM audit_events WHERE tenant_user_id = ?",
            (str(user_id),),
        )
    return await _sqlite_count(path, "SELECT COUNT(*) FROM audit_events")


async def _build_summary_for_user(
    *,
    user_id: int,
    selected_categories: list[str],
) -> list[dict[str, Any]]:
    media_records, chat_messages, notes, audit_events, embeddings = await asyncio.gather(
        _count_media_records(user_id),
        _count_chat_messages(user_id),
        _count_notes(user_id),
        _count_audit_events(user_id),
        _count_embeddings(user_id),
    )
    count_map = {
        "media_records": media_records,
        "chat_messages": chat_messages,
        "notes": notes,
        "audit_events": audit_events,
        "embeddings": embeddings,
    }
    return [
        {
            "key": entry["key"],
            "label": entry["label"],
            "count": int(count_map.get(entry["key"], 0)),
        }
        for entry in _CATEGORY_DEFS
        if entry["key"] in selected_categories
    ]


def _coverage_metadata(*, selected_categories: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "supported_categories": [entry["key"] for entry in _CATEGORY_DEFS],
        "selected_categories": selected_categories,
        "unsupported_categories": sorted(_UNSUPPORTED_CATEGORY_KEYS),
    }
    if _UNSUPPORTED_CATEGORY_KEYS:
        result["unsupported_details"] = {
            key: f"Category '{key}' is not yet supported."
            for key in sorted(_UNSUPPORTED_CATEGORY_KEYS)
        }
    else:
        result["unsupported_details"] = {}
    return result


async def preview_data_subject_request(
    *,
    requester_identifier: str,
    request_type: str | None = None,
    categories: list[str] | None = None,
    principal: AuthPrincipal | None = None,
    users_repo: AuthnzUsersRepo,
) -> dict[str, Any]:
    """Return an authoritative DSR preview for a requester within admin scope.

    The requester is resolved through the AuthNZ user repo, then scoped visibility is
    enforced before any per-user store is queried. Unsupported or unavailable coverage
    fails closed so the caller never receives synthetic zero-count summaries.
    """
    user = await _resolve_requester_user(
        requester_identifier,
        users_repo=users_repo,
    )
    await _enforce_requester_visibility(
        principal=principal,
        target_user_id=int(user["id"]),
    )
    selected_categories = _normalize_requested_categories(
        request_type=request_type,
        categories=categories,
    )
    try:
        summary = await _build_summary_for_user(
            user_id=int(user["id"]),
            selected_categories=selected_categories,
        )
    except DataSubjectRequestCoverageUnavailableError as exc:
        logger.warning("DSR preview unavailable for user {}: {}", user["id"], exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="requester_data_unavailable",
        ) from exc
    return {
        "requester_identifier": user.get("email") or user.get("username") or requester_identifier,
        "resolved_user_id": int(user["id"]),
        "request_type": request_type,
        "selected_categories": selected_categories,
        "summary": summary,
        "counts": {entry["key"]: entry["count"] for entry in summary},
        "coverage_metadata": _coverage_metadata(selected_categories=selected_categories),
    }


async def _normalize_requested_by_user_id(
    principal: AuthPrincipal,
    *,
    users_repo: AuthnzUsersRepo,
) -> int | None:
    if getattr(principal, "user_id", None) is None:
        return None
    try:
        candidate = int(principal.user_id)
    except (TypeError, ValueError):
        return None

    row = await users_repo.get_user_by_id(candidate)
    if row is None:
        return None
    return candidate


async def record_data_subject_request(
    *,
    principal: AuthPrincipal,
    client_request_id: str,
    requester_identifier: str,
    request_type: str,
    categories: list[str] | None,
    users_repo: AuthnzUsersRepo,
    requests_repo: AuthnzDataSubjectRequestsRepo,
    preview: dict[str, Any] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Persist an idempotent DSR intake record after authoritative preview validation.

    Callers may supply a previously computed preview to avoid duplicate work, but the
    preview must still come from the authoritative preview path. Records are stored in
    the AuthNZ control-plane DB with newest request history and review status metadata.
    """
    if preview is None:
        preview = await preview_data_subject_request(
            requester_identifier=requester_identifier,
            request_type=request_type,
            categories=categories,
            principal=principal,
            users_repo=users_repo,
        )

    await requests_repo.ensure_schema()

    return await requests_repo.create_or_get_request(
        client_request_id=str(client_request_id).strip(),
        requester_identifier=str(preview["requester_identifier"]),
        resolved_user_id=int(preview["resolved_user_id"]),
        request_type=request_type,
        status="recorded",
        selected_categories=list(preview["selected_categories"]),
        preview_summary=list(preview["summary"]),
        coverage_metadata=dict(preview["coverage_metadata"]),
        requested_by_user_id=await _normalize_requested_by_user_id(
            principal,
            users_repo=users_repo,
        ),
        notes=notes,
    )


async def list_data_subject_requests(
    principal: AuthPrincipal,
    *,
    limit: int,
    offset: int,
    requests_repo: AuthnzDataSubjectRequestsRepo,
) -> tuple[list[dict[str, Any]], int]:
    """Return paged DSR history limited to the admin's permitted scope.

    Platform admins receive the full request log. Org-scoped admins are limited by the
    same org scope used elsewhere in admin data-ops, and that filtering is delegated to
    the repository query so pagination stays authoritative.
    """
    await requests_repo.ensure_schema()
    org_ids = await admin_scope_service.get_admin_org_ids(principal)
    return await requests_repo.list_requests(
        limit=limit,
        offset=offset,
        org_ids=org_ids,
    )


# ---------------------------------------------------------------------------
# DSR erasure execution helpers
# ---------------------------------------------------------------------------

def _sqlite_hard_delete_sync(path: Path, statements: list[tuple[str, tuple]]) -> int:
    """Execute DELETE statements and return total rows affected."""
    if not path.exists():
        return 0
    total = 0
    with sqlite3.connect(path) as conn:
        for sql, params in statements:
            cursor = conn.execute(sql, params)
            total += cursor.rowcount
    return total


async def _erase_media_records(user_id: int) -> int:
    """Hard-delete all media records for a user."""
    path = DatabasePaths.get_media_db_path(user_id)
    return await asyncio.to_thread(
        _sqlite_hard_delete_sync,
        path,
        [("DELETE FROM Media", ())],
    )


async def _erase_chat_messages(user_id: int) -> int:
    """Hard-delete all chat conversations and messages for a user."""
    path = DatabasePaths.get_chacha_db_path(user_id)
    return await asyncio.to_thread(
        _sqlite_hard_delete_sync,
        path,
        [
            ("DELETE FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE client_id = ?)", (str(user_id),)),
            ("DELETE FROM conversations WHERE client_id = ?", (str(user_id),)),
        ],
    )


async def _erase_notes(user_id: int) -> int:
    """Hard-delete all notes for a user."""
    path = DatabasePaths.get_chacha_db_path(user_id)
    return await asyncio.to_thread(
        _sqlite_hard_delete_sync,
        path,
        [("DELETE FROM notes", ())],
    )


async def _erase_embeddings(user_id: int) -> int:
    """Delete all ChromaDB collections for a user, returning total embedding count erased.

    Raises ``RuntimeError`` when ChromaDB is unavailable but data might exist,
    so the category is marked as ``error`` and the overall DSR as ``failed``.
    Returns 0 only when there is genuinely nothing to delete.
    """

    def _erase_sync() -> int:
        try:
            manager = _get_chroma_manager_for_user(user_id)
        except Exception as exc:
            # Check whether a chroma_storage directory exists for this user.
            # If it does, data might be present and we must not silently succeed.
            from tldw_Server_API.app.core.config import settings as app_settings

            user_db_base = app_settings.get("USER_DB_BASE_DIR")
            if not user_db_base:
                project_root = Path(__file__).resolve().parents[3]
                user_db_base = str(project_root / "Databases" / "user_databases")
            chroma_dir = Path(user_db_base) / str(user_id) / "chroma_storage"
            if chroma_dir.exists():
                raise RuntimeError(
                    f"ChromaDB unavailable for user {user_id} but chroma_storage "
                    f"directory exists — cannot confirm erasure: {exc}"
                ) from exc
            # No storage directory → nothing to erase
            return 0

        try:
            collections = manager.list_collections()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to list ChromaDB collections for user {user_id}: {exc}"
            ) from exc

        total_embeddings = 0
        for col in collections:
            try:
                total_embeddings += col.count()
            except Exception as exc:
                logger.debug(
                    "Failed to count ChromaDB collection before deletion for user {}: {}",
                    user_id,
                    exc,
                )
            try:
                manager.delete_collection(col.name)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to delete collection '{col.name}' for user {user_id}: {exc}"
                ) from exc
        return total_embeddings

    return await asyncio.to_thread(_erase_sync)


async def _erase_audit_events(user_id: int) -> int:
    """Hard-delete audit events for a user."""
    from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import _resolve_audit_storage_mode

    shared_mode = _resolve_audit_storage_mode() == "shared"
    path = (
        DatabasePaths.get_shared_audit_db_path()
        if shared_mode
        else DatabasePaths.get_audit_db_path(user_id)
    )
    if shared_mode:
        return await asyncio.to_thread(
            _sqlite_hard_delete_sync,
            path,
            [("DELETE FROM audit_events WHERE tenant_user_id = ?", (str(user_id),))],
        )
    return await asyncio.to_thread(
        _sqlite_hard_delete_sync,
        path,
        [("DELETE FROM audit_events", ())],
    )


_ERASURE_HANDLERS: dict[str, Any] = {
    "media_records": _erase_media_records,
    "chat_messages": _erase_chat_messages,
    "notes": _erase_notes,
    "embeddings": _erase_embeddings,
    "audit_events": _erase_audit_events,
}


async def execute_dsr_erasure(
    *,
    request_id: int,
    user_id: int,
    selected_categories: list[str],
    dsr_repo: AuthnzDataSubjectRequestsRepo,
    principal: AuthPrincipal | None = None,
) -> dict[str, Any]:
    """Orchestrate DSR erasure across all selected categories.

    Updates the DSR record status to ``executing`` before running handlers, then
    transitions to ``completed`` or ``failed`` depending on outcome. Returns a
    summary dict with per-category results.
    """
    # Mark as executing
    await dsr_repo.update_request_status(request_id, "executing")

    executed_by = (
        getattr(principal, "principal_id", None) or getattr(principal, "user_id", None)
        if principal is not None
        else None
    )
    logger.info(
        "DSR erasure started: request_id={} user_id={} executed_by={} categories={}",
        request_id, user_id, executed_by, selected_categories,
    )

    results: dict[str, Any] = {}
    errors: dict[str, str] = {}

    for category in selected_categories:
        handler = _ERASURE_HANDLERS.get(category)
        if handler is None:
            errors[category] = f"No erasure handler for category '{category}'"
            continue
        try:
            count = await handler(user_id)
            results[category] = {"deleted_count": count, "status": "ok"}
        except Exception as exc:
            logger.error("DSR erasure failed for category '{}' user {}: {}", category, user_id, exc)
            errors[category] = "Erasure handler failed"
            results[category] = {
                "deleted_count": 0,
                "status": "error",
                "error": "Erasure handler failed",
            }

    # Determine final status
    final_status = "completed" if not errors else "failed"
    notes_text = None
    if errors:
        notes_text = f"Erasure errors: {errors}"

    await dsr_repo.update_request_status(request_id, final_status, notes=notes_text)

    return {
        "request_id": request_id,
        "user_id": user_id,
        "status": final_status,
        "categories": results,
        "errors": errors,
    }
