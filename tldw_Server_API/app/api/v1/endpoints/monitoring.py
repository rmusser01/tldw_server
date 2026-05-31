from __future__ import annotations

"""
Topic Monitoring API (Phase 1)

Permission-gated endpoints requiring SYSTEM_LOGS to manage watchlists, alerts, and notifications.
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import RequirePermission, get_auth_principal
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.schemas.monitoring_schemas import (
    AlertItem,
    AlertsListResponse,
    MarkReadResponse,
    NotificationSettings,
    NotificationSettingsUpdate,
    NotificationTestRequest,
    NotificationTestResponse,
    RecentNotificationsResponse,
    Watchlist,
    WatchlistDeleteResponse,
    WatchlistListResponse,
    WatchlistsReloadResponse,
    WatchlistUpsertResponse,
)
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_LOGS
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos.admin_monitoring_repo import (
    AuthnzAdminMonitoringRepo,
)
from tldw_Server_API.app.core.DB_Management.TopicMonitoring_DB import TopicAlert, TopicMonitoringDB
from tldw_Server_API.app.core.Monitoring.notification_service import get_notification_service
from tldw_Server_API.app.core.Monitoring.topic_monitoring_service import get_topic_monitoring_service
from tldw_Server_API.app.core.testing import is_test_mode as _is_test_mode
from tldw_Server_API.app.services.admin_monitoring_alerts_service import (
    build_alert_identity,
    merge_runtime_alert_with_overlay,
)

# Cached TopicMonitoringDB instance (initialized on first use).
_TOPIC_MONITORING_DB: TopicMonitoringDB | None = None

# All monitoring routes require SYSTEM_LOGS permission via this dependency.
router = APIRouter(
    dependencies=[Depends(RequirePermission(SYSTEM_LOGS))],
)


def _find_project_root(start: Path) -> Path | None:
    """Best-effort search for the repository root starting from a file/dir path."""
    start_dir = start if start.is_dir() else start.parent

    for candidate in (start_dir, *start_dir.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
        if (candidate / "AGENTS.md").is_file() and (candidate / "tldw_Server_API").is_dir():
            return candidate
        if (candidate / ".git").exists():
            return candidate
    return None

def get_topic_monitoring_db() -> TopicMonitoringDB:
    """Return a TopicMonitoringDB instance for alert reads/updates."""
    global _TOPIC_MONITORING_DB
    test_mode = _is_test_mode()
    if _TOPIC_MONITORING_DB is not None and not test_mode:
        return _TOPIC_MONITORING_DB

    raw_db_path = os.getenv("MONITORING_ALERTS_DB", "Databases/monitoring_alerts.db")
    try:
        db_path = Path(raw_db_path)
    except (TypeError, ValueError) as exc:
        msg = f"Invalid MONITORING_ALERTS_DB={raw_db_path!r}: {exc}"
        logger.error(msg)
        raise RuntimeError(msg) from exc
    if not db_path.is_absolute() and db_path.parent == Path("."):
        msg = (
            "MONITORING_ALERTS_DB must include a directory when using a relative path "
            f"(got {raw_db_path!r}). Use an absolute path or include a directory component."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    if db_path.is_absolute():
        db = TopicMonitoringDB(db_path=str(db_path.resolve()))
        if not test_mode:
            _TOPIC_MONITORING_DB = db
        return db

    try:
        from tldw_Server_API.app.core.Utils.Utils import get_project_root as _gpr
    except ImportError as exc:
        start = Path(__file__).resolve()
        root = _find_project_root(start)
        if root is None:
            msg = (
                "Unable to resolve monitoring alerts DB path: MONITORING_ALERTS_DB is relative "
                f"({raw_db_path!r}), and importing get_project_root failed: {exc}. "
                f"Searched parents of {start} for root markers (pyproject.toml, AGENTS.md, .git) "
                "but none were found."
            )
            logger.error(msg)
            raise RuntimeError(msg) from exc
        logger.debug(
            "monitoring: get_project_root unavailable ({}); using fallback root {}",
            exc,
            root,
        )
    else:
        root = Path(_gpr()).resolve()

    db_path = (root / db_path).resolve()

    db = TopicMonitoringDB(db_path=str(db_path))
    if not test_mode:
        _TOPIC_MONITORING_DB = db
    return db


async def _get_monitoring_repo() -> AuthnzAdminMonitoringRepo:
    """Return the shared admin monitoring repository for overlay state reads/writes."""
    pool = await get_db_pool()
    return AuthnzAdminMonitoringRepo(pool)


@router.on_event("startup")
async def _monitoring_startup() -> None:
    """Ensure admin monitoring tables exist before monitoring routes serve requests."""
    repo = await _get_monitoring_repo()
    await repo.ensure_schema_ready_once()


def _principal_actor_id(principal: AuthPrincipal) -> int | None:
    try:
        return int(principal.user_id) if principal.user_id is not None else None
    except (TypeError, ValueError):
        return None


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _decode_alert_metadata(row: dict[str, object]) -> dict[str, object]:
    """Return an alert row copy with runtime DB metadata normalized for AlertItem.

    TopicMonitoringDB alert rows expose metadata as SQLite JSON text, a decoded
    mapping, or an empty/null value depending on the source path. AlertItem
    accepts a mapping or None, so this helper preserves the original row and
    normalizes only the metadata field before response construction.
    """
    normalized = dict(row)
    meta = normalized.get("metadata")
    if isinstance(meta, str) and meta:
        try:
            parsed = json.loads(meta)
        except (TypeError, json.JSONDecodeError) as e:
            logger.debug(
                "monitoring: failed to parse alert metadata JSON: {}",
                e,
            )
            normalized["metadata"] = {"raw": meta}
        else:
            if parsed is None:
                normalized["metadata"] = None
            else:
                normalized["metadata"] = parsed if isinstance(parsed, dict) else {"value": parsed}
    elif meta == "":
        normalized["metadata"] = None
    return normalized


async def _build_alert_mutation_response(
    *,
    alert_id: int,
    db: TopicMonitoringDB,
    repo: AuthnzAdminMonitoringRepo,
) -> MarkReadResponse:
    """Build the mutation response with the authoritative merged alert state.

    The public read, acknowledge, and dismiss endpoints mutate overlay state,
    then return the runtime alert row merged with the matching admin overlay row
    so clients do not need to re-list alerts to see the updated state.
    """
    raw_alert = await asyncio.to_thread(db.get_alert, alert_id)
    if raw_alert is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")

    normalized_alert = _decode_alert_metadata(raw_alert)
    alert_identity = build_alert_identity(normalized_alert)
    overlay_rows = await repo.list_alert_states([alert_identity])
    merged_row = merge_runtime_alert_with_overlay(
        normalized_alert,
        overlay_rows[0] if overlay_rows else None,
    )
    return MarkReadResponse(
        status="ok",
        id=alert_id,
        item=AlertItem(**merged_row),
    )


async def _emit_admin_audit_event(
    request: Request,
    principal: AuthPrincipal,
    *,
    action: str,
    resource_id: str,
    metadata: dict[str, object],
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod

    await admin_mod._emit_admin_audit_event(
        request,
        principal,
        event_type="data.update",
        category="system",
        resource_type="monitoring_alert",
        resource_id=resource_id,
        action=action,
        metadata=metadata,
    )


@router.get(
    "/monitoring/watchlists",
    response_model=WatchlistListResponse,
    tags=["monitoring"],
    summary="List all watchlists",
)
async def list_watchlists() -> WatchlistListResponse:
    """List all configured topic monitoring watchlists."""
    watchlists = await asyncio.to_thread(
        lambda: get_topic_monitoring_service().list_watchlists()
    )
    return WatchlistListResponse(watchlists=watchlists)


@router.post(
    "/monitoring/watchlists",
    response_model=WatchlistUpsertResponse,
    tags=["monitoring"],
    summary="Create or update a watchlist",
)
async def upsert_watchlist(payload: Watchlist) -> WatchlistUpsertResponse:
    """Create a new watchlist or update an existing one."""
    try:
        wl = await asyncio.to_thread(
            lambda: get_topic_monitoring_service().upsert_watchlist(payload)
        )
        return WatchlistUpsertResponse(watchlist=wl, status="ok")
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except HTTPException:
        # Propagate existing HTTP errors without masking them
        raise
    except Exception as e:  # Generic 500 handler
        logger.error("Failed to upsert watchlist")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upsert watchlist",
        ) from e


@router.delete(
    "/monitoring/watchlists/{watchlist_id}",
    response_model=WatchlistDeleteResponse,
    tags=["monitoring"],
    summary="Delete a watchlist",
)
async def delete_watchlist(watchlist_id: str) -> WatchlistDeleteResponse:
    """Delete a watchlist by ID and return the deletion status."""
    ok = await asyncio.to_thread(
        lambda: get_topic_monitoring_service().delete_watchlist(watchlist_id)
    )
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Watchlist not found or failed to delete")
    return WatchlistDeleteResponse(status="deleted", id=watchlist_id)


@router.post(
    "/monitoring/reload",
    response_model=WatchlistsReloadResponse,
    tags=["monitoring"],
    summary="Reload watchlists from file",
)
async def reload_watchlists(
    delete_missing: bool = Query(False, description="Delete config-managed watchlists missing from file"),
    disable_missing: bool = Query(False, description="Disable config-managed watchlists missing from file"),
    include_unmanaged: bool = Query(False, description="Apply reload to non-config watchlists"),
) -> WatchlistsReloadResponse:
    """Reload watchlist definitions from the backing configuration source."""
    if delete_missing and disable_missing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="delete_missing and disable_missing are mutually exclusive",
        )
    await asyncio.to_thread(
        lambda: get_topic_monitoring_service().reload(
            delete_missing=delete_missing,
            disable_missing=disable_missing,
            include_unmanaged=include_unmanaged,
        )
    )
    return WatchlistsReloadResponse(status="ok")


@router.get(
    "/monitoring/alerts",
    response_model=AlertsListResponse,
    tags=["monitoring"],
    summary="List monitoring alerts",
)
async def list_alerts(
    user_id: str | None = Query(None, description="Filter by user id"),
    source: str | None = Query(None, description="Filter by source"),
    rule_category: str | None = Query(None, description="Filter by rule category"),
    rule_severity: str | None = Query(None, description="Filter by rule severity"),
    scope_type: str | None = Query(None, description="Filter by scope type"),
    scope_id: str | None = Query(None, description="Filter by scope id"),
    since: str | None = Query(None, description="ISO 8601 timestamp inclusive lower bound"),
    unread_only: bool = Query(False, description="Only unread alerts"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: TopicMonitoringDB = Depends(get_topic_monitoring_db),  # noqa: B008
) -> AlertsListResponse:
    """List monitoring alerts with optional filters and pagination."""

    repo = await _get_monitoring_repo()
    rows = await asyncio.to_thread(
        db.list_alerts,
        user_id=user_id,
        source=source,
        rule_category=rule_category,
        rule_severity=rule_severity,
        scope_type=scope_type,
        scope_id=scope_id,
        since_iso=since,
        unread_only=unread_only,
        limit=limit,
        offset=offset,
    )
    alert_identities = [build_alert_identity(row) for row in rows]
    overlay_rows = await repo.list_alert_states(alert_identities)
    overlay_by_identity = {
        str(row["alert_identity"]): row for row in overlay_rows if row.get("alert_identity")
    }
    items: list[AlertItem] = []
    for r in rows:
        normalized_row = _decode_alert_metadata(r)
        alert_identity = build_alert_identity(normalized_row)
        merged_row = merge_runtime_alert_with_overlay(
            normalized_row,
            overlay_by_identity.get(alert_identity),
        )
        items.append(AlertItem(**merged_row))
    return AlertsListResponse(
        items=items,
        total=None,
        pagination=build_offset_pagination_meta(
            limit=limit,
            offset=offset,
            total=None,
            count=len(items),
        ),
    )


@router.post(
    "/monitoring/alerts/{alert_id}/read",
    response_model=MarkReadResponse,
    tags=["monitoring"],
    summary="Mark an alert as read",
)
async def mark_alert_read(
    alert_id: int,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: TopicMonitoringDB = Depends(get_topic_monitoring_db),  # noqa: B008
) -> MarkReadResponse:
    """Mark a single alert as read by ID."""

    ok = await asyncio.to_thread(db.mark_read, alert_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")

    alert_identity = f"alert:{alert_id}"
    actor_id = _principal_actor_id(principal)
    read_at = _utcnow_iso()
    repo = await _get_monitoring_repo()
    await repo.append_alert_event(
        alert_identity=alert_identity,
        action="read",
        actor_user_id=actor_id,
        details_json=json.dumps({"alert_id": alert_id}),
        created_at=read_at,
    )
    await _emit_admin_audit_event(
        request,
        principal,
        action="monitoring.alert.read",
        resource_id=alert_identity,
        metadata={"alert_id": alert_id},
    )
    return await _build_alert_mutation_response(alert_id=alert_id, db=db, repo=repo)


@router.post(
    "/monitoring/alerts/{alert_id}/acknowledge",
    response_model=MarkReadResponse,
    tags=["monitoring"],
    summary="Acknowledge an alert",
)
async def acknowledge_alert(
    alert_id: int,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: TopicMonitoringDB = Depends(get_topic_monitoring_db),  # noqa: B008
) -> MarkReadResponse:
    """Acknowledge a monitoring alert using the authoritative overlay state."""

    ok = await asyncio.to_thread(db.mark_read, alert_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")

    alert_identity = f"alert:{alert_id}"
    actor_id = _principal_actor_id(principal)
    acknowledged_at = _utcnow_iso()
    repo = await _get_monitoring_repo()
    await repo.upsert_alert_state(
        alert_identity=alert_identity,
        acknowledged_at=acknowledged_at,
        updated_by_user_id=actor_id,
    )
    await repo.append_alert_event(
        alert_identity=alert_identity,
        action="acknowledged",
        actor_user_id=actor_id,
        details_json=json.dumps({"alert_id": alert_id}),
        created_at=acknowledged_at,
    )
    await _emit_admin_audit_event(
        request,
        principal,
        action="monitoring.alert.acknowledge",
        resource_id=alert_identity,
        metadata={"alert_id": alert_id},
    )
    return await _build_alert_mutation_response(alert_id=alert_id, db=db, repo=repo)


@router.delete(
    "/monitoring/alerts/{alert_id}",
    response_model=MarkReadResponse,
    tags=["monitoring"],
    summary="Dismiss an alert",
)
async def dismiss_alert(
    alert_id: int,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: TopicMonitoringDB = Depends(get_topic_monitoring_db),  # noqa: B008
) -> MarkReadResponse:
    """Dismiss a monitoring alert using the authoritative overlay state."""

    ok = await asyncio.to_thread(db.mark_read, alert_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")

    alert_identity = f"alert:{alert_id}"
    actor_id = _principal_actor_id(principal)
    dismissed_at = _utcnow_iso()
    repo = await _get_monitoring_repo()
    await repo.upsert_alert_state(
        alert_identity=alert_identity,
        dismissed_at=dismissed_at,
        updated_by_user_id=actor_id,
    )
    await repo.append_alert_event(
        alert_identity=alert_identity,
        action="dismissed",
        actor_user_id=actor_id,
        details_json=json.dumps({"alert_id": alert_id}),
        created_at=dismissed_at,
    )
    await _emit_admin_audit_event(
        request,
        principal,
        action="monitoring.alert.dismiss",
        resource_id=alert_identity,
        metadata={"alert_id": alert_id},
    )
    return await _build_alert_mutation_response(alert_id=alert_id, db=db, repo=repo)


@router.get(
    "/monitoring/notifications/settings",
    response_model=NotificationSettings,
    tags=["monitoring"],
    summary="Get notification settings",
)
async def get_notifications_settings() -> NotificationSettings:
    """Return the current in-memory notification settings."""

    svc = get_notification_service()
    return NotificationSettings(**svc.get_settings())


@router.put(
    "/monitoring/notifications/settings",
    response_model=NotificationSettings,
    tags=["monitoring"],
    summary="Update notification settings (runtime only)",
)
async def update_notifications_settings(
    payload: NotificationSettingsUpdate,
) -> NotificationSettings:
    """Update notification settings for the running process."""

    svc = get_notification_service()
    data = payload.model_dump(exclude_unset=True)
    if "file" in data:
        file_val = data.get("file")
        if file_val is None or not str(file_val).strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Notification file path must be non-empty",
            )
    updated = svc.update_settings(**data)
    return NotificationSettings(**updated)


@router.post(
    "/monitoring/notifications/test",
    response_model=NotificationTestResponse,
    tags=["monitoring"],
    summary="Send a test notification (critical by default)",
)
async def send_test_notification(payload: NotificationTestRequest) -> NotificationTestResponse:
    """Send a synthetic test notification using the current settings."""

    notifier = get_notification_service()
    alert = TopicAlert(
        user_id=payload.user_id or "admin",
        scope_type="user",
        scope_id=payload.user_id or "admin",
        source="monitoring.test",
        watchlist_id="test",
        rule_category="test",
        rule_severity=payload.severity,
        pattern="test",
        text_snippet=payload.message or "Test notification",
        metadata={"source": "admin_panel"},
    )
    try:
        # notifier.notify performs file I/O; offload to a thread to keep the event loop responsive.
        result = await asyncio.to_thread(notifier.notify, alert)
    except Exception as e:  # Generic 500 handler
        logger.exception("monitoring: failed to send test notification")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send test notification",
        ) from e
    status_value = result if isinstance(result, str) and result else "ok"
    return NotificationTestResponse(status=status_value)


def _tail_jsonl_file(path: str, limit: int) -> list[str]:
    """
    Return the last ``limit`` lines from a JSONL file without scanning the entire file.

    Reads from the end of the file in fixed-size blocks until enough newline-delimited
    records have been collected.
    """
    if limit <= 0:
        return []

    block_size = 4096
    lines: list[bytes] = []
    buffer = b""

    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        remaining = file_size

        while remaining > 0 and len(lines) < limit:
            read_size = block_size if remaining >= block_size else remaining
            remaining -= read_size
            f.seek(remaining)
            chunk = f.read(read_size)
            if not chunk:
                break

            buffer = chunk + buffer

            while True:
                newline_pos = buffer.rfind(b"\n")
                if newline_pos == -1:
                    break
                line = buffer[newline_pos + 1 :]
                buffer = buffer[:newline_pos]
                if line:
                    lines.append(line)
                if len(lines) >= limit:
                    break

        if buffer and len(lines) < limit:
            lines.append(buffer)

    # lines are collected from the end; reverse to restore chronological order
    decoded: list[str] = []
    for raw in reversed(lines[:limit]):
        decoded.append(raw.decode("utf-8", errors="replace").rstrip("\n"))
    return decoded


@router.get(
    "/monitoring/notifications/recent",
    response_model=RecentNotificationsResponse,
    tags=["monitoring"],
    summary="Tail recent notifications from file (JSONL)",
)
async def get_recent_notifications(
    limit: int = Query(50, ge=1, le=500),
) -> RecentNotificationsResponse:
    """Return a bounded tail of recent notification events from the JSONL log file."""

    svc = get_notification_service()
    path = svc.get_notification_file_path()
    items: list[dict] = []
    if not path or not os.path.exists(path):
        return RecentNotificationsResponse(items=[])
    try:
        # Offload blocking file I/O to a worker thread to keep the event loop responsive.
        lines = await asyncio.to_thread(_tail_jsonl_file, path, limit)
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            try:
                items.append(json.loads(ln))
            except (TypeError, json.JSONDecodeError) as e:
                logger.debug(
                    "monitoring: failed to parse recent notification JSONL line: {}",
                    e,
                )
                items.append({"raw": ln})
    except Exception as e:  # Generic 500 handler
        logger.error(
            "monitoring: failed to read recent notifications from {}: {}",
            path,
            e,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read recent notifications",
        ) from e
    return RecentNotificationsResponse(items=items)
