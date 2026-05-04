from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import get_audit_service_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal, get_request_user, RequirePermission, User
from tldw_Server_API.app.core.Audit.unified_audit_service import (
    AuditEventCategory,
    AuditEventType,
    UnifiedAuditService,
)
from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_LOGS
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core import config as app_config
from tldw_Server_API.app.core.testing import is_truthy

router = APIRouter()

_DEFAULT_STREAM_AUTO_THRESHOLD = 5000
raw_stream_auto = None
try:
    raw_stream_auto = app_config.settings.get("AUDIT_EXPORT_STREAM_AUTO_MAX_ROWS", None)
    if raw_stream_auto is None:
        STREAM_AUTO_MAX_ROWS_THRESHOLD = _DEFAULT_STREAM_AUTO_THRESHOLD
    else:
        raw_str = str(raw_stream_auto).strip()
        STREAM_AUTO_MAX_ROWS_THRESHOLD = _DEFAULT_STREAM_AUTO_THRESHOLD if raw_str == "" else int(raw_str)
except Exception:
    logger.warning(
        "Invalid AUDIT_EXPORT_STREAM_AUTO_MAX_ROWS value {}; using default {}",
        raw_stream_auto,
        _DEFAULT_STREAM_AUTO_THRESHOLD,
    )
    STREAM_AUTO_MAX_ROWS_THRESHOLD = _DEFAULT_STREAM_AUTO_THRESHOLD


def _parse_dt(val: str | None, *, field_name: str) -> datetime | None:
    """Parse an optional ISO8601 timestamp.

    Raises:
        HTTPException: if a non-empty value is provided but cannot be parsed.
    """
    if val is None:
        return None
    s = str(val).strip()
    if s == "":
        return None
    try:
        # Handle trailing 'Z' (UTC) which fromisoformat doesn't accept directly
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {field_name}; expected ISO8601 timestamp (e.g. 2025-01-01T00:00:00Z)",
        ) from e
    return dt


def _coerce_bool(value: object | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return is_truthy(str(value).strip().lower())


def _shared_storage_enabled() -> bool:
    settings = app_config.settings
    if _coerce_bool(settings.get("AUDIT_STORAGE_ROLLBACK"), False):
        return False
    mode = str(settings.get("AUDIT_STORAGE_MODE", "per_user")).strip().lower()
    return mode == "shared"


def _principal_is_admin(principal: AuthPrincipal) -> bool:
    try:
        roles = {
            str(role).strip().lower()
            for role in (principal.roles or [])
            if str(role).strip()
        }
        permissions = {
            str(perm).strip().lower()
            for perm in (principal.permissions or [])
            if str(perm).strip()
        }
        if "admin" in roles:
            return True
        if "*" in permissions:
            return True
        if "system.configure" in permissions:
            return True
    except Exception:
        return False
    return False


def _resolve_request_user_id(principal: AuthPrincipal, current_user: User) -> str:
    principal_user_id = getattr(principal, "user_id", None)
    if principal_user_id is not None:
        return str(principal_user_id)
    return current_user.id_str or str(current_user.id)


def _map_event_types(values: list[str] | None | str | None) -> list[AuditEventType] | None:
    if not values:
        return None
    # Accept comma-separated string or list of strings
    if isinstance(values, str):
        raw_vals = [s.strip() for s in values.split(',') if s.strip()]
    else:
        raw_vals = [v for v in values if v]
    mapped: list[AuditEventType] = []
    invalid: list[str] = []
    for v in raw_vals:
        if not v:
            continue
        # Accept either enum name (AUTH_LOGIN_SUCCESS) or value (auth.login.success)
        try:
            mapped.append(AuditEventType[v])
            continue
        except Exception:
            logger.debug("Audit event type enum-key mapping failed")
        try:
            mapped.append(AuditEventType(v))
        except Exception:
            invalid.append(str(v))
            continue
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid event_type value(s): {', '.join(invalid)}",
        )
    return mapped or None


def _map_categories(values: list[str] | None | str | None) -> list[AuditEventCategory] | None:
    if not values:
        return None
    if isinstance(values, str):
        raw_vals = [s.strip() for s in values.split(',') if s.strip()]
    else:
        raw_vals = [v for v in values if v]
    mapped: list[AuditEventCategory] = []
    invalid: list[str] = []
    for v in raw_vals:
        if not v:
            continue
        try:
            mapped.append(AuditEventCategory[v])
            continue
        except Exception:
            logger.debug("Audit category enum-key mapping failed")
        try:
            mapped.append(AuditEventCategory(v))
        except Exception:
            invalid.append(str(v))
            continue
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category value(s): {', '.join(invalid)}",
        )
    return mapped or None


def _sanitize_filename(name: str, default_name: str) -> str:
    """Sanitize a user-supplied filename for Content-Disposition.

    - Drop directories
    - Allow only alphanumerics, dash, underscore, and dot
    - Fallback to default if empty
    """
    base = Path(name).name
    safe = []
    for ch in base:
        if ch.isalnum() or ch in {"-", "_", "."}:
            safe.append(ch)
        else:
            safe.append("_")
    sanitized = "".join(safe).strip("._")
    return sanitized or default_name


@router.get(
    "/audit/export",
    responses={
        200: {
            "description": "Audit export as JSON, NDJSON, or CSV.",
            "content": {
                "application/json": {},
                "application/x-ndjson": {},
                "text/csv": {},
            },
        },
    },
    summary="Export audit events (JSON/JSONL/CSV)",
    dependencies=[Depends(RequirePermission(SYSTEM_LOGS))],
)
async def export_audit_events(
    format: str = Query("json"),
    start_time: str | None = Query(None, description="ISO8601 start timestamp"),
    end_time: str | None = Query(None, description="ISO8601 end timestamp"),
    event_type: str | None = Query(None, description="Event types (enum name or value), comma-separated"),
    category: str | None = Query(None, description="Categories (enum name or value), comma-separated"),
    min_risk_score: int | None = Query(None, ge=0, le=100, description="Minimum risk score (0-100)"),
    user_id: str | None = Query(None),
    request_id: str | None = Query(None),
    correlation_id: str | None = Query(None),
    ip_address: str | None = Query(None, description="Filter by IP address"),
    session_id: str | None = Query(None, description="Filter by session id"),
    endpoint: str | None = Query(None, description="Filter by endpoint path"),
    method: str | None = Query(None, description="Filter by HTTP method"),
    max_rows: int | None = Query(None, ge=1, description="Hard maximum rows to export"),
    filename: str | None = Query(None),
    stream: bool = Query(False, description="Stream JSON/JSONL/CSV output incrementally"),
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    audit_service: UnifiedAuditService = Depends(get_audit_service_for_user),
):
    """Export audit events (JSON, JSONL, CSV). Requires SYSTEM_LOGS permission.

    Parameters (filters):
    - start_time, end_time: ISO8601; accepts trailing Z (UTC)
    - event_type, category: comma-separated enum names or values
    - min_risk_score: minimum risk score
    - user_id, request_id, correlation_id
    - ip_address, session_id, endpoint, method (context-based filters)
    - max_rows: hard cap to limit output size

    Output control:
    - format: json (default), jsonl (NDJSON), csv
    - stream: true enables streaming for JSON/JSONL/CSV
    - filename: suggested download name (sanitized; extension normalized)
    """
    fmt = format.lower()
    if fmt not in {"json", "csv", "jsonl"}:
        raise HTTPException(status_code=400, detail="format must be 'json', 'csv', or 'jsonl'")

    st = _parse_dt(start_time, field_name="start_time")
    et = _parse_dt(end_time, field_name="end_time")
    ets = _map_event_types(event_type)
    cats = _map_categories(category)

    streamable = {"json", "jsonl", "csv"}
    force_stream = bool(max_rows is not None and max_rows > STREAM_AUTO_MAX_ROWS_THRESHOLD)
    do_stream = bool((stream or force_stream) and fmt in streamable)

    user_id_filter = user_id
    allow_cross_tenant = False
    principal_is_admin = _principal_is_admin(principal)
    if principal_is_admin and _shared_storage_enabled():
        allow_cross_tenant = True
    elif not principal_is_admin:
        request_user_id = _resolve_request_user_id(principal, current_user)
        user_id_filter = request_user_id
        if user_id and str(user_id) != request_user_id:
            logger.warning("Ignoring audit export user_id override from non-admin user.")
    content = await audit_service.export_events(
        start_time=st,
        end_time=et,
        event_types=ets,
        categories=cats,
        user_id=user_id_filter,
        request_id=request_id,
        correlation_id=correlation_id,
        ip_address=ip_address,
        session_id=session_id,
        endpoint=endpoint,
        method=method,
        min_risk_score=min_risk_score,
        format=fmt,
        file_path=None,
        stream=do_stream,
        max_rows=max_rows,
        allow_cross_tenant=allow_cross_tenant,
    )

    if fmt == "json":
        media = "application/json"
        default_name = "audit_export.json"
    elif fmt == "jsonl":
        media = "application/x-ndjson"
        default_name = "audit_export.jsonl"
    else:
        media = "text/csv"
        default_name = "audit_export.csv"

    def _normalize_ext(name: str, expected_ext: str) -> str:
        # Preserve known compound extensions in the stem (e.g., archive.tar.gz -> archive.tar)
        name_l = name.lower()
        compound = [".tar.gz", ".tar.bz2", ".tar.xz", ".tar.Z"]
        for ce in compound:
            if name_l.endswith(ce):
                base = name[: -len(ce)]  # keep without compound ext
                stem = base
                break
        else:
            p = Path(name)
            stem = p.stem or name
        if not expected_ext.startswith("."):
            expected_ext = "." + expected_ext
        return f"{stem}{expected_ext}"

    fname = _sanitize_filename(filename, default_name) if filename else default_name
    # Normalize extension to match format to reduce client confusion
    if fmt == "json":
        fname = _normalize_ext(fname, "json")
    elif fmt == "jsonl":
        fname = _normalize_ext(fname, "jsonl")
    else:
        fname = _normalize_ext(fname, "csv")
    headers = {"Content-Disposition": f"attachment; filename={fname}"}

    if not do_stream and max_rows is None:
        default_limit = getattr(audit_service, "non_stream_max_rows", None)
        count_events_fn = getattr(audit_service, "count_events", None)
        if isinstance(default_limit, int) and default_limit > 0 and callable(count_events_fn):
            total_count = await count_events_fn(
                start_time=st,
                end_time=et,
                event_types=ets,
                categories=cats,
                user_id=user_id_filter,
                request_id=request_id,
                correlation_id=correlation_id,
                ip_address=ip_address,
                session_id=session_id,
                endpoint=endpoint,
                method=method,
                min_risk_score=min_risk_score,
                allow_cross_tenant=allow_cross_tenant,
            )
            if int(total_count) > default_limit:
                headers["X-Audit-Export-Truncated"] = "true"
                headers["X-Audit-Export-Row-Limit"] = str(default_limit)
                headers["X-Audit-Export-Total-Count"] = str(int(total_count))

    if do_stream:
        # content is an async generator from export_events
        return StreamingResponse(content=content, media_type=media, headers=headers)
    return Response(content=content, media_type=media, headers=headers)


@router.get(
    "/audit/count",
    summary="Count audit events for pagination",
    dependencies=[Depends(RequirePermission(SYSTEM_LOGS))],
)
async def count_audit_events(
    start_time: str | None = Query(None, description="ISO8601 start timestamp"),
    end_time: str | None = Query(None, description="ISO8601 end timestamp"),
    event_type: str | None = Query(None, description="Event types (enum name or value), comma-separated"),
    category: str | None = Query(None, description="Categories (enum name or value), comma-separated"),
    min_risk_score: int | None = Query(None, ge=0, le=100, description="Minimum risk score (0-100)"),
    user_id: str | None = Query(None),
    request_id: str | None = Query(None),
    correlation_id: str | None = Query(None),
    ip_address: str | None = Query(None, description="Filter by IP address"),
    session_id: str | None = Query(None, description="Filter by session id"),
    endpoint: str | None = Query(None, description="Filter by endpoint path"),
    method: str | None = Query(None, description="Filter by HTTP method"),
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    audit_service: UnifiedAuditService = Depends(get_audit_service_for_user),
):
    """Count audit events for pagination UIs. Requires SYSTEM_LOGS permission.

    Accepts the same filters as the export endpoint (except format/stream/filename).
    Returns a JSON object: {"count": <int>}.
    """
    st = _parse_dt(start_time, field_name="start_time")
    et = _parse_dt(end_time, field_name="end_time")
    ets = _map_event_types(event_type)
    cats = _map_categories(category)

    user_id_filter = user_id
    allow_cross_tenant = False
    principal_is_admin = _principal_is_admin(principal)
    if principal_is_admin and _shared_storage_enabled():
        allow_cross_tenant = True
    elif not principal_is_admin:
        request_user_id = _resolve_request_user_id(principal, current_user)
        user_id_filter = request_user_id
        if user_id and str(user_id) != request_user_id:
            logger.warning("Ignoring audit count user_id override from non-admin user.")
    count = await audit_service.count_events(
        start_time=st,
        end_time=et,
        event_types=ets,
        categories=cats,
        user_id=user_id_filter,
        request_id=request_id,
        correlation_id=correlation_id,
        ip_address=ip_address,
        session_id=session_id,
        endpoint=endpoint,
        method=method,
        min_risk_score=min_risk_score,
        allow_cross_tenant=allow_cross_tenant,
    )
    return {"count": int(count)}
