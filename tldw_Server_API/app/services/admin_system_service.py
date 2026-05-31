from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from fastapi import HTTPException
from loguru import logger

from tldw_Server_API.app.api.v1.utils.pagination import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    ActivitySummaryResponse,
    AuditLogResponse,
    ErrorBreakdownItem,
    ErrorBreakdownResponse,
    RateLimitPolicyHeadroom,
    RateLimitSummaryResponse,
    RateLimitThrottledEntity,
    SecurityAlertSinkStatus,
    SecurityAlertStatusResponse,
    SystemLogEntry,
    SystemLogsResponse,
    SystemStatsResponse,
)
from tldw_Server_API.app.core.AuthNZ.alerting import get_security_alert_dispatcher
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.rbac import get_effective_permissions
from tldw_Server_API.app.core.Logging.system_log_buffer import query_system_logs
from tldw_Server_API.app.core.Metrics import get_metrics_registry
from tldw_Server_API.app.core.testing import is_test_mode
from tldw_Server_API.app.services import admin_scope_service
from tldw_Server_API.app.services.admin_data_ops_service import (
    build_audit_log_csv as svc_build_audit_log_csv,
)
from tldw_Server_API.app.services.admin_data_ops_service import (
    build_audit_log_json as svc_build_audit_log_json,
)


def _is_db_pool_object(db: Any) -> bool:
    return isinstance(db, DatabasePool)


def _is_postgres_connection(db: Any) -> bool:
    """Resolve backend mode from connection/adapter shape without global probes."""
    if _is_db_pool_object(db):
        return getattr(db, "pool", None) is not None

    sqlite_hint = getattr(db, "_is_sqlite", None)
    if isinstance(sqlite_hint, bool):
        return not sqlite_hint

    if getattr(db, "_c", None) is not None:
        return False

    module_name = getattr(type(db), "__module__", "")
    if isinstance(module_name, str) and module_name.startswith("asyncpg"):
        return True

    return callable(getattr(db, "fetchrow", None))


def _parse_date_param(value: str | None, label: str, end_of_day: bool = False) -> datetime | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    raw = raw.replace("Z", "+00:00")
    try:
        if re.match(r"^\d{4}-\d{2}-\d{2}$", raw):
            dt = datetime.fromisoformat(raw)
            if end_of_day:
                dt = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
        else:
            dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid {label} date format") from None


async def get_security_alert_status() -> SecurityAlertStatusResponse:
    """Return configuration and last-known status for AuthNZ security alerts."""
    dispatcher = get_security_alert_dispatcher()
    status = dispatcher.get_status()

    sink_status_map: dict[str, bool | None] = status.get("last_sink_status", {})
    sink_error_map: dict[str, str | None] = status.get("last_sink_errors", {})
    sink_threshold_map: dict[str, str | None] = status.get("sink_thresholds", {})
    sink_backoff_map: dict[str, str | None] = status.get("sink_backoff_until", {})

    sink_rows = []
    for sink_name, configured in (
        ("file", status.get("file_sink_configured", False)),
        ("webhook", status.get("webhook_configured", False)),
        ("email", status.get("email_configured", False)),
    ):
        sink_rows.append(
            SecurityAlertSinkStatus(
                sink=sink_name,
                configured=bool(configured),
                min_severity=sink_threshold_map.get(sink_name),
                last_status=sink_status_map.get(sink_name),
                last_error=sink_error_map.get(sink_name),
                backoff_until=sink_backoff_map.get(sink_name),
            )
        )

    overall_health = "ok"
    if status.get("enabled", False):
        if status.get("last_validation_errors"):
            overall_health = "errors"
        else:
            configured_rows = [row for row in sink_rows if row.configured]
            if status.get("last_dispatch_success") is False or any(row.last_error for row in configured_rows) or configured_rows and all(row.last_status is False for row in configured_rows):
                overall_health = "degraded"

    return SecurityAlertStatusResponse(
        enabled=status.get("enabled", False),
        min_severity=status.get("min_severity", "high"),
        last_dispatch_time=status.get("last_dispatch_time"),
        last_dispatch_success=status.get("last_dispatch_success"),
        last_dispatch_error=status.get("last_dispatch_error"),
        dispatch_count=status.get("dispatch_count", 0),
        last_validation_time=status.get("last_validation_time"),
        validation_errors=status.get("last_validation_errors"),
        sinks=sink_rows,
        health=overall_health,
    )


def _empty_system_stats_response() -> SystemStatsResponse:
    return SystemStatsResponse(
        users={"total": 0, "active": 0, "verified": 0, "admins": 0, "new_last_30d": 0},
        storage={"total_used_mb": 0.0, "total_quota_mb": 0.0, "average_used_mb": 0.0, "max_used_mb": 0.0},
        sessions={"active": 0, "unique_users": 0},
    )


async def get_system_stats(db) -> SystemStatsResponse:
    """Get system statistics."""
    try:
        def _row_to_dict(row, keys: list[str]) -> dict:
            if row is None:
                return {}
            if isinstance(row, dict):
                return row
            if hasattr(row, "keys"):
                row_keys = list(row.keys())
                return {key: row[key] for key in row_keys}
            return {key: row[idx] if idx < len(row) else None for idx, key in enumerate(keys)}

        is_pg = _is_postgres_connection(db)
        if is_pg:
            # PostgreSQL
            user_stats = await db.fetchrow(
                """
                SELECT
                    COUNT(*) as total_users,
                    COUNT(*) FILTER (WHERE is_active = TRUE) as active_users,
                    COUNT(*) FILTER (WHERE is_verified = TRUE) as verified_users,
                    COUNT(*) FILTER (WHERE role = 'admin') as admin_users,
                    COUNT(*) FILTER (WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '30 days') as new_users_30d
                FROM users
                """
            )

            storage_stats = await db.fetchrow(
                """
                SELECT
                    SUM(storage_used_mb) as total_used_mb,
                    SUM(storage_quota_mb) as total_quota_mb,
                    AVG(storage_used_mb) as avg_used_mb,
                    MAX(storage_used_mb) as max_used_mb
                FROM users
                WHERE is_active = TRUE
                """
            )

            session_stats = await db.fetchrow(
                """
                SELECT
                    COUNT(*) as active_sessions,
                    COUNT(DISTINCT user_id) as unique_users
                FROM sessions
                WHERE is_active = TRUE AND expires_at > CURRENT_TIMESTAMP
                """
            )

        else:
            # SQLite
            cursor = await db.execute(
                """
                SELECT
                    COUNT(*) as total_users,
                    SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END) as active_users,
                    SUM(CASE WHEN is_verified = 1 THEN 1 ELSE 0 END) as verified_users,
                    SUM(CASE WHEN role = 'admin' THEN 1 ELSE 0 END) as admin_users,
                    SUM(CASE WHEN datetime(created_at) > datetime('now', '-30 days') THEN 1 ELSE 0 END) as new_users_30d
                FROM users
                """
            )
            user_stats = await cursor.fetchone()

            cursor = await db.execute(
                """
                SELECT
                    SUM(storage_used_mb) as total_used_mb,
                    SUM(storage_quota_mb) as total_quota_mb,
                    AVG(storage_used_mb) as avg_used_mb,
                    MAX(storage_used_mb) as max_used_mb
                FROM users
                WHERE is_active = 1
                """
            )
            storage_stats = await cursor.fetchone()

            cursor = await db.execute(
                """
                SELECT
                    COUNT(*) as active_sessions,
                    COUNT(DISTINCT user_id) as unique_users
                FROM sessions
                WHERE is_active = 1 AND datetime(expires_at) > datetime('now')
                """
            )
            session_stats = await cursor.fetchone()

        user_keys = ["total_users", "active_users", "verified_users", "admin_users", "new_users_30d"]
        storage_keys = ["total_used_mb", "total_quota_mb", "avg_used_mb", "max_used_mb"]
        session_keys = ["active_sessions", "unique_users"]
        us = _row_to_dict(user_stats, user_keys)
        ss = _row_to_dict(storage_stats, storage_keys)
        se = _row_to_dict(session_stats, session_keys)

        # Gather optional extended stats (ACP sessions, token usage)
        active_acp = None
        tokens_today = None
        try:
            from tldw_Server_API.app.services.admin_acp_sessions_service import get_acp_session_store
            store = await get_acp_session_store()
            _records, active_count = await store.list_sessions(status="active", limit=0, offset=0)
            active_acp = active_count
        except Exception as exc:
            logger.debug(f"Skipping ACP session stats in system overview: {exc}")

        try:
            if is_pg:
                token_row = await db.fetchrow(
                    """
                    SELECT
                        COALESCE(SUM(prompt_tokens), 0) as prompt,
                        COALESCE(SUM(completion_tokens), 0) as completion,
                        COALESCE(SUM(total_tokens), 0) as total
                    FROM llm_usage_v2
                    WHERE date(created_at) = CURRENT_DATE
                    """
                )
            else:
                cursor = await db.execute(
                    """
                    SELECT
                        COALESCE(SUM(prompt_tokens), 0) as prompt,
                        COALESCE(SUM(completion_tokens), 0) as completion,
                        COALESCE(SUM(total_tokens), 0) as total
                    FROM llm_usage_v2
                    WHERE date(created_at) = date('now')
                    """
                )
                token_row = await cursor.fetchone()
            if token_row:
                td = _row_to_dict(token_row, ["prompt", "completion", "total"])
                tokens_today = {
                    "prompt": int(td.get("prompt") or 0),
                    "completion": int(td.get("completion") or 0),
                    "total": int(td.get("total") or 0),
                }
        except Exception as exc:
            logger.debug(f"Skipping token usage stats in system overview: {exc}")

        return SystemStatsResponse(
            users={
                "total": int(us.get("total_users") or 0),
                "active": int(us.get("active_users") or 0),
                "verified": int(us.get("verified_users") or 0),
                "admins": int(us.get("admin_users") or 0),
                "new_last_30d": int(us.get("new_users_30d") or 0),
            },
            storage={
                "total_used_mb": float(ss.get("total_used_mb") or 0.0),
                "total_quota_mb": float(ss.get("total_quota_mb") or 0.0),
                "average_used_mb": float(ss.get("avg_used_mb") or 0.0),
                "max_used_mb": float(ss.get("max_used_mb") or 0.0),
            },
            sessions={
                "active": int(se.get("active_sessions") or 0),
                "unique_users": int(se.get("unique_users") or 0),
            },
            active_acp_sessions=active_acp,
            tokens_today=tokens_today,
            mcp_invocations_today=None,  # MCP metrics are Prometheus-only; defer until parsed
        )

    except Exception as exc:
        logger.error(f"Failed to get system stats: {exc}")
        try:
            if is_test_mode():
                return _empty_system_stats_response()
        except Exception as db_summary_error:
            logger.debug("Admin system service failed to resolve test-mode fallback", exc_info=db_summary_error)
        logger.warning("Returning empty system stats snapshot after backend query failure")
        return _empty_system_stats_response()


async def get_dashboard_activity(
    days: int,
    db,
    granularity: Literal["hour", "day", "auto"] = "auto",
) -> ActivitySummaryResponse:
    """Return recent request/user activity for the admin dashboard."""
    now_utc = datetime.now(timezone.utc)
    resolved_granularity: Literal["hour", "day"] = (
        "hour" if granularity == "auto" and days <= 1
        else "day" if granularity == "auto"
        else granularity
    )
    warnings: list[str] = []
    if resolved_granularity == "hour":
        end_hour = now_utc.replace(minute=0, second=0, microsecond=0)
        buckets = [
            end_hour - timedelta(hours=offset)
            for offset in range(23, -1, -1)
        ]
    else:
        end_day = now_utc.date()
        start_day = end_day - timedelta(days=days - 1)
        buckets = [
            datetime.combine(start_day + timedelta(days=offset), datetime.min.time(), tzinfo=timezone.utc)
            for offset in range(days)
        ]
    start_dt = buckets[0]
    activity_by_bucket = {
        bucket: {
            "date": bucket.date().isoformat(),
            "bucket_start": bucket.isoformat(),
            "requests": 0,
            "users": 0,
        }
        for bucket in buckets
    }

    def _bucket_for_datetime(timestamp: datetime) -> datetime:
        dt = timestamp.astimezone(timezone.utc)
        if resolved_granularity == "hour":
            return dt.replace(minute=0, second=0, microsecond=0)
        return datetime.combine(dt.date(), datetime.min.time(), tzinfo=timezone.utc)

    def _parse_row_bucket(value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
            return _bucket_for_datetime(dt)
        raw = str(value).strip()
        if not raw:
            return None
        if resolved_granularity == "hour" and "T" not in raw and " " in raw:
            raw = raw.replace(" ", "T")
        if raw.endswith("Z"):
            raw = raw.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return _bucket_for_datetime(parsed)

    try:
        registry = get_metrics_registry()
        request_values = registry.values.get("http_requests_total", [])
        for metric_value in list(request_values):
            try:
                metric_day = datetime.fromtimestamp(
                    metric_value.timestamp,
                    timezone.utc,
                ).date()
            except (OSError, OverflowError, ValueError):
                continue
            metric_dt = datetime.combine(metric_day, datetime.min.time(), tzinfo=timezone.utc)
            if resolved_granularity == "hour":
                try:
                    metric_dt = datetime.fromtimestamp(
                        metric_value.timestamp,
                        timezone.utc,
                    )
                except (OSError, OverflowError, ValueError):
                    continue
            bucket = _bucket_for_datetime(metric_dt)
            if bucket in activity_by_bucket:
                activity_by_bucket[bucket]["requests"] += int(metric_value.value or 0)
    except Exception as exc:
        logger.warning("Admin activity request metrics unavailable: {}", exc)
        warnings.append("request_metrics_unavailable")

    try:
        is_pg = _is_postgres_connection(db)
        if is_pg:
            if resolved_granularity == "hour":
                rows = await db.fetch(
                    """
                    SELECT date_trunc('hour', created_at AT TIME ZONE 'UTC') as bucket,
                           COUNT(DISTINCT user_id) as active_users
                    FROM sessions
                    WHERE created_at >= $1
                    GROUP BY bucket
                    ORDER BY bucket
                    """,
                    start_dt,
                )
            else:
                rows = await db.fetch(
                    """
                    SELECT DATE(created_at) as bucket,
                           COUNT(DISTINCT user_id) as active_users
                    FROM sessions
                    WHERE created_at >= $1
                    GROUP BY bucket
                    ORDER BY bucket
                    """,
                    start_dt,
                )
        else:
            if resolved_granularity == "hour":
                cursor = await db.execute(
                    """
                    SELECT strftime('%Y-%m-%dT%H:00:00', datetime(created_at)) as bucket,
                           COUNT(DISTINCT user_id) as active_users
                    FROM sessions
                    WHERE datetime(created_at) >= datetime(?)
                    GROUP BY bucket
                    ORDER BY bucket
                    """,
                    (start_dt.isoformat(),),
                )
            else:
                cursor = await db.execute(
                    """
                    SELECT date(created_at) as bucket,
                           COUNT(DISTINCT user_id) as active_users
                    FROM sessions
                    WHERE datetime(created_at) >= datetime(?)
                    GROUP BY bucket
                    ORDER BY bucket
                    """,
                    (start_dt.isoformat(),),
                )
            rows = await cursor.fetchall()
        for row in rows:
            if isinstance(row, dict):
                bucket_value = row.get("bucket")
                active_users = row.get("active_users")
            else:
                bucket_value = row[0] if len(row) > 0 else None
                active_users = row[1] if len(row) > 1 else None
            bucket = _parse_row_bucket(bucket_value)
            if bucket in activity_by_bucket:
                activity_by_bucket[bucket]["users"] = int(active_users or 0)
    except Exception as exc:
        logger.warning("Admin activity user metrics unavailable: {}", exc)
        warnings.append("user_metrics_unavailable")

    points = [activity_by_bucket[bucket] for bucket in buckets]
    return ActivitySummaryResponse(
        days=days,
        granularity=resolved_granularity,
        points=points,
        warnings=warnings or None,
    )


async def get_audit_log(
    *,
    user_id: int | None,
    action: str | None,
    resource: str | None,
    start: str | None,
    end: str | None,
    days: int,
    limit: int,
    offset: int,
    org_id: int | None,
    principal: AuthPrincipal,
    db,
) -> AuditLogResponse:
    try:
        is_pg = _is_postgres_connection(db)
        conditions: list[str] = []
        params: list[Any] = []
        param_count = 0

        start_dt = _parse_date_param(start, "start")
        end_dt = _parse_date_param(end, "end", end_of_day=True)
        if start_dt and end_dt and start_dt > end_dt:
            raise HTTPException(status_code=400, detail="Start date must be on or before end date")

        org_ids = await admin_scope_service.get_admin_org_ids(principal)
        if org_id is not None:
            org_ids = [org_id] if org_ids is None else [org_id] if org_id in org_ids else []
        if org_ids is not None and len(org_ids) == 0:
            return AuditLogResponse(
                entries=[],
                total=0,
                limit=limit,
                offset=offset,
                pagination=build_offset_pagination_meta(total=0, limit=limit, offset=offset, count=0),
            )

        if user_id:
            await admin_scope_service.enforce_admin_user_scope(principal, user_id, require_hierarchy=False)
            param_count += 1
            conditions.append(f"a.user_id = ${param_count}" if is_pg else "a.user_id = ?")
            params.append(user_id)

        if action:
            if action.endswith("*"):
                # Prefix match: e.g. "admin*" matches "admin.update", "admin.delete", etc.
                prefix = action[:-1]
                param_count += 1
                if is_pg:
                    conditions.append(f"a.action LIKE ${param_count}")
                else:
                    conditions.append("a.action LIKE ?")
                params.append(f"{prefix}%")
            else:
                param_count += 1
                conditions.append(f"a.action = ${param_count}" if is_pg else "a.action = ?")
                params.append(action)

        if resource:
            resource_filter = resource.strip()
            if resource_filter:
                if ":" in resource_filter:
                    resource_type, resource_id = resource_filter.split(":", 1)
                    resource_type = resource_type.strip()
                    resource_id = resource_id.strip()
                    if resource_type:
                        param_count += 1
                        conditions.append(f"a.resource_type = ${param_count}" if is_pg else "a.resource_type = ?")
                        params.append(resource_type)
                    if resource_id.isdigit():
                        param_count += 1
                        conditions.append(f"a.resource_id = ${param_count}" if is_pg else "a.resource_id = ?")
                        params.append(int(resource_id))
                else:
                    param_count += 1
                    if is_pg:
                        conditions.append(f"a.resource_type ILIKE ${param_count}")
                        params.append(f"%{resource_filter}%")
                    else:
                        conditions.append("LOWER(a.resource_type) LIKE ?")
                        params.append(f"%{resource_filter.lower()}%")

        if start_dt or end_dt:
            if start_dt:
                param_count += 1
                conditions.append(f"a.created_at >= ${param_count}" if is_pg else "datetime(a.created_at) >= datetime(?)")
                params.append(start_dt.isoformat())
            if end_dt:
                param_count += 1
                conditions.append(f"a.created_at <= ${param_count}" if is_pg else "datetime(a.created_at) <= datetime(?)")
                params.append(end_dt.isoformat())
        else:
            if is_pg:
                conditions.append(f"a.created_at > CURRENT_TIMESTAMP - INTERVAL '{days} days'")
            else:
                conditions.append("datetime(a.created_at) > datetime('now', ? || ' days')")
                params.append(f"-{days}")

        join_clause = ""
        if org_ids is not None:
            join_clause = " JOIN org_members om ON om.user_id = a.user_id"
            if is_pg:
                param_count += 1
                conditions.append(f"om.org_id = ANY(${param_count})")
                params.append(org_ids)
            else:
                placeholders = ",".join("?" for _ in org_ids)
                conditions.append(f"om.org_id IN ({placeholders})")
                params.extend(org_ids)

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

        def _format_resource(resource_type: str | None, resource_id: int | None) -> str | None:
            if not resource_type and resource_id is None:
                return None
            if resource_type and resource_id is not None:
                return f"{resource_type}:{resource_id}"
            if resource_type:
                return str(resource_type)
            return str(resource_id)

        if is_pg:
            limit_pos = param_count + 1
            offset_pos = param_count + 2
            count_query_template = """
                SELECT COUNT(*)
                FROM audit_logs a
                {join_clause}
                {where_clause}
            """
            count_query = count_query_template.format_map(locals())  # nosec B608
            total = await db.fetchval(count_query, *params)
            query_template = """
                SELECT a.id, a.user_id, u.username, a.action, a.resource_type, a.resource_id, a.details,
                       a.ip_address, a.created_at
                FROM audit_logs a
                LEFT JOIN users u ON a.user_id = u.id
                {join_clause}
                {where_clause}
                ORDER BY a.created_at DESC
                LIMIT ${limit_pos}
                OFFSET ${offset_pos}
            """
            query = query_template.format_map(locals())  # nosec B608
            query_params = list(params)
            query_params.append(limit)
            query_params.append(offset)
            rows = await db.fetch(query, *query_params)
        else:
            count_query_template = """
                SELECT COUNT(*)
                FROM audit_logs a
                {join_clause}
                {where_clause}
            """
            count_query = count_query_template.format_map(locals())  # nosec B608
            count_cursor = await db.execute(count_query, params)
            count_row = await count_cursor.fetchone()
            total = int(count_row[0]) if count_row and count_row[0] is not None else 0
            query_template = """
                SELECT a.id, a.user_id, u.username, a.action, a.resource_type, a.resource_id, a.details,
                       a.ip_address, a.created_at
                FROM audit_logs a
                LEFT JOIN users u ON a.user_id = u.id
                {join_clause}
                {where_clause}
                ORDER BY a.created_at DESC
                LIMIT ?
                OFFSET ?
            """
            query = query_template.format_map(locals())  # nosec B608
            query_params = list(params)
            query_params.append(limit)
            query_params.append(offset)
            cursor = await db.execute(query, query_params)
            rows = await cursor.fetchall()

        entries = []
        for row in rows:
            if isinstance(row, dict):
                resource_value = _format_resource(row.get("resource_type"), row.get("resource_id"))
                row["resource"] = resource_value
                entries.append(row)
            else:
                resource_value = _format_resource(row[4], row[5])
                entry = {
                    "id": row[0],
                    "user_id": row[1],
                    "username": row[2],
                    "action": row[3],
                    "resource": resource_value,
                    "details": row[6],
                    "ip_address": row[7],
                    "created_at": row[8],
                }
                entries.append(entry)

        total_count = int(total or 0)
        return AuditLogResponse(
            entries=entries,
            total=total_count,
            limit=limit,
            offset=offset,
            pagination=build_offset_pagination_meta(
                total=total_count,
                limit=limit,
                offset=offset,
                count=len(entries),
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to get audit log: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve audit log",
        ) from exc


async def export_audit_log(
    *,
    user_id: int | None,
    action: str | None,
    resource: str | None,
    start: str | None,
    end: str | None,
    days: int,
    limit: int,
    offset: int,
    org_id: int | None,
    format: str,
    principal: AuthPrincipal,
    db,
) -> tuple[str, str, str]:
    audit = await get_audit_log(
        user_id=user_id,
        action=action,
        resource=resource,
        start=start,
        end=end,
        days=days,
        limit=limit,
        offset=offset,
        org_id=org_id,
        principal=principal,
        db=db,
    )
    if format == "json":
        content = svc_build_audit_log_json(audit.entries, total=audit.total, limit=limit, offset=offset)
        return content, "application/json", "audit_log.json"
    content = svc_build_audit_log_csv(audit.entries)
    return content, "text/csv", "audit_log.csv"


async def list_system_logs(
    *,
    start: str | None,
    end: str | None,
    level: str | None,
    service: str | None,
    query: str | None,
    org_id: int | None,
    user_id: int | None,
    limit: int,
    offset: int,
    principal: AuthPrincipal,
) -> SystemLogsResponse:
    start_dt = _parse_date_param(start, "start")
    end_dt = _parse_date_param(end, "end", end_of_day=True)
    if start_dt and end_dt and start_dt > end_dt:
        raise HTTPException(status_code=400, detail="Start date must be on or before end date")

    org_ids = await admin_scope_service.get_admin_org_ids(principal)
    if org_id is not None:
        org_ids = [org_id] if org_ids is None else [org_id] if org_id in org_ids else []
    if org_ids is not None and len(org_ids) == 0:
        return SystemLogsResponse(
            items=[],
            total=0,
            limit=limit,
            offset=offset,
            pagination=build_offset_pagination_meta(total=0, limit=limit, offset=offset, count=0),
        )

    items, total = query_system_logs(
        start=start_dt,
        end=end_dt,
        level=level,
        service=service,
        query=query,
        org_id=org_id if org_ids is None else None,
        org_ids=org_ids,
        user_id=user_id,
        limit=limit,
        offset=offset,
    )
    return SystemLogsResponse(
        items=[SystemLogEntry(**item) for item in items],
        total=total,
        limit=limit,
        offset=offset,
        pagination=build_offset_pagination_meta(
            total=total,
            limit=limit,
            offset=offset,
            count=len(items),
        ),
    )


# ── Error breakdown aggregation (10.4) ──────────────────────────────────────

async def get_error_breakdown(
    *,
    principal: AuthPrincipal,
    db,
    hours: int = 24,
) -> ErrorBreakdownResponse:
    """Aggregate recent audit-log entries that look like errors (failed/denied/error actions).

    Groups by action (used as endpoint proxy) and a synthetic status_code derived
    from the action name. This is lightweight in-memory processing over the
    existing audit_log table.
    """
    try:
        is_pg = _is_postgres_connection(db)
        org_ids = await admin_scope_service.get_admin_org_ids(principal)

        # Build time constraint
        if is_pg:
            time_clause = f"a.created_at > CURRENT_TIMESTAMP - INTERVAL '{hours} hours'"
            time_params: list[Any] = []
        else:
            time_clause = "datetime(a.created_at) > datetime('now', ? || ' hours')"
            time_params = [f"-{hours}"]

        # Filter actions that indicate errors
        error_actions_clause = (
            "(LOWER(a.action) LIKE '%error%'"
            " OR LOWER(a.action) LIKE '%fail%'"
            " OR LOWER(a.action) LIKE '%denied%'"
            " OR LOWER(a.action) LIKE '%reject%'"
            " OR LOWER(a.action) LIKE '%unauthorized%')"
        )

        # Org scoping join
        join_clause = ""
        org_params: list[Any] = []
        org_condition = ""
        if org_ids is not None:
            if len(org_ids) == 0:
                return ErrorBreakdownResponse(items=[], total_errors=0, period=f"{hours}h")
            join_clause = " LEFT JOIN org_members om ON om.user_id = a.user_id"
            if is_pg:
                org_condition = f" AND om.org_id = ANY(${len(time_params) + 1})"
                org_params = [org_ids]
            else:
                placeholders = ",".join("?" for _ in org_ids)
                org_condition = f" AND om.org_id IN ({placeholders})"
                org_params = list(org_ids)

        # Clauses are server-defined fragments; org values remain bound parameters.
        query = (
            f"SELECT a.action, COUNT(*) as cnt, MAX(a.created_at) as last_at"  # nosec B608
            f" FROM audit_log a{join_clause}"  # nosec B608
            f" WHERE {time_clause} AND {error_actions_clause}{org_condition}"  # nosec B608
            f" GROUP BY a.action ORDER BY cnt DESC LIMIT 50"  # nosec B608
        )
        params = time_params + org_params

        if is_pg:
            rows = await db.fetch(query, *params)
        else:
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

        items: list[ErrorBreakdownItem] = []
        total_errors = 0
        for row in rows:
            if isinstance(row, dict):
                action = row.get("action", "unknown")
                count = int(row.get("cnt", 0))
                last_at = row.get("last_at")
            else:
                action = row[0] if len(row) > 0 else "unknown"
                count = int(row[1]) if len(row) > 1 else 0
                last_at = row[2] if len(row) > 2 else None

            # Derive a synthetic status code from the action name
            action_lower = str(action).lower()
            if "unauthorized" in action_lower or "denied" in action_lower:
                status_code = 403
            elif "not_found" in action_lower:
                status_code = 404
            elif "rate_limit" in action_lower or "throttl" in action_lower:
                status_code = 429
            elif "timeout" in action_lower:
                status_code = 504
            else:
                status_code = 500

            # Parse last_at
            last_occurred = None
            if last_at:
                try:
                    if isinstance(last_at, datetime):
                        last_occurred = last_at
                    else:
                        last_occurred = datetime.fromisoformat(str(last_at).replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass

            total_errors += count
            items.append(ErrorBreakdownItem(
                endpoint=str(action),
                status_code=status_code,
                count=count,
                last_occurred=last_occurred,
            ))

        return ErrorBreakdownResponse(
            items=items,
            total_errors=total_errors,
            period=f"{hours}h",
        )
    except Exception as exc:
        logger.warning("Error breakdown aggregation failed: {}", exc)
        return ErrorBreakdownResponse(items=[], total_errors=0, period=f"{hours}h")


# ── Rate limit summary aggregation (10.5) ───────────────────────────────────

async def get_rate_limit_summary(
    *,
    hours: int = 24,
) -> RateLimitSummaryResponse:
    """Aggregate rate-limit denial metrics from the metrics registry.

    Uses the in-memory Prometheus-style counters (rg_denials_total,
    rg_denials_by_entity_total) already maintained by the Resource Governor.
    """
    total_throttle_events = 0
    entity_rejections: dict[str, dict[str, Any]] = {}
    policy_stats: dict[str, dict[str, Any]] = {}

    try:
        reg = get_metrics_registry()
        # Try to get denial metrics from the registry
        for metric_name in ("rg_denials_total", "rg_denials_by_entity_total", "mcp_rate_limit_hits_total"):
            try:
                samples = reg.get_samples(metric_name)
            except (AttributeError, KeyError, TypeError):
                samples = []

            if not samples:
                continue

            for sample in samples:
                labels = {}
                value = 0
                if isinstance(sample, dict):
                    labels = sample.get("labels", {})
                    value = int(sample.get("value", 0))
                elif hasattr(sample, "labels"):
                    labels = getattr(sample, "labels", {})
                    value = int(getattr(sample, "value", 0))
                elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
                    labels = sample[0] if isinstance(sample[0], dict) else {}
                    value = int(sample[1]) if len(sample) > 1 else 0

                if value <= 0:
                    continue

                total_throttle_events += value

                # Track by entity
                entity = (
                    labels.get("entity")
                    or labels.get("scope", "unknown")
                )
                if entity not in entity_rejections:
                    entity_rejections[entity] = {"rejections": 0, "last_rejected_at": None}
                entity_rejections[entity]["rejections"] += value

                # Track by policy
                policy_id = labels.get("policy_id", "unknown")
                if policy_id not in policy_stats:
                    policy_stats[policy_id] = {
                        "resource_type": labels.get("category"),
                        "scope": labels.get("scope"),
                        "total_denials": 0,
                        "total_decisions": 0,
                    }
                policy_stats[policy_id]["total_denials"] += value

        # Also try to get total decisions for utilization calculation
        for metric_name in ("rg_decisions_total",):
            try:
                samples = reg.get_samples(metric_name)
            except (AttributeError, KeyError, TypeError):
                samples = []

            if not samples:
                continue

            for sample in samples:
                labels = {}
                value = 0
                if isinstance(sample, dict):
                    labels = sample.get("labels", {})
                    value = int(sample.get("value", 0))
                elif hasattr(sample, "labels"):
                    labels = getattr(sample, "labels", {})
                    value = int(getattr(sample, "value", 0))
                elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
                    labels = sample[0] if isinstance(sample[0], dict) else {}
                    value = int(sample[1]) if len(sample) > 1 else 0

                if value <= 0:
                    continue

                policy_id = labels.get("policy_id", "unknown")
                if policy_id in policy_stats:
                    policy_stats[policy_id]["total_decisions"] += value

    except Exception as exc:
        logger.warning("Rate limit summary metrics collection failed: {}", exc)

    # Build top throttled entities
    sorted_entities = sorted(entity_rejections.items(), key=lambda x: x[1]["rejections"], reverse=True)
    top_throttled = [
        RateLimitThrottledEntity(
            entity=entity,
            rejections=data["rejections"],
            last_rejected_at=data.get("last_rejected_at"),
        )
        for entity, data in sorted_entities[:20]
    ]

    # Build policy headroom
    headroom = []
    for pid, stats in sorted(policy_stats.items(), key=lambda x: x[1]["total_denials"], reverse=True):
        total_decisions = stats["total_decisions"]
        total_denials = stats["total_denials"]
        utilization = (total_denials / total_decisions * 100.0) if total_decisions > 0 else 0.0
        headroom.append(RateLimitPolicyHeadroom(
            policy_id=pid,
            resource_type=stats.get("resource_type"),
            scope=stats.get("scope"),
            total_decisions=total_decisions,
            total_denials=total_denials,
            utilization_pct=round(utilization, 2),
        ))

    return RateLimitSummaryResponse(
        total_throttle_events=total_throttle_events,
        period=f"{hours}h",
        top_throttled_entities=top_throttled,
        policy_headroom=headroom[:20],
    )


# ── Key age stats (dev) ────────────────────────────────────────────────────

async def get_key_age_stats(db) -> dict:
    """Get API key age distribution without per-user fan-out.

    Returns counts of keys in age buckets: 0-30d, 31-90d, 91-180d, 180d+.
    """
    try:
        is_pg = _is_postgres_connection(db)
        if is_pg:
            row = await db.fetchrow("""
                SELECT
                    COUNT(*) FILTER (WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '30 days') as age_0_30,
                    COUNT(*) FILTER (WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '90 days'
                                     AND created_at <= CURRENT_TIMESTAMP - INTERVAL '30 days') as age_31_90,
                    COUNT(*) FILTER (WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '180 days'
                                     AND created_at <= CURRENT_TIMESTAMP - INTERVAL '90 days') as age_91_180,
                    COUNT(*) FILTER (WHERE created_at <= CURRENT_TIMESTAMP - INTERVAL '180 days') as age_180_plus,
                    COUNT(*) as total
                FROM api_keys
                WHERE revoked_at IS NULL
            """)
        else:
            cursor = await db.execute("""
                SELECT
                    SUM(CASE WHEN datetime(created_at) > datetime('now', '-30 days') THEN 1 ELSE 0 END) as age_0_30,
                    SUM(CASE WHEN datetime(created_at) > datetime('now', '-90 days')
                              AND datetime(created_at) <= datetime('now', '-30 days') THEN 1 ELSE 0 END) as age_31_90,
                    SUM(CASE WHEN datetime(created_at) > datetime('now', '-180 days')
                              AND datetime(created_at) <= datetime('now', '-90 days') THEN 1 ELSE 0 END) as age_91_180,
                    SUM(CASE WHEN datetime(created_at) <= datetime('now', '-180 days') THEN 1 ELSE 0 END) as age_180_plus,
                    COUNT(*) as total
                FROM api_keys
                WHERE revoked_at IS NULL
            """)
            row = await cursor.fetchone()

        if row is None:
            return {"buckets": [], "total": 0}

        def _val(key: str) -> int:
            if isinstance(row, dict):
                return int(row.get(key) or 0)
            # tuple-like
            idx = {"age_0_30": 0, "age_31_90": 1, "age_91_180": 2, "age_180_plus": 3, "total": 4}
            return int(row[idx[key]] or 0) if key in idx else 0

        return {
            "buckets": [
                {"label": "0-30 days", "count": _val("age_0_30"), "color": "green"},
                {"label": "31-90 days", "count": _val("age_31_90"), "color": "green"},
                {"label": "91-180 days", "count": _val("age_91_180"), "color": "yellow"},
                {"label": "180+ days", "count": _val("age_180_plus"), "color": "red"},
            ],
            "total": _val("total"),
        }
    except Exception as exc:
        logger.warning(f"Failed to get key age stats: {exc}")
        return {"buckets": [], "total": 0}


async def debug_resolve_permissions(user_id: int, db) -> dict:
    """Resolve effective permissions for a user by querying roles + overrides."""
    try:
        is_pg = _is_postgres_connection(db)
        if is_pg:
            roles_row = await db.fetch(
                "SELECT r.name FROM user_roles ur JOIN roles r ON ur.role_id = r.id WHERE ur.user_id = $1",
                user_id,
            )
            perms_row = await db.fetch("""
                SELECT DISTINCT p.name
                FROM user_roles ur
                JOIN role_permissions rp ON ur.role_id = rp.role_id
                JOIN permissions p ON rp.permission_id = p.id
                WHERE ur.user_id = $1
            """, user_id)
        else:
            cursor = await db.execute(
                "SELECT r.name FROM user_roles ur JOIN roles r ON ur.role_id = r.id WHERE ur.user_id = ?",
                (user_id,),
            )
            roles_row = await cursor.fetchall()

        roles = [r["name"] if isinstance(r, dict) else r[0] for r in (roles_row or [])]
        loop = asyncio.get_running_loop()
        permissions = await loop.run_in_executor(None, get_effective_permissions, int(user_id))
        normalized_permissions = sorted({str(permission) for permission in permissions if str(permission).strip()})

        return {
            "user_id": user_id,
            "roles": roles,
            "effective_permissions": normalized_permissions,
            "permission_count": len(normalized_permissions),
        }
    except Exception as exc:
        logger.warning(f"Failed to resolve permissions for user {user_id}")
        return {
            "user_id": user_id,
            "roles": [],
            "effective_permissions": [],
            "error": "Failed to resolve permissions.",
        }


async def debug_decode_token(token: str) -> dict:
    """Decode a JWT token without claiming signature verification."""
    import base64
    import json

    try:
        # Split JWT into parts
        parts = token.split(".")
        if len(parts) != 3:
            return {"decoded": False, "signature_verified": False, "error": "Not a valid JWT format (expected 3 parts)"}

        # Decode header
        header_b64 = parts[0] + "=" * (4 - len(parts[0]) % 4)
        header = json.loads(base64.urlsafe_b64decode(header_b64))

        # Decode payload
        payload_b64 = parts[1] + "=" * (4 - len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))

        # Check expiration
        from datetime import datetime, timezone
        exp = payload.get("exp")
        is_expired = False
        if exp:
            exp_dt = datetime.fromtimestamp(exp, tz=timezone.utc)
            is_expired = exp_dt < datetime.now(timezone.utc)

        return {
            "decoded": True,
            "signature_verified": False,
            "header": header,
            "payload": {k: v for k, v in payload.items() if k not in ("password", "secret")},
            "expired": is_expired,
            "expires_at": datetime.fromtimestamp(exp, tz=timezone.utc).isoformat() if exp else None,
            "issuer": payload.get("iss"),
            "subject": payload.get("sub"),
        }
    except Exception as exc:
        return {"decoded": False, "signature_verified": False, "error": "Failed to decode token."}


async def debug_validate_token(token: str) -> dict:
    """Backward-compatible alias for token decoding debug logic."""
    return await debug_decode_token(token)


async def get_billing_analytics(db) -> dict:
    """Compute billing analytics: MRR, active subscribers, churn rate."""
    try:
        is_pg = _is_postgres_connection(db)

        # Count subscriptions by status
        if is_pg:
            row = await db.fetchrow("""
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE status = 'active') as active,
                    COUNT(*) FILTER (WHERE status = 'trialing') as trialing,
                    COUNT(*) FILTER (WHERE status = 'past_due') as past_due,
                    COUNT(*) FILTER (WHERE status = 'canceled') as canceled
                FROM subscriptions
            """)
        else:
            cursor = await db.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active,
                    SUM(CASE WHEN status = 'trialing' THEN 1 ELSE 0 END) as trialing,
                    SUM(CASE WHEN status = 'past_due' THEN 1 ELSE 0 END) as past_due,
                    SUM(CASE WHEN status = 'canceled' THEN 1 ELSE 0 END) as canceled
                FROM subscriptions
            """)
            row = await cursor.fetchone()

        if row is None:
            return {"analytics_available": False}

        def _v(key: str) -> int:
            if isinstance(row, dict):
                return int(row.get(key) or 0)
            idx = {"total": 0, "active": 1, "trialing": 2, "past_due": 3, "canceled": 4}
            return int(row[idx.get(key, 0)] or 0) if key in idx else 0

        active = _v("active")
        canceled = _v("canceled")
        total = _v("total")
        churn_rate = round(canceled / max(total, 1) * 100, 1)

        return {
            "analytics_available": True,
            "total_subscriptions": total,
            "active_subscriptions": active,
            "trialing": _v("trialing"),
            "past_due": _v("past_due"),
            "canceled": canceled,
            "churn_rate_pct": churn_rate,
            "trial_conversion_rate_pct": None,  # Would need historical data
            "mrr_cents": None,  # Would need plan price join
        }
    except Exception as exc:
        logger.warning(f"Failed to get billing analytics: {exc}")
        return {"analytics_available": False}


async def get_all_dependencies_health() -> dict:
    """Probe all external service dependencies and return combined health status.

    Calls the existing health endpoints for each service and returns a unified view.
    """
    import asyncio
    import time

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

    services = []

    # Database health
    async def _check_db() -> dict:
        start = time.monotonic()
        try:
            pool = await get_db_pool()
            if pool:
                services.append(True)  # Just mark as checked
            return {
                "name": "Database",
                "status": "healthy",
                "latency_ms": round((time.monotonic() - start) * 1000),
            }
        except Exception as exc:
            return {
                "name": "Database",
                "status": "down",
                "latency_ms": round((time.monotonic() - start) * 1000),
                "error": str(exc)[:200],
            }

    # ACP Session Store health
    async def _check_acp() -> dict:
        start = time.monotonic()
        try:
            from tldw_Server_API.app.services.admin_acp_sessions_service import get_acp_session_store
            store = await get_acp_session_store()
            _records, total = await store.list_sessions(limit=0, offset=0)
            return {
                "name": "ACP Sessions",
                "status": "healthy",
                "latency_ms": round((time.monotonic() - start) * 1000),
                "detail": f"{total} sessions",
            }
        except Exception as exc:
            return {
                "name": "ACP Sessions",
                "status": "down",
                "latency_ms": round((time.monotonic() - start) * 1000),
                "error": str(exc)[:200],
            }

    results = await asyncio.gather(
        _check_db(),
        _check_acp(),
        return_exceptions=True,
    )

    deps = []
    for r in results:
        if isinstance(r, dict):
            deps.append(r)
        elif isinstance(r, Exception):
            deps.append({"name": "Unknown", "status": "error", "error": str(r)[:200]})

    overall = "healthy" if all(d.get("status") == "healthy" for d in deps) else "degraded"
    from datetime import datetime, timezone
    return {"status": overall, "dependencies": deps, "checked_at": datetime.now(timezone.utc).isoformat()}
