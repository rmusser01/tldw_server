"""
Audit shared DB migration utility.

Merges per-user audit databases and the legacy Databases/unified_audit.db
into the shared audit database, deduplicating by event_id.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiosqlite
from loguru import logger

from tldw_Server_API.app.core.Audit.unified_audit_service import (
    HIGH_RISK_SCORE,
    UnifiedAuditService,
)
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.sqlite_policy import (
    configure_sqlite_connection_async,
)
from tldw_Server_API.app.core.testing import is_truthy
from tldw_Server_API.app.core.Utils.Utils import get_project_root

_UNIDENTIFIED_TENANT_ID = "unidentified_user"

_AUDIT_COERCE_EXCEPTIONS = (
    TypeError,
    ValueError,
    AttributeError,
)

_AUDIT_JSON_EXCEPTIONS = (
    TypeError,
    ValueError,
    OverflowError,
)

_AUDIT_DB_EXCEPTIONS = (
    aiosqlite.Error,
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
)

_AUDIT_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
)


@dataclass(frozen=True)
class AuditMigrationSource:
    path: Path
    tenant_id: str | None
    label: str


@dataclass
class AuditMigrationCounts:
    source: AuditMigrationSource
    events_read: int = 0
    events_inserted: int = 0
    events_skipped: int = 0
    # stats_* are event-level counters for audit_daily_stats derivation.
    # stats_read = source events considered for stats, stats_inserted =
    # events that contributed to a daily stats bucket, stats_skipped =
    # events that could not contribute (duplicate event IDs or invalid timestamp).
    stats_read: int = 0
    stats_inserted: int = 0
    stats_skipped: int = 0
    failed: bool = False
    error: str | None = None


@dataclass(frozen=True)
class AuditStatsUpdateResult:
    events_contributed: int = 0
    events_skipped: int = 0
    buckets_updated: int = 0


@dataclass
class AuditMigrationReport:
    shared_db_path: Path
    sources: list[AuditMigrationCounts]

    @property
    def total_events_inserted(self) -> int:
        return sum(c.events_inserted for c in self.sources)

    @property
    def total_events_skipped(self) -> int:
        return sum(c.events_skipped for c in self.sources)

    @property
    def total_stats_read(self) -> int:
        return sum(c.stats_read for c in self.sources)

    @property
    def total_stats_inserted(self) -> int:
        return sum(c.stats_inserted for c in self.sources)

    @property
    def total_stats_skipped(self) -> int:
        return sum(c.stats_skipped for c in self.sources)

    @property
    def failed_sources(self) -> list[AuditMigrationCounts]:
        return [c for c in self.sources if c.failed]

    @property
    def total_failures(self) -> int:
        return len(self.failed_sources)


def _normalize_subpath(raw: str | None) -> Path | None:
    if raw is None:
        return None
    try:
        value = str(raw).strip()
    except _AUDIT_COERCE_EXCEPTIONS:
        return None
    if not value:
        return None
    cleaned = value.lstrip("/\\")
    if not cleaned:
        return None
    path = Path(cleaned)
    if any(part in {"..", "."} for part in path.parts):
        return None
    return path


def discover_audit_sources(
    *,
    user_db_base_dir: Path | None = None,
    default_db_path: Path | None = None,
) -> list[AuditMigrationSource]:
    base_dir = user_db_base_dir or DatabasePaths.get_user_db_base_dir()
    sources: list[AuditMigrationSource] = []
    seen_paths: set[Path] = set()

    def _add_source(path: Path, tenant_id: str | None, label: str) -> None:
        resolved = path.resolve()
        if resolved in seen_paths:
            return
        seen_paths.add(resolved)
        sources.append(
            AuditMigrationSource(
                path=resolved,
                tenant_id=tenant_id,
                label=label,
            )
        )

    def _scan_base(root: Path) -> None:
        try:
            for entry in sorted(root.iterdir()):
                if not entry.is_dir():
                    continue
                audit_path = entry / DatabasePaths.AUDIT_SUBDIR / DatabasePaths.AUDIT_DB_NAME
                if audit_path.exists():
                    _add_source(audit_path, entry.name, f"user:{entry.name}")
        except FileNotFoundError:
            return

    _scan_base(base_dir)

    subpath_raw = settings.get("AUDIT_ETL_USER_SUBPATH")
    subpath = _normalize_subpath(subpath_raw)
    if subpath is not None:
        _scan_base(base_dir / subpath)

    if default_db_path is not None and default_db_path.exists():
        _add_source(default_db_path, None, "default")

    return sources


async def _ensure_checkpoint_table(db: aiosqlite.Connection) -> None:
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_migration_checkpoints (
            source_path TEXT PRIMARY KEY,
            last_rowid INTEGER,
            last_event_id TEXT,
            last_timestamp TEXT,
            updated_at TEXT
        )
        """
    )
    # Back-compat: add last_timestamp if it doesn't exist yet.
    try:
        await db.execute(
            "ALTER TABLE audit_migration_checkpoints ADD COLUMN last_timestamp TEXT"
        )
    except Exception as exc:
        msg = str(exc).lower()
        if "duplicate column" in msg:
            logger.debug("audit_migration_checkpoints already has last_timestamp column")
        else:
            logger.exception("Failed adding last_timestamp column: {}", exc)
            raise


async def _load_checkpoint(
    db: aiosqlite.Connection, source_path: Path
) -> tuple[int, str | None, str | None]:
    try:
        async with db.execute(
            "SELECT last_rowid, last_event_id, last_timestamp FROM audit_migration_checkpoints WHERE source_path = ?",
            (str(source_path),),
        ) as cur:
            row = await cur.fetchone()
    except Exception as exc:
        if "no such column" not in str(exc).lower():
            raise
        # Older schema without last_timestamp
        async with db.execute(
            "SELECT last_rowid, last_event_id FROM audit_migration_checkpoints WHERE source_path = ?",
            (str(source_path),),
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return 0, None, None
        last_rowid = int(row[0]) if row[0] is not None else 0
        last_event_id = str(row[1]) if row[1] else None
        return last_rowid, last_event_id, None
    if not row:
        return 0, None, None
    last_rowid = int(row[0]) if row[0] is not None else 0
    last_event_id = str(row[1]) if row[1] else None
    last_timestamp = str(row[2]) if row[2] else None
    return last_rowid, last_event_id, last_timestamp


async def _save_checkpoint(
    db: aiosqlite.Connection,
    source_path: Path,
    last_rowid: int,
    last_event_id: str | None,
    last_timestamp: str | None,
) -> None:
    await db.execute(
        """
        INSERT INTO audit_migration_checkpoints (source_path, last_rowid, last_event_id, last_timestamp, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(source_path) DO UPDATE SET
            last_rowid=excluded.last_rowid,
            last_event_id=excluded.last_event_id,
            last_timestamp=excluded.last_timestamp,
            updated_at=excluded.updated_at
        """,
        (
            str(source_path),
            last_rowid,
            last_event_id,
            last_timestamp,
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def _normalize_tenant_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(value).strip()
    except _AUDIT_COERCE_EXCEPTIONS:
        return ""


def _is_system_event(event_type: Any, category: Any) -> bool:
    if category is not None:
        try:
            cat_val = str(category).lower()
        except _AUDIT_COERCE_EXCEPTIONS:
            cat_val = ""
        if cat_val == "system":
            return True
    if event_type is not None:
        try:
            event_val = str(event_type).lower()
        except _AUDIT_COERCE_EXCEPTIONS:
            event_val = ""
        if event_val.startswith("system"):
            return True
    return False


def _resolve_tenant_id(
    *,
    tenant_override: str | None,
    raw_tenant: Any,
    context_user_id: Any,
    event_type: Any,
    category: Any,
    system_tenant_id: str,
    unidentified_tenant_id: str,
) -> str:
    if _is_system_event(event_type, category):
        return system_tenant_id

    candidate = tenant_override
    if candidate is None or str(candidate).strip() == "":
        candidate = raw_tenant
    if candidate is None or str(candidate).strip() == "":
        candidate = context_user_id

    normalized = _normalize_tenant_value(candidate)
    if not normalized:
        return unidentified_tenant_id

    lowered = normalized.lower()
    if lowered == system_tenant_id:
        return system_tenant_id
    if lowered == unidentified_tenant_id:
        return unidentified_tenant_id
    if not normalized.isdigit():
        logger.warning(
            "Audit migration: non-numeric tenant id {} preserved as-is",
            normalized,
        )
        return normalized
    return normalized


def _coerce_timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc).isoformat()
        return value.astimezone(timezone.utc).isoformat()
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    try:
        s = str(value).strip()
        if not s:
            return datetime.now(timezone.utc).isoformat()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt_val = datetime.fromisoformat(s)
        if dt_val.tzinfo is None:
            dt_val = dt_val.replace(tzinfo=timezone.utc)
        return dt_val.astimezone(timezone.utc).isoformat()
    except _AUDIT_COERCE_EXCEPTIONS:
        return str(value)


def _checkpoint_timestamp(value: Any, previous: str | None) -> str | None:
    """Return a safe checkpoint timestamp while preserving source text ordering."""
    try:
        if value is None:
            return previous
        s = str(value).strip()
        if not s:
            return previous
        # Preserve source-side timestamp text so resume comparisons match
        # ORDER BY timestamp,event_id semantics in the source database.
        return s
    except _AUDIT_COERCE_EXCEPTIONS:
        return previous


def _parse_timestamp_to_date(value: Any) -> date | None:
    try:
        if isinstance(value, datetime):
            dt_val = value
        elif value is None:
            return None
        else:
            s = str(value).strip()
            if not s:
                return None
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt_val = datetime.fromisoformat(s)
        if dt_val.tzinfo is None:
            dt_val = dt_val.replace(tzinfo=timezone.utc)
        return dt_val.astimezone(timezone.utc).date()
    except (TypeError, ValueError):
        return None


def _infer_category(event_type: str | None) -> str:
    if not event_type:
        return "system"
    val = str(event_type).strip().lower()
    if val.startswith("auth"):
        return "authentication"
    if val.startswith("user"):
        return "authorization"
    if val.startswith("data"):
        if any(val.endswith(suffix) for suffix in ("write", "update", "delete", "import", "export")):
            return "data_modification"
        return "data_access"
    if val.startswith("rag"):
        return "rag"
    if val.startswith("eval"):
        return "evaluation"
    if val.startswith("api"):
        return "api_call"
    if val.startswith("security"):
        return "security"
    if val.startswith("system"):
        return "system"
    return "system"


def _normalize_severity(value: Any) -> str:
    if value is None:
        return "info"
    val = str(value).strip().lower()
    if val in {"debug", "info", "warning", "error", "critical"}:
        return val
    mapping = {
        "low": "info",
        "medium": "warning",
        "high": "error",
        "critical": "critical",
    }
    return mapping.get(val, "info")


def _safe_json_text(value: Any, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except _AUDIT_JSON_EXCEPTIONS:
        return str(value)


def _pick(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def _build_event_record(
    row: dict[str, Any],
    columns: list[str],
    *,
    tenant_override: str | None,
    system_tenant_id: str,
    unidentified_tenant_id: str,
) -> dict[str, Any]:
    event_type_val = _pick(row, "event_type", "event") or "system.start"
    record: dict[str, Any] = dict.fromkeys(columns)

    record["event_id"] = str(_pick(row, "event_id") or uuid4())
    record["timestamp"] = _coerce_timestamp(_pick(row, "timestamp"))
    record["category"] = _pick(row, "category") or _infer_category(event_type_val)
    record["event_type"] = str(event_type_val)
    record["severity"] = _normalize_severity(_pick(row, "severity"))

    context_user_id = _pick(row, "context_user_id", "user_id")
    if "tenant_user_id" in columns:
        record["tenant_user_id"] = _resolve_tenant_id(
            tenant_override=tenant_override,
            raw_tenant=_pick(row, "tenant_user_id"),
            context_user_id=context_user_id,
            event_type=event_type_val,
            category=record["category"],
            system_tenant_id=system_tenant_id,
            unidentified_tenant_id=unidentified_tenant_id,
        )

    record["context_request_id"] = _pick(row, "context_request_id", "request_id")
    record["context_correlation_id"] = _pick(row, "context_correlation_id", "correlation_id")
    record["context_session_id"] = _pick(row, "context_session_id", "session_id")
    record["context_user_id"] = context_user_id
    record["context_api_key_hash"] = _pick(row, "context_api_key_hash", "api_key_hash")
    record["context_ip_address"] = _pick(row, "context_ip_address", "ip_address")
    record["context_user_agent"] = _pick(row, "context_user_agent", "user_agent")
    record["context_endpoint"] = _pick(row, "context_endpoint", "endpoint")
    record["context_method"] = _pick(row, "context_method", "method")

    record["resource_type"] = _pick(row, "resource_type")
    record["resource_id"] = _pick(row, "resource_id")
    record["action"] = _pick(row, "action")
    record["result"] = _pick(row, "result", "outcome") or "success"
    record["error_message"] = _pick(row, "error_message", "details")

    record["duration_ms"] = _pick(row, "duration_ms")
    record["tokens_used"] = _pick(row, "tokens_used")
    record["estimated_cost"] = _pick(row, "estimated_cost")
    record["result_count"] = _pick(row, "result_count")

    record["risk_score"] = _pick(row, "risk_score") or 0
    record["pii_detected"] = _safe_bool(_pick(row, "pii_detected"), False)
    record["compliance_flags"] = _safe_json_text(_pick(row, "compliance_flags"), "[]")
    record["metadata"] = _safe_json_text(_pick(row, "metadata"), "{}")

    return record


async def _fetch_existing_event_ids(
    db: aiosqlite.Connection,
    event_ids: list[str],
    *,
    chunk_size: int = 500,
) -> set[str]:
    if not event_ids:
        return set()
    existing: set[str] = set()
    for i in range(0, len(event_ids), chunk_size):
        chunk = event_ids[i : i + chunk_size]
        placeholders = ",".join("?" * len(chunk))
        event_ids_clause = f"({placeholders})"
        query_template = "SELECT event_id FROM audit_events WHERE event_id IN {event_ids_clause}"
        query = query_template.format_map(locals())  # nosec B608
        async with db.execute(query, chunk) as cursor:
            rows = await cursor.fetchall()
            existing.update(str(row[0]) for row in rows if row and row[0])
    return existing


async def _table_exists(db: aiosqlite.Connection, name: str) -> bool:
    async with db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ) as cur:
        return await cur.fetchone() is not None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or isinstance(value, bool):
            return default
        if isinstance(value, (int, float)):
            return int(value)
        text = str(value).strip()
        if not text:
            return default
        return int(float(text))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or isinstance(value, bool):
            return default
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if is_truthy(normalized):
        return True
    if normalized in {"0", "false", "no", "off", "n", ""}:
        return False
    return default


def _normalize_result(value: Any) -> str:
    """Normalize result strings for stats; missing/invalid values become empty string."""
    try:
        if value is None:
            return ""
        return str(value).strip().lower()
    except _AUDIT_COERCE_EXCEPTIONS:
        return ""


async def _update_shared_daily_stats_from_records(
    db: aiosqlite.Connection,
    records: list[dict[str, Any]],
    *,
    unidentified_tenant_id: str,
) -> AuditStatsUpdateResult:
    if not records:
        return AuditStatsUpdateResult()
    stats = defaultdict(lambda: {
        "total": 0,
        "high_risk": 0,
        "failed": 0,
        "cost": 0.0,
        "tokens": 0,
        "durations": [],
    })
    stats_events_contributed = 0
    stats_events_skipped = 0

    for record in records:
        tenant_id = record.get("tenant_user_id") or unidentified_tenant_id
        date_val = _parse_timestamp_to_date(record.get("timestamp"))
        if date_val is None:
            stats_events_skipped += 1
            continue
        stats_events_contributed += 1
        category = str(record.get("category") or "system")
        key = (tenant_id, date_val, category)
        stats[key]["total"] += 1
        if _safe_int(record.get("risk_score"), 0) >= HIGH_RISK_SCORE:
            stats[key]["high_risk"] += 1
        if _normalize_result(record.get("result")) in {"failure", "error"}:
            stats[key]["failed"] += 1
        stats[key]["cost"] += _safe_float(record.get("estimated_cost"), 0.0)
        stats[key]["tokens"] += _safe_int(record.get("tokens_used"), 0)
        if record.get("duration_ms") is not None:
            stats[key]["durations"].append(_safe_float(record.get("duration_ms"), 0.0))

    updated_rows = 0
    for key, data in stats.items():
        tenant_id, date_val, category = key
        duration_count = len(data["durations"])
        avg_duration = (
            sum(data["durations"]) / duration_count
            if duration_count > 0 else None
        )
        await db.execute(
            """
            INSERT INTO audit_daily_stats (
                tenant_user_id, date, category, total_events, high_risk_events,
                failed_events, total_cost, total_tokens, avg_duration_ms,
                duration_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(tenant_user_id, date, category) DO UPDATE SET
                total_events = total_events + excluded.total_events,
                high_risk_events = high_risk_events + excluded.high_risk_events,
                failed_events = failed_events + excluded.failed_events,
                total_cost = total_cost + excluded.total_cost,
                total_tokens = total_tokens + excluded.total_tokens,
                duration_count = COALESCE(duration_count, 0) + COALESCE(excluded.duration_count, 0),
                avg_duration_ms = CASE
                    WHEN COALESCE(duration_count, 0) + COALESCE(excluded.duration_count, 0) = 0 THEN NULL
                    WHEN COALESCE(duration_count, 0) = 0 THEN excluded.avg_duration_ms
                    WHEN COALESCE(excluded.duration_count, 0) = 0 THEN avg_duration_ms
                    ELSE (
                        COALESCE(avg_duration_ms, 0) * COALESCE(duration_count, 0) +
                        COALESCE(excluded.avg_duration_ms, 0) * COALESCE(excluded.duration_count, 0)
                    ) / (COALESCE(duration_count, 0) + COALESCE(excluded.duration_count, 0))
                END
            """,
            (
                tenant_id,
                date_val,
                category,
                data["total"],
                data["high_risk"],
                data["failed"],
                data["cost"],
                data["tokens"],
                avg_duration,
                duration_count,
            ),
        )
        updated_rows += 1
    return AuditStatsUpdateResult(
        events_contributed=stats_events_contributed,
        events_skipped=stats_events_skipped,
        buckets_updated=updated_rows,
    )


async def _migrate_source(
    shared_db: aiosqlite.Connection,
    source: AuditMigrationSource,
    *,
    columns: list[str],
    insert_sql: str,
    system_tenant_id: str,
    unidentified_tenant_id: str,
    chunk_size: int,
) -> AuditMigrationCounts:
    counts = AuditMigrationCounts(source=source)
    if not source.path.exists():
        return counts
    source_key = source.path.resolve()
    try:
        async with aiosqlite.connect(source.path) as source_db:
            source_db.row_factory = aiosqlite.Row
            if not await _table_exists(source_db, "audit_events"):
                return counts

            last_rowid, last_event_id, last_timestamp = await _load_checkpoint(shared_db, source_key)
            # Existing checkpoints may contain normalized UTC timestamps from older
            # builds. Refresh from the source event_id to keep resume comparisons
            # aligned with source timestamp text.
            if last_timestamp is not None and last_event_id:
                try:
                    async with source_db.execute(
                        "SELECT timestamp FROM audit_events WHERE event_id = ? LIMIT 1",
                        (last_event_id,),
                    ) as cur:
                        checkpoint_row = await cur.fetchone()
                    if checkpoint_row:
                        refreshed = _checkpoint_timestamp(checkpoint_row["timestamp"], last_timestamp)
                        if refreshed is not None:
                            last_timestamp = refreshed
                except _AUDIT_DB_EXCEPTIONS:
                    pass
            if last_timestamp is None and last_rowid > 0:
                # Attempt to upgrade legacy rowid checkpoint to timestamp+event_id.
                async with source_db.execute(
                    "SELECT timestamp, event_id FROM audit_events WHERE rowid = ?",
                    (last_rowid,),
                ) as cur:
                    row = await cur.fetchone()
                if row:
                    try:
                        last_timestamp = str(row["timestamp"]) if row["timestamp"] else None
                    except (KeyError, TypeError):
                        last_timestamp = None
                    if last_event_id is None:
                        try:
                            last_event_id = str(row["event_id"]) if row["event_id"] else None
                        except (KeyError, TypeError):
                            last_event_id = None
                else:
                    # If the rowid no longer exists, reset to full scan; dedupe by event_id protects us.
                    last_rowid = 0
                    last_event_id = None
                    last_timestamp = None

            if last_timestamp:
                query = (
                    "SELECT rowid, * FROM audit_events "
                    "WHERE (timestamp > ? OR (timestamp = ? AND event_id > ?)) "
                    "ORDER BY timestamp, event_id"
                )
                params = (last_timestamp, last_timestamp, last_event_id or "")
            else:
                query = "SELECT rowid, * FROM audit_events ORDER BY timestamp, event_id"
                params = ()

            async with source_db.execute(query, params) as cur:
                while True:
                    rows = await cur.fetchmany(chunk_size)
                    if not rows:
                        break

                    counts.events_read += len(rows)
                    counts.stats_read += len(rows)
                    records: list[dict[str, Any]] = []
                    seen: set[str] = set()
                    duplicates_in_chunk: list[str] = []
                    for row in rows:
                        record = _build_event_record(
                            dict(row),
                            columns,
                            tenant_override=source.tenant_id,
                            system_tenant_id=system_tenant_id,
                            unidentified_tenant_id=unidentified_tenant_id,
                        )
                        ev_id = record.get("event_id")
                        if ev_id:
                            if ev_id in seen:
                                duplicates_in_chunk.append(ev_id)
                                continue
                            seen.add(ev_id)
                        records.append(record)

                    existing = await _fetch_existing_event_ids(
                        shared_db, [r["event_id"] for r in records if r.get("event_id")]
                    )
                    filtered = [r for r in records if r.get("event_id") not in existing]
                    duplicates_existing = [r["event_id"] for r in records if r.get("event_id") in existing]
                    duplicates_total = len(duplicates_in_chunk) + len(duplicates_existing)

                    chunk_events_inserted = len(filtered)
                    chunk_events_skipped = duplicates_total
                    chunk_stats_inserted = 0
                    chunk_stats_skipped = duplicates_total

                    if duplicates_in_chunk:
                        logger.warning(
                            "Audit migration: skipped {} duplicate event_id(s) within batch from {} (tenant {}). Sample={}",
                            len(duplicates_in_chunk),
                            source.label,
                            source.tenant_id or "unknown",
                            duplicates_in_chunk[:5],
                        )
                    if duplicates_existing:
                        logger.warning(
                            "Audit migration: skipped {} duplicate event_id(s) already in shared DB from {} (tenant {}). Sample={}",
                            len(duplicates_existing),
                            source.label,
                            source.tenant_id or "unknown",
                            duplicates_existing[:5],
                        )

                    if filtered:
                        await shared_db.executemany(insert_sql, filtered)
                        stats_result = await _update_shared_daily_stats_from_records(
                            shared_db,
                            filtered,
                            unidentified_tenant_id=unidentified_tenant_id,
                        )
                        chunk_stats_inserted = stats_result.events_contributed
                        chunk_stats_skipped += stats_result.events_skipped

                    last_row = rows[-1]
                    with contextlib.suppress(_AUDIT_COERCE_EXCEPTIONS):
                        last_rowid = int(last_row["rowid"])
                    try:
                        last_event = last_row["event_id"]
                    except (KeyError, TypeError, IndexError):
                        last_event = None
                    if last_event:
                        last_event_id = str(last_event)
                    try:
                        last_timestamp = _checkpoint_timestamp(last_row["timestamp"], last_timestamp)
                    except _AUDIT_NONCRITICAL_EXCEPTIONS:
                        last_timestamp = last_timestamp
                    await _save_checkpoint(shared_db, source_key, last_rowid, last_event_id, last_timestamp)
                    await shared_db.commit()
                    counts.events_inserted += chunk_events_inserted
                    counts.events_skipped += chunk_events_skipped
                    counts.stats_inserted += chunk_stats_inserted
                    counts.stats_skipped += chunk_stats_skipped
    except _AUDIT_DB_EXCEPTIONS as exc:
        with contextlib.suppress(_AUDIT_NONCRITICAL_EXCEPTIONS):
            await shared_db.rollback()
        counts.failed = True
        counts.error = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "Audit migration: failed to process source {} (tenant {}): {}",
            source.label,
            source.tenant_id or "unknown",
            counts.error,
        )
        return counts

    return counts


async def migrate_to_shared_audit_db(
    *,
    shared_db_path: Path | None = None,
    user_db_base_dir: Path | None = None,
    default_db_path: Path | None = None,
    system_tenant_id: str = "system",
    unidentified_tenant_id: str = _UNIDENTIFIED_TENANT_ID,
    chunk_size: int = 5000,
) -> AuditMigrationReport:
    system_tenant_id = system_tenant_id.strip().lower() or "system"
    unidentified_tenant_id = unidentified_tenant_id.strip().lower() or _UNIDENTIFIED_TENANT_ID
    if unidentified_tenant_id == system_tenant_id:
        logger.warning(
            "Unidentified tenant id matches system tenant id; falling back to {}",
            _UNIDENTIFIED_TENANT_ID,
        )
        unidentified_tenant_id = _UNIDENTIFIED_TENANT_ID

    shared_path = shared_db_path or DatabasePaths.get_shared_audit_db_path()
    shared_path.parent.mkdir(parents=True, exist_ok=True)
    service = UnifiedAuditService(
        db_path=str(shared_path),
        storage_mode="shared",
        enable_pii_detection=False,
        enable_risk_scoring=False,
        buffer_size=chunk_size,
        flush_interval=1.0,
        system_tenant_id=system_tenant_id,
        unidentified_tenant_id=unidentified_tenant_id,
    )
    await service.initialize(start_background_tasks=False)
    await service.stop()

    sources = discover_audit_sources(
        user_db_base_dir=user_db_base_dir,
        default_db_path=default_db_path,
    )
    if not sources:
        logger.warning("No audit databases found for migration.")
        return AuditMigrationReport(shared_db_path=shared_path, sources=[])

    counts: list[AuditMigrationCounts] = []
    async with aiosqlite.connect(shared_path) as shared_db:
        shared_db.row_factory = aiosqlite.Row
        try:
            await configure_sqlite_connection_async(
                shared_db,
                use_wal=True,
                synchronous="NORMAL",
                busy_timeout_ms=5000,
                foreign_keys=True,
                temp_store="MEMORY",
            )
        except _AUDIT_DB_EXCEPTIONS:
            pass
        await _ensure_checkpoint_table(shared_db)
        await shared_db.commit()

        for source in sources:
            logger.info(f"Migrating audit DB: {source.label} ({source.path})")
            src_counts = await _migrate_source(
                shared_db,
                source,
                columns=list(service._event_columns),
                insert_sql=service._event_insert_sql,
                system_tenant_id=system_tenant_id,
                unidentified_tenant_id=unidentified_tenant_id,
                chunk_size=chunk_size,
            )
            counts.append(src_counts)
            await shared_db.commit()

    report = AuditMigrationReport(shared_db_path=shared_path, sources=counts)
    logger.info(
        "Audit migration complete. sources={}, failures={}, events_inserted={}, events_skipped={}, stats_read={}, stats_inserted={}, stats_skipped={}",
        len(report.sources),
        report.total_failures,
        report.total_events_inserted,
        report.total_events_skipped,
        report.total_stats_read,
        report.total_stats_inserted,
        report.total_stats_skipped,
    )
    return report


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate per-user audit DBs into shared audit DB.")
    parser.add_argument(
        "--shared-db",
        dest="shared_db",
        default=None,
        help="Path to shared audit DB (default: AUDIT_SHARED_DB_PATH or Databases/audit_shared.db)",
    )
    parser.add_argument(
        "--user-db-base",
        dest="user_db_base",
        default=None,
        help="Base directory for user DBs (default: USER_DB_BASE_DIR)",
    )
    parser.add_argument(
        "--default-db",
        dest="default_db",
        default=None,
        help="Legacy unified audit DB path (default: Databases/unified_audit.db if present)",
    )
    parser.add_argument(
        "--system-tenant-id",
        dest="system_tenant_id",
        default="system",
        help="Reserved tenant id for system events",
    )
    parser.add_argument(
        "--unidentified-tenant-id",
        dest="unidentified_tenant_id",
        default=_UNIDENTIFIED_TENANT_ID,
        help="Reserved tenant id for anonymous/unidentified events",
    )
    parser.add_argument(
        "--chunk-size",
        dest="chunk_size",
        type=int,
        default=5000,
        help="Row batch size for migration",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    shared_db = Path(args.shared_db).expanduser() if args.shared_db else None
    user_db_base = Path(args.user_db_base).expanduser() if args.user_db_base else None
    if args.default_db:
        default_db = Path(args.default_db).expanduser()
    else:
        project_root = Path(get_project_root())
        candidate = project_root / "Databases" / "unified_audit.db"
        default_db = candidate if candidate.exists() else None

    asyncio.run(
        migrate_to_shared_audit_db(
            shared_db_path=shared_db,
            user_db_base_dir=user_db_base,
            default_db_path=default_db,
            system_tenant_id=args.system_tenant_id,
            unidentified_tenant_id=args.unidentified_tenant_id,
            chunk_size=args.chunk_size,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
