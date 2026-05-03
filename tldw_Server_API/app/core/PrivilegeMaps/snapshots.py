from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from loguru import logger

from tldw_Server_API.app.api.v1.utils.pagination import build_page_pagination_meta
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.exceptions import TransactionError

MAX_SNAPSHOT_DETAIL_ROWS = 50_000
DETAIL_INSERT_BATCH_SIZE = 500
_SNAPSHOT_JSON_EXCEPTIONS: tuple[type[BaseException], ...] = (
    TypeError,
    ValueError,
    json.JSONDecodeError,
    UnicodeDecodeError,
)
_SNAPSHOT_NUMERIC_EXCEPTIONS: tuple[type[BaseException], ...] = (
    OverflowError,
    TypeError,
    ValueError,
)
_SNAPSHOT_DB_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    OSError,
    RuntimeError,
    TimeoutError,
    TransactionError,
    TypeError,
    ValueError,
    sqlite3.Error,
)


class PrivilegeSnapshotStore:
    """Database-backed snapshot store for privilege maps."""

    def __init__(self, pool: DatabasePool | None = None) -> None:
        self._pool = pool
        self._initialized = False

    async def list_snapshots(
        self,
        *,
        page: int,
        page_size: int,
        date_from: datetime | None,
        date_to: datetime | None,
        generated_by: str | None,
        org_id: str | None,
        team_id: str | None,
        catalog_version: str | None,
        scope: str | None,
        include_counts: bool,
    ) -> dict[str, Any]:
        pool = await self._get_pool()
        await self._ensure_schema(pool)

        filters: list[str] = []
        params: list[Any] = []

        if org_id:
            filters.append("org_id = ?")
            params.append(org_id)
        if team_id:
            filters.append("team_id = ?")
            params.append(team_id)
        if generated_by:
            filters.append("generated_by = ?")
            params.append(generated_by)
        if catalog_version:
            filters.append("catalog_version = ?")
            params.append(catalog_version)
        if date_from:
            filters.append("generated_at >= ?")
            params.append(self._to_iso(date_from))
        if date_to:
            filters.append("generated_at <= ?")
            params.append(self._to_iso(date_to))
        if scope:
            filters.append("scope_index LIKE ?")
            params.append(f"%|{scope}|%")

        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        count_sql_template = "SELECT COUNT(*) AS total FROM privilege_snapshots {where_clause}"
        count_sql = count_sql_template.format_map(locals())  # nosec B608
        count_row = await pool.fetchone(
            count_sql,
            tuple(params),
        )
        if not count_row:
            total_items = 0
        elif isinstance(count_row, dict):
            total_items = int(count_row.get("total", 0))
        elif hasattr(count_row, "keys"):
            total_items = int(count_row["total"])
        else:
            total_items = int(count_row[0])

        page = max(page, 1)
        page_size = max(min(page_size, 200), 1)
        offset = (page - 1) * page_size
        data_params = list(params) + [page_size, offset]

        list_sql_template = """
            SELECT snapshot_id, generated_at, generated_by, target_scope, org_id, team_id,
                   catalog_version, summary_json
            FROM privilege_snapshots
            {where_clause}
            ORDER BY generated_at DESC
            LIMIT ? OFFSET ?
            """
        list_sql = list_sql_template.format_map(locals())  # nosec B608
        rows = await pool.fetchall(
            list_sql,
            tuple(data_params),
        )

        items: list[dict[str, Any]] = []
        for row in rows:
            record = self._row_to_dict(row)
            if not record:
                continue
            summary_obj = None
            if include_counts and record.get("summary_json"):
                try:
                    summary_obj = json.loads(record["summary_json"])
                except _SNAPSHOT_JSON_EXCEPTIONS as exc:
                    logger.warning("Failed to parse snapshot summary JSON: {}", exc)
                    summary_obj = None

            generated_at_dt = self._parse_datetime(record.get("generated_at"))

            items.append(
                {
                    "snapshot_id": record.get("snapshot_id"),
                    "generated_at": generated_at_dt,
                    "generated_by": record.get("generated_by"),
                    "target_scope": record.get("target_scope"),
                    "org_id": record.get("org_id"),
                    "team_id": record.get("team_id"),
                    "catalog_version": record.get("catalog_version"),
                    "summary": summary_obj,
                }
            )

        return {
            "page": page,
            "page_size": page_size,
            "total_items": total_items,
            "pagination": build_page_pagination_meta(
                page=page,
                per_page=page_size,
                total=total_items,
                total_pages=(total_items + page_size - 1) // page_size,
            ),
            "items": items,
            "filters": {
                "date_from": date_from.isoformat() if date_from else None,
                "date_to": date_to.isoformat() if date_to else None,
                "generated_by": generated_by,
                "org_id": org_id,
                "team_id": team_id,
                "catalog_version": catalog_version,
                "scope": scope,
                "include_counts": include_counts,
            },
        }

    async def add_snapshot(
        self,
        snapshot: dict[str, Any],
        detail_items: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        snapshot_id = snapshot.get("snapshot_id")
        if not snapshot_id:
            raise ValueError("snapshot must include snapshot_id")

        pool = await self._get_pool()
        await self._ensure_schema(pool)

        generated_at = snapshot.get("generated_at")
        generated_at_iso = self._to_iso(generated_at)
        generated_by = snapshot.get("generated_by")
        target_scope = snapshot.get("target_scope")
        org_id = snapshot.get("org_id")
        team_id = snapshot.get("team_id")
        catalog_version = snapshot.get("catalog_version")
        summary = snapshot.get("summary")
        summary_json = json.dumps(summary) if summary is not None else None
        scope_index = self._build_scope_index(summary)
        now_iso = self._to_iso(datetime.now(timezone.utc))

        detail_payload: list[dict[str, Any]] = []
        if detail_items is not None:
            # Clamp to prevent runaway storage; log if truncated.
            detail_payload = list(detail_items)[:MAX_SNAPSHOT_DETAIL_ROWS]
            if len(detail_items) > MAX_SNAPSHOT_DETAIL_ROWS:
                logger.warning(
                    'Truncated snapshot detail rows for {} to {} entries',
                    snapshot_id,
                    MAX_SNAPSHOT_DETAIL_ROWS,
                )

        async with pool.transaction() as conn:
            await conn.execute(
                """
                INSERT INTO privilege_snapshots (
                    snapshot_id,
                    generated_at,
                    generated_by,
                    target_scope,
                    org_id,
                    team_id,
                    catalog_version,
                    summary_json,
                    scope_index,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_id) DO UPDATE SET
                    generated_at = excluded.generated_at,
                    generated_by = excluded.generated_by,
                    target_scope = excluded.target_scope,
                    org_id = excluded.org_id,
                    team_id = excluded.team_id,
                    catalog_version = excluded.catalog_version,
                    summary_json = excluded.summary_json,
                    scope_index = excluded.scope_index,
                    updated_at = excluded.updated_at
                """,
                (
                    snapshot_id,
                    generated_at_iso,
                    generated_by,
                    target_scope,
                    org_id,
                    team_id,
                    catalog_version,
                    summary_json,
                    scope_index,
                    now_iso,
                    now_iso,
                ),
            )
            await conn.execute(
                "DELETE FROM privilege_snapshot_details WHERE snapshot_id = ?",
                (snapshot_id,),
            )
            if detail_payload:
                await self._insert_snapshot_details(conn, snapshot_id, detail_payload, now_iso)

    async def clear(self) -> None:
        pool = await self._get_pool()
        await self._ensure_schema(pool)
        try:
            async with pool.transaction() as conn:
                await conn.execute("DELETE FROM privilege_snapshots")
                await conn.execute("DELETE FROM privilege_snapshot_details")
        except TransactionError as exc:
            detail = str(exc).lower()
            if "no such table" in detail:
                logger.debug("Snapshot clear skipped: tables not present yet")
            else:
                raise

    async def _insert_snapshot_details(
        self,
        conn: Any,
        snapshot_id: str,
        detail_rows: Sequence[dict[str, Any]],
        created_at_iso: str,
    ) -> None:
        row_index = 0
        for chunk_start in range(0, len(detail_rows), DETAIL_INSERT_BATCH_SIZE):
            chunk = detail_rows[chunk_start : chunk_start + DETAIL_INSERT_BATCH_SIZE]
            for item in chunk:
                row_json = self._encode_detail_item(item)
                await conn.execute(
                    """
                    INSERT INTO privilege_snapshot_details (
                        snapshot_id,
                        row_index,
                        row_json,
                        created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        snapshot_id,
                        row_index,
                        row_json,
                        created_at_iso,
                    ),
                )
                row_index += 1

    async def get_snapshot(
        self,
        *,
        snapshot_id: str,
        page: int,
        page_size: int,
    ) -> dict[str, Any] | None:
        pool = await self._get_pool()
        await self._ensure_schema(pool)
        row = await pool.fetchone(
            """
            SELECT snapshot_id,
                   generated_at,
                   generated_by,
                   target_scope,
                   org_id,
                   team_id,
                   catalog_version,
                   summary_json,
                   COALESCE(downsampled, 0) AS downsampled
            FROM privilege_snapshots
            WHERE snapshot_id = ?
            """,
            (snapshot_id,),
        )
        record = self._row_to_dict(row)
        if not record:
            return None

        if record.get("downsampled"):
            return {"_downsampled": True, "snapshot_id": record.get("snapshot_id")}

        summary_obj = None
        if record.get("summary_json"):
            try:
                summary_obj = json.loads(record["summary_json"])
            except _SNAPSHOT_JSON_EXCEPTIONS as exc:
                logger.warning("Failed to parse snapshot summary JSON: {}", exc)
                summary_obj = None
        page = max(page, 1)
        page_size = max(min(page_size, 500), 1)
        offset = (page - 1) * page_size
        total_items_raw = await pool.fetchval(
            "SELECT COUNT(*) FROM privilege_snapshot_details WHERE snapshot_id = ?",
            (snapshot_id,),
        )
        try:
            total_items = int(total_items_raw or 0)
        except _SNAPSHOT_NUMERIC_EXCEPTIONS:
            total_items = 0
        detail_items: list[dict[str, Any]] = []
        if total_items and offset < total_items:
            rows = await pool.fetchall(
                """
                SELECT row_index, row_json
                FROM privilege_snapshot_details
                WHERE snapshot_id = ?
                ORDER BY row_index
                LIMIT ? OFFSET ?
                """,
                (snapshot_id, page_size, offset),
            )
            for row in rows:
                payload = self._decode_detail_json(self._row_to_dict(row).get("row_json"))
                if payload is not None:
                    detail_items.append(payload)
        detail = {
            "page": page,
            "page_size": page_size,
            "total_items": total_items,
            "pagination": build_page_pagination_meta(
                page=page,
                per_page=page_size,
                total=total_items,
                total_pages=(total_items + page_size - 1) // page_size,
            ),
            "items": detail_items,
        }

        return {
            "snapshot_id": record.get("snapshot_id"),
            "catalog_version": record.get("catalog_version"),
            "generated_at": self._parse_datetime(record.get("generated_at")),
            "generated_by": record.get("generated_by"),
            "target_scope": record.get("target_scope"),
            "org_id": record.get("org_id"),
            "team_id": record.get("team_id"),
            "summary": summary_obj,
            "detail": detail,
            "etag": f'W/"{record.get("snapshot_id")}-v{total_items or 0}"',
        }

    async def export_snapshot(
        self,
        *,
        snapshot_id: str,
    ) -> dict[str, Any] | None:
        pool = await self._get_pool()
        await self._ensure_schema(pool)

        row = await pool.fetchone(
            """
            SELECT snapshot_id,
                   generated_at,
                   generated_by,
                   target_scope,
                   org_id,
                   team_id,
                   catalog_version,
                   summary_json
            FROM privilege_snapshots
            WHERE snapshot_id = ?
            """,
            (snapshot_id,),
        )
        record = self._row_to_dict(row)
        if not record:
            return None

        summary_obj = None
        if record.get("summary_json"):
            try:
                summary_obj = json.loads(record["summary_json"])
            except _SNAPSHOT_JSON_EXCEPTIONS as exc:
                logger.warning("Failed to parse snapshot summary JSON during export: {}", exc)
                summary_obj = None

        rows = await pool.fetchall(
            """
            SELECT row_index, row_json
            FROM privilege_snapshot_details
            WHERE snapshot_id = ?
            ORDER BY row_index
            """,
            (snapshot_id,),
        )

        detail_items: list[dict[str, Any]] = []
        for row in rows:
            payload = self._decode_detail_json(self._row_to_dict(row).get("row_json"))
            if payload is not None:
                detail_items.append(payload)

        total_items = len(detail_items)
        return {
            "snapshot_id": record.get("snapshot_id"),
            "catalog_version": record.get("catalog_version"),
            "generated_at": self._parse_datetime(record.get("generated_at")),
            "generated_by": record.get("generated_by"),
            "target_scope": record.get("target_scope"),
            "org_id": record.get("org_id"),
            "team_id": record.get("team_id"),
            "summary": summary_obj,
            "detail_items": detail_items,
            "total_items": total_items,
            "etag": f'W/"{record.get("snapshot_id")}-v{total_items}"',
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    async def _get_pool(self) -> DatabasePool:
        if self._pool is None:
            self._pool = await get_db_pool()
        return self._pool

    async def _ensure_schema(self, pool: DatabasePool) -> None:
        if self._initialized:
            return
        async with pool.transaction() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS privilege_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    generated_at TEXT NOT NULL,
                    generated_by TEXT NOT NULL,
                    target_scope TEXT,
                    org_id TEXT,
                    team_id TEXT,
                    catalog_version TEXT NOT NULL,
                    summary_json TEXT,
                    scope_index TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS privilege_snapshot_details (
                    snapshot_id TEXT NOT NULL,
                    row_index INTEGER NOT NULL,
                    row_json TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (snapshot_id, row_index)
                )
                """
            )

        # Ensure legacy deployments add scope_index column
        try:
            async with pool.transaction() as conn:
                await conn.execute(
                    "ALTER TABLE privilege_snapshots ADD COLUMN scope_index TEXT"
                )
        except _SNAPSHOT_DB_NONCRITICAL_EXCEPTIONS:
            pass

        try:
            async with pool.transaction() as conn:
                await conn.execute(
                    "ALTER TABLE privilege_snapshots ADD COLUMN target_scope TEXT"
                )
        except _SNAPSHOT_DB_NONCRITICAL_EXCEPTIONS:
            pass

        try:
            async with pool.transaction() as conn:
                await conn.execute(
                    "ALTER TABLE privilege_snapshots ADD COLUMN downsampled INTEGER DEFAULT 0"
                )
        except _SNAPSHOT_DB_NONCRITICAL_EXCEPTIONS:
            pass

        try:
            async with pool.transaction() as conn:
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_priv_snapshots_generated_at ON privilege_snapshots(generated_at)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_priv_snapshots_org ON privilege_snapshots(org_id)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_priv_snapshots_team ON privilege_snapshots(team_id)"
                )
        except _SNAPSHOT_DB_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Privilege snapshot index creation skipped: {}", exc)

        self._initialized = True

    @staticmethod
    def _encode_detail_item(item: dict[str, Any]) -> str:
        def _default(obj: Any) -> Any:
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, set):
                return sorted(obj)
            return str(obj)

        try:
            return json.dumps(item, default=_default)
        except (TypeError, ValueError):
            sanitized = {str(k): str(v) for k, v in item.items()}
            return json.dumps(sanitized)

    @staticmethod
    def _decode_detail_json(value: str | None) -> dict[str, Any] | None:
        if not value:
            return None
        try:
            payload = json.loads(value)
            return payload if isinstance(payload, dict) else None
        except _SNAPSHOT_JSON_EXCEPTIONS as exc:
            logger.warning("Failed to decode snapshot detail payload: {}", exc)
            return None

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any] | None:
        if row is None:
            return None
        if isinstance(row, dict):
            return row
        if hasattr(row, "_mapping"):
            return dict(row._mapping)  # type: ignore[attr-defined]
        if hasattr(row, "keys"):
            return {key: row[key] for key in row.keys()}
        return None

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_iso(value: Any) -> str:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc).isoformat()
        if isinstance(value, str):
            return value
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _build_scope_index(summary: dict[str, Any] | None) -> str | None:
        if not summary:
            return None
        scope_ids = summary.get("scope_ids")
        if not scope_ids:
            return None
        ordered = sorted(set(scope_ids))
        return "|" + "|".join(ordered) + "|"


@lru_cache
def get_privilege_snapshot_store() -> PrivilegeSnapshotStore:
    return PrivilegeSnapshotStore()
