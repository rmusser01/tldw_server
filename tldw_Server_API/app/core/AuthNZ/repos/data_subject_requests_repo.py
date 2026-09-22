from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool


@dataclass
class AuthnzDataSubjectRequestsRepo:
    """Repository for AuthNZ data subject request records."""

    db_pool: DatabasePool

    def _is_postgres_backend(self) -> bool:
        """Return True when the current AuthNZ backend is PostgreSQL."""
        return bool(getattr(self.db_pool, "pool", None))

    async def ensure_schema(self) -> None:
        """Ensure the DSR table exists for the current backend."""
        if self._is_postgres_backend():
            from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
                ensure_data_subject_requests_table_pg,
            )

            await ensure_data_subject_requests_table_pg(self.db_pool)
            return

        from pathlib import Path

        from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables

        db_fs_path = getattr(self.db_pool, "_sqlite_fs_path", None) or getattr(
            self.db_pool, "db_path", None
        )
        if db_fs_path:
            ensure_authnz_tables(Path(str(db_fs_path)))

    @staticmethod
    def _parse_json_field(value: Any, *, fallback: Any) -> Any:
        """Decode a stored JSON blob, clamped to the container the caller asked for.

        The four sibling copies in this package end with
        ``dict(parsed) if isinstance(parsed, dict) else {}``; this one returned
        ``json.loads(...)`` raw, so a column holding "[]", "null" or "123" yielded a
        list / None / int where ``fallback`` promised a dict. coverage_metadata is
        declared ``dict[str, Any]`` on the response models
        (api/v1/schemas/admin_schemas.py:1027, :1082), so that surfaced as a validation
        error on a GDPR data-subject-request read rather than a benign default.
        """
        if value is None:
            return fallback

        if isinstance(value, str):
            try:
                parsed: Any = json.loads(value)
            except json.JSONDecodeError:
                return fallback
        elif isinstance(value, (list, dict)):
            parsed = value
        else:
            return fallback

        if isinstance(fallback, dict):
            return parsed if isinstance(parsed, dict) else fallback
        if isinstance(fallback, list):
            return parsed if isinstance(parsed, list) else fallback
        return parsed

    @classmethod
    def _normalize_record(cls, row: Any) -> dict[str, Any]:
        """Normalize backend row types into JSON-friendly dicts."""
        if row is None:
            return {}

        try:
            record = dict(row) if hasattr(row, "keys") or isinstance(row, dict) else {}
        except Exception:
            record = {}

        for field in ("id", "resolved_user_id", "requested_by_user_id"):
            if field in record and record[field] is not None:
                with contextlib.suppress(Exception):
                    record[field] = int(record[field])

        record["selected_categories"] = cls._parse_json_field(
            record.get("selected_categories"),
            fallback=[],
        )
        record["preview_summary"] = cls._parse_json_field(
            record.get("preview_summary"),
            fallback=[],
        )
        record["coverage_metadata"] = cls._parse_json_field(
            record.get("coverage_metadata"),
            fallback={},
        )

        if "requested_at" in record and record["requested_at"] is not None:
            with contextlib.suppress(Exception):
                if hasattr(record["requested_at"], "isoformat"):
                    record["requested_at"] = record["requested_at"].isoformat()

        return record

    async def create_or_get_request(
        self,
        *,
        client_request_id: str,
        requester_identifier: str,
        resolved_user_id: int | None,
        request_type: str,
        status: str,
        selected_categories: list[str],
        preview_summary: list[dict[str, Any]],
        coverage_metadata: dict[str, Any] | None,
        requested_by_user_id: int | None,
        notes: str | None,
    ) -> dict[str, Any]:
        """Persist a request unless the client idempotency key already exists."""
        now = datetime.now(timezone.utc)
        selected_categories_json = json.dumps(selected_categories or [])
        preview_summary_json = json.dumps(preview_summary or [])
        coverage_metadata_json = json.dumps(coverage_metadata or {})

        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    row = await conn.fetchrow(
                        """
                        INSERT INTO data_subject_requests (
                            client_request_id,
                            requester_identifier,
                            resolved_user_id,
                            request_type,
                            status,
                            selected_categories,
                            preview_summary,
                            coverage_metadata,
                            requested_by_user_id,
                            requested_at,
                            notes
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8::jsonb, $9, $10, $11
                        )
                        ON CONFLICT (client_request_id) DO NOTHING
                        RETURNING *
                        """,
                        client_request_id,
                        requester_identifier,
                        resolved_user_id,
                        request_type,
                        status,
                        selected_categories_json,
                        preview_summary_json,
                        coverage_metadata_json,
                        requested_by_user_id,
                        now,
                        notes,
                    )
                    if row is None:
                        row = await conn.fetchrow(
                            "SELECT * FROM data_subject_requests WHERE client_request_id = $1",
                            client_request_id,
                        )
                    return self._normalize_record(row)

                await conn.execute(
                    """
                    INSERT OR IGNORE INTO data_subject_requests (
                        client_request_id,
                        requester_identifier,
                        resolved_user_id,
                        request_type,
                        status,
                        selected_categories,
                        preview_summary,
                        coverage_metadata,
                        requested_by_user_id,
                        requested_at,
                        notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        client_request_id,
                        requester_identifier,
                        resolved_user_id,
                        request_type,
                        status,
                        selected_categories_json,
                        preview_summary_json,
                        coverage_metadata_json,
                        requested_by_user_id,
                        now.isoformat(),
                        notes,
                    ),
                )
                cursor = await conn.execute(
                    "SELECT * FROM data_subject_requests WHERE client_request_id = ?",
                    (client_request_id,),
                )
                row = await cursor.fetchone()
                if row is None:
                    raise RuntimeError("Failed to fetch persisted data subject request")
                return self._normalize_record(row)
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzDataSubjectRequestsRepo.create_or_get_request failed: {exc}")
            raise

    _VALID_STATUSES = {"pending", "recorded", "executing", "completed", "failed"}

    async def get_request_by_id(self, request_id: int) -> dict[str, Any] | None:
        """Return a single DSR record by primary key, or None if not found."""
        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres_backend():
                    row = await conn.fetchrow(
                        "SELECT * FROM data_subject_requests WHERE id = $1",
                        request_id,
                    )
                    return self._normalize_record(row) if row else None

                cursor = await conn.execute(
                    "SELECT * FROM data_subject_requests WHERE id = ?",
                    (request_id,),
                )
                row = await cursor.fetchone()
                return self._normalize_record(row) if row else None
        except Exception as exc:
            logger.error(f"AuthnzDataSubjectRequestsRepo.get_request_by_id failed: {exc}")
            raise

    async def update_request_status(
        self,
        request_id: int,
        new_status: str,
        *,
        notes: str | None = None,
    ) -> dict[str, Any] | None:
        """Update the status (and optionally notes) of a DSR record.

        Returns the updated record as a dict, or None if *request_id* does not
        exist.  Raises ``ValueError`` when *new_status* is not one of the
        recognised lifecycle states.
        """
        if new_status not in self._VALID_STATUSES:
            raise ValueError(
                f"Invalid DSR status '{new_status}'. "
                f"Must be one of: {', '.join(sorted(self._VALID_STATUSES))}"
            )

        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    if notes is not None:
                        row = await conn.fetchrow(
                            "UPDATE data_subject_requests SET status = $1, notes = $2 WHERE id = $3 RETURNING *",
                            new_status,
                            notes,
                            request_id,
                        )
                    else:
                        row = await conn.fetchrow(
                            "UPDATE data_subject_requests SET status = $1 WHERE id = $2 RETURNING *",
                            new_status,
                            request_id,
                        )
                    return self._normalize_record(row) if row else None

                if notes is not None:
                    await conn.execute(
                        "UPDATE data_subject_requests SET status = ?, notes = ? WHERE id = ?",
                        (new_status, notes, request_id),
                    )
                else:
                    await conn.execute(
                        "UPDATE data_subject_requests SET status = ? WHERE id = ?",
                        (new_status, request_id),
                    )
                cursor = await conn.execute(
                    "SELECT * FROM data_subject_requests WHERE id = ?",
                    (request_id,),
                )
                row = await cursor.fetchone()
                return self._normalize_record(row) if row else None
        except ValueError:
            raise
        except Exception as exc:
            logger.error(f"AuthnzDataSubjectRequestsRepo.update_request_status failed: {exc}")
            raise

    async def list_requests(
        self,
        *,
        limit: int,
        offset: int,
        resolved_user_ids: list[int] | None = None,
        org_ids: list[int] | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return paged request rows ordered newest-first."""
        try:
            if resolved_user_ids is not None and not resolved_user_ids:
                return [], 0
            if org_ids is not None and not org_ids:
                return [], 0

            async with self.db_pool.acquire() as conn:
                if self._is_postgres_backend():
                    where_clauses: list[str] = []
                    params: list[Any] = []

                    if resolved_user_ids is not None:
                        where_clauses.append(f"resolved_user_id = ANY(${len(params) + 1})")
                        params.append([int(value) for value in resolved_user_ids])

                    if org_ids is not None:
                        org_scope_param_index = len(params) + 1
                        # Placeholder index is int-derived; values remain parameterized.
                        org_scope_clause = f"EXISTS (SELECT 1 FROM org_members om WHERE om.user_id = data_subject_requests.resolved_user_id AND om.org_id = ANY(${org_scope_param_index}) AND om.status = 'active')"  # nosec
                        where_clauses.append(org_scope_clause)
                        params.append([int(value) for value in org_ids])

                    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
                    # where_sql is built only from fixed clauses with parameter placeholders.
                    query = f"SELECT * FROM data_subject_requests {where_sql} ORDER BY requested_at DESC, id DESC LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}"  # nosec
                    count_query = f"SELECT COUNT(*) FROM data_subject_requests {where_sql}"  # nosec
                    rows = await conn.fetch(query, *params, limit, offset)
                    total = await conn.fetchval(count_query, *params)
                    return [self._normalize_record(row) for row in rows], int(total or 0)

                where_clauses: list[str] = []
                params: list[Any] = []

                if resolved_user_ids is not None:
                    placeholders = ", ".join(["?"] * len(resolved_user_ids))
                    where_clauses.append(f"resolved_user_id IN ({placeholders})")
                    params.extend(int(value) for value in resolved_user_ids)

                if org_ids is not None:
                    placeholders = ", ".join(["?"] * len(org_ids))
                    # Placeholder count is int-derived; values remain parameterized.
                    org_scope_clause = f"EXISTS (SELECT 1 FROM org_members om WHERE om.user_id = data_subject_requests.resolved_user_id AND om.org_id IN ({placeholders}) AND om.status = 'active')"  # nosec
                    where_clauses.append(org_scope_clause)
                    params.extend(int(value) for value in org_ids)

                where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
                # where_sql is built only from fixed clauses with parameter placeholders.
                query = f"SELECT * FROM data_subject_requests {where_sql} ORDER BY requested_at DESC, id DESC LIMIT ? OFFSET ?"  # nosec
                cursor = await conn.execute(
                    query,
                    (*params, limit, offset),
                )
                rows = await cursor.fetchall()
                count_query = f"SELECT COUNT(*) FROM data_subject_requests {where_sql}"  # nosec
                total_cursor = await conn.execute(
                    count_query,
                    tuple(params),
                )
                total_row = await total_cursor.fetchone()
                total = int(total_row[0]) if total_row else 0
                return [self._normalize_record(row) for row in rows], total
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzDataSubjectRequestsRepo.list_requests failed: {exc}")
            raise
