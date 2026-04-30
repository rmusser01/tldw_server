from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
    normalize_provider_name,
    normalize_secret_owner_scope_type,
)


@dataclass
class ManagedSecretRefsRepo:
    """Repository for logical secret references and backend registrations."""

    db_pool: DatabasePool

    async def ensure_tables(self) -> None:
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
                    ensure_secret_backend_tables_pg,
                )

                ok = await ensure_secret_backend_tables_pg(self.db_pool)
                if not ok:
                    raise RuntimeError("PostgreSQL secret backend schema ensure failed")
                return

            for table_name in ("secret_backends", "managed_secret_refs"):
                row = await self.db_pool.fetchone(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,),
                )
                if not row:
                    raise RuntimeError(
                        f"SQLite {table_name} table is missing. "
                        "Run the AuthNZ migrations/bootstrap (see "
                        "'python -m tldw_Server_API.app.core.AuthNZ.initialize')."
                    )
        except Exception as exc:
            logger.error("ManagedSecretRefsRepo.ensure_tables failed: {}", exc)
            raise

    @staticmethod
    def _normalize_datetime_for_postgres(dt: datetime | None) -> datetime | None:
        if dt is None:
            return None
        return dt.replace(tzinfo=None) if getattr(dt, "tzinfo", None) else dt

    @staticmethod
    def _parse_json_field(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError):
                return {}
            return dict(parsed) if isinstance(parsed, dict) else {}
        return {}

    @classmethod
    def _row_to_dict(cls, row: Any) -> dict[str, Any]:
        if isinstance(row, dict):
            raw = dict(row)
        else:
            try:
                raw = {key: row[key] for key in row.keys()}
            except Exception as row_keys_error:
                logger.bind(error_type=type(row_keys_error).__name__).debug(
                    "Managed secret ref row key materialization failed; falling back to dict(row)",
                )
                raw = dict(row)

        if "capabilities_json" in raw:
            raw["capabilities"] = cls._parse_json_field(raw.get("capabilities_json"))
        if "metadata_json" in raw:
            raw["metadata"] = cls._parse_json_field(raw.get("metadata_json"))
        return raw

    async def ensure_backend_registration(
        self,
        *,
        name: str,
        display_name: str,
        capabilities: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        status: str = "enabled",
    ) -> dict[str, Any]:
        capabilities_json = json.dumps(capabilities or {})
        metadata_json = json.dumps(metadata or {})
        now = datetime.now(timezone.utc)
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                ts = self._normalize_datetime_for_postgres(now)
                row = await self.db_pool.fetchone(
                    """
                    INSERT INTO secret_backends (
                        name, display_name, status, capabilities_json, metadata_json, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $6)
                    ON CONFLICT (name) DO UPDATE SET
                        display_name = EXCLUDED.display_name,
                        status = EXCLUDED.status,
                        capabilities_json = EXCLUDED.capabilities_json,
                        metadata_json = EXCLUDED.metadata_json,
                        updated_at = EXCLUDED.updated_at
                    RETURNING name, display_name, status, capabilities_json, metadata_json, created_at, updated_at
                    """,
                    name,
                    display_name,
                    status,
                    capabilities_json,
                    metadata_json,
                    ts,
                )
                return self._row_to_dict(row) if row else {}

            await self.db_pool.execute(
                """
                INSERT INTO secret_backends (
                    name, display_name, status, capabilities_json, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    display_name = excluded.display_name,
                    status = excluded.status,
                    capabilities_json = excluded.capabilities_json,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    name,
                    display_name,
                    status,
                    capabilities_json,
                    metadata_json,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            row = await self.db_pool.fetchone(
                """
                SELECT name, display_name, status, capabilities_json, metadata_json, created_at, updated_at
                FROM secret_backends
                WHERE name = ?
                """,
                (name,),
            )
            return self._row_to_dict(row) if row else {}
        except Exception as exc:
            logger.error("ManagedSecretRefsRepo.ensure_backend_registration failed: {}", exc)
            raise

    async def upsert_ref(
        self,
        *,
        backend_name: str,
        owner_scope_type: str,
        owner_scope_id: int,
        provider_key: str,
        backend_ref: str | None,
        metadata: dict[str, Any] | None,
        display_name: str | None = None,
        status: str = "active",
        expires_at: datetime | None = None,
        created_by: int | None = None,
        updated_by: int | None = None,
    ) -> dict[str, Any]:
        scope_type = normalize_secret_owner_scope_type(owner_scope_type)
        provider = normalize_provider_name(provider_key)
        metadata_json = json.dumps(metadata or {})
        now = datetime.now(timezone.utc)
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                ts = self._normalize_datetime_for_postgres(now)
                expires_ts = self._normalize_datetime_for_postgres(expires_at)
                row = await self.db_pool.fetchone(
                    """
                    INSERT INTO managed_secret_refs (
                        backend_name, owner_scope_type, owner_scope_id, provider_key, backend_ref,
                        display_name, status, metadata_json, expires_at,
                        created_by, updated_by, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $12)
                    ON CONFLICT (backend_name, owner_scope_type, owner_scope_id, provider_key) DO UPDATE SET
                        backend_ref = EXCLUDED.backend_ref,
                        display_name = EXCLUDED.display_name,
                        status = EXCLUDED.status,
                        metadata_json = EXCLUDED.metadata_json,
                        expires_at = EXCLUDED.expires_at,
                        revoked_at = NULL,
                        revoked_by = NULL,
                        updated_by = EXCLUDED.updated_by,
                        updated_at = EXCLUDED.updated_at
                    RETURNING id, backend_name, owner_scope_type, owner_scope_id, provider_key, backend_ref,
                              display_name, status, metadata_json, last_resolved_at, expires_at,
                              created_by, updated_by, revoked_by, revoked_at, created_at, updated_at
                    """,
                    backend_name,
                    scope_type,
                    int(owner_scope_id),
                    provider,
                    backend_ref,
                    display_name,
                    status,
                    metadata_json,
                    expires_ts,
                    created_by,
                    updated_by,
                    ts,
                )
                return self._row_to_dict(row) if row else {}

            await self.db_pool.execute(
                """
                INSERT INTO managed_secret_refs (
                    backend_name, owner_scope_type, owner_scope_id, provider_key, backend_ref,
                    display_name, status, metadata_json, expires_at,
                    created_by, updated_by, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(backend_name, owner_scope_type, owner_scope_id, provider_key) DO UPDATE SET
                    backend_ref = excluded.backend_ref,
                    display_name = excluded.display_name,
                    status = excluded.status,
                    metadata_json = excluded.metadata_json,
                    expires_at = excluded.expires_at,
                    revoked_at = NULL,
                    revoked_by = NULL,
                    updated_by = excluded.updated_by,
                    updated_at = excluded.updated_at
                """,
                (
                    backend_name,
                    scope_type,
                    int(owner_scope_id),
                    provider,
                    backend_ref,
                    display_name,
                    status,
                    metadata_json,
                    expires_at.isoformat() if expires_at else None,
                    created_by,
                    updated_by,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            row = await self.db_pool.fetchone(
                """
                SELECT id, backend_name, owner_scope_type, owner_scope_id, provider_key, backend_ref,
                       display_name, status, metadata_json, last_resolved_at, expires_at,
                       created_by, updated_by, revoked_by, revoked_at, created_at, updated_at
                FROM managed_secret_refs
                WHERE backend_name = ? AND owner_scope_type = ? AND owner_scope_id = ? AND provider_key = ?
                """,
                (backend_name, scope_type, int(owner_scope_id), provider),
            )
            return self._row_to_dict(row) if row else {}
        except Exception as exc:
            logger.error("ManagedSecretRefsRepo.upsert_ref failed: {}", exc)
            raise

    async def get_ref(
        self,
        ref_id: int,
        *,
        include_revoked: bool = False,
    ) -> dict[str, Any] | None:
        revoked_clause = "" if include_revoked else " AND revoked_at IS NULL"
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                get_ref_sql_template = """
                    SELECT id, backend_name, owner_scope_type, owner_scope_id, provider_key, backend_ref,
                           display_name, status, metadata_json, last_resolved_at, expires_at,
                           created_by, updated_by, revoked_by, revoked_at, created_at, updated_at
                    FROM managed_secret_refs
                    WHERE id = $1{revoked_clause}
                    """
                get_ref_sql = get_ref_sql_template.format_map(locals())  # nosec B608
                row = await self.db_pool.fetchone(get_ref_sql, int(ref_id))
            else:
                get_ref_sql_template = """
                    SELECT id, backend_name, owner_scope_type, owner_scope_id, provider_key, backend_ref,
                           display_name, status, metadata_json, last_resolved_at, expires_at,
                           created_by, updated_by, revoked_by, revoked_at, created_at, updated_at
                    FROM managed_secret_refs
                    WHERE id = ?{revoked_clause}
                    """
                get_ref_sql = get_ref_sql_template.format_map(locals())  # nosec B608
                row = await self.db_pool.fetchone(get_ref_sql, (int(ref_id),))
            return self._row_to_dict(row) if row else None
        except Exception as exc:
            logger.error("ManagedSecretRefsRepo.get_ref failed: {}", exc)
            raise

    async def list_refs_by_ids(
        self,
        ref_ids: list[int],
        *,
        include_revoked: bool = False,
    ) -> dict[int, dict[str, Any]]:
        """Return managed secret refs keyed by id for the supplied identifiers."""
        normalized_ids = sorted({int(ref_id) for ref_id in ref_ids if int(ref_id) > 0})
        if not normalized_ids:
            return {}

        try:
            if getattr(self.db_pool, "pool", None) is not None:
                list_refs_sql = """
                    SELECT id, backend_name, owner_scope_type, owner_scope_id, provider_key, backend_ref,
                           display_name, status, metadata_json, last_resolved_at, expires_at,
                           created_by, updated_by, revoked_by, revoked_at, created_at, updated_at
                    FROM managed_secret_refs
                    WHERE id = ANY($1::int[]) AND revoked_at IS NULL
                    """
                if include_revoked:
                    list_refs_sql = """
                        SELECT id, backend_name, owner_scope_type, owner_scope_id, provider_key, backend_ref,
                               display_name, status, metadata_json, last_resolved_at, expires_at,
                               created_by, updated_by, revoked_by, revoked_at, created_at, updated_at
                        FROM managed_secret_refs
                        WHERE id = ANY($1::int[])
                        """
                rows = await self.db_pool.fetchall(list_refs_sql, normalized_ids)
            else:
                list_refs_sql = """
                    SELECT id, backend_name, owner_scope_type, owner_scope_id, provider_key, backend_ref,
                           display_name, status, metadata_json, last_resolved_at, expires_at,
                           created_by, updated_by, revoked_by, revoked_at, created_at, updated_at
                    FROM managed_secret_refs
                    WHERE id IN (SELECT CAST(value AS INTEGER) FROM json_each(?)) AND revoked_at IS NULL
                    """
                if include_revoked:
                    list_refs_sql = """
                        SELECT id, backend_name, owner_scope_type, owner_scope_id, provider_key, backend_ref,
                               display_name, status, metadata_json, last_resolved_at, expires_at,
                               created_by, updated_by, revoked_by, revoked_at, created_at, updated_at
                        FROM managed_secret_refs
                        WHERE id IN (SELECT CAST(value AS INTEGER) FROM json_each(?))
                        """
                rows = await self.db_pool.fetchall(list_refs_sql, (json.dumps(normalized_ids),))
            return {
                int(row_dict["id"]): row_dict
                for row in rows
                for row_dict in [self._row_to_dict(row)]
                if row_dict.get("id") is not None
            }
        except Exception as exc:
            logger.error("ManagedSecretRefsRepo.list_refs_by_ids failed: {}", exc)
            raise

    async def touch_last_resolved(
        self,
        ref_id: int,
        *,
        resolved_at: datetime,
        expires_at: datetime | None = None,
    ) -> None:
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                ts = self._normalize_datetime_for_postgres(resolved_at)
                expires_ts = self._normalize_datetime_for_postgres(expires_at)
                await self.db_pool.execute(
                    """
                    UPDATE managed_secret_refs
                    SET last_resolved_at = $1, expires_at = $2, updated_at = $1
                    WHERE id = $3
                    """,
                    ts,
                    expires_ts,
                    int(ref_id),
                )
                return

            await self.db_pool.execute(
                """
                UPDATE managed_secret_refs
                SET last_resolved_at = ?, expires_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    resolved_at.isoformat(),
                    expires_at.isoformat() if expires_at else None,
                    resolved_at.isoformat(),
                    int(ref_id),
                ),
            )
        except Exception as exc:
            logger.error("ManagedSecretRefsRepo.touch_last_resolved failed: {}", exc)
            raise

    async def delete_ref(
        self,
        ref_id: int,
        *,
        revoked_by: int | None = None,
        revoked_at: datetime | None = None,
    ) -> bool:
        revoked_ts = revoked_at or datetime.now(timezone.utc)
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                ts = self._normalize_datetime_for_postgres(revoked_ts)
                result = await self.db_pool.execute(
                    """
                    UPDATE managed_secret_refs
                    SET revoked_at = $1, revoked_by = $2, updated_at = $1, updated_by = $3
                    WHERE id = $4 AND revoked_at IS NULL
                    """,
                    ts,
                    revoked_by,
                    revoked_by,
                    int(ref_id),
                )
                if isinstance(result, str):
                    parts = result.split()
                    if parts and parts[-1].isdigit():
                        return int(parts[-1]) > 0
                return True

            cursor = await self.db_pool.execute(
                """
                UPDATE managed_secret_refs
                SET revoked_at = ?, revoked_by = ?, updated_at = ?, updated_by = ?
                WHERE id = ? AND revoked_at IS NULL
                """,
                (
                    revoked_ts.isoformat(),
                    revoked_by,
                    revoked_ts.isoformat(),
                    revoked_by,
                    int(ref_id),
                ),
            )
            rowcount = getattr(cursor, "rowcount", 0)
            return rowcount > 0
        except Exception as exc:
            logger.error("ManagedSecretRefsRepo.delete_ref failed: {}", exc)
            raise
