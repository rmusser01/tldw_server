from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import normalize_provider_name


def _normalize_scope_type(scope_type: str) -> str:
    st = (scope_type or "").strip().lower()
    if st in {"org", "organization", "orgs"}:
        return "org"
    if st in {"team", "teams"}:
        return "team"
    raise ValueError(f"Invalid scope_type: {scope_type}")


@dataclass
class AuthnzOrgProviderSecretsRepo:
    """Repository for org/team shared provider secrets (BYOK)."""

    db_pool: DatabasePool

    async def ensure_tables(self) -> None:
        """Ensure org_provider_secrets schema exists."""
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
                    ensure_org_provider_secrets_pg,
                )

                ok = await ensure_org_provider_secrets_pg(self.db_pool)
                if not ok:
                    raise RuntimeError("PostgreSQL org_provider_secrets schema ensure failed")
                return

            row = await self.db_pool.fetchone(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='org_provider_secrets'"
            )
            if not row:
                raise RuntimeError(
                    "SQLite org_provider_secrets table is missing. "
                    "Run the AuthNZ migrations/bootstrap (see "
                    "'python -m tldw_Server_API.app.core.AuthNZ.initialize')."
                )
        except Exception as exc:
            logger.error(f"AuthnzOrgProviderSecretsRepo.ensure_tables failed: {exc}")
            raise

    @staticmethod
    def _normalize_datetime_for_postgres(dt: datetime) -> datetime:
        return dt.replace(tzinfo=None) if getattr(dt, "tzinfo", None) else dt

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        if isinstance(row, dict):
            return dict(row)
        try:
            keys = row.keys()
            return {key: row[key] for key in keys}
        except Exception as row_keys_error:
            logger.bind(error_type=type(row_keys_error).__name__).debug(
                "Org provider secret row key materialization failed; falling back to dict(row)"
            )
        return dict(row)

    async def upsert_secret(
        self,
        *,
        scope_type: str,
        scope_id: int,
        provider: str,
        encrypted_blob: str,
        key_hint: str | None,
        metadata: dict[str, Any] | None,
        updated_at: datetime,
        created_by: int | None = None,
        updated_by: int | None = None,
    ) -> dict[str, Any]:
        scope_norm = _normalize_scope_type(scope_type)
        provider_norm = normalize_provider_name(provider)
        metadata_json = json.dumps(metadata) if metadata is not None else None
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                ts = self._normalize_datetime_for_postgres(updated_at)
                row = await self.db_pool.fetchone(
                    """
                    INSERT INTO org_provider_secrets (
                        scope_type, scope_id, provider, encrypted_blob, key_hint, metadata,
                        created_by, updated_by, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $9)
                    ON CONFLICT (scope_type, scope_id, provider) DO UPDATE SET
                        encrypted_blob = EXCLUDED.encrypted_blob,
                        key_hint = EXCLUDED.key_hint,
                        metadata = EXCLUDED.metadata,
                        updated_at = EXCLUDED.updated_at,
                        updated_by = EXCLUDED.updated_by,
                        revoked_at = NULL,
                        revoked_by = NULL
                    RETURNING id, scope_type, scope_id, provider, key_hint, metadata, created_at, updated_at,
                              last_used_at, created_by, updated_by, revoked_by, revoked_at
                    """,
                    scope_norm,
                    int(scope_id),
                    provider_norm,
                    encrypted_blob,
                    key_hint,
                    metadata_json,
                    created_by,
                    updated_by,
                    ts,
                )
                return self._row_to_dict(row) if row else {}

            await self.db_pool.execute(
                """
                INSERT INTO org_provider_secrets (
                    scope_type, scope_id, provider, encrypted_blob, key_hint, metadata,
                    created_by, updated_by, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope_type, scope_id, provider) DO UPDATE SET
                    encrypted_blob = excluded.encrypted_blob,
                    key_hint = excluded.key_hint,
                    metadata = excluded.metadata,
                    updated_at = excluded.updated_at,
                    updated_by = excluded.updated_by,
                    revoked_at = NULL,
                    revoked_by = NULL
                """,
                (
                    scope_norm,
                    int(scope_id),
                    provider_norm,
                    encrypted_blob,
                    key_hint,
                    metadata_json,
                    created_by,
                    updated_by,
                    updated_at.isoformat(),
                    updated_at.isoformat(),
                ),
            )
            row = await self.db_pool.fetchone(
                """
                SELECT id, scope_type, scope_id, provider, key_hint, metadata, created_at, updated_at, last_used_at,
                       created_by, updated_by, revoked_by, revoked_at
                FROM org_provider_secrets
                WHERE scope_type = ? AND scope_id = ? AND provider = ?
                """,
                (scope_norm, int(scope_id), provider_norm),
            )
            return self._row_to_dict(row) if row else {}
        except Exception as exc:
            logger.error(f"AuthnzOrgProviderSecretsRepo.upsert_secret failed: {exc}")
            raise

    async def fetch_secret(
        self,
        scope_type: str,
        scope_id: int,
        provider: str,
        *,
        include_revoked: bool = False,
    ) -> dict[str, Any] | None:
        scope_norm = _normalize_scope_type(scope_type)
        provider_norm = normalize_provider_name(provider)
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                revoked_clause = "" if include_revoked else " AND revoked_at IS NULL"
                # Bandit B105 false positive: SQL template, not a hardcoded secret.
                fetch_secret_sql_template = (  # nosec B105
                    """
                    SELECT id, scope_type, scope_id, provider, encrypted_blob, key_hint, metadata,
                           created_at, updated_at, last_used_at, created_by, updated_by, revoked_by, revoked_at
                    FROM org_provider_secrets
                    WHERE scope_type = $1 AND scope_id = $2 AND provider = $3{revoked_clause}
                    """
                )
                fetch_secret_sql = fetch_secret_sql_template.format_map(locals())  # nosec B608
                row = await self.db_pool.fetchone(
                    fetch_secret_sql,
                    scope_norm,
                    int(scope_id),
                    provider_norm,
                )
            else:
                revoked_clause = "" if include_revoked else " AND revoked_at IS NULL"
                # Bandit B105 false positive: SQL template, not a hardcoded secret.
                fetch_secret_sql_template = (  # nosec B105
                    """
                    SELECT id, scope_type, scope_id, provider, encrypted_blob, key_hint, metadata,
                           created_at, updated_at, last_used_at, created_by, updated_by, revoked_by, revoked_at
                    FROM org_provider_secrets
                    WHERE scope_type = ? AND scope_id = ? AND provider = ?{revoked_clause}
                    """
                )
                fetch_secret_sql = fetch_secret_sql_template.format_map(locals())  # nosec B608
                row = await self.db_pool.fetchone(
                    fetch_secret_sql,
                    (scope_norm, int(scope_id), provider_norm),
                )
            return self._row_to_dict(row) if row else None
        except Exception as exc:
            logger.error(f"AuthnzOrgProviderSecretsRepo.fetch_secret failed: {exc}")
            raise

    async def list_secrets(
        self,
        *,
        scope_type: str | None = None,
        scope_id: int | None = None,
        provider: str | None = None,
        include_revoked: bool = False,
    ) -> list[dict[str, Any]]:
        if scope_id is not None and not scope_type:
            raise ValueError("scope_type is required when scope_id is provided")

        scope_norm = _normalize_scope_type(scope_type) if scope_type else None
        provider_norm = normalize_provider_name(provider) if provider else None

        try:
            if getattr(self.db_pool, "pool", None) is not None:
                clauses = []
                params: list[Any] = []
                idx = 1
                if scope_norm:
                    clauses.append(f"scope_type = ${idx}")
                    params.append(scope_norm)
                    idx += 1
                if scope_id is not None:
                    clauses.append(f"scope_id = ${idx}")
                    params.append(int(scope_id))
                    idx += 1
                if provider_norm:
                    clauses.append(f"provider = ${idx}")
                    params.append(provider_norm)
                    idx += 1
                if not include_revoked:
                    clauses.append("revoked_at IS NULL")
                where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
                list_secrets_sql_template = """
                    SELECT id, scope_type, scope_id, provider, key_hint, metadata, created_at, updated_at, last_used_at,
                           created_by, updated_by, revoked_by, revoked_at
                    FROM org_provider_secrets
                    {where}
                    ORDER BY scope_type, scope_id, provider
                    """
                list_secrets_sql = list_secrets_sql_template.format_map(locals())  # nosec B608
                rows = await self.db_pool.fetchall(
                    list_secrets_sql,
                    *params,
                )
            else:
                clauses = []
                params = []
                if scope_norm:
                    clauses.append("scope_type = ?")
                    params.append(scope_norm)
                if scope_id is not None:
                    clauses.append("scope_id = ?")
                    params.append(int(scope_id))
                if provider_norm:
                    clauses.append("provider = ?")
                    params.append(provider_norm)
                if not include_revoked:
                    clauses.append("revoked_at IS NULL")
                where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
                list_secrets_sql_template = """
                    SELECT id, scope_type, scope_id, provider, key_hint, metadata, created_at, updated_at, last_used_at,
                           created_by, updated_by, revoked_by, revoked_at
                    FROM org_provider_secrets
                    {where}
                    ORDER BY scope_type, scope_id, provider
                    """
                list_secrets_sql = list_secrets_sql_template.format_map(locals())  # nosec B608
                rows = await self.db_pool.fetchall(
                    list_secrets_sql,
                    tuple(params),
                )

            return [self._row_to_dict(row) for row in rows]
        except Exception as exc:
            logger.error(f"AuthnzOrgProviderSecretsRepo.list_secrets failed: {exc}")
            raise

    async def delete_secret(
        self,
        scope_type: str,
        scope_id: int,
        provider: str,
        *,
        revoked_by: int | None = None,
        revoked_at: datetime | None = None,
    ) -> bool:
        scope_norm = _normalize_scope_type(scope_type)
        provider_norm = normalize_provider_name(provider)
        revoked_ts = revoked_at or datetime.now(timezone.utc)
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                ts = self._normalize_datetime_for_postgres(revoked_ts)
                result = await self.db_pool.execute(
                    """
                    UPDATE org_provider_secrets
                    SET revoked_at = $1, revoked_by = $2, updated_at = $1, updated_by = $3
                    WHERE scope_type = $4 AND scope_id = $5 AND provider = $6 AND revoked_at IS NULL
                    """,
                    ts,
                    revoked_by,
                    revoked_by,
                    scope_norm,
                    int(scope_id),
                    provider_norm,
                )
                if isinstance(result, str):
                    parts = result.split()
                    if parts and parts[-1].isdigit():
                        return int(parts[-1]) > 0
                return True

            cursor = await self.db_pool.execute(
                """
                UPDATE org_provider_secrets
                SET revoked_at = ?, revoked_by = ?, updated_at = ?, updated_by = ?
                WHERE scope_type = ? AND scope_id = ? AND provider = ? AND revoked_at IS NULL
                """,
                (
                    revoked_ts.isoformat(),
                    revoked_by,
                    revoked_ts.isoformat(),
                    revoked_by,
                    scope_norm,
                    int(scope_id),
                    provider_norm,
                ),
            )
            rowcount = getattr(cursor, "rowcount", 0)
            return rowcount > 0
        except Exception as exc:
            logger.error(f"AuthnzOrgProviderSecretsRepo.delete_secret failed: {exc}")
            raise

    async def touch_last_used(self, scope_type: str, scope_id: int, provider: str, used_at: datetime) -> None:
        scope_norm = _normalize_scope_type(scope_type)
        provider_norm = normalize_provider_name(provider)
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                ts = self._normalize_datetime_for_postgres(used_at)
                await self.db_pool.execute(
                    """
                    UPDATE org_provider_secrets
                    SET last_used_at = $1, updated_at = $1
                    WHERE scope_type = $2 AND scope_id = $3 AND provider = $4
                    """,
                    ts,
                    scope_norm,
                    int(scope_id),
                    provider_norm,
                )
                return

            await self.db_pool.execute(
                """
                UPDATE org_provider_secrets
                SET last_used_at = ?, updated_at = ?
                WHERE scope_type = ? AND scope_id = ? AND provider = ?
                """,
                (used_at.isoformat(), used_at.isoformat(), scope_norm, int(scope_id), provider_norm),
            )
        except Exception as exc:
            logger.error(f"AuthnzOrgProviderSecretsRepo.touch_last_used failed: {exc}")
            raise
