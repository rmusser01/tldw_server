from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
    ProviderCredentialAliasConflictError,
)
from tldw_Server_API.app.core.LLM_Calls.provider_identity import (
    canonical_provider_name,
    provider_lookup_names,
)


@dataclass
class AuthnzUserProviderSecretsRepo:
    """Repository for per-user provider secrets (BYOK)."""

    db_pool: DatabasePool

    async def ensure_tables(self) -> None:
        """Ensure user_provider_secrets schema exists."""
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
                    ensure_user_provider_secrets_pg,
                )

                ok = await ensure_user_provider_secrets_pg(self.db_pool)
                if not ok:
                    raise RuntimeError("PostgreSQL user_provider_secrets schema ensure failed")
                return

            row = await self.db_pool.fetchone(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='user_provider_secrets'"
            )
            if not row:
                raise RuntimeError(
                    "SQLite user_provider_secrets table is missing. "
                    "Run the AuthNZ migrations/bootstrap (see "
                    "'python -m tldw_Server_API.app.core.AuthNZ.initialize')."
                )
        except Exception as exc:
            logger.error(f"AuthnzUserProviderSecretsRepo.ensure_tables failed: {exc}")
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
                "User provider secret row key materialization failed; falling back to dict(row)"
            )
        return dict(row)

    async def upsert_secret(
        self,
        *,
        user_id: int,
        provider: str,
        encrypted_blob: str,
        key_hint: str | None,
        metadata: dict[str, Any] | None,
        updated_at: datetime,
        created_by: int | None = None,
        updated_by: int | None = None,
    ) -> dict[str, Any]:
        provider_norm = canonical_provider_name(provider)
        metadata_json = json.dumps(metadata) if metadata is not None else None
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                ts = self._normalize_datetime_for_postgres(updated_at)
                row = await self.db_pool.fetchone(
                    """
                    INSERT INTO user_provider_secrets (
                        user_id, provider, encrypted_blob, key_hint, metadata,
                        created_by, updated_by, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $8)
                    ON CONFLICT (user_id, provider) DO UPDATE SET
                        encrypted_blob = EXCLUDED.encrypted_blob,
                        key_hint = EXCLUDED.key_hint,
                        metadata = EXCLUDED.metadata,
                        updated_at = EXCLUDED.updated_at,
                        updated_by = EXCLUDED.updated_by,
                        revoked_at = NULL,
                        revoked_by = NULL
                    RETURNING id, user_id, provider, key_hint, metadata, created_at, updated_at, last_used_at,
                              created_by, updated_by, revoked_by, revoked_at
                    """,
                    user_id,
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
                INSERT INTO user_provider_secrets (
                    user_id, provider, encrypted_blob, key_hint, metadata,
                    created_by, updated_by, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, provider) DO UPDATE SET
                    encrypted_blob = excluded.encrypted_blob,
                    key_hint = excluded.key_hint,
                    metadata = excluded.metadata,
                    updated_at = excluded.updated_at,
                    updated_by = excluded.updated_by,
                    revoked_at = NULL,
                    revoked_by = NULL
                """,
                (
                    user_id,
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
                SELECT id, user_id, provider, key_hint, metadata, created_at, updated_at, last_used_at,
                       created_by, updated_by, revoked_by, revoked_at
                FROM user_provider_secrets
                WHERE user_id = ? AND provider = ?
                """,
                (user_id, provider_norm),
            )
            return self._row_to_dict(row) if row else {}
        except Exception as exc:
            logger.error(f"AuthnzUserProviderSecretsRepo.upsert_secret failed: {exc}")
            raise

    async def fetch_secret_for_user(
        self,
        user_id: int,
        provider: str,
        *,
        include_revoked: bool = False,
    ) -> dict[str, Any] | None:
        try:
            canonical, *legacy_names = provider_lookup_names(provider)
            row = await self._fetch_secret_for_exact_provider(
                user_id,
                canonical,
                include_revoked=True,
            )
            if row is not None:
                return row if include_revoked or row.get("revoked_at") is None else None

            legacy_rows = [
                legacy_row
                for legacy_name in legacy_names
                if (
                    legacy_row := await self._fetch_secret_for_exact_provider(
                        user_id,
                        legacy_name,
                        include_revoked=True,
                    )
                )
                is not None
            ]
            if len(legacy_rows) > 1:
                raise ProviderCredentialAliasConflictError("conflicting legacy provider credentials")
            if not legacy_rows:
                return None
            legacy_row = legacy_rows[0]
            return legacy_row if include_revoked or legacy_row.get("revoked_at") is None else None
        except Exception as exc:
            logger.error(f"AuthnzUserProviderSecretsRepo.fetch_secret_for_user failed: {exc}")
            raise

    async def update_secret_if_active_and_unchanged(
        self,
        *,
        user_id: int,
        provider: str,
        encrypted_blob: str,
        expected_encrypted_blob: str,
        key_hint: str | None,
        metadata: dict[str, Any] | None,
        updated_at: datetime,
        updated_by: int | None = None,
    ) -> bool:
        """Update an active secret only when its encrypted payload is unchanged."""

        provider_norm = canonical_provider_name(provider)
        metadata_json = json.dumps(metadata) if metadata is not None else None
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                ts = self._normalize_datetime_for_postgres(updated_at)
                row = await self.db_pool.fetchone(
                    """
                    UPDATE user_provider_secrets
                    SET encrypted_blob = $1, key_hint = $2, metadata = $3,
                        updated_at = $4, updated_by = $5
                    WHERE user_id = $6 AND provider = $7
                      AND revoked_at IS NULL AND encrypted_blob = $8
                    RETURNING id
                    """,
                    encrypted_blob,
                    key_hint,
                    metadata_json,
                    ts,
                    updated_by,
                    user_id,
                    provider_norm,
                    expected_encrypted_blob,
                )
                return row is not None

            cursor = await self.db_pool.execute(
                """
                UPDATE user_provider_secrets
                SET encrypted_blob = ?, key_hint = ?, metadata = ?,
                    updated_at = ?, updated_by = ?
                WHERE user_id = ? AND provider = ?
                  AND revoked_at IS NULL AND encrypted_blob = ?
                """,
                (
                    encrypted_blob,
                    key_hint,
                    metadata_json,
                    updated_at.isoformat(),
                    updated_by,
                    user_id,
                    provider_norm,
                    expected_encrypted_blob,
                ),
            )
            return getattr(cursor, "rowcount", 0) > 0
        except Exception as exc:
            logger.error("AuthnzUserProviderSecretsRepo.update_secret_if_active_and_unchanged " f"failed: {exc}")
            raise

    async def _fetch_secret_for_exact_provider(
        self,
        user_id: int,
        provider: str,
        *,
        include_revoked: bool,
    ) -> dict[str, Any] | None:
        revoked_clause = "" if include_revoked else " AND revoked_at IS NULL"
        if getattr(self.db_pool, "pool", None) is not None:
            sql_template = """
                SELECT id, user_id, provider, encrypted_blob, key_hint, metadata,
                       created_at, updated_at, last_used_at, created_by, updated_by, revoked_by, revoked_at
                FROM user_provider_secrets
                WHERE user_id = $1 AND provider = $2{revoked_clause}
                """
            sql = sql_template.format_map(locals())  # nosec B608
            row = await self.db_pool.fetchone(sql, user_id, provider)
        else:
            sql_template = """
                SELECT id, user_id, provider, encrypted_blob, key_hint, metadata,
                       created_at, updated_at, last_used_at, created_by, updated_by, revoked_by, revoked_at
                FROM user_provider_secrets
                WHERE user_id = ? AND provider = ?{revoked_clause}
                """
            sql = sql_template.format_map(locals())  # nosec B608
            row = await self.db_pool.fetchone(sql, (user_id, provider))
        return self._row_to_dict(row) if row else None

    async def list_secrets_for_user(
        self,
        user_id: int,
        *,
        include_revoked: bool = False,
    ) -> list[dict[str, Any]]:
        try:
            revoked_clause = "" if include_revoked else " AND revoked_at IS NULL"
            if getattr(self.db_pool, "pool", None) is not None:
                list_user_secrets_sql_template = """
                    SELECT id, user_id, provider, key_hint, metadata, created_at, updated_at, last_used_at,
                           created_by, updated_by, revoked_by, revoked_at
                    FROM user_provider_secrets
                    WHERE user_id = $1{revoked_clause}
                    ORDER BY provider
                    """
                list_user_secrets_sql = list_user_secrets_sql_template.format_map(locals())  # nosec B608
                rows = await self.db_pool.fetchall(
                    list_user_secrets_sql,
                    user_id,
                )
            else:
                list_user_secrets_sql_template = """
                    SELECT id, user_id, provider, key_hint, metadata, created_at, updated_at, last_used_at,
                           created_by, updated_by, revoked_by, revoked_at
                    FROM user_provider_secrets
                    WHERE user_id = ?{revoked_clause}
                    ORDER BY provider
                    """
                list_user_secrets_sql = list_user_secrets_sql_template.format_map(locals())  # nosec B608
                rows = await self.db_pool.fetchall(
                    list_user_secrets_sql,
                    (user_id,),
                )
            return [self._row_to_dict(row) for row in rows]
        except Exception as exc:
            logger.error(f"AuthnzUserProviderSecretsRepo.list_secrets_for_user failed: {exc}")
            raise

    async def delete_secret(
        self,
        user_id: int,
        provider: str,
        *,
        revoked_by: int | None = None,
        revoked_at: datetime | None = None,
    ) -> bool:
        revoked_ts = revoked_at or datetime.now(timezone.utc)
        try:
            row = await self.fetch_secret_for_user(user_id, provider)
            if row is None:
                return False
            stored_provider = str(row.get("provider") or "")
            if getattr(self.db_pool, "pool", None) is not None:
                ts = self._normalize_datetime_for_postgres(revoked_ts)
                result = await self.db_pool.execute(
                    """
                    UPDATE user_provider_secrets
                    SET revoked_at = $1, revoked_by = $2, updated_at = $1, updated_by = $3
                    WHERE user_id = $4 AND provider = $5 AND revoked_at IS NULL
                    """,
                    ts,
                    revoked_by,
                    revoked_by,
                    user_id,
                    stored_provider,
                )
                if isinstance(result, str):
                    parts = result.split()
                    if parts and parts[-1].isdigit():
                        return int(parts[-1]) > 0
                return True

            cursor = await self.db_pool.execute(
                """
                UPDATE user_provider_secrets
                SET revoked_at = ?, revoked_by = ?, updated_at = ?, updated_by = ?
                WHERE user_id = ? AND provider = ? AND revoked_at IS NULL
                """,
                (
                    revoked_ts.isoformat(),
                    revoked_by,
                    revoked_ts.isoformat(),
                    revoked_by,
                    user_id,
                    stored_provider,
                ),
            )
            rowcount = getattr(cursor, "rowcount", 0)
            return rowcount > 0
        except Exception as exc:
            logger.error(f"AuthnzUserProviderSecretsRepo.delete_secret failed: {exc}")
            raise

    async def touch_last_used(self, user_id: int, provider: str, used_at: datetime) -> None:
        try:
            row = await self.fetch_secret_for_user(user_id, provider)
            if row is None:
                return
            stored_provider = str(row.get("provider") or "")
            if getattr(self.db_pool, "pool", None) is not None:
                ts = self._normalize_datetime_for_postgres(used_at)
                await self.db_pool.execute(
                    """
                    UPDATE user_provider_secrets
                    SET last_used_at = $1, updated_at = $1
                    WHERE user_id = $2 AND provider = $3 AND revoked_at IS NULL
                    """,
                    ts,
                    user_id,
                    stored_provider,
                )
                return

            await self.db_pool.execute(
                """
                UPDATE user_provider_secrets
                SET last_used_at = ?, updated_at = ?
                WHERE user_id = ? AND provider = ? AND revoked_at IS NULL
                """,
                (used_at.isoformat(), used_at.isoformat(), user_id, stored_provider),
            )
        except Exception as exc:
            logger.error(f"AuthnzUserProviderSecretsRepo.touch_last_used failed: {exc}")
            raise
