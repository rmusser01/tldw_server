from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.exceptions import TransactionError
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
    ProviderCredentialAliasConflictError,
    fold_provider_credential_rows,
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
            logger.bind(error_type=type(exc).__name__).error(
                "AuthnzUserProviderSecretsRepo.ensure_tables failed"
            )
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

    @staticmethod
    def _select_authoritative_row(
        rows: list[dict[str, Any]],
        canonical: str,
    ) -> dict[str, Any] | None:
        """Choose the authoritative stored row without rewriting its spelling."""
        canonical_rows = [
            row
            for row in rows
            if str(row.get("provider") or "").strip().lower() == canonical
        ]
        if len(canonical_rows) > 1:
            raise ProviderCredentialAliasConflictError(
                "conflicting canonical provider credentials"
            )
        if canonical_rows:
            return canonical_rows[0]
        if len(rows) > 1:
            raise ProviderCredentialAliasConflictError(
                "conflicting legacy provider credentials"
            )
        return rows[0] if rows else None

    async def _canonicalize_provider_identity(
        self,
        conn: Any,
        user_id: int,
        provider: str,
        *,
        postgres: bool,
    ) -> None:
        """Lock and collapse every stored spelling for one provider identity."""
        provider_norm = canonical_provider_name(provider)
        lookup_names = provider_lookup_names(provider_norm)

        if postgres:
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtext($1), hashtext($2))",
                "user_provider_secrets",
                f"{int(user_id)}:{provider_norm}",
            )
            rows = await conn.fetch(
                """
                SELECT provider, revoked_at
                FROM user_provider_secrets
                WHERE user_id = $1 AND provider = ANY($2::text[])
                ORDER BY provider
                FOR UPDATE
                """,
                int(user_id),
                list(lookup_names),
            )
        else:
            placeholders = ", ".join("?" for _ in lookup_names)
            cursor = await conn.execute(
                f"""
                SELECT provider, revoked_at
                FROM user_provider_secrets
                WHERE user_id = ? AND provider IN ({placeholders})
                ORDER BY provider
                """,  # nosec B608
                (int(user_id), *lookup_names),
            )
            rows = await cursor.fetchall()

        materialized_rows = [self._row_to_dict(row) for row in rows]
        fold_provider_credential_rows(materialized_rows, include_revoked=True)
        stored_names = {
            str(row.get("provider") or "") for row in materialized_rows
        }

        if provider_norm in stored_names:
            legacy_names = tuple(name for name in lookup_names if name != provider_norm)
            if not legacy_names:
                return
            if postgres:
                await conn.execute(
                    """
                    DELETE FROM user_provider_secrets
                    WHERE user_id = $1 AND provider = ANY($2::text[])
                    """,
                    int(user_id),
                    list(legacy_names),
                )
            else:
                legacy_placeholders = ", ".join("?" for _ in legacy_names)
                await conn.execute(
                    f"""
                    DELETE FROM user_provider_secrets
                    WHERE user_id = ? AND provider IN ({legacy_placeholders})
                    """,  # nosec B608
                    (int(user_id), *legacy_names),
                )
            return

        if not materialized_rows:
            return

        stored_provider = str(materialized_rows[0].get("provider") or "")
        if postgres:
            await conn.execute(
                """
                UPDATE user_provider_secrets
                SET provider = $1
                WHERE user_id = $2 AND provider = $3
                """,
                provider_norm,
                int(user_id),
                stored_provider,
            )
        else:
            await conn.execute(
                """
                UPDATE user_provider_secrets
                SET provider = ?
                WHERE user_id = ? AND provider = ?
                """,
                (provider_norm, int(user_id), stored_provider),
            )

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
            postgres = getattr(self.db_pool, "pool", None) is not None
            async with self.db_pool.transaction() as conn:
                await self._canonicalize_provider_identity(
                    conn,
                    user_id,
                    provider_norm,
                    postgres=postgres,
                )

                if postgres:
                    ts = self._normalize_datetime_for_postgres(updated_at)
                    row = await conn.fetchrow(
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

                await conn.execute(
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
                cursor = await conn.execute(
                    """
                    SELECT id, user_id, provider, key_hint, metadata, created_at, updated_at, last_used_at,
                           created_by, updated_by, revoked_by, revoked_at
                    FROM user_provider_secrets
                    WHERE user_id = ? AND provider = ?
                    """,
                    (user_id, provider_norm),
                )
                row = await cursor.fetchone()
                return self._row_to_dict(row) if row else {}
        except TransactionError as exc:
            if isinstance(exc.__cause__, ProviderCredentialAliasConflictError):
                raise exc.__cause__ from None
            logger.bind(error_type=type(exc).__name__).error(
                "AuthnzUserProviderSecretsRepo.upsert_secret failed"
            )
            raise
        except Exception as exc:
            logger.bind(error_type=type(exc).__name__).error(
                "AuthnzUserProviderSecretsRepo.upsert_secret failed"
            )
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
            rows = await self._fetch_secrets_for_providers(
                user_id,
                (canonical, *legacy_names),
                active_user=False,
            )
            row = self._select_authoritative_row(rows, canonical)
            return row if row is not None and (include_revoked or row.get("revoked_at") is None) else None
        except Exception as exc:
            logger.bind(error_type=type(exc).__name__).error(
                "AuthnzUserProviderSecretsRepo.fetch_secret_for_user failed"
            )
            raise

    async def fetch_secret_for_active_user(
        self,
        user_id: int,
        provider: str,
        *,
        include_revoked: bool = False,
    ) -> dict[str, Any] | None:
        """Fetch a user secret only while its owning user is active."""
        try:
            canonical, *legacy_names = provider_lookup_names(provider)
            rows = await self._fetch_secrets_for_providers(
                user_id,
                (canonical, *legacy_names),
                active_user=True,
            )
            row = self._select_authoritative_row(rows, canonical)
            return row if row is not None and (include_revoked or row.get("revoked_at") is None) else None
        except Exception as exc:
            logger.bind(error_type=type(exc).__name__).error(
                "AuthnzUserProviderSecretsRepo.fetch_secret_for_active_user failed"
            )
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
            postgres = getattr(self.db_pool, "pool", None) is not None
            async with self.db_pool.transaction() as conn:
                await self._canonicalize_provider_identity(
                    conn,
                    user_id,
                    provider_norm,
                    postgres=postgres,
                )
                if postgres:
                    ts = self._normalize_datetime_for_postgres(updated_at)
                    row = await conn.fetchrow(
                        """
                        UPDATE user_provider_secrets
                        SET encrypted_blob = $1, key_hint = $2, metadata = $3,
                            updated_at = $4, updated_by = $5
                        WHERE user_id = $6 AND provider = $7
                          AND revoked_at IS NULL AND encrypted_blob = $8
                          AND EXISTS (
                              SELECT 1
                              FROM users AS active_user
                              WHERE active_user.id = user_provider_secrets.user_id
                                AND active_user.is_active = TRUE
                          )
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

                cursor = await conn.execute(
                    """
                    UPDATE user_provider_secrets
                    SET encrypted_blob = ?, key_hint = ?, metadata = ?,
                        updated_at = ?, updated_by = ?
                    WHERE user_id = ? AND provider = ?
                      AND revoked_at IS NULL AND encrypted_blob = ?
                      AND EXISTS (
                          SELECT 1
                          FROM users AS active_user
                          WHERE active_user.id = user_provider_secrets.user_id
                            AND active_user.is_active = 1
                      )
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
        except TransactionError as exc:
            if isinstance(exc.__cause__, ProviderCredentialAliasConflictError):
                raise exc.__cause__ from None
            logger.bind(error_type=type(exc).__name__).error(
                "AuthnzUserProviderSecretsRepo.update_secret_if_active_and_unchanged failed"
            )
            raise
        except Exception as exc:
            logger.bind(error_type=type(exc).__name__).error(
                "AuthnzUserProviderSecretsRepo.update_secret_if_active_and_unchanged failed"
            )
            raise

    async def _fetch_secrets_for_providers(
        self,
        user_id: int,
        providers: tuple[str, ...],
        *,
        active_user: bool,
    ) -> list[dict[str, Any]]:
        """Read every alias for one user through a single database snapshot."""
        if getattr(self.db_pool, "pool", None) is not None:
            active_join = (
                """
                JOIN users u
                  ON u.id = s.user_id
                 AND u.is_active = TRUE
                """
                if active_user
                else ""
            )
            sql = f"""
                SELECT s.id, s.user_id, s.provider, s.encrypted_blob, s.key_hint,
                       s.metadata, s.created_at, s.updated_at, s.last_used_at,
                       s.created_by, s.updated_by, s.revoked_by, s.revoked_at
                FROM user_provider_secrets s
                {active_join}
                WHERE s.user_id = $1 AND s.provider = ANY($2::text[])
                """  # nosec B608
            rows = await self.db_pool.fetchall(sql, user_id, list(providers))
        else:
            active_join = (
                """
                JOIN users u
                  ON u.id = s.user_id
                 AND u.is_active = 1
                """
                if active_user
                else ""
            )
            placeholders = ", ".join("?" for _provider in providers)
            sql = f"""
                SELECT s.id, s.user_id, s.provider, s.encrypted_blob, s.key_hint,
                       s.metadata, s.created_at, s.updated_at, s.last_used_at,
                       s.created_by, s.updated_by, s.revoked_by, s.revoked_at
                FROM user_provider_secrets s
                {active_join}
                WHERE s.user_id = ? AND s.provider IN ({placeholders})
                """  # nosec B608
            rows = await self.db_pool.fetchall(sql, (user_id, *providers))
        return [self._row_to_dict(row) for row in rows]

    async def list_secrets_for_user(
        self,
        user_id: int,
        *,
        include_revoked: bool = False,
    ) -> list[dict[str, Any]]:
        try:
            revoked_clause = ""
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
            return fold_provider_credential_rows(
                [self._row_to_dict(row) for row in rows],
                include_revoked=include_revoked,
            )
        except Exception as exc:
            logger.bind(error_type=type(exc).__name__).error(
                "AuthnzUserProviderSecretsRepo.list_secrets_for_user failed"
            )
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
            provider_norm = canonical_provider_name(provider)
            postgres = getattr(self.db_pool, "pool", None) is not None
            async with self.db_pool.transaction() as conn:
                await self._canonicalize_provider_identity(
                    conn,
                    user_id,
                    provider_norm,
                    postgres=postgres,
                )
                if postgres:
                    ts = self._normalize_datetime_for_postgres(revoked_ts)
                    result = await conn.execute(
                        """
                        UPDATE user_provider_secrets
                        SET revoked_at = $1, revoked_by = $2, updated_at = $1, updated_by = $3
                        WHERE user_id = $4 AND provider = $5 AND revoked_at IS NULL
                        """,
                        ts,
                        revoked_by,
                        revoked_by,
                        user_id,
                        provider_norm,
                    )
                    if isinstance(result, str):
                        parts = result.split()
                        if parts and parts[-1].isdigit():
                            return int(parts[-1]) > 0
                    return True

                cursor = await conn.execute(
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
                        provider_norm,
                    ),
                )
                rowcount = getattr(cursor, "rowcount", 0)
                return rowcount > 0
        except TransactionError as exc:
            if isinstance(exc.__cause__, ProviderCredentialAliasConflictError):
                raise exc.__cause__ from None
            logger.bind(error_type=type(exc).__name__).error(
                "AuthnzUserProviderSecretsRepo.delete_secret failed"
            )
            raise
        except Exception as exc:
            logger.bind(error_type=type(exc).__name__).error(
                "AuthnzUserProviderSecretsRepo.delete_secret failed"
            )
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
            logger.bind(error_type=type(exc).__name__).error(
                "AuthnzUserProviderSecretsRepo.touch_last_used failed"
            )
            raise
