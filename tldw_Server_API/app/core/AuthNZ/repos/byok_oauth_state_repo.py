from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.repos.datetime_utils import _strip_tzinfo
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import normalize_provider_name


@dataclass
class AuthnzByokOAuthStateRepo:
    """Repository for BYOK OAuth state records used during provider callback flows."""

    db_pool: DatabasePool

    async def ensure_tables(self) -> None:
        """Ensure byok_oauth_state schema exists."""
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
                    ensure_byok_oauth_state_pg,
                )

                ok = await ensure_byok_oauth_state_pg(self.db_pool)
                if not ok:
                    raise RuntimeError("PostgreSQL byok_oauth_state schema ensure failed")
                return

            row = await self.db_pool.fetchone(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='byok_oauth_state'"
            )
            if not row:
                raise RuntimeError(
                    "SQLite byok_oauth_state table is missing. "
                    "Run the AuthNZ migrations/bootstrap (see "
                    "'python -m tldw_Server_API.app.core.AuthNZ.initialize')."
                )
        except Exception as exc:
            logger.error(f"AuthnzByokOAuthStateRepo.ensure_tables failed: {exc}")
            raise

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        if isinstance(row, dict):
            return dict(row)
        try:
            keys = row.keys()
            return {key: row[key] for key in keys}
        except Exception as row_keys_error:
            logger.bind(error_type=type(row_keys_error).__name__).debug(
                "BYOK OAuth state row key materialization failed; falling back to dict(row)"
            )
        return dict(row)

    @staticmethod
    def _command_touched_rows(result: Any) -> bool:
        if isinstance(result, str):
            parts = result.split()
            if parts and parts[-1].isdigit():
                return int(parts[-1]) > 0
            return True
        rowcount = getattr(result, "rowcount", None)
        if isinstance(rowcount, int):
            return rowcount != 0
        return True

    async def create_state(
        self,
        *,
        state: str,
        user_id: int,
        provider: str,
        auth_session_id: str,
        redirect_uri: str,
        pkce_verifier_encrypted: str,
        expires_at: datetime,
        return_path: str | None = None,
        created_at: datetime | None = None,
    ) -> dict[str, Any]:
        provider_norm = normalize_provider_name(provider)
        created_ts = created_at or datetime.now(timezone.utc)
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                expires_ts = _strip_tzinfo(expires_at)
                created_pg = _strip_tzinfo(created_ts)
                await self.db_pool.execute(
                    """
                    INSERT INTO byok_oauth_state (
                        state, user_id, provider, auth_session_id, redirect_uri,
                        pkce_verifier_encrypted, created_at, expires_at, consumed_at, return_path
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NULL, $9)
                    ON CONFLICT (state, user_id) DO UPDATE SET
                        provider = EXCLUDED.provider,
                        auth_session_id = EXCLUDED.auth_session_id,
                        redirect_uri = EXCLUDED.redirect_uri,
                        pkce_verifier_encrypted = EXCLUDED.pkce_verifier_encrypted,
                        created_at = EXCLUDED.created_at,
                        expires_at = EXCLUDED.expires_at,
                        consumed_at = NULL,
                        return_path = EXCLUDED.return_path
                    """,
                    state,
                    int(user_id),
                    provider_norm,
                    auth_session_id,
                    redirect_uri,
                    pkce_verifier_encrypted,
                    created_pg,
                    expires_ts,
                    return_path,
                )
                row = await self.db_pool.fetchone(
                    """
                    SELECT state, user_id, provider, auth_session_id, redirect_uri,
                           pkce_verifier_encrypted, created_at, expires_at, consumed_at, return_path
                    FROM byok_oauth_state
                    WHERE state = $1 AND user_id = $2 AND provider = $3
                    """,
                    state,
                    int(user_id),
                    provider_norm,
                )
                return self._row_to_dict(row) if row else {}

            await self.db_pool.execute(
                """
                INSERT INTO byok_oauth_state (
                    state, user_id, provider, auth_session_id, redirect_uri,
                    pkce_verifier_encrypted, created_at, expires_at, consumed_at, return_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
                ON CONFLICT(state, user_id) DO UPDATE SET
                    provider = excluded.provider,
                    auth_session_id = excluded.auth_session_id,
                    redirect_uri = excluded.redirect_uri,
                    pkce_verifier_encrypted = excluded.pkce_verifier_encrypted,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at,
                    consumed_at = NULL,
                    return_path = excluded.return_path
                """,
                (
                    state,
                    int(user_id),
                    provider_norm,
                    auth_session_id,
                    redirect_uri,
                    pkce_verifier_encrypted,
                    created_ts.isoformat(),
                    expires_at.isoformat(),
                    return_path,
                ),
            )
            row = await self.db_pool.fetchone(
                """
                SELECT state, user_id, provider, auth_session_id, redirect_uri,
                       pkce_verifier_encrypted, created_at, expires_at, consumed_at, return_path
                FROM byok_oauth_state
                WHERE state = ? AND user_id = ? AND provider = ?
                """,
                (state, int(user_id), provider_norm),
            )
            return self._row_to_dict(row) if row else {}
        except Exception as exc:
            logger.error(f"AuthnzByokOAuthStateRepo.create_state failed: {exc}")
            raise

    async def fetch_state(
        self,
        *,
        state: str,
        provider: str,
        now: datetime | None = None,
        include_consumed: bool = False,
        include_expired: bool = False,
    ) -> dict[str, Any] | None:
        provider_norm = normalize_provider_name(provider)
        lookup_time = now or datetime.now(timezone.utc)
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                params: list[Any] = [state, provider_norm]
                where = ["state = $1", "provider = $2"]
                idx = 3
                if not include_consumed:
                    where.append("consumed_at IS NULL")
                if not include_expired:
                    where.append(f"expires_at > ${idx}")
                    params.append(_strip_tzinfo(lookup_time))
                    idx += 1
                where_clause = " AND ".join(where)
                fetch_state_sql_template = """
                    SELECT state, user_id, provider, auth_session_id, redirect_uri,
                           pkce_verifier_encrypted, created_at, expires_at, consumed_at, return_path
                    FROM byok_oauth_state
                    WHERE {where_clause}
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                fetch_state_sql = fetch_state_sql_template.format_map(locals())  # nosec B608
                row = await self.db_pool.fetchone(
                    fetch_state_sql,
                    *params,
                )
                return self._row_to_dict(row) if row else None

            params_sqlite: list[Any] = [state, provider_norm]
            where_sqlite = ["state = ?", "provider = ?"]
            if not include_consumed:
                where_sqlite.append("consumed_at IS NULL")
            if not include_expired:
                where_sqlite.append("datetime(expires_at) > datetime(?)")
                params_sqlite.append(lookup_time.isoformat())
            where_clause = " AND ".join(where_sqlite)
            fetch_state_sql_template = """
                SELECT state, user_id, provider, auth_session_id, redirect_uri,
                       pkce_verifier_encrypted, created_at, expires_at, consumed_at, return_path
                FROM byok_oauth_state
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT 1
                """
            fetch_state_sql = fetch_state_sql_template.format_map(locals())  # nosec B608
            row = await self.db_pool.fetchone(
                fetch_state_sql,
                tuple(params_sqlite),
            )
            return self._row_to_dict(row) if row else None
        except Exception as exc:
            logger.error(f"AuthnzByokOAuthStateRepo.fetch_state failed: {exc}")
            raise

    async def consume_state(
        self,
        *,
        state: str,
        provider: str,
        consumed_at: datetime | None = None,
    ) -> dict[str, Any] | None:
        provider_norm = normalize_provider_name(provider)
        consume_ts = consumed_at or datetime.now(timezone.utc)
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                consume_pg = _strip_tzinfo(consume_ts)
                async with self.db_pool.transaction() as conn:
                    row = await conn.fetchrow(
                        """
                        SELECT state, user_id, provider, auth_session_id, redirect_uri,
                               pkce_verifier_encrypted, created_at, expires_at, consumed_at, return_path
                        FROM byok_oauth_state
                        WHERE state = $1
                          AND provider = $2
                          AND consumed_at IS NULL
                          AND expires_at > $3
                        ORDER BY created_at DESC
                        LIMIT 1
                        FOR UPDATE
                        """,
                        state,
                        provider_norm,
                        consume_pg,
                    )
                    if not row:
                        return None

                    result = await conn.execute(
                        """
                        UPDATE byok_oauth_state
                        SET consumed_at = $1
                        WHERE state = $2
                          AND user_id = $3
                          AND provider = $4
                          AND consumed_at IS NULL
                        """,
                        consume_pg,
                        row["state"],
                        int(row["user_id"]),
                        provider_norm,
                    )
                    if not self._command_touched_rows(result):
                        return None

                    item = self._row_to_dict(row)
                    item["consumed_at"] = consume_pg
                    return item

            async with self.db_pool.transaction() as conn:
                cursor = await conn.execute(
                    """
                    SELECT state, user_id, provider, auth_session_id, redirect_uri,
                           pkce_verifier_encrypted, created_at, expires_at, consumed_at, return_path
                    FROM byok_oauth_state
                    WHERE state = ?
                      AND provider = ?
                      AND consumed_at IS NULL
                      AND datetime(expires_at) > datetime(?)
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (state, provider_norm, consume_ts.isoformat()),
                )
                row = await cursor.fetchone()
                if not row:
                    return None

                update_cursor = await conn.execute(
                    """
                    UPDATE byok_oauth_state
                    SET consumed_at = ?
                    WHERE state = ?
                      AND user_id = ?
                      AND provider = ?
                      AND consumed_at IS NULL
                    """,
                    (
                        consume_ts.isoformat(),
                        row["state"],
                        int(row["user_id"]),
                        provider_norm,
                    ),
                )
                if not self._command_touched_rows(update_cursor):
                    return None

                item = self._row_to_dict(row)
                item["consumed_at"] = consume_ts.isoformat()
                return item
        except Exception as exc:
            logger.error(f"AuthnzByokOAuthStateRepo.consume_state failed: {exc}")
            raise

    async def count_outstanding(
        self,
        *,
        user_id: int,
        provider: str,
        now: datetime | None = None,
    ) -> int:
        provider_norm = normalize_provider_name(provider)
        lookup_time = now or datetime.now(timezone.utc)
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                row = await self.db_pool.fetchone(
                    """
                    SELECT COUNT(*) AS count
                    FROM byok_oauth_state
                    WHERE user_id = $1
                      AND provider = $2
                      AND consumed_at IS NULL
                      AND expires_at > $3
                    """,
                    int(user_id),
                    provider_norm,
                    _strip_tzinfo(lookup_time),
                )
            else:
                row = await self.db_pool.fetchone(
                    """
                    SELECT COUNT(*) AS count
                    FROM byok_oauth_state
                    WHERE user_id = ?
                      AND provider = ?
                      AND consumed_at IS NULL
                      AND datetime(expires_at) > datetime(?)
                    """,
                    (int(user_id), provider_norm, lookup_time.isoformat()),
                )
            if not row:
                return 0
            value = row["count"] if isinstance(row, dict) else row[0]
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return 0
            return parsed if parsed > 0 else 0
        except Exception as exc:
            logger.error(f"AuthnzByokOAuthStateRepo.count_outstanding failed: {exc}")
            raise

    async def enforce_outstanding_cap(
        self,
        *,
        user_id: int,
        provider: str,
        max_outstanding: int,
        now: datetime | None = None,
    ) -> int:
        cap = int(max_outstanding)
        if cap <= 0:
            return 0

        lookup_time = now or datetime.now(timezone.utc)
        keep_active = max(0, cap - 1)
        outstanding = await self.count_outstanding(
            user_id=int(user_id),
            provider=provider,
            now=lookup_time,
        )
        overflow = outstanding - keep_active
        if overflow <= 0:
            return 0

        provider_norm = normalize_provider_name(provider)
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                async with self.db_pool.transaction() as conn:
                    rows = await conn.fetch(
                        """
                        WITH stale AS (
                            SELECT state, user_id
                            FROM byok_oauth_state
                            WHERE user_id = $1
                              AND provider = $2
                              AND consumed_at IS NULL
                              AND expires_at > $3
                            ORDER BY created_at ASC
                            LIMIT $4
                        )
                        DELETE FROM byok_oauth_state target
                        USING stale
                        WHERE target.state = stale.state
                          AND target.user_id = stale.user_id
                        RETURNING target.state
                        """,
                        int(user_id),
                        provider_norm,
                        _strip_tzinfo(lookup_time),
                        int(overflow),
                    )
                    return len(rows or [])

            async with self.db_pool.transaction() as conn:
                cursor = await conn.execute(
                    """
                    DELETE FROM byok_oauth_state
                    WHERE rowid IN (
                        SELECT rowid
                        FROM byok_oauth_state
                        WHERE user_id = ?
                          AND provider = ?
                          AND consumed_at IS NULL
                          AND datetime(expires_at) > datetime(?)
                        ORDER BY created_at ASC
                        LIMIT ?
                    )
                    """,
                    (
                        int(user_id),
                        provider_norm,
                        lookup_time.isoformat(),
                        int(overflow),
                    ),
                )
                rowcount = getattr(cursor, "rowcount", 0)
                if isinstance(rowcount, int) and rowcount >= 0:
                    return rowcount
                changes_cursor = await conn.execute("SELECT changes()")
                change_row = await changes_cursor.fetchone()
                return int(change_row[0]) if change_row else 0
        except Exception as exc:
            logger.error(f"AuthnzByokOAuthStateRepo.enforce_outstanding_cap failed: {exc}")
            raise

    async def purge_expired(self, *, now: datetime | None = None) -> int:
        purge_time = now or datetime.now(timezone.utc)
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                purge_pg = _strip_tzinfo(purge_time)
                async with self.db_pool.transaction() as conn:
                    rows = await conn.fetch(
                        """
                        DELETE FROM byok_oauth_state
                        WHERE consumed_at IS NOT NULL OR expires_at < $1
                        RETURNING state, user_id
                        """,
                        purge_pg,
                    )
                    return len(rows or [])

            async with self.db_pool.transaction() as conn:
                cursor = await conn.execute(
                    """
                    DELETE FROM byok_oauth_state
                    WHERE consumed_at IS NOT NULL OR datetime(expires_at) < datetime(?)
                    """,
                    (purge_time.isoformat(),),
                )
                rowcount = getattr(cursor, "rowcount", 0)
                if isinstance(rowcount, int) and rowcount >= 0:
                    return rowcount
                changes_cursor = await conn.execute("SELECT changes()")
                change_row = await changes_cursor.fetchone()
                return int(change_row[0]) if change_row else 0
        except Exception as exc:
            logger.error(f"AuthnzByokOAuthStateRepo.purge_expired failed: {exc}")
            raise
