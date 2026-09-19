from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.repos.datetime_utils import _strip_tzinfo


@dataclass
class AuthnzTokenBlacklistRepo:
    """
    Repository for AuthNZ token blacklist persistence.

    This repo centralizes the DB interactions for the ``token_blacklist``
    table so higher-level services (TokenBlacklist, SessionManager) do not
    need to embed backend-specific SQL or DDL logic.
    """

    db_pool: DatabasePool

    def _is_postgres_backend(self) -> bool:
        """Return True when the underlying DatabasePool is using PostgreSQL."""
        return bool(getattr(self.db_pool, "pool", None))

    async def ensure_schema(self) -> None:
        """
        Ensure the ``token_blacklist`` table and key indexes exist.

        This is an idempotent bootstrap/backstop helper for deployments and
        test setups that may not have run the full AuthNZ migrations yet.
        """
        ddl_postgres = [
            """
            CREATE TABLE IF NOT EXISTS token_blacklist (
                id SERIAL PRIMARY KEY,
                jti VARCHAR(255) UNIQUE NOT NULL,
                user_id INTEGER,
                token_type VARCHAR(50),
                revoked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                reason VARCHAR(255),
                revoked_by INTEGER,
                ip_address VARCHAR(45),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_blacklist_jti ON token_blacklist(jti)",
            "CREATE INDEX IF NOT EXISTS idx_blacklist_expires ON token_blacklist(expires_at)",
            "CREATE INDEX IF NOT EXISTS idx_blacklist_user ON token_blacklist(user_id)",
        ]

        ddl_sqlite = [
            """
            CREATE TABLE IF NOT EXISTS token_blacklist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                jti TEXT UNIQUE NOT NULL,
                user_id INTEGER,
                token_type TEXT,
                revoked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                reason TEXT,
                revoked_by INTEGER,
                ip_address TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_blacklist_jti ON token_blacklist(jti)",
            "CREATE INDEX IF NOT EXISTS idx_blacklist_expires ON token_blacklist(expires_at)",
            "CREATE INDEX IF NOT EXISTS idx_blacklist_user ON token_blacklist(user_id)",
        ]

        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    # IF NOT EXISTS does not serialize concurrent catalog writes.
                    # The transaction releases this lock on commit or rollback.
                    await conn.execute(
                        "SELECT pg_advisory_xact_lock(hashtext($1), hashtext($2))",
                        "authnz_schema",
                        "token_blacklist",
                    )
                    for sql in ddl_postgres:
                        await conn.execute(sql)
                else:
                    for sql in ddl_sqlite:
                        await conn.execute(sql)
                    try:
                        await conn.commit()
                    except Exception:
                        logger.debug(
                            "SQLite commit skipped (likely auto-committed by transaction shim)"
                        )
        except Exception as exc:
            logger.error(f"AuthnzTokenBlacklistRepo.ensure_schema failed: {exc}")
            raise

    async def insert_blacklisted_token(
        self,
        *,
        jti: str,
        user_id: int | None,
        token_type: str,
        expires_at: datetime,
        reason: str | None,
        revoked_by: int | None,
        ip_address: str | None,
    ) -> None:
        """
        Insert a blacklisted token row (or no-op on duplicate JTI).
        """
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    exp = _strip_tzinfo(expires_at)
                    await conn.execute(
                        """
                        INSERT INTO token_blacklist
                        (jti, user_id, token_type, expires_at, reason, revoked_by, ip_address)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT (jti) DO NOTHING
                        """,
                        jti,
                        user_id,
                        token_type,
                        exp,
                        reason,
                        revoked_by,
                        ip_address,
                    )
                else:
                    try:
                        await conn.execute(
                            """
                            INSERT OR IGNORE INTO token_blacklist
                            (jti, user_id, token_type, expires_at, reason, revoked_by, ip_address)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                jti,
                                user_id,
                                token_type,
                                expires_at.isoformat(),
                                reason,
                                revoked_by,
                                ip_address,
                            ),
                        )
                        try:
                            await conn.commit()
                        except Exception:
                            # Transaction shim may auto-commit; log at debug level
                            logger.debug(
                                "SQLite commit skipped (likely auto-committed by transaction shim)"
                            )
                    except Exception as sqlite_err:
                        if "FOREIGN KEY constraint failed" in str(sqlite_err):
                            logger.warning(
                                "FK constraint failed for user_id={} in token_blacklist; retrying with NULL user_id",
                                user_id,
                            )
                            try:
                                await conn.execute(
                                    """
                                    INSERT OR IGNORE INTO token_blacklist
                                    (jti, user_id, token_type, expires_at, reason, revoked_by, ip_address)
                                    VALUES (?, NULL, ?, ?, ?, ?, ?)
                                    """,
                                    (
                                        jti,
                                        token_type,
                                        expires_at.isoformat(),
                                        reason,
                                        revoked_by,
                                        ip_address,
                                    ),
                                )
                                try:
                                    await conn.commit()
                                except Exception:
                                    logger.debug(
                                        "SQLite commit skipped (likely auto-committed)"
                                    )
                            except Exception as inner_exc:
                                logger.error(
                                    "Retry insert with NULL user_id failed: {}", inner_exc
                                )
                                raise
                        else:
                            raise
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(
                f"AuthnzTokenBlacklistRepo.insert_blacklisted_token failed: {exc}"
            )
            raise

    async def get_active_expiry_for_jti(
        self, jti: str, now: datetime
    ) -> datetime | None:
        """
        Return the latest non-expired ``expires_at`` for a given JTI, or None.
        """
        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres_backend():
                    now_param = _strip_tzinfo(now)
                    row = await conn.fetchrow(
                        """
                        SELECT expires_at
                        FROM token_blacklist
                        WHERE jti = $1 AND expires_at > $2
                        ORDER BY expires_at DESC
                        LIMIT 1
                        """,
                        jti,
                        now_param,
                    )
                    return row["expires_at"] if row else None

                cursor = await conn.execute(
                    """
                    SELECT expires_at
                    FROM token_blacklist
                    WHERE jti = ? AND expires_at > ?
                    ORDER BY expires_at DESC
                    LIMIT 1
                    """,
                    (jti, now.isoformat()),
                )
                result = await cursor.fetchone()
                return result[0] if result else None
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(
                f"AuthnzTokenBlacklistRepo.get_active_expiry_for_jti failed: {exc}"
            )
            raise

    async def cleanup_expired(self, now: datetime) -> int:
        """
        Delete expired blacklist rows and return the approximate count.
        """
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    now_param = _strip_tzinfo(now)
                    rows = await conn.fetch(
                        "DELETE FROM token_blacklist WHERE expires_at < $1 RETURNING id",
                        now_param,
                    )
                    return len(rows or [])

                cursor = await conn.execute(
                    "DELETE FROM token_blacklist WHERE expires_at < ?",
                    (now.isoformat(),),
                )
                deleted = getattr(cursor, "rowcount", 0) or 0
                try:
                    await conn.commit()
                except Exception:
                    logger.debug(
                        "SQLite commit skipped (likely auto-committed by transaction shim)"
                    )
                return int(deleted)
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(
                f"AuthnzTokenBlacklistRepo.cleanup_expired failed: {exc}"
            )
            raise

    async def get_blacklist_stats(
        self,
        *,
        now: datetime,
        user_id: int | None = None,
    ) -> dict[str, Any]:
        """
        Return blacklist statistics for all users or a single user.
        """
        try:
            async with self.db_pool.acquire() as conn:
                now_param = _strip_tzinfo(now)
                if user_id:
                    if self._is_postgres_backend():
                        row = await conn.fetchrow(
                            """
                            SELECT
                                COUNT(*) as total,
                                SUM(CASE WHEN token_type = 'access' THEN 1 ELSE 0 END) as access_tokens,
                                SUM(CASE WHEN token_type = 'refresh' THEN 1 ELSE 0 END) as refresh_tokens,
                                MIN(revoked_at) as earliest_revocation,
                                MAX(revoked_at) as latest_revocation
                            FROM token_blacklist
                            WHERE user_id = $1 AND expires_at > $2
                            """,
                            user_id,
                            now_param,
                        )
                    else:
                        cursor = await conn.execute(
                            """
                            SELECT
                                COUNT(*) as total,
                                SUM(CASE WHEN token_type = 'access' THEN 1 ELSE 0 END) as access_tokens,
                                SUM(CASE WHEN token_type = 'refresh' THEN 1 ELSE 0 END) as refresh_tokens,
                                MIN(revoked_at) as earliest_revocation,
                                MAX(revoked_at) as latest_revocation
                            FROM token_blacklist
                            WHERE user_id = ? AND expires_at > ?
                            """,
                            (user_id, now.isoformat()),
                        )
                        row = await cursor.fetchone()
                else:
                    if self._is_postgres_backend():
                        row = await conn.fetchrow(
                            """
                            SELECT
                                COUNT(*) as total,
                                COUNT(DISTINCT user_id) as unique_users,
                                SUM(CASE WHEN token_type = 'access' THEN 1 ELSE 0 END) as access_tokens,
                                SUM(CASE WHEN token_type = 'refresh' THEN 1 ELSE 0 END) as refresh_tokens
                            FROM token_blacklist
                            WHERE expires_at > $1
                            """,
                            now_param,
                        )
                    else:
                        cursor = await conn.execute(
                            """
                            SELECT
                                COUNT(*) as total,
                                COUNT(DISTINCT user_id) as unique_users,
                                SUM(CASE WHEN token_type = 'access' THEN 1 ELSE 0 END) as access_tokens,
                                SUM(CASE WHEN token_type = 'refresh' THEN 1 ELSE 0 END) as refresh_tokens
                            FROM token_blacklist
                            WHERE expires_at > ?
                            """,
                            (now.isoformat(),),
                        )
                        row = await cursor.fetchone()

                if not row:
                    return {
                        "total": 0,
                        "unique_users": 0 if user_id is None else 1,
                        "access_tokens": 0,
                        "refresh_tokens": 0,
                    }

                if hasattr(row, "keys"):
                    stats: dict[str, Any] = dict(row)
                else:
                    if user_id:
                        stats = {
                            "total": row[0],
                            "access_tokens": row[1],
                            "refresh_tokens": row[2],
                            "earliest_revocation": row[3],
                            "latest_revocation": row[4],
                        }
                    else:
                        stats = {
                            "total": row[0],
                            "unique_users": row[1],
                            "access_tokens": row[2],
                            "refresh_tokens": row[3],
                        }
                if user_id is not None:
                    stats.setdefault("unique_users", 1 if stats.get("total") else 0)
                else:
                    stats.setdefault("unique_users", 0)
                stats.setdefault("access_tokens", 0)
                stats.setdefault("refresh_tokens", 0)
                return stats
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(
                f"AuthnzTokenBlacklistRepo.get_blacklist_stats failed: {exc}"
            )
            raise
