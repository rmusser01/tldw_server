from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.exceptions import DatabaseError


@dataclass
class AuthnzSessionsRepo:
    """
    Repository for AuthNZ session persistence.

    This repo centralizes the core read/write paths for the ``sessions``
    table so that SessionManager does not need to embed backend-specific
    SQL or DDL checks for PostgreSQL vs SQLite.
    """

    db_pool: DatabasePool

    def _is_postgres_backend(self) -> bool:
        """
        Return True when the underlying DatabasePool is using PostgreSQL.

        Backend routing should use pool state instead of inferring from
        connection method availability.
        """
        return bool(getattr(self.db_pool, "pool", None))

    @staticmethod
    def _normalize_datetime_for_postgres(dt: datetime | None) -> datetime | None:
        """Strip timezone info for PostgreSQL TIMESTAMP columns (not TIMESTAMPTZ)."""
        if dt is None:
            return None
        return dt.replace(tzinfo=None) if getattr(dt, "tzinfo", None) else dt

    def _normalize_session_details(self, details: dict[str, Any]) -> dict[str, Any]:
        """
        Normalize datetime fields across backends so callers always
        see ``expires_at`` and ``refresh_expires_at`` as datetime objects
        when they are valid ISO strings.
        """
        for field in ("expires_at", "refresh_expires_at"):
            value = details.get(field)
            if value is None:
                continue
            if isinstance(value, str):
                try:
                    details[field] = datetime.fromisoformat(value)
                except ValueError:
                    # Leave as-is on parse failure
                    pass
        return details

    async def create_session_record(
        self,
        *,
        user_id: int,
        token_hash: str,
        refresh_token_hash: str,
        encrypted_token: str,
        encrypted_refresh: str,
        expires_at: datetime,
        refresh_expires_at: datetime | None,
        ip_address: str,
        user_agent: str,
        device_id: str,
        access_jti: str | None,
        refresh_jti: str | None,
    ) -> int:
        """
        Insert a new session row and return its ``id``.
        """
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    exp = expires_at.replace(tzinfo=None) if getattr(expires_at, "tzinfo", None) else expires_at
                    ref = (
                        refresh_expires_at.replace(tzinfo=None)
                        if refresh_expires_at is not None and getattr(refresh_expires_at, "tzinfo", None)
                        else refresh_expires_at
                    )
                    session_id = await conn.fetchval(
                        """
                        INSERT INTO sessions (
                            user_id, token_hash, refresh_token_hash,
                            encrypted_token, encrypted_refresh,
                            expires_at, refresh_expires_at,
                            ip_address, user_agent, device_id,
                            access_jti, refresh_jti
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                        RETURNING id
                        """,
                        user_id,
                        token_hash,
                        refresh_token_hash,
                        encrypted_token,
                        encrypted_refresh,
                        exp,
                        ref,
                        ip_address,
                        user_agent,
                        device_id,
                        access_jti,
                        refresh_jti,
                    )
                    return int(session_id)

                cursor = await conn.execute(
                    """
                    INSERT INTO sessions (
                        user_id, token_hash, refresh_token_hash,
                        encrypted_token, encrypted_refresh,
                        expires_at, refresh_expires_at,
                        ip_address, user_agent, device_id,
                        access_jti, refresh_jti
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        token_hash,
                        refresh_token_hash,
                        encrypted_token,
                        encrypted_refresh,
                        expires_at.isoformat(),
                        refresh_expires_at.isoformat() if refresh_expires_at else None,
                        ip_address,
                        user_agent,
                        device_id,
                        access_jti,
                        refresh_jti,
                    ),
                )
                session_id = getattr(cursor, "lastrowid", None)
                if session_id is None:
                    raise RuntimeError("Failed to obtain session id for new session row")
                return int(session_id)
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzSessionsRepo.create_session_record failed: {exc}")
            raise

    async def revoke_session_record(
        self,
        *,
        session_id: int,
        expected_user_id: int | None = None,
        revoked_by: int | None,
        reason: str | None,
    ) -> dict[str, Any] | None:
        """
        Mark a single session as revoked and return its details for blacklist use.
        """
        try:
            async with self.db_pool.transaction() as conn:
                session_details: dict[str, Any] | None = None

                if self._is_postgres_backend():
                    if expected_user_id is None:
                        session_row = await conn.fetchrow(
                            """
                            SELECT id, user_id, access_jti, refresh_jti, expires_at, refresh_expires_at
                            FROM sessions
                            WHERE id = $1
                            """,
                            session_id,
                        )
                    else:
                        session_row = await conn.fetchrow(
                            """
                            SELECT id, user_id, access_jti, refresh_jti, expires_at, refresh_expires_at
                            FROM sessions
                            WHERE id = $1 AND user_id = $2
                            """,
                            session_id,
                            expected_user_id,
                        )
                    if session_row:
                        session_details = self._normalize_session_details(dict(session_row))

                    if expected_user_id is None:
                        await conn.execute(
                            """
                            UPDATE sessions
                            SET is_active = FALSE,
                                is_revoked = TRUE,
                                revoked_at = CURRENT_TIMESTAMP,
                                revoked_by = $2,
                                revoke_reason = $3
                            WHERE id = $1
                            """,
                            session_id,
                            revoked_by,
                            reason,
                        )
                    else:
                        await conn.execute(
                            """
                            UPDATE sessions
                            SET is_active = FALSE,
                                is_revoked = TRUE,
                                revoked_at = CURRENT_TIMESTAMP,
                                revoked_by = $3,
                                revoke_reason = $4
                            WHERE id = $1 AND user_id = $2
                            """,
                            session_id,
                            expected_user_id,
                            revoked_by,
                            reason,
                        )
                else:
                    if expected_user_id is None:
                        cursor = await conn.execute(
                            """
                            SELECT id, user_id, access_jti, refresh_jti, expires_at, refresh_expires_at
                            FROM sessions
                            WHERE id = ?
                            """,
                            (session_id,),
                        )
                    else:
                        cursor = await conn.execute(
                            """
                            SELECT id, user_id, access_jti, refresh_jti, expires_at, refresh_expires_at
                            FROM sessions
                            WHERE id = ? AND user_id = ?
                            """,
                            (session_id, expected_user_id),
                        )
                    row = await cursor.fetchone()
                    if row:
                        session_details = self._normalize_session_details(
                            {
                                "id": row[0],
                                "user_id": row[1],
                                "access_jti": row[2],
                                "refresh_jti": row[3],
                                "expires_at": row[4],
                                "refresh_expires_at": row[5],
                            }
                        )
                    if expected_user_id is None:
                        await conn.execute(
                            """
                            UPDATE sessions
                            SET is_active = 0,
                                is_revoked = 1,
                                revoked_at = datetime('now'),
                                revoked_by = ?,
                                revoke_reason = ?
                            WHERE id = ?
                            """,
                            (revoked_by, reason, session_id),
                        )
                    else:
                        await conn.execute(
                            """
                            UPDATE sessions
                            SET is_active = 0,
                                is_revoked = 1,
                                revoked_at = datetime('now'),
                                revoked_by = ?,
                                revoke_reason = ?
                            WHERE id = ? AND user_id = ?
                            """,
                            (revoked_by, reason, session_id, expected_user_id),
                        )

                return session_details
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.bind(
                operation="revoke_session_record",
                exception_type=type(exc).__name__,
            ).error("AuthNZ session persistence failed")
            raise DatabaseError("Session persistence operation failed") from None

    async def revoke_all_sessions_for_user(
        self,
        *,
        user_id: int,
        except_session_id: int | None = None,
        revoked_by: int | None = None,
        reason: str | None = None,
    ) -> int:
        """
        Mark all sessions for a user as revoked (optionally excluding one).

        Returns the approximate number of rows affected (best-effort).
        Note: Some backends may return 0 even when rows were updated; do not
        depend on this count for critical validation logic.
        """
        try:
            async with self.db_pool.transaction() as conn:
                affected = 0
                if self._is_postgres_backend():
                    if except_session_id:
                        result = await conn.execute(
                            """
                            UPDATE sessions
                            SET is_active = FALSE,
                                is_revoked = TRUE,
                                revoked_at = CURRENT_TIMESTAMP,
                                revoked_by = $3,
                                revoke_reason = $4
                            WHERE user_id = $1 AND id != $2
                            """,
                            user_id,
                            except_session_id,
                            revoked_by,
                            reason,
                        )
                    else:
                        result = await conn.execute(
                            """
                            UPDATE sessions
                            SET is_active = FALSE,
                                is_revoked = TRUE,
                                revoked_at = CURRENT_TIMESTAMP,
                                revoked_by = $2,
                                revoke_reason = $3
                            WHERE user_id = $1
                            """,
                            user_id,
                            revoked_by,
                            reason,
                        )
                    try:
                        affected = int(result.split()[-1]) if isinstance(result, str) else 0
                    except Exception:
                        affected = 0
                else:
                    if except_session_id:
                        cursor = await conn.execute(
                            """
                            UPDATE sessions
                            SET is_active = 0,
                                is_revoked = 1,
                                revoked_at = datetime('now'),
                                revoked_by = ?,
                                revoke_reason = ?
                            WHERE user_id = ? AND id != ?
                            """,
                            (revoked_by, reason, user_id, except_session_id),
                        )
                    else:
                        cursor = await conn.execute(
                            """
                            UPDATE sessions
                            SET is_active = 0,
                                is_revoked = 1,
                                revoked_at = datetime('now'),
                                revoked_by = ?,
                                revoke_reason = ?
                            WHERE user_id = ?
                            """,
                            (revoked_by, reason, user_id),
                        )
                    affected = getattr(cursor, "rowcount", 0) or 0

                return int(affected)
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.bind(
                operation="revoke_all_sessions_for_user",
                exception_type=type(exc).__name__,
            ).error("AuthNZ session persistence failed")
            raise DatabaseError("Session persistence operation failed") from None

    async def fetch_session_token_metadata_for_user(
        self,
        user_id: int,
        *,
        except_session_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch session token metadata for a user.

        Returns a list of mappings with ``id``, ``access_jti``, ``refresh_jti``,
        ``expires_at``, and ``refresh_expires_at`` suitable for bulk
        blacklist operations.
        """
        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres_backend():
                    if except_session_id is None:
                        rows = await conn.fetch(
                            """
                            SELECT id, access_jti, refresh_jti, expires_at, refresh_expires_at
                            FROM sessions
                            WHERE user_id = $1
                            """,
                            user_id,
                        )
                    else:
                        rows = await conn.fetch(
                            """
                            SELECT id, access_jti, refresh_jti, expires_at, refresh_expires_at
                            FROM sessions
                            WHERE user_id = $1 AND id != $2
                            """,
                            user_id,
                            except_session_id,
                        )
                    return [
                        self._normalize_session_details(dict(row))
                        for row in rows
                    ]

                if except_session_id is None:
                    cursor = await conn.execute(
                        """
                        SELECT id, access_jti, refresh_jti, expires_at, refresh_expires_at
                        FROM sessions
                        WHERE user_id = ?
                        """,
                        (user_id,),
                    )
                else:
                    cursor = await conn.execute(
                        """
                        SELECT id, access_jti, refresh_jti, expires_at, refresh_expires_at
                        FROM sessions
                        WHERE user_id = ? AND id != ?
                        """,
                        (user_id, except_session_id),
                    )
                sqlite_rows = await cursor.fetchall()
                sessions: list[dict[str, Any]] = []
                for row in sqlite_rows:
                    sessions.append(
                        self._normalize_session_details(
                            {
                                "id": row[0],
                                "access_jti": row[1],
                                "refresh_jti": row[2],
                                "expires_at": row[3],
                                "refresh_expires_at": row[4],
                            }
                        )
                    )
                return sessions
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(
                f"AuthnzSessionsRepo.fetch_session_token_metadata_for_user failed: {exc}"
            )
            raise

    async def has_revoked_session_for_token_hash_candidates(
        self,
        token_hash_candidates: list[str],
    ) -> bool:
        """
        Return True if any revoked session exists for the provided token hashes.

        Mirrors the blacklist check previously embedded in
        ``SessionManager.is_token_blacklisted`` while keeping the SQL
        backend-agnostic and centralized.
        """
        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres_backend():
                    primary_query = """
                        SELECT COUNT(*)
                        FROM sessions
                        WHERE is_revoked = TRUE
                          AND (token_hash = $1 OR refresh_token_hash = $1)
                    """
                    legacy_query = """
                        SELECT COUNT(*)
                        FROM sessions
                        WHERE token_hash = $1 AND is_revoked = TRUE
                    """
                    try:
                        for candidate in token_hash_candidates:
                            result = await conn.fetchval(primary_query, candidate)
                            if result:
                                return True
                    except Exception as exc:
                        logger.debug(
                            "has_revoked_session_for_token_hash_candidates: "
                            "falling back to legacy token_hash-only query (PostgreSQL): {}",
                            exc,
                        )
                        for candidate in token_hash_candidates:
                            result = await conn.fetchval(legacy_query, candidate)
                            if result:
                                return True
                    return False

                primary_query = """
                    SELECT COUNT(*)
                    FROM sessions
                    WHERE is_revoked = 1
                      AND (token_hash = ? OR refresh_token_hash = ?)
                """
                legacy_query = """
                    SELECT COUNT(*)
                    FROM sessions
                    WHERE token_hash = ? AND is_revoked = 1
                """
                try:
                    for candidate in token_hash_candidates:
                        cursor = await conn.execute(
                            primary_query,
                            (candidate, candidate),
                        )
                        row = await cursor.fetchone()
                        count = row[0] if row else 0
                        if count:
                            return True
                except Exception as exc:
                    logger.debug(
                        "has_revoked_session_for_token_hash_candidates: "
                        "falling back to legacy token_hash-only query (SQLite): {}",
                        exc,
                    )
                    for candidate in token_hash_candidates:
                        cursor = await conn.execute(
                            legacy_query,
                            (candidate,),
                        )
                        row = await cursor.fetchone()
                        count = row[0] if row else 0
                        if count:
                            return True

                return False
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(
                f"AuthnzSessionsRepo.has_revoked_session_for_token_hash_candidates failed: {exc}"
            )
            raise

    async def mark_sessions_revoked_for_user_with_audit(
        self,
        *,
        user_id: int,
        revoked_by: int | None,
        reason: str | None,
        except_session_id: int | None = None,
    ) -> int:
        """
        Mark all sessions for a user as revoked with audit metadata.

        This mirrors the semantics previously embedded in
        ``token_blacklist.revoke_all_user_tokens`` while keeping the
        logic backend-agnostic.
        """
        try:
            async with self.db_pool.transaction() as conn:
                affected = 0
                if self._is_postgres_backend():
                    if except_session_id is None:
                        result = await conn.execute(
                            """
                            UPDATE sessions
                            SET is_revoked = TRUE,
                                is_active = FALSE,
                                revoked_at = COALESCE(revoked_at, CURRENT_TIMESTAMP),
                                revoked_by = COALESCE($2, revoked_by),
                                revoke_reason = COALESCE($3, revoke_reason)
                            WHERE user_id = $1
                            """,
                            user_id,
                            revoked_by,
                            reason,
                        )
                    else:
                        result = await conn.execute(
                            """
                            UPDATE sessions
                            SET is_revoked = TRUE,
                                is_active = FALSE,
                                revoked_at = COALESCE(revoked_at, CURRENT_TIMESTAMP),
                                revoked_by = COALESCE($3, revoked_by),
                                revoke_reason = COALESCE($4, revoke_reason)
                            WHERE user_id = $1 AND id != $2
                            """,
                            user_id,
                            except_session_id,
                            revoked_by,
                            reason,
                        )
                    try:
                        affected = int(result.split()[-1]) if isinstance(result, str) else 0
                    except Exception:
                        affected = 0
                else:
                    if except_session_id is None:
                        cursor = await conn.execute(
                            """
                            UPDATE sessions
                            SET is_revoked = 1,
                                is_active = 0,
                                revoked_at = COALESCE(revoked_at, CURRENT_TIMESTAMP),
                                revoked_by = COALESCE(?, revoked_by),
                                revoke_reason = COALESCE(?, revoke_reason)
                            WHERE user_id = ?
                            """,
                            (revoked_by, reason, user_id),
                        )
                    else:
                        cursor = await conn.execute(
                            """
                            UPDATE sessions
                            SET is_revoked = 1,
                                is_active = 0,
                                revoked_at = COALESCE(revoked_at, CURRENT_TIMESTAMP),
                                revoked_by = COALESCE(?, revoked_by),
                                revoke_reason = COALESCE(?, revoke_reason)
                            WHERE user_id = ? AND id != ?
                            """,
                            (revoked_by, reason, user_id, except_session_id),
                        )
                    affected = getattr(cursor, "rowcount", 0) or 0

                return int(affected)
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(
                f"AuthnzSessionsRepo.mark_sessions_revoked_for_user_with_audit failed: {exc}"
            )
            raise

    async def get_active_sessions_for_user(self, user_id: int) -> list[dict[str, Any]]:
        """
        Return active sessions for a user, ordered by last activity.
        """
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                rows = await self.db_pool.fetch(
                    """
                    SELECT id, ip_address, user_agent, device_id,
                           created_at, last_activity, expires_at
                    FROM sessions
                    WHERE user_id = $1 AND is_active = TRUE
                    ORDER BY last_activity DESC
                    """,
                    user_id,
                )
                return [dict(r) for r in rows]

            # SQLite path – support deployments where ``last_activity`` has not
            # yet been added by selecting a synthetic last-activity column.
            async with self.db_pool.acquire() as conn:
                try:
                    cursor = await conn.execute(
                        """
                        SELECT id, ip_address, user_agent, device_id,
                               created_at, last_activity, expires_at
                        FROM sessions
                        WHERE user_id = ? AND is_active = 1
                        ORDER BY last_activity DESC
                        """,
                        (user_id,),
                    )
                except Exception as exc:
                    msg = str(exc).lower()
                    if "no such column" in msg and "last_activity" in msg:
                        cursor = await conn.execute(
                            """
                            SELECT id, ip_address, user_agent, device_id,
                                   created_at, created_at AS last_activity, expires_at
                            FROM sessions
                            WHERE user_id = ? AND is_active = 1
                            ORDER BY created_at DESC
                            """,
                            (user_id,),
                        )
                    else:
                        raise

                rows = await cursor.fetchall()
                sessions: list[dict[str, Any]] = []
                for row in rows:
                    sessions.append(
                        {
                            "id": row[0],
                            "ip_address": row[1],
                            "user_agent": row[2],
                            "device_id": row[3],
                            "created_at": row[4],
                            "last_activity": row[5],
                            "expires_at": row[6],
                        }
                    )
                return sessions
        except DatabaseError:
            raise
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.bind(
                operation="get_active_sessions_for_user",
                exception_type=type(exc).__name__,
            ).error("AuthNZ session read failed")
            raise DatabaseError("Session read operation failed") from None

    async def cleanup_expired_sessions(self) -> int:
        """
        Delete expired or long-revoked sessions.

        Returns the number of deleted rows (best-effort).
        """
        try:
            async with self.db_pool.transaction() as conn:
                # First check if the sessions table exists
                if self._is_postgres_backend():
                    table_exists = await conn.fetchval(
                        """
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_name = 'sessions'
                        )
                        """
                    )
                else:
                    cursor = await conn.execute(
                        """
                        SELECT name FROM sqlite_master
                        WHERE type='table' AND name='sessions'
                        """
                    )
                    result = await cursor.fetchone()
                    table_exists = result is not None

                if not table_exists:
                    logger.debug("Sessions table does not exist, skipping cleanup")
                    return 0

                deleted = 0
                if self._is_postgres_backend():
                    rows = await conn.fetch(
                        """
                        DELETE FROM sessions
                        WHERE expires_at < CURRENT_TIMESTAMP - INTERVAL '1 day'
                        OR (is_active = FALSE AND revoked_at < CURRENT_TIMESTAMP - INTERVAL '7 days')
                        RETURNING id
                        """
                    )
                    deleted = len(rows or [])
                else:
                    cursor = await conn.execute(
                        """
                        DELETE FROM sessions
                        WHERE datetime(expires_at) < datetime('now', '-1 day')
                        OR (is_active = 0 AND datetime(revoked_at) < datetime('now', '-7 days'))
                        """
                    )
                    deleted = getattr(cursor, "rowcount", 0) or 0

                return int(deleted or 0)
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(
                f"AuthnzSessionsRepo.cleanup_expired_sessions failed: {exc}"
            )
            raise

    async def update_last_activity(self, session_id: int) -> None:
        """
        Best-effort last-activity update for a session.
        """
        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres_backend():
                    await conn.execute(
                        "UPDATE sessions SET last_activity = CURRENT_TIMESTAMP WHERE id = $1",
                        session_id,
                    )
                else:
                    await conn.execute(
                        "UPDATE sessions SET last_activity = datetime('now') WHERE id = ?",
                        (session_id,),
                    )
        except Exception:
            # Do not fail callers on activity update errors
            return

    async def fetch_session_for_validation_by_id(
        self,
        session_id: int,
    ) -> dict[str, Any] | None:
        """
        Fetch an active, non-expired session joined with user state by session id.

        Mirrors the previous SessionManager._fetch_session_record(session_id=...)
        semantics, including the user_active flag needed for validation.
        """
        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres_backend():
                    row = await conn.fetchrow(
                        """
                        SELECT s.id,
                               s.token_hash,
                               s.user_id,
                               s.device_id,
                               s.expires_at,
                               s.is_active,
                               s.revoked_at,
                               u.username,
                               u.role,
                               u.is_active AS user_active
                        FROM sessions s
                        JOIN users u ON s.user_id = u.id
                        WHERE s.id = $1
                          AND s.is_active = TRUE
                          AND s.expires_at > CURRENT_TIMESTAMP
                        """,
                        session_id,
                    )
                    return dict(row) if row else None

                cursor = await conn.execute(
                    """
                    SELECT s.id,
                           s.token_hash,
                           s.user_id,
                           s.device_id,
                           s.expires_at,
                           s.is_active,
                           s.revoked_at,
                           u.username,
                           u.role,
                           u.is_active AS user_active
                    FROM sessions s
                    JOIN users u ON s.user_id = u.id
                    WHERE s.id = ?
                      AND s.is_active = 1
                      AND datetime(s.expires_at) > datetime('now')
                    """,
                    (session_id,),
                )
                row = await cursor.fetchone()
                if not row:
                    return None
                return {
                    "id": row[0],
                    "token_hash": row[1],
                    "user_id": row[2],
                    "device_id": row[3],
                    "expires_at": row[4],
                    "is_active": row[5],
                    "revoked_at": row[6],
                    "username": row[7],
                    "role": row[8],
                    "user_active": row[9],
                }
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(
                f"AuthnzSessionsRepo.fetch_session_for_validation_by_id failed: {exc}"
            )
            raise

    async def fetch_session_for_validation_by_token_hash(
        self,
        token_hash: str,
    ) -> dict[str, Any] | None:
        """
        Fetch an active, non-expired session joined with user state by token hash.

        Mirrors the previous SessionManager._fetch_session_record(token_hash=...)
        semantics used during session validation.
        """
        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres_backend():
                    row = await conn.fetchrow(
                        """
                        SELECT s.id,
                               s.token_hash,
                               s.user_id,
                               s.device_id,
                               s.expires_at,
                               s.is_active,
                               s.revoked_at,
                               u.username,
                               u.role,
                               u.is_active AS user_active
                        FROM sessions s
                        JOIN users u ON s.user_id = u.id
                        WHERE s.token_hash = $1
                          AND s.is_active = TRUE
                          AND s.expires_at > CURRENT_TIMESTAMP
                        """,
                        token_hash,
                    )
                    return dict(row) if row else None

                cursor = await conn.execute(
                    """
                    SELECT s.id,
                           s.token_hash,
                           s.user_id,
                           s.device_id,
                           s.expires_at,
                           s.is_active,
                           s.revoked_at,
                           u.username,
                           u.role,
                           u.is_active AS user_active
                    FROM sessions s
                    JOIN users u ON s.user_id = u.id
                    WHERE s.token_hash = ?
                      AND s.is_active = 1
                      AND datetime(s.expires_at) > datetime('now')
                    """,
                    (token_hash,),
                )
                row = await cursor.fetchone()
                if not row:
                    return None
                return {
                    "id": row[0],
                    "token_hash": row[1],
                    "user_id": row[2],
                    "device_id": row[3],
                    "expires_at": row[4],
                    "is_active": row[5],
                    "revoked_at": row[6],
                    "username": row[7],
                    "role": row[8],
                    "user_active": row[9],
                }
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(
                f"AuthnzSessionsRepo.fetch_session_for_validation_by_token_hash failed: {exc}"
            )
            raise

    async def normalize_session_token_hash(
        self,
        *,
        session_id: int,
        new_token_hash: str,
    ) -> None:
        """
        Normalize a session's token_hash to the canonical value.

        This is used when a legacy hash candidate matched during validation and
        we want to store the primary hash going forward.
        """
        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres_backend():
                    await conn.execute(
                        "UPDATE sessions SET token_hash = $1 WHERE id = $2",
                        new_token_hash,
                        session_id,
                    )
                else:
                    await conn.execute(
                        "UPDATE sessions SET token_hash = ? WHERE id = ?",
                        (new_token_hash, session_id),
                    )
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(
                f"AuthnzSessionsRepo.normalize_session_token_hash failed: {exc}"
            )
            raise

    async def update_session_tokens_after_creation(
        self,
        *,
        session_id: int,
        access_token_hash: str,
        refresh_token_hash: str,
        access_jti: str | None,
        refresh_jti: str | None,
        access_expires_at: datetime | None,
        refresh_expires_at: datetime | None,
        encrypted_access_token: str,
        encrypted_refresh_token: str,
    ) -> int | None:
        """
        Update a newly created session with finalized token hashes and encrypted tokens.

        This mirrors the behavior previously embedded in
        ``SessionManager.update_session_tokens`` and returns the associated user_id
        when available.
        """
        try:
            async with self.db_pool.transaction() as conn:
                session_row: Any | None = None

                if self._is_postgres_backend():
                    # Normalize datetimes for PostgreSQL TIMESTAMP columns
                    pg_access_expires = self._normalize_datetime_for_postgres(access_expires_at)
                    pg_refresh_expires = self._normalize_datetime_for_postgres(refresh_expires_at)
                    await conn.execute(
                        """
                        UPDATE sessions
                        SET token_hash = $2,
                            refresh_token_hash = $3,
                            access_jti = COALESCE($4, access_jti),
                            refresh_jti = COALESCE($5, refresh_jti),
                            expires_at = COALESCE($6, expires_at),
                            refresh_expires_at = COALESCE($7, refresh_expires_at),
                            encrypted_token = $8,
                            encrypted_refresh = $9
                        WHERE id = $1
                        """,
                        session_id,
                        access_token_hash,
                        refresh_token_hash,
                        access_jti,
                        refresh_jti,
                        pg_access_expires,
                        pg_refresh_expires,
                        encrypted_access_token,
                        encrypted_refresh_token,
                    )
                    session_row = await conn.fetchrow(
                        "SELECT user_id FROM sessions WHERE id = $1",
                        session_id,
                    )
                else:
                    await conn.execute(
                        """
                        UPDATE sessions
                        SET token_hash = ?,
                            refresh_token_hash = ?,
                            access_jti = COALESCE(?, access_jti),
                            refresh_jti = COALESCE(?, refresh_jti),
                            expires_at = COALESCE(?, expires_at),
                            refresh_expires_at = COALESCE(?, refresh_expires_at),
                            encrypted_token = ?,
                            encrypted_refresh = ?
                        WHERE id = ?
                        """,
                        (
                            access_token_hash,
                            refresh_token_hash,
                            access_jti,
                            refresh_jti,
                            access_expires_at.isoformat()
                            if access_expires_at
                            else None,
                            refresh_expires_at.isoformat()
                            if refresh_expires_at
                            else None,
                            encrypted_access_token,
                            encrypted_refresh_token,
                            session_id,
                        ),
                    )
                    cursor = await conn.execute(
                        "SELECT user_id FROM sessions WHERE id = ?",
                        (session_id,),
                    )
                    session_row = await cursor.fetchone()

                if not session_row:
                    return None

                if isinstance(session_row, dict):
                    return session_row.get("user_id")
                if hasattr(session_row, "get"):
                    return session_row.get("user_id")
                return session_row[0]
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(
                f"AuthnzSessionsRepo.update_session_tokens_after_creation failed: {exc}"
            )
            raise

    async def find_active_session_by_refresh_hash_candidates(
        self,
        refresh_hash_candidates: list[str],
    ) -> dict[str, Any] | None:
        """
        Locate an active, unexpired session by trying multiple refresh-token-hash candidates.

        Used by SessionManager.refresh_session() to support legacy hash formats.
        Returns a mapping containing ``id``, ``user_id``, ``token_hash``, and
        ``refresh_token_hash`` or ``None``.
        """
        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres_backend():
                    for candidate in refresh_hash_candidates:
                        row = await conn.fetchrow(
                            """
                            SELECT id, user_id, token_hash, refresh_token_hash
                            FROM sessions
                            WHERE refresh_token_hash = $1
                              AND is_active = TRUE
                              AND refresh_expires_at IS NOT NULL
                              AND refresh_expires_at > CURRENT_TIMESTAMP
                            """,
                            candidate,
                        )
                        if row:
                            data = dict(row)
                            return {
                                "id": data["id"],
                                "user_id": data["user_id"],
                                "token_hash": data.get("token_hash"),
                                "refresh_token_hash": data.get("refresh_token_hash"),
                            }
                    else:
                        return None
                else:
                    for candidate in refresh_hash_candidates:
                        cursor = await conn.execute(
                            """
                            SELECT id, user_id, token_hash, refresh_token_hash
                            FROM sessions
                            WHERE refresh_token_hash = ?
                              AND is_active = 1
                              AND refresh_expires_at IS NOT NULL
                              AND datetime(refresh_expires_at) > datetime('now')
                            """,
                            (candidate,),
                        )
                        row = await cursor.fetchone()
                        if row:
                            return {
                                "id": row[0],
                                "user_id": row[1],
                                "token_hash": row[2],
                                "refresh_token_hash": row[3],
                            }
                    else:
                        return None
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(
                f"AuthnzSessionsRepo.find_active_session_by_refresh_hash_candidates failed: {exc}"
            )
            raise

    async def update_session_tokens_for_refresh(
        self,
        *,
        session_id: int,
        expected_access_hash: str,
        expected_refresh_hash: str,
        new_access_hash: str,
        access_jti: str | None,
        expires_at: datetime,
        encrypted_access_token: str,
        refresh_hash_update: str,
        refresh_jti: str | None,
        refresh_expires_at: datetime | None,
        encrypted_refresh_token: str,
    ) -> bool:
        """
        Update a session row with refreshed access/refresh token material.

        Uses compare-and-swap semantics to prevent concurrent refresh requests
        from both succeeding: the row is only updated if both expected token
        hashes still match the current session row at write time.

        Returns True when exactly one row was updated, False otherwise.
        """
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    exp = expires_at.replace(tzinfo=None) if getattr(expires_at, "tzinfo", None) else expires_at
                    ref = (
                        refresh_expires_at.replace(tzinfo=None)
                        if refresh_expires_at is not None and getattr(refresh_expires_at, "tzinfo", None)
                        else refresh_expires_at
                    )
                    row = await conn.fetchrow(
                        """
                        UPDATE sessions
                        SET token_hash = $2,
                            access_jti = COALESCE($3, access_jti),
                            expires_at = $4,
                            encrypted_token = $5,
                            refresh_token_hash = COALESCE($6, refresh_token_hash),
                            refresh_jti = COALESCE($7, refresh_jti),
                            refresh_expires_at = COALESCE($8, refresh_expires_at),
                            encrypted_refresh = COALESCE($9, encrypted_refresh),
                            last_activity = CURRENT_TIMESTAMP
                        WHERE id = $1
                          AND token_hash = $10
                          AND refresh_token_hash = $11
                          AND is_active = TRUE
                          AND refresh_expires_at IS NOT NULL
                          AND refresh_expires_at > CURRENT_TIMESTAMP
                        RETURNING id
                        """,
                        session_id,
                        new_access_hash,
                        access_jti,
                        exp,
                        encrypted_access_token,
                        refresh_hash_update,
                        refresh_jti,
                        ref,
                        encrypted_refresh_token,
                        expected_access_hash,
                        expected_refresh_hash,
                    )
                    return row is not None
                else:
                    cursor = None
                    try:
                        cursor = await conn.execute(
                            """
                            UPDATE sessions
                            SET token_hash = ?,
                                access_jti = COALESCE(?, access_jti),
                                expires_at = ?,
                                encrypted_token = ?,
                                refresh_token_hash = COALESCE(?, refresh_token_hash),
                                refresh_jti = COALESCE(?, refresh_jti),
                                refresh_expires_at = COALESCE(?, refresh_expires_at),
                                encrypted_refresh = COALESCE(?, encrypted_refresh),
                                last_activity = datetime('now')
                            WHERE id = ?
                              AND token_hash = ?
                              AND refresh_token_hash = ?
                              AND is_active = 1
                              AND refresh_expires_at IS NOT NULL
                              AND datetime(refresh_expires_at) > datetime('now')
                            """,
                            (
                                new_access_hash,
                                access_jti,
                                expires_at.isoformat(),
                                encrypted_access_token,
                                refresh_hash_update,
                                refresh_jti,
                                refresh_expires_at.isoformat()
                                if refresh_expires_at
                                else None,
                                encrypted_refresh_token,
                                session_id,
                                expected_access_hash,
                                expected_refresh_hash,
                            ),
                        )
                    except Exception as exc:
                        msg = str(exc).lower()
                        if "no such column" in msg and "last_activity" in msg:
                            cursor = await conn.execute(
                                """
                                UPDATE sessions
                                SET token_hash = ?,
                                    access_jti = COALESCE(?, access_jti),
                                    expires_at = ?,
                                    encrypted_token = ?,
                                    refresh_token_hash = COALESCE(?, refresh_token_hash),
                                    refresh_jti = COALESCE(?, refresh_jti),
                                    refresh_expires_at = COALESCE(?, refresh_expires_at),
                                    encrypted_refresh = COALESCE(?, encrypted_refresh)
                                WHERE id = ?
                                  AND token_hash = ?
                                  AND refresh_token_hash = ?
                                  AND is_active = 1
                                  AND refresh_expires_at IS NOT NULL
                                  AND datetime(refresh_expires_at) > datetime('now')
                                """,
                                (
                                    new_access_hash,
                                    access_jti,
                                    expires_at.isoformat(),
                                    encrypted_access_token,
                                    refresh_hash_update,
                                    refresh_jti,
                                    refresh_expires_at.isoformat()
                                    if refresh_expires_at
                                    else None,
                                    encrypted_refresh_token,
                                    session_id,
                                    expected_access_hash,
                                    expected_refresh_hash,
                                ),
                            )
                        else:
                            raise
                    if cursor is None:
                        return False
                    rowcount = getattr(cursor, "rowcount", -1)
                    if isinstance(rowcount, int) and rowcount >= 0:
                        return rowcount > 0
                    changes_cursor = await conn.execute("SELECT changes()")
                    changes_row = await changes_cursor.fetchone()
                    if not changes_row:
                        return False
                    return int(changes_row[0]) > 0
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(
                f"AuthnzSessionsRepo.update_session_tokens_for_refresh failed: {exc}"
            )
            raise
