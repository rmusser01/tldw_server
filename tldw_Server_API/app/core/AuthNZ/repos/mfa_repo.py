from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.exceptions import DatabaseError, UserNotFoundError
from tldw_Server_API.app.core.AuthNZ.profile_version import (
    ProfileVersionNotFound,
    VersionedUserWriteGateway,
)


@dataclass
class AuthnzMfaRepo:
    """
    Repository for MFA-related fields stored on the ``users`` table.

    This repo centralizes the small set of reads and updates used by
    ``MFAService`` so that backend-specific SQL for PostgreSQL vs SQLite
    is not embedded directly in the service logic.
    """

    db_pool: DatabasePool

    def _is_postgres_backend(self) -> bool:
        """Return True when the underlying DatabasePool is using PostgreSQL."""
        return bool(getattr(self.db_pool, "pool", None))

    @staticmethod
    def _normalize_datetime_for_postgres(dt: datetime) -> datetime:
        """Return an aware UTC value for PostgreSQL TIMESTAMPTZ columns."""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _assert_user_row_updated(result: Any, user_id: int, *, operation: str) -> None:
        """
        Ensure an UPDATE affected at least one row; raise if the user is missing.

        For PostgreSQL (asyncpg), ``result`` is a status string like ``\"UPDATE 1\"``.
        For SQLite (aiosqlite), ``result`` is a cursor with a ``rowcount`` attribute.
        """
        try:
            # asyncpg status string: e.g. "UPDATE 0", "UPDATE 1"
            if isinstance(result, str):
                parts = result.split()
                if parts and parts[-1].isdigit() and int(parts[-1]) == 0:
                    msg = f"User {user_id} not found during {operation}"
                    logger.warning(msg)
                    raise UserNotFoundError(msg)
                return

            # aiosqlite cursor-style result
            rowcount = getattr(result, "rowcount", None)
            if rowcount == 0:
                msg = f"User {user_id} not found during {operation}"
                logger.warning(msg)
                raise UserNotFoundError(msg)
        except UserNotFoundError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            # Introspection failures must not hide the underlying DB behavior.
            logger.debug(f"AuthnzMfaRepo._assert_user_row_updated introspection failed: {exc}")

    async def set_mfa_config(
        self,
        *,
        user_id: int,
        encrypted_secret: str,
        backup_codes_json: str,
        updated_at: datetime,
    ) -> None:
        """
        Enable MFA for a user by updating the TOTP secret, backup codes,
        and two-factor flag.
        """
        try:
            async with self.db_pool.transaction() as conn:
                gateway = VersionedUserWriteGateway(
                    "postgres" if self._is_postgres_backend() else "sqlite"
                )
                if self._is_postgres_backend():
                    ts = self._normalize_datetime_for_postgres(updated_at)
                    await gateway.execute_update(
                        conn,
                        user_id=user_id,
                        profile_visible_fields=("two_factor_enabled",),
                        statement=
                        """
                        UPDATE users
                        SET totp_secret = $1,
                            two_factor_enabled = TRUE,
                            backup_codes = $2,
                            updated_at = $3
                        WHERE id = $4
                        """,
                        parameters=(encrypted_secret, backup_codes_json, ts, user_id),
                    )
                else:
                    await gateway.execute_update(
                        conn,
                        user_id=user_id,
                        profile_visible_fields=("two_factor_enabled",),
                        statement=
                        """
                        UPDATE users
                        SET totp_secret = ?,
                            two_factor_enabled = 1,
                            backup_codes = ?,
                            updated_at = ?
                        WHERE id = ?
                        """,
                        parameters=(
                            encrypted_secret,
                            backup_codes_json,
                            updated_at.isoformat(),
                            user_id,
                        ),
                    )
        except ProfileVersionNotFound:
            msg = f"User {user_id} not found during set_mfa_config"
            logger.warning(msg)
            raise UserNotFoundError(msg) from None
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(f"AuthnzMfaRepo.set_mfa_config failed: {exc}")
            raise

    async def clear_mfa_config(
        self,
        *,
        user_id: int,
        updated_at: datetime,
    ) -> None:
        """
        Disable MFA for a user and clear secret/backup codes.
        """
        try:
            async with self.db_pool.transaction() as conn:
                gateway = VersionedUserWriteGateway(
                    "postgres" if self._is_postgres_backend() else "sqlite"
                )
                if self._is_postgres_backend():
                    ts = self._normalize_datetime_for_postgres(updated_at)
                    await gateway.execute_update(
                        conn,
                        user_id=user_id,
                        profile_visible_fields=("two_factor_enabled",),
                        statement=
                        """
                        UPDATE users
                        SET totp_secret = NULL,
                            two_factor_enabled = FALSE,
                            backup_codes = NULL,
                            updated_at = $1
                        WHERE id = $2
                        """,
                        parameters=(ts, user_id),
                    )
                else:
                    await gateway.execute_update(
                        conn,
                        user_id=user_id,
                        profile_visible_fields=("two_factor_enabled",),
                        statement=
                        """
                        UPDATE users
                        SET totp_secret = NULL,
                            two_factor_enabled = 0,
                            backup_codes = NULL,
                            updated_at = ?
                        WHERE id = ?
                        """,
                        parameters=(updated_at.isoformat(), user_id),
                    )
        except ProfileVersionNotFound:
            msg = f"User {user_id} not found during clear_mfa_config"
            logger.warning(msg)
            raise UserNotFoundError(msg) from None
        except UserNotFoundError:
            raise
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.bind(
                operation="clear_mfa_config",
                exception_type=type(exc).__name__,
            ).error("AuthNZ MFA configuration clear failed")
            raise DatabaseError("Failed to clear MFA configuration") from None

    async def get_encrypted_totp_secret(self, user_id: int) -> str | None:
        """
        Return the encrypted TOTP secret for a user, if present.
        """
        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres_backend():
                    encrypted = await conn.fetchval(
                        "SELECT totp_secret FROM users WHERE id = $1",
                        user_id,
                    )
                else:
                    cursor = await conn.execute(
                        "SELECT totp_secret FROM users WHERE id = ?",
                        (user_id,),
                    )
                    row = await cursor.fetchone()
                    encrypted = row[0] if row else None
            return encrypted
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(
                f"AuthnzMfaRepo.get_encrypted_totp_secret failed for user {user_id}: {exc}"
            )
            raise

    async def get_mfa_status_row(self, user_id: int) -> dict[str, Any] | None:
        """
        Fetch raw MFA status fields for a user.

        Returns a mapping with keys:
        - ``two_factor_enabled``
        - ``has_secret``
        - ``has_backup_codes``
        or ``None`` if the user row does not exist.
        """
        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres_backend():
                    row = await conn.fetchrow(
                        """
                        SELECT two_factor_enabled,
                               totp_secret IS NOT NULL AS has_secret,
                               backup_codes IS NOT NULL AS has_backup_codes
                        FROM users
                        WHERE id = $1
                        """,
                        user_id,
                    )
                    return dict(row) if row else None

                cursor = await conn.execute(
                    """
                    SELECT two_factor_enabled,
                           totp_secret IS NOT NULL AS has_secret,
                           backup_codes IS NOT NULL AS has_backup_codes
                    FROM users
                    WHERE id = ?
                    """,
                    (user_id,),
                )
                row = await cursor.fetchone()
                if not row:
                    return None
                return {
                    "two_factor_enabled": row[0],
                    "has_secret": row[1],
                    "has_backup_codes": row[2],
                }
        except DatabaseError:
            raise
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.bind(
                operation="get_mfa_status_row",
                exception_type=type(exc).__name__,
            ).error("AuthNZ MFA status read failed")
            raise DatabaseError("MFA status read failed") from None

    async def get_backup_codes_json(self, user_id: int) -> str | None:
        """
        Return the raw ``backup_codes`` JSON for a user, if present.
        """
        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres_backend():
                    value = await conn.fetchval(
                        "SELECT backup_codes FROM users WHERE id = $1",
                        user_id,
                    )
                else:
                    cursor = await conn.execute(
                        "SELECT backup_codes FROM users WHERE id = ?",
                        (user_id,),
                    )
                    row = await cursor.fetchone()
                    value = row[0] if row else None
            return value
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(
                f"AuthnzMfaRepo.get_backup_codes_json failed for user {user_id}: {exc}"
            )
            raise

    async def update_backup_codes_json(
        self,
        *,
        user_id: int,
        backup_codes_json: str,
    ) -> None:
        """
        Persist an updated ``backup_codes`` JSON payload for a user.

        This mirrors the semantics used when consuming a single backup code
        during verification and intentionally does not modify ``updated_at``.
        """
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    result = await conn.execute(
                        "UPDATE users SET backup_codes = $1 WHERE id = $2",
                        backup_codes_json,
                        user_id,
                    )
                else:
                    result = await conn.execute(
                        "UPDATE users SET backup_codes = ? WHERE id = ?",
                        (backup_codes_json, user_id),
                    )
                self._assert_user_row_updated(result, user_id, operation="update_backup_codes_json")
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(
                f"AuthnzMfaRepo.update_backup_codes_json failed for user {user_id}: {exc}"
            )
            raise

    async def consume_backup_codes_json(
        self,
        *,
        user_id: int,
        expected_backup_codes_json: str,
        updated_backup_codes_json: str,
    ) -> bool:
        """
        Atomically persist updated ``backup_codes`` JSON when the current value matches.

        This implements a simple compare-and-swap pattern:

        - The caller provides the previously observed JSON payload
          (``expected_backup_codes_json``) and the desired updated payload
          (``updated_backup_codes_json``).
        - The UPDATE succeeds only when the row exists and ``backup_codes`` still
          equals the expected value; exactly one row is affected in that case.
        - When another concurrent operation has already changed ``backup_codes``,
          the UPDATE affects zero rows and this method returns ``False`` without
          raising, signalling that the code was already consumed or refreshed.

        Returns:
            True when the backup codes were updated; False when no row matched the
            expected JSON (concurrent update / already-consumed code).
        """
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    result = await conn.execute(
                        "UPDATE users SET backup_codes = $1 WHERE id = $2 AND backup_codes = $3",
                        updated_backup_codes_json,
                        user_id,
                        expected_backup_codes_json,
                    )
                else:
                    result = await conn.execute(
                        "UPDATE users SET backup_codes = ? WHERE id = ? AND backup_codes = ?",
                        (updated_backup_codes_json, user_id, expected_backup_codes_json),
                    )

                # Interpret result without raising on zero-row updates: this path is
                # used to guard against concurrent consumption of the same code.
                try:
                    if isinstance(result, str):
                        parts = result.split()
                        if parts and parts[-1].isdigit():
                            return int(parts[-1]) > 0
                        return False
                    rowcount = getattr(result, "rowcount", None)
                    return bool(rowcount)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug(f"AuthnzMfaRepo.consume_backup_codes_json introspection failed: {exc}")
                    # In doubt, treat as no-op rather than raising; caller will
                    # interpret this as a failed consume.
                    return False
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(
                f"AuthnzMfaRepo.consume_backup_codes_json failed for user {user_id}: {exc}"
            )
            raise

    async def set_backup_codes_with_timestamp(
        self,
        *,
        user_id: int,
        backup_codes_json: str,
        updated_at: datetime,
    ) -> None:
        """
        Set ``backup_codes`` and bump ``updated_at`` for a user.

        Used when regenerating backup codes so callers can distinguish
        the refresh event in audit-style views.
        """
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    ts = self._normalize_datetime_for_postgres(updated_at)
                    result = await conn.execute(
                        "UPDATE users SET backup_codes = $1, updated_at = $2 WHERE id = $3",
                        backup_codes_json,
                        ts,
                        user_id,
                    )
                else:
                    result = await conn.execute(
                        "UPDATE users SET backup_codes = ?, updated_at = ? WHERE id = ?",
                        (backup_codes_json, updated_at.isoformat(), user_id),
                    )
                self._assert_user_row_updated(result, user_id, operation="set_backup_codes_with_timestamp")
        except Exception as exc:  # pragma: no cover - surfaced via callers
            logger.error(
                f"AuthnzMfaRepo.set_backup_codes_with_timestamp failed for user {user_id}: {exc}"
            )
            raise
