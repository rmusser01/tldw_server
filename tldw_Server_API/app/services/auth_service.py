from __future__ import annotations

import contextlib
import inspect
import re
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import (
    _profile_user_backend,
)
from tldw_Server_API.app.core.AuthNZ.profile_version import (
    ProfileVersionNotFound,
    VersionedUserWriteGateway,
)
from tldw_Server_API.app.core.deprecations import log_runtime_deprecation

_USER_ROW_FALLBACK_COLUMNS = (
    "id",
    "uuid",
    "username",
    "email",
    "password_hash",
    "role",
    "is_active",
    "is_verified",
    "created_at",
    "updated_at",
    "last_login",
    "storage_quota_mb",
    "storage_used_mb",
)


_SQL_PARAM_RE = re.compile(r"\$\d+")


def _normalize_user_row(row: Any) -> dict[str, Any] | None:
    if not row:
        return None
    if isinstance(row, dict):
        return dict(row)
    with contextlib.suppress(Exception):
        return dict(row)
    with contextlib.suppress(Exception):
        return {key: row[key] for key in row}
    return {
        _USER_ROW_FALLBACK_COLUMNS[i]: row[i]
        for i in range(min(len(_USER_ROW_FALLBACK_COLUMNS), len(row)))
    }


def _normalize_sqlite_placeholders(query: str) -> str:
    return _SQL_PARAM_RE.sub("?", query)


async def _execute_compat(db: Any, query: str, *params: Any) -> Any:
    execute = db.execute
    try:
        return await execute(query, *params)
    except TypeError:
        log_runtime_deprecation(
            "auth_db_execute_compat",
            message=(
                "Auth DB execute compatibility adapter was used for sqlite-like "
                "execute(query, tuple(params)) signature."
            ),
        )
        # Compatibility for sqlite-like execute(query, tuple(params))
        return await execute(_normalize_sqlite_placeholders(query), tuple(params))


async def _fetchrow_compat(db: Any, query: str, *params: Any) -> Any:
    fetchrow = getattr(db, "fetchrow", None)
    if callable(fetchrow):
        return await fetchrow(query, *params)
    log_runtime_deprecation(
        "auth_db_execute_compat",
        message="Auth DB fetchrow compatibility adapter used execute()+fetchone() path.",
    )
    cursor = await _execute_compat(db, query, *params)
    return await cursor.fetchone()


async def _maybe_commit(db: Any) -> None:
    if _profile_user_backend(db) is not None:
        return
    commit = getattr(db, "commit", None)
    if callable(commit):
        with contextlib.suppress(Exception):
            await commit()


def _extract_update_count(result: Any) -> int:
    if isinstance(result, str):
        parts = result.split()
        if parts and parts[-1].isdigit():
            return int(parts[-1])
        return 0
    rowcount = getattr(result, "rowcount", None)
    with contextlib.suppress(TypeError, ValueError):
        if rowcount is not None:
            return int(rowcount)
    return 0


def _extract_row_value(row: Any, key: str, index: int) -> Any:
    if row is None:
        return None
    if isinstance(row, dict):
        return row.get(key)
    with contextlib.suppress(Exception):
        return row[key]
    with contextlib.suppress(Exception):
        return row[index]
    return None


def _versioned_user_gateway(db: Any) -> VersionedUserWriteGateway:
    """Resolve the connection's established async PostgreSQL/SQLite contract."""
    backend = _profile_user_backend(db)
    if backend is None:
        fetch = getattr(db, "fetch", None)
        backend = "postgres" if callable(fetch) else "sqlite"
    return VersionedUserWriteGateway(backend)


@contextlib.asynccontextmanager
async def _managed_profile_write_transaction(
    db: Any,
    *,
    backend: str,
) -> AsyncIterator[None]:
    """Make a managed compound write atomic unless its caller already did."""
    if _profile_user_backend(db) is None:
        yield
        return
    if inspect.getattr_static(db, "transaction", None) is None:
        yield
        return

    if backend == "sqlite":
        if inspect.getattr_static(db, "in_transaction", None) is None:
            yield
            return
        in_transaction = db.in_transaction
    else:
        if inspect.getattr_static(db, "is_in_transaction", None) is None:
            yield
            return
        is_in_transaction = db.is_in_transaction
        in_transaction = is_in_transaction()

    if type(in_transaction) is not bool:
        raise TypeError("Managed connection returned an invalid transaction state")
    if in_transaction:
        yield
        return

    transaction = db.transaction
    if not callable(transaction):
        raise TypeError("Managed connection transaction factory is not callable")
    async with transaction():
        yield


def _normalize_datetime_for_backend(value: datetime, *, backend: str) -> datetime:
    if backend != "postgres":
        return value
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


async def fetch_user_by_login_identifier(db, identifier: str) -> dict[str, Any] | None:
    """Fetch a user row by username or email (case-insensitive)."""
    ident_l = identifier.strip().lower()
    try:
        row = await _fetchrow_compat(
            db,
            "SELECT * FROM users WHERE lower(username) = $1 OR lower(email) = $2",
            ident_l,
            ident_l,
        )
        return _normalize_user_row(row)
    except Exception as e:
        logger.error(f"auth_service.fetch_user_by_login_identifier failed: {e}")
        raise


async def update_user_password_hash(db, user_id: int, new_hash: str) -> None:
    """Persist a new password hash for the user."""
    try:
        await _execute_compat(
            db,
            "UPDATE users SET password_hash = $1 WHERE id = $2",
            new_hash,
            user_id,
        )
        await _maybe_commit(db)
    except Exception as e:
        logger.error(f"auth_service.update_user_password_hash failed for user {user_id}: {e}")
        raise


async def update_user_last_login(db, user_id: int, now: datetime | None = None) -> None:
    """Update last_login timestamp for the user."""
    now = now or datetime.now(timezone.utc)
    try:
        gateway = _versioned_user_gateway(db)
        now = _normalize_datetime_for_backend(now, backend=gateway.backend)
        statement = (
            "UPDATE users SET last_login = $1 WHERE id = $2"
            if gateway.backend == "postgres"
            else "UPDATE users SET last_login = ? WHERE id = ?"
        )
        async with _managed_profile_write_transaction(
            db,
            backend=gateway.backend,
        ):
            await gateway.execute_update(
                db,
                user_id=user_id,
                profile_visible_fields=("last_login",),
                statement=statement,
                parameters=(now, user_id),
            )
        await _maybe_commit(db)
    except Exception as e:
        logger.error(f"auth_service.update_user_last_login failed for user {user_id}: {e}")
        raise


async def fetch_active_user_by_id(db, user_id: int) -> dict[str, Any] | None:
    """Fetch an active user by id, normalized to dict, or None if not found."""
    try:
        row = await _fetchrow_compat(
            db,
            "SELECT * FROM users WHERE id = $1 AND is_active = $2",
            user_id,
            True,
        )
        user = _normalize_user_row(row)
        if user and "is_active" in user:
            user["is_active"] = bool(user.get("is_active"))
        return user
    except Exception as e:
        logger.error(f"auth_service.fetch_active_user_by_id failed: {e}")
        raise


async def fetch_user_by_email_for_password_reset(db: Any, email: str) -> dict[str, Any] | None:
    """Fetch reset-eligible user fields by email, case-insensitive."""
    email_l = str(email or "").strip().lower()
    try:
        row = await _fetchrow_compat(
            db,
            "SELECT id, username, email, is_active FROM users WHERE lower(email) = $1",
            email_l,
        )
        user = _normalize_user_row(row)
        if user and "is_active" in user:
            user["is_active"] = bool(user.get("is_active"))
        return user
    except Exception as exc:
        logger.error(f"auth_service.fetch_user_by_email_for_password_reset failed: {exc}")
        raise


async def store_password_reset_token(
    db: Any,
    *,
    user_id: int,
    token_hash: str,
    expires_at: datetime,
    ip_address: str,
) -> None:
    """Insert a password reset token record."""
    try:
        await _execute_compat(
            db,
            """
            INSERT INTO password_reset_tokens (user_id, token_hash, expires_at, ip_address)
            VALUES ($1, $2, $3, $4)
            """,
            user_id,
            token_hash,
            expires_at,
            ip_address,
        )
        await _maybe_commit(db)
    except Exception as exc:
        logger.error(f"auth_service.store_password_reset_token failed for user {user_id}: {exc}")
        raise


async def fetch_password_reset_token_record(
    db: Any,
    *,
    user_id: int,
    hash_candidates: list[str],
) -> tuple[int | None, Any | None]:
    """Fetch latest password-reset token record matching one of the token hash candidates."""
    if not hash_candidates:
        return None, None
    try:
        placeholders = ", ".join(f"${idx}" for idx in range(2, len(hash_candidates) + 2))
        hash_candidates_clause = f"({placeholders})"
        query_template = """
            SELECT id, used_at
            FROM password_reset_tokens
            WHERE user_id = $1 AND token_hash IN {hash_candidates_clause}
            ORDER BY expires_at DESC
            LIMIT 1
        """
        query = query_template.format_map(locals())  # nosec B608
        row = await _fetchrow_compat(db, query, user_id, *hash_candidates)
        if not row:
            return None, None
        token_record_id = _extract_row_value(row, "id", 0)
        used_at = _extract_row_value(row, "used_at", 1)
        return int(token_record_id) if token_record_id is not None else None, used_at
    except Exception as exc:
        logger.error(f"auth_service.fetch_password_reset_token_record failed for user {user_id}: {exc}")
        raise


async def apply_password_reset(
    db: Any,
    *,
    user_id: int,
    new_password_hash: str,
    token_record_id: int,
    now_utc: datetime,
) -> None:
    """Apply password reset and mark reset token as used."""
    try:
        backend = _versioned_user_gateway(db).backend
        now_utc = _normalize_datetime_for_backend(now_utc, backend=backend)
        await _execute_compat(
            db,
            "UPDATE users SET password_hash = $1, updated_at = $2 WHERE id = $3",
            new_password_hash,
            now_utc,
            user_id,
        )
        await _execute_compat(
            db,
            "UPDATE password_reset_tokens SET used_at = $1 WHERE id = $2",
            now_utc,
            token_record_id,
        )
        await _maybe_commit(db)
    except Exception as exc:
        logger.error(f"auth_service.apply_password_reset failed for user {user_id}: {exc}")
        raise


async def verify_user_email_once(
    db: Any,
    *,
    user_id: int,
    email: str,
    now_utc: datetime,
) -> int:
    """Mark user email as verified once; return number of updated rows."""
    try:
        gateway = _versioned_user_gateway(db)
        now_utc = _normalize_datetime_for_backend(now_utc, backend=gateway.backend)
        if gateway.backend == "postgres":
            statement = """
                UPDATE users
                   SET is_verified = $1, updated_at = $2
                 WHERE id = $3
                   AND lower(email) = lower($4)
                   AND COALESCE(is_verified, $5) != $6
                """
        else:
            statement = """
                UPDATE users
                   SET is_verified = ?, updated_at = ?
                 WHERE id = ?
                   AND lower(email) = lower(?)
                   AND COALESCE(is_verified, ?) != ?
                """
        write_result = await gateway.execute_update(
            db,
            user_id=user_id,
            profile_visible_fields=("is_verified",),
            statement=statement,
            parameters=(True, now_utc, user_id, email, False, True),
        )
        await _maybe_commit(db)
        return len(write_result.affected_user_ids)
    except ProfileVersionNotFound:
        await _maybe_commit(db)
        return 0
    except Exception as exc:
        logger.error(f"auth_service.verify_user_email_once failed for user {user_id}: {exc}")
        raise


async def fetch_user_by_email_for_verification(db: Any, email: str) -> dict[str, Any] | None:
    """Fetch email-verification fields by email, case-insensitive."""
    email_l = str(email or "").strip().lower()
    try:
        row = await _fetchrow_compat(
            db,
            "SELECT id, username, email, is_verified FROM users WHERE lower(email) = $1",
            email_l,
        )
        user = _normalize_user_row(row)
        if user and "is_verified" in user:
            user["is_verified"] = bool(user.get("is_verified"))
        return user
    except Exception as exc:
        logger.error(f"auth_service.fetch_user_by_email_for_verification failed: {exc}")
        raise


async def mark_user_verified(db: Any, user_id: int, now_utc: datetime) -> None:
    """Mark user email as verified."""
    try:
        gateway = _versioned_user_gateway(db)
        now_utc = _normalize_datetime_for_backend(now_utc, backend=gateway.backend)
        statement = (
            "UPDATE users SET is_verified = $1, updated_at = $2 WHERE id = $3"
            if gateway.backend == "postgres"
            else "UPDATE users SET is_verified = ?, updated_at = ? WHERE id = ?"
        )
        await gateway.execute_update(
            db,
            user_id=user_id,
            profile_visible_fields=("is_verified",),
            statement=statement,
            parameters=(True, now_utc, user_id),
        )
        await _maybe_commit(db)
    except Exception as exc:
        logger.error(f"auth_service.mark_user_verified failed for user {user_id}: {exc}")
        raise
