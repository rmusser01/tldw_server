# auth_deps.py
# Description: FastAPI dependency injection for authentication services
#
import asyncio
import inspect
import os
import re
import threading
import time
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Mapping
from datetime import datetime, timezone
from typing import Annotated, Any, Callable, Optional
from weakref import WeakKeyDictionary

#
# 3rd-party imports
from fastapi import Depends, Header, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.api_key_manager import get_api_key_manager
from tldw_Server_API.app.core.AuthNZ.auth_governor import get_auth_governor
from tldw_Server_API.app.core.AuthNZ.auth_principal_resolver import (
    get_auth_principal as _resolve_auth_principal,
)

#
# Local imports
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    DatabaseError,
    DatabaseLockError,
    InvalidTokenError,
    RegistrationError,
    TokenExpiredError,
    TransactionError,
    WeakPasswordError,
)
from tldw_Server_API.app.core.AuthNZ.ip_allowlist import (
    is_single_user_ip_allowed,
    resolve_client_ip,
)
from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService, get_jwt_service
from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService, get_password_service
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal, is_single_user_principal
from tldw_Server_API.app.core.AuthNZ.rate_limiter import RateLimiter, get_rate_limiter
from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager, get_session_manager
from tldw_Server_API.app.core.AuthNZ.settings import (
    get_settings,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import (
    authenticate_api_key_user,
    get_single_user_instance,
    get_request_user,
    resolve_user_id_for_request as resolve_user_id_for_request,
    User as User,
    verify_jwt_and_fetch_user,
)
from tldw_Server_API.app.core.DB_Management.scope_context import set_scope
from tldw_Server_API.app.core.exceptions import InactiveUserError
from tldw_Server_API.app.core.External_Sources.connectors_service import get_policy
from tldw_Server_API.app.core.External_Sources.policy import get_default_policy_from_env
from tldw_Server_API.app.core.MCP_unified.monitoring import metrics
from tldw_Server_API.app.core.testing import (
    env_flag_enabled as _env_flag_enabled,
    is_explicit_pytest_runtime as _is_explicit_pytest_runtime,
    is_production_like_env as _is_production_like_env,
)
from tldw_Server_API.app.core.testing import is_test_mode as _is_test_mode
from tldw_Server_API.app.services.registration_service import RegistrationService, get_registration_service
from tldw_Server_API.app.services.storage_quota_service import StorageQuotaService, get_storage_service

# Narrowed exception tuple for auth dependency safety (BLE001)
_AUTH_DEPS_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
    InvalidTokenError,
    TokenExpiredError,
    DatabaseError,
    TransactionError,
    RegistrationError,
    WeakPasswordError,
    InactiveUserError,
)

# Test stub shared state (persist across dependency calls under TEST_MODE/pytest)
_TEST_SESSION_STATE: dict = {"sid": 1000, "sessions": {}}
_TEST_SESSION_LOCKS: "WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = WeakKeyDictionary()
_TEST_SESSION_LOCK_GUARD = threading.Lock()
_TEST_SESSION_STATE_GUARD = threading.Lock()
_TEST_EPHEMERAL_STATE: dict = {"values": {}}
_TEST_EPHEMERAL_STATE_GUARD = threading.Lock()

_SENSITIVE_USER_KEY_PATTERN = re.compile(
    r"(password|secret|token|api[_-]?key|ssn|totp|otp|mfa|backup_codes|recovery_codes)",
    re.IGNORECASE,
)
_AUTH_DEPS_RG_DIAGNOSTICS_ONLY_LOGGED: set[str] = set()
_AUTH_DEPS_FALLBACK_RATE_WINDOWS: dict[tuple[str, str], deque[float]] = {}
_AUTH_DEPS_FALLBACK_RATE_WINDOWS_LOCK = threading.Lock()


def _read_non_negative_int_env(name: str, default: int) -> int:
    """Read an integer environment variable and clamp to non-negative."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        logger.warning(
            "Invalid integer env for {}: {!r}; using default={}",
            name,
            raw,
            default,
        )
        return default
    return max(parsed, 0)


def _read_non_negative_float_env(name: str, default: float) -> float:
    """Read a float environment variable and clamp to non-negative."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        parsed = float(raw)
    except ValueError:
        logger.warning(
            "Invalid float env for {}: {!r}; using default={}",
            name,
            raw,
            default,
        )
        return default
    return max(parsed, 0.0)


def _authnz_sqlite_lock_retry_config() -> tuple[int, float, float, int]:
    """Return retry/backoff config for transient AuthNZ SQLite lock contention."""
    max_retries = _read_non_negative_int_env(
        "AUTHNZ_SQLITE_LOCK_MAX_RETRIES",
        2,
    )
    retry_after_seconds = _read_non_negative_int_env(
        "AUTHNZ_SQLITE_LOCK_RETRY_AFTER_SECONDS",
        1,
    )
    base_backoff_seconds = _read_non_negative_float_env(
        "AUTHNZ_SQLITE_LOCK_RETRY_BASE_SECONDS",
        0.05,
    )
    max_backoff_seconds = _read_non_negative_float_env(
        "AUTHNZ_SQLITE_LOCK_RETRY_MAX_SECONDS",
        0.25,
    )
    if max_backoff_seconds < base_backoff_seconds:
        max_backoff_seconds = base_backoff_seconds
    return (
        max_retries,
        base_backoff_seconds,
        max_backoff_seconds,
        retry_after_seconds,
    )


def _authnz_busy_http_exception(retry_after_seconds: int) -> HTTPException:
    """Create a consistent 503 response for temporary AuthNZ DB contention."""
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Authentication database is busy. Please retry shortly.",
        headers={"Retry-After": str(max(retry_after_seconds, 0))},
    )


def _public_user_dict(user: Mapping[str, Any]) -> dict[str, Any]:
    """
    Return a sanitized shallow copy of a user mapping.

    The AuthNZ stack may temporarily cache full database rows (including secrets)
    in `request.state._auth_user`. This helper strips common secret-bearing keys
    (passwords, tokens, secrets, API keys, SSNs) so dependency returns never
    expose sensitive fields.
    """
    safe: dict[str, Any] = {}
    for key, value in dict(user).items():
        if isinstance(key, str) and _SENSITIVE_USER_KEY_PATTERN.search(key):
            continue
        safe[key] = value
    return safe


def _looks_like_jwt(token: Optional[str]) -> bool:
    if not isinstance(token, str):
        return False
    return token.count(".") == 2


async def _authenticate_api_key_from_request(request: Request, api_key: str) -> dict[str, Any]:
    test_mode = _is_test_mode()
    if test_mode:
        # SECURITY: Warn loudly about TEST_MODE and block in production unless explicitly allowed
        allow_test_in_prod = _env_flag_enabled("ALLOW_TEST_MODE_IN_PRODUCTION")
        environment = os.getenv("ENVIRONMENT", "").strip().lower()
        prod_flag = _env_flag_enabled("tldw_production")
        is_production = environment in {"production", "prod"} or prod_flag
        if is_production:
            if not allow_test_in_prod:
                logger.critical(
                    "TEST_MODE is enabled in production environment! "
                    "This is a SEVERE security risk. Set ALLOW_TEST_MODE_IN_PRODUCTION=1 to override (NOT recommended)."
                )
                raise HTTPException(
                    status_code=500,
                    detail="Server configuration error"
                )
            else:
                logger.warning(
                    "TEST_MODE is enabled in production with explicit override. "
                    "This bypasses normal authentication and should only be used for debugging."
                )
        else:
            logger.debug("TEST_MODE is enabled for non-production environment")
        try:
            settings = get_settings()
        except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS:
            settings = None
        allowed_keys: set[str] = set()
        test_key = os.getenv("SINGLE_USER_TEST_API_KEY")
        if test_key:
            allowed_keys.add(test_key)
        if settings and settings.SINGLE_USER_API_KEY:
            allowed_keys.add(settings.SINGLE_USER_API_KEY)
        if api_key in allowed_keys:
            client_ip = resolve_client_ip(request, settings)
            if settings and settings.AUTH_MODE == "single_user":
                if not is_single_user_ip_allowed(client_ip, settings):
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid or missing API Key",
                    )
            try:
                if settings and isinstance(settings.DATABASE_URL, str) and settings.DATABASE_URL.startswith("sqlite:///"):
                    from pathlib import Path as _Path

                    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables as _ensure_authnz_tables
                    db_path = settings.DATABASE_URL.replace("sqlite:///", "")
                    _ensure_authnz_tables(_Path(db_path))
            except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as _ensure_err:
                logger.debug(
                    "AuthNZ test fallback: ensure_authnz_tables skipped/failed: error_type={}",
                    type(_ensure_err).__name__,
                )
            fixed_id = getattr(settings, "SINGLE_USER_FIXED_ID", 1)
            user = {
                "id": fixed_id,
                "username": "single_user",
                "email": None,
                "role": "admin",
                "roles": ["admin"],
                "permissions": ["*"],
                "is_active": True,
                "is_verified": True,
            }
            user = _public_user_dict(user)
            try:
                request.state.user_id = fixed_id
                request.state.team_ids = []
                request.state.org_ids = []
            except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as state_exc:
                logger.debug(
                    "API key test-mode path: unable to attach state context: {}",
                    state_exc,
                )
            return user
    try:
        # Force API key manager resolution through this module so tests can
        # monkeypatch `auth_deps.get_api_key_manager` and assert error logging
        # does not leak exception messages outside TEST_MODE.
        await get_api_key_manager()
        user_obj = await authenticate_api_key_user(request, api_key)
        # Normalize User model to a plain dict for response serialization.
        if hasattr(user_obj, "model_dump"):
            user_dict = user_obj.model_dump()  # type: ignore[call-arg]
        elif hasattr(user_obj, "dict"):
            user_dict = user_obj.dict()  # type: ignore[call-arg]
        elif isinstance(user_obj, Mapping):
            user_dict = dict(user_obj)
        else:
            user_dict = {"id": getattr(user_obj, "id", None)}
        return _public_user_dict(user_dict)
    except HTTPException:
        # Propagate explicit HTTP errors unchanged (401/403, etc.)
        raise
    except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as e:
        if _is_test_mode():
            logger.exception("API key authentication error in get_current_user (TEST_MODE)")
        else:
            logger.error(
                "API key authentication error in get_current_user (type={})",
                type(e).__name__,
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate API key",
        ) from e


def _get_test_session_lock() -> asyncio.Lock:
    """
    Lazily initialize and return a test session lock scoped to the current event loop.

    The lock is created on first use within the currently running event loop
    instead of at module import time to avoid binding it to the wrong loop
    in test environments.
    """
    loop = asyncio.get_running_loop()
    with _TEST_SESSION_LOCK_GUARD:
        lock = _TEST_SESSION_LOCKS.get(loop)
        if lock is None:
            lock = asyncio.Lock()
            _TEST_SESSION_LOCKS[loop] = lock
        return lock


def reset_test_ephemeral_state() -> None:
    """Clear the test-only in-process ephemeral KV store (pytest/TEST_MODE)."""
    with _TEST_EPHEMERAL_STATE_GUARD:
        _TEST_EPHEMERAL_STATE["values"].clear()

#######################################################################################################################
#
# Security scheme for JWT bearer tokens

security = HTTPBearer(auto_error=False)


#######################################################################################################################
#
# Service Dependency Functions

def _activate_scope_context(
    request: Request,
    *,
    user_id: Optional[int],
    org_ids: Optional[list[int]],
    team_ids: Optional[list[int]],
    is_admin: bool,
) -> None:
    """Record content scope information for downstream database access."""
    try:
        active_org = getattr(request.state, "active_org_id", None)
        active_team = getattr(request.state, "active_team_id", None)
        token = set_scope(
            user_id=user_id,
            org_ids=org_ids or (),
            team_ids=team_ids or (),
            active_org_id=active_org,
            active_team_id=active_team,
            is_admin=is_admin,
        )
        request.state._content_scope_token = token
    except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(
            "Unable to establish content scope context: {}",
            exc,
        )

async def get_db_transaction() -> AsyncGenerator[Any, None]:
    """Get database connection in transaction mode.

    Always behaves as an async generator for FastAPI compatibility. In explicit test mode,
    yields a lightweight pool adapter that runs queries without holding a long-lived
    transaction to avoid event-loop and teardown issues. Otherwise, yields a
    request-scoped transaction connection.
    """
    db_pool = await get_db_pool()

    # Decide whether to use the lightweight adapter (explicit test mode) or a real transaction
    use_adapter = False
    try:
        if _is_test_mode():
            use_adapter = True
    except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS:
        # If any detection fails, fall back to default transaction behavior
        use_adapter = False

    if use_adapter:
        # Keep a single connection open for the request lifetime so cursors remain valid
        is_postgres_backend = bool(getattr(db_pool, "pool", None) is not None)
        conn_cm = db_pool.acquire()
        conn = await conn_cm.__aenter__()

        # NOTE: This adapter normalizes asyncpg-style ($1, fetch*) and SQLite-style
        # query/return semantics for tests. If other modules need similar behavior,
        # it can be extracted into the DB_Management layer.
        class _ConnAdapter:
            def __init__(self, _conn: Any, *, is_postgres: bool) -> None:
                self._conn = _conn
                self._is_sqlite = not is_postgres
                self._dollar_param = re.compile(r"\$\d+")

            def _normalize_sqlite_sql(self, query: str) -> str:
                if not self._is_sqlite or "$" not in query:
                    return query
                # Replace $1, $2 ... with '?'
                return self._dollar_param.sub("?", query)

            async def execute(self, query: str, *args: object) -> Any:
                # Postgres (asyncpg) supports variadic args; SQLite expects a sequence
                if not self._is_sqlite:
                    # asyncpg connection
                    return await self._conn.execute(query, *args)
                else:
                    params = args[0] if (len(args) == 1 and isinstance(args[0], (list, tuple))) else args
                    q = self._normalize_sqlite_sql(query)
                    cur = await self._conn.execute(q, params)
                    try:
                        await self._conn.commit()
                    except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as exc:
                        logger.debug(
                            "Test DB adapter: sqlite commit failed: {}",
                            exc,
                        )
                        raise
                    return cur

            async def fetchval(self, query: str, *args: object) -> Any | None:
                if not self._is_sqlite:
                    # asyncpg connection
                    return await self._conn.fetchval(query, *args)
                else:
                    params = args[0] if (len(args) == 1 and isinstance(args[0], (list, tuple))) else args
                    q = self._normalize_sqlite_sql(query)
                    cur = await self._conn.execute(q, params)
                    row = await cur.fetchone()
                    return row[0] if row else None

            async def fetch(self, query: str, *args: object) -> list[Any]:
                if not self._is_sqlite:
                    # asyncpg connection
                    rows = await self._conn.fetch(query, *args)
                    return [dict(r) for r in rows]
                else:
                    params = args[0] if (len(args) == 1 and isinstance(args[0], (list, tuple))) else args
                    q = self._normalize_sqlite_sql(query)
                    cur = await self._conn.execute(q, params)
                    rows = await cur.fetchall()
                    normalized_rows: list[Any] = []
                    for row in rows:
                        try:
                            keys = row.keys()  # sqlite3.Row / aiosqlite.Row
                            normalized_rows.append({key: row[key] for key in keys})
                            continue
                        except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS:
                            pass
                        normalized_rows.append(row)
                    return normalized_rows

            async def fetchrow(self, query: str, *args: object) -> Any | None:
                if not self._is_sqlite:
                    # asyncpg connection
                    return await self._conn.fetchrow(query, *args)
                else:
                    params = args[0] if (len(args) == 1 and isinstance(args[0], (list, tuple))) else args
                    q = self._normalize_sqlite_sql(query)
                    cur = await self._conn.execute(q, params)
                    row = await cur.fetchone()
                    try:
                        if not row:
                            return None
                        keys = row.keys()  # sqlite3.Row / aiosqlite.Row
                        return {key: row[key] for key in keys}
                    except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS:
                        return row

            async def commit(self) -> None:
                if self._is_sqlite:
                    try:
                        await self._conn.commit()
                    except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as exc:
                        logger.debug(
                            "Test DB adapter: sqlite commit failed: {}",
                            exc,
                        )
                        raise

        adapter = _ConnAdapter(conn, is_postgres=is_postgres_backend)
        try:
            yield adapter
        finally:
            await conn_cm.__aexit__(None, None, None)
    else:
        # Default: yield a request-scoped transaction so writes commit reliably.
        # For SQLite lock contention, retry only transaction-entry failures.
        max_retries, backoff_base, backoff_max, retry_after = _authnz_sqlite_lock_retry_config()
        entry_attempt = 0
        txn_cm = None
        conn = None

        while True:
            txn_cm = db_pool.transaction()
            try:
                conn = await txn_cm.__aenter__()
                break
            except DatabaseLockError as lock_exc:
                if entry_attempt >= max_retries:
                    logger.warning(
                        "AuthNZ DB lock contention exhausted entry retries (attempts={})",
                        entry_attempt + 1,
                    )
                    raise _authnz_busy_http_exception(retry_after) from lock_exc
                sleep_seconds = min(backoff_base * (2 ** entry_attempt), backoff_max)
                logger.debug(
                    "AuthNZ DB lock contention on transaction entry; retrying (attempt={} sleep={}s)",
                    entry_attempt + 1,
                    sleep_seconds,
                )
                entry_attempt += 1
                await asyncio.sleep(sleep_seconds)

        if txn_cm is None:
            raise RuntimeError("AuthNZ transaction context manager was not initialized")

        try:
            yield conn
        except BaseException as exc:
            try:
                await txn_cm.__aexit__(type(exc), exc, exc.__traceback__)
            except DatabaseLockError as lock_exc:
                raise _authnz_busy_http_exception(retry_after) from lock_exc
            raise
        else:
            try:
                await txn_cm.__aexit__(None, None, None)
            except DatabaseLockError as lock_exc:
                raise _authnz_busy_http_exception(retry_after) from lock_exc


async def get_password_service_dep() -> PasswordService:
    """Get password service dependency"""
    return get_password_service()


async def get_jwt_service_dep() -> JWTService:
    """Get JWT service dependency"""
    return get_jwt_service()


async def get_session_manager_dep() -> SessionManager:
    """Get session manager dependency"""
    # In explicit pytest runtime + test mode, return a lightweight stub to avoid heavy init.
    # Never use this stub in production-like environments.
    try:
        force_real = _env_flag_enabled("AUTHNZ_FORCE_REAL_SESSION_MANAGER")
        in_production_like = bool(_is_production_like_env())
        if (
            not force_real
            and not in_production_like
            and _is_explicit_pytest_runtime()
            and _is_test_mode()
        ):
            class _StubSessionManager:
                enabled = True

                async def is_token_blacklisted(
                    self,
                    token: str,
                    jti: Optional[str] = None,
                    *,
                    token_type: Optional[str] = None,
                    user_id: Optional[int] = None,
                ) -> bool:
                    return False

                async def create_session(
                    self,
                    user_id: int,
                    access_token: str,
                    refresh_token: str,
                    ip_address: Optional[str] = None,
                    user_agent: Optional[str] = None,
                    device_id: Optional[str] = None,
                    expires_at_override: Optional[datetime] = None,
                    refresh_expires_at_override: Optional[datetime] = None,
                    **_kwargs: Any,
                ) -> dict[str, Any]:
                    async with _get_test_session_lock():
                        with _TEST_SESSION_STATE_GUARD:
                            _TEST_SESSION_STATE["sid"] += 1
                            sid = _TEST_SESSION_STATE["sid"]
                            now = datetime.now(timezone.utc)
                            expires_at = expires_at_override or now
                            refresh_expires_at = refresh_expires_at_override or now
                            sess = {
                                "id": sid,
                                "session_id": sid,
                                "user_id": user_id,
                                "ip_address": ip_address or "",
                                "user_agent": user_agent or "",
                                "device_id": device_id,
                                "created_at": now,
                                "last_activity": now,
                                "expires_at": expires_at,
                                "refresh_expires_at": refresh_expires_at,
                                "is_active": True,
                                "is_revoked": False,
                            }
                            _TEST_SESSION_STATE["sessions"][sid] = sess
                            return sess

                async def update_session_tokens(
                    self,
                    _session_id: int = 0,
                    _access_token: str = "",
                    _refresh_token: str = "",
                    **_kwargs: object,
                ) -> bool:
                    # No-op in stub
                    return True

                async def refresh_session(self, *_args: object, **kwargs: object) -> dict[str, Any]:
                    return {
                        "session_id": kwargs.get("session_id") or 1,
                        "user_id": kwargs.get("user_id") or 1,
                        "expires_at": datetime.now(timezone.utc).isoformat(),
                    }

                async def get_user_sessions(self, user_id: int) -> list[dict[str, Any]]:
                    async with _get_test_session_lock():
                        with _TEST_SESSION_STATE_GUARD:
                            sessions = list(_TEST_SESSION_STATE["sessions"].values())
                        return [s for s in sessions if s.get("user_id") == user_id]

                async def revoke_session(self, session_id: int, *_args: object, **_kwargs: object) -> bool:
                    async with _get_test_session_lock():
                        with _TEST_SESSION_STATE_GUARD:
                            sess = _TEST_SESSION_STATE["sessions"].get(session_id)
                            if sess is None:
                                return False
                            sess["is_revoked"] = True
                            sess["is_active"] = False
                            return True

                async def revoke_all_user_sessions(self, user_id: int) -> int:
                    async with _get_test_session_lock():
                        with _TEST_SESSION_STATE_GUARD:
                            changed = 0
                            for s in _TEST_SESSION_STATE["sessions"].values():
                                if s.get("user_id") == user_id:
                                    s["is_revoked"] = True
                                    s["is_active"] = False
                                    changed += 1
                            return changed

                async def store_ephemeral_value(self, key: str, value: str, ttl_seconds: int) -> None:
                    ttl = max(int(ttl_seconds), 1)
                    now = time.monotonic()
                    expires_at = now + ttl
                    with _TEST_EPHEMERAL_STATE_GUARD:
                        # Opportunistic pruning to keep long-running test suites bounded.
                        values = _TEST_EPHEMERAL_STATE["values"]
                        for k, (_v, exp) in list(values.items()):
                            if exp <= now:
                                values.pop(k, None)
                        values[key] = (value, expires_at)

                async def get_ephemeral_value(self, key: str) -> Optional[str]:
                    now = time.monotonic()
                    with _TEST_EPHEMERAL_STATE_GUARD:
                        entry = _TEST_EPHEMERAL_STATE["values"].get(key)
                        if not entry:
                            return None
                        value, expires_at = entry
                        if expires_at <= now:
                            _TEST_EPHEMERAL_STATE["values"].pop(key, None)
                            return None
                        return value

                async def consume_ephemeral_value(self, key: str) -> Optional[str]:
                    now = time.monotonic()
                    with _TEST_EPHEMERAL_STATE_GUARD:
                        entry = _TEST_EPHEMERAL_STATE["values"].get(key)
                        if not entry:
                            return None
                        value, expires_at = entry
                        if expires_at <= now:
                            _TEST_EPHEMERAL_STATE["values"].pop(key, None)
                            return None
                        _TEST_EPHEMERAL_STATE["values"].pop(key, None)
                        return value

                async def delete_ephemeral_value(self, key: str) -> None:
                    with _TEST_EPHEMERAL_STATE_GUARD:
                        _TEST_EPHEMERAL_STATE["values"].pop(key, None)

            return _StubSessionManager()  # type: ignore[return-value]
    except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(
            "get_session_manager_dep: test stub resolution failed; falling back to real SessionManager: {}",
            exc,
        )
    return await get_session_manager()


async def get_rate_limiter_dep(request: Request) -> RateLimiter:
    """Get AuthNZ rate limiter dependency (lockout only)."""
    _ = request
    return get_rate_limiter()


async def get_registration_service_dep() -> RegistrationService:
    """Get registration service dependency"""
    return await get_registration_service()


async def get_storage_service_dep() -> StorageQuotaService:
    """Get storage service dependency"""
    return await get_storage_service()


#######################################################################################################################
#
# User Authentication Dependencies

async def get_current_user(
    request: Request,
    response: Response,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    session_manager: SessionManager = Depends(get_session_manager_dep),
    db_pool: DatabasePool = Depends(get_db_pool),
    x_api_key: Optional[str] = Header(None, alias="X-API-KEY")
) -> dict[str, Any]:
    """
    Legacy compatibility shim that resolves and returns a user dictionary.

    New or migrated route code should prefer ``CurrentPrincipal`` /
    ``get_auth_principal`` unless the endpoint still requires dictionary-shaped
    user data for backwards compatibility.

    Supports Bearer JWT authentication and API keys via `X-API-KEY` or
    Authorization Bearer (non-JWT tokens). If an upstream dependency already
    populated `request.state.auth` and
    `request.state._auth_user`, this function reuses that request-scoped cache to
    avoid repeating token/API-key validation within a single request.

    Args:
        request: FastAPI request object.
        response: FastAPI response object.
        credentials: Bearer token from Authorization header (if present).
        session_manager: Session manager instance.
        db_pool: Database pool instance.
        x_api_key: API key from `X-API-KEY` header (if present).

    Returns:
        User dictionary with all user information.

    Raises:
        HTTPException: If authentication fails.
    """
    # TEST_MODE diagnostics: log auth header presence
    try:
        if _is_test_mode():
            logger.info(
                "get_current_user: has_bearer={} has_api_key={} path={}",
                bool(credentials),
                bool(x_api_key),
                request.url.path,
            )
    except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(
            "get_current_user: TEST_MODE auth header diagnostics failed; continuing without diagnostics: {}",
            exc,
        )

    # Fast-path: if an AuthPrincipal/AuthContext has already been resolved for this
    # request (e.g., via get_auth_principal or budget guards), reuse the cached user
    # representation instead of re-running JWT/API-key logic.
    try:
        existing_ctx = getattr(request.state, "auth", None)
        cached_user = getattr(request.state, "_auth_user", None)
        if isinstance(existing_ctx, AuthContext) and cached_user is not None:
            logger.debug("get_current_user: Reusing cached AuthPrincipal/_auth_user from request.state.")
            # Validate cached user type before processing; if it is not a mapping or
            # Pydantic-style model, fall through to standard auth.
            if not (isinstance(cached_user, Mapping) or hasattr(cached_user, "model_dump") or hasattr(cached_user, "dict")):
                logger.debug(
                    "get_current_user: cached _auth_user is not a mapping/Pydantic model; "
                    "skipping fast-path reuse."
                )
            else:
                # Normalize to a plain dict to preserve existing return shape
                if isinstance(cached_user, Mapping):
                    user_dict: dict[str, Any] = dict(cached_user)
                else:
                    dump = getattr(cached_user, "model_dump", None) or getattr(cached_user, "dict", None)
                    user_dict = dict(dump())
                safe_user = _public_user_dict(user_dict)
                # If the cached user was a full DB row mapping, replace the cache with
                # the sanitized representation to avoid re-exposing secrets later in
                # this request.
                try:
                    if isinstance(cached_user, Mapping):
                        request.state._auth_user = safe_user
                except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as exc:
                    logger.debug(
                        "Fast-path: unable to update request.state._auth_user with sanitized user: {}",
                        exc,
                    )
                # Ensure request.state.user_id is populated for downstream consumers
                try:
                    uid = safe_user.get("id")
                    if uid is not None:
                        request.state.user_id = int(uid)
                except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as exc:
                    logger.debug(
                        "Fast-path: unable to attach user_id to request.state: {}",
                        exc,
                    )
                # Scope context should already be set by upstream auth, but be defensive.
                # Prefer org/team ids from the existing principal, with request.state as fallback.
                try:
                    principal = getattr(existing_ctx, "principal", None)
                    org_ids = getattr(principal, "org_ids", None) if principal is not None else None
                    team_ids = getattr(principal, "team_ids", None) if principal is not None else None
                    _activate_scope_context(
                        request,
                        user_id=getattr(principal, "user_id", None),
                        org_ids=org_ids if org_ids is not None else getattr(request.state, "org_ids", None),
                        team_ids=team_ids if team_ids is not None else getattr(request.state, "team_ids", None),
                        # Stage 4 claim-first: admin scope context derives from claims, not boolean flags.
                        is_admin=_principal_has_admin_bypass_claims(principal),
                    )
                except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as exc:
                    logger.debug(
                        "Fast-path: unable to (re)establish content scope context: {}",
                        exc,
                    )
                return safe_user
    except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as exc:
        # Fall through to standard auth behavior if any issue occurs
        logger.debug(
            "get_current_user: Fast-path AuthContext reuse failed, falling back to standard auth: {}",
            exc,
        )

    # Single-user compatibility: accept Authorization Bearer as API key when no X-API-KEY is present.
    if credentials and not x_api_key:
        try:
            settings = get_settings()
        except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS:
            settings = None
        if settings and getattr(settings, "AUTH_MODE", None) == "single_user":
            x_api_key = credentials.credentials
            credentials = None

    bearer_token = credentials.credentials if credentials else None
    bearer_is_jwt = _looks_like_jwt(bearer_token) if bearer_token else False

    api_key_candidate = x_api_key
    if not api_key_candidate and bearer_token and not bearer_is_jwt:
        api_key_candidate = bearer_token

    # If Authorization is absent or not a JWT but API key present, attempt API-key auth.
    if api_key_candidate and (not credentials or not bearer_is_jwt):
        return await _authenticate_api_key_from_request(request, api_key_candidate)

    # Otherwise, require Bearer token
    if not credentials:
        # TEST_MODE: surface why we failed
        extra_headers = {"WWW-Authenticate": "Bearer"}
        if _is_test_mode():
            try:
                present_headers = ",".join(h for h in ("Authorization", "X-API-KEY") if request.headers.get(h)) or "none"
                extra_headers["X-TLDW-Auth-Reason"] = "missing-bearer"
                extra_headers["X-TLDW-Auth-Headers"] = present_headers
            except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as exc:  # noqa: BLE001
                logger.debug(
                    "get_current_user: failed to set TEST_MODE missing-bearer diagnostic headers: {}",
                    exc,
                )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers=extra_headers
        )

    # Bearer token path (JWT-based authentication)
    token = credentials.credentials
    try:
        # Delegate JWT validation and user enrichment to the shared helper so
        # roles/permissions/admin flags stay aligned with AuthPrincipal/User flows.
        user_obj = await verify_jwt_and_fetch_user(request, token)
    except InactiveUserError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user",
        ) from exc
    except HTTPException as exc:
        # If a JWT fails but an explicit X-API-KEY is present, fall back to API key auth.
        if exc.status_code == status.HTTP_401_UNAUTHORIZED and x_api_key:
            try:
                return await _authenticate_api_key_from_request(request, x_api_key)
            except HTTPException:
                # If API key auth fails, continue with the JWT failure handling below.
                pass
        if exc.status_code == status.HTTP_401_UNAUTHORIZED:
            # Normalize 401 semantics for callers: detail must contain
            # "Authentication required" and include WWW-Authenticate header.
            extra_headers = {"WWW-Authenticate": "Bearer"}
            if _is_test_mode():
                try:
                    extra_headers["X-TLDW-Auth-Reason"] = f"auth-error:{exc.detail}"
                except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as inner_exc:  # noqa: BLE001
                    logger.debug(
                        "get_current_user: failed to set TEST_MODE auth-error diagnostic header: {}",
                        inner_exc,
                    )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers=extra_headers,
            ) from exc
        # Propagate non-401 HTTP errors unchanged.
        raise
    except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as e:
        if _is_test_mode():
            logger.exception("Authentication error in get_current_user (TEST_MODE)")
        else:
            logger.error(
                "Authentication error in get_current_user (type={})",
                type(e).__name__,
            )
        extra_headers = {"WWW-Authenticate": "Bearer"}
        if _is_test_mode():
            try:
                extra_headers["X-TLDW-Auth-Reason"] = f"auth-error:{e}"
            except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as exc:  # noqa: BLE001
                logger.debug(
                    "get_current_user: failed to set TEST_MODE auth-error diagnostic header: {}",
                    exc,
                )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers=extra_headers,
        ) from e

    # Successful JWT authentication: normalize the User model to a public dict.
    if hasattr(user_obj, "model_dump"):
        user_dict = user_obj.model_dump()  # type: ignore[call-arg]
    elif hasattr(user_obj, "dict"):
        user_dict = user_obj.dict()  # type: ignore[call-arg]
    elif isinstance(user_obj, Mapping):
        user_dict = dict(user_obj)
    else:
        user_dict = {"id": getattr(user_obj, "id", None)}

    safe_user = _public_user_dict(user_dict)

    # Ensure request.state.user_id is populated for downstream consumers, even
    # though verify_jwt_and_fetch_user already attempts to do this.
    try:
        uid = safe_user.get("id")
        if uid is not None:
            request.state.user_id = int(uid)
    except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as state_exc:
        logger.debug(
            "JWT path: unable to attach user_id from verify_jwt_and_fetch_user: {}",
            state_exc,
        )

    return safe_user


#######################################################################################################################
#
# Claim-First Principal Dependencies


def _coerce_optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _normalize_claim_values(values: Any) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple, set)):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(text)
    return normalized


def _mapping_from_user_like(user_obj: Any) -> dict[str, Any]:
    if isinstance(user_obj, Mapping):
        return dict(user_obj)
    if hasattr(user_obj, "model_dump"):
        try:
            dumped = user_obj.model_dump()  # type: ignore[attr-defined]
            if isinstance(dumped, Mapping):
                return dict(dumped)
        except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS:
            pass
    if hasattr(user_obj, "dict"):
        try:
            dumped = user_obj.dict()  # type: ignore[attr-defined]
            if isinstance(dumped, Mapping):
                return dict(dumped)
        except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS:
            pass

    return {
        "id": getattr(user_obj, "id", None),
        "user_id": getattr(user_obj, "user_id", None),
        "username": getattr(user_obj, "username", None),
        "email": getattr(user_obj, "email", None),
        "role": getattr(user_obj, "role", None),
        "roles": getattr(user_obj, "roles", []),
        "permissions": getattr(user_obj, "permissions", []),
        "is_admin": getattr(user_obj, "is_admin", None),
        "org_ids": getattr(user_obj, "org_ids", []),
        "team_ids": getattr(user_obj, "team_ids", []),
        "active_org_id": getattr(user_obj, "active_org_id", None),
        "active_team_id": getattr(user_obj, "active_team_id", None),
        "subject": getattr(user_obj, "subject", None),
        "token_type": getattr(user_obj, "token_type", None),
        "jti": getattr(user_obj, "jti", None),
    }


def _principal_from_legacy_active_user_override(
    request: Request,
    user_obj: Any,
) -> AuthPrincipal:
    data = _mapping_from_user_like(user_obj)
    user_id = _coerce_optional_int(data.get("id"))
    if user_id is None:
        user_id = _coerce_optional_int(data.get("user_id"))

    roles = _normalize_claim_values(data.get("roles"))
    legacy_role = str(data.get("role") or "").strip()
    if legacy_role:
        roles = _normalize_claim_values([legacy_role, *roles])

    permissions = _normalize_claim_values(data.get("permissions"))
    permissions_lc = {str(value).strip().lower() for value in permissions}

    is_admin = bool(data.get("is_admin"))
    if not is_admin:
        roles_lc = {str(value).strip().lower() for value in roles}
        is_admin = ("admin" in roles_lc) or bool(permissions_lc & {"*", "system.configure"})

    org_ids = [
        int(org_id)
        for org_id in (data.get("org_ids") or getattr(request.state, "org_ids", []) or [])
        if _coerce_optional_int(org_id) is not None
    ]
    team_ids = [
        int(team_id)
        for team_id in (data.get("team_ids") or getattr(request.state, "team_ids", []) or [])
        if _coerce_optional_int(team_id) is not None
    ]
    active_org_id = _coerce_optional_int(
        data.get("active_org_id", getattr(request.state, "active_org_id", None))
    )
    active_team_id = _coerce_optional_int(
        data.get("active_team_id", getattr(request.state, "active_team_id", None))
    )
    api_key_id = _coerce_optional_int(data.get("api_key_id"))

    username = data.get("username")
    email = data.get("email")
    subject = data.get("subject")
    token_type = data.get("token_type")
    jti = data.get("jti")

    return AuthPrincipal(
        kind="user",
        user_id=user_id,
        api_key_id=api_key_id,
        username=str(username) if username is not None else None,
        email=str(email) if email is not None else None,
        subject=str(subject) if subject else None,
        token_type=str(token_type) if token_type else "access",
        jti=str(jti) if jti else None,
        roles=roles,
        permissions=permissions,
        is_admin=is_admin,
        org_ids=org_ids,
        team_ids=team_ids,
        active_org_id=active_org_id,
        active_team_id=active_team_id,
    )


def _build_auth_context_from_principal(
    request: Request,
    principal: AuthPrincipal,
) -> AuthContext:
    try:
        ip = request.client.host if request.client else None  # type: ignore[union-attr]
    except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS:
        ip = None
    try:
        user_agent = request.headers.get("User-Agent")
    except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS:
        user_agent = None
    try:
        request_id = (
            request.headers.get("X-Request-ID")
            or getattr(request.state, "request_id", None)
        )
    except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS:
        request_id = None
    return AuthContext(
        principal=principal,
        ip=ip,
        user_agent=user_agent,
        request_id=request_id,
    )


async def _get_legacy_active_user_override_principal(
    request: Request,
) -> AuthPrincipal | None:
    """
    Compatibility shim for tests overriding ``get_current_active_user``.

    Some legacy tests still monkeypatch get_current_active_user while newer
    routes depend directly on get_auth_principal. When an explicit dependency
    override for get_current_active_user is present, honor it and synthesize a
    principal from the override payload.
    """
    app = getattr(request, "app", None)
    if app is None:
        return None
    overrides = getattr(app, "dependency_overrides", None)
    if not isinstance(overrides, Mapping):
        return None
    override_fn = overrides.get(get_current_active_user)
    if override_fn is None:
        return None

    try:
        sig = inspect.signature(override_fn)
        has_no_params = len(sig.parameters) == 0
    except (TypeError, ValueError):
        has_no_params = False

    try:
        result = override_fn() if has_no_params else override_fn(request)
    except TypeError:
        result = override_fn()

    if inspect.isawaitable(result):
        result = await result
    if result is None:
        return None

    principal = (
        result
        if isinstance(result, AuthPrincipal)
        else _principal_from_legacy_active_user_override(request, result)
    )
    ctx = _build_auth_context_from_principal(request, principal)
    try:
        request.state.auth = ctx
        request.state._auth_user = result
        request.state.user_id = principal.user_id
        request.state.api_key_id = principal.api_key_id
        request.state.org_ids = list(principal.org_ids or [])
        request.state.team_ids = list(principal.team_ids or [])
        request.state.active_org_id = principal.active_org_id
        request.state.active_team_id = principal.active_team_id
    except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as state_exc:
        logger.debug(
            "Legacy active-user override principal context attach failed: {}",
            state_exc,
        )
    return principal


def _has_legacy_request_user_override(request: Request) -> bool:
    """
    Return True when tests explicitly override get_request_user.

    This preserves compatibility for tests that inject user context via
    dependency overrides while exercising routes that also include
    route-level require_token_scope dependencies.
    """
    app = getattr(request, "app", None)
    if app is None:
        return False
    overrides = getattr(app, "dependency_overrides", None)
    if not isinstance(overrides, Mapping):
        return False
    return overrides.get(get_request_user) is not None


def _should_promote_claimless_request_user_override(
    user_obj: Any,
) -> bool:
    """Return True when a test override should mirror single-user admin claims."""

    try:
        settings = get_settings()
    except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS:
        return False

    if str(getattr(settings, "AUTH_MODE", "")).strip().lower() != "single_user":
        return False

    data = _mapping_from_user_like(user_obj)
    roles = _normalize_claim_values(data.get("roles"))
    permissions = _normalize_claim_values(data.get("permissions"))
    legacy_role = str(data.get("role") or "").strip().lower()
    subject = str(data.get("subject") or "").strip()
    explicit_admin = data.get("is_admin")

    return (
        not roles
        and not permissions
        and legacy_role in {"", "user"}
        and not subject
        and explicit_admin in (None, False, 0, "0", "")
    )


def _build_single_user_override_payload(user_obj: Any) -> dict[str, Any]:
    """Promote a claimless get_request_user override to single-user admin semantics."""

    data = _mapping_from_user_like(user_obj)
    single_user = get_single_user_instance()
    permissions = list(getattr(single_user, "permissions", []) or ["*"])

    user_id = data.get("id")
    if _coerce_optional_int(user_id) is None:
        fallback_id = data.get("user_id")
        user_id = fallback_id if _coerce_optional_int(fallback_id) is not None else getattr(single_user, "id", 1)

    promoted = dict(data)
    promoted["id"] = user_id
    promoted["username"] = promoted.get("username") or getattr(single_user, "username", "single_user")
    promoted["role"] = "admin"
    promoted["roles"] = ["admin"]
    promoted["permissions"] = permissions
    promoted["is_admin"] = True
    promoted["subject"] = promoted.get("subject") or "single_user"
    return promoted


async def _get_legacy_request_user_override_principal(
    request: Request,
) -> AuthPrincipal | None:
    """Honor explicit get_request_user overrides when get_auth_principal is requested."""

    app = getattr(request, "app", None)
    if app is None:
        return None
    overrides = getattr(app, "dependency_overrides", None)
    if not isinstance(overrides, Mapping):
        return None
    override_fn = overrides.get(get_request_user)
    if override_fn is None:
        return None

    try:
        sig = inspect.signature(override_fn)
        has_no_params = len(sig.parameters) == 0
    except (TypeError, ValueError):
        has_no_params = False

    try:
        result = override_fn() if has_no_params else override_fn(request)
    except TypeError:
        result = override_fn()

    if inspect.isawaitable(result):
        result = await result
    if result is None:
        return None

    principal_source = (
        _build_single_user_override_payload(result)
        if _should_promote_claimless_request_user_override(result)
        else result
    )
    principal = (
        principal_source
        if isinstance(principal_source, AuthPrincipal)
        else _principal_from_legacy_active_user_override(request, principal_source)
    )
    ctx = _build_auth_context_from_principal(request, principal)
    try:
        request.state.auth = ctx
        request.state._auth_user = principal_source
        request.state.user_id = principal.user_id
        request.state.api_key_id = principal.api_key_id
        request.state.org_ids = list(principal.org_ids or [])
        request.state.team_ids = list(principal.team_ids or [])
        request.state.active_org_id = principal.active_org_id
        request.state.active_team_id = principal.active_team_id
    except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as state_exc:
        logger.debug(
            "Legacy request-user override principal context attach failed: {}",
            state_exc,
        )
    return principal


async def get_auth_principal(
    request: Request,
) -> AuthPrincipal:
    """
    FastAPI dependency that returns the AuthPrincipal for the current request.

    This delegates to the core auth_principal_resolver and reuses any existing
    AuthContext attached to request.state.auth when present.
    """
    legacy_override_principal = await _get_legacy_active_user_override_principal(request)
    if legacy_override_principal is not None:
        return legacy_override_principal

    legacy_request_user_principal = await _get_legacy_request_user_override_principal(request)
    if legacy_request_user_principal is not None:
        return legacy_request_user_principal

    principal = await _resolve_auth_principal(request)
    try:
        from tldw_Server_API.app.services.admin_system_ops_service import (
            get_maintenance_state as _get_maintenance_state,
        )

        state = _get_maintenance_state()
        if state.get("enabled"):
            if _principal_has_admin_bypass_claims(principal):
                return principal
            allowlist_ids = set()
            for val in state.get("allowlist_user_ids") or []:
                try:
                    if val is not None:
                        allowlist_ids.add(int(val))
                except (TypeError, ValueError):
                    continue
            allowlist_emails = {
                str(val).strip().lower()
                for val in (state.get("allowlist_emails") or [])
                if val
            }
            if principal.user_id is not None and principal.user_id in allowlist_ids:
                return principal
            if principal.email and principal.email.lower() in allowlist_emails:
                return principal
            message = state.get("message") or "Service temporarily unavailable for maintenance."
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"code": "maintenance_mode", "message": message},
            )
    except HTTPException:
        raise
    except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug("Maintenance guard skipped: error_type={}", type(exc).__name__)
    return principal


def require_permissions(*permissions: str) -> Callable[[AuthPrincipal], Awaitable[AuthPrincipal]]:
    """
    Dependency factory that enforces required permission claims on the principal.

    Admin-style principals (explicit role/permission claims) are allowed
    regardless of specific permissions. On failure, raises HTTP 403 with a
    descriptive message.

    Note: Uses AND semantics - the principal must have all specified
    permissions (unlike require_roles, which uses OR semantics for roles).
    """

    perms = [str(p) for p in permissions if str(p).strip()]

    async def _checker(principal: AuthPrincipal = Depends(get_auth_principal)) -> AuthPrincipal:  # noqa: B008
        if _principal_has_admin_bypass_claims(principal):
            return principal
        missing = [p for p in perms if p not in principal.permissions]
        if missing:
            if _is_test_mode():
                logger.debug("require_permissions denied principal; missing={}", missing)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: missing {', '.join(missing)}",
            )
        return principal

    return _checker


def require_api_key_scope(
    *scopes: str,
    allow_jwt_bypass: bool = True,
    allow_admin_bypass: bool = True,
) -> Callable[..., Awaitable[AuthPrincipal]]:
    """
    Dependency factory that enforces API key scope requirements.

    Creates a FastAPI dependency that verifies the authenticated principal
    has one of the required scopes. This is specifically for API key scope
    enforcement (separate from JWT token scopes or role-based permissions).
    If no scopes are provided, no additional scope constraint is applied.

    Args:
        *scopes: One or more scope strings that satisfy the requirement (OR logic).
                 Valid values: "read", "write", "admin", "service"
        allow_jwt_bypass: If True, JWT-authenticated users bypass this check (default: True)
        allow_admin_bypass: If True, principals with explicit admin role/permission
            claims bypass this check (default: True)

    Returns:
        Dependency function that raises HTTP 403 if scope requirement not met

    Example:
        @router.get("/media/{id}")
        async def get_media(
            id: int,
            _: AuthPrincipal = Depends(require_api_key_scope("read")),
        ):
            ...

        @router.post("/media/process")
        async def process_media(
            request: MediaProcessRequest,
            _: AuthPrincipal = Depends(require_api_key_scope("write", "admin")),
        ):
            ...
    """
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import has_scope, normalize_scope

    required_scopes = frozenset(s.strip().lower() for s in scopes if s)

    async def _checker(
        request: Request,
        principal: AuthPrincipal = Depends(get_auth_principal),  # noqa: B008
    ) -> AuthPrincipal:
        if not required_scopes:
            return principal

        # Admin bypass
        if allow_admin_bypass and _principal_has_admin_bypass_claims(principal):
            # AUDIT: Log when admin bypasses API key scope check for security visibility
            logger.info(
                "Admin user {} bypassing API key scope check for scopes {} on endpoint {}",
                principal.user_id, list(required_scopes), request.url.path
            )
            return principal

        # JWT bypass (for JWT-authenticated users without API key context)
        if allow_jwt_bypass and principal.kind == "user" and principal.api_key_id is None:
            return principal

        # API key scope check
        if principal.api_key_id is not None:
            # Retrieve scope from request.state (populated during auth)
            key_scope = getattr(request.state, "_api_key_scope", None)
            if key_scope is None:
                # Fallback: allow only for explicit single-user principals.
                if getattr(principal, "subject", None) == "single_user":
                    return principal
                # In multi-user mode, missing scope is an error
                if _is_test_mode():
                    logger.debug("require_api_key_scope: scope info unavailable for key {}", principal.api_key_id)
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="API key scope information unavailable",
                )

            key_scopes = normalize_scope(key_scope)

            if not any(has_scope(key_scopes, rs) for rs in required_scopes):
                if _is_test_mode():
                    logger.debug(
                        "require_api_key_scope denied: key_scopes={}, required_any_of={}",
                        key_scopes, required_scopes
                    )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"API key lacks required scope. Required: {', '.join(required_scopes)}",
                )

        return principal

    return _checker


def require_roles(*roles: str) -> Callable[[AuthPrincipal], Awaitable[AuthPrincipal]]:
    """
    Dependency factory that enforces required role claims on the principal.

    Admin-style principals (explicit role/permission claims) are allowed
    regardless of specific roles. On failure, raises HTTP 403 with a
    descriptive message.

    Note: Uses OR semantics - the principal must have at least one of the
    specified roles (unlike require_permissions, which requires all listed
    permissions).
    """

    role_list = [str(r) for r in roles if str(r).strip()]

    async def _checker(principal: AuthPrincipal = Depends(get_auth_principal)) -> AuthPrincipal:  # noqa: B008
        if _principal_has_admin_bypass_claims(principal):
            return principal
        if not role_list:
            return principal
        if not any(r in principal.roles for r in role_list):
            if _is_test_mode():
                logger.debug("require_roles denied principal; required_any_of={}", role_list)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role(s): {', '.join(role_list)}",
            )
        return principal

    return _checker


async def require_service_principal(
    principal: AuthPrincipal = Depends(get_auth_principal),  # noqa: B008
) -> AuthPrincipal:
    """
    Dependency that enforces a service principal.

    Behavior:
    - Relies on get_auth_principal for authentication (401 on failure).
    - When a principal is present but principal.kind is not "service", raises
      HTTP 403 with a stable, descriptive message.
    - When principal.kind == "service", returns the principal unchanged.
    """
    if principal.kind != "service":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Service principal required",
        )
    return principal


async def get_current_active_user(
    current_user: dict[str, Any] = Depends(get_current_user)
) -> dict[str, Any]:
    """
    Legacy compatibility shim that returns an active user dictionary.

    New or migrated route code should prefer ``CurrentPrincipal`` /
    ``get_auth_principal`` unless dictionary-shaped user data is still needed.

    Args:
        current_user: Current authenticated user

    Returns:
        User dictionary if active and verified

    Raises:
        HTTPException: If user is inactive or unverified
    """
    if not current_user.get("is_active"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )

    if not current_user.get("is_verified"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email verification required"
        )

    return current_user


async def _load_org_policy(db: Any, org_id: int) -> dict[str, Any]:
    """
    Internal helper to load an organization policy with consistent error handling.
    """
    try:
        pol = await get_policy(db, org_id)
    except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as exc:
        logger.exception(
            "Failed to load organization policy for org_id={}",
            org_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to load organization policy",
        ) from exc
    else:
        if not pol:
            pol = get_default_policy_from_env(org_id)
        return pol


async def get_org_policy_from_principal(
    db: Any = Depends(get_db_transaction),  # noqa: B008
    principal: AuthPrincipal = Depends(get_auth_principal),  # noqa: B008
) -> dict[str, Any]:
    """
    Resolve organization policy from ``AuthPrincipal`` org claims.

    Preference order:
    1. First org_id in ``principal.org_ids`` (claim-first).
    2. Synthetic org_id=1 for explicit single-user principals (subject=single_user).
    3. HTTP 400 when no organization can be resolved.

    This helper is intended for code paths that already depend on
    ``get_auth_principal``.
    """
    def _should_use_synthetic_single_user_org(p: AuthPrincipal) -> bool:
        """
        Decide whether to fall back to synthetic org_id=1.

        Stage 4 claim-first tightening: org-policy fallback is derived from
        principal identity only (no mode/profile helper branch).
        """
        # Principal-first: only explicit single-user principals qualify. For the
        # org-policy fallback, we deliberately require the principal to be
        # tagged with subject \"single_user\" instead of relying on numeric
        # fixed-id fallbacks to avoid misclassifying arbitrary principals that
        # happen to share the single-user id.
        try:
            return getattr(p, "subject", None) == "single_user"
        except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS:
            return False

    # 1) Claim-first: use principal.org_ids when available.
    org_ids = list(getattr(principal, "org_ids", []) or [])
    if org_ids:
        org_id = org_ids[0]
    else:
        if _should_use_synthetic_single_user_org(principal):
            # Single-user environment: synthetic org_id=1 to mirror legacy behaviour.
            org_id = 1
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User has no organization memberships",
            )

    if org_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Organization membership is missing org_id",
        )
    return await _load_org_policy(db, org_id)


_ADMIN_BYPASS_PERMISSIONS = frozenset({"*"})
_ADMIN_CLAIM_PERMISSIONS = frozenset({"*", "system.configure"})


def _normalized_claim_values(values: list[Any] | tuple[Any, ...] | None) -> set[str]:
    return {
        str(value).strip().lower()
        for value in (values or [])
        if str(value).strip()
    }


def _principal_has_admin_bypass_claims(principal: AuthPrincipal | None) -> bool:
    if principal is None:
        return False
    # Claim-first behavior: do not treat the legacy boolean `is_admin` flag as
    # an authorization bypass on its own. Require explicit admin role/permission
    # claims for bypass.
    roles = _normalized_claim_values(principal.roles)
    permissions = _normalized_claim_values(principal.permissions)
    if "admin" in roles:
        return True
    return bool(permissions & _ADMIN_BYPASS_PERMISSIONS)


def _principal_has_admin_claims(principal: AuthPrincipal | None) -> bool:
    if principal is None:
        return False
    roles = _normalized_claim_values(principal.roles)
    permissions = _normalized_claim_values(principal.permissions)
    if "admin" in roles:
        return True
    return bool(permissions & _ADMIN_CLAIM_PERMISSIONS)


#######################################################################################################################
#
# Rate Limiting Dependencies

def _rg_enabled_for_request(request: Request) -> bool:
    state = getattr(request, "state", None)
    return bool(state is not None and getattr(state, "rg_policy_id", None))


def _rg_enabled_flag() -> bool:
    raw = os.getenv("RG_ENABLED", "")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _log_auth_deps_rg_diagnostics_only_shim(
    *,
    dependency: str,
    endpoint: str,
    reason: str,
) -> None:
    key = f"{dependency}:{reason}"
    if key in _AUTH_DEPS_RG_DIAGNOSTICS_ONLY_LOGGED:
        return
    _AUTH_DEPS_RG_DIAGNOSTICS_ONLY_LOGGED.add(key)
    logger.warning(
        "Auth dependency {} using diagnostics-only shim (reason={}); endpoint={}",
        dependency,
        reason,
        endpoint,
    )


def _auth_deps_rate_limit_identifier(request: Request, endpoint: str) -> str:
    """Build a stable fallback limiter key from request principal/user/ip."""
    try:
        state = getattr(request, "state", None)
        if state is not None:
            auth_ctx = getattr(state, "auth", None)
            if isinstance(auth_ctx, AuthContext):
                principal = getattr(auth_ctx, "principal", None)
                principal_id = str(getattr(principal, "principal_id", "") or "").strip()
                if principal_id:
                    return f"principal:{principal_id}:{endpoint}"
            user_id = str(getattr(state, "user_id", "") or "").strip()
            if user_id:
                return f"user:{user_id}:{endpoint}"
    except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS:
        pass

    try:
        client = getattr(request, "client", None)
        host = str(getattr(client, "host", "") or "").strip()
    except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS:
        host = ""
    if host:
        return f"ip:{host}:{endpoint}"
    return f"anonymous:{endpoint}"


def _consume_auth_deps_fallback_rate_token(
    *,
    dependency: str,
    identifier: str,
    limit: int,
    window_seconds: float,
) -> tuple[bool, int]:
    """Consume one token from an in-process fallback fixed-window limiter."""
    safe_limit = max(1, int(limit))
    safe_window = max(1.0, float(window_seconds))
    now = time.monotonic()
    cutoff = now - safe_window
    key = (dependency, identifier)

    with _AUTH_DEPS_FALLBACK_RATE_WINDOWS_LOCK:
        bucket = _AUTH_DEPS_FALLBACK_RATE_WINDOWS.get(key)
        if bucket is None:
            bucket = deque()
            _AUTH_DEPS_FALLBACK_RATE_WINDOWS[key] = bucket

        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

        if len(bucket) >= safe_limit:
            retry_after = int(max(1.0, (bucket[0] + safe_window) - now))
            return False, retry_after

        bucket.append(now)
        return True, 0


def _auth_deps_request_principal(request: Request) -> AuthPrincipal | None:
    """Best-effort principal lookup from request state."""
    try:
        ctx = getattr(request.state, "auth", None)
        if isinstance(ctx, AuthContext):
            return ctx.principal
    except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS:
        return None
    return None


async def _enforce_auth_deps_ingress_guard(
    *,
    request: Request,
    dependency: str,
    endpoint: str,
    fallback_limit_env: str,
    fallback_window_env: str,
    fallback_limit_default: int,
    fallback_window_default: float,
    limit_exceeded_detail: str,
) -> None:
    """Apply fail-closed ingress fallback limits when RG policy metadata is absent."""
    if _rg_enabled_for_request(request):
        return

    principal = _auth_deps_request_principal(request)
    if is_single_user_principal(principal):
        return

    # Claim-first governor hook (primarily for test invariants)
    await get_auth_governor()

    # In test mode, bypass rate limiting entirely for deterministic tests.
    if _is_test_mode():
        return

    try:
        fallback_limit = _read_non_negative_int_env(fallback_limit_env, fallback_limit_default)
        fallback_window = _read_non_negative_float_env(fallback_window_env, fallback_window_default)
        identifier = _auth_deps_rate_limit_identifier(request, endpoint)
        allowed, retry_after = _consume_auth_deps_fallback_rate_token(
            dependency=dependency,
            identifier=identifier,
            limit=fallback_limit,
            window_seconds=fallback_window,
        )
    except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(
            "Auth dependency {} fallback enforcement failed (endpoint={}): {}",
            dependency,
            endpoint,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Rate limiting temporarily unavailable",
        ) from exc

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=limit_exceeded_detail,
            headers={"Retry-After": str(retry_after)},
        )


async def check_rate_limit(request: Request, rate_limiter=None) -> None:
    """General ingress rate-limit guard with fail-closed fallback enforcement."""
    _ = rate_limiter
    endpoint = request.url.path if getattr(request, "url", None) else "unknown"
    await _enforce_auth_deps_ingress_guard(
        request=request,
        dependency="check_rate_limit",
        endpoint=endpoint,
        fallback_limit_env="AUTH_DEPS_FALLBACK_RATE_LIMIT",
        fallback_window_env="AUTH_DEPS_FALLBACK_RATE_WINDOW_SECONDS",
        fallback_limit_default=120,
        fallback_window_default=60.0,
        limit_exceeded_detail=f"Rate limit exceeded for endpoint: {endpoint}",
    )


async def check_auth_rate_limit(request: Request, rate_limiter=None) -> None:
    """Auth endpoint ingress rate-limit guard with fail-closed fallback enforcement."""
    _ = rate_limiter
    endpoint = request.url.path if getattr(request, "url", None) else "auth"
    await _enforce_auth_deps_ingress_guard(
        request=request,
        dependency="check_auth_rate_limit",
        endpoint=endpoint,
        fallback_limit_env="AUTH_DEPS_AUTH_FALLBACK_RATE_LIMIT",
        fallback_window_env="AUTH_DEPS_AUTH_FALLBACK_RATE_WINDOW_SECONDS",
        fallback_limit_default=30,
        fallback_window_default=60.0,
        limit_exceeded_detail=f"Auth rate limit exceeded for endpoint: {endpoint}",
    )


# ---------------------------------------------------------------------------------
# RBAC resource-aware rate limit (stub - logs selected limits, no enforcement yet)

async def enforce_rbac_rate_limit(
    request: Request,
    resource: str,
    db_pool: DatabasePool = Depends(get_db_pool)
):
    """
    Resource-aware rate limit selector (stub).

    Reads the strictest configured limit for the current user from rbac_user_rate_limits
    and rbac_role_rate_limits. Currently logs selected limits without enforcing.
    """
    try:
        user_id = getattr(request.state, 'user_id', None)
        if not user_id:
            # Unknown user context; skip
            return

        # User-level limit
        user_limit = None
        role_limit = None

        # SQLite vs Postgres param binding
        if db_pool.pool:  # Postgres
            user_limit = await db_pool.fetchone(
                "SELECT limit_per_min, burst FROM rbac_user_rate_limits WHERE user_id = $1 AND resource = $2",
                user_id, resource
            )
            # Get roles for user
            role_ids = await db_pool.fetchall(
                """
                SELECT role_id FROM user_roles
                WHERE user_id = $1 AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                """,
                user_id
            )
            if role_ids:
                role_ids_list = [r['role_id'] for r in role_ids]
                # Take the strictest (min) among role limits
                role_limit = await db_pool.fetchone(
                    """
                    SELECT MIN(limit_per_min) as limit_per_min, MIN(burst) as burst
                    FROM rbac_role_rate_limits WHERE role_id = ANY($1) AND resource = $2
                    """,
                    role_ids_list, resource
                )
        else:  # SQLite
            async with db_pool.acquire() as conn:
                c1 = await conn.execute(
                    "SELECT limit_per_min, burst FROM rbac_user_rate_limits WHERE user_id = ? AND resource = ?",
                    (user_id, resource)
                )
                user_limit = await c1.fetchone()
                c2 = await conn.execute(
                    """
                    SELECT MIN(rl.limit_per_min), MIN(rl.burst)
                    FROM rbac_role_rate_limits rl
                    JOIN user_roles ur ON ur.role_id = rl.role_id
                    WHERE ur.user_id = ? AND (ur.expires_at IS NULL OR ur.expires_at > CURRENT_TIMESTAMP)
                      AND rl.resource = ?
                    """,
                    (user_id, resource)
                )
                role_limit = await c2.fetchone()

        # Choose strictest (lowest) effective limits
        candidates = []
        if user_limit:
            lp = user_limit[0] if not isinstance(user_limit, dict) else user_limit.get('limit_per_min')
            bp = user_limit[1] if not isinstance(user_limit, dict) else user_limit.get('burst')
            candidates.append((lp, bp))
        if role_limit:
            lp = role_limit[0] if not isinstance(role_limit, dict) else role_limit.get('limit_per_min')
            bp = role_limit[1] if not isinstance(role_limit, dict) else role_limit.get('burst')
            candidates.append((lp, bp))

        if candidates:
            limit_per_min = min([c[0] for c in candidates if c[0] is not None]) if any(c[0] for c in candidates) else None
            burst = min([c[1] for c in candidates if c[1] is not None]) if any(c[1] for c in candidates) else None
            logger.debug(
                "RBAC rate-limit selected for user {}, resource {}: rpm={}, burst={}",
                user_id,
                resource,
                limit_per_min,
                burst,
            )
        else:
            logger.debug("RBAC rate-limit: no configured limits for user {}, resource {}", user_id, resource)
    except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as e:
        logger.debug("RBAC rate-limit selection failed: error_type={}", type(e).__name__)


def rbac_rate_limit(resource: str):
    """Factory returning a dependency that logs selected RBAC limits for the given resource."""
    async def _dep(request: Request, db_pool: DatabasePool = Depends(get_db_pool)):
        await enforce_rbac_rate_limit(request, resource, db_pool)
    try:
        _dep._tldw_rate_limit_resource = resource
    except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(
            "rbac_rate_limit: unable to attach rate-limit metadata to dependency: {}",
            exc,
        )
    return _dep


#######################################################################################################################
#
# Scoped Virtual-Key Enforcement

_VK_USAGE: dict = {}
_VK_USAGE_LOCK = threading.Lock()


def _vk_usage_ttl_seconds() -> int:
    try:
        ttl = int(os.getenv("VK_USAGE_TTL_SECONDS", "3600"))
    except (TypeError, ValueError):
        ttl = 3600
    return max(ttl, 60)


def _vk_usage_check_and_increment(key: object, limit: int) -> bool:
    """Process-local quota fallback with TTL to avoid unbounded growth."""
    now = time.monotonic()
    ttl = _vk_usage_ttl_seconds()
    with _VK_USAGE_LOCK:
        # Opportunistic pruning on each fallback use.
        if _VK_USAGE:
            expired = [k for k, (_cnt, exp) in _VK_USAGE.items() if exp <= now]
            for k in expired:
                _VK_USAGE.pop(k, None)
        entry = _VK_USAGE.get(key)
        if entry:
            count, expires_at = entry
            if expires_at <= now:
                count = 0
        else:
            count = 0
        if count >= int(limit):
            return False
        _VK_USAGE[key] = (count + 1, now + ttl)
        return True


def require_token_scope(
    scope: str,
    *,
    require_if_present: bool = True,
    require_schedule_match: bool = False,
    schedule_path_param: str = "schedule_id",
    schedule_header: str = "X-Workflow-Schedule-Id",
    allow_admin_bypass: bool = True,
    endpoint_id: Optional[str] = None,
    count_as: Optional[str] = None,
):
    """
    Create a dependency that enforces a scoped JWT ("virtual key").

    Behavior:
    - If the Authorization bearer token contains a 'scope' claim and require_if_present=True,
      it must match the provided `scope` or 403.
    - If `require_schedule_match=True` and the token includes 'schedule_id', the value must
      match the request path param `schedule_path_param` when present, or header `schedule_header`.
    - When `require_if_present=True`, missing/invalid credentials fail closed (401).
      API keys are accepted via `X-API-KEY` or Authorization bearer (non-JWT).
    - If `allow_admin_bypass=True`, admin users skip this enforcement.
    - If the bearer token is not a JWT, enforce API key constraints using that token.
    """
    async def _checker(
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
        jwt_service: JWTService = Depends(get_jwt_service_dep),
        db_pool: DatabasePool = Depends(get_db_pool),
    ) -> None:
        # Allow direct invocation (e.g., pytest unit tests) by resolving Depends defaults manually
        try:
            from fastapi.params import Depends as _Depends  # Local import to avoid top-level dependency

            if isinstance(jwt_service, _Depends) or jwt_service is None:
                jwt_service = await get_jwt_service_dep()
            if isinstance(db_pool, _Depends) or db_pool is None:
                db_pool = await get_db_pool()
        except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as exc:  # noqa: BLE001
            # Best effort: if resolution fails, leave as-is; downstream code handles missing services.
            logger.debug(
                "require_token_scope: dependency resolution failed; continuing with provided services: {}",
                exc,
            )

        token = credentials.credentials if credentials else None
        token_is_jwt = _looks_like_jwt(token) if token else False
        legacy_request_user_override = False
        if not token:
            try:
                legacy_request_user_override = _has_legacy_request_user_override(request)
            except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(
                    "require_token_scope: override detection failed; continuing with normal auth: {}",
                    exc,
                )
                legacy_request_user_override = False

        try:
            if endpoint_id is not None:
                request.state._auth_endpoint_id = str(endpoint_id)
            if count_as is not None:
                request.state._auth_action = str(count_as)
            if endpoint_id is not None or count_as is not None:
                request.state._auth_scope_name = str(scope)
        except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as exc:  # noqa: BLE001
            logger.debug(
                "require_token_scope: unable to attach endpoint/action context to request.state: {}",
                exc,
            )

        # Optional admin bypass based on explicit principal claims (not token claims).
        if allow_admin_bypass:
            principal = None
            try:
                ctx = getattr(request.state, "auth", None)
                if isinstance(ctx, AuthContext):
                    principal = ctx.principal
            except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS:
                principal = None
            if principal is None and credentials:
                try:
                    resolver = get_auth_principal
                    try:
                        app = getattr(request, "app", None)
                        if app is not None:
                            override_fn = app.dependency_overrides.get(get_auth_principal)
                            if override_fn is not None:
                                resolver = override_fn
                    except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS:
                        resolver = get_auth_principal

                    if resolver is get_auth_principal:
                        principal = await resolver(request)
                    else:
                        try:
                            sig = inspect.signature(resolver)
                            has_no_params = len(sig.parameters) == 0
                        except (TypeError, ValueError):
                            has_no_params = False
                        try:
                            result = resolver() if has_no_params else resolver(request)
                        except TypeError:
                            result = resolver()
                        if inspect.isawaitable(result):
                            result = await result
                        principal = result
                except HTTPException:
                    principal = None
                except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS:
                    principal = None
            if _principal_has_admin_bypass_claims(principal):
                return None

        # If we have Authorization bearer and it looks like a JWT, apply JWT-based checks.
        if token and token_is_jwt:
            try:
                payload = jwt_service.decode_access_token(token)
            except (InvalidTokenError, TokenExpiredError) as exc:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Could not validate credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                ) from exc
            # Enforce revocation/blacklist checks for scoped JWTs.
            try:
                from tldw_Server_API.app.core.AuthNZ.session_manager import get_session_manager

                session_manager = await get_session_manager()
                if await session_manager.is_token_blacklisted(token, payload.get("jti")):
                    raise HTTPException(status_code=401, detail="Token has been revoked")
            except HTTPException:
                raise
            except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as exc:  # noqa: BLE001
                logger.debug(
                    "require_token_scope: token revocation check failed; denying: {}",
                    exc,
                )
                raise HTTPException(status_code=401, detail="Could not validate credentials") from exc
            tok_scope = str(payload.get("scope") or "").strip()
            if tok_scope:
                try:
                    request.state._token_scope_enforced = True
                    request.state._token_scope_claim = tok_scope
                    request.state._token_scope_required = str(scope)
                except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS:  # noqa: BLE001
                    logger.debug("require_token_scope: failed to attach scope enforcement marker to request.state")
            if tok_scope and require_if_present and tok_scope != str(scope):
                raise HTTPException(status_code=403, detail="Forbidden: invalid token scope")

            # Enforce endpoint allowlist
            try:
                if endpoint_id:
                    allowed_eps = payload.get("allowed_endpoints")
                    if isinstance(allowed_eps, list) and allowed_eps:
                        if endpoint_id not in [str(x) for x in allowed_eps]:
                            raise HTTPException(status_code=403, detail="Forbidden: endpoint not permitted for token")
            except HTTPException:
                raise
            except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as exc:  # noqa: BLE001
                logger.debug(
                    "require_token_scope: endpoint allowlist enforcement failed; continuing: {}",
                    exc,
                )

            # Enforce HTTP method allowlist
            try:
                am = payload.get("allowed_methods")
                if isinstance(am, list) and am:
                    method = str(getattr(request, "method", "")).upper()
                    if method and method not in [str(x).upper() for x in am]:
                        raise HTTPException(status_code=403, detail="Forbidden: method not permitted for token")
            except HTTPException:
                raise
            except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as exc:  # noqa: BLE001
                logger.debug(
                    "require_token_scope: method allowlist enforcement failed; continuing: {}",
                    exc,
                )

            # Enforce path prefix allowlist
            try:
                ap = payload.get("allowed_paths")
                if isinstance(ap, list) and ap:
                    path = getattr(getattr(request, "url", None), "path", None) or getattr(request, "scope", {}).get("path")
                    if path and not any(str(path).startswith(str(pfx)) for pfx in ap):
                        raise HTTPException(status_code=403, detail="Forbidden: path not permitted for token")
            except HTTPException:
                raise
            except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as exc:  # noqa: BLE001
                logger.debug(
                    "require_token_scope: path allowlist enforcement failed; continuing: {}",
                    exc,
                )

            # Quotas: simple per-token counters (DB-backed; fallback to process-local)
            try:
                if count_as:
                    jti = payload.get("jti")
                    if jti:
                        key = (f"jwt:{jti}", str(count_as))
                        max_calls = None
                        if str(count_as) == "run":
                            max_calls = payload.get("max_runs")
                        if max_calls is None:
                            max_calls = payload.get("max_calls")
                        if isinstance(max_calls, int) and max_calls >= 0:
                            try:
                                from tldw_Server_API.app.core.AuthNZ.quotas import increment_and_check_jwt_quota
                                allowed, _cnt = await increment_and_check_jwt_quota(
                                    db_pool=db_pool,
                                    jti=str(jti),
                                    counter_type=str(count_as),
                                    limit=int(max_calls),
                                )
                                if not allowed:
                                    raise HTTPException(status_code=403, detail="Forbidden: token quota exceeded")
                            except HTTPException:
                                raise
                            except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as err:
                                # Defensive: fall back to process-local counters if quota backend fails.
                                if not _vk_usage_check_and_increment(key, int(max_calls)):
                                    raise HTTPException(
                                        status_code=403,
                                        detail="Forbidden: token quota exceeded",
                                    ) from err
            except HTTPException:
                raise
            except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as exc:  # noqa: BLE001
                logger.debug(
                    "require_token_scope: token constraints evaluation failed; continuing: {}",
                    exc,
                )

            if require_schedule_match:
                try:
                    tok_sid = payload.get("schedule_id")
                except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS:  # noqa: BLE001
                    tok_sid = None
                expected = None
                try:
                    expected = request.path_params.get(schedule_path_param)
                except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS:  # noqa: BLE001
                    expected = None
                if expected is None:
                    try:
                        expected = request.headers.get(schedule_header)
                    except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS:  # noqa: BLE001
                        expected = None
                if tok_sid is not None and expected is not None and str(tok_sid) != str(expected):
                    raise HTTPException(status_code=403, detail="Forbidden: schedule scope mismatch")

            return None

        # Fallback: X-API-KEY constraints enforcement (if header present and key is valid)
        try:
            api_key = request.headers.get("X-API-KEY") if getattr(request, "headers", None) else None
        except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS:  # noqa: BLE001
            # Defensive: request headers access should never block the fallback path.
            api_key = None
        if not api_key and token and not token_is_jwt:
            api_key = token
        if not api_key:
            if legacy_request_user_override:
                logger.debug(
                    "require_token_scope: allowing missing credentials due test override of get_request_user"
                )
                return None
            if require_if_present:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return None
        if api_key:
            try:
                from tldw_Server_API.app.core.AuthNZ.api_key_manager import (
                    VALID_SCOPE_VALUES as _VALID_SCOPE_VALUES,
                )
                from tldw_Server_API.app.core.AuthNZ.api_key_manager import (
                    has_scope as _has_scope,
                )
                from tldw_Server_API.app.core.AuthNZ.api_key_manager import (
                    normalize_scope as _normalize_scope,
                )

                def _required_scope_for_api_key(scope_value: str, method: str) -> Optional[str]:
                    scope_norm = str(scope_value or "").strip().lower()
                    if scope_norm in _VALID_SCOPE_VALUES:
                        return scope_norm
                    if method in {"GET", "HEAD", "OPTIONS"}:
                        return "read"
                    if method:
                        return "write"
                    return None

                # Single-user compatibility: allow the configured primary/test API key
                # without requiring a persisted API key record in AuthNZ tables.
                # Keep the same IP allowlist guard as regular single-user auth paths.
                settings = get_settings()
                if getattr(settings, "AUTH_MODE", None) == "single_user":
                    allowed_keys: set[str] = set()
                    primary_key = getattr(settings, "SINGLE_USER_API_KEY", None)
                    if primary_key:
                        allowed_keys.add(primary_key)
                    env_primary_key = os.getenv("SINGLE_USER_API_KEY") or os.getenv("API_KEY")
                    if env_primary_key:
                        allowed_keys.add(env_primary_key)
                    if _is_test_mode():
                        test_key = os.getenv("SINGLE_USER_TEST_API_KEY")
                        if test_key:
                            allowed_keys.add(test_key)
                    if api_key in allowed_keys:
                        client_ip = resolve_client_ip(request, settings)
                        if not is_single_user_ip_allowed(client_ip, settings):
                            raise HTTPException(
                                status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="Invalid or missing API Key",
                            )
                        if allow_admin_bypass:
                            return None

                api_mgr = await get_api_key_manager()
                client_ip = resolve_client_ip(request, settings)
                info = await api_mgr.validate_api_key(
                    api_key=api_key,
                    ip_address=client_ip,
                    record_usage=False,
                )
                if not info:
                    if require_if_present:
                        raise HTTPException(
                            status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Could not validate credentials",
                            headers={"WWW-Authenticate": "Bearer"},
                        )
                    # Optional mode: let upstream auth fail.
                    return None
                # Admin bypass via scope 'admin'
                key_scopes = _normalize_scope(info.get("scope"))
                if allow_admin_bypass and (_has_scope(key_scopes, "admin") or _has_scope(key_scopes, "service")):
                    return None
                required_scope = _required_scope_for_api_key(scope, str(getattr(request, "method", "")).upper())
                if required_scope and not _has_scope(key_scopes, required_scope):
                    raise HTTPException(
                        status_code=403,
                        detail="Forbidden: API key lacks required scope",
                    )
                # Allowed endpoints from llm_allowed_endpoints
                allowed_eps = info.get("llm_allowed_endpoints")
                if isinstance(allowed_eps, str):
                    import json as _json
                    try:
                        allowed_eps = _json.loads(allowed_eps)
                    except (ValueError, TypeError) as parse_exc:
                        logger.debug(
                            "require_token_scope: malformed llm_allowed_endpoints for key {}; denying",
                            info.get("id"),
                        )
                        raise HTTPException(
                            status_code=403,
                            detail="Forbidden: invalid API key endpoint constraints",
                        ) from parse_exc
                if allowed_eps is not None and not isinstance(allowed_eps, list):
                    raise HTTPException(
                        status_code=403,
                        detail="Forbidden: invalid API key endpoint constraints",
                    )
                if endpoint_id and isinstance(allowed_eps, list) and allowed_eps:
                    if endpoint_id not in [str(x) for x in allowed_eps]:
                        raise HTTPException(status_code=403, detail="Forbidden: endpoint not permitted for API key")
                # Metadata-based constraints
                meta = info.get("metadata")
                if isinstance(meta, str):
                    import json as _json
                    try:
                        meta = _json.loads(meta)
                    except (ValueError, TypeError) as parse_exc:
                        logger.debug(
                            "require_token_scope: malformed metadata constraints for key {}; denying",
                            info.get("id"),
                        )
                        raise HTTPException(
                            status_code=403,
                            detail="Forbidden: invalid API key metadata constraints",
                        ) from parse_exc
                if meta is not None and not isinstance(meta, dict):
                    raise HTTPException(
                        status_code=403,
                        detail="Forbidden: invalid API key metadata constraints",
                    )
                if isinstance(meta, dict):
                    am = meta.get("allowed_methods")
                    if am is not None and not isinstance(am, list):
                        raise HTTPException(
                            status_code=403,
                            detail="Forbidden: invalid API key metadata constraints",
                        )
                    if isinstance(am, list) and am:
                        method = str(getattr(request, "method", "")).upper()
                        if method and method not in [str(x).upper() for x in am]:
                            raise HTTPException(status_code=403, detail="Forbidden: method not permitted for API key")
                    ap = meta.get("allowed_paths")
                    if ap is not None and not isinstance(ap, list):
                        raise HTTPException(
                            status_code=403,
                            detail="Forbidden: invalid API key metadata constraints",
                        )
                    if isinstance(ap, list) and ap:
                        path = getattr(getattr(request, "url", None), "path", None) or getattr(request, "scope", {}).get("path")
                        if path and not any(str(path).startswith(str(pfx)) for pfx in ap):
                            raise HTTPException(status_code=403, detail="Forbidden: path not permitted for API key")
                    if count_as:
                        key_id = info.get("id")
                        if key_id is not None:
                            quota = None
                            if str(count_as) == "run":
                                quota = meta.get("max_runs")
                            if quota is None:
                                quota = meta.get("max_calls")
                            if isinstance(quota, int) and quota >= 0:
                                try:
                                    from tldw_Server_API.app.core.AuthNZ.quotas import increment_and_check_api_key_quota
                                    allowed, _cnt = await increment_and_check_api_key_quota(
                                        db_pool=db_pool,
                                        api_key_id=int(key_id),
                                        counter_type=str(count_as),
                                        limit=int(quota),
                                    )
                                    if not allowed:
                                        raise HTTPException(status_code=403, detail="Forbidden: API key quota exceeded")
                                except HTTPException:
                                    raise
                                except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as err:
                                    # Defensive: fall back to process-local counters if quota backend fails.
                                    key = (f"apikey:{key_id}", str(count_as))
                                    if not _vk_usage_check_and_increment(key, int(quota)):
                                        raise HTTPException(
                                            status_code=403,
                                            detail="Forbidden: API key quota exceeded",
                                        ) from err
            except HTTPException:
                raise
            except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as exc:  # noqa: BLE001
                logger.debug(
                    "require_token_scope: API key constraint evaluation failed; denying: {}",
                    type(exc).__name__,
                )
                raise HTTPException(
                    status_code=403,
                    detail="Forbidden: unable to validate API key constraints",
                ) from exc
        return None

    try:
        _checker._tldw_endpoint_id = endpoint_id
        _checker._tldw_count_as = count_as
        _checker._tldw_scope_name = scope
        _checker._tldw_token_scope = True
        _checker._tldw_token_scope_required = str(scope)
    except _AUTH_DEPS_NONCRITICAL_EXCEPTIONS as exc:  # noqa: BLE001
        logger.debug(
            "require_token_scope: unable to attach metadata to dependency: {}",
            exc,
        )

    return _checker


#######################################################################################################################
#
# Phase 3.4 Standard Auth Dependency Surface

CurrentPrincipal = Annotated[AuthPrincipal, Depends(get_auth_principal)]
"""Route dependency alias that resolves and returns the current AuthPrincipal."""

CurrentUserDict = Annotated[dict[str, Any], Depends(get_current_active_user)]
"""Legacy route dependency alias for endpoints that still require user dictionaries."""

_require_admin_principal = require_roles("admin")
AdminPrincipal = Annotated[AuthPrincipal, Depends(_require_admin_principal)]
"""Route dependency alias that returns an AuthPrincipal after admin role checks."""

ServicePrincipal = Annotated[AuthPrincipal, Depends(require_service_principal)]
"""Route dependency alias that returns an AuthPrincipal after service-principal checks."""

RequireRole = require_roles
"""Standard role-guard factory; returns an AuthPrincipal on success."""

RequirePermission = require_permissions
"""Standard permission-guard factory; returns an AuthPrincipal on success."""

RequireApiKeyScope = require_api_key_scope
"""Standard API-key-scope guard factory; returns an AuthPrincipal on success."""

TokenScopeGuard = require_token_scope
"""Standard token-scope guard factory for dependency lists; returns None on success."""
#
# End of auth_deps.py
#######################################################################################################################
