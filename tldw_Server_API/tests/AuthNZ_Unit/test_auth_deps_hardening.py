from datetime import timezone
import io
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException, Response
from fastapi.security import HTTPAuthorizationCredentials
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.core.AuthNZ import migrations
from tldw_Server_API.app.core.AuthNZ.exceptions import DatabaseLockError
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal


class _FailingCommitConn:
    async def execute(self, query: str, params: Any) -> Any:
        return SimpleNamespace()

    async def commit(self) -> None:
        raise RuntimeError("sqlite commit failed")


class _AcquireCM:
    async def __aenter__(self) -> _FailingCommitConn:
        return _FailingCommitConn()

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeDBPool:
    def acquire(self) -> _AcquireCM:
        return _AcquireCM()


class _ExplodingPoolProperty:
    @property
    def pool(self) -> object:
        raise RuntimeError("db pool secret /tmp/authnz.sqlite token=user-123")


class _DummyRequest:
    def __init__(self) -> None:
        self.state = SimpleNamespace()
        self.client = SimpleNamespace(host="127.0.0.1")
        self.method = "GET"
        self.url = SimpleNamespace(path="/test")
        self.headers: dict[str, str] = {}


class _LockingTxnCM:
    def __init__(self, pool: "_LockingPool") -> None:
        self._pool = pool

    async def __aenter__(self) -> object:
        self._pool.enter_calls += 1
        if self._pool.lock_on_enter_remaining > 0:
            self._pool.lock_on_enter_remaining -= 1
            raise DatabaseLockError()
        return self._pool.conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        self._pool.exit_calls += 1
        if self._pool.raise_lock_on_exit:
            raise DatabaseLockError()
        return False


class _LockingPool:
    def __init__(self, *, lock_on_enter_count: int = 0, raise_lock_on_exit: bool = False) -> None:
        self.lock_on_enter_remaining = lock_on_enter_count
        self.raise_lock_on_exit = raise_lock_on_exit
        self.enter_calls = 0
        self.exit_calls = 0
        self.conn = object()

    def transaction(self) -> _LockingTxnCM:
        return _LockingTxnCM(self)

    def acquire(self) -> object:
        raise AssertionError("adapter path should not be used for lock retry tests")


@pytest.mark.asyncio
async def test_test_db_adapter_execute_propagates_sqlite_commit_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_get_db_pool() -> _FakeDBPool:
        return _FakeDBPool()

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setattr(auth_deps, "get_db_pool", _fake_get_db_pool)

    agen = auth_deps.get_db_transaction()
    adapter = await agen.__anext__()
    try:
        with pytest.raises(RuntimeError, match="sqlite commit failed"):
            await adapter.execute("SELECT 1")
    finally:
        await agen.aclose()


@pytest.mark.asyncio
async def test_test_db_adapter_commit_propagates_sqlite_commit_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_get_db_pool() -> _FakeDBPool:
        return _FakeDBPool()

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setattr(auth_deps, "get_db_pool", _fake_get_db_pool)

    agen = auth_deps.get_db_transaction()
    adapter = await agen.__anext__()
    try:
        with pytest.raises(RuntimeError, match="sqlite commit failed"):
            await adapter.commit()
    finally:
        await agen.aclose()


@pytest.mark.asyncio
async def test_stub_session_manager_uses_timezone_aware_timestamps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.delenv("AUTHNZ_FORCE_REAL_SESSION_MANAGER", raising=False)

    sm = await auth_deps.get_session_manager_dep()
    sess = await sm.create_session(
        user_id=1,
        access_token="access",
        refresh_token="refresh",
        ip_address="127.0.0.1",
        user_agent="pytest",
    )
    for field in ("created_at", "last_activity", "expires_at"):
        dt = sess[field]
        assert getattr(dt, "tzinfo", None) is timezone.utc

    refreshed = await sm.refresh_session("unused-positional", session_id=1, user_id=1)
    assert str(refreshed["expires_at"]).endswith("+00:00")

@pytest.mark.asyncio
async def test_get_current_user_fast_path_sanitizes_cached_user(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_MODE", "0")

    request = _DummyRequest()
    request.state._auth_user = {
        "id": 42,
        "username": "alice",
        "email": "alice@example.com",
        "role": "user",
        "password_hash": "super-secret",
        "hashed_password": "super-secret",
        "two_factor_secret": "2fa-secret",
        "totp_secret": "totp-secret",
        "backup_codes": "backup-secret",
        "access_token": "access-secret",
        "refresh_token": "refresh-secret",
        "api_key": "api-key-secret",
        "ssn": "123-45-6789",
    }
    request.state.auth = AuthContext(
        principal=AuthPrincipal(kind="user", user_id=42, is_admin=True),
    )

    user = await auth_deps.get_current_user(
        request=request,
        response=Response(),
        credentials=None,
        session_manager=object(),
        db_pool=object(),
        x_api_key=None,
    )

    assert user["id"] == 42
    assert user["username"] == "alice"
    assert "password_hash" not in user
    assert "hashed_password" not in user
    assert "two_factor_secret" not in user
    assert "totp_secret" not in user
    assert "backup_codes" not in user
    assert "access_token" not in user
    assert "refresh_token" not in user
    assert "api_key" not in user
    assert "ssn" not in user

    cached = request.state._auth_user
    assert isinstance(cached, dict)
    assert "password_hash" not in cached


@pytest.mark.asyncio
async def test_api_key_auth_error_logging_does_not_leak_exception_message_outside_test_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "__auth_deps_secret__"

    async def _boom_api_key_mgr() -> Any:
        raise RuntimeError(f"boom: {secret}")

    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")
    monkeypatch.setattr(auth_deps, "get_api_key_manager", _boom_api_key_mgr)

    sink = io.StringIO()
    token = logger.add(sink, level="ERROR")
    try:
        with pytest.raises(HTTPException) as exc_info:
            await auth_deps.get_current_user(
                request=_DummyRequest(),
                response=Response(),
                credentials=None,
                session_manager=object(),
                db_pool=object(),
                x_api_key="not-a-real-key",
            )
        assert exc_info.value.status_code == 401
    finally:
        logger.remove(token)

    captured = sink.getvalue()
    assert "API key authentication error in get_current_user" in captured
    assert secret not in captured


@pytest.mark.asyncio
async def test_test_mode_ensure_authnz_tables_fallback_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "/tmp/authnz-users.db token=single-user-secret"

    def _raise_secret_path(_path: object) -> None:
        raise RuntimeError(secret)

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("SINGLE_USER_TEST_API_KEY", "test-key")
    monkeypatch.setattr(
        auth_deps,
        "get_settings",
        lambda: SimpleNamespace(
            AUTH_MODE="single_user",
            DATABASE_URL="sqlite:////tmp/authnz-users.db",
            SINGLE_USER_API_KEY=None,
            SINGLE_USER_FIXED_ID=7,
        ),
    )
    monkeypatch.setattr(auth_deps, "resolve_client_ip", lambda request, settings: "127.0.0.1")
    monkeypatch.setattr(auth_deps, "is_single_user_ip_allowed", lambda client_ip, settings: True)
    monkeypatch.setattr(migrations, "ensure_authnz_tables", _raise_secret_path)

    sink = io.StringIO()
    token = logger.add(sink, level="DEBUG")
    try:
        user = await auth_deps._authenticate_api_key_from_request(_DummyRequest(), "test-key")
    finally:
        logger.remove(token)

    captured = sink.getvalue()
    assert user["id"] == 7
    assert "AuthNZ test fallback: ensure_authnz_tables skipped/failed" in captured
    assert "RuntimeError" in captured
    assert secret not in captured
    assert "Traceback" not in captured


@pytest.mark.asyncio
async def test_maintenance_guard_skipped_log_is_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "maintenance secret path=/tmp/maintenance.json user_id=444"

    async def _principal(_request: object) -> AuthPrincipal:
        return AuthPrincipal(kind="user", user_id=444, email="person@example.com")

    def _raise_secret_state() -> dict[str, object]:
        raise RuntimeError(secret)

    from tldw_Server_API.app.services import admin_system_ops_service

    monkeypatch.setattr(auth_deps, "_resolve_auth_principal", _principal)
    monkeypatch.setattr(admin_system_ops_service, "get_maintenance_state", _raise_secret_state)

    sink = io.StringIO()
    token = logger.add(sink, level="DEBUG")
    try:
        principal = await auth_deps.get_auth_principal(_DummyRequest())
    finally:
        logger.remove(token)

    captured = sink.getvalue()
    assert principal.user_id == 444
    assert "Maintenance guard skipped" in captured
    assert "RuntimeError" in captured
    assert secret not in captured
    assert "444" not in captured
    assert "Traceback" not in captured


@pytest.mark.asyncio
async def test_rbac_rate_limit_selection_failure_log_is_sanitized() -> None:
    request = _DummyRequest()
    request.state.user_id = "user-secret-123"
    resource = "resource:/private/path?token=secret"

    sink = io.StringIO()
    token = logger.add(sink, level="DEBUG")
    try:
        await auth_deps.enforce_rbac_rate_limit(request, resource, _ExplodingPoolProperty())
    finally:
        logger.remove(token)

    captured = sink.getvalue()
    assert "RBAC rate-limit selection failed" in captured
    assert "RuntimeError" in captured
    assert "user-secret-123" not in captured
    assert resource not in captured
    assert "/tmp/authnz.sqlite" not in captured
    assert "Traceback" not in captured


@pytest.mark.asyncio
async def test_get_current_user_prefers_jwt_then_falls_back_to_api_key_on_401(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"jwt": 0, "api_key": 0}

    async def _fake_verify_jwt_and_fetch_user(request, token: str = ""):
        calls["jwt"] += 1
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    async def _fake_api_key_auth(request, api_key: str):
        calls["api_key"] += 1
        return {
            "id": 99,
            "username": "api-user",
            "is_active": True,
            "is_verified": True,
        }

    monkeypatch.setattr(auth_deps, "verify_jwt_and_fetch_user", _fake_verify_jwt_and_fetch_user)
    monkeypatch.setattr(auth_deps, "_authenticate_api_key_from_request", _fake_api_key_auth)

    request = _DummyRequest()
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="aaa.bbb.ccc")

    user = await auth_deps.get_current_user(
        request=request,
        response=Response(),
        credentials=creds,
        session_manager=object(),
        db_pool=object(),
        x_api_key="api-key",
    )

    assert user["id"] == 99
    assert calls["jwt"] == 1
    assert calls["api_key"] == 1


@pytest.mark.asyncio
async def test_get_current_user_does_not_fall_back_when_jwt_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_verify_jwt_and_fetch_user(request, token: str = ""):
        return {
            "id": 42,
            "username": "jwt-user",
            "is_active": True,
            "is_verified": True,
        }

    async def _fake_api_key_auth(request, api_key: str):
        raise AssertionError("API key auth should not be used when JWT succeeds")

    monkeypatch.setattr(auth_deps, "verify_jwt_and_fetch_user", _fake_verify_jwt_and_fetch_user)
    monkeypatch.setattr(auth_deps, "_authenticate_api_key_from_request", _fake_api_key_auth)

    request = _DummyRequest()
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="aaa.bbb.ccc")

    user = await auth_deps.get_current_user(
        request=request,
        response=Response(),
        credentials=creds,
        session_manager=object(),
        db_pool=object(),
        x_api_key="api-key",
    )

    assert user["id"] == 42


def _profile_helper_should_not_be_called() -> bool:
    raise AssertionError("Profile helper should not be used for rate-limit bypass")

def _mode_helper_should_not_be_called() -> bool:
    raise AssertionError("Mode helper should not be used for rate-limit bypass")


@pytest.mark.asyncio
@pytest.mark.parametrize("func_name", ["check_rate_limit", "check_auth_rate_limit"])
async def test_admin_rate_limit_bypass_is_principal_first(
    monkeypatch: pytest.MonkeyPatch,
    func_name: str,
) -> None:
    async def _boom_auth_governor() -> Any:
        raise RuntimeError("auth_governor_called")

    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")
    # Compatibility: these helpers may no longer be imported by auth_deps in
    # claim-first paths; patch with raising=False so the assertion remains valid
    # regardless of symbol exposure.
    monkeypatch.setattr(auth_deps, "is_single_user_mode", _mode_helper_should_not_be_called, raising=False)
    monkeypatch.setattr(auth_deps, "is_single_user_profile_mode", _profile_helper_should_not_be_called, raising=False)
    monkeypatch.setattr(auth_deps, "get_auth_governor", _boom_auth_governor)

    calls = {"count": 0}

    def _fake_is_single_user_principal(principal: AuthPrincipal | None) -> bool:
        calls["count"] += 1
        if not isinstance(principal, AuthPrincipal):
            return False
        return getattr(principal, "subject", None) == "single_user"

    monkeypatch.setattr(auth_deps, "is_single_user_principal", _fake_is_single_user_principal)

    request = _DummyRequest()
    request.state.auth = AuthContext(
        principal=AuthPrincipal(kind="user", is_admin=True, subject=None),
    )

    func = getattr(auth_deps, func_name)

    with pytest.raises(RuntimeError, match="auth_governor_called"):
        await func(request=request, rate_limiter=object())
    assert calls["count"] == 1

    request.state.auth = AuthContext(
        principal=AuthPrincipal(kind="user", is_admin=True, subject="single_user"),
    )
    await func(request=request, rate_limiter=object())
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_check_rate_limit_enforces_fallback_limiter_when_rg_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RG_ENABLED", "0")
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")
    monkeypatch.setenv("TESTING", "0")
    monkeypatch.setenv("AUTH_DEPS_FALLBACK_RATE_LIMIT", "1")
    monkeypatch.setenv("AUTH_DEPS_FALLBACK_RATE_WINDOW_SECONDS", "60")

    async def _fake_get_auth_governor() -> object:
        return object()

    monkeypatch.setattr(auth_deps, "get_auth_governor", _fake_get_auth_governor)
    auth_deps._AUTH_DEPS_FALLBACK_RATE_WINDOWS.clear()

    request = _DummyRequest()
    request.state.user_id = "not-an-int"
    request.url.path = "/api/v1/rag/search"

    class _StubLimiter:
        enabled = True

        async def check_user_rate_limit(self, **kwargs):
            _ = kwargs
            raise AssertionError("legacy fallback limiter should be bypassed when RG is disabled")

        async def check_rate_limit(self, **kwargs):
            _ = kwargs
            raise AssertionError("legacy limiter should not be used by auth_deps fallback")

    await auth_deps.check_rate_limit(request=request, rate_limiter=_StubLimiter())
    with pytest.raises(HTTPException) as exc_info:
        await auth_deps.check_rate_limit(request=request, rate_limiter=_StubLimiter())
    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_check_rate_limit_enforces_fallback_limiter_when_rg_enabled_without_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RG_ENABLED", "1")
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")
    monkeypatch.setenv("TESTING", "0")
    monkeypatch.setenv("AUTH_DEPS_FALLBACK_RATE_LIMIT", "1")
    monkeypatch.setenv("AUTH_DEPS_FALLBACK_RATE_WINDOW_SECONDS", "60")

    async def _fake_get_auth_governor() -> object:
        return object()

    monkeypatch.setattr(auth_deps, "get_auth_governor", _fake_get_auth_governor)
    auth_deps._AUTH_DEPS_FALLBACK_RATE_WINDOWS.clear()

    request = _DummyRequest()
    request.url.path = "/api/v1/rag/search"

    class _StubLimiter:
        enabled = True

        async def check_user_rate_limit(self, **kwargs):
            raise AssertionError("legacy fallback limiter should be bypassed when RG is enabled")

        async def check_rate_limit(self, **kwargs):
            raise AssertionError("legacy fallback limiter should be bypassed when RG is enabled")

    await auth_deps.check_rate_limit(request=request, rate_limiter=_StubLimiter())
    with pytest.raises(HTTPException) as exc_info:
        await auth_deps.check_rate_limit(request=request, rate_limiter=_StubLimiter())
    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_check_auth_rate_limit_enforces_fallback_limiter_when_rg_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RG_ENABLED", "0")
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")
    monkeypatch.setenv("TESTING", "0")
    monkeypatch.setenv("AUTH_DEPS_AUTH_FALLBACK_RATE_LIMIT", "1")
    monkeypatch.setenv("AUTH_DEPS_AUTH_FALLBACK_RATE_WINDOW_SECONDS", "60")

    async def _fake_get_auth_governor() -> object:
        return object()

    monkeypatch.setattr(auth_deps, "get_auth_governor", _fake_get_auth_governor)
    auth_deps._AUTH_DEPS_FALLBACK_RATE_WINDOWS.clear()

    request = _DummyRequest()
    request.url.path = "/api/v1/auth/forgot-password"

    class _StubLimiter:
        enabled = True

        async def check_rate_limit_fallback(self, **kwargs):
            _ = kwargs
            raise AssertionError("legacy fallback limiter should be bypassed when RG is disabled")

        async def check_rate_limit(self, **kwargs):
            _ = kwargs
            raise AssertionError("legacy limiter should not be used by auth_deps fallback")

    await auth_deps.check_auth_rate_limit(request=request, rate_limiter=_StubLimiter())
    with pytest.raises(HTTPException) as exc_info:
        await auth_deps.check_auth_rate_limit(request=request, rate_limiter=_StubLimiter())
    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_check_auth_rate_limit_enforces_fallback_limiter_when_rg_enabled_without_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RG_ENABLED", "1")
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")
    monkeypatch.setenv("TESTING", "0")
    monkeypatch.setenv("AUTH_DEPS_AUTH_FALLBACK_RATE_LIMIT", "1")
    monkeypatch.setenv("AUTH_DEPS_AUTH_FALLBACK_RATE_WINDOW_SECONDS", "60")

    async def _fake_get_auth_governor() -> object:
        return object()

    monkeypatch.setattr(auth_deps, "get_auth_governor", _fake_get_auth_governor)
    auth_deps._AUTH_DEPS_FALLBACK_RATE_WINDOWS.clear()

    request = _DummyRequest()
    request.url.path = "/api/v1/auth/forgot-password"

    class _StubLimiter:
        enabled = True

        async def check_rate_limit_fallback(self, **kwargs):
            raise AssertionError("legacy fallback limiter should be bypassed when RG is enabled")

        async def check_rate_limit(self, **kwargs):
            raise AssertionError("legacy fallback limiter should be bypassed when RG is enabled")

    await auth_deps.check_auth_rate_limit(request=request, rate_limiter=_StubLimiter())
    with pytest.raises(HTTPException) as exc_info:
        await auth_deps.check_auth_rate_limit(request=request, rate_limiter=_StubLimiter())
    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_get_session_manager_dep_requires_explicit_test_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.delenv("AUTHNZ_FORCE_REAL_SESSION_MANAGER", raising=False)

    sentinel = object()

    async def _fake_get_session_manager() -> object:
        return sentinel

    monkeypatch.setattr(auth_deps, "get_session_manager", _fake_get_session_manager)

    resolved = await auth_deps.get_session_manager_dep()
    assert resolved is sentinel


@pytest.mark.asyncio
async def test_get_session_manager_dep_does_not_use_stub_without_explicit_pytest_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("AUTHNZ_FORCE_REAL_SESSION_MANAGER", raising=False)

    sentinel = object()

    async def _fake_get_session_manager() -> object:
        return sentinel

    monkeypatch.setattr(auth_deps, "get_session_manager", _fake_get_session_manager)

    resolved = await auth_deps.get_session_manager_dep()
    assert resolved is sentinel


@pytest.mark.asyncio
async def test_get_db_transaction_requires_explicit_test_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")

    sentinel = object()

    class _TxnCM:
        async def __aenter__(self) -> object:
            return sentinel

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

    class _Pool:
        def transaction(self) -> _TxnCM:
            return _TxnCM()

        def acquire(self) -> object:
            raise AssertionError("adapter path should not be used when TEST_MODE=0")

    async def _fake_get_db_pool() -> _Pool:
        return _Pool()

    monkeypatch.setattr(auth_deps, "get_db_pool", _fake_get_db_pool)

    agen = auth_deps.get_db_transaction()
    try:
        conn = await agen.__anext__()
        assert conn is sentinel
    finally:
        await agen.aclose()


@pytest.mark.asyncio
async def test_get_db_transaction_retries_lock_contention_on_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")
    monkeypatch.setenv("AUTHNZ_SQLITE_LOCK_MAX_RETRIES", "2")
    monkeypatch.setenv("AUTHNZ_SQLITE_LOCK_RETRY_BASE_SECONDS", "0")
    monkeypatch.setenv("AUTHNZ_SQLITE_LOCK_RETRY_MAX_SECONDS", "0")

    pool = _LockingPool(lock_on_enter_count=1, raise_lock_on_exit=False)

    async def _fake_get_db_pool() -> _LockingPool:
        return pool

    sleep_calls: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr(auth_deps, "get_db_pool", _fake_get_db_pool)
    monkeypatch.setattr(auth_deps.asyncio, "sleep", _fake_sleep)

    agen = auth_deps.get_db_transaction()
    try:
        conn = await agen.__anext__()
        assert conn is pool.conn
    finally:
        await agen.aclose()

    assert pool.enter_calls == 2
    assert pool.exit_calls == 1
    assert sleep_calls == [0.0]


@pytest.mark.asyncio
async def test_get_db_transaction_returns_503_when_lock_retries_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")
    monkeypatch.setenv("AUTHNZ_SQLITE_LOCK_MAX_RETRIES", "1")
    monkeypatch.setenv("AUTHNZ_SQLITE_LOCK_RETRY_BASE_SECONDS", "0")
    monkeypatch.setenv("AUTHNZ_SQLITE_LOCK_RETRY_MAX_SECONDS", "0")
    monkeypatch.setenv("AUTHNZ_SQLITE_LOCK_RETRY_AFTER_SECONDS", "7")

    pool = _LockingPool(lock_on_enter_count=5, raise_lock_on_exit=False)

    async def _fake_get_db_pool() -> _LockingPool:
        return pool

    sleep_calls: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr(auth_deps, "get_db_pool", _fake_get_db_pool)
    monkeypatch.setattr(auth_deps.asyncio, "sleep", _fake_sleep)

    agen = auth_deps.get_db_transaction()
    with pytest.raises(HTTPException) as exc_info:
        await agen.__anext__()

    assert exc_info.value.status_code == 503
    assert exc_info.value.headers is not None
    assert exc_info.value.headers.get("Retry-After") == "7"
    assert pool.enter_calls == 2
    assert pool.exit_calls == 0
    assert sleep_calls == [0.0]


@pytest.mark.asyncio
async def test_get_db_transaction_maps_cleanup_lock_error_to_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")
    monkeypatch.setenv("AUTHNZ_SQLITE_LOCK_RETRY_AFTER_SECONDS", "3")

    pool = _LockingPool(lock_on_enter_count=0, raise_lock_on_exit=True)

    async def _fake_get_db_pool() -> _LockingPool:
        return pool

    monkeypatch.setattr(auth_deps, "get_db_pool", _fake_get_db_pool)

    agen = auth_deps.get_db_transaction()
    conn = await agen.__anext__()
    assert conn is pool.conn

    with pytest.raises(HTTPException) as exc_info:
        await agen.__anext__()

    assert exc_info.value.status_code == 503
    assert exc_info.value.headers is not None
    assert exc_info.value.headers.get("Retry-After") == "3"
    assert pool.enter_calls == 1
    assert pool.exit_calls == 1


@pytest.mark.asyncio
async def test_get_db_transaction_uses_adapter_when_tldw_test_mode_is_y(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "y")

    class _Conn:
        async def execute(self, query: str, params: Any) -> Any:
            return SimpleNamespace()

        async def commit(self) -> None:
            return None

    class _Acquire:
        async def __aenter__(self) -> _Conn:
            return _Conn()

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

    class _Pool:
        def acquire(self) -> _Acquire:
            return _Acquire()

        def transaction(self) -> object:
            raise AssertionError("transaction() path should not be used when TLDW_TEST_MODE=y")

    async def _fake_get_db_pool() -> _Pool:
        return _Pool()

    monkeypatch.setattr(auth_deps, "get_db_pool", _fake_get_db_pool)

    agen = auth_deps.get_db_transaction()
    try:
        conn = await agen.__anext__()
        assert hasattr(conn, "execute")
    finally:
        await agen.aclose()
