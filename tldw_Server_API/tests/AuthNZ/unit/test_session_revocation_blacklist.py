import asyncio
from datetime import datetime, timedelta
from typing import Any, Optional

import pytest

from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager
from tldw_Server_API.app.core.AuthNZ.settings import Settings


class _FakeTransaction:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakePool:
    def __init__(self, conn):
        self.pool = object()  # Marker so SessionManager treats as Postgres path
        self._conn = conn

    def transaction(self):

        return _FakeTransaction(self._conn)


class _FakeConn:
    def __init__(self, session_record: dict[str, Any]):
        self._session_record = session_record
        self.fetchrow_calls = 0

    async def fetchrow(self, query: str, *args):
        if "SELECT id, user_id" in query:
            self.fetchrow_calls += 1
            return dict(self._session_record)
        return None

    async def execute(self, *args, **kwargs):
        return None


class _StubBlacklist:
    def __init__(self):
        self.calls = []

    def hint_blacklisted(self, jti: str, expires_at: datetime):
        self.calls.append(("hint", jti, expires_at))

    async def revoke_token(
        self,
        *,
        jti: str,
        expires_at: datetime,
        user_id: Optional[int],
        token_type: str,
        reason: Optional[str],
        revoked_by: Optional[int],
        ip_address: Optional[str],
    ) -> bool:
        self.calls.append(
            ("revoke", jti, expires_at, user_id, token_type, reason, revoked_by, ip_address)
        )
        return True


@pytest.mark.asyncio
async def test_revoke_session_blacklists_tokens(monkeypatch):
    now = datetime.utcnow()
    session_record = {
        "id": 123,
        "user_id": 456,
        "access_jti": "access-jti-xyz",
        "refresh_jti": "refresh-jti-xyz",
        "expires_at": now + timedelta(minutes=15),
        "refresh_expires_at": now + timedelta(days=2),
    }

    settings = Settings(AUTH_MODE="multi_user", JWT_SECRET_KEY="rotation-new-secret-1234567890abcd")
    manager = SessionManager(settings=settings)
    manager._initialized = True
    manager._external_db_pool = True

    fake_conn = _FakeConn(session_record)
    fake_pool = _FakePool(fake_conn)

    async def _fake_ensure_db_pool():
        return fake_pool

    monkeypatch.setattr(manager, "_ensure_db_pool", _fake_ensure_db_pool)

    stub_blacklist = _StubBlacklist()
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.session_manager.get_token_blacklist",
        lambda: stub_blacklist,
    )

    await manager.revoke_session(session_id=123, revoked_by=42, reason="unit-test")

    revoke_events = [
        call for call in stub_blacklist.calls if call and call[0] == "revoke"
    ]
    assert len(revoke_events) == 2
    access_event = next(evt for evt in revoke_events if evt[4] == "access")
    refresh_event = next(evt for evt in revoke_events if evt[4] == "refresh")
    assert access_event[1] == session_record["access_jti"]
    assert refresh_event[1] == session_record["refresh_jti"]


@pytest.mark.asyncio
async def test_revoke_session_ignores_malformed_post_commit_cache_entry(monkeypatch):
    now = datetime.utcnow()
    session_record = {
        "id": 123,
        "user_id": 456,
        "access_jti": "access-jti-xyz",
        "refresh_jti": "refresh-jti-xyz",
        "expires_at": now + timedelta(minutes=15),
        "refresh_expires_at": now + timedelta(days=2),
    }

    class _MalformedRedis:
        async def scan_iter(self, _pattern: str):
            yield "session:malformed"

        async def get(self, _key: str) -> str:
            return "{not-json"

    manager = SessionManager(
        settings=Settings(
            AUTH_MODE="multi_user",
            JWT_SECRET_KEY="rotation-new-secret-1234567890abcd",
        )
    )
    manager._initialized = True
    manager._external_db_pool = True
    manager.redis_client = _MalformedRedis()
    fake_pool = _FakePool(_FakeConn(session_record))

    async def _fake_ensure_db_pool():
        return fake_pool

    monkeypatch.setattr(manager, "_ensure_db_pool", _fake_ensure_db_pool)
    stub_blacklist = _StubBlacklist()
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.session_manager.get_token_blacklist",
        lambda: stub_blacklist,
    )

    revoked = await manager.revoke_session(
        session_id=123,
        revoked_by=42,
        reason="unit-test",
    )

    assert revoked is True
    assert len([call for call in stub_blacklist.calls if call[0] == "revoke"]) == 2


@pytest.mark.asyncio
async def test_tenant_mismatch_does_not_invalidate_session_cache(monkeypatch):
    cache_scans: list[str] = []

    class _Redis:
        async def scan_iter(self, pattern: str):
            cache_scans.append(pattern)
            if False:
                yield ""

    class _MissingRepo:
        def __init__(self, _db_pool) -> None:
            return None

        async def revoke_session_record(self, **_kwargs):
            return None

    manager = SessionManager(
        settings=Settings(
            AUTH_MODE="multi_user",
            JWT_SECRET_KEY="rotation-new-secret-1234567890abcd",
        )
    )
    manager._initialized = True
    manager.redis_client = _Redis()

    async def _fake_ensure_db_pool():
        return object()

    monkeypatch.setattr(manager, "_ensure_db_pool", _fake_ensure_db_pool)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.session_manager.AuthnzSessionsRepo",
        _MissingRepo,
    )

    revoked = await manager.revoke_session(
        session_id=123,
        expected_user_id=456,
        revoked_by=42,
    )

    assert revoked is False
    assert cache_scans == []


@pytest.mark.asyncio
async def test_revoke_all_user_sessions_returns_count_and_forwards_reason(monkeypatch):
    settings = Settings(AUTH_MODE="multi_user", JWT_SECRET_KEY="rotation-new-secret-1234567890abcd")
    manager = SessionManager(settings=settings)
    manager._initialized = True
    manager.redis_client = None

    captured: dict[str, Any] = {}

    class _StubRepo:
        def __init__(self, _db_pool):
            return None

        async def revoke_all_sessions_for_user(
            self,
            *,
            user_id: int,
            except_session_id: Optional[int] = None,
            reason: str | None = None,
            revoked_by: int | None = None,
        ) -> int:
            captured["repo_user_id"] = user_id
            captured["repo_except_session_id"] = except_session_id
            captured["repo_reason"] = reason
            captured["repo_revoked_by"] = revoked_by
            return 4

    class _StubBlacklistAll:
        async def revoke_all_user_tokens(
            self,
            user_id: int,
            reason: str = "User requested logout from all devices",
            revoked_by: Optional[int] = None,
            ip_address: Optional[str] = None,
            except_session_id: Optional[int] = None,
        ) -> int:
            captured["blacklist_user_id"] = user_id
            captured["blacklist_reason"] = reason
            captured["blacklist_revoked_by"] = revoked_by
            captured["blacklist_ip_address"] = ip_address
            captured["blacklist_except_session_id"] = except_session_id
            return 8

    async def _fake_ensure_db_pool():
        return object()

    monkeypatch.setattr(manager, "_ensure_db_pool", _fake_ensure_db_pool)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.session_manager.AuthnzSessionsRepo",
        _StubRepo,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.session_manager.get_token_blacklist",
        lambda: _StubBlacklistAll(),
    )

    count = await manager.revoke_all_user_sessions(
        user_id=999,
        except_session_id=55,
        reason="unit-test-revoke-all",
        revoked_by=77,
    )

    assert count == 4
    assert captured["repo_user_id"] == 999
    assert captured["repo_except_session_id"] == 55
    assert captured["repo_reason"] == "unit-test-revoke-all"
    assert captured["repo_revoked_by"] == 77
    assert captured["blacklist_user_id"] == 999
    assert captured["blacklist_reason"] == "unit-test-revoke-all"
    assert captured["blacklist_revoked_by"] == 77
    assert captured["blacklist_except_session_id"] == 55


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["single", "all"])
async def test_session_revocation_preserves_repository_cancellation_identity(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    cancellation = asyncio.CancelledError()
    manager = SessionManager(
        settings=Settings(
            AUTH_MODE="multi_user",
            JWT_SECRET_KEY="rotation-new-secret-1234567890abcd",
        )
    )
    manager._initialized = True
    manager.redis_client = None

    class _CancellingRepo:
        def __init__(self, _db_pool) -> None:
            return None

        async def revoke_session_record(self, **_kwargs):
            raise cancellation

        async def revoke_all_sessions_for_user(self, **_kwargs):
            raise cancellation

    async def _fake_ensure_db_pool():
        return object()

    monkeypatch.setattr(manager, "_ensure_db_pool", _fake_ensure_db_pool)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.session_manager.AuthnzSessionsRepo",
        _CancellingRepo,
    )

    with pytest.raises(asyncio.CancelledError) as raised:
        if operation == "single":
            await manager.revoke_session(session_id=123, revoked_by=42)
        else:
            await manager.revoke_all_user_sessions(user_id=123)

    assert raised.value is cancellation


@pytest.mark.asyncio
async def test_revoke_all_preserves_post_commit_blacklist_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancellation = asyncio.CancelledError()
    manager = SessionManager(
        settings=Settings(
            AUTH_MODE="multi_user",
            JWT_SECRET_KEY="rotation-new-secret-1234567890abcd",
        )
    )
    manager._initialized = True
    manager.redis_client = None

    class _Repo:
        def __init__(self, _db_pool) -> None:
            return None

        async def revoke_all_sessions_for_user(self, **_kwargs) -> int:
            return 1

    class _CancellingBlacklist:
        async def revoke_all_user_tokens(self, *_args, **_kwargs) -> int:
            raise cancellation

    async def _fake_ensure_db_pool():
        return object()

    monkeypatch.setattr(manager, "_ensure_db_pool", _fake_ensure_db_pool)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.session_manager.AuthnzSessionsRepo",
        _Repo,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.session_manager.get_token_blacklist",
        lambda: _CancellingBlacklist(),
    )

    with pytest.raises(asyncio.CancelledError) as raised:
        await manager.revoke_all_user_sessions(user_id=123)

    assert raised.value is cancellation


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["single", "all"])
@pytest.mark.parametrize("cache_stage", ["scan", "get", "delete"])
async def test_post_commit_cache_invalidation_preserves_cancellation_identity(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    cache_stage: str,
) -> None:
    cancellation = asyncio.CancelledError()
    manager = SessionManager(
        settings=Settings(
            AUTH_MODE="multi_user",
            JWT_SECRET_KEY="rotation-new-secret-1234567890abcd",
        )
    )
    manager._initialized = True

    class _Repo:
        def __init__(self, _db_pool) -> None:
            return None

        async def revoke_session_record(self, **_kwargs):
            return {
                "id": 123,
                "user_id": 456,
                "access_jti": None,
                "refresh_jti": None,
            }

        async def revoke_all_sessions_for_user(self, **_kwargs) -> int:
            return 1

    class _CancellingRedis:
        async def scan_iter(self, _pattern: str):
            if cache_stage == "scan":
                raise cancellation
            yield "session:123"

        async def get(self, _key: str) -> str:
            if cache_stage == "get":
                raise cancellation
            return '{"session_id": 123, "user_id": 456}'

        async def delete(self, _key: str) -> None:
            if cache_stage == "delete":
                raise cancellation

    async def _fake_ensure_db_pool():
        return object()

    manager.redis_client = _CancellingRedis()
    monkeypatch.setattr(manager, "_ensure_db_pool", _fake_ensure_db_pool)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.session_manager.AuthnzSessionsRepo",
        _Repo,
    )

    with pytest.raises(asyncio.CancelledError) as raised:
        if operation == "single":
            await manager.revoke_session(session_id=123, expected_user_id=456)
        else:
            await manager.revoke_all_user_sessions(user_id=456)

    assert raised.value is cancellation
