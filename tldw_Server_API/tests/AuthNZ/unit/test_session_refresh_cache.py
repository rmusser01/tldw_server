import asyncio
import json
from collections import defaultdict
from datetime import datetime, timedelta
import types
import os

import pytest

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.exceptions import InvalidSessionError, SessionError
from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService
from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager
from tldw_Server_API.app.core.AuthNZ.settings import Settings, reset_settings


class FakeRedis:
    def __init__(self):
        self.kv: dict[str, str] = {}
        self.sets: defaultdict[str, set[int]] = defaultdict(set)

    async def setex(self, key, ttl, value):
        self.kv[key] = value

    async def sadd(self, key, value):
        self.sets[key].add(value)

    async def expire(self, key, ttl):
        # TTL tracking not required for test assertions
        return True

    async def get(self, key):
        return self.kv.get(key)

    async def exists(self, key):
        return 1 if key in self.kv else 0

    async def delete(self, key):
        self.kv.pop(key, None)

    async def scan_iter(self, pattern):
        if pattern != "session:*":
            return
        # Iterate over a snapshot to allow mutation during iteration
        for key in list(self.kv.keys()):
            if key.startswith("session:"):
                yield key


class StubTransaction:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class StubPool:
    def __init__(self, conn):
        self.conn = conn
        # Sessions repo backend routing keys off db_pool.pool presence.
        # Use a truthy sentinel so this stub follows the Postgres fetchrow path.
        self.pool = object()

    def transaction(self):

        return StubTransaction(self.conn)

    def acquire(self):

             # Match DatabasePool.acquire() context manager shape for repo helpers
        return StubTransaction(self.conn)


class StubConn:
    def __init__(self, expected_refresh_hash, expected_access_hash, session_id, user_id):
        self.expected_refresh_hash = expected_refresh_hash
        self.expected_access_hash = expected_access_hash
        self.session_id = session_id
        self.user_id = user_id
        self.updated = False

    async def fetchrow(self, query, *params):
        if "SELECT id, user_id, token_hash, refresh_token_hash" in query:
            refresh_hash = params[0]
            assert refresh_hash == self.expected_refresh_hash
            return {
                "id": self.session_id,
                "user_id": self.user_id,
                "token_hash": self.expected_access_hash,
                "refresh_token_hash": self.expected_refresh_hash,
            }
        if "UPDATE sessions" in query:
            expected_access_hash = params[9]
            expected_refresh_hash = params[10]
            if (
                expected_access_hash == self.expected_access_hash
                and expected_refresh_hash == self.expected_refresh_hash
            ):
                self.expected_access_hash = params[1]
                self.expected_refresh_hash = params[5]
                self.updated = True
                return {"id": self.session_id}
            return None
        return None

    async def execute(self, *args, **kwargs):
        self.updated = True
        return None


@pytest.mark.asyncio
async def test_validate_session_strict_surfaces_infrastructure_failure():
    manager = object.__new__(SessionManager)
    manager._initialized = True
    manager.redis_client = None
    manager._token_hash_candidates = types.MethodType(
        lambda self, token: [f"h:{token}"], manager
    )

    async def fail_pool(self):
        raise RuntimeError("session store unavailable")

    manager._ensure_db_pool = types.MethodType(fail_pool, manager)

    assert await manager.validate_session("opaque-token") is None
    with pytest.raises(SessionError, match="Failed to validate session"):
        await manager.validate_session("opaque-token", raise_on_error=True)


@pytest.mark.asyncio
async def test_refresh_session_replaces_cached_access_token(monkeypatch):
    reset_settings()
    manager = SessionManager()
    manager._initialized = True

    # Simplify hashing/encryption for deterministic testing
    manager._token_hash_candidates = types.MethodType(
        lambda self, token: [f"h:{token}"], manager
    )
    manager.hash_token = types.MethodType(lambda self, token: f"h:{token}", manager)
    manager.encrypt_token = types.MethodType(
        lambda self, token: f"enc:{token}" if token else None, manager
    )

    def _extract_stub(self, token):

        if not token:
            return (None, None)
        return (f"jti:{token}", datetime.utcnow() + timedelta(minutes=30))

    manager._extract_token_metadata = types.MethodType(_extract_stub, manager)

    fake_redis = FakeRedis()
    manager.redis_client = fake_redis

    old_refresh = "old-refresh-token"
    old_refresh_hash = manager.hash_token(old_refresh)
    session_id = 555
    user_id = 42

    # Token binding validation now requires JWT-like claims aligned with the session.
    manager._get_unverified_claims = types.MethodType(
        lambda self, _token: {"sub": user_id, "session_id": session_id},
        manager,
    )

    # Seed redis cache with previous access token entry
    old_access_hash = manager.hash_token("old-access-token")
    fake_redis.kv[f"session:{old_access_hash}"] = json.dumps(
        {
            "session_id": session_id,
            "user_id": user_id,
            "expires_at": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
            "user_active": True,
            "revoked": False,
        }
    )

    stub_conn = StubConn(old_refresh_hash, old_access_hash, session_id, user_id)
    stub_pool = StubPool(stub_conn)

    async def _ensure_db_pool_stub(self):
        return stub_pool

    manager.db_pool = stub_pool
    manager._ensure_db_pool = types.MethodType(_ensure_db_pool_stub, manager)

    new_access = "new-access-token"
    new_refresh = "new-refresh-token"

    result = await manager.refresh_session(
        refresh_token=old_refresh,
        new_access_token=new_access,
        new_refresh_token=new_refresh,
    )

    assert result["session_id"] == session_id
    new_access_hash = manager.hash_token(new_access)

    # Old cache entry should be removed and replaced with new token hash
    assert f"session:{old_access_hash}" not in fake_redis.kv
    assert f"session:{new_access_hash}" in fake_redis.kv

    cached_payload = json.loads(fake_redis.kv[f"session:{new_access_hash}"])
    assert cached_payload["session_id"] == session_id
    assert cached_payload["user_id"] == user_id


@pytest.mark.asyncio
async def test_refresh_session_accepts_legacy_refresh_hash(monkeypatch):
    reset_settings()
    manager = SessionManager()
    manager._initialized = True

    def _token_hash_candidates(self, token):

        return [f"new-h:{token}", f"old-h:{token}"]

    manager._token_hash_candidates = types.MethodType(_token_hash_candidates, manager)

    manager.hash_token = types.MethodType(
        lambda self, token: self._token_hash_candidates(token)[0] if token else None,
        manager,
    )
    manager.encrypt_token = types.MethodType(
        lambda self, token: f"enc:{token}" if token else None, manager
    )

    def _extract_stub(self, token):

        if not token:
            return (None, None)
        return (f"jti:{token}", datetime.utcnow() + timedelta(minutes=5))

    manager._extract_token_metadata = types.MethodType(_extract_stub, manager)

    legacy_refresh = "legacy-refresh-token"
    legacy_hash = manager._token_hash_candidates(legacy_refresh)[1]
    session_id = 777
    user_id = 101

    manager._get_unverified_claims = types.MethodType(
        lambda self, _token: {"sub": user_id, "session_id": session_id},
        manager,
    )

    class CandidateStubConn:
        def __init__(self, expected_refresh_hash):
            self.expected_refresh_hash = expected_refresh_hash
            self.expected_access_hash = "hash:existing-access"
            self.updated = False
            self.fetch_calls: list[str] = []

        async def fetchrow(self, query, *params):
            if "SELECT id, user_id, token_hash, refresh_token_hash" in query:
                candidate_hash = params[0]
                self.fetch_calls.append(candidate_hash)
                if candidate_hash == self.expected_refresh_hash:
                    return {
                        "id": session_id,
                        "user_id": user_id,
                        "token_hash": self.expected_access_hash,
                        "refresh_token_hash": self.expected_refresh_hash,
                    }
                return None
            if "UPDATE sessions" in query:
                expected_access_hash = params[9]
                expected_refresh_hash = params[10]
                if (
                    expected_access_hash == self.expected_access_hash
                    and expected_refresh_hash == self.expected_refresh_hash
                ):
                    self.expected_access_hash = params[1]
                    self.expected_refresh_hash = params[5]
                    self.updated = True
                    return {"id": session_id}
                return None
            return None

        async def execute(self, *args, **kwargs):
            self.updated = True
            return None

    stub_conn = CandidateStubConn(legacy_hash)
    stub_pool = StubPool(stub_conn)

    async def _ensure_db_pool_stub(self):
        return stub_pool

    manager.db_pool = stub_pool
    manager._ensure_db_pool = types.MethodType(_ensure_db_pool_stub, manager)

    result = await manager.refresh_session(
        refresh_token=legacy_refresh,
        new_access_token="fresh-access",
        new_refresh_token="fresh-refresh",
    )

    assert result["session_id"] == 777
    assert result["user_id"] == 101
    assert stub_conn.updated is True
    assert stub_conn.fetch_calls == [
        manager._token_hash_candidates(legacy_refresh)[0],
        legacy_hash,
    ]


@pytest.mark.asyncio
async def test_validate_session_rewrites_legacy_hash(monkeypatch):
    reset_settings()
    manager = SessionManager()
    manager._initialized = True
    manager.redis_client = FakeRedis()

    def _token_hash_candidates(self, token):

        return [f"new-h:{token}", f"old-h:{token}"]

    manager._token_hash_candidates = types.MethodType(_token_hash_candidates, manager)

    access_token = "legacy-access-token"
    candidate_hashes = manager._token_hash_candidates(access_token)
    primary_hash, legacy_hash = candidate_hashes

    expires_at = datetime.utcnow() + timedelta(minutes=45)

    class AcquireCtx:
        def __init__(self, conn):
            self.conn = conn

        async def __aenter__(self):
            return self.conn

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class LegacyConn:
        def __init__(self, session_id, user_id, stored_hash):
            self.session_id = session_id
            self.user_id = user_id
            self.stored_hash = stored_hash
            self.fetch_calls: list[tuple[str, str | int]] = []
            self.update_calls: list[tuple[str, tuple]] = []

        async def fetchrow(self, query, param):
            self.fetch_calls.append((query, param))
            if "s.token_hash = $1" in query:
                if param == self.stored_hash:
                    return {
                        "id": self.session_id,
                        "token_hash": self.stored_hash,
                        "user_id": self.user_id,
                        "expires_at": expires_at,
                        "is_active": True,
                        "revoked_at": None,
                        "username": "legacy-user",
                        "role": "user",
                        "user_active": True,
                    }
                return None
            return None

        async def execute(self, query, *params):
            params_tuple = tuple(params)
            self.update_calls.append((query, params_tuple))
            if "SET token_hash" in query:
                self.stored_hash = params_tuple[0]
            return None

    class PoolStub:
        def __init__(self, conn):
            self.pool = object()
            self._conn = conn

        def acquire(self):

            return AcquireCtx(self._conn)

    stub_conn = LegacyConn(session_id=777, user_id=42, stored_hash=legacy_hash)
    stub_pool = PoolStub(stub_conn)

    async def _ensure_db_pool_stub(self):
        return stub_pool

    manager.db_pool = stub_pool
    manager._ensure_db_pool = types.MethodType(_ensure_db_pool_stub, manager)

    result = await manager.validate_session(access_token)
    assert result is not None
    assert result["id"] == 777
    assert any("SET token_hash" in q for q, _ in stub_conn.update_calls)
    assert stub_conn.stored_hash == primary_hash

    # After normalization, drop the legacy candidate and ensure validation still succeeds.
    def _token_hash_primary_only(self, token):
        return [primary_hash]

    manager._token_hash_candidates = types.MethodType(_token_hash_primary_only, manager)

    update_calls_before = len([q for q, _ in stub_conn.update_calls if "SET token_hash" in q])
    result_again = await manager.validate_session(access_token)
    assert result_again is not None
    update_calls_after = len([q for q, _ in stub_conn.update_calls if "SET token_hash" in q])
    assert update_calls_after == update_calls_before


@pytest.mark.asyncio
async def test_refresh_session_concurrent_rotation_allows_single_winner(isolated_test_environment):
    reset_settings()
    _client, _db_name = isolated_test_environment  # ensure DB/env is provisioned for this test
    db_url = os.getenv("DATABASE_URL", "")

    settings = Settings(
        AUTH_MODE="multi_user",
        DATABASE_URL=db_url,
        JWT_SECRET_KEY="session-refresh-concurrency-secret-1234567890",
        ENABLE_REGISTRATION=True,
        REQUIRE_REGISTRATION_CODE=False,
        ROTATE_REFRESH_TOKENS=True,
    )

    pool = DatabasePool(settings)
    await pool.initialize()
    manager = SessionManager(db_pool=pool, settings=settings)
    await manager.initialize()

    try:
        # Ensure a user exists for FK constraints
        async with pool.transaction() as conn:
            await conn.execute(
                """
                INSERT INTO users (id, username, email, password_hash, is_active)
                VALUES ($1, $2, $3, $4, TRUE)
                """,
                1,
                "concurrency-user",
                "concurrency-user@example.com",
                "hashed-password",
            )

        jwt_service = JWTService(settings=settings)
        original_access = jwt_service.create_access_token(user_id=1, username="concurrency-user", role="user")
        original_refresh = jwt_service.create_refresh_token(user_id=1, username="concurrency-user")

        created = await manager.create_session(
            user_id=1,
            access_token=original_access,
            refresh_token=original_refresh,
            ip_address="127.0.0.1",
            user_agent="pytest-concurrency",
        )
        session_id = int(created["session_id"])

        token_pair_a = (
            jwt_service.create_access_token(user_id=1, username="concurrency-user", role="user"),
            jwt_service.create_refresh_token(user_id=1, username="concurrency-user"),
        )
        token_pair_b = (
            jwt_service.create_access_token(user_id=1, username="concurrency-user", role="user"),
            jwt_service.create_refresh_token(user_id=1, username="concurrency-user"),
        )

        async def _attempt_refresh(new_access: str, new_refresh: str) -> tuple[str, dict | None]:
            try:
                refreshed = await manager.refresh_session(
                    refresh_token=original_refresh,
                    new_access_token=new_access,
                    new_refresh_token=new_refresh,
                )
                return "ok", refreshed
            except InvalidSessionError:
                return "invalid", None

        outcomes = await asyncio.gather(
            _attempt_refresh(*token_pair_a),
            _attempt_refresh(*token_pair_b),
        )
        statuses = sorted([status for status, _ in outcomes])
        assert statuses == ["invalid", "ok"]

        row = await pool.fetchone(
            "SELECT refresh_token_hash, token_hash FROM sessions WHERE id = $1",
            session_id,
        )
        assert row is not None
        winner_refresh_hashes = {
            manager.hash_token(token_pair_a[1]),
            manager.hash_token(token_pair_b[1]),
        }
        assert row["refresh_token_hash"] in winner_refresh_hashes
        assert row["refresh_token_hash"] != manager.hash_token(original_refresh)
    finally:
        await manager.shutdown()
        await pool.close()
        reset_settings()

@pytest.mark.asyncio
async def test_is_token_blacklisted_uses_jti_redis_key(monkeypatch):
    reset_settings()
    manager = SessionManager()
    manager._initialized = True

    # Simplify hashing to avoid crypto dependencies
    manager.hash_token = types.MethodType(lambda self, token: f"h:{token}", manager)

    fake_redis = FakeRedis()
    manager.redis_client = fake_redis

    jti = "revoked-jti"
    fake_redis.kv[f"blacklist:{jti}"] = json.dumps({"revoked": True})

    class _StubPool:
        pool = None

        async def fetchval(self, *args, **kwargs):
            pytest.fail("Database should not be queried when Redis indicates revocation")

    async def _ensure_db_pool_stub(self):
        return _StubPool()

    manager._ensure_db_pool = types.MethodType(_ensure_db_pool_stub, manager)

    assert await manager.is_token_blacklisted("dummy-token", jti=jti) is True


@pytest.mark.asyncio
async def test_is_token_blacklisted_checks_refresh_hash_fallback(monkeypatch):
    reset_settings()
    manager = SessionManager()
    manager._initialized = True
    manager.redis_client = None

    revoked_refresh_hash = "hash:refresh-token"

    def _token_hash_candidates(self, token):

        return ["hash:access-token", revoked_refresh_hash]

    manager._token_hash_candidates = types.MethodType(_token_hash_candidates, manager)

    class FallbackPool:
        def __init__(self, revoked_hash: str):
            self.pool = object()  # mimic Postgres pool path
            self.revoked_hash = revoked_hash
            self.calls: list[tuple[str, str]] = []

        def acquire(self):

                     # Match DatabasePool.acquire() context manager shape for repo helpers
            return StubTransaction(self)

        async def fetchval(self, query: str, candidate: str):
            self.calls.append((query, candidate))
            if "refresh_token_hash" in query and candidate == self.revoked_hash:
                return 1
            return 0

    pool = FallbackPool(revoked_refresh_hash)

    async def _ensure_db_pool_stub(self):
        return pool

    manager._ensure_db_pool = types.MethodType(_ensure_db_pool_stub, manager)

    class DummyBlacklist:
        async def is_blacklisted(self, _jti: str) -> bool:
            return False

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.session_manager.get_token_blacklist",
        lambda: DummyBlacklist(),
    )

    assert (
        await manager.is_token_blacklisted("refresh-token-content", jti="refresh-jti")
        is True
    )
    assert any("refresh_token_hash" in query for query, _ in pool.calls)


@pytest.mark.asyncio
async def test_refresh_session_survives_hmac_rotation(isolated_test_environment, monkeypatch):
    reset_settings()
    # Provision a per-test PostgreSQL database via the shared AuthNZ fixture
    _client, _db_name = isolated_test_environment  # client unused; ensures DB is created and env is set
    old_secret = "old-session-secret-key-for-tests-001"
    new_secret = "new-session-secret-key-for-tests-002"

    db_url = os.getenv("DATABASE_URL", "")
    old_settings = Settings(
        AUTH_MODE="multi_user",
        DATABASE_URL=db_url,
        JWT_SECRET_KEY=old_secret,
        ENABLE_REGISTRATION=True,
        REQUIRE_REGISTRATION_CODE=False,
    )

    pool = DatabasePool(old_settings)
    await pool.initialize()

    manager_old = SessionManager(db_pool=pool, settings=old_settings)
    await manager_old.initialize()
    manager_rotated: SessionManager | None = None

    # Ensure a user exists for FK constraints
    async with pool.transaction() as conn:
        await conn.execute(
            """
            INSERT INTO users (id, username, email, password_hash, is_active)
            VALUES ($1, $2, $3, $4, TRUE)
            """,
            1,
            "alice",
            "alice@example.com",
            "hashed-password",
        )

    jwt_old = JWTService(settings=old_settings)
    access_token_old = jwt_old.create_access_token(user_id=1, username="alice", role="user")
    refresh_token = jwt_old.create_refresh_token(user_id=1, username="alice")

    session_info = await manager_old.create_session(
        user_id=1,
        access_token=access_token_old,
        refresh_token=refresh_token,
        ip_address="127.0.0.1",
        user_agent="pytest-suite",
    )

    row_before = await pool.fetchone(
        "SELECT refresh_token_hash FROM sessions WHERE id = ?",
        session_info["session_id"],
    )
    assert row_before is not None
    original_hash = row_before["refresh_token_hash"]

    try:
        # Simulate rotation by updating environment-backed settings
        monkeypatch.setenv("JWT_SECRET_KEY", new_secret)
        monkeypatch.setenv("JWT_SECONDARY_SECRET", old_secret)
        reset_settings()

        rotated_settings = Settings(
            AUTH_MODE="multi_user",
            DATABASE_URL=db_url,
            JWT_SECRET_KEY=new_secret,
            JWT_SECONDARY_SECRET=old_secret,
            ENABLE_REGISTRATION=True,
            REQUIRE_REGISTRATION_CODE=False,
        )

        manager_rotated = SessionManager(db_pool=pool, settings=rotated_settings)
        await manager_rotated.initialize()

        jwt_new = JWTService(settings=rotated_settings)
        new_access_token = jwt_new.create_access_token(user_id=1, username="alice", role="user")

        candidate_hashes = manager_rotated._token_hash_candidates(refresh_token)
        assert candidate_hashes, "expected at least one hash candidate"
        assert rotated_settings.JWT_SECRET_KEY == new_secret
        assert candidate_hashes[0] != original_hash, "primary hash should rotate to new secret"

        result = await manager_rotated.refresh_session(
            refresh_token=refresh_token,
            new_access_token=new_access_token,
            new_refresh_token=None,
        )

        assert result["session_id"] == session_info["session_id"]

        row_after = await pool.fetchone(
            "SELECT refresh_token_hash FROM sessions WHERE id = $1",
            session_info["session_id"],
        )
        assert row_after is not None
        rotated_hash = row_after["refresh_token_hash"]

        assert candidate_hashes[0] == manager_rotated.hash_token(refresh_token)
        assert candidate_hashes[-1] != candidate_hashes[0] or len(candidate_hashes) == 1

        assert rotated_hash != original_hash
        assert rotated_hash == manager_rotated.hash_token(refresh_token)
    finally:
        reset_settings()
        if manager_rotated:
            await manager_rotated.shutdown()
        await manager_old.shutdown()
        await pool.close()


@pytest.mark.asyncio
async def test_validate_session_persists_last_activity(isolated_test_environment):
    reset_settings()
    # Use per-test PostgreSQL database from AuthNZ fixture
    _client, _db_name = isolated_test_environment  # client unused; ensures DB is created and env is set
    db_url = os.getenv("DATABASE_URL", "")
    settings = Settings(
        AUTH_MODE="multi_user",
        DATABASE_URL=db_url,
        JWT_SECRET_KEY="session-last-activity-secret-1234567890abcd",
        ENABLE_REGISTRATION=True,
        REQUIRE_REGISTRATION_CODE=False,
    )

    pool = DatabasePool(settings)
    await pool.initialize()

    # Ensure a user exists for FK constraints
    async with pool.transaction() as conn:
        await conn.execute(
            """
            INSERT INTO users (id, username, email, password_hash, is_active)
            VALUES ($1, $2, $3, $4, TRUE)
            """,
            1,
            "bob",
            "bob@example.com",
            "bob-hashed-password",
        )

    manager = SessionManager(db_pool=pool, settings=settings)
    await manager.initialize()

    jwt_service = JWTService(settings=settings)
    access_token = jwt_service.create_access_token(user_id=1, username="bob", role="user")
    refresh_token = jwt_service.create_refresh_token(user_id=1, username="bob")

    session_info = await manager.create_session(
        user_id=1,
        access_token=access_token,
        refresh_token=refresh_token,
        ip_address="127.0.0.1",
        user_agent="pytest-suite",
    )

    row_before = await pool.fetchone(
        "SELECT last_activity FROM sessions WHERE id = $1",
        session_info["session_id"],
    )
    before_value = None
    if row_before is not None:
        try:
            before_value = row_before["last_activity"]
        except Exception:
            before_value = row_before[0]

    result = await manager.validate_session(access_token)
    assert result is not None

    row_after = await pool.fetchone(
        "SELECT last_activity FROM sessions WHERE id = $1",
        session_info["session_id"],
    )
    assert row_after is not None
    try:
        after_value = row_after["last_activity"]
    except Exception:
        after_value = row_after[0]

    assert after_value is not None
    if before_value is not None:
        assert after_value != before_value

    await manager.shutdown()
    await pool.close()
    reset_settings()
