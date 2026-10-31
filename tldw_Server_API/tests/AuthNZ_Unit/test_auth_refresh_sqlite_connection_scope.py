"""Refresh requests must not lock their own atomic session-token update."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import auth
from tldw_Server_API.app.core.AuthNZ import database, session_manager, token_blacklist
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    ConnectionPoolExhaustedError,
    DatabaseConcurrencyConflict,
    DatabaseLockError,
    TransactionError,
)
from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService
from tldw_Server_API.app.core.AuthNZ.repos.sessions_repo import AuthnzSessionsRepo
from tldw_Server_API.app.core.AuthNZ.repos.token_blacklist_repo import AuthnzTokenBlacklistRepo
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.app.core.AuthNZ.settings import Settings, reset_settings


@dataclass
class RefreshHarness:
    client: httpx.AsyncClient
    pool: database.DatabasePool
    manager: session_manager.SessionManager
    blacklist: token_blacklist.TokenBlacklist
    jwt: JWTService
    user_id: int
    other_user_id: int

    async def create_session(self, *, user_id: int | None = None) -> dict:
        owner = self.user_id if user_id is None else user_id
        access = self.jwt.create_access_token(owner, "refresh-user", "user", expires_delta=timedelta(minutes=-1))
        refresh = self.jwt.create_refresh_token(owner, "refresh-user")
        return await self.manager.create_session(owner, access, refresh)

    async def refresh(self, value: str) -> httpx.Response:
        return await self.client.post("/api/v1/auth/refresh", json={"refresh_token": value})


@pytest.fixture
async def refresh_harness(tmp_path, monkeypatch):
    """Use production SQLite transactions and services, with only local resources."""
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'refresh.db'}")
    monkeypatch.setenv("AUTHNZ_SQLITE_LOCK_MAX_RETRIES", "0")
    monkeypatch.setenv("AUTHNZ_SQLITE_LOCK_RETRY_AFTER_SECONDS", "1")
    reset_settings()
    settings = Settings(
        AUTH_MODE="multi_user",
        DATABASE_URL=f"sqlite:///{tmp_path / 'refresh.db'}",
        JWT_SECRET_KEY="refresh-regression-signing-key-" * 2,
        JWT_ALGORITHM="HS256",
        SESSION_ENCRYPTION_KEY=Fernet.generate_key().decode(),
        REDIS_URL=None,
        ROTATE_REFRESH_TOKENS=True,
        PII_REDACT_LOGS=True,
    )
    monkeypatch.setattr(session_manager, "get_settings", lambda: settings)
    monkeypatch.setattr(token_blacklist, "get_settings", lambda: settings)
    pool = database.DatabasePool(settings=settings)
    await pool.initialize()
    monkeypatch.setattr(database, "_db_pool", pool)
    # Keep the route's real dependency, forcing its production transaction branch.
    monkeypatch.setattr(auth_deps, "_is_test_mode", lambda: False)
    monkeypatch.setattr(auth, "_is_test_mode", lambda: False)
    original_configure = database.configure_sqlite_connection_async

    async def configure_for_bounded_lock_test(conn):
        await original_configure(conn, busy_timeout_ms=50)

    monkeypatch.setattr(database, "configure_sqlite_connection_async", configure_for_bounded_lock_test)
    users = AuthnzUsersRepo(pool)
    user_id = await users.create_user(  # nosec B106 - deliberately unusable synthetic fixture hash
        username="refresh-user", email="refresh-user@example.com", password_hash="unused-test-hash"
    )
    other_user_id = await users.create_user(  # nosec B106 - deliberately unusable synthetic fixture hash
        username="other-user", email="other-user@example.com", password_hash="unused-test-hash"
    )
    manager = session_manager.SessionManager(db_pool=pool, settings=settings)
    await manager.initialize()
    blacklist = token_blacklist.TokenBlacklist(db_pool=pool, settings=settings)
    await blacklist.initialize()
    monkeypatch.setattr(session_manager, "get_token_blacklist", lambda: blacklist)
    monkeypatch.setattr(token_blacklist, "get_token_blacklist", lambda: blacklist)
    jwt = JWTService(settings=settings)
    app = FastAPI()
    app.include_router(auth.router, prefix="/api/v1")
    app.dependency_overrides[auth.get_jwt_service_dep] = lambda: jwt
    app.dependency_overrides[auth.get_session_manager_dep] = lambda: manager
    app.dependency_overrides[auth.get_settings] = lambda: settings
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://refresh.test",
        ) as client:
            yield RefreshHarness(client, pool, manager, blacklist, jwt, user_id, other_user_id)
    finally:
        await pool.close()
        reset_settings()


@pytest.mark.asyncio
async def test_expired_access_refreshes_through_real_sqlite_request_dependency(refresh_harness):
    session = await refresh_harness.create_session()

    response = await refresh_harness.refresh(session["refresh_token"])

    assert response.status_code == 200
    rotated = response.json()
    assert rotated["refresh_token"] != session["refresh_token"]
    repo = AuthnzSessionsRepo(refresh_harness.pool)
    rows = await repo.get_active_sessions_for_user(refresh_harness.user_id)
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_competing_sqlite_writer_is_retryable_without_spending_refresh(refresh_harness):
    h = refresh_harness
    session = await h.create_session()

    async with h.pool.transaction():
        response = await h.refresh(session["refresh_token"])

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "1"
    assert (await h.refresh(session["refresh_token"])).status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure, expected_status",
    [
        (DatabaseLockError(), 503),
        (ConnectionPoolExhaustedError(), 503),
        (DatabaseConcurrencyConflict(), 503),
        (TransactionError("commit acknowledgement"), 500),
        (RuntimeError("unexpected database failure"), 500),
    ],
    ids=["busy", "pool", "conflict", "ambiguous-commit", "unknown"],
)
async def test_refresh_storage_failure_is_not_invalid_credentials(
    refresh_harness, monkeypatch, failure, expected_status
):
    h = refresh_harness
    session = await h.create_session()

    async def fail_update(*args, **kwargs):
        raise failure

    with monkeypatch.context() as patch:
        patch.setattr(AuthnzSessionsRepo, "update_session_tokens_for_refresh", fail_update)
        response = await h.refresh(session["refresh_token"])

    assert response.status_code == expected_status
    assert ("Retry-After" in response.headers) == (expected_status == 503)
    assert (await h.refresh(session["refresh_token"])).status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("failure, expected_status", [(DatabaseLockError(), 503), (RuntimeError("offline"), 500)])
@pytest.mark.parametrize("boundary", ["blacklist", "revoked-session"])
async def test_refresh_blacklist_lookup_failure_is_distinct_from_revocation(
    refresh_harness, monkeypatch, failure, expected_status, boundary
):
    h = refresh_harness
    session = await h.create_session()

    async def fail_lookup(*args, **kwargs):
        raise failure

    with monkeypatch.context() as patch:
        if boundary == "blacklist":
            patch.setattr(AuthnzTokenBlacklistRepo, "get_active_expiry_for_jti", fail_lookup)
        else:
            patch.setattr(AuthnzSessionsRepo, "has_revoked_session_for_token_hash_candidates", fail_lookup)
        response = await h.refresh(session["refresh_token"])
        # Existing callers retain the fail-closed contract.
        assert await h.manager.is_token_blacklisted(session["refresh_token"])
        if boundary == "blacklist":
            assert await h.blacklist.is_blacklisted(h.jwt.decode_refresh_token(session["refresh_token"])["jti"])

    assert response.status_code == expected_status
    assert ("Retry-After" in response.headers) == (expected_status == 503)
    assert (await h.refresh(session["refresh_token"])).status_code == 200


@pytest.mark.asyncio
async def test_postcommit_cache_failure_has_no_safe_retry_claim(refresh_harness, monkeypatch):
    h = refresh_harness
    session = await h.create_session()

    class Cache:
        async def exists(self, key):
            return False

    async def fail_cache(*args, **kwargs):
        raise DatabaseLockError()

    monkeypatch.setattr(h.manager, "redis_client", Cache())
    monkeypatch.setattr(h.manager, "_clear_session_cache", fail_cache)
    response = await h.refresh(session["refresh_token"])

    assert response.status_code == 500
    assert "Retry-After" not in response.headers
    repo = AuthnzSessionsRepo(h.pool)
    assert (
        await repo.find_active_session_by_refresh_hash_candidates(
            h.manager._token_hash_candidates(session["refresh_token"])
        )
        is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["scope", "transaction", "blacklist-write"])
async def test_refresh_cancellation_propagates_with_transaction_rollback(refresh_harness, monkeypatch, boundary):
    h = refresh_harness
    session = await h.create_session()

    async def cancel(*args, **kwargs):
        raise asyncio.CancelledError()

    with monkeypatch.context() as patch:
        if boundary == "scope":
            patch.setattr(auth, "list_memberships_for_user", cancel)
        elif boundary == "blacklist-write":
            patch.setattr(h.blacklist, "revoke_token", cancel)
        else:
            transaction = h.pool.transaction

            @asynccontextmanager
            async def cancel_before_commit(*args, **kwargs):
                async with transaction(*args, **kwargs) as conn:
                    yield conn
                    raise asyncio.CancelledError()

            patch.setattr(h.pool, "transaction", cancel_before_commit)
        with pytest.raises(asyncio.CancelledError):
            await h.refresh(session["refresh_token"])

    if boundary != "blacklist-write":
        assert (await h.refresh(session["refresh_token"])).status_code == 200
    else:
        # Cancellation after commit must not restore the already-spent token.
        assert (await h.refresh(session["refresh_token"])).status_code == 401


@pytest.mark.asyncio
async def test_repeated_rotation_rejects_each_prior_token(refresh_harness):
    h = refresh_harness
    session = await h.create_session()
    current = session["refresh_token"]
    for _ in range(2):
        response = await h.refresh(current)
        assert response.status_code == 200
        replacement = response.json()["refresh_token"]
        assert replacement != current
        assert (await h.refresh(current)).status_code == 401
        assert await h.blacklist.is_blacklisted(h.jwt.decode_refresh_token(current)["jti"])
        current = replacement


@pytest.mark.asyncio
async def test_concurrent_refresh_has_one_atomic_winner(refresh_harness, monkeypatch):
    h = refresh_harness
    session = await h.create_session()
    lookup = AuthnzSessionsRepo.find_active_session_by_refresh_hash_candidates
    arrived = 0
    ready = asyncio.Event()

    async def synchronized_lookup(*args, **kwargs):
        nonlocal arrived
        record = await lookup(*args, **kwargs)
        arrived += 1
        if arrived == 2:
            ready.set()
        await asyncio.wait_for(ready.wait(), timeout=5)
        return record

    with monkeypatch.context() as patch:
        patch.setattr(AuthnzSessionsRepo, "find_active_session_by_refresh_hash_candidates", synchronized_lookup)
        responses = await asyncio.gather(h.refresh(session["refresh_token"]), h.refresh(session["refresh_token"]))

    assert sorted(response.status_code for response in responses) == [200, 401]
    winner = next(response for response in responses if response.status_code == 200)
    assert (await h.refresh(winner.json()["refresh_token"])).status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("denial", ["revoked", "wrong-subject", "expired-jwt", "expired-session"])
async def test_invalid_refresh_stays_unauthorized(refresh_harness, denial):
    h = refresh_harness
    access = h.jwt.create_access_token(h.user_id, "refresh-user", "user")
    owner = h.other_user_id if denial == "wrong-subject" else h.user_id
    issuer = h.jwt
    if denial == "expired-jwt":
        issuer = JWTService(settings=h.jwt.settings.model_copy(update={"REFRESH_TOKEN_EXPIRE_DAYS": -1}))
    refresh = issuer.create_refresh_token(owner, "refresh-user")
    options = {}
    if denial == "expired-session":
        options["refresh_expires_at_override"] = datetime.now(timezone.utc) - timedelta(minutes=1)
    session = await h.manager.create_session(h.user_id, access, refresh, **options)
    if denial == "revoked":
        await AuthnzSessionsRepo(h.pool).revoke_session_record(
            session_id=session["session_id"],
            expected_user_id=h.user_id,
            revoked_by=None,
            reason="regression test",
        )

    response = await h.refresh(refresh)

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert "Retry-After" not in response.headers


@pytest.mark.asyncio
async def test_rotation_disabled_preserves_refresh_token(refresh_harness):
    h = refresh_harness
    h.manager.settings.ROTATE_REFRESH_TOKENS = False
    session = await h.create_session()
    for _ in range(2):
        response = await h.refresh(session["refresh_token"])
        assert response.status_code == 200
        assert h.jwt.decode_refresh_token(response.json()["refresh_token"])["jti"] == (
            h.jwt.decode_refresh_token(session["refresh_token"])["jti"]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure, expected_status",
    [(ConnectionPoolExhaustedError(), 503), (DatabaseLockError(), 503), (RuntimeError("unavailable"), 500)],
    ids=["pool", "busy", "unknown"],
)
async def test_request_connection_acquisition_classifies_only_known_transient_failures(
    refresh_harness, monkeypatch, failure, expected_status
):
    h = refresh_harness
    session = await h.create_session()

    @asynccontextmanager
    async def fail_acquire(*args, **kwargs):
        raise failure
        yield  # pragma: no cover - define an async context manager that fails on entry

    with monkeypatch.context() as patch:
        patch.setattr(h.pool, "acquire_statement_autocommit", fail_acquire)
        response = await h.refresh(session["refresh_token"])

    assert response.status_code == expected_status
    assert ("Retry-After" in response.headers) == (expected_status == 503)
    assert (await h.refresh(session["refresh_token"])).status_code == 200


@pytest.mark.asyncio
async def test_request_connection_does_not_make_post_yield_failure_retryable(refresh_harness):
    connection = auth_deps.get_login_db_connection()
    await connection.__anext__()
    with pytest.raises(DatabaseLockError):
        await connection.athrow(DatabaseLockError())


@pytest.mark.asyncio
async def test_revocation_between_lookup_and_atomic_update_still_denies(refresh_harness, monkeypatch):
    h = refresh_harness
    session = await h.create_session()
    lookup = AuthnzSessionsRepo.find_active_session_by_refresh_hash_candidates

    async def revoke_after_lookup(repo, *args, **kwargs):
        record = await lookup(repo, *args, **kwargs)
        await repo.revoke_session_record(
            session_id=session["session_id"],
            expected_user_id=h.user_id,
            revoked_by=None,
            reason="concurrent revocation regression",
        )
        return record

    monkeypatch.setattr(AuthnzSessionsRepo, "find_active_session_by_refresh_hash_candidates", revoke_after_lookup)
    response = await h.refresh(session["refresh_token"])

    assert response.status_code == 401
    assert await h.manager.is_token_blacklisted(session["refresh_token"], strict=True)


@pytest.mark.asyncio
async def test_rotation_remains_replay_safe_if_best_effort_blacklist_write_fails(refresh_harness, monkeypatch):
    h = refresh_harness
    session = await h.create_session()

    async def fail_write(*args, **kwargs):
        raise DatabaseLockError()

    monkeypatch.setattr(h.blacklist, "revoke_token", fail_write)
    response = await h.refresh(session["refresh_token"])

    assert response.status_code == 200
    assert (await h.refresh(session["refresh_token"])).status_code == 401
    assert (await h.refresh(response.json()["refresh_token"])).status_code == 200
