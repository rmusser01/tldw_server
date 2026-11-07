"""One-time magic-link redemption over HTTP against a real SQLite database."""

import asyncio
import sqlite3
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from tldw_Server_API.app.api.v1.endpoints import auth
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.tests.AuthNZ_SQLite._user_fixtures import create_authnz_test_user


@pytest.fixture
async def sign_in(tmp_path, monkeypatch, fresh_services):
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'magic.db'}")
    monkeypatch.setenv("JWT_SECRET_KEY", "synthetic-signing-key-for-regression-tests-only")
    monkeypatch.setenv("REDIS_URL", "")
    reset_settings()
    pool = await get_db_pool()
    email = "reader@example.test"
    user_id = await create_authnz_test_user(pool, username="reader", email=email, is_verified=True)
    jwt = JWTService()
    token = jwt.create_magic_link_token(email=email, user_id=user_id, expires_in_minutes=5)
    app = FastAPI()
    app.include_router(auth.router, prefix="/api/v1")
    app.dependency_overrides[auth.get_jwt_service_dep] = lambda: jwt
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test",
        ) as client:
            async def submit():
                return await client.post("/api/v1/auth/magic-link/verify", json={"token": token})

            yield SimpleNamespace(submit=submit, pool=pool, jwt=jwt, user_id=user_id)
    finally:
        reset_settings()


@pytest.mark.asyncio
async def test_concurrent_verified_submissions_create_one_session(sign_in, monkeypatch):
    verify = sign_in.jwt.verify_token_async
    both_verified = asyncio.Event()
    arrivals = 0

    async def verify_before_claim(*args, **kwargs):
        nonlocal arrivals
        claims = await verify(*args, **kwargs)
        arrivals += 1
        if arrivals == 2:
            both_verified.set()
        await asyncio.wait_for(both_verified.wait(), timeout=5)
        return claims

    monkeypatch.setattr(sign_in.jwt, "verify_token_async", verify_before_claim)
    results = await asyncio.gather(sign_in.submit(), sign_in.submit())
    assert sorted(result.status_code for result in results) == [200, 400]
    assert await sign_in.pool.fetchval("SELECT count(*) FROM sessions WHERE user_id = ?", sign_in.user_id) == 1
    assert (await sign_in.submit()).status_code == 400


@pytest.mark.asyncio
async def test_blacklist_write_failure_cannot_grant_credentials(sign_in):
    # Raise only on insertion: JWT verification still sees a healthy blacklist.
    with sqlite3.connect(sign_in.pool.db_path) as db:
        db.execute(
            "CREATE TRIGGER reject_claim BEFORE INSERT ON token_blacklist "
            "BEGIN SELECT RAISE(ABORT, 'synthetic write failure'); END"
        )
    response = await sign_in.submit()
    assert response.status_code == 503
    assert "access_token" not in response.json()
    assert await sign_in.pool.fetchval("SELECT count(*) FROM sessions WHERE user_id = ?", sign_in.user_id) == 0
    assert await sign_in.pool.fetchval("SELECT last_login FROM users WHERE id = ?", sign_in.user_id) is None
    with sqlite3.connect(sign_in.pool.db_path) as db:
        db.execute("DROP TRIGGER reject_claim")
    assert (await sign_in.submit()).status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("claim", ["jti", "exp"])
async def test_missing_redemption_claim_denies_sign_in(sign_in, monkeypatch, claim):
    verify = sign_in.jwt.verify_token_async

    async def verify_without_claim(*args, **kwargs):
        claims = await verify(*args, **kwargs)
        del claims[claim]
        return claims

    monkeypatch.setattr(sign_in.jwt, "verify_token_async", verify_without_claim)
    assert (await sign_in.submit()).status_code == 400
    assert await sign_in.pool.fetchval("SELECT count(*) FROM sessions WHERE user_id = ?", sign_in.user_id) == 0


@pytest.mark.asyncio
async def test_downstream_failure_leaves_link_spent(sign_in, monkeypatch):
    async def unavailable(*args):
        raise RuntimeError("synthetic membership failure")

    with monkeypatch.context() as patch:
        patch.setattr(auth, "_ensure_user_org_membership", unavailable)
        assert (await sign_in.submit()).status_code == 500
    assert (await sign_in.submit()).status_code == 400
    assert await sign_in.pool.fetchval("SELECT count(*) FROM sessions WHERE user_id = ?", sign_in.user_id) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("driver", ["postgres", "sqlite"])
async def test_unconfirmed_insert_result_is_unavailable(sign_in, monkeypatch, driver):
    from unittest.mock import AsyncMock

    consume = auth.consume_magic_link
    result = "UNKNOWN" if driver == "postgres" else SimpleNamespace(rowcount=-1, close=AsyncMock())

    async def consume_with_unknown_result(db, **kwargs):
        return await consume(SimpleNamespace(execute=AsyncMock(return_value=result)), **kwargs)

    monkeypatch.setattr(auth, "consume_magic_link", consume_with_unknown_result)
    assert (await sign_in.submit()).status_code == 503
    assert await sign_in.pool.fetchval("SELECT count(*) FROM sessions WHERE user_id = ?", sign_in.user_id) == 0
