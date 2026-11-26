"""Live-session predicate, cap-row cleanup and the per-user session cap (eviction, ordering,
concurrency, backlog, revoke-reason lookups) against isolated PostgreSQL."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.repos.sessions_repo import AuthnzSessionsRepo
from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

pytestmark = pytest.mark.integration


async def _create_user(pool, username: str) -> int:
    users_db = UsersDB(pool)
    await users_db.initialize()
    created = await users_db.create_user(
        username=username,
        email=f"{username}@example.test",
        password_hash="hash",
        role="user",
        is_active=True,
        is_superuser=False,
        storage_quota_mb=5120,
    )
    return int(created["id"])


async def _row(repo, uid, tag, *, access_in, refresh_in) -> int:
    now = datetime.now(timezone.utc)
    return await repo.create_session_record(
        user_id=uid,
        token_hash=f"h-{tag}",
        refresh_token_hash=f"r-{tag}",
        encrypted_token="enc",
        encrypted_refresh="enc",
        expires_at=now + access_in,
        refresh_expires_at=now + refresh_in,
        ip_address="127.0.0.1",
        user_agent="pytest-pg",
        device_id=None,
        access_jti=f"aj-{tag}",
        refresh_jti=f"rj-{tag}",
    )


async def _revoke(pool, repo, sid, reason, minutes_ago=0) -> None:
    await repo.revoke_session_record(session_id=sid, revoked_by=None, reason=reason)
    await pool.execute(
        "UPDATE sessions SET revoked_at = $1 WHERE id = $2",
        (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).replace(tzinfo=None),
        sid,
    )


@pytest.mark.asyncio
async def test_live_predicate_and_cleanup_postgres(
    isolated_test_environment: tuple[TestClient, str],
) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

    pool = await get_db_pool()
    uid = await _create_user(pool, "pg_session_cap")
    repo = AuthnzSessionsRepo(pool)
    h, d = timedelta(hours=1), timedelta(days=1)

    live = await _row(repo, uid, "live", access_in=h, refresh_in=7 * d)
    idle = await _row(repo, uid, "idle", access_in=-2 * d, refresh_in=5 * d)
    dead = await _row(repo, uid, "dead", access_in=-3 * d, refresh_in=-2 * d)
    revoked = await _row(repo, uid, "revoked", access_in=h, refresh_in=7 * d)
    await _revoke(pool, repo, revoked, "User logout")

    assert {r["id"] for r in await repo.get_active_sessions_for_user(uid)} == {live, idle}

    cap_61 = await _row(repo, uid, "cap61", access_in=h, refresh_in=7 * d)
    cap_30 = await _row(repo, uid, "cap30", access_in=h, refresh_in=7 * d)
    await _revoke(pool, repo, cap_61, "session_cap", 61)
    await _revoke(pool, repo, cap_30, "session_cap", 30)

    await repo.cleanup_expired_sessions(session_cap_retention=timedelta(hours=1))

    rows = await pool.fetch("SELECT id FROM sessions WHERE user_id = $1", uid)
    remaining = {int(r["id"]) for r in rows}
    assert cap_61 not in remaining and cap_30 in remaining
    assert dead not in remaining and idle in remaining
    assert revoked in remaining


# --- the per-user cap on session creation ------------------------------------


async def _capped(repo, uid, tag, *, device_id=None) -> int:
    now = datetime.now(timezone.utc)
    new_id, _evicted = await repo.create_session_record_with_cap(
        user_id=uid,
        token_hash=f"h-{tag}",
        refresh_token_hash=f"r-{tag}",
        encrypted_token="enc",
        encrypted_refresh="enc",
        expires_at=now + timedelta(hours=1),
        refresh_expires_at=now + timedelta(days=7),
        ip_address="127.0.0.1",
        user_agent="pytest-pg",
        device_id=device_id,
        access_jti=f"aj-{tag}",
        refresh_jti=f"rj-{tag}",
        max_live_sessions=4,
    )
    return new_id


async def _live(repo, uid) -> set[int]:
    return {r["id"] for r in await repo.get_active_sessions_for_user(uid)}


@pytest.mark.asyncio
async def test_fifth_session_evicts_the_oldest_postgres(
    isolated_test_environment: tuple[TestClient, str],
) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

    pool = await get_db_pool()
    uid = await _create_user(pool, "pg_cap_fifth")
    other_uid = await _create_user(pool, "pg_cap_other")
    repo = AuthnzSessionsRepo(pool)

    other = await _capped(repo, other_uid, "other")
    ids = [await _capped(repo, uid, f"s{i}") for i in range(5)]

    assert await _live(repo, uid) == set(ids[1:])
    row = await pool.fetchone("SELECT is_active, revoke_reason FROM sessions WHERE id = $1", ids[0])
    assert row["revoke_reason"] == "session_cap" and not row["is_active"]
    assert other in await _live(repo, other_uid)


@pytest.mark.asyncio
async def test_same_device_id_replaces_only_its_own_row_postgres(
    isolated_test_environment: tuple[TestClient, str],
) -> None:
    from collections import Counter

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

    pool = await get_db_pool()
    uid = await _create_user(pool, "pg_cap_device")
    repo = AuthnzSessionsRepo(pool)
    a, b, c = [await _capped(repo, uid, d, device_id=d) for d in ("dev-aaaaaaaa", "dev-bbbbbbbb", "dev-cccccccc")]
    for i in range(6):
        await _capped(repo, uid, f"d{i}", device_id="dev-dddddddd")

    rows = await repo.get_active_sessions_for_user(uid)
    assert Counter(r["device_id"] for r in rows) == {
        "dev-aaaaaaaa": 1, "dev-bbbbbbbb": 1, "dev-cccccccc": 1, "dev-dddddddd": 1
    }
    assert {a, b, c} <= await _live(repo, uid)


@pytest.mark.asyncio
async def test_cap_evicts_the_least_recently_active_session_postgres(
    isolated_test_environment: tuple[TestClient, str],
) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

    pool = await get_db_pool()
    uid = await _create_user(pool, "pg_cap_lru")
    repo = AuthnzSessionsRepo(pool)
    ids = [await _capped(repo, uid, f"s{i}") for i in range(4)]
    await pool.execute(
        "UPDATE sessions SET created_at = NOW() - INTERVAL '2 hours', "
        "last_activity = NOW() - INTERVAL '2 hours' WHERE user_id = $1",
        uid,
    )
    await pool.execute("UPDATE sessions SET last_activity = NOW() WHERE id = $1", ids[0])

    new = await _capped(repo, uid, "new")

    assert await _live(repo, uid) == {ids[0], ids[2], ids[3], new}  # ids[1] was least recently active


@pytest.mark.asyncio
async def test_headerless_logins_replace_each_other_before_browsers_postgres(
    isolated_test_environment: tuple[TestClient, str],
) -> None:
    from collections import Counter

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

    pool = await get_db_pool()
    uid = await _create_user(pool, "pg_cap_headerless")
    repo = AuthnzSessionsRepo(pool)
    for device in ("dev-aaaaaaaa", "dev-bbbbbbbb", "dev-cccccccc"):
        await _capped(repo, uid, device, device_id=device)
    for i in range(3):
        await _capped(repo, uid, f"headerless{i}")  # no install id

    rows = await repo.get_active_sessions_for_user(uid)
    assert Counter(r["device_id"] for r in rows) == {
        "dev-aaaaaaaa": 1, "dev-bbbbbbbb": 1, "dev-cccccccc": 1, None: 1
    }


@pytest.mark.asyncio
async def test_concurrent_creates_at_the_cap_never_exceed_it_postgres(
    isolated_test_environment: tuple[TestClient, str],
) -> None:
    import asyncio

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

    pool = await get_db_pool()
    uid = await _create_user(pool, "pg_cap_concurrent")
    repo = AuthnzSessionsRepo(pool)
    for i in range(4):
        await _capped(repo, uid, f"seed{i}")

    results = await asyncio.gather(*[_capped(repo, uid, f"race{i}") for i in range(6)])

    assert len(results) == 6
    assert len(await _live(repo, uid)) == 4


@pytest.mark.asyncio
async def test_expired_refresh_rows_do_not_count_toward_the_cap_postgres(
    isolated_test_environment: tuple[TestClient, str],
) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

    pool = await get_db_pool()
    uid = await _create_user(pool, "pg_cap_expired")
    repo = AuthnzSessionsRepo(pool)
    h, d = timedelta(hours=1), timedelta(days=1)
    for i in range(4):
        await _row(repo, uid, f"dead{i}", access_in=-3 * d, refresh_in=-2 * d)
    for i in range(3):
        await _row(repo, uid, f"live{i}", access_in=h, refresh_in=7 * d)

    await _capped(repo, uid, "new")

    assert len(await _live(repo, uid)) == 4
    rows = await pool.fetch("SELECT id FROM sessions WHERE revoke_reason = $1", "session_cap")
    assert not rows


@pytest.mark.asyncio
async def test_cap_backlog_is_evicted_in_one_creation_postgres(
    isolated_test_environment: tuple[TestClient, str],
) -> None:
    import time

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

    pool = await get_db_pool()
    uid = await _create_user(pool, "pg_cap_backlog")
    repo = AuthnzSessionsRepo(pool)
    later = (datetime.now(timezone.utc) + timedelta(days=1)).replace(tzinfo=None)
    # 33,000 rows: more than asyncpg's 32,767-argument limit, so a per-victim IN list cannot work.
    await pool.execute(
        """
        INSERT INTO sessions (
            user_id, token_hash, refresh_token_hash, encrypted_token, encrypted_refresh,
            expires_at, refresh_expires_at, device_id
        )
        SELECT $1, 'backlog-h-' || g, 'backlog-r-' || g, 'enc', 'enc', $2, $2,
               'single-user-cookie:v1'
        FROM generate_series(1, 33000) AS g
        """,
        uid,
        later,
    )
    assert len(await _live(repo, uid)) == 33000

    started = time.monotonic()
    new_id = await _capped(repo, uid, "new")
    elapsed = time.monotonic() - started
    print(f"postgres backlog capped insert: {elapsed:.3f}s")

    assert elapsed < 5
    live = await _live(repo, uid)
    assert len(live) == 4 and new_id in live


# --- why a revoked token was revoked (session_replaced lookups) ---------------


@pytest.mark.asyncio
async def test_get_reason_for_jti_postgres(
    isolated_test_environment: tuple[TestClient, str],
) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.repos.token_blacklist_repo import AuthnzTokenBlacklistRepo

    pool = await get_db_pool()
    uid = await _create_user(pool, "pg_reason_jti")
    repo = AuthnzTokenBlacklistRepo(pool)
    await repo.ensure_schema()
    await repo.insert_blacklisted_token(
        jti="jti-capped",
        user_id=uid,
        token_type="access",
        expires_at=datetime.utcnow() + timedelta(hours=1),
        reason="session_cap",
        revoked_by=None,
        ip_address=None,
    )

    assert await repo.get_reason_for_jti("jti-capped") == "session_cap"
    assert await repo.get_reason_for_jti("jti-unknown") is None


@pytest.mark.asyncio
async def test_fetch_revoke_reason_for_token_hash_candidates_postgres(
    isolated_test_environment: tuple[TestClient, str],
) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

    pool = await get_db_pool()
    uid = await _create_user(pool, "pg_reason_hash")
    repo = AuthnzSessionsRepo(pool)
    h = timedelta(hours=1)
    revoked = await _row(repo, uid, "rv", access_in=h, refresh_in=7 * h)
    await _revoke(pool, repo, revoked, "session_cap")
    await _row(repo, uid, "lv", access_in=h, refresh_in=7 * h)

    fetch = repo.fetch_revoke_reason_for_token_hash_candidates
    assert await fetch(["h-rv"]) == "session_cap"
    assert await fetch(["nope", "r-rv"]) == "session_cap"
    assert await fetch(["h-lv"]) is None
    assert await fetch(["r-lv"]) is None
    assert await fetch(["no-match"]) is None


@pytest.mark.asyncio
async def test_cap_evicted_session_gets_session_replaced_over_http_postgres(
    isolated_test_environment: tuple[TestClient, str], monkeypatch
) -> None:
    client, _db_name = isolated_test_environment
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("JWT_SECRET_KEY", "X" * 40)
    password = "Str0ngP@ssw0rd!"
    reg = client.post(
        "/api/v1/auth/register",
        json={"username": "pgcapuser", "email": "pgcap@example.com", "password": password},
    )
    assert reg.status_code == 200

    tokens = []
    for _ in range(5):  # relies on the default cap of 4: this fixture ignores MAX_SESSIONS_PER_USER
        login = client.post("/api/v1/auth/login", data={"username": "pgcapuser", "password": password})
        assert login.status_code == 200
        tokens.append(login.json())
    first, last = tokens[0], tokens[-1]

    r = client.get(
        "/api/v1/auth/sessions", headers={"Authorization": f"Bearer {first['access_token']}"}
    )
    assert r.status_code == 401 and r.json()["detail"]["code"] == "session_replaced"
    ok = client.get(
        "/api/v1/auth/sessions", headers={"Authorization": f"Bearer {last['access_token']}"}
    )
    assert ok.status_code == 200

    rr = client.post("/api/v1/auth/refresh", json={"refresh_token": first["refresh_token"]})
    assert rr.status_code == 401 and rr.json()["detail"]["code"] == "session_replaced"
