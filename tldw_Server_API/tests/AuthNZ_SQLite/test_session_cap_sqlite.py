"""Live-session predicate (listing, cleanup, the callers that gate on it) and the per-user
session cap (eviction order, bounded blacklisting, revoke reasons) on SQLite."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest_plugins = ("tldw_Server_API.tests._plugins.authnz_full_fixtures",)

pytestmark = pytest.mark.asyncio


def _aware(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


@pytest.fixture
async def authnz_sqlite(tmp_path, monkeypatch, fresh_services):
    """Yield ``(pool, user_id, other_user_id)`` on a fresh SQLite AuthNZ DB."""
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

    db_path = tmp_path / "session_cap.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("SESSION_ENCRYPTION_KEY", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
    monkeypatch.setenv("MAX_SESSIONS_PER_USER", "4")  # the cap tests below assume 4
    reset_settings()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))
    users_db = UsersDB(pool)
    await users_db.initialize()
    ids = []
    for name in ("alice", "bob"):
        created = await users_db.create_user(
            username=name,
            email=f"{name}@example.com",
            password_hash="x",
            role="user",
            is_active=True,
            is_superuser=False,
            storage_quota_mb=5120,
            uuid_value=uuid.uuid4(),
        )
        ids.append(int(created["id"]))
    return pool, ids[0], ids[1]


async def _row(pool, user_id, tag, *, access_in, refresh_in, device_id=None) -> int:
    """Insert a session via the uncapped repo path; ``*_in`` are timedeltas from now."""
    from tldw_Server_API.app.core.AuthNZ.repos.sessions_repo import AuthnzSessionsRepo

    now = datetime.now(timezone.utc)
    return await AuthnzSessionsRepo(pool).create_session_record(
        user_id=user_id,
        token_hash=f"h-{tag}",
        refresh_token_hash=f"r-{tag}",
        encrypted_token="enc",
        encrypted_refresh="enc",
        expires_at=now + access_in,
        refresh_expires_at=now + refresh_in,
        ip_address="127.0.0.1",
        user_agent="pytest",
        device_id=device_id,
        access_jti=f"aj-{tag}",
        refresh_jti=f"rj-{tag}",
    )


async def _revoke(pool, session_id, reason, *, minutes_ago=0):
    from tldw_Server_API.app.core.AuthNZ.repos.sessions_repo import AuthnzSessionsRepo

    await AuthnzSessionsRepo(pool).revoke_session_record(
        session_id=session_id, revoked_by=None, reason=reason
    )
    backdated = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    async with pool.transaction() as conn:
        await conn.execute(
            "UPDATE sessions SET revoked_at = $1 WHERE id = $2", backdated, session_id
        )


async def _fetch(pool, session_id) -> dict:
    row = await pool.fetchone(
        "SELECT is_active, revoke_reason, device_id FROM sessions WHERE id = $1", session_id
    )
    if row is None:
        return {}
    r = dict(row) if isinstance(row, dict) or hasattr(row, "keys") else {
        "is_active": row[0], "revoke_reason": row[1], "device_id": row[2]
    }
    return {"is_active": bool(r["is_active"]), "revoke_reason": r["revoke_reason"], "device_id": r["device_id"]}


async def live_ids(pool, user_id) -> set[int]:
    from tldw_Server_API.app.core.AuthNZ.repos.sessions_repo import AuthnzSessionsRepo

    return {r["id"] for r in await AuthnzSessionsRepo(pool).get_active_sessions_for_user(user_id)}


async def _seed(pool, uid, *, only=("live", "idle")):
    """Seed dead + revoked plus the ``only`` rows of live/idle; returns ``{name: id}``."""
    h, d = timedelta(hours=1), timedelta(days=1)
    ids = {"dead": await _row(pool, uid, "dead", access_in=-3 * d, refresh_in=-2 * d)}
    if "live" in only:
        ids["live"] = await _row(pool, uid, "live", access_in=h, refresh_in=7 * d)
    if "idle" in only:
        ids["idle"] = await _row(pool, uid, "idle", access_in=-2 * d, refresh_in=5 * d)
    ids["revoked"] = await _row(pool, uid, "revoked", access_in=h, refresh_in=7 * d)
    await _revoke(pool, ids["revoked"], "User logout")
    return ids


async def _remaining(pool, uid) -> set[int]:
    rows = await pool.fetchall("SELECT id FROM sessions WHERE user_id = $1", uid)
    return {int(r["id"] if hasattr(r, "keys") else r[0]) for r in rows}


async def test_listing_uses_live_predicate(authnz_sqlite):
    pool, uid, _ = authnz_sqlite
    ids = await _seed(pool, uid)
    assert await live_ids(pool, uid) == {ids["live"], ids["idle"]}


async def test_listing_reports_session_end_as_expires_at(authnz_sqlite):
    from tldw_Server_API.app.core.AuthNZ.repos.sessions_repo import AuthnzSessionsRepo

    pool, uid, _ = authnz_sqlite
    now = datetime.now(timezone.utc)
    ids = await _seed(pool, uid)
    rows = await AuthnzSessionsRepo(pool).get_active_sessions_for_user(uid)
    row = next(r for r in rows if r["id"] == ids["idle"])
    assert abs((_aware(row["expires_at"]) - (now + timedelta(days=5))).total_seconds()) < 10


async def test_cleanup_keeps_idle_refreshable_and_deletes_dead_after_grace(authnz_sqlite):
    from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager

    pool, uid, _ = authnz_sqlite
    ids = await _seed(pool, uid)
    await SessionManager(db_pool=pool).cleanup_expired_sessions()
    remaining = await _remaining(pool, uid)
    assert remaining >= {ids["live"], ids["idle"]}
    assert ids["dead"] not in remaining


@pytest.mark.parametrize(
    "access_minutes,minutes_ago,kept", [(30, 61, False), (30, 30, True), (120, 90, True)]
)
async def test_cleanup_purges_cap_evicted_rows_after_retention(
    authnz_sqlite, monkeypatch, access_minutes, minutes_ago, kept
):
    from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    pool, uid, _ = authnz_sqlite
    monkeypatch.setenv("ACCESS_TOKEN_EXPIRE_MINUTES", str(access_minutes))
    reset_settings()
    h, d = timedelta(hours=1), timedelta(days=1)
    cap_row = await _row(pool, uid, "cap", access_in=h, refresh_in=7 * d)
    logout_row = await _row(pool, uid, "logout", access_in=h, refresh_in=7 * d)
    await _revoke(pool, cap_row, "session_cap", minutes_ago=minutes_ago)
    await _revoke(pool, logout_row, "User logout", minutes_ago=61)
    await SessionManager(db_pool=pool).cleanup_expired_sessions()
    remaining = await _remaining(pool, uid)
    assert (cap_row in remaining) is kept
    assert logout_row in remaining  # ordinary revocations still keep 7 days


async def test_cleanup_purges_logged_out_rows_a_day_after_access_expiry(authnz_sqlite):
    from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager

    pool, uid, _ = authnz_sqlite
    h, d = timedelta(hours=1), timedelta(days=1)
    stale_logout = await _row(pool, uid, "slo", access_in=-2 * d, refresh_in=5 * d)
    fresh_logout = await _row(pool, uid, "flo", access_in=h, refresh_in=5 * d)
    stale_cap = await _row(pool, uid, "scap", access_in=-2 * d, refresh_in=5 * d)
    await _revoke(pool, stale_logout, "User logout")
    await _revoke(pool, fresh_logout, "User logout")
    await _revoke(pool, stale_cap, "session_cap", minutes_ago=30)
    await SessionManager(db_pool=pool).cleanup_expired_sessions()
    remaining = await _remaining(pool, uid)
    assert stale_logout not in remaining
    assert fresh_logout in remaining
    assert stale_cap in remaining  # cap rows keep their own retention


async def test_token_update_is_refused_for_a_cap_evicted_row(authnz_sqlite):
    from tldw_Server_API.app.core.AuthNZ.exceptions import SessionError
    from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager

    pool, uid, _ = authnz_sqlite
    sid = await _row(pool, uid, "temp", access_in=timedelta(hours=1), refresh_in=7 * timedelta(days=1))
    await _revoke(pool, sid, "session_cap")
    with pytest.raises(SessionError):
        await SessionManager(db_pool=pool).update_session_tokens(sid, "access.real", "refresh.real")
    row = await pool.fetchone("SELECT token_hash FROM sessions WHERE id = $1", sid)
    assert dict(row)["token_hash"] == "h-temp"


async def test_failed_sign_in_steps_revoke_the_placeholder_row(authnz_sqlite):
    from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager

    pool, uid, _ = authnz_sqlite
    sm = SessionManager(db_pool=pool)
    made = {}
    for tag in ("fails", "works"):
        made[tag] = (await sm.create_session(user_id=uid, access_token=f"a-{tag}", refresh_token=f"r-{tag}"))[
            "session_id"
        ]
    with pytest.raises(ValueError, match="boom"):
        async with sm.revoke_on_failure(made["fails"], uid):
            raise ValueError("boom")
    async with sm.revoke_on_failure(made["works"], uid):
        pass
    assert await live_ids(pool, uid) == {made["works"]}


async def test_last_login_failure_leaves_no_live_session(authnz_sqlite, monkeypatch):
    from starlette.requests import Request

    import tldw_Server_API.app.api.v1.endpoints.auth as auth
    from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings

    pool, uid, _ = authnz_sqlite

    async def _fail_last_login(*args, **kwargs):
        raise RuntimeError("last login write failed")

    class _JWT:
        def create_access_token(self, **kwargs):
            return "access.real"

        def create_refresh_token(self, **kwargs):
            return "refresh.real"

    monkeypatch.setattr(
        "tldw_Server_API.app.services.auth_service.update_user_last_login", _fail_last_login
    )
    request = Request({
        "type": "http", "method": "POST", "path": "/x", "headers": [],
        "client": ("198.51.100.44", 1234), "scheme": "http", "query_string": b"",
        "server": ("testserver", 80),
    })
    with pytest.raises(RuntimeError, match="last login"):
        await auth._issue_multi_user_tokens(
            request=request,
            db=None,
            user={"id": uid, "username": "alice", "role": "user"},
            jwt_service=_JWT(),
            session_manager=SessionManager(db_pool=pool),
            settings=get_settings(),
        )
    assert await live_ids(pool, uid) == set()


# --- callers that gate on the live predicate ---------------------------------


@pytest.mark.parametrize("with_idle,expect_run", [(False, False), (True, True)])
async def test_workflows_require_online_uses_live_predicate(
    authnz_sqlite, monkeypatch, with_idle, expect_run
):
    import dataclasses
    from typing import Any

    from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager
    from tldw_Server_API.app.services import workflows_scheduler as mod
    from tldw_Server_API.tests.Workflows.test_workflows_scheduler import _schedule

    pool, uid, _ = authnz_sqlite
    await _seed(pool, uid, only=("idle",) if with_idle else ())
    monkeypatch.delenv("WORKFLOWS_MINT_VIRTUAL_KEYS", raising=False)

    async def _sm():
        return SessionManager(db_pool=pool)

    monkeypatch.setattr(mod, "get_session_manager", _sm)
    schedule = dataclasses.replace(_schedule("s1", user_id=str(uid)), require_online=True)
    history: list[dict[str, Any]] = []

    class _StubDB:
        def get_schedule(self, schedule_id):
            return schedule

        def set_history(self, _sid, **kwargs):
            history.append(kwargs)

    svc = mod._WFRecurringScheduler()
    monkeypatch.setattr(svc, "_get_db", lambda _uid: _StubDB())

    # No core scheduler is attached: a schedule that passes the presence gate stops at
    # "pending" (nothing to submit to); one that fails it is marked "skipped_offline".
    await svc._run_schedule("s1", user_id=uid)  # type: ignore[attr-defined]

    assert history[-1].get("last_status") == ("pending" if expect_run else "skipped_offline")


@pytest.mark.parametrize("with_idle,expect_run", [(False, False), (True, True)])
async def test_reading_digest_require_online_uses_live_predicate(
    authnz_sqlite, monkeypatch, with_idle, expect_run
):
    from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager
    from tldw_Server_API.app.services import reading_digest_scheduler as mod

    pool, uid, _ = authnz_sqlite
    await _seed(pool, uid, only=("idle",) if with_idle else ())

    async def _sm():
        return SessionManager(db_pool=pool)

    monkeypatch.setattr(mod, "get_session_manager", _sm)
    schedule = SimpleNamespace(
        id="d1", enabled=True, cron="0 * * * *", timezone="UTC", next_run_at=None,
        require_online=True, user_id=str(uid), tenant_id="default",
    )

    class _StubDB:
        def get_reading_digest_schedule(self, _sid):
            return schedule

        def try_claim_reading_digest_run(self, *_a, **_k):
            return True

        def set_reading_digest_history(self, *_a, **_k):
            return None

    class _Jobs:
        created: list[dict] = []

        def create_job(self, **kwargs):
            self.created.append(kwargs)
            return {"id": 1}

    jobs = _Jobs()
    sched = mod._ReadingDigestScheduler()
    monkeypatch.setattr(sched, "_get_db", lambda _uid: _StubDB())
    sched._jobs = jobs  # type: ignore[assignment]

    await sched._run_schedule("d1", user_id=uid)

    assert bool(jobs.created) is expect_run


async def test_admin_session_listing_uses_live_predicate(authnz_sqlite, monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager
    from tldw_Server_API.app.services import admin_sessions_mfa_service as svc

    pool, uid, _ = authnz_sqlite
    ids = await _seed(pool, uid)

    async def _noop(*_a, **_k):
        return None

    monkeypatch.setattr(svc.admin_scope_service, "enforce_admin_user_scope", _noop)
    out = await svc.list_user_sessions(object(), uid, SessionManager(db_pool=pool))
    assert {s.id for s in out} == {ids["live"], ids["idle"]}


async def test_profile_security_counts_live_sessions(authnz_sqlite, monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager
    from tldw_Server_API.app.core.UserProfiles import service
    from tldw_Server_API.app.core.UserProfiles.service import UserProfileService

    pool, uid, _ = authnz_sqlite
    await _seed(pool, uid)

    class _Mfa:
        async def get_user_mfa_status(self, _uid):
            return {"enabled": False}

    class _Keys:
        async def list_user_keys(self, **_k):
            return []

    monkeypatch.setattr(service, "get_mfa_service", lambda: _Mfa())
    sec = await UserProfileService(pool).build_security(
        user_id=uid, session_manager=SessionManager(db_pool=pool), api_key_manager=_Keys()
    )
    assert sec["active_sessions"] == 2


# --- the per-user cap on session creation ------------------------------------


async def _create(manager, uid, device=None) -> int:
    result = await manager.create_session(
        user_id=uid,
        access_token=f"temp_access_{uuid.uuid4().hex}",
        refresh_token=f"temp_refresh_{uuid.uuid4().hex}",
        ip_address="127.0.0.1",
        user_agent="pytest",
        device_id=device,
    )
    return int(result["session_id"])


def _manager(pool):
    from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager

    return SessionManager(db_pool=pool)


async def _ids_with_reason(pool, reason) -> set[int]:
    rows = await pool.fetchall("SELECT id FROM sessions WHERE revoke_reason = $1", reason)
    return {int(r["id"] if hasattr(r, "keys") else r[0]) for r in rows}


async def test_fifth_session_evicts_the_oldest(authnz_sqlite):
    pool, uid, other_uid = authnz_sqlite
    manager = _manager(pool)
    other = await _create(manager, other_uid)
    ids = [await _create(manager, uid) for _ in range(5)]
    assert await live_ids(pool, uid) == set(ids[1:])
    evicted = await _fetch(pool, ids[0])
    assert evicted["revoke_reason"] == "session_cap" and not evicted["is_active"]
    assert other in await live_ids(pool, other_uid)


async def test_cap_evicts_the_least_recently_active_session(authnz_sqlite):
    pool, uid, _ = authnz_sqlite
    manager = _manager(pool)
    ids = [await _create(manager, uid) for _ in range(4)]
    async with pool.transaction() as conn:
        await conn.execute(
            "UPDATE sessions SET created_at = datetime('now', '-2 hours'), "
            "last_activity = datetime('now', '-2 hours') WHERE user_id = $1",
            uid,
        )
        await conn.execute("UPDATE sessions SET last_activity = datetime('now') WHERE id = $1", ids[0])
    new = await _create(manager, uid)
    assert await live_ids(pool, uid) == {ids[0], ids[2], ids[3], new}  # ids[1] was least recently active


async def test_headerless_logins_replace_each_other_before_browsers(authnz_sqlite):
    from collections import Counter

    from tldw_Server_API.app.core.AuthNZ.repos.sessions_repo import AuthnzSessionsRepo

    pool, uid, _ = authnz_sqlite
    manager = _manager(pool)
    for device in ("dev-aaaaaaaa", "dev-bbbbbbbb", "dev-cccccccc"):
        await _create(manager, uid, device)
    for _ in range(3):
        await _create(manager, uid)  # no install id
    rows = await AuthnzSessionsRepo(pool).get_active_sessions_for_user(uid)
    assert Counter(r["device_id"] for r in rows) == {
        "dev-aaaaaaaa": 1, "dev-bbbbbbbb": 1, "dev-cccccccc": 1, None: 1
    }


async def test_same_device_id_replaces_only_its_own_row(authnz_sqlite):
    from collections import Counter

    from tldw_Server_API.app.core.AuthNZ.repos.sessions_repo import AuthnzSessionsRepo

    pool, uid, _ = authnz_sqlite
    manager = _manager(pool)
    a, b, c = [await _create(manager, uid, d) for d in ("dev-aaaaaaaa", "dev-bbbbbbbb", "dev-cccccccc")]
    for _ in range(6):
        await _create(manager, uid, "dev-dddddddd")
    rows = await AuthnzSessionsRepo(pool).get_active_sessions_for_user(uid)
    assert Counter(r["device_id"] for r in rows) == {
        "dev-aaaaaaaa": 1, "dev-bbbbbbbb": 1, "dev-cccccccc": 1, "dev-dddddddd": 1
    }
    assert {a, b, c} <= await live_ids(pool, uid)


async def test_evicted_rows_returned_by_the_repo_carry_their_device_id(authnz_sqlite):
    from datetime import datetime, timedelta, timezone

    from tldw_Server_API.app.core.AuthNZ.repos.sessions_repo import AuthnzSessionsRepo

    pool, uid, _ = authnz_sqlite
    repo = AuthnzSessionsRepo(pool)
    now = datetime.now(timezone.utc)

    async def capped(tag, device):
        return await repo.create_session_record_with_cap(
            user_id=uid, token_hash=f"h-{tag}", refresh_token_hash=f"r-{tag}", encrypted_token="enc",
            encrypted_refresh="enc", expires_at=now + timedelta(hours=1), refresh_expires_at=now + timedelta(days=7),
            ip_address="127.0.0.1", user_agent="pytest", device_id=device, access_jti=f"aj-{tag}",
            refresh_jti=f"rj-{tag}", max_live_sessions=1,
        )

    await capped("one", "dev-aaaaaaaa")
    _new_id, evicted = await capped("two", "dev-bbbbbbbb")
    assert [e["device_id"] for e in evicted] == ["dev-aaaaaaaa"]


async def test_expired_refresh_rows_do_not_count_toward_the_cap(authnz_sqlite):
    pool, uid, _ = authnz_sqlite
    h, d = timedelta(hours=1), timedelta(days=1)
    for i in range(4):
        await _row(pool, uid, f"dead{i}", access_in=-3 * d, refresh_in=-2 * d)
    for i in range(3):
        await _row(pool, uid, f"live{i}", access_in=h, refresh_in=7 * d)
    await _create(_manager(pool), uid)
    assert len(await live_ids(pool, uid)) == 4
    assert not await _ids_with_reason(pool, "session_cap")


async def test_concurrent_creates_at_the_cap_never_exceed_it(authnz_sqlite):
    import asyncio

    pool, uid, _ = authnz_sqlite
    manager = _manager(pool)
    for _ in range(4):
        await _create(manager, uid)
    results = await asyncio.gather(*[_create(manager, uid) for _ in range(6)])
    assert len(results) == 6 and len(await live_ids(pool, uid)) == 4


async def test_cap_backlog_is_evicted_in_one_creation(authnz_sqlite):
    import time

    pool, uid, _ = authnz_sqlite
    now = datetime.now(timezone.utc)
    # 33,000 rows: more than SQLite's 32,766 bound-variable limit, so a per-victim IN list cannot work.
    async with pool.transaction() as conn:
        await conn.execute(
            """
            INSERT INTO sessions (
                user_id, token_hash, refresh_token_hash, encrypted_token, encrypted_refresh,
                expires_at, refresh_expires_at, device_id
            )
            WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 33000)
            SELECT $1, 'backlog-h-' || i, 'backlog-r-' || i, 'enc', 'enc', $2, $3,
                   'single-user-cookie:v1'
            FROM n
            """,
            uid,
            (now + timedelta(days=1)).isoformat(),
            (now + timedelta(days=1)).isoformat(),
        )
    assert len(await live_ids(pool, uid)) == 33000
    manager = _manager(pool)
    started = time.monotonic()
    new_id = await _create(manager, uid)
    elapsed = time.monotonic() - started
    print(f"sqlite backlog capped insert: {elapsed:.3f}s")
    assert elapsed < 5  # SQLite busy_timeout; a per-row loop cannot pass
    live = await live_ids(pool, uid)
    assert len(live) == 4 and new_id in live


async def test_cap_of_one_replaces_the_only_session(authnz_sqlite, monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    pool, uid, _ = authnz_sqlite
    monkeypatch.setenv("MAX_SESSIONS_PER_USER", "1")
    reset_settings()
    manager = _manager(pool)
    first = await _create(manager, uid)
    second = await _create(manager, uid)
    assert first != second
    assert await live_ids(pool, uid) == {second}


async def test_evicted_tokens_are_blacklisted_after_commit(authnz_sqlite, monkeypatch):
    pool, uid, _ = authnz_sqlite
    manager = _manager(pool)
    calls = []

    async def _record(*args, **kwargs):
        # Read on another connection: the eviction must already be committed.
        committed = [await _fetch(pool, s["id"]) for s in args[0]]
        calls.append(SimpleNamespace(args=args, kwargs=kwargs, committed=committed))

    monkeypatch.setattr(manager, "_blacklist_session_tokens", _record)
    ids = [await _create(manager, uid) for _ in range(5)]
    assert len(calls) == 1  # only the creation that evicted
    assert [s["id"] for s in calls[-1].args[0]] == [ids[0]]
    assert calls[-1].kwargs["reason"] == "session_cap"
    assert calls[-1].kwargs["revoked_by"] is None
    assert [c["revoke_reason"] for c in calls[-1].committed] == ["session_cap"]


async def test_inline_blacklist_is_bounded_to_the_most_recently_active(authnz_sqlite, monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import session_manager as sm

    pool, uid, _ = authnz_sqlite
    manager = _manager(pool)
    h, d = timedelta(hours=1), timedelta(days=1)
    for i in range(150):
        await _row(pool, uid, f"bulk{i}", access_in=h, refresh_in=7 * d)
    async with pool.transaction() as conn:  # higher id = less recently active
        await conn.execute(
            "UPDATE sessions SET last_activity = datetime('now', '-' || id || ' minutes') WHERE user_id = $1",
            uid,
        )
    calls = []

    async def _record(*args, **kwargs):
        calls.append(args[0])

    monkeypatch.setattr(manager, "_blacklist_session_tokens", _record)
    await _create(manager, uid)

    evicted = sorted(await _ids_with_reason(pool, "session_cap"))
    assert len(evicted) == 147 and sm.SESSION_CAP_INLINE_BLACKLIST_LIMIT == 100
    assert len(calls) == 1 and len(calls[0]) == 100
    # the 3 most recent rows survive; of the evicted, the lowest ids are the most recently active
    assert {s["id"] for s in calls[0]} == set(evicted[:100])


async def test_skipped_evicted_session_is_still_rejected_by_validation(authnz_sqlite, monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import session_manager as sm
    from tldw_Server_API.app.core.AuthNZ.exceptions import SessionRevokedException

    pool, uid, _ = authnz_sqlite
    manager = _manager(pool)
    monkeypatch.setattr(sm, "SESSION_CAP_INLINE_BLACKLIST_LIMIT", 0)
    first = await manager.create_session(
        user_id=uid, access_token="temp_access_skipped", refresh_token="temp_refresh_skipped"
    )
    for _ in range(4):
        await _create(manager, uid)
    assert (await _fetch(pool, int(first["session_id"])))["revoke_reason"] == "session_cap"
    try:  # the request path treats a revoked row as "no valid session", raised or None
        assert await manager.validate_session("temp_access_skipped", raise_on_error=True) is None
    except SessionRevokedException:
        pass


# --- why a revoked token was revoked -----------------------------------------


async def test_revoke_reason_falls_back_to_the_session_row(authnz_sqlite):
    pool, uid, _ = authnz_sqlite
    manager = _manager(pool)
    await manager.create_session(
        user_id=uid, access_token="temp_access_first", refresh_token="temp_refresh_first"
    )
    for _ in range(4):
        await _create(manager, uid)
    assert await manager.get_revoke_reason("temp_access_first") == "session_cap"
    assert await manager.get_revoke_reason("temp_refresh_first") == "session_cap"


async def test_revoke_reason_reports_other_reasons_and_is_none_on_unknown_or_error(authnz_sqlite, monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.repos.sessions_repo import AuthnzSessionsRepo

    pool, uid, _ = authnz_sqlite
    manager = _manager(pool)
    result = await manager.create_session(
        user_id=uid, access_token="temp_access_out", refresh_token="temp_refresh_out"
    )
    await _revoke(pool, int(result["session_id"]), "User logout")
    assert await manager.get_revoke_reason("temp_access_out") == "User logout"
    assert await manager.get_revoke_reason("unknown-token") is None

    async def _boom(*_a, **_k):
        raise RuntimeError("db down")

    monkeypatch.setattr(AuthnzSessionsRepo, "fetch_revoke_reason_for_token_hash_candidates", _boom)
    assert await manager.get_revoke_reason("temp_access_out") is None
    assert await manager.get_revoke_reason("") is None
