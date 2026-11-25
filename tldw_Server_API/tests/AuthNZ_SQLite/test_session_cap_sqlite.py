"""Live-session predicate: listing, cleanup and the callers that gate on it (SQLite)."""

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
    captured: dict[str, Any] = {}
    history: list[dict[str, Any]] = []

    class _StubDB:
        def get_schedule(self, schedule_id):
            return schedule

        def set_history(self, _sid, **kwargs):
            history.append(kwargs)

    class _StubScheduler:
        def __init__(self):
            self.worker_pool = self

        def get_status(self):
            return {"workers_by_queue": {"workflows": 1}}

        async def submit(self, *args, **kwargs):
            captured["payload"] = kwargs.get("payload")
            return "task-1"

    svc = mod._WFRecurringScheduler()
    monkeypatch.setattr(svc, "_get_db", lambda _uid: _StubDB())
    svc._core_scheduler = _StubScheduler()  # type: ignore[attr-defined]

    await svc._run_schedule("s1", user_id=uid)  # type: ignore[attr-defined]

    assert ("payload" in captured) is expect_run
    assert (history[-1].get("last_status") == "skipped_offline") is (not expect_run)


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
