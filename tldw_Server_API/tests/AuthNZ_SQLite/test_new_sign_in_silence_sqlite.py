"""Only an older browser session may silence a new-sign-in alert (TASK-15649)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.AuthNZ import interactive_session
from tldw_Server_API.app.core.AuthNZ.sign_in_alert import caller_may_silence, drain_pending_alerts
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase, UserNotificationRow
from tldw_Server_API.app.services.notifications_prune_service import DEFAULT_RETENTION_DAYS
from tldw_Server_API.tests.AuthNZ_SQLite._sessions_http_fixtures import (
    api_key_for,
    bearer,
    login,
    session_id_of,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.sign_in_alerts]


def item(client, tokens, notification_id: int) -> dict:
    body = client.get("/api/v1/notifications?include_archived=true", headers=bearer(tokens)).json()
    return next(i for i in body["items"] if i["id"] == notification_id)


def backdate(ctx, notification_id: int, hours: int) -> None:
    iso = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with CollectionsDatabase.for_user(user_id=ctx.alice_id) as db:
        db.backend.execute("UPDATE user_notifications SET created_at = ? WHERE id = ?", (iso, notification_id))


def _other(ctx):
    with CollectionsDatabase.for_user(user_id=ctx.alice_id) as db:
        return db.create_user_notification(kind="job_completed", title="Done", message="m", severity="info")


async def _setup(ctx):
    """Alice signs in three times (A, B, C); returns tokens and the alert ids for B and C."""
    client, pw = ctx.client, ctx.password
    A = login(client, "alice", pw, device="dev-aaaaaaaa")
    B = login(client, "alice", pw, device="dev-bbbbbbbb")
    C = login(client, "alice", pw, device="dev-cccccccc")
    await drain_pending_alerts()
    items = client.get("/api/v1/notifications?kind=new_sign_in", headers=bearer(A)).json()["items"]
    by_source = {i["source_job_id"]: i["id"] for i in items}
    return A, B, C, by_source[str(session_id_of(B))], by_source[str(session_id_of(C))]


async def test_older_session_can_mark_read_and_dismiss(two_user_client):
    ctx = two_user_client
    client = ctx.client
    A, B, C, nB, nC = await _setup(ctx)
    assert client.post(f"/api/v1/notifications/{nB}/dismiss", headers=bearer(A)).json() == {"dismissed": True}
    assert client.post("/api/v1/notifications/mark-read", json={"ids": [nC]}, headers=bearer(A)).json()["updated"] == 1


async def test_source_session_cannot_dismiss_its_own_alert(two_user_client):
    ctx = two_user_client
    client = ctx.client
    A, B, C, nB, nC = await _setup(ctx)
    r = client.post(f"/api/v1/notifications/{nB}/dismiss", headers=bearer(B))
    assert r.status_code == 403 and r.json()["detail"]["code"] == "new_sign_in_protected"
    assert item(client, A, nB)["dismissed_at"] is None


async def test_later_session_cannot_mark_read_or_snooze(two_user_client):
    ctx = two_user_client
    client = ctx.client
    A, B, C, nB, nC = await _setup(ctx)
    r = client.post("/api/v1/notifications/mark-read", json={"ids": [nB]}, headers=bearer(C))
    assert r.status_code == 403 and r.json()["detail"]["code"] == "new_sign_in_protected"
    assert item(client, A, nB)["read_at"] is None
    s = client.post(f"/api/v1/notifications/{nB}/snooze", json={"minutes": 30}, headers=bearer(C))
    assert s.status_code == 403 and s.json()["detail"]["code"] == "new_sign_in_protected"


async def test_api_key_cannot_silence_even_an_old_alert(two_user_client):
    ctx = two_user_client
    client = ctx.client
    A, B, C, nB, nC = await _setup(ctx)
    backdate(ctx, nB, hours=25)
    r = client.post(f"/api/v1/notifications/{nB}/dismiss", headers={"X-API-KEY": api_key_for(client, A)})
    assert r.status_code == 403 and r.json()["detail"]["code"] == "new_sign_in_protected"
    assert item(client, A, nB)["dismissed_at"] is None


async def test_after_24_hours_any_browser_session_may_dismiss(two_user_client):
    ctx = two_user_client
    client = ctx.client
    A, B, C, nB, nC = await _setup(ctx)
    assert client.post(f"/api/v1/notifications/{nB}/dismiss", headers=bearer(C)).status_code == 403  # new
    backdate(ctx, nB, hours=25)
    assert client.post(f"/api/v1/notifications/{nB}/dismiss", headers=bearer(C)).json() == {"dismissed": True}


async def test_other_kinds_are_unaffected(two_user_client):
    ctx = two_user_client
    client = ctx.client
    A, B, C, nB, nC = await _setup(ctx)
    other = _other(ctx)
    r = client.post(f"/api/v1/notifications/{other.id}/dismiss", headers={"X-API-KEY": api_key_for(client, A)})
    assert r.json() == {"dismissed": True}


async def test_user_cannot_dismiss_another_users_alert(two_user_client):
    ctx = two_user_client
    client = ctx.client
    A, B, C, nB, nC = await _setup(ctx)
    bob = login(client, "bob", ctx.password)
    assert client.post(f"/api/v1/notifications/{nB}/dismiss", headers=bearer(bob)).json() == {"dismissed": False}
    assert item(client, A, nB)["dismissed_at"] is None


async def test_preferences_expose_no_switch_for_the_alert(two_user_client):
    ctx = two_user_client
    A = login(ctx.client, "alice", ctx.password, device="dev-aaaaaaaa")
    keys = set(ctx.client.get("/api/v1/notifications/preferences", headers=bearer(A)).json())
    assert keys == {"user_id", "reminder_enabled", "job_completed_enabled", "job_failed_enabled", "updated_at"}


async def test_mixed_mark_read_updates_the_allowed_ids_and_reports_the_refused(two_user_client):
    ctx = two_user_client
    client = ctx.client
    A, B, C, nB, nC = await _setup(ctx)
    other = _other(ctx)
    r = client.post("/api/v1/notifications/mark-read", json={"ids": [nB, other.id]}, headers=bearer(C))
    assert r.json() == {"updated": 1, "refused_ids": [nB]}
    assert item(client, A, nB)["read_at"] is None and item(client, A, other.id)["read_at"] is not None


async def test_list_filters_by_kind_and_reports_the_filtered_total(two_user_client):
    ctx = two_user_client
    client = ctx.client
    A, B, C, nB, nC = await _setup(ctx)
    with CollectionsDatabase.for_user(user_id=ctx.alice_id) as db:
        for _ in range(3):
            db.create_user_notification(kind="job_completed", title="Done", message="m", severity="info")
    body = client.get("/api/v1/notifications?kind=new_sign_in&limit=1", headers=bearer(A)).json()
    assert [i["kind"] for i in body["items"]] == ["new_sign_in"] and body["total"] == 2
    assert client.get("/api/v1/notifications?kind=not_a_kind", headers=bearer(A)).status_code == 422


def test_new_sign_in_alerts_expire_after_30_days():
    assert DEFAULT_RETENTION_DAYS["new_sign_in"] == 30


@pytest.mark.parametrize(
    "created_at,caller,expected",
    [
        ("garbage", 9, False),
        ("", 9, False),
        ("2026-11-23T00:00:00+00:00", 9, True),
        ("2026-11-25T11:00:00+00:00", 9, False),
        ("2026-11-25T11:00:00+00:00", 3, True),
        ("2026-11-25T11:00:00+00:00", None, False),
    ],
)
def test_caller_may_silence_rule(created_at, caller, expected):
    now = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)
    row = UserNotificationRow(
        id=1, user_id="1", kind="new_sign_in", title="t", message="m", severity="warning",
        source_task_id=None, source_task_run_id=None, source_job_id="7", source_domain="auth_session",
        source_job_type=None, link_type=None, link_id=None, link_url=None, dedupe_key=None,
        retention_until=None, archived_at=None, created_at=created_at, read_at=None, dismissed_at=None,
    )
    assert caller_may_silence(row, caller, now) is expected


async def test_other_kinds_never_resolve_the_session_manager(two_user_client, monkeypatch):
    ctx = two_user_client
    client = ctx.client
    A = login(client, "alice", ctx.password, device="dev-aaaaaaaa")

    async def boom(*_a, **_k):
        raise AssertionError("caller resolved for a kind that needs no check")

    monkeypatch.setattr(interactive_session, "resolve_current_session_id", boom)
    other = _other(ctx)
    assert client.post(f"/api/v1/notifications/{other.id}/dismiss", headers=bearer(A)).json() == {"dismissed": True}
