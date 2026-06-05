import asyncio
import importlib
import json
import os
import shutil
from typing import Optional
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from apscheduler.triggers.cron import CronTrigger
from zoneinfo import ZoneInfo

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.Collections.reading_digest_jobs import (
    READING_DIGEST_DOMAIN,
    READING_DIGEST_JOB_TYPE,
    _normalize_suggestions_config,
    _score_suggestion_candidate,
    handle_reading_digest_job,
    reading_digest_queue,
)
from tldw_Server_API.app.services.reading_digest_scheduler import _ReadingDigestScheduler
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.Collections_DB import ContentItemRow
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.config import settings


pytestmark = pytest.mark.unit


@pytest.fixture()
def client_with_user(monkeypatch):
    async def override_user():
        return User(id=222, username="reader", email=None, is_active=True)

    monkeypatch.setenv("MINIMAL_TEST_APP", "0")
    monkeypatch.setenv("ULTRA_MINIMAL_APP", "0")
    monkeypatch.setenv("ROUTES_ENABLE", "reading")

    base_dir = Path.cwd() / "Databases" / "test_reading_digest"
    shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    prev_base_dir = settings.get("USER_DB_BASE_DIR")
    settings.USER_DB_BASE_DIR = str(base_dir)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    jobs_db_path = base_dir / "jobs.db"
    monkeypatch.setenv("JOBS_DB_PATH", str(jobs_db_path))
    os.environ["JOBS_DB_PATH"] = str(jobs_db_path)

    from tldw_Server_API.app import main as app_main

    importlib.reload(app_main)
    fastapi_app = app_main.app

    fastapi_app.dependency_overrides[get_request_user] = override_user
    try:
        with TestClient(fastapi_app) as client:
            yield client
    finally:
        fastapi_app.dependency_overrides.clear()
        if prev_base_dir is not None:
            settings.USER_DB_BASE_DIR = prev_base_dir
        else:
            try:
                del settings.USER_DB_BASE_DIR
            except AttributeError:
                pass
        os.environ.pop("JOBS_DB_PATH", None)


def _run_digest_job(job_id: int) -> None:
    async def _runner() -> None:
        jm = JobManager()
        job = jm.get_job(job_id)
        assert job is not None
        result = await handle_reading_digest_job(job)
        jm.complete_job(job_id, result=result, enforce=False)

    asyncio.run(_runner())


def _make_content_item_row(
    item_id: int,
    *,
    status: str,
    tags: list[str],
    favorite: bool,
    updated_at: str,
    created_at: Optional[str] = None,
) -> ContentItemRow:
    return ContentItemRow(
        id=item_id,
        user_id="222",
        origin="reading",
        origin_type=None,
        origin_id=None,
        url="https://example.com",
        canonical_url="https://example.com",
        domain="example.com",
        title=f"Item {item_id}",
        summary=None,
        notes=None,
        content_hash=None,
        word_count=1200,
        published_at=None,
        status=status,
        favorite=favorite,
        metadata_json=None,
        media_id=None,
        job_id=None,
        run_id=None,
        source_id=None,
        read_at=None,
        created_at=created_at or updated_at,
        updated_at=updated_at,
        tags=tags,
    )


def test_reading_digest_suggestion_scoring():
    now = datetime(2026, 1, 20, 8, 0, tzinfo=timezone.utc)
    recent = _make_content_item_row(
        1,
        status="reading",
        tags=["ai"],
        favorite=True,
        updated_at=now.isoformat(),
    )
    old = _make_content_item_row(
        2,
        status="saved",
        tags=[],
        favorite=False,
        updated_at=(now - timedelta(days=45)).isoformat(),
    )
    score_recent, reasons = _score_suggestion_candidate(recent, {"ai"}, now)
    score_old, _ = _score_suggestion_candidate(old, {"ai"}, now)
    assert score_recent > score_old
    assert "favorite" in reasons


def test_reading_digest_suggestions_flags_accept_y():
    cfg = _normalize_suggestions_config(
        {
            "suggestions": {
                "enabled": "y",
                "include_read": "y",
                "include_archived": "y",
                "limit": 2,
                "status": ["saved"],
            }
        }
    )

    assert cfg is not None
    assert cfg["enabled"] is True
    assert cfg["include_read"] is True
    assert cfg["include_archived"] is True
    assert "read" in cfg["status"]
    assert "archived" in cfg["status"]


def test_reading_digest_schedule_crud(client_with_user):
    client = client_with_user
    payload = {
        "name": "Morning Digest",
        "cron": "0 8 * * *",
        "timezone": "UTC",
        "format": "md",
        "filters": {"status": ["saved"], "tags": ["ai"], "limit": 5},
    }
    create_resp = client.post("/api/v1/reading/digests/schedules", json=payload)
    assert create_resp.status_code == 201, create_resp.text
    schedule_id = create_resp.json()["id"]

    list_resp = client.get("/api/v1/reading/digests/schedules")
    assert list_resp.status_code == 200
    schedules = list_resp.json()
    assert any(row["id"] == schedule_id for row in schedules)

    get_resp = client.get(f"/api/v1/reading/digests/schedules/{schedule_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["name"] == "Morning Digest"

    patch_resp = client.patch(
        f"/api/v1/reading/digests/schedules/{schedule_id}",
        json={"name": "Updated Digest", "enabled": False},
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["name"] == "Updated Digest"
    assert patch_resp.json()["enabled"] is False

    delete_resp = client.delete(f"/api/v1/reading/digests/schedules/{schedule_id}")
    assert delete_resp.status_code == 200
    assert delete_resp.json()["ok"] is True


def test_reading_digest_job_creates_output(client_with_user):
    client = client_with_user
    save_resp = client.post(
        "/api/v1/reading/save",
        json={"url": "https://example.com/article", "title": "Example", "content": "Body"},
    )
    assert save_resp.status_code == 200, save_resp.text

    db = CollectionsDatabase.for_user(user_id=222)
    schedule_id = "digest_test"
    db.create_reading_digest_schedule(
        id=schedule_id,
        tenant_id="default",
        name="Test Digest",
        cron="0 8 * * *",
        timezone="UTC",
        enabled=True,
        require_online=False,
        filters={"status": ["saved"], "limit": 10},
        template_id=None,
        template_name=None,
        format="md",
        retention_days=7,
    )

    jm = JobManager()
    job = jm.create_job(
        domain=READING_DIGEST_DOMAIN,
        queue=reading_digest_queue(),
        job_type=READING_DIGEST_JOB_TYPE,
        payload={"schedule_id": schedule_id, "user_id": 222},
        owner_user_id=222,
    )
    _run_digest_job(job["id"])

    outputs, total = db.list_output_artifacts(type_="reading_digest", limit=10, offset=0)
    assert total == 1
    row = outputs[0]
    meta = json.loads(row.metadata_json or "{}")
    assert meta.get("schedule_id") == schedule_id
    output_path = DatabasePaths.get_user_outputs_dir(222) / row.storage_path
    assert output_path.exists()


def test_reading_digest_job_includes_suggestions(client_with_user):
    client = client_with_user
    digest_a = client.post(
        "/api/v1/reading/save",
        json={
            "url": "https://example.com/a",
            "title": "Digest A",
            "tags": ["ai"],
            "status": "saved",
            "content": "Digest A body",
        },
    ).json()
    digest_b = client.post(
        "/api/v1/reading/save",
        json={
            "url": "https://example.com/b",
            "title": "Digest B",
            "tags": ["ai"],
            "status": "saved",
            "content": "Digest B body",
        },
    ).json()
    assert digest_a["id"] != digest_b["id"]

    suggestion_a = client.post(
        "/api/v1/reading/save",
        json={
            "url": "https://example.com/c",
            "title": "Suggestion A",
            "tags": ["ai"],
            "status": "reading",
            "content": "Suggestion A body",
        },
    ).json()
    suggestion_b = client.post(
        "/api/v1/reading/save",
        json={
            "url": "https://example.com/d",
            "title": "Suggestion B",
            "tags": ["ai"],
            "status": "reading",
            "content": "Suggestion B body",
        },
    ).json()
    suggestion_c = client.post(
        "/api/v1/reading/save",
        json={
            "url": "https://example.com/e",
            "title": "Suggestion C",
            "tags": ["misc"],
            "status": "reading",
            "content": "Suggestion C body",
        },
    ).json()

    db = CollectionsDatabase.for_user(user_id=222)
    template = db.create_output_template(
        name="Digest Suggestions Template",
        type_="newsletter_markdown",
        format_="md",
        body=(
            "# {{ title }}\n\n"
            "Items:\n"
            "{% for item in items %}- {{ item.title }}\n{% endfor %}\n\n"
            "Suggestions:\n"
            "{% for item in suggestions %}- {{ item.title }}\n{% endfor %}\n"
        ),
        description=None,
        is_default=False,
    )

    schedule_id = "digest_suggestions"
    db.create_reading_digest_schedule(
        id=schedule_id,
        tenant_id="default",
        name="Digest Suggestions",
        cron="0 8 * * *",
        timezone="UTC",
        enabled=True,
        require_online=False,
        filters={
            "status": ["saved"],
            "tags": ["ai"],
            "limit": 10,
            "suggestions": {"enabled": True, "limit": 2, "status": ["reading"]},
        },
        template_id=template.id,
        template_name=None,
        format="md",
        retention_days=7,
    )

    jm = JobManager()
    job = jm.create_job(
        domain=READING_DIGEST_DOMAIN,
        queue=reading_digest_queue(),
        job_type=READING_DIGEST_JOB_TYPE,
        payload={"schedule_id": schedule_id, "user_id": 222},
        owner_user_id=222,
    )
    _run_digest_job(job["id"])

    outputs, total = db.list_output_artifacts(type_="reading_digest", limit=10, offset=0)
    assert total == 1
    row = outputs[0]
    meta = json.loads(row.metadata_json or "{}")
    assert meta.get("suggestions_count") == 2
    assert set(meta.get("suggestions_item_ids") or []) == {suggestion_a["id"], suggestion_b["id"]}
    assert meta.get("suggestions_config", {}).get("enabled") is True

    output_path = DatabasePaths.get_user_outputs_dir(222) / row.storage_path
    content = output_path.read_text(encoding="utf-8")
    assert "Suggestions:" in content
    assert suggestion_a["title"] in content
    assert suggestion_b["title"] in content
    assert suggestion_c["title"] not in content


def test_reading_digest_scheduler_claims_single_enqueue(client_with_user):
    db = CollectionsDatabase.for_user(user_id=222)
    schedule_id = "digest_claim"
    db.create_reading_digest_schedule(
        id=schedule_id,
        tenant_id="default",
        name="Claim Digest",
        cron="*/5 * * * *",
        timezone="UTC",
        enabled=True,
        require_online=False,
        filters={"status": ["saved"], "limit": 10},
        template_id=None,
        template_name=None,
        format="md",
        retention_days=7,
    )
    tz = ZoneInfo("UTC")
    trigger = CronTrigger.from_crontab("*/5 * * * *", timezone=tz)
    scheduled_dt = trigger.get_next_fire_time(None, datetime.now(tz))
    assert scheduled_dt is not None
    db.set_reading_digest_history(schedule_id, next_run_at=scheduled_dt.isoformat())

    async def _runner() -> None:
        sched_a = _ReadingDigestScheduler()
        sched_b = _ReadingDigestScheduler()
        await asyncio.gather(
            sched_a._run_schedule(schedule_id, user_id=222),
            sched_b._run_schedule(schedule_id, user_id=222),
        )

    asyncio.run(_runner())

    jm = JobManager()
    jobs = jm.list_jobs(domain=READING_DIGEST_DOMAIN, job_type=READING_DIGEST_JOB_TYPE)
    assert len(jobs) == 1
    updated = db.get_reading_digest_schedule(schedule_id)
    assert updated.next_run_at is not None
    updated_dt = datetime.fromisoformat(updated.next_run_at)
    if updated_dt.tzinfo is None:
        updated_dt = updated_dt.replace(tzinfo=timezone.utc)
    if scheduled_dt.tzinfo is None:
        scheduled_dt = scheduled_dt.replace(tzinfo=timezone.utc)
    assert updated_dt > scheduled_dt


@pytest.mark.parametrize(
    ("tz_name", "cron"),
    [
        ("UTC", "0 8 * * *"),
        ("America/New_York", "0 8 * * *"),
    ],
)
def test_reading_digest_next_run_at_timezone(client_with_user, tz_name, cron):
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        pytest.skip(f"Timezone {tz_name} not available in test environment")
    db = CollectionsDatabase.for_user(user_id=222)
    schedule_id = f"digest_tz_{tz_name.replace('/', '_')}"
    db.create_reading_digest_schedule(
        id=schedule_id,
        tenant_id="default",
        name="TZ Digest",
        cron=cron,
        timezone=tz_name,
        enabled=True,
        require_online=False,
        filters={"status": ["saved"], "limit": 5},
        template_id=None,
        template_name=None,
        format="md",
        retention_days=7,
    )
    trigger = CronTrigger.from_crontab(cron, timezone=tz)
    scheduled_dt = trigger.get_next_fire_time(None, datetime.now(tz))
    assert scheduled_dt is not None
    db.set_reading_digest_history(schedule_id, next_run_at=scheduled_dt.isoformat())

    async def _runner() -> None:
        sched = _ReadingDigestScheduler()
        await sched._run_schedule(schedule_id, user_id=222)

    asyncio.run(_runner())

    updated = db.get_reading_digest_schedule(schedule_id)
    assert updated.next_run_at is not None
    updated_dt = datetime.fromisoformat(updated.next_run_at)
    if updated_dt.tzinfo is None:
        pytest.fail("next_run_at missing timezone info")
    if tz_name != "UTC":
        assert updated_dt.utcoffset() is not None
        assert updated_dt.utcoffset().total_seconds() != 0
    assert updated_dt > scheduled_dt


def test_reading_digest_scheduler_skips_disabled_after_claim(client_with_user, monkeypatch):
    db = CollectionsDatabase.for_user(user_id=222)
    schedule_id = "digest_disable_after_claim"
    db.create_reading_digest_schedule(
        id=schedule_id,
        tenant_id="default",
        name="Disable Digest",
        cron="*/15 * * * *",
        timezone="UTC",
        enabled=True,
        require_online=False,
        filters={"status": ["saved"], "limit": 5},
        template_id=None,
        template_name=None,
        format="md",
        retention_days=7,
    )
    tz = ZoneInfo("UTC")
    trigger = CronTrigger.from_crontab("*/15 * * * *", timezone=tz)
    scheduled_dt = trigger.get_next_fire_time(None, datetime.now(tz))
    assert scheduled_dt is not None
    db.set_reading_digest_history(schedule_id, next_run_at=scheduled_dt.isoformat())

    original_claim = CollectionsDatabase.try_claim_reading_digest_run

    def _patched_claim(self, schedule_id: str, **kwargs):
        ok = original_claim(self, schedule_id, **kwargs)
        if ok:
            self.update_reading_digest_schedule(schedule_id, {"enabled": False})
        return ok

    monkeypatch.setattr(CollectionsDatabase, "try_claim_reading_digest_run", _patched_claim)

    async def _runner() -> None:
        sched = _ReadingDigestScheduler()
        await sched._run_schedule(schedule_id, user_id=222)

    asyncio.run(_runner())

    jm = JobManager()
    jobs = [
        job
        for job in jm.list_jobs(domain=READING_DIGEST_DOMAIN, job_type=READING_DIGEST_JOB_TYPE)
        if (job.get("payload") or {}).get("schedule_id") == schedule_id
    ]
    assert jobs == []
    updated = db.get_reading_digest_schedule(schedule_id)
    assert updated.enabled is False
    assert updated.last_status == "skipped_disabled"
