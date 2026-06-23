from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterator

import pytest

from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Personalization.companion_user_ids import (
    resolve_companion_storage_user_id,
)
from tldw_Server_API.app.core.Personalization.companion_reflection_jobs import (
    COMPANION_REBUILD_JOB_TYPE,
    handle_companion_reflection_job,
    run_companion_reflection_job,
)


pytestmark = pytest.mark.unit


@pytest.fixture()
def companion_reflection_env(monkeypatch, tmp_path) -> Iterator[Path]:
    base_dir = tmp_path / "test_companion_reflection_jobs"
    base_dir.mkdir(parents=True, exist_ok=True)
    prev_base_dir = settings.get("USER_DB_BASE_DIR")
    settings.USER_DB_BASE_DIR = str(base_dir)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    try:
        yield Path(base_dir)
    finally:
        if prev_base_dir is not None:
            settings.USER_DB_BASE_DIR = prev_base_dir
        else:
            try:
                del settings.USER_DB_BASE_DIR
            except AttributeError:
                pass


def _legacy_storage_user_id(user_id: str) -> str:
    return resolve_companion_storage_user_id(user_id)


def _seed_companion_context(user_id: str) -> tuple[PersonalizationDB, CollectionsDatabase]:
    personalization_db = PersonalizationDB(str(DatabasePaths.get_personalization_db_path(user_id)))
    collections_db = CollectionsDatabase.for_user(user_id=int(user_id))
    personalization_db.update_profile(user_id, enabled=1)
    personalization_db.insert_companion_activity_event(
        user_id=user_id,
        event_type="reading_item_saved",
        source_type="reading_item",
        source_id="101",
        surface="reading",
        dedupe_key="reading_item_saved:101",
        tags=["project-alpha", "research"],
        provenance={"capture_mode": "explicit", "source_ids": ["101"]},
        metadata={"title": "Alpha kickoff"},
    )
    personalization_db.insert_companion_activity_event(
        user_id=user_id,
        event_type="watchlist_source_created",
        source_type="watchlist_source",
        source_id="202",
        surface="watchlists",
        dedupe_key="watchlist_source_created:202",
        tags=["project-alpha", "monitoring"],
        provenance={"capture_mode": "explicit", "source_ids": ["202"]},
        metadata={"title": "Alpha monitoring feed"},
    )
    personalization_db.upsert_companion_knowledge_card(
        user_id=user_id,
        card_type="project_focus",
        title="Current focus",
        summary="Recent explicit activity clusters around 'project-alpha'.",
        evidence=[{"source_id": "101"}, {"source_id": "202"}],
        score=0.9,
    )
    return personalization_db, collections_db


def test_companion_reflection_job_creates_notification_and_persists_reflection(companion_reflection_env) -> None:
    personalization_db, collections_db = _seed_companion_context("1")

    result = run_companion_reflection_job(
        user_id="1",
        cadence="daily",
        job_id="501",
        now=datetime(2026, 3, 10, 15, 0, tzinfo=timezone.utc),
        personalization_db=personalization_db,
        collections_db=collections_db,
    )

    assert result["status"] == "completed"
    assert result["reflection_id"]

    rows, _total = personalization_db.list_companion_activity_events("1", limit=20, offset=0)
    reflection = next(row for row in rows if row["id"] == result["reflection_id"])
    assert reflection["event_type"] == "companion_reflection_generated"
    assert reflection["source_type"] == "companion_reflection"
    assert reflection["metadata"]["cadence"] == "daily"
    assert reflection["metadata"]["summary"]
    assert reflection["provenance"]["signal_count"] >= 2

    notifications = collections_db.list_user_notifications(limit=10, offset=0)
    assert len(notifications) == 1
    assert notifications[0].kind == "companion_reflection"
    assert notifications[0].link_type == "companion_reflection"
    assert notifications[0].link_id == result["reflection_id"]
    assert notifications[0].source_job_id == "501"


def test_companion_reflection_job_includes_goal_and_stale_signals(companion_reflection_env) -> None:
    personalization_db, collections_db = _seed_companion_context("1")
    goal_id = personalization_db.create_companion_goal(
        user_id="1",
        title="Resume alpha review",
        description="Return to the alpha review backlog.",
        goal_type="manual",
        config={},
        progress={"percent": 40},
        origin_kind="manual",
        progress_mode="computed",
        evidence=[{"source_id": "101"}],
        status="active",
    )
    personalization_db.upsert_companion_knowledge_card(
        user_id="1",
        card_type="stale_followup",
        title="Stale follow-up",
        summary="No fresh explicit activity has touched 'project-alpha' this week.",
        evidence=[{"source_id": "101"}, {"source_id": "202"}],
        score=0.8,
    )

    result = run_companion_reflection_job(
        user_id="1",
        cadence="daily",
        now=datetime(2026, 3, 10, 15, 0, tzinfo=timezone.utc),
        personalization_db=personalization_db,
        collections_db=collections_db,
    )

    assert result["status"] == "completed"

    rows, _total = personalization_db.list_companion_activity_events("1", limit=20, offset=0)
    reflection = next(row for row in rows if row["id"] == result["reflection_id"])
    assert any(
        item["kind"] == "knowledge_card" and item.get("card_type") == "stale_followup"
        for item in reflection["metadata"]["evidence"]
    )
    assert any(
        item["kind"] == "goal" and item.get("goal_id") == goal_id
        for item in reflection["metadata"]["evidence"]
    )
    assert goal_id in reflection["provenance"]["goal_ids"]


def test_companion_reflection_job_persists_delivery_metadata_and_prompts(
    companion_reflection_env,
) -> None:
    personalization_db, collections_db = _seed_companion_context("1")

    result = run_companion_reflection_job(
        user_id="1",
        cadence="daily",
        now=datetime(2026, 3, 10, 15, 0, tzinfo=timezone.utc),
        personalization_db=personalization_db,
        collections_db=collections_db,
    )

    assert result["status"] == "completed"

    rows, _total = personalization_db.list_companion_activity_events("1", limit=20, offset=0)
    reflection = next(row for row in rows if row["id"] == result["reflection_id"])
    assert reflection["metadata"]["delivery_decision"] == "delivered"
    assert reflection["metadata"]["theme_key"]
    assert reflection["metadata"]["signal_strength"] >= 1
    assert reflection["metadata"]["follow_up_prompts"]


def test_companion_reflection_job_skips_when_quiet_hours_active(companion_reflection_env) -> None:
    personalization_db, collections_db = _seed_companion_context("1")
    personalization_db.update_profile(
        "1",
        quiet_hours_start="01:00",
        quiet_hours_end="02:00",
    )

    result = run_companion_reflection_job(
        user_id="1",
        cadence="daily",
        now=datetime(2026, 3, 10, 1, 30, tzinfo=timezone.utc),
        personalization_db=personalization_db,
        collections_db=collections_db,
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "quiet_hours"

    rows, _total = personalization_db.list_companion_activity_events("1", limit=20, offset=0)
    assert not any(row["event_type"] == "companion_reflection_generated" for row in rows)
    assert collections_db.list_user_notifications(limit=10, offset=0) == []


@pytest.mark.parametrize(
    ("cadence", "profile_fields", "reason"),
    [
        ("daily", {"companion_reflections_enabled": 0}, "companion_reflections_disabled"),
        ("daily", {"companion_daily_reflections_enabled": 0}, "daily_reflections_disabled"),
        ("weekly", {"companion_weekly_reflections_enabled": 0}, "weekly_reflections_disabled"),
    ],
)
def test_companion_reflection_job_honors_reflection_preferences(
    companion_reflection_env,
    cadence: str,
    profile_fields: dict[str, int],
    reason: str,
) -> None:
    personalization_db, collections_db = _seed_companion_context("1")
    personalization_db.update_profile("1", **profile_fields)

    result = run_companion_reflection_job(
        user_id="1",
        cadence=cadence,
        now=datetime(2026, 3, 10, 15, 0, tzinfo=timezone.utc),
        personalization_db=personalization_db,
        collections_db=collections_db,
    )

    assert result["status"] == "skipped"
    assert result["reason"] == reason
    assert collections_db.list_user_notifications(limit=10, offset=0) == []


def test_companion_reflection_job_reuses_existing_reflection_outside_recent_window(
    companion_reflection_env,
) -> None:
    personalization_db = PersonalizationDB(str(DatabasePaths.get_personalization_db_path("1")))
    collections_db = CollectionsDatabase.for_user(user_id=1)
    personalization_db.update_profile("1", enabled=1)

    reflection_id = personalization_db.insert_companion_activity_event(
        user_id="1",
        event_type="companion_reflection_generated",
        source_type="companion_reflection",
        source_id="2026-03-10",
        surface="jobs.companion",
        dedupe_key="companion.reflection:daily:2026-03-10",
        provenance={"capture_mode": "explicit"},
        metadata={"title": "Daily reflection", "summary": "Existing reflection"},
    )
    for index in range(205):
        personalization_db.insert_companion_activity_event(
            user_id="1",
            event_type="reading_item_saved",
            source_type="reading_item",
            source_id=f"reading-{index}",
            surface="api.reading",
            dedupe_key=f"reading.save:{index}",
            tags=["project-alpha"],
            provenance={"capture_mode": "explicit"},
            metadata={"title": f"Reading item {index}"},
        )

    result = run_companion_reflection_job(
        user_id="1",
        cadence="daily",
        now=datetime(2026, 3, 10, 16, 0, tzinfo=timezone.utc),
        personalization_db=personalization_db,
        collections_db=collections_db,
    )

    assert result["status"] == "completed"
    assert result["reflection_id"] == reflection_id
    rows, total = personalization_db.list_companion_activity_events("1", limit=500, offset=0)
    reflections = [row for row in rows if row["event_type"] == "companion_reflection_generated"]
    assert total == 206
    assert len(reflections) == 1


def test_companion_reflection_job_normalizes_cadence_before_deduping(
    companion_reflection_env,
) -> None:
    personalization_db, collections_db = _seed_companion_context("1")
    now = datetime(2026, 3, 10, 16, 0, tzinfo=timezone.utc)

    first = run_companion_reflection_job(
        user_id="1",
        cadence=" Weekly ",
        now=now,
        personalization_db=personalization_db,
        collections_db=collections_db,
    )
    second = run_companion_reflection_job(
        user_id="1",
        cadence="weekly",
        now=now,
        personalization_db=personalization_db,
        collections_db=collections_db,
    )

    assert first["status"] == "completed"
    assert second["status"] == "completed"
    assert second["reflection_id"] == first["reflection_id"]
    rows, _total = personalization_db.list_companion_activity_events("1", limit=20, offset=0)
    reflections = [row for row in rows if row["event_type"] == "companion_reflection_generated"]
    assert len(reflections) == 1
    assert reflections[0]["metadata"]["cadence"] == "weekly"
    assert reflections[0]["source_id"] == "2026-W11"


def test_companion_reflection_job_rejects_unknown_cadence(
    companion_reflection_env,
) -> None:
    personalization_db, collections_db = _seed_companion_context("1")

    result = run_companion_reflection_job(
        user_id="1",
        cadence="monthly",
        now=datetime(2026, 3, 10, 16, 0, tzinfo=timezone.utc),
        personalization_db=personalization_db,
        collections_db=collections_db,
    )

    assert result == {"status": "skipped", "reason": "invalid_cadence"}
    rows, _total = personalization_db.list_companion_activity_events("1", limit=20, offset=0)
    assert not any(row["event_type"] == "companion_reflection_generated" for row in rows)


@pytest.mark.asyncio
async def test_handle_companion_reflection_job_rejects_unknown_job_type(
    companion_reflection_env,
) -> None:
    result = await handle_companion_reflection_job(
        {
            "job_type": "companion_archive",
            "payload": {"user_id": "1", "cadence": "daily"},
        }
    )

    assert result == {
        "status": "skipped",
        "reason": "unsupported_job_type",
        "job_type": "companion_archive",
    }


@pytest.mark.asyncio
async def test_handle_companion_job_dispatches_rebuild_scope(companion_reflection_env) -> None:
    personalization_db = PersonalizationDB(str(DatabasePaths.get_personalization_db_path("1")))
    personalization_db.update_profile("1", enabled=1)
    personalization_db.insert_companion_activity_event(
        user_id="1",
        event_type="reading_item_saved",
        source_type="reading_item",
        source_id="101",
        surface="api.reading",
        dedupe_key="reading_item_saved:101",
        tags=["project-alpha", "research"],
        provenance={"capture_mode": "explicit"},
        metadata={"title": "Alpha kickoff"},
    )
    personalization_db.insert_companion_activity_event(
        user_id="1",
        event_type="note_updated",
        source_type="note",
        source_id="202",
        surface="api.notes",
        dedupe_key="note_updated:202",
        tags=["project-alpha", "research"],
        provenance={"capture_mode": "explicit"},
        metadata={"title": "Backlog review notes"},
    )

    result = await handle_companion_reflection_job(
        {
            "job_type": COMPANION_REBUILD_JOB_TYPE,
            "payload": {"user_id": "1", "scope": "knowledge"},
        }
    )

    assert result["status"] == "completed"
    assert result["scope"] == "knowledge"
    cards = personalization_db.list_companion_knowledge_cards("1", status="active")
    assert cards


@pytest.mark.asyncio
async def test_handle_companion_rebuild_job_uses_legacy_storage_for_string_user_ids(
    companion_reflection_env,
) -> None:
    user_id = "user-alpha"
    storage_user_id = _legacy_storage_user_id(user_id)
    personalization_db = PersonalizationDB(
        str(DatabasePaths.get_personalization_db_path(storage_user_id))
    )
    personalization_db.update_profile(user_id, enabled=1)
    personalization_db.insert_companion_activity_event(
        user_id=user_id,
        event_type="reading_item_saved",
        source_type="reading_item",
        source_id="101",
        surface="api.reading",
        dedupe_key="reading_item_saved:101",
        tags=["project-alpha", "research"],
        provenance={"capture_mode": "explicit"},
        metadata={"title": "Alpha kickoff"},
    )
    personalization_db.insert_companion_activity_event(
        user_id=user_id,
        event_type="note_updated",
        source_type="note",
        source_id="202",
        surface="api.notes",
        dedupe_key="note_updated:202",
        tags=["project-alpha", "research"],
        provenance={"capture_mode": "explicit"},
        metadata={"title": "Alpha notes"},
    )

    result = await handle_companion_reflection_job(
        {
            "owner_user_id": user_id,
            "job_type": COMPANION_REBUILD_JOB_TYPE,
            "payload": {"user_id": user_id, "scope": "knowledge"},
        }
    )

    assert result["status"] == "completed"
    assert result["scope"] == "knowledge"
    cards = personalization_db.list_companion_knowledge_cards(user_id, status="active")
    assert cards
