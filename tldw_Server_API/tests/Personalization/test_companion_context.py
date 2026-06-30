import sqlite3
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Personalization.companion_user_ids import resolve_companion_storage_user_id
from tldw_Server_API.app.core.Personalization.companion_context import load_companion_context
from tldw_Server_API.app.core.config import settings


pytestmark = pytest.mark.unit


@pytest.fixture()
def companion_context_env(monkeypatch, tmp_path):
    base_dir = tmp_path / "user_dbs"
    base_dir.mkdir(parents=True, exist_ok=True)
    prev_base_dir = settings.get("USER_DB_BASE_DIR")
    settings.USER_DB_BASE_DIR = str(base_dir)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    monkeypatch.setenv("TEST_MODE", "1")
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


def test_load_companion_context_returns_compact_knowledge_and_explicit_activity(
    companion_context_env,
):
    user_id = "71"
    db = PersonalizationDB(str(DatabasePaths.get_personalization_db_path(user_id)))
    db.update_profile(user_id, enabled=1)
    db.upsert_companion_knowledge_card(
        user_id=user_id,
        card_type="project_focus",
        title="Current focus",
        summary="Recent explicit activity clusters around 'pytest'.",
        evidence=[{"source_id": "reading-1"}],
        score=0.9,
        status="active",
    )
    db.insert_companion_activity_event(
        user_id=user_id,
        event_type="reading_item_saved",
        source_type="reading_item",
        source_id="reading-1",
        surface="api.reading",
        dedupe_key="reading.save:reading-1",
        tags=["pytest"],
        provenance={"capture_mode": "explicit", "route": "/api/v1/reading/save"},
        metadata={"title": "Pytest Patterns"},
    )
    db.insert_companion_activity_event(
        user_id=user_id,
        event_type="companion_reflection_generated",
        source_type="companion_reflection",
        source_id="daily:2026-03-10",
        surface="jobs.companion",
        dedupe_key="companion.reflection:daily:2026-03-10",
        provenance={"generated_at": "2026-03-10T05:00:00Z"},
        metadata={"title": "Daily reflection"},
    )

    payload = load_companion_context(user_id=user_id)

    assert payload["card_count"] == 1
    assert payload["activity_count"] == 1
    assert payload["knowledge_lines"] == [
        "- Current focus: Recent explicit activity clusters around 'pytest'."
    ]
    assert payload["activity_lines"] == [
        "- Pytest Patterns (reading item saved)"
    ]


def test_load_companion_context_uses_resolved_storage_user_id_for_text_users(monkeypatch):
    class _FakeDB:
        def get_or_create_profile(self, user_id_value):
            assert user_id_value == "user-alpha"
            return {"enabled": 0}

    opened_user_ids: list[str] = []

    def _fake_for_user(cls, user_id_value):
        opened_user_ids.append(str(user_id_value))
        return _FakeDB()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Personalization.companion_context.PersonalizationDB.for_user",
        classmethod(_fake_for_user),
    )

    payload = load_companion_context(user_id="user-alpha")

    assert payload == {
        "knowledge_lines": [],
        "activity_lines": [],
        "card_count": 0,
        "activity_count": 0,
    }
    assert opened_user_ids == [resolve_companion_storage_user_id("user-alpha")]


def test_load_companion_context_returns_empty_when_profile_disabled(companion_context_env):
    user_id = "72"
    db = PersonalizationDB(str(DatabasePaths.get_personalization_db_path(user_id)))
    db.update_profile(user_id, enabled=0)
    db.upsert_companion_knowledge_card(
        user_id=user_id,
        card_type="project_focus",
        title="Current focus",
        summary="Should not be returned.",
        evidence=[],
        score=0.1,
        status="active",
    )

    payload = load_companion_context(user_id=user_id)

    assert payload == {
        "knowledge_lines": [],
        "activity_lines": [],
        "card_count": 0,
        "activity_count": 0,
    }


def test_load_companion_context_returns_empty_for_recoverable_db_errors(
    companion_context_env,
    monkeypatch,
):
    user_id = "73"
    db = PersonalizationDB(str(DatabasePaths.get_personalization_db_path(user_id)))
    db.update_profile(user_id, enabled=1)

    def _raise_sqlite_error(self, user_id_value):
        raise sqlite3.OperationalError("database is busy")

    monkeypatch.setattr(PersonalizationDB, "get_or_create_profile", _raise_sqlite_error)

    payload = load_companion_context(user_id=user_id)

    assert payload == {
        "knowledge_lines": [],
        "activity_lines": [],
        "card_count": 0,
        "activity_count": 0,
    }


def test_load_companion_context_does_not_suppress_unexpected_errors(
    companion_context_env,
    monkeypatch,
):
    user_id = "74"
    db = PersonalizationDB(str(DatabasePaths.get_personalization_db_path(user_id)))
    db.update_profile(user_id, enabled=1)

    def _raise_runtime_error(self, user_id_value):
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(PersonalizationDB, "get_or_create_profile", _raise_runtime_error)

    with pytest.raises(RuntimeError, match="unexpected failure"):
        load_companion_context(user_id=user_id)
