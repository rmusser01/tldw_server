from pathlib import Path
import sqlite3
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Personalization import companion_activity as companion_activity_module
from tldw_Server_API.app.core.Personalization.companion_activity import (
    build_note_bulk_import_activity,
    build_watchlist_source_bulk_import_activity,
    record_companion_activity,
    record_companion_activity_events_bulk,
    record_note_created,
    record_note_deleted,
    record_note_restored,
    record_note_updated,
    record_persona_session_started,
    record_persona_session_summarized,
    record_persona_tool_executed,
    record_reminder_task_deleted,
    record_reminder_task_updated,
    record_watchlist_item_added,
    record_watchlist_item_updated,
    record_watchlist_source_deleted,
    record_watchlist_source_restored,
    record_watchlist_source_updated,
)
from tldw_Server_API.app.core.config import settings


pytestmark = pytest.mark.unit


def _capture_companion_activity_logs() -> tuple[list[str], int]:
    messages: list[str] = []
    sink_id = companion_activity_module.logger.add(
        lambda message: messages.append(str(message.record.get("message") or "")),
        level="DEBUG",
    )
    return messages, sink_id


@pytest.fixture()
def companion_db_env(monkeypatch, tmp_path):
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


def test_record_companion_activity_requires_opt_in_and_dedupes(companion_db_env):
    user_id = "77"
    db = PersonalizationDB(str(DatabasePaths.get_personalization_db_path(user_id)))

    skipped = record_companion_activity(
        user_id=user_id,
        event_type="reading_item_saved",
        source_type="reading_item",
        source_id="123",
        surface="api.reading",
        dedupe_key="reading.save:123",
        tags=["ai"],
        provenance={"capture_mode": "explicit", "route": "/api/v1/reading/save"},
        metadata={"title": "Example"},
    )
    assert skipped is None
    _, skipped_total = db.list_companion_activity_events(user_id)
    assert skipped_total == 0

    db.update_profile(user_id, enabled=1)
    created = record_companion_activity(
        user_id=user_id,
        event_type="reading_item_saved",
        source_type="reading_item",
        source_id="123",
        surface="api.reading",
        dedupe_key="reading.save:123",
        tags=["ai"],
        provenance={"capture_mode": "explicit", "route": "/api/v1/reading/save"},
        metadata={"title": "Example"},
    )
    assert created

    items, total = db.list_companion_activity_events(user_id)
    assert total == 1
    assert items[0]["event_type"] == "reading_item_saved"
    assert items[0]["provenance"]["capture_mode"] == "explicit"

    duplicate = record_companion_activity(
        user_id=user_id,
        event_type="reading_item_saved",
        source_type="reading_item",
        source_id="123",
        surface="api.reading",
        dedupe_key="reading.save:123",
        tags=["ai"],
        provenance={"capture_mode": "explicit", "route": "/api/v1/reading/save"},
        metadata={"title": "Example"},
    )
    assert duplicate is None
    _, total_after_duplicate = db.list_companion_activity_events(user_id)
    assert total_after_duplicate == 1


def test_insert_companion_activity_events_bulk_skips_duplicate_dedupe_keys(companion_db_env):
    user_id = "77-bulk"
    db = PersonalizationDB(str(DatabasePaths.get_personalization_db_path(user_id)))
    db.update_profile(user_id, enabled=1)

    inserted = db.insert_companion_activity_events_bulk(
        user_id=user_id,
        events=[
            {
                "event_type": "note_created",
                "source_type": "note",
                "source_id": "n1",
                "surface": "api.notes.import",
                "dedupe_key": "notes.create:n1",
                "provenance": {"capture_mode": "explicit"},
                "metadata": {"title": "One"},
            },
            {
                "event_type": "note_created",
                "source_type": "note",
                "source_id": "n1",
                "surface": "api.notes.import",
                "dedupe_key": "notes.create:n1",
                "provenance": {"capture_mode": "explicit"},
                "metadata": {"title": "One duplicate"},
            },
        ],
    )

    assert len(inserted) == 1
    rows, total = db.list_companion_activity_events(user_id, limit=10)
    assert total == 1
    assert rows[0]["source_id"] == "n1"


def test_insert_companion_activity_events_bulk_keeps_unique_rows_when_one_conflicts(companion_db_env):
    user_id = "77-bulk-existing"
    db = PersonalizationDB(str(DatabasePaths.get_personalization_db_path(user_id)))
    db.update_profile(user_id, enabled=1)
    db.insert_companion_activity_event(
        user_id=user_id,
        event_type="note_created",
        source_type="note",
        source_id="n1",
        surface="api.notes.import",
        dedupe_key="notes.create:n1",
        provenance={"capture_mode": "explicit"},
        metadata={"title": "Existing"},
    )

    inserted = db.insert_companion_activity_events_bulk(
        user_id=user_id,
        events=[
            {
                "event_type": "note_created",
                "source_type": "note",
                "source_id": "n1",
                "surface": "api.notes.import",
                "dedupe_key": "notes.create:n1",
                "provenance": {"capture_mode": "explicit"},
                "metadata": {"title": "Duplicate"},
            },
            {
                "event_type": "note_created",
                "source_type": "note",
                "source_id": "n2",
                "surface": "api.notes.import",
                "dedupe_key": "notes.create:n2",
                "provenance": {"capture_mode": "explicit"},
                "metadata": {"title": "Fresh"},
            },
        ],
    )

    assert len(inserted) == 1
    rows, total = db.list_companion_activity_events(user_id, limit=10)
    assert total == 2
    source_ids = {row["source_id"] for row in rows}
    assert source_ids == {"n1", "n2"}


def test_insert_companion_activity_events_bulk_keeps_unique_rows_when_conflict_appears_at_insert_time(
    companion_db_env,
    monkeypatch,
):
    user_id = "77-bulk-race"
    db = PersonalizationDB(str(DatabasePaths.get_personalization_db_path(user_id)))
    db.update_profile(user_id, enabled=1)
    real_connect = db._connect

    class _RaceInjectingConnection:
        def __init__(self, inner: sqlite3.Connection) -> None:
            self._inner = inner
            self._injected = False

        def execute(self, sql: str, params: tuple[Any, ...] = ()) -> Any:
            normalized_sql = " ".join(sql.split()).upper()
            if (
                normalized_sql.startswith("INSERT")
                and "COMPANION_ACTIVITY_EVENTS" in normalized_sql
                and len(params) >= 7
                and str(params[6]) == "notes.create:n1"
                and not self._injected
            ):
                self._injected = True
                competing = sqlite3.connect(db.db_path, timeout=10, isolation_level=None)
                try:
                    competing.execute("PRAGMA journal_mode=WAL")
                    competing.execute("PRAGMA foreign_keys=ON")
                    competing.execute("PRAGMA busy_timeout=5000")
                    competing.execute(sql, params)
                    competing.commit()
                finally:
                    competing.close()
            return self._inner.execute(sql, params)

        def executemany(self, sql: str, seq_of_params: Any) -> Any:
            for row in seq_of_params:
                self.execute(sql, row)
            return None

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    monkeypatch.setattr(db, "_connect", lambda: _RaceInjectingConnection(real_connect()))

    inserted = db.insert_companion_activity_events_bulk(
        user_id=user_id,
        events=[
            {
                "event_type": "note_created",
                "source_type": "note",
                "source_id": "n1",
                "surface": "api.notes.import",
                "dedupe_key": "notes.create:n1",
                "provenance": {"capture_mode": "explicit"},
                "metadata": {"title": "Conflicted"},
            },
            {
                "event_type": "note_created",
                "source_type": "note",
                "source_id": "n2",
                "surface": "api.notes.import",
                "dedupe_key": "notes.create:n2",
                "provenance": {"capture_mode": "explicit"},
                "metadata": {"title": "Fresh"},
            },
        ],
    )

    assert len(inserted) == 1
    rows, total = db.list_companion_activity_events(user_id, limit=10)
    assert total == 2
    source_ids = {row["source_id"] for row in rows}
    assert source_ids == {"n1", "n2"}


def test_record_companion_activity_capture_skip_log_omits_exception_text(monkeypatch):
    messages, sink_id = _capture_companion_activity_logs()
    monkeypatch.setattr(companion_activity_module, "is_personalization_enabled", lambda: True)
    monkeypatch.setattr(
        companion_activity_module,
        "_open_db_for_user",
        lambda user_id: (_ for _ in ()).throw(
            RuntimeError("companion backend exploded /private/companion.db")
        ),
    )

    try:
        result = record_companion_activity(
            user_id="77-capture-sanitizer",
            event_type="reading_item_saved",
            source_type="reading_item",
            source_id="123",
            surface="api.reading",
            dedupe_key="reading.save:123",
        )
    finally:
        companion_activity_module.logger.remove(sink_id)

    assert result is None
    log_text = "\n".join(messages)
    assert "companion activity capture skipped" in log_text
    assert "companion backend exploded" not in log_text
    assert "/private/companion.db" not in log_text


def test_record_companion_activity_insert_skip_log_omits_exception_text(monkeypatch):
    class _ExplodingInsertDB:
        def get_or_create_profile(self, user_id: str) -> dict[str, int]:
            return {"enabled": 1}

        def insert_companion_activity_event(self, **kwargs: Any) -> str:
            raise sqlite3.IntegrityError("database disk image is malformed /private/companion.db")

    messages, sink_id = _capture_companion_activity_logs()
    monkeypatch.setattr(companion_activity_module, "is_personalization_enabled", lambda: True)
    monkeypatch.setattr(
        companion_activity_module,
        "_open_db_for_user",
        lambda user_id: (_ExplodingInsertDB(), str(user_id)),
    )

    try:
        result = record_companion_activity(
            user_id="77-insert-sanitizer",
            event_type="reading_item_saved",
            source_type="reading_item",
            source_id="123",
            surface="api.reading",
            dedupe_key="reading.save:123",
        )
    finally:
        companion_activity_module.logger.remove(sink_id)

    assert result is None
    log_text = "\n".join(messages)
    assert "companion activity insert skipped" in log_text
    assert "database disk image is malformed" not in log_text
    assert "/private/companion.db" not in log_text


def test_record_companion_activity_bulk_capture_skip_log_omits_exception_text(monkeypatch):
    messages, sink_id = _capture_companion_activity_logs()
    monkeypatch.setattr(companion_activity_module, "is_personalization_enabled", lambda: True)
    monkeypatch.setattr(
        companion_activity_module,
        "_open_db_for_user",
        lambda user_id: (_ for _ in ()).throw(
            RuntimeError("companion backend exploded /private/companion.db")
        ),
    )

    try:
        result = record_companion_activity_events_bulk(
            user_id="77-bulk-sanitizer",
            events=[
                {
                    "event_type": "note_created",
                    "source_type": "note",
                    "source_id": "n1",
                    "surface": "api.notes.import",
                    "dedupe_key": "notes.create:n1",
                }
            ],
        )
    finally:
        companion_activity_module.logger.remove(sink_id)

    assert result == []
    log_text = "\n".join(messages)
    assert "companion activity bulk capture skipped" in log_text
    assert "companion backend exploded" not in log_text
    assert "/private/companion.db" not in log_text


def test_watchlist_item_tags_lookup_skip_log_omits_exception_text():
    class _ItemWithExplodingTags:
        def tags(self) -> list[str]:
            raise RuntimeError("watchlist tags exploded /private/watchlist.json")

    messages, sink_id = _capture_companion_activity_logs()
    try:
        tags = companion_activity_module._watchlist_item_tags(_ItemWithExplodingTags())
    finally:
        companion_activity_module.logger.remove(sink_id)

    assert tags == []
    log_text = "\n".join(messages)
    assert "watchlist item tags() lookup skipped" in log_text
    assert "watchlist tags exploded" not in log_text
    assert "/private/watchlist.json" not in log_text


def test_watchlist_item_tags_json_parse_skip_log_omits_parse_error_text():
    class _ItemWithMalformedTagsJson:
        tags_json = '["security", "/private/watchlist.json",'

    messages, sink_id = _capture_companion_activity_logs()
    try:
        tags = companion_activity_module._watchlist_item_tags(_ItemWithMalformedTagsJson())
    finally:
        companion_activity_module.logger.remove(sink_id)

    assert tags == []
    log_text = "\n".join(messages)
    assert "watchlist item tags_json parse skipped" in log_text
    assert "Expecting value" not in log_text
    assert "/private/watchlist.json" not in log_text


def test_build_note_import_created_activity_uses_import_surface_and_route():
    note = {
        "id": "note-1",
        "title": "Imported note",
        "content": "Imported content",
        "version": 1,
        "created_at": "2026-03-12T00:00:00+00:00",
        "last_modified": "2026-03-12T00:00:00+00:00",
        "keywords": [{"keyword": "import"}],
    }

    payload = build_note_bulk_import_activity(
        note=note,
        operation="import_create",
        route="/api/v1/notes/import",
        surface="api.notes.import",
    )

    assert payload["event_type"] == "note_created"
    assert payload["source_type"] == "note"
    assert payload["source_id"] == "note-1"
    assert payload["surface"] == "api.notes.import"
    assert payload["provenance"]["route"] == "/api/v1/notes/import"
    assert payload["provenance"]["action"] == "import_create"
    assert payload["metadata"]["title"] == "Imported note"
    assert payload["tags"] == ["import"]


def test_build_watchlist_source_bulk_activity_uses_bulk_surface_and_route():
    source = {
        "id": 12,
        "name": "Bulk source",
        "url": "https://example.com/feed.xml",
        "source_type": "rss",
        "active": True,
        "status": None,
        "group_ids": [2],
        "tags": ["feeds"],
        "created_at": "2026-03-12T00:00:00+00:00",
        "updated_at": "2026-03-12T00:00:00+00:00",
    }

    payload = build_watchlist_source_bulk_import_activity(
        source=source,
        operation="bulk_create",
        route="/api/v1/watchlists/sources/bulk",
        surface="api.watchlists.sources.bulk",
    )

    assert payload["event_type"] == "watchlist_source_created"
    assert payload["source_type"] == "watchlist_source"
    assert payload["source_id"] == "12"
    assert payload["surface"] == "api.watchlists.sources.bulk"
    assert payload["provenance"]["route"] == "/api/v1/watchlists/sources/bulk"
    assert payload["provenance"]["action"] == "bulk_create"
    assert payload["metadata"]["name"] == "Bulk source"
    assert payload["tags"] == ["feeds"]


def test_note_activity_adapters_capture_compact_metadata(companion_db_env):
    user_id = "78"
    db = PersonalizationDB(str(DatabasePaths.get_personalization_db_path(user_id)))
    db.update_profile(user_id, enabled=1)

    note = {
        "id": "note-1",
        "title": "Focus Note",
        "content": "This is a note body that should be compactly represented in the companion activity ledger.",
        "created_at": "2026-03-09T10:00:00Z",
        "last_modified": "2026-03-09T10:05:00Z",
        "version": 2,
        "conversation_id": "conv-1",
        "message_id": "msg-1",
        "keywords": [{"keyword": "research"}, {"keyword": "planning"}],
    }

    created = record_note_created(user_id=user_id, note=note)
    updated = record_note_updated(
        user_id=user_id,
        note=note,
        route="/api/v1/notes/note-1",
        action="patch",
        patch={"title": "Focus Note", "content": "Updated body"},
    )

    assert created
    assert updated

    events, total = db.list_companion_activity_events(user_id, limit=10)
    assert total == 2

    updated_event = events[0]
    assert updated_event["event_type"] == "note_updated"
    assert updated_event["tags"] == ["research", "planning"]
    assert updated_event["metadata"]["version"] == 2
    assert updated_event["metadata"]["changed_fields"] == ["content", "title"]
    assert "Updated body" not in str(updated_event["metadata"])
    assert updated_event["provenance"]["action"] == "patch"

    created_event = events[1]
    assert created_event["event_type"] == "note_created"
    assert created_event["provenance"]["route"] == "/api/v1/notes/"
    assert created_event["metadata"]["content_preview"].startswith("This is a note body")


def test_note_delete_and_restore_adapters_capture_state_changes(companion_db_env):
    user_id = "79"
    db = PersonalizationDB(str(DatabasePaths.get_personalization_db_path(user_id)))
    db.update_profile(user_id, enabled=1)

    deleted_note = {
        "id": "note-2",
        "title": "Archive Me",
        "content": "A note that is about to be removed from the active list.",
        "created_at": "2026-03-09T11:00:00Z",
        "last_modified": "2026-03-09T11:05:00Z",
        "version": 2,
        "keywords": [{"keyword": "cleanup"}],
    }
    restored_note = {
        **deleted_note,
        "last_modified": "2026-03-09T11:10:00Z",
        "version": 4,
    }

    deleted = record_note_deleted(user_id=user_id, note=deleted_note, deleted_version=3)
    restored = record_note_restored(user_id=user_id, note=restored_note)

    assert deleted
    assert restored

    events, total = db.list_companion_activity_events(user_id, limit=10)
    assert total == 2

    restored_event = events[0]
    assert restored_event["event_type"] == "note_restored"
    assert restored_event["metadata"]["version"] == 4
    assert restored_event["metadata"]["deleted"] is False
    assert restored_event["provenance"]["action"] == "restore"

    deleted_event = events[1]
    assert deleted_event["event_type"] == "note_deleted"
    assert deleted_event["tags"] == ["cleanup"]
    assert deleted_event["metadata"]["version"] == 3
    assert deleted_event["metadata"]["deleted"] is True
    assert deleted_event["metadata"]["hard_delete"] is False
    assert deleted_event["metadata"]["content_preview"].startswith("A note that is about to be removed")
    assert "content" not in deleted_event["metadata"]


def test_reminder_task_update_and_delete_adapters_capture_compact_metadata(companion_db_env):
    user_id = "80"
    db = PersonalizationDB(str(DatabasePaths.get_personalization_db_path(user_id)))
    db.update_profile(user_id, enabled=1)

    updated_task = {
        "id": "task-1",
        "title": "Review companion signals",
        "body": "Check the explicit activity backlog before the next reflection run.",
        "schedule_kind": "one_time",
        "run_at": "2026-03-10T18:00:00Z",
        "cron": None,
        "timezone": None,
        "enabled": False,
        "link_type": "note",
        "link_id": "note-22",
        "link_url": None,
        "created_at": "2026-03-10T09:00:00Z",
        "updated_at": "2026-03-10T09:30:00Z",
    }

    updated = record_reminder_task_updated(
        user_id=user_id,
        task=updated_task,
        patch={"enabled": False, "title": "Review companion signals"},
    )
    deleted = record_reminder_task_deleted(user_id=user_id, task=updated_task)

    assert updated
    assert deleted

    events, total = db.list_companion_activity_events(user_id, limit=10)
    assert total == 2

    deleted_event = events[0]
    assert deleted_event["event_type"] == "reminder_task_deleted"
    assert deleted_event["source_type"] == "reminder_task"
    assert deleted_event["surface"] == "api.tasks"
    assert deleted_event["provenance"]["action"] == "delete"
    assert deleted_event["metadata"]["title"] == "Review companion signals"
    assert deleted_event["metadata"]["hard_delete"] is True
    assert deleted_event["metadata"]["enabled"] is False
    assert deleted_event["metadata"]["body_preview"].startswith("Check the explicit activity backlog")
    assert "body" not in deleted_event["metadata"]

    updated_event = events[1]
    assert updated_event["event_type"] == "reminder_task_updated"
    assert updated_event["provenance"]["route"] == "/api/v1/tasks/task-1"
    assert updated_event["metadata"]["changed_fields"] == ["enabled", "title"]
    assert updated_event["metadata"]["schedule_kind"] == "one_time"
    assert updated_event["metadata"]["link_type"] == "note"


def test_persona_summary_and_tool_adapters_capture_compact_metadata(companion_db_env):
    user_id = "81"
    db = PersonalizationDB(str(DatabasePaths.get_personalization_db_path(user_id)))
    db.update_profile(user_id, enabled=1)

    summary_text = (
        "Wrapped up the session with a concise plan for companion capture parity across "
        "notes, reminders, watchlists, and persona. " * 3
    ).strip()
    summarized = record_persona_session_summarized(
        user_id=user_id,
        session_id="sess-81",
        persona_id="research_assistant",
        plan_id="plan-81",
        step_idx=2,
        runtime_mode="session_scoped",
        scope_snapshot_id="scope-81",
        summary_text=summary_text,
    )
    executed = record_persona_tool_executed(
        user_id=user_id,
        session_id="sess-81",
        persona_id="research_assistant",
        plan_id="plan-81",
        step_idx=1,
        step_type="mcp_tool",
        tool_name="ingest_url",
        runtime_mode="session_scoped",
        scope_snapshot_id="scope-81",
        outcome={
            "ok": True,
            "output": {
                "saved": True,
                "url": "https://example.com/article",
                "secret": "do-not-store-this-raw-output",
            },
        },
    )
    skipped = record_persona_tool_executed(
        user_id=user_id,
        session_id="sess-81",
        persona_id="research_assistant",
        plan_id="plan-81",
        step_idx=3,
        step_type="rag_query",
        tool_name="rag_search",
        runtime_mode="session_scoped",
        scope_snapshot_id="scope-81",
        outcome={"ok": True, "output": {"matches": 4}},
    )

    assert summarized
    assert executed
    assert skipped is None

    events, total = db.list_companion_activity_events(user_id, limit=10)
    assert total == 2

    tool_event = events[0]
    assert tool_event["event_type"] == "persona_tool_executed"
    assert tool_event["source_type"] == "persona_tool_step"
    assert tool_event["surface"] == "api.persona"
    assert tool_event["tags"] == ["research_assistant", "ingest_url"]
    assert tool_event["metadata"]["tool_name"] == "ingest_url"
    assert tool_event["metadata"]["step_type"] == "mcp_tool"
    assert tool_event["metadata"]["ok"] is True
    assert tool_event["metadata"]["output_type"] == "dict"
    assert tool_event["metadata"]["output_item_count"] == 3
    assert tool_event["provenance"]["route"] == "/api/v1/persona/stream"
    assert tool_event["provenance"]["action"] == "tool_outcome"
    assert "do-not-store-this-raw-output" not in str(tool_event["metadata"])

    summary_event = events[1]
    assert summary_event["event_type"] == "persona_session_summarized"
    assert summary_event["source_type"] == "persona_session"
    assert summary_event["source_id"] == "sess-81"
    assert summary_event["tags"] == ["research_assistant"]
    assert summary_event["metadata"]["persona_id"] == "research_assistant"
    assert summary_event["metadata"]["plan_id"] == "plan-81"
    assert summary_event["metadata"]["step_idx"] == 2
    assert summary_event["metadata"]["summary_char_count"] == len(summary_text)
    assert summary_event["metadata"]["summary_preview"].startswith("Wrapped up the session")
    assert len(summary_event["metadata"]["summary_preview"]) < len(summary_text)
    assert summary_event["provenance"]["route"] == "/api/v1/persona/stream"
    assert summary_event["provenance"]["action"] == "session_summary"


def test_persona_activity_adapters_accept_surface_overrides_and_normalize_invalid_values(
    companion_db_env,
):
    user_id = "81b"
    db = PersonalizationDB(str(DatabasePaths.get_personalization_db_path(user_id)))
    db.update_profile(user_id, enabled=1)

    started = record_persona_session_started(
        user_id=user_id,
        session_id="sess-81b",
        persona_id="research_assistant",
        runtime_mode="session_scoped",
        scope_snapshot_id="scope-81b",
        surface="invalid.surface",
    )
    summarized = record_persona_session_summarized(
        user_id=user_id,
        session_id="sess-81b",
        persona_id="research_assistant",
        plan_id="plan-81b",
        step_idx=2,
        runtime_mode="session_scoped",
        scope_snapshot_id="scope-81b",
        summary_text="Summarized companion work for the session.",
        surface="companion.conversation",
    )
    executed = record_persona_tool_executed(
        user_id=user_id,
        session_id="sess-81b",
        persona_id="research_assistant",
        plan_id="plan-81b",
        step_idx=1,
        step_type="mcp_tool",
        tool_name="ingest_url",
        runtime_mode="session_scoped",
        scope_snapshot_id="scope-81b",
        surface="companion.conversation",
        outcome={"ok": True, "output": {"saved": True}},
    )

    assert started
    assert summarized
    assert executed

    events, total = db.list_companion_activity_events(user_id, limit=10)
    assert total == 3

    started_event = next(event for event in events if event["event_type"] == "persona_session_started")
    assert started_event["surface"] == "api.persona"

    summary_event = next(
        event for event in events if event["event_type"] == "persona_session_summarized"
    )
    assert summary_event["surface"] == "companion.conversation"

    tool_event = next(event for event in events if event["event_type"] == "persona_tool_executed")
    assert tool_event["surface"] == "companion.conversation"


def test_watchlist_source_update_delete_and_restore_adapters_capture_compact_metadata(companion_db_env):
    user_id = "81"
    db = PersonalizationDB(str(DatabasePaths.get_personalization_db_path(user_id)))
    db.update_profile(user_id, enabled=1)

    source = {
        "id": 202,
        "name": "Security Feed",
        "url": "https://example.com/security.xml",
        "source_type": "rss",
        "active": True,
        "tags": ["security", "feeds"],
        "group_ids": [7, 9],
        "settings": {"poll_minutes": 30, "include_summary": True},
        "status": "ok",
        "created_at": "2026-03-10T08:00:00Z",
        "updated_at": "2026-03-10T09:00:00Z",
    }

    updated = record_watchlist_source_updated(
        user_id=user_id,
        source={**source, "active": False, "tags": ["security", "analysis"]},
        patch={"active": False, "tags": ["security", "analysis"]},
        event_timestamp="2026-03-10T09:15:00Z",
    )
    deleted = record_watchlist_source_deleted(
        user_id=user_id,
        source={**source, "active": False, "tags": ["security", "analysis"]},
        event_timestamp="2026-03-10T09:30:00Z",
        restore_window_seconds=300,
    )
    restored = record_watchlist_source_restored(
        user_id=user_id,
        source={**source, "active": False, "tags": ["security", "analysis"]},
        event_timestamp="2026-03-10T09:35:00Z",
    )

    assert updated
    assert deleted
    assert restored

    events, total = db.list_companion_activity_events(user_id, limit=10)
    assert total == 3

    restored_event = events[0]
    assert restored_event["event_type"] == "watchlist_source_restored"
    assert restored_event["metadata"]["deleted"] is False
    assert restored_event["metadata"]["hard_delete"] is False
    assert restored_event["provenance"]["route"] == "/api/v1/watchlists/sources/202/restore"

    deleted_event = events[1]
    assert deleted_event["event_type"] == "watchlist_source_deleted"
    assert deleted_event["source_type"] == "watchlist_source"
    assert deleted_event["surface"] == "api.watchlists"
    assert deleted_event["metadata"]["deleted"] is True
    assert deleted_event["metadata"]["hard_delete"] is False
    assert deleted_event["metadata"]["restore_window_seconds"] == 300
    assert deleted_event["metadata"]["settings_keys"] == ["include_summary", "poll_minutes"]

    updated_event = events[2]
    assert updated_event["event_type"] == "watchlist_source_updated"
    assert updated_event["tags"] == ["security", "analysis"]
    assert updated_event["metadata"]["changed_fields"] == ["active", "tags"]
    assert updated_event["metadata"]["group_ids"] == [7, 9]
    assert updated_event["provenance"]["action"] == "update"


def test_watchlist_item_add_and_update_adapters_capture_compact_metadata(companion_db_env):
    user_id = "82"
    db = PersonalizationDB(str(DatabasePaths.get_personalization_db_path(user_id)))
    db.update_profile(user_id, enabled=1)

    item = {
        "id": 404,
        "run_id": 12,
        "job_id": 21,
        "source_id": 202,
        "media_id": 11,
        "media_uuid": "1d2602cb-2a5a-4c9c-b253-c8090bce89d8",
        "url": "https://example.com/watch/security-story",
        "title": "Security Story",
        "summary": (
            "A long summary that should be compacted before it is written into the "
            "companion activity ledger so the ledger does not store the full item body. "
            * 3
        ).strip(),
        "published_at": "2026-03-10T12:00:00Z",
        "status": "ingested",
        "reviewed": 0,
        "queued_for_briefing": 0,
        "created_at": "2026-03-10T12:01:00Z",
        "tags": ["security", "flagged"],
    }

    added = record_watchlist_item_added(
        user_id=user_id,
        item=item,
        route="/api/v1/watchlists/jobs/21/run",
    )
    updated = record_watchlist_item_updated(
        user_id=user_id,
        item={**item, "reviewed": 1, "queued_for_briefing": 1},
        patch={"reviewed": True, "queued_for_briefing": True},
        event_timestamp="2026-03-10T12:05:00Z",
    )

    assert added
    assert updated

    events, total = db.list_companion_activity_events(user_id, limit=10)
    assert total == 2

    updated_event = events[0]
    assert updated_event["event_type"] == "watchlist_item_updated"
    assert updated_event["source_type"] == "watchlist_item"
    assert updated_event["surface"] == "api.watchlists"
    assert updated_event["metadata"]["run_id"] == 12
    assert updated_event["metadata"]["job_id"] == 21
    assert updated_event["metadata"]["source_id"] == 202
    assert updated_event["metadata"]["reviewed"] is True
    assert updated_event["metadata"]["queued_for_briefing"] is True
    assert updated_event["metadata"]["changed_fields"] == ["queued_for_briefing", "reviewed"]
    assert updated_event["metadata"]["summary_preview"].startswith("A long summary")
    assert len(updated_event["metadata"]["summary_preview"]) < len(item["summary"])
    assert "summary" not in updated_event["metadata"]
    assert updated_event["provenance"]["route"] == "/api/v1/watchlists/items/404"
    assert updated_event["provenance"]["action"] == "update"

    added_event = events[1]
    assert added_event["event_type"] == "watchlist_item_added"
    assert added_event["tags"] == ["security", "flagged"]
    assert added_event["metadata"]["status"] == "ingested"
    assert added_event["metadata"]["media_id"] == 11
    assert added_event["metadata"]["media_uuid"] == "1d2602cb-2a5a-4c9c-b253-c8090bce89d8"
    assert added_event["metadata"]["summary_preview"].startswith("A long summary")
    assert len(added_event["metadata"]["summary_preview"]) < len(item["summary"])
    assert added_event["provenance"]["route"] == "/api/v1/watchlists/jobs/21/run"
    assert added_event["provenance"]["action"] == "item_ingested"
