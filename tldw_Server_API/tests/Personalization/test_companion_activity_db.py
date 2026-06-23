from __future__ import annotations

from types import SimpleNamespace

import pytest

import tldw_Server_API.app.api.v1.API_Deps.personalization_deps as personalization_deps
import tldw_Server_API.app.core.DB_Management.Personalization_DB as personalization_db_module
from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB


pytestmark = pytest.mark.unit


def test_companion_activity_round_trip(tmp_path) -> None:
    db = PersonalizationDB(str(tmp_path / "personalization.db"))

    event_id = db.insert_companion_activity_event(
        user_id="1",
        event_type="reading.saved",
        source_type="reading_item",
        source_id="42",
        surface="reading",
        dedupe_key="reading.saved:42",
        tags=["research", "paper"],
        provenance={"source_ids": ["42"]},
        metadata={"title": "Example"},
    )

    rows, total = db.list_companion_activity_events("1", limit=10, offset=0)

    assert event_id
    assert total == 1
    assert rows[0]["id"] == event_id
    assert rows[0]["event_type"] == "reading.saved"
    assert rows[0]["source_type"] == "reading_item"
    assert rows[0]["source_id"] == "42"
    assert rows[0]["surface"] == "reading"
    assert rows[0]["tags"] == ["research", "paper"]
    assert rows[0]["provenance"] == {"source_ids": ["42"]}
    assert rows[0]["metadata"] == {"title": "Example"}


def test_personalization_dependency_uses_safe_for_user_factory(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        personalization_deps.PersonalizationDB,
        "for_user",
        classmethod(lambda cls, user_id: sentinel),
        raising=False,
    )

    result = personalization_deps.get_personalization_db_for_user(
        user=SimpleNamespace(id="7")
    )

    assert result is sentinel


def test_personalization_db_constructor_does_not_resolve_input_path(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "personalization.db"

    def fake_require_trusted_database_parent_exists(path, **kwargs):
        assert path == db_path
        return path

    monkeypatch.setattr(
        personalization_db_module,
        "require_trusted_database_parent_exists",
        fake_require_trusted_database_parent_exists,
    )

    db = PersonalizationDB(db_path)

    assert db.db_path == str(db_path)


def test_companion_knowledge_card_upsert_updates_existing_card(tmp_path) -> None:
    db = PersonalizationDB(str(tmp_path / "personalization.db"))

    first_id = db.upsert_companion_knowledge_card(
        user_id="1",
        card_type="project_focus",
        title="Current focus",
        summary="Initial summary",
        evidence=[{"source_id": "42"}],
        score=0.5,
    )

    second_id = db.upsert_companion_knowledge_card(
        user_id="1",
        card_type="project_focus",
        title="Current focus",
        summary="Updated summary",
        evidence=[{"source_id": "99"}],
        score=0.9,
    )

    rows = db.list_companion_knowledge_cards("1")

    assert first_id == second_id
    assert len(rows) == 1
    assert rows[0]["summary"] == "Updated summary"
    assert rows[0]["evidence"] == [{"source_id": "99"}]
    assert rows[0]["score"] == pytest.approx(0.9)


def test_companion_goal_create_and_list(tmp_path) -> None:
    db = PersonalizationDB(str(tmp_path / "personalization.db"))

    goal_id = db.create_companion_goal(
        user_id="1",
        title="Finish reading queue",
        description="Read 3 saved papers this week",
        goal_type="reading_backlog",
        config={"target_count": 3},
        progress={"completed_count": 0},
    )

    rows = db.list_companion_goals("1")

    assert goal_id
    assert len(rows) == 1
    assert rows[0]["id"] == goal_id
    assert rows[0]["title"] == "Finish reading queue"
    assert rows[0]["goal_type"] == "reading_backlog"
    assert rows[0]["config"] == {"target_count": 3}
    assert rows[0]["progress"] == {"completed_count": 0}
    assert rows[0]["status"] == "active"


def test_companion_activity_bulk_insert_and_lookup_by_dedupe_key(tmp_path) -> None:
    db = PersonalizationDB(str(tmp_path / "personalization.db"))

    event_ids = db.insert_companion_activity_events_bulk(
        user_id="1",
        events=[
            {
                "event_type": "watchlist_item_added",
                "source_type": "watchlist_item",
                "source_id": "42",
                "surface": "api.watchlists",
                "dedupe_key": "watchlists.item.add:42",
                "tags": ["security"],
                "provenance": {"capture_mode": "explicit"},
                "metadata": {"title": "First item"},
            },
            {
                "event_type": "watchlist_item_added",
                "source_type": "watchlist_item",
                "source_id": "43",
                "surface": "api.watchlists",
                "dedupe_key": "watchlists.item.add:43",
                "tags": ["security"],
                "provenance": {"capture_mode": "explicit"},
                "metadata": {"title": "Second item"},
            },
        ],
    )

    rows, total = db.list_companion_activity_events("1", limit=10, offset=0)

    assert len(event_ids) == 2
    assert total == 2
    assert {row["id"] for row in rows} == set(event_ids)
    assert (
        db.get_companion_activity_event_id_by_dedupe_key(
            user_id="1",
            dedupe_key="watchlists.item.add:42",
        )
        in event_ids
    )
