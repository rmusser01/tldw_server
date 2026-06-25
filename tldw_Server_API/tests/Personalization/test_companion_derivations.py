from datetime import datetime, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Personalization.companion_derivations import derive_companion_knowledge_cards
from tldw_Server_API.app.core.Personalization.companion_user_ids import (
    resolve_companion_storage_user_id,
)
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.services.personalization_consolidation import PersonalizationConsolidationService


pytestmark = pytest.mark.unit


def _insert_event(
    db: PersonalizationDB,
    *,
    user_id: str,
    event_type: str,
    source_id: str,
    tags: list[str],
    title: str,
    source_type: str = "reading_item",
) -> None:
    db.insert_companion_activity_event(
        user_id=user_id,
        event_type=event_type,
        source_type=source_type,
        source_id=source_id,
        surface="api.reading",
        dedupe_key=f"{event_type}:{source_id}",
        tags=tags,
        provenance={"capture_mode": "explicit"},
        metadata={"title": title},
    )


def _set_event_created_at(
    db: PersonalizationDB,
    *,
    user_id: str,
    source_id: str,
    created_at: datetime,
) -> None:
    with db._lock:
        conn = db._connect()
        try:
            conn.execute(
                """
                UPDATE companion_activity_events
                SET created_at = ?
                WHERE user_id = ? AND source_id = ?
                """,
                (
                    created_at.astimezone(timezone.utc).replace(microsecond=0).isoformat(),
                    str(user_id),
                    str(source_id),
                ),
            )
            conn.commit()
        finally:
            conn.close()


@pytest.fixture()
def companion_derivations_env(monkeypatch, tmp_path):
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


def test_derive_companion_knowledge_cards_builds_project_focus_card(companion_derivations_env):
    user_id = "1"
    db = PersonalizationDB(str(DatabasePaths.get_personalization_db_path(user_id)))

    _insert_event(
        db,
        user_id=user_id,
        event_type="reading_item_saved",
        source_id="101",
        tags=["project-alpha", "research"],
        title="Alpha kickoff",
    )
    _insert_event(
        db,
        user_id=user_id,
        event_type="reading_item_updated",
        source_id="102",
        tags=["project-alpha"],
        title="Alpha follow-up",
    )
    _insert_event(
        db,
        user_id=user_id,
        event_type="reading_item_saved",
        source_id="103",
        tags=["project-beta"],
        title="Beta note",
    )

    cards = derive_companion_knowledge_cards(db, user_id=user_id)

    card_types = {card["card_type"] for card in cards}
    assert "project_focus" in card_types
    assert "source_focus" in card_types
    card = next(card for card in cards if card["card_type"] == "project_focus")
    assert card["card_type"] == "project_focus"
    assert card["title"] == "Current focus"
    assert "project-alpha" in card["summary"]
    assert len(card["evidence"]) == 2


def test_consolidation_upserts_companion_knowledge_cards(companion_derivations_env):
    user_id = "1"
    db = PersonalizationDB(str(DatabasePaths.get_personalization_db_path(user_id)))

    _insert_event(
        db,
        user_id=user_id,
        event_type="watchlist_source_created",
        source_id="201",
        tags=["security", "alerting"],
        title="Security feeds",
    )
    _insert_event(
        db,
        user_id=user_id,
        event_type="watchlist_source_created",
        source_id="202",
        tags=["security", "triage"],
        title="Second security feed",
    )

    service = PersonalizationConsolidationService()
    service._consolidate_user(user_id)

    cards = db.list_companion_knowledge_cards(user_id)
    card_types = {card["card_type"] for card in cards}
    assert "project_focus" in card_types
    assert "source_focus" in card_types
    project_card = next(card for card in cards if card["card_type"] == "project_focus")
    assert "security" in project_card["summary"]


def test_consolidation_pairs_storage_dirs_with_logical_profile_ids(companion_derivations_env):
    user_id = "user-alpha"
    storage_user_id = resolve_companion_storage_user_id(user_id)
    db = PersonalizationDB(str(DatabasePaths.get_personalization_db_path(storage_user_id)))
    db.update_profile(user_id, enabled=1)
    _insert_event(
        db,
        user_id=user_id,
        event_type="reading_item_saved",
        source_id="501",
        tags=["storage-safe", "research"],
        title="Storage-safe profile event",
    )
    _insert_event(
        db,
        user_id=user_id,
        event_type="reading_item_updated",
        source_id="502",
        tags=["storage-safe", "research"],
        title="Storage-safe follow-up",
    )

    service = PersonalizationConsolidationService()
    targets = service._enumerate_user_targets()

    assert [
        (target.logical_user_id, target.storage_user_id)
        for target in targets
    ] == [(user_id, storage_user_id)]

    service._consolidate_user(user_id, storage_user_id=storage_user_id)

    cards = db.list_companion_knowledge_cards(user_id)
    assert cards
    assert not db.list_companion_knowledge_cards(storage_user_id)


def test_derive_companion_knowledge_cards_emits_multiple_card_families(companion_derivations_env):
    user_id = "1"
    db = PersonalizationDB(str(DatabasePaths.get_personalization_db_path(user_id)))

    _insert_event(
        db,
        user_id=user_id,
        event_type="reading_item_saved",
        source_id="301",
        tags=["project-alpha", "research"],
        title="Alpha research brief",
    )
    _insert_event(
        db,
        user_id=user_id,
        event_type="reading_item_updated",
        source_id="302",
        tags=["project-alpha", "research"],
        title="Alpha research updates",
    )
    _insert_event(
        db,
        user_id=user_id,
        event_type="reading_item_saved",
        source_id="303",
        tags=["project-alpha", "planning"],
        title="Alpha planning note",
        source_type="watchlist_item",
    )
    _set_event_created_at(
        db,
        user_id=user_id,
        source_id="301",
        created_at=datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc),
    )
    _set_event_created_at(
        db,
        user_id=user_id,
        source_id="302",
        created_at=datetime(2026, 3, 1, 13, 0, tzinfo=timezone.utc),
    )
    _set_event_created_at(
        db,
        user_id=user_id,
        source_id="303",
        created_at=datetime(2026, 3, 1, 14, 0, tzinfo=timezone.utc),
    )
    db.create_companion_goal(
        user_id=user_id,
        title="Resume alpha review",
        description="Review the alpha backlog and open questions.",
        goal_type="manual",
        config={},
        progress={"percent": 25},
        origin_kind="manual",
        progress_mode="computed",
        evidence=[{"source_id": "301"}],
        status="active",
    )

    cards = derive_companion_knowledge_cards(
        db,
        user_id=user_id,
        now=datetime(2026, 3, 10, 15, 0, tzinfo=timezone.utc),
    )
    card_types = {card["card_type"] for card in cards}

    assert "project_focus" in card_types
    assert "topic_focus" in card_types
    assert "stale_followup" in card_types
    assert "source_focus" in card_types
    assert "active_goal_signal" in card_types
