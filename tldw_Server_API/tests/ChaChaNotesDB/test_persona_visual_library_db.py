import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


pytestmark = pytest.mark.unit


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "persona_visual_library.sqlite"


@pytest.fixture()
def db_instance(db_path: Path) -> Iterator[CharactersRAGDB]:
    db = CharactersRAGDB(db_path, "persona-visual-library-test-client")
    yield db
    db.close_connection()


def _create_persona_and_pack(
    db: CharactersRAGDB,
    *,
    user_id: str = "user-1",
    persona_name: str = "Research Buddy",
    pack_title: str = "Warm desk assistant",
) -> tuple[str, dict]:
    persona_id = db.create_persona_profile({"user_id": user_id, "name": persona_name})
    pack = db.create_persona_visual_pack(
        persona_id=persona_id,
        user_id=user_id,
        title=pack_title,
        manifest={
            "manifest_version": 1,
            "renderer_type": "sprite_frames",
            "states": {},
            "animations": {},
        },
    )
    return persona_id, pack


def test_migration_v45_to_latest_creates_persona_visual_library_table(db_path: Path) -> None:
    seeded = CharactersRAGDB(db_path, "persona-visual-library-seed")
    seeded.close_connection()

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute(
            "UPDATE db_schema_version SET version = ? WHERE schema_name = ?",
            (45, CharactersRAGDB._SCHEMA_NAME),
        )
        conn.execute("DROP TABLE IF EXISTS persona_visual_library_items")
        conn.commit()

    migrated = CharactersRAGDB(db_path, "persona-visual-library-migration")
    conn = migrated.get_connection()

    version = conn.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (CharactersRAGDB._SCHEMA_NAME,),
    ).fetchone()["version"]
    assert version == CharactersRAGDB._CURRENT_SCHEMA_VERSION

    tables = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    }
    assert "persona_visual_library_items" in tables

    indexes = {
        row["name"]
        for row in conn.execute("PRAGMA index_list('persona_visual_library_items')").fetchall()
    }
    assert "idx_persona_visual_library_items_user_time" in indexes
    assert "idx_persona_visual_library_items_live_source" in indexes
    migrated.close_connection()


def test_upsert_and_list_persona_visual_library_item_with_source_status(
    db_instance: CharactersRAGDB,
) -> None:
    persona_id, pack = _create_persona_and_pack(db_instance)

    first = db_instance.upsert_persona_visual_library_item(
        user_id="user-1",
        source_persona_id=persona_id,
        source_pack_id=pack["id"],
        title="Desk helper",
        notes="Good for focused research.",
        tags=["Research", "calm", "research"],
    )
    second = db_instance.upsert_persona_visual_library_item(
        user_id="user-1",
        source_persona_id=persona_id,
        source_pack_id=pack["id"],
        title="Warm desk assistant",
        notes="Updated notes",
        tags=["calm"],
    )
    assert second["id"] == first["id"]

    updated_pack = db_instance.update_persona_visual_pack_manifest(
        pack_id=pack["id"],
        persona_id=persona_id,
        user_id="user-1",
        manifest={
            "manifest_version": 1,
            "renderer_type": "sprite_frames",
            "states": {"idle": {"animation_id": "idle"}},
            "animations": {"idle": {"frames": []}},
        },
        expected_version=pack["version"],
    )
    assert updated_pack is not None

    items = db_instance.list_persona_visual_library_items(user_id="user-1")

    assert len(items) == 1
    item = items[0]
    assert item["id"] == first["id"]
    assert item["source_persona_id"] == persona_id
    assert item["source_pack_id"] == pack["id"]
    assert item["source_persona_name"] == "Research Buddy"
    assert item["source_pack_title"] == "Warm desk assistant"
    assert item["source_persona_name_snapshot"] == "Research Buddy"
    assert item["source_pack_title_snapshot"] == "Warm desk assistant"
    assert item["source_pack_version"] == pack["version"]
    assert item["source_current_version"] == updated_pack["version"]
    assert item["source_changed"] is True
    assert item["source_available"] is True
    assert item["title"] == "Warm desk assistant"
    assert item["notes"] == "Updated notes"
    assert item["tags"] == ["calm"]


def test_stale_source_library_item_lists_unavailable_and_remains_removable(
    db_instance: CharactersRAGDB,
) -> None:
    persona_id, pack = _create_persona_and_pack(db_instance)
    item = db_instance.upsert_persona_visual_library_item(
        user_id="user-1",
        source_persona_id=persona_id,
        source_pack_id=pack["id"],
        title="Reusable helper",
        notes=None,
        tags=[],
    )

    assert db_instance.soft_delete_persona_visual_pack_with_assets(
        pack_id=pack["id"],
        persona_id=persona_id,
        user_id="user-1",
    )

    listed = db_instance.list_persona_visual_library_items(user_id="user-1")

    assert len(listed) == 1
    assert listed[0]["id"] == item["id"]
    assert listed[0]["source_available"] is False
    assert listed[0]["source_persona_name"] == "Research Buddy"
    assert listed[0]["source_pack_title"] == "Warm desk assistant"

    assert db_instance.soft_delete_persona_visual_library_item(
        item_id=item["id"],
        user_id="user-1",
    )
    assert db_instance.list_persona_visual_library_items(user_id="user-1") == []


def test_library_items_are_scoped_to_user(db_instance: CharactersRAGDB) -> None:
    persona_id, pack = _create_persona_and_pack(db_instance)
    item = db_instance.upsert_persona_visual_library_item(
        user_id="user-1",
        source_persona_id=persona_id,
        source_pack_id=pack["id"],
        title="Private helper",
        notes="Private notes",
        tags=["private"],
    )

    assert db_instance.list_persona_visual_library_items(user_id="user-2") == []
    assert (
        db_instance.update_persona_visual_library_item(
            item_id=item["id"],
            user_id="user-2",
            title="Other title",
            notes=None,
            tags=[],
        )
        is None
    )
    assert not db_instance.soft_delete_persona_visual_library_item(
        item_id=item["id"],
        user_id="user-2",
    )
    assert db_instance.get_persona_visual_library_item(
        item_id=item["id"],
        user_id="user-1",
    )["title"] == "Private helper"
