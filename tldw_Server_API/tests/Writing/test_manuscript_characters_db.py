# test_manuscript_characters_db.py
# Unit tests for manuscript character, relationship, and scene-character CRUD.
#
from __future__ import annotations

import json

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
)
from tldw_Server_API.app.core.DB_Management.ManuscriptDB import ManuscriptDBHelper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mdb(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "test.db"), client_id="test_client")
    return ManuscriptDBHelper(db)


def _sync_log_payloads(mdb: ManuscriptDBHelper, entity_id: str) -> list[tuple[str, dict[str, object]]]:
    with mdb.db.transaction() as conn:
        rows = conn.execute(
            """
            SELECT operation, payload
            FROM sync_log
            WHERE entity = 'manuscript_characters' AND entity_id = ?
            ORDER BY rowid
            """,
            (entity_id,),
        ).fetchall()
    return [(row["operation"], json.loads(row["payload"])) for row in rows]


# ---------------------------------------------------------------------------
# Character CRUD
# ---------------------------------------------------------------------------

class TestCharacterCRUD:
    def test_create_and_get(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_character(pid, "Alice", role="protagonist")
        ch = mdb.get_character(cid)
        assert ch is not None
        assert ch["name"] == "Alice"
        assert ch["role"] == "protagonist"
        assert ch["project_id"] == pid
        assert ch["version"] == 1
        assert ch["deleted"] == 0

    def test_create_with_custom_id(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_character(pid, "Bob", character_id="custom-char-1")
        assert cid == "custom-char-1"
        ch = mdb.get_character("custom-char-1")
        assert ch is not None
        assert ch["name"] == "Bob"

    def test_get_missing_returns_none(self, mdb):
        assert mdb.get_character("nonexistent") is None

    def test_list_characters(self, mdb):
        pid = mdb.create_project("Novel")
        mdb.create_character(pid, "Alice", role="protagonist")
        mdb.create_character(pid, "Bob", role="antagonist")
        mdb.create_character(pid, "Charlie", role="supporting")
        chars = mdb.list_characters(pid)
        assert len(chars) == 3

    def test_list_characters_filter_by_role(self, mdb):
        pid = mdb.create_project("Novel")
        mdb.create_character(pid, "Alice", role="protagonist")
        mdb.create_character(pid, "Bob", role="supporting")
        chars = mdb.list_characters(pid, role_filter="protagonist")
        assert len(chars) == 1
        assert chars[0]["name"] == "Alice"

    def test_list_characters_filter_by_cast_group(self, mdb):
        pid = mdb.create_project("Novel")
        mdb.create_character(pid, "Alice", cast_group="heroes")
        mdb.create_character(pid, "Bob", cast_group="villains")
        chars = mdb.list_characters(pid, cast_group_filter="heroes")
        assert len(chars) == 1
        assert chars[0]["name"] == "Alice"

    def test_list_characters_ordered_by_sort_order(self, mdb):
        pid = mdb.create_project("Novel")
        mdb.create_character(pid, "Charlie", sort_order=3)
        mdb.create_character(pid, "Alice", sort_order=1)
        mdb.create_character(pid, "Bob", sort_order=2)
        chars = mdb.list_characters(pid)
        names = [c["name"] for c in chars]
        assert names == ["Alice", "Bob", "Charlie"]

    def test_update_character(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_character(pid, "Alice")
        mdb.update_character(cid, {"name": "Alicia", "backstory": "Born in..."}, expected_version=1)
        ch = mdb.get_character(cid)
        assert ch["name"] == "Alicia"
        assert ch["backstory"] == "Born in..."
        assert ch["version"] == 2

    def test_update_character_version_conflict(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_character(pid, "Alice")
        with pytest.raises(ConflictError):
            mdb.update_character(cid, {"name": "Nope"}, expected_version=99)

    def test_update_character_empty_updates(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_character(pid, "Alice")
        # Should be a no-op
        mdb.update_character(cid, {}, expected_version=1)
        ch = mdb.get_character(cid)
        assert ch["version"] == 1

    def test_soft_delete_character(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_character(pid, "Alice")
        mdb.soft_delete_character(cid, expected_version=1)
        assert mdb.get_character(cid) is None

    def test_soft_delete_character_version_conflict(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_character(pid, "Alice")
        with pytest.raises(ConflictError):
            mdb.soft_delete_character(cid, expected_version=99)

    def test_soft_deleted_not_in_list(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_character(pid, "Alice")
        mdb.create_character(pid, "Bob")
        mdb.soft_delete_character(cid, expected_version=1)
        chars = mdb.list_characters(pid)
        assert len(chars) == 1
        assert chars[0]["name"] == "Bob"

    def test_custom_fields_roundtrip(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_character(pid, "Alice", custom_fields={"hair": "red", "magic": True})
        ch = mdb.get_character(cid)
        assert ch["custom_fields"]["hair"] == "red"
        assert ch["custom_fields"]["magic"] is True

    def test_custom_fields_default_empty(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_character(pid, "Bob")
        ch = mdb.get_character(cid)
        assert ch["custom_fields"] == {}

    def test_update_custom_fields(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_character(pid, "Alice", custom_fields={"hair": "red"})
        mdb.update_character(cid, {"custom_fields": {"hair": "blue", "eyes": "green"}}, expected_version=1)
        ch = mdb.get_character(cid)
        assert ch["custom_fields"]["hair"] == "blue"
        assert ch["custom_fields"]["eyes"] == "green"

    def test_full_character_fields(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_character(
            pid, "Alice",
            role="protagonist",
            cast_group="heroes",
            full_name="Alice Wonderland",
            age="25",
            gender="female",
            appearance="Tall, blonde",
            personality="Curious and brave",
            backstory="Fell down a rabbit hole",
            motivation="Find her way home",
            arc_summary="Learns courage",
            notes="Main character",
        )
        ch = mdb.get_character(cid)
        assert ch["full_name"] == "Alice Wonderland"
        assert ch["age"] == "25"
        assert ch["gender"] == "female"
        assert ch["appearance"] == "Tall, blonde"
        assert ch["personality"] == "Curious and brave"
        assert ch["backstory"] == "Fell down a rabbit hole"
        assert ch["motivation"] == "Find her way home"
        assert ch["arc_summary"] == "Learns courage"
        assert ch["notes"] == "Main character"

    def test_character_sync_log_payload_includes_extended_fields_for_create_update_and_undelete(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_character(
            pid,
            "Alice",
            role="protagonist",
            cast_group="heroes",
            full_name="Alice Liddell",
            age="25",
            gender="female",
            appearance="Tall",
            personality="Curious",
            backstory="Rabbit hole",
            motivation="Get home",
            arc_summary="Learns courage",
            notes="Original notes",
            custom_fields={"hair": "blonde"},
        )

        create_op, create_payload = _sync_log_payloads(mdb, cid)[-1]
        assert create_op == "create"
        assert create_payload["full_name"] == "Alice Liddell"
        assert create_payload["age"] == "25"
        assert create_payload["gender"] == "female"
        assert create_payload["appearance"] == "Tall"
        assert create_payload["personality"] == "Curious"
        assert create_payload["backstory"] == "Rabbit hole"
        assert create_payload["motivation"] == "Get home"
        assert create_payload["arc_summary"] == "Learns courage"
        assert create_payload["notes"] == "Original notes"
        assert json.loads(create_payload["custom_fields_json"]) == {"hair": "blonde"}

        mdb.update_character(
            cid,
            {
                "age": "26",
                "gender": "woman",
                "notes": "Updated notes",
                "custom_fields": {"hair": "auburn", "eyes": "green"},
            },
            expected_version=1,
        )
        update_op, update_payload = _sync_log_payloads(mdb, cid)[-1]
        assert update_op == "update"
        assert update_payload["age"] == "26"
        assert update_payload["gender"] == "woman"
        assert update_payload["notes"] == "Updated notes"
        assert json.loads(update_payload["custom_fields_json"]) == {"hair": "auburn", "eyes": "green"}

        mdb.soft_delete_character(cid, expected_version=2)
        with mdb.db.transaction() as conn:
            conn.execute(
                """
                UPDATE manuscript_characters
                SET deleted = 0, last_modified = CURRENT_TIMESTAMP, version = ?, client_id = ?
                WHERE id = ?
                """,
                (4, mdb.db.client_id, cid),
            )

        undelete_op, undelete_payload = _sync_log_payloads(mdb, cid)[-1]
        assert undelete_op == "update"
        assert undelete_payload["deleted"] == 0
        assert undelete_payload["full_name"] == "Alice Liddell"
        assert undelete_payload["age"] == "26"
        assert undelete_payload["gender"] == "woman"
        assert undelete_payload["appearance"] == "Tall"
        assert undelete_payload["personality"] == "Curious"
        assert undelete_payload["backstory"] == "Rabbit hole"
        assert undelete_payload["motivation"] == "Get home"
        assert undelete_payload["arc_summary"] == "Learns courage"
        assert undelete_payload["notes"] == "Updated notes"
        assert json.loads(undelete_payload["custom_fields_json"]) == {"hair": "auburn", "eyes": "green"}


# ---------------------------------------------------------------------------
# Character Relationships
# ---------------------------------------------------------------------------

class TestCharacterRelationships:
    def test_create_and_list(self, mdb):
        pid = mdb.create_project("Novel")
        c1 = mdb.create_character(pid, "Alice")
        c2 = mdb.create_character(pid, "Bob")
        rel_id = mdb.create_relationship(pid, c1, c2, "sibling", description="Twins")
        rels = mdb.list_relationships(pid)
        assert len(rels) == 1
        assert rels[0]["relationship_type"] == "sibling"
        assert rels[0]["from_character_id"] == c1
        assert rels[0]["to_character_id"] == c2
        assert rels[0]["description"] == "Twins"

    def test_create_with_custom_id(self, mdb):
        pid = mdb.create_project("Novel")
        c1 = mdb.create_character(pid, "Alice")
        c2 = mdb.create_character(pid, "Bob")
        rid = mdb.create_relationship(pid, c1, c2, "friend", relationship_id="rel-1")
        assert rid == "rel-1"

    def test_bidirectional_flag(self, mdb):
        pid = mdb.create_project("Novel")
        c1 = mdb.create_character(pid, "Alice")
        c2 = mdb.create_character(pid, "Bob")
        mdb.create_relationship(pid, c1, c2, "mentor", bidirectional=False)
        rels = mdb.list_relationships(pid)
        assert rels[0]["bidirectional"] == 0

    def test_delete_relationship(self, mdb):
        pid = mdb.create_project("Novel")
        c1 = mdb.create_character(pid, "Alice")
        c2 = mdb.create_character(pid, "Bob")
        rel_id = mdb.create_relationship(pid, c1, c2, "rival")
        mdb.soft_delete_relationship(rel_id, expected_version=1)
        rels = mdb.list_relationships(pid)
        assert len(rels) == 0

    def test_delete_relationship_version_conflict(self, mdb):
        pid = mdb.create_project("Novel")
        c1 = mdb.create_character(pid, "Alice")
        c2 = mdb.create_character(pid, "Bob")
        rel_id = mdb.create_relationship(pid, c1, c2, "rival")
        with pytest.raises(ConflictError):
            mdb.soft_delete_relationship(rel_id, expected_version=99)

    def test_multiple_relationships(self, mdb):
        pid = mdb.create_project("Novel")
        c1 = mdb.create_character(pid, "Alice")
        c2 = mdb.create_character(pid, "Bob")
        c3 = mdb.create_character(pid, "Charlie")
        mdb.create_relationship(pid, c1, c2, "friend")
        mdb.create_relationship(pid, c1, c3, "rival")
        mdb.create_relationship(pid, c2, c3, "sibling")
        rels = mdb.list_relationships(pid)
        assert len(rels) == 3

    def test_get_relationship(self, mdb):
        pid = mdb.create_project("Novel")
        c1 = mdb.create_character(pid, "Alice")
        c2 = mdb.create_character(pid, "Bob")
        rel_id = mdb.create_relationship(pid, c1, c2, "sibling", description="Twins")
        rel = mdb.get_relationship(rel_id)
        assert rel is not None
        assert rel["relationship_type"] == "sibling"
        assert rel["from_character_id"] == c1
        assert rel["to_character_id"] == c2
        assert rel["description"] == "Twins"

    def test_get_relationship_missing(self, mdb):
        assert mdb.get_relationship("nonexistent") is None

    def test_get_relationship_deleted(self, mdb):
        pid = mdb.create_project("Novel")
        c1 = mdb.create_character(pid, "Alice")
        c2 = mdb.create_character(pid, "Bob")
        rel_id = mdb.create_relationship(pid, c1, c2, "rival")
        mdb.soft_delete_relationship(rel_id, expected_version=1)
        assert mdb.get_relationship(rel_id) is None

    def test_update_character_rejects_unknown_column(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_character(pid, "Alice")
        with pytest.raises(ValueError, match="Unknown update column"):
            mdb.update_character(cid, {"malicious_col": "x"}, expected_version=1)

# ---------------------------------------------------------------------------
# Scene-Character Linking
# ---------------------------------------------------------------------------

class TestSceneCharacterLinking:
    def test_link_table_has_sync_metadata_columns(self, mdb):
        columns = {
            row["name"]
            for row in mdb.db.execute_query(
                "PRAGMA table_info('manuscript_scene_characters')"
            ).fetchall()
        }

        assert {"deleted", "client_id", "version", "last_modified"}.issubset(columns)

    def test_link_and_list(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        scene_id = mdb.create_scene(ch_id, pid, title="Opening", content_plain="hello")
        char_id = mdb.create_character(pid, "Alice", role="protagonist")
        mdb.link_scene_character(scene_id, char_id)
        linked = mdb.list_scene_characters(scene_id)
        assert len(linked) == 1
        assert linked[0]["character_id"] == char_id
        assert linked[0]["name"] == "Alice"
        assert linked[0]["role"] == "protagonist"

    def test_link_with_pov(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        scene_id = mdb.create_scene(ch_id, pid, title="Opening", content_plain="hello")
        char_id = mdb.create_character(pid, "Alice")
        mdb.link_scene_character(scene_id, char_id, is_pov=True)
        linked = mdb.list_scene_characters(scene_id)
        assert linked[0]["is_pov"] == 1

    def test_relink_updates_is_pov(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        scene_id = mdb.create_scene(ch_id, pid, title="Opening", content_plain="hello")
        char_id = mdb.create_character(pid, "Alice")

        mdb.link_scene_character(scene_id, char_id, is_pov=False)
        mdb.link_scene_character(scene_id, char_id, is_pov=True)

        linked = mdb.list_scene_characters(scene_id)
        assert linked[0]["is_pov"] == 1

    def test_link_writes_sync_log_entry(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        scene_id = mdb.create_scene(ch_id, pid, title="Opening", content_plain="hello")
        char_id = mdb.create_character(pid, "Alice")

        before = mdb.db.execute_query(
            "SELECT COUNT(*) AS count FROM sync_log WHERE entity = ?",
            ("manuscript_scene_characters",),
        ).fetchone()["count"]

        mdb.link_scene_character(scene_id, char_id, is_pov=True)

        after = mdb.db.execute_query(
            "SELECT COUNT(*) AS count FROM sync_log WHERE entity = ?",
            ("manuscript_scene_characters",),
        ).fetchone()["count"]
        assert after == before + 1

    def test_link_idempotent(self, mdb):
        """INSERT OR IGNORE should not raise on duplicate."""
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        scene_id = mdb.create_scene(ch_id, pid, title="Opening", content_plain="hello")
        char_id = mdb.create_character(pid, "Alice")
        mdb.link_scene_character(scene_id, char_id)
        mdb.link_scene_character(scene_id, char_id)  # Should not raise
        linked = mdb.list_scene_characters(scene_id)
        assert len(linked) == 1

    def test_unlink(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        scene_id = mdb.create_scene(ch_id, pid, title="Opening", content_plain="hello")
        char_id = mdb.create_character(pid, "Alice")
        mdb.link_scene_character(scene_id, char_id)
        mdb.unlink_scene_character(scene_id, char_id)
        assert len(mdb.list_scene_characters(scene_id)) == 0

    def test_multiple_characters_per_scene(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        scene_id = mdb.create_scene(ch_id, pid, title="Opening", content_plain="hello")
        c1 = mdb.create_character(pid, "Alice")
        c2 = mdb.create_character(pid, "Bob")
        mdb.link_scene_character(scene_id, c1)
        mdb.link_scene_character(scene_id, c2)
        linked = mdb.list_scene_characters(scene_id)
        assert len(linked) == 2

    def test_deleted_character_excluded_from_list(self, mdb):
        """If a character is soft-deleted, it should not appear in scene links."""
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        scene_id = mdb.create_scene(ch_id, pid, title="Opening", content_plain="hello")
        char_id = mdb.create_character(pid, "Alice")
        mdb.link_scene_character(scene_id, char_id)
        mdb.soft_delete_character(char_id, expected_version=1)
        linked = mdb.list_scene_characters(scene_id)
        assert len(linked) == 0
