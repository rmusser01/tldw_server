# test_manuscript_world_plot_db.py
# Unit tests for world info, plot tracking, and citations CRUD.
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


def _sync_log_payloads(mdb: ManuscriptDBHelper, entity: str, entity_id: str) -> list[tuple[str, dict[str, object]]]:
    with mdb.db.transaction() as conn:
        rows = conn.execute(
            """
            SELECT operation, payload
            FROM sync_log
            WHERE entity = ? AND entity_id = ?
            ORDER BY rowid
            """,
            (entity, entity_id),
        ).fetchall()
    return [(row["operation"], json.loads(row["payload"])) for row in rows]


# ---------------------------------------------------------------------------
# World Info CRUD
# ---------------------------------------------------------------------------

class TestWorldInfoCRUD:
    def test_create_and_get(self, mdb):
        pid = mdb.create_project("Novel")
        wid = mdb.create_world_info(pid, kind="location", name="Mordor", description="Dark land")
        wi = mdb.get_world_info(wid)
        assert wi is not None
        assert wi["name"] == "Mordor"
        assert wi["kind"] == "location"
        assert wi["description"] == "Dark land"
        assert wi["project_id"] == pid
        assert wi["version"] == 1

    def test_create_with_custom_id(self, mdb):
        pid = mdb.create_project("Novel")
        wid = mdb.create_world_info(pid, kind="item", name="Ring", world_info_id="wi-1")
        assert wid == "wi-1"

    def test_get_missing_returns_none(self, mdb):
        assert mdb.get_world_info("nonexistent") is None

    def test_list_all(self, mdb):
        pid = mdb.create_project("Novel")
        mdb.create_world_info(pid, kind="location", name="Mordor")
        mdb.create_world_info(pid, kind="item", name="Ring")
        mdb.create_world_info(pid, kind="location", name="Shire")
        items = mdb.list_world_info(pid)
        assert len(items) == 3

    def test_list_by_kind(self, mdb):
        pid = mdb.create_project("Novel")
        mdb.create_world_info(pid, kind="location", name="Mordor")
        mdb.create_world_info(pid, kind="item", name="Ring")
        mdb.create_world_info(pid, kind="location", name="Shire")
        items = mdb.list_world_info(pid, kind_filter="location")
        assert len(items) == 2
        names = {i["name"] for i in items}
        assert names == {"Mordor", "Shire"}

    def test_list_ordered_by_sort_order(self, mdb):
        pid = mdb.create_project("Novel")
        mdb.create_world_info(pid, kind="location", name="Shire", sort_order=2)
        mdb.create_world_info(pid, kind="location", name="Mordor", sort_order=1)
        items = mdb.list_world_info(pid)
        assert items[0]["name"] == "Mordor"
        assert items[1]["name"] == "Shire"

    def test_hierarchical(self, mdb):
        pid = mdb.create_project("Novel")
        parent = mdb.create_world_info(pid, kind="location", name="Middle Earth")
        child = mdb.create_world_info(pid, kind="location", name="Shire", parent_id=parent)
        wi = mdb.get_world_info(child)
        assert wi["parent_id"] == parent

    def test_properties_and_tags(self, mdb):
        pid = mdb.create_project("Novel")
        wid = mdb.create_world_info(
            pid, kind="item", name="Ring",
            properties={"power": "invisibility"}, tags=["artifact", "danger"],
        )
        wi = mdb.get_world_info(wid)
        assert wi["properties"]["power"] == "invisibility"
        assert "artifact" in wi["tags"]
        assert "danger" in wi["tags"]

    def test_world_info_sync_payload_includes_properties_and_tags(self, mdb):
        pid = mdb.create_project("Novel")
        wid = mdb.create_world_info(
            pid,
            kind="item",
            name="Ring",
            properties={"power": "invisibility"},
            tags=["artifact", "danger"],
        )

        row = mdb.db.execute_query(
            "SELECT payload FROM sync_log WHERE entity = ? AND entity_id = ? "
            "ORDER BY change_id DESC LIMIT 1",
            ("manuscript_world_info", wid),
        ).fetchone()

        payload = json.loads(row["payload"])
        assert payload["properties_json"] == json.dumps({"power": "invisibility"})
        assert payload["tags_json"] == json.dumps(["artifact", "danger"])

    def test_properties_default_empty(self, mdb):
        pid = mdb.create_project("Novel")
        wid = mdb.create_world_info(pid, kind="location", name="Mordor")
        wi = mdb.get_world_info(wid)
        assert wi["properties"] == {}
        assert wi["tags"] == []

    def test_update_world_info(self, mdb):
        pid = mdb.create_project("Novel")
        wid = mdb.create_world_info(pid, kind="faction", name="Elves")
        mdb.update_world_info(wid, {"description": "Ancient race"}, expected_version=1)
        wi = mdb.get_world_info(wid)
        assert wi["description"] == "Ancient race"
        assert wi["version"] == 2

    def test_update_properties_and_tags(self, mdb):
        pid = mdb.create_project("Novel")
        wid = mdb.create_world_info(pid, kind="item", name="Sword")
        mdb.update_world_info(
            wid,
            {"properties": {"material": "steel"}, "tags": ["weapon"]},
            expected_version=1,
        )
        wi = mdb.get_world_info(wid)
        assert wi["properties"]["material"] == "steel"
        assert wi["tags"] == ["weapon"]

    def test_update_world_info_rejects_cross_project_parent(self, mdb):
        left = mdb.create_project("Left")
        right = mdb.create_project("Right")
        local = mdb.create_world_info(left, kind="location", name="Local")
        foreign_parent = mdb.create_world_info(right, kind="location", name="Foreign")

        with pytest.raises(ConflictError, match="different project"):
            mdb.update_world_info(local, {"parent_id": foreign_parent}, expected_version=1)

    def test_update_version_conflict(self, mdb):
        pid = mdb.create_project("Novel")
        wid = mdb.create_world_info(pid, kind="location", name="Mordor")
        with pytest.raises(ConflictError):
            mdb.update_world_info(wid, {"name": "Nope"}, expected_version=99)

    def test_update_empty_updates(self, mdb):
        pid = mdb.create_project("Novel")
        wid = mdb.create_world_info(pid, kind="location", name="Mordor")
        mdb.update_world_info(wid, {}, expected_version=1)
        wi = mdb.get_world_info(wid)
        assert wi["version"] == 1

    def test_soft_delete(self, mdb):
        pid = mdb.create_project("Novel")
        wid = mdb.create_world_info(pid, kind="faction", name="Elves")
        mdb.update_world_info(wid, {"description": "Ancient race"}, expected_version=1)
        mdb.soft_delete_world_info(wid, expected_version=2)
        assert mdb.get_world_info(wid) is None

    def test_soft_delete_version_conflict(self, mdb):
        pid = mdb.create_project("Novel")
        wid = mdb.create_world_info(pid, kind="location", name="Mordor")
        with pytest.raises(ConflictError):
            mdb.soft_delete_world_info(wid, expected_version=99)

    def test_soft_deleted_not_in_list(self, mdb):
        pid = mdb.create_project("Novel")
        mdb.create_world_info(pid, kind="location", name="Mordor")
        wid2 = mdb.create_world_info(pid, kind="location", name="Shire")
        mdb.soft_delete_world_info(wid2, expected_version=1)
        items = mdb.list_world_info(pid)
        assert len(items) == 1
        assert items[0]["name"] == "Mordor"

    def test_scene_linking(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        scene_id = mdb.create_scene(ch_id, pid, title="S1", content_plain="text")
        wid = mdb.create_world_info(pid, kind="location", name="Shire")
        mdb.link_scene_world_info(scene_id, wid)
        linked = mdb.list_scene_world_info(scene_id)
        assert len(linked) == 1
        assert linked[0]["world_info_id"] == wid
        assert linked[0]["name"] == "Shire"
        assert linked[0]["kind"] == "location"

    def test_scene_world_info_link_table_has_sync_metadata_columns(self, mdb):
        columns = {
            row["name"]
            for row in mdb.db.execute_query(
                "PRAGMA table_info('manuscript_scene_world_info')"
            ).fetchall()
        }

        assert {"deleted", "client_id", "version", "last_modified"}.issubset(columns)

    def test_scene_world_info_link_writes_sync_log_entry(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        scene_id = mdb.create_scene(ch_id, pid, title="S1", content_plain="text")
        wid = mdb.create_world_info(pid, kind="location", name="Shire")

        before = mdb.db.execute_query(
            "SELECT COUNT(*) AS count FROM sync_log WHERE entity = ?",
            ("manuscript_scene_world_info",),
        ).fetchone()["count"]

        mdb.link_scene_world_info(scene_id, wid)

        after = mdb.db.execute_query(
            "SELECT COUNT(*) AS count FROM sync_log WHERE entity = ?",
            ("manuscript_scene_world_info",),
        ).fetchone()["count"]
        assert after == before + 1

    def test_scene_unlink(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        scene_id = mdb.create_scene(ch_id, pid, title="S1", content_plain="text")
        wid = mdb.create_world_info(pid, kind="location", name="Shire")
        mdb.link_scene_world_info(scene_id, wid)
        mdb.unlink_scene_world_info(scene_id, wid)
        assert len(mdb.list_scene_world_info(scene_id)) == 0

    def test_scene_link_idempotent(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        scene_id = mdb.create_scene(ch_id, pid, title="S1", content_plain="text")
        wid = mdb.create_world_info(pid, kind="location", name="Shire")
        mdb.link_scene_world_info(scene_id, wid)
        mdb.link_scene_world_info(scene_id, wid)  # Should not raise
        assert len(mdb.list_scene_world_info(scene_id)) == 1

    def test_deleted_world_info_excluded_from_scene_list(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        scene_id = mdb.create_scene(ch_id, pid, title="S1", content_plain="text")
        wid = mdb.create_world_info(pid, kind="location", name="Shire")
        mdb.link_scene_world_info(scene_id, wid)
        mdb.soft_delete_world_info(wid, expected_version=1)
        assert len(mdb.list_scene_world_info(scene_id)) == 0

    def test_world_info_sync_log_payload_includes_properties_and_tags(self, mdb):
        pid = mdb.create_project("Novel")
        wid = mdb.create_world_info(
            pid,
            kind="item",
            name="Ring",
            properties={"power": "invisibility"},
            tags=["artifact", "danger"],
        )

        create_op, create_payload = _sync_log_payloads(mdb, "manuscript_world_info", wid)[-1]
        assert create_op == "create"
        assert json.loads(create_payload["properties_json"]) == {"power": "invisibility"}
        assert json.loads(create_payload["tags_json"]) == ["artifact", "danger"]

        mdb.update_world_info(
            wid,
            {"properties": {"power": "dominion"}, "tags": ["artifact", "cursed"]},
            expected_version=1,
        )
        update_op, update_payload = _sync_log_payloads(mdb, "manuscript_world_info", wid)[-1]
        assert update_op == "update"
        assert json.loads(update_payload["properties_json"]) == {"power": "dominion"}
        assert json.loads(update_payload["tags_json"]) == ["artifact", "cursed"]

        mdb.soft_delete_world_info(wid, expected_version=2)
        with mdb.db.transaction() as conn:
            conn.execute(
                """
                UPDATE manuscript_world_info
                SET deleted = 0, last_modified = CURRENT_TIMESTAMP, version = ?, client_id = ?
                WHERE id = ?
                """,
                (4, mdb.db.client_id, wid),
            )

        undelete_op, undelete_payload = _sync_log_payloads(mdb, "manuscript_world_info", wid)[-1]
        assert undelete_op == "update"
        assert undelete_payload["deleted"] == 0
        assert json.loads(undelete_payload["properties_json"]) == {"power": "dominion"}
        assert json.loads(undelete_payload["tags_json"]) == ["artifact", "cursed"]


# ---------------------------------------------------------------------------
# Plot Line CRUD
# ---------------------------------------------------------------------------

class TestPlotLineCRUD:
    def test_create_and_list(self, mdb):
        pid = mdb.create_project("Novel")
        pl_id = mdb.create_plot_line(pid, "Main Quest", description="Destroy the ring")
        lines = mdb.list_plot_lines(pid)
        assert len(lines) == 1
        assert lines[0]["title"] == "Main Quest"
        assert lines[0]["description"] == "Destroy the ring"
        assert lines[0]["status"] == "active"

    def test_create_with_custom_id(self, mdb):
        pid = mdb.create_project("Novel")
        pl_id = mdb.create_plot_line(pid, "Side Quest", plot_line_id="pl-1")
        assert pl_id == "pl-1"

    def test_get_plot_line(self, mdb):
        pid = mdb.create_project("Novel")
        pl_id = mdb.create_plot_line(pid, "Main Quest")
        pl = mdb.get_plot_line(pl_id)
        assert pl is not None
        assert pl["title"] == "Main Quest"
        assert pl["version"] == 1

    def test_get_missing_returns_none(self, mdb):
        assert mdb.get_plot_line("nonexistent") is None

    def test_list_ordered_by_sort_order(self, mdb):
        pid = mdb.create_project("Novel")
        mdb.create_plot_line(pid, "Side Quest", sort_order=2)
        mdb.create_plot_line(pid, "Main Quest", sort_order=1)
        lines = mdb.list_plot_lines(pid)
        assert lines[0]["title"] == "Main Quest"
        assert lines[1]["title"] == "Side Quest"

    def test_update_status(self, mdb):
        pid = mdb.create_project("Novel")
        pl_id = mdb.create_plot_line(pid, "Side Quest")
        mdb.update_plot_line(pl_id, {"status": "resolved"}, expected_version=1)
        pl = mdb.get_plot_line(pl_id)
        assert pl["status"] == "resolved"
        assert pl["version"] == 2

    def test_update_version_conflict(self, mdb):
        pid = mdb.create_project("Novel")
        pl_id = mdb.create_plot_line(pid, "Quest")
        with pytest.raises(ConflictError):
            mdb.update_plot_line(pl_id, {"title": "Nope"}, expected_version=99)

    def test_soft_delete(self, mdb):
        pid = mdb.create_project("Novel")
        pl_id = mdb.create_plot_line(pid, "Doomed Quest")
        mdb.soft_delete_plot_line(pl_id, expected_version=1)
        assert mdb.get_plot_line(pl_id) is None
        lines = mdb.list_plot_lines(pid)
        assert len(lines) == 0

    def test_soft_delete_version_conflict(self, mdb):
        pid = mdb.create_project("Novel")
        pl_id = mdb.create_plot_line(pid, "Quest")
        with pytest.raises(ConflictError):
            mdb.soft_delete_plot_line(pl_id, expected_version=99)

    def test_with_color(self, mdb):
        pid = mdb.create_project("Novel")
        pl_id = mdb.create_plot_line(pid, "Romance", color="#ff0000")
        pl = mdb.get_plot_line(pl_id)
        assert pl["color"] == "#ff0000"


# ---------------------------------------------------------------------------
# Plot Event CRUD
# ---------------------------------------------------------------------------

class TestPlotEventCRUD:
    def test_create_and_list(self, mdb):
        pid = mdb.create_project("Novel")
        pl_id = mdb.create_plot_line(pid, "Main Quest")
        pe_id = mdb.create_plot_event(pid, pl_id, "Ring found", event_type="setup")
        events = mdb.list_plot_events(pl_id)
        assert len(events) == 1
        assert events[0]["title"] == "Ring found"
        assert events[0]["event_type"] == "setup"

    def test_create_with_custom_id(self, mdb):
        pid = mdb.create_project("Novel")
        pl_id = mdb.create_plot_line(pid, "Main Quest")
        eid = mdb.create_plot_event(pid, pl_id, "Event", event_id="pe-1")
        assert eid == "pe-1"

    def test_list_ordered_by_sort_order(self, mdb):
        pid = mdb.create_project("Novel")
        pl_id = mdb.create_plot_line(pid, "Main Quest")
        mdb.create_plot_event(pid, pl_id, "Event B", sort_order=2)
        mdb.create_plot_event(pid, pl_id, "Event A", sort_order=1)
        events = mdb.list_plot_events(pl_id)
        assert events[0]["title"] == "Event A"
        assert events[1]["title"] == "Event B"

    def test_event_with_scene_and_chapter(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        scene_id = mdb.create_scene(ch_id, pid, title="S1", content_plain="text")
        pl_id = mdb.create_plot_line(pid, "Main Quest")
        pe_id = mdb.create_plot_event(
            pid, pl_id, "Key moment",
            scene_id=scene_id, chapter_id=ch_id,
        )
        events = mdb.list_plot_events(pl_id)
        assert events[0]["scene_id"] == scene_id
        assert events[0]["chapter_id"] == ch_id

    def test_update_plot_event(self, mdb):
        pid = mdb.create_project("Novel")
        pl_id = mdb.create_plot_line(pid, "Main Quest")
        pe_id = mdb.create_plot_event(pid, pl_id, "Old Title")
        mdb.update_plot_event(pe_id, {"title": "New Title", "description": "Updated"}, expected_version=1)
        events = mdb.list_plot_events(pl_id)
        assert events[0]["title"] == "New Title"
        assert events[0]["description"] == "Updated"

    def test_update_version_conflict(self, mdb):
        pid = mdb.create_project("Novel")
        pl_id = mdb.create_plot_line(pid, "Main Quest")
        pe_id = mdb.create_plot_event(pid, pl_id, "Event")
        with pytest.raises(ConflictError):
            mdb.update_plot_event(pe_id, {"title": "Nope"}, expected_version=99)

    def test_update_plot_event_rejects_scene_chapter_mismatch(self, mdb):
        pid = mdb.create_project("Novel")
        chapter_one = mdb.create_chapter(pid, "Ch1")
        chapter_two = mdb.create_chapter(pid, "Ch2", sort_order=2)
        scene_one = mdb.create_scene(chapter_one, pid, title="S1", content_plain="one")
        scene_two = mdb.create_scene(chapter_two, pid, title="S2", content_plain="two")
        pl_id = mdb.create_plot_line(pid, "Main Quest")
        pe_id = mdb.create_plot_event(
            pid,
            pl_id,
            "Key moment",
            scene_id=scene_one,
            chapter_id=chapter_one,
        )

        with pytest.raises(ValueError, match="does not belong to chapter"):
            mdb.update_plot_event(pe_id, {"scene_id": scene_two}, expected_version=1)

    def test_update_plot_event_rejects_cross_project_plot_line(self, mdb):
        pid_one = mdb.create_project("Novel One")
        pid_two = mdb.create_project("Novel Two")
        plot_line_one = mdb.create_plot_line(pid_one, "Main Quest")
        plot_line_two = mdb.create_plot_line(pid_two, "Other Quest")
        pe_id = mdb.create_plot_event(pid_one, plot_line_one, "Key moment")

        with pytest.raises(ConflictError, match="different project"):
            mdb.update_plot_event(pe_id, {"plot_line_id": plot_line_two}, expected_version=1)

    def test_update_plot_event_plot_line_change_writes_sync_log(self, mdb):
        pid = mdb.create_project("Novel")
        plot_line_one = mdb.create_plot_line(pid, "Main Quest")
        plot_line_two = mdb.create_plot_line(pid, "Side Quest", sort_order=2)
        pe_id = mdb.create_plot_event(pid, plot_line_one, "Key moment")
        before = mdb.db.execute_query(
            "SELECT COUNT(*) AS count FROM sync_log WHERE entity = ?",
            ("manuscript_plot_events",),
        ).fetchone()["count"]

        mdb.update_plot_event(pe_id, {"plot_line_id": plot_line_two}, expected_version=1)

        after = mdb.db.execute_query(
            "SELECT COUNT(*) AS count FROM sync_log WHERE entity = ?",
            ("manuscript_plot_events",),
        ).fetchone()["count"]
        assert after == before + 1

    def test_soft_delete(self, mdb):
        pid = mdb.create_project("Novel")
        pl_id = mdb.create_plot_line(pid, "Main Quest")
        pe_id = mdb.create_plot_event(pid, pl_id, "Doomed Event")
        mdb.soft_delete_plot_event(pe_id, expected_version=1)
        events = mdb.list_plot_events(pl_id)
        assert len(events) == 0

    def test_soft_delete_version_conflict(self, mdb):
        pid = mdb.create_project("Novel")
        pl_id = mdb.create_plot_line(pid, "Quest")
        pe_id = mdb.create_plot_event(pid, pl_id, "Event")
        with pytest.raises(ConflictError):
            mdb.soft_delete_plot_event(pe_id, expected_version=99)

    def test_default_event_type(self, mdb):
        pid = mdb.create_project("Novel")
        pl_id = mdb.create_plot_line(pid, "Main Quest")
        mdb.create_plot_event(pid, pl_id, "Generic Event")
        events = mdb.list_plot_events(pl_id)
        assert events[0]["event_type"] == "plot"

    def test_get_plot_event(self, mdb):
        pid = mdb.create_project("Novel")
        pl_id = mdb.create_plot_line(pid, "Main Quest")
        pe_id = mdb.create_plot_event(pid, pl_id, "Ring found", event_type="setup")
        event = mdb.get_plot_event(pe_id)
        assert event is not None
        assert event["title"] == "Ring found"
        assert event["event_type"] == "setup"

    def test_get_plot_event_missing(self, mdb):
        assert mdb.get_plot_event("nonexistent") is None

    def test_get_plot_event_deleted(self, mdb):
        pid = mdb.create_project("Novel")
        pl_id = mdb.create_plot_line(pid, "Quest")
        pe_id = mdb.create_plot_event(pid, pl_id, "Event")
        mdb.soft_delete_plot_event(pe_id, expected_version=1)
        assert mdb.get_plot_event(pe_id) is None


# ---------------------------------------------------------------------------
# Plot Hole CRUD
# ---------------------------------------------------------------------------

class TestPlotHoleCRUD:
    def test_create_and_list(self, mdb):
        pid = mdb.create_project("Novel")
        ph_id = mdb.create_plot_hole(pid, "Timeline inconsistency", severity="high")
        holes = mdb.list_plot_holes(pid)
        assert len(holes) == 1
        assert holes[0]["severity"] == "high"
        assert holes[0]["status"] == "open"
        assert holes[0]["title"] == "Timeline inconsistency"

    def test_create_with_custom_id(self, mdb):
        pid = mdb.create_project("Novel")
        ph_id = mdb.create_plot_hole(pid, "Gap", plot_hole_id="ph-1")
        assert ph_id == "ph-1"

    def test_get_plot_hole(self, mdb):
        pid = mdb.create_project("Novel")
        ph_id = mdb.create_plot_hole(pid, "Gap", severity="low")
        ph = mdb.get_plot_hole(ph_id)
        assert ph is not None
        assert ph["title"] == "Gap"
        assert ph["severity"] == "low"

    def test_get_missing_returns_none(self, mdb):
        assert mdb.get_plot_hole("nonexistent") is None

    def test_list_with_status_filter(self, mdb):
        pid = mdb.create_project("Novel")
        mdb.create_plot_hole(pid, "Open hole")
        ph2 = mdb.create_plot_hole(pid, "Resolved hole")
        mdb.update_plot_hole(ph2, {"status": "resolved"}, expected_version=1)
        open_holes = mdb.list_plot_holes(pid, status_filter="open")
        assert len(open_holes) == 1
        assert open_holes[0]["title"] == "Open hole"

    def test_resolve_plot_hole(self, mdb):
        pid = mdb.create_project("Novel")
        ph_id = mdb.create_plot_hole(pid, "Plot gap")
        mdb.update_plot_hole(ph_id, {"status": "resolved", "resolution": "Added scene"}, expected_version=1)
        ph = mdb.get_plot_hole(ph_id)
        assert ph["status"] == "resolved"
        assert ph["resolution"] == "Added scene"
        assert ph["version"] == 2

    def test_update_version_conflict(self, mdb):
        pid = mdb.create_project("Novel")
        ph_id = mdb.create_plot_hole(pid, "Gap")
        with pytest.raises(ConflictError):
            mdb.update_plot_hole(ph_id, {"title": "Nope"}, expected_version=99)

    def test_update_plot_hole_rejects_scene_chapter_mismatch(self, mdb):
        pid = mdb.create_project("Novel")
        chapter_one = mdb.create_chapter(pid, "Ch1")
        chapter_two = mdb.create_chapter(pid, "Ch2", sort_order=2)
        scene_one = mdb.create_scene(chapter_one, pid, title="S1", content_plain="one")
        scene_two = mdb.create_scene(chapter_two, pid, title="S2", content_plain="two")
        ph_id = mdb.create_plot_hole(
            pid,
            "Gap",
            scene_id=scene_one,
            chapter_id=chapter_one,
        )

        with pytest.raises(ValueError, match="does not belong to chapter"):
            mdb.update_plot_hole(ph_id, {"scene_id": scene_two}, expected_version=1)

    def test_update_plot_hole_detected_by_change_writes_sync_log(self, mdb):
        pid = mdb.create_project("Novel")
        ph_id = mdb.create_plot_hole(pid, "Gap", detected_by="manual")
        before = mdb.db.execute_query(
            "SELECT COUNT(*) AS count FROM sync_log WHERE entity = ?",
            ("manuscript_plot_holes",),
        ).fetchone()["count"]

        mdb.update_plot_hole(ph_id, {"detected_by": "ai"}, expected_version=1)

        after = mdb.db.execute_query(
            "SELECT COUNT(*) AS count FROM sync_log WHERE entity = ?",
            ("manuscript_plot_holes",),
        ).fetchone()["count"]
        assert after == before + 1

    def test_soft_delete(self, mdb):
        pid = mdb.create_project("Novel")
        ph_id = mdb.create_plot_hole(pid, "Doomed hole")
        mdb.soft_delete_plot_hole(ph_id, expected_version=1)
        assert mdb.get_plot_hole(ph_id) is None
        holes = mdb.list_plot_holes(pid)
        assert len(holes) == 0

    def test_soft_delete_version_conflict(self, mdb):
        pid = mdb.create_project("Novel")
        ph_id = mdb.create_plot_hole(pid, "Gap")
        with pytest.raises(ConflictError):
            mdb.soft_delete_plot_hole(ph_id, expected_version=99)

    def test_with_references(self, mdb):
        """Plot hole can reference scene, chapter, and plot line."""
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        scene_id = mdb.create_scene(ch_id, pid, title="S1", content_plain="text")
        pl_id = mdb.create_plot_line(pid, "Quest")
        ph_id = mdb.create_plot_hole(
            pid, "Inconsistency",
            scene_id=scene_id, chapter_id=ch_id, plot_line_id=pl_id,
        )
        ph = mdb.get_plot_hole(ph_id)
        assert ph["scene_id"] == scene_id
        assert ph["chapter_id"] == ch_id
        assert ph["plot_line_id"] == pl_id

    def test_detected_by_ai(self, mdb):
        pid = mdb.create_project("Novel")
        ph_id = mdb.create_plot_hole(pid, "AI found this", detected_by="ai")
        ph = mdb.get_plot_hole(ph_id)
        assert ph["detected_by"] == "ai"

    def test_default_detected_by(self, mdb):
        pid = mdb.create_project("Novel")
        ph_id = mdb.create_plot_hole(pid, "Manual hole")
        ph = mdb.get_plot_hole(ph_id)
        assert ph["detected_by"] == "manual"


# ---------------------------------------------------------------------------
# Citation CRUD
# ---------------------------------------------------------------------------

class TestCitationCRUD:
    def test_create_and_list(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        scene_id = mdb.create_scene(ch_id, pid, title="S1", content_plain="text")
        cit_id = mdb.create_citation(
            pid, scene_id, source_type="media_db",
            source_title="Wikipedia", excerpt="Key fact",
        )
        cits = mdb.list_citations(scene_id)
        assert len(cits) == 1
        assert cits[0]["source_title"] == "Wikipedia"
        assert cits[0]["source_type"] == "media_db"
        assert cits[0]["excerpt"] == "Key fact"

    def test_create_with_custom_id(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        scene_id = mdb.create_scene(ch_id, pid, title="S1", content_plain="text")
        cit_id = mdb.create_citation(pid, scene_id, source_type="note", citation_id="cit-1")
        assert cit_id == "cit-1"

    def test_create_with_all_fields(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        scene_id = mdb.create_scene(ch_id, pid, title="S1", content_plain="text")
        cit_id = mdb.create_citation(
            pid, scene_id, source_type="rag",
            source_id="media-123", source_title="Research Paper",
            excerpt="Important finding", query_used="medieval weapons",
            anchor_offset=42,
        )
        cits = mdb.list_citations(scene_id)
        assert cits[0]["source_id"] == "media-123"
        assert cits[0]["query_used"] == "medieval weapons"
        assert cits[0]["anchor_offset"] == 42

    def test_multiple_citations_per_scene(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        scene_id = mdb.create_scene(ch_id, pid, title="S1", content_plain="text")
        mdb.create_citation(pid, scene_id, source_type="media_db", source_title="Source 1")
        mdb.create_citation(pid, scene_id, source_type="note", source_title="Source 2")
        cits = mdb.list_citations(scene_id)
        assert len(cits) == 2

    def test_delete_citation(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        scene_id = mdb.create_scene(ch_id, pid, title="S1", content_plain="text")
        cit_id = mdb.create_citation(pid, scene_id, source_type="note", source_title="My notes")
        mdb.soft_delete_citation(cit_id, expected_version=1)
        assert len(mdb.list_citations(scene_id)) == 0

    def test_delete_citation_version_conflict(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        scene_id = mdb.create_scene(ch_id, pid, title="S1", content_plain="text")
        cit_id = mdb.create_citation(pid, scene_id, source_type="note")
        with pytest.raises(ConflictError):
            mdb.soft_delete_citation(cit_id, expected_version=99)

    def test_get_citation(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        scene_id = mdb.create_scene(ch_id, pid, title="S1", content_plain="text")
        cit_id = mdb.create_citation(
            pid, scene_id, source_type="media_db",
            source_title="Wikipedia", excerpt="Key fact",
        )
        cit = mdb.get_citation(cit_id)
        assert cit is not None
        assert cit["source_title"] == "Wikipedia"
        assert cit["source_type"] == "media_db"
        assert cit["excerpt"] == "Key fact"

    def test_get_citation_missing(self, mdb):
        assert mdb.get_citation("nonexistent") is None

    def test_get_citation_deleted(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        scene_id = mdb.create_scene(ch_id, pid, title="S1", content_plain="text")
        cit_id = mdb.create_citation(pid, scene_id, source_type="note")
        mdb.soft_delete_citation(cit_id, expected_version=1)
        assert mdb.get_citation(cit_id) is None

    def test_citations_scoped_to_scene(self, mdb):
        """Citations listed for one scene should not include those from another."""
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        s1 = mdb.create_scene(ch_id, pid, title="S1", content_plain="text")
        s2 = mdb.create_scene(ch_id, pid, title="S2", content_plain="other")
        mdb.create_citation(pid, s1, source_type="note", source_title="For S1")
        mdb.create_citation(pid, s2, source_type="note", source_title="For S2")
        cits_s1 = mdb.list_citations(s1)
        assert len(cits_s1) == 1
        assert cits_s1[0]["source_title"] == "For S1"
