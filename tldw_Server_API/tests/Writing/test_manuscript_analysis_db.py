# test_manuscript_analysis_db.py
# Tests for manuscript AI analysis CRUD.
#
from __future__ import annotations

import json

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
)
from tldw_Server_API.app.core.DB_Management.ManuscriptDB import ManuscriptDBHelper


@pytest.fixture()
def mdb(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "test.db"), client_id="test_client")
    return ManuscriptDBHelper(db)


def _analysis_row(mdb, analysis_id):
    return mdb.db.execute_query(
        "SELECT * FROM manuscript_ai_analyses WHERE id = ?",
        (analysis_id,),
    ).fetchone()


class TestAnalysisCRUD:
    def test_create_and_get(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        scene_id = mdb.create_scene(ch_id, pid, title="S1", content_json="{}", content_plain="text")
        aid = mdb.create_analysis(pid, "scene", scene_id, "pacing",
                                  {"pacing": 0.7, "tension": 0.5}, score=0.7)
        analysis = mdb.get_analysis(aid)
        assert analysis is not None
        assert analysis["analysis_type"] == "pacing"
        assert analysis["result"]["pacing"] == 0.7
        assert analysis["score"] == 0.7
        assert analysis["stale"] == 0

    def test_create_with_provider_and_model(self, mdb):
        pid = mdb.create_project("Novel")
        aid = mdb.create_analysis(
            pid, "project", pid, "consistency", {"ok": True},
            provider="openai", model="gpt-4o",
        )
        analysis = mdb.get_analysis(aid)
        assert analysis["provider"] == "openai"
        assert analysis["model"] == "gpt-4o"

    def test_analysis_sync_payload_includes_result_json(self, mdb):
        pid = mdb.create_project("Novel")
        aid = mdb.create_analysis(
            pid,
            "project",
            pid,
            "consistency",
            {"ok": True, "score": 0.9},
        )

        row = mdb.db.execute_query(
            "SELECT payload FROM sync_log WHERE entity = ? AND entity_id = ? "
            "ORDER BY change_id DESC LIMIT 1",
            ("manuscript_ai_analyses", aid),
        ).fetchone()

        payload = json.loads(row["payload"])
        assert payload["result_json"] == json.dumps({"ok": True, "score": 0.9})

    def test_get_missing_returns_none(self, mdb):
        assert mdb.get_analysis("nonexistent") is None

    def test_list_by_project(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        sid = mdb.create_scene(ch_id, pid, title="S1", content_json="{}", content_plain="text")
        mdb.create_analysis(pid, "scene", sid, "pacing", {"pacing": 0.5})
        mdb.create_analysis(pid, "scene", sid, "tension", {"tension": 0.8})
        analyses = mdb.list_analyses(pid)
        assert len(analyses) == 2

    def test_list_filter_by_type(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        sid = mdb.create_scene(ch_id, pid, title="S1", content_json="{}", content_plain="text")
        mdb.create_analysis(pid, "scene", sid, "pacing", {})
        mdb.create_analysis(pid, "scene", sid, "tension", {})
        analyses = mdb.list_analyses(pid, analysis_type="pacing")
        assert len(analyses) == 1

    def test_list_filter_by_scope(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        s1 = mdb.create_scene(ch_id, pid, title="S1", content_json="{}", content_plain="a")
        s2 = mdb.create_scene(ch_id, pid, title="S2", content_json="{}", content_plain="b")
        mdb.create_analysis(pid, "scene", s1, "pacing", {})
        mdb.create_analysis(pid, "scene", s2, "pacing", {})
        analyses = mdb.list_analyses(pid, scope_type="scene", scope_id=s1)
        assert len(analyses) == 1

    def test_list_excludes_stale_by_default(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        sid = mdb.create_scene(ch_id, pid, title="S1", content_json="{}", content_plain="text")
        mdb.create_analysis(pid, "scene", sid, "pacing", {})
        mdb.mark_analyses_stale("scene", sid)
        analyses = mdb.list_analyses(pid)
        assert len(analyses) == 0
        analyses_with_stale = mdb.list_analyses(pid, include_stale=True)
        assert len(analyses_with_stale) == 1

    def test_list_excludes_deleted(self, mdb):
        pid = mdb.create_project("Novel")
        aid = mdb.create_analysis(pid, "project", pid, "consistency", {})
        mdb.soft_delete_analysis(aid, expected_version=1)
        analyses = mdb.list_analyses(pid)
        assert len(analyses) == 0

    def test_deleted_project_hides_cached_analysis_reads(self, mdb):
        pid = mdb.create_project("Novel")
        aid = mdb.create_analysis(pid, "project", pid, "consistency", {})

        mdb.soft_delete_project(pid, expected_version=1)

        assert mdb.get_analysis(aid) is None
        assert mdb.list_analyses(pid) == []

    def test_active_part_scope_analysis_remains_visible(self, mdb):
        pid = mdb.create_project("Novel")
        part_id = mdb.create_part(pid, "Part I")
        aid = mdb.create_analysis(pid, "part", part_id, "structure", {"notes": []})

        analysis = mdb.get_analysis(aid)

        assert analysis is not None
        assert analysis["scope_type"] == "part"
        assert [item["id"] for item in mdb.list_analyses(pid)] == [aid]

    def test_deleted_scene_scope_hides_cached_analysis_reads(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        sid = mdb.create_scene(ch_id, pid, title="S1", content_json="{}", content_plain="text")
        aid = mdb.create_analysis(pid, "scene", sid, "pacing", {"pacing": 0.5})

        mdb.soft_delete_scene(sid, expected_version=1)

        assert mdb.get_analysis(aid) is None
        assert mdb.list_analyses(pid) == []

    def test_deleted_part_scope_hides_cached_analysis_reads(self, mdb):
        pid = mdb.create_project("Novel")
        part_id = mdb.create_part(pid, "Part I")
        aid = mdb.create_analysis(pid, "part", part_id, "structure", {"notes": []})

        mdb.soft_delete_part(part_id, expected_version=1)

        assert mdb.get_analysis(aid) is None
        assert mdb.list_analyses(pid, include_stale=True) == []

    def test_mark_stale(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        sid = mdb.create_scene(ch_id, pid, title="S1", content_json="{}", content_plain="text")
        aid = mdb.create_analysis(pid, "scene", sid, "pacing", {})
        count = mdb.mark_analyses_stale("scene", sid)
        assert count == 1
        analysis = mdb.get_analysis(aid)
        assert analysis["stale"] == 1
        assert analysis["version"] == 2

    def test_mark_stale_idempotent(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        sid = mdb.create_scene(ch_id, pid, title="S1", content_json="{}", content_plain="text")
        mdb.create_analysis(pid, "scene", sid, "pacing", {})
        assert mdb.mark_analyses_stale("scene", sid) == 1
        # Second call should find no fresh rows to mark
        assert mdb.mark_analyses_stale("scene", sid) == 0

    def test_soft_delete(self, mdb):
        pid = mdb.create_project("Novel")
        aid = mdb.create_analysis(pid, "project", pid, "consistency", {})
        mdb.soft_delete_analysis(aid, expected_version=1)
        assert mdb.get_analysis(aid) is None

    def test_soft_delete_version_conflict(self, mdb):
        pid = mdb.create_project("Novel")
        aid = mdb.create_analysis(pid, "project", pid, "consistency", {})
        with pytest.raises(ConflictError):
            mdb.soft_delete_analysis(aid, expected_version=99)

    def test_scene_update_marks_analyses_stale(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        sid = mdb.create_scene(ch_id, pid, title="S1", content_json="{}", content_plain="old text")
        aid = mdb.create_analysis(pid, "scene", sid, "pacing", {"pacing": 0.5})
        # Verify not stale initially
        assert mdb.get_analysis(aid)["stale"] == 0
        # Update scene content
        mdb.update_scene(sid, {"content_plain": "new text entirely different"}, expected_version=1)
        # Verify analysis is now stale
        analysis = mdb.get_analysis(aid)
        assert analysis["stale"] == 1

    def test_scene_content_json_update_marks_analyses_stale(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        sid = mdb.create_scene(ch_id, pid, title="S1", content_json='{"type":"doc"}', content_plain="old text")
        aid = mdb.create_analysis(pid, "scene", sid, "pacing", {"pacing": 0.5})
        assert mdb.get_analysis(aid)["stale"] == 0

        mdb.update_scene(
            sid,
            {"content_json": '{"type":"doc","content":[{"type":"paragraph"}]}'},
            expected_version=1,
        )

        assert mdb.get_analysis(aid)["stale"] == 1

    def test_scene_update_non_content_does_not_mark_stale(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        sid = mdb.create_scene(ch_id, pid, title="S1", content_json="{}", content_plain="text")
        aid = mdb.create_analysis(pid, "scene", sid, "pacing", {"pacing": 0.5})
        # Update only the title, not content
        mdb.update_scene(sid, {"title": "S1 Renamed"}, expected_version=1)
        # Analysis should still be fresh
        assert mdb.get_analysis(aid)["stale"] == 0

    def test_project_analyses_stale_after_character_role_and_world_kind_changes(self, mdb):
        pid = mdb.create_project("Novel")
        character_id = mdb.create_character(pid, "Alice", role="supporting")
        world_info_id = mdb.create_world_info(pid, "location", "Town")

        character_analysis_id = mdb.create_analysis(pid, "project", pid, "consistency", {})
        mdb.update_character(character_id, {"role": "protagonist"}, expected_version=1)
        assert mdb.get_analysis(character_analysis_id)["stale"] == 1

        world_analysis_id = mdb.create_analysis(pid, "project", pid, "consistency", {})
        mdb.update_world_info(world_info_id, {"kind": "faction"}, expected_version=1)
        assert mdb.get_analysis(world_analysis_id)["stale"] == 1

    def test_part_delete_stales_project_analysis(self, mdb):
        pid = mdb.create_project("Novel")
        part_id = mdb.create_part(pid, "Part I")
        chapter_one = mdb.create_chapter(pid, "Chapter 1", part_id=part_id)
        chapter_two = mdb.create_chapter(pid, "Chapter 2", part_id=part_id)
        project_analysis_id = mdb.create_analysis(pid, "project", pid, "consistency", {})
        chapter_analysis_one_id = mdb.create_analysis(pid, "chapter", chapter_one, "structure", {})
        chapter_analysis_two_id = mdb.create_analysis(pid, "chapter", chapter_two, "structure", {})

        mdb.soft_delete_part(part_id, expected_version=1)

        assert mdb.get_analysis(project_analysis_id)["stale"] == 1
        assert _analysis_row(mdb, chapter_analysis_one_id)["stale"] == 1
        assert _analysis_row(mdb, chapter_analysis_two_id)["stale"] == 1

    def test_chapter_delete_stales_chapter_and_project_analyses(self, mdb):
        pid = mdb.create_project("Novel")
        chapter_id = mdb.create_chapter(pid, "Chapter 1")
        project_analysis_id = mdb.create_analysis(pid, "project", pid, "consistency", {})
        chapter_analysis_one_id = mdb.create_analysis(pid, "chapter", chapter_id, "structure", {})
        chapter_analysis_two_id = mdb.create_analysis(pid, "chapter", chapter_id, "pacing", {})

        mdb.soft_delete_chapter(chapter_id, expected_version=1)

        assert mdb.get_analysis(project_analysis_id)["stale"] == 1
        assert _analysis_row(mdb, chapter_analysis_one_id)["stale"] == 1
        assert _analysis_row(mdb, chapter_analysis_two_id)["stale"] == 1

    def test_chapter_reparent_stales_project_analysis(self, mdb):
        pid = mdb.create_project("Novel")
        part_a = mdb.create_part(pid, "Part A")
        part_b = mdb.create_part(pid, "Part B")
        chapter_id = mdb.create_chapter(pid, "Chapter 1", part_id=part_a)
        analysis_id = mdb.create_analysis(pid, "project", pid, "consistency", {})

        mdb.reorder_items(
            "chapter",
            [{"id": chapter_id, "sort_order": 0, "part_id": part_b}],
            project_id=pid,
        )

        assert mdb.get_analysis(analysis_id)["stale"] == 1

    def test_analysis_sync_log_includes_result_json_on_create_and_update(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        sid = mdb.create_scene(ch_id, pid, title="S1", content_json="{}", content_plain="text")
        result = {"pacing": 0.8, "assessment": "Fast"}
        aid = mdb.create_analysis(pid, "scene", sid, "pacing", result)

        with mdb.db.transaction() as conn:
            create_row = conn.execute(
                "SELECT operation, payload FROM sync_log WHERE entity = ? AND entity_id = ? ORDER BY change_id ASC LIMIT 1",
                ("manuscript_ai_analyses", aid),
            ).fetchone()
        assert create_row is not None
        create_payload = json.loads(create_row["payload"])
        assert create_row["operation"] == "create"
        assert json.loads(create_payload["result_json"]) == result

        assert mdb.mark_analyses_stale("scene", sid) == 1

        with mdb.db.transaction() as conn:
            update_row = conn.execute(
                "SELECT operation, payload FROM sync_log WHERE entity = ? AND entity_id = ? ORDER BY change_id DESC LIMIT 1",
                ("manuscript_ai_analyses", aid),
            ).fetchone()
        assert update_row is not None
        update_payload = json.loads(update_row["payload"])
        assert update_row["operation"] == "update"
        assert json.loads(update_payload["result_json"]) == result
