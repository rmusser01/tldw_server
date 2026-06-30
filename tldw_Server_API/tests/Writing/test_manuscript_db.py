# test_manuscript_db.py
# Unit tests for ManuscriptDBHelper CRUD operations.
#
from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
)
from tldw_Server_API.app.core.DB_Management.ManuscriptDB import (
    ManuscriptDBHelper,
    _word_count,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mdb(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "test.db"), client_id="test_client")
    return ManuscriptDBHelper(db)


# ---------------------------------------------------------------------------
# Word-count helper
# ---------------------------------------------------------------------------

class TestWordCount:
    def test_none(self):
        assert _word_count(None) == 0

    def test_empty(self):
        assert _word_count("") == 0

    def test_whitespace_only(self):
        assert _word_count("   \n\t  ") == 0

    def test_simple(self):
        assert _word_count("hello world") == 2

    def test_multiline(self):
        assert _word_count("one two\nthree four five") == 5


# ---------------------------------------------------------------------------
# Project CRUD
# ---------------------------------------------------------------------------

class TestProjectCRUD:
    def test_create_and_get(self, mdb):
        pid = mdb.create_project(
            "My Novel",
            author="Alice",
            genre="Fantasy",
            settings={"theme": "noir", "view": {"mode": "board"}},
        )
        proj = mdb.get_project(pid)
        assert proj is not None
        assert proj["title"] == "My Novel"
        assert proj["author"] == "Alice"
        assert proj["genre"] == "Fantasy"
        assert proj["status"] == "draft"
        assert proj["settings"] == {"theme": "noir", "view": {"mode": "board"}}
        assert "settings_json" not in proj
        assert proj["word_count"] == 0
        assert proj["version"] == 1
        assert proj["deleted"] == 0

    def test_create_with_custom_id(self, mdb):
        pid = mdb.create_project("Titled", project_id="custom-123")
        assert pid == "custom-123"
        proj = mdb.get_project("custom-123")
        assert proj is not None
        assert proj["title"] == "Titled"

    def test_get_missing_returns_none(self, mdb):
        assert mdb.get_project("nonexistent") is None

    def test_list_projects(self, mdb):
        mdb.create_project("A")
        mdb.create_project("B")
        mdb.create_project("C", status="complete")

        projects, total = mdb.list_projects()
        assert total == 3
        assert len(projects) == 3

    def test_list_projects_with_status_filter(self, mdb):
        mdb.create_project("Draft1")
        mdb.create_project("Done", status="complete")

        projects, total = mdb.list_projects(status_filter="complete")
        assert total == 1
        assert projects[0]["title"] == "Done"

    def test_list_projects_pagination(self, mdb):
        for i in range(5):
            mdb.create_project(f"Project {i}")

        projects, total = mdb.list_projects(limit=2, offset=0)
        assert total == 5
        assert len(projects) == 2

        projects2, total2 = mdb.list_projects(limit=2, offset=2)
        assert total2 == 5
        assert len(projects2) == 2

    def test_update_project(self, mdb):
        pid = mdb.create_project("Original")
        mdb.update_project(pid, {"title": "Updated", "status": "writing"}, expected_version=1)

        proj = mdb.get_project(pid)
        assert proj["title"] == "Updated"
        assert proj["status"] == "writing"
        assert proj["version"] == 2

    def test_update_project_settings(self, mdb):
        pid = mdb.create_project("With Settings")
        mdb.update_project(pid, {"settings": {"theme": "dark"}}, expected_version=1)

        proj = mdb.get_project(pid)
        assert proj["settings"] == {"theme": "dark"}
        assert "settings_json" not in proj

    def test_project_settings_roundtrip(self, mdb):
        pid = mdb.create_project("With Settings", settings={"font": "serif", "columns": 2})
        proj = mdb.get_project(pid)
        assert proj["settings"] == {"font": "serif", "columns": 2}
        assert "settings_json" not in proj

    def test_list_projects_settings_deserialized(self, mdb):
        mdb.create_project("P1", settings={"a": 1})
        projects, _ = mdb.list_projects()
        assert projects[0]["settings"] == {"a": 1}
        assert "settings_json" not in projects[0]

    def test_list_projects_exposes_settings_dict(self, mdb):
        mdb.create_project("With Settings", settings={"theme": "dark", "autosave": True})

        projects, total = mdb.list_projects()

        assert total == 1
        assert projects[0]["settings"] == {"theme": "dark", "autosave": True}
        assert "settings_json" not in projects[0]

    def test_update_project_version_conflict(self, mdb):
        pid = mdb.create_project("Conflicted")

        with pytest.raises(ConflictError):
            mdb.update_project(pid, {"title": "Nope"}, expected_version=999)

    def test_update_project_empty_updates(self, mdb):
        pid = mdb.create_project("No Change")
        # Should be a no-op, no error
        mdb.update_project(pid, {}, expected_version=1)
        proj = mdb.get_project(pid)
        assert proj["version"] == 1

    def test_soft_delete_project(self, mdb):
        pid = mdb.create_project("To Delete")
        mdb.soft_delete_project(pid, expected_version=1)

        # Should not be found after deletion
        assert mdb.get_project(pid) is None

    def test_soft_delete_project_version_conflict(self, mdb):
        pid = mdb.create_project("To Delete Conflict")

        with pytest.raises(ConflictError):
            mdb.soft_delete_project(pid, expected_version=99)

    def test_soft_deleted_not_in_list(self, mdb):
        pid = mdb.create_project("Deleted One")
        mdb.create_project("Kept")
        mdb.soft_delete_project(pid, expected_version=1)

        projects, total = mdb.list_projects()
        assert total == 1
        assert projects[0]["title"] == "Kept"

    def test_invalid_project_status(self, mdb):
        with pytest.raises(ValueError, match="Invalid project status"):
            mdb.create_project("Bad Status", status="bogus")


# ---------------------------------------------------------------------------
# Part CRUD
# ---------------------------------------------------------------------------

class TestPartCRUD:
    def test_create_and_get(self, mdb):
        pid = mdb.create_project("Novel")
        part_id = mdb.create_part(pid, "Part One", sort_order=1)

        part = mdb.get_part(part_id)
        assert part is not None
        assert part["title"] == "Part One"
        assert part["project_id"] == pid
        assert part["sort_order"] == 1
        assert part["version"] == 1

    def test_list_parts_ordered(self, mdb):
        pid = mdb.create_project("Novel")
        mdb.create_part(pid, "Part B", sort_order=2)
        mdb.create_part(pid, "Part A", sort_order=1)

        parts = mdb.list_parts(pid)
        assert len(parts) == 2
        assert parts[0]["title"] == "Part A"
        assert parts[1]["title"] == "Part B"

    def test_update_part(self, mdb):
        pid = mdb.create_project("Novel")
        part_id = mdb.create_part(pid, "Old Title")

        mdb.update_part(part_id, {"title": "New Title"}, expected_version=1)
        part = mdb.get_part(part_id)
        assert part["title"] == "New Title"
        assert part["version"] == 2

    def test_update_part_version_conflict(self, mdb):
        pid = mdb.create_project("Novel")
        part_id = mdb.create_part(pid, "Part")

        with pytest.raises(ConflictError):
            mdb.update_part(part_id, {"title": "Nope"}, expected_version=42)

    def test_soft_delete_part(self, mdb):
        pid = mdb.create_project("Novel")
        part_id = mdb.create_part(pid, "Deleted Part")

        mdb.soft_delete_part(part_id, expected_version=1)
        assert mdb.get_part(part_id) is None

    def test_soft_delete_part_cascades_scenes(self, mdb):
        project_id = mdb.create_project("Novel")
        part_id = mdb.create_part(project_id, "Part I")
        chapter_id = mdb.create_chapter(project_id, "Chapter 1", part_id=part_id)
        scene_id = mdb.create_scene(chapter_id, project_id, title="Scene 1", content_plain="alpha beta")
        part = mdb.get_part(part_id)

        mdb.soft_delete_part(part_id, expected_version=part["version"])

        assert mdb.get_part(part_id) is None
        assert mdb.get_chapter(chapter_id) is None
        assert mdb.get_scene(scene_id) is None

    def test_get_missing_part(self, mdb):
        assert mdb.get_part("no-such-id") is None


# ---------------------------------------------------------------------------
# Chapter CRUD
# ---------------------------------------------------------------------------

class TestChapterCRUD:
    def test_create_without_part(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_chapter(pid, "Chapter 1")

        ch = mdb.get_chapter(cid)
        assert ch is not None
        assert ch["title"] == "Chapter 1"
        assert ch["part_id"] is None
        assert ch["project_id"] == pid

    def test_create_with_part(self, mdb):
        pid = mdb.create_project("Novel")
        part_id = mdb.create_part(pid, "Part I")
        cid = mdb.create_chapter(pid, "Chapter 1", part_id=part_id)

        ch = mdb.get_chapter(cid)
        assert ch["part_id"] == part_id

    def test_create_chapter_rejects_cross_project_part(self, mdb):
        left = mdb.create_project("Left")
        right = mdb.create_project("Right")
        foreign_part = mdb.create_part(right, "Foreign")

        with pytest.raises(ConflictError, match="different project"):
            mdb.create_chapter(left, "Intruder", part_id=foreign_part)

    def test_list_chapters(self, mdb):
        pid = mdb.create_project("Novel")
        mdb.create_chapter(pid, "Ch B", sort_order=2)
        mdb.create_chapter(pid, "Ch A", sort_order=1)

        chapters = mdb.list_chapters(pid)
        assert len(chapters) == 2
        assert chapters[0]["title"] == "Ch A"
        assert chapters[1]["title"] == "Ch B"

    def test_list_chapters_by_part(self, mdb):
        pid = mdb.create_project("Novel")
        part1 = mdb.create_part(pid, "Part 1")
        part2 = mdb.create_part(pid, "Part 2")
        mdb.create_chapter(pid, "Ch in Part 1", part_id=part1)
        mdb.create_chapter(pid, "Ch in Part 2", part_id=part2)
        mdb.create_chapter(pid, "Ch Unassigned")

        chapters = mdb.list_chapters(pid, part_id=part1)
        assert len(chapters) == 1
        assert chapters[0]["title"] == "Ch in Part 1"

    def test_update_chapter(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_chapter(pid, "Old")
        mdb.update_chapter(cid, {"title": "New"}, expected_version=1)

        ch = mdb.get_chapter(cid)
        assert ch["title"] == "New"
        assert ch["version"] == 2

    def test_update_chapter_part_id(self, mdb):
        pid = mdb.create_project("Novel")
        part_a = mdb.create_part(pid, "Part A")
        part_b = mdb.create_part(pid, "Part B")
        cid = mdb.create_chapter(pid, "Movable", part_id=part_a)

        mdb.update_chapter(cid, {"part_id": part_b}, expected_version=1)

        ch = mdb.get_chapter(cid)
        assert ch["part_id"] == part_b
        assert ch["version"] == 2

    def test_update_chapter_rejects_cross_project_part(self, mdb):
        left = mdb.create_project("Left")
        right = mdb.create_project("Right")
        local_part = mdb.create_part(left, "Local")
        foreign_part = mdb.create_part(right, "Foreign")
        chapter_id = mdb.create_chapter(left, "Movable", part_id=local_part)

        with pytest.raises(ConflictError, match="different project"):
            mdb.update_chapter(chapter_id, {"part_id": foreign_part}, expected_version=1)

    def test_update_chapter_version_conflict(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_chapter(pid, "Ch")

        with pytest.raises(ConflictError):
            mdb.update_chapter(cid, {"title": "Nope"}, expected_version=99)

    def test_soft_delete_chapter(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_chapter(pid, "Doomed")
        mdb.soft_delete_chapter(cid, expected_version=1)

        assert mdb.get_chapter(cid) is None

    def test_invalid_chapter_status(self, mdb):
        pid = mdb.create_project("Novel")
        with pytest.raises(ValueError, match="Invalid chapter status"):
            mdb.create_chapter(pid, "Bad", status="nonexistent")


# ---------------------------------------------------------------------------
# Scene CRUD
# ---------------------------------------------------------------------------

class TestSceneCRUD:
    def test_create_and_get(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_chapter(pid, "Ch1")
        sid = mdb.create_scene(
            cid, pid,
            title="Opening",
            content_plain="It was a dark and stormy night.",
        )

        scene = mdb.get_scene(sid)
        assert scene is not None
        assert scene["title"] == "Opening"
        assert scene["word_count"] == 7
        assert scene["chapter_id"] == cid
        assert scene["project_id"] == pid

    def test_create_scene_empty_content(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_chapter(pid, "Ch1")
        sid = mdb.create_scene(cid, pid)

        scene = mdb.get_scene(sid)
        assert scene["word_count"] == 0

    def test_list_scenes_ordered(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_chapter(pid, "Ch1")
        mdb.create_scene(cid, pid, title="Scene B", sort_order=2)
        mdb.create_scene(cid, pid, title="Scene A", sort_order=1)

        scenes = mdb.list_scenes(cid)
        assert len(scenes) == 2
        assert scenes[0]["title"] == "Scene A"
        assert scenes[1]["title"] == "Scene B"

    def test_update_scene_content(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_chapter(pid, "Ch1")
        sid = mdb.create_scene(cid, pid, content_plain="one two three")

        scene = mdb.get_scene(sid)
        assert scene["word_count"] == 3

        mdb.update_scene(sid, {"content_plain": "a b c d e"}, expected_version=1)
        scene = mdb.get_scene(sid)
        assert scene["word_count"] == 5
        assert scene["version"] == 2

    def test_update_scene_non_content_field(self, mdb):
        """Updating a non-content field should not change word_count."""
        pid = mdb.create_project("Novel")
        cid = mdb.create_chapter(pid, "Ch1")
        sid = mdb.create_scene(cid, pid, content_plain="one two three")

        mdb.update_scene(sid, {"title": "Renamed"}, expected_version=1)
        scene = mdb.get_scene(sid)
        assert scene["word_count"] == 3
        assert scene["title"] == "Renamed"

    def test_update_scene_version_conflict(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_chapter(pid, "Ch1")
        sid = mdb.create_scene(cid, pid)

        with pytest.raises(ConflictError):
            mdb.update_scene(sid, {"title": "Nope"}, expected_version=99)

    def test_soft_delete_scene(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_chapter(pid, "Ch1")
        sid = mdb.create_scene(cid, pid, content_plain="some words here")

        mdb.soft_delete_scene(sid, expected_version=1)
        assert mdb.get_scene(sid) is None

    def test_soft_delete_scene_version_conflict(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_chapter(pid, "Ch1")
        sid = mdb.create_scene(cid, pid)

        with pytest.raises(ConflictError):
            mdb.soft_delete_scene(sid, expected_version=99)

    def test_invalid_scene_status(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_chapter(pid, "Ch1")
        with pytest.raises(ValueError, match="Invalid scene status"):
            mdb.create_scene(cid, pid, status="bogus")


# ---------------------------------------------------------------------------
# Word count propagation
# ---------------------------------------------------------------------------

class TestWordCountPropagation:
    def test_scene_to_chapter(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_chapter(pid, "Ch1")
        mdb.create_scene(cid, pid, content_plain="one two three")  # 3 words
        mdb.create_scene(cid, pid, content_plain="four five")      # 2 words

        ch = mdb.get_chapter(cid)
        assert ch["word_count"] == 5

    def test_scene_to_project(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_chapter(pid, "Ch1")
        mdb.create_scene(cid, pid, content_plain="one two three")

        proj = mdb.get_project(pid)
        assert proj["word_count"] == 3

    def test_scene_to_part_to_project(self, mdb):
        pid = mdb.create_project("Novel")
        part_id = mdb.create_part(pid, "Part I")
        cid = mdb.create_chapter(pid, "Ch1", part_id=part_id)
        mdb.create_scene(cid, pid, content_plain="alpha beta gamma delta")  # 4 words

        part = mdb.get_part(part_id)
        assert part["word_count"] == 4

        proj = mdb.get_project(pid)
        assert proj["word_count"] == 4

    def test_update_propagates(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_chapter(pid, "Ch1")
        sid = mdb.create_scene(cid, pid, content_plain="one two")  # 2 words

        mdb.update_scene(sid, {"content_plain": "one two three four five"}, expected_version=1)

        ch = mdb.get_chapter(cid)
        assert ch["word_count"] == 5

        proj = mdb.get_project(pid)
        assert proj["word_count"] == 5

    def test_delete_propagates(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_chapter(pid, "Ch1")
        s1 = mdb.create_scene(cid, pid, content_plain="one two three")  # 3
        mdb.create_scene(cid, pid, content_plain="four five")            # 2

        mdb.soft_delete_scene(s1, expected_version=1)

        ch = mdb.get_chapter(cid)
        assert ch["word_count"] == 2

        proj = mdb.get_project(pid)
        assert proj["word_count"] == 2

    def test_multi_chapter_propagation(self, mdb):
        pid = mdb.create_project("Novel")
        c1 = mdb.create_chapter(pid, "Ch1")
        c2 = mdb.create_chapter(pid, "Ch2")
        mdb.create_scene(c1, pid, content_plain="one two")      # 2
        mdb.create_scene(c2, pid, content_plain="three four five")  # 3

        proj = mdb.get_project(pid)
        assert proj["word_count"] == 5


# ---------------------------------------------------------------------------
# Project structure
# ---------------------------------------------------------------------------

class TestProjectStructure:
    def test_full_hierarchy(self, mdb):
        pid = mdb.create_project("Novel")
        part_id = mdb.create_part(pid, "Part I")
        ch1 = mdb.create_chapter(pid, "Ch1", part_id=part_id)
        ch2 = mdb.create_chapter(pid, "Ch2")  # unassigned
        mdb.create_scene(ch1, pid, title="S1")
        mdb.create_scene(ch1, pid, title="S2")
        mdb.create_scene(ch2, pid, title="S3")

        structure = mdb.get_project_structure(pid)

        assert structure["project_id"] == pid
        assert len(structure["parts"]) == 1
        assert structure["parts"][0]["title"] == "Part I"

        part_chapters = structure["parts"][0]["chapters"]
        assert len(part_chapters) == 1
        assert part_chapters[0]["title"] == "Ch1"
        assert len(part_chapters[0]["scenes"]) == 2

        unassigned = structure["unassigned_chapters"]
        assert len(unassigned) == 1
        assert unassigned[0]["title"] == "Ch2"
        assert len(unassigned[0]["scenes"]) == 1

    def test_empty_project(self, mdb):
        pid = mdb.create_project("Empty")
        structure = mdb.get_project_structure(pid)

        assert structure["project_id"] == pid
        assert structure["parts"] == []
        assert structure["unassigned_chapters"] == []


# ---------------------------------------------------------------------------
# FTS Search
# ---------------------------------------------------------------------------

class TestFTSSearch:
    def test_search_by_content(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_chapter(pid, "Ch1")
        mdb.create_scene(cid, pid, title="Opening", content_plain="The dragon flew over the mountain.")
        mdb.create_scene(cid, pid, title="Middle", content_plain="The knight rode through the valley.")

        results = mdb.search_scenes(pid, "dragon")
        assert len(results) == 1
        assert results[0]["title"] == "Opening"

    def test_search_by_title(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_chapter(pid, "Ch1")
        mdb.create_scene(cid, pid, title="Epilogue", content_plain="The end.")
        mdb.create_scene(cid, pid, title="Prologue", content_plain="The beginning.")

        results = mdb.search_scenes(pid, "Epilogue")
        assert len(results) == 1
        assert results[0]["title"] == "Epilogue"

    def test_search_no_results(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_chapter(pid, "Ch1")
        mdb.create_scene(cid, pid, content_plain="Nothing relevant here.")

        results = mdb.search_scenes(pid, "unicorn")
        assert results == []

    def test_search_excludes_deleted(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_chapter(pid, "Ch1")
        sid = mdb.create_scene(cid, pid, title="Gone", content_plain="The cat sat on the mat.")
        mdb.soft_delete_scene(sid, expected_version=1)

        results = mdb.search_scenes(pid, "cat")
        assert results == []

    def test_search_has_snippet(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_chapter(pid, "Ch1")
        mdb.create_scene(cid, pid, content_plain="The wizard cast a powerful spell.")

        results = mdb.search_scenes(pid, "wizard")
        assert len(results) == 1
        assert "snippet" in results[0]

    def test_search_respects_limit(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_chapter(pid, "Ch1")
        for i in range(5):
            mdb.create_scene(cid, pid, content_plain=f"The dragon number {i} appeared.")

        results = mdb.search_scenes(pid, "dragon", limit=2)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Reorder
# ---------------------------------------------------------------------------

class TestReorder:
    def test_reorder_chapters(self, mdb):
        pid = mdb.create_project("Novel")
        c1 = mdb.create_chapter(pid, "Ch A", sort_order=1)
        c2 = mdb.create_chapter(pid, "Ch B", sort_order=2)
        c3 = mdb.create_chapter(pid, "Ch C", sort_order=3)

        # Reverse order
        mdb.reorder_items("chapter", [
            {"id": c3, "sort_order": 1},
            {"id": c2, "sort_order": 2},
            {"id": c1, "sort_order": 3},
        ])

        chapters = mdb.list_chapters(pid)
        assert chapters[0]["title"] == "Ch C"
        assert chapters[1]["title"] == "Ch B"
        assert chapters[2]["title"] == "Ch A"

    def test_reorder_scenes(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_chapter(pid, "Ch1")
        s1 = mdb.create_scene(cid, pid, title="S1", sort_order=1)
        s2 = mdb.create_scene(cid, pid, title="S2", sort_order=2)

        mdb.reorder_items("scene", [
            {"id": s2, "sort_order": 1},
            {"id": s1, "sort_order": 2},
        ])

        scenes = mdb.list_scenes(cid)
        assert scenes[0]["title"] == "S2"
        assert scenes[1]["title"] == "S1"

    def test_reorder_with_reparent(self, mdb):
        pid = mdb.create_project("Novel")
        part1 = mdb.create_part(pid, "Part 1")
        part2 = mdb.create_part(pid, "Part 2")
        cid = mdb.create_chapter(pid, "Moved Chapter", part_id=part1)

        mdb.reorder_items("chapter", [
            {"id": cid, "sort_order": 0, "part_id": part2},
        ])

        ch = mdb.get_chapter(cid)
        assert ch["part_id"] == part2

    def test_reorder_with_reparent_rejects_cross_project_part(self, mdb):
        left = mdb.create_project("Left")
        right = mdb.create_project("Right")
        local_part = mdb.create_part(left, "Local")
        foreign_part = mdb.create_part(right, "Foreign")
        chapter_id = mdb.create_chapter(left, "Movable", part_id=local_part)

        with pytest.raises(ConflictError, match="different project"):
            mdb.reorder_items(
                "chapter",
                [{"id": chapter_id, "sort_order": 0, "part_id": foreign_part}],
                project_id=left,
            )

    def test_reorder_with_reparent_rejects_cross_project_part_without_project_id(self, mdb):
        left = mdb.create_project("Left")
        right = mdb.create_project("Right")
        local_part = mdb.create_part(left, "Local")
        foreign_part = mdb.create_part(right, "Foreign")
        chapter_id = mdb.create_chapter(left, "Movable", part_id=local_part)

        with pytest.raises(ConflictError, match="different project"):
            mdb.reorder_items(
                "chapter",
                [{"id": chapter_id, "sort_order": 0, "part_id": foreign_part}],
            )

    def test_reorder_parts(self, mdb):
        pid = mdb.create_project("Novel")
        p1 = mdb.create_part(pid, "P1", sort_order=1)
        p2 = mdb.create_part(pid, "P2", sort_order=2)

        mdb.reorder_items("part", [
            {"id": p2, "sort_order": 1},
            {"id": p1, "sort_order": 2},
        ])

        parts = mdb.list_parts(pid)
        assert parts[0]["title"] == "P2"
        assert parts[1]["title"] == "P1"

    def test_invalid_entity_type(self, mdb):
        with pytest.raises(ValueError, match="Invalid entity_type"):
            mdb.reorder_items("bogus", [])
