# test_manuscript_annotations_db.py
# DB tests for manuscript annotation persistence and migration wiring.
#
from __future__ import annotations

import json
import sqlite3

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
)
from tldw_Server_API.app.core.DB_Management.ManuscriptDB import ManuscriptDBHelper


@pytest.fixture()
def raw_db(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "annotations.db"), client_id="test_client")
    try:
        yield db
    finally:
        db.close_connection()


@pytest.fixture()
def mdb(raw_db):
    return ManuscriptDBHelper(raw_db)


@pytest.fixture()
def manuscript(mdb):
    project_id = mdb.create_project("Annotated Novel")
    chapter_id = mdb.create_chapter(project_id, "Chapter 1")
    scene_id = mdb.create_scene(
        chapter_id,
        project_id,
        title="Opening",
        content_plain="Alpha beta gamma. Alpha beta delta.",
    )
    return {
        "project_id": project_id,
        "chapter_id": chapter_id,
        "scene_id": scene_id,
    }


def _table_names(db: CharactersRAGDB) -> set[str]:
    with db.transaction() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    return {row["name"] for row in rows}


def _index_names(db: CharactersRAGDB) -> set[str]:
    with db.transaction() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        ).fetchall()
    return {row["name"] for row in rows}


def _trigger_names(db: CharactersRAGDB) -> set[str]:
    with db.transaction() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger'"
        ).fetchall()
    return {row["name"] for row in rows}


def _table_sql(db: CharactersRAGDB, table_name: str) -> str:
    with db.transaction() as conn:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
    return row["sql"] if row is not None else ""


def _schema_version(db: CharactersRAGDB) -> int:
    with db.transaction() as conn:
        row = conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()
    return int(row["version"])


def test_fresh_db_creates_manuscript_annotations_table(raw_db):
    assert _schema_version(raw_db) == 51
    assert "manuscript_annotations" in _table_names(raw_db)
    table_sql = _table_sql(raw_db, "manuscript_annotations")
    assert "CHECK(anchor_status IN ('scene_level','attached','reattached','needs_review'))" in table_sql
    assert {
        "idx_mann_project_target",
        "idx_mann_project_status",
        "idx_mann_source",
        "idx_mann_deleted",
    }.issubset(_index_names(raw_db))
    assert {
        "manuscript_annotations_sync_create",
        "manuscript_annotations_sync_update",
        "manuscript_annotations_sync_delete",
        "manuscript_annotations_sync_undelete",
    }.issubset(_trigger_names(raw_db))


def test_sqlite_v50_migration_routes_to_v51(tmp_path):
    db_path = tmp_path / "migrating.db"
    db = CharactersRAGDB(str(db_path), client_id="test_client")
    db.close_connection()

    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(
            """
            DROP TRIGGER IF EXISTS manuscript_annotations_sync_create;
            DROP TRIGGER IF EXISTS manuscript_annotations_sync_update;
            DROP TRIGGER IF EXISTS manuscript_annotations_sync_delete;
            DROP TRIGGER IF EXISTS manuscript_annotations_sync_undelete;
            DROP INDEX IF EXISTS idx_mann_project_target;
            DROP INDEX IF EXISTS idx_mann_project_status;
            DROP INDEX IF EXISTS idx_mann_source;
            DROP INDEX IF EXISTS idx_mann_deleted;
            DROP TABLE IF EXISTS manuscript_annotations;
            UPDATE db_schema_version
               SET version = 50
             WHERE schema_name = 'rag_char_chat_schema';
            """
        )

    migrated = CharactersRAGDB(str(db_path), client_id="test_client")
    try:
        assert _schema_version(migrated) == 51
        assert "manuscript_annotations" in _table_names(migrated)
        table_sql = _table_sql(migrated, "manuscript_annotations")
        assert "CHECK(anchor_status IN ('scene_level','attached','reattached','needs_review'))" in table_sql
        assert "idx_mann_project_target" in _index_names(migrated)
        assert "manuscript_annotations_sync_create" in _trigger_names(migrated)
    finally:
        migrated.close_connection()


def test_postgres_v50_migration_script_contract_and_routing():
    script = getattr(CharactersRAGDB, "_MIGRATION_SQL_V50_TO_V51_POSTGRES", "")
    assert "CREATE TABLE IF NOT EXISTS manuscript_annotations" in script
    assert "TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP" in script
    assert "BOOLEAN NOT NULL DEFAULT FALSE" in script
    assert "CHECK(anchor_status IN ('scene_level','attached','reattached','needs_review'))" in script
    assert "manuscript_annotations_sync_log_fn" in script
    assert "DROP TRIGGER IF EXISTS manuscript_annotations_sync_log" in script
    assert "CREATE TRIGGER manuscript_annotations_sync_log" in script
    assert "json_build_object(" in script
    assert "OLD.deleted = FALSE AND NEW.deleted = TRUE" in script
    assert "OLD.deleted = TRUE AND NEW.deleted = FALSE" in script
    assert "json_object(" not in script
    assert "UPDATE db_schema_version" in script
    assert "SET version = 51" in script

    steps = CharactersRAGDB(":memory:", client_id="test_client")._sqlite_linear_migration_steps()
    assert 50 in steps
    assert steps[50].__name__ == "_migrate_from_v50_to_v51"


def test_manuscript_annotations_rejects_unknown_stored_anchor_status(mdb, manuscript):
    with pytest.raises(sqlite3.IntegrityError):
        with mdb.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO manuscript_annotations (
                    id, project_id, target_type, target_id, status, category,
                    tags_json, source, body, metadata_json, anchor_status, client_id
                )
                VALUES (?, ?, 'project', ?, 'open', 'other', '[]', 'user', ?, '{}', ?, ?)
                """,
                (
                    "bad-anchor-status",
                    manuscript["project_id"],
                    manuscript["project_id"],
                    "Invalid stored anchor status.",
                    "unknown",
                    "test_client",
                ),
            )


def test_create_get_scene_range_annotation_validates_anchor(mdb, manuscript):
    scene = mdb.get_scene(manuscript["scene_id"])
    start = scene["content_plain"].index("beta")
    end = start + len("beta")

    annotation_id = mdb.create_annotation(
        project_id=manuscript["project_id"],
        target_type="scene",
        target_id=manuscript["scene_id"],
        category="clarity",
        source="ai_selected_text",
        body="Clarify this word.",
        scene_version=scene["version"],
        anchor_start=start,
        anchor_end=end,
        selected_text="beta",
        tags=["line"],
        metadata={"review_id": "r1"},
        suggested_fix="Use a concrete image.",
    )

    annotation = mdb.get_annotation(annotation_id)
    assert annotation is not None
    assert annotation["target_type"] == "scene"
    assert annotation["target_id"] == manuscript["scene_id"]
    assert annotation["selected_text"] == "beta"
    assert annotation["anchor_start"] == start
    assert annotation["anchor_end"] == end
    assert annotation["anchor_status"] == "attached"
    assert annotation["derived_start"] == start
    assert annotation["derived_end"] == end
    assert annotation["tags"] == ["line"]
    assert annotation["metadata"] == {"review_id": "r1"}
    assert "tags_json" not in annotation
    assert "metadata_json" not in annotation

    with pytest.raises(ValueError, match="selected text"):
        mdb.create_annotation(
            project_id=manuscript["project_id"],
            target_type="scene",
            target_id=manuscript["scene_id"],
            category="clarity",
            source="ai_selected_text",
            body="Bad anchor.",
            scene_version=scene["version"],
            anchor_start=start,
            anchor_end=end,
            selected_text="wrong",
        )

    with pytest.raises(ConflictError, match="scene version"):
        mdb.create_annotation(
            project_id=manuscript["project_id"],
            target_type="scene",
            target_id=manuscript["scene_id"],
            category="clarity",
            source="ai_selected_text",
            body="Stale anchor.",
            scene_version=scene["version"] + 1,
            anchor_start=start,
            anchor_end=end,
            selected_text="beta",
        )


def test_chapter_and_project_notes_reject_ranges_and_return_scene_level(mdb, manuscript):
    chapter_annotation_id = mdb.create_annotation(
        project_id=manuscript["project_id"],
        target_type="chapter",
        target_id=manuscript["chapter_id"],
        category="structure",
        source="user",
        body="Chapter-level note.",
    )
    project_annotation_id = mdb.create_annotation(
        project_id=manuscript["project_id"],
        target_type="project",
        target_id=manuscript["project_id"],
        category="other",
        source="user",
        body="Project-level note.",
    )

    assert mdb.get_annotation(chapter_annotation_id)["anchor_status"] == "scene_level"
    assert mdb.get_annotation(project_annotation_id)["anchor_status"] == "scene_level"
    assert mdb.get_annotation(chapter_annotation_id)["scene_level"] is True

    with pytest.raises(ValueError, match="range fields"):
        mdb.create_annotation(
            project_id=manuscript["project_id"],
            target_type="chapter",
            target_id=manuscript["chapter_id"],
            category="structure",
            source="user",
            body="Invalid range.",
            anchor_start=0,
            anchor_end=4,
            selected_text="test",
        )


def test_update_and_delete_use_optimistic_locking(mdb, manuscript):
    annotation_id = mdb.create_annotation(
        project_id=manuscript["project_id"],
        target_type="project",
        target_id=manuscript["project_id"],
        category="other",
        source="user",
        body="Original.",
    )

    with pytest.raises(ConflictError):
        mdb.update_annotation(annotation_id, {"body": "Nope."}, expected_version=99)

    mdb.update_annotation(
        annotation_id,
        {"body": "Resolved.", "status": "resolved", "followup_note": "Handled."},
        expected_version=1,
    )
    updated = mdb.get_annotation(annotation_id)
    assert updated["body"] == "Resolved."
    assert updated["status"] == "resolved"
    assert updated["followup_note"] == "Handled."
    assert updated["version"] == 2

    with pytest.raises(ConflictError):
        mdb.soft_delete_annotation(annotation_id, expected_version=1)

    mdb.soft_delete_annotation(annotation_id, expected_version=2)
    assert mdb.get_annotation(annotation_id) is None


def test_deleted_scene_target_hides_annotation_from_get_and_project_list(mdb, manuscript):
    scene = mdb.get_scene(manuscript["scene_id"])
    start = scene["content_plain"].index("beta")
    annotation_id = mdb.create_annotation(
        project_id=manuscript["project_id"],
        target_type="scene",
        target_id=manuscript["scene_id"],
        category="clarity",
        source="ai_selected_text",
        body="Clarify this word.",
        scene_version=scene["version"],
        anchor_start=start,
        anchor_end=start + len("beta"),
        selected_text="beta",
    )

    mdb.soft_delete_scene(manuscript["scene_id"], expected_version=scene["version"])

    assert mdb.get_annotation(annotation_id) is None
    rows, total = mdb.list_annotations(manuscript["project_id"])
    assert rows == []
    assert total == 0
    with mdb.db.transaction() as conn:
        stored = conn.execute(
            "SELECT deleted FROM manuscript_annotations WHERE id = ?",
            (annotation_id,),
        ).fetchone()
    assert stored["deleted"] == 0


def test_deleted_chapter_target_hides_annotation_but_preserves_project_annotation(mdb, manuscript):
    chapter_annotation_id = mdb.create_annotation(
        project_id=manuscript["project_id"],
        target_type="chapter",
        target_id=manuscript["chapter_id"],
        category="structure",
        source="user",
        body="Chapter-level note.",
    )
    project_annotation_id = mdb.create_annotation(
        project_id=manuscript["project_id"],
        target_type="project",
        target_id=manuscript["project_id"],
        category="other",
        source="user",
        body="Project-level note.",
    )

    chapter = mdb.get_chapter(manuscript["chapter_id"])
    mdb.soft_delete_chapter(manuscript["chapter_id"], expected_version=chapter["version"])

    assert mdb.get_annotation(chapter_annotation_id) is None
    assert mdb.get_annotation(project_annotation_id) is not None
    rows, total = mdb.list_annotations(manuscript["project_id"])
    assert [row["id"] for row in rows] == [project_annotation_id]
    assert total == 1
    with mdb.db.transaction() as conn:
        stored = conn.execute(
            "SELECT deleted FROM manuscript_annotations WHERE id = ?",
            (chapter_annotation_id,),
        ).fetchone()
    assert stored["deleted"] == 0


def test_list_annotations_filters_by_target_status_category_and_source(mdb, manuscript):
    mdb.create_annotation(
        project_id=manuscript["project_id"],
        target_type="chapter",
        target_id=manuscript["chapter_id"],
        category="structure",
        source="user",
        body="Chapter note.",
    )
    resolved_id = mdb.create_annotation(
        project_id=manuscript["project_id"],
        target_type="project",
        target_id=manuscript["project_id"],
        category="other",
        source="ai_scene_review",
        body="Project note.",
    )
    mdb.update_annotation(resolved_id, {"status": "resolved"}, expected_version=1)

    rows, total = mdb.list_annotations(
        manuscript["project_id"],
        target_type="chapter",
        target_id=manuscript["chapter_id"],
        status="open",
        category="structure",
        source="user",
    )
    assert total == 1
    assert rows[0]["body"] == "Chapter note."

    rows, total = mdb.list_annotations(
        manuscript["project_id"],
        status="resolved",
        category="other",
        source="ai_scene_review",
    )
    assert total == 1
    assert rows[0]["id"] == resolved_id


def test_list_annotations_bulk_loads_scene_rows_for_anchor_derivation(mdb, manuscript):
    scene = mdb.get_scene(manuscript["scene_id"])
    gamma_start = scene["content_plain"].index("gamma")
    delta_start = scene["content_plain"].index("delta")
    for body, start, selected_text in (
        ("Review gamma.", gamma_start, "gamma"),
        ("Review delta.", delta_start, "delta"),
    ):
        mdb.create_annotation(
            project_id=manuscript["project_id"],
            target_type="scene",
            target_id=manuscript["scene_id"],
            category="clarity",
            source="ai_scene_review",
            body=body,
            scene_version=scene["version"],
            anchor_start=start,
            anchor_end=start + len(selected_text),
            selected_text=selected_text,
        )

    traced_sql: list[str] = []
    conn = mdb.db.get_connection()
    conn.set_trace_callback(traced_sql.append)
    try:
        rows, total = mdb.list_annotations(
            manuscript["project_id"],
            target_type="scene",
            target_id=manuscript["scene_id"],
        )
    finally:
        conn.set_trace_callback(None)

    assert total == 2
    assert len(rows) == 2
    scene_lookup_count = sum(
        "FROM manuscript_scenes" in statement
        and "content_plain" in statement
        and "version" in statement
        for statement in traced_sql
    )
    assert scene_lookup_count == 1


def test_list_annotations_rejects_unknown_anchor_status_filter(mdb, manuscript):
    mdb.create_annotation(
        project_id=manuscript["project_id"],
        target_type="project",
        target_id=manuscript["project_id"],
        category="other",
        source="user",
        body="Project note.",
    )

    with pytest.raises(ValueError, match="Invalid annotation anchor_status"):
        mdb.list_annotations(
            manuscript["project_id"],
            anchor_status="unknown",
        )


def test_derived_anchor_status_changes_after_scene_edit_without_mutating_row(mdb, manuscript):
    scene = mdb.get_scene(manuscript["scene_id"])
    start = scene["content_plain"].index("gamma")
    end = start + len("gamma")
    annotation_id = mdb.create_annotation(
        project_id=manuscript["project_id"],
        target_type="scene",
        target_id=manuscript["scene_id"],
        category="pacing",
        source="ai_selected_text",
        body="Check emphasis.",
        scene_version=scene["version"],
        anchor_start=start,
        anchor_end=end,
        selected_text="gamma",
    )

    mdb.update_scene(
        manuscript["scene_id"],
        {"content_plain": "Intro. Alpha beta gamma. Alpha beta delta."},
        expected_version=scene["version"],
    )
    annotation = mdb.get_annotation(annotation_id)
    assert annotation["anchor_status"] == "reattached"
    assert annotation["derived_start"] == "Intro. Alpha beta gamma. Alpha beta delta.".index("gamma")

    with mdb.db.transaction() as conn:
        stored = conn.execute(
            "SELECT anchor_start, anchor_status FROM manuscript_annotations WHERE id = ?",
            (annotation_id,),
        ).fetchone()
    assert stored["anchor_start"] == start
    assert stored["anchor_status"] == "attached"


def test_bounded_scene_anchor_status_filter_derives_before_pagination(mdb, manuscript):
    scene = mdb.get_scene(manuscript["scene_id"])
    gamma_start = scene["content_plain"].index("gamma")
    delta_start = scene["content_plain"].index("delta")
    attached_ids = [
        mdb.create_annotation(
            project_id=manuscript["project_id"],
            target_type="scene",
            target_id=manuscript["scene_id"],
            category="pacing",
            source="ai_selected_text",
            body="Attached gamma.",
            scene_version=scene["version"],
            anchor_start=gamma_start,
            anchor_end=gamma_start + len("gamma"),
            selected_text="gamma",
        ),
        mdb.create_annotation(
            project_id=manuscript["project_id"],
            target_type="scene",
            target_id=manuscript["scene_id"],
            category="clarity",
            source="ai_selected_text",
            body="Attached delta.",
            scene_version=scene["version"],
            anchor_start=delta_start,
            anchor_end=delta_start + len("delta"),
            selected_text="delta",
        ),
    ]
    scene_level_id = mdb.create_annotation(
        project_id=manuscript["project_id"],
        target_type="scene",
        target_id=manuscript["scene_id"],
        category="other",
        source="user",
        body="Newest scene-level note.",
    )
    mdb.update_annotation(scene_level_id, {"body": "Newest scene-level note updated."}, expected_version=1)

    rows, total = mdb.list_annotations(
        manuscript["project_id"],
        target_type="scene",
        target_id=manuscript["scene_id"],
        anchor_status="attached",
        limit=1,
    )

    assert total == 2
    assert len(rows) == 1
    assert rows[0]["id"] in attached_ids
    assert rows[0]["anchor_status"] == "attached"


def test_unbounded_anchor_status_filter_rejects_when_candidate_set_exceeds_cap(mdb, manuscript):
    for i in range(501):
        mdb.create_annotation(
            project_id=manuscript["project_id"],
            target_type="project",
            target_id=manuscript["project_id"],
            category="other",
            source="user",
            body=f"Project note {i}.",
        )

    with pytest.raises(ValueError, match="candidate set"):
        mdb.list_annotations(
            manuscript["project_id"],
            anchor_status="scene_level",
            limit=1,
        )


def test_create_and_update_annotation_reject_non_structured_tags_and_metadata(mdb, manuscript):
    with pytest.raises(ValueError, match="tags must be a list"):
        mdb.create_annotation(
            project_id=manuscript["project_id"],
            target_type="project",
            target_id=manuscript["project_id"],
            category="other",
            source="user",
            body="Invalid tags.",
            tags="not-a-list",
        )

    with pytest.raises(ValueError, match="tags must contain only strings"):
        mdb.create_annotation(
            project_id=manuscript["project_id"],
            target_type="project",
            target_id=manuscript["project_id"],
            category="other",
            source="user",
            body="Invalid tag entry.",
            tags=["valid", 42],
        )

    with pytest.raises(ValueError, match="metadata must be a dict"):
        mdb.create_annotation(
            project_id=manuscript["project_id"],
            target_type="project",
            target_id=manuscript["project_id"],
            category="other",
            source="user",
            body="Invalid metadata.",
            metadata=["not-a-dict"],
        )

    annotation_id = mdb.create_annotation(
        project_id=manuscript["project_id"],
        target_type="project",
        target_id=manuscript["project_id"],
        category="other",
        source="user",
        body="Original.",
    )

    with pytest.raises(ValueError, match="tags must be a list"):
        mdb.update_annotation(annotation_id, {"tags": "not-a-list"}, expected_version=1)

    with pytest.raises(ValueError, match="tags must contain only strings"):
        mdb.update_annotation(annotation_id, {"tags": ["valid", None]}, expected_version=1)

    with pytest.raises(ValueError, match="metadata must be a dict"):
        mdb.update_annotation(annotation_id, {"metadata": ["not-a-dict"]}, expected_version=1)


def test_duplicate_annotation_suppression_preserves_repeated_text_at_distinct_offsets(mdb, manuscript):
    scene = mdb.get_scene(manuscript["scene_id"])
    first_start = scene["content_plain"].index("beta")
    second_start = scene["content_plain"].index("beta", first_start + 1)
    body = "Clarify this repeated word."

    mdb.create_annotation(
        project_id=manuscript["project_id"],
        target_type="scene",
        target_id=manuscript["scene_id"],
        category="clarity",
        source="ai_selected_text",
        body=body,
        scene_version=scene["version"],
        anchor_start=first_start,
        anchor_end=first_start + len("beta"),
        selected_text="beta",
    )

    same_anchor_candidate = {
        "target_type": "scene",
        "target_id": manuscript["scene_id"],
        "category": "clarity",
        "source": "ai_selected_text",
        "body": body,
        "scene_version": scene["version"],
        "anchor_start": first_start,
        "anchor_end": first_start + len("beta"),
        "selected_text": "beta",
    }
    distinct_anchor_candidate = {
        **same_anchor_candidate,
        "anchor_start": second_start,
        "anchor_end": second_start + len("beta"),
    }

    retained = mdb.suppress_duplicate_annotation_candidates(
        manuscript["project_id"],
        [same_anchor_candidate, distinct_anchor_candidate],
    )

    assert retained == [distinct_anchor_candidate]


def test_duplicate_annotation_suppression_dedupes_candidates_in_same_batch(mdb, manuscript):
    scene = mdb.get_scene(manuscript["scene_id"])
    anchor_start = scene["content_plain"].index("gamma")
    candidate = {
        "target_type": "scene",
        "target_id": manuscript["scene_id"],
        "category": "clarity",
        "source": "ai_scene_review",
        "body": "Clarify this moment.",
        "scene_version": scene["version"],
        "anchor_start": anchor_start,
        "anchor_end": anchor_start + len("gamma"),
        "selected_text": "gamma",
    }

    retained = mdb.suppress_duplicate_annotation_candidates(
        manuscript["project_id"],
        [candidate, dict(candidate)],
    )

    assert retained == [candidate]


def test_sync_log_records_create_update_and_delete_for_annotations(mdb, manuscript):
    annotation_id = mdb.create_annotation(
        project_id=manuscript["project_id"],
        target_type="project",
        target_id=manuscript["project_id"],
        category="other",
        source="user",
        body="Sync me.",
        metadata={"k": "v"},
    )
    mdb.update_annotation(annotation_id, {"body": "Sync update."}, expected_version=1)
    mdb.soft_delete_annotation(annotation_id, expected_version=2)

    with mdb.db.transaction() as conn:
        rows = conn.execute(
            """
            SELECT operation, payload
              FROM sync_log
             WHERE entity = 'manuscript_annotations'
               AND entity_id = ?
             ORDER BY change_id
            """,
            (annotation_id,),
        ).fetchall()

    assert [row["operation"] for row in rows] == ["create", "update", "delete"]
    create_payload = json.loads(rows[0]["payload"])
    assert create_payload["metadata_json"] == '{"k": "v"}'
    assert {
        "scene_version",
        "anchor_start",
        "anchor_end",
        "selected_text",
        "document_fingerprint",
        "anchor_prefix",
        "anchor_suffix",
        "anchor_status",
    }.issubset(create_payload)
