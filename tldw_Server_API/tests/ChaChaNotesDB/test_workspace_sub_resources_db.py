"""Tests for workspace sub-resource tables: sources, artifacts, notes."""
import sqlite3
from datetime import datetime

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError, InputError


@pytest.fixture
def db(tmp_path):
    d = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    d.upsert_workspace("ws-1", "Test WS")
    try:
        yield d
    finally:
        d.close_all_connections()


class TestWorkspaceSources:
    def test_add_source(self, db):
        src = db.add_workspace_source("ws-1", {
            "id": "src-1", "media_id": 42, "title": "My Video",
            "source_type": "video",
        })
        assert src["id"] == "src-1"
        assert src["media_id"] == 42
        assert src["review_state"] == "unset"
        assert src["review_state_updated_at"]
        assert src["reviewed_at"] is None
        assert src["reviewed_by_user_id"] is None
        assert src["version"] == 1

    def test_add_source_with_needs_review_state(self, db):
        src = db.add_workspace_source("ws-1", {
            "id": "src-1", "media_id": 42, "title": "My Video",
            "source_type": "video", "review_state": "needs_review",
        })

        assert src["review_state"] == "needs_review"
        assert src["review_state_updated_at"]
        assert src["reviewed_at"] is None
        assert src["reviewed_by_user_id"] is None

    @pytest.mark.parametrize("review_state", [None, "", "  "])
    def test_add_source_normalizes_blank_review_state_to_unset(self, db, review_state):
        src = db.add_workspace_source("ws-1", {
            "id": f"src-{review_state!r}", "media_id": 42, "title": "My Video",
            "source_type": "video", "review_state": review_state,
        })

        assert src["review_state"] == "unset"

    @pytest.mark.parametrize("review_state", ["reviewed", "invalid"])
    def test_add_source_rejects_reviewed_and_invalid_review_states(self, db, review_state):
        with pytest.raises(InputError):
            db.add_workspace_source("ws-1", {
                "id": "src-1", "media_id": 42, "title": "My Video",
                "source_type": "video", "review_state": review_state,
            })

    @pytest.mark.parametrize(
        ("review_state", "source_id"),
        [
            (False, "src-false"),
            (0, "src-zero"),
            ([], "src-list"),
            ({}, "src-dict"),
        ],
    )
    def test_add_source_rejects_falsey_non_string_review_states(self, db, review_state, source_id):
        with pytest.raises(InputError):
            db.add_workspace_source("ws-1", {
                "id": source_id, "media_id": 42, "title": "My Video",
                "source_type": "video", "review_state": review_state,
            })

    def test_list_sources_ordered_by_position(self, db):
        db.add_workspace_source("ws-1", {
            "id": "src-a", "media_id": 1, "title": "A",
            "source_type": "video", "position": 2,
        })
        db.add_workspace_source("ws-1", {
            "id": "src-b", "media_id": 2, "title": "B",
            "source_type": "pdf", "position": 1,
        })
        sources = db.list_workspace_sources("ws-1")
        assert sources[0]["id"] == "src-b"
        assert sources[1]["id"] == "src-a"

    def test_update_source_with_version_check(self, db):
        db.add_workspace_source("ws-1", {
            "id": "src-1", "media_id": 1, "title": "Old",
            "source_type": "video",
        })
        updated = db.update_workspace_source("ws-1", "src-1", {"title": "New"}, expected_version=1)
        assert updated["title"] == "New"
        assert updated["version"] == 2

    def test_update_source_stale_version_raises(self, db):
        db.add_workspace_source("ws-1", {
            "id": "src-1", "media_id": 1, "title": "X",
            "source_type": "video",
        })
        db.update_workspace_source("ws-1", "src-1", {"title": "Y"}, expected_version=1)
        with pytest.raises(ConflictError):
            db.update_workspace_source("ws-1", "src-1", {"title": "Z"}, expected_version=1)

    def test_update_source_review_state_sets_and_clears_review_metadata(self, db, monkeypatch):
        timestamps = iter(
            [
                "2026-07-18T18:18:10.000Z",
                "2026-07-18T18:18:10.001Z",
                "2026-07-18T18:18:10.002Z",
                "2026-07-18T18:18:10.003Z",
            ]
        )
        monkeypatch.setattr(db, "_get_current_utc_timestamp_iso", lambda: next(timestamps))
        created = db.add_workspace_source("ws-1", {
            "id": "src-1", "media_id": 1, "title": "X",
            "source_type": "video", "review_state": "needs_review",
        })

        reviewed = db.update_workspace_source(
            "ws-1",
            "src-1",
            {"review_state": "reviewed"},
            expected_version=created["version"],
            actor_user_id=" reviewer-1 ",
        )

        assert reviewed["review_state"] == "reviewed"
        assert reviewed["review_state_updated_at"] != created["review_state_updated_at"]
        assert reviewed["reviewed_at"] == reviewed["review_state_updated_at"]
        assert reviewed["reviewed_by_user_id"] == "reviewer-1"
        assert reviewed["version"] == created["version"] + 1

        needs_review = db.update_workspace_source(
            "ws-1",
            "src-1",
            {"review_state": "needs_review"},
            expected_version=reviewed["version"],
        )

        assert needs_review["review_state"] == "needs_review"
        assert needs_review["review_state_updated_at"] != reviewed["review_state_updated_at"]
        assert needs_review["reviewed_at"] is None
        assert needs_review["reviewed_by_user_id"] is None
        assert needs_review["version"] == reviewed["version"] + 1

    @pytest.mark.parametrize(
        "actor_user_id",
        [None, "", "   ", 123],
        ids=["none", "empty", "blank", "non-string"],
    )
    def test_update_source_reviewed_requires_non_empty_string_actor(self, db, actor_user_id):
        created = db.add_workspace_source("ws-1", {
            "id": "src-1", "media_id": 1, "title": "X",
            "source_type": "video", "review_state": "needs_review",
        })

        with pytest.raises(InputError):
            db.update_workspace_source(
                "ws-1",
                "src-1",
                {"review_state": "reviewed"},
                expected_version=created["version"],
                actor_user_id=actor_user_id,
            )

        unchanged = db.get_workspace_source("ws-1", "src-1")
        assert unchanged["review_state"] == "needs_review"
        assert unchanged["version"] == created["version"]

    def test_update_source_rejects_invalid_review_state(self, db):
        created = db.add_workspace_source("ws-1", {
            "id": "src-1", "media_id": 1, "title": "X",
            "source_type": "video",
        })

        with pytest.raises(InputError):
            db.update_workspace_source(
                "ws-1",
                "src-1",
                {"review_state": "invalid"},
                expected_version=created["version"],
            )

    def test_delete_source(self, db):
        db.add_workspace_source("ws-1", {
            "id": "src-1", "media_id": 1, "title": "X",
            "source_type": "video",
        })
        db.delete_workspace_source("ws-1", "src-1")
        assert db.list_workspace_sources("ws-1") == []

    def test_batch_update_selection(self, db):
        db.add_workspace_source("ws-1", {
            "id": "src-a", "media_id": 1, "title": "A",
            "source_type": "video",
        })
        db.add_workspace_source("ws-1", {
            "id": "src-b", "media_id": 2, "title": "B",
            "source_type": "pdf",
        })
        db.update_workspace_source_selection("ws-1", selected_ids=["src-a"])
        sources = db.list_workspace_sources("ws-1")
        sel = {s["id"]: s["selected"] for s in sources}
        assert sel["src-a"] in (True, 1)
        assert sel["src-b"] in (False, 0)

    def test_batch_update_review_state_deduplicates_ids_and_leaves_unrelated_rows_unchanged(self, db):
        for source_id in ("src-a", "src-b", "src-c"):
            db.add_workspace_source("ws-1", {
                "id": source_id, "media_id": 1, "title": source_id,
                "source_type": "video", "review_state": "needs_review",
            })

        updated = db.update_workspace_source_review_states(
            "ws-1",
            ["src-a", "src-a", "src-b"],
            "reviewed",
            " reviewer-1 ",
        )

        assert [source["id"] for source in updated] == ["src-a", "src-b"]
        assert all(source["review_state"] == "reviewed" for source in updated)
        assert all(source["reviewed_at"] == source["review_state_updated_at"] for source in updated)
        assert all(source["reviewed_by_user_id"] == "reviewer-1" for source in updated)
        assert all(source["version"] == 2 for source in updated)
        unrelated = db.get_workspace_source("ws-1", "src-c")
        assert unrelated["review_state"] == "needs_review"
        assert unrelated["version"] == 1

    def test_batch_update_review_state_preserves_exact_source_ids(self, db):
        for source_id in ("src", " src "):
            db.add_workspace_source("ws-1", {
                "id": source_id, "media_id": 1, "title": repr(source_id),
                "source_type": "video",
            })

        updated = db.update_workspace_source_review_states(
            "ws-1",
            [" src "],
            "needs_review",
            None,
        )

        assert [source["id"] for source in updated] == [" src "]
        assert db.get_workspace_source("ws-1", " src ")["review_state"] == "needs_review"
        assert db.get_workspace_source("ws-1", "src")["review_state"] == "unset"

    @pytest.mark.parametrize(
        "source_id",
        ["", "   ", None, 0],
        ids=["empty", "blank", "none", "non-string"],
    )
    def test_batch_update_review_state_rejects_invalid_source_ids(self, db, source_id):
        with pytest.raises(InputError, match="source_ids"):
            db.update_workspace_source_review_states(
                "ws-1",
                [source_id],
                "needs_review",
                None,
            )

    def test_postgres_review_backfill_statement_emits_iso_utc_text(self):
        class RecordingBackend:
            def __init__(self):
                self.statements = []

            def execute(self, statement, *, connection):
                self.statements.append(statement)

        backend = RecordingBackend()
        db = type("RecordingDB", (), {"backend": backend})()

        CharactersRAGDB._ensure_workspace_subresource_schema_postgres(db, object())

        backfill = next(
            statement
            for statement in backend.statements
            if "SET review_state_updated_at" in statement
        )
        assert "to_char(" in backfill
        assert 'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"' in backfill

    def test_batch_update_review_state_fails_atomically_for_missing_source(self, db):
        created = db.add_workspace_source("ws-1", {
            "id": "src-a", "media_id": 1, "title": "A",
            "source_type": "video", "review_state": "needs_review",
        })

        with pytest.raises(ConflictError):
            db.update_workspace_source_review_states(
                "ws-1",
                ["src-a", "missing"],
                "reviewed",
                "reviewer-1",
            )

        unchanged = db.get_workspace_source("ws-1", "src-a")
        assert unchanged["review_state"] == "needs_review"
        assert unchanged["version"] == created["version"]

    @pytest.mark.parametrize(
        "actor_user_id",
        [None, "", "   ", 123],
        ids=["none", "empty", "blank", "non-string"],
    )
    def test_batch_update_reviewed_requires_non_empty_string_actor(self, db, actor_user_id):
        created = db.add_workspace_source("ws-1", {
            "id": "src-1", "media_id": 1, "title": "X",
            "source_type": "video", "review_state": "needs_review",
        })

        with pytest.raises(InputError):
            db.update_workspace_source_review_states(
                "ws-1",
                ["src-1"],
                "reviewed",
                actor_user_id,
            )

        unchanged = db.get_workspace_source("ws-1", "src-1")
        assert unchanged["review_state"] == "needs_review"
        assert unchanged["version"] == created["version"]

    def test_batch_reorder(self, db):
        db.add_workspace_source("ws-1", {
            "id": "src-a", "media_id": 1, "title": "A",
            "source_type": "video",
        })
        db.add_workspace_source("ws-1", {
            "id": "src-b", "media_id": 2, "title": "B",
            "source_type": "pdf",
        })
        db.reorder_workspace_sources("ws-1", ["src-b", "src-a"])
        sources = db.list_workspace_sources("ws-1")
        assert sources[0]["id"] == "src-b"
        assert sources[0]["position"] == 0
        assert sources[1]["id"] == "src-a"
        assert sources[1]["position"] == 1

    def test_existing_workspace_source_schema_is_backfilled(self, tmp_path):
        db_path = tmp_path / "pre-review.db"
        seed = CharactersRAGDB(db_path=str(db_path), client_id="user-1")
        seed.upsert_workspace("ws-1", "Test WS")
        seed.close_all_connections()

        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("DROP TABLE workspace_sources")
            conn.execute("""
                CREATE TABLE workspace_sources (
                    id            TEXT NOT NULL,
                    workspace_id  TEXT NOT NULL,
                    media_id      INTEGER NOT NULL,
                    title         TEXT NOT NULL,
                    source_type   TEXT NOT NULL,
                    url           TEXT,
                    position      INTEGER NOT NULL DEFAULT 0,
                    selected      BOOLEAN NOT NULL DEFAULT 1,
                    added_at      TEXT NOT NULL,
                    version       INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (workspace_id, id)
                )
            """)
            conn.executemany(
                "INSERT INTO workspace_sources "
                "(id, workspace_id, media_id, title, source_type, added_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    ("src-1", "ws-1", 1, "Existing", "pdf", "2026-01-02T03:04:05.000Z"),
                    ("src-blank-added-at", "ws-1", 2, "Malformed", "pdf", "   "),
                ],
            )

        migrated = CharactersRAGDB(db_path=str(db_path), client_id="user-1")
        try:
            source = migrated.get_workspace_source("ws-1", "src-1")
            assert source["review_state"] == "unset"
            assert source["review_state_updated_at"] == source["added_at"]
            assert source["reviewed_at"] is None
            assert source["reviewed_by_user_id"] is None

            malformed = migrated.get_workspace_source("ws-1", "src-blank-added-at")
            assert malformed["review_state_updated_at"].strip()
            assert malformed["review_state_updated_at"] != malformed["added_at"]
            assert datetime.fromisoformat(
                malformed["review_state_updated_at"].replace("Z", "+00:00")
            ).tzinfo is not None
        finally:
            migrated.close_all_connections()

    def test_workspace_source_schema_ensure_is_operationally_idempotent(self, db):
        db.add_workspace_source("ws-1", {
            "id": "src-1", "media_id": 1, "title": "X",
            "source_type": "video",
        })

        with db.transaction() as conn:
            changes_before = conn.total_changes
            db._ensure_workspace_subresource_schema_sqlite(conn)
            assert conn.total_changes == changes_before


class TestWorkspaceArtifacts:
    def test_add_artifact(self, db):
        art = db.add_workspace_artifact("ws-1", {
            "id": "art-1", "artifact_type": "summary", "title": "Summary",
        })
        assert art["id"] == "art-1"
        assert art["artifact_type"] == "summary"

    def test_add_traceable_artifact_contract_fields_and_initial_version(self, db):
        art = db.add_workspace_artifact("ws-1", {
            "id": "brief-1",
            "artifact_type": "workspace_brief",
            "title": "ACP Research Brief",
            "status": "completed",
            "content": "# Brief\nGrounded answer.",
            "content_type": "text/markdown",
            "preview_text": "Grounded answer.",
            "summary": "Executive summary",
            "review_state": "accepted",
            "owner_scope": "workspace",
            "owner_id": "ws-1",
            "producer_metadata": {
                "producer_type": "acp",
                "producer_id": "task-42",
                "run_id": "run-7",
                "session_id": "session-abc",
            },
            "source_lineage": {
                "sources": [
                    {
                        "source_id": "src-1",
                        "source_type": "media",
                        "label": "Transcript",
                        "citation_spans": [{"start": 0, "end": 18}],
                    }
                ]
            },
            "review_metadata": {"reviewer_id": "reviewer-1", "decision": "accepted"},
            "version_metadata": {"revision_reason": "initial"},
            "export_refs": [{"format": "md", "file_id": 101}],
            "redaction": {"support_safe": True, "redacted": False, "retention_class": "standard"},
        })

        assert art["review_state"] == "accepted"
        assert art["content_type"] == "text/markdown"
        assert art["preview_text"] == "Grounded answer."
        assert art["summary"] == "Executive summary"
        assert art["owner_scope"] == "workspace"
        assert art["owner_id"] == "ws-1"
        assert art["root_artifact_id"] == "brief-1"
        assert art["artifact_version_id"] == "brief-1:v1"
        assert art["previous_version_id"] is None
        assert art["producer_metadata"]["producer_type"] == "acp"
        assert art["source_lineage"]["sources"][0]["source_id"] == "src-1"
        assert art["review_metadata"]["decision"] == "accepted"
        assert art["version_metadata"]["revision_reason"] == "initial"
        assert art["export_refs"][0]["file_id"] == 101
        assert art["redaction"]["support_safe"] is True
        assert art["schema_version"] == 1

        versions = db.list_workspace_artifact_versions("ws-1", "brief-1")
        assert len(versions) == 1
        assert versions[0]["artifact_version_id"] == "brief-1:v1"
        assert versions[0]["review_state"] == "accepted"
        assert versions[0]["source_lineage"]["sources"][0]["label"] == "Transcript"

    def test_add_artifact_returns_inserted_row_without_post_commit_reload(
        self,
        db,
        monkeypatch,
    ):
        monkeypatch.setattr(
            db,
            "_get_workspace_artifact",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("late reload")),
        )

        artifact = db.add_workspace_artifact(
            "ws-1",
            {"id": "art-inline", "artifact_type": "summary", "title": "Inline"},
        )

        assert artifact["id"] == "art-inline"
        assert artifact["artifact_version_id"] == "art-inline:v1"

    def test_artifact_version_ids_are_server_owned_on_create_and_update(self, db):
        created = db.add_workspace_artifact("ws-1", {
            "id": "brief-1",
            "artifact_type": "workspace_brief",
            "title": "Draft Brief",
            "root_artifact_id": "forged-root",
            "artifact_version_id": "forged:v99",
            "previous_version_id": "forged:v98",
        })

        assert created["root_artifact_id"] == "brief-1"
        assert created["artifact_version_id"] == "brief-1:v1"
        assert created["previous_version_id"] is None

        updated = db.update_workspace_artifact(
            "ws-1",
            "brief-1",
            {
                "title": "Updated Brief",
                "root_artifact_id": "rewired-root",
                "artifact_version_id": "rewired:v100",
                "previous_version_id": "rewired:v99",
            },
            expected_version=created["version"],
        )

        assert updated["root_artifact_id"] == "brief-1"
        assert updated["artifact_version_id"] == "brief-1:v2"
        assert updated["previous_version_id"] == "brief-1:v1"
        versions = db.list_workspace_artifact_versions("ws-1", "brief-1")
        assert [version["artifact_version_id"] for version in versions] == ["brief-1:v1", "brief-1:v2"]

    def test_workspace_artifact_json_decode_failure_is_logged(self, monkeypatch):
        warnings: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def _capture_warning(*args: object, **kwargs: object) -> None:
            warnings.append((args, kwargs))

        monkeypatch.setattr(
            "tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB.logger.warning",
            _capture_warning,
        )

        loaded = CharactersRAGDB._load_workspace_artifact_json("{bad-json", {}, field_name="source_lineage_json")

        assert loaded == {}
        assert warnings
        assert warnings[0][0][1] == "source_lineage_json"

    def test_list_artifacts(self, db):
        db.add_workspace_artifact("ws-1", {
            "id": "art-1", "artifact_type": "summary", "title": "S1",
        })
        db.add_workspace_artifact("ws-1", {
            "id": "art-2", "artifact_type": "podcast", "title": "P1",
        })
        arts = db.list_workspace_artifacts("ws-1")
        assert len(arts) == 2

    def test_update_artifact_with_version_check(self, db):
        db.add_workspace_artifact("ws-1", {
            "id": "art-1", "artifact_type": "summary", "title": "Old",
        })
        updated = db.update_workspace_artifact("ws-1", "art-1", {"title": "New"}, expected_version=1)
        assert updated["title"] == "New"
        assert updated["version"] == 2

    def test_update_artifact_creates_new_traceable_version_record(self, db):
        created = db.add_workspace_artifact("ws-1", {
            "id": "brief-1",
            "artifact_type": "workspace_brief",
            "title": "Draft Brief",
            "content": "Draft",
            "review_state": "draft",
            "producer_metadata": {"producer_type": "acp", "producer_id": "task-42"},
            "source_lineage": {"sources": [{"source_id": "src-1", "label": "Transcript"}]},
        })

        updated = db.update_workspace_artifact(
            "ws-1",
            "brief-1",
            {
                "title": "Revised Brief",
                "content": "Needs citations.",
                "review_state": "needs_revision",
                "review_metadata": {"decision": "needs_revision", "reason_code": "missing_citation"},
                "version_metadata": {"revision_reason": "reviewer_requested_changes"},
            },
            expected_version=created["version"],
        )

        assert updated["version"] == 2
        assert updated["review_state"] == "needs_revision"
        assert updated["artifact_version_id"] == "brief-1:v2"
        assert updated["previous_version_id"] == "brief-1:v1"
        assert updated["review_metadata"]["reason_code"] == "missing_citation"
        assert updated["version_metadata"]["revision_reason"] == "reviewer_requested_changes"

        versions = db.list_workspace_artifact_versions("ws-1", "brief-1")
        assert [version["artifact_version_id"] for version in versions] == ["brief-1:v1", "brief-1:v2"]
        assert versions[0]["review_state"] == "draft"
        assert versions[1]["review_state"] == "needs_revision"

    def test_update_artifact_stale_version_raises(self, db):
        db.add_workspace_artifact("ws-1", {
            "id": "art-1", "artifact_type": "summary", "title": "X",
        })
        db.update_workspace_artifact("ws-1", "art-1", {"title": "Y"}, expected_version=1)
        with pytest.raises(ConflictError):
            db.update_workspace_artifact("ws-1", "art-1", {"title": "Z"}, expected_version=1)

    def test_delete_artifact(self, db):
        db.add_workspace_artifact("ws-1", {
            "id": "art-1", "artifact_type": "summary", "title": "X",
        })
        db.delete_workspace_artifact("ws-1", "art-1")
        assert db.list_workspace_artifacts("ws-1") == []

    def test_delete_artifact_removes_version_history_for_recreate(self, db):
        db.add_workspace_artifact("ws-1", {
            "id": "art-1", "artifact_type": "summary", "title": "X",
        })

        db.delete_workspace_artifact("ws-1", "art-1")
        recreated = db.add_workspace_artifact("ws-1", {
            "id": "art-1", "artifact_type": "summary", "title": "Recreated",
        })

        assert recreated["artifact_version_id"] == "art-1:v1"
        assert len(db.list_workspace_artifact_versions("ws-1", "art-1")) == 1


class TestWorkspaceNotes:
    def test_add_note(self, db):
        note = db.add_workspace_note("ws-1", {
            "title": "My Note", "content": "Hello",
        })
        assert note["title"] == "My Note"
        assert note["version"] == 1

    def test_add_note_returns_inserted_row_without_post_commit_reload(
        self,
        db,
        monkeypatch,
    ):
        monkeypatch.setattr(
            db,
            "_get_workspace_note",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("late reload")),
        )

        note = db.add_workspace_note(
            "ws-1",
            {"title": "Inline", "content": "Committed", "keywords": ["one"]},
        )

        assert note["title"] == "Inline"
        assert note["keywords_json"] == '["one"]'

    def test_list_notes_excludes_deleted(self, db):
        db.add_workspace_note("ws-1", {"title": "N1", "content": ""})
        n2 = db.add_workspace_note("ws-1", {"title": "N2", "content": ""})
        db.delete_workspace_note("ws-1", n2["id"])
        notes = db.list_workspace_notes("ws-1")
        assert len(notes) == 1
        assert notes[0]["title"] == "N1"

    def test_update_note_with_version_check(self, db):
        note = db.add_workspace_note("ws-1", {"title": "Old", "content": ""})
        updated = db.update_workspace_note("ws-1", note["id"], {"title": "New"}, expected_version=1)
        assert updated["title"] == "New"
        assert updated["version"] == 2


class TestWorkspaceSettings:
    def test_update_workspace_banner_settings(self, db):
        ws = db.update_workspace("ws-1", {
            "banner_title": "My Project",
            "banner_subtitle": "Research notes",
        }, expected_version=1)
        assert ws["banner_title"] == "My Project"
        assert ws["banner_subtitle"] == "Research notes"

    def test_update_workspace_audio_settings(self, db):
        ws = db.update_workspace("ws-1", {
            "audio_provider": "openai",
            "audio_model": "tts-1",
            "audio_voice": "alloy",
        }, expected_version=1)
        assert ws["audio_provider"] == "openai"
        assert ws["audio_model"] == "tts-1"


class TestFKCascadeOnHardDelete:
    def test_hard_delete_workspace_cascades_to_sources(self, db):
        db.add_workspace_source("ws-1", {
            "id": "src-1", "media_id": 1, "title": "X",
            "source_type": "video",
        })
        db.hard_delete_workspace("ws-1")
        assert db.list_workspace_sources("ws-1") == []

    def test_hard_delete_workspace_cascades_to_artifacts(self, db):
        db.add_workspace_artifact("ws-1", {
            "id": "art-1", "artifact_type": "summary", "title": "X",
        })
        db.hard_delete_workspace("ws-1")
        assert db.list_workspace_artifacts("ws-1") == []

    def test_hard_delete_workspace_cascades_to_notes(self, db):
        db.add_workspace_note("ws-1", {"title": "N", "content": ""})
        db.hard_delete_workspace("ws-1")
        assert db.list_workspace_notes("ws-1") == []
