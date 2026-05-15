"""Tests for workspace sub-resource tables: sources, artifacts, notes."""
import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError


@pytest.fixture
def db(tmp_path):
    d = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    d.upsert_workspace("ws-1", "Test WS")
    return d


class TestWorkspaceSources:
    def test_add_source(self, db):
        src = db.add_workspace_source("ws-1", {
            "id": "src-1", "media_id": 42, "title": "My Video",
            "source_type": "video",
        })
        assert src["id"] == "src-1"
        assert src["media_id"] == 42
        assert src["version"] == 1

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
