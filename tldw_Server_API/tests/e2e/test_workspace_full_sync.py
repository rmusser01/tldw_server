"""End-to-end tests for full workspace sync (Phase 2)."""
import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError


@pytest.fixture
def db(tmp_path):
    d = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    return d


class TestFullWorkspaceSync:
    def test_workspace_with_all_sub_resources(self, db):
        db.workspace_store.upsert_workspace("ws-1", "Full WS")
        db.workspace_store.add_workspace_source("ws-1", {
            "id": "src-1", "media_id": 1, "title": "Video",
            "source_type": "video",
        })
        db.workspace_store.add_workspace_artifact("ws-1", {
            "id": "art-1", "artifact_type": "summary", "title": "Summary",
        })
        db.workspace_store.add_workspace_note("ws-1", {"title": "Note", "content": "body"})

        assert len(db.workspace_store.list_workspace_sources("ws-1")) == 1
        assert len(db.workspace_store.list_workspace_artifacts("ws-1")) == 1
        assert len(db.workspace_store.list_workspace_notes("ws-1")) == 1

        ws = db.workspace_store.update_workspace("ws-1", {
            "banner_title": "Project X",
            "audio_provider": "openai",
        }, expected_version=1)
        assert ws["banner_title"] == "Project X"
        assert ws["audio_provider"] == "openai"

    def test_hard_delete_cascades_sub_resources(self, db):
        db.workspace_store.upsert_workspace("ws-1", "WS")
        db.workspace_store.add_workspace_source("ws-1", {
            "id": "src-1", "media_id": 1, "title": "V",
            "source_type": "video",
        })
        db.workspace_store.add_workspace_artifact("ws-1", {
            "id": "art-1", "artifact_type": "summary", "title": "S",
        })
        db.workspace_store.add_workspace_note("ws-1", {"title": "N", "content": ""})

        db.workspace_store.hard_delete_workspace("ws-1")

        assert db.workspace_store.list_workspace_sources("ws-1") == []
        assert db.workspace_store.list_workspace_artifacts("ws-1") == []
        assert db.workspace_store.list_workspace_notes("ws-1") == []

    def test_optimistic_locking_across_sub_resources(self, db):
        db.workspace_store.upsert_workspace("ws-1", "WS")
        db.workspace_store.add_workspace_source("ws-1", {
            "id": "src-1", "media_id": 1, "title": "V",
            "source_type": "video",
        })

        db.workspace_store.update_workspace_source("ws-1", "src-1", {"title": "V2"}, expected_version=1)
        with pytest.raises(ConflictError):
            db.workspace_store.update_workspace_source("ws-1", "src-1", {"title": "V3"}, expected_version=1)
