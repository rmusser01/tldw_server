"""End-to-end tests for workspace chat scope isolation at the DB level."""
import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def db(tmp_path):
    d = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    d.add_character_card({"name": "Test Char"})
    return d


class TestWorkspaceScopeEndToEnd:
    def test_full_isolation_workflow(self, db):
        # Create global chat
        global_id = db.add_conversation({"title": "Global", "character_id": 1})

        # Create workspace A with chat
        db.workspace_store.upsert_workspace("ws-a", "Workspace A")
        ws_a_id = db.add_conversation({
            "title": "WS-A Chat", "character_id": 1,
            "scope_type": "workspace", "workspace_id": "ws-a",
        })

        # Create workspace B with chat
        db.workspace_store.upsert_workspace("ws-b", "Workspace B")
        ws_b_id = db.add_conversation({
            "title": "WS-B Chat", "character_id": 1,
            "scope_type": "workspace", "workspace_id": "ws-b",
        })

        # Verify isolation
        global_results = db.search_conversations(None, scope_type="global")
        ws_a_results = db.search_conversations(None, scope_type="workspace", workspace_id="ws-a")
        ws_b_results = db.search_conversations(None, scope_type="workspace", workspace_id="ws-b")

        global_ids = [r["id"] for r in global_results]
        ws_a_ids = [r["id"] for r in ws_a_results]
        ws_b_ids = [r["id"] for r in ws_b_results]

        assert global_id in global_ids
        assert ws_a_id not in global_ids
        assert ws_b_id not in global_ids

        assert ws_a_id in ws_a_ids
        assert global_id not in ws_a_ids
        assert ws_b_id not in ws_a_ids

        assert ws_b_id in ws_b_ids
        assert global_id not in ws_b_ids
        assert ws_a_id not in ws_b_ids

    def test_workspace_delete_cascade(self, db):
        db.workspace_store.upsert_workspace("ws-a", "A")
        ws_conv = db.add_conversation({
            "title": "WS Chat", "character_id": 1,
            "scope_type": "workspace", "workspace_id": "ws-a",
        })
        global_conv = db.add_conversation({"title": "Global", "character_id": 1})

        ws = db.workspace_store.get_workspace("ws-a")
        db.workspace_store.delete_workspace("ws-a", ws["version"])

        # Workspace chat should be soft-deleted (not returned by default)
        ws_results = db.search_conversations(None, scope_type="workspace", workspace_id="ws-a")
        assert ws_conv not in [r["id"] for r in ws_results]
        # Global chat untouched
        assert db.get_conversation_by_id(global_conv) is not None
        # Workspace itself is soft-deleted (not listed)
        assert db.workspace_store.get_workspace("ws-a") is None

    def test_existing_conversations_default_to_global(self, db):
        """Pre-existing conversations (before scope was added) should be global."""
        conv_id = db.add_conversation({"title": "Old chat", "character_id": 1})
        conv = db.get_conversation_by_id(conv_id)
        assert conv["scope_type"] == "global"
        assert conv["workspace_id"] is None
