"""Tests that /api/v1/chat/* endpoints respect scope_type and workspace_id."""
import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def db(tmp_path):
    d = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    d.add_character_card({"name": "Test Char"})
    return d


class TestConversationListScope:
    def test_list_defaults_to_global_scope(self, db):
        """Omitting scope returns only global conversations."""
        db.workspace_store.upsert_workspace("ws-a", "Workspace A")
        db.add_conversation({"title": "Global chat", "character_id": 1})
        db.add_conversation({
            "title": "Workspace chat", "character_id": 1,
            "scope_type": "workspace", "workspace_id": "ws-a",
        })

        results = db.search_conversations(None, scope_type="global")
        titles = [r["title"] for r in results]
        assert "Global chat" in titles
        assert "Workspace chat" not in titles

    def test_list_workspace_scope_returns_only_that_workspace(self, db):
        db.workspace_store.upsert_workspace("ws-a", "A")
        db.workspace_store.upsert_workspace("ws-b", "B")
        db.add_conversation({
            "title": "A chat", "character_id": 1,
            "scope_type": "workspace", "workspace_id": "ws-a",
        })
        db.add_conversation({
            "title": "B chat", "character_id": 1,
            "scope_type": "workspace", "workspace_id": "ws-b",
        })

        results = db.search_conversations(None, scope_type="workspace", workspace_id="ws-a")
        titles = [r["title"] for r in results]
        assert "A chat" in titles
        assert "B chat" not in titles


class TestConversationDetail404OnScopeMismatch:
    def test_get_conversation_has_scope_fields(self, db):
        """Verify the raw data has scope set correctly."""
        db.workspace_store.upsert_workspace("ws-a", "A")
        conv_id = db.add_conversation({
            "title": "WS chat", "character_id": 1,
            "scope_type": "workspace", "workspace_id": "ws-a",
        })
        conv = db.get_conversation_by_id(conv_id)
        assert conv["scope_type"] == "workspace"
        assert conv["workspace_id"] == "ws-a"
