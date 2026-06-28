# tldw_Server_API/tests/kanban/test_kanban_vector_search.py
"""
Unit tests for Kanban vector search helper methods.

Tests the KanbanVectorSearch class helper methods and utility functions
without requiring ChromaDB to be available.
"""
import hashlib

import pytest

from tldw_Server_API.app.core.DB_Management.kanban_vector_search import (
    get_kanban_collection_name,
    is_vector_search_available,
    KanbanVectorSearch,
    KANBAN_COLLECTION_PREFIX,
)


class TestGetKanbanCollectionName:
    """Tests for get_kanban_collection_name utility function."""

    def test_basic_user_id(self):

        """Test with a simple alphanumeric user_id."""
        result = get_kanban_collection_name("user123")
        assert result == f"{KANBAN_COLLECTION_PREFIX}user123"

    def test_user_id_with_hyphens(self):

        """Test that hyphens are replaced with underscores."""
        result = get_kanban_collection_name("user-123-abc")
        assert result == f"{KANBAN_COLLECTION_PREFIX}user_123_abc"

    def test_user_id_with_spaces(self):

        """Test that spaces are replaced with underscores."""
        result = get_kanban_collection_name("user 123 abc")
        assert result == f"{KANBAN_COLLECTION_PREFIX}user_123_abc"

    def test_user_id_with_mixed_chars(self):

        """Test with mixed hyphens and spaces."""
        result = get_kanban_collection_name("user-123 abc-def ghi")
        assert result == f"{KANBAN_COLLECTION_PREFIX}user_123_abc_def_ghi"

    def test_long_user_id_truncation(self):

        """Test that user_id longer than 50 chars uses a hash suffix."""
        long_id = "a" * 100
        result = get_kanban_collection_name(long_id)
        hash_suffix = hashlib.sha256(long_id.encode("utf-8")).hexdigest()[:16]
        expected_safe_user_id = f"{'a' * 33}_{hash_suffix}"
        assert result == f"{KANBAN_COLLECTION_PREFIX}{expected_safe_user_id}"
        assert len(result) == len(KANBAN_COLLECTION_PREFIX) + 50

    def test_long_user_id_hash_avoids_collision(self):

        """Test that very long user IDs remain unique even with shared prefixes."""
        user_id_1 = ("a" * 100) + "1"
        user_id_2 = ("a" * 100) + "2"
        result_1 = get_kanban_collection_name(user_id_1)
        result_2 = get_kanban_collection_name(user_id_2)
        assert result_1 != result_2
        assert len(result_1) == len(KANBAN_COLLECTION_PREFIX) + 50
        assert len(result_2) == len(KANBAN_COLLECTION_PREFIX) + 50

    def test_numeric_user_id(self):

        """Test with numeric user_id (gets converted to string)."""
        result = get_kanban_collection_name(12345)
        assert result == f"{KANBAN_COLLECTION_PREFIX}12345"

    def test_uuid_style_user_id(self):

        """Test with UUID-style user_id."""
        uuid_id = "550e8400-e29b-41d4-a716-446655440000"
        result = get_kanban_collection_name(uuid_id)
        expected = f"{KANBAN_COLLECTION_PREFIX}550e8400_e29b_41d4_a716_446655440000"
        assert result == expected


class TestIsVectorSearchAvailable:
    """Tests for is_vector_search_available utility function."""

    def test_returns_boolean(self):

        """Test that the function returns a boolean."""
        result = is_vector_search_available()
        assert isinstance(result, bool)


class TestKanbanVectorSearchHelpers:
    """Tests for KanbanVectorSearch helper methods."""

    @pytest.fixture
    def vector_search_instance(self):
        """Create a KanbanVectorSearch instance without embedding config (disabled)."""
        return KanbanVectorSearch(user_id="test_user", embedding_config=None)

    def test_available_property_false_without_config(self, vector_search_instance):

        """Test that available is False when no embedding_config is provided."""
        assert vector_search_instance.available is False

    def test_build_document_with_title_only(self, vector_search_instance):

        """Test _build_document with only a title."""
        card = {"title": "Test Card"}
        result = vector_search_instance._build_document(card)
        assert result == "Test Card"

    def test_build_document_with_title_and_description(self, vector_search_instance):

        """Test _build_document with title and description."""
        card = {
            "title": "Test Card",
            "description": "This is a test description"
        }
        result = vector_search_instance._build_document(card)
        assert result == "Test Card This is a test description"

    def test_build_document_with_labels(self, vector_search_instance):

        """Test _build_document with title, description, and labels."""
        card = {
            "title": "Test Card",
            "description": "Test description",
            "labels": [
                {"id": 1, "name": "Bug"},
                {"id": 2, "name": "Urgent"}
            ]
        }
        result = vector_search_instance._build_document(card)
        assert "Test Card" in result
        assert "Test description" in result
        assert "Labels: Bug, Urgent" in result

    def test_build_document_with_empty_labels(self, vector_search_instance):

        """Test _build_document with empty labels list."""
        card = {
            "title": "Test Card",
            "labels": []
        }
        result = vector_search_instance._build_document(card)
        assert result == "Test Card"
        assert "Labels:" not in result

    def test_build_document_with_labels_missing_name(self, vector_search_instance):

        """Test _build_document filters out labels without names."""
        card = {
            "title": "Test Card",
            "labels": [
                {"id": 1, "name": "Bug"},
                {"id": 2},  # No name
                {"id": 3, "name": ""}  # Empty name
            ]
        }
        result = vector_search_instance._build_document(card)
        assert "Labels: Bug" in result
        # Should only include "Bug" since others have no valid name

    def test_build_document_with_checklist_items(self, vector_search_instance):

        """Test _build_document includes checklist item names."""
        card = {
            "title": "Test Card",
            "checklist_items": ["First task", "", "Second task"]
        }
        result = vector_search_instance._build_document(card)
        assert "Test Card" in result
        assert "Checklist: First task; Second task" in result

    def test_build_metadata_basic(self, vector_search_instance):

        """Test _build_metadata with basic card data."""
        card = {
            "id": 123,
            "board_id": 1,
            "list_id": 5
        }
        result = vector_search_instance._build_metadata(card)
        assert result["card_id"] == 123
        assert result["board_id"] == 1
        assert result["list_id"] == 5

    def test_build_metadata_with_optional_fields(self, vector_search_instance):

        """Test _build_metadata includes optional fields when present."""
        card = {
            "id": 123,
            "board_id": 1,
            "list_id": 5,
            "priority": "high",
            "due_date": "2024-12-31",
            "created_at": "2024-01-01T00:00:00"
        }
        result = vector_search_instance._build_metadata(card)
        assert result["card_id"] == 123
        assert result["priority"] == "high"
        assert result["due_date"] == "2024-12-31"
        assert result["created_at"] == "2024-01-01T00:00:00"

    def test_build_metadata_with_labels(self, vector_search_instance):

        """Test _build_metadata includes label names when present."""
        card = {
            "id": 123,
            "board_id": 1,
            "list_id": 5,
            "labels": [
                {"id": 1, "name": "Bug"},
                {"id": 2, "name": "Urgent"},
                {"id": 3, "name": ""}
            ]
        }
        result = vector_search_instance._build_metadata(card)
        assert result["labels"] == ["Bug", "Urgent"]

    def test_build_metadata_without_optional_fields(self, vector_search_instance):

        """Test _build_metadata doesn't include missing optional fields."""
        card = {
            "id": 123,
            "board_id": 1,
            "list_id": 5
        }
        result = vector_search_instance._build_metadata(card)
        assert "priority" not in result
        assert "due_date" not in result
        assert "created_at" not in result

    def test_search_returns_empty_when_unavailable(self, vector_search_instance):

        """Test that search returns empty list when vector search is unavailable."""
        result = vector_search_instance.search(query="test")
        assert result == []

    def test_index_card_enqueues_embeddings_job(self, monkeypatch):
        from tldw_Server_API.app.core.DB_Management import kanban_vector_search as kvs

        instance = kvs.KanbanVectorSearch(
            user_id="1",
            embedding_config={"embedding_model": "test-model", "embedding_provider": "test-provider"},
        )
        instance._available = True
        instance._manager = object()

        captured: dict[str, object] = {}

        class _StubJobManager:
            def create_job(self, **kwargs):
                captured["root_payload"] = kwargs.get("payload")
                return {"id": 1, "uuid": "root-uuid"}

        def _fake_enqueue_content_job(*, payload, root_job_uuid, **_kwargs):
            captured["stage_payload"] = payload
            captured["root_uuid"] = root_job_uuid
            return "stream-id"

        monkeypatch.setattr(kvs, "_jobs_manager", lambda: _StubJobManager())
        monkeypatch.setattr(kvs.redis_pipeline, "enqueue_content_job", _fake_enqueue_content_job)
        monkeypatch.setattr(kvs.redis_pipeline, "allow_stub", lambda: True)
        monkeypatch.delenv("TEST_MODE", raising=False)

        card = {
            "id": 10,
            "title": "Queue Test",
            "description": "Queue content",
            "board_id": 1,
            "list_id": 2,
            "version": 7,
        }
        assert instance.index_card(card) is True

        stage_payload = captured["stage_payload"]
        assert stage_payload["collection_name"] == instance._collection_name
        assert stage_payload["document_id"] == "card_10"
        assert stage_payload["card_id"] == 10
        assert stage_payload["card_version"] == 7

    def test_index_card_fails_root_job_when_enqueue_fails(self, monkeypatch):
        from tldw_Server_API.app.core.DB_Management import kanban_vector_search as kvs

        instance = kvs.KanbanVectorSearch(
            user_id="1",
            embedding_config={"embedding_model": "test-model", "embedding_provider": "test-provider"},
        )
        instance._available = True
        instance._manager = object()

        captured: dict[str, object] = {}

        class _StubJobManager:
            def create_job(self, **_kwargs):
                return {"id": 99, "uuid": "root-uuid"}

            def fail_job(self, job_id, **kwargs):
                captured["failed_job_id"] = job_id
                captured["fail_kwargs"] = kwargs

        def _raise_enqueue_error(**_kwargs):
            raise kvs.redis_pipeline.RedisEnqueueError("redis unavailable")

        monkeypatch.setattr(kvs, "_jobs_manager", lambda: _StubJobManager())
        monkeypatch.setattr(kvs.redis_pipeline, "enqueue_content_job", _raise_enqueue_error)
        monkeypatch.setattr(kvs.redis_pipeline, "allow_stub", lambda: False)
        monkeypatch.delenv("TEST_MODE", raising=False)

        card = {
            "id": 10,
            "title": "Queue Test",
            "description": "Queue content",
            "board_id": 1,
            "list_id": 2,
            "version": 7,
        }
        assert instance.index_card(card) is False
        assert captured["failed_job_id"] == 99
        assert captured["fail_kwargs"]["retryable"] is False

    def test_index_card_returns_false_when_unavailable(self, vector_search_instance):

        """Test that index_card returns False when vector search is unavailable."""
        card = {"id": 1, "title": "Test"}
        result = vector_search_instance.index_card(card)
        assert result is False

    def test_remove_card_returns_false_when_unavailable(self, vector_search_instance):

        """Test that remove_card returns False when vector search is unavailable."""
        result = vector_search_instance.remove_card(card_id=1)
        assert result is False

    def test_reindex_all_cards_returns_failures_when_unavailable(self, vector_search_instance):

        """Test that reindex_all_cards reports all as failures when unavailable."""
        cards = [
            {"id": 1, "title": "Card 1"},
            {"id": 2, "title": "Card 2"},
            {"id": 3, "title": "Card 3"}
        ]
        success, failure = vector_search_instance.reindex_all_cards(cards)
        assert success == 0
        assert failure == 3

    def test_context_manager(self, vector_search_instance):

        """Test that the class works as a context manager."""
        with KanbanVectorSearch(user_id="test", embedding_config=None) as vs:
            assert vs.available is False
        # After exit, should still be clean (no error)

    def test_close_method(self, vector_search_instance):

        """Test that close() can be called safely even when unavailable."""
        vector_search_instance.close()
        assert vector_search_instance.available is False

    def test_factory_degrades_on_pyo3_style_panic(self, monkeypatch):

        from tldw_Server_API.app.core.DB_Management import kanban_vector_search as kvs

        class PanicException(BaseException):
            pass

        class _RaisingVectorSearch:
            def __init__(self, *_args, **_kwargs):
                raise PanicException("panic from rust backend")

        monkeypatch.setattr(kvs, "_CHROMADB_AVAILABLE", True)
        monkeypatch.setattr(kvs, "KanbanVectorSearch", _RaisingVectorSearch)

        result = kvs.create_kanban_vector_search(
            user_id="panic-user",
            embedding_config={"embedding_model": "test-model"},
        )

        assert result is None
