from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import chat as chat_router
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError, InputError
from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry


pytestmark = pytest.mark.unit


class _FakeMetricsRegistry:
    def __init__(self) -> None:
        self.increment_calls: list[tuple[str, dict[str, str] | None, int | float | None]] = []
        self.observe_calls: list[tuple[str, float, dict[str, str] | None]] = []

    def increment(self, name: str, value: int | float = 1, labels: dict[str, str] | None = None) -> None:
        self.increment_calls.append((name, labels, value))

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        self.observe_calls.append((name, value, labels))


def _build_app(db: CharactersRAGDB) -> TestClient:
    app = FastAPI()
    app.include_router(chat_router.router, prefix="/api/v1/chat")
    app.include_router(chat_router.conversations_alias_router, prefix="/api/v1/chats")
    app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id="user-1")
    return TestClient(app)


def _install_conversation_observability_spies(monkeypatch: pytest.MonkeyPatch):
    registry = _FakeMetricsRegistry()
    debug_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    error_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def capture_debug(*args, **kwargs):  # noqa: ANN002, ANN003
        if args and isinstance(args[0], str) and args[0].startswith("Conversation search completed:"):
            debug_calls.append((args, kwargs))

    def capture_error(*args, **kwargs):  # noqa: ANN002, ANN003
        if args and isinstance(args[0], str) and args[0].startswith("Conversation list failed:"):
            error_calls.append((args, kwargs))

    monkeypatch.setattr(chat_router, "get_metrics_registry", lambda: registry, raising=False)
    monkeypatch.setattr(chat_router.logger, "debug", capture_debug)
    monkeypatch.setattr(chat_router.logger, "error", capture_error)
    return registry, debug_calls, error_calls


def test_conversation_list_bm25_and_keywords(tmp_path):
    db_path = tmp_path / "chacha.db"
    db = CharactersRAGDB(db_path=str(db_path), client_id="user-1")
    app = _build_app(db)

    char_id = db.add_character_card(
        {
            "name": "Test Character",
            "description": "desc",
            "personality": "helpful",
            "system_prompt": "You are helpful.",
            "client_id": "user-1",
        }
    )
    conv1 = db.add_conversation(
        {
            "character_id": char_id,
            "title": "alpha alpha alpha",
            "client_id": "user-1",
        }
    )
    conv2 = db.add_conversation(
        {
            "character_id": char_id,
            "title": "alpha beta",
            "client_id": "user-1",
        }
    )
    kw_id = db.add_keyword("triage")
    db.link_conversation_to_keyword(conv1, kw_id)

    resp = app.get("/api/v1/chat/conversations", params={"query": "alpha", "order_by": "bm25"})
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["pagination"]["total"] >= 2
    items = payload["items"]
    assert items[0]["id"] == conv1
    assert items[0]["bm25_norm"] == pytest.approx(1.0, rel=1e-6)
    assert "triage" in items[0]["keywords"]


def test_chat_analytics_buckets(tmp_path):
    db_path = tmp_path / "chacha.db"
    db = CharactersRAGDB(db_path=str(db_path), client_id="user-1")
    app = _build_app(db)

    char_id = db.add_character_card(
        {
            "name": "Test Character",
            "description": "desc",
            "personality": "helpful",
            "system_prompt": "You are helpful.",
            "client_id": "user-1",
        }
    )
    db.add_conversation(
        {
            "character_id": char_id,
            "title": "Analytics A",
            "state": "in-progress",
            "client_id": "user-1",
        }
    )
    db.add_conversation(
        {
            "character_id": char_id,
            "title": "Analytics B",
            "state": "resolved",
            "client_id": "user-1",
        }
    )

    today = datetime.now(timezone.utc).date()
    start_date = (today - timedelta(days=1)).isoformat()
    end_date = (today + timedelta(days=1)).isoformat()

    resp = app.get(
        "/api/v1/chat/analytics",
        params={"start_date": start_date, "end_date": end_date, "bucket_granularity": "day"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["pagination"]["total"] >= 1
    assert data["bucket_granularity"] == "day"


def test_conversation_endpoints_expose_normalized_assistant_identity(tmp_path):
    db_path = tmp_path / "chacha.db"
    db = CharactersRAGDB(db_path=str(db_path), client_id="user-1")
    app = _build_app(db)

    conversation_id = db.add_conversation(
        {
            "assistant_kind": "persona",
            "assistant_id": "garden-helper",
            "persona_memory_mode": "read_only",
            "title": "Persona conversation",
            "client_id": "user-1",
        }
    )
    assert conversation_id is not None

    list_resp = app.get("/api/v1/chat/conversations")
    assert list_resp.status_code == 200, list_resp.text
    items = list_resp.json()["items"]
    item = next(entry for entry in items if entry["id"] == conversation_id)
    assert item["assistant_kind"] == "persona"
    assert item["assistant_id"] == "garden-helper"
    assert item["character_id"] is None
    assert item["persona_memory_mode"] == "read_only"

    conversation = db.get_conversation_by_id(conversation_id)
    assert conversation is not None

    patch_resp = app.patch(
        f"/api/v1/chat/conversations/{conversation_id}",
        json={
            "version": conversation["version"],
            "source": "api",
        },
    )
    assert patch_resp.status_code == 200, patch_resp.text
    patched = patch_resp.json()
    assert patched["assistant_kind"] == "persona"
    assert patched["assistant_id"] == "garden-helper"
    assert patched["character_id"] is None
    assert patched["persona_memory_mode"] == "read_only"

    tree_resp = app.get(f"/api/v1/chat/conversations/{conversation_id}/tree")
    assert tree_resp.status_code == 200, tree_resp.text
    metadata = tree_resp.json()["conversation"]
    assert metadata["assistant_kind"] == "persona"
    assert metadata["assistant_id"] == "garden-helper"
    assert metadata["character_id"] is None


def test_update_conversation_maps_conflict_error(tmp_path, monkeypatch):
    db_path = tmp_path / "chacha.db"
    db = CharactersRAGDB(db_path=str(db_path), client_id="user-1")
    app = _build_app(db)
    conversation_id = db.add_conversation(
        {
            "assistant_kind": "persona",
            "assistant_id": "conflict-helper",
            "persona_memory_mode": "read_only",
            "title": "Conflict thread",
            "client_id": "user-1",
        }
    )
    conversation = db.get_conversation_by_id(conversation_id)
    assert conversation is not None

    def _raise_conflict(_conversation_id: str, _update_data: dict, _expected_version: int):
        raise ConflictError("conversation update conflict")

    monkeypatch.setattr(db, "update_conversation", _raise_conflict)

    response = app.patch(
        f"/api/v1/chat/conversations/{conversation_id}",
        json={"version": conversation["version"], "title": "Updated"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "conversation update conflict"


def test_update_conversation_maps_input_error(tmp_path, monkeypatch):
    db_path = tmp_path / "chacha.db"
    db = CharactersRAGDB(db_path=str(db_path), client_id="user-1")
    app = _build_app(db)
    conversation_id = db.add_conversation(
        {
            "assistant_kind": "persona",
            "assistant_id": "invalid-helper",
            "persona_memory_mode": "read_only",
            "title": "Invalid thread",
            "client_id": "user-1",
        }
    )
    conversation = db.get_conversation_by_id(conversation_id)
    assert conversation is not None

    def _raise_input_error(_conversation_id: str, _update_data: dict, _expected_version: int):
        raise InputError("invalid conversation update")

    monkeypatch.setattr(db, "update_conversation", _raise_input_error)

    response = app.patch(
        f"/api/v1/chat/conversations/{conversation_id}",
        json={"version": conversation["version"], "title": "Updated"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid conversation update"


def test_get_chat_conversation_returns_metadata_for_knowledge_clients(tmp_path):
    db_path = tmp_path / "chacha.db"
    db = CharactersRAGDB(db_path=str(db_path), client_id="user-1")
    app = _build_app(db)

    conversation_id = db.add_conversation(
        {
            "assistant_kind": "persona",
            "assistant_id": "knowledge-helper",
            "persona_memory_mode": "read_only",
            "title": "Knowledge QA thread",
            "client_id": "user-1",
        }
    )
    assert conversation_id is not None
    keyword_id = db.add_keyword("__knowledge_QA__")
    db.link_conversation_to_keyword(conversation_id, keyword_id)

    response = app.get(f"/api/v1/chat/conversations/{conversation_id}")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["id"] == conversation_id
    assert payload["assistant_kind"] == "persona"
    assert payload["assistant_id"] == "knowledge-helper"
    assert payload["persona_memory_mode"] == "read_only"
    assert payload["keywords"] == ["__knowledge_QA__"]


def test_conversation_alias_filters_character_scope(tmp_path):
    db_path = tmp_path / "chacha.db"
    db = CharactersRAGDB(db_path=str(db_path), client_id="user-1")
    app = _build_app(db)

    char_id = db.add_character_card(
        {
            "name": "Character Scope",
            "description": "desc",
            "personality": "helpful",
            "system_prompt": "You are helpful.",
            "client_id": "user-1",
        }
    )
    db.add_conversation(
        {
            "id": "character-conv",
            "character_id": char_id,
            "title": "Quota review",
            "client_id": "user-1",
        }
    )
    db.add_conversation(
        {
            "id": "plain-conv",
            "character_id": None,
            "assistant_kind": "persona",
            "assistant_id": "plain-helper",
            "persona_memory_mode": "read_only",
            "title": "Quota review",
            "client_id": "user-1",
        }
    )

    resp = app.get(
        "/api/v1/chats/conversations",
        params={"query": "Quota", "character_scope": "non_character"},
    )

    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert [item["id"] for item in payload["items"]] == ["plain-conv"]
    assert payload["pagination"]["total"] == 1


def test_conversation_list_defaults_omitted_scope_to_global(tmp_path):
    db_path = tmp_path / "chacha.db"
    db = CharactersRAGDB(db_path=str(db_path), client_id="user-1")
    app = _build_app(db)

    char_id = db.add_character_card(
        {
            "name": "Scope Character",
            "description": "desc",
            "personality": "helpful",
            "system_prompt": "You are helpful.",
            "client_id": "user-1",
        }
    )
    db.upsert_workspace("ws-1", "Workspace One")
    db.add_conversation(
        {
            "id": "global-conv",
            "character_id": char_id,
            "title": "Global conversation",
            "client_id": "user-1",
        }
    )
    db.add_conversation(
        {
            "id": "workspace-conv",
            "character_id": char_id,
            "title": "Workspace conversation",
            "client_id": "user-1",
            "scope_type": "workspace",
            "workspace_id": "ws-1",
        }
    )

    response = app.get("/api/v1/chat/conversations")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert [item["id"] for item in payload["items"]] == ["global-conv"]
    assert payload["pagination"]["total"] == 1


def test_conversation_tree_requires_exact_scope_match(tmp_path):
    db_path = tmp_path / "chacha.db"
    db = CharactersRAGDB(db_path=str(db_path), client_id="user-1")
    app = _build_app(db)

    char_id = db.add_character_card(
        {
            "name": "Scope Tree Character",
            "description": "desc",
            "personality": "helpful",
            "system_prompt": "You are helpful.",
            "client_id": "user-1",
        }
    )
    db.upsert_workspace("ws-1", "Workspace One")
    conversation_id = db.add_conversation(
        {
            "id": "workspace-tree",
            "character_id": char_id,
            "title": "Workspace tree",
            "client_id": "user-1",
            "scope_type": "workspace",
            "workspace_id": "ws-1",
        }
    )
    assert conversation_id == "workspace-tree"

    missing_scope = app.get(f"/api/v1/chat/conversations/{conversation_id}/tree")
    assert missing_scope.status_code == 404

    wrong_scope = app.get(
        f"/api/v1/chat/conversations/{conversation_id}/tree",
        params={"scope_type": "workspace", "workspace_id": "ws-2"},
    )
    assert wrong_scope.status_code == 404

    correct_scope = app.get(
        f"/api/v1/chat/conversations/{conversation_id}/tree",
        params={"scope_type": "workspace", "workspace_id": "ws-1"},
    )
    assert correct_scope.status_code == 200, correct_scope.text
    payload = correct_scope.json()
    assert payload["conversation"]["scope_type"] == "workspace"
    assert payload["conversation"]["workspace_id"] == "ws-1"


def test_conversation_alias_rejects_incompatible_character_scope_and_character_id(tmp_path):
    db_path = tmp_path / "chacha.db"
    db = CharactersRAGDB(db_path=str(db_path), client_id="user-1")
    app = _build_app(db)

    resp = app.get(
        "/api/v1/chats/conversations",
        params={"character_scope": "non_character", "character_id": 12},
    )

    assert resp.status_code == 400
    assert "character_scope" in resp.json()["detail"]


def test_conversation_alias_uses_paged_db_search(tmp_path, monkeypatch):
    db_path = tmp_path / "chacha.db"
    db = CharactersRAGDB(db_path=str(db_path), client_id="user-1")
    app = _build_app(db)

    observed: dict[str, object] = {}
    now = datetime.now(timezone.utc)

    def fail_full_search(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("full search helper should not be used by the conversations endpoint")

    def fake_page_search(
        query: str | None,
        *,
        client_id: str | None = None,
        character_id: int | None = None,
        character_scope: str | None = None,
        state: str | None = None,
        topic_label: str | None = None,
        topic_prefix: bool = False,
        cluster_id: str | None = None,
        keywords: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        date_field: str = "last_modified",
        order_by: str = "recency",
        limit: int = 50,
        offset: int = 0,
        as_of: datetime | None = None,
        **kwargs,
    ):
        observed.update(
            {
                "query": query,
                "client_id": client_id,
                "character_id": character_id,
                "character_scope": character_scope,
                "order_by": order_by,
                "limit": limit,
                "offset": offset,
                "as_of": as_of,
                "date_field": date_field,
            }
        )
        return (
            [
                {
                    "id": "plain-conv",
                    "assistant_kind": "persona",
                    "assistant_id": "plain-helper",
                    "persona_memory_mode": "read_only",
                    "character_id": None,
                    "title": "Quota review",
                    "state": "in-progress",
                    "topic_label": None,
                    "bm25_norm": 0.42,
                    "last_modified": now.isoformat(),
                    "created_at": now.isoformat(),
                    "version": 3,
                    "cluster_id": None,
                    "source": None,
                    "external_ref": None,
                }
            ],
            7,
            0.91,
        )

    monkeypatch.setattr(db, "search_conversations", fail_full_search)
    monkeypatch.setattr(db, "search_conversations_page", fake_page_search, raising=False)

    resp = app.get(
        "/api/v1/chats/conversations",
        params={
            "query": "Quota",
            "character_scope": "non_character",
            "order_by": "hybrid",
            "limit": 1,
            "offset": 1,
        },
    )

    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["pagination"]["total"] == 7
    assert payload["pagination"]["limit"] == 1
    assert payload["pagination"]["offset"] == 1
    assert [item["id"] for item in payload["items"]] == ["plain-conv"]
    assert payload["items"][0]["bm25_norm"] == pytest.approx(0.42, rel=1e-6)
    assert observed["query"] == "Quota"
    assert observed["client_id"] == "user-1"
    assert observed["character_scope"] == "non_character"
    assert observed["order_by"] == "hybrid"
    assert observed["limit"] == 1
    assert observed["offset"] == 1
    assert observed["date_field"] == "last_modified"
    assert isinstance(observed["as_of"], datetime)


def test_conversation_alias_related_lookups_only_use_page_rows(tmp_path, monkeypatch):
    db_path = tmp_path / "chacha.db"
    db = CharactersRAGDB(db_path=str(db_path), client_id="user-1")
    app = _build_app(db)

    now = datetime.now(timezone.utc)
    requested_ids: dict[str, list[str]] = {}

    def fail_full_search(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("full search helper should not be used by the conversations endpoint")

    def fake_page_search(*args, **kwargs):  # noqa: ANN002, ANN003
        return (
            [
                {
                    "id": "page-only",
                    "assistant_kind": "persona",
                    "assistant_id": "page-helper",
                    "persona_memory_mode": "read_only",
                    "character_id": None,
                    "title": "Quota review",
                    "state": "in-progress",
                    "topic_label": "quota",
                    "bm25_norm": 1.0,
                    "last_modified": now.isoformat(),
                    "created_at": now.isoformat(),
                    "version": 1,
                    "cluster_id": None,
                    "source": None,
                    "external_ref": None,
                }
            ],
            12,
            1.0,
        )

    def capture_keywords(conversation_ids: list[str]):
        requested_ids["keywords"] = list(conversation_ids)
        return {"page-only": [{"keyword": "quota"}]}

    def capture_message_counts(conversation_ids: list[str], include_deleted: bool = False):
        requested_ids["message_counts"] = list(conversation_ids)
        return {"page-only": 4}

    monkeypatch.setattr(db, "search_conversations", fail_full_search)
    monkeypatch.setattr(db, "search_conversations_page", fake_page_search, raising=False)
    monkeypatch.setattr(db, "get_keywords_for_conversations", capture_keywords)
    monkeypatch.setattr(db, "count_messages_for_conversations", capture_message_counts)

    resp = app.get(
        "/api/v1/chats/conversations",
        params={"query": "Quota", "limit": 1, "offset": 0},
    )

    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["items"][0]["keywords"] == ["quota"]
    assert payload["items"][0]["message_count"] == 4
    assert requested_ids["keywords"] == ["page-only"]
    assert requested_ids["message_counts"] == ["page-only"]


def test_conversation_alias_passes_deleted_filters_to_paged_search(tmp_path, monkeypatch):
    db_path = tmp_path / "chacha.db"
    db = CharactersRAGDB(db_path=str(db_path), client_id="user-1")
    app = _build_app(db)

    observed: dict[str, object] = {}
    now = datetime.now(timezone.utc)

    def fake_page_search(
        query: str | None,
        *,
        include_deleted: bool = False,
        deleted_only: bool = False,
        **kwargs,
    ):
        observed.update(
            {
                "query": query,
                "include_deleted": include_deleted,
                "deleted_only": deleted_only,
            }
        )
        return (
            [
                {
                    "id": "deleted-conv",
                    "assistant_kind": "persona",
                    "assistant_id": "trash-helper",
                    "persona_memory_mode": "read_only",
                    "character_id": None,
                    "title": "Quota cleanup",
                    "state": "resolved",
                    "topic_label": "trash",
                    "bm25_norm": 1.0,
                    "last_modified": now.isoformat(),
                    "created_at": now.isoformat(),
                    "version": 2,
                    "cluster_id": None,
                    "source": None,
                    "external_ref": None,
                }
            ],
            1,
            1.0,
        )

    monkeypatch.setattr(db, "search_conversations_page", fake_page_search, raising=False)

    resp = app.get(
        "/api/v1/chats/conversations",
        params={"query": "Quota", "deleted_only": "true"},
    )

    assert resp.status_code == 200, resp.text
    assert [item["id"] for item in resp.json()["items"]] == ["deleted-conv"]
    assert observed == {
        "query": "Quota",
        "include_deleted": True,
        "deleted_only": True,
    }


def test_conversation_search_metrics_registered():
    registry = get_metrics_registry()

    assert "chat_conversation_search_requests_total" in registry.metrics
    assert "chat_conversation_search_duration_seconds" in registry.metrics


def test_conversation_alias_emits_success_metrics_and_debug_shape(tmp_path, monkeypatch):
    db_path = tmp_path / "chacha.db"
    db = CharactersRAGDB(db_path=str(db_path), client_id="user-1")
    app = _build_app(db)
    registry, debug_calls, _error_calls = _install_conversation_observability_spies(monkeypatch)

    now = datetime.now(timezone.utc)

    def fake_page_search(*args, **kwargs):  # noqa: ANN002, ANN003
        return (
            [
                {
                    "id": "plain-conv",
                    "assistant_kind": "persona",
                    "assistant_id": "plain-helper",
                    "persona_memory_mode": "read_only",
                    "character_id": None,
                    "title": "Quota review",
                    "state": "in-progress",
                    "topic_label": "quota",
                    "bm25_norm": 0.5,
                    "last_modified": now.isoformat(),
                    "created_at": now.isoformat(),
                    "version": 1,
                    "cluster_id": None,
                    "source": None,
                    "external_ref": None,
                }
            ],
            3,
            1.0,
        )

    monkeypatch.setattr(db, "search_conversations_page", fake_page_search, raising=False)

    resp = app.get(
        "/api/v1/chats/conversations",
        params={"query": "Quota", "order_by": "recency", "limit": 1, "offset": 0},
    )

    assert resp.status_code == 200, resp.text
    assert registry.increment_calls == [
        (
            "chat_conversation_search_requests_total",
            {
                "query_strategy": "fts",
                "order_by": "recency",
                "deleted_scope": "active",
                "outcome": "success",
            },
            1,
        )
    ]
    assert len(registry.observe_calls) == 1
    metric_name, metric_value, metric_labels = registry.observe_calls[0]
    assert metric_name == "chat_conversation_search_duration_seconds"
    assert metric_value >= 0.0
    assert metric_labels == {
        "query_strategy": "fts",
        "order_by": "recency",
        "deleted_scope": "active",
        "outcome": "success",
    }
    assert len(debug_calls) == 1
    debug_repr = repr(debug_calls[0])
    assert "query_strategy" in debug_repr
    assert "returned" in debug_repr
    assert "total" in debug_repr
    assert "Quota" not in debug_repr


def test_conversation_alias_emits_validation_outcome_metric(tmp_path, monkeypatch):
    db_path = tmp_path / "chacha.db"
    db = CharactersRAGDB(db_path=str(db_path), client_id="user-1")
    app = _build_app(db)
    registry, debug_calls, error_calls = _install_conversation_observability_spies(monkeypatch)

    resp = app.get(
        "/api/v1/chats/conversations",
        params={"character_scope": "non_character", "character_id": 12},
    )

    assert resp.status_code == 400
    assert registry.increment_calls == [
        (
            "chat_conversation_search_requests_total",
            {
                "query_strategy": "none",
                "order_by": "recency",
                "deleted_scope": "active",
                "outcome": "validation",
            },
            1,
        )
    ]
    assert len(registry.observe_calls) == 1
    assert registry.observe_calls[0][2] == {
        "query_strategy": "none",
        "order_by": "recency",
        "deleted_scope": "active",
        "outcome": "validation",
    }
    assert debug_calls == []
    assert error_calls == []


def test_conversation_alias_maps_db_input_error_with_validation_metric(tmp_path, monkeypatch):
    db_path = tmp_path / "chacha.db"
    db = CharactersRAGDB(db_path=str(db_path), client_id="user-1")
    app = _build_app(db)
    registry, debug_calls, error_calls = _install_conversation_observability_spies(monkeypatch)

    def raise_input_error(*args, **kwargs):  # noqa: ANN002, ANN003
        raise InputError("invalid conversation filter")

    monkeypatch.setattr(db, "search_conversations_page", raise_input_error, raising=False)

    resp = app.get(
        "/api/v1/chats/conversations",
        params={"query": "Quota"},
    )

    assert resp.status_code == 400
    assert resp.json()["detail"] == "invalid conversation filter"
    assert registry.increment_calls == [
        (
            "chat_conversation_search_requests_total",
            {
                "query_strategy": "fts",
                "order_by": "recency",
                "deleted_scope": "active",
                "outcome": "validation",
            },
            1,
        )
    ]
    assert len(registry.observe_calls) == 1
    assert registry.observe_calls[0][2] == {
        "query_strategy": "fts",
        "order_by": "recency",
        "deleted_scope": "active",
        "outcome": "validation",
    }
    assert debug_calls == []
    assert error_calls == []


def test_conversation_alias_emits_server_error_outcome_metric(tmp_path, monkeypatch):
    db_path = tmp_path / "chacha.db"
    db = CharactersRAGDB(db_path=str(db_path), client_id="user-1")
    app = _build_app(db)
    registry, _debug_calls, error_calls = _install_conversation_observability_spies(monkeypatch)

    def blow_up(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("db exploded")

    monkeypatch.setattr(db, "search_conversations_page", blow_up, raising=False)

    resp = app.get(
        "/api/v1/chats/conversations",
        params={"query": "Quota", "deleted_only": "true"},
    )

    assert resp.status_code == 500
    assert registry.increment_calls == [
        (
            "chat_conversation_search_requests_total",
            {
                "query_strategy": "deleted_text",
                "order_by": "recency",
                "deleted_scope": "deleted_only",
                "outcome": "server_error",
            },
            1,
        )
    ]
    assert len(registry.observe_calls) == 1
    assert registry.observe_calls[0][2] == {
        "query_strategy": "deleted_text",
        "order_by": "recency",
        "deleted_scope": "deleted_only",
        "outcome": "server_error",
    }
    assert len(error_calls) == 1
    assert "deleted_text" in repr(error_calls[0])
