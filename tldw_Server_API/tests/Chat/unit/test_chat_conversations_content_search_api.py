"""Conversation search over HTTP finds a chat by what was said in it (CS-02)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import chat as chat_router
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.unit

SEARCH_PATHS = ("/api/v1/chats/conversations", "/api/v1/chat/conversations")


class _FakeMetricsRegistry:
    """Record the search metrics an endpoint call emits."""

    def __init__(self) -> None:
        self.increment_calls: list[tuple[str, dict[str, str] | None]] = []

    def increment(self, name: str, value: int | float = 1, labels: dict[str, str] | None = None) -> None:
        self.increment_calls.append((name, labels))

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        return None


def _add_chat(db: CharactersRAGDB, conversation_id: str, title: str, *, owner: str = "user-1") -> str:
    """Create an owned persona conversation."""
    db.add_conversation(
        {
            "id": conversation_id,
            "root_id": conversation_id,
            "assistant_kind": "persona",
            "assistant_id": "persona-search",
            "persona_memory_mode": "read_only",
            "title": title,
            "client_id": owner,
        }
    )
    return conversation_id


def _say(db: CharactersRAGDB, conversation_id: str, content: str, *, owner: str = "user-1") -> str:
    """Append one live message and return its id."""
    message_id = db.add_message(
        {"conversation_id": conversation_id, "sender": "user", "content": content, "client_id": owner}
    )
    assert message_id is not None
    return message_id


@pytest.fixture
def ctx(tmp_path: Path) -> Iterator[SimpleNamespace]:
    """Serve the conversation routes for user-1 over a real SQLite chat store."""
    db = CharactersRAGDB(db_path=str(tmp_path / "content-search-api.db"), client_id="user-1")
    app = FastAPI()
    app.include_router(chat_router.router, prefix="/api/v1/chat")
    app.include_router(chat_router.conversations_alias_router, prefix="/api/v1/chats")
    app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id="user-1")
    try:
        with TestClient(app) as client:
            yield SimpleNamespace(db=db, client=client)
    finally:
        db.close_all_connections()


def _items(response: Any) -> list[dict[str, Any]]:
    assert response.status_code == 200, response.text
    return response.json()["items"]


@pytest.mark.parametrize("path", SEARCH_PATHS)
def test_search_finds_a_chat_by_a_word_from_its_messages(ctx: SimpleNamespace, path: str) -> None:
    """A bland title with the memorable word only in the messages is found, with a snippet."""
    _add_chat(ctx.db, "weekly", "Weekly sync")
    message_id = _say(ctx.db, "weekly", "Remember that the launch codeword is zebrafinch.")
    _add_chat(ctx.db, "unrelated", "Lunch plans")
    _say(ctx.db, "unrelated", "Sandwiches again?")

    response = ctx.client.get(path, params={"query": "zebrafinch", "search_in": "title,content"})

    items = _items(response)
    assert [item["id"] for item in items] == ["weekly"]
    assert items[0]["matched_in"] == ["content"]
    assert "zebrafinch" in items[0]["match_snippet"]
    assert items[0]["match_message_id"] == message_id
    assert response.json()["pagination"]["total"] == 1


def test_search_without_search_in_keeps_title_only_results(ctx: SimpleNamespace) -> None:
    """Existing clients see the same results; the new fields are present but empty."""
    _add_chat(ctx.db, "weekly", "Weekly sync")
    _say(ctx.db, "weekly", "The launch codeword is zebrafinch.")

    by_content = ctx.client.get("/api/v1/chats/conversations", params={"query": "zebrafinch"})
    by_title = ctx.client.get("/api/v1/chats/conversations", params={"query": "Weekly"})

    assert _items(by_content) == []
    title_items = _items(by_title)
    assert [item["id"] for item in title_items] == ["weekly"]
    assert title_items[0]["matched_in"] is None
    assert title_items[0]["match_snippet"] is None
    assert title_items[0]["match_message_id"] is None


def test_title_matches_come_first_and_paging_keeps_the_total(ctx: SimpleNamespace) -> None:
    """Title hits lead, content hits follow, and every page reports the full total."""
    _add_chat(ctx.db, "content-a", "Standup")
    _say(ctx.db, "content-a", "zebrafinch seen")
    _add_chat(ctx.db, "title-hit", "Zebrafinch notes")
    _add_chat(ctx.db, "content-b", "Retro")
    _say(ctx.db, "content-b", "another zebrafinch")
    params = {"query": "zebrafinch", "search_in": "title,content", "limit": 1}

    pages = [ctx.client.get("/api/v1/chats/conversations", params={**params, "offset": offset}) for offset in range(3)]

    ordered = [_items(page)[0]["id"] for page in pages]
    assert ordered[0] == "title-hit"
    assert set(ordered[1:]) == {"content-a", "content-b"}
    assert [page.json()["pagination"]["total"] for page in pages] == [3, 3, 3]
    assert [page.json()["pagination"]["has_more"] for page in pages] == [True, True, False]


def test_trashed_chats_deleted_messages_and_other_owners_are_excluded(ctx: SimpleNamespace) -> None:
    """Only the caller's live chats with a live matching message are returned."""
    _add_chat(ctx.db, "live", "Weekly sync")
    _say(ctx.db, "live", "zebrafinch stays")
    _add_chat(ctx.db, "trashed", "Planning")
    _say(ctx.db, "trashed", "zebrafinch in the trash")
    trashed = ctx.db.get_conversation_by_id("trashed")
    ctx.db.soft_delete_conversation("trashed", expected_version=trashed["version"])
    _add_chat(ctx.db, "message-deleted", "Retro")
    doomed = _say(ctx.db, "message-deleted", "zebrafinch removed")
    ctx.db.soft_delete_message(doomed, expected_version=1)
    _add_chat(ctx.db, "foreign", "Weekly sync", owner="user-2")
    _say(ctx.db, "foreign", "zebrafinch from someone else", owner="user-2")

    response = ctx.client.get(
        "/api/v1/chats/conversations",
        params={"query": "zebrafinch", "search_in": "title,content"},
    )

    assert [item["id"] for item in _items(response)] == ["live"]
    assert response.json()["pagination"]["total"] == 1
    assert "someone else" not in response.text
    assert "in the trash" not in response.text


@pytest.mark.parametrize("search_in", ["body", "title,messages", ","])
def test_unknown_search_in_is_a_client_error(ctx: SimpleNamespace, search_in: str) -> None:
    """A typo is reported instead of silently searching titles only."""
    response = ctx.client.get(
        "/api/v1/chats/conversations",
        params={"query": "zebrafinch", "search_in": search_in},
    )

    assert response.status_code == 400, response.text
    assert "search fields" in response.json()["detail"]


def test_content_search_is_labelled_in_search_metrics(ctx: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """Content searches are distinguishable from title searches when watching latency."""
    registry = _FakeMetricsRegistry()
    monkeypatch.setattr(chat_router, "get_metrics_registry", lambda: registry, raising=False)
    _add_chat(ctx.db, "weekly", "Weekly sync")
    _say(ctx.db, "weekly", "zebrafinch")

    ctx.client.get("/api/v1/chats/conversations", params={"query": "zebrafinch", "search_in": "title,content"})
    ctx.client.get("/api/v1/chats/conversations", params={"query": "zebrafinch"})
    ctx.client.get(
        "/api/v1/chats/conversations",
        params={"query": "zebrafinch", "search_in": "title,content", "deleted_only": "true"},
    )

    assert [labels["query_strategy"] for _name, labels in registry.increment_calls] == [
        "fts_content",
        "fts",
        "deleted_text",
    ]
