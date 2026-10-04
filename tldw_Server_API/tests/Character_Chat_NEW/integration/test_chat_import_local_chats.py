"""POST /api/v1/chats/import: lossless import of an "On this device" chat (D7 P8).

Contract:

* The body names the conversation (a UUID) and carries its title, dates and the
  whole message graph: ids, parents, roles, text, timestamps, images and
  allow-listed generation metadata.
* The chat is written in one transaction, to the authenticated account. The
  body has no owner field, and any unknown field is refused.
* Ids, parents, order and timestamps are kept; nothing is re-stamped with the
  time of the import. The result is an ordinary server chat: its messages read
  back through the message API and the history-selection capture, and it can be
  continued.
* Repeating the same import returns the existing chat with 200 and
  ``Idempotency-Replayed: true``. Any other use of a taken id is 409
  ``chat_id_conflict``; a repeat after the chat was trashed is 410
  ``chat_deleted``.
* A refused import writes nothing.
"""

from __future__ import annotations

import base64
import io
import json
import threading
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as deps
from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions as chats
from tldw_Server_API.app.api.v1.endpoints import character_messages as messages_api
from tldw_Server_API.app.api.v1.endpoints import chat as chat_api
from tldw_Server_API.app.core.Character_Chat.character_rate_limiter import CharacterRateLimiter
from tldw_Server_API.app.core.Chat.history_selection import resolve_history_selection
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError

pytestmark = pytest.mark.integration

PATH = "/api/v1/chats/import"
CHATS = "/api/v1/chats/"
CHAT_ID = "6b0f8c1e-2d4a-4f3b-9a71-5c2e8d9f0a14"
SECRET_TITLE = "Owner one private plans"
SECRET_TEXT = "owner one private message"
TABLES = ("conversations", "messages", "message_metadata", "message_images")


def _png(color: tuple[int, int, int] = (200, 30, 30)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (3, 2), color).save(buffer, format="PNG")
    return buffer.getvalue()


RED_PNG = _png()
BLUE_PNG = _png((30, 30, 200))


def _data_url(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _ts(minute: int, second: int = 0) -> str:
    """A fixed past time: imports refuse timestamps ahead of the server clock."""
    return f"2025-03-01T10:{minute:02d}:{second:02d}.250Z"


def _message(mid: str, parent: str | None, role: str, minute: int, **extra: Any) -> dict[str, Any]:
    return {"id": mid, "parent_message_id": parent, "role": role, "content": f"text of {mid}", "timestamp": _ts(minute), **extra}


USAGE = {"prompt_tokens": 12, "completion_tokens": 30, "total_tokens": 42}
COMPLETE = {"model_id": "gpt-4o-mini", "provider": "openai", "finish_reason": "stop", "generation_status": "complete", "usage": USAGE}
STOPPED = {"model_id": "claude-sonnet", "provider": "anthropic", "generation_status": "stopped"}


def _branching_messages() -> list[dict[str, Any]]:
    """Two roots (an edited first message), a regenerated reply and a continued branch.

    pa_q1 ── pa_a1            (first answer, complete)
          └─ pa_a2 ── pa_q2 ── pa_a3     (regenerated answer, continued)
    pa_q1b ── pa_a4           (edited first message, stopped reply)
    """
    return [
        _message("pa_q1", None, "user", 0, images=[_data_url(RED_PNG), _data_url(BLUE_PNG)]),
        _message("pa_a1", "pa_q1", "assistant", 1, metadata=COMPLETE),
        _message("pa_a2", "pa_q1", "assistant", 2, metadata={"model_id": "gpt-4o", "generation_status": "length"}),
        _message("pa_q2", "pa_a2", "user", 3),
        _message("pa_a3", "pa_q2", "assistant", 4, metadata=COMPLETE),
        _message("pa_q1b", None, "user", 5),
        _message("pa_a4", "pa_q1b", "assistant", 6, metadata=STOPPED),
    ]


def _body(messages: list[dict[str, Any]] | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "id": CHAT_ID,
        "title": "Trip planning",
        "state": "resolved",
        "created_at": "2025-03-01T09:59:30.000Z",
        "last_modified": "2025-03-02T08:30:00.000Z",
        "messages": _branching_messages() if messages is None else messages,
        **extra,
    }


def _simple(**extra: Any) -> dict[str, Any]:
    return _body([_message("pa_q1", None, "user", 0), _message("pa_a1", "pa_q1", "assistant", 1)], **extra)


def _counts(db: CharactersRAGDB) -> dict[str, int]:
    return {
        table: int(db.execute_query(f"SELECT COUNT(*) AS n FROM {table}", read_only=True).fetchone()["n"])  # nosec B608
        for table in TABLES
    }


def _limiter(**limits: Any) -> CharacterRateLimiter:
    return CharacterRateLimiter(enabled=False, **{"max_chats_per_user": 100, "max_messages_per_chat": 1000, **limits})


def _client(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
    *,
    user_id: int = 1,
    sync_service: Any = None,
    limiter: CharacterRateLimiter | None = None,
) -> TestClient:
    app = FastAPI()
    app.include_router(chats.router, prefix="/api/v1/chats")
    app.include_router(messages_api.router, prefix="/api/v1")
    app.include_router(chat_api.router, prefix="/api/v1/chat")
    app.dependency_overrides[deps.get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[chats.get_request_user] = lambda: SimpleNamespace(id=user_id)
    app.dependency_overrides[chats.require_expected_user] = lambda: None
    active = limiter or _limiter()
    monkeypatch.setattr(chats, "get_character_rate_limiter", lambda: active)
    monkeypatch.setattr(messages_api, "get_character_rate_limiter", lambda: active)
    monkeypatch.setattr(chats, "_active_chat_sync_service", lambda *_args: sync_service)
    monkeypatch.setattr(messages_api, "_active_message_sync_service", lambda *_args: None)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def db(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[CharactersRAGDB]:
    """One owner's store. Both backends ship, so the whole contract runs on both."""
    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    database = CharactersRAGDB(tmp_path / "1" / "ChaChaNotes.db", client_id="1", backend=backend)
    try:
        yield database
    finally:
        database.close_all_connections()
        if backend is not None:
            backend.get_pool().close_all()


@pytest.fixture
def client(db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    return _client(db, monkeypatch)


def _error(response: Any) -> str:
    detail = response.json()["detail"]
    assert isinstance(detail, dict), response.text
    return detail["error_code"]


def _capture(client: TestClient, cursor: dict[str, str]) -> dict[str, Any]:
    response = client.post(
        f"/api/v1/chat/conversations/{CHAT_ID}/history/selection",
        json={
            "purpose": "send",
            "view": {
                "view_session_id": "view-one",
                "conversation_id": CHAT_ID,
                "interpretation": {"kind": "parent_graph_v1"},
                "cursor": cursor,
                "selection_revision": 1,
            },
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _stored_messages(client: TestClient) -> list[dict[str, Any]]:
    response = client.get(
        f"{CHATS}{CHAT_ID}/messages",
        params={"limit": 200, "include_metadata": True, "include_images": True, "render_placeholders": False},
    )
    assert response.status_code == 200, response.text
    assert response.json()["total"] == len(response.json()["messages"])
    return response.json()["messages"]


def _instant(value: Any) -> datetime:
    """A stored or returned timestamp as a point in time: SQLite keeps text, PostgreSQL returns datetimes."""
    moment = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_multi_branch_chat_round_trips_through_messages_and_history_capture(
    client: TestClient, db: CharactersRAGDB
) -> None:
    sent = _body()
    imported = client.post(PATH, json=sent)
    assert imported.status_code == 201, imported.text
    assert "Idempotency-Replayed" not in imported.headers
    chat = imported.json()
    assert (chat["id"], chat["title"], chat["state"], chat["scope_type"]) == (CHAT_ID, "Trip planning", "resolved", "global")
    assert (chat["message_count"], chat["version"], chat["root_id"]) == (7, 1, CHAT_ID)
    assert _instant(chat["created_at"]) == _instant(sent["created_at"])
    assert _instant(chat["last_modified"]) == _instant(sent["last_modified"])
    assert chat["tail"] == {"message_id": "pa_a4", "message_version": 1}
    assert db.get_conversation_by_id(CHAT_ID)["client_id"] == "1"

    # The same chat through GET /chats/{id}.
    fetched = client.get(f"{CHATS}{CHAT_ID}")
    assert fetched.status_code == 200, fetched.text
    assert _instant(fetched.json()["created_at"]) == _instant(sent["created_at"])
    assert fetched.json()["message_count"] == 7

    # Messages: same ids, parents, roles, text, order, timestamps, images and metadata.
    stored = _stored_messages(client)
    expected = sent["messages"]
    assert [row["id"] for row in stored] == [message["id"] for message in expected]
    for row, message in zip(stored, expected, strict=True):
        assert row["conversation_id"] == CHAT_ID
        assert row["parent_message_id"] == message["parent_message_id"]
        assert row["sender"] == message["role"]
        assert row["content"] == message["content"]
        assert _instant(row["timestamp"]) == _instant(message["timestamp"])
        assert row["version"] == 1
        assert row["images"] == message.get("images", [])
        assert row["has_image"] is bool(message.get("images"))
        assert row["metadata_extra"] == {"sender_role": message["role"], **message.get("metadata", {})}

    # History capture: one parent graph, every node settled, same parents.
    whole = _capture(client, {"kind": "empty"})
    assert whole["status"] == "captured"
    snapshot = whole["snapshot"]
    assert snapshot["interpretation_status"] == {"kind": "parent_graph_v1"}
    assert [(node["id"], node["parent_id"], node["role"], node["settled"]) for node in snapshot["nodes"]] == [
        (message["id"], message["parent_message_id"], message["role"], True) for message in expected
    ]

    # Each leaf resolves to its own branch, with the stored text, images and metadata.
    by_id = {message["id"]: message for message in expected}
    for leaf, path in (
        ("pa_a1", ["pa_q1", "pa_a1"]),
        ("pa_a3", ["pa_q1", "pa_a2", "pa_q2", "pa_a3"]),
        ("pa_a4", ["pa_q1b", "pa_a4"]),
    ):
        branch = _capture(client, {"kind": "after_message", "message_id": leaf})
        assert branch["status"] == "captured", branch
        assert [row["id"] for row in branch["rows"]] == path
        content = branch["selected_content"]
        assert [item["id"] for item in content] == path
        for item in content:
            source = by_id[item["id"]]
            assert item["message"] == source["content"]
            assert item["images"] == source.get("images", [])
            assert item["extra_metadata"] == {"sender_role": source["role"], **source.get("metadata", {})}


def test_imported_chat_can_be_continued_through_the_versioned_send_path(client: TestClient, db: CharactersRAGDB) -> None:
    assert client.post(PATH, json=_body()).status_code == 201
    captured = _capture(client, {"kind": "after_message", "message_id": "pa_a3"})
    selection = resolve_history_selection(captured["snapshot"], captured["view"], "send", "client-only")["selection"]
    sent = client.post(
        f"{CHATS}{CHAT_ID}/messages",
        json={"id": "pa_q3", "role": "user", "content": "and then?", "tldw_history_selection_v1": selection},
    )
    assert sent.status_code == 201, sent.text
    assert db.get_message_by_id("pa_q3")["parent_message_id"] == "pa_a3"
    assert _capture(client, {"kind": "empty"})["snapshot"]["interpretation_status"] == {"kind": "parent_graph_v1"}


def test_legacy_linear_chat_with_explicit_parents_needs_no_legacy_review(client: TestClient) -> None:
    linear = [
        _message(f"pa_{index}", f"pa_{index - 1}" if index else None, "user" if index % 2 == 0 else "assistant", index)
        for index in range(6)
    ]
    assert client.post(PATH, json=_body(linear)).status_code == 201
    captured = _capture(client, {"kind": "after_message", "message_id": "pa_5"})
    assert captured["status"] == "captured"
    assert [row["id"] for row in captured["rows"]] == [f"pa_{index}" for index in range(6)]


def test_timestamps_are_not_restamped_with_the_time_of_the_import(client: TestClient, db: CharactersRAGDB) -> None:
    started = datetime.now(timezone.utc)
    body = _simple(last_modified=None)
    assert client.post(PATH, json=body).status_code == 201
    row = db.get_conversation_by_id(CHAT_ID)
    assert _instant(row["created_at"]) == _instant("2025-03-01T09:59:30.000Z")
    # No last_modified sent: the newest message, not the import time.
    assert _instant(row["last_modified"]) == _instant("2025-03-01T10:01:00.250Z")
    assert [_instant(message["timestamp"]) for message in db.get_messages_for_conversation(CHAT_ID)] == [
        _instant("2025-03-01T10:00:00.250Z"), _instant("2025-03-01T10:01:00.250Z"),
    ]
    assert all(_instant(value) < started for value in (row["created_at"], row["last_modified"]))
    listed = client.get(CHATS)
    assert listed.status_code == 200, listed.text
    assert _instant(listed.json()["chats"][0]["last_modified"]) == _instant("2025-03-01T10:01:00.250Z")


def test_offset_timestamps_are_stored_as_the_same_instant(client: TestClient, db: CharactersRAGDB) -> None:
    messages = [{**_message("pa_q1", None, "user", 0), "timestamp": "2025-03-01T12:00:00.250+02:00"}]
    assert client.post(PATH, json=_body(messages, created_at="2025-03-01T05:59:30-04:00")).status_code == 201
    assert _instant(db.get_message_by_id("pa_q1")["timestamp"]) == _instant("2025-03-01T10:00:00.250Z")
    assert _instant(db.get_conversation_by_id(CHAT_ID)["created_at"]) == _instant("2025-03-01T09:59:30.000Z")


def test_imported_text_and_title_are_searchable(client: TestClient) -> None:
    messages = [{**_message("pa_q1", None, "user", 0), "content": "the heliotrope itinerary"}]
    assert client.post(PATH, json=_body(messages, title="Heliotrope trip")).status_code == 201
    found = client.get(f"{CHATS}{CHAT_ID}/messages/search", params={"query": "heliotrope"})
    assert found.status_code == 200, found.text
    assert [row["id"] for row in found.json()["messages"]] == ["pa_q1"]


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_replay_returns_the_same_chat_and_writes_nothing(client: TestClient, db: CharactersRAGDB) -> None:
    first = client.post(PATH, json=_body())
    after_first = _counts(db)
    second = client.post(PATH, json=_body())
    assert first.status_code == 201, first.text
    assert second.status_code == 200, second.text
    assert second.headers["Idempotency-Replayed"] == "true"
    assert second.json() == first.json()
    assert _counts(db) == after_first == {"conversations": 1, "messages": 7, "message_metadata": 7, "message_images": 2}
    # The stored fingerprint is internal: no response carries it.
    fingerprint = db.chat_imports.get_import_state(CHAT_ID)[1]
    for payload in (first.text, second.text, client.get(CHATS).text, client.get(f"{CHATS}{CHAT_ID}").text):
        assert fingerprint not in payload
    assert fingerprint not in client.get(f"{CHATS}{CHAT_ID}/messages", params={"include_metadata": True}).text


@pytest.mark.parametrize(
    "equivalent",
    [
        lambda body: {key: body[key] for key in reversed(list(body))},
        lambda body: {**body, "id": body["id"].upper()},
        lambda body: {**body, "state": "RESOLVED"},
        lambda body: {**body, "created_at": "2025-03-01T11:59:30+02:00"},
        lambda body: {**body, "character_id": None, "parent_conversation_id": None},
        lambda body: {**body, "messages": body["messages"][::-1]},
        lambda body: {**body, "messages": [{**message, "images": message.get("images", []), "metadata": message.get("metadata")} for message in body["messages"]]},
    ],
    ids=[
        "key-order", "uppercase-id", "normalized-state", "same-instant", "explicit-nulls", "messages-listed-backwards",
        "explicit-defaults",
    ],
)
def test_replay_matches_equivalent_request_bodies(client: TestClient, db: CharactersRAGDB, equivalent: Any) -> None:
    assert client.post(PATH, json=_body()).status_code == 201
    replay = client.post(PATH, json=equivalent(_body()))
    assert replay.status_code == 200, replay.text
    assert _counts(db)["conversations"] == 1


def test_replay_compares_the_original_request_not_the_current_chat(client: TestClient, db: CharactersRAGDB) -> None:
    """Renaming, editing or continuing the chat does not turn a late retry into a conflict."""
    created = client.post(PATH, json=_body())
    renamed = client.put(f"{CHATS}{CHAT_ID}", params={"expected_version": created.json()["version"]}, json={"title": "Renamed"})
    assert renamed.status_code == 200, renamed.text
    db.update_message("pa_a1", {"content": "edited on the server"}, expected_version=1)
    appended = client.post(f"{CHATS}{CHAT_ID}/messages", json={"role": "user", "content": "one more", "parent_message_id": "pa_a3"})
    assert appended.status_code == 201, appended.text
    replay = client.post(PATH, json=_body())
    assert replay.status_code == 200, replay.text
    assert replay.headers["Idempotency-Replayed"] == "true"
    assert (replay.json()["title"], replay.json()["message_count"]) == ("Renamed", 8)
    assert db.get_message_by_id("pa_a1")["content"] == "edited on the server"
    assert _counts(db)["messages"] == 8


@pytest.mark.parametrize(
    "change",
    [
        lambda body: {**body, "title": "Different title"},
        lambda body: {**body, "state": "backlog"},
        lambda body: {**body, "created_at": "2025-03-01T09:59:31.000Z"},
        lambda body: {**body, "messages": body["messages"][:-1]},
        lambda body: {**body, "messages": [{**body["messages"][0], "content": "edited"}, *body["messages"][1:]]},
        lambda body: {**body, "messages": [*body["messages"], _message("pa_new", "pa_a4", "user", 7)]},
        lambda body: {**body, "messages": [body["messages"][0], {**body["messages"][1], "metadata": {**COMPLETE, "model_id": "other"}}, *body["messages"][2:]]},
    ],
    ids=["title", "state", "created_at", "fewer-messages", "edited-message", "extra-message", "metadata"],
)
def test_same_id_with_a_different_body_is_409_and_changes_nothing(client: TestClient, db: CharactersRAGDB, change: Any) -> None:
    assert client.post(PATH, json=_body()).status_code == 201
    before = _counts(db)
    conflict = client.post(PATH, json=change(_body()))
    assert conflict.status_code == 409, conflict.text
    assert _error(conflict) == "chat_id_conflict"
    assert "Idempotency-Replayed" not in conflict.headers
    assert _counts(db) == before
    assert db.get_conversation_by_id(CHAT_ID)["title"] == "Trip planning"
    assert db.get_message_by_id("pa_q1")["content"] == "text of pa_q1"
    assert db.get_message_by_id("pa_new") is None


def test_id_of_a_chat_that_was_not_imported_is_409(client: TestClient, db: CharactersRAGDB) -> None:
    """Chats created another way were never this import, so they are never replayed or merged into."""
    db.add_conversation({"id": CHAT_ID, "title": "Server chat", "client_id": "1"})
    db.add_message({"id": "existing", "conversation_id": CHAT_ID, "sender": "user", "content": "already here"})
    before = _counts(db)
    conflict = client.post(PATH, json=_simple())
    assert conflict.status_code == 409, conflict.text
    assert _error(conflict) == "chat_id_conflict"
    assert _counts(db) == before
    assert db.get_message_by_id("pa_q1") is None


def _hide_existing_chat_from_first_lookup(db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a request whose pre-insert read ran before a concurrent import committed."""
    real = db.chat_imports.get_import_state
    calls = {"n": 0}

    def stale(conversation_id: str) -> tuple[dict[str, Any] | None, str | None]:
        calls["n"] += 1
        return (None, None) if calls["n"] == 1 else real(conversation_id)

    monkeypatch.setattr(db.chat_imports, "get_import_state", stale)


def test_lost_race_with_the_same_import_replays(client: TestClient, db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch) -> None:
    assert client.post(PATH, json=_body()).status_code == 201
    before = _counts(db)
    _hide_existing_chat_from_first_lookup(db, monkeypatch)
    raced = client.post(PATH, json=_body())
    assert raced.status_code == 200, raced.text
    assert raced.headers["Idempotency-Replayed"] == "true"
    assert _counts(db) == before


def test_lost_race_with_a_different_import_is_409(client: TestClient, db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch) -> None:
    assert client.post(PATH, json=_body()).status_code == 201
    before = _counts(db)
    _hide_existing_chat_from_first_lookup(db, monkeypatch)
    raced = client.post(PATH, json=_body(title="Different title"))
    assert raced.status_code == 409, raced.text
    assert _error(raced) == "chat_id_conflict"
    assert _counts(db) == before


def test_concurrent_identical_imports_produce_one_chat(client: TestClient, db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every request passes the pre-insert read before any of them writes."""
    workers = 5
    barrier = threading.Barrier(workers, timeout=30)
    real = db.chat_imports.get_import_state
    lock = threading.Lock()
    lookups = {"n": 0}

    def gated(conversation_id: str) -> tuple[dict[str, Any] | None, str | None]:
        state = real(conversation_id)
        with lock:
            lookups["n"] += 1
            first_lookup_of_a_request = lookups["n"] <= workers
        if first_lookup_of_a_request:
            barrier.wait()
        return state

    monkeypatch.setattr(db.chat_imports, "get_import_state", gated)
    results: list[Any] = [None] * workers

    def worker(index: int) -> None:
        results[index] = client.post(PATH, json=_body())

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    statuses = sorted(response.status_code for response in results)
    assert statuses == [200] * (workers - 1) + [201], [response.text for response in results]
    assert _counts(db) == {"conversations": 1, "messages": 7, "message_metadata": 7, "message_images": 2}


# ---------------------------------------------------------------------------
# Invalid graphs and metadata: 422, nothing written
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "messages,code",
    [
        ([_message("pa_q1", None, "user", 0), _message("pa_a1", "pa_missing", "assistant", 1)], "missing_parent"),
        ([_message("pa_q1", "pa_q1", "user", 0)], "cyclic_ancestry"),
        (
            [_message("pa_1", "pa_3", "user", 0), _message("pa_2", "pa_1", "assistant", 1), _message("pa_3", "pa_2", "user", 2)],
            "cyclic_ancestry",
        ),
        ([_message("pa_q1", None, "user", 0), _message("pa_q1", None, "user", 1)], "duplicate_message_id"),
    ],
    ids=["missing-parent", "self-parent", "cycle", "duplicate-id"],
)
def test_invalid_graph_is_422_and_writes_nothing(client: TestClient, db: CharactersRAGDB, messages: list, code: str) -> None:
    response = client.post(PATH, json=_body(messages))
    assert response.status_code == 422, response.text
    assert _error(response) == code
    assert _counts(db) == dict.fromkeys(TABLES, 0)


def test_parent_in_another_conversation_is_not_a_valid_parent(client: TestClient, db: CharactersRAGDB) -> None:
    """Parents must be part of the import; an existing server message cannot be adopted as one."""
    other = db.add_conversation({"title": "Other chat", "client_id": "1"})
    db.add_message({"id": "server_msg", "conversation_id": other, "sender": "user", "content": "elsewhere"})
    before = _counts(db)
    response = client.post(PATH, json=_body([_message("pa_q1", "server_msg", "user", 0)]))
    assert response.status_code == 422, response.text
    assert _error(response) == "missing_parent"
    assert _counts(db) == before


@pytest.mark.parametrize(
    "metadata",
    [
        {"sources": [{"url": "https://example.com"}]},
        {**COMPLETE, "reasoning_time_taken": 12},
        {"generation_status": "finished"},
        {"usage": {"prompt_tokens": -1}},
        {"sender_role": "system"},
        {"model_id": " gpt-4o "},
        {"cut \ud83d key": 1},
        {"k" * 5_000: 1},
    ],
    ids=["sources", "extra-key", "bad-status", "bad-usage", "reserved-key", "padded-value", "surrogate-key", "long-key"],
)
def test_metadata_outside_the_allow_list_is_422_and_writes_nothing(client: TestClient, db: CharactersRAGDB, metadata: dict) -> None:
    messages = [_message("pa_q1", None, "user", 0), _message("pa_a1", "pa_q1", "assistant", 1, metadata=metadata)]
    # ASCII-escaped JSON, as a browser sends it.
    response = client.post(PATH, content=json.dumps(_body(messages)), headers={"Content-Type": "application/json"})
    assert response.status_code == 422, response.text
    assert _error(response) == "unsupported_metadata"
    assert response.json()["detail"]["message_id"] == "pa_a1"
    assert len(response.content) < 600
    assert _counts(db) == dict.fromkeys(TABLES, 0)


def test_generation_metadata_on_a_user_message_is_422(client: TestClient, db: CharactersRAGDB) -> None:
    response = client.post(PATH, json=_body([_message("pa_q1", None, "user", 0, metadata={"model_id": "gpt-4o"})]))
    assert response.status_code == 422, response.text
    assert _error(response) == "unsupported_metadata"
    assert _counts(db) == dict.fromkeys(TABLES, 0)


@pytest.mark.parametrize(
    "change",
    [
        lambda body: {**body, "id": "not-a-uuid"},
        lambda body: {**body, "id": "00000000-0000-0000-0000-000000000000"},
        lambda body: {key: value for key, value in body.items() if key != "id"},
        lambda body: {key: value for key, value in body.items() if key != "title"},
        lambda body: {key: value for key, value in body.items() if key != "created_at"},
        lambda body: {**body, "title": "   "},
        lambda body: {**body, "state": "archived"},
        lambda body: {**body, "created_at": "2025-03-01T09:59:30"},
        lambda body: {**body, "created_at": 1788256770000},
        lambda body: {**body, "created_at": "1740823170"},
        lambda body: {**body, "messages": [{**body["messages"][0], "timestamp": "1740823170000"}]},
        lambda body: {**body, "created_at": "9999-12-31T23:59:59-10:00"},
        lambda body: {**body, "messages": [{**body["messages"][0], "timestamp": "0001-01-01T00:00:00+10:00"}]},
        lambda body: {**body, "character_id": 99999999999999999999},
        lambda body: {**body, "assistant_kind": "character", "assistant_id": "99999999999999999999"},
        lambda body: {**body, "messages": []},
        lambda body: {**body, "messages": [{**body["messages"][0], "role": "tool"}]},
        lambda body: {**body, "messages": [{**body["messages"][0], "id": "has space"}]},
        lambda body: {**body, "messages": [{**body["messages"][0], "id": "../etc"}]},
        lambda body: {**body, "messages": [{**body["messages"][0], "id": "pa_q1\n"}]},
        lambda body: {**body, "messages": [{**body["messages"][0], "id": "x" * 256}]},
        lambda body: {**body, "messages": [{**body["messages"][0], "timestamp": "yesterday"}]},
        lambda body: {**body, "messages": [{key: value for key, value in body["messages"][0].items() if key != "timestamp"}]},
        lambda body: {**body, "messages": [{**body["messages"][0], "content": "   ", "images": []}]},
        lambda body: {**body, "messages": [{**body["messages"][0], "content": "a\u0000b"}]},
        lambda body: {**body, "messages": [{**body["messages"][0], "content": "cut \ud83d emoji"}]},
        lambda body: {**body, "forked_from_message_id": "pa_q1"},
        lambda body: {**body, "assistant_kind": "persona"},
        lambda body: {**body, "persona_memory_mode": "read_write", "character_id": 1},
        lambda body: {**body, "assistant_kind": "persona", "assistant_id": "p1", "character_id": 1},
        lambda body: {**body, "parent_conversation_id": "a\u0000b"},
        lambda body: {**body, "assistant_kind": "persona", "assistant_id": "cut \ud83d id"},
        lambda body: {**body, "messages": [{**body["messages"][0], "id": "cut\ud83did"}]},
        lambda body: {**body, "state": "cut \ud83d state"},
        lambda body: {**body, "cut \ud83d key": 1},
    ],
    ids=[
        "bad-uuid", "nil-uuid", "no-id", "no-title", "no-created-at", "blank-title", "bad-state", "naive-date",
        "epoch-number", "epoch-seconds-text", "epoch-millis-text", "year-9999", "year-1", "huge-character-id",
        "huge-assistant-id", "no-messages", "tool-role", "id-with-space", "id-with-slash", "id-with-newline", "id-too-long", "bad-timestamp",
        "no-timestamp", "empty-message", "nul-in-text", "lone-surrogate", "fork-without-parent", "persona-without-id",
        "memory-mode-on-character", "persona-with-character", "control-char-in-parent-id", "surrogate-in-assistant-id",
        "surrogate-in-message-id", "surrogate-in-state", "surrogate-in-unknown-key",
    ],
)
def test_malformed_request_is_422_and_writes_nothing(client: TestClient, db: CharactersRAGDB, change: Any) -> None:
    # ASCII-escaped JSON, as a browser sends it: half a surrogate pair arrives as a \uXXXX escape.
    raw = json.dumps(change(_simple()))
    response = client.post(PATH, content=raw, headers={"Content-Type": "application/json"})
    assert response.status_code == 422, response.text
    assert _counts(db) == dict.fromkeys(TABLES, 0)


def test_schema_errors_name_the_field_without_echoing_the_chat(client: TestClient, db: CharactersRAGDB) -> None:
    """A whole chat is sent, so a validation error must not send it back."""
    body = _body([{**_message("pa_q1", None, "user", 0), "content": SECRET_TEXT * 400, "images": [_data_url(RED_PNG)]}])
    del body["title"]
    body["messages"].append({**_message("pa_a1", "pa_q1", "assistant", 1), "role": "narrator", "content": SECRET_TEXT})
    response = client.post(PATH, json=body)
    assert response.status_code == 422, response.text
    errors = response.json()["detail"]
    assert {(tuple(error["loc"]), error["type"]) for error in errors} == {
        (("body", "title"), "missing"),
        (("body", "messages", 1, "role"), "literal_error"),
    }
    assert all(set(error) == {"loc", "msg", "type"} for error in errors)
    assert SECRET_TEXT not in response.text and "base64" not in response.text
    assert len(response.content) < 1_000
    assert _counts(db) == dict.fromkeys(TABLES, 0)


def test_schema_error_list_is_bounded(client: TestClient, db: CharactersRAGDB) -> None:
    messages = [{"id": f"m{index}", "role": "narrator", "content": "x", "timestamp": _ts(0)} for index in range(500)]
    response = client.post(PATH, json=_body(messages))
    assert response.status_code == 422, response.text
    assert len(response.json()["detail"]) == 100
    assert _counts(db) == dict.fromkeys(TABLES, 0)


def test_timestamp_in_the_future_is_422(client: TestClient, db: CharactersRAGDB) -> None:
    future = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    response = client.post(PATH, json=_body([{**_message("pa_q1", None, "user", 0), "timestamp": future}]))
    assert response.status_code == 422, response.text
    assert _error(response) == "timestamp_in_future"
    assert _counts(db) == dict.fromkeys(TABLES, 0)


# ---------------------------------------------------------------------------
# Security: the owner is the caller, and nothing else in the body can say otherwise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("client_id", "2"), ("user_id", 2), ("owner_user_id", "2"), ("owner", "2"), ("deleted", True),
        ("version", 9), ("scope_type", "workspace"), ("workspace_id", "ws"), ("root_id", "x"), ("rating", 5),
        ("create_request_fingerprint", "a" * 64), ("history_version", 99),
    ],
)
def test_owner_and_other_server_owned_conversation_fields_are_refused(
    client: TestClient, db: CharactersRAGDB, field: str, value: Any
) -> None:
    response = client.post(PATH, json=_simple(**{field: value}))
    assert response.status_code == 422, response.text
    assert _counts(db) == dict.fromkeys(TABLES, 0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("client_id", "2"), ("conversation_id", "another-chat"), ("history_id", "pa_local"), ("deleted", True),
        ("version", 9), ("sender", "assistant"), ("ranking", 3), ("last_modified", "2025-03-01T10:00:00Z"),
        ("tool_calls", []), ("history_admission_json", "{}"), ("tldw_history_admission_v1", {}),
    ],
)
def test_owner_and_other_server_owned_message_fields_are_refused(
    client: TestClient, db: CharactersRAGDB, field: str, value: Any
) -> None:
    response = client.post(PATH, json=_body([{**_message("pa_q1", None, "user", 0), field: value}]))
    assert response.status_code == 422, response.text
    assert _counts(db) == dict.fromkeys(TABLES, 0)


def test_import_always_goes_to_the_authenticated_user(db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch) -> None:
    second = _client(db, monkeypatch, user_id=2)
    assert second.post(PATH, json=_simple()).status_code == 201
    assert db.get_conversation_by_id(CHAT_ID)["client_id"] == "2"
    assert {row["client_id"] for row in db.get_messages_for_conversation(CHAT_ID)} == {"2"}


def test_another_owner_reusing_the_id_in_a_shared_store_gets_409_without_a_leak(
    db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = _client(db, monkeypatch, user_id=1)
    body = _body([{**_message("pa_q1", None, "user", 0), "content": SECRET_TEXT}], title=SECRET_TITLE)
    assert owner.post(PATH, json=body).status_code == 201
    before = _counts(db)
    other = _client(db, monkeypatch, user_id=2)
    # The identical body (a guessed fingerprint) and a different one are refused the same way.
    different = _body([_message("pa_other", None, "user", 0)], title="Something else")
    for attempt in (body, different):
        response = other.post(PATH, json=attempt)
        assert response.status_code == 409, response.text
        assert _error(response) == "chat_id_conflict"
        assert set(response.json()["detail"]) == {"error_code", "message"}
        assert SECRET_TITLE not in response.text and SECRET_TEXT not in response.text
        assert "Idempotency-Replayed" not in response.headers
    assert _counts(db) == before
    assert db.get_conversation_by_id(CHAT_ID)["client_id"] == "1"
    assert db.get_message_by_id("pa_other") is None
    assert other.get(f"{CHATS}{CHAT_ID}").status_code in {403, 404}
    assert other.get(f"{CHATS}{CHAT_ID}/messages").status_code in {403, 404}


def test_message_id_taken_by_another_owner_is_409_without_a_leak_and_writes_nothing(
    db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = _client(db, monkeypatch, user_id=1)
    first = _body([{**_message("pa_shared", None, "user", 0), "content": SECRET_TEXT}], title=SECRET_TITLE)
    assert owner.post(PATH, json=first).status_code == 201
    before = _counts(db)
    other = _client(db, monkeypatch, user_id=2)
    other_chat = "9d2f6a70-1c3b-4e5d-8f90-a1b2c3d4e5f6"
    attempt = _body(
        [_message("pa_mine", None, "user", 0), _message("pa_shared", "pa_mine", "assistant", 1)], id=other_chat
    )
    response = other.post(PATH, json=attempt)
    assert response.status_code == 409, response.text
    assert _error(response) == "message_id_conflict"
    assert SECRET_TITLE not in response.text and SECRET_TEXT not in response.text and CHAT_ID not in response.text
    assert _counts(db) == before
    assert db.get_conversation_by_id(other_chat, include_deleted=True) is None
    assert db.get_message_by_id("pa_mine") is None
    stored = db.get_message_by_id("pa_shared")
    assert (stored["conversation_id"], stored["content"], stored["client_id"]) == (CHAT_ID, SECRET_TEXT, "1")


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def owners(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[SimpleNamespace]:
    """Two real owners: PostgreSQL shares one database; SQLite uses per-user files.

    The PostgreSQL arm runs as a role that cannot bypass row-level security. The
    plain test role is a superuser and would see the other owner's rows, which a
    deployed server's role does not.
    """
    backend = request.getfixturevalue("pg_restricted_backend") if request.param == "postgres" else None
    first = CharactersRAGDB(tmp_path / "1" / "ChaChaNotes.db", client_id="1", backend=backend)
    second = CharactersRAGDB(tmp_path / "2" / "ChaChaNotes.db", client_id="2", backend=backend)
    try:
        yield SimpleNamespace(first=first, second=second, postgres=request.param == "postgres")
    finally:
        first.close_all_connections()
        second.close_all_connections()


def test_another_owner_reusing_ids_never_touches_the_first_owners_chat(
    owners: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PostgreSQL ids share one key space (409); per-user SQLite files do not collide."""
    first = _client(owners.first, monkeypatch, user_id=1)
    body = _body([{**_message("pa_q1", None, "user", 0), "content": SECRET_TEXT}], title=SECRET_TITLE)
    assert first.post(PATH, json=body).status_code == 201
    second = _client(owners.second, monkeypatch, user_id=2)
    response = second.post(PATH, json=body)
    if owners.postgres:
        # The second owner cannot see the chat at all: the primary key alone refuses the id.
        assert owners.second.get_conversation_by_id(CHAT_ID, include_deleted=True) is None
        assert owners.second.chat_imports.get_import_state(CHAT_ID) == (None, None)
        assert response.status_code == 409, response.text
        assert _error(response) == "chat_id_conflict"
        assert set(response.json()["detail"]) == {"error_code", "message"}
        assert SECRET_TITLE not in response.text and SECRET_TEXT not in response.text
        # A fresh conversation id still cannot take the first owner's message id.
        other_chat = "9d2f6a70-1c3b-4e5d-8f90-a1b2c3d4e5f6"
        retry = second.post(PATH, json={**body, "id": other_chat})
        assert retry.status_code == 409, retry.text
        assert _error(retry) == "message_id_conflict"
        assert SECRET_TEXT not in retry.text and CHAT_ID not in retry.text
        # Nothing of the refused imports was kept, and the second owner still has no chats.
        assert owners.second.get_conversation_by_id(other_chat, include_deleted=True) is None
        assert owners.second.count_conversations_for_user("2", scope_type="global", workspace_id=None) == 0
        assert second.get(f"{CHATS}{CHAT_ID}").status_code in {403, 404}
        # With ids of their own the second owner imports normally.
        own = _body([_message("pa_second", None, "user", 0)], id=other_chat)
        assert second.post(PATH, json=own).status_code == 201
        assert owners.first.get_conversation_by_id(other_chat, include_deleted=True) is None
    else:
        # Each SQLite owner has a separate store, so the ids name the caller's own new chat.
        assert response.status_code == 201, response.text
        assert owners.second.get_conversation_by_id(CHAT_ID)["client_id"] == "2"
    row = owners.first.get_conversation_by_id(CHAT_ID)
    assert (row["client_id"], row["title"]) == ("1", SECRET_TITLE)
    stored = owners.first.get_message_by_id("pa_q1")
    assert (stored["client_id"], stored["content"]) == ("1", SECRET_TEXT)
    assert first.post(PATH, json=body).status_code == 200


def test_import_is_rate_limited_as_a_chat_creation(db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch) -> None:
    limiter = _limiter()
    seen: list[tuple[int, str]] = []

    async def record(user_id: int, operation: str = "character_op") -> tuple[bool, int]:
        seen.append((user_id, operation))
        return True, 0

    monkeypatch.setattr(limiter, "check_rate_limit", record)
    client = _client(db, monkeypatch, user_id=7, limiter=limiter)
    assert client.post(PATH, json=_simple()).status_code == 201
    assert client.post(PATH, json=_simple()).status_code == 200
    assert seen == [(7, "chat_create"), (7, "chat_create")]


def test_rate_limited_import_is_429_and_writes_nothing(db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import HTTPException

    limiter = _limiter()

    async def deny(*_args: Any, **_kwargs: Any) -> tuple[bool, int]:
        raise HTTPException(status_code=429, detail="Rate limit exceeded.", headers={"Retry-After": "60"})

    monkeypatch.setattr(limiter, "check_rate_limit", deny)
    response = _client(db, monkeypatch, limiter=limiter).post(PATH, json=_simple())
    assert response.status_code == 429, response.text
    assert response.headers["Retry-After"] == "60"
    assert _counts(db) == dict.fromkeys(TABLES, 0)


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------


def test_more_messages_than_a_chat_may_hold_is_refused(db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(db, monkeypatch, limiter=_limiter(max_messages_per_chat=6))
    response = client.post(PATH, json=_body())
    assert response.status_code == 403, response.text
    assert _error(response) == "message_limit_exceeded"
    assert response.json()["detail"]["limit"] == 6
    assert _counts(db) == dict.fromkeys(TABLES, 0)
    allowed = _client(db, monkeypatch, limiter=_limiter(max_messages_per_chat=7))
    assert allowed.post(PATH, json=_body()).status_code == 201


def test_message_limit_is_checked_before_any_message_is_decoded(
    db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An oversized import is refused on its message count alone, without decoding its images."""
    inspected: list[int] = []
    real = chats._chat_import_image_inspector

    def counting() -> Any:
        inspect = real()

        def wrapped(data: bytes) -> str:
            inspected.append(len(data))
            return inspect(data)

        return wrapped

    monkeypatch.setattr(chats, "_chat_import_image_inspector", counting)
    client = _client(db, monkeypatch, limiter=_limiter(max_messages_per_chat=6))
    response = client.post(PATH, json=_body())
    assert response.status_code == 403, response.text
    assert _error(response) == "message_limit_exceeded"
    assert inspected == []
    allowed = _client(db, monkeypatch, limiter=_limiter(max_messages_per_chat=7))
    assert allowed.post(PATH, json=_body()).status_code == 201
    assert len(inspected) == 2


def test_message_count_beyond_the_parse_ceiling_is_422(client: TestClient, db: CharactersRAGDB) -> None:
    from tldw_Server_API.app.api.v1.schemas.chat_import_schemas import CHAT_IMPORT_MAX_MESSAGES

    messages = [
        {"id": f"m{index}", "role": "user", "content": "x", "timestamp": _ts(0)}
        for index in range(CHAT_IMPORT_MAX_MESSAGES + 1)
    ]
    response = client.post(PATH, json=_body(messages))
    assert response.status_code == 422, response.text
    assert _counts(db) == dict.fromkeys(TABLES, 0)


def test_message_text_over_the_persisted_content_limit_is_413(
    client: TestClient, db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(settings, "MAX_PERSIST_CONTENT_LENGTH", 40)
    too_long = _body([{**_message("pa_q1", None, "user", 0), "content": "x" * 41}])
    response = client.post(PATH, json=too_long)
    assert response.status_code == 413, response.text
    assert _error(response) == "message_content_too_large"
    assert response.json()["detail"]["message_id"] == "pa_q1"
    assert _counts(db) == dict.fromkeys(TABLES, 0)
    # Placeholders are measured as the normal message path measures them.
    within = _body([{**_message("pa_q1", None, "user", 0), "content": "{{char}}" * 40}])
    assert client.post(PATH, json=within).status_code == 201


def test_image_over_the_message_image_limit_is_413(
    client: TestClient, db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(settings, "MAX_MESSAGE_IMAGE_BYTES", len(RED_PNG) - 1)
    response = client.post(PATH, json=_body([_message("pa_q1", None, "user", 0, images=[_data_url(RED_PNG)])]))
    assert response.status_code == 413, response.text
    assert _error(response) == "image_too_large"
    assert _counts(db) == dict.fromkeys(TABLES, 0)


def test_images_over_the_import_image_budget_are_413(
    client: TestClient, db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chats, "MAX_CHAT_IMPORT_IMAGE_BYTES", len(RED_PNG) + len(BLUE_PNG) - 1)
    response = client.post(PATH, json=_body())
    assert response.status_code == 413, response.text
    assert _error(response) == "images_too_large"
    assert _counts(db) == dict.fromkeys(TABLES, 0)


def test_images_over_the_import_pixel_budget_are_413(
    client: TestClient, db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Decoding work is bounded too: small files can still hold very large pictures."""
    # The chat has two 3x2 images: 12 pixels in total.
    monkeypatch.setattr(chats, "MAX_CHAT_IMPORT_IMAGE_PIXELS", 11)
    response = client.post(PATH, json=_body())
    assert response.status_code == 413, response.text
    assert _error(response) == "images_too_large"
    assert response.json()["detail"]["limit_pixels"] == 11
    assert _counts(db) == dict.fromkeys(TABLES, 0)
    monkeypatch.setattr(chats, "MAX_CHAT_IMPORT_IMAGE_PIXELS", 12)
    assert client.post(PATH, json=_body()).status_code == 201


def test_one_picture_over_the_pixel_limit_is_413(client: TestClient, db: CharactersRAGDB) -> None:
    """A picture larger than the message API will decode is refused like an oversized file."""
    buffer = io.BytesIO()
    Image.new("L", (4097, 4096)).save(buffer, format="PNG", optimize=True)
    huge = buffer.getvalue()
    assert len(huge) < 100_000
    response = client.post(PATH, json=_body([_message("pa_q1", None, "user", 0, images=[_data_url(huge)])]))
    assert response.status_code == 413, response.text
    assert _error(response) == "image_too_large"
    assert response.json()["detail"]["message_id"] == "pa_q1"
    assert _counts(db) == dict.fromkeys(TABLES, 0)


@pytest.mark.parametrize(
    "image",
    [
        "data:image/png;base64,",
        "not base64 !!",
        base64.b64encode(b"plain text pretending to be an image").decode("ascii"),
        _data_url(RED_PNG[:40]),
    ],
    ids=["empty", "not-base64", "not-an-image", "truncated-png"],
)
def test_image_that_could_not_be_read_back_is_422(client: TestClient, db: CharactersRAGDB, image: str) -> None:
    response = client.post(PATH, json=_body([_message("pa_q1", None, "user", 0, images=[image])]))
    assert response.status_code == 422, response.text
    assert _error(response) == "invalid_image"
    assert response.json()["detail"]["message_id"] == "pa_q1"
    assert _counts(db) == dict.fromkeys(TABLES, 0)


def test_too_many_images_on_one_message_is_422(client: TestClient, db: CharactersRAGDB) -> None:
    from tldw_Server_API.app.api.v1.schemas.chat_import_schemas import CHAT_IMPORT_MAX_IMAGES_PER_MESSAGE

    images = [_data_url(RED_PNG)] * (CHAT_IMPORT_MAX_IMAGES_PER_MESSAGE + 1)
    response = client.post(PATH, json=_body([_message("pa_q1", None, "user", 0, images=images)]))
    assert response.status_code == 422, response.text
    assert _counts(db) == dict.fromkeys(TABLES, 0)


def test_request_body_over_the_import_size_limit_is_413_before_parsing(
    client: TestClient, db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(settings, "CHAT_IMPORT_MAX_BODY_BYTES", 600)
    big = _body([{**_message("pa_q1", None, "user", 0), "content": "x" * 2_000}])
    response = client.post(PATH, json=big)
    assert response.status_code == 413, response.text
    assert _error(response) == "import_body_too_large"
    assert response.json()["detail"]["max_bytes"] == 600
    # A body that is not even JSON is refused on size alone, declared or streamed.
    assert client.post(PATH, content=b"{" * 601, headers={"Content-Type": "application/json"}).status_code == 413

    def chunks() -> Iterator[bytes]:
        yield from (b"{" * 200 for _ in range(4))

    streamed = client.post(PATH, content=chunks(), headers={"Content-Type": "application/json"})
    assert streamed.status_code == 413, streamed.text
    assert _counts(db) == dict.fromkeys(TABLES, 0)
    small = _body([_message("pa_q1", None, "user", 0)])
    assert client.post(PATH, json=small).status_code == 201


def test_body_size_limit_prefers_the_environment_and_ignores_unusable_values(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints.chat_import_transport import (
        DEFAULT_CHAT_IMPORT_MAX_BODY_BYTES,
        chat_import_max_body_bytes,
    )

    monkeypatch.delenv("CHAT_IMPORT_MAX_BODY_BYTES", raising=False)
    assert chat_import_max_body_bytes() == DEFAULT_CHAT_IMPORT_MAX_BODY_BYTES == 64 * 1024 * 1024
    monkeypatch.setitem(settings, "CHAT_IMPORT_MAX_BODY_BYTES", 2_000)
    assert chat_import_max_body_bytes() == 2_000
    monkeypatch.setenv("CHAT_IMPORT_MAX_BODY_BYTES", "1000")
    assert chat_import_max_body_bytes() == 1_000
    for unusable in ("0", "-5", "lots", ""):
        monkeypatch.setenv("CHAT_IMPORT_MAX_BODY_BYTES", unusable)
        assert chat_import_max_body_bytes() == 2_000


def test_chat_count_quota_applies_to_new_imports_but_not_to_replays(db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(db, monkeypatch, limiter=_limiter(max_chats_per_user=1))
    assert client.post(PATH, json=_simple()).status_code == 201
    before = _counts(db)
    other = "9d2f6a70-1c3b-4e5d-8f90-a1b2c3d4e5f6"
    refused = client.post(PATH, json=_body([_message("pa_x", None, "user", 0)], id=other))
    assert refused.status_code == 403, refused.text
    assert _error(refused) == "chat_limit_exceeded"
    assert _counts(db) == before
    assert db.get_message_by_id("pa_x") is None
    # A replay creates nothing, so it is answered even at the limit.
    assert client.post(PATH, json=_simple()).status_code == 200


# ---------------------------------------------------------------------------
# Atomicity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fail_at", [1, 4, 7])
def test_failure_part_way_through_rolls_back_everything(
    client: TestClient, db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, fail_at: int
) -> None:
    real = db.message_store._write_history_authority
    calls = {"n": 0}

    def fail_later(*args: Any, **kwargs: Any) -> None:
        calls["n"] += 1
        if calls["n"] == fail_at:
            raise CharactersRAGDBError("disk full")
        real(*args, **kwargs)

    monkeypatch.setattr(db.message_store, "_write_history_authority", fail_later)
    failed = client.post(PATH, json=_body())
    assert failed.status_code == 500, failed.text
    assert "disk full" not in failed.text
    assert _counts(db) == dict.fromkeys(TABLES, 0)
    assert db.get_conversation_by_id(CHAT_ID, include_deleted=True) is None
    assert db.search_messages_by_content("text of pa_q1") == []
    assert client.get(f"{CHATS}{CHAT_ID}").status_code == 404
    # Nothing was left behind, so the retry is a clean first import, not a replay.
    monkeypatch.undo()
    retry = _client(db, monkeypatch).post(PATH, json=_body())
    assert retry.status_code == 201, retry.text
    assert _counts(db)["messages"] == 7


def test_message_id_already_on_the_server_is_409_and_writes_nothing(client: TestClient, db: CharactersRAGDB) -> None:
    other = db.add_conversation({"title": "Existing", "client_id": "1"})
    db.add_message({"id": "pa_q2", "conversation_id": other, "sender": "user", "content": "already here"})
    before = _counts(db)
    response = client.post(PATH, json=_body())
    assert response.status_code == 409, response.text
    assert _error(response) == "message_id_conflict"
    assert _counts(db) == before
    assert db.get_conversation_by_id(CHAT_ID, include_deleted=True) is None
    assert db.get_message_by_id("pa_q1") is None
    assert db.get_message_by_id("pa_q2")["conversation_id"] == other


# ---------------------------------------------------------------------------
# Trash, Sync v2, bindings and forks
# ---------------------------------------------------------------------------


def _trash(client: TestClient, version: int) -> None:
    response = client.delete(f"{CHATS}{CHAT_ID}", params={"expected_version": version})
    assert response.status_code == 204, response.text


def test_replaying_a_trashed_import_is_410_and_does_not_restore_or_recreate(client: TestClient, db: CharactersRAGDB) -> None:
    created = client.post(PATH, json=_body())
    _trash(client, created.json()["version"])
    before = _counts(db)
    gone = client.post(PATH, json=_body())
    assert gone.status_code == 410, gone.text
    assert _error(gone) == "chat_deleted"
    assert db.get_conversation_by_id(CHAT_ID) is None
    assert db.get_conversation_by_id(CHAT_ID, include_deleted=True)["deleted"]
    assert _counts(db) == before
    different = client.post(PATH, json=_body(title="Different title"))
    assert different.status_code == 409, different.text
    assert _error(different) == "chat_id_conflict"


def test_restored_import_replays_again(client: TestClient, db: CharactersRAGDB) -> None:
    created = client.post(PATH, json=_body())
    _trash(client, created.json()["version"])
    trashed = db.get_conversation_by_id(CHAT_ID, include_deleted=True)
    restored = client.post(f"{CHATS}{CHAT_ID}/restore", params={"expected_version": trashed["version"]})
    assert restored.status_code == 200, restored.text
    replay = client.post(PATH, json=_body())
    assert replay.status_code == 200, replay.text
    assert replay.headers["Idempotency-Replayed"] == "true"
    assert _counts(db)["conversations"] == 1


def test_permanently_deleted_import_frees_its_ids(client: TestClient, db: CharactersRAGDB) -> None:
    """Purging the chat removes its messages too, so the same import is a first import again.

    The chat is trashed in the store, flagging only the conversation. The trash
    endpoint on this base also soft-deletes every message, after which purging
    any chat, imported or not, fails on the message search index (#3153, #3154).
    """
    created = client.post(PATH, json=_body())
    assert db.soft_delete_conversation(CHAT_ID, expected_version=created.json()["version"])
    trashed = db.get_conversation_by_id(CHAT_ID, include_deleted=True)
    assert client.post(PATH, json=_body()).status_code == 410
    purged = client.delete(f"{CHATS}{CHAT_ID}", params={"expected_version": trashed["version"], "hard_delete": True})
    assert purged.status_code == 204, purged.text
    assert _counts(db) == dict.fromkeys(TABLES, 0)
    assert db.search_messages_by_content("text of pa_q1") == []
    again = client.post(PATH, json=_body())
    assert again.status_code == 201, again.text
    assert "Idempotency-Replayed" not in again.headers
    assert _counts(db)["messages"] == 7


def test_import_is_refused_while_sync_v2_is_active(db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(db, monkeypatch, sync_service=object())
    response = client.post(PATH, json=_body())
    assert response.status_code == 409, response.text
    assert _error(response) == "sync_chat_import_unsupported"
    assert _counts(db) == dict.fromkeys(TABLES, 0)


def test_character_binding_is_kept_and_an_unknown_character_is_404(client: TestClient, db: CharactersRAGDB) -> None:
    missing = client.post(PATH, json=_simple(character_id=4242))
    assert missing.status_code == 404, missing.text
    assert _error(missing) == "character_not_found"
    assert _counts(db) == dict.fromkeys(TABLES, 0)
    character_id = db.add_character_card({"name": "Archivist", "first_message": "Welcome back."})
    bound = client.post(PATH, json=_simple(character_id=character_id))
    assert bound.status_code == 201, bound.text
    chat = bound.json()
    assert (chat["character_id"], chat["assistant_kind"], chat["assistant_id"]) == (character_id, "character", str(character_id))
    assert chat["character_name"] == "Archivist"
    # No greeting is seeded: the import is exactly the messages that were sent.
    assert chat["message_count"] == 2
    aliased = client.post(PATH, json=_simple(assistant_kind="character", assistant_id=str(character_id)))
    assert aliased.status_code == 200, aliased.text


def test_unknown_persona_is_404(client: TestClient, db: CharactersRAGDB) -> None:
    response = client.post(PATH, json=_simple(assistant_kind="persona", assistant_id="no-such-persona"))
    assert response.status_code == 404, response.text
    assert _error(response) == "persona_not_found"
    assert _counts(db) == dict.fromkeys(TABLES, 0)


def test_fork_keeps_its_lineage_when_the_parent_is_on_the_server(client: TestClient, db: CharactersRAGDB) -> None:
    assert client.post(PATH, json=_body()).status_code == 201
    child = "9d2f6a70-1c3b-4e5d-8f90-a1b2c3d4e5f6"
    fork = _body(
        [_message("pa_f1", None, "user", 0), _message("pa_f2", "pa_f1", "assistant", 1)],
        id=child, title="Fork", parent_conversation_id=CHAT_ID, forked_from_message_id="pa_a2",
    )
    response = client.post(PATH, json=fork)
    assert response.status_code == 201, response.text
    chat = response.json()
    assert (chat["parent_conversation_id"], chat["root_id"], chat["forked_from_message_id"]) == (CHAT_ID, CHAT_ID, "pa_a2")


def test_fork_of_a_missing_or_foreign_parent_is_404_and_writes_nothing(
    db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = _client(db, monkeypatch, user_id=1)
    assert owner.post(PATH, json=_body(title=SECRET_TITLE)).status_code == 201
    before = _counts(db)
    child = "9d2f6a70-1c3b-4e5d-8f90-a1b2c3d4e5f6"
    fork = _body([_message("pa_f1", None, "user", 0)], id=child, parent_conversation_id=CHAT_ID)
    other = _client(db, monkeypatch, user_id=2)
    foreign = other.post(PATH, json=fork)
    missing = other.post(PATH, json={**fork, "parent_conversation_id": "1b2c3d4e-0000-4000-8000-000000000001"})
    for response in (foreign, missing):
        assert response.status_code == 404, response.text
        assert _error(response) == "parent_conversation_not_found"
        assert SECRET_TITLE not in response.text
    # The two cases are indistinguishable to the caller.
    assert foreign.json()["detail"]["message"] == missing.json()["detail"]["message"]
    assert _counts(db) == before


def test_fork_source_must_be_a_message_of_the_parent(client: TestClient, db: CharactersRAGDB) -> None:
    assert client.post(PATH, json=_body()).status_code == 201
    before = _counts(db)
    child = "9d2f6a70-1c3b-4e5d-8f90-a1b2c3d4e5f6"
    fork = _body([_message("pa_f1", None, "user", 0)], id=child, parent_conversation_id=CHAT_ID, forked_from_message_id="pa_nope")
    response = client.post(PATH, json=fork)
    assert response.status_code == 422, response.text
    assert _error(response) == "invalid_fork_source"
    assert _counts(db) == before
