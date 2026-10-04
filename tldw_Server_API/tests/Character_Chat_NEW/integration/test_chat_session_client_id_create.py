"""POST /api/v1/chats/ with a client-supplied conversation id (D7 P3).

Contract:

* ``id`` is optional. When present it must be a canonical UUID and becomes the
  chat id.
* The same owner repeating the same request gets the existing chat back with
  200 and ``Idempotency-Replayed: true``. No second chat is created.
* Any other use of a taken id (a different request, another owner, or a chat
  that was not created through a client id) is 409 ``chat_id_conflict`` and
  reveals nothing about the existing chat.
* Repeating the create after the chat was moved to trash is 410
  ``chat_deleted``; the id stays taken until the chat is permanently deleted.
* Without ``id`` the endpoint behaves exactly as before.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as deps
from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions as chats
from tldw_Server_API.app.core.Character_Chat.character_rate_limiter import CharacterRateLimiter
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration

PATH = "/api/v1/chats/"
CHAT_ID = "6b0f8c1e-2d4a-4f3b-9a71-5c2e8d9f0a14"
BODY = {"title": "Planning notes", "state": "in-progress", "source": "webui-chat"}
SECRET_TITLE = "Owner one private plans"


def _count(db: CharactersRAGDB, *, include_deleted: bool = True) -> int:
    query = "SELECT COUNT(*) AS n FROM conversations"
    if not include_deleted:
        query += " WHERE deleted = 0"
    return int(db.execute_query(query, read_only=True).fetchone()["n"])


def _client(db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, *, user_id: int = 1, sync_service: Any = None):
    app = FastAPI()
    app.include_router(chats.router, prefix="/api/v1/chats")
    app.dependency_overrides[deps.get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[chats.get_request_user] = lambda: SimpleNamespace(id=user_id)
    app.dependency_overrides[chats.require_expected_user] = lambda: None
    limiter = CharacterRateLimiter(enabled=False, max_chats_per_user=100)
    monkeypatch.setattr(chats, "get_character_rate_limiter", lambda: limiter)
    monkeypatch.setattr(chats, "_active_chat_sync_service", lambda *_args: sync_service)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def db(tmp_path: Path) -> Iterator[CharactersRAGDB]:
    database = CharactersRAGDB(tmp_path / "1" / "ChaChaNotes.db", client_id="1")
    try:
        yield database
    finally:
        database.close_all_connections()


@pytest.fixture
def client(db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    return _client(db, monkeypatch)


def _create(client: TestClient, body: dict[str, Any] | None = None, **params: Any):
    return client.post(PATH, json={"id": CHAT_ID, **(BODY if body is None else body)}, params=params or None)


# ---------------------------------------------------------------------------
# Create and replay
# ---------------------------------------------------------------------------


def test_client_id_becomes_the_chat_id(client: TestClient, db: CharactersRAGDB) -> None:
    response = _create(client)
    assert response.status_code == 201, response.text
    assert response.json()["id"] == CHAT_ID
    assert response.json()["title"] == BODY["title"]
    assert "Idempotency-Replayed" not in response.headers
    assert db.get_conversation_by_id(CHAT_ID)["client_id"] == "1"


def test_replay_returns_the_same_chat_and_creates_no_duplicate(client: TestClient, db: CharactersRAGDB) -> None:
    first = _create(client)
    second = _create(client)
    assert first.status_code == 201, first.text
    assert second.status_code == 200, second.text
    assert second.headers["Idempotency-Replayed"] == "true"
    assert second.json()["id"] == first.json()["id"] == CHAT_ID
    assert second.json()["created_at"] == first.json()["created_at"]
    assert _count(db) == 1
    # The stored fingerprint is internal: no response carries it.
    listed = client.get(PATH)
    assert listed.status_code == 200, listed.text
    for payload in (first.text, second.text, listed.text, client.get(f"{PATH}{CHAT_ID}").text):
        assert "create_request_fingerprint" not in payload
        assert db.get_conversation_by_id(CHAT_ID)["create_request_fingerprint"] not in payload


def test_replay_compares_the_original_request_not_the_current_chat(client: TestClient, db: CharactersRAGDB) -> None:
    """A rename after creation does not turn a late retry into a conflict."""
    created = _create(client)
    renamed = client.put(
        f"{PATH}{CHAT_ID}", params={"expected_version": created.json()["version"]}, json={"title": "Renamed"}
    )
    assert renamed.status_code == 200, renamed.text
    replay = _create(client)
    assert replay.status_code == 200, replay.text
    assert replay.json()["title"] == "Renamed"
    assert _count(db) == 1


@pytest.mark.parametrize(
    "equivalent",
    [
        {"source": "webui-chat", "title": "Planning notes", "state": "in-progress"},
        {"title": "Planning notes", "state": "IN-PROGRESS", "source": "webui-chat"},
        {"title": "Planning notes", "state": "in-progress", "source": "webui-chat", "unknown_extra": 1},
    ],
    ids=["key-order", "normalized-state", "ignored-extra-field"],
)
def test_replay_matches_equivalent_request_bodies(client: TestClient, db: CharactersRAGDB, equivalent: dict) -> None:
    assert _create(client).status_code == 201
    replay = _create(client, equivalent)
    assert replay.status_code == 200, replay.text
    assert _count(db) == 1


def test_uppercase_client_id_is_stored_in_canonical_form(client: TestClient, db: CharactersRAGDB) -> None:
    response = client.post(PATH, json={"id": CHAT_ID.upper(), **BODY})
    assert response.status_code == 201, response.text
    assert response.json()["id"] == CHAT_ID
    assert _create(client).status_code == 200
    assert _count(db) == 1


def test_character_chat_replay_keeps_one_chat_and_one_greeting(
    client: TestClient, db: CharactersRAGDB
) -> None:
    character_id = db.add_character_card({"name": "Archivist", "first_message": "Welcome back."})
    body = {"character_id": character_id, "title": "Archive visit"}
    first = _create(client, body, seed_first_message=True)
    second = _create(client, body, seed_first_message=True)
    assert first.status_code == 201, first.text
    assert second.status_code == 200, second.text
    assert first.json()["message_count"] == 1
    assert second.json()["character_id"] == character_id
    assert _count(db) == 1
    assert len(db.get_messages_for_conversation(CHAT_ID)) == 1


def test_character_id_aliases_replay_as_the_same_request(client: TestClient, db: CharactersRAGDB) -> None:
    character_id = db.add_character_card({"name": "Navigator"})
    assert _create(client, {"character_id": character_id}).status_code == 201
    aliased = _create(client, {"assistant_kind": "character", "assistant_id": str(character_id)})
    assert aliased.status_code == 200, aliased.text
    assert _count(db) == 1


def test_workspace_chat_with_client_id_replays(client: TestClient, db: CharactersRAGDB) -> None:
    db.upsert_workspace("ws", "Workspace")
    body = {"scope_type": "workspace", "workspace_id": "ws", "assistant_kind": None, "title": "Workspace chat"}
    first = _create(client, body)
    second = _create(client, body)
    assert first.status_code == 201, first.text
    assert second.status_code == 200, second.text
    assert first.json()["workspace_id"] == "ws"
    assert _count(db) == 1


# ---------------------------------------------------------------------------
# Conflicts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "changed,params",
    [
        ({"title": "Different title"}, {}),
        # An explicit null is a choice (for example "no assistant" in a Workspace), not an omission.
        ({"assistant_kind": None}, {}),
        ({"state": "resolved"}, {}),
        ({}, {"seed_first_message": True}),
    ],
    ids=["title", "explicit-null-choice", "state", "query-option"],
)
def test_same_id_with_a_different_request_is_409_and_changes_nothing(
    client: TestClient, db: CharactersRAGDB, changed: dict, params: dict
) -> None:
    assert _create(client).status_code == 201
    conflict = _create(client, {**BODY, **changed}, **params)
    assert conflict.status_code == 409, conflict.text
    assert conflict.json()["detail"]["error_code"] == "chat_id_conflict"
    assert db.get_conversation_by_id(CHAT_ID)["title"] == BODY["title"]
    assert _count(db) == 1


def test_id_of_a_chat_created_without_client_id_is_409(client: TestClient, db: CharactersRAGDB) -> None:
    """Server-named chats were never created by this request, so they are never replayed."""
    existing = client.post(PATH, json=BODY)
    assert existing.status_code == 201
    conflict = client.post(PATH, json={"id": existing.json()["id"], **BODY})
    assert conflict.status_code == 409, conflict.text
    assert conflict.json()["detail"]["error_code"] == "chat_id_conflict"
    assert _count(db) == 1


def test_another_owner_reusing_an_id_in_a_shared_store_gets_409_without_a_leak(
    db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = _client(db, monkeypatch, user_id=1)
    body = {"title": SECRET_TITLE}
    assert _create(owner, body).status_code == 201
    other = _client(db, monkeypatch, user_id=2)
    for attempt in (body, {"title": "Something else"}):
        response = _create(other, attempt)
        assert response.status_code == 409, response.text
        assert response.json()["detail"]["error_code"] == "chat_id_conflict"
        assert SECRET_TITLE not in response.text
        assert "Idempotency-Replayed" not in response.headers
    assert db.get_conversation_by_id(CHAT_ID)["client_id"] == "1"
    assert _count(db) == 1
    assert other.get(f"{PATH}{CHAT_ID}").status_code in {403, 404}


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def owners(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[SimpleNamespace]:
    """Two real owners: PostgreSQL shares one database; SQLite uses per-user files."""
    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    first = CharactersRAGDB(tmp_path / "1" / "ChaChaNotes.db", client_id="1", backend=backend)
    second = CharactersRAGDB(tmp_path / "2" / "ChaChaNotes.db", client_id="2", backend=backend)
    try:
        yield SimpleNamespace(first=first, second=second, postgres=request.param == "postgres")
    finally:
        first.close_all_connections()
        second.close_all_connections()
        if backend is not None:
            backend.get_pool().close_all()


def test_another_owner_reusing_an_id_never_sees_the_first_owners_chat(
    owners: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PostgreSQL ids share one key space (409); per-user SQLite files do not collide."""
    first = _client(owners.first, monkeypatch, user_id=1)
    assert _create(first, {"title": SECRET_TITLE}).status_code == 201
    second = _client(owners.second, monkeypatch, user_id=2)
    response = _create(second, {"title": SECRET_TITLE})
    if owners.postgres:
        assert response.status_code == 409, response.text
        assert response.json()["detail"]["error_code"] == "chat_id_conflict"
        assert SECRET_TITLE not in response.text
        assert owners.second.get_conversation_by_id(CHAT_ID, include_deleted=True) is None
    else:
        # Each SQLite owner has a separate store, so the id names the caller's own new chat.
        assert response.status_code == 201, response.text
        assert owners.second.get_conversation_by_id(CHAT_ID)["client_id"] == "2"
    row = owners.first.get_conversation_by_id(CHAT_ID)
    assert (row["client_id"], row["title"]) == ("1", SECRET_TITLE)


# ---------------------------------------------------------------------------
# Validation and the no-id path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_id",
    [
        "not-a-uuid",
        "",
        "6b0f8c1e2d4a4f3b9a715c2e8d9f0a14",
        "{6b0f8c1e-2d4a-4f3b-9a71-5c2e8d9f0a14}",
        "urn:uuid:6b0f8c1e-2d4a-4f3b-9a71-5c2e8d9f0a14",
        " 6b0f8c1e-2d4a-4f3b-9a71-5c2e8d9f0a14",
        "00000000-0000-0000-0000-000000000000",
        "ffffffff-ffff-ffff-ffff-ffffffffffff",
        12345,
    ],
)
def test_invalid_client_id_is_422_and_creates_nothing(client: TestClient, db: CharactersRAGDB, bad_id: Any) -> None:
    response = client.post(PATH, json={"id": bad_id, **BODY})
    assert response.status_code == 422, response.text
    assert _count(db) == 0


def test_without_id_each_create_makes_a_new_server_named_chat(client: TestClient, db: CharactersRAGDB) -> None:
    first = client.post(PATH, json=BODY)
    second = client.post(PATH, json=BODY)
    assert first.status_code == second.status_code == 201
    assert first.json()["id"] != second.json()["id"]
    for response in (first, second):
        uuid.UUID(response.json()["id"])
        assert "Idempotency-Replayed" not in response.headers
        assert db.get_conversation_by_id(response.json()["id"])["create_request_fingerprint"] is None
    assert _count(db) == 2


def test_explicit_null_id_behaves_like_no_id(client: TestClient, db: CharactersRAGDB) -> None:
    assert client.post(PATH, json={"id": None, **BODY}).status_code == 201
    assert client.post(PATH, json={"id": None, **BODY}).status_code == 201
    assert _count(db) == 2


def test_client_id_is_refused_while_sync_v2_is_active(db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(db, monkeypatch, sync_service=object())
    response = _create(client)
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["error_code"] == "sync_client_chat_id_unsupported"
    assert _count(db) == 0


# ---------------------------------------------------------------------------
# Trash
# ---------------------------------------------------------------------------


def _trash(client: TestClient, version: int) -> None:
    response = client.delete(f"{PATH}{CHAT_ID}", params={"expected_version": version})
    assert response.status_code == 204, response.text


def test_replaying_a_trashed_chat_is_410_and_does_not_restore_or_recreate(
    client: TestClient, db: CharactersRAGDB
) -> None:
    created = _create(client)
    _trash(client, created.json()["version"])
    gone = _create(client)
    assert gone.status_code == 410, gone.text
    assert gone.json()["detail"]["error_code"] == "chat_deleted"
    assert db.get_conversation_by_id(CHAT_ID) is None
    assert db.get_conversation_by_id(CHAT_ID, include_deleted=True)["deleted"]
    assert _count(db) == 1
    different = _create(client, {"title": "Different title"})
    assert different.status_code == 409, different.text


def test_restored_chat_replays_again(client: TestClient, db: CharactersRAGDB) -> None:
    created = _create(client)
    _trash(client, created.json()["version"])
    trashed = db.get_conversation_by_id(CHAT_ID, include_deleted=True)
    restored = client.post(f"{PATH}{CHAT_ID}/restore", params={"expected_version": trashed["version"]})
    assert restored.status_code == 200, restored.text
    assert _create(client).status_code == 200
    assert _count(db) == 1


def test_permanently_deleted_chat_frees_its_id(client: TestClient, db: CharactersRAGDB) -> None:
    created = _create(client)
    _trash(client, created.json()["version"])
    trashed = db.get_conversation_by_id(CHAT_ID, include_deleted=True)
    purged = client.delete(
        f"{PATH}{CHAT_ID}", params={"expected_version": trashed["version"], "hard_delete": True}
    )
    assert purged.status_code == 204, purged.text
    assert _create(client).status_code == 201
    assert _count(db) == 1


# ---------------------------------------------------------------------------
# Concurrency: the primary key, not the pre-insert read, decides
# ---------------------------------------------------------------------------


def _hide_existing_chat_from_first_lookup(db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a request whose pre-insert read ran before a concurrent insert committed."""
    real = db.get_conversation_by_id
    calls = {"n": 0}

    def stale(conversation_id: str, include_deleted: bool = False) -> dict | None:
        if conversation_id == CHAT_ID:
            calls["n"] += 1
            if calls["n"] == 1:
                return None
        return real(conversation_id, include_deleted=include_deleted)

    monkeypatch.setattr(db, "get_conversation_by_id", stale)


def test_lost_race_with_the_same_request_replays(
    client: TestClient, db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _create(client).status_code == 201
    _hide_existing_chat_from_first_lookup(db, monkeypatch)
    raced = _create(client)
    assert raced.status_code == 200, raced.text
    assert raced.headers["Idempotency-Replayed"] == "true"
    assert _count(db) == 1


def test_lost_race_with_a_different_request_is_409(
    client: TestClient, db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _create(client).status_code == 201
    _hide_existing_chat_from_first_lookup(db, monkeypatch)
    raced = _create(client, {"title": "Different title"})
    assert raced.status_code == 409, raced.text
    assert db.get_conversation_by_id(CHAT_ID)["title"] == BODY["title"]
    assert _count(db) == 1


def test_concurrent_duplicate_creates_produce_one_chat(
    client: TestClient, db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every request passes the pre-insert read before any of them inserts."""
    workers = 6
    barrier = threading.Barrier(workers, timeout=30)
    real = db.get_conversation_by_id
    seen = threading.local()

    def gated(conversation_id: str, include_deleted: bool = False) -> dict | None:
        row = real(conversation_id, include_deleted=include_deleted)
        if conversation_id == CHAT_ID and not getattr(seen, "waited", False):
            seen.waited = True
            barrier.wait()
        return row

    monkeypatch.setattr(db, "get_conversation_by_id", gated)
    results: list[Any] = [None] * workers

    def worker(index: int) -> None:
        results[index] = _create(client)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    statuses = sorted(response.status_code for response in results)
    assert statuses == [200] * (workers - 1) + [201], [response.text for response in results]
    assert {response.json()["id"] for response in results} == {CHAT_ID}
    assert _count(db) == 1
