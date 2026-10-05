"""The delete rule for chat rows Sync v2 never saw, with PostgreSQL on either side (#3181).

``test_sync_v2_unenrolled_delete.py`` covers the rule on SQLite. Here the same
routes run once with the owner's chat database on PostgreSQL and once with the
Sync store on PostgreSQL, because the rule takes the dataset projection fence,
resolves a conflict inside a savepoint, and deletes owner rows inside that
fence. The tests skip when no PostgreSQL server is reachable.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service
from tldw_Server_API.tests.Sync.test_sync_v2_native_history_capture import (
    USER,
    _active_profile,
    _client,
    _dataset_id,
    _log,
)
from tldw_Server_API.tests.Sync.test_sync_v2_unenrolled_delete import (
    LEGACY_ANSWER,
    LEGACY_CHAT,
    LEGACY_QUESTION,
    _blocker,
    _delete_chat,
    _delete_message,
    _has_sync_history,
    _is_deleted,
    _leave_the_old_bug_behind,
    _legacy_chat,
    _legacy_message,
)

pytestmark = [pytest.mark.integration, pytest.mark.postgres]


@pytest.fixture(params=["chat_database_on_postgres", "sync_store_on_postgres"])
def stack(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pg_database_config: DatabaseConfig,
) -> Iterator[tuple[TestClient, CharactersRAGDB, SyncV2Service]]:
    """The routes with an active profile, one of the two databases on PostgreSQL."""
    backends = []

    def postgres_backend():
        backend = DatabaseBackendFactory.create_backend(pg_database_config)
        backends.append(backend)
        return backend

    if request.param == "chat_database_on_postgres":
        chacha_db = CharactersRAGDB(db_path=":memory:", client_id=USER, backend=postgres_backend())
        # The Sync materializers' own handle on the same database, as the service factory wires it.
        projection_db = CharactersRAGDB(db_path=":memory:", client_id=USER, backend=postgres_backend())
        sync_database = SyncDatabase(sqlite_path=tmp_path / "Sync_v2.db")
    else:
        chacha_db = CharactersRAGDB(db_path=str(tmp_path / "ChaChaNotes.db"), client_id=USER)
        projection_db = CharactersRAGDB(db_path=chacha_db.db_path_str, client_id=USER)
        sync_database = SyncDatabase(backend=postgres_backend())
    try:
        sync_service = _active_profile(sync_database, projection_db)
        yield _client(monkeypatch, chacha_db, sync_service), chacha_db, sync_service
    finally:
        for database in (projection_db, chacha_db):
            database.close_all_connections()
        for backend in backends:
            backend.get_pool().close_all()


def _published_chat(client: TestClient) -> tuple[str, str]:
    """Create a chat and one message through the ordinary Sync route: both are in the log."""
    created = client.post("/api/v1/chats/", json={"title": "Published"})
    assert created.status_code == 201, created.text
    chat_id = created.json()["id"]
    sent = client.post(f"/api/v1/chats/{chat_id}/messages", json={"role": "user", "content": "published"})
    assert sent.status_code == 201, sent.text
    return chat_id, sent.json()["id"]


def test_postgres_message_sync_never_saw_is_deleted_directly_and_writes_continue(
    stack: tuple[TestClient, CharactersRAGDB, SyncV2Service],
) -> None:
    client, chacha_db, sync_service = stack
    _legacy_chat(chacha_db, messages=(LEGACY_QUESTION, LEGACY_ANSWER))

    deleted = _delete_message(client, chacha_db, LEGACY_QUESTION)

    assert deleted.status_code == 204, deleted.text
    assert _is_deleted(chacha_db, LEGACY_QUESTION) and not _is_deleted(chacha_db, LEGACY_ANSWER)
    assert _log(sync_service) == []
    assert not _has_sync_history(sync_service, "chat.message", LEGACY_QUESTION)
    chat_id, message_id = _published_chat(client)
    assert _log(sync_service) == [
        ("chat.conversation", "upsert", chat_id, "applied"),
        ("chat.message", "append", message_id, "applied"),
    ]
    assert _blocker(sync_service) is None


def test_postgres_chat_sync_never_saw_is_deleted_without_any_envelope(
    stack: tuple[TestClient, CharactersRAGDB, SyncV2Service],
) -> None:
    client, chacha_db, sync_service = stack
    _legacy_chat(chacha_db, messages=(LEGACY_QUESTION, LEGACY_ANSWER))

    deleted = _delete_chat(client, chacha_db, LEGACY_CHAT)

    assert deleted.status_code == 204, deleted.text
    assert chacha_db.get_conversation_by_id(LEGACY_CHAT) is None
    assert _is_deleted(chacha_db, LEGACY_QUESTION) and _is_deleted(chacha_db, LEGACY_ANSWER)
    assert _log(sync_service) == []
    assert sync_service.store.list_conflicts(_dataset_id(sync_service)) == []


def test_postgres_chat_with_both_kinds_of_row_tombstones_only_the_published_ones(
    stack: tuple[TestClient, CharactersRAGDB, SyncV2Service],
) -> None:
    client, chacha_db, sync_service = stack
    chat_id, published_id = _published_chat(client)
    _legacy_message(chacha_db, "unpublished-save", chat_id, sender="assistant")

    deleted = _delete_chat(client, chacha_db, chat_id)

    assert deleted.status_code == 204, deleted.text
    assert _log(sync_service)[2:] == [
        ("chat.message", "tombstone", published_id, "applied"),
        ("chat.conversation", "tombstone", chat_id, "applied"),
    ]
    assert _is_deleted(chacha_db, published_id) and _is_deleted(chacha_db, "unpublished-save")
    assert chacha_db.get_conversation_by_id(chat_id) is None
    assert not _has_sync_history(sync_service, "chat.message", "unpublished-save")
    assert sync_service.store.list_conflicts(_dataset_id(sync_service)) == []


def test_postgres_dataset_blocked_by_the_old_delete_recovers_on_the_next_write(
    stack: tuple[TestClient, CharactersRAGDB, SyncV2Service],
) -> None:
    client, chacha_db, sync_service = stack
    _legacy_chat(chacha_db, messages=(LEGACY_QUESTION,))
    stranded = _leave_the_old_bug_behind(sync_service, LEGACY_QUESTION, LEGACY_CHAT)

    created = client.post("/api/v1/chats/", json={"title": "After the block"})

    assert created.status_code == 201, created.text
    assert _blocker(sync_service) is None
    superseded = sync_service.store.get_envelope_by_server_cursor(stranded.server_cursor)
    assert (superseded.apply_status, superseded.apply_error_code) == ("superseded", "sync_conflict_skipped")
    assert not _is_deleted(chacha_db, LEGACY_QUESTION)
    # The delete that caused the block now works, and so does the delete of a published row.
    assert _delete_message(client, chacha_db, LEGACY_QUESTION).status_code == 204
    assert _is_deleted(chacha_db, LEGACY_QUESTION)
    assert _delete_chat(client, chacha_db, created.json()["id"]).status_code == 204
    assert _log(sync_service)[-1] == ("chat.conversation", "tombstone", created.json()["id"], "applied")


def test_postgres_stranded_delete_recovers_when_it_is_retried(
    stack: tuple[TestClient, CharactersRAGDB, SyncV2Service],
) -> None:
    """The dismissal runs inside the fence the retried delete already holds."""
    client, chacha_db, sync_service = stack
    _legacy_chat(chacha_db, messages=(LEGACY_QUESTION,))
    _leave_the_old_bug_behind(sync_service, LEGACY_QUESTION, LEGACY_CHAT)

    retried = _delete_message(client, chacha_db, LEGACY_QUESTION)

    assert retried.status_code == 204, retried.text
    assert _is_deleted(chacha_db, LEGACY_QUESTION)
    assert _blocker(sync_service) is None
