"""Deletes of chat rows Sync v2 never saw, with PostgreSQL on either side (#3181).

``test_sync_v2_unenrolled_delete.py`` covers the server's delete routes on
SQLite, ``test_sync_v2_satisfied_tombstone.py`` a device's tombstone of a
message the server does not have, and
``test_sync_v2_satisfied_tombstone_objects.py`` the same for a chat and a note
(§5.4, Q7). Here they run once with the owner's database on PostgreSQL and once
with the Sync store on PostgreSQL, because they take the dataset projection
fence, settle a conflict inside a savepoint, delete owner rows inside that
fence, and publish the id-only tombstone of a direct delete in the same Sync
transaction (§5.2, Q8). The tests skip when no PostgreSQL server is reachable.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service
from tldw_Server_API.tests.Sync.test_sync_v2_native_history_capture import (
    USER,
    _active_profile,
    _block_dataset,
    _client,
    _dataset_id,
    _log,
)
from tldw_Server_API.tests.Sync.test_sync_v2_satisfied_tombstone import (
    GONE,
    THIRD,
    _envelope,
    _push,
    _push_tombstone,
    _state,
    _strand_device_tombstone,
)
from tldw_Server_API.tests.Sync.test_sync_v2_satisfied_tombstone_objects import (
    OBJECT_DOMAINS,
    _copy_outside_sync,
    _row,
    _service,
)
from tldw_Server_API.tests.Sync.test_sync_v2_satisfied_tombstone_objects import (
    _is_deleted as _object_is_deleted,
)
from tldw_Server_API.tests.Sync.test_sync_v2_satisfied_tombstone_objects import (
    _push_tombstone as _push_object_tombstone,
)
from tldw_Server_API.tests.Sync.test_sync_v2_satisfied_tombstone_objects import (
    _state as _object_state,
)
from tldw_Server_API.tests.Sync.test_sync_v2_satisfied_tombstone_objects import (
    _strand as _strand_object_tombstone,
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
    assert _log(sync_service) == [("chat.message", "tombstone", LEGACY_QUESTION, "applied")]
    assert _state(sync_service, LEGACY_QUESTION).deleted is True
    chat_id, message_id = _published_chat(client)
    assert _log(sync_service)[1:] == [
        ("chat.conversation", "upsert", chat_id, "applied"),
        ("chat.message", "append", message_id, "applied"),
    ]
    assert _blocker(sync_service) is None


def test_postgres_chat_sync_never_saw_is_deleted_and_published_by_id(
    stack: tuple[TestClient, CharactersRAGDB, SyncV2Service],
) -> None:
    client, chacha_db, sync_service = stack
    _legacy_chat(chacha_db, messages=(LEGACY_QUESTION, LEGACY_ANSWER))

    deleted = _delete_chat(client, chacha_db, LEGACY_CHAT)

    assert deleted.status_code == 204, deleted.text
    assert chacha_db.get_conversation_by_id(LEGACY_CHAT) is None
    assert _is_deleted(chacha_db, LEGACY_QUESTION) and _is_deleted(chacha_db, LEGACY_ANSWER)
    assert _log(sync_service) == [
        ("chat.message", "tombstone", LEGACY_QUESTION, "applied"),
        ("chat.message", "tombstone", LEGACY_ANSWER, "applied"),
        ("chat.conversation", "tombstone", LEGACY_CHAT, "applied"),
    ]
    assert sync_service.store.list_conflicts(_dataset_id(sync_service)) == []


def test_postgres_a_blocked_dataset_still_takes_a_direct_delete_and_publishes_nothing(
    stack: tuple[TestClient, CharactersRAGDB, SyncV2Service],
) -> None:
    client, chacha_db, sync_service = stack
    _legacy_chat(chacha_db, messages=(LEGACY_QUESTION,))
    _block_dataset(sync_service)
    before = _log(sync_service)

    deleted = _delete_message(client, chacha_db, LEGACY_QUESTION)

    assert deleted.status_code == 204, deleted.text
    assert _is_deleted(chacha_db, LEGACY_QUESTION)
    assert _log(sync_service) == before
    assert not _has_sync_history(sync_service, "chat.message", LEGACY_QUESTION)


def test_postgres_chat_with_both_kinds_of_row_tombstones_only_the_published_ones(
    stack: tuple[TestClient, CharactersRAGDB, SyncV2Service],
) -> None:
    client, chacha_db, sync_service = stack
    chat_id, published_id = _published_chat(client)
    _legacy_message(chacha_db, "unpublished-save", chat_id, sender="assistant")

    deleted = _delete_chat(client, chacha_db, chat_id)

    assert deleted.status_code == 204, deleted.text
    assert _log(sync_service)[2:] == [
        ("chat.message", "tombstone", "unpublished-save", "applied"),
        ("chat.message", "tombstone", published_id, "applied"),
        ("chat.conversation", "tombstone", chat_id, "applied"),
    ]
    assert _is_deleted(chacha_db, published_id) and _is_deleted(chacha_db, "unpublished-save")
    assert chacha_db.get_conversation_by_id(chat_id) is None
    assert _state(sync_service, "unpublished-save").deleted is True
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


# ---------------------------------------------------------------------------
# A device's tombstone of a message the server does not have
# ---------------------------------------------------------------------------


def _third_device(sync_service: SyncV2Service) -> str:
    sync_service.register_device(user_id=USER, display_name="Phone", client_type="chatbook", device_id=THIRD)
    return THIRD


def test_postgres_device_tombstone_of_an_unknown_message_is_applied_and_delivered(
    stack: tuple[TestClient, CharactersRAGDB, SyncV2Service],
) -> None:
    client, chacha_db, sync_service = stack
    third = _third_device(sync_service)

    pushed = _push_tombstone(sync_service)

    assert pushed.conflicts == [] and pushed.rejected == []
    assert [item.apply_status for item in pushed.accepted] == ["applied"]
    assert _state(sync_service).deleted is True
    assert chacha_db.get_message_by_id(GONE, include_deleted=True) is None
    assert _blocker(sync_service) is None
    pulled = sync_service.pull(
        user_id=USER,
        dataset_id=_dataset_id(sync_service),
        device_id=third,
        cursor="0",
        domains=["chat.conversation", "chat.message"],
    ).envelopes
    assert [(item.operation, item.object_id) for item in pulled] == [("tombstone", GONE)]
    assert client.post("/api/v1/chats/", json={"title": "After the tombstone"}).status_code == 201


def test_postgres_device_tombstone_deletes_the_copy_the_server_holds_outside_sync(
    stack: tuple[TestClient, CharactersRAGDB, SyncV2Service],
) -> None:
    _client, chacha_db, sync_service = stack
    _legacy_chat(chacha_db, messages=(LEGACY_QUESTION, LEGACY_ANSWER))

    pushed = _push_tombstone(sync_service, LEGACY_QUESTION)

    assert pushed.conflicts == []
    assert [item.apply_status for item in pushed.accepted] == ["applied"]
    assert _is_deleted(chacha_db, LEGACY_QUESTION) and not _is_deleted(chacha_db, LEGACY_ANSWER)
    assert _state(sync_service, LEGACY_QUESTION).deleted is True


def test_postgres_dataset_blocked_by_a_device_tombstone_recovers_on_the_next_push(
    stack: tuple[TestClient, CharactersRAGDB, SyncV2Service],
) -> None:
    _client, chacha_db, sync_service = stack
    third = _third_device(sync_service)
    _legacy_chat(chacha_db)
    stranded = _strand_device_tombstone(sync_service)

    pushed = _push(sync_service, _envelope(sync_service, "append", "from-the-phone", device=third))

    assert pushed.conflicts == []
    assert [item.apply_status for item in pushed.accepted] == ["applied"]
    assert _blocker(sync_service) is None
    assert sync_service.store.get_envelope_by_server_cursor(stranded.server_cursor).apply_status == "applied"
    assert sync_service.store.get_conflict("conflict-device-tombstone").status == "resolved"
    assert _state(sync_service).deleted is True


def test_postgres_device_tombstone_never_reaches_another_owners_message(
    tmp_path: Path,
    pg_database_config: DatabaseConfig,
) -> None:
    """One database holds both owners' chats, and the documented role bypasses row-level security.

    The tombstone names a message by id only. It must find nothing for an owner
    who does not hold the chat, even though the connection can read the row.
    """
    backends = [DatabaseBackendFactory.create_backend(pg_database_config) for _ in range(3)]
    first = CharactersRAGDB(db_path=":memory:", client_id=USER, backend=backends[0])
    first_projection = CharactersRAGDB(db_path=":memory:", client_id=USER, backend=backends[1])
    second = CharactersRAGDB(db_path=":memory:", client_id="user-2", backend=backends[2])
    try:
        second.add_conversation({"id": "their-chat", "character_id": None, "title": "Theirs", "client_id": "user-2"})
        second.add_message({"id": "their-message", "conversation_id": "their-chat", "sender": "user", "content": "x"})
        sync_service = _active_profile(SyncDatabase(sqlite_path=tmp_path / "Sync_v2.db"), first_projection)

        pushed = _push_tombstone(sync_service, "their-message")

        assert pushed.conflicts == []
        assert [item.apply_status for item in pushed.accepted] == ["applied"]
        assert not _is_deleted(second, "their-message")
        assert _state(sync_service, "their-message").deleted is True
    finally:
        for database in (first_projection, first, second):
            database.close_connection()
        for backend in backends:
            backend.get_pool().close_all()


# ---------------------------------------------------------------------------
# A device's tombstone of a chat or a note the server does not have (Q7)
# ---------------------------------------------------------------------------


@pytest.fixture(params=["chat_database_on_postgres", "sync_store_on_postgres"])
def object_stack(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    pg_database_config: DatabaseConfig,
) -> Iterator[tuple[CharactersRAGDB, SyncV2Service]]:
    """The chat and notes projection, one of the two databases on PostgreSQL."""
    backends = []

    def postgres_backend():
        backend = DatabaseBackendFactory.create_backend(pg_database_config)
        backends.append(backend)
        return backend

    if request.param == "chat_database_on_postgres":
        chacha_db = CharactersRAGDB(db_path=":memory:", client_id=USER, backend=postgres_backend())
        projection_db = CharactersRAGDB(db_path=":memory:", client_id=USER, backend=postgres_backend())
        sync_database = SyncDatabase(sqlite_path=tmp_path / "Sync_v2.db")
    else:
        chacha_db = CharactersRAGDB(db_path=str(tmp_path / "ChaChaNotes.db"), client_id=USER)
        projection_db = CharactersRAGDB(db_path=chacha_db.db_path_str, client_id=USER)
        sync_database = SyncDatabase(backend=postgres_backend())
    try:
        yield chacha_db, _service(sync_database, projection_db)
    finally:
        for database in (projection_db, chacha_db):
            database.close_all_connections()
        for backend in backends:
            backend.get_pool().close_all()


@pytest.mark.parametrize("domain", OBJECT_DOMAINS)
def test_postgres_device_tombstone_of_an_unknown_object_is_applied_and_its_copy_deleted(
    object_stack: tuple[CharactersRAGDB, SyncV2Service],
    domain: str,
) -> None:
    chacha_db, sync_service = object_stack
    _copy_outside_sync(chacha_db, domain, "kept-outside-sync")

    absent = _push_object_tombstone(sync_service, domain)
    copy = _push_object_tombstone(sync_service, domain, "kept-outside-sync")

    for pushed in (absent, copy):
        assert pushed.conflicts == [] and pushed.rejected == []
        assert [item.apply_status for item in pushed.accepted] == ["applied"]
    assert _row(chacha_db, domain) is None
    assert _object_is_deleted(chacha_db, domain, "kept-outside-sync")
    assert _object_state(sync_service, domain).deleted is True
    assert _object_state(sync_service, domain, "kept-outside-sync").deleted is True
    assert sync_service.store.get_unresolved_materialization_conflict(_dataset_id(sync_service)) is None


@pytest.mark.parametrize("domain", OBJECT_DOMAINS)
def test_postgres_stranded_object_tombstone_recovers_on_the_next_push(
    object_stack: tuple[CharactersRAGDB, SyncV2Service],
    domain: str,
) -> None:
    chacha_db, sync_service = object_stack
    _copy_outside_sync(chacha_db, domain)
    stranded = _strand_object_tombstone(sync_service, domain, with_base=True)

    again = _push_object_tombstone(sync_service, domain, "something-else")

    assert again.conflicts == []
    assert sync_service.store.get_envelope_by_server_cursor(stranded.server_cursor).apply_status == "applied"
    assert _object_is_deleted(chacha_db, domain)
    assert sync_service.store.get_unresolved_materialization_conflict(_dataset_id(sync_service)) is None


@pytest.mark.parametrize("domain", OBJECT_DOMAINS)
def test_postgres_object_tombstone_never_reaches_another_owners_row(
    tmp_path: Path,
    pg_database_config: DatabaseConfig,
    domain: str,
) -> None:
    """One database holds both owners' rows, and the documented role bypasses row-level security."""
    backends = [DatabaseBackendFactory.create_backend(pg_database_config) for _ in range(3)]
    first = CharactersRAGDB(db_path=":memory:", client_id=USER, backend=backends[0])
    first_projection = CharactersRAGDB(db_path=":memory:", client_id=USER, backend=backends[1])
    second = CharactersRAGDB(db_path=":memory:", client_id="user-2", backend=backends[2])
    try:
        if domain == "chat.conversation":
            second.add_conversation({"id": "theirs", "character_id": None, "title": "Theirs", "client_id": "user-2"})
        else:
            second.add_note(title="Theirs", content="not yours to delete", note_id="theirs")
        sync_service = _service(SyncDatabase(sqlite_path=tmp_path / "Sync_v2.db"), first_projection)

        pushed = _push_object_tombstone(sync_service, domain, "theirs")

        assert pushed.conflicts == []
        assert [item.apply_status for item in pushed.accepted] == ["applied"]
        assert not _object_is_deleted(second, domain, "theirs")
        assert _object_state(sync_service, domain, "theirs").deleted is True
    finally:
        for database in (first_projection, first, second):
            database.close_connection()
        for backend in backends:
            backend.get_pool().close_all()


@pytest.mark.parametrize("domain", OBJECT_DOMAINS)
def test_postgres_object_tombstone_under_forced_row_level_security(
    pg_database_config: DatabaseConfig,
    domain: str,
) -> None:
    """The store methods behind the rule, run as a role that does not bypass row-level security.

    The owner's copy kept outside Sync is deleted, and the other owner's row with
    the same table is neither seen nor changed: the forced tenant policy and the
    owner filter agree. Before the rule, these methods did not exist.
    """
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    owner = CharactersRAGDB(db_path=":memory:", client_id=USER, backend=backend)
    other = CharactersRAGDB(db_path=":memory:", client_id="user-2", backend=backend)
    role = backend.escape_identifier(f"sync_tombstone_rls_{uuid4().hex[:12]}")
    role_created = False

    @contextmanager
    def restricted(database: CharactersRAGDB) -> Iterator[None]:
        with chacha_operation(independent=True), database.transaction() as conn:
            conn.execute(f"SET LOCAL ROLE {role}")
            flags = dict(
                conn.execute("SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user").fetchone()
            )
            assert flags == {"rolsuper": False, "rolbypassrls": False}
            table = "conversations" if domain == "chat.conversation" else "notes"
            policy = dict(
                conn.execute(
                    "SELECT relrowsecurity, relforcerowsecurity FROM pg_class WHERE relname = ?", (table,)
                ).fetchone()
            )
            assert policy == {"relrowsecurity": True, "relforcerowsecurity": True}
            yield

    def tombstone(object_id: str) -> str:
        if domain == "chat.conversation":
            return owner.tombstone_unsynced_conversation_from_sync(
                conversation_id=object_id, sync_client_id=USER, object_revision=1, object_hash="sha256:gone"
            )
        return owner.tombstone_unsynced_note_from_sync(
            note_id=object_id, sync_client_id=USER, object_revision=1, object_hash="sha256:gone"
        )

    try:
        _copy_outside_sync(owner, domain, "mine")
        if domain == "chat.conversation":
            other.add_conversation({"id": "theirs", "character_id": None, "title": "Theirs", "client_id": "user-2"})
        else:
            other.add_note(title="Theirs", content="not yours to delete", note_id="theirs")
        with backend.transaction() as conn:
            backend.execute(f"CREATE ROLE {role} NOLOGIN NOSUPERUSER NOBYPASSRLS", connection=conn)
            backend.execute(f"GRANT USAGE ON SCHEMA public TO {role}", connection=conn)
            backend.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {role}", connection=conn)
            backend.execute(f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {role}", connection=conn)
            backend.execute(f"GRANT {role} TO CURRENT_USER", connection=conn)
        role_created = True

        with restricted(owner):
            assert _row(owner, domain, "theirs") is None  # the policy hides the other owner's row
            assert tombstone("mine") == "deleted"
            assert tombstone("theirs") == "absent"
            assert tombstone("mine") == "deleted"  # a replay changes nothing

        assert _object_is_deleted(owner, domain, "mine")
        assert not _object_is_deleted(other, domain, "theirs")
    finally:
        try:
            if role_created:
                with backend.transaction() as conn:
                    backend.execute(f"DROP OWNED BY {role}", connection=conn)
                    backend.execute(f"DROP ROLE {role}", connection=conn)
        finally:
            for database in (owner, other):
                database.close_connection()
            backend.get_pool().close_all()
