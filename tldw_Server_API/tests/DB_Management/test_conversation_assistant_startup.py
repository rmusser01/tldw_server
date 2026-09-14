"""Local conversation startup provenance storage and lifecycle contracts."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event
from time import monotonic
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tldw_Server_API.app.core.Chat.assistant_startup import (
    AssistantStartup,
    decode_assistant_startup,
    encode_assistant_startup,
)
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)

pytestmark = pytest.mark.integration

ORIGIN_COLUMN = "assistant_startup_json"


@pytest.fixture(params=["sqlite", "postgres"])
def db_factory(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Callable[[], CharactersRAGDB]]:
    """Use the official backend fixtures and close every fresh/reopened handle."""
    config = request.getfixturevalue("pg_database_config") if request.param == "postgres" else None
    opened: list[CharactersRAGDB] = []

    def create() -> CharactersRAGDB:
        """Open the isolated database through its real initializer."""
        backend = DatabaseBackendFactory.create_backend(config) if config else None
        database = CharactersRAGDB(tmp_path / "startup.db", client_id="user-1", backend=backend)
        opened.append(database)
        return database

    try:
        yield create
    finally:
        for database in opened:
            database.close_all_connections()


def test_fresh_untrusted_insertion_stores_no_origin(db_factory: Callable[[], CharactersRAGDB]) -> None:
    """Ordinary insertion must not invent creation history from identity."""
    db = db_factory()
    cid = db.add_conversation({"title": "Legacy caller", "client_id": "user-1"})
    assert db.get_conversation_by_id(cid)[ORIGIN_COLUMN] is None
    db.close_all_connections()
    assert db_factory().get_conversation_by_id(cid)[ORIGIN_COLUMN] is None


def test_v68_upgrade_keeps_existing_identity_and_unknown_origin(
    db_factory: Callable[[], CharactersRAGDB],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Upgrade genuine old storage without fabricating or changing its identity."""
    with monkeypatch.context() as historical:
        historical.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 68)
        historical.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 68)
        legacy = db_factory()
    with legacy.transaction() as conn:
        conn.execute(
            "INSERT INTO conversations (id, root_id, title, client_id, assistant_kind, assistant_id, "
            "persona_memory_mode, version) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("legacy", "legacy", "Original", "user-1", "persona", "persona-a", "read_only", 7),
        )
    assert ORIGIN_COLUMN not in legacy.get_conversation_by_id("legacy")
    legacy.close_all_connections()
    upgraded = db_factory()
    row = upgraded.get_conversation_by_id("legacy")
    assert row[ORIGIN_COLUMN] is None
    assert (row["title"], row["assistant_kind"], row["assistant_id"], row["persona_memory_mode"], row["version"]) == (
        "Original",
        "persona",
        "persona-a",
        "read_only",
        7,
    )


def test_storage_limit_counts_utf8_bytes_and_rolls_back_rejected_update(
    db_factory: Callable[[], CharactersRAGDB],
) -> None:
    """SQL rejects more than 1024 bytes even when the character count is smaller."""
    db = db_factory()
    cid = db.add_conversation({"title": "Bounded", "client_id": "user-1"})
    accepted = "\u00e9" * 512
    with db.transaction() as conn:
        conn.execute("UPDATE conversations SET assistant_startup_json = ? WHERE id = ?", (accepted, cid))
    with pytest.raises((sqlite3.IntegrityError, DatabaseError, CharactersRAGDBError)):
        with db.transaction() as conn:
            conn.execute("UPDATE conversations SET assistant_startup_json = ? WHERE id = ?", (accepted + "x", cid))
    assert db.get_conversation_by_id(cid)[ORIGIN_COLUMN] == accepted


def test_failed_v69_migration_rolls_back_column_and_preserves_old_row(
    db_factory: Callable[[], CharactersRAGDB],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An error after real DDL leaves v68 storage readable and unchanged."""
    with monkeypatch.context() as historical:
        historical.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 68)
        historical.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 68)
        legacy = db_factory()
    with legacy.transaction() as conn:
        conn.execute(
            "INSERT INTO conversations (id, root_id, title, client_id) VALUES (?, ?, ?, ?)",
            ("legacy", "legacy", "Retained", "user-1"),
        )
    method = (
        "_migrate_from_v68_to_v69_postgres" if legacy.backend_type.value == "postgresql" else "_migrate_from_v68_to_v69"
    )
    migrate = getattr(CharactersRAGDB, method)
    legacy.close_all_connections()

    def fail_after_ddl(self: CharactersRAGDB, conn: Any) -> None:
        """Fail only after the real registered migration has executed."""
        migrate(self, conn)
        raise CharactersRAGDBError("injected startup migration failure")

    with monkeypatch.context() as failing:
        failing.setattr(CharactersRAGDB, method, fail_after_ddl)
        with pytest.raises(CharactersRAGDBError, match="injected startup migration failure"):
            db_factory()
    with monkeypatch.context() as historical:
        historical.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 68)
        historical.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 68)
        rolled_back = db_factory()
    row = rolled_back.get_conversation_by_id("legacy")
    assert ORIGIN_COLUMN not in row
    assert row["title"] == "Retained"


def test_trusted_origin_is_atomic_with_identity_and_outer_rollback(
    db_factory: Callable[[], CharactersRAGDB],
) -> None:
    """The trusted keyword inserts once on the caller's transaction connection."""
    db = db_factory()
    payload = {"id": "atomic", "assistant_kind": "persona", "assistant_id": "persona-a", "client_id": "user-1"}
    with pytest.raises(RuntimeError, match="abort creation"):
        with db.transaction() as conn:
            db.add_conversation(payload, conn=conn, assistant_startup=AssistantStartup(source="explicit"))
            raise RuntimeError("abort creation")
    assert db.get_conversation_by_id("atomic", include_deleted=True) is None
    cid = db.add_conversation(payload, assistant_startup=AssistantStartup(source="explicit"))
    row = db.get_conversation_by_id(cid)
    assert row["assistant_id"] == "persona-a"
    assert decode_assistant_startup(row[ORIGIN_COLUMN]).source == "explicit"


@pytest.mark.parametrize("key", ["assistant_startup", "assistant_startup_json"])
@pytest.mark.parametrize("value", [None, {"source": "explicit"}])
def test_untrusted_dictionary_cannot_write_origin(
    db_factory: Callable[[], CharactersRAGDB],
    key: str,
    value: Any,
) -> None:
    """Both reserved keys are rejected even when their supplied value is null."""
    db = db_factory()
    with pytest.raises(InputError):
        db.add_conversation({"id": "forged", "client_id": "user-1", key: value})
    assert db.get_conversation_by_id("forged", include_deleted=True) is None
    cid = db.add_conversation({"title": "Original", "client_id": "user-1"})
    before = db.get_conversation_by_id(cid)
    with pytest.raises(InputError):
        db.update_conversation(cid, {"title": "Forged", key: value}, before["version"])
    assert db.get_conversation_by_id(cid) == before


def test_metadata_and_normalized_noops_preserve_origin_until_actual_identity_change(
    db_factory: Callable[[], CharactersRAGDB],
) -> None:
    """Origin survives metadata/no-op writes, but memory changes invalidate it permanently."""
    db = db_factory()
    cid = db.add_conversation(
        {"assistant_kind": "persona", "assistant_id": "persona-a", "persona_memory_mode": "read_only"},
        assistant_startup=AssistantStartup(source="explicit"),
    )
    raw = db.get_conversation_by_id(cid)[ORIGIN_COLUMN]
    for update in ({"title": "Renamed"}, {}, {"assistant_kind": " PERSONA ", "assistant_id": " persona-a "}):
        row = db.get_conversation_by_id(cid)
        db.update_conversation(cid, update, row["version"])
        assert db.get_conversation_by_id(cid)[ORIGIN_COLUMN] == raw
    for mode in ("read_write", "read_only"):
        row = db.get_conversation_by_id(cid)
        db.update_conversation(cid, {"persona_memory_mode": mode}, row["version"])
        changed = db.get_conversation_by_id(cid)
        assert changed["persona_memory_mode"] == mode
        assert changed[ORIGIN_COLUMN] is None


def test_sync_scope_only_move_preserves_origin_but_rebinding_clears_it(
    db_factory: Callable[[], CharactersRAGDB],
) -> None:
    """Whole-object Sync compares actual normalized binding, not revision or scope alone."""
    db = db_factory()
    db.upsert_workspace("origin", "Origin")
    db.upsert_workspace("destination", "Destination")
    cid = db.add_conversation(
        {
            "id": "synced",
            "assistant_kind": "persona",
            "assistant_id": "persona-a",
            "scope_type": "workspace",
            "workspace_id": "origin",
        },
        assistant_startup=AssistantStartup(source="workspace_default", workspace_id="origin", workspace_version=1),
    )
    raw = db.get_conversation_by_id(cid)[ORIGIN_COLUMN]
    replacement = {
        "conversation_id": cid,
        "title": "Moved",
        "sync_client_id": "user-1",
        "object_revision": 1,
        "object_hash": "ignored",
        "assistant_kind": "persona",
        "assistant_id": "persona-a",
        "scope_type": "workspace",
        "workspace_id": "destination",
    }
    db.upsert_conversation_from_sync(**replacement)
    moved = db.get_conversation_by_id(cid)
    assert moved["workspace_id"] == "destination"
    assert moved[ORIGIN_COLUMN] == raw
    replacement["assistant_id"] = "persona-b"
    db.upsert_conversation_from_sync(**replacement)
    assert db.get_conversation_by_id(cid)[ORIGIN_COLUMN] is None
    replacement["assistant_id"] = "persona-a"
    db.upsert_conversation_from_sync(**replacement)
    assert db.get_conversation_by_id(cid)[ORIGIN_COLUMN] is None


def test_soft_delete_restore_preserve_origin_and_hard_delete_removes_it(
    db_factory: Callable[[], CharactersRAGDB],
) -> None:
    """Deletion keeps historical origin only while the conversation row exists."""
    db = db_factory()
    cid = db.add_conversation({"title": "Lifecycle"}, assistant_startup=AssistantStartup(source="explicit_none"))
    initial = db.get_conversation_by_id(cid)
    db.soft_delete_conversation(cid, initial["version"])
    deleted = db.get_conversation_by_id(cid, include_deleted=True)
    assert deleted[ORIGIN_COLUMN] == initial[ORIGIN_COLUMN]
    db.restore_conversation(cid, deleted["version"])
    assert db.get_conversation_by_id(cid)[ORIGIN_COLUMN] == initial[ORIGIN_COLUMN]
    db.hard_delete_conversation(cid)
    assert db.get_conversation_by_id(cid, include_deleted=True) is None


def test_failed_identity_writes_leave_origin_and_binding_unchanged(db_factory: Callable[[], CharactersRAGDB]) -> None:
    """CAS conflicts, invalid identities and enclosing rollback cannot clear origin."""
    db = db_factory()
    cid = db.add_conversation(
        {"assistant_kind": "persona", "assistant_id": "persona-a"},
        assistant_startup=AssistantStartup(source="explicit"),
    )
    before = db.get_conversation_by_id(cid)
    with pytest.raises(ConflictError):
        db.update_conversation(cid, {"assistant_id": "persona-b"}, before["version"] + 1)
    with pytest.raises(InputError):
        db.update_conversation(cid, {"assistant_id": None}, before["version"])
    with pytest.raises(RuntimeError, match="rollback"):
        with db.transaction():
            db.update_conversation(cid, {"assistant_id": "persona-b"}, before["version"])
            raise RuntimeError("rollback")
    assert db.get_conversation_by_id(cid) == before


def test_settings_and_message_history_preserve_origin(db_factory: Callable[[], CharactersRAGDB]) -> None:
    """Independent settings/history counters must not be provenance invalidators."""
    db = db_factory()
    cid = db.add_conversation({"title": "Settings"}, assistant_startup=AssistantStartup(source="explicit_none"))
    before = db.get_conversation_by_id(cid)
    assert db.upsert_conversation_settings(cid, {"temperature": 0.5})
    mid = db.add_message({"conversation_id": cid, "sender": "user", "content": "Hello"})
    assert mid
    after = db.get_conversation_by_id(cid)
    assert after["version"] > before["version"]
    assert after["history_version"] > before["history_version"]
    assert after[ORIGIN_COLUMN] == before[ORIGIN_COLUMN]


def test_metadata_on_invalid_prior_identity_preserves_origin_until_repair(
    db_factory: Callable[[], CharactersRAGDB],
) -> None:
    """Unrelated metadata must remain writable; repairing an invalid binding clears origin."""
    db = db_factory()
    cid = db.add_conversation({"title": "Corrupt"})
    raw = encode_assistant_startup(AssistantStartup(source="explicit"))
    with db.transaction() as conn:
        conn.execute(
            "UPDATE conversations SET assistant_kind = 'persona', assistant_id = NULL, "
            "assistant_startup_json = ? WHERE id = ?",
            (raw, cid),
        )
    db.update_conversation(cid, {"title": "Still editable"}, 1)
    assert db.get_conversation_by_id(cid)[ORIGIN_COLUMN] == raw
    db.update_conversation(cid, {"assistant_id": "repaired"}, 2)
    assert db.get_conversation_by_id(cid)[ORIGIN_COLUMN] is None


@pytest.mark.parametrize(
    "kind,assistant_id,memory,retained",
    [
        (None, None, None, True),
        (None, "999", None, True),
        ("character", " ", None, True),
        (None, None, "read_only", False),
        ("persona", None, None, False),
        ("persona", " ", None, False),
    ],
)
def test_reopen_repairs_compare_normalized_binding(
    db_factory: Callable[[], CharactersRAGDB],
    kind: str | None,
    assistant_id: str | None,
    memory: str | None,
    retained: bool,
) -> None:
    """Historical Character repairs preserve equivalent origins and clear invalid bindings."""
    db = db_factory()
    character_id = db.add_character_card({"name": "Repair source"})
    cid = db.add_conversation({"character_id": character_id, "title": "Repair"})
    raw = encode_assistant_startup(AssistantStartup(source="explicit"))
    with db.transaction() as conn:
        conn.execute(
            "UPDATE conversations SET assistant_kind = ?, assistant_id = ?, persona_memory_mode = ?, "
            "assistant_startup_json = ? WHERE id = ?",
            (kind, assistant_id, memory, raw, cid),
        )
    db.close_all_connections()
    reopened = db_factory()
    row = reopened.get_conversation_by_id(cid)
    assert row["assistant_kind"] == (kind or "character")
    assert row["assistant_id"] == (assistant_id if assistant_id and assistant_id.strip() else str(character_id))
    assert row[ORIGIN_COLUMN] == (raw if retained else None)
    reopened.close_all_connections()
    assert db_factory().get_conversation_by_id(cid)[ORIGIN_COLUMN] == (raw if retained else None)


@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    st.lists(
        st.sampled_from(["title", "empty", "noop", "identity", "memory", "scope", "tombstone", "restore"]),
        min_size=1,
        max_size=25,
    )
)
def test_generated_lifecycle_never_restores_invalidated_origin(
    db_factory: Callable[[], CharactersRAGDB],
    operations: list[str],
) -> None:
    """Generated mutation sequences preserve history only until the first actual rebinding."""
    db = db_factory()
    cid = db.add_conversation(
        {"assistant_kind": "persona", "assistant_id": "a", "persona_memory_mode": "read_only"},
        assistant_startup=AssistantStartup(source="explicit"),
    )
    raw = db.get_conversation_by_id(cid)[ORIGIN_COLUMN]
    identity, memory, invalidated = "a", "read_only", False
    for operation in operations:
        row = db.get_conversation_by_id(cid, include_deleted=True)
        if operation == "tombstone":
            db.tombstone_conversation_from_sync(
                conversation_id=cid, sync_client_id="user-1", object_revision=row["version"] + 1, object_hash="h"
            )
        elif operation == "restore":
            if row["deleted"]:
                db.restore_conversation(cid, row["version"])
        elif not row["deleted"]:
            if operation == "scope":
                db.upsert_workspace("sequence-ws", "Sequence")
                db.upsert_conversation_from_sync(
                    conversation_id=cid,
                    title="Moved",
                    sync_client_id="user-1",
                    object_revision=row["version"],
                    object_hash="h",
                    assistant_kind="persona",
                    assistant_id=identity,
                    persona_memory_mode=memory,
                    scope_type="workspace",
                    workspace_id="sequence-ws",
                )
            else:
                update = {}
                if operation == "title":
                    update = {"title": "Renamed"}
                elif operation == "noop":
                    update = {"assistant_kind": " PERSONA ", "assistant_id": f" {identity} "}
                elif operation == "identity":
                    identity = "b" if identity == "a" else "a"
                    update, invalidated = {"assistant_id": identity}, True
                elif operation == "memory":
                    memory = "read_write" if memory == "read_only" else "read_only"
                    update, invalidated = {"persona_memory_mode": memory}, True
                db.update_conversation(cid, update, row["version"])
        assert db.get_conversation_by_id(cid, include_deleted=True)[ORIGIN_COLUMN] == (None if invalidated else raw)
    db.hard_delete_conversation(cid)
    assert db.get_conversation_by_id(cid, include_deleted=True) is None
    db.close_all_connections()


def test_workspace_cascade_keeps_history_on_soft_delete_and_moved_rows(
    db_factory: Callable[[], CharactersRAGDB],
) -> None:
    """Workspace deletion must not erase origin on a surviving historical conversation."""
    db = db_factory()
    workspace = db.upsert_workspace("cascade", "Cascade")
    origin = AssistantStartup(source="workspace_default", workspace_id="cascade", workspace_version=1)
    scoped = db.add_conversation({"scope_type": "workspace", "workspace_id": "cascade"}, assistant_startup=origin)
    moved = db.add_conversation({"title": "Already moved"}, assistant_startup=origin)
    db.delete_workspace("cascade", workspace["version"])
    assert decode_assistant_startup(db.get_conversation_by_id(scoped, include_deleted=True)[ORIGIN_COLUMN]) == origin
    db.hard_delete_workspace("cascade")
    assert db.get_conversation_by_id(scoped, include_deleted=True) is None
    assert decode_assistant_startup(db.get_conversation_by_id(moved)[ORIGIN_COLUMN]) == origin


def _wait_for_blocked_writer(conn: Any, pid: int, future: Future[Any]) -> None:
    """Observe PostgreSQL's lock graph rather than treating elapsed time as race evidence."""
    deadline = monotonic() + 10
    while monotonic() < deadline and not future.done():
        blockers = conn.execute("SELECT pg_blocking_pids(?) AS blockers", (pid,)).fetchone()["blockers"]
        if blockers:
            return
    pytest.fail("second writer never waited for the first transaction")


def _assert_blocked_writer(conn: Any, writer: CharactersRAGDB, operation: Callable[[], Any]) -> Any:
    """Prove a real second connection is blocked, then release the first transaction."""
    ready = Event()
    pid: list[int] = []

    def write() -> Any:
        """Bound the worker's SQL and expose its backend PID before entering the writer."""
        with writer.transaction() as worker_conn:
            worker_conn.execute("SET LOCAL statement_timeout = '15s'")
            pid.append(worker_conn.execute("SELECT pg_backend_pid() AS pid").fetchone()["pid"])
            ready.set()
            return operation()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(write)
        try:
            assert ready.wait(10), "second connection did not start"
            _wait_for_blocked_writer(conn, pid[0], future)
        finally:
            conn.commit()
        return future.result(timeout=20)


@pytest.mark.parametrize("db_factory", ["postgres"], indirect=True)
@pytest.mark.parametrize("same_binding", [True, False])
def test_postgres_insert_winner_is_compared_after_sync_wait(
    db_factory: Callable[[], CharactersRAGDB],
    same_binding: bool,
) -> None:
    """An absent-row Sync race must compare the committed trusted insertion winner."""
    first, second = db_factory(), db_factory()
    with first.transaction() as conn:
        cid = first.add_conversation(
            {"id": "insert-race", "assistant_kind": "persona", "assistant_id": "a"},
            conn=conn,
            assistant_startup=AssistantStartup(source="explicit"),
        )
        _assert_blocked_writer(
            conn,
            second,
            lambda: second.upsert_conversation_from_sync(
                conversation_id=cid,
                title="Remote",
                sync_client_id="user-1",
                object_revision=1,
                object_hash="h",
                assistant_kind="persona",
                assistant_id="a" if same_binding else "b",
            ),
        )
    row = first.get_conversation_by_id(cid)
    assert row["assistant_id"] == ("a" if same_binding else "b")
    assert decode_assistant_startup(row[ORIGIN_COLUMN]).source == ("explicit" if same_binding else "unknown")


@pytest.mark.parametrize("db_factory", ["postgres"], indirect=True)
@pytest.mark.parametrize("sync_first", [True, False])
def test_postgres_local_update_and_same_revision_sync_serialize_bindings(
    db_factory: Callable[[], CharactersRAGDB],
    sync_first: bool,
) -> None:
    """Both blocking orders compare current identity even when Sync keeps the revision."""
    first, second = db_factory(), db_factory()
    cid = first.add_conversation(
        {"assistant_kind": "persona", "assistant_id": "a", "persona_memory_mode": "read_only"},
        assistant_startup=AssistantStartup(source="explicit"),
    )

    def sync(db: CharactersRAGDB) -> bool:
        """Replace the whole identity without advancing the caller's observed revision."""
        return db.upsert_conversation_from_sync(
            conversation_id=cid,
            title="Remote",
            sync_client_id="user-1",
            object_revision=1,
            object_hash="h",
            assistant_kind="persona",
            assistant_id="b",
            persona_memory_mode="read_write",
        )

    with first.transaction() as conn:
        if sync_first:
            sync(first)
            _assert_blocked_writer(conn, second, lambda: second.update_conversation(cid, {"assistant_id": "a"}, 1))
        else:
            first.update_conversation(cid, {"assistant_id": "b"}, 1)
            _assert_blocked_writer(conn, second, lambda: sync(second))
    row = first.get_conversation_by_id(cid)
    assert (row["assistant_id"], row["persona_memory_mode"]) == ("a" if sync_first else "b", "read_write")
    assert row[ORIGIN_COLUMN] is None


@pytest.mark.parametrize("db_factory", ["postgres"], indirect=True)
def test_postgres_disappearing_sync_conflict_is_retryable_without_origin(
    db_factory: Callable[[], CharactersRAGDB],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deletion between insert conflict and lock must not fabricate a prior row or origin."""
    first, second = db_factory(), db_factory()
    cid = first.add_conversation({"id": "disappearing"}, assistant_startup=AssistantStartup(source="explicit_none"))
    execute = first.backend.execute
    deleted = Event()

    def delete_after_conflict(query: str, *args: Any, **kwargs: Any) -> Any:
        """Delete the actual conflicting row on another connection before its locking read."""
        result = execute(query, *args, **kwargs)
        if "INSERT INTO conversations" in query and result.rowcount == 0 and not deleted.is_set():
            second.hard_delete_conversation(cid)
            deleted.set()
        return result

    payload = {
        "conversation_id": cid,
        "title": "Retry",
        "sync_client_id": "user-1",
        "object_revision": 1,
        "object_hash": "h",
    }
    with monkeypatch.context() as racing:
        racing.setattr(first.backend, "execute", delete_after_conflict)
        with pytest.raises(ConflictError, match="retry"):
            first.upsert_conversation_from_sync(**payload)
    assert deleted.is_set()
    assert first.get_conversation_by_id(cid) is None
    first.upsert_conversation_from_sync(**payload)
    assert first.get_conversation_by_id(cid)[ORIGIN_COLUMN] is None


@pytest.mark.parametrize("db_factory", ["postgres"], indirect=True)
def test_postgres_sync_insert_winner_cannot_acquire_losing_local_origin(
    db_factory: Callable[[], CharactersRAGDB],
) -> None:
    """A trusted creator losing to a committed Sync insert must leave the remote row unknown."""
    first, second = db_factory(), db_factory()
    with first.transaction() as conn:
        first.upsert_conversation_from_sync(
            conversation_id="sync-winner",
            title="Remote",
            sync_client_id="user-1",
            object_revision=1,
            object_hash="h",
        )
        with pytest.raises((DatabaseError, CharactersRAGDBError)):
            _assert_blocked_writer(
                conn,
                second,
                lambda: second.add_conversation(
                    {"id": "sync-winner", "title": "Local"},
                    assistant_startup=AssistantStartup(source="explicit_none"),
                ),
            )
    row = first.get_conversation_by_id("sync-winner")
    assert row["title"] == "Remote"
    assert row[ORIGIN_COLUMN] is None


@pytest.mark.parametrize("db_factory", ["postgres"], indirect=True)
def test_postgres_sync_read_holds_lock_until_replacement_commits(
    db_factory: Callable[[], CharactersRAGDB],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local rebinding after the Sync read cannot be overwritten with stale retained origin."""
    sync_db, local_db = db_factory(), db_factory()
    cid = sync_db.add_conversation(
        {"assistant_kind": "persona", "assistant_id": "a"},
        assistant_startup=AssistantStartup(source="explicit"),
    )
    read_finished, release_read, local_ready = Event(), Event(), Event()
    local_pid: list[int] = []
    execute = sync_db.backend.execute

    def pause_after_sync_read(query: str, *args: Any, **kwargs: Any) -> Any:
        """Pause after the real locking read, before the real replacement UPDATE."""
        result = execute(query, *args, **kwargs)
        if "assistant_startup_json FROM conversations WHERE id" in query and not read_finished.is_set():
            read_finished.set()
            assert release_read.wait(15), "Sync read was not released"
        return result

    def replace() -> bool:
        """Run the actual Sync writer on its own connection."""
        return sync_db.upsert_conversation_from_sync(
            conversation_id=cid,
            title="Synced",
            sync_client_id="user-1",
            object_revision=1,
            object_hash="h",
            assistant_kind="persona",
            assistant_id="a",
        )

    def rebind() -> bool | None:
        """Bound the competing update and expose its PID to the read-only lock observer."""
        with local_db.transaction() as conn:
            conn.execute("SET LOCAL statement_timeout = '15s'")
            local_pid.append(conn.execute("SELECT pg_backend_pid() AS pid").fetchone()["pid"])
            local_ready.set()
            return local_db.update_conversation(cid, {"assistant_id": "b"}, 1)

    monkeypatch.setattr(sync_db.backend, "execute", pause_after_sync_read)
    with ThreadPoolExecutor(max_workers=2) as executor:
        synced = executor.submit(replace)
        try:
            assert read_finished.wait(10), "Sync did not reach its identity read"
            rebound = executor.submit(rebind)
            assert local_ready.wait(10), "local writer did not start"
            with sync_db.transaction() as observer:
                _wait_for_blocked_writer(observer, local_pid[0], rebound)
        finally:
            release_read.set()
        assert synced.result(timeout=20)
        assert rebound.result(timeout=20)
    row = sync_db.get_conversation_by_id(cid)
    assert row["assistant_id"] == "b"
    assert row[ORIGIN_COLUMN] is None


@pytest.mark.parametrize("sync", [False, True])
def test_sql_constraint_failure_rolls_back_identity_and_origin(
    db_factory: Callable[[], CharactersRAGDB], sync: bool
) -> None:
    """A rejected Character foreign key must roll back the same UPDATE that cleared origin."""
    db = db_factory()
    cid = db.add_conversation(
        {"assistant_kind": "persona", "assistant_id": "a"},
        assistant_startup=AssistantStartup(source="explicit"),
    )
    before = db.get_conversation_by_id(cid)
    with pytest.raises((DatabaseError, CharactersRAGDBError)):
        if sync:
            db.upsert_conversation_from_sync(
                conversation_id=cid,
                title="Rejected",
                sync_client_id="user-1",
                object_revision=1,
                object_hash="h",
                assistant_kind="character",
                character_id=999999999,
            )
        else:
            db.update_conversation(cid, {"assistant_kind": "character", "character_id": 999999999}, 1)
    assert db.get_conversation_by_id(cid) == before


def test_typed_keyword_rejects_unvalidated_dictionary(db_factory: Callable[[], CharactersRAGDB]) -> None:
    """The trusted keyword still requires a validated bounded model rather than raw JSON."""
    db = db_factory()
    with pytest.raises(TypeError):
        db.add_conversation({"id": "not-typed"}, assistant_startup={"source": "explicit"})
    assert db.get_conversation_by_id("not-typed") is None


def test_character_normalization_preserves_origin_for_redundant_id_changes(
    db_factory: Callable[[], CharactersRAGDB],
) -> None:
    """Character ID precedence makes a redundant assistant ID change an identity no-op."""
    db = db_factory()
    character = db.add_character_card({"name": "Canonical"})
    cid = db.add_conversation({"character_id": character}, assistant_startup=AssistantStartup(source="explicit"))
    before = db.get_conversation_by_id(cid)
    db.update_conversation(cid, {"assistant_id": "999", "assistant_kind": " CHARACTER "}, 1)
    after = db.get_conversation_by_id(cid)
    assert after["assistant_id"] == str(character)
    assert after[ORIGIN_COLUMN] == before[ORIGIN_COLUMN]


def test_sync_replacement_keeps_created_at_but_replaces_root_revision_and_deletion(
    db_factory: Callable[[], CharactersRAGDB],
) -> None:
    """The new conflict path remains whole-object replacement, not a metadata PATCH."""
    db = db_factory()
    cid = db.add_conversation(
        {"root_id": "old-root", "topic_label": "Old", "assistant_kind": "persona", "assistant_id": "a"},
        assistant_startup=AssistantStartup(source="explicit"),
    )
    before = db.get_conversation_by_id(cid)
    db.soft_delete_conversation(cid, 1)
    db.upsert_conversation_from_sync(
        conversation_id=cid,
        title="Replacement",
        sync_client_id="remote",
        object_revision=1,
        object_hash="h",
        assistant_kind="persona",
        assistant_id="a",
        state="archived",
    )
    row = db.get_conversation_by_id(cid)
    assert (row["root_id"], row["version"], row["deleted"], row["state"], row["topic_label"], row["client_id"]) == (
        cid,
        1,
        False,
        "resolved",
        None,
        "remote",
    )
    assert row["created_at"] == before["created_at"]
    assert row[ORIGIN_COLUMN] == before[ORIGIN_COLUMN]
