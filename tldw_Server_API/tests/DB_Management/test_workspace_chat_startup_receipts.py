"""Inactive receipt storage respects owner authority and caller transactions."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.tests.DB_Management.test_conversation_assistant_startup import _wait_for_blocked_writer
from tldw_Server_API.tests.DB_Management.test_conversation_assistant_startup import db_factory as db_factory

pytestmark = pytest.mark.integration


def _store(db: CharactersRAGDB) -> Any:
    """Fail on the missing storage contract, not an unrelated collection import."""
    assert hasattr(db, "workspace_chat_startups"), "Workspace receipt store must be installed"
    return db.workspace_chat_startups


def test_receipt_count_includes_invalidated_and_deleted_targets(db_factory: Callable[[], CharactersRAGDB]) -> None:
    """Tombstones permanently consume owner capacity, independently of writer attribution."""
    db = db_factory()
    store = _store(db)
    db.upsert_workspace("ws", "Workspace")
    cid = db.add_conversation({"scope_type": "workspace", "workspace_id": "ws"})
    with db.transaction() as conn:
        store.lock_owner("user-1", conn=conn)
        store.insert_receipt("user-1", "a" * 64, "b" * 64, "c" * 64, "ws", cid, conn=conn)
        assert store.has_receipt(cid, conn=conn)
        assert store.count_live_chats("user-1", "ws", conn=conn) == 1
        store.mark_hard_deleted(cid, conn=conn)
        receipt = store.get_receipt("user-1", "a" * 64, conn=conn, for_update=True)
        assert receipt["conversation_id"] is None
        assert store.count_receipts("user-1", conn=conn) == 1
        db.client_id = "different-device"
        assert store.count_receipts("user-1", conn=conn) == 1


def test_wrong_owner_rejected_before_sql(db_factory: Callable[[], CharactersRAGDB]) -> None:
    """Receipt helpers do not let a caller choose another owner's namespace."""
    db = db_factory()
    store = _store(db)
    with db.transaction() as conn:
        with pytest.raises(ValueError, match="workspace_chat_startup_owner_mismatch"):
            store.count_receipts("other", conn=conn)


def test_outermost_guard_preserves_managed_caller_transaction(db_factory: Callable[[], CharactersRAGDB]) -> None:
    """An outer caller remains able to commit its unrelated changes after rejection."""
    db = db_factory()
    store = _store(db)
    store.require_outermost()
    with db.transaction() as conn:
        cid = db.add_conversation({"title": "Caller work"}, conn=conn)
        with pytest.raises(ValueError, match="workspace_chat_startup_transaction_required"):
            store.require_outermost()
        assert conn.in_transaction
    store.require_outermost()
    assert db.get_conversation_by_id(cid)["title"] == "Caller work"


@pytest.mark.parametrize("db_factory", ["sqlite"], indirect=True)
def test_outermost_guard_does_not_probe_caller_connection(db_factory: Callable[[], CharactersRAGDB]) -> None:
    """Reject before SQL, including the connection getter's SQLite liveness probe."""
    db = db_factory()
    with db.transaction() as conn:
        statements: list[str] = []
        conn.set_trace_callback(statements.append)
        try:
            with pytest.raises(ValueError, match="workspace_chat_startup_transaction_required"):
                db.workspace_chat_startups.require_outermost()
            assert statements == []
        finally:
            conn.set_trace_callback(None)


@pytest.mark.parametrize("db_factory", ["postgres"], indirect=True)
def test_outermost_unallocated_shared_handle_does_not_refresh(
    db_factory: Callable[[], CharactersRAGDB], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pure preflight cannot resolve, bootstrap or acquire a changed shared target."""
    db = db_factory()
    expired = db.get_connection()
    db.close_connection()
    state = db._connection_state()
    assert getattr(state, "conn", None) is None
    assert getattr(state, "backend_ref", None) is None

    def unexpected_access(*args: Any, **kwargs: Any) -> Any:
        """Make any refreshing accessor fail before it could initialize storage."""
        pytest.fail("Outermost preflight must not refresh or initialize a backend")

    with monkeypatch.context() as patch:
        patch.setattr(db, "_uses_shared_content_backend", True)
        patch.setattr(db, "_backend_refresh_suspended", False)
        for method in ("_resolve_backend", "_ensure_bootstrap_for_backend", "_initialize_schema", "get_connection"):
            patch.setattr(db, method, unexpected_access)
        db.workspace_chat_startups.require_outermost()
        with pytest.raises(ValueError, match="workspace_chat_startup_transaction_required"):
            db.workspace_chat_startups.get_receipt("user-1", "a" * 64, conn=expired)


def test_workspace_lock_returns_closure_and_system_state(db_factory: Callable[[], CharactersRAGDB]) -> None:
    """Strict lookup can classify closure without leaking another tenant's Workspace."""
    db = db_factory()
    store = _store(db)
    db.upsert_workspace("ws", "Workspace")
    with db.transaction() as conn:
        conn.execute("UPDATE workspaces SET native_chat_admission_closed = ? WHERE id = ?", (True, "ws"))
        row = store.lock_receipt_workspace("ws", conn=conn)
        assert row["native_chat_admission_closed"]
        assert store.lock_receipt_workspace("missing", conn=conn) is None


def test_binding_digest_uses_normalized_identity_not_metadata(db_factory: Callable[[], CharactersRAGDB]) -> None:
    """Existing canonical identity/scope rules govern immutable receipt bindings."""
    db = db_factory()
    _store(db)
    module = importlib.import_module("tldw_Server_API.app.core.DB_Management.chacha.workspace_chat_startup_store")
    row = {
        "assistant_kind": None,
        "assistant_id": None,
        "character_id": None,
        "persona_memory_mode": None,
        "assistant_startup_json": None,
        "scope_type": "workspace",
        "workspace_id": "ws",
    }
    assert module.startup_binding_digest(db, row) == module.startup_binding_digest(
        db, dict(row, title="Edited", state="archived", client_id="device", version=8)
    )
    assert module.startup_binding_digest(db, row) != module.startup_binding_digest(db, dict(row, workspace_id="other"))


@pytest.mark.parametrize(
    "bad", [{"assistant_kind": "persona"}, {"scope_type": "bad"}, {"assistant_startup_json": "{}oops"}]
)
def test_corrupt_binding_is_a_bounded_changed_error(
    db_factory: Callable[[], CharactersRAGDB], bad: dict[str, Any]
) -> None:
    """Replay identity does not use the display projection's corrupt-origin fallback."""
    db = db_factory()
    module = importlib.import_module("tldw_Server_API.app.core.DB_Management.chacha.workspace_chat_startup_store")
    with pytest.raises(module.WorkspaceStartupError) as caught:
        module.startup_binding_digest(db, {"scope_type": "workspace", "workspace_id": "ws", **bad})
    assert caught.value.code == "workspace_chat_startup_changed"


def test_invalidation_noop_permanence_and_rollback(db_factory: Callable[[], CharactersRAGDB]) -> None:
    """No-op preserves receipt, actual change invalidates once, rollback restores both."""
    db = db_factory()
    store = _store(db)
    db.upsert_workspace("ws", "Workspace")
    cid = db.add_conversation({"scope_type": "workspace", "workspace_id": "ws"})
    module = importlib.import_module("tldw_Server_API.app.core.DB_Management.chacha.workspace_chat_startup_store")
    with db.transaction() as conn:
        before = store.lock_conversation(cid, conn=conn)
        store.insert_receipt(
            "user-1", "a" * 64, "b" * 64, module.startup_binding_digest(db, before), "ws", cid, conn=conn
        )
        store.invalidate_changed_binding(cid, before, dict(before, title="Edited"), conn=conn)
        assert store.get_receipt("user-1", "a" * 64, conn=conn)["invalidated_at"] is None
    with pytest.raises(RuntimeError, match="rollback"):
        with db.transaction() as conn:
            store.invalidate_changed_binding(cid, before, dict(before, workspace_id="other"), conn=conn)
            raise RuntimeError("rollback")
    with db.transaction() as conn:
        assert store.get_receipt("user-1", "a" * 64, conn=conn)["invalidated_at"] is None
        store.invalidate_changed_binding(cid, before, dict(before, workspace_id="other"), conn=conn)
        timestamp = store.get_receipt("user-1", "a" * 64, conn=conn)["invalidated_at"]
        assert timestamp is not None
        store.invalidate_changed_binding(cid, dict(before, workspace_id="other"), before, conn=conn)
        assert store.get_receipt("user-1", "a" * 64, conn=conn)["invalidated_at"] == timestamp
        assert store.count_receipts("user-1", conn=conn) == 1


def test_helpers_do_not_open_their_own_transaction(db_factory: Callable[[], CharactersRAGDB]) -> None:
    """Lock/read/write primitives reject an idle connection instead of implicit reads."""
    db = db_factory()
    store = _store(db)
    conn = db.get_connection()
    for operation in (
        lambda: store.get_receipt("user-1", "a" * 64, conn=conn),
        lambda: store.lock_conversation("missing", conn=conn),
        lambda: store.count_receipts("user-1", conn=conn),
        lambda: store.lock_owner("user-1", conn=conn),
        lambda: store.lock_receipt_workspace("missing", conn=conn),
        lambda: store.has_receipt("missing", conn=conn),
        lambda: store.count_live_chats("user-1", "missing", conn=conn),
        lambda: store.insert_receipt("user-1", "a" * 64, "b" * 64, "c" * 64, "ws", "missing", conn=conn),
        lambda: store.mark_hard_deleted("missing", conn=conn),
        lambda: store.invalidate_changed_binding("missing", {}, {}, conn=conn),
    ):
        with pytest.raises(ValueError, match="workspace_chat_startup_transaction_required"):
            operation()
        if db.backend_type.value == "postgresql":
            assert conn._connection.info.transaction_status.name == "IDLE"
        else:
            assert not conn.in_transaction


@pytest.mark.parametrize("db_factory", ["postgres"], indirect=True)
def test_postgres_driver_transaction_is_preserved(db_factory: Callable[[], CharactersRAGDB]) -> None:
    """A raw driver-open transaction is caller-owned even at wrapper depth zero."""
    db = db_factory()
    store = _store(db)
    conn = db.get_connection()
    conn.execute("SELECT 1")
    with pytest.raises(ValueError, match="workspace_chat_startup_transaction_required"):
        store.require_outermost()
    assert conn.in_transaction
    conn.rollback()
    store.require_outermost()


@pytest.mark.parametrize("db_factory", ["postgres"], indirect=True)
def test_postgres_failed_transaction_is_rejected_without_settling(db_factory: Callable[[], CharactersRAGDB]) -> None:
    """A failed managed transaction is not authority to issue another receipt query."""
    db = db_factory()
    with db.transaction() as conn:
        with pytest.raises(DatabaseError):
            conn.execute("SELECT 1 / 0")
        assert conn._connection.info.transaction_status.name == "INERROR"
        with pytest.raises(ValueError, match="workspace_chat_startup_transaction_required"):
            db.workspace_chat_startups.count_receipts("user-1", conn=conn)
        assert conn._connection.info.transaction_status.name == "INERROR"


@pytest.mark.parametrize("db_factory", ["postgres"], indirect=True)
def test_postgres_autocommit_is_not_a_receipt_transaction(db_factory: Callable[[], CharactersRAGDB]) -> None:
    """A wrapper context cannot make autocommitted advisory locks transaction-local."""
    db = db_factory()
    raw = db.get_connection()._connection
    raw.autocommit = True
    try:
        with db.transaction() as conn:
            with pytest.raises(ValueError, match="workspace_chat_startup_transaction_required"):
                db.workspace_chat_startups.lock_owner("user-1", conn=conn)
        assert raw.autocommit
        assert raw.info.transaction_status.name == "IDLE"
    finally:
        raw.autocommit = False


@pytest.mark.parametrize("db_factory", ["postgres"], indirect=True)
def test_postgres_other_owner_lock_is_independent(db_factory: Callable[[], CharactersRAGDB]) -> None:
    """A different owner's acceptance lock completes while the first remains held."""
    first = db_factory()
    other = CharactersRAGDB(
        first.db_path_str, client_id="user-2", backend=DatabaseBackendFactory.create_backend(first.backend.config)
    )
    first.upsert_workspace("ws-1", "First owner")
    other.upsert_workspace("ws-2", "Other owner")
    first_cid = first.add_conversation({"scope_type": "workspace", "workspace_id": "ws-1"})
    other_cid = other.add_conversation({"scope_type": "workspace", "workspace_id": "ws-2"})

    def contender() -> None:
        """Acquire another owner namespace on a separate officially provisioned handle."""
        with other.transaction() as conn:
            conn.execute("SET LOCAL statement_timeout = '10s'")
            other.workspace_chat_startups.lock_owner("user-2", conn=conn)
            other.workspace_chat_startups.insert_receipt(
                "user-2", "a" * 64, "b" * 64, "c" * 64, "ws-2", other_cid, conn=conn
            )

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            with first.transaction() as conn:
                first.workspace_chat_startups.lock_owner("user-1", conn=conn)
                first.workspace_chat_startups.insert_receipt(
                    "user-1", "a" * 64, "b" * 64, "c" * 64, "ws-1", first_cid, conn=conn
                )
                executor.submit(contender).result(timeout=15)
                assert first.workspace_chat_startups.count_receipts("user-1", conn=conn) == 1
        with other.transaction() as conn:
            assert other.workspace_chat_startups.count_receipts("user-2", conn=conn) == 1
    finally:
        other.close_all_connections()


@pytest.mark.parametrize("db_factory", ["postgres"], indirect=True)
def test_postgres_owner_lock_blocks_same_owner_only(db_factory: Callable[[], CharactersRAGDB]) -> None:
    """Actual backend lock graph proves same-owner serialization across handles."""
    first, second = db_factory(), db_factory()
    ready = Event()
    pid: list[int] = []

    def contender() -> None:
        """Publish an initialized handle before entering the tested advisory lock."""
        with second.transaction() as conn:
            conn.execute("SET LOCAL statement_timeout = '15s'")
            pid.append(conn.execute("SELECT pg_backend_pid() AS pid").fetchone()["pid"])
            ready.set()
            second.workspace_chat_startups.lock_owner("user-1", conn=conn)

    with ThreadPoolExecutor(max_workers=1) as executor:
        with first.transaction() as conn:
            first.workspace_chat_startups.lock_owner("user-1", conn=conn)
            future = executor.submit(contender)
            try:
                assert ready.wait(10)
                _wait_for_blocked_writer(conn, pid[0], future)
            finally:
                conn.commit()
        future.result(timeout=20)
