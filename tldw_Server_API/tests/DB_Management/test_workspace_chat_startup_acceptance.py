"""Strict startup accepts and replays one committed owner-bound Workspace chat."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.schemas.workspace_chat_startup_schemas import WorkspaceChatStartupRequest
from tldw_Server_API.app.core.Chat.assistant_startup import decode_assistant_startup
from tldw_Server_API.app.core.DB_Management.chacha.workspace_chat_startup_store import WorkspaceStartupError
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_Server_API.tests.DB_Management.test_conversation_assistant_startup import db_factory as db_factory
from tldw_Server_API.tests.DB_Management.test_workspace_assistant_creation_atomic import creation_db as creation_db

pytestmark = pytest.mark.integration


def _start(db: CharactersRAGDB, *, payload: dict[str, Any] | None = None, **options: Any) -> Any:
    """Exercise the orchestrator without making a missing module a collection error."""
    module = "tldw_Server_API.app.core.Workspaces.chat_startup"
    assert importlib.util.find_spec(module) is not None, "Strict Workspace startup orchestrator must exist"
    request = WorkspaceChatStartupRequest.model_validate(
        payload
        or {
            "scope_type": "workspace",
            "workspace_id": "ws",
            "workspace_assistant_selection": "inherit",
            "workspace_assistant_default_version": 2,
        }
    )
    kwargs = {
        "owner_id": "user-1",
        "idempotency_key": "accepted-1",
        "receipt_limit": 1,
        "chat_limit": None,
        "title_timestamp": "first",
        **options,
    }
    return importlib.import_module(module).start_workspace_chat(db, request=request, **kwargs)


def _counts(db: CharactersRAGDB) -> tuple[int, int]:
    """Inspect both records in one explicitly owned observer read."""
    with db.transaction() as conn:
        return (
            db.workspace_chat_startups.count_receipts("user-1", conn=conn),
            db.workspace_chat_startups.count_live_chats("user-1", "ws", conn=conn),
        )


def test_accepted_replay_at_capacity_keeps_original_chat(creation_db: CharactersRAGDB) -> None:
    """Lifetime capacity, exhausted live quota and a changed clock cannot affect replay."""
    first = _start(creation_db)
    replay = _start(creation_db, chat_limit=0, title_timestamp="different")
    assert first.replayed is False and replay.replayed is True
    assert (replay.conversation["id"], replay.conversation["title"]) == (first.conversation["id"], "First Chat (first)")
    assert _counts(creation_db) == (1, 1)


def test_replay_does_not_resolve_current_default(creation_db: CharactersRAGDB) -> None:
    """An accepted key binds the original identity/version, not today's default."""
    first = _start(creation_db)
    with creation_db.transaction():
        creation_db.update_workspace("ws", {"assistant_defaults_json": None}, 2)
    replay = _start(creation_db)
    assert replay.conversation["assistant_id"] == "persona-a"
    assert decode_assistant_startup(replay.conversation["assistant_startup_json"]).workspace_version == 2
    assert replay.conversation["id"] == first.conversation["id"]


def test_none_selection_never_resolves_default(creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit None remains available even when Persona/default lookup cannot run."""
    from tldw_Server_API.app.core import feature_flags

    monkeypatch.setattr(feature_flags, "is_persona_enabled", lambda: False)
    first = _start(
        creation_db,
        payload={
            "scope_type": "workspace",
            "workspace_id": "ws",
            "workspace_assistant_selection": "none",
        },
    )
    assert first.conversation["assistant_id"] is None
    assert decode_assistant_startup(first.conversation["assistant_startup_json"]).source == "explicit_none"


@pytest.mark.parametrize("change", [{"title": None}, {"workspace_id": "other"}])
def test_key_request_conflict_does_not_create(creation_db: CharactersRAGDB, change: dict[str, Any]) -> None:
    """Presence changes and cross-Workspace key reuse cannot create a second receipt."""
    _start(creation_db)
    with creation_db.transaction():
        creation_db.upsert_workspace("other", "Other")
    with pytest.raises(WorkspaceStartupError) as caught:
        _start(
            creation_db,
            payload={
                "scope_type": "workspace",
                "workspace_id": "ws",
                "workspace_assistant_selection": "inherit",
                "workspace_assistant_default_version": 2,
                **change,
            },
        )
    assert caught.value.code == "idempotency_key_conflict"
    assert _counts(creation_db) == (1, 1)


@pytest.mark.parametrize("gate", ["version", "capacity", "quota", "archive", "closure", "system", "access"])
def test_new_acceptance_rejection_is_atomic(creation_db: CharactersRAGDB, gate: str) -> None:
    """Version, lifecycle, capacity and quota failures leave no partially accepted chat."""
    options: dict[str, Any] = {}
    codes = {
        "version": "workspace_assistant_version_conflict",
        "capacity": "workspace_chat_receipt_capacity_exceeded",
        "quota": "workspace_chat_quota_exceeded",
        "archive": "workspace_archived",
        "closure": "workspace_chat_admission_closed",
        "system": "workspace_chat_admission_closed",
        "access": "workspace_not_found",
    }
    if gate == "capacity":
        _start(creation_db)
        options["idempotency_key"] = "unseen"
    elif gate == "quota":
        options["chat_limit"] = 0
    else:
        updates = {
            "version": ("version", 3),
            "archive": ("archived", True),
            "closure": ("native_chat_admission_closed", True),
            "system": ("system_operation_state", "staged"),
            "access": ("client_id", "other"),
        }
        column, value = updates[gate]
        with creation_db.transaction() as conn:
            # Column comes only from the fixed test-owned schema map above.
            conn.execute(f"UPDATE workspaces SET {column} = ? WHERE id = ?", (value, "ws"))  # nosec B608
    with pytest.raises(WorkspaceStartupError) as caught:
        _start(creation_db, **options)
    assert caught.value.code == codes[gate]
    assert _counts(creation_db) == ((1, 1) if gate == "capacity" else (0, 0))


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("rollback", [False, True])
def test_nested_startup_preserves_caller_work(
    creation_db: CharactersRAGDB, db_factory: Callable[[], CharactersRAGDB], replay: bool, rollback: bool
) -> None:
    """A strict call must not commit, roll back or savepoint inside its caller."""
    if replay:
        _start(creation_db)
    observer = db_factory()
    try:
        with creation_db.transaction() as conn:
            # Keep the caller-write probe independent of conversation FTS setup.
            conn.execute("UPDATE workspaces SET name = ? WHERE id = ?", ("Caller sentinel", "ws"))
            with pytest.raises(WorkspaceStartupError) as caught:
                _start(creation_db)
            assert caught.value.code == "workspace_chat_startup_transaction_required"
            assert creation_db.workspace_chat_startups.count_receipts("user-1", conn=conn) == int(replay)
            if rollback:
                raise RuntimeError("caller rolls back")
    except RuntimeError as error:
        assert rollback and str(error) == "caller rolls back"
    with observer.transaction():
        row = observer.get_workspace("ws")
        assert row["name"] == ("Workspace" if rollback else "Caller sentinel")


@pytest.mark.parametrize("after", [False, True])
def test_failed_receipt_insert_rolls_back_chat_and_receipt(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, after: bool
) -> None:
    """A failure at either side of real receipt insertion rolls back the entire acceptance."""
    insert = creation_db.workspace_chat_startups.insert_receipt

    def fail(*args: Any, **kwargs: Any) -> None:
        """Fault injection retains real database work before the selected failure boundary."""
        if after:
            insert(*args, **kwargs)
        raise RuntimeError("receipt insertion failed")

    monkeypatch.setattr(creation_db.workspace_chat_startups, "insert_receipt", fail)
    with pytest.raises(RuntimeError, match="receipt insertion failed"):
        _start(creation_db)
    assert _counts(creation_db) == (0, 0)


@pytest.mark.parametrize(
    "options",
    [
        {"receipt_limit": 0},
        {"receipt_limit": -1},
        {"receipt_limit": True},
        {"receipt_limit": "1"},
        {"chat_limit": -1},
        {"chat_limit": True},
        {"chat_limit": "1"},
    ],
)
def test_invalid_configuration_rejects_before_transaction(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, options: dict[str, Any]
) -> None:
    """Misconfiguration fails closed without reads, writes or silent capacity relaxation."""

    def unexpected_transaction() -> Any:
        """Configuration errors must not reach storage."""
        pytest.fail("No transaction is permitted for invalid configuration")

    with monkeypatch.context() as patch:
        patch.setattr(creation_db, "transaction", unexpected_transaction)
        with pytest.raises(WorkspaceStartupError) as caught:
            _start(creation_db, **options)
    assert (caught.value.status_code, caught.value.code) == (503, "workspace_chat_startup_configuration_invalid")
    assert _counts(creation_db) == (0, 0)


@pytest.mark.parametrize("key", ["", "_first", "private space", "private\u00e9", "a" * 129, None, True])
def test_invalid_key_rejects_without_acceptance(creation_db: CharactersRAGDB, key: Any) -> None:
    """The internal entry point enforces the same bounded ASCII key contract."""
    with pytest.raises(WorkspaceStartupError) as caught:
        _start(creation_db, idempotency_key=key)
    assert (caught.value.status_code, caught.value.code) == (422, "invalid_idempotency_key")
    assert _counts(creation_db) == (0, 0)


def test_return_means_committed_even_after_response_loss(
    creation_db: CharactersRAGDB, db_factory: Callable[[], CharactersRAGDB]
) -> None:
    """A preinitialized observer sees both records before any response/retry processing."""
    observer = db_factory()
    first = _start(creation_db, idempotency_key="a" * 128)
    assert _counts(observer) == (1, 1)
    with observer.transaction():
        assert observer.get_conversation_by_id(first.conversation["id"])["assistant_id"] == "persona-a"
    replay = _start(observer, idempotency_key="a" * 128)
    assert (replay.replayed, replay.conversation["id"]) == (True, first.conversation["id"])
    assert _counts(creation_db) == (1, 1)


@pytest.mark.parametrize("replay", [False, True])
def test_commit_failure_cannot_return_acceptance(
    creation_db: CharactersRAGDB, db_factory: Callable[[], CharactersRAGDB], monkeypatch: pytest.MonkeyPatch, replay: bool
) -> None:
    """Inject failure in the real context manager's commit, retaining its real rollback."""
    if replay:
        _start(creation_db)
    observer = db_factory()
    transaction = creation_db.transaction

    class CommitFailure:
        """Delegate driver operations except the physical commit boundary."""

        def __init__(self, raw: Any) -> None:
            """Retain the real driver for every non-commit operation."""
            self.raw = raw

        def commit(self) -> None:
            """Simulate an unsuccessful commit before durable acceptance."""
            raise sqlite3.OperationalError("injected commit failure")

        def __getattr__(self, name: str) -> Any:
            """Preserve the actual driver's rollback and state inspection."""
            return getattr(self.raw, name)

    @contextmanager
    def failing_transaction() -> Iterator[Any]:
        """Leave all work on the actual driver and fault only context exit."""
        context = transaction()
        with context as conn:
            yield conn
            attribute = "_raw_conn" if creation_db.backend_type.value == "postgresql" else "conn"
            setattr(context, attribute, CommitFailure(getattr(context, attribute)))

    with monkeypatch.context() as patch:
        patch.setattr(creation_db, "transaction", failing_transaction)
        with pytest.raises(CharactersRAGDBError, match="commit fail"):
            _start(creation_db)
    assert _counts(observer) == ((1, 1) if replay else (0, 0))
    assert _start(creation_db).replayed is replay


@pytest.mark.parametrize("replay", [False, True])
def test_driver_open_caller_transaction_is_preserved(creation_db: CharactersRAGDB, replay: bool) -> None:
    """Wrapper-depth zero does not grant ownership over a driver-open caller unit."""
    if replay:
        _start(creation_db)
    conn = creation_db.get_connection()
    if creation_db.backend_type.value != "postgresql":
        conn.execute("BEGIN")
    conn.execute("INSERT INTO conversations (id, root_id, title, client_id) VALUES (?, ?, ?, ?)",
                 ("caller", "caller", "Sentinel", "user-1"))
    with pytest.raises(WorkspaceStartupError) as caught:
        _start(creation_db)
    assert caught.value.code == "workspace_chat_startup_transaction_required"
    if creation_db.backend_type.value == "postgresql":
        assert conn._connection.info.transaction_status.name == "INTRANS"
    else:
        assert conn.in_transaction
    assert conn.execute("SELECT title FROM conversations WHERE id = ?", ("caller",)).fetchone()["title"] == "Sentinel"
    conn.rollback()
    assert _counts(creation_db) == ((1, 1) if replay else (0, 0))


@pytest.mark.parametrize("gate", ["inactive", "deleted", "disabled"])
def test_replay_requires_original_current_persona(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, gate: str
) -> None:
    """Revocation cannot replace the original Persona with a newly selected default."""
    from tldw_Server_API.app.core import feature_flags

    _start(creation_db)
    with creation_db.transaction() as conn:
        creation_db.update_workspace("ws", {"assistant_defaults_json": None}, 2)
        if gate == "inactive":
            conn.execute("UPDATE persona_profiles SET is_active = ? WHERE id = ?", (False, "persona-a"))
        elif gate == "deleted":
            conn.execute("UPDATE persona_profiles SET deleted = ? WHERE id = ?", (True, "persona-a"))
    if gate == "disabled":
        monkeypatch.setattr(feature_flags, "is_persona_enabled", lambda: False)
    expected = {
        "inactive": (409, "persona_unavailable"), "deleted": (404, "persona_not_found"),
        "disabled": (503, "persona_feature_disabled"),
    }[gate]
    with pytest.raises(WorkspaceStartupError) as caught:
        _start(creation_db)
    assert (caught.value.status_code, caught.value.code) == expected
    assert _counts(creation_db) == (1, 1)


def test_replay_returns_current_metadata_and_allows_archive(creation_db: CharactersRAGDB) -> None:
    """Title edits and archival do not change the accepted binding or recreate it."""
    first = _start(creation_db)
    with creation_db.transaction():
        creation_db.update_conversation(first.conversation["id"], {"title": "Edited"}, first.conversation["version"])
        creation_db.update_workspace("ws", {"archived": True}, 2)
    replay = _start(creation_db)
    assert (replay.replayed, replay.conversation["id"], replay.conversation["title"]) == (
        True, first.conversation["id"], "Edited",
    )
    with pytest.raises(WorkspaceStartupError) as caught:
        _start(creation_db, idempotency_key="new")
    assert caught.value.code == "workspace_archived"
    assert _counts(creation_db) == (1, 1)


@pytest.mark.parametrize("hard", [False, True])
def test_deleted_chat_precedes_revocation_and_never_recreates(creation_db: CharactersRAGDB, hard: bool) -> None:
    """Deletion is 410 even when current profile admission would also fail."""
    first = _start(creation_db)
    with creation_db.transaction() as conn:
        conn.execute("UPDATE persona_profiles SET is_active = ? WHERE id = ?", (False, "persona-a"))
        if hard:
            creation_db.hard_delete_conversation(first.conversation["id"])
        else:
            creation_db.soft_delete_conversation(first.conversation["id"], first.conversation["version"])
    with pytest.raises(WorkspaceStartupError) as caught:
        _start(creation_db)
    assert (caught.value.status_code, caught.value.code) == (410, "workspace_chat_deleted")
    assert _counts(creation_db) == (1, 0)


@pytest.mark.parametrize("gate", ["access", "deleted", "closing", "system"])
def test_workspace_authority_precedes_receipt_target_disclosure(creation_db: CharactersRAGDB, gate: str) -> None:
    """Inaccessible/closed Workspace errors take precedence over a deleted target."""
    first = _start(creation_db)
    column, value = {
        "access": ("client_id", "other"), "deleted": ("deleted", True),
        "closing": ("native_chat_admission_closed", True), "system": ("system_operation_state", "staged"),
    }[gate]
    with creation_db.transaction() as conn:
        creation_db.soft_delete_conversation(first.conversation["id"], first.conversation["version"])
        # Column comes only from the fixed test-owned schema map above.
        conn.execute(f"UPDATE workspaces SET {column} = ? WHERE id = ?", (value, "ws"))  # nosec B608
    with pytest.raises(WorkspaceStartupError) as caught:
        _start(creation_db)
    assert (caught.value.status_code, caught.value.code) == (
        (404, "workspace_not_found") if gate in {"access", "deleted"} else (409, "workspace_chat_admission_closed")
    )


@pytest.mark.parametrize("hidden_reads", [1, 2, 3, 4])
def test_discovered_winner_replays_after_owned_rollback(
    creation_db: CharactersRAGDB, db_factory: Callable[[], CharactersRAGDB],
    monkeypatch: pytest.MonkeyPatch, hidden_reads: int,
) -> None:
    """Post-lock discovery or a real unique violation settles failed work before replay."""
    observer = db_factory()
    first = _start(creation_db)
    getter = creation_db.workspace_chat_startups.get_receipt
    calls: list[Any] = []
    transaction = creation_db.transaction
    entered: list[Any] = []
    insert = creation_db.workspace_chat_startups.insert_receipt
    insertions: list[str] = []

    def hide_winner(*args: Any, **kwargs: Any) -> Any:
        """Inject stale preflight visibility, never fabricate the winning row."""
        row = getter(*args, **kwargs)
        calls.append(row)
        return None if len(calls) <= hidden_reads else row

    @contextmanager
    def record_transaction() -> Iterator[Any]:
        """Record real context entries without changing commit or rollback behavior."""
        with transaction() as conn:
            entered.append(conn)
            yield conn

    def record_insert(*args: Any, **kwargs: Any) -> None:
        """Keep the real unique violation rather than fabricating its exception."""
        insertions.append(args[5])
        insert(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(creation_db.workspace_chat_startups, "get_receipt", hide_winner)
        patch.setattr(creation_db, "transaction", record_transaction)
        patch.setattr(creation_db.workspace_chat_startups, "insert_receipt", record_insert)
        replay = _start(creation_db, receipt_limit=2)
    assert (replay.replayed, replay.conversation["id"]) == (True, first.conversation["id"])
    assert len(entered) == 2
    assert len(insertions) == (1 if hidden_reads == 4 else 0)
    assert _counts(observer) == (1, 1)


@pytest.mark.parametrize("failure_kind", ["http", "storage"])
def test_untyped_resolver_errors_propagate_without_acceptance(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, failure_kind: str
) -> None:
    """Generic exceptions cannot be parsed into a bounded default-unavailability reason."""
    from fastapi import HTTPException

    from tldw_Server_API.app.core.Workspaces import chat_startup

    error = HTTPException(status_code=409, detail="private failure") if failure_kind == "http" else RuntimeError("storage")

    def fail(*args: Any, **kwargs: Any) -> Any:
        """Fault only the resolver; leave owned storage/rollback in use."""
        raise error

    monkeypatch.setattr(chat_startup, "resolve_workspace_assistant_startup", fail)
    with pytest.raises(type(error)) as caught:
        _start(creation_db)
    assert caught.value is error
    assert _counts(creation_db) == (0, 0)


@pytest.mark.parametrize(
    "reason", ["persona_deleted", "persona_unavailable", "persona_feature_disabled",
               "permission_denied", "invalid_default", "unsupported_assistant_kind"],
)
def test_typed_default_failure_has_exact_strict_mapping(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    """Strict translation catches the real resolver's bounded type, never its message."""
    from tldw_Server_API.app.api.v1.schemas.workspace_schemas import WorkspaceEffectiveAssistantDefault
    from tldw_Server_API.app.core import feature_flags
    from tldw_Server_API.app.core.Workspaces import assistant_defaults

    version = 2
    with creation_db.transaction() as conn:
        if reason == "persona_deleted":
            creation_db.soft_delete_persona_profile(persona_id="persona-a", user_id="user-1", expected_version=1)
        elif reason == "persona_unavailable":
            conn.execute("UPDATE persona_profiles SET is_active = ? WHERE id = ?", (False, "persona-a"))
        elif reason in {"permission_denied", "invalid_default"}:
            creation_db.update_workspace("ws", {"assistant_defaults_json": {
                "assistant_kind": "persona", "assistant_id": "" if reason == "invalid_default" else "private-missing",
            }}, 2)
            version = 3
    if reason == "persona_feature_disabled":
        monkeypatch.setattr(feature_flags, "is_persona_enabled", lambda: False)
    elif reason == "unsupported_assistant_kind":
        # Storage admits only Persona; use the existing schema-valid future-kind result.
        monkeypatch.setattr(assistant_defaults, "resolve_effective_workspace_assistant_default",
                            lambda *args, **kwargs: WorkspaceEffectiveAssistantDefault(
                                status="unavailable", source="workspace", degraded_reason="unsupported_assistant_kind",
                            ))
    with pytest.raises(WorkspaceStartupError) as caught:
        _start(creation_db, payload={
            "scope_type": "workspace", "workspace_id": "ws", "workspace_assistant_selection": "inherit",
            "workspace_assistant_default_version": version,
        })
    assert (caught.value.status_code, caught.value.code, caught.value.reason) == (
        (503, "persona_feature_disabled", reason) if reason == "persona_feature_disabled" else
        (409, "workspace_assistant_unavailable", reason)
    )
    assert _counts(creation_db) == (0, 0)


@pytest.mark.parametrize("mutation", ["binding", "scope", "invalidated", "malformed"])
def test_replay_rejects_changed_or_invalid_binding(creation_db: CharactersRAGDB, mutation: str) -> None:
    """Digest/current tombstone authority cannot be replaced by a newly valid identity."""
    first = _start(creation_db)
    with creation_db.transaction() as conn:
        if mutation == "binding":
            creation_db.update_conversation(first.conversation["id"], {"assistant_id": "persona-b"},
                                            first.conversation["version"])
        elif mutation == "scope":
            creation_db.upsert_conversation_from_sync(
                conversation_id=first.conversation["id"], title="Moved", sync_client_id="user-1",
                object_revision=1, object_hash="ignored", scope_type="global",
                assistant_kind="persona", assistant_id="persona-a", persona_memory_mode="read_only",
            )
            assert creation_db.get_conversation_by_id(first.conversation["id"])["scope_type"] == "global"
        elif mutation == "invalidated":
            conn.execute("UPDATE workspace_chat_startup_receipts SET invalidated_at = ? WHERE conversation_id = ?",
                         ("2026-09-27T00:00:00Z", first.conversation["id"]))
        else:
            conn.execute("UPDATE conversations SET assistant_startup_json = ? WHERE id = ?",
                         ("not-json", first.conversation["id"]))
    with pytest.raises(WorkspaceStartupError) as caught:
        _start(creation_db)
    assert (caught.value.status_code, caught.value.code) == (409, "workspace_chat_startup_changed")


def test_create_delete_cycles_do_not_recycle_lifetime_capacity(creation_db: CharactersRAGDB) -> None:
    """Deleted conversations free live-chat quota, not the permanent key budget."""
    for index in range(2):
        accepted = _start(creation_db, idempotency_key=f"key-{index}", receipt_limit=2, chat_limit=1)
        with creation_db.transaction():
            creation_db.hard_delete_conversation(accepted.conversation["id"])
    assert _counts(creation_db) == (2, 0)
    with pytest.raises(WorkspaceStartupError) as caught:
        _start(creation_db, idempotency_key="unseen", receipt_limit=2, chat_limit=1)
    assert caught.value.code == "workspace_chat_receipt_capacity_exceeded"
    assert _counts(creation_db) == (2, 0)


def test_owner_mismatch_precedes_all_storage(creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch) -> None:
    """Writer attribution cannot replace immutable ownership at this internal boundary."""
    creation_db.client_id = "other"

    def unexpected() -> Any:
        """An unauthorized namespace must never enter a transaction."""
        pytest.fail("Owner mismatch must not read or write")

    monkeypatch.setattr(creation_db, "transaction", unexpected)
    with pytest.raises(WorkspaceStartupError) as caught:
        _start(creation_db, owner_id="other")
    assert (caught.value.status_code, caught.value.code) == (404, "workspace_chat_startup_owner_mismatch")


def test_unique_violation_without_winner_propagates_original_failure(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recovery may replay only an actual winner for this key, never accept afresh."""
    from tldw_Server_API.app.core.DB_Management.backends.base import UniqueConstraintError

    _start(creation_db, idempotency_key="taken")
    insert = creation_db.workspace_chat_startups.insert_receipt

    def collide(owner: str, key: str, *args: Any, **kwargs: Any) -> None:
        """Trigger a real driver's unique constraint on another accepted key."""
        insert(owner, hashlib.sha256(b"taken").hexdigest(), *args, **kwargs)

    monkeypatch.setattr(creation_db.workspace_chat_startups, "insert_receipt", collide)
    with pytest.raises((sqlite3.IntegrityError, UniqueConstraintError)):
        _start(creation_db, idempotency_key="new", receipt_limit=2)
    assert _counts(creation_db) == (1, 1)


def test_winner_discovery_precedes_post_lock_default_failure(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A winner discovered after Persona selection must take the replay error path."""
    _start(creation_db)
    with creation_db.transaction() as conn:
        conn.execute("UPDATE persona_profiles SET is_active = ? WHERE id = ?", (False, "persona-a"))
    getter = creation_db.workspace_chat_startups.get_receipt
    reads = 0

    def stale_preflight(*args: Any, **kwargs: Any) -> Any:
        """Hide only pre-lock observations; later reads expose the real accepted receipt."""
        nonlocal reads
        reads += 1
        row = getter(*args, **kwargs)
        return None if reads <= 2 else row

    monkeypatch.setattr(creation_db.workspace_chat_startups, "get_receipt", stale_preflight)
    with pytest.raises(WorkspaceStartupError) as caught:
        _start(creation_db)
    assert (caught.value.status_code, caught.value.code) == (409, "persona_unavailable")
    assert _counts(creation_db) == (1, 1)


def test_deleted_hint_becoming_live_cannot_skip_persona_admission(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed lock hint cannot grant replay when no current Persona was admitted."""
    first = _start(creation_db)
    with creation_db.transaction() as conn:
        creation_db.soft_delete_conversation(first.conversation["id"], first.conversation["version"])
        conn.execute("UPDATE persona_profiles SET is_active = ? WHERE id = ?", (False, "persona-a"))
    lock = creation_db.workspace_chat_startups.lock_conversation

    def resurrect_between_reads(conversation_id: str, *, conn: Any) -> Any:
        """Fault the hint/final-row boundary using actual row mutation, not fake results."""
        conn.execute("UPDATE conversations SET deleted = ? WHERE id = ?", (False, conversation_id))
        return lock(conversation_id, conn=conn)

    monkeypatch.setattr(creation_db.workspace_chat_startups, "lock_conversation", resurrect_between_reads)
    with pytest.raises(WorkspaceStartupError) as caught:
        _start(creation_db)
    assert (caught.value.status_code, caught.value.code) == (409, "workspace_chat_startup_changed")
    assert _counts(creation_db) == (1, 0)


def test_matching_owner_admits_independently_of_writer_attribution(creation_db: CharactersRAGDB) -> None:
    """An owned default and replay remain valid after a device attribution change."""
    creation_db.client_id = "different-device"
    first = _start(creation_db)
    replay = _start(creation_db)
    assert first.conversation["assistant_id"] == "persona-a"
    assert (replay.replayed, replay.conversation["id"]) == (True, first.conversation["id"])


@pytest.mark.parametrize("winner", [False, True])
def test_core_admission_failure_rechecks_winner_before_mapping(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, winner: bool
) -> None:
    """Core admission failure either rolls back/replays the winner or rejects atomically."""
    from tldw_Server_API.app.core.Workspaces import chat_startup

    first = _start(creation_db) if winner else None
    getter = creation_db.workspace_chat_startups.get_receipt
    admission = chat_startup.require_current_persona
    transaction = creation_db.transaction
    reads, admissions = 0, 0
    transactions: list[Any] = []

    def stale_preflight(*args: Any, **kwargs: Any) -> Any:
        """Keep the real winning receipt; hide the three observations before core admission."""
        nonlocal reads
        reads += 1
        row = getter(*args, **kwargs)
        return None if winner and reads <= 3 else row

    def revoke_before_guard(db: CharactersRAGDB, **kwargs: Any) -> Any:
        """Mutate actual profile state only in the first attempt's real owned unit."""
        nonlocal admissions
        admissions += 1
        if admissions == 1:
            kwargs["conn"].execute("UPDATE persona_profiles SET is_active = ? WHERE id = ?", (False, "persona-a"))
        return admission(db, **kwargs)

    @contextmanager
    def record_transaction() -> Iterator[Any]:
        """Observe distinct owned attempts while retaining physical settlement."""
        with transaction() as conn:
            transactions.append(conn)
            yield conn

    with monkeypatch.context() as patch:
        patch.setattr(creation_db.workspace_chat_startups, "get_receipt", stale_preflight)
        patch.setattr(chat_startup, "require_current_persona", revoke_before_guard)
        patch.setattr(creation_db, "transaction", record_transaction)
        if winner:
            result = _start(creation_db)
            assert (result.replayed, result.conversation["id"]) == (True, first.conversation["id"])
        else:
            with pytest.raises(WorkspaceStartupError) as caught:
                _start(creation_db)
            assert (caught.value.status_code, caught.value.code, caught.value.reason) == (
                409, "persona_unavailable", "persona_unavailable",
            )
    assert len(transactions) == (2 if winner else 1)
    assert _counts(creation_db) == ((1, 1) if winner else (0, 0))
    with creation_db.transaction():
        assert creation_db.get_persona_profile("persona-a", user_id="user-1")["is_active"]
