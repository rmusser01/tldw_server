"""Atomic Workspace startup selection using real SQLite and PostgreSQL stores."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import ChatSessionCreate
from tldw_Server_API.app.api.v1.schemas.workspace_schemas import (
    WorkspaceAssistantDefaultDegradedReason,
    WorkspaceEffectiveAssistantDefault,
)
from tldw_Server_API.app.core import feature_flags
from tldw_Server_API.app.core.Character_Chat.character_conversation_factory import create_character_conversation
from tldw_Server_API.app.core.Chat.assistant_startup import AssistantStartup, decode_assistant_startup
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, InputError
from tldw_Server_API.app.core.Workspaces import assistant_defaults
from tldw_Server_API.tests.DB_Management.test_conversation_assistant_startup import (
    _assert_blocked_writer,
    _wait_for_blocked_writer,
)
from tldw_Server_API.tests.DB_Management.test_conversation_assistant_startup import (
    db_factory as db_factory,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def creation_db(db_factory: Callable[[], CharactersRAGDB], monkeypatch: pytest.MonkeyPatch) -> CharactersRAGDB:
    """Seed saved defaults without mocking the resolver or transactional stores."""
    monkeypatch.setattr(feature_flags, "is_persona_enabled", lambda: True)
    db = db_factory()
    character = db.add_character_card({"name": "Source"})
    for suffix, name in (("a", "First"), ("b", "Second")):
        db.create_persona_profile({
            "id": f"persona-{suffix}", "name": name, "user_id": "user-1",
            "character_card_id": character, "mode": "session_scoped", "is_active": True,
        })
    db.upsert_workspace("ws", "Workspace")
    db.update_workspace("ws", {"assistant_defaults_json": {
        "assistant_kind": "persona", "assistant_id": "persona-a", "persona_memory_mode": "read_only",
    }}, 1)
    # Finish the read transaction left by the update's returned Workspace row
    # before a second handle performs the existing schema preflight.
    with db.transaction():
        pass
    return db


def _create(db: CharactersRAGDB, **overrides: Any) -> str:
    """Call the actual helper with the endpoint's validated payload shape."""
    return assistant_defaults.create_workspace_persona_conversation(
        db, user_id="user-1", request=ChatSessionCreate(scope_type="workspace", workspace_id="ws"),
        conversation_data={
            "id": "created", "root_id": "created", "scope_type": "workspace", "workspace_id": "ws",
            "client_id": "user-1", "parent_conversation_id": None, "forked_from_message_id": None,
            "title": "Stale preflight title", **overrides,
        },
        title_timestamp="stamp",
    )


def test_helper_replaces_preflight_identity_and_title(creation_db: CharactersRAGDB) -> None:
    """The committed title, identity and origin come from the transaction's selection."""
    cid = _create(creation_db, assistant_kind="persona", assistant_id="persona-b")
    row = creation_db.get_conversation_by_id(cid)
    assert (row["assistant_id"], row["persona_memory_mode"], row["title"]) == (
        "persona-a", "read_only", "First Chat (stamp)",
    )
    assert decode_assistant_startup(row["assistant_startup_json"]).model_dump() == {
        "schema_version": 1, "source": "workspace_default", "workspace_id": "ws", "workspace_version": 2,
    }


@pytest.mark.parametrize("request_fields, clear_default, expected_identity, expected_title, expected_origin", [
    pytest.param({}, False, ("persona", "persona-a", None, "read_only"), "First Chat (stamp)",
                 ("workspace_default", "ws", 2), id="inherited"),
    pytest.param({"title": "Chosen title"}, False, ("persona", "persona-a", None, "read_only"), "Chosen title",
                 ("workspace_default", "ws", 2), id="explicit-title"),
    pytest.param({"title": None}, False, ("persona", "persona-a", None, "read_only"), "First Chat (stamp)",
                 ("workspace_default", "ws", 2), id="null-title"),
    pytest.param({"title": ""}, False, ("persona", "persona-a", None, "read_only"), "First Chat (stamp)",
                 ("workspace_default", "ws", 2), id="empty-title"),
    pytest.param({"assistant_kind": None}, False, (None, None, None, None), "Chat (stamp)",
                 ("explicit_none", None, None), id="explicit-none"),
    pytest.param({"assistant_kind": "persona", "assistant_id": "persona-b", "persona_memory_mode": "read_write"},
                 False, ("persona", "persona-b", None, "read_write"), "Second Chat (stamp)",
                 ("explicit", None, None), id="explicit-persona"),
    pytest.param({}, True, (None, None, None, None), "Chat (stamp)",
                 ("system_fallback", "ws", 3), id="cleared-default"),
])
def test_resolved_insertion_preserves_identity_title_and_origin(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch,
    request_fields: dict[str, Any], clear_default: bool,
    expected_identity: tuple[Any, ...], expected_title: str, expected_origin: tuple[Any, ...],
) -> None:
    """Insert the selected result once without resolving again or opening a transaction."""
    insert = getattr(assistant_defaults, "insert_resolved_workspace_conversation", None)
    assert callable(insert), "Missing shared resolved-conversation insertion helper"
    if clear_default:
        creation_db.update_workspace("ws", {"assistant_defaults_json": None}, 2)
    request = ChatSessionCreate(scope_type="workspace", workspace_id="ws", **request_fields)
    payload = {
        "id": "resolved", "root_id": "resolved", "client_id": "user-1",
        "scope_type": "workspace", "workspace_id": "ws", "title": "Stale preflight title",
        "assistant_kind": "persona", "assistant_id": "persona-b", "character_id": None,
        "persona_memory_mode": "read_write", "topic_label": "Keep metadata",
    }
    original_payload = dict(payload)

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        """Selection and transaction ownership must stay with the caller."""
        raise AssertionError("Insertion must not resolve or open a transaction")

    with creation_db.transaction() as conn:
        resolved = assistant_defaults.resolve_workspace_assistant_startup(
            creation_db, user_id="user-1", request=request, conn=conn,
        )
        with monkeypatch.context() as patch:
            patch.setattr(assistant_defaults, "resolve_workspace_assistant_startup", unexpected)
            patch.setattr(creation_db, "transaction", unexpected)
            cid = insert(
                creation_db, resolved=resolved, conversation_data=payload, title_timestamp="stamp", conn=conn,
            )
    row = creation_db.get_conversation_by_id(cid)
    assert cid == "resolved"
    assert tuple(row[field] for field in (
        "assistant_kind", "assistant_id", "character_id", "persona_memory_mode",
    )) == expected_identity
    assert row["title"] == expected_title
    origin = decode_assistant_startup(row["assistant_startup_json"])
    assert (origin.source, origin.workspace_id, origin.workspace_version) == expected_origin
    assert (row["scope_type"], row["workspace_id"], row["root_id"], row["topic_label"]) == (
        "workspace", "ws", "resolved", "Keep metadata",
    )
    assert payload == original_payload


@pytest.mark.parametrize("identity", [{}, {"assistant_kind": None}, {
    "assistant_kind": "persona", "assistant_id": "persona-b",
}])
def test_resolver_accepts_mapping_without_changing_selection_intent(
    creation_db: CharactersRAGDB, identity: dict[str, Any],
) -> None:
    """Core callers can pass fields while sharing the same validated legacy resolver."""
    payload = {"scope_type": "workspace", "workspace_id": "ws", **identity}
    with creation_db.transaction() as conn:
        expected = assistant_defaults.resolve_workspace_assistant_startup(
            creation_db, user_id="user-1", request=ChatSessionCreate.model_validate(payload), conn=conn,
        )
        actual = assistant_defaults.resolve_workspace_assistant_startup(
            creation_db, user_id="user-1", request=payload, conn=conn,
        )
    assert (actual.request.model_dump(), actual.request.model_fields_set, actual.startup, actual.display_name) == (
        expected.request.model_dump(), expected.request.model_fields_set, expected.startup, expected.display_name,
    )


def test_resolved_insertion_obeys_caller_rollback(
    creation_db: CharactersRAGDB, db_factory: Callable[[], CharactersRAGDB],
) -> None:
    """The helper cannot commit identity or provenance ahead of its caller."""
    insert = getattr(assistant_defaults, "insert_resolved_workspace_conversation", None)
    assert callable(insert), "Missing shared resolved-conversation insertion helper"
    observer = db_factory()
    with pytest.raises(RuntimeError, match="rollback caller unit"):
        with creation_db.transaction() as conn:
            resolved = assistant_defaults.resolve_workspace_assistant_startup(
                creation_db, user_id="user-1",
                request=ChatSessionCreate(scope_type="workspace", workspace_id="ws"), conn=conn,
            )
            cid = insert(
                creation_db, resolved=resolved,
                conversation_data={
                    "id": "rolled-back", "root_id": "rolled-back", "client_id": "user-1",
                    "scope_type": "workspace", "workspace_id": "ws",
                }, title_timestamp="stamp", conn=conn,
            )
            assert creation_db.get_conversation_by_id(cid)["assistant_id"] == "persona-a"
            raise RuntimeError("rollback caller unit")
    assert observer.get_conversation_by_id("rolled-back", include_deleted=True) is None


def test_legacy_creation_resolves_once_and_delegates_insertion(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guarded wrapper shares insertion without a second default selection."""
    insert = getattr(assistant_defaults, "insert_resolved_workspace_conversation", None)
    assert callable(insert), "Missing shared resolved-conversation insertion helper"
    resolve = assistant_defaults.resolve_workspace_assistant_startup
    selections: list[assistant_defaults.ResolvedConversationAssistant] = []
    inserted: list[assistant_defaults.ResolvedConversationAssistant] = []

    def resolve_once(*args: Any, **kwargs: Any) -> assistant_defaults.ResolvedConversationAssistant:
        """Keep real locking selection, tracking the result handed to insertion."""
        resolved = resolve(*args, **kwargs)
        selections.append(resolved)
        return resolved

    def insert_once(*args: Any, **kwargs: Any) -> str:
        """Keep the actual insertion on the wrapper's supplied connection."""
        inserted.append(kwargs["resolved"])
        return insert(*args, **kwargs)

    monkeypatch.setattr(assistant_defaults, "resolve_workspace_assistant_startup", resolve_once)
    monkeypatch.setattr(assistant_defaults, "insert_resolved_workspace_conversation", insert_once)
    cid = _create(creation_db)
    assert len(selections) == len(inserted) == 1
    assert inserted[0] is selections[0]
    assert creation_db.get_conversation_by_id(cid)["title"] == "First Chat (stamp)"


@pytest.mark.parametrize("reason, expected_status", [
    ("persona_deleted", 409), ("persona_unavailable", 409), ("persona_feature_disabled", 503),
    ("permission_denied", 409), ("invalid_default", 409), ("unsupported_assistant_kind", 409),
])
def test_unavailable_default_exposes_typed_reason_with_legacy_http_contract(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch,
    reason: WorkspaceAssistantDefaultDegradedReason, expected_status: int,
) -> None:
    """Only unavailable defaults expose a bounded reason, preserving legacy wire behavior."""
    if reason == "persona_deleted":
        creation_db.soft_delete_persona_profile(persona_id="persona-a", user_id="user-1", expected_version=1)
    elif reason == "persona_unavailable":
        _mutate(creation_db, "deactivate")
    elif reason == "persona_feature_disabled":
        monkeypatch.setattr(feature_flags, "is_persona_enabled", lambda: False)
    elif reason == "permission_denied":
        creation_db.update_workspace("ws", {"assistant_defaults_json": {
            "assistant_kind": "persona", "assistant_id": "inaccessible-persona", "persona_memory_mode": "read_only",
        }}, 2)
    elif reason == "invalid_default":
        creation_db.update_workspace("ws", {"assistant_defaults_json": {
            "assistant_kind": "persona", "assistant_id": "",
        }}, 2)
    else:
        # Stored defaults currently admit only Persona; this future-kind branch
        # is unreachable through schema-safe storage, but its bounded result is valid.
        monkeypatch.setattr(
            assistant_defaults, "resolve_effective_workspace_assistant_default",
            lambda *args, **kwargs: WorkspaceEffectiveAssistantDefault(
                status="unavailable", source="workspace", degraded_reason="unsupported_assistant_kind",
            ),
        )
    with pytest.raises(HTTPException) as error:
        _create(creation_db)
    assert type(error.value) is getattr(assistant_defaults, "WorkspaceDefaultUnavailable", None)
    assert error.value.reason == reason
    assert error.value.status_code == expected_status
    assert error.value.detail == "Workspace default Persona is unavailable; choose an assistant explicitly"
    assert creation_db.get_conversation_by_id("created", include_deleted=True) is None


@pytest.mark.parametrize("explicit_persona", [False, True], ids=["missing-workspace", "missing-explicit-persona"])
def test_missing_selection_target_is_not_typed_default_unavailability(
    creation_db: CharactersRAGDB, explicit_persona: bool,
) -> None:
    """Workspace and explicit Persona misses retain their original 404 types/details."""
    request = ChatSessionCreate(
        scope_type="workspace", workspace_id="ws" if explicit_persona else "missing-workspace",
        **({"assistant_kind": "persona", "assistant_id": "missing-persona"} if explicit_persona else {}),
    )
    with pytest.raises(HTTPException) as error:
        with creation_db.transaction() as conn:
            assistant_defaults.resolve_workspace_assistant_startup(
                creation_db, user_id="user-1", request=request, conn=conn,
            )
    assert type(error.value) is HTTPException
    assert (error.value.status_code, error.value.detail) == (
        404, "Persona not found" if explicit_persona else "Workspace not found",
    )


@pytest.mark.parametrize("getter", ["get_workspace", "get_persona_profile"])
@pytest.mark.parametrize("failure_kind", ["http", "storage"])
def test_selection_errors_are_not_relabeled_as_default_unavailability(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, getter: str, failure_kind: str,
) -> None:
    """Unexpected HTTP/storage failures propagate unchanged, never becoming fallback/default errors."""
    failure = HTTPException(status_code=502, detail="upstream failure") if failure_kind == "http" else (
        DatabaseError("injected storage failure")
    )

    def fail(*args: Any, **kwargs: Any) -> Any:
        """Inject only the failed read; the real transaction still owns rollback."""
        raise failure

    with monkeypatch.context() as patch:
        patch.setattr(creation_db, getter, fail)
        with pytest.raises(type(failure)) as error:
            _create(creation_db)
    assert error.value is failure
    assert creation_db.get_conversation_by_id("created", include_deleted=True) is None


@pytest.mark.parametrize("overrides", [
    {"client_id": "other-owner"}, {"scope_type": "global"}, {"workspace_id": "other"},
    {"parent_conversation_id": "forged-parent"}, {"forked_from_message_id": "forged-message"},
    {"root_id": "forged-root"},
])
def test_helper_rejects_payload_authority_changes(creation_db: CharactersRAGDB, overrides: dict[str, Any]) -> None:
    """An internal mapping cannot swap validated scope, owner or lineage after selection."""
    with pytest.raises(InputError):
        _create(creation_db, **overrides)
    assert creation_db.get_conversation_by_id("created", include_deleted=True) is None


@pytest.mark.parametrize("after_insert", [False, True])
def test_creation_failure_leaves_no_partial_row(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, after_insert: bool,
) -> None:
    """Failures after selection and after real INSERT both roll back the whole unit."""
    insert = creation_db.add_conversation

    def fail(*args: Any, **kwargs: Any) -> str:
        """Inject failure at the actual insertion boundary, not a mocked resolver."""
        if after_insert:
            insert(*args, **kwargs)
        raise RuntimeError("injected creation failure")

    monkeypatch.setattr(creation_db, "add_conversation", fail)
    with pytest.raises(RuntimeError, match="injected creation failure"):
        _create(creation_db)
    assert creation_db.get_conversation_by_id("created", include_deleted=True) is None
    assert creation_db.get_workspace("ws")["version"] == 2


@pytest.mark.parametrize("getter", ["workspace", "persona"])
def test_locking_getter_requires_connection(creation_db: CharactersRAGDB, getter: str) -> None:
    """A locking read must not silently open and release its own transaction."""
    with pytest.raises(InputError):
        if getter == "workspace":
            creation_db.get_workspace("ws", for_update=True)
        else:
            creation_db.get_persona_profile("persona-a", user_id="user-1", for_update=True)


@pytest.mark.parametrize("getter", ["workspace", "persona"])
def test_locking_getter_rejects_finished_transaction(creation_db: CharactersRAGDB, getter: str) -> None:
    """An expired transaction wrapper cannot authorize a fresh implicit locking read."""
    with creation_db.transaction() as conn:
        pass
    with pytest.raises(InputError):
        if getter == "workspace":
            creation_db.get_workspace("ws", conn=conn, for_update=True)
        else:
            creation_db.get_persona_profile("persona-a", user_id="user-1", conn=conn, for_update=True)


@pytest.mark.parametrize("db_factory", ["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)], indirect=True)
def test_factory_failure_rolls_back_trusted_origin_and_all_artifacts(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late snapshot failure cannot leave origin, conversation, settings or greeting."""
    db = creation_db
    character_id = db.get_persona_profile("persona-a", user_id="user-1")["character_card_id"]
    put = db.conversation_resume_store.put_behavior_snapshot

    def fail(*args: Any, **kwargs: Any) -> Any:
        """Fail after the real snapshot writer has run in the factory transaction."""
        put(*args, **kwargs)
        raise RuntimeError("injected snapshot failure")

    monkeypatch.setattr(db.conversation_resume_store, "put_behavior_snapshot", fail)
    with pytest.raises(RuntimeError, match="injected snapshot failure"):
        create_character_conversation(
            db, conversation_data={"id": "failed-character", "character_id": character_id},
            assistant_startup=AssistantStartup(source="explicit"),
            initial_messages=[{"sender": "user", "content": "Initial message"}],
        )
    counts = tuple(db.execute_query(query).fetchone()["cnt"] for query in (
        "SELECT COUNT(*) AS cnt FROM conversations", "SELECT COUNT(*) AS cnt FROM messages",
        "SELECT COUNT(*) AS cnt FROM conversation_settings",
        "SELECT COUNT(*) AS cnt FROM conversation_behavior_snapshots",
    ))
    assert counts == (0, 0, 0, 0)


def test_connection_reads_preserve_owner_and_visibility(creation_db: CharactersRAGDB) -> None:
    """Transaction-aware reads retain deleted, staged and per-Persona owner predicates."""
    with creation_db.transaction() as conn:
        conn.execute("UPDATE workspaces SET deleted = 1 WHERE id = ?", ("ws",))
        conn.execute("UPDATE persona_profiles SET deleted = 1 WHERE id = ?", ("persona-a",))
        assert creation_db.get_workspace("ws", conn=conn, for_update=True) is None
        assert creation_db.get_workspace("ws", include_deleted=True, conn=conn, for_update=True)["deleted"]
        assert creation_db.get_persona_profile("persona-a", user_id="user-1", conn=conn, for_update=True) is None
        assert creation_db.get_persona_profile(
            "persona-a", user_id="other", include_deleted=True, conn=conn, for_update=True,
        ) is None
        assert creation_db.get_persona_profile(
            "persona-a", user_id="user-1", include_deleted=True, conn=conn, for_update=True,
        )["deleted"]
        conn.execute("UPDATE workspaces SET system_operation_state = 'staged' WHERE id = ?", ("ws",))
        assert creation_db.get_workspace("ws", include_deleted=True, conn=conn, for_update=True) is None


def _mutate(db: CharactersRAGDB, mutation: str) -> None:
    """Use existing public stores for the competing settings or availability write."""
    if mutation == "deactivate":
        profile = db.get_persona_profile("persona-a", user_id="user-1")
        db.update_persona_profile(
            persona_id="persona-a", update_data={"is_active": False},
            expected_version=profile["version"], user_id="user-1",
        )
    else:
        defaults = None if mutation == "clear" else {
            "assistant_kind": "persona", "assistant_id": "persona-b", "persona_memory_mode": "read_write",
        }
        db.update_workspace("ws", {"assistant_defaults_json": defaults}, 2)


def _assert_after_mutation(db: CharactersRAGDB, mutation: str, cid: str = "created") -> None:
    """Check literal post-mutation identity and provenance, including fail-closed creation."""
    if mutation == "deactivate":
        with pytest.raises(HTTPException) as error:
            _create(db, id=cid, root_id=cid)
        assert error.value.status_code == 409
        assert db.get_conversation_by_id(cid) is None
        return
    _create(db, id=cid, root_id=cid)
    row = db.get_conversation_by_id(cid)
    assert (row["assistant_id"], row["persona_memory_mode"]) == (
        (None, None) if mutation == "clear" else ("persona-b", "read_write")
    )
    origin = decode_assistant_startup(row["assistant_startup_json"])
    assert (origin.source, origin.workspace_version) == (
        "system_fallback" if mutation == "clear" else "workspace_default", 3,
    )


@pytest.mark.parametrize("db_factory", ["postgres"], indirect=True)
@pytest.mark.parametrize("mutation", ["clear", "rebind", "deactivate"])
def test_postgres_mutation_wins_before_creation_selection(
    creation_db: CharactersRAGDB, db_factory: Callable[[], CharactersRAGDB], mutation: str,
) -> None:
    """Creation waits on real row locks and then sees the committed new state."""
    creator = db_factory()
    with creation_db.transaction() as conn:
        _mutate(creation_db, mutation)
        _assert_blocked_writer(conn, creator, lambda: _assert_after_mutation(creator, mutation))


@pytest.mark.parametrize("db_factory", ["postgres"], indirect=True)
@pytest.mark.parametrize("mutation", ["clear", "rebind", "deactivate"])
def test_postgres_creation_holds_selection_locks_through_insert(
    creation_db: CharactersRAGDB, db_factory: Callable[[], CharactersRAGDB],
    monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    """Competing mutation cannot commit between selected identity and origin INSERT."""
    writer = db_factory()
    ready = Event()
    pid: list[int] = []
    insert = creation_db.add_conversation
    futures = []

    def mutate() -> None:
        """Expose the real backend PID with a bounded SQL wait."""
        with writer.transaction() as conn:
            conn.execute("SET LOCAL statement_timeout = '15s'")
            pid.append(conn.execute("SELECT pg_backend_pid() AS pid").fetchone()["pid"])
            ready.set()
            _mutate(writer, mutation)

    with ThreadPoolExecutor(max_workers=1) as executor:
        def insert_while_mutation_waits(*args: Any, **kwargs: Any) -> str:
            """Observe the server lock graph while the helper still owns its transaction."""
            future = executor.submit(mutate)
            futures.append(future)
            assert ready.wait(10)
            _wait_for_blocked_writer(kwargs["conn"], pid[0], future)
            return insert(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(creation_db, "add_conversation", insert_while_mutation_waits)
            _create(creation_db)
        futures[0].result(timeout=20)
    row = creation_db.get_conversation_by_id("created")
    assert (row["assistant_id"], row["persona_memory_mode"], row["title"]) == (
        "persona-a", "read_only", "First Chat (stamp)",
    )
    assert decode_assistant_startup(row["assistant_startup_json"]).workspace_version == 2
    _assert_after_mutation(creation_db, mutation, cid="later")
