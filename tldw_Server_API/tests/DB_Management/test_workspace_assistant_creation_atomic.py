"""Atomic Workspace startup selection using real SQLite and PostgreSQL stores."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import ChatSessionCreate
from tldw_Server_API.app.core import feature_flags
from tldw_Server_API.app.core.Character_Chat.character_conversation_factory import create_character_conversation
from tldw_Server_API.app.core.Chat.assistant_startup import AssistantStartup, decode_assistant_startup
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
