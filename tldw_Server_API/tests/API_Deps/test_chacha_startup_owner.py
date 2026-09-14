"""Startup authorization uses the owner of a real alias-initialized cached DB."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio

from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as deps
from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import ChatSessionCreate
from tldw_Server_API.app.core import feature_flags
from tldw_Server_API.app.core.Character_Chat.modules.character_chat import (
    get_conversation_metadata,
    list_character_conversations,
)
from tldw_Server_API.app.core.Chat.assistant_startup import AssistantStartup, decode_assistant_startup
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, InputError
from tldw_Server_API.app.core.Workspaces import assistant_defaults

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest_asyncio.fixture
async def owner_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AsyncIterator[None]:
    """Use normal path resolution, initialization, health probes and lifecycle."""
    from tldw_Server_API.app.main import app

    await deps.shutdown_chacha_resources()
    deps.reset_chacha_shutdown_state()
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    monkeypatch.delitem(app.dependency_overrides, deps.get_chacha_db_for_user, raising=False)
    monkeypatch.setattr(feature_flags, "is_persona_enabled", lambda: True)
    try:
        yield
    finally:
        await deps.shutdown_chacha_resources()
        deps.reset_chacha_shutdown_state()


@pytest_asyncio.fixture(params=["voice_assistant", "study-pack-worker-42"])
async def alias_db(owner_cache: None, request: pytest.FixtureRequest) -> CharactersRAGDB:
    """Initialize as voice/a worker, then use the real REST dependency cache hit."""
    first = await deps.get_chacha_db_for_user_id(42, client_id=request.param)
    rest = await deps.get_chacha_db_for_user(SimpleNamespace(id=42))
    assert rest is first
    assert rest.client_id == request.param
    return rest


def _seed(db: CharactersRAGDB) -> int:
    """Store a real owned Persona default and return its Character source."""
    character_id = db.add_character_card({"name": "Source"})
    db.create_persona_profile({
        "id": "persona", "name": "Researcher", "user_id": "42",
        "character_card_id": character_id, "mode": "session_scoped", "is_active": True,
    })
    db.upsert_workspace("origin", "Origin")
    db.update_workspace("origin", {"assistant_defaults_json": {
        "assistant_kind": "persona", "assistant_id": "persona", "persona_memory_mode": "read_only",
    }}, 1)
    return character_id


def _create(db: CharactersRAGDB, user_id: str = "42", **overrides: Any) -> str:
    """Pass the endpoint's validated scope and owner payload to the real helper."""
    return assistant_defaults.create_workspace_persona_conversation(
        db, user_id=user_id, request=ChatSessionCreate(scope_type="workspace", workspace_id="origin"),
        conversation_data={
            "id": "created", "root_id": "created", "scope_type": "workspace", "workspace_id": "origin",
            "client_id": user_id, "parent_conversation_id": None, "forked_from_message_id": None,
            **overrides,
        },
        title_timestamp="stamp",
    )


def _insert_history(db: CharactersRAGDB, character_id: int | None = None) -> str:
    """Seed stored history independently so projection tests do not rely on creation."""
    return db.add_conversation(
        {"id": "history", "title": "History", "client_id": "42", "character_id": character_id},
        assistant_startup=AssistantStartup(source="workspace_default", workspace_id="origin", workspace_version=2),
    )


async def test_alias_first_rest_creation_preserves_owner_and_client_attribution(alias_db: CharactersRAGDB) -> None:
    """The owner can start a Persona without replacing the cached writer alias."""
    db = alias_db
    alias = db.client_id
    _seed(db)
    legacy = db.add_conversation({"title": "Legacy", "client_id": "42"})
    assert db.get_conversation_by_id(legacy)["client_id"] == "42"
    row = db.get_conversation_by_id(_create(db))
    assert (row["client_id"], row["assistant_id"], row["title"]) == ("42", "persona", "Researcher Chat (stamp)")
    assert decode_assistant_startup(row["assistant_startup_json"]).workspace_id == "origin"
    assert db.client_id == alias
    attributed = db.add_conversation({"title": "Attributed"})
    assert db.get_conversation_by_id(attributed)["client_id"] == alias


async def test_alias_first_rest_projection_reads_real_history(alias_db: CharactersRAGDB) -> None:
    """REST owner projection succeeds on the healthy cached alias handle."""
    _seed(alias_db)
    cid = _insert_history(alias_db)
    raw = alias_db.get_conversation_by_id(cid)["assistant_startup_json"]
    projected = assistant_defaults.project_assistant_startup(alias_db, raw=raw, user_id="42")
    assert projected.workspace_id == "origin"
    assert alias_db.get_conversation_by_id(cid)["assistant_startup_json"] == raw


@pytest.mark.parametrize("surface", ["detail", "list"])
async def test_alias_first_character_library_projects_and_redacts(
    alias_db: CharactersRAGDB, surface: str,
) -> None:
    """Library projection uses DB ownership while retaining explicit client filters."""
    db = alias_db
    character_id = _seed(db)
    cid = _insert_history(db, character_id)
    stored = db.get_conversation_by_id(cid)

    def read() -> dict[str, Any]:
        """Exercise the public read facade, not its private projection helper."""
        if surface == "detail":
            return get_conversation_metadata(db, cid)
        assert list_character_conversations(db, character_id, client_id=db.client_id) == []
        return list_character_conversations(db, character_id, client_id="42")[0]

    assert read()["assistant_startup"]["workspace_id"] == "origin"
    db.delete_workspace("origin", expected_version=2)
    result = read()
    assert result["assistant_startup"]["source"] == "unknown"
    assert "assistant_startup_json" not in result
    assert db.get_conversation_by_id(cid) == stored


@pytest.mark.parametrize("wrong_owner", ["43", "caller-alias"])
async def test_alias_is_not_owner_authority(alias_db: CharactersRAGDB, wrong_owner: str) -> None:
    """Neither another owner nor the writer alias may authorize this DB's origin."""
    db = alias_db
    _seed(db)
    cid = _insert_history(db)
    raw = db.get_conversation_by_id(cid)["assistant_startup_json"]
    caller = db.client_id if wrong_owner == "caller-alias" else wrong_owner
    with pytest.raises(InputError, match="owner"):
        _create(db, caller)
    with pytest.raises(InputError, match="owner"):
        assistant_defaults.project_assistant_startup(
            db, raw=raw, user_id=caller, workspace_visibility_cache={"origin": True},
        )
    assert db.get_conversation_by_id("created") is None


async def test_payload_cannot_replace_trusted_owner(alias_db: CharactersRAGDB) -> None:
    """Untrusted owner-shaped fields do not authorize a different payload client."""
    _seed(alias_db)
    with pytest.raises(InputError, match="owner"):
        _create(alias_db, client_id="43", owner_user_id="43")
    assert alias_db.get_conversation_by_id("created") is None


async def test_same_alias_keeps_distinct_owner_databases(alias_db: CharactersRAGDB) -> None:
    """An identical service alias never coalesces different owners' cache entries."""
    _seed(alias_db)
    cid = _insert_history(alias_db)
    other = await deps.get_chacha_db_for_user_id(43, client_id=alias_db.client_id)
    assert other is not alias_db
    assert other.get_conversation_by_id(cid) is None
    assert other.get_workspace("origin") is None
    assert (alias_db.owner_user_id, other.owner_user_id) == ("42", "43")


async def test_shutdown_reopen_rebinds_owner_without_rewriting_history(alias_db: CharactersRAGDB) -> None:
    """A fresh handle gets the same owner and its new first caller attribution."""
    _seed(alias_db)
    cid = _insert_history(alias_db)
    stored = alias_db.get_conversation_by_id(cid)
    await deps.shutdown_chacha_resources()
    deps.reset_chacha_shutdown_state()
    reopened = await deps.get_chacha_db_for_user_id(42, client_id="reopened-worker")
    assert reopened is not alias_db
    assert reopened.client_id == "reopened-worker"
    assert reopened.get_conversation_by_id(cid) == stored
    projected = assistant_defaults.project_assistant_startup(reopened, raw=stored["assistant_startup_json"], user_id="42")
    assert projected.workspace_id == "origin"


async def test_concurrent_rest_waiter_receives_owner_bound_alias_handle(
    owner_cache: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observe the real initialization waiter before publishing the prepared DB."""
    created = threading.Event()
    release = threading.Event()
    waiting = threading.Event()
    original_create = deps._create_and_prepare_db
    handles: list[CharactersRAGDB] = []

    def held_create(user_id: int, client_id: str) -> CharactersRAGDB:
        """Pause actual construction before the cache can publish its result."""
        db = original_create(user_id, client_id)
        handles.append(db)
        created.set()
        assert release.wait(30)
        return db

    monkeypatch.setattr(deps, "_create_and_prepare_db", held_create)
    first = asyncio.create_task(deps.get_chacha_db_for_user_id(42, client_id="voice_assistant"))
    tasks = [first]
    try:
        assert await asyncio.to_thread(created.wait, 30)
        event = deps._chacha_db_init_events[str(deps.DatabasePaths.get_user_base_directory(42))]
        original_wait = event.wait

        def observed_wait(timeout: float | None = None) -> bool:
            """Confirm REST is waiting on the existing initialization event."""
            waiting.set()
            return original_wait(timeout)

        monkeypatch.setattr(event, "wait", observed_wait)
        rest = asyncio.create_task(deps.get_chacha_db_for_user(SimpleNamespace(id=42)))
        tasks.append(rest)
        assert await asyncio.to_thread(waiting.wait, 10)
        release.set()
        first_db, rest_db = await asyncio.gather(*tasks)
        assert first_db is rest_db
        assert handles == [rest_db]
        assert (rest_db.owner_user_id, rest_db.client_id) == ("42", "voice_assistant")
        _seed(rest_db)
        assert rest_db.get_conversation_by_id(_create(rest_db))["assistant_id"] == "persona"
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
