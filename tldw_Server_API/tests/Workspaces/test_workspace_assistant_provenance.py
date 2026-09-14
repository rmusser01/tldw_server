"""Workspace chat startup records the actual choice without accepting caller provenance."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions as endpoint
from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
from tldw_Server_API.app.api.v1.schemas.chat_conversation_schemas import ConversationUpdateRequest
from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import ChatSessionCreate, ChatSessionUpdate
from tldw_Server_API.app.core import feature_flags
from tldw_Server_API.app.core.Character_Chat.character_conversation_factory import create_character_conversation
from tldw_Server_API.app.core.Chat.assistant_startup import AssistantStartup, decode_assistant_startup
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service
from tldw_Server_API.tests.Sync.test_sync_v2_server_origin_capture import (
    _chat_messages_app,
)
from tldw_Server_API.tests.Sync.test_sync_v2_server_origin_capture import (
    chacha_db as chacha_db,
)
from tldw_Server_API.tests.Sync.test_sync_v2_server_origin_capture import (
    sync_service as sync_service,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("model,payload", [
    (ChatSessionCreate, {"assistant_kind": None, "title": "Kept"}),
    (ChatSessionUpdate, {"title": "Kept"}),
    (ConversationUpdateRequest, {"version": 1, "state": "resolved"}),
])
def test_origin_rejection_does_not_forbid_other_legacy_extras(model: Any, payload: dict[str, Any]) -> None:
    """Targeted read-only validation preserves existing ignored-extra compatibility."""
    request = model.model_validate({**payload, "unrelated_legacy_option": "ignored"})
    assert request.model_dump(exclude_unset=True) == payload


@pytest.fixture
def startup_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    """Exercise real route/schema/store behavior with only auth and quotas isolated."""
    db = CharactersRAGDB(tmp_path / "startup-api.db", client_id="1")
    character_id = db.add_character_card({"name": "Source", "system_prompt": "Help.", "first_message": "Hello."})
    db.create_persona_profile({
        "id": "persona-a", "user_id": "1", "name": "First Persona", "character_card_id": character_id,
        "mode": "session_scoped", "is_active": True,
    })
    workspace = db.upsert_workspace("ws", "Workspace")
    db.update_workspace("ws", {"assistant_defaults_json": {
        "assistant_kind": "persona", "assistant_id": "persona-a", "persona_memory_mode": "read_only",
    }}, workspace["version"])
    limiter = SimpleNamespace(check_rate_limit=AsyncMock(), check_chat_limit=AsyncMock())
    monkeypatch.setattr(endpoint, "get_character_rate_limiter", lambda: limiter)
    monkeypatch.setattr(endpoint, "_active_chat_sync_service", lambda *args: None)
    monkeypatch.setattr(feature_flags, "is_persona_enabled", lambda: True)
    app = FastAPI()
    app.include_router(endpoint.router, prefix="/api/v1/chats")
    app.include_router(chat_endpoint.router, prefix="/api/v1/chat")
    app.dependency_overrides[endpoint.get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[endpoint.get_request_user] = lambda: SimpleNamespace(id=1)
    app.dependency_overrides[chat_endpoint.get_request_user] = lambda: SimpleNamespace(id=1)
    app.dependency_overrides[endpoint.require_expected_user] = lambda: None
    try:
        with TestClient(app) as client:
            yield SimpleNamespace(db=db, client=client, character_id=character_id, limiter=limiter)
    finally:
        db.close_all_connections()


@pytest.mark.parametrize("choice,source,assistant_id,memory_mode", [
    ({}, "workspace_default", "persona-a", "read_only"),
    ({"assistant_kind": None}, "explicit_none", None, None),
    ({"assistant_id": None}, "explicit_none", None, None),
    ({"character_id": None}, "explicit_none", None, None),
    ({"assistant_kind": "persona", "assistant_id": "persona-a"}, "explicit", "persona-a", None),
    ({"assistant_kind": "persona", "assistant_id": "persona-a", "persona_memory_mode": "read_write"},
     "explicit", "persona-a", "read_write"),
])
def test_creation_records_omission_null_and_explicit_choices(
    startup_api: SimpleNamespace, choice: dict[str, Any], source: str,
    assistant_id: str | None, memory_mode: str | None,
) -> None:
    """Explicit intent is not inferred from equality with the current default."""
    response = startup_api.client.post("/api/v1/chats/", json={"scope_type": "workspace", "workspace_id": "ws", **choice})
    assert response.status_code == 201, response.text
    row = startup_api.db.get_conversation_by_id(response.json()["id"])
    assert row["assistant_id"] == assistant_id
    assert row["persona_memory_mode"] == memory_mode
    origin = decode_assistant_startup(row["assistant_startup_json"]).model_dump()
    assert origin == {
        "schema_version": 1, "source": source,
        "workspace_id": "ws" if source == "workspace_default" else None,
        "workspace_version": startup_api.db.get_workspace("ws")["version"] if source == "workspace_default" else None,
    }


def test_global_creation_remains_unknown(startup_api: SimpleNamespace) -> None:
    """Workspace adoption does not expand to the existing global creation path."""
    response = startup_api.client.post("/api/v1/chats/", json={})
    assert response.status_code == 201, response.text
    row = startup_api.db.get_conversation_by_id(response.json()["id"])
    assert row["assistant_id"] is None
    assert row["assistant_startup_json"] is None


@pytest.mark.parametrize("cleared", [False, True])
def test_unset_and_cleared_defaults_record_examined_fallback(startup_api: SimpleNamespace, cleared: bool) -> None:
    """A Workspace setting opt-out is not a request-level explicit None."""
    workspace = startup_api.db.upsert_workspace("empty", "Empty")
    if cleared:
        workspace = startup_api.db.update_workspace("empty", {"assistant_defaults_json": None}, workspace["version"])
    response = startup_api.client.post("/api/v1/chats/", json={"scope_type": "workspace", "workspace_id": "empty"})
    assert response.status_code == 201, response.text
    row = startup_api.db.get_conversation_by_id(response.json()["id"])
    assert row["assistant_id"] is None
    assert decode_assistant_startup(row["assistant_startup_json"]).model_dump() == {
        "schema_version": 1, "source": "system_fallback", "workspace_id": "empty", "workspace_version": workspace["version"],
    }


@pytest.mark.parametrize("caller_title", [None, "My title"])
def test_default_changed_during_quota_check_is_not_reused_from_preflight(
    startup_api: SimpleNamespace, caller_title: str | None,
) -> None:
    """Identity, title and origin version must all describe the final selection."""
    db = startup_api.db
    db.create_persona_profile({
        "id": "persona-b", "user_id": "1", "name": "Second Persona", "character_card_id": startup_api.character_id,
        "mode": "session_scoped", "is_active": True,
    })

    def change_default(*args: Any) -> None:
        """Interleave a real settings write after the old preflight boundary."""
        workspace = db.get_workspace("ws")
        db.update_workspace("ws", {"assistant_defaults_json": {
            "assistant_kind": "persona", "assistant_id": "persona-b", "persona_memory_mode": "read_write",
        }}, workspace["version"])

    startup_api.limiter.check_chat_limit.side_effect = change_default
    payload = {"scope_type": "workspace", "workspace_id": "ws"}
    if caller_title:
        payload["title"] = caller_title
    response = startup_api.client.post("/api/v1/chats/", json=payload)
    assert response.status_code == 201, response.text
    row = db.get_conversation_by_id(response.json()["id"])
    assert (row["assistant_id"], row["persona_memory_mode"]) == ("persona-b", "read_write")
    if caller_title:
        assert row["title"] == caller_title
    else:
        assert row["title"].startswith("Second Persona Chat (")
    origin = decode_assistant_startup(row["assistant_startup_json"])
    assert (origin.source, origin.workspace_id, origin.workspace_version) == ("workspace_default", "ws", db.get_workspace("ws")["version"])


@pytest.mark.parametrize("character", [False, True])
def test_validated_fork_records_lineage_without_inheriting_default(startup_api: SimpleNamespace, character: bool) -> None:
    """Fork classification follows valid lineage, not today's Workspace default."""
    parent = startup_api.db.add_conversation({"id": "parent", "scope_type": "workspace", "workspace_id": "ws"})
    payload: dict[str, Any] = {"scope_type": "workspace", "workspace_id": "ws", "parent_conversation_id": parent}
    if character:
        payload["character_id"] = startup_api.character_id
    response = startup_api.client.post("/api/v1/chats/", json=payload)
    assert response.status_code == 201, response.text
    row = startup_api.db.get_conversation_by_id(response.json()["id"])
    assert (row["parent_conversation_id"], row["root_id"]) == (parent, parent)
    assert row["assistant_id"] == (str(startup_api.character_id) if character else None)
    assert decode_assistant_startup(row["assistant_startup_json"]).model_dump() == {
        "schema_version": 1, "source": "fork", "workspace_id": None, "workspace_version": None,
    }


@pytest.mark.parametrize("visibility", ["missing", "deleted", "staged"])
@pytest.mark.parametrize("fork", [False, True])
def test_character_creation_cannot_bypass_workspace_visibility(
    startup_api: SimpleNamespace, visibility: str, fork: bool,
) -> None:
    """The Character factory must not turn a hidden Workspace FK into authorization."""
    db = startup_api.db
    parent = db.add_conversation({"id": "parent", "scope_type": "workspace", "workspace_id": "ws"})
    workspace_id = "ws"
    if visibility == "missing":
        workspace_id = "missing"
    elif visibility == "deleted":
        db.delete_workspace("ws", expected_version=db.get_workspace("ws")["version"])
    else:
        with db.transaction() as conn:
            conn.execute("UPDATE workspaces SET system_operation_state = 'staged' WHERE id = ?", ("ws",))
    before = _creation_counts(db)
    payload = {"scope_type": "workspace", "workspace_id": workspace_id, "character_id": startup_api.character_id}
    if fork:
        payload["parent_conversation_id"] = parent
    response = startup_api.client.post("/api/v1/chats/", json=payload)
    assert response.status_code == 404, response.text
    assert _creation_counts(db) == before


@pytest.mark.parametrize("key", ["assistant_startup", "assistant_startup_json"])
@pytest.mark.parametrize("value", [None, {"source": "explicit"}])
def test_create_and_update_reject_caller_origin_without_writes(
    startup_api: SimpleNamespace, key: str, value: Any,
) -> None:
    """Targeted rejection covers null too while unrelated legacy extras remain separate."""
    db = startup_api.db
    before_count = db.execute_query("SELECT COUNT(*) FROM conversations").fetchone()[0]
    response = startup_api.client.post("/api/v1/chats/", json={key: value})
    assert response.status_code == 422, response.text
    assert db.execute_query("SELECT COUNT(*) FROM conversations").fetchone()[0] == before_count
    cid = db.add_conversation({"title": "Original"})
    before = db.get_conversation_by_id(cid)
    for method, path, params, payload in (
        ("put", f"/api/v1/chats/{cid}", {"expected_version": before["version"]}, {"title": "Changed", key: value}),
        ("patch", f"/api/v1/chat/conversations/{cid}", {}, {"state": "resolved", "version": before["version"], key: value}),
    ):
        response = getattr(startup_api.client, method)(path, params=params, json=payload)
        assert response.status_code == 422, response.text
        assert db.get_conversation_by_id(cid) == before


def _creation_counts(db: CharactersRAGDB) -> tuple[int, ...]:
    """Count all creation artifacts so error assertions catch orphaned side effects."""
    return tuple(db.execute_query(query).fetchone()[0] for query in (
        "SELECT COUNT(*) FROM conversations", "SELECT COUNT(*) FROM messages",
        "SELECT COUNT(*) FROM conversation_settings", "SELECT COUNT(*) FROM conversation_behavior_snapshots",
    ))


@pytest.mark.parametrize("archived", [False, True])
@pytest.mark.parametrize("character", [False, True])
def test_archived_and_active_workspaces_keep_legacy_creation(
    startup_api: SimpleNamespace, archived: bool, character: bool,
) -> None:
    """Archiving does not introduce a new admission rule for any supported identity."""
    db = startup_api.db
    if archived:
        db.update_workspace("ws", {"archived": True}, db.get_workspace("ws")["version"])
    choice = {"character_id": startup_api.character_id, "assistant_id": None} if character else {}
    response = startup_api.client.post("/api/v1/chats/", json={"scope_type": "workspace", "workspace_id": "ws", **choice})
    assert response.status_code == 201, response.text
    row = db.get_conversation_by_id(response.json()["id"])
    assert row["assistant_id"] == (str(startup_api.character_id) if character else "persona-a")
    assert decode_assistant_startup(row["assistant_startup_json"]).source == ("explicit" if character else "workspace_default")


@pytest.mark.parametrize("state,status_code", [
    ("disabled", 503), ("inactive", 409), ("deleted", 409), ("foreign", 409), ("invalid", 409),
])
def test_unavailable_default_is_bounded_and_creates_nothing(
    startup_api: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, state: str, status_code: int,
) -> None:
    """Unavailable defaults fail closed without disclosing their ID or writing rows."""
    db = startup_api.db
    if state == "disabled":
        monkeypatch.setattr(feature_flags, "is_persona_enabled", lambda: False)
    else:
        with db.transaction() as conn:
            if state == "inactive":
                conn.execute("UPDATE persona_profiles SET is_active = 0 WHERE id = ?", ("persona-a",))
            elif state == "deleted":
                conn.execute("UPDATE persona_profiles SET deleted = 1 WHERE id = ?", ("persona-a",))
            elif state == "foreign":
                conn.execute("UPDATE persona_profiles SET user_id = ? WHERE id = ?", ("other", "persona-a"))
            else:
                conn.execute("UPDATE workspaces SET assistant_defaults_json = ? WHERE id = ?", ('{"unexpected":true}', "ws"))
    before = _creation_counts(db)
    response = startup_api.client.post("/api/v1/chats/", json={"scope_type": "workspace", "workspace_id": "ws"})
    assert response.status_code == status_code, response.text
    assert "persona-a" not in response.text
    assert len(response.text) < 256
    assert _creation_counts(db) == before


@pytest.mark.parametrize("field", ["assistant_kind", "assistant_id", "character_id"])
def test_explicit_null_bypasses_disabled_defaults(
    startup_api: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, field: str,
) -> None:
    """Each original null selector suppresses inheritance even with Persona disabled."""
    monkeypatch.setattr(feature_flags, "is_persona_enabled", lambda: False)
    response = startup_api.client.post("/api/v1/chats/", json={"scope_type": "workspace", "workspace_id": "ws", field: None})
    assert response.status_code == 201, response.text
    row = startup_api.db.get_conversation_by_id(response.json()["id"])
    assert row["assistant_id"] is None
    assert decode_assistant_startup(row["assistant_startup_json"]).source == "explicit_none"


@pytest.mark.parametrize("choice", ["persona", "character", "fork", "global"])
def test_explicit_and_lineage_paths_do_not_consult_default_availability(
    startup_api: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, choice: str,
) -> None:
    """Disabled defaults do not reject independent explicit, fork or global intent."""
    monkeypatch.setattr(feature_flags, "is_persona_enabled", lambda: False)
    payload: dict[str, Any] = {"scope_type": "workspace", "workspace_id": "ws"}
    if choice == "persona":
        payload.update(assistant_kind="persona", assistant_id="persona-a", character_id=None)
    elif choice == "character":
        payload["character_id"] = startup_api.character_id
    elif choice == "fork":
        payload["parent_conversation_id"] = startup_api.db.add_conversation(payload)
    else:
        payload = {"character_id": startup_api.character_id}
    response = startup_api.client.post("/api/v1/chats/", json=payload)
    assert response.status_code == 201, response.text
    row = startup_api.db.get_conversation_by_id(response.json()["id"])
    assert decode_assistant_startup(row["assistant_startup_json"]).source == (
        "unknown" if choice == "global" else "fork" if choice == "fork" else "explicit"
    )


@pytest.mark.parametrize("invalid", ["missing_parent", "foreign_parent", "scope", "missing_message", "other_message", "no_parent"])
@pytest.mark.parametrize("character", [False, True])
def test_unvalidated_lineage_creates_no_artifacts(
    startup_api: SimpleNamespace, invalid: str, character: bool,
) -> None:
    """A supplied parent string is never itself authority to persist fork origin."""
    db = startup_api.db
    parent = db.add_conversation({"scope_type": "workspace", "workspace_id": "ws", "client_id": "1"})
    other = db.add_conversation({"client_id": "1"})
    message = db.add_message({"conversation_id": other, "sender": "user", "content": "Other message"})
    payload: dict[str, Any] = {"scope_type": "workspace", "workspace_id": "ws", "parent_conversation_id": parent}
    if character:
        payload["character_id"] = startup_api.character_id
    if invalid == "missing_parent":
        payload["parent_conversation_id"] = "missing"
    elif invalid == "foreign_parent":
        with db.transaction() as conn:
            conn.execute("UPDATE conversations SET client_id = ? WHERE id = ?", ("other", parent))
    elif invalid == "scope":
        payload["parent_conversation_id"] = other
    else:
        payload["forked_from_message_id"] = "missing" if invalid == "missing_message" else message
        if invalid == "no_parent":
            payload.pop("parent_conversation_id")
    before = _creation_counts(db)
    response = startup_api.client.post("/api/v1/chats/", json=payload)
    assert response.status_code in {400, 403, 404}, response.text
    assert _creation_counts(db) == before


def test_confirmed_read_write_default_is_inherited(startup_api: SimpleNamespace) -> None:
    """Creation uses the saved confirmed mode, not the request model's default."""
    db = startup_api.db
    db.update_workspace("ws", {"assistant_defaults_json": {
        "assistant_kind": "persona", "assistant_id": "persona-a", "persona_memory_mode": "read_write",
    }}, db.get_workspace("ws")["version"])
    response = startup_api.client.post("/api/v1/chats/", json={"scope_type": "workspace", "workspace_id": "ws"})
    assert response.status_code == 201, response.text
    row = db.get_conversation_by_id(response.json()["id"])
    assert (row["assistant_id"], row["persona_memory_mode"]) == ("persona-a", "read_write")
    assert decode_assistant_startup(row["assistant_startup_json"]).workspace_version == 3


@pytest.mark.parametrize("fork", [False, True])
def test_character_origin_preserves_factory_behavior_and_lineage(startup_api: SimpleNamespace, fork: bool) -> None:
    """Trusted forwarding retains greetings, participants, settings, snapshot and cross-Character forks."""
    db = startup_api.db
    secondary = db.add_character_card({"name": "Second", "first_message": "Second hello"})
    payload: dict[str, Any] = {
        "scope_type": "workspace", "workspace_id": "ws", "character_id": startup_api.character_id,
        "assistant_id": None, "participant_character_ids": [secondary], "prompt_preset_id": "st_default",
        "provider": "local-llm", "model": "local-test", "temperature": 0.0,
        "memory_by_character_id": {str(startup_api.character_id): "Remember the key"},
    }
    if fork:
        parent = db.add_conversation({"id": "parent", "root_id": "parent", "scope_type": "workspace", "workspace_id": "ws", "character_id": secondary})
        message = db.add_message({"conversation_id": parent, "sender": "user", "content": "Fork here"})
        payload.update(parent_conversation_id=parent, forked_from_message_id=message)
    response = startup_api.client.post("/api/v1/chats/?seed_first_message=true", json=payload)
    assert response.status_code == 201, response.text
    cid = response.json()["id"]
    row = db.get_conversation_by_id(cid)
    assert decode_assistant_startup(row["assistant_startup_json"]).source == ("fork" if fork else "explicit")
    assert (row["root_id"], row["parent_conversation_id"], row["forked_from_message_id"]) == (
        (parent, parent, message) if fork else (cid, None, None)
    )
    resume = db.get_roleplay_resume_state(cid)
    participants = resume["behavior_snapshot"]["payload"]["participants"]
    assert [p["source"]["id"] for p in participants] == [str(startup_api.character_id), str(secondary)]
    assert participants[0]["greeting"]["content"] == "Hello."
    assert participants[0]["default_memory"]["content"] == "Remember the key"
    settings = db.get_conversation_settings(cid)["settings"]
    assert settings["promptPreset"] == "st_default"
    completion = settings["roleplayResumeV1"]["effectiveCompletion"]
    assert (completion["provider"], completion["model"], completion["sampling"]["temperature"]) == ("local-llm", "local-test", 0.0)
    messages = db.get_messages_for_conversation(cid)
    assert [m["content"] for m in messages] == ["Hello."]


@pytest.mark.parametrize("trusted", [False, True])
def test_factory_keyword_is_separate_from_untrusted_payload(startup_api: SimpleNamespace, trusted: bool) -> None:
    """Direct legacy callers remain unknown while the typed keyword persists explicit origin."""
    kwargs = {"assistant_startup": AssistantStartup(source="explicit")} if trusted else {}
    cid = create_character_conversation(
        startup_api.db, conversation_data={"character_id": startup_api.character_id}, **kwargs,
    )
    row = startup_api.db.get_conversation_by_id(cid)
    assert decode_assistant_startup(row["assistant_startup_json"]).source == ("explicit" if trusted else "unknown")


def test_unrepresentable_origin_is_mapped_without_echoing_private_workspace_id(startup_api: SimpleNamespace) -> None:
    """A legacy long Workspace ID remains readable but cannot become an unsafe origin."""
    db = startup_api.db
    workspace_id = "PRIVATE-" + "x" * 1024
    db.upsert_workspace(workspace_id, "Legacy Workspace")
    before = _creation_counts(db)
    response = startup_api.client.post("/api/v1/chats/", json={"scope_type": "workspace", "workspace_id": workspace_id})
    assert response.status_code == 400, response.text
    assert "PRIVATE" not in response.text
    assert len(response.text) < 256
    assert _creation_counts(db) == before


@pytest.mark.parametrize("padded_owner", [False, True])
def test_explicit_persona_fork_retains_validated_root_and_memory_mode(
    startup_api: SimpleNamespace, padded_owner: bool,
) -> None:
    """Persona forks preserve legacy owner normalization and never copy parent identity."""
    db = startup_api.db
    root = db.add_conversation({"scope_type": "workspace", "workspace_id": "ws"})
    parent = db.add_conversation({
        "root_id": root, "scope_type": "workspace", "workspace_id": "ws",
        "character_id": startup_api.character_id, "client_id": " 1 " if padded_owner else "1",
    })
    message = db.add_message({"conversation_id": parent, "sender": "user", "content": "Fork here"})
    response = startup_api.client.post("/api/v1/chats/", json={
        "scope_type": "workspace", "workspace_id": "ws", "parent_conversation_id": parent,
        "forked_from_message_id": message, "assistant_kind": "persona", "assistant_id": "persona-a",
        "persona_memory_mode": "read_write", "character_id": None,
    })
    assert response.status_code == 201, response.text
    row = db.get_conversation_by_id(response.json()["id"])
    assert (row["root_id"], row["parent_conversation_id"], row["forked_from_message_id"]) == (root, parent, message)
    assert (row["assistant_id"], row["persona_memory_mode"]) == ("persona-a", "read_write")
    assert row["title"].startswith("First Persona Chat (")
    assert decode_assistant_startup(row["assistant_startup_json"]).source == "fork"


@pytest.mark.parametrize("field", ["parent_conversation_id", "forked_from_message_id"])
def test_empty_optional_lineage_keeps_legacy_omission_semantics(startup_api: SimpleNamespace, field: str) -> None:
    """Empty optional lineage strings must not become a new create rejection."""
    response = startup_api.client.post("/api/v1/chats/", json={"scope_type": "workspace", "workspace_id": "ws", field: ""})
    assert response.status_code == 201, response.text
    row = startup_api.db.get_conversation_by_id(response.json()["id"])
    assert row[field] is None
    assert decode_assistant_startup(row["assistant_startup_json"]).source == "workspace_default"


@pytest.mark.parametrize("character", [False, True])
def test_global_sync_creation_keeps_unknown_origin(
    monkeypatch: pytest.MonkeyPatch, chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service, character: bool,
) -> None:
    """Real global HTTP Sync capture/materialization never acquires local Workspace authority."""
    payload = {"character_id": chacha_db.add_character_card({"name": "Global"})} if character else {}
    with _chat_messages_app(monkeypatch, chacha_db=chacha_db, sync_service=sync_service) as client:
        response = client.post("/api/v1/chats/", json=payload)
    assert response.status_code == 201, response.text
    row = chacha_db.get_conversation_by_id(response.json()["id"])
    assert row["assistant_startup_json"] is None
    assert row["scope_type"] == "global"
    dataset_id = sync_service.profile(user_id="user-1").active_dataset_id
    envelopes = sync_service.store.list_envelopes_after(dataset_id, 0, domains=["chat.conversation"], limit=10)
    assert len(envelopes) == 1
    assert "assistant_startup" not in envelopes[0].payload
    assert "assistant_startup_json" not in envelopes[0].payload
    chacha_db.close_all_connections()
