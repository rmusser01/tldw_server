from datetime import datetime, timezone

import pytest
from fastapi import FastAPI, HTTPException

from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions as sessions
from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import GreetingSelectRequest
from tldw_Server_API.app.core.Character_Chat import character_conversation_factory
from tldw_Server_API.app.core.Character_Chat.character_conversation_factory import (
    build_materialized_behavior_controls,
)
from tldw_Server_API.app.core.DB_Management.chacha.conversation_resume_store import (
    build_materialized_behavior_settings,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import ConflictError, InputError


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "settings_row",
    [
        None,
        {
            "settings": {"greetingsChecksum": "bootstrap-only"},
            "last_modified": datetime(2026, 8, 25, tzinfo=timezone.utc),
        },
    ],
)
async def test_get_chat_settings_returns_empty_settings_without_user_overrides(
    settings_row: dict[str, object] | None,
) -> None:
    """Bootstrap-only settings are omitted from the user override response."""
    class _StubDB:
        """Return a known chat with the parametrized settings row."""

        def get_conversation_by_id(self, chat_id: str) -> dict[str, object]:
            """Return the requested global conversation."""
            return {
                "id": chat_id,
                "client_id": "1",
                "scope_type": "global",
                "character_id": None,
            }

        def get_conversation_settings(
            self,
            chat_id: str,
        ) -> dict[str, object] | None:
            """Return the configured persisted settings row."""
            return settings_row

    class _StubUser:
        """Represent the owner of the stub conversation."""

        id = "1"

    response = await sessions.get_chat_settings(
        chat_id="chat-with-default-settings",
        scope_type=None,
        workspace_id=None,
        db=_StubDB(),  # type: ignore[arg-type]
        current_user=_StubUser(),  # type: ignore[arg-type]
    )

    assert response.settings == {}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_chat_settings_keeps_unknown_chat_not_found() -> None:
    """A missing conversation remains a 404 instead of an empty response."""
    class _StubDB:
        """Return no conversation for the requested identifier."""

        def get_conversation_by_id(self, chat_id: str) -> None:
            """Report that the conversation does not exist."""
            return None

    class _StubUser:
        """Represent the caller of the missing-chat request."""

        id = "1"

    with pytest.raises(HTTPException) as exc_info:
        await sessions.get_chat_settings(
            chat_id="missing-chat",
            scope_type=None,
            workspace_id=None,
            db=_StubDB(),  # type: ignore[arg-type]
            current_user=_StubUser(),  # type: ignore[arg-type]
        )

    assert exc_info.value.status_code == 404


@pytest.mark.unit
def test_merge_conversation_settings_server_wins_on_equal_updated_at():
    server = {
        "schemaVersion": 2,
        "updatedAt": "2026-02-06T00:00:00Z",
        "authorNote": "server-note",
        "memoryScope": "shared",
    }
    incoming = {
        "schemaVersion": 2,
        "updatedAt": "2026-02-06T00:00:00Z",
        "authorNote": "incoming-note",
        "memoryScope": "both",
    }

    merged = sessions._merge_conversation_settings(server, incoming)

    assert merged["authorNote"] == "server-note"
    assert merged["memoryScope"] == "shared"
    assert merged["updatedAt"] == "2026-02-06T00:00:00Z"


@pytest.mark.unit
def test_merge_conversation_settings_incoming_wins_when_newer():
    server = {
        "schemaVersion": 2,
        "updatedAt": "2026-02-06T00:00:00Z",
        "greetingEnabled": True,
    }
    incoming = {
        "updatedAt": "2026-02-06T00:00:01Z",
        "greetingEnabled": False,
    }

    merged = sessions._merge_conversation_settings(server, incoming)

    assert merged["greetingEnabled"] is False
    assert merged["updatedAt"] == "2026-02-06T00:00:01Z"
    assert merged["schemaVersion"] == 2


@pytest.mark.unit
def test_merge_conversation_settings_incoming_loses_when_older():
    server = {
        "schemaVersion": 2,
        "updatedAt": "2026-02-06T00:00:10Z",
        "authorNote": "server-note",
    }
    incoming = {
        "updatedAt": "2026-02-06T00:00:00Z",
        "authorNote": "incoming-stale-note",
    }

    merged = sessions._merge_conversation_settings(server, incoming)

    assert merged["authorNote"] == "server-note"
    assert merged["updatedAt"] == "2026-02-06T00:00:10Z"


@pytest.mark.unit
def test_merge_conversation_settings_applies_untimestamped_patch_update():
    server = {
        "schemaVersion": 2,
        "updatedAt": "2026-02-06T00:00:00Z",
        "authorNote": "server-note",
        "memoryScope": "shared",
    }
    incoming = {
        "authorNote": "incoming-note-without-timestamp",
    }

    merged = sessions._merge_conversation_settings(server, incoming)

    assert merged["authorNote"] == "incoming-note-without-timestamp"
    assert merged["memoryScope"] == "shared"
    assert merged["schemaVersion"] == 2
    assert merged["updatedAt"] != "2026-02-06T00:00:00Z"
    assert sessions._parse_iso_timestamp(merged["updatedAt"]) is not None


@pytest.mark.unit
def test_merge_conversation_settings_character_memory_per_entry_timestamp():
    server = {
        "schemaVersion": 2,
        "updatedAt": "2026-02-06T00:00:00Z",
        "characterMemoryById": {
            "1": {"note": "server-primary", "updatedAt": "2026-02-06T00:00:00Z"},
            "2": {"note": "server-secondary", "updatedAt": "2026-02-06T00:00:00Z"},
        },
    }
    incoming = {
        "schemaVersion": 2,
        "updatedAt": "2026-02-06T00:00:00Z",
        "characterMemoryById": {
            "1": {"note": "incoming-primary-older", "updatedAt": "2026-02-05T23:59:59Z"},
            "2": {"note": "incoming-secondary-newer", "updatedAt": "2026-02-06T00:00:01Z"},
            "3": {"note": "incoming-third", "updatedAt": "2026-02-06T00:00:01Z"},
        },
    }

    merged = sessions._merge_conversation_settings(server, incoming)
    memory = merged["characterMemoryById"]

    assert memory["1"]["note"] == "server-primary"
    assert memory["2"]["note"] == "incoming-secondary-newer"
    assert memory["3"]["note"] == "incoming-third"


@pytest.mark.unit
def test_merge_conversation_settings_preserves_unknown_keys():
    server = {
        "schemaVersion": 2,
        "updatedAt": "2026-02-06T00:00:00Z",
        "serverCustomFlag": "keep-me",
    }
    incoming = {
        "updatedAt": "2026-02-06T00:00:10Z",
        "clientCustomBlock": {"a": 1},
    }

    merged = sessions._merge_conversation_settings(server, incoming)

    assert merged["serverCustomFlag"] == "keep-me"
    assert merged["clientCustomBlock"] == {"a": 1}
    assert merged["schemaVersion"] == 2


@pytest.mark.unit
def test_public_chat_settings_hides_internal_resume_contract_keys_without_mutation():
    stored = {
        "schemaVersion": 2,
        "authorNote": "visible",
        "roleplayResumeV1": {"resumeEligible": True},
        "roleplayBehaviorV1": {"schemaVersion": 1, "values": {}},
    }

    public = sessions._public_chat_settings(stored)

    assert public == {"schemaVersion": 2, "authorNote": "visible"}
    assert "roleplayResumeV1" in stored
    assert "roleplayBehaviorV1" in stored


@pytest.mark.unit
def test_merge_conversation_settings_preserves_assistant_overlay_payload():
    server = {
        "schemaVersion": 2,
        "updatedAt": "2026-05-22T20:00:00Z",
        "assistantOverlay": {
            "kind": "character",
            "id": "char-1",
            "name": "Server Overlay",
            "system_prompt_snapshot": "server snapshot",
            "updatedAt": "2026-05-22T20:00:00Z",
        },
    }
    incoming = {
        "schemaVersion": 2,
        "updatedAt": "2026-05-22T20:00:01Z",
        "assistantOverlay": {
            "kind": "persona",
            "id": "persona-9",
            "name": "Incoming Overlay",
            "system_prompt_snapshot": "incoming snapshot",
            "updatedAt": "2026-05-22T20:00:01Z",
        },
    }

    merged = sessions._merge_conversation_settings(server, incoming)

    assert merged["assistantOverlay"] == incoming["assistantOverlay"]
    assert merged["updatedAt"] == "2026-05-22T20:00:01Z"


@pytest.mark.unit
def test_persist_auto_summary_settings_upsert_does_not_touch_conversation_metadata():
    class _StubDB:
        def __init__(self) -> None:
            self.upsert_calls = 0
            self.update_conversation_calls = 0
            self.expected_settings_version: int | None = None

        def upsert_conversation_settings(
            self,
            conversation_id: str,
            settings: dict[str, object],
            *,
            expected_settings_version: int | None = None,
        ) -> bool:
            self.upsert_calls += 1
            self.expected_settings_version = expected_settings_version
            return True

        def update_conversation(self, conversation_id: str, update_data: dict[str, object], expected_version: int) -> bool:
            self.update_conversation_calls += 1
            return True

    db = _StubDB()
    settings = {"schemaVersion": 2, "updatedAt": "2026-02-06T00:00:00Z"}

    sessions._persist_auto_summary_to_settings(
        db=db,
        chat_id="chat-1",
        settings=settings,
        content="summary content",
        source_from_id="msg-1",
        source_to_id="msg-2",
        threshold=10,
        window=20,
        compressed_count=3,
        expected_settings_version=7,
    )

    assert db.upsert_calls == 1
    assert db.expected_settings_version == 7
    assert db.update_conversation_calls == 0


@pytest.mark.unit
def test_convert_db_conversation_to_response_includes_settings_payload():
    conv = {
        "id": "chat-1",
        "character_id": 7,
        "created_at": datetime.now(timezone.utc),
        "last_modified": datetime.now(timezone.utc),
        "version": 3,
        "message_count": 12,
    }
    settings = {"greetingEnabled": True, "authorNote": "test"}

    response = sessions._convert_db_conversation_to_response(conv, settings=settings)

    assert response.id == "chat-1"
    assert response.settings == settings


@pytest.mark.unit
def test_convert_db_conversation_to_response_defaults_settings_none():
    conv = {
        "id": "chat-2",
        "character_id": 9,
        "created_at": datetime.now(timezone.utc),
        "last_modified": datetime.now(timezone.utc),
        "version": 1,
    }

    response = sessions._convert_db_conversation_to_response(conv)

    assert response.id == "chat-2"
    assert response.settings is None


@pytest.mark.unit
def test_convert_db_conversation_to_response_does_not_infer_tracked_identity_from_character_id():
    conv = {
        "id": "chat-3",
        "character_id": 11,
        "assistant_kind": None,
        "assistant_id": None,
        "created_at": datetime.now(timezone.utc),
        "last_modified": datetime.now(timezone.utc),
        "version": 1,
    }

    response = sessions._convert_db_conversation_to_response(conv)

    assert response.character_id == 11
    assert response.assistant_kind is None
    assert response.assistant_id is None


@pytest.mark.unit
def test_openapi_exposes_include_settings_query_params():
    app = FastAPI()
    app.include_router(sessions.router, prefix="/api/v1/chats")
    schema = app.openapi()

    detail_params = schema["paths"]["/api/v1/chats/{chat_id}"]["get"]["parameters"]
    detail_param_names = {param["name"] for param in detail_params}
    assert "include_settings" in detail_param_names

    list_params = schema["paths"]["/api/v1/chats/"]["get"]["parameters"]
    list_param_names = {param["name"] for param in list_params}
    assert "include_settings" in list_param_names


@pytest.mark.unit
def test_openapi_exposes_chat_trash_query_params_and_routes():
    app = FastAPI()
    app.include_router(sessions.router, prefix="/api/v1/chats")
    schema = app.openapi()

    list_params = schema["paths"]["/api/v1/chats/"]["get"]["parameters"]
    list_param_names = {param["name"] for param in list_params}
    assert "include_deleted" in list_param_names
    assert "deleted_only" in list_param_names

    delete_params = schema["paths"]["/api/v1/chats/{chat_id}"]["delete"]["parameters"]
    delete_param_names = {param["name"] for param in delete_params}
    assert "hard_delete" in delete_param_names

    assert "/api/v1/chats/{chat_id}/restore" in schema["paths"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_select_greeting_returns_500_when_settings_persist_fails():
    class _StubDB:
        def __init__(self) -> None:
            self.expected_settings_version: int | None = None

        def get_conversation_by_id(self, chat_id: str) -> dict[str, object]:
            return {"id": chat_id, "client_id": "1", "character_id": 7}

        def get_character_card_by_id(self, character_id: int) -> dict[str, object]:
            return {"id": character_id, "name": "Test Character", "first_message": "Hello!", "alternate_greetings": ["Hi!"]}

        def get_conversation_settings(self, chat_id: str) -> dict[str, object]:
            return {"settings": {}, "settings_version": 4}

        def upsert_conversation_settings(
            self,
            chat_id: str,
            settings: dict[str, object],
            *,
            expected_settings_version: int | None = None,
        ) -> bool:
            self.expected_settings_version = expected_settings_version
            return False

    class _StubUser:
        id = "1"

    db = _StubDB()
    with pytest.raises(HTTPException) as exc_info:
        await sessions.select_greeting(
            chat_id="chat-1",
            body=GreetingSelectRequest(index=0),
            db=db,  # type: ignore[arg-type]
            current_user=_StubUser(),  # type: ignore[arg-type]
        )

    assert exc_info.value.status_code == 500
    assert "Failed to persist greeting selection" in str(exc_info.value.detail)
    assert db.expected_settings_version == 4


@pytest.mark.unit
@pytest.mark.asyncio
async def test_select_greeting_returns_409_on_concurrent_settings_change():
    class _StubDB:
        def get_conversation_by_id(self, chat_id: str) -> dict[str, object]:
            return {"id": chat_id, "client_id": "1", "character_id": 7}

        def get_character_card_by_id(self, character_id: int) -> dict[str, object]:
            return {
                "id": character_id,
                "name": "Test Character",
                "first_message": "Hello!",
            }

        def get_conversation_settings(self, chat_id: str) -> dict[str, object]:
            return {"settings": {}, "settings_version": 4}

        def upsert_conversation_settings(
            self,
            chat_id: str,
            settings: dict[str, object],
            *,
            expected_settings_version: int | None = None,
        ) -> bool:
            raise ConflictError("stale settings")

    class _StubUser:
        id = "1"

    with pytest.raises(HTTPException) as exc_info:
        await sessions.select_greeting(
            chat_id="chat-1",
            body=GreetingSelectRequest(index=0),
            db=_StubDB(),  # type: ignore[arg-type]
            current_user=_StubUser(),  # type: ignore[arg-type]
        )

    assert exc_info.value.status_code == 409


@pytest.mark.unit
def test_prompt_completion_settings_inventory_classifies_every_consumed_control():
    expected_behavior_fields = {
        "assistantOverlay",
        "authorNote",
        "authorNoteEnabled",
        "authorNoteExcludeFromPrompt",
        "authorNoteGmOnly",
        "authorNoteInjectionPosition",
        "authorNotePlacement",
        "authorNotePosition",
        "autoSummaryEnabled",
        "autoSummaryMessageThreshold",
        "autoSummaryRecentWindow",
        "autoSummaryThresholdMessages",
        "autoSummaryWindowMessages",
        "characterMemoryById",
        "chatGenerationOverride",
        "chatPresetOverrideId",
        "conversationContext",
        "generationOverrides",
        "greetingEnabled",
        "greetingScope",
        "greetingSelectionId",
        "memoryScope",
        "model",
        "participantCharacterIds",
        "participant_character_ids",
        "pinnedMessageIds",
        "presetScope",
        "promptPreset",
        "prompt_preset",
        "provider",
        "summary",
        "turnTakingMode",
        "useCharacterDefault",
    }
    inventory = getattr(sessions, "PROMPT_COMPLETION_SETTING_CLASSIFICATION", {})
    assert expected_behavior_fields <= {
        key for key, classification in inventory.items() if classification == "behavior"
    }


@pytest.mark.unit
def test_materialized_behavior_record_rejects_oversize_payload():
    values = {
        "base_snapshot": {
            "schema_version": 1,
            "digest": "sha256:" + ("0" * 64),
        },
        "behavior_controls": {
            "applied_overrides": {},
            "author_note": {
                "enabled": True,
                "gm_only": False,
                "exclude_from_prompt": False,
                "position": "before_system",
            },
            "auto_summary": {
                "enabled": False,
                "threshold_messages": 40,
                "window_messages": 12,
            },
            "greeting": {
                "enabled": True,
                "scope": "chat",
                "selection_id": None,
                "use_character_default": True,
            },
            "memory_scope": "shared",
            "pinned_message_ids": [],
            "preset_scope": "character",
            "prompt_context": {},
            "turn_taking_mode": "single",
        },
        "effective_completion": {
            "provider": "local-llm",
            "model": "local-test",
            "sampling": {
                "temperature": 0.7,
                "top_p": 1.0,
                "repetition_penalty": 1.0,
                "stop": [],
            },
        },
        "memory": {"author_note": "x" * (1024 * 1024)},
    }
    with pytest.raises(InputError, match="exceeds maximum"):
        build_materialized_behavior_settings(values)


@pytest.mark.unit
def test_materialized_reference_ids_are_deduplicated_and_bounded_before_lookup():
    normalize_participants = getattr(
        character_conversation_factory,
        "normalize_materialized_participant_ids",
        None,
    )
    normalize_world_books = getattr(
        character_conversation_factory,
        "normalize_materialized_world_book_ids",
        None,
    )
    assert callable(normalize_participants)
    assert callable(normalize_world_books)
    assert normalize_participants(1, [1, "2", 2, 3, "3"]) == [1, 2, 3]
    assert normalize_world_books([1, "1", 2, 2, 3]) == [1, 2, 3]
    with pytest.raises(InputError, match="at most 33"):
        normalize_participants(1, list(range(2, 35)))
    with pytest.raises(InputError, match="at most 64"):
        normalize_world_books(list(range(1, 66)))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("authorNoteEnabled", "false"),
        ("authorNoteGmOnly", 1),
        ("authorNoteExcludeFromPrompt", 0),
        ("greetingEnabled", "true"),
        ("useCharacterDefault", 1),
        ("autoSummaryEnabled", "false"),
    ],
)
def test_materialized_behavior_controls_reject_non_boolean_known_flags(
    field: str,
    value: object,
) -> None:
    with pytest.raises(InputError, match=field):
        build_materialized_behavior_controls({field: value})


@pytest.mark.unit
def test_settings_endpoint_rejects_non_boolean_behavior_flags() -> None:
    with pytest.raises(HTTPException) as exc_info:
        sessions._validate_chat_settings_payload({"greetingEnabled": "false"})

    assert exc_info.value.status_code == 422
    assert "greetingEnabled" in str(exc_info.value.detail)
