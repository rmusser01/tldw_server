"""Creation-time behavior-snapshot and readiness integration coverage."""

from __future__ import annotations

import json
from dataclasses import replace
from unittest import mock

import pytest

from tldw_Server_API.app.core.Character_Chat import character_conversation_factory
from tldw_Server_API.app.core.Character_Chat.character_behavior_snapshot import (
    build_behavior_snapshot,
)
from tldw_Server_API.app.core.Character_Chat.character_conversation_factory import (
    create_character_conversation,
)
from tldw_Server_API.app.core.Character_Chat.modules.character_chat import (
    start_new_chat_session,
)
from tldw_Server_API.app.core.Character_Chat.modules.character_io import (
    load_chat_history_from_file_and_save_to_db,
)
from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService
from tldw_Server_API.app.core.DB_Management.chacha.conversation_resume_store import (
    ConversationResumeStore,
    build_materialized_behavior_settings,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    BackendType,
    InputError,
)


def _create_behavior_sources(db):
    assert db.upsert_prompt_preset(
        preset_id="snapshot-cinematic",
        name="Snapshot Cinematic",
        section_order=["identity", "system_prompt", "scenario"],
        section_templates={
            "identity": "Act as {{char}} for {{user}}.",
            "system_prompt": "{{system_prompt}}",
            "scenario": "Scene: {{scenario}}",
        },
    )
    primary_id = db.add_character_card(
        {
            "name": "Ari",
            "description": "A patient archivist",
            "personality": "Exacting and warm",
            "scenario": "A rain-soaked archive",
            "system_prompt": "Preserve continuity.",
            "first_message": "Welcome back, {{user}}.",
            "alternate_greetings": ["The archive remembers you, {{user}}."],
            "message_example": "{{char}}: The record is intact.",
            "post_history_instructions": "Never invent prior events.",
            "extensions": {
                "token": "legitimate-token-content",
                "token_budget": 777,
                "promptToken": "prompt-behavior-id",
                "pageToken": "page-opaque-id",
                "nextPageToken": "next-page-opaque-id",
                "validationToken": "validation-behavior-id",
                "undoToken": "undo-behavior-id",
                "tldw": {
                    "generation": {
                        "temperature": "0.35",
                        "topP": "0.8",
                        "repetitionPenalty": "1.15",
                        "stopSequences": "<END>; <DONE>",
                    }
                }
            },
        }
    )
    second_id = db.add_character_card(
        {
            "name": "Bex",
            "description": "A skeptical field researcher",
            "personality": "Direct",
            "scenario": "The same archive",
            "system_prompt": "Challenge uncertain claims.",
            "first_message": "Show me the evidence.",
            "message_example": "{{char}}: Which source proves that?",
            "post_history_instructions": "Ask for sources.",
            "extensions": {
                "tldw": {
                    "generation": {
                        "temperature": "0.45",
                        "topP": "0.65",
                        "repetitionPenalty": "1.05",
                        "stopSequences": ["<BEX>"],
                    }
                }
            },
        }
    )
    exemplar = db.add_character_exemplar(
        primary_id,
        {
            "id": "ari-exemplar-v1",
            "text": "Ari checks the accession number before answering.",
            "scenario": "small_talk",
            "emotion": "neutral",
        },
    )
    world_books = WorldBookService(db)
    world_book_id = world_books.create_world_book(
        "Archive Canon",
        description="Immutable archive facts",
        scan_depth=4,
        token_budget=321,
    )
    entry_id = world_books.add_world_book_entry(
        world_book_id,
        keywords=["archive"],
        content="The east vault flooded in 2041.",
        priority=9,
    )
    assert world_books.attach_to_character(world_book_id, primary_id)["success"] is True
    return {
        "primary_id": primary_id,
        "second_id": second_id,
        "exemplar_id": exemplar["id"],
        "world_book_id": world_book_id,
        "world_book_entry_id": entry_id,
    }


def _snapshot_storage_bytes(db, conversation_id: str) -> tuple[str, str]:
    with db.transaction() as conn:
        snapshot = conn.execute(
            "SELECT canonical_json FROM conversation_behavior_snapshots WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        settings = conn.execute(
            "SELECT settings_json FROM conversation_settings WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
    return str(snapshot[0]), str(settings[0])


@pytest.mark.integration
def test_api_creation_captures_all_sources_and_redacts_snapshot_body(
    test_client,
    auth_headers,
    character_db,
):
    sources = _create_behavior_sources(character_db)
    response = test_client.post(
        "/api/v1/chats/?seed_first_message=true&greeting_strategy=alternate_index&alternate_index=0",
        headers=auth_headers,
        json={
            "character_id": sources["primary_id"],
            "participant_character_ids": [sources["second_id"]],
            "prompt_preset_id": "snapshot-cinematic",
            "memory_by_character_id": {
                str(sources["primary_id"]): "Ari remembers the brass key.",
                str(sources["second_id"]): "Bex remembers the flooded stairs.",
            },
            "provider": "local-llm",
            "model": "local-test",
            "temperature": 0.0,
            "top_p": 0.0,
            "repetition_penalty": 0.0,
            "stop": [],
            "title": "Immutable archive chat",
        },
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["behavior_snapshot"]["status"] == "valid"
    assert body["behavior_snapshot"]["schema_version"] == 1
    assert body["behavior_snapshot"]["digest"].startswith("sha256:")
    assert body["resume_eligible"] is True
    assert body["resume_ineligible_reason"] is None
    assert body["settings_version"] == 1
    assert body["history_version"] == 2
    assert body["message_count"] == 1
    assert body["tail"]["message_id"]
    assert body["tail"]["message_version"] == 1
    assert "canonical_json" not in body["behavior_snapshot"]
    assert "payload" not in body["behavior_snapshot"]

    state = character_db.get_roleplay_resume_state(body["id"])
    participants = state["behavior_snapshot"]["payload"]["participants"]
    assert [participant["source"]["id"] for participant in participants] == [
        str(sources["primary_id"]),
        str(sources["second_id"]),
    ]
    primary = participants[0]
    assert primary["prompt"]["prompt_relevant_extensions"]["prompt_preset"]["preset_id"] == "snapshot-cinematic"
    assert primary["exemplars"][0]["id"] == sources["exemplar_id"]
    assert primary["world_books"][0]["id"] == sources["world_book_id"]
    assert primary["world_books"][0]["entries"][0]["id"] == sources["world_book_entry_id"]
    assert primary["default_memory"]["content"] == "Ari remembers the brass key."
    assert primary["generation_defaults"] == {
        "source": primary["source"],
        "sampling": {
            "temperature": 0.35,
            "top_p": 0.8,
            "repetition_penalty": 1.15,
            "stop": ["<END>", "<DONE>"],
        },
    }
    secondary = participants[1]
    assert secondary["generation_defaults"] == {
        "source": secondary["source"],
        "sampling": {
            "temperature": 0.45,
            "top_p": 0.65,
            "repetition_penalty": 1.05,
            "stop": ["<BEX>"],
        },
    }
    character_extensions = primary["prompt"]["prompt_relevant_extensions"][
        "character_extensions"
    ]
    assert character_extensions["token"] == "legitimate-token-content"
    assert character_extensions["token_budget"] == 777
    assert character_extensions["promptToken"] == "prompt-behavior-id"
    assert character_extensions["pageToken"] == "page-opaque-id"
    assert character_extensions["nextPageToken"] == "next-page-opaque-id"
    assert character_extensions["validationToken"] == "validation-behavior-id"
    assert character_extensions["undoToken"] == "undo-behavior-id"
    assert primary["greeting"] == {
        "content": "The archive remembers you, {{user}}.",
        "source": "alternate",
        "source_index": 0,
    }
    assert state["effective_completion"] == {
        "provider": "local-llm",
        "model": "local-test",
        "sampling": {
            "temperature": 0.0,
            "top_p": 0.0,
            "repetition_penalty": 0.0,
            "stop": [],
        },
    }
    snapshot_bytes, settings_bytes = _snapshot_storage_bytes(character_db, body["id"])
    assert "legitimate-token-content" in snapshot_bytes
    assert "legitimate-token-content" not in settings_bytes


@pytest.mark.integration
def test_configured_defaults_and_generation_aliases_create_valid_effective_settings(
    character_db,
    monkeypatch,
):
    monkeypatch.delenv("DEFAULT_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("DEFAULT_MODEL_LOCAL_LLM", raising=False)
    monkeypatch.delenv("CHAR_CHAT_MODEL", raising=False)
    app_config = {
        "llm_api_settings": {"default_api": "local_llm"},
        "local_llm": {"model": "configured-local-model"},
    }
    monkeypatch.setattr(
        character_conversation_factory,
        "ensure_app_config",
        lambda: app_config,
        raising=False,
    )
    character_id = character_db.add_character_card(
        {
            "name": "Configured Ari",
            "extensions": {
                "tldw": {
                    "generation": {
                        "temperature": "0.25",
                        "topP": "0.55",
                        "repetitionPenalty": "1.2",
                        "stopSequences": "<ONE>; <TWO>",
                    }
                }
            },
        }
    )

    conversation_id = create_character_conversation(
        character_db,
        conversation_data={
            "character_id": character_id,
            "title": "Configured defaults",
            "client_id": str(character_db.client_id),
        },
        conversation_settings={
            "token": "legitimate-settings-token",
            "token_budget": 2048,
        },
    )

    state = character_db.get_roleplay_resume_state(conversation_id)
    expected_sampling = {
        "temperature": 0.25,
        "top_p": 0.55,
        "repetition_penalty": 1.2,
        "stop": ["<ONE>", "<TWO>"],
    }
    assert state["resume_eligible"] is True
    assert state["effective_completion"] == {
        "provider": "local-llm",
        "model": "configured-local-model",
        "sampling": expected_sampling,
    }
    assert state["settings"]["token"] == "legitimate-settings-token"
    assert state["settings"]["token_budget"] == 2048
    generation = state["behavior_snapshot"]["payload"]["participants"][0][
        "generation_defaults"
    ]
    assert generation == {
        "source": {"kind": "character", "id": str(character_id), "version": 1},
        "sampling": expected_sampling,
    }


@pytest.mark.integration
def test_api_omitted_sampling_preserves_character_generation_defaults(
    test_client,
    auth_headers,
    character_db,
):
    character_id = character_db.add_character_card(
        {
            "name": "Endpoint Defaults Ari",
            "extensions": {
                "tldw": {
                    "generation": {
                        "temperature": "0.25",
                        "topP": "0.55",
                        "repetitionPenalty": "1.2",
                        "stopSequences": "<ONE>; <TWO>",
                    }
                }
            },
        }
    )

    response = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={
            "character_id": character_id,
            "provider": "local-llm",
            "model": "local-test",
        },
    )

    assert response.status_code == 201, response.text
    state = character_db.get_roleplay_resume_state(response.json()["id"])
    assert state["effective_completion"]["sampling"] == {
        "temperature": 0.25,
        "top_p": 0.55,
        "repetition_penalty": 1.2,
        "stop": ["<ONE>", "<TWO>"],
    }


@pytest.mark.integration
def test_api_partial_sampling_override_merges_with_character_generation_defaults(
    test_client,
    auth_headers,
    character_db,
):
    character_id = character_db.add_character_card(
        {
            "name": "Partial Override Ari",
            "extensions": {
                "tldw": {
                    "generation": {
                        "temperature": "0.25",
                        "topP": "0.55",
                        "repetitionPenalty": "1.2",
                        "stopSequences": "<ONE>; <TWO>",
                    }
                }
            },
        }
    )

    response = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={
            "character_id": character_id,
            "provider": "local-llm",
            "model": "local-test",
            "temperature": 0.9,
        },
    )

    assert response.status_code == 201, response.text
    state = character_db.get_roleplay_resume_state(response.json()["id"])
    assert state["effective_completion"]["sampling"] == {
        "temperature": 0.9,
        "top_p": 0.55,
        "repetition_penalty": 1.2,
        "stop": ["<ONE>", "<TWO>"],
    }


@pytest.mark.integration
@pytest.mark.parametrize(
    ("extensions", "expected_preset_id", "expected_selection_source"),
    [
        ({}, "default", "default"),
        ({"tldw": {"promptPreset": "default"}}, "default", "character"),
        ({"tldw": {"promptPreset": "st_default"}}, "st_default", "character"),
    ],
)
def test_creation_materializes_effective_builtin_prompt_preset(
    character_db,
    extensions,
    expected_preset_id,
    expected_selection_source,
):
    from tldw_Server_API.app.core.Character_Chat.modules.character_prompt_presets import (
        get_builtin_presets,
    )

    character_id = character_db.add_character_card(
        {"name": "Preset Ari", "extensions": extensions}
    )
    conversation_id = create_character_conversation(
        character_db,
        conversation_data={
            "character_id": character_id,
            "title": "Preset snapshot",
            "client_id": str(character_db.client_id),
        },
        provider="local-llm",
        model="local-test",
    )

    state = character_db.get_roleplay_resume_state(conversation_id)
    materialized = state["behavior_snapshot"]["payload"]["participants"][0][
        "prompt"
    ]["prompt_relevant_extensions"]["prompt_preset"]
    expected = next(
        preset
        for preset in get_builtin_presets()
        if preset["preset_id"] == expected_preset_id
    )
    assert materialized["preset_id"] == expected_preset_id
    assert materialized["name"] == expected["name"]
    assert materialized["section_order"] == expected["section_order"]
    assert materialized["section_templates"] == expected["section_templates"]
    assert materialized["selection_source"] == expected_selection_source
    assert materialized["source"] == {
        "kind": "builtin_prompt_preset",
        "id": expected_preset_id,
        "version": 1,
    }


@pytest.mark.integration
def test_prompt_preset_id_is_trimmed_or_rejected_before_materialization(
    test_client,
    auth_headers,
    character_db,
):
    character_id = character_db.add_character_card({"name": "Preset Boundary Ari"})

    rejected = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={
            "character_id": character_id,
            "provider": "local-llm",
            "model": "local-test",
            "prompt_preset_id": "   ",
        },
    )
    assert rejected.status_code == 400, rejected.text
    assert character_db.get_conversations_for_character(character_id) == []

    conversation_id = create_character_conversation(
        character_db,
        conversation_data={
            "character_id": character_id,
            "title": "Normalized preset",
            "client_id": str(character_db.client_id),
        },
        prompt_preset_id="  st_default  ",
        provider="local-llm",
        model="local-test",
    )
    state = character_db.get_roleplay_resume_state(conversation_id)
    assert state["settings"]["chatPresetOverrideId"] == "st_default"
    materialized = state["behavior_snapshot"]["payload"]["participants"][0][
        "prompt"
    ]["prompt_relevant_extensions"]["prompt_preset"]
    assert materialized["preset_id"] == "st_default"
    assert materialized["selection_source"] == "creation_request"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("provider", "model", "catalog_result"),
    [
        ("definitely-not-a-provider", "model", None),
        ("local-llm", "known-invalid-model", False),
    ],
)
def test_invalid_provider_or_model_is_explicitly_ineligible(
    character_db,
    monkeypatch,
    provider,
    model,
    catalog_result,
):
    monkeypatch.setattr(
        character_conversation_factory,
        "is_model_known_for_provider",
        lambda *_args: catalog_result,
        raising=False,
    )
    character_id = character_db.add_character_card({"name": "Invalid Settings Ari"})

    conversation_id = create_character_conversation(
        character_db,
        conversation_data={
            "character_id": character_id,
            "title": "Invalid settings",
            "client_id": str(character_db.client_id),
        },
        provider=provider,
        model=model,
        sampling={
            "temperature": 0.0,
            "top_p": 0.0,
            "repetition_penalty": 0.0,
            "stop": [],
        },
    )

    state = character_db.get_roleplay_resume_state(conversation_id)
    assert state["resume_eligible"] is False
    assert state["resume_ineligible_reason"] == "incomplete_effective_settings"
    assert state["effective_completion"] is None


@pytest.mark.integration
def test_explicit_provider_model_mutation_repairs_incomplete_resume_authority(
    test_client,
    auth_headers,
    character_db,
):
    character_id = character_db.add_character_card({"name": "Repairable Ari"})
    conversation_id = create_character_conversation(
        character_db,
        conversation_data={
            "character_id": character_id,
            "title": "Repair incomplete completion",
            "client_id": str(character_db.client_id),
        },
        provider="definitely-not-a-provider",
        model="missing-model",
    )
    before = character_db.get_roleplay_resume_state(conversation_id)
    assert before["resume_eligible"] is False
    assert before["materialized_settings"] is None

    response = test_client.put(
        f"/api/v1/chats/{conversation_id}/settings",
        headers=auth_headers,
        json={
            "settings": {
                "provider": "local-llm",
                "model": "local-test",
            }
        },
    )

    assert response.status_code == 200, response.text
    after = character_db.get_roleplay_resume_state(conversation_id)
    assert after["resume_eligible"] is True
    assert after["settings_version"] == before["settings_version"] + 1
    assert after["effective_completion"]["provider"] == "local-llm"
    assert after["effective_completion"]["model"] == "local-test"
    assert after["materialized_settings"]["values"]["effective_completion"] == (
        after["effective_completion"]
    )


@pytest.mark.integration
def test_incomplete_greeting_selection_freezes_text_for_later_completion_repair(
    test_client,
    auth_headers,
    character_db,
) -> None:
    character_id = character_db.add_character_card(
        {
            "name": "Frozen Greeting Repair",
            "first_message": "Default greeting",
            "alternate_greetings": ["Selected before repair"],
        }
    )
    conversation_id = create_character_conversation(
        character_db,
        conversation_data={
            "character_id": character_id,
            "title": "Freeze greeting before repair",
            "client_id": str(character_db.client_id),
        },
        provider="definitely-not-a-provider",
        model="missing-model",
    )

    selected = test_client.put(
        f"/api/v1/chats/{conversation_id}/greetings/select",
        headers=auth_headers,
        json={"index": 1},
    )
    assert selected.status_code == 200, selected.text
    selected_state = character_db.get_roleplay_resume_state(conversation_id)
    assert selected_state["resume_eligible"] is False
    assert selected_state["materialized_settings"] is None
    assert "roleplayPendingGreetingV1" in selected_state["settings"]
    public_settings = test_client.get(
        f"/api/v1/chats/{conversation_id}/settings",
        headers=auth_headers,
    )
    assert public_settings.status_code == 200, public_settings.text
    assert "roleplayPendingGreetingV1" not in public_settings.json()["settings"]

    assert character_db.update_character_card(
        character_id,
        {"alternate_greetings": ["Mutated after selection"]},
        expected_version=1,
    )
    repaired = test_client.put(
        f"/api/v1/chats/{conversation_id}/settings",
        headers=auth_headers,
        json={
            "settings": {
                "provider": "local-llm",
                "model": "local-test",
            }
        },
    )

    assert repaired.status_code == 200, repaired.text
    repaired_state = character_db.get_roleplay_resume_state(conversation_id)
    assert repaired_state["resume_eligible"] is True
    assert "roleplayPendingGreetingV1" not in repaired_state["settings"]
    greeting = repaired_state["materialized_settings"]["values"]["greeting"]
    assert greeting == {
        "content": "Selected before repair",
        "selection_id": "greeting:1:selected",
        "source": "alternate_greeting",
        "source_index": 1,
        "character_version": 1,
    }


@pytest.mark.integration
def test_tampered_incomplete_greeting_freeze_fails_repair_atomically(
    test_client,
    auth_headers,
    character_db,
) -> None:
    character_id = character_db.add_character_card(
        {
            "name": "Tampered Greeting Repair",
            "first_message": "Default greeting",
            "alternate_greetings": ["Authorized selection"],
        }
    )
    conversation_id = create_character_conversation(
        character_db,
        conversation_data={
            "character_id": character_id,
            "title": "Reject tampered greeting freeze",
            "client_id": str(character_db.client_id),
        },
        provider="definitely-not-a-provider",
        model="missing-model",
    )
    selected = test_client.put(
        f"/api/v1/chats/{conversation_id}/greetings/select",
        headers=auth_headers,
        json={"index": 1},
    )
    assert selected.status_code == 200, selected.text
    selected_state = character_db.get_roleplay_resume_state(conversation_id)
    tampered_settings = dict(selected_state["settings"])
    pending = dict(tampered_settings["roleplayPendingGreetingV1"])
    pending_values = dict(pending["values"])
    pending_greeting = dict(pending_values["greeting"])
    pending_greeting["content"] = "Injected greeting"
    pending_values["greeting"] = pending_greeting
    pending["values"] = pending_values
    tampered_settings["roleplayPendingGreetingV1"] = pending
    with character_db.transaction() as conn:
        conn.execute(
            "UPDATE conversation_settings SET settings_json = ? WHERE conversation_id = ?",
            (json.dumps(tampered_settings), conversation_id),
        )
    before_repair = character_db.get_roleplay_resume_state(conversation_id)

    repaired = test_client.put(
        f"/api/v1/chats/{conversation_id}/settings",
        headers=auth_headers,
        json={
            "settings": {
                "provider": "local-llm",
                "model": "local-test",
            }
        },
    )

    assert repaired.status_code in {400, 409, 422}, repaired.text
    after_repair = character_db.get_roleplay_resume_state(conversation_id)
    assert after_repair["settings_version"] == before_repair["settings_version"]
    assert after_repair["settings"] == before_repair["settings"]
    assert after_repair["resume_eligible"] is False


@pytest.mark.integration
def test_client_cannot_supply_pending_greeting_authority(
    test_client,
    auth_headers,
    character_db,
) -> None:
    character_id = character_db.add_character_card({"name": "Reserved Greeting State"})
    conversation_id = create_character_conversation(
        character_db,
        conversation_data={
            "character_id": character_id,
            "title": "Reserved greeting state",
            "client_id": str(character_db.client_id),
        },
        provider="definitely-not-a-provider",
        model="missing-model",
    )
    before = character_db.get_roleplay_resume_state(conversation_id)

    response = test_client.put(
        f"/api/v1/chats/{conversation_id}/settings",
        headers=auth_headers,
        json={
            "settings": {
                "roleplayPendingGreetingV1": {
                    "schemaVersion": 1,
                    "digest": "sha256:" + ("0" * 64),
                    "values": {},
                }
            }
        },
    )

    assert response.status_code == 422, response.text
    after = character_db.get_roleplay_resume_state(conversation_id)
    assert after["settings_version"] == before["settings_version"]
    assert after["settings"] == before["settings"]


@pytest.mark.integration
def test_pin_writer_preserves_ineligibility_when_completion_is_unresolved(
    test_client,
    auth_headers,
    character_db,
):
    character_id = character_db.add_character_card({"name": "Incomplete Pin Ari"})
    conversation_id = create_character_conversation(
        character_db,
        conversation_data={
            "character_id": character_id,
            "title": "Incomplete pin",
            "client_id": str(character_db.client_id),
        },
        provider="definitely-not-a-provider",
        model="missing-model",
        initial_messages=[
            {"id": "incomplete-pin-target", "sender": "user", "content": "Pin me"},
        ],
    )
    before = character_db.get_roleplay_resume_state(conversation_id)
    assert before["resume_eligible"] is False

    response = test_client.put(
        "/api/v1/messages/incomplete-pin-target",
        params={"expected_version": 1},
        headers=auth_headers,
        json={"pinned": True},
    )

    assert response.status_code == 200, response.text
    after = character_db.get_roleplay_resume_state(conversation_id)
    assert after["resume_eligible"] is False
    assert after["resume_ineligible_reason"] == "incomplete_effective_settings"
    assert after["effective_completion"] is None
    assert after["settings_version"] == before["settings_version"] + 1
    assert after["history_version"] == before["history_version"] + 1
    assert after["settings"]["pinnedMessageIds"] == ["incomplete-pin-target"]
    assert character_db.get_message_by_id("incomplete-pin-target")["version"] == 2


@pytest.mark.integration
def test_incomplete_completion_does_not_bypass_reference_validation(
    test_client,
    auth_headers,
    character_db,
) -> None:
    character_id = character_db.add_character_card({"name": "Incomplete Reference"})
    conversation_id = create_character_conversation(
        character_db,
        conversation_data={
            "character_id": character_id,
            "title": "Incomplete reference",
            "client_id": str(character_db.client_id),
        },
        provider="definitely-not-a-provider",
        model="missing-model",
    )
    before = character_db.get_roleplay_resume_state(conversation_id)

    response = test_client.put(
        f"/api/v1/chats/{conversation_id}/settings",
        headers=auth_headers,
        json={
            "settings": {
                "conversationContext": {"world_book_ids": [999_999_999]},
            }
        },
    )

    assert response.status_code in {400, 404, 422}, response.text
    after = character_db.get_roleplay_resume_state(conversation_id)
    assert after["settings_version"] == before["settings_version"]
    assert after["settings"] == before["settings"]


@pytest.mark.integration
def test_drift_retry_freezes_initial_provider_model_and_sampling(
    character_db,
    monkeypatch,
):
    app_config = {
        "llm_api_settings": {"default_api": "local-llm"},
        "local_llm": {"model": "initial-model"},
    }
    monkeypatch.setattr(
        character_conversation_factory,
        "ensure_app_config",
        lambda: app_config,
        raising=False,
    )
    character_id = character_db.add_character_card(
        {
            "name": "Frozen Defaults Ari",
            "extensions": {
                "tldw": {
                    "generation": {
                        "temperature": "0.2",
                        "topP": "0.4",
                        "repetitionPenalty": "1.1",
                        "stopSequences": ["<INITIAL>"],
                    }
                }
            },
        }
    )
    real_materialize = character_conversation_factory._materialize_behavior
    call_count = 0

    def drift_once(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        materialized = real_materialize(*args, **kwargs)
        if call_count >= 2:
            payload = materialized.snapshot.payload
            payload["participants"][0]["prompt"]["description"] += " drift"
            materialized = replace(
                materialized,
                snapshot=build_behavior_snapshot(
                    payload,
                    max_bytes=kwargs["max_snapshot_bytes"],
                ),
            )
        if call_count == 2:
            app_config["llm_api_settings"]["default_api"] = "definitely-not-a-provider"
            app_config["local_llm"]["model"] = "changed-model"
            monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "definitely-not-a-provider")
            monkeypatch.setenv("CHAR_CHAT_MODEL", "changed-model")
        return materialized

    monkeypatch.setattr(
        character_conversation_factory,
        "_materialize_behavior",
        drift_once,
    )

    conversation_id = create_character_conversation(
        character_db,
        conversation_data={
            "character_id": character_id,
            "title": "Frozen defaults",
            "client_id": str(character_db.client_id),
        },
    )

    assert call_count == 4
    assert character_db.get_roleplay_resume_state(conversation_id)[
        "effective_completion"
    ] == {
        "provider": "local-llm",
        "model": "initial-model",
        "sampling": {
            "temperature": 0.2,
            "top_p": 0.4,
            "repetition_penalty": 1.1,
            "stop": ["<INITIAL>"],
        },
    }


@pytest.mark.integration
@pytest.mark.parametrize(
    "credential_key",
    [
        "api.key",
        "accessToken",
        "authToken",
        "bearerToken",
        "clientSecret",
        "privateKey",
        "refreshToken",
        "xApiKey",
        "secretKey",
        "awsSecretAccessKey",
        "openaiApiKey",
        "apiToken",
        "consumerSecret",
        "signingSecret",
        "vendorAccessToken",
        "vendorAuthToken",
        "vendorBearerToken",
        "vendorRefreshToken",
        "vendorClientSecret",
        "vendorApiKey",
        "vendorApiToken",
        "vendorPrivateKey",
        "vendorXApiKey",
        "oauthToken",
        "sessionToken",
        "csrfToken",
        "idToken",
        "oauthAccessToken",
        "hfToken",
        "huggingfaceToken",
        "githubToken",
    ],
)
def test_credential_settings_rejection_rolls_back_creation(
    character_db,
    credential_key,
):
    character_id = character_db.add_character_card({"name": "Credential Ari"})

    with pytest.raises(InputError, match="credential"):
        create_character_conversation(
            character_db,
            conversation_data={
                "character_id": character_id,
                "title": "Credential settings",
                "client_id": str(character_db.client_id),
            },
            provider="local-llm",
            model="local-test",
            sampling={
                "temperature": 0.0,
                "top_p": 0.0,
                "repetition_penalty": 0.0,
                "stop": [],
            },
            conversation_settings={
                "nested": {credential_key: "must-not-persist"},
            },
        )

    with character_db.transaction() as conn:
        assert conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM conversation_settings").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM conversation_behavior_snapshots").fetchone()[0] == 0


@pytest.mark.integration
def test_ordinary_token_settings_remain_allowed(character_db):
    character_id = character_db.add_character_card({"name": "Ordinary Token Ari"})

    conversation_id = create_character_conversation(
        character_db,
        conversation_data={
            "character_id": character_id,
            "title": "Ordinary token settings",
            "client_id": str(character_db.client_id),
        },
        provider="local-llm",
        model="local-test",
        conversation_settings={
            "token": "ordinary prompt metadata",
            "nested": {
                "token_budget": 512,
                "tokenBudget": 256,
                "max_tokens": 1024,
                "promptToken": "prompt-behavior-id",
                "pageToken": "page-opaque-id",
                "nextPageToken": "next-page-opaque-id",
                "validationToken": "validation-behavior-id",
                "undoToken": "undo-behavior-id",
            },
        },
    )

    settings = character_db.get_roleplay_resume_state(conversation_id)["settings"]
    assert settings["token"] == "ordinary prompt metadata"
    assert settings["nested"]["token_budget"] == 512
    assert settings["nested"]["tokenBudget"] == 256
    assert settings["nested"]["max_tokens"] == 1024
    assert settings["nested"]["promptToken"] == "prompt-behavior-id"
    assert settings["nested"]["pageToken"] == "page-opaque-id"
    assert settings["nested"]["nextPageToken"] == "next-page-opaque-id"
    assert settings["nested"]["validationToken"] == "validation-behavior-id"
    assert settings["nested"]["undoToken"] == "undo-behavior-id"


@pytest.mark.integration
def test_reserved_roleplay_settings_update_is_rejected(
    test_client,
    auth_headers,
    character_db,
):
    character_id = character_db.add_character_card({"name": "Reserved Ari"})
    created = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={
            "character_id": character_id,
            "provider": "local-llm",
            "model": "local-test",
        },
    )
    assert created.status_code == 201, created.text
    conversation_id = created.json()["id"]
    before = _snapshot_storage_bytes(character_db, conversation_id)[1]

    response = test_client.put(
        f"/api/v1/chats/{conversation_id}/settings",
        headers=auth_headers,
        json={
            "settings": {
                "roleplayResumeV1": {
                    "resumeEligible": True,
                    "resumeIneligibleReason": None,
                    "effectiveCompletion": {},
                }
            }
        },
    )

    assert response.status_code == 422
    assert _snapshot_storage_bytes(character_db, conversation_id)[1] == before


@pytest.mark.integration
@pytest.mark.parametrize(
    "forged_effective",
    [
        {},
        {
            "provider": "",
            "model": "forged-model",
            "sampling": {
                "temperature": 0.0,
                "top_p": 0.0,
                "repetition_penalty": 0.0,
                "stop": [],
            },
        },
    ],
)
def test_forged_readiness_is_invalid_on_store_and_api_reads(
    test_client,
    auth_headers,
    character_db,
    forged_effective,
):
    character_id = character_db.add_character_card({"name": "Forged Ari"})
    created = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={
            "character_id": character_id,
            "provider": "local-llm",
            "model": "local-test",
        },
    )
    assert created.status_code == 201, created.text
    conversation_id = created.json()["id"]
    forged = {
        "roleplayResumeV1": {
            "resumeEligible": True,
            "resumeIneligibleReason": None,
            "effectiveCompletion": forged_effective,
        }
    }
    with character_db.transaction() as conn:
        conn.execute(
            "UPDATE conversation_settings SET settings_json = ? WHERE conversation_id = ?",
            (json.dumps(forged), conversation_id),
        )

    state = character_db.get_roleplay_resume_state(conversation_id)
    assert state["resume_eligible"] is False
    assert state["resume_ineligible_reason"] == "invalid_effective_settings"
    assert state["effective_completion"] is None

    response = test_client.get(
        f"/api/v1/chats/{conversation_id}",
        headers=auth_headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["resume_eligible"] is False
    assert body["resume_ineligible_reason"] == "invalid_effective_settings"
    assert "canonical_json" not in json.dumps(body)
    assert "payload" not in json.dumps(body)


@pytest.mark.integration
def test_stored_readiness_does_not_depend_on_current_provider_inventory(
    test_client,
    auth_headers,
    character_db,
    monkeypatch,
):
    character_id = character_db.add_character_card({"name": "Stable Readiness Ari"})
    created = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={
            "character_id": character_id,
            "provider": "local-llm",
            "model": "local-test",
        },
    )
    assert created.status_code == 201, created.text
    conversation_id = created.json()["id"]

    from tldw_Server_API.app.core.Chat import chat_service
    from tldw_Server_API.app.core.LLM_Calls import adapter_registry

    def deployment_probe(*_args, **_kwargs):
        raise AssertionError("stored readiness must not consult live provider state")

    monkeypatch.setattr(adapter_registry, "get_registry", deployment_probe)
    monkeypatch.setattr(chat_service, "is_model_known_for_provider", deployment_probe)

    state = character_db.get_roleplay_resume_state(conversation_id)
    assert state["resume_eligible"] is True
    assert state["effective_completion"]["model"] == "local-test"
    response = test_client.get(
        f"/api/v1/chats/{conversation_id}",
        headers=auth_headers,
    )
    assert response.status_code == 200, response.text
    assert response.json()["resume_eligible"] is True


@pytest.mark.integration
def test_source_mutation_and_deployment_default_changes_do_not_change_creation_bytes(
    test_client,
    auth_headers,
    character_db,
    monkeypatch,
):
    sources = _create_behavior_sources(character_db)
    response = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={
            "character_id": sources["primary_id"],
            "participant_character_ids": [sources["second_id"]],
            "prompt_preset_id": "snapshot-cinematic",
            "memory_by_character_id": {str(sources["primary_id"]): "Remember the brass key."},
            "provider": "local-llm",
            "model": "local-test",
            "temperature": 0.0,
            "top_p": 0.0,
            "repetition_penalty": 0.0,
            "stop": [],
        },
    )
    assert response.status_code == 201, response.text
    conversation_id = response.json()["id"]
    before_bytes = _snapshot_storage_bytes(character_db, conversation_id)
    before_state = character_db.get_roleplay_resume_state(conversation_id)
    before_effective = before_state["effective_completion"]
    settings = before_state["settings"]
    assert settings["presetScope"] == "chat"
    assert settings["chatPresetOverrideId"] == "snapshot-cinematic"

    from tldw_Server_API.app.api.v1.endpoints.character_chat_sessions import (
        _resolve_effective_prompt_preset,
    )

    primary_before = character_db.get_character_card_by_id(sources["primary_id"])
    assert _resolve_effective_prompt_preset(
        settings,
        primary_before,
        db=character_db,
    ) == "snapshot-cinematic"
    materialized_preset = before_state["behavior_snapshot"]["payload"]["participants"][
        0
    ]["prompt"]["prompt_relevant_extensions"]["prompt_preset"]
    assert materialized_preset["selection_source"] == "creation_request"
    assert materialized_preset["source"] == {
        "kind": "prompt_preset",
        "id": "snapshot-cinematic",
        "version": 1,
    }

    assert character_db.update_character_card(
        sources["primary_id"],
        {
            "system_prompt": "Changed after creation.",
            "first_message": "Changed greeting.",
            "extensions": {"tldw": {"generation": {"temperature": 1.9}}},
        },
        expected_version=primary_before["version"],
    )
    second = character_db.get_character_card_by_id(sources["second_id"])
    assert character_db.soft_delete_character_card(sources["second_id"], expected_version=second["version"])
    assert character_db.delete_prompt_preset("snapshot-cinematic")
    with character_db.transaction() as conn:
        conn.execute(
            "UPDATE character_exemplars SET text = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            ("Mutated exemplar", sources["exemplar_id"]),
        )
        conn.execute("DELETE FROM world_book_entries WHERE id = ?", (sources["world_book_entry_id"],))
        conn.execute("DELETE FROM world_books WHERE id = ?", (sources["world_book_id"],))
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("CHAR_CHAT_MODEL", "changed-model")
    monkeypatch.setenv("CHAR_CHAT_TEMPERATURE", "1.8")

    assert _snapshot_storage_bytes(character_db, conversation_id) == before_bytes
    assert character_db.get_roleplay_resume_state(conversation_id)["effective_completion"] == before_effective


@pytest.mark.integration
def test_list_omits_detail_only_resume_authority_without_loading_snapshot_bodies(
    test_client,
    auth_headers,
    character_db,
):
    character_id = character_db.add_character_card(
        {"name": "List Projection Ari", "first_message": "Welcome back."}
    )
    resumable_id = create_character_conversation(
        character_db,
        conversation_data={
            "character_id": character_id,
            "title": "Resumable list item",
            "client_id": str(character_db.client_id),
        },
        provider="local-llm",
        model="local-test",
        initial_messages=[
            {
                "sender": "List Projection Ari",
                "content": "Welcome back.",
                "role": "assistant",
            }
        ],
    )
    legacy_id = character_db.add_conversation(
        {
            "character_id": character_id,
            "title": "Legacy list item",
            "client_id": str(character_db.client_id),
        }
    )
    store = character_db.conversation_resume_store

    with (
        mock.patch.object(
            store,
            "get_roleplay_resume_summaries",
            side_effect=AssertionError("list must not load resume summaries"),
            create=True,
        ) as summary_read,
        mock.patch.object(
            store,
            "get_roleplay_resume_state",
            side_effect=AssertionError("list must not load full resume state"),
        ) as full_read,
        mock.patch.object(
            store,
            "get_conversation_behavior_snapshot",
            side_effect=AssertionError("list must not load snapshot bodies"),
        ) as snapshot_read,
    ):
        response = test_client.get(
            "/api/v1/chats/?limit=10",
            headers=auth_headers,
        )

    assert response.status_code == 200, response.text
    assert summary_read.call_count == 0
    assert full_read.call_count == 0
    assert snapshot_read.call_count == 0
    by_id = {chat["id"]: chat for chat in response.json()["chats"]}
    detail_only_fields = {
        "behavior_snapshot",
        "resume_eligible",
        "resume_ineligible_reason",
        "settings_version",
        "history_version",
        "tail",
    }
    assert detail_only_fields.isdisjoint(by_id[resumable_id])
    assert detail_only_fields.isdisjoint(by_id[legacy_id])

    schema = test_client.app.openapi()["components"]["schemas"]["ChatSessionListItem"]
    assert detail_only_fields.isdisjoint(schema["properties"])

    valid_detail = test_client.get(
        f"/api/v1/chats/{resumable_id}",
        headers=auth_headers,
    )
    assert valid_detail.status_code == 200, valid_detail.text
    assert valid_detail.json()["resume_eligible"] is True
    assert valid_detail.json()["tail"]["message_id"] is not None
    assert valid_detail.json()["tail"]["message_version"] == 1

    with character_db.transaction() as conn:
        row = conn.execute(
            "SELECT canonical_json FROM conversation_behavior_snapshots WHERE conversation_id = ?",
            (resumable_id,),
        ).fetchone()
        canonical_json = str(row[0])
        tampered_json = canonical_json.replace(
            "List Projection Ari",
            "Mist Projection Ari",
            1,
        )
        assert tampered_json != canonical_json
        assert len(tampered_json) == len(canonical_json)
        conn.execute(
            "UPDATE conversation_behavior_snapshots SET canonical_json = ? WHERE conversation_id = ?",
            (tampered_json, resumable_id),
        )

    list_after_tamper = test_client.get(
        "/api/v1/chats/?limit=10&include_settings=true",
        headers=auth_headers,
    )
    assert list_after_tamper.status_code == 200, list_after_tamper.text
    tampered_list_item = next(
        item
        for item in list_after_tamper.json()["chats"]
        if item["id"] == resumable_id
    )
    assert "roleplayResumeV1" not in tampered_list_item["settings"]
    assert detail_only_fields.isdisjoint(tampered_list_item)
    persisted_settings = character_db.get_conversation_settings(resumable_id)[
        "settings"
    ]
    assert "roleplayResumeV1" in persisted_settings

    tampered_detail = test_client.get(
        f"/api/v1/chats/{resumable_id}",
        headers=auth_headers,
    )
    assert tampered_detail.status_code == 200, tampered_detail.text
    assert tampered_detail.json()["behavior_snapshot"]["status"] == "invalid"
    assert tampered_detail.json()["resume_eligible"] is False
    assert tampered_detail.json()["resume_ineligible_reason"] == (
        "behavior_snapshot_invalid"
    )


@pytest.mark.integration
def test_creation_snapshot_persistence_failure_rolls_back_every_row(
    test_client,
    auth_headers,
    character_db,
):
    sources = _create_behavior_sources(character_db)
    with mock.patch.object(
        ConversationResumeStore,
        "put_behavior_snapshot",
        side_effect=InputError("snapshot persistence failed"),
    ):
        response = test_client.post(
            "/api/v1/chats/?seed_first_message=true",
            headers=auth_headers,
            json={
                "character_id": sources["primary_id"],
                "provider": "local-llm",
                "model": "local-test",
                "temperature": 0.0,
                "top_p": 0.0,
                "repetition_penalty": 0.0,
                "stop": [],
            },
        )

    assert response.status_code >= 400
    with character_db.transaction() as conn:
        assert conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM conversation_settings").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM conversation_behavior_snapshots").fetchone()[0] == 0


@pytest.mark.integration
def test_creation_oversize_snapshot_rolls_back(character_db):
    sources = _create_behavior_sources(character_db)

    with pytest.raises(ValueError, match="exceeds maximum"):
        create_character_conversation(
            character_db,
            conversation_data={
                "character_id": sources["primary_id"],
                "title": "Too large",
                "client_id": str(character_db.client_id),
            },
            participant_character_ids=[sources["second_id"]],
            provider="local-llm",
            model="local-test",
            sampling={
                "temperature": 0.0,
                "top_p": 0.0,
                "repetition_penalty": 0.0,
                "stop": [],
            },
            max_snapshot_bytes=64,
        )

    assert character_db.get_conversations_for_character(sources["primary_id"]) == []


@pytest.mark.integration
def test_creation_partial_sampling_merges_character_defaults_and_remains_stable(
    character_db,
    monkeypatch,
):
    character_id = character_db.add_character_card({"name": "Incomplete Ari"})
    conversation_id = create_character_conversation(
        character_db,
        conversation_data={
            "character_id": character_id,
            "title": "Incomplete",
            "client_id": str(character_db.client_id),
        },
        provider="local-llm",
        model="local-test",
        sampling={"temperature": 0.0},
    )
    before = _snapshot_storage_bytes(character_db, conversation_id)

    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "changed-provider")
    monkeypatch.setenv("CHAR_CHAT_MODEL", "changed-model")
    state = character_db.get_roleplay_resume_state(conversation_id)

    assert state["behavior_snapshot"]["status"] == "valid"
    assert state["resume_eligible"] is True
    assert state["resume_ineligible_reason"] is None
    assert state["effective_completion"]["sampling"] == {
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "stop": [],
    }
    assert _snapshot_storage_bytes(character_db, conversation_id) == before


@pytest.mark.integration
def test_creation_source_drift_exhaustion_rolls_back(character_db, monkeypatch):
    character_id = character_db.add_character_card({"name": "Drifting Ari"})
    real_materialize = character_conversation_factory._materialize_behavior
    call_count = 0

    def alternating_materialize(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        materialized = real_materialize(*args, **kwargs)
        if call_count % 2:
            return materialized
        payload = materialized.snapshot.payload
        payload["participants"][0]["prompt"]["description"] += " drift"
        return replace(
            materialized,
            snapshot=build_behavior_snapshot(
                payload,
                max_bytes=kwargs["max_snapshot_bytes"],
            ),
        )

    monkeypatch.setattr(
        character_conversation_factory,
        "_materialize_behavior",
        alternating_materialize,
    )

    with pytest.raises(InputError, match="changed during conversation creation"):
        create_character_conversation(
            character_db,
            conversation_data={
                "character_id": character_id,
                "title": "Drifting",
                "client_id": str(character_db.client_id),
            },
            provider="local-llm",
            model="local-test",
            sampling={
                "temperature": 0.0,
                "top_p": 0.0,
                "repetition_penalty": 0.0,
                "stop": [],
            },
        )

    assert call_count == 4
    assert character_db.get_conversations_for_character(character_id) == []


@pytest.mark.integration
def test_postgres_style_creation_drift_retries_one_complete_source_view_and_sampling(
    character_db,
    monkeypatch,
) -> None:
    character_id = character_db.add_character_card({"name": "Drifting Defaults"})
    real_materialize = character_conversation_factory._materialize_behavior
    call_count = 0

    def interleaved_materialize(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        materialized = real_materialize(*args, **kwargs)
        if call_count == 1:
            return materialized
        payload = materialized.snapshot.payload
        payload["participants"][0]["generation_defaults"]["sampling"][
            "temperature"
        ] = 0.2
        refreshed_character = dict(materialized.primary_character)
        refreshed_character["extensions"] = {
            "tldw": {"generation": {"temperature": 0.2}}
        }
        return replace(
            materialized,
            snapshot=build_behavior_snapshot(
                payload,
                max_bytes=kwargs["max_snapshot_bytes"],
            ),
            primary_character=refreshed_character,
        )

    monkeypatch.setattr(
        character_conversation_factory,
        "_materialize_behavior",
        interleaved_materialize,
    )
    conversation_id = create_character_conversation(
        character_db,
        conversation_data={
            "character_id": character_id,
            "title": "Coherent retry",
            "client_id": str(character_db.client_id),
        },
        provider="local-llm",
        model="local-test",
    )

    state = character_db.get_roleplay_resume_state(conversation_id)
    assert call_count == 4
    assert state["behavior_snapshot"]["payload"]["participants"][0][
        "generation_defaults"
    ]["sampling"]["temperature"] == 0.2
    assert state["effective_completion"]["sampling"]["temperature"] == 0.2


@pytest.mark.integration
def test_creation_greeting_seed_uses_same_stabilized_source_as_snapshot(
    test_client,
    auth_headers,
    character_db,
    monkeypatch,
) -> None:
    character_id = character_db.add_character_card(
        {
            "name": "Greeting Drift",
            "first_message": "Old greeting",
            "alternate_greetings": ["Old alternate"],
        }
    )
    real_materialize = character_conversation_factory._materialize_behavior
    call_count = 0

    def _interleaved_materialize(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        materialized = real_materialize(*args, **kwargs)
        if call_count == 1:
            return materialized
        payload = materialized.snapshot.payload
        participant = payload["participants"][0]
        participant["source"]["version"] += 1
        participant["greeting"] = {
            "content": "New accepted greeting",
            "source": "first_message",
            "source_index": 0,
        }
        refreshed_character = dict(materialized.primary_character)
        refreshed_character["version"] = int(
            refreshed_character.get("version") or 1
        ) + 1
        refreshed_character["first_message"] = "New accepted greeting"
        return replace(
            materialized,
            snapshot=build_behavior_snapshot(
                payload,
                max_bytes=kwargs["max_snapshot_bytes"],
            ),
            primary_character=refreshed_character,
        )

    monkeypatch.setattr(
        character_conversation_factory,
        "_materialize_behavior",
        _interleaved_materialize,
    )

    created = test_client.post(
        "/api/v1/chats/?seed_first_message=true&greeting_strategy=default",
        headers=auth_headers,
        json={
            "character_id": character_id,
            "provider": "local-llm",
            "model": "local-test",
        },
    )

    assert created.status_code == 201, created.text
    conversation_id = created.json()["id"]
    state = character_db.get_roleplay_resume_state(conversation_id)
    messages = character_db.get_messages_for_conversation(conversation_id)
    assert call_count == 4
    assert state["behavior_snapshot"]["payload"]["participants"][0]["greeting"][
        "content"
    ] == "New accepted greeting"
    assert [message["content"] for message in messages] == ["New accepted greeting"]
    expected_checksum = character_conversation_factory.compute_character_greetings_checksum(
        {
            "first_message": "New accepted greeting",
            "alternate_greetings": ["Old alternate"],
        }
    )
    assert state["settings"]["greetingsChecksum"] == expected_checksum


@pytest.mark.integration
def test_module_creator_greeting_seed_uses_same_stabilized_source_as_snapshot(
    character_db,
    monkeypatch,
) -> None:
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "local-llm")
    monkeypatch.setenv("CHAR_CHAT_MODEL", "local-test")
    character_id = character_db.add_character_card(
        {
            "name": "Module Greeting Drift",
            "first_message": "Old module greeting",
        }
    )
    real_materialize = character_conversation_factory._materialize_behavior
    call_count = 0

    def _interleaved_materialize(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        materialized = real_materialize(*args, **kwargs)
        if call_count == 1:
            return materialized
        payload = materialized.snapshot.payload
        participant = payload["participants"][0]
        participant["source"]["version"] += 1
        participant["greeting"] = {
            "content": "New accepted module greeting",
            "source": "first_message",
            "source_index": 0,
        }
        refreshed_character = dict(materialized.primary_character)
        refreshed_character["version"] = int(
            refreshed_character.get("version") or 1
        ) + 1
        refreshed_character["first_message"] = "New accepted module greeting"
        return replace(
            materialized,
            snapshot=build_behavior_snapshot(
                payload,
                max_bytes=kwargs["max_snapshot_bytes"],
            ),
            primary_character=refreshed_character,
        )

    monkeypatch.setattr(
        character_conversation_factory,
        "_materialize_behavior",
        _interleaved_materialize,
    )

    conversation_id, _character, history, _image = start_new_chat_session(
        character_db,
        character_id,
        "Alice",
        greeting_strategy="default",
    )

    assert conversation_id is not None
    state = character_db.get_roleplay_resume_state(conversation_id)
    messages = character_db.get_messages_for_conversation(conversation_id)
    assert call_count == 4
    assert state["behavior_snapshot"]["payload"]["participants"][0]["greeting"][
        "content"
    ] == "New accepted module greeting"
    assert [message["content"] for message in messages] == [
        "New accepted module greeting"
    ]
    assert history == [(None, "New accepted module greeting")]


@pytest.mark.integration
def test_legacy_and_invalid_storage_are_non_resumable_and_body_redacted(
    test_client,
    auth_headers,
    character_db,
):
    character_id = character_db.add_character_card({"name": "Legacy Ari"})
    legacy_id = character_db.add_conversation(
        {
            "character_id": character_id,
            "title": "Legacy",
            "client_id": str(character_db.client_id),
        }
    )
    invalid_id = character_db.add_conversation(
        {
            "character_id": character_id,
            "title": "Invalid",
            "client_id": str(character_db.client_id),
        }
    )
    with character_db.transaction() as conn:
        conn.execute(
            "INSERT INTO conversation_behavior_snapshots(conversation_id, status) VALUES (?, 'invalid')",
            (invalid_id,),
        )

    legacy = test_client.get(f"/api/v1/chats/{legacy_id}", headers=auth_headers)
    invalid = test_client.get(f"/api/v1/chats/{invalid_id}", headers=auth_headers)

    assert legacy.status_code == 200
    assert legacy.json()["behavior_snapshot"] == {
        "status": "missing",
        "schema_version": None,
        "digest": None,
    }
    assert legacy.json()["resume_eligible"] is False
    assert legacy.json()["resume_ineligible_reason"] == "behavior_snapshot_missing"
    assert invalid.status_code == 200
    assert invalid.json()["behavior_snapshot"]["status"] == "invalid"
    assert invalid.json()["resume_eligible"] is False
    assert invalid.json()["resume_ineligible_reason"] == "behavior_snapshot_invalid"
    assert "canonical_json" not in json.dumps(invalid.json())


@pytest.mark.integration
def test_legacy_import_workflow_delegates_to_atomic_creation(character_db, monkeypatch):
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "local-llm")
    monkeypatch.setenv("CHAR_CHAT_MODEL", "local-test")
    character_id = character_db.add_character_card(
        {"name": "Import Ari", "first_message": "Imported greeting"}
    )

    conversation_id, returned_character_id = load_chat_history_from_file_and_save_to_db(
        character_db,
        character_id,
        file_content=json.dumps({"messages": [{"role": "user", "content": "Hello"}]}),
        user_name_for_placeholders="Reader",
    )

    assert returned_character_id == character_id
    state = character_db.get_roleplay_resume_state(conversation_id)
    assert state["behavior_snapshot"]["status"] == "valid"
    assert state["resume_eligible"] is True


@pytest.mark.integration
def test_legacy_sync_projection_stays_explicitly_non_resumable(
    character_db,
):
    character_id = character_db.add_character_card({"name": "Sync Ari"})
    assert character_db.upsert_conversation_from_sync(
        conversation_id="sync-character-conversation",
        title="Synced character chat",
        sync_client_id="sync-device",
        object_revision=1,
        object_hash="sha256:sync",
        assistant_kind="character",
        assistant_id=str(character_id),
        character_id=character_id,
    )

    state = character_db.get_roleplay_resume_state("sync-character-conversation")

    assert state["behavior_snapshot"]["status"] == "missing"
    assert state["resume_eligible"] is False
    assert state["resume_ineligible_reason"] == "behavior_snapshot_missing"


@pytest.mark.integration
def test_cross_user_detail_never_reveals_resume_metadata(test_client, auth_headers, character_db):
    character_id = character_db.add_character_card({"name": "Private Ari"})
    conversation_id = character_db.add_conversation(
        {"character_id": character_id, "title": "Private", "client_id": "other-user"}
    )

    response = test_client.get(f"/api/v1/chats/{conversation_id}", headers=auth_headers)

    assert response.status_code in {403, 404}
    serialized = response.text
    assert "behavior_snapshot" not in serialized
    assert "resume_eligible" not in serialized


@pytest.mark.integration
def test_behavior_settings_materialize_references_and_noop_advances_version(
    test_client,
    auth_headers,
    character_db,
):
    sources = _create_behavior_sources(character_db)
    created = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={
            "character_id": sources["primary_id"],
            "participant_character_ids": [sources["second_id"]],
            "provider": "local-llm",
            "model": "local-test",
        },
    )
    assert created.status_code == 201, created.text
    conversation_id = created.json()["id"]
    mutation = {
        "provider": "local-llm",
        "model": "local-test",
        "chatGenerationOverride": {
            "enabled": True,
            "temperature": 0.2,
            "top_p": 0.6,
            "repetition_penalty": 1.1,
            "stop": ["<STOP>"],
        },
        "presetScope": "chat",
        "chatPresetOverrideId": "snapshot-cinematic",
        "participantCharacterIds": [
            sources["primary_id"],
            sources["second_id"],
        ],
        "characterMemoryById": {
            str(sources["primary_id"]): {
                "note": "Remember the materialized brass key.",
                "updatedAt": "2026-08-28T00:00:00Z",
            }
        },
        "assistantOverlay": {
            "kind": "character",
            "id": str(sources["primary_id"]),
            "name": "Ari in the east vault",
            "system_prompt_snapshot": "Speak as the east-vault archivist.",
            "updatedAt": "2026-08-28T00:00:00Z",
        },
        "conversationContext": {
            "world_book_ids": [sources["world_book_id"]],
        },
    }

    updated = test_client.put(
        f"/api/v1/chats/{conversation_id}/settings",
        headers=auth_headers,
        json={"settings": mutation},
    )

    assert updated.status_code == 200, updated.text
    assert "roleplayResumeV1" not in updated.json()["settings"]
    assert "roleplayBehaviorV1" not in updated.json()["settings"]
    state = character_db.get_roleplay_resume_state(conversation_id)
    assert state["settings_version"] == 2
    materialized = state["materialized_settings"]
    assert materialized["schema_version"] == 1
    assert materialized["digest"].startswith("sha256:")
    values = materialized["values"]
    assert values["effective_completion"] == {
        "provider": "local-llm",
        "model": "local-test",
        "sampling": {
            "temperature": 0.2,
            "top_p": 0.6,
            "repetition_penalty": 1.1,
            "stop": ["<STOP>"],
        },
    }
    assert values["prompt_preset"]["preset_id"] == "snapshot-cinematic"
    assert values["prompt_preset"]["section_templates"]["scenario"] == "Scene: {{scenario}}"
    assert values["world_books"][0]["id"] == sources["world_book_id"]
    assert values["world_books"][0]["entries"][0]["content"] == (
        "The east vault flooded in 2041."
    )
    assert [item["source"]["id"] for item in values["participants"]] == [
        str(sources["primary_id"]),
        str(sources["second_id"]),
    ]
    assert values["memory"]["character_memory_by_id"] == {
        str(sources["primary_id"]): "Remember the materialized brass key."
    }
    assert values["assistant_overlay"] == {
        "source": {
            "kind": "character",
            "id": str(sources["primary_id"]),
            "version": 1,
        },
        "name": "Ari in the east vault",
        "system_prompt": "Speak as the east-vault archivist.",
    }

    character = character_db.get_character_card_by_id(sources["primary_id"])
    assert character_db.update_character_card(
        sources["primary_id"],
        {"system_prompt": "Changed after settings materialization."},
        expected_version=character["version"],
    )
    assert character_db.delete_prompt_preset("snapshot-cinematic")
    with character_db.transaction() as conn:
        conn.execute(
            "UPDATE world_book_entries SET content = ? WHERE id = ?",
            ("Changed after settings materialization.", sources["world_book_entry_id"]),
        )
    assert character_db.get_roleplay_resume_state(conversation_id)[
        "materialized_settings"
    ] == materialized

    replay = test_client.put(
        f"/api/v1/chats/{conversation_id}/settings",
        headers=auth_headers,
        json={"settings": mutation},
    )
    assert replay.status_code == 200, replay.text
    replay_state = character_db.get_roleplay_resume_state(conversation_id)
    assert replay_state["settings_version"] == 3
    assert replay_state["materialized_settings"] == materialized


@pytest.mark.integration
@pytest.mark.parametrize(
    "invalid_patch",
    [
        {"chatPresetOverrideId": "missing-preset", "presetScope": "chat"},
        {"participantCharacterIds": [999_999]},
        {"conversationContext": {"world_book_ids": [999_999]}},
        {
            "assistantOverlay": {
                "kind": "character",
                "id": "999999",
                "name": "Unknown",
                "updatedAt": "2026-08-28T00:00:00Z",
            }
        },
        {"provider": "definitely-not-a-provider", "model": "missing-model"},
    ],
)
def test_unknown_behavior_reference_rejects_without_settings_version_change(
    test_client,
    auth_headers,
    character_db,
    invalid_patch,
):
    character_id = character_db.add_character_card({"name": "Reference Guard Ari"})
    created = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={
            "character_id": character_id,
            "provider": "local-llm",
            "model": "local-test",
        },
    )
    assert created.status_code == 201, created.text
    conversation_id = created.json()["id"]
    before_state = character_db.get_roleplay_resume_state(conversation_id)
    before_settings_bytes = _snapshot_storage_bytes(character_db, conversation_id)[1]

    response = test_client.put(
        f"/api/v1/chats/{conversation_id}/settings",
        headers=auth_headers,
        json={"settings": invalid_patch},
    )

    assert response.status_code in {400, 404, 409, 422}, response.text
    after_state = character_db.get_roleplay_resume_state(conversation_id)
    assert after_state["settings_version"] == before_state["settings_version"] == 1
    assert _snapshot_storage_bytes(character_db, conversation_id)[1] == before_settings_bytes


@pytest.mark.integration
def test_history_version_advances_for_ancestor_and_branch_with_stable_tail_and_rollback(
    character_db,
):
    character_id = character_db.add_character_card({"name": "History Fence Ari"})
    conversation_id = create_character_conversation(
        character_db,
        conversation_data={
            "character_id": character_id,
            "title": "History fences",
            "client_id": str(character_db.client_id),
        },
        provider="local-llm",
        model="local-test",
        initial_messages=[
            {"id": "history-root", "sender": "user", "content": "Root"},
            {
                "id": "history-tail",
                "sender": "assistant",
                "content": "Tail",
                "parent_message_id": "history-root",
            },
        ],
    )
    initial = character_db.get_roleplay_resume_state(conversation_id)
    assert initial["history_version"] == 3
    assert initial["tail"] == {"message_id": "history-tail", "message_version": 1}

    assert character_db.update_message(
        "history-root",
        {"content": "Edited ancestor"},
        expected_version=1,
    )
    edited = character_db.get_roleplay_resume_state(conversation_id)
    assert edited["history_version"] == 4
    assert edited["tail"] == initial["tail"]

    assert character_db.update_message(
        "history-tail",
        {"parent_message_id": None},
        expected_version=1,
    )
    branched = character_db.get_roleplay_resume_state(conversation_id)
    assert branched["history_version"] == 5
    assert branched["tail"] == {
        "message_id": "history-tail",
        "message_version": 2,
    }

    with pytest.raises(RuntimeError, match="rollback history mutation"):
        with character_db.transaction() as conn:
            assert character_db.soft_delete_message(
                "history-root",
                expected_version=2,
                conn=conn,
            )
            assert character_db.get_roleplay_resume_state(
                conversation_id,
                conn=conn,
            )["history_version"] == 6
            raise RuntimeError("rollback history mutation")
    assert character_db.get_roleplay_resume_state(conversation_id) == branched


@pytest.mark.integration
def test_creation_v1_materialized_authority_binds_immutable_snapshot_and_defaults(
    test_client,
    auth_headers,
    character_db,
):
    sources = _create_behavior_sources(character_db)
    created = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={
            "character_id": sources["primary_id"],
            "participant_character_ids": [sources["second_id"]],
            "prompt_preset_id": "snapshot-cinematic",
            "memory_by_character_id": {
                str(sources["primary_id"]): "Creation memory",
            },
            "provider": "local-llm",
            "model": "local-test",
        },
    )
    assert created.status_code == 201, created.text
    conversation_id = created.json()["id"]

    state = character_db.get_roleplay_resume_state(conversation_id)
    materialized = state["materialized_settings"]
    assert state["settings_version"] == 1
    assert materialized["schema_version"] == 1
    assert materialized["values"]["base_snapshot"] == {
        "schema_version": state["behavior_snapshot"]["schema_version"],
        "digest": state["behavior_snapshot"]["digest"],
    }
    assert materialized["values"]["effective_completion"] == state[
        "effective_completion"
    ]
    assert materialized["values"]["behavior_controls"]["author_note"][
        "position"
    ] == "before_system"
    assert materialized["values"]["behavior_controls"]["preset_scope"] == "chat"

    with character_db.transaction() as conn:
        conn.execute(
            "UPDATE world_book_entries SET content = ? WHERE id = ?",
            ("Mutable source changed", sources["world_book_entry_id"]),
        )
    assert character_db.delete_prompt_preset("snapshot-cinematic")

    provider_only = test_client.put(
        f"/api/v1/chats/{conversation_id}/settings",
        headers=auth_headers,
        json={"settings": {"provider": "local-llm", "model": "local-test"}},
    )
    assert provider_only.status_code == 200, provider_only.text
    after = character_db.get_roleplay_resume_state(conversation_id)
    assert after["settings_version"] == 2
    assert after["materialized_settings"]["values"]["base_snapshot"] == materialized[
        "values"
    ]["base_snapshot"]


@pytest.mark.integration
def test_resumable_sources_and_settings_reject_credentials_recursively_and_atomically(
    test_client,
    auth_headers,
    character_db,
):
    secret_character_id = character_db.add_character_card(
        {
            "name": "Secret Source",
            "extensions": {"nested": {"clientSecret": "must-not-materialize"}},
        }
    )
    rejected_create = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={
            "character_id": secret_character_id,
            "provider": "local-llm",
            "model": "local-test",
        },
    )
    assert rejected_create.status_code in {400, 422}, rejected_create.text
    assert character_db.get_conversations_for_character(secret_character_id) == []

    assert character_db.upsert_prompt_preset(
        preset_id="secret-preset",
        name="Secret preset",
        section_order=["identity"],
        section_templates={"identity": "safe", "clientSecret": "must-reject"},
    )
    preset_character_id = character_db.add_character_card({"name": "Preset Source"})
    rejected_preset = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={
            "character_id": preset_character_id,
            "prompt_preset_id": "secret-preset",
            "provider": "local-llm",
            "model": "local-test",
        },
    )
    assert rejected_preset.status_code in {400, 422}, rejected_preset.text
    assert character_db.get_conversations_for_character(preset_character_id) == []

    lore_character_id = character_db.add_character_card({"name": "Lore Source"})
    world_books = WorldBookService(character_db)
    secret_book_id = world_books.create_world_book("Secret metadata canon")
    world_books.add_world_book_entry(
        secret_book_id,
        keywords=["secret"],
        content="Visible lore",
        metadata={"nested": {"apiToken": "must-reject"}},
    )
    assert world_books.attach_to_character(secret_book_id, lore_character_id)[
        "success"
    ] is True
    rejected_lore = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={
            "character_id": lore_character_id,
            "provider": "local-llm",
            "model": "local-test",
        },
    )
    assert rejected_lore.status_code in {400, 422}, rejected_lore.text
    assert character_db.get_conversations_for_character(lore_character_id) == []

    character_id = character_db.add_character_card({"name": "Clean Source"})
    created = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={
            "character_id": character_id,
            "provider": "local-llm",
            "model": "local-test",
        },
    )
    assert created.status_code == 201, created.text
    conversation_id = created.json()["id"]
    before = character_db.get_roleplay_resume_state(conversation_id)

    rejected_patch = test_client.put(
        f"/api/v1/chats/{conversation_id}/settings",
        headers=auth_headers,
        json={
            "settings": {
                "authorNote": "safe",
                "nested": {
                    "apiToken": "must-not-materialize",
                    "token": "legitimate-behavior-token",
                },
            }
        },
    )
    assert rejected_patch.status_code in {400, 422}, rejected_patch.text
    after = character_db.get_roleplay_resume_state(conversation_id)
    assert after["settings_version"] == before["settings_version"] == 1
    assert after["settings"] == before["settings"]


@pytest.mark.integration
def test_materialized_controls_cover_prompt_consumers_and_direct_writers(
    test_client,
    auth_headers,
    character_db,
):
    character_id = character_db.add_character_card(
        {
            "name": "Control Ari",
            "first_message": "Default greeting",
            "alternate_greetings": ["Alternate greeting"],
        }
    )
    created = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={
            "character_id": character_id,
            "provider": "local-llm",
            "model": "local-test",
        },
    )
    assert created.status_code == 201, created.text
    conversation_id = created.json()["id"]

    patch = {
        "turnTakingMode": "round_robin",
        "greetingEnabled": False,
        "greetingScope": "character",
        "greetingSelectionId": "greeting:1:selected",
        "useCharacterDefault": False,
        "authorNote": "GM note",
        "authorNoteEnabled": True,
        "authorNoteGmOnly": True,
        "authorNoteExcludeFromPrompt": True,
        "authorNotePlacement": {"mode": "depth", "depth": 3},
        "autoSummaryEnabled": True,
        "autoSummaryMessageThreshold": 30,
        "autoSummaryRecentWindow": 6,
        "pinnedMessageIds": ["m-1", "m-1", "m-2"],
        "presetScope": "character",
    }
    updated = test_client.put(
        f"/api/v1/chats/{conversation_id}/settings",
        headers=auth_headers,
        json={"settings": patch},
    )
    assert updated.status_code == 200, updated.text
    state = character_db.get_roleplay_resume_state(conversation_id)
    controls = state["materialized_settings"]["values"]["behavior_controls"]
    assert controls["turn_taking_mode"] == "round_robin"
    assert controls["greeting"] == {
        "enabled": False,
        "scope": "character",
        "selection_id": "greeting:1:selected",
        "use_character_default": False,
    }
    assert controls["author_note"] == {
        "enabled": True,
        "gm_only": True,
        "exclude_from_prompt": True,
        "position": {"mode": "depth", "depth": 3},
    }
    assert controls["auto_summary"] == {
        "enabled": True,
        "threshold_messages": 30,
        "window_messages": 6,
    }
    assert controls["pinned_message_ids"] == ["m-1", "m-2"]
    assert controls["preset_scope"] == "character"

    selected = test_client.put(
        f"/api/v1/chats/{conversation_id}/greetings/select",
        headers=auth_headers,
        json={"index": 1},
    )
    assert selected.status_code == 200, selected.text
    after_greeting = character_db.get_roleplay_resume_state(conversation_id)
    assert after_greeting["settings_version"] == state["settings_version"] + 1
    assert after_greeting["materialized_settings"]["values"]["behavior_controls"][
        "greeting"
    ]["selection_id"] == "greeting:1:selected"

    sessions = __import__(
        "tldw_Server_API.app.api.v1.endpoints.character_chat_sessions",
        fromlist=["_persist_auto_summary_to_settings"],
    )
    sessions._persist_auto_summary_to_settings(
        character_db,
        conversation_id,
        dict(after_greeting["settings"]),
        "Frozen summary",
        "m-1",
        "m-2",
        30,
        6,
        2,
        expected_settings_version=after_greeting["settings_version"],
    )
    after_summary = character_db.get_roleplay_resume_state(conversation_id)
    assert after_summary["settings_version"] == after_greeting["settings_version"] + 1
    assert after_summary["materialized_settings"]["values"]["behavior_controls"][
        "auto_summary"
    ]["summary"]["content"] == "Frozen summary"


@pytest.mark.integration
def test_persona_overlay_reference_enforces_explicit_owner_atomically(
    test_client,
    auth_headers,
    character_db,
):
    character_id = character_db.add_character_card({"name": "Owned Overlay"})
    created = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={
            "character_id": character_id,
            "provider": "local-llm",
            "model": "local-test",
        },
    )
    assert created.status_code == 201, created.text
    conversation_id = created.json()["id"]
    with character_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO persona_profiles(id, user_id, name, mode, version)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("other-persona", "different-user", "Private Persona", "session_scoped", 1),
        )
    before = character_db.get_roleplay_resume_state(conversation_id)

    response = test_client.put(
        f"/api/v1/chats/{conversation_id}/settings",
        headers=auth_headers,
        json={
            "settings": {
                "assistantOverlay": {
                    "kind": "persona",
                    "id": "other-persona",
                    "name": "Private Persona",
                    "updatedAt": "2026-08-28T00:00:00Z",
                }
            }
        },
    )

    assert response.status_code in {400, 404, 422}, response.text
    after = character_db.get_roleplay_resume_state(conversation_id)
    assert after["settings_version"] == before["settings_version"]
    assert after["settings"] == before["settings"]


@pytest.mark.integration
def test_detail_settings_come_from_coherent_resume_state(
    test_client,
    auth_headers,
    character_db,
    monkeypatch,
):
    character_id = character_db.add_character_card({"name": "Coherent Detail"})
    created = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={
            "character_id": character_id,
            "provider": "local-llm",
            "model": "local-test",
        },
    )
    assert created.status_code == 201, created.text
    conversation_id = created.json()["id"]

    def fail_independent_settings_read(*_args, **_kwargs):
        raise AssertionError("detail must use coherent resume-state settings")

    monkeypatch.setattr(
        character_db,
        "get_conversation_settings",
        fail_independent_settings_read,
    )
    response = test_client.get(
        f"/api/v1/chats/{conversation_id}?include_settings=true",
        headers=auth_headers,
    )
    assert response.status_code == 200, response.text
    assert response.json()["settings"]["participantCharacterIds"] == [character_id]


@pytest.mark.integration
def test_pin_writer_rebuilds_materialized_controls_once(
    test_client,
    auth_headers,
    character_db,
):
    character_id = character_db.add_character_card({"name": "Pinned Control"})
    conversation_id = create_character_conversation(
        character_db,
        conversation_data={
            "character_id": character_id,
            "title": "Pinned settings",
            "client_id": str(character_db.client_id),
        },
        provider="local-llm",
        model="local-test",
        initial_messages=[
            {"id": "pin-target", "sender": "user", "content": "Keep this turn"},
        ],
    )
    before = character_db.get_roleplay_resume_state(conversation_id)

    response = test_client.put(
        "/api/v1/messages/pin-target",
        params={"expected_version": 1},
        headers=auth_headers,
        json={"pinned": True},
    )
    assert response.status_code == 200, response.text
    after = character_db.get_roleplay_resume_state(conversation_id)
    message = character_db.get_message_by_id("pin-target")
    assert message["version"] == 2
    assert after["history_version"] == before["history_version"] + 1
    assert after["settings_version"] == before["settings_version"] + 1
    assert after["materialized_settings"]["values"]["behavior_controls"][
        "pinned_message_ids"
    ] == ["pin-target"]


@pytest.mark.integration
def test_pin_writer_rejects_stale_repeat_without_advancing_any_fence(
    test_client,
    auth_headers,
    character_db,
) -> None:
    character_id = character_db.add_character_card({"name": "Pinned Stale"})
    conversation_id = create_character_conversation(
        character_db,
        conversation_data={
            "character_id": character_id,
            "title": "Pinned stale",
            "client_id": str(character_db.client_id),
        },
        provider="local-llm",
        model="local-test",
        initial_messages=[
            {"id": "pin-stale-target", "sender": "user", "content": "Keep me"},
        ],
    )
    first = test_client.put(
        "/api/v1/messages/pin-stale-target",
        params={"expected_version": 1},
        headers=auth_headers,
        json={"pinned": True},
    )
    assert first.status_code == 200, first.text
    before_repeat = character_db.get_roleplay_resume_state(conversation_id)

    repeated = test_client.put(
        "/api/v1/messages/pin-stale-target",
        params={"expected_version": 1},
        headers=auth_headers,
        json={"pinned": True},
    )

    assert repeated.status_code == 409, repeated.text
    after_repeat = character_db.get_roleplay_resume_state(conversation_id)
    assert character_db.get_message_by_id("pin-stale-target")["version"] == 2
    assert after_repeat["history_version"] == before_repeat["history_version"]
    assert after_repeat["settings_version"] == before_repeat["settings_version"]


@pytest.mark.integration
def test_pin_writer_semantic_noop_still_advances_each_fence_once(
    test_client,
    auth_headers,
    character_db,
) -> None:
    character_id = character_db.add_character_card({"name": "Pinned No-op"})
    conversation_id = create_character_conversation(
        character_db,
        conversation_data={
            "character_id": character_id,
            "title": "Pinned no-op",
            "client_id": str(character_db.client_id),
        },
        provider="local-llm",
        model="local-test",
        initial_messages=[
            {"id": "pin-noop-target", "sender": "user", "content": "Keep me"},
        ],
    )
    before = character_db.get_roleplay_resume_state(conversation_id)

    response = test_client.put(
        "/api/v1/messages/pin-noop-target",
        params={"expected_version": 1},
        headers=auth_headers,
        json={"pinned": False},
    )

    assert response.status_code == 200, response.text
    after = character_db.get_roleplay_resume_state(conversation_id)
    assert character_db.get_message_by_id("pin-noop-target")["version"] == 2
    assert character_db.get_message_metadata("pin-noop-target")["extra"]["pinned"] is False
    assert after["history_version"] == before["history_version"] + 1
    assert after["settings_version"] == before["settings_version"] + 1
    assert after["settings"]["pinnedMessageIds"] == []


@pytest.mark.integration
def test_pin_writer_delete_race_rolls_back_all_fences(
    test_client,
    auth_headers,
    character_db,
    monkeypatch,
) -> None:
    character_id = character_db.add_character_card({"name": "Pinned Race"})
    conversation_id = create_character_conversation(
        character_db,
        conversation_data={
            "character_id": character_id,
            "title": "Pinned race",
            "client_id": str(character_db.client_id),
        },
        provider="local-llm",
        model="local-test",
        initial_messages=[
            {"id": "pin-race-target", "sender": "user", "content": "Keep me"},
        ],
    )
    before = character_db.get_roleplay_resume_state(conversation_id)
    real_update_message = character_db.update_message

    def delete_before_conditional_update(
        message_id,
        message_data,
        expected_version,
        *,
        conn=None,
    ):
        assert conn is not None
        conn.execute(
            "UPDATE messages SET deleted = 1, version = version + 1 WHERE id = ?",
            (message_id,),
        )
        return real_update_message(
            message_id,
            message_data,
            expected_version,
            conn=conn,
        )

    monkeypatch.setattr(character_db, "update_message", delete_before_conditional_update)

    response = test_client.put(
        "/api/v1/messages/pin-race-target",
        params={"expected_version": 1},
        headers=auth_headers,
        json={"pinned": True},
    )

    assert response.status_code == 409, response.text
    after = character_db.get_roleplay_resume_state(conversation_id)
    message = character_db.get_message_by_id("pin-race-target")
    assert bool(message["deleted"]) is False
    assert message["version"] == 1
    assert character_db.get_message_metadata("pin-race-target") is None
    assert after["history_version"] == before["history_version"]
    assert after["settings_version"] == before["settings_version"]


@pytest.mark.integration
def test_pin_writer_rejects_deleted_message_without_mutating_settings(
    test_client,
    auth_headers,
    character_db,
) -> None:
    character_id = character_db.add_character_card({"name": "Pinned Deleted"})
    conversation_id = create_character_conversation(
        character_db,
        conversation_data={
            "character_id": character_id,
            "title": "Pinned deleted",
            "client_id": str(character_db.client_id),
        },
        provider="local-llm",
        model="local-test",
        initial_messages=[
            {"id": "pin-deleted-target", "sender": "user", "content": "Delete me"},
        ],
    )
    assert character_db.soft_delete_message("pin-deleted-target", 1)
    before = character_db.get_roleplay_resume_state(conversation_id)

    response = test_client.put(
        "/api/v1/messages/pin-deleted-target",
        params={"expected_version": 2},
        headers=auth_headers,
        json={"pinned": True},
    )

    assert response.status_code == 409, response.text
    after = character_db.get_roleplay_resume_state(conversation_id)
    deleted = character_db.get_message_by_id(
        "pin-deleted-target",
        include_deleted=True,
    )
    assert deleted["deleted"]
    assert deleted["version"] == 2
    assert character_db.get_message_metadata("pin-deleted-target") is None
    assert after["history_version"] == before["history_version"]
    assert after["settings_version"] == before["settings_version"]


@pytest.mark.integration
def test_combined_content_pin_update_rolls_back_on_materialization_failure(
    test_client,
    auth_headers,
    character_db,
    monkeypatch,
) -> None:
    character_id = character_db.add_character_card({"name": "Atomic Pin Failure"})
    conversation_id = create_character_conversation(
        character_db,
        conversation_data={
            "character_id": character_id,
            "title": "Atomic pin failure",
            "client_id": str(character_db.client_id),
        },
        provider="local-llm",
        model="local-test",
        initial_messages=[
            {"id": "pin-atomic-failure", "sender": "user", "content": "Original"},
        ],
    )
    before = character_db.get_roleplay_resume_state(conversation_id)
    import tldw_Server_API.app.api.v1.endpoints.character_messages as messages_endpoint

    def _fail_materialization(*_args, **_kwargs):
        raise InputError("forced pin materialization failure")

    monkeypatch.setattr(
        messages_endpoint,
        "materialize_roleplay_behavior_settings",
        _fail_materialization,
    )

    response = test_client.put(
        "/api/v1/messages/pin-atomic-failure",
        params={"expected_version": 1},
        headers=auth_headers,
        json={"content": "Must roll back", "pinned": True},
    )

    assert response.status_code in {400, 409, 422}, response.text
    message = character_db.get_message_by_id("pin-atomic-failure")
    after = character_db.get_roleplay_resume_state(conversation_id)
    assert message["content"] == "Original"
    assert message["version"] == 1
    assert character_db.get_message_metadata("pin-atomic-failure") is None
    assert after["history_version"] == before["history_version"]
    assert after["settings_version"] == before["settings_version"]


@pytest.mark.integration
def test_combined_content_pin_update_advances_each_fence_once(
    test_client,
    auth_headers,
    character_db,
) -> None:
    character_id = character_db.add_character_card({"name": "Atomic Pin Success"})
    conversation_id = create_character_conversation(
        character_db,
        conversation_data={
            "character_id": character_id,
            "title": "Atomic pin success",
            "client_id": str(character_db.client_id),
        },
        provider="local-llm",
        model="local-test",
        initial_messages=[
            {"id": "pin-atomic-success", "sender": "user", "content": "Original"},
        ],
    )
    before = character_db.get_roleplay_resume_state(conversation_id)

    response = test_client.put(
        "/api/v1/messages/pin-atomic-success",
        params={"expected_version": 1},
        headers=auth_headers,
        json={"content": "Updated atomically", "pinned": True},
    )

    assert response.status_code == 200, response.text
    message = character_db.get_message_by_id("pin-atomic-success")
    metadata = character_db.get_message_metadata("pin-atomic-success")
    after = character_db.get_roleplay_resume_state(conversation_id)
    assert message["content"] == "Updated atomically"
    assert message["version"] == 2
    assert metadata["extra"]["pinned"] is True
    assert after["history_version"] == before["history_version"] + 1
    assert after["settings_version"] == before["settings_version"] + 1


@pytest.mark.integration
def test_oversize_materialized_mutation_rolls_back_settings_version(
    test_client,
    auth_headers,
    character_db,
):
    character_id = character_db.add_character_card({"name": "Bounded Control"})
    created = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={
            "character_id": character_id,
            "provider": "local-llm",
            "model": "local-test",
        },
    )
    assert created.status_code == 201, created.text
    conversation_id = created.json()["id"]
    before = character_db.get_roleplay_resume_state(conversation_id)

    response = test_client.put(
        f"/api/v1/chats/{conversation_id}/settings",
        headers=auth_headers,
        json={
            "settings": {
                "autoSummaryEnabled": True,
                "summary": {
                    "enabled": True,
                    "content": "x" * (1024 * 1024),
                    "thresholdMessages": 30,
                    "windowMessages": 6,
                },
            }
        },
    )
    assert response.status_code in {400, 413, 422}, response.text
    after = character_db.get_roleplay_resume_state(conversation_id)
    assert after["settings_version"] == before["settings_version"]
    assert after["settings"] == before["settings"]


@pytest.mark.integration
def test_duplicate_world_book_references_materialize_with_one_lookup(
    test_client,
    auth_headers,
    character_db,
):
    character_id = character_db.add_character_card({"name": "Bounded Lore"})
    world_books = WorldBookService(character_db)
    world_book_id = world_books.create_world_book("One lookup canon")
    created = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={
            "character_id": character_id,
            "provider": "local-llm",
            "model": "local-test",
        },
    )
    assert created.status_code == 201, created.text
    conversation_id = created.json()["id"]

    real_materialize = character_conversation_factory._materialize_world_books_by_id
    with mock.patch.object(
        character_conversation_factory,
        "_materialize_world_books_by_id",
        wraps=real_materialize,
    ) as materialize_book:
        response = test_client.put(
            f"/api/v1/chats/{conversation_id}/settings",
            headers=auth_headers,
            json={
                "settings": {
                    "conversationContext": {
                        "world_book_ids": [
                            world_book_id,
                            str(world_book_id),
                            world_book_id,
                        ]
                    }
                }
            },
        )

    assert response.status_code == 200, response.text
    assert materialize_book.call_count == 1
    assert materialize_book.call_args.args[1] == [world_book_id]
    state = character_db.get_roleplay_resume_state(conversation_id)
    assert [
        book["id"]
        for book in state["materialized_settings"]["values"]["world_books"]
    ] == [world_book_id]


@pytest.mark.integration
def test_participant_removal_prunes_nonparticipant_memory_authority(
    test_client,
    auth_headers,
    character_db,
):
    sources = _create_behavior_sources(character_db)
    created = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={
            "character_id": sources["primary_id"],
            "participant_character_ids": [sources["second_id"]],
            "provider": "local-llm",
            "model": "local-test",
        },
    )
    assert created.status_code == 201, created.text
    conversation_id = created.json()["id"]

    memory = test_client.put(
        f"/api/v1/chats/{conversation_id}/settings",
        headers=auth_headers,
        json={
            "settings": {
                "characterMemoryById": {
                    str(sources["second_id"]): {"note": "Secondary-only memory"}
                }
            }
        },
    )
    assert memory.status_code == 200, memory.text
    before = character_db.get_roleplay_resume_state(conversation_id)

    removed = test_client.put(
        f"/api/v1/chats/{conversation_id}/settings",
        headers=auth_headers,
        json={
            "settings": {
                "participantCharacterIds": [sources["primary_id"]],
            }
        },
    )

    assert removed.status_code == 200, removed.text
    after = character_db.get_roleplay_resume_state(conversation_id)
    assert after["settings_version"] == before["settings_version"] + 1
    values = after["materialized_settings"]["values"]
    assert [item["source"]["id"] for item in values["participants"]] == [
        str(sources["primary_id"])
    ]
    assert values["memory"]["character_memory_by_id"] == {}
    assert values["behavior_controls"]["applied_overrides"][
        "characterMemoryById"
    ] == {}


@pytest.mark.integration
def test_replayed_greeting_selection_reuses_frozen_content_and_rejects_malformed_id(
    test_client,
    auth_headers,
    character_db,
):
    character_id = character_db.add_character_card(
        {
            "name": "Stable Greeting Ari",
            "first_message": "Default greeting",
            "alternate_greetings": ["Frozen alternate"],
        }
    )
    created = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={
            "character_id": character_id,
            "provider": "local-llm",
            "model": "local-test",
        },
    )
    assert created.status_code == 201, created.text
    conversation_id = created.json()["id"]

    selected = test_client.put(
        f"/api/v1/chats/{conversation_id}/greetings/select",
        headers=auth_headers,
        json={"index": 1},
    )
    assert selected.status_code == 200, selected.text
    frozen = character_db.get_roleplay_resume_state(conversation_id)
    frozen_greeting = frozen["materialized_settings"]["values"]["greeting"]
    assert frozen_greeting["content"] == "Frozen alternate"

    character = character_db.get_character_card_by_id(character_id)
    assert character_db.update_character_card(
        character_id,
        {"alternate_greetings": ["Mutable replacement"]},
        expected_version=character["version"],
    )
    replayed = test_client.put(
        f"/api/v1/chats/{conversation_id}/greetings/select",
        headers=auth_headers,
        json={"index": 1},
    )
    assert replayed.status_code == 200, replayed.text
    replayed_state = character_db.get_roleplay_resume_state(conversation_id)
    assert replayed_state["settings_version"] == frozen["settings_version"] + 1
    assert replayed_state["materialized_settings"]["values"]["greeting"] == frozen_greeting

    whitespace_replay = test_client.put(
        f"/api/v1/chats/{conversation_id}/settings",
        headers=auth_headers,
        json={
            "settings": {"greetingSelectionId": "  greeting:1:selected  "}
        },
    )
    assert whitespace_replay.status_code == 200, whitespace_replay.text
    whitespace_state = character_db.get_roleplay_resume_state(conversation_id)
    assert whitespace_state["settings_version"] == replayed_state["settings_version"] + 1
    whitespace_values = whitespace_state["materialized_settings"]["values"]
    assert whitespace_values["greeting"] == frozen_greeting
    assert whitespace_values["behavior_controls"]["greeting"][
        "selection_id"
    ] == "greeting:1:selected"

    for malformed_selection_id in ("not-a-greeting", "greeting:1"):
        malformed = test_client.put(
            f"/api/v1/chats/{conversation_id}/settings",
            headers=auth_headers,
            json={
                "settings": {"greetingSelectionId": malformed_selection_id}
            },
        )
        assert malformed.status_code in {400, 422}, malformed.text
        assert character_db.get_roleplay_resume_state(conversation_id)[
            "settings_version"
        ] == whitespace_state["settings_version"]


@pytest.mark.integration
def test_creation_caps_world_books_before_entry_expansion(
    test_client,
    auth_headers,
    character_db,
):
    character_id = character_db.add_character_card({"name": "Bounded Creation Lore"})
    world_books = WorldBookService(character_db)
    for index in range(character_conversation_factory.MAX_MATERIALIZED_WORLD_BOOKS + 1):
        world_book_id = world_books.create_world_book(f"Creation lore {index}")
        assert world_books.attach_to_character(world_book_id, character_id)[
            "success"
        ] is True

    real_select = character_conversation_factory._select_rows
    entry_query_count = 0

    def _counting_select(conn, query, params):
        nonlocal entry_query_count
        if "FROM world_book_entries" in query:
            entry_query_count += 1
        return real_select(conn, query, params)

    with mock.patch.object(
        character_conversation_factory,
        "_select_rows",
        side_effect=_counting_select,
    ):
        response = test_client.post(
            "/api/v1/chats/",
            headers=auth_headers,
            json={
                "character_id": character_id,
                "provider": "local-llm",
                "model": "local-test",
            },
        )

    assert response.status_code in {400, 413, 422}, response.text
    assert entry_query_count == 0


@pytest.mark.integration
def test_creation_materializes_shared_world_book_entries_once_per_snapshot_pass(
    test_client,
    auth_headers,
    character_db,
):
    primary_id = character_db.add_character_card({"name": "Shared Lore Primary"})
    secondary_id = character_db.add_character_card({"name": "Shared Lore Secondary"})
    world_books = WorldBookService(character_db)
    world_book_id = world_books.create_world_book("Shared creation lore")
    world_books.add_world_book_entry(
        world_book_id,
        ["shared"],
        "One canonical entry expansion.",
    )
    assert world_books.attach_to_character(world_book_id, primary_id)["success"] is True
    assert world_books.attach_to_character(world_book_id, secondary_id)["success"] is True

    real_select = character_conversation_factory._select_rows
    entry_query_count = 0

    def _counting_select(conn, query, params):
        nonlocal entry_query_count
        if "FROM world_book_entries" in query:
            entry_query_count += 1
        return real_select(conn, query, params)

    with mock.patch.object(
        character_conversation_factory,
        "_select_rows",
        side_effect=_counting_select,
    ):
        response = test_client.post(
            "/api/v1/chats/",
            headers=auth_headers,
            json={
                "character_id": primary_id,
                "participant_character_ids": [secondary_id],
                "provider": "local-llm",
                "model": "local-test",
            },
        )

    assert response.status_code == 201, response.text
    # Creation performs one drift-check pass and one transactional persistence pass.
    assert entry_query_count == 2


@pytest.mark.integration
def test_creation_rejects_total_exemplar_count_before_expansion(
    test_client,
    auth_headers,
    character_db,
    monkeypatch,
) -> None:
    primary_id = character_db.add_character_card({"name": "Exemplar Count Primary"})
    secondary_id = character_db.add_character_card({"name": "Exemplar Count Secondary"})
    for character_id, indexes in ((primary_id, range(1)), (secondary_id, range(2))):
        for index in indexes:
            character_db.add_character_exemplar(
                character_id,
                {
                    "id": f"bounded-exemplar-{character_id}-{index}",
                    "text": f"Exemplar {character_id}-{index}",
                },
            )
    monkeypatch.setattr(
        character_conversation_factory,
        "MAX_MATERIALIZED_EXEMPLARS",
        2,
        raising=False,
    )
    real_select = character_conversation_factory._select_rows
    detail_query_count = 0

    def _counting_select(conn, query, params):
        nonlocal detail_query_count
        if "FROM character_exemplars" in query:
            detail_query_count += 1
        return real_select(conn, query, params)

    with mock.patch.object(
        character_conversation_factory,
        "_select_rows",
        side_effect=_counting_select,
    ):
        response = test_client.post(
            "/api/v1/chats/",
            headers=auth_headers,
            json={
                "character_id": primary_id,
                "participant_character_ids": [secondary_id],
                "provider": "local-llm",
                "model": "local-test",
            },
        )

    assert response.status_code in {400, 413, 422}, response.text
    assert "exemplar" in response.text.lower()
    assert detail_query_count == 0
    assert character_db.get_conversations_for_character(primary_id) == []


@pytest.mark.integration
def test_creation_rejects_total_exemplar_bytes_before_expansion(
    test_client,
    auth_headers,
    character_db,
    monkeypatch,
) -> None:
    character_id = character_db.add_character_card({"name": "Exemplar Bytes"})
    character_db.add_character_exemplar(
        character_id,
        {
            "id": "oversized-exemplar",
            "text": "x" * 128,
        },
    )
    monkeypatch.setattr(
        character_conversation_factory,
        "MAX_MATERIALIZED_EXEMPLAR_BYTES",
        64,
        raising=False,
    )
    real_select = character_conversation_factory._select_rows
    detail_query_count = 0

    def _counting_select(conn, query, params):
        nonlocal detail_query_count
        if "FROM character_exemplars" in query:
            detail_query_count += 1
        return real_select(conn, query, params)

    with mock.patch.object(
        character_conversation_factory,
        "_select_rows",
        side_effect=_counting_select,
    ):
        response = test_client.post(
            "/api/v1/chats/",
            headers=auth_headers,
            json={
                "character_id": character_id,
                "provider": "local-llm",
                "model": "local-test",
            },
        )

    assert response.status_code in {400, 413, 422}, response.text
    assert "exemplar" in response.text.lower()
    assert "byte" in response.text.lower()
    assert detail_query_count == 0
    assert character_db.get_conversations_for_character(character_id) == []


@pytest.mark.integration
def test_creation_rejects_total_world_book_entry_count_before_expansion(
    test_client,
    auth_headers,
    character_db,
    monkeypatch,
) -> None:
    primary_id = character_db.add_character_card({"name": "Lore Count Primary"})
    secondary_id = character_db.add_character_card({"name": "Lore Count Secondary"})
    world_books = WorldBookService(character_db)
    for character_id, entry_count in ((primary_id, 1), (secondary_id, 2)):
        book_id = world_books.create_world_book(f"Count lore {character_id}")
        assert world_books.attach_to_character(book_id, character_id)["success"] is True
        for index in range(entry_count):
            world_books.add_world_book_entry(
                book_id,
                [f"count-{index}"],
                f"Count lore entry {character_id}-{index}",
            )
    monkeypatch.setattr(
        character_conversation_factory,
        "MAX_MATERIALIZED_WORLD_BOOK_ENTRIES",
        2,
        raising=False,
    )
    real_select = character_conversation_factory._select_rows
    detail_query_count = 0

    def _counting_select(conn, query, params):
        nonlocal detail_query_count
        if "FROM world_book_entries" in query:
            detail_query_count += 1
        return real_select(conn, query, params)

    with mock.patch.object(
        character_conversation_factory,
        "_select_rows",
        side_effect=_counting_select,
    ):
        response = test_client.post(
            "/api/v1/chats/",
            headers=auth_headers,
            json={
                "character_id": primary_id,
                "participant_character_ids": [secondary_id],
                "provider": "local-llm",
                "model": "local-test",
            },
        )

    assert response.status_code in {400, 413, 422}, response.text
    assert "world-book entries" in response.text.lower()
    assert detail_query_count == 0
    assert character_db.get_conversations_for_character(primary_id) == []


@pytest.mark.integration
def test_creation_rejects_total_world_book_entry_bytes_before_expansion(
    test_client,
    auth_headers,
    character_db,
    monkeypatch,
) -> None:
    primary_id = character_db.add_character_card({"name": "Lore Bytes Primary"})
    secondary_id = character_db.add_character_card({"name": "Lore Bytes Secondary"})
    world_books = WorldBookService(character_db)
    for character_id in (primary_id, secondary_id):
        book_id = world_books.create_world_book(f"Byte lore {character_id}")
        assert world_books.attach_to_character(book_id, character_id)["success"] is True
        world_books.add_world_book_entry(
            book_id,
            ["byte-budget"],
            "y" * 64,
        )
    monkeypatch.setattr(
        character_conversation_factory,
        "MAX_MATERIALIZED_WORLD_BOOK_ENTRY_BYTES",
        96,
        raising=False,
    )
    real_select = character_conversation_factory._select_rows
    detail_query_count = 0

    def _counting_select(conn, query, params):
        nonlocal detail_query_count
        if "FROM world_book_entries" in query:
            detail_query_count += 1
        return real_select(conn, query, params)

    with mock.patch.object(
        character_conversation_factory,
        "_select_rows",
        side_effect=_counting_select,
    ):
        response = test_client.post(
            "/api/v1/chats/",
            headers=auth_headers,
            json={
                "character_id": primary_id,
                "participant_character_ids": [secondary_id],
                "provider": "local-llm",
                "model": "local-test",
            },
        )

    assert response.status_code in {400, 413, 422}, response.text
    assert "world-book entries" in response.text.lower()
    assert "byte" in response.text.lower()
    assert detail_query_count == 0
    assert character_db.get_conversations_for_character(primary_id) == []


class _PostgresBudgetBackend:
    backend_type = BackendType.POSTGRESQL


class _PostgresBudgetResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)

    def keys(self):
        return list(self._rows[0]) if self._rows else []


class _PostgresWorldBookBudgetConnection:
    def __init__(self, *, row_count: int, byte_count: int):
        self._backend = _PostgresBudgetBackend()
        self.row_count = row_count
        self.byte_count = byte_count
        self.queries: list[tuple[str, tuple]] = []

    def execute(self, query, params=()):
        normalized = " ".join(query.split())
        self.queries.append((normalized, tuple(params)))
        if "FROM world_books wb" in normalized:
            return _PostgresBudgetResult(
                [
                    {
                        "id": int(params[0]),
                        "name": f"Owned lore {params[0]}",
                        "deleted": False,
                        "attachment_enabled": True,
                        "attachment_priority": 0,
                    }
                ]
            )
        if "COUNT(*) AS row_count" in normalized:
            return _PostgresBudgetResult(
                [{"row_count": self.row_count, "byte_count": self.byte_count}]
            )
        if "FROM world_book_entries" in normalized:
            raise AssertionError("entry detail rows must not be expanded after overflow")
        raise AssertionError(f"Unexpected query: {normalized}")


class _PostgresForeignCharacterConnection:
    def __init__(self) -> None:
        self._backend = _PostgresBudgetBackend()
        self.queries: list[tuple[str, tuple]] = []

    def execute(self, query, params=()):
        normalized = " ".join(query.split())
        self.queries.append((normalized, tuple(params)))
        if "FROM character_exemplars" in normalized:
            raise AssertionError(
                "foreign character exemplar rows must not be queried before ownership proof"
            )
        if "FROM world_books wb" in normalized:
            return _PostgresBudgetResult([])
        if "FROM character_cards" in normalized:
            return _PostgresBudgetResult([])
        raise AssertionError(f"Unexpected query: {normalized}")


@pytest.mark.integration
def test_postgres_foreign_character_is_rejected_before_exemplar_queries() -> None:
    conn = _PostgresForeignCharacterConnection()

    with pytest.raises(InputError, match="Character ID 999 not found"):
        character_conversation_factory._materialize_behavior(
            conn,
            participant_character_ids=[999],
            prompt_preset_id=None,
            memory_by_character_id={},
            primary_greeting=None,
            owner_user_id="alice",
            max_snapshot_bytes=1024 * 1024,
        )

    assert all("FROM character_exemplars" not in query for query, _params in conn.queries)
    character_query, params = next(
        (query, params)
        for query, params in conn.queries
        if "FROM character_cards" in query
    )
    assert "client_id = ?" in character_query
    assert params == (999, "alice")


@pytest.mark.integration
@pytest.mark.parametrize(
    ("row_count", "byte_count", "message"),
    [(3, 1, "count"), (1, 97, "byte")],
)
def test_postgres_world_book_entry_budget_preserves_owner_linkage_before_expansion(
    monkeypatch,
    row_count: int,
    byte_count: int,
    message: str,
) -> None:
    monkeypatch.setattr(
        character_conversation_factory,
        "MAX_MATERIALIZED_WORLD_BOOK_ENTRIES",
        2,
        raising=False,
    )
    monkeypatch.setattr(
        character_conversation_factory,
        "MAX_MATERIALIZED_WORLD_BOOK_ENTRY_BYTES",
        96,
        raising=False,
    )
    conn = _PostgresWorldBookBudgetConnection(
        row_count=row_count,
        byte_count=byte_count,
    )

    with pytest.raises(InputError, match=f"world-book entries.*{message}"):
        character_conversation_factory._load_world_books_for_participants(
            conn,
            [11, 12],
            owner_user_id="alice",
        )

    attachment_queries = [
        (query, params)
        for query, params in conn.queries
        if "FROM world_books wb" in query
    ]
    assert len(attachment_queries) == 2
    assert all("cc.client_id = ?" in query for query, _params in attachment_queries)
    assert [params for _query, params in attachment_queries] == [
        (11, "alice"),
        (12, "alice"),
    ]


@pytest.mark.integration
def test_participant_mutation_caps_combined_new_world_books_before_entry_expansion(
    test_client,
    auth_headers,
    character_db,
):
    primary_id = character_db.add_character_card({"name": "Mutation Lore Primary"})
    second_id = character_db.add_character_card({"name": "Mutation Lore Second"})
    third_id = character_db.add_character_card({"name": "Mutation Lore Third"})
    world_books = WorldBookService(character_db)
    for index in range(33):
        world_book_id = world_books.create_world_book(f"Second lore {index}")
        assert world_books.attach_to_character(world_book_id, second_id)["success"] is True
    for index in range(32):
        world_book_id = world_books.create_world_book(f"Third lore {index}")
        assert world_books.attach_to_character(world_book_id, third_id)["success"] is True

    created = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={
            "character_id": primary_id,
            "provider": "local-llm",
            "model": "local-test",
        },
    )
    assert created.status_code == 201, created.text
    conversation_id = created.json()["id"]
    before = character_db.get_roleplay_resume_state(conversation_id)

    real_select = character_conversation_factory._select_rows
    entry_query_count = 0

    def _counting_select(conn, query, params):
        nonlocal entry_query_count
        if "FROM world_book_entries" in query:
            entry_query_count += 1
        return real_select(conn, query, params)

    with mock.patch.object(
        character_conversation_factory,
        "_select_rows",
        side_effect=_counting_select,
    ):
        response = test_client.put(
            f"/api/v1/chats/{conversation_id}/settings",
            headers=auth_headers,
            json={
                "settings": {
                    "participantCharacterIds": [primary_id, second_id, third_id]
                }
            },
        )

    assert response.status_code in {400, 413, 422}, response.text
    assert entry_query_count == 0
    after = character_db.get_roleplay_resume_state(conversation_id)
    assert after["settings_version"] == before["settings_version"]
    assert after["settings"] == before["settings"]


@pytest.mark.integration
@pytest.mark.parametrize("budget_kind", ["count", "bytes"])
def test_participant_addition_budgets_frozen_and_new_exemplars_together(
    test_client,
    auth_headers,
    character_db,
    monkeypatch,
    budget_kind: str,
) -> None:
    primary_id = character_db.add_character_card({"name": "Frozen Exemplar Primary"})
    secondary_id = character_db.add_character_card({"name": "New Exemplar Secondary"})
    character_db.add_character_exemplar(
        primary_id,
        {"id": "frozen-exemplar", "text": "f" * 64},
    )
    new_count = 2 if budget_kind == "count" else 1
    for index in range(new_count):
        character_db.add_character_exemplar(
            secondary_id,
            {"id": f"new-exemplar-{index}", "text": "n" * 64},
        )

    created = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={
            "character_id": primary_id,
            "provider": "local-llm",
            "model": "local-test",
        },
    )
    assert created.status_code == 201, created.text
    conversation_id = created.json()["id"]
    before = character_db.get_roleplay_resume_state(conversation_id)
    if budget_kind == "count":
        monkeypatch.setattr(
            character_conversation_factory,
            "MAX_MATERIALIZED_EXEMPLARS",
            2,
        )
    else:
        frozen_count, frozen_bytes = (
            character_conversation_factory._materialized_exemplar_usage(
                before["behavior_snapshot"]["payload"]["participants"]
            )
        )
        assert frozen_count == 1
        with character_db.transaction() as conn:
            new_rows, new_bytes = character_conversation_factory._source_collection_stats(
                conn,
                table="character_exemplars",
                foreign_key="character_id",
                source_ids=[secondary_id],
                text_columns=(
                    character_conversation_factory._MATERIALIZED_EXEMPLAR_TEXT_FIELDS
                ),
                extra_predicate="AND is_deleted = FALSE",
            )
        assert new_rows == 1
        monkeypatch.setattr(
            character_conversation_factory,
            "MAX_MATERIALIZED_EXEMPLAR_BYTES",
            frozen_bytes + new_bytes - 1,
        )

    real_select = character_conversation_factory._select_rows
    detail_query_count = 0

    def _counting_select(conn, query, params):
        nonlocal detail_query_count
        if "FROM character_exemplars" in query:
            detail_query_count += 1
        return real_select(conn, query, params)

    with mock.patch.object(
        character_conversation_factory,
        "_select_rows",
        side_effect=_counting_select,
    ):
        response = test_client.put(
            f"/api/v1/chats/{conversation_id}/settings",
            headers=auth_headers,
            json={
                "settings": {
                    "participantCharacterIds": [primary_id, secondary_id],
                }
            },
        )

    assert response.status_code in {400, 413, 422}, response.text
    assert "exemplar" in response.text.lower()
    assert detail_query_count == 0
    after = character_db.get_roleplay_resume_state(conversation_id)
    assert after["settings_version"] == before["settings_version"]
    assert after["settings"] == before["settings"]


@pytest.mark.integration
@pytest.mark.parametrize("budget_kind", ["count", "bytes"])
def test_participant_addition_budgets_frozen_and_new_world_book_entries_together(
    test_client,
    auth_headers,
    character_db,
    monkeypatch,
    budget_kind: str,
) -> None:
    primary_id = character_db.add_character_card({"name": "Frozen Lore Primary"})
    secondary_id = character_db.add_character_card({"name": "New Lore Secondary"})
    world_books = WorldBookService(character_db)
    primary_book_id = world_books.create_world_book("Frozen lore")
    assert world_books.attach_to_character(primary_book_id, primary_id)["success"] is True
    world_books.add_world_book_entry(primary_book_id, ["frozen"], "f" * 64)
    secondary_book_id = world_books.create_world_book("New lore")
    assert world_books.attach_to_character(secondary_book_id, secondary_id)["success"] is True
    new_count = 2 if budget_kind == "count" else 1
    for index in range(new_count):
        world_books.add_world_book_entry(
            secondary_book_id,
            [f"new-{index}"],
            "n" * 64,
        )

    created = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={
            "character_id": primary_id,
            "provider": "local-llm",
            "model": "local-test",
        },
    )
    assert created.status_code == 201, created.text
    conversation_id = created.json()["id"]
    before = character_db.get_roleplay_resume_state(conversation_id)
    if budget_kind == "count":
        monkeypatch.setattr(
            character_conversation_factory,
            "MAX_MATERIALIZED_WORLD_BOOK_ENTRIES",
            2,
        )
    else:
        frozen_books = before["behavior_snapshot"]["payload"]["participants"][0][
            "world_books"
        ]
        frozen_count, frozen_bytes = (
            character_conversation_factory._materialized_world_book_entry_usage(
                frozen_books
            )
        )
        assert frozen_count == 1
        with character_db.transaction() as conn:
            new_rows, new_bytes = character_conversation_factory._source_collection_stats(
                conn,
                table="world_book_entries",
                foreign_key="world_book_id",
                source_ids=[secondary_book_id],
                text_columns=("keywords", "content", "metadata"),
            )
        assert new_rows == 1
        monkeypatch.setattr(
            character_conversation_factory,
            "MAX_MATERIALIZED_WORLD_BOOK_ENTRY_BYTES",
            frozen_bytes + new_bytes - 1,
        )

    real_select = character_conversation_factory._select_rows
    detail_query_count = 0

    def _counting_select(conn, query, params):
        nonlocal detail_query_count
        if "FROM world_book_entries" in query:
            detail_query_count += 1
        return real_select(conn, query, params)

    with mock.patch.object(
        character_conversation_factory,
        "_select_rows",
        side_effect=_counting_select,
    ):
        response = test_client.put(
            f"/api/v1/chats/{conversation_id}/settings",
            headers=auth_headers,
            json={
                "settings": {
                    "participantCharacterIds": [primary_id, secondary_id],
                }
            },
        )

    assert response.status_code in {400, 413, 422}, response.text
    assert "world-book entries" in response.text.lower()
    assert detail_query_count == 0
    after = character_db.get_roleplay_resume_state(conversation_id)
    assert after["settings_version"] == before["settings_version"]
    assert after["settings"] == before["settings"]


@pytest.mark.integration
def test_participant_addition_reuses_frozen_shared_world_book_without_expansion(
    test_client,
    auth_headers,
    character_db,
    monkeypatch,
) -> None:
    primary_id = character_db.add_character_card({"name": "Shared Lore Primary"})
    secondary_id = character_db.add_character_card({"name": "Shared Lore Secondary"})
    world_books = WorldBookService(character_db)
    world_book_id = world_books.create_world_book("Frozen shared lore")
    world_books.add_world_book_entry(world_book_id, ["shared"], "One frozen entry")
    assert world_books.attach_to_character(world_book_id, primary_id)["success"] is True
    assert world_books.attach_to_character(world_book_id, secondary_id)["success"] is True
    created = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={
            "character_id": primary_id,
            "provider": "local-llm",
            "model": "local-test",
        },
    )
    assert created.status_code == 201, created.text
    conversation_id = created.json()["id"]
    monkeypatch.setattr(
        character_conversation_factory,
        "MAX_MATERIALIZED_WORLD_BOOK_ENTRIES",
        1,
    )
    real_select = character_conversation_factory._select_rows
    detail_query_count = 0

    def _counting_select(conn, query, params):
        nonlocal detail_query_count
        if "FROM world_book_entries" in query:
            detail_query_count += 1
        return real_select(conn, query, params)

    with mock.patch.object(
        character_conversation_factory,
        "_select_rows",
        side_effect=_counting_select,
    ):
        response = test_client.put(
            f"/api/v1/chats/{conversation_id}/settings",
            headers=auth_headers,
            json={
                "settings": {
                    "participantCharacterIds": [primary_id, secondary_id],
                }
            },
        )

    assert response.status_code == 200, response.text
    assert detail_query_count == 0
    participants = character_db.get_roleplay_resume_state(conversation_id)[
        "materialized_settings"
    ]["values"]["participants"]
    assert [participant["world_books"][0]["id"] for participant in participants] == [
        world_book_id,
        world_book_id,
    ]


@pytest.mark.integration
def test_context_mutation_applies_one_entry_budget_across_direct_world_books(
    test_client,
    auth_headers,
    character_db,
    monkeypatch,
) -> None:
    character_id = character_db.add_character_card({"name": "Direct Lore Budget"})
    created = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={
            "character_id": character_id,
            "provider": "local-llm",
            "model": "local-test",
        },
    )
    assert created.status_code == 201, created.text
    conversation_id = created.json()["id"]
    before = character_db.get_roleplay_resume_state(conversation_id)
    world_books = WorldBookService(character_db)
    book_ids = []
    for index in range(2):
        book_id = world_books.create_world_book(f"Direct lore {index}")
        world_books.add_world_book_entry(
            book_id,
            [f"direct-{index}"],
            f"Direct entry {index}",
        )
        book_ids.append(book_id)
    monkeypatch.setattr(
        character_conversation_factory,
        "MAX_MATERIALIZED_WORLD_BOOK_ENTRIES",
        1,
    )
    real_select = character_conversation_factory._select_rows
    detail_query_count = 0

    def _counting_select(conn, query, params):
        nonlocal detail_query_count
        if "FROM world_book_entries" in query:
            detail_query_count += 1
        return real_select(conn, query, params)

    with mock.patch.object(
        character_conversation_factory,
        "_select_rows",
        side_effect=_counting_select,
    ):
        updated = test_client.put(
            f"/api/v1/chats/{conversation_id}/settings",
            headers=auth_headers,
            json={
                "settings": {
                    "conversationContext": {"world_book_ids": book_ids}
                }
            },
        )

    assert updated.status_code in {400, 413, 422}, updated.text
    assert "world-book entries" in updated.text.lower()
    # The aggregate preflight covers both IDs before either book's entries expand.
    assert detail_query_count == 0
    after = character_db.get_roleplay_resume_state(conversation_id)
    assert after["settings_version"] == before["settings_version"]
    assert after["settings"] == before["settings"]


@pytest.mark.integration
@pytest.mark.parametrize(
    "binding_mutation",
    ["missing", "digest_mismatch", "version_mismatch"],
)
def test_resume_readiness_requires_materialized_authority_bound_to_snapshot(
    test_client,
    auth_headers,
    character_db,
    binding_mutation: str,
) -> None:
    character_id = character_db.add_character_card({"name": "Bound Authority"})
    created = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={
            "character_id": character_id,
            "provider": "local-llm",
            "model": "local-test",
        },
    )
    assert created.status_code == 201, created.text
    conversation_id = created.json()["id"]
    before = character_db.get_roleplay_resume_state(conversation_id)
    settings = dict(before["settings"])

    if binding_mutation == "missing":
        settings.pop("roleplayBehaviorV1")
    else:
        values = dict(before["materialized_settings"]["values"])
        values["base_snapshot"] = dict(values["base_snapshot"])
        if binding_mutation == "digest_mismatch":
            values["base_snapshot"]["digest"] = "sha256:" + ("f" * 64)
            settings["roleplayBehaviorV1"] = build_materialized_behavior_settings(values)
        else:
            values["base_snapshot"]["schema_version"] = 2
            raw = dict(settings["roleplayBehaviorV1"])
            raw["values"] = values
            settings["roleplayBehaviorV1"] = raw

    with character_db.transaction() as conn:
        conn.execute(
            "UPDATE conversation_settings SET settings_json = ? WHERE conversation_id = ?",
            (json.dumps(settings), conversation_id),
        )

    state = character_db.get_roleplay_resume_state(conversation_id)
    assert state["resume_eligible"] is False
    assert state["resume_ineligible_reason"] == "invalid_effective_settings"
    assert state["effective_completion"] is None


@pytest.mark.integration
def test_greeting_materialization_uses_live_nonempty_ordered_deduped_picker(
    test_client,
    auth_headers,
    character_db,
) -> None:
    character_id = character_db.add_character_card(
        {
            "name": "Greeting Parity",
            "first_message": "",
            "alternate_greetings": ["", "Hello", "Hello", "Goodbye"],
        }
    )
    created = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={
            "character_id": character_id,
            "provider": "local-llm",
            "model": "local-test",
        },
    )
    assert created.status_code == 201, created.text
    conversation_id = created.json()["id"]

    listed = test_client.get(
        f"/api/v1/chats/{conversation_id}/greetings",
        headers=auth_headers,
    )
    assert listed.status_code == 200, listed.text
    assert [item["text"] for item in listed.json()["greetings"]] == [
        "Hello",
        "Goodbye",
    ]

    selected = test_client.put(
        f"/api/v1/chats/{conversation_id}/greetings/select",
        headers=auth_headers,
        json={"index": 1},
    )
    assert selected.status_code == 200, selected.text
    greeting = character_db.get_roleplay_resume_state(conversation_id)[
        "materialized_settings"
    ]["values"]["greeting"]
    assert greeting["content"] == "Goodbye"


@pytest.mark.integration
def test_preset_scope_chat_without_override_materializes_builtin_default(
    test_client,
    auth_headers,
    character_db,
) -> None:
    character_id = character_db.add_character_card({"name": "Default Preset"})
    created = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={
            "character_id": character_id,
            "provider": "local-llm",
            "model": "local-test",
        },
    )
    assert created.status_code == 201, created.text
    conversation_id = created.json()["id"]

    updated = test_client.put(
        f"/api/v1/chats/{conversation_id}/settings",
        headers=auth_headers,
        json={"settings": {"presetScope": "chat"}},
    )
    assert updated.status_code == 200, updated.text
    preset = character_db.get_roleplay_resume_state(conversation_id)[
        "materialized_settings"
    ]["values"]["prompt_preset"]
    assert preset["preset_id"] == "default"
    assert preset["source"] == {
        "kind": "builtin_prompt_preset",
        "id": "default",
        "version": 1,
    }


@pytest.mark.integration
def test_preset_scope_character_ignores_retained_chat_override(
    test_client,
    auth_headers,
    character_db,
) -> None:
    sources = _create_behavior_sources(character_db)
    created = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={
            "character_id": sources["primary_id"],
            "prompt_preset_id": "snapshot-cinematic",
            "provider": "local-llm",
            "model": "local-test",
        },
    )
    assert created.status_code == 201, created.text
    conversation_id = created.json()["id"]
    before = character_db.get_roleplay_resume_state(conversation_id)
    assert before["settings"]["chatPresetOverrideId"] == "snapshot-cinematic"
    initial_preset = before["behavior_snapshot"]["payload"]["participants"][0][
        "prompt"
    ]["prompt_relevant_extensions"]["prompt_preset"]
    assert initial_preset["preset_id"] == "snapshot-cinematic"

    updated = test_client.put(
        f"/api/v1/chats/{conversation_id}/settings",
        headers=auth_headers,
        json={"settings": {"presetScope": "character"}},
    )

    assert updated.status_code == 200, updated.text
    after = character_db.get_roleplay_resume_state(conversation_id)
    assert after["settings"]["chatPresetOverrideId"] == "snapshot-cinematic"
    assert after["materialized_settings"]["values"]["behavior_controls"][
        "preset_scope"
    ] == "character"
    preset = after["materialized_settings"]["values"]["prompt_preset"]
    assert preset["preset_id"] == "default"
    assert preset["selection_source"] == "default"


@pytest.mark.integration
def test_creation_freezes_active_persona_memory_entries(
    test_client,
    auth_headers,
    character_db,
) -> None:
    character_id = character_db.add_character_card({"name": "Memory Freeze"})
    persona_id = f"char:{character_id}"
    character_db.create_persona_profile(
        {"id": persona_id, "user_id": "1", "name": "Memory Freeze"}
    )
    memory_id = character_db.add_persona_memory_entry(
        {
            "persona_id": persona_id,
            "user_id": "1",
            "memory_type": "fact",
            "content": "The original immutable memory.",
            "salience": 0.9,
        }
    )
    created = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={
            "character_id": character_id,
            "provider": "local-llm",
            "model": "local-test",
        },
    )
    assert created.status_code == 201, created.text
    conversation_id = created.json()["id"]
    before = character_db.get_roleplay_resume_state(conversation_id)
    frozen = before["behavior_snapshot"]["payload"]["participants"][0][
        "default_memory"
    ]["persona_memory_entries"]
    assert [entry["content"] for entry in frozen] == [
        "The original immutable memory."
    ]

    with character_db.transaction() as conn:
        conn.execute(
            "UPDATE persona_memory_entries SET content = ?, version = version + 1 WHERE id = ?",
            ("Mutable replacement.", memory_id),
        )
    after = character_db.get_roleplay_resume_state(conversation_id)
    assert after["behavior_snapshot"]["payload"] == before["behavior_snapshot"][
        "payload"
    ]


@pytest.mark.integration
def test_participant_mutation_freezes_active_persona_memory_entries(
    test_client,
    auth_headers,
    character_db,
) -> None:
    primary_id = character_db.add_character_card({"name": "Memory Primary"})
    secondary_id = character_db.add_character_card({"name": "Memory Secondary"})
    persona_id = f"char:{secondary_id}"
    character_db.create_persona_profile(
        {"id": persona_id, "user_id": "1", "name": "Memory Secondary"}
    )
    memory_id = character_db.add_persona_memory_entry(
        {
            "persona_id": persona_id,
            "user_id": "1",
            "memory_type": "relationship",
            "content": "The secondary remembers the original pact.",
            "salience": 0.8,
        }
    )
    created = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={
            "character_id": primary_id,
            "provider": "local-llm",
            "model": "local-test",
        },
    )
    assert created.status_code == 201, created.text
    conversation_id = created.json()["id"]

    updated = test_client.put(
        f"/api/v1/chats/{conversation_id}/settings",
        headers=auth_headers,
        json={"settings": {"participantCharacterIds": [primary_id, secondary_id]}},
    )
    assert updated.status_code == 200, updated.text
    frozen_state = character_db.get_roleplay_resume_state(conversation_id)
    secondary = next(
        participant
        for participant in frozen_state["materialized_settings"]["values"][
            "participants"
        ]
        if participant["source"]["id"] == str(secondary_id)
    )
    assert [
        entry["content"]
        for entry in secondary["default_memory"]["persona_memory_entries"]
    ] == ["The secondary remembers the original pact."]

    with character_db.transaction() as conn:
        conn.execute(
            "UPDATE persona_memory_entries SET content = ?, version = version + 1 WHERE id = ?",
            ("Mutable replacement pact.", memory_id),
        )
    assert character_db.get_roleplay_resume_state(conversation_id)[
        "materialized_settings"
    ] == frozen_state["materialized_settings"]


@pytest.mark.integration
def test_world_book_cap_applies_to_participant_and_conversation_context_union(
    test_client,
    auth_headers,
    character_db,
) -> None:
    character_id = character_db.add_character_card({"name": "Total Lore Cap"})
    world_books = WorldBookService(character_db)
    for index in range(32):
        book_id = world_books.create_world_book(f"Attached total lore {index}")
        assert world_books.attach_to_character(book_id, character_id)["success"] is True
    explicit_ids = [
        world_books.create_world_book(f"Explicit total lore {index}")
        for index in range(33)
    ]
    created = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={
            "character_id": character_id,
            "provider": "local-llm",
            "model": "local-test",
        },
    )
    assert created.status_code == 201, created.text
    conversation_id = created.json()["id"]
    before = character_db.get_roleplay_resume_state(conversation_id)

    with mock.patch.object(
        character_conversation_factory,
        "_materialize_world_books_by_id",
        wraps=character_conversation_factory._materialize_world_books_by_id,
    ) as materialize_book:
        updated = test_client.put(
            f"/api/v1/chats/{conversation_id}/settings",
            headers=auth_headers,
            json={
                "settings": {
                    "conversationContext": {"world_book_ids": explicit_ids}
                }
            },
        )

    assert updated.status_code in {400, 413, 422}, updated.text
    assert materialize_book.call_count == 0
    after = character_db.get_roleplay_resume_state(conversation_id)
    assert after["settings_version"] == before["settings_version"]
    assert after["settings"] == before["settings"]


@pytest.mark.integration
def test_postgres_style_settings_materialization_rejects_sustained_source_drift(
    test_client,
    auth_headers,
    character_db,
    monkeypatch,
) -> None:
    character_id = character_db.add_character_card({"name": "Mutation Drift"})
    created = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={
            "character_id": character_id,
            "provider": "local-llm",
            "model": "local-test",
        },
    )
    assert created.status_code == 201, created.text
    conversation_id = created.json()["id"]
    before = character_db.get_roleplay_resume_state(conversation_id)
    call_count = 0

    def _drifting_book(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        return {
            9: {
                "id": 9,
                "name": "Drifting lore",
                "version": call_count,
                "entries": [{"id": 1, "content": f"revision-{call_count}"}],
            }
        }

    monkeypatch.setattr(
        character_conversation_factory,
        "_is_postgres_connection",
        lambda _conn: True,
        raising=False,
    )
    monkeypatch.setattr(
        character_conversation_factory,
        "_materialize_world_books_by_id",
        _drifting_book,
    )
    updated = test_client.put(
        f"/api/v1/chats/{conversation_id}/settings",
        headers=auth_headers,
        json={"settings": {"conversationContext": {"world_book_ids": [9]}}},
    )

    assert updated.status_code in {400, 409, 422}, updated.text
    assert call_count == 4
    after = character_db.get_roleplay_resume_state(conversation_id)
    assert after["settings_version"] == before["settings_version"]
    assert after["settings"] == before["settings"]


@pytest.mark.integration
def test_module_creator_materializes_database_owner_memory(
    character_db,
    monkeypatch,
) -> None:
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "local-llm")
    monkeypatch.setenv("CHAR_CHAT_MODEL", "local-test")
    owner_user_id = str(character_db.client_id)
    character_id = character_db.add_character_card({"name": "Owned Memory Creator"})
    persona_id = f"char:{character_id}"
    character_db.create_persona_profile(
        {"id": persona_id, "user_id": owner_user_id, "name": "Owned Memory Creator"}
    )
    character_db.add_persona_memory_entry(
        {
            "persona_id": persona_id,
            "user_id": owner_user_id,
            "memory_type": "fact",
            "content": "Only the database owner can materialize this memory.",
            "salience": 1.0,
        }
    )

    conversation_id, _character, _history, _image = start_new_chat_session(
        character_db,
        character_id,
        "Alice",
    )

    assert conversation_id is not None
    conversation = character_db.get_conversation_by_id(conversation_id)
    assert conversation["client_id"] == owner_user_id
    participant = character_db.get_roleplay_resume_state(conversation_id)[
        "behavior_snapshot"
    ]["payload"]["participants"][0]
    assert [
        entry["content"]
        for entry in participant["default_memory"]["persona_memory_entries"]
    ] == ["Only the database owner can materialize this memory."]


@pytest.mark.integration
def test_factory_derives_owner_from_scoped_database(
    character_db,
) -> None:
    owner_user_id = str(character_db.client_id)
    character_id = character_db.add_character_card({"name": "Derived Factory Owner"})
    persona_id = f"char:{character_id}"
    character_db.create_persona_profile(
        {"id": persona_id, "user_id": owner_user_id, "name": "Derived Factory Owner"}
    )
    character_db.add_persona_memory_entry(
        {
            "persona_id": persona_id,
            "user_id": owner_user_id,
            "memory_type": "fact",
            "content": "The factory must derive this scoped memory owner.",
            "salience": 1.0,
        }
    )

    conversation_id = create_character_conversation(
        character_db,
        conversation_data={"character_id": character_id, "title": "Derived owner"},
        provider="local-llm",
        model="local-test",
    )

    conversation = character_db.get_conversation_by_id(conversation_id)
    assert conversation["client_id"] == owner_user_id
    participant = character_db.get_roleplay_resume_state(conversation_id)[
        "behavior_snapshot"
    ]["payload"]["participants"][0]
    assert [
        entry["content"]
        for entry in participant["default_memory"]["persona_memory_entries"]
    ] == ["The factory must derive this scoped memory owner."]


@pytest.mark.integration
def test_factory_rejects_conflicting_conversation_owner(character_db) -> None:
    character_id = character_db.add_character_card({"name": "Conflicting Owner"})

    with pytest.raises(InputError, match="scoped database owner"):
        create_character_conversation(
            character_db,
            conversation_data={
                "character_id": character_id,
                "title": "Wrong owner",
                "client_id": "mallory",
            },
            provider="local-llm",
            model="local-test",
        )

    assert character_db.get_conversations_for_character(character_id) == []


@pytest.mark.integration
def test_module_creator_uses_database_owner_for_postgres_custom_preset(
    character_db,
    monkeypatch,
) -> None:
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "local-llm")
    monkeypatch.setenv("CHAR_CHAT_MODEL", "local-test")
    assert character_db.upsert_prompt_preset(
        preset_id="owner-only-preset",
        name="Owner-only preset",
        section_order=["system_prompt"],
        section_templates={"system_prompt": "{{system_prompt}}"},
    )
    character_id = character_db.add_character_card({"name": "Owned Preset Creator"})
    monkeypatch.setattr(
        character_conversation_factory,
        "resolve_character_prompt_preset",
        lambda _character: "owner-only-preset",
    )
    monkeypatch.setattr(
        character_conversation_factory,
        "_is_postgres_connection",
        lambda _conn: True,
    )

    conversation_id, _character, _history, _image = start_new_chat_session(
        character_db,
        character_id,
        "Alice",
    )

    assert conversation_id is not None
    preset = character_db.get_roleplay_resume_state(conversation_id)[
        "behavior_snapshot"
    ]["payload"]["participants"][0]["prompt"]["prompt_relevant_extensions"][
        "prompt_preset"
    ]
    assert preset["preset_id"] == "owner-only-preset"
    assert preset["source"] == {
        "kind": "prompt_preset",
        "id": "owner-only-preset",
        "version": 1,
    }
