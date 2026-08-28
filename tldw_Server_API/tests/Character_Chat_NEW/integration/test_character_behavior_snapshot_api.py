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
from tldw_Server_API.app.core.Character_Chat.modules.character_io import (
    load_chat_history_from_file_and_save_to_db,
)
from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService
from tldw_Server_API.app.core.DB_Management.chacha.conversation_resume_store import (
    ConversationResumeStore,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError


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
                "api_key": "never-store-this-credential",
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
    assert "never-store-this-credential" not in snapshot_bytes
    assert "never-store-this-credential" not in settings_bytes


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
            "client_id": "1",
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
            "client_id": "1",
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
            "client_id": "1",
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
            "client_id": "1",
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
            "client_id": "1",
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
                "client_id": "1",
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
            "client_id": "1",
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
            "client_id": "1",
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
            "client_id": "1",
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
                "client_id": "test_client",
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
            "client_id": "test_client",
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
                "client_id": "test_client",
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
def test_legacy_and_invalid_storage_are_non_resumable_and_body_redacted(
    test_client,
    auth_headers,
    character_db,
):
    character_id = character_db.add_character_card({"name": "Legacy Ari"})
    legacy_id = character_db.add_conversation(
        {"character_id": character_id, "title": "Legacy", "client_id": "1"}
    )
    invalid_id = character_db.add_conversation(
        {"character_id": character_id, "title": "Invalid", "client_id": "1"}
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
            "client_id": "1",
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
