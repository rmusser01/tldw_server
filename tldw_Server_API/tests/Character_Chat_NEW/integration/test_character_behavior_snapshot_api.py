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
                "tldw": {
                    "generation": {
                        "temperature": 0.35,
                        "top_p": 0.8,
                        "repetition_penalty": 1.15,
                        "stop": ["<END>"],
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
            "provider": "openai",
            "model": "gpt-snapshot",
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
    assert primary["greeting"] == {
        "content": "The archive remembers you, {{user}}.",
        "source": "alternate",
        "source_index": 0,
    }
    assert state["effective_completion"] == {
        "provider": "openai",
        "model": "gpt-snapshot",
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
            "provider": "openai",
            "model": "gpt-snapshot",
            "temperature": 0.0,
            "top_p": 0.0,
            "repetition_penalty": 0.0,
            "stop": [],
        },
    )
    assert response.status_code == 201, response.text
    conversation_id = response.json()["id"]
    before_bytes = _snapshot_storage_bytes(character_db, conversation_id)
    before_effective = character_db.get_roleplay_resume_state(conversation_id)["effective_completion"]

    primary = character_db.get_character_card_by_id(sources["primary_id"])
    assert character_db.update_character_card(
        sources["primary_id"],
        {
            "system_prompt": "Changed after creation.",
            "first_message": "Changed greeting.",
            "extensions": {"tldw": {"generation": {"temperature": 1.9}}},
        },
        expected_version=primary["version"],
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
                "provider": "openai",
                "model": "gpt-snapshot",
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
            provider="openai",
            model="gpt-snapshot",
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
def test_creation_with_incomplete_settings_is_explicitly_non_resumable(
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
        provider="openai",
        model="gpt-snapshot",
        sampling={"temperature": 0.0},
    )
    before = _snapshot_storage_bytes(character_db, conversation_id)

    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "changed-provider")
    monkeypatch.setenv("CHAR_CHAT_MODEL", "changed-model")
    state = character_db.get_roleplay_resume_state(conversation_id)

    assert state["behavior_snapshot"]["status"] == "valid"
    assert state["resume_eligible"] is False
    assert state["resume_ineligible_reason"] == "incomplete_effective_settings"
    assert state["effective_completion"] is None
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
            provider="openai",
            model="gpt-snapshot",
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
