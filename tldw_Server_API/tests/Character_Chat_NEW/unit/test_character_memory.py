"""Unit tests for cross-session character memory.

Covers:
- Schema validation
- Extraction prompt building and response parsing
- Deduplication logic
- Context injection formatting
- Persona profile auto-creation helper
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestCharacterMemorySchemas:
    """Pydantic schema validation tests."""

    def test_create_defaults(self):
        from tldw_Server_API.app.api.v1.schemas.character_memory_schemas import CharacterMemoryCreate
        m = CharacterMemoryCreate(content="User likes cats")
        assert m.memory_type == "manual"
        assert m.salience == 0.7

    def test_create_all_types(self):
        from tldw_Server_API.app.api.v1.schemas.character_memory_schemas import CharacterMemoryCreate
        for mt in ("fact", "relationship", "event", "preference", "manual"):
            m = CharacterMemoryCreate(content="test", memory_type=mt)
            assert m.memory_type == mt

    def test_create_salience_bounds(self):
        from tldw_Server_API.app.api.v1.schemas.character_memory_schemas import CharacterMemoryCreate
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            CharacterMemoryCreate(content="test", salience=-0.1)
        with pytest.raises(ValidationError):
            CharacterMemoryCreate(content="test", salience=1.5)

    def test_create_empty_content_rejected(self):
        from tldw_Server_API.app.api.v1.schemas.character_memory_schemas import CharacterMemoryCreate
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            CharacterMemoryCreate(content="")

    def test_update_partial(self):
        from tldw_Server_API.app.api.v1.schemas.character_memory_schemas import CharacterMemoryUpdate
        m = CharacterMemoryUpdate(content="new content")
        assert m.content == "new content"
        assert m.memory_type is None
        assert m.salience is None

    def test_extract_request_defaults(self):
        from tldw_Server_API.app.api.v1.schemas.character_memory_schemas import CharacterMemoryExtractRequest
        r = CharacterMemoryExtractRequest(chat_id="abc-123")
        assert r.message_limit == 50
        assert r.provider is None

    def test_extract_request_limit_bounds(self):
        from tldw_Server_API.app.api.v1.schemas.character_memory_schemas import CharacterMemoryExtractRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            CharacterMemoryExtractRequest(chat_id="x", message_limit=0)
        with pytest.raises(ValidationError):
            CharacterMemoryExtractRequest(chat_id="x", message_limit=201)


# ---------------------------------------------------------------------------
# Extraction module tests
# ---------------------------------------------------------------------------


class TestBuildExtractionPrompt:

    def test_basic_prompt(self):
        from tldw_Server_API.app.core.Character_Chat.modules.character_memory_extraction import (
            build_extraction_prompt,
        )
        messages = [
            {"role": "user", "content": "I work as an engineer"},
            {"role": "assistant", "content": "That's interesting!"},
        ]
        system_msg, user_msg = build_extraction_prompt(
            messages, "Luna", "Alex", [],
        )
        assert "Alex" in system_msg
        assert "Luna" in system_msg
        assert "(none)" in system_msg  # no existing memories
        assert "Alex: I work as an engineer" in user_msg
        assert "Luna: That's interesting!" in user_msg

    def test_existing_memories_in_prompt(self):
        from tldw_Server_API.app.core.Character_Chat.modules.character_memory_extraction import (
            build_extraction_prompt,
        )
        existing = [
            {"memory_type": "fact", "content": "User is an engineer"},
        ]
        system_msg, _ = build_extraction_prompt([], "Luna", "Alex", existing)
        assert "[fact] User is an engineer" in system_msg


class TestParseExtractionResponse:

    def test_clean_json(self):
        from tldw_Server_API.app.core.Character_Chat.modules.character_memory_extraction import (
            parse_extraction_response,
        )
        raw = json.dumps([
            {"category": "fact", "content": "User is 30", "salience": 0.8},
        ])
        result = parse_extraction_response(raw)
        assert len(result) == 1
        assert result[0]["category"] == "fact"
        assert result[0]["content"] == "User is 30"
        assert result[0]["salience"] == 0.8

    def test_markdown_fenced(self):
        from tldw_Server_API.app.core.Character_Chat.modules.character_memory_extraction import (
            parse_extraction_response,
        )
        raw = '```json\n[{"category": "event", "content": "Got promoted", "salience": 0.9}]\n```'
        result = parse_extraction_response(raw)
        assert len(result) == 1
        assert result[0]["category"] == "event"

    def test_empty_array(self):
        from tldw_Server_API.app.core.Character_Chat.modules.character_memory_extraction import (
            parse_extraction_response,
        )
        assert parse_extraction_response("[]") == []

    def test_invalid_json(self):
        from tldw_Server_API.app.core.Character_Chat.modules.character_memory_extraction import (
            parse_extraction_response,
        )
        assert parse_extraction_response("not json at all") == []

    def test_embedded_json(self):
        from tldw_Server_API.app.core.Character_Chat.modules.character_memory_extraction import (
            parse_extraction_response,
        )
        raw = 'Here are the memories: [{"category": "preference", "content": "Likes tea"}] done.'
        result = parse_extraction_response(raw)
        assert len(result) == 1
        assert result[0]["content"] == "Likes tea"

    def test_invalid_category_normalized(self):
        from tldw_Server_API.app.core.Character_Chat.modules.character_memory_extraction import (
            parse_extraction_response,
        )
        raw = json.dumps([{"category": "unknown_cat", "content": "test", "salience": 0.5}])
        result = parse_extraction_response(raw)
        assert result[0]["category"] == "fact"  # normalized to default

    def test_salience_clamped(self):
        from tldw_Server_API.app.core.Character_Chat.modules.character_memory_extraction import (
            parse_extraction_response,
        )
        raw = json.dumps([
            {"category": "fact", "content": "a", "salience": 2.0},
            {"category": "fact", "content": "b", "salience": -1.0},
        ])
        result = parse_extraction_response(raw)
        assert result[0]["salience"] == 1.0
        assert result[1]["salience"] == 0.0

    def test_empty_content_skipped(self):
        from tldw_Server_API.app.core.Character_Chat.modules.character_memory_extraction import (
            parse_extraction_response,
        )
        raw = json.dumps([{"category": "fact", "content": "", "salience": 0.5}])
        assert parse_extraction_response(raw) == []


class TestDeduplicateMemories:

    def test_exact_duplicate_removed(self):
        from tldw_Server_API.app.core.Character_Chat.modules.character_memory_extraction import (
            deduplicate_memories,
        )
        new = [{"content": "User is an engineer"}]
        existing = [{"content": "User is an engineer"}]
        assert deduplicate_memories(new, existing) == []

    def test_near_duplicate_removed(self):
        from tldw_Server_API.app.core.Character_Chat.modules.character_memory_extraction import (
            deduplicate_memories,
        )
        new = [{"content": "User is a software engineer"}]
        existing = [{"content": "User is a software engineeer"}]  # typo = near match
        assert deduplicate_memories(new, existing) == []

    def test_unique_kept(self):
        from tldw_Server_API.app.core.Character_Chat.modules.character_memory_extraction import (
            deduplicate_memories,
        )
        new = [{"content": "User has a cat named Whiskers"}]
        existing = [{"content": "User likes tea"}]
        result = deduplicate_memories(new, existing)
        assert len(result) == 1

    def test_empty_existing(self):
        from tldw_Server_API.app.core.Character_Chat.modules.character_memory_extraction import (
            deduplicate_memories,
        )
        new = [{"content": "fact1"}, {"content": "fact2"}]
        result = deduplicate_memories(new, [])
        assert len(result) == 2

    def test_custom_threshold(self):
        from tldw_Server_API.app.core.Character_Chat.modules.character_memory_extraction import (
            deduplicate_memories,
        )
        new = [{"content": "abc def"}]
        existing = [{"content": "abc xyz"}]
        # Low threshold: should consider as dup
        assert deduplicate_memories(new, existing, threshold=0.3) == []
        # High threshold: should keep
        result = deduplicate_memories(new, existing, threshold=0.99)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Context injection tests
# ---------------------------------------------------------------------------


class TestInjectCharacterMemory:

    def _make_mock_db(self, rows):
        db = MagicMock()
        db.list_persona_memory_entries.return_value = rows
        return db

    def test_no_memories_returns_unchanged(self):
        from tldw_Server_API.app.api.v1.endpoints.character_chat_sessions import (
            _inject_character_memory_from_db,
        )
        db = self._make_mock_db([])
        messages = [{"role": "user", "content": "hi"}]
        result = _inject_character_memory_from_db(
            messages, db=db, user_id="1", character_id="42",
            char_name="Luna", user_name="Alex",
        )
        assert result == messages

    def test_memories_appended_as_system_message(self):
        from tldw_Server_API.app.api.v1.endpoints.character_chat_sessions import (
            _inject_character_memory_from_db,
        )
        rows = [
            {
                "id": "m1", "memory_type": "fact", "content": "User is 30 years old",
                "salience": 0.8, "last_modified": "2026-01-01",
            },
            {
                "id": "m2", "memory_type": "preference", "content": "User prefers tea",
                "salience": 0.6, "last_modified": "2026-01-02",
            },
        ]
        db = self._make_mock_db(rows)
        messages = [{"role": "user", "content": "hello"}]
        result = _inject_character_memory_from_db(
            messages, db=db, user_id="1", character_id="42",
            char_name="Luna", user_name="Alex",
        )
        assert len(result) == 2
        mem_msg = result[-1]
        assert mem_msg["role"] == "system"
        assert "Character memory about Alex" in mem_msg["content"]
        assert "User is 30 years old" in mem_msg["content"]
        assert "User prefers tea" in mem_msg["content"]
        assert "[Facts]" in mem_msg["content"]
        assert "[Preferences]" in mem_msg["content"]

    def test_token_budget_respected(self):
        from tldw_Server_API.app.api.v1.endpoints.character_chat_sessions import (
            _inject_character_memory_from_db,
        )
        # Create many long memories
        rows = [
            {
                "id": f"m{i}", "memory_type": "fact",
                "content": f"This is a very long memory entry number {i} " * 10,
                "salience": 0.5, "last_modified": "2026-01-01",
            }
            for i in range(50)
        ]
        db = self._make_mock_db(rows)
        result = _inject_character_memory_from_db(
            [{"role": "user", "content": "hi"}],
            db=db, user_id="1", character_id="42",
            char_name="Luna", user_name="Alex",
            token_budget=50,  # very small budget
        )
        if len(result) > 1:
            mem_content = result[-1]["content"]
            assert len(mem_content) <= 50 * 4 + 100  # some tolerance

    def test_db_error_gracefully_skipped(self):
        from tldw_Server_API.app.api.v1.endpoints.character_chat_sessions import (
            _inject_character_memory_from_db,
        )
        db = MagicMock()
        db.list_persona_memory_entries.side_effect = RuntimeError("DB broken")
        messages = [{"role": "user", "content": "hi"}]
        result = _inject_character_memory_from_db(
            messages, db=db, user_id="1", character_id="42",
            char_name="Luna", user_name="Alex",
        )
        assert result == messages  # unchanged on error


# ---------------------------------------------------------------------------
# Persona profile helper tests
# ---------------------------------------------------------------------------


class TestPersonaIdHelpers:

    def test_persona_id_for_character(self):
        from tldw_Server_API.app.api.v1.endpoints.character_memory import (
            _persona_id_for_character,
            _character_id_from_persona,
        )
        assert _persona_id_for_character("42") == "char:42"
        assert _character_id_from_persona("char:42") == "42"
        assert _character_id_from_persona("other") == "other"


class TestGetOrCreatePersonaProfile:

    def test_existing_profile_returned(self):
        from tldw_Server_API.app.api.v1.endpoints.character_memory import (
            get_or_create_character_persona_profile,
        )
        db = MagicMock()
        db.get_persona_profile.return_value = {"id": "char:42", "user_id": "1"}
        result = get_or_create_character_persona_profile(db, "42", "Luna", "1")
        assert result == "char:42"
        db.create_persona_profile.assert_not_called()

    def test_new_profile_created(self):
        from tldw_Server_API.app.api.v1.endpoints.character_memory import (
            get_or_create_character_persona_profile,
        )
        db = MagicMock()
        db.get_persona_profile.return_value = None
        db.create_persona_profile.return_value = "char:42"
        result = get_or_create_character_persona_profile(db, "42", "Luna", "1")
        assert result == "char:42"
        db.create_persona_profile.assert_called_once()
        call_data = db.create_persona_profile.call_args[0][0]
        assert call_data["id"] == "char:42"
        assert call_data["name"] == "char_memory:42"
        assert call_data["origin_character_name"] == "Luna"

    def test_profile_creation_failure_log_is_sanitized(self, monkeypatch):
        from fastapi import HTTPException

        from tldw_Server_API.app.api.v1.endpoints import character_memory

        db = MagicMock()
        db.get_persona_profile.side_effect = [None, None]
        db.create_persona_profile.side_effect = RuntimeError(
            "profile create exploded at /private/character-memory.db"
        )
        fake_logger = MagicMock()
        monkeypatch.setattr(character_memory, "logger", fake_logger)

        with pytest.raises(HTTPException) as exc_info:
            character_memory.get_or_create_character_persona_profile(db, "42", "Luna", "1")

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to initialize character memory storage."
        fake_logger.error.assert_called_once_with("Failed to create persona profile")


# ---------------------------------------------------------------------------
# Endpoint DB error mapping tests
# ---------------------------------------------------------------------------


class TestCharacterMemoryEndpointDbErrorMapping:

    @pytest.mark.asyncio
    async def test_list_character_memories_maps_db_error_to_500(self):
        from fastapi import HTTPException

        from tldw_Server_API.app.api.v1.endpoints.character_memory import (
            list_character_memories,
        )
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
        from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError

        class _BrokenListDb:
            def get_character_card_by_id(self, character_id: int):
                return {"id": character_id, "name": "Luna"}

            def list_persona_memory_entries(self, **_kwargs):
                raise CharactersRAGDBError("list failed")

        with pytest.raises(HTTPException) as exc_info:
            await list_character_memories(
                character_id="42",
                db=_BrokenListDb(),
                current_user=User(id=1, username="tester", email=None, is_active=True),
            )

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to list character memories"

    @pytest.mark.asyncio
    async def test_archive_character_memory_maps_input_error_to_400(self):
        from fastapi import HTTPException

        from tldw_Server_API.app.api.v1.endpoints.character_memory import (
            archive_character_memory,
        )
        from tldw_Server_API.app.api.v1.schemas.character_memory_schemas import (
            CharacterMemoryArchiveRequest,
        )
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
        from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError

        class _InputErrorArchiveDb:
            def get_character_card_by_id(self, character_id: int):
                return {"id": character_id, "name": "Luna"}

            def set_persona_memory_archived(self, **_kwargs):
                raise InputError("invalid archive transition")

        with pytest.raises(HTTPException) as exc_info:
            await archive_character_memory(
                character_id="42",
                memory_id="mem-1",
                body=CharacterMemoryArchiveRequest(archived=True),
                db=_InputErrorArchiveDb(),
                current_user=User(id=1, username="tester", email=None, is_active=True),
            )

        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "invalid archive transition"

    @pytest.mark.asyncio
    async def test_archive_character_memory_maps_db_error_to_500(self):
        from fastapi import HTTPException

        from tldw_Server_API.app.api.v1.endpoints.character_memory import (
            archive_character_memory,
        )
        from tldw_Server_API.app.api.v1.schemas.character_memory_schemas import (
            CharacterMemoryArchiveRequest,
        )
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
        from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError

        class _BrokenArchiveDb:
            def get_character_card_by_id(self, character_id: int):
                return {"id": character_id, "name": "Luna"}

            def set_persona_memory_archived(self, **_kwargs):
                raise CharactersRAGDBError("archive failed")

        with pytest.raises(HTTPException) as exc_info:
            await archive_character_memory(
                character_id="42",
                memory_id="mem-1",
                body=CharacterMemoryArchiveRequest(archived=True),
                db=_BrokenArchiveDb(),
                current_user=User(id=1, username="tester", email=None, is_active=True),
            )

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to update character memory archive state"

    @pytest.mark.asyncio
    async def test_create_character_memory_maps_db_error_to_500(self):
        from fastapi import HTTPException

        from tldw_Server_API.app.api.v1.endpoints.character_memory import (
            create_character_memory,
        )
        from tldw_Server_API.app.api.v1.schemas.character_memory_schemas import (
            CharacterMemoryCreate,
        )
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
        from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError

        class _BrokenCreateDb:
            def get_character_card_by_id(self, character_id: int):
                return {"id": character_id, "name": "Luna"}

            def get_persona_profile(self, persona_id: str, user_id: str):
                return {"id": persona_id, "user_id": user_id}

            def add_persona_memory_entry(self, _payload):
                raise CharactersRAGDBError("create failed")

        with pytest.raises(HTTPException) as exc_info:
            await create_character_memory(
                character_id="42",
                body=CharacterMemoryCreate(content="User likes tea."),
                db=_BrokenCreateDb(),
                current_user=User(id=1, username="tester", email=None, is_active=True),
            )

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to create character memory"

    @pytest.mark.asyncio
    async def test_update_character_memory_maps_db_error_to_500(self):
        from fastapi import HTTPException

        from tldw_Server_API.app.api.v1.endpoints.character_memory import (
            update_character_memory,
        )
        from tldw_Server_API.app.api.v1.schemas.character_memory_schemas import (
            CharacterMemoryUpdate,
        )
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
        from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError

        class _BrokenUpdateDb:
            def get_character_card_by_id(self, character_id: int):
                return {"id": character_id, "name": "Luna"}

            def update_persona_memory_entry(self, **_kwargs):
                raise CharactersRAGDBError("update failed")

        with pytest.raises(HTTPException) as exc_info:
            await update_character_memory(
                character_id="42",
                memory_id="mem-1",
                body=CharacterMemoryUpdate(content="Updated memory."),
                db=_BrokenUpdateDb(),
                current_user=User(id=1, username="tester", email=None, is_active=True),
            )

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to update character memory"

    @pytest.mark.asyncio
    async def test_update_character_memory_maps_fetch_db_error_to_500(self):
        from fastapi import HTTPException

        from tldw_Server_API.app.api.v1.endpoints.character_memory import (
            update_character_memory,
        )
        from tldw_Server_API.app.api.v1.schemas.character_memory_schemas import (
            CharacterMemoryUpdate,
        )
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
        from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError

        class _BrokenUpdateFetchDb:
            def get_character_card_by_id(self, character_id: int):
                return {"id": character_id, "name": "Luna"}

            def update_persona_memory_entry(self, **_kwargs):
                return True

            def get_persona_memory_entry_by_id(self, **_kwargs):
                raise CharactersRAGDBError("fetch failed")

        with pytest.raises(HTTPException) as exc_info:
            await update_character_memory(
                character_id="42",
                memory_id="mem-1",
                body=CharacterMemoryUpdate(content="Updated memory."),
                db=_BrokenUpdateFetchDb(),
                current_user=User(id=1, username="tester", email=None, is_active=True),
            )

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to fetch character memory"

    @pytest.mark.asyncio
    async def test_delete_character_memory_maps_db_error_to_500(self):
        from fastapi import HTTPException

        from tldw_Server_API.app.api.v1.endpoints.character_memory import (
            delete_character_memory,
        )
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
        from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError

        class _BrokenDeleteDb:
            def get_character_card_by_id(self, character_id: int):
                return {"id": character_id, "name": "Luna"}

            def update_persona_memory_entry(self, **_kwargs):
                raise CharactersRAGDBError("delete failed")

        with pytest.raises(HTTPException) as exc_info:
            await delete_character_memory(
                character_id="42",
                memory_id="mem-1",
                db=_BrokenDeleteDb(),
                current_user=User(id=1, username="tester", email=None, is_active=True),
            )

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to delete character memory"

    @pytest.mark.asyncio
    async def test_extract_character_memories_persist_failure_log_is_sanitized(self, monkeypatch):
        from tldw_Server_API.app.api.v1.endpoints import character_memory
        from tldw_Server_API.app.api.v1.schemas.character_memory_schemas import (
            CharacterMemoryExtractRequest,
        )
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
        from tldw_Server_API.app.core.Character_Chat.modules import character_memory_extraction

        class _PersistFailDb:
            def get_character_card_by_id(self, character_id: int):
                return {"id": character_id, "name": "Luna"}

            def get_conversation_by_id(self, chat_id: str):
                assert chat_id == "chat-1"
                return {"id": chat_id, "client_id": 1, "character_id": 42, "user_name": "Alex"}

            def get_persona_profile(self, persona_id: str, user_id: str):
                return {"id": persona_id, "user_id": user_id}

            def get_messages_for_conversation(self, chat_id: str, limit: int, offset: int):
                assert chat_id == "chat-1"
                return [{"role": "user", "content": "I like tea.", "deleted": False}]

            def list_persona_memory_entries(self, **_kwargs):
                return []

            def add_persona_memory_entry(self, _payload):
                raise RuntimeError("memory persist exploded at /private/character-memory.db")

        def _fake_extract_character_memories(**_kwargs):
            return SimpleNamespace(
                unique=[{"category": "fact", "content": "User likes tea.", "salience": 0.8}],
                duplicates_skipped=0,
            )

        fake_logger = MagicMock()
        monkeypatch.setattr(character_memory, "logger", fake_logger)
        monkeypatch.setattr(
            character_memory_extraction,
            "extract_character_memories",
            _fake_extract_character_memories,
        )

        response = await character_memory.extract_character_memories_endpoint(
            character_id="42",
            body=CharacterMemoryExtractRequest(chat_id="chat-1"),
            db=_PersistFailDb(),
            current_user=User(id=1, username="tester", email=None, is_active=True),
        )

        assert response.extracted == 0
        assert response.skipped_duplicates == 0
        assert response.memories == []
        fake_logger.warning.assert_called_once_with("Failed to persist extracted memory")


# ---------------------------------------------------------------------------
# Extraction trigger tests
# ---------------------------------------------------------------------------


class TestExtractionTrigger:

    def test_no_background_tasks_skips(self):
        from tldw_Server_API.app.api.v1.endpoints.character_chat_sessions import (
            _maybe_trigger_character_memory_extraction,
        )
        # Should not raise
        _maybe_trigger_character_memory_extraction(
            background_tasks=None, db=MagicMock(), chat_id="c1",
            settings_row={}, conversation={}, character_id="42",
            char_name="Luna", user_id="1",
        )

    def test_disabled_extraction_skips(self):
        from tldw_Server_API.app.api.v1.endpoints.character_chat_sessions import (
            _maybe_trigger_character_memory_extraction,
            _extraction_message_counters,
        )
        bt = MagicMock()
        _maybe_trigger_character_memory_extraction(
            background_tasks=bt, db=MagicMock(), chat_id="c1",
            settings_row={"settings": {"characterMemoryExtraction": {"enabled": False}}},
            conversation={}, character_id="42", char_name="Luna", user_id="1",
        )
        bt.add_task.assert_not_called()

    def test_counter_increments(self):
        from tldw_Server_API.app.api.v1.endpoints.character_chat_sessions import (
            _maybe_trigger_character_memory_extraction,
            _extraction_message_counters,
        )
        key = "test_user:test_chat"
        _extraction_message_counters.pop(key, None)

        bt = MagicMock()
        settings = {"settings": {"characterMemoryExtraction": {"enabled": True, "intervalMessages": 5}}}
        for i in range(4):
            _maybe_trigger_character_memory_extraction(
                background_tasks=bt, db=MagicMock(), chat_id="test_chat",
                settings_row=settings, conversation={}, character_id="42",
                char_name="Luna", user_id="test_user",
            )
        assert bt.add_task.call_count == 0
        assert _extraction_message_counters[key] == 4

        # 5th message triggers extraction
        _maybe_trigger_character_memory_extraction(
            background_tasks=bt, db=MagicMock(), chat_id="test_chat",
            settings_row=settings, conversation={}, character_id="42",
            char_name="Luna", user_id="test_user",
        )
        assert bt.add_task.call_count == 1
        assert _extraction_message_counters[key] == 0  # reset

        # Clean up
        _extraction_message_counters.pop(key, None)
