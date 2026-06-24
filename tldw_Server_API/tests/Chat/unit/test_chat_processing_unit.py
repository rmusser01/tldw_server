"""
Unit tests for chat input processing and content update helpers.

These tests were consolidated from Chat_NEW to avoid duplication and
live alongside other Chat unit tests.
"""

import json
from unittest.mock import MagicMock

import pytest

from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAPIError,
    ChatRateLimitError,
    ChatAuthenticationError,
    ChatProviderError,
)
from tldw_Server_API.app.core.Chat.chat_dictionary import (
    ChatDictionary,
    TokenBudgetExceededWarning,
    apply_strategy,
    enforce_token_budget,
    process_user_input,
)
from tldw_Server_API.app.core.Chat.chat_history import update_chat_content


# ========================================================================
# process_user_input tests
# ========================================================================

class TestProcessUserInput:
    @pytest.mark.unit
    def test_process_simple_text_input(self):
        result = process_user_input("Hello, how are you?", entries=[])
        assert isinstance(result, str)
        assert result == "Hello, how are you?"

    @pytest.mark.unit
    def test_process_empty_input(self):
        result = process_user_input("", entries=[])
        assert isinstance(result, str)
        assert result == ""

    @pytest.mark.unit
    def test_process_multiline_input(self):
        input_text = """Line 1
        Line 2
        Line 3"""
        result = process_user_input(input_text, entries=[])
        assert isinstance(result, str)
        assert "Line 1" in result and "Line 2" in result and "Line 3" in result

    @pytest.mark.unit
    def test_process_input_with_special_characters(self):
        special_input = "Test with special chars: !@#$%^&*()[]{}\"'<>"
        result = process_user_input(special_input, entries=[])
        assert isinstance(result, str)
        assert result == special_input

    @pytest.mark.unit
    def test_process_json_like_input(self):
        json_input = '{"key": "value", "number": 123}'
        result = process_user_input(json_input, entries=[])
        assert isinstance(result, str)
        assert result == json_input


class TestChatDictionaryUtilities:
    @pytest.mark.unit
    def test_enforce_token_budget_truncates_entries(self):
        small_entry = ChatDictionary(key="hello", content="hi there")
        large_entry = ChatDictionary(key="world", content="this entry should be trimmed out")

        with pytest.warns(TokenBudgetExceededWarning):
            filtered = enforce_token_budget([small_entry, large_entry], max_tokens=3)

        assert len(filtered) == 1
        assert filtered[0].key_raw == "hello"

    @pytest.mark.unit
    def test_apply_strategy_prioritizes_global_group(self):
        global_entry = ChatDictionary(key="alpha", content="global", group="global")
        character_entry = ChatDictionary(key="beta", content="character", group="character")
        default_entry = ChatDictionary(key="gamma", content="default")

        ordered = apply_strategy([character_entry, default_entry, global_entry], strategy="global_lore_first")

        assert [entry.key_raw for entry in ordered] == ["alpha", "beta", "gamma"]

    @pytest.mark.unit
    def test_chat_dictionary_rejects_unsafe_regex_key_at_runtime(self) -> None:
        with pytest.raises(ValueError, match="Unsafe chat dictionary regex"):
            ChatDictionary(key="/^(a+)+$/", content="unsafe")
# ========================================================================
# update_chat_content tests
# ========================================================================

class TestUpdateChatContent:
    @pytest.mark.unit
    def test_update_content_basic(self):
        mock_db = MagicMock()
        mock_db.get_note_by_id.return_value = {
            'content': '{"content": "Note content", "summary": "Note summary", "prompt": "Note prompt"}',
        }

        result, tags = update_chat_content(
            selected_item="Test Item",
            use_content=True,
            use_summary=False,
            use_prompt=False,
            item_mapping={"Test Item": "1"},
            db_instance=mock_db,
        )

        assert isinstance(result, dict)
        assert isinstance(tags, list)
        assert 'content' in result
        mock_db.get_note_by_id.assert_called_once_with("1")

    @pytest.mark.unit
    def test_update_content_with_summary(self):
        mock_db = MagicMock()
        mock_db.get_note_by_id.return_value = {
            'content': '{"content": "Note content", "summary": "Note summary", "prompt": "Note prompt"}',
        }

        result, tags = update_chat_content(
            selected_item="Test Item",
            use_content=False,
            use_summary=True,
            use_prompt=False,
            item_mapping={"Test Item": "2"},
            db_instance=mock_db,
        )

        assert 'summary' in result
        assert result['summary'] == 'Note summary'

    @pytest.mark.unit
    def test_update_content_no_selection(self):
        mock_db = MagicMock()
        result, tags = update_chat_content(
            selected_item=None,
            use_content=True,
            use_summary=False,
            use_prompt=False,
            item_mapping={},
            db_instance=mock_db,
        )
        assert result == {}
        assert tags == []
        mock_db.get_note_by_id.assert_not_called()

    @pytest.mark.unit
    def test_update_content_all_options(self):
        mock_db = MagicMock()
        mock_db.get_note_by_id.return_value = {
            'content': '{"content": "Note content", "summary": "Note summary", "prompt": "Note prompt"}',
            'keywords': 'tag1, tag2',
        }

        result, tags = update_chat_content(
            selected_item="Test Item",
            use_content=True,
            use_summary=True,
            use_prompt=True,
            item_mapping={"Test Item": "3"},
            db_instance=mock_db,
        )

        assert 'content' in result and 'summary' in result and 'prompt' in result
        assert len(tags) > 0


# ========================================================================
# Error object property sanity checks (non-duplicative)
# ========================================================================

class TestErrorHandling:
    @pytest.mark.unit
    def test_chat_api_error_creation(self):
        error = ChatAPIError("API call failed", status_code=500)
        assert str(error) == "API call failed"
        assert error.status_code == 500

    @pytest.mark.unit
    def test_rate_limit_error_properties(self):
        error = ChatRateLimitError("Too many requests", provider="openai")
        assert "Too many requests" in str(error)
        assert error.provider == "openai"

    @pytest.mark.unit
    def test_auth_error_properties(self):
        error = ChatAuthenticationError("Invalid credentials", provider="openai")
        assert "Invalid credentials" in str(error)
        assert error.provider == "openai"
        assert error.status_code == 401

    @pytest.mark.unit
    def test_provider_error_properties(self):
        error = ChatProviderError("Provider unavailable", provider="openai")
        assert "Provider unavailable" in str(error)
        assert error.provider == "openai"


# ========================================================================
# Simple message formatting checks
# ========================================================================

class TestMessageFormatting:
    @pytest.mark.unit
    def test_format_single_message(self):
        message = {"role": "user", "content": "Hello"}
        formatted = json.dumps(message)
        assert '"role": "user"' in formatted
        assert '"content": "Hello"' in formatted

    @pytest.mark.unit
    def test_format_message_list(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ]
        formatted = json.dumps(messages)
        parsed = json.loads(formatted)
        assert len(parsed) == 2
        assert parsed[0]["role"] == "system"
        assert parsed[1]["role"] == "user"

    @pytest.mark.unit
    def test_format_message_with_metadata(self):
        message = {
            "role": "assistant",
            "content": "Response",
            "name": "Assistant",
            "metadata": {"timestamp": "2024-01-01T00:00:00"},
        }
        formatted = json.dumps(message)
        parsed = json.loads(formatted)
        assert parsed["name"] == "Assistant"
        assert "metadata" in parsed
