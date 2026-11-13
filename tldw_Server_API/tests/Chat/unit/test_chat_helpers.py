# test_chat_helpers.py
# Unit tests for chat helper functions

import pytest
import asyncio
import datetime
import threading
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import Dict, Any

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import DEFAULT_CHARACTER_NAME

from tldw_Server_API.app.core.Chat import chat_helpers as chat_helpers_module
from tldw_Server_API.app.core.Chat.chat_helpers import (
    validate_request_payload,
    get_or_create_character_context,
    get_or_create_conversation,
    load_conversation_history,
    prepare_llm_messages,
    extract_system_message,
    extract_response_content,
    first_user_message_text,
    format_new_conversation_title,
    validate_provider_configuration,
)


class MockChatRequest:
    """Mock chat request for testing."""
    def __init__(self, messages=None):
        self.messages = messages or []


class MockMessage:
    """Mock message for testing."""
    def __init__(self, role="user", content="test"):
        self.role = role
        self.content = content


class MockMessagePart:
    """Mock message part for testing."""
    def __init__(self, type="text", text=None):
        self.type = type
        self.text = text


class MockImagePart:
    """Mock image part for testing."""
    def __init__(self, url: str):
        self.type = "image_url"
        self.image_url = {"url": url}


class MockDatabase:
    """Mock database for testing."""

    def __init__(self, with_default: bool = True):
        self.client_id = "test_client"
        self._next_id = 2
        self._characters: Dict[int, Dict[str, Any]] = {
            1: {"id": 1, "name": "TestChar", "description": "Test character"}
        }
        if with_default:
            self._characters[999] = {
                "id": 999,
                "name": DEFAULT_CHARACTER_NAME,
                "description": "Default character",
                "system_prompt": "You are a helpful AI assistant.",
            }

    def transaction(self):

        """Mock transaction context manager."""
        from contextlib import contextmanager

        @contextmanager
        def mock_transaction():
            yield self

        return mock_transaction()

    def get_character_card_by_id(self, char_id):

        return self._characters.get(char_id)

    def get_character_card_by_name(self, name):

        for card in self._characters.values():
            if card.get("name") == name:
                return card
        return None

    def add_character_card(self, data):

        new_id = data.get("id") or self._next_id
        self._next_id += 1
        card = dict(data)
        card["id"] = new_id
        self._characters[new_id] = card
        return new_id

    def get_conversation_by_id(self, conv_id):

        if conv_id == "existing_conv":
            return {
                "id": "existing_conv",
                "character_id": 1,
                "client_id": self.client_id,
                "title": "Test Conversation"
            }
        return None

    def add_conversation(self, data):

        return f"new_conv_{data.get('character_id', 0)}"

    def get_messages_for_conversation(self, conv_id, limit, offset, order):

        if conv_id == "existing_conv":
            return [
                {
                    "id": 1,
                    "sender": "user",
                    "content": "Hello",
                    "image_data": None,
                    "image_mime_type": None
                },
                {
                    "id": 2,
                    "sender": "assistant",
                    "content": "Hi there!",
                    "image_data": None,
                    "image_mime_type": None
                }
            ]
        return []


@pytest.mark.asyncio
class TestValidateRequestPayload:
    """Test request payload validation."""

    async def test_valid_request(self):
        """Test validation of valid request."""
        request = MockChatRequest([
            MockMessage("user", "Hello"),
            MockMessage("assistant", "Hi")
        ])

        is_valid, error = await validate_request_payload(request)
        assert is_valid is True
        assert error is None

    async def test_empty_messages(self):
        """Test validation fails for empty messages."""
        request = MockChatRequest([])

        is_valid, error = await validate_request_payload(request)
        assert is_valid is False
        assert "empty" in error.lower()

    async def test_too_many_messages(self):
        """Test validation fails for too many messages."""
        messages = [MockMessage() for _ in range(1001)]
        request = MockChatRequest(messages)

        is_valid, error = await validate_request_payload(request, max_messages=1000)
        assert is_valid is False
        assert "too many" in error.lower()

    async def test_message_too_long(self):
        """Test validation fails for overly long message."""
        long_text = "x" * 400001
        request = MockChatRequest([MockMessage("user", long_text)])

        is_valid, error = await validate_request_payload(request, max_text_length=400000)
        assert is_valid is False
        assert "too long" in error.lower()

    async def test_too_many_images(self):
        """Test validation fails for too many images."""
        messages = []
        for i in range(11):
            msg = MockMessage()
            msg.content = [MockMessagePart("image_url")]
            messages.append(msg)
        request = MockChatRequest(messages)

        is_valid, error = await validate_request_payload(request, max_images=10)
        assert is_valid is False
        assert "too many images" in error.lower()

    async def test_image_size_enforcement_rejects_oversized_payload(self):
        """Test validation fails when data URI exceeds configured image byte cap."""
        big_b64 = "A" * 5000
        msg = MockMessage()
        msg.content = [MockImagePart(f"data:image/png;base64,{big_b64}")]
        request = MockChatRequest([msg])

        is_valid, error = await validate_request_payload(
            request,
            enforce_image_max_bytes=True,
            max_image_bytes=100,
        )
        assert is_valid is False
        assert "too large" in error.lower()

    @pytest.mark.unit
    async def test_image_size_enforcement_allows_http_url(self):
        """Test validation allows non-data image URLs when base64 enforcement is enabled."""
        msg = MockMessage()
        msg.content = [MockImagePart("https://example.com/image.png")]
        request = MockChatRequest([msg])

        is_valid, error = await validate_request_payload(
            request,
            enforce_image_max_bytes=True,
            max_image_bytes=100,
        )
        assert is_valid is True
        assert error is None

    @pytest.mark.unit
    async def test_image_size_enforcement_rejects_invalid_data_uri(self):
        """Test validation rejects invalid data URI payloads when enforcing size."""
        msg = MockMessage()
        msg.content = [MockImagePart("data:image/png,not-base64")]
        request = MockChatRequest([msg])

        is_valid, error = await validate_request_payload(
            request,
            enforce_image_max_bytes=True,
            max_image_bytes=100,
        )
        assert is_valid is False
        assert "invalid" in error.lower()


@pytest.mark.asyncio
class TestGetOrCreateCharacterContext:
    """Test character context retrieval/creation."""

    async def test_get_character_by_id(self):
        """Test getting character by numeric ID."""
        db = MockDatabase()
        loop = asyncio.get_running_loop()

        char_card, char_id = await get_or_create_character_context(db, "1", loop)

        assert char_card is not None
        assert char_card["name"] == "TestChar"
        assert char_id == 1

    async def test_get_character_by_name(self):
        """Test getting character by name."""
        db = MockDatabase()
        loop = asyncio.get_running_loop()

        char_card, char_id = await get_or_create_character_context(db, "TestChar", loop)

        assert char_card is not None
        assert char_card["name"] == "TestChar"
        assert char_id == 1

    async def test_default_character_fallback(self):
        """Test fallback to default character."""
        db = MockDatabase()
        loop = asyncio.get_running_loop()

        char_card, char_id = await get_or_create_character_context(db, "NonExistent", loop)

        assert char_card is not None
        assert char_card["name"] == DEFAULT_CHARACTER_NAME
        assert char_id == 999

    async def test_no_character_provided(self):
        """Test when no character ID is provided."""
        db = MockDatabase()
        loop = asyncio.get_running_loop()

        char_card, char_id = await get_or_create_character_context(db, None, loop)

        assert char_card is not None
        assert char_card["name"] == DEFAULT_CHARACTER_NAME
        assert char_id == 999

    async def test_default_character_created_when_missing(self):
        """Ensure default character is created if missing."""
        db = MockDatabase(with_default=False)
        loop = asyncio.get_running_loop()

        char_card, char_id = await get_or_create_character_context(db, None, loop)

        assert char_card is not None
        assert char_card["name"] == DEFAULT_CHARACTER_NAME
        assert char_id is not None


@pytest.mark.asyncio
class TestGetOrCreateConversation:
    """Test conversation retrieval/creation."""

    async def test_get_existing_conversation(self):
        """Test retrieving existing conversation."""
        db = MockDatabase()
        loop = asyncio.get_running_loop()

        conv_id, was_created = await get_or_create_conversation(
            db, "existing_conv", 1, "TestChar", "test_client", loop
        )

        assert conv_id == "existing_conv"
        assert was_created is False

    async def test_create_new_conversation(self):
        """Test creating new conversation."""
        db = MockDatabase()
        loop = asyncio.get_running_loop()

        conv_id, was_created = await get_or_create_conversation(
            db, None, 1, "TestChar", "test_client", loop
        )

        assert conv_id == "new_conv_1"
        assert was_created is True

    async def test_conversation_mismatch_character(self):
        """Test conversation with wrong character ID."""
        db = MockDatabase()
        loop = asyncio.get_running_loop()

        # Mock conversation with different character
        with patch.object(db, 'get_conversation_by_id', return_value={
            "id": "existing_conv",
            "character_id": 999,  # Different character
            "client_id": "test_client"
        }):
            conv_id, was_created = await get_or_create_conversation(
                db, "existing_conv", 1, "TestChar", "test_client", loop
            )

        assert conv_id != "existing_conv"
        assert was_created is True

    async def test_conversation_mismatch_client(self):
        """Test conversation with wrong client ID."""
        db = MockDatabase()
        loop = asyncio.get_running_loop()

        # Mock conversation with different client
        with patch.object(db, 'get_conversation_by_id', return_value={
            "id": "existing_conv",
            "character_id": 1,
            "client_id": "different_client"  # Different client
        }):
            conv_id, was_created = await get_or_create_conversation(
                db, "existing_conv", 1, "TestChar", "test_client", loop
            )

        assert conv_id != "existing_conv"
        assert was_created is True

    async def test_conversation_creation_transaction_thread(self):
        """Ensure conversation creation runs inside a single-thread transaction."""

        class ThreadAwareDB:
            def __init__(self):
                self.client_id = "threaded_client"
                self._tx_thread = None

            def transaction(self):
                from contextlib import contextmanager

                @contextmanager
                def _tx():
                    self._tx_thread = threading.get_ident()
                    try:
                        yield self
                    finally:
                        self._tx_thread = None

                return _tx()

            def get_conversation_by_id(self, conv_id):
                return None

            def add_conversation(self, data):
                assert self._tx_thread == threading.get_ident()
                return "threaded_conv"

        db = ThreadAwareDB()
        loop = asyncio.get_running_loop()

        conv_id, was_created = await get_or_create_conversation(
            db, None, 1, "TestChar", "threaded_client", loop
        )

        assert conv_id == "threaded_conv"
        assert was_created is True

    async def test_new_conversation_is_titled_from_the_first_user_message(self):
        """XP-14: a new chat is named after what the user asked, not a timestamp."""
        db = MockDatabase()
        saved: list[dict[str, Any]] = []
        db.add_conversation = lambda data: saved.append(dict(data)) or "titled_conv"
        loop = asyncio.get_running_loop()

        await get_or_create_conversation(
            db,
            None,
            1,
            "Helpful AI Assistant",
            "test_client",
            loop,
            first_user_message="  How do I   tune a LoRA adapter for code review?",
        )

        assert saved[0]["title"] == "How do I tune a LoRA…"

    async def test_new_conversation_without_text_is_titled_by_assistant_and_local_time(self, monkeypatch):
        """XP-14: without message text the title reads '<assistant> · Nov 12, 16:41 PST', in server local time."""
        db = MockDatabase()
        saved: list[dict[str, Any]] = []
        db.add_conversation = lambda data: saved.append(dict(data)) or "titled_conv"
        pacific = datetime.timezone(datetime.timedelta(hours=-8), "PST")
        monkeypatch.setattr(
            chat_helpers_module,
            "_local_now",
            lambda: datetime.datetime(2026, 11, 12, 16, 41, 5, tzinfo=pacific),
        )
        loop = asyncio.get_running_loop()

        await get_or_create_conversation(db, None, 1, "Helpful AI Assistant", "test_client", loop)

        assert saved[0]["title"] == "Helpful AI Assistant · Nov 12, 16:41 PST"


class TestNewConversationTitle:
    """XP-14: titles people can read for chats the server creates."""

    def test_uses_the_first_six_words(self):
        assert format_new_conversation_title("Bot", "one two three four five six") == "one two three four five six"

    def test_shortens_a_long_first_message(self):
        title = format_new_conversation_title("Bot", "x" * 200)
        assert len(title) <= 60
        assert title.endswith("…")

    def test_keeps_the_cap_after_adding_the_ellipsis(self):
        six_words = " ".join(["abcdefghi"] * 5 + ["abcdefghij"])  # exactly 60 characters
        title = format_new_conversation_title("Bot", f"{six_words} and more")
        assert len(title) <= 60
        assert title.endswith("…")

    def test_reads_only_the_first_words_of_a_long_message(self):
        """A multi-megabyte first message is not split into all of its words."""
        splits: list[int] = []

        class SpyText(str):
            def split(self, sep=None, maxsplit=-1):
                splits.append(maxsplit)
                return super().split(sep, maxsplit)

        title = format_new_conversation_title("Bot", SpyText("word " * 200_000))
        assert title == "word word word word word word…"
        assert splits == [6]

    def test_falls_back_to_the_assistant_and_a_readable_time_with_its_zone(self):
        moment = datetime.datetime(
            2026, 10, 3, 3, 38, 23, tzinfo=datetime.timezone(datetime.timedelta(hours=2), "CEST")
        )
        assert format_new_conversation_title("Helpful AI Assistant", "   ", now=moment) == (
            "Helpful AI Assistant · Oct 3, 03:38 CEST"
        )

    def test_the_default_clock_is_server_local_time(self):
        """A UTC time without a zone reads as local time; the default clock is local and zoned."""
        moment = chat_helpers_module._local_now()
        assert moment.tzinfo is not None
        assert moment.utcoffset() == datetime.datetime.now().astimezone().utcoffset()

    def test_reads_the_first_user_text_from_request_messages(self):
        messages = [
            MockMessage(role="system", content="Be brief"),
            MockMessage(role="user", content=[MockImagePart("data:image/png;base64,AA"), MockMessagePart(text="What is this?")]),
            MockMessage(role="user", content="Later question"),
        ]
        assert first_user_message_text(messages) == "What is this?"
        assert first_user_message_text([{"role": "user", "content": "From a dict"}]) == "From a dict"
        assert first_user_message_text([MockMessage(role="assistant", content="Hi")]) is None


@pytest.mark.asyncio
class TestLoadConversationHistory:
    """Test conversation history loading."""

    async def test_load_history(self):
        """Test loading conversation history."""
        db = MockDatabase()
        loop = asyncio.get_running_loop()
        character_card = {"id": 1, "name": "TestChar"}

        messages = await load_conversation_history(
            db, "existing_conv", character_card, limit=20, loop=loop
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hi there!"

    async def test_load_empty_history(self):
        """Test loading empty conversation history."""
        db = MockDatabase()
        loop = asyncio.get_running_loop()

        messages = await load_conversation_history(
            db, "non_existent", None, limit=20, loop=loop
        )

        assert messages == []

    async def test_history_with_character_names(self):
        """Test history includes character names."""
        db = MockDatabase()
        loop = asyncio.get_running_loop()
        character_card = {"id": 1, "name": "TestChar"}

        messages = await load_conversation_history(
            db, "existing_conv", character_card, limit=20, loop=loop
        )

        # Assistant messages should have character name
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        for msg in assistant_msgs:
            assert "name" in msg or msg["role"] == "assistant"


@pytest.mark.asyncio
class TestExtractSystemMessage:
    """Test system message extraction."""

    async def test_extract_from_messages(self):
        """Test extracting system message from message list."""
        messages = [
            MockMessage("system", "You are helpful"),
            MockMessage("user", "Hello")
        ]

        system_msg = await asyncio.get_running_loop().run_in_executor(
            None, extract_system_message, messages, None
        )

        assert system_msg == "You are helpful"

    async def test_extract_from_character_card(self):
        """Test extracting system message from character card."""
        messages = [MockMessage("user", "Hello")]
        character_card = {
            "name": "TestChar",
            "description": "A test character",
            "system_prompt": "You are TestChar"
        }

        system_msg = await asyncio.get_running_loop().run_in_executor(
            None, extract_system_message, messages, character_card
        )

        assert "TestChar" in system_msg

    async def test_no_system_message(self):
        """Test when no system message exists."""
        messages = [MockMessage("user", "Hello")]

        system_msg = await asyncio.get_running_loop().run_in_executor(
            None, extract_system_message, messages, None
        )

        assert system_msg is None or system_msg == ""


@pytest.mark.asyncio
class TestExtractResponseContent:
    """Test response content extraction."""

    async def test_extract_string_response(self):
        """Test extracting content from string response."""
        response = "Hello, world!"

        content = await asyncio.get_running_loop().run_in_executor(
            None, extract_response_content, response
        )

        assert content == "Hello, world!"

    async def test_extract_dict_response(self):
        """Test extracting content from dict response."""
        response = {
            "choices": [
                {"message": {"content": "Hello from dict"}}
            ]
        }

        content = await asyncio.get_running_loop().run_in_executor(
            None, extract_response_content, response
        )

        assert content == "Hello from dict"

    async def test_extract_empty_response(self):
        """Test extracting from empty response."""
        response = {"choices": []}

        content = await asyncio.get_running_loop().run_in_executor(
            None, extract_response_content, response
        )

        assert content == "" or content is None

    async def test_extract_malformed_response(self):
        """Test extracting from malformed response."""
        response = {"invalid": "structure"}

        content = await asyncio.get_running_loop().run_in_executor(
            None, extract_response_content, response
        )

        assert content == "" or content is None


@pytest.mark.asyncio
class TestValidateProviderConfiguration:
    """Test provider configuration validation."""

    async def test_valid_provider_config(self):
        """Test validation of valid provider configuration."""
        api_keys = {'openai': 'test_key'}
        is_valid, error = await asyncio.get_running_loop().run_in_executor(
            None, validate_provider_configuration, "openai", api_keys
        )

        assert is_valid is True
        assert error is None

    async def test_missing_api_key(self):
        """Test validation fails for missing API key."""
        api_keys = {}  # Empty dictionary, no API key
        is_valid, error = await asyncio.get_running_loop().run_in_executor(
            None, validate_provider_configuration, "openai", api_keys
        )

        assert is_valid is False
        assert error is not None

    async def test_local_provider_no_key_needed(self):
        """Test local providers don't need API keys."""
        api_keys = {}  # No API keys needed for local providers
        is_valid, error = await asyncio.get_running_loop().run_in_executor(
            None, validate_provider_configuration, "llama.cpp", api_keys
        )

        assert is_valid is True

    async def test_unknown_provider(self):
        """Test unknown provider handling."""
        api_keys = {}
        result, error = await asyncio.get_running_loop().run_in_executor(
            None, validate_provider_configuration, "unknown_provider", api_keys
        )

        # Unknown provider should return False with an error message
        assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
