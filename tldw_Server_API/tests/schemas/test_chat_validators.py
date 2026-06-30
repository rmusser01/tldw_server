# test_chat_validators.py
# Unit tests for chat request validators

import pytest
import json
from unittest.mock import patch

from tldw_Server_API.app.api.v1.schemas.chat_validators import (
    validate_conversation_id,
    validate_character_id,
    validate_tool_definitions,
    validate_temperature,
    validate_max_tokens,
    validate_request_size,
    validate_stop_sequences,
    validate_model_name,
    validate_provider_name,
    CONVERSATION_ID_PATTERN,
    CHARACTER_ID_PATTERN,
    MAX_TOOL_DEFINITION_SIZE,
    MAX_REQUEST_SIZE,
)


class TestValidateConversationId:
    """Test conversation ID validation."""

    def test_valid_uuid(self):
        """Test valid UUID format."""
        uuid = "550e8400-e29b-41d4-a716-446655440000"
        result = validate_conversation_id(uuid)
        assert result == uuid

    def test_valid_alphanumeric(self):
        """Test valid alphanumeric format."""
        conv_id = "conv_123_abc-XYZ"
        result = validate_conversation_id(conv_id)
        assert result == conv_id

    def test_none_value(self):
        """Test None value is allowed."""
        result = validate_conversation_id(None)
        assert result is None

    def test_invalid_special_chars(self):
        """Test invalid special characters."""
        with pytest.raises(ValueError, match="Invalid conversation_id format"):
            validate_conversation_id("conv@123#test")

    def test_too_long(self):
        """Test ID that's too long."""
        long_id = "a" * 101
        with pytest.raises(ValueError, match="Invalid conversation_id format"):
            validate_conversation_id(long_id)

    def test_empty_string(self):
        """Test empty string is invalid."""
        with pytest.raises(ValueError):
            validate_conversation_id("")


class TestValidateCharacterId:
    """Test character ID validation."""

    def test_numeric_id(self):
        """Test numeric character ID."""
        result = validate_character_id("123")
        assert result == "123"

    def test_character_name(self):
        """Test character name format."""
        result = validate_character_id("Test Character-Name_123")
        assert result == "Test Character-Name_123"

    def test_none_value(self):
        """Test None value is allowed."""
        result = validate_character_id(None)
        assert result is None

    def test_invalid_special_chars(self):
        """Test invalid special characters."""
        with pytest.raises(ValueError, match="Invalid character_id format"):
            validate_character_id("char@#$%")

    def test_too_long(self):
        """Test ID that's too long."""
        long_id = "a" * 101
        with pytest.raises(ValueError):
            validate_character_id(long_id)


class TestValidateToolDefinitions:
    """Test tool definitions validation."""

    def test_valid_tool_definition(self):
        """Test valid tool definition."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather info",
                    "parameters": {}
                }
            }
        ]
        result = validate_tool_definitions(tools)
        assert result == tools

    def test_google_allows_gemini_native_tools(self):
        """Test Gemini-native tools are allowed for Google provider."""
        tools = [
            {
                "function_declarations": [{"name": "lookup"}]
            }
        ]
        result = validate_tool_definitions(tools, provider="google")
        assert result == tools

    def test_none_value(self):
        """Test None value is allowed."""
        result = validate_tool_definitions(None)
        assert result is None

    def test_not_a_list(self):
        """Test non-list input."""
        with pytest.raises(ValueError, match="Tools must be a list"):
            validate_tool_definitions("not a list")

    def test_too_many_tools(self):
        """Test too many tools."""
        tools = [{"type": "function", "function": {"name": f"tool_{i}"}} for i in range(129)]
        with pytest.raises(ValueError, match="Too many tools"):
            validate_tool_definitions(tools)

    def test_missing_type(self):
        """Test tool missing type field."""
        tools = [{"function": {"name": "test"}}]
        with pytest.raises(ValueError, match="missing 'type' field"):
            validate_tool_definitions(tools)

    def test_invalid_type(self):
        """Test invalid tool type."""
        tools = [{"type": "invalid", "function": {"name": "test"}}]
        with pytest.raises(ValueError, match="invalid type"):
            validate_tool_definitions(tools)

    def test_missing_function(self):
        """Test tool missing function field."""
        tools = [{"type": "function"}]
        with pytest.raises(ValueError, match="missing 'function' field"):
            validate_tool_definitions(tools)

    def test_missing_function_name(self):
        """Test function missing name."""
        tools = [{"type": "function", "function": {"description": "test"}}]
        with pytest.raises(ValueError, match="missing 'name' field"):
            validate_tool_definitions(tools)

    def test_invalid_function_name(self):
        """Test invalid function name format."""
        tools = [{"type": "function", "function": {"name": "invalid name!"}}]
        with pytest.raises(ValueError, match="function name must be alphanumeric"):
            validate_tool_definitions(tools)

    def test_tool_too_large(self):
        """Test tool definition that's too large."""
        large_description = "x" * MAX_TOOL_DEFINITION_SIZE
        tools = [{
            "type": "function",
            "function": {
                "name": "test",
                "description": large_description
            }
        }]
        with pytest.raises(ValueError, match="definition too large"):
            validate_tool_definitions(tools)


class TestValidateTemperature:
    """Test temperature validation."""

    def test_valid_temperature(self):
        """Test valid temperature values."""
        assert validate_temperature(0.0) == 0.0
        assert validate_temperature(1.0) == 1.0
        assert validate_temperature(2.0) == 2.0
        assert validate_temperature(0.7) == 0.7

    def test_none_value(self):
        """Test None value is allowed."""
        result = validate_temperature(None)
        assert result is None

    def test_negative_temperature(self):
        """Test negative temperature."""
        with pytest.raises(ValueError, match="Temperature must be between"):
            validate_temperature(-0.1)

    def test_too_high_temperature(self):
        """Test temperature too high."""
        with pytest.raises(ValueError, match="Temperature must be between"):
            validate_temperature(2.1)


class TestValidateMaxTokens:
    """Test max_tokens validation."""

    def test_valid_max_tokens(self):
        """Test valid max_tokens values."""
        assert validate_max_tokens(1) == 1
        assert validate_max_tokens(1000) == 1000
        assert validate_max_tokens(128000) == 128000

    def test_none_value(self):
        """Test None value is allowed."""
        result = validate_max_tokens(None)
        assert result is None

    def test_zero_tokens(self):
        """Test zero tokens."""
        with pytest.raises(ValueError, match="max_tokens must be at least"):
            validate_max_tokens(0)

    def test_negative_tokens(self):
        """Test negative tokens."""
        with pytest.raises(ValueError, match="max_tokens must be at least"):
            validate_max_tokens(-1)

    def test_too_many_tokens(self):
        """Test too many tokens."""
        with pytest.raises(ValueError, match="max_tokens too large"):
            validate_max_tokens(128001)


class TestValidateRequestSize:
    """Test request size validation."""

    def test_valid_request_size(self):
        """Test valid request size."""
        request = json.dumps({"test": "data"})
        result = validate_request_size(request)
        assert result is True

    def test_request_too_large(self):
        """Test request that's too large."""
        large_request = "x" * (MAX_REQUEST_SIZE + 1)
        with pytest.raises(ValueError, match="Failed to validate request size"):
            validate_request_size(large_request)

    def test_empty_request(self):
        """Test empty request."""
        result = validate_request_size("")
        assert result is True


class TestValidateStopSequences:
    """Test stop sequences validation."""

    def test_valid_stop_sequences(self):
        """Test valid stop sequences."""
        sequences = ["\\n", "END", "STOP"]
        result = validate_stop_sequences(sequences)
        assert result == sequences

    def test_none_value(self):
        """Test None value is allowed."""
        result = validate_stop_sequences(None)
        assert result is None

    def test_empty_list(self):
        """Test empty list is valid."""
        result = validate_stop_sequences([])
        assert result == []

    def test_string_stop_sequence(self):
        """Test string stop sequence is valid."""
        result = validate_stop_sequences("STOP")
        assert result == "STOP"

    def test_too_many_sequences(self):
        """Test too many stop sequences."""
        sequences = [f"stop_{i}" for i in range(5)]
        with pytest.raises(ValueError, match="Too many stop sequences"):
            validate_stop_sequences(sequences)

    def test_non_string_sequence(self):
        """Test non-string sequence."""
        with pytest.raises(ValueError, match="must be a string"):
            validate_stop_sequences([123])

    def test_sequence_too_long(self):
        """Test sequence that's too long."""
        long_sequence = "x" * 501
        with pytest.raises(ValueError, match="too long"):
            validate_stop_sequences([long_sequence])

    def test_invalid_type(self):
        """Test invalid type for stop sequences."""
        with pytest.raises(ValueError, match="must be string or list"):
            validate_stop_sequences(123)


class TestValidateModelName:
    """Test model name validation."""

    def test_valid_model_names(self):
        """Test valid model names."""
        assert validate_model_name("gpt-4") == "gpt-4"
        assert validate_model_name("claude-opus-4.1") == "claude-opus-4.1"
        assert validate_model_name("llama2_70b") == "llama2_70b"

    def test_none_value(self):
        """Test None value handling."""
        result = validate_model_name(None)
        assert result is None

    def test_invalid_characters(self):
        """Test invalid characters in model name."""
        with pytest.raises(ValueError, match="Model name contains invalid characters"):
            validate_model_name("model@#$%")

    def test_model_name_too_long(self):
        """Test model name that's too long."""
        long_name = "a" * 101
        with pytest.raises(ValueError, match="Model name too long"):
            validate_model_name(long_name)

    def test_empty_model_name(self):
        """Test empty model name."""
        # Empty string is explicitly rejected before pattern check
        with pytest.raises(ValueError, match="Model name cannot be empty"):
            validate_model_name("")


class TestValidateProviderName:
    """Test provider name validation."""

    def test_known_providers(self):
        """Test known provider names."""
        # Updated list to match ALLOWED_PROVIDERS in chat_validators.py
        known_providers = [
            "openai", "anthropic", "cohere", "groq",
            "openrouter", "deepseek", "mistral", "google", "qwen",
            "novita", "poe", "together",
            "llama.cpp", "kobold", "ooba", "ollama", "bedrock",
            "huggingface", "tabbyapi", "vllm", "local-llm", "aphrodite",
            "custom-openai-api", "custom-openai-api-2", "custom-openai-api-37",
            "custom-openai-api-99", "moonshot", "zai"
        ]
        for provider in known_providers:
            result = validate_provider_name(provider)
            assert result == provider

    def test_numbered_custom_openai_providers_stop_at_99(self):
        """Test additional named custom providers are accepted through 99 only."""
        assert validate_provider_name("custom-openai-api-99") == "custom-openai-api-99"
        with pytest.raises(ValueError, match="Unknown provider"):
            validate_provider_name("custom-openai-api-100")

    def test_none_value(self):
        """Test None value handling."""
        result = validate_provider_name(None)
        assert result is None

    def test_unknown_provider(self):
        """Test unknown provider raises error in strict mode."""
        # Default is strict=True, so unknown providers raise ValueError
        with pytest.raises(ValueError, match="Unknown provider"):
            validate_provider_name("unknown_provider")

    def test_unknown_provider_non_strict(self):
        """Test unknown provider with warning in non-strict mode."""
        with patch('tldw_Server_API.app.api.v1.schemas.chat_validators.logger') as mock_logger:
            result = validate_provider_name("unknown_provider", strict=False)
            assert result == "unknown_provider"
            mock_logger.warning.assert_called()

    def test_invalid_provider_name(self):
        """Test invalid provider name format."""
        # Provider validator now validates character format
        with pytest.raises(ValueError, match="invalid characters"):
            validate_provider_name("provider@#$%")

    def test_provider_name_too_long(self):
        """Test provider name that's too long."""
        long_name = "a" * 51
        # Provider validator now checks length
        with pytest.raises(ValueError, match="too long"):
            validate_provider_name(long_name)

    def test_empty_provider_name(self):
        """Test empty provider name."""
        # Empty strings after strip are converted to None
        result = validate_provider_name("")
        assert result is None


class TestPatternConstants:
    """Test regex pattern constants."""

    def test_conversation_id_pattern(self):
        """Test conversation ID pattern."""
        assert CONVERSATION_ID_PATTERN.match("conv_123")
        assert CONVERSATION_ID_PATTERN.match("abc-def-123")
        assert not CONVERSATION_ID_PATTERN.match("conv@123")
        assert not CONVERSATION_ID_PATTERN.match("")

    def test_character_id_pattern(self):
        """Test character ID pattern."""
        assert CHARACTER_ID_PATTERN.match("123")
        assert CHARACTER_ID_PATTERN.match("Character Name 123")
        assert not CHARACTER_ID_PATTERN.match("char@#$")
        assert not CHARACTER_ID_PATTERN.match("")


class TestSizeConstants:
    """Test size limit constants."""

    def test_max_tool_definition_size(self):
        """Test max tool definition size constant (reduced for security)."""
        assert MAX_TOOL_DEFINITION_SIZE == 5000

    def test_max_request_size(self):
        """Test max request size constant."""
        assert MAX_REQUEST_SIZE == 1000000  # 1MB


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
