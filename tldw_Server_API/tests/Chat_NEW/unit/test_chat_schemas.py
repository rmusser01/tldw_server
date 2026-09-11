"""
Unit tests for chat request/response schemas and message validation.

Tests focus on the OpenAI-compatible schema validation, message formats,
and request parameter validation without any external dependencies.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionRequest,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    FunctionDefinition,
    ResponseFormat,
    ToolDefinition,
)

# ========================================================================
# Message Validation Tests
# ========================================================================

class TestMessageValidation:
    """Test message parameter validation."""

    @pytest.mark.unit
    def test_valid_user_message(self):
        """Test valid user message creation."""
        msg = ChatCompletionUserMessageParam(
            role="user",
            content="Hello, how are you?"
        )
        assert msg.role == "user"
        assert msg.content == "Hello, how are you?"

    @pytest.mark.unit
    def test_valid_system_message(self):
        """Test valid system message creation."""
        msg = ChatCompletionSystemMessageParam(
            role="system",
            content="You are a helpful assistant."
        )
        assert msg.role == "system"
        assert msg.content == "You are a helpful assistant."

    @pytest.mark.unit
    def test_valid_assistant_message(self):
        """Test valid assistant message creation."""
        msg = ChatCompletionAssistantMessageParam(
            role="assistant",
            content="I'm doing well, thank you!"
        )
        assert msg.role == "assistant"
        assert msg.content == "I'm doing well, thank you!"

    @pytest.mark.unit
    def test_message_with_name(self):
        """Test message with optional name field."""
        msg = ChatCompletionUserMessageParam(
            role="user",
            content="Test message",
            name="user_123"
        )
        assert msg.name == "user_123"

    @pytest.mark.unit
    def test_multimodal_user_message(self):
        """Test user message with multimodal content (text + image)."""
        # User messages can have List content for multimodal
        msg = ChatCompletionUserMessageParam(
            role="user",
            content=[
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
            ]
        )
        assert msg.role == "user"
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2

# ========================================================================
# ChatCompletionRequest Validation Tests
# ========================================================================

class TestChatCompletionRequest:
    """Test ChatCompletionRequest model validation."""

    @staticmethod
    def _bounded_research_context_payload() -> dict[str, Any]:
        return {
            "run_id": "run_123",
            "query": "battery recycling supply chain",
            "question": "What changed in the battery recycling market?",
            "outline": [{"title": "Overview"}],
            "key_claims": [{"text": "Claim one"}],
            "unresolved_questions": ["What changed in Europe?"],
            "verification_summary": {"unsupported_claim_count": 0},
            "source_trust_summary": {"high_trust_count": 3},
            "research_url": "/research?run=run_123",
        }

    @pytest.mark.unit
    def test_minimal_valid_request(self):
        """Test minimal valid chat completion request."""
        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Hello"}
            ]
        )
        assert request.model == "gpt-3.5-turbo"
        assert len(request.messages) == 1
        assert request.messages[0].role == "user"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "unsafe_value",
        [
            "allowed\r\nX-Injected: attacker",
            "allowed\x00attacker",
            "allowed\x7fattacker",
        ],
    )
    def test_extra_header_values_reject_controls(
        self,
        unsafe_value: str,
    ) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                extra_headers={"X-Provider-Extension": unsafe_value},
            )

        assert exc_info.value.errors()[0]["loc"] == (
            "extra_headers",
            "X-Provider-Extension",
        )
        assert "attacker" not in str(exc_info.value)

    @pytest.mark.unit
    @pytest.mark.parametrize("api_provider", ["llama", "llama.cpp"])
    def test_llamacpp_aliases_normalize_to_canonical_provider(
        self: "TestChatCompletionRequest",
        api_provider: str,
    ) -> None:
        """Test WebUI llama provider ids route through the canonical llama.cpp adapter."""
        request = ChatCompletionRequest(
            api_provider=api_provider,
            model="local-model",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert request.api_provider == "llama.cpp"

    @pytest.mark.unit
    def test_request_accepts_typed_research_context(self):
        """Test request accepts a bounded typed research_context payload."""
        request = ChatCompletionRequest(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Use the attached research."}],
            research_context=self._bounded_research_context_payload(),
        )

        assert request.research_context.run_id == "run_123"
        assert request.research_context.key_claims[0].text == "Claim one"

    @pytest.mark.unit
    def test_request_without_research_context_preserves_current_behavior(self):
        """Test omitting research_context still produces the current request shape."""
        request = ChatCompletionRequest(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert not hasattr(request, "research_context") or request.research_context is None

    @pytest.mark.unit
    def test_request_rejects_unknown_nested_research_context_keys(self):
        """Test bounded research_context rejects unknown nested keys."""
        bad_context = self._bounded_research_context_payload()
        bad_context["unexpected"] = "nope"

        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Use the attached research."}],
                research_context=bad_context,
            )

    @pytest.mark.unit
    def test_request_rejects_oversized_research_context(self):
        """Test bounded research_context enforces caps instead of accepting arbitrarily large payloads."""
        bad_context = self._bounded_research_context_payload()
        bad_context["outline"] = [{"title": f"Section {index}"} for index in range(8)]

        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Use the attached research."}],
                research_context=bad_context,
            )

    @pytest.mark.unit
    def test_request_with_all_parameters(self):
        """Test request with all optional parameters."""
        request = ChatCompletionRequest(
            api_provider="openai",
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"}
            ],
            temperature=0.7,
            max_tokens=150,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            stop=["\n", "END"],
            stream=False,
            n=1,
            user="user_123"
        )
        assert request.api_provider == "openai"
        assert request.temperature == 0.7
        assert request.max_tokens == 150
        assert request.stop == ["\n", "END"]

    @pytest.mark.unit
    def test_request_without_messages_fails(self):
        """Test that request without messages fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(model="gpt-3.5-turbo")

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("messages",) for e in errors)

    @pytest.mark.unit
    def test_request_with_empty_messages_fails(self):
        """Test that request with empty messages list fails."""
        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(
                model="gpt-3.5-turbo",
                messages=[]
            )

        errors = exc_info.value.errors()
        assert any("at least 1 item" in str(e) for e in errors)

    @pytest.mark.unit
    def test_temperature_bounds(self):
        """Test temperature parameter bounds (0.0 to 2.0)."""
        # Valid temperatures
        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            temperature=0.0
        )
        assert request.temperature == 0.0

        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            temperature=2.0
        )
        assert request.temperature == 2.0

        # Invalid temperature (too high)
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                temperature=2.1
            )

        # Invalid temperature (negative)
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                temperature=-0.1
            )

    @pytest.mark.unit
    def test_provider_validation(self):
        """Test API provider validation."""
        valid_providers = ["openai", "anthropic", "cohere", "groq", "mistral"]

        for provider in valid_providers:
            request = ChatCompletionRequest(
                api_provider=provider,
                model="test-model",
                messages=[{"role": "user", "content": "test"}]
            )
            assert request.api_provider == provider

    @pytest.mark.unit
    def test_streaming_parameter(self):
        """Test streaming parameter."""
        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            stream=True
        )
        assert request.stream is True

        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            stream=False
        )
        assert request.stream is False

    @pytest.mark.unit
    def test_llamacpp_extension_fields_are_first_class_schema_fields(self):
        """Test llama.cpp extension fields round-trip through the request schema."""
        request = ChatCompletionRequest(
            model="llama.cpp/local-model",
            messages=[{"role": "user", "content": "test"}],
            grammar_mode="library",
            grammar_id="grammar_1",
            grammar_override='root ::= "ok"',
            thinking_budget_tokens=64,
        )
        dumped = request.model_dump()
        assert dumped["grammar_mode"] == "library"
        assert dumped["grammar_id"] == "grammar_1"
        assert dumped["thinking_budget_tokens"] == 64

    @pytest.mark.unit
    def test_llamacpp_inline_mode_requires_inline_grammar(self):
        """Test llama.cpp inline grammar mode validation."""
        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(
                model="llama.cpp/local-model",
                messages=[{"role": "user", "content": "test"}],
                grammar_mode="inline",
            )

        assert "grammar_inline is required" in str(exc_info.value)

# ========================================================================
# Response Format Tests
# ========================================================================

class TestResponseFormat:
    """Test response format specifications."""

    @pytest.mark.unit
    def test_json_response_format(self):
        """Test JSON response format specification."""
        format_spec = ResponseFormat(type="json_object")
        assert format_spec.type == "json_object"

        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Return JSON"}],
            response_format={"type": "json_object"}
        )
        assert request.response_format.type == "json_object"

    @pytest.mark.unit
    def test_text_response_format(self):
        """Test text response format (default)."""
        format_spec = ResponseFormat(type="text")
        assert format_spec.type == "text"

    @pytest.mark.unit
    def test_json_schema_response_format(self):
        """Test JSON schema response format specification."""
        format_spec = ResponseFormat(
            type="json_schema",
            json_schema={
                "name": "answer_schema",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            },
        )
        assert format_spec.type == "json_schema"
        assert format_spec.json_schema is not None
        assert format_spec.json_schema.name == "answer_schema"

        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Return structured"}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer_schema",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                },
            },
        )
        assert request.response_format is not None
        assert request.response_format.type == "json_schema"

    @pytest.mark.unit
    def test_json_schema_response_format_requires_schema(self):
        """Test JSON schema response format requires json_schema.schema."""
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="gpt-4",
                messages=[{"role": "user", "content": "Return structured"}],
                response_format={"type": "json_schema", "json_schema": {"name": "bad"}},
            )

# ========================================================================
# Tool/Function Calling Tests
# ========================================================================

class TestToolsAndFunctions:
    """Test tool and function definitions."""

    @pytest.mark.unit
    def test_function_definition(self):
        """Test function definition for tool calling."""
        func = FunctionDefinition(
            name="get_weather",
            description="Get weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        )
        assert func.name == "get_weather"
        assert "location" in func.parameters["required"]

    @pytest.mark.unit
    def test_tool_definition(self):
        """Test tool definition with function."""
        tool = ToolDefinition(
            type="function",
            function={
                "name": "search",
                "description": "Search for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            }
        )
        assert tool.type == "function"
        assert tool.function.name == "search"

    @pytest.mark.unit
    def test_request_with_tools(self):
        """Test request with tool definitions."""
        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"}
                            }
                        }
                    }
                }
            ],
            tool_choice="auto"
        )
        assert len(request.tools) == 1
        assert request.tool_choice == "auto"

    @pytest.mark.unit
    def test_request_with_gemini_tools(self):
        """Test request accepts Gemini-native tools."""
        request = ChatCompletionRequest(
            api_provider="google",
            model="gemini-2.5-pro",
            messages=[{"role": "user", "content": "Hello"}],
            tools=[{"function_declarations": [{"name": "lookup"}]}],
        )
        assert len(request.tools) == 1

# ========================================================================
# Edge Cases and Error Handling
# ========================================================================

class TestEdgeCasesAndErrors:
    """Test edge cases and error conditions."""

    @pytest.mark.unit
    def test_invalid_role_fails(self):
        """Test that invalid message role fails validation."""
        with pytest.raises(ValidationError):
            ChatCompletionUserMessageParam(
                role="invalid_role",  # Should be "user"
                content="test"
            )

    @pytest.mark.unit
    def test_missing_content_fails(self):
        """Test that message without content fails."""
        with pytest.raises(ValidationError):
            ChatCompletionUserMessageParam(role="user")

    @pytest.mark.unit
    def test_max_tokens_bounds(self):
        """Test max_tokens parameter bounds."""
        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        assert request.max_tokens == 1

        # Negative max_tokens should fail
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=-1
            )

    @pytest.mark.unit
    def test_n_parameter_bounds(self):
        """Test n parameter bounds (1 to 128)."""
        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            n=1
        )
        assert request.n == 1

        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            n=128
        )
        assert request.n == 128

        # Too many completions
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                n=129
            )

    @pytest.mark.unit
    def test_extra_fields_allowed(self):
        """Test that undeclared fields remain accepted for parse compatibility."""
        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            custom_param="custom_value",
            another_param=123,
        )
        # Dispatch only forwards declared fields and the explicit extra_body container.
        assert request.model_extra == {
            "custom_param": "custom_value",
            "another_param": 123,
        }


class TestContinuationSchema:
    """Test continuation extension schema and validation."""

    @pytest.mark.unit
    def test_tldw_continuation_branch_is_accepted(self):
        request = ChatCompletionRequest(
            model="gpt-4o-mini",
            conversation_id="conv-123",
            messages=[{"role": "user", "content": "Continue from here"}],
            tldw_continuation={
                "from_message_id": "msg-456",
                "mode": "branch",
                "assistant_prefill": "Partial response ",
            },
        )

        assert request.tldw_continuation is not None
        assert request.tldw_continuation.mode == "branch"
        assert request.tldw_continuation.from_message_id == "msg-456"

    @pytest.mark.unit
    def test_tldw_continuation_requires_conversation_id(self):
        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Continue from here"}],
                tldw_continuation={
                    "from_message_id": "msg-456",
                    "mode": "append",
                },
            )

        assert "conversation_id is required" in str(exc_info.value)
