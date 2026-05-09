# tldw_Server_API/tests/Chat/test_chat_request_schemas.py
import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    ChatCompletionRequest,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestMessageContentPartImage,
    ChatCompletionRequestMessageContentPartImageURL,
    FunctionDefinition,
    ToolDefinition,
    ToolChoiceOption,
    ToolChoiceFunction,
    ResponseFormat,
    SUPPORTED_API_ENDPOINTS,
)


# --- Tests for Message Content Parts ---
@pytest.mark.unit
def test_chat_message_content_part_image_url_valid():
    valid_http_url = "http://example.com/image.png"
    valid_data_url = "data:image/png;base64,abcdef12345="

    # HTTP URL directly passed (Pydantic v2 might convert it)
    # This depends on HttpUrl type strictness from Pydantic
    # For safety, let's assume HttpUrl objects are constructed if needed for schema compliance
    # However, the schema allows Union[HttpUrl, str] and validates the str part.

    # Valid data URI string
    image_part = ChatCompletionRequestMessageContentPartImageURL(url=valid_data_url)
    assert str(image_part.url) == valid_data_url  # Pydantic might keep it as str if it passes validation

    # Valid HttpUrl object (less direct for this field's typical use in requests)
    # from pydantic import HttpUrl
    # image_part_http = ChatCompletionRequestMessageContentPartImageURL(url=HttpUrl(valid_http_url))
    # assert str(image_part_http.url) == valid_http_url


@pytest.mark.unit
def test_chat_message_content_part_image_url_invalid():
    invalid_string_url = "example.com/image.png"  # Not a data URI
    with pytest.raises(ValidationError) as exc_info:
        ChatCompletionRequestMessageContentPartImageURL(url=invalid_string_url)
    assert "data URI for base64 encoded images" in str(exc_info.value)


# --- Tests for Message Types ---
@pytest.mark.unit
def test_chat_completion_assistant_message_param_validation():
    # Valid: content only
    msg_content = ChatCompletionAssistantMessageParam(role="assistant", content="Hello")
    assert msg_content.content == "Hello"

    # Valid: tool_calls only
    tool_call = {"id": "call1", "type": "function", "function": {"name": "func", "description": "d", "parameters": {}}}
    msg_tools = ChatCompletionAssistantMessageParam(role="assistant", tool_calls=[tool_call])
    assert msg_tools.tool_calls[0].id == "call1"

    # Valid: tool_calls with OpenAI-style arguments
    tool_call_args = {
        "id": "call2",
        "type": "function",
        "function": {"name": "func", "arguments": "{\"foo\": \"bar\"}"},
    }
    msg_tools_args = ChatCompletionAssistantMessageParam(role="assistant", tool_calls=[tool_call_args])
    assert msg_tools_args.tool_calls[0].function.arguments == "{\"foo\": \"bar\"}"

    # Invalid: neither content nor tool_calls
    with pytest.raises(ValidationError) as exc_info:
        ChatCompletionAssistantMessageParam(role="assistant")
    assert "Assistant message must have content or tool_calls" in str(exc_info.value)


# --- Tests for ChatCompletionRequest ---
@pytest.mark.unit
def test_chat_completion_request_logprobs_validation():
    base_messages = [ChatCompletionUserMessageParam(role="user", content="hi")]
    # Valid: logprobs=True, top_logprobs=5
    req_valid = ChatCompletionRequest(model="m", messages=base_messages, logprobs=True, top_logprobs=5)
    assert req_valid.logprobs is True
    assert req_valid.top_logprobs == 5

    # Invalid: top_logprobs without logprobs=True
    with pytest.raises(ValidationError) as exc_info:
        ChatCompletionRequest(model="m", messages=base_messages, logprobs=False, top_logprobs=5)
    assert "If top_logprobs is specified, logprobs must be set to true" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info_none:
        ChatCompletionRequest(model="m", messages=base_messages, top_logprobs=5)  # logprobs defaults to False
    assert "If top_logprobs is specified, logprobs must be set to true" in str(exc_info_none.value)


@pytest.mark.unit
def test_chat_completion_request_valid_api_provider():
    # Assuming "openai" is in SUPPORTED_API_ENDPOINTS
    req = ChatCompletionRequest(
        model="test-m",
        messages=[ChatCompletionUserMessageParam(role="user", content="hi")],
        api_provider="openai"
    )
    assert req.api_provider == "openai"


@pytest.mark.unit
def test_supported_api_endpoints_uses_python310_safe_runtime_validation():
    assert SUPPORTED_API_ENDPOINTS is str


@pytest.mark.unit
def test_chat_completion_request_accepts_numbered_custom_openai_provider():
    req = ChatCompletionRequest(
        model="test-m",
        messages=[ChatCompletionUserMessageParam(role="user", content="hi")],
        api_provider="custom-openai-api-99",
    )

    assert req.api_provider == "custom-openai-api-99"


@pytest.mark.unit
def test_chat_completion_request_invalid_api_provider():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="test-m",
            messages=[ChatCompletionUserMessageParam(role="user", content="hi")],
            api_provider="non_existent_provider_literal_test"
        )


@pytest.mark.unit
def test_chat_completion_request_accepts_tldw_continuation_branch():
    req = ChatCompletionRequest(
        model="test-m",
        conversation_id="conv-1",
        messages=[ChatCompletionUserMessageParam(role="user", content="continue")],
        tldw_continuation={
            "from_message_id": "msg-1",
            "mode": "branch",
            "assistant_prefill": "Draft: ",
        },
    )
    assert req.tldw_continuation is not None
    assert req.tldw_continuation.mode == "branch"
    assert req.tldw_continuation.from_message_id == "msg-1"


@pytest.mark.unit
def test_chat_completion_request_requires_conversation_for_tldw_continuation():
    with pytest.raises(ValidationError) as exc_info:
        ChatCompletionRequest(
            model="test-m",
            messages=[ChatCompletionUserMessageParam(role="user", content="continue")],
            tldw_continuation={
                "from_message_id": "msg-1",
                "mode": "append",
            },
        )
    assert "conversation_id is required" in str(exc_info.value)


@pytest.mark.unit
def test_chat_completion_request_accepts_json_schema_response_format():
    req = ChatCompletionRequest(
        model="test-m",
        messages=[ChatCompletionUserMessageParam(role="user", content="Return structured")],
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
    assert req.response_format is not None
    assert req.response_format.type == "json_schema"
    assert req.response_format.json_schema is not None
    assert req.response_format.json_schema.name == "answer_schema"


@pytest.mark.unit
def test_chat_completion_request_rejects_json_schema_without_schema_object():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="test-m",
            messages=[ChatCompletionUserMessageParam(role="user", content="Return structured")],
            response_format={"type": "json_schema", "json_schema": {"name": "bad"}},
        )


@pytest.mark.unit
def test_chat_completion_request_accepts_llamacpp_extension_fields():
    req = ChatCompletionRequest(
        model="llama.cpp/local-model",
        messages=[ChatCompletionUserMessageParam(role="user", content="reply in JSON")],
        grammar_mode="library",
        grammar_id="grammar_123",
        grammar_override='root ::= "ok"',
        thinking_budget_tokens=128,
    )
    assert req.grammar_mode == "library"
    assert req.grammar_id == "grammar_123"
    assert req.thinking_budget_tokens == 128


@pytest.mark.unit
def test_chat_completion_request_rejects_llamacpp_library_mode_without_grammar_id():
    with pytest.raises(ValidationError) as exc_info:
        ChatCompletionRequest(
            model="llama.cpp/local-model",
            messages=[ChatCompletionUserMessageParam(role="user", content="x")],
            grammar_mode="library",
        )
    assert "grammar_id is required" in str(exc_info.value)


@pytest.mark.unit
def test_chat_completion_request_rejects_llamacpp_inline_mode_without_grammar_inline():
    with pytest.raises(ValidationError) as exc_info:
        ChatCompletionRequest(
            model="llama.cpp/local-model",
            messages=[ChatCompletionUserMessageParam(role="user", content="x")],
            grammar_mode="inline",
        )
    assert "grammar_inline is required" in str(exc_info.value)


# --- Tests for FunctionDefinition Parameter Validation ---

@pytest.mark.unit
def test_function_definition_valid_parameters():
    """Test that valid JSON Schema parameters are accepted."""
    valid_params = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The name to greet"},
            "age": {"type": "integer", "description": "Age in years"}
        },
        "required": ["name"]
    }
    func = FunctionDefinition(name="greet", parameters=valid_params)
    assert func.parameters == valid_params


@pytest.mark.unit
def test_function_definition_empty_parameters():
    """Test that empty/None parameters are accepted."""
    func1 = FunctionDefinition(name="noop", parameters={})
    assert func1.parameters == {}

    func2 = FunctionDefinition(name="noop2", parameters=None)
    assert func2.parameters is None


@pytest.mark.unit
def test_function_definition_invalid_type():
    """Test that invalid JSON Schema types are rejected."""
    invalid_params = {
        "type": "invalid_type",  # Not a valid JSON Schema type
        "properties": {}
    }
    with pytest.raises(ValidationError) as exc_info:
        FunctionDefinition(name="test", parameters=invalid_params)
    assert "Invalid JSON Schema type" in str(exc_info.value)


@pytest.mark.unit
def test_function_definition_excessive_depth():
    """Test that deeply nested parameters are rejected (DoS prevention)."""
    # Build a structure deeper than MAX_PARAMETER_DEPTH (10)
    deep_params = {"type": "object", "properties": {}}
    current = deep_params["properties"]
    for i in range(15):  # Create 15 levels of nesting
        current["nested"] = {"type": "object", "properties": {}}
        current = current["nested"]["properties"]

    with pytest.raises(ValidationError) as exc_info:
        FunctionDefinition(name="deep", parameters=deep_params)
    assert "maximum nesting depth" in str(exc_info.value)


@pytest.mark.unit
def test_function_definition_excessive_size():
    """Test that oversized parameters are rejected (DoS prevention)."""
    # Create a large but shallow structure (> 5KB)
    large_params = {
        "type": "object",
        "properties": {
            f"field_{i}": {"type": "string", "description": "x" * 100}
            for i in range(100)
        }
    }
    with pytest.raises(ValidationError) as exc_info:
        FunctionDefinition(name="large", parameters=large_params)
    assert "maximum size" in str(exc_info.value)


@pytest.mark.unit
def test_function_definition_invalid_required_field():
    """Test that invalid 'required' field structure is rejected."""
    invalid_params = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": "name"  # Should be an array, not a string
    }
    with pytest.raises(ValidationError) as exc_info:
        FunctionDefinition(name="test", parameters=invalid_params)
    assert "'required' must be an array" in str(exc_info.value)


@pytest.mark.unit
def test_function_definition_invalid_properties_field():
    """Test that invalid 'properties' field structure is rejected."""
    invalid_params = {
        "type": "object",
        "properties": ["name", "age"]  # Should be an object, not an array
    }
    with pytest.raises(ValidationError) as exc_info:
        FunctionDefinition(name="test", parameters=invalid_params)
    assert "'properties' must be an object" in str(exc_info.value)


@pytest.mark.unit
def test_function_definition_array_items_validation():
    """Test that array items schema is validated."""
    valid_array_params = {
        "type": "array",
        "items": {"type": "string"}
    }
    func = FunctionDefinition(name="list_func", parameters=valid_array_params)
    assert func.parameters == valid_array_params

    invalid_array_params = {
        "type": "array",
        "items": {"type": "invalid_type"}
    }
    with pytest.raises(ValidationError) as exc_info:
        FunctionDefinition(name="list_func", parameters=invalid_array_params)
    assert "Invalid JSON Schema type" in str(exc_info.value)


@pytest.mark.unit
def test_function_definition_type_array():
    """Test that type arrays (union types) are validated."""
    valid_union_params = {
        "type": ["string", "null"],
        "description": "Optional string"
    }
    func = FunctionDefinition(name="optional", parameters=valid_union_params)
    assert func.parameters == valid_union_params

    invalid_union_params = {
        "type": ["string", "invalid"],
        "description": "Invalid union"
    }
    with pytest.raises(ValidationError) as exc_info:
        FunctionDefinition(name="invalid_union", parameters=invalid_union_params)
    assert "Invalid JSON Schema type" in str(exc_info.value)
