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
    # SUPPORTED_API_ENDPOINTS # Not directly testable other than through ChatCompletionRequest
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
    assert "String url must be a data URI" in str(exc_info.value)


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
def test_chat_completion_request_invalid_api_provider():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="test-m",
            messages=[ChatCompletionUserMessageParam(role="user", content="hi")],
            api_provider="non_existent_provider_literal_test"
        )