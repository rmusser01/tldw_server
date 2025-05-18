# tests/unit/api/v1/endpoints/test_chat_endpoint.py
import pytest
from unittest.mock import patch, MagicMock, ANY
from fastapi import status, HTTPException
from fastapi.testclient import TestClient
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import your FastAPI app instance
from tldw_Server_API.app.main import app
# Import schemas from your actual file path
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    ChatCompletionRequest,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestMessageContentPartImage,
    ChatCompletionRequestMessageContentPartImageURL,
    ResponseFormat,
    ToolDefinition,
    FunctionDefinition,
    ToolChoiceOption,
    ToolChoiceFunction
)
from tldw_Server_API.app.core.Chat.Chat_Functions import (
    ChatAuthenticationError, ChatRateLimitError, ChatBadRequestError,
    ChatConfigurationError, ChatProviderError, ChatAPIError
)
from tldw_Server_API.app.core.Chat.prompt_template_manager import PromptTemplate, DEFAULT_RAW_PASSTHROUGH_TEMPLATE
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user, DEFAULT_CHARACTER_NAME


# Fixture for TestClient
@pytest.fixture(scope="function")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def valid_auth_token() -> str:
    # This MUST match the value of os.getenv("API_BEARER") in the app's runtime environment.
    token = "default-secret-key-for-single-user"
    if os.getenv("API_BEARER") != token:
        pytest.fail(f"MISMATCH: API_BEARER is '{os.getenv('API_BEARER')}' but test token is '{token}'")
    return token


# --- Test Data defined locally in this file ---
DEFAULT_MODEL_NAME = "test-model-unit"  # Changed name to avoid potential clash if importing
DEFAULT_USER_MESSAGES_FOR_SCHEMA = [
    ChatCompletionUserMessageParam(role="user", content="Hello from unit test")
]  # This was the missing declaration


# Fixture to provide default chat request data
@pytest.fixture
def default_chat_request_data():
    """Provides a default ChatCompletionRequest object for tests."""
    return ChatCompletionRequest(
        model=DEFAULT_MODEL_NAME,
        messages=DEFAULT_USER_MESSAGES_FOR_SCHEMA  # Use the locally defined constant
    )

@pytest.fixture
def default_chat_request_data_error_stream():
    """Provides a default ChatCompletionRequest object for error streaming tests."""
    return ChatCompletionRequest(
        model=DEFAULT_MODEL_NAME_UNIT,
        messages=DEFAULT_USER_MESSAGES_FOR_SCHEMA_UNIT,
        stream=True # Ensure stream is true for this test
    )


# Mocks for DB dependencies
@pytest.fixture
def mock_chat_db():
    db_mock = MagicMock(spec=CharactersRAGDB)
    db_mock.get_character_card_by_id.return_value = None
    # Add default character card for by_name lookup for default character
    db_mock.get_character_card_by_name.return_value = {
        'id': 'mock_default_char_id_123', # Or an int if your DB returns int
        'name': DEFAULT_CHARACTER_NAME, # Assuming DEFAULT_CHARACTER_NAME is importable or defined
        'system_prompt': 'Mock default system prompt'
        # Add any other fields your endpoint/templating might try to access from the default char
    }
    db_mock.add_conversation.return_value = "mock_conv_id_xyz" # For new conversation creation
    db_mock.add_message.return_value = "mock_message_id_abc" # Or whatever it should return

    # >>> ADD THIS LINE <<<
    db_mock.client_id = "test_client_unit"
    return db_mock


@pytest.fixture
def mock_media_db():
    return MagicMock()


# --- Unit Tests for the Endpoint ---

@pytest.mark.unit
@patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")
@patch(
    "tldw_Server_API.app.api.v1.endpoints.chat.apply_template_to_string")  # Keep patch in case logic changes, but expect no calls
def test_create_chat_completion_no_template(
        mock_apply_template, mock_load_template, mock_chat_api_call,
        client, valid_auth_token, mock_media_db, mock_chat_db, default_chat_request_data
):
    # Simulate that when the endpoint tries to load the default template (or any specified one),
    # it's not found, making active_template=None in the endpoint.
    mock_load_template.return_value = None

    mock_response_data = {"id": "chatcmpl-no-template",
                          "choices": [{"message": {"role": "assistant", "content": "Raw response"}}]}
    mock_chat_api_call.return_value = mock_response_data

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    request_data_dict = default_chat_request_data.model_dump()
    # In default_chat_request_data, prompt_template_name is None.
    # The endpoint will try to load DEFAULT_RAW_PASSTHROUGH_TEMPLATE.name.

    response = client.post("/api/v1/chat/completions", json=request_data_dict, headers={"Token": valid_auth_token})

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["choices"][0]["message"]["content"] == mock_response_data["choices"][0]["message"][
        "content"]  # Check relevant part

    mock_chat_api_call.assert_called_once()
    called_kwargs = mock_chat_api_call.call_args.kwargs

    # Since default_chat_request_data has no system message and active_template is None,
    # final_system_message should be empty.
    assert "system_message" not in called_kwargs

    # Messages should be passed through as is because active_template is None
    expected_payload_messages_as_dicts = [msg.model_dump(exclude_none=True) for msg in DEFAULT_USER_MESSAGES_FOR_SCHEMA]
    actual_payload_messages = called_kwargs["messages_payload"]
    assert actual_payload_messages == expected_payload_messages_as_dicts

    # load_template IS called because request_data.prompt_template_name is None,
    # so it tries to load DEFAULT_RAW_PASSTHROUGH_TEMPLATE.
    mock_load_template.assert_called_once_with(DEFAULT_RAW_PASSTHROUGH_TEMPLATE.name)

    # apply_template_to_string should NOT be called because active_template is None
    # (due to mock_load_template.return_value = None) and sys_msg_from_req is empty.
    mock_apply_template.assert_not_called()

    app.dependency_overrides = {}


# (The rest of the tests in test_chat_endpoint.py remain the same as the corrected version from the previous response)
# Ensure they use the `default_chat_request_data` fixture where appropriate.

@pytest.mark.unit
@patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.apply_template_to_string")
def test_create_chat_completion_success_streaming(  # Added default_chat_request_data fixture
        mock_apply_template, mock_load_template, mock_chat_api_call,
        client, default_chat_request_data, valid_auth_token, mock_media_db, mock_chat_db
):
    mock_load_template.return_value = None  # Default passthrough
    mock_apply_template.side_effect = lambda template_str, data: data.get("message_content", data.get(
        "original_system_message_from_request", ""))

    with patch(
            "tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call") as mock_chat_api_call_inner:  # Renamed for clarity
        def mock_stream_generator():  # This mock is for the return value of perform_chat_api_call
            yield "Hello"  # Just the content delta
            yield " World"

        mock_chat_api_call_inner.return_value = mock_stream_generator()  # Corrected to use the inner mock

        app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
        app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

        streaming_request_data = default_chat_request_data.model_copy(update={"stream": True})
        response = client.post(
            "/api/v1/chat/completions",
            json=streaming_request_data.model_dump(),
            headers={"token": valid_auth_token}
        )

        assert response.status_code == status.HTTP_200_OK
        assert "text/event-stream" in response.headers["content-type"].lower()

        stream_content = response.text
        events = [line for line in stream_content.split("\n\n") if line.strip()]
        assert events[0].startswith("event: tldw_metadata")
        # Assuming mock_conv_id_xyz is correct for conversation_id
        assert json.loads(events[0].split("data: ", 1)[1])["conversation_id"] == "mock_conv_id_xyz"
        # events[1] would be 'data: {"choices": [{"delta": {"content": "Hello"}}]}' (after metadata)
        # events[2] would be 'data: {"choices": [{"delta": {"content": " World"}}]}'
        assert json.loads(events[1].split("data: ", 1)[1])["choices"][0]["delta"]["content"] == "Hello"
        assert json.loads(events[2].split("data: ", 1)[1])["choices"][0]["delta"]["content"] == " World"
        assert "data: " in events[-1] and json.loads(events[-1].split("data: ", 1)[1]).get("choices")[0].get("finish_reason") == "stop"

        mock_chat_api_call_inner.assert_called_once()
        call_args = mock_chat_api_call_inner.call_args[1]
        assert call_args["streaming"] is True

    app.dependency_overrides = {}


@pytest.mark.unit
@patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.apply_template_to_string")
def test_system_message_extraction(
        mock_apply_template, mock_load_template, mock_chat_api_call,
        client, valid_auth_token, mock_media_db, mock_chat_db
):
    mock_load_template.return_value = None  # Default passthrough
    mock_apply_template.side_effect = lambda template_str, data: data.get("message_content", data.get(
        "original_system_message_from_request", ""))
    mock_chat_api_call.return_value = {"id": "chatcmpl-123",
                                       "choices": [{"message": {"role": "assistant", "content": "Test response"}}]}

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    messages_with_system = [
        ChatCompletionSystemMessageParam(role="system", content="You are a helpful assistant."),
        ChatCompletionUserMessageParam(role="user", content="Hello there.")
    ]
    # Use the specific Pydantic models from your schema
    request_data_obj = ChatCompletionRequest(model="test-model", messages=messages_with_system)

    client.post("/api/v1/chat/completions", json=request_data_obj.model_dump(), headers={"Token": valid_auth_token})

    mock_chat_api_call.assert_called_once()
    call_args = mock_chat_api_call.call_args.kwargs
    assert call_args["system_message"] == "You are a helpful assistant."
    assert len(call_args["messages_payload"]) == 1
    assert call_args["messages_payload"][0]["role"] == "user"
    assert call_args["messages_payload"][0]["content"] == "Hello there."  # Assuming passthrough template
    app.dependency_overrides = {}


@pytest.mark.unit
@patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.apply_template_to_string")
def test_no_system_message_in_payload(
        mock_apply_template, mock_load_template, mock_chat_api_call,
        client, default_chat_request_data, valid_auth_token, mock_media_db, mock_chat_db
        # Added default_chat_request_data
):
    mock_load_template.return_value = None  # Default passthrough because prompt_template_name is None in default_chat_request_data
    # Simulate passthrough for apply_template when default template is used
    mock_apply_template.side_effect = lambda template_str, data: data.get("message_content", data.get(
        "original_system_message_from_request", ""))

    mock_chat_api_call.return_value = {"id": "chatcmpl-123",
                                       "choices": [{"message": {"role": "assistant", "content": "Test response"}}]}

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    client.post("/api/v1/chat/completions", json=default_chat_request_data.model_dump(),
                headers={"Token": valid_auth_token})

    mock_chat_api_call.assert_called_once()
    # This line was missing:
    called_kwargs = mock_chat_api_call.call_args.kwargs  # Get the keyword arguments passed to the mock

    # Assert that system_message passed to chat_api_call is None because:
    # 1. default_chat_request_data has no system message.
    # 2. No template is specified, so DEFAULT_RAW_PASSTHROUGH_TEMPLATE is used.
    # 3. DEFAULT_RAW_PASSTHROUGH_TEMPLATE.system_message_template is likely empty or just "{original_system_message_from_request}".
    # 4. original_system_message_from_request will be "" or None.
    # So, final_system_message_for_provider should be None or empty, leading to system_message=None in chat_args_cleaned.
    assert mock_chat_api_call.get("system_message") == ""

    # Ensure the messages in the payload are dictionaries and match the input (since it's passthrough)
    expected_payload_messages_as_dicts = [msg.model_dump(exclude_none=True) for msg in DEFAULT_USER_MESSAGES_FOR_SCHEMA]
    # The endpoint logic (after templating with passthrough) should result in this payload
    actual_payload_messages = called_kwargs["messages_payload"]

    assert len(actual_payload_messages) == len(expected_payload_messages_as_dicts)
    for actual_msg, expected_msg in zip(actual_payload_messages, expected_payload_messages_as_dicts):
        assert actual_msg["role"] == expected_msg["role"]
        # The content from the user message (a string in this case) should pass through the template
        # The template application might wrap it in a list of content parts if it wasn't already
        # Your endpoint logic: `msg_dict["content"] = new_content_str` for string content
        # So it should be a direct string comparison if the passthrough template just returns the message_content
        assert actual_msg["content"] == expected_msg["content"]

    app.dependency_overrides = {}

VALID_ALTERNATIVE_PROVIDER_FOR_TEST = "groq"


@pytest.mark.unit
# Update the patch.dict to use this valid alternative provider name
@patch.dict("tldw_Server_API.app.api.v1.schemas.chat_request_schemas.API_KEYS", {
    "openai": "key_from_config",
    VALID_ALTERNATIVE_PROVIDER_FOR_TEST: "alternative_key_for_test",  # Use the valid name
    "cohere": "cohere_test_key_if_needed_separately"  # If you still have a cohere specific part
})
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.apply_template_to_string")
def test_api_key_used_from_config(
        mock_apply_template, mock_load_template,
        client, default_chat_request_data, valid_auth_token, mock_media_db, mock_chat_db
):
    mock_load_template.return_value = None
    mock_apply_template.side_effect = lambda template_str, data: data.get("message_content", data.get(
        "original_system_message_from_request", ""))

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    with patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call") as mock_chat_api_call:
        mock_chat_api_call.return_value = {"id": "res_openai"}

        # Test 1: Default provider (should pick up "openai" key from patched dict)
        # default_chat_request_data has api_provider=None, so it will use DEFAULT_LLM_PROVIDER ("openai")
        client.post("/api/v1/chat/completions", json=default_chat_request_data.model_dump(),
                    headers={"Token": valid_auth_token})
        mock_chat_api_call.assert_called_once()
        assert mock_chat_api_call.call_args.kwargs["api_key"] == "key_from_config"
        assert mock_chat_api_call.call_args.kwargs["api_endpoint"] == "openai"  # Check target endpoint

        mock_chat_api_call.reset_mock()
        mock_chat_api_call.return_value = {"id": "res_alternative"}

        # Test 2: Specific valid alternative provider
        request_data_alternative = default_chat_request_data.model_copy(
            update={
                "api_provider": VALID_ALTERNATIVE_PROVIDER_FOR_TEST,
                # Ensure a model is provided if the alternative provider requires it,
                # default_chat_request_data already includes a model='test-model-unit'
                # which might be fine if the mock_chat_api_call doesn't care about model validity for this unit test.
                # If it were an integration test, a valid model for the provider would be needed.
            }
        )
        response_alternative = client.post("/api/v1/chat/completions", json=request_data_alternative.model_dump(),
                                           headers={"Token": valid_auth_token})

        # This assertion should now pass if VALID_ALTERNATIVE_PROVIDER_FOR_TEST is correctly handled
        assert response_alternative.status_code == status.HTTP_200_OK, \
            f"Alternative provider '{VALID_ALTERNATIVE_PROVIDER_FOR_TEST}' failed: {response_alternative.text}"

        mock_chat_api_call.assert_called_once()
        assert mock_chat_api_call.call_args.kwargs["api_key"] == "alternative_key_for_test"
        assert mock_chat_api_call.call_args.kwargs["api_endpoint"] == VALID_ALTERNATIVE_PROVIDER_FOR_TEST

        # If you had a third part of the test for "cohere" specifically:
        mock_chat_api_call.reset_mock()
        mock_chat_api_call.return_value = {"id": "res_cohere"}
        request_data_cohere = default_chat_request_data.model_copy(
            update={"api_provider": "cohere", "model": "command-r"}  # Assuming 'cohere' is in SUPPORTED_API_ENDPOINTS
        )
        response_cohere = client.post("/api/v1/chat/completions", json=request_data_cohere.model_dump(),
                                      headers={"Token": valid_auth_token})
        assert response_cohere.status_code == status.HTTP_200_OK, f"Cohere provider failed: {response_cohere.text}"
        mock_chat_api_call.assert_called_once()
        assert mock_chat_api_call.call_args.kwargs["api_key"] == "cohere_test_key_if_needed_separately"
        assert mock_chat_api_call.call_args.kwargs["api_endpoint"] == "cohere"

    app.dependency_overrides = {}


@pytest.mark.unit
@patch.dict("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": ""})  # Empty key for openai
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.apply_template_to_string")
def test_missing_api_key_for_required_provider(
        mock_apply_template, mock_load_template,
        client, default_chat_request_data, valid_auth_token, mock_media_db, mock_chat_db
):
    # Simulate that the default template is found and is a passthrough
    mock_load_template.return_value = DEFAULT_RAW_PASSTHROUGH_TEMPLATE
    mock_apply_template.side_effect = lambda template_str, data: data.get("message_content", data.get(
        "original_system_message_from_request", ""))

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    # We don't need to mock chat_api_call here as the error should occur before it's called.
    # However, if any part of the setup before the key check calls it, a basic mock might be needed.
    # The key check happens quite early.

    # default_chat_request_data uses 'openai' as the default provider if api_provider is None.
    # We've patched API_KEYS so "openai" has an empty string key.
    # The endpoint's providers_requiring_keys list includes "openai".
    request_data_openai = default_chat_request_data.model_copy(update={"api_provider": "openai"})

    response = client.post(
        "/api/v1/chat/completions",
        json=request_data_openai.model_dump(),
        headers={"token": valid_auth_token}
    )

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    expected_detail = "Service for 'openai' is not configured (key missing)."  # Or the relevant provider
    assert response.json()["detail"] == expected_detail

    app.dependency_overrides = {}


@pytest.mark.unit
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")  # Mock template deps
@patch("tldw_Server_API.app.api.v1.endpoints.chat.apply_template_to_string")
def test_keyless_provider_proceeds_without_key(  # Added default_chat_request_data
        mock_apply_template, mock_load_template,
        client, default_chat_request_data, valid_auth_token, mock_media_db, mock_chat_db
):
    mock_load_template.return_value = None  # Default passthrough
    mock_apply_template.side_effect = lambda template_str, data: data.get("message_content", data.get(
        "original_system_message_from_request", ""))

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    with patch.dict("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {}, clear=True), \
            patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call") as mock_chat_api_call:
        mock_chat_api_call.return_value = {"id": "res_ollama"}
        request_data_ollama = default_chat_request_data.model_copy(update={"api_provider": "ollama"})

        response = client.post(
            "/api/v1/chat/completions",
            json=request_data_ollama.model_dump(),
            headers={"token": valid_auth_token}
        )
        assert response.status_code == status.HTTP_200_OK
        mock_chat_api_call.assert_called_once()
        assert mock_chat_api_call.call_args[1].get(
            "api_key") is None  # Check that api_key was indeed None or not passed
    app.dependency_overrides = {}


@pytest.mark.unit
@patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call")  # Corrected patch target
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.apply_template_to_string")
@pytest.mark.parametrize("error_type, expected_status, expected_detail_substring", [
    (ChatAuthenticationError(provider="test", message="Auth failed detail from lib"),
     # Error from perform_chat_api_call
     status.HTTP_401_UNAUTHORIZED,
     "Auth failed detail from lib"),  # Endpoint uses the lib's message for < 500 errors

    (ChatRateLimitError(provider="test", message="Rate limit detail from lib"),
     status.HTTP_429_TOO_MANY_REQUESTS,
     "Rate limit detail from lib"),

    (ChatBadRequestError(provider="test", message="Bad request detail from lib"),
     status.HTTP_400_BAD_REQUEST,
     "Bad request detail from lib"),

    (ChatConfigurationError(provider="test", message="Config error from lib"),  # This is a 5xx type error
     status.HTTP_503_SERVICE_UNAVAILABLE,  # Endpoint maps ChatConfigurationError to 503
     "An error occurred with the chat provider or service configuration."),  # Endpoint masks 5xx details

    (ChatProviderError(provider="test", message="Provider issue from lib", status_code=503),  # This is a 5xx type error
     status.HTTP_503_SERVICE_UNAVAILABLE,
     "An error occurred with the chat provider or service configuration."),

    (ChatProviderError(provider="test", message="Provider non-HTTP issue from lib", status_code=502),
     # This is a 5xx type error
     status.HTTP_502_BAD_GATEWAY,
     "An error occurred with the chat provider or service configuration."),

    (ChatAPIError(provider="test", message="Generic API issue from lib", status_code=500),  # This is a 5xx type error
     status.HTTP_500_INTERNAL_SERVER_ERROR,
     "An error occurred with the chat provider or service configuration."),

    # Case: A non-library, non-HTTPException error from perform_chat_api_call (e.g., a raw ValueError)
    # The endpoint's final `except Exception` catches this.
    (ValueError("Value error from shim"),
     status.HTTP_500_INTERNAL_SERVER_ERROR,  # Endpoint's generic catch-all
     "An unexpected internal server error occurred."),

    # Case: An HTTPException raised directly by perform_chat_api_call
    # The endpoint's `except HTTPException` should re-raise it.
    (HTTPException(status_code=418, detail="I'm a teapot from shim"),
     418,  # The status code from the raised HTTPException
     "I'm a teapot from shim")  # The detail from the raised HTTPException
])
def test_chat_api_call_exception_handling_unit(
        mock_apply_template, mock_load_template, mock_perform_chat_api_call,  # Corrected mock name
        client, valid_auth_token, mock_media_db, mock_chat_db,
        default_chat_request_data,
        error_type, expected_status, expected_detail_substring
):
    mock_load_template.return_value = DEFAULT_RAW_PASSTHROUGH_TEMPLATE
    mock_apply_template.side_effect = lambda template_str, data: data.get("message_content", data.get(
        "original_system_message_from_request", ""))

    # This is the mock for perform_chat_api_call inside the endpoint
    mock_perform_chat_api_call.side_effect = error_type

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    response = client.post(
        "/api/v1/chat/completions",
        json=default_chat_request_data.model_dump(),
        headers={"Token": valid_auth_token}  # Ensure correct header name
    )

    assert response.status_code == expected_status

    response_json = response.json()
    assert "detail" in response_json, "Response JSON should contain a 'detail' field"
    response_detail_text = response_json["detail"]

    # For string details, check if the expected substring is present.
    # For dict details (e.g. Pydantic validation errors), this check might need adjustment,
    # but for these specific exception handlings, detail is expected to be a string.
    assert isinstance(response_detail_text,
                      str), f"Response detail should be a string, got {type(response_detail_text)}"
    assert expected_detail_substring.lower() in response_detail_text.lower(), \
        f"Expected detail '{expected_detail_substring}' not found in actual detail '{response_detail_text}'"

    app.dependency_overrides = {}


@pytest.mark.unit
@patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.apply_template_to_string")
def test_non_iterable_stream_generator_from_shim(
        mock_apply_template, mock_load_template, mock_chat_api_call,
        client, default_chat_request_data, valid_auth_token, mock_media_db, mock_chat_db
):
    # Simulate that the default template is found and is a passthrough
    mock_load_template.return_value = DEFAULT_RAW_PASSTHROUGH_TEMPLATE
    mock_apply_template.side_effect = lambda template_str, data: data.get("message_content", data.get(
        "original_system_message_from_request", ""))

    # Simulate chat_api_call (perform_chat_api_call) returning a non-iterable/non-async-iterable
    mock_chat_api_call.return_value = 123

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    streaming_request_data = default_chat_request_data.model_copy(update={"stream": True})
    response = client.post(
        "/api/v1/chat/completions",
        json=streaming_request_data.model_dump(),
        headers={"token": valid_auth_token}
    )

    assert response.status_code == status.HTTP_502_BAD_GATEWAY
    assert response.json()["detail"] == "Provider did not return a valid stream."

    app.dependency_overrides = {}

DEFAULT_MODEL_NAME_UNIT = "test-model-unit-error-stream"
DEFAULT_USER_MESSAGES_FOR_SCHEMA_UNIT = [
    ChatCompletionUserMessageParam(role="user", content="Test stream with error")
]
@pytest.mark.unit
@patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call")  # Patches the alias used in chat.py
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.apply_template_to_string")
def test_error_within_stream_generator(
        mock_apply_template, mock_load_template, mock_chat_api_call,
        client, default_chat_request_data, valid_auth_token, mock_media_db, mock_chat_db
, default_chat_request_data_error_stream):
    # Simulate that the default template (or any) is found and is a passthrough
    # or correctly loaded if it's DEFAULT_RAW_PASSTHROUGH_TEMPLATE.name
    mock_load_template.return_value = DEFAULT_RAW_PASSTHROUGH_TEMPLATE
    mock_apply_template.side_effect = lambda template_str, data: data.get("message_content", data.get(
        "original_system_message_from_request", ""))

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    # This inner patch is fine if you want to ensure the outer mock_perform_chat_api_call
    # doesn't interfere, but the outer one should suffice.
    # For clarity, using the already patched mock_perform_chat_api_call directly.

    def faulty_stream_generator():  # This is what perform_chat_api_call (mocked) will return
        yield "Good start..."  # Raw content, not full SSE
        raise ValueError("Something broke mid-stream!")

    mock_chat_api_call.return_value = faulty_stream_generator()

    # Use the specific request data for this test
    request_data_dict = default_chat_request_data_error_stream.model_dump()

    response = client.post(
        "/api/v1/chat/completions",
        json=request_data_dict,
        headers={"Token": valid_auth_token}  # Corrected header name
    )
    assert response.status_code == status.HTTP_200_OK  # Stream should still start with 200

    # Process the SSE stream
    raw_stream_text = response.text
    print(f"DEBUG Raw Stream Output:\n{raw_stream_text}")  # For debugging

    events = []
    current_event_data = []
    for line in raw_stream_text.splitlines():
        if not line.strip():  # End of an event
            if current_event_data:
                events.append("\n".join(current_event_data))
                current_event_data = []
        else:
            current_event_data.append(line)
    if current_event_data:  # Append last event if any
        events.append("\n".join(current_event_data))

    print(f"DEBUG Parsed SSE Events: {events}")

    assert len(events) >= 3  # Expect metadata, data/error, DONE

    # 1. Check metadata event
    assert events[0].startswith("event: tldw_metadata")
    metadata_json = json.loads(events[0].split("data: ", 1)[1])
    assert metadata_json["conversation_id"] == "mock_conv_id_xyz"
    assert metadata_json["model"] == DEFAULT_MODEL_NAME_UNIT

    # 2. Check for "Good start..." data chunk formatted by the endpoint
    # This part might be tricky if the error occurs before the first data yield is fully processed by sse_event_generator
    # The sse_event_generator will try to yield the "Good start..." data first.
    first_data_event_found = False
    error_event_found = False
    done_event_found = False

    for event_str in events[1:]:  # Skip metadata
        if event_str.startswith("data: "):
            try:
                payload_str = event_str.split("data: ", 1)[1]
                payload = json.loads(payload_str)

                if "choices" in payload and payload["choices"]:
                    delta = payload["choices"][0].get("delta", {})
                    if "content" in delta and delta["content"] == "Good start...":
                        first_data_event_found = True
                    if payload["choices"][0].get("finish_reason") == "stop":
                        done_event_found = True
                        assert payload.get("tldw_conversation_id") == "mock_conv_id_stream_error"

                if "error" in payload and "message" in payload["error"]:
                    if "Something broke mid-stream!" in payload["error"]["message"] or \
                            "Stream failed due to provider error." in payload["error"][
                        "message"]:  # Endpoint wraps the error
                        error_event_found = True
            except json.JSONDecodeError:
                print(f"WARN: Could not decode JSON from event: {payload_str}")
                continue

    # Depending on precise timing of exception in generator, "Good start..." might or might not appear
    # The error and DONE message are more crucial for this test.
    # If faulty_stream_generator raises immediately after yielding "Good start...",
    # the endpoint's sse_event_generator should process that yield, then catch the error.
    assert first_data_event_found, "The 'Good start...' data chunk was not found or not correctly formatted."
    assert error_event_found, "The SSE error message was not found or not correctly formatted."
    assert done_event_found, "The SSE DONE message was not found or not correctly formatted."

    # Ensure the background task (if any was triggered by the stream ending) is handled gracefully.
    # The test mock for _save_message_turn_to_db will be called by final_save_bg_task.
    # We can check its call if needed, but the main focus is the stream content.
    # mock_chat_db.add_message.assert_called() # Or more specific checks on what was saved.
    # The `_sse_state['full_reply']` would contain "Good start..." in this case.

    app.dependency_overrides = {}


@pytest.mark.unit
@patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")
def test_create_chat_completion_with_optional_params(
        mock_load_template, mock_chat_api_call,
        client, valid_auth_token, mock_media_db, mock_chat_db, default_chat_request_data
):
    mock_load_template.return_value = DEFAULT_RAW_PASSTHROUGH_TEMPLATE  # Use the actual default
    mock_chat_api_call.return_value = {"id": "chatcmpl-optional", "choices": [
        {"message": {"role": "assistant", "content": "Response with optionals"}}]}

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    request_with_optionals = default_chat_request_data.model_copy(update={
        "frequency_penalty": 0.5,
        "presence_penalty": -0.5,
        "logprobs": True,
        "top_logprobs": 5,
        "max_tokens": 150,
        "n": 2,
        "response_format": ResponseFormat(type="json_object"),
        "seed": 12345,
        "stop": ["\n", "stopword"],
        "user": "test-user-id",
        "minp": 0.05,  # Custom extension
        "topk": 50  # Custom extension
    })
    request_data_dict = request_with_optionals.model_dump(exclude_none=True)

    response = client.post("/api/v1/chat/completions", json=request_data_dict, headers={"Token": valid_auth_token})

    assert response.status_code == status.HTTP_200_OK
    mock_chat_api_call.assert_called_once()
    called_kwargs = mock_chat_api_call.call_args.kwargs

    assert called_kwargs["frequency_penalty"] == 0.5
    assert called_kwargs["presence_penalty"] == -0.5
    assert called_kwargs["logprobs"] is True
    assert called_kwargs["top_logprobs"] == 5
    # max_tokens is not directly mapped by chat_args, but by the schema to chat_api_call's provider logic
    # For a unit test of the endpoint, we check it's passed to chat_api_call if chat_api_call accepts it
    # The current chat_api_call doesn't explicitly list max_tokens in its signature,
    # so it depends on the underlying provider functions.
    # For now, we'll assume it's NOT directly passed by chat_api_call's main args
    # unless PROVIDER_PARAM_MAP is updated.
    # Let's focus on params explicitly in chat_api_call signature.
    # assert called_kwargs["n"] == 2 # 'n' is also not directly in chat_api_call signature
    # assert called_kwargs["response_format"] == {"type": "json_object"} # also not direct
    # assert called_kwargs["seed"] == 12345 # also not direct
    # assert called_kwargs["stop"] == ["\n", "stopword"] # also not direct
    assert called_kwargs["minp"] == 0.05
    assert called_kwargs["topk"] == 50
    # user is not directly passed to chat_api_call

    app.dependency_overrides = {}


@pytest.mark.unit
@patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")
def test_create_chat_completion_with_tools_unit(
        mock_load_template, mock_chat_api_call,
        client, valid_auth_token, mock_media_db, mock_chat_db, default_chat_request_data
):
    mock_load_template.return_value = DEFAULT_RAW_PASSTHROUGH_TEMPLATE
    mock_chat_api_call.return_value = {"id": "chatcmpl-tools",
                                       "choices": [{"message": {"role": "assistant", "tool_calls": []}}]}

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    tools_payload = [
        ToolDefinition(type="function",
                       function=FunctionDefinition(name="get_current_weather", description="Get weather",
                                                   parameters={"type": "object",
                                                               "properties": {"location": {"type": "string"}}}))
    ]
    tool_choice_payload = ToolChoiceOption(type="function", function=ToolChoiceFunction(name="get_current_weather"))

    request_with_tools = default_chat_request_data.model_copy(update={
        "tools": [t.model_dump(exclude_none=True) for t in tools_payload],  # Must be dicts
        "tool_choice": tool_choice_payload.model_dump(exclude_none=True)  # Must be dict
    })
    request_data_dict = request_with_tools.model_dump(exclude_none=True)

    response = client.post("/api/v1/chat/completions", json=request_data_dict, headers={"Token": valid_auth_token})
    assert response.status_code == status.HTTP_200_OK
    mock_chat_api_call.assert_called_once()
    called_kwargs = mock_chat_api_call.call_args.kwargs
    # tool_choice from request_data.model_dump() is a dict
    assert called_kwargs.get("tool_choice") == tool_choice_payload.model_dump(exclude_none=True)
    assert called_kwargs.get("tools") == [t.model_dump(exclude_none=True) for t in tools_payload]

    app.dependency_overrides = {}


@pytest.mark.unit
@patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")  # Mock this
def test_create_chat_completion_character_not_found_uses_defaults(
        mock_load_template, mock_chat_api_call_shim,
        client, valid_auth_token, mock_media_db, mock_chat_db, default_chat_request_data
):
    # Mock DB to return None for character
    mock_chat_db.get_character_card_by_id.return_value = None
    mock_load_template.return_value = DEFAULT_RAW_PASSTHROUGH_TEMPLATE  # Or a specific test template

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    mock_chat_api_call_shim.return_value = {"id": "res", "choices": [{"message": {"content": "default response"}}]}

    request_with_char = default_chat_request_data.model_copy(update={
        "character_id": "non_existent_char_id",
        "prompt_template_name": "some_template_that_uses_char_vars"  # Assume this template exists for the test
    })
    # If some_template_that_uses_char_vars is mocked by mock_load_template to use {char_name}
    # And char_name is not found, it will use the default "Character" from template_data initialization.

    client.post("/api/v1/chat/completions", json=request_with_char.model_dump(), headers={"Token": valid_auth_token})

    mock_chat_api_call_shim.assert_called_once()
    called_args_to_shim = mock_chat_api_call_shim.call_args.kwargs
    # Check that the system_message sent to chat_api_call uses default character values if the template was applied.
    # This depends on how DEFAULT_RAW_PASSTHROUGH_TEMPLATE is structured or the mock for "some_template_that_uses_char_vars"
    # If system_message_template in the active_template was "{char_name} says: {original_system_message_from_request}"
    # And no char was found, and no original system message, it might become "Character says: "
    # For the default passthrough, if original system message is empty, this would be empty.
    assert called_args_to_shim.get(
        "system_message") is not None  # It will be at least "" if DEFAULT_RAW_PASSTHROUGH_TEMPLATE is used

    # Verify DB was called
    mock_chat_db.get_character_card_by_id.assert_called_once_with("non_existent_char_id")

    app.dependency_overrides = {}


@pytest.mark.unit
@patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")
def test_create_chat_completion_template_file_not_found(
        mock_load_template, mock_chat_api_call_shim,
        client, valid_auth_token, mock_media_db, mock_chat_db, default_chat_request_data
):
    # Simulate load_template returning None (template not found)
    mock_load_template.return_value = None
    # chat_api_call should still be called with DEFAULT_RAW_PASSTHROUGH_TEMPLATE logic
    mock_chat_api_call_shim.return_value = {"id": "res", "choices": [{"message": {"content": "passthrough response"}}]}

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    request_data = default_chat_request_data.model_copy(update={
        "prompt_template_name": "definitely_missing_template"
    })

    response = client.post("/api/v1/chat/completions", json=request_data.model_dump(),
                           headers={"Token": valid_auth_token})
    assert response.status_code == status.HTTP_200_OK  # Should fall back to default template
    mock_load_template.assert_called_once_with("definitely_missing_template")
    mock_chat_api_call_shim.assert_called_once()
    # Further assertions could check if the payload sent to chat_api_call reflects the passthrough template.

    app.dependency_overrides = {}


