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
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionRequestMessageContentPartText,  # For multimodal user messages
    ChatCompletionRequestMessageContentPartImage,
    ChatCompletionRequestMessageContentPartImageURL
)
from tldw_Server_API.app.core.Chat.Chat_Functions import (
    ChatAuthenticationError, ChatRateLimitError, ChatBadRequestError,
    ChatConfigurationError, ChatProviderError, ChatAPIError
)
from tldw_Server_API.app.core.Chat.prompt_template_manager import PromptTemplate, DEFAULT_RAW_PASSTHROUGH_TEMPLATE
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user  # Import your actual DB deps
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user


# Fixture for TestClient
@pytest.fixture(scope="function")
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture
def valid_auth_token() -> str:
    token = os.getenv("TEST_AUTH_TOKEN", "test-user-token-if-auth-is-mocked")
    if not token and not os.getenv("CI"):
        pytest.skip("TEST_AUTH_TOKEN not set.")
    return token

# --- Provider lists and helpers (same as your previous standalone version) ---
try:
    from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import API_KEYS as APP_API_KEYS_FROM_SCHEMA
    from tldw_Server_API.app.core.Chat.Chat_Functions import API_CALL_HANDLERS as APP_API_CALL_HANDLERS
    ALL_CONFIGURED_PROVIDERS_FROM_APP = list(APP_API_CALL_HANDLERS.keys())
except ImportError:
    APP_API_KEYS_FROM_SCHEMA = {}
    ALL_CONFIGURED_PROVIDERS_FROM_APP = []

def get_commercial_providers_with_keys_integration():
    # ... (same as before)
    potentially_commercial = ["openai", "anthropic", "cohere", "groq", "openrouter", "deepseek", "mistral", "google", "huggingface"]
    return [p for p in potentially_commercial if p in ALL_CONFIGURED_PROVIDERS_FROM_APP and APP_API_KEYS_FROM_SCHEMA.get(p)]

def get_local_providers_integration():
    # ... (same as before)
    local_provider_names = ["llama.cpp", "kobold", "ooba", "tabbyapi", "vllm", "local-llm", "ollama", "aphrodite", "custom-openai-api", "custom-openai-api-2"]
    return [p for p in local_provider_names if p in ALL_CONFIGURED_PROVIDERS_FROM_APP]


# Test data using your schema
INTEGRATION_MESSAGES_NO_SYS_SCHEMA = [
    ChatCompletionUserMessageParam(role="user", content="Explain the theory of relativity simply.")
]
INTEGRATION_MESSAGES_WITH_SYS_SCHEMA = [
    ChatCompletionSystemMessageParam(role="system", content="You are Albert Einstein. Explain things from your perspective."),
    ChatCompletionUserMessageParam(role="user", content="Explain the theory of relativity simply.")
]
STREAM_INTEGRATION_MESSAGES_SCHEMA = [
    ChatCompletionUserMessageParam(role="user", content="Stream a very short poem about space. Max 3 lines.")
]

COMMERCIAL_PROVIDERS_FOR_TEST = get_commercial_providers_with_keys_integration()
LOCAL_PROVIDERS_FOR_TEST = get_local_providers_integration()

# Fixture to mock DB dependencies for integration tests
@pytest.fixture
def default_chat_request_data():
    return ChatCompletionRequest(
        model=DEFAULT_MODEL,
        messages=DEFAULT_USER_MESSAGES_SCHEMA
    )


# Mocks for DB dependencies
@pytest.fixture
def mock_chat_db():
    db_mock = MagicMock(spec=CharactersRAGDB)
    db_mock.get_character_card_by_id.return_value = None
    return db_mock


@pytest.fixture
def mock_media_db():
    return MagicMock()


# --- Unit Tests for the Endpoint ---

@pytest.mark.unit
@patch("tldw_Server_API.app.api.v1.endpoints.chat.chat_api_call")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.apply_template_to_string")
def test_create_chat_completion_no_template(
        mock_apply_template, mock_load_template, mock_chat_api_call,
        client, valid_auth_token, mock_media_db, mock_chat_db, default_chat_request_data
):
    mock_load_template.return_value = None
    mock_apply_template.side_effect = lambda template_str, data: data.get("message_content", data.get(
        "original_system_message_from_request", ""))
    mock_response_data = {"id": "chatcmpl-no-template",
                          "choices": [{"message": {"role": "assistant", "content": "Raw response"}}]}
    mock_chat_api_call.return_value = mock_response_data

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    request_data_dict = default_chat_request_data.model_dump()
    response = client.post("/api/v1/chat/completions", json=request_data_dict, headers={"token": valid_auth_token})

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == mock_response_data
    mock_chat_api_call.assert_called_once()
    called_kwargs = mock_chat_api_call.call_args.kwargs
    assert called_kwargs.get("system_message") is None

    # Compare content of messages payload
    expected_payload_content = [msg.model_dump(exclude_none=True) for msg in DEFAULT_USER_MESSAGES_SCHEMA]
    # The endpoint sends messages_payload as list of dicts to chat_api_call
    actual_payload_content = called_kwargs["messages_payload"]
    assert actual_payload_content[0]["content"] == expected_payload_content[0][
        "content"]  # Assuming single message here

    app.dependency_overrides = {}


@pytest.mark.unit
@patch("tldw_Server_API.app.api.v1.endpoints.chat.chat_api_call")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.apply_template_to_string")
def test_create_chat_completion_with_simple_template_no_char(
        mock_apply_template, mock_load_template, mock_chat_api_call,
        client, valid_auth_token, mock_media_db, mock_chat_db
):
    template_name = "simple_test_template"
    mock_template = PromptTemplate(
        name=template_name,
        system_message_template="System: {original_system_message_from_request} Be brief.",
        user_message_content_template="User asks: {message_content}",
        assistant_message_content_template="Assistant said: {message_content}"
    )
    mock_load_template.return_value = mock_template

    def apply_side_effect(template_str, data):
        if template_str == mock_template.system_message_template: return f"System: {data.get('original_system_message_from_request', '')} Be brief."
        if template_str == mock_template.user_message_content_template: return f"User asks: {data.get('message_content', '')}"
        return data.get('message_content', '')

    mock_apply_template.side_effect = apply_side_effect
    mock_chat_api_call.return_value = {"id": "chatcmpl-simple-template",
                                       "choices": [{"message": {"role": "assistant", "content": "Templated response"}}]}

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    messages = [
        ChatCompletionSystemMessageParam(role="system", content="Original sys info."),
        ChatCompletionUserMessageParam(role="user", content="What is the weather?")
    ]
    request_obj = ChatCompletionRequest(model="test-model", messages=messages, prompt_template_name=template_name)

    response = client.post("/api/v1/chat/completions", json=request_obj.model_dump(),
                           headers={"token": valid_auth_token})

    assert response.status_code == status.HTTP_200_OK
    mock_load_template.assert_called_once_with(template_name)
    mock_chat_api_call.assert_called_once()
    called_kwargs = mock_chat_api_call.call_args.kwargs
    assert called_kwargs["system_message"] == "System: Original sys info. Be brief."
    assert len(called_kwargs["messages_payload"]) == 1
    assert called_kwargs["messages_payload"][0]["role"] == "user"
    assert called_kwargs["messages_payload"][0]["content"] == "User asks: What is the weather?"
    mock_chat_db.get_character_card_by_id.assert_not_called()
    app.dependency_overrides = {}


@pytest.mark.unit
@patch("tldw_Server_API.app.api.v1.endpoints.chat.chat_api_call")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.apply_template_to_string")
def test_create_chat_completion_with_template_and_character(
        mock_apply_template, mock_load_template, mock_chat_api_call,
        client, valid_auth_token, mock_media_db, mock_chat_db
):
    template_name = "char_template"
    char_id_to_test = "char123"
    mock_char_template = PromptTemplate(
        name=template_name,
        system_message_template="System for {char_name}: {character_system_prompt}. User says: {original_system_message_from_request}",
        user_message_content_template="{char_name} is asked: {message_content}"
    )
    mock_load_template.return_value = mock_char_template
    mock_character_data = {"id": char_id_to_test, "name": "TestChar", "system_prompt": "Be a pirate!"}
    mock_chat_db.get_character_card_by_id.return_value = mock_character_data

    def apply_side_effect_char(template_str, data):
        if template_str == mock_char_template.system_message_template:
            return f"System for {data.get('char_name')}: {data.get('character_system_prompt')}. User says: {data.get('original_system_message_from_request', '')}"
        elif template_str == mock_char_template.user_message_content_template:
            return f"{data.get('char_name')} is asked: {data.get('message_content', '')}"
        return data.get('message_content', '')

    mock_apply_template.side_effect = apply_side_effect_char
    mock_chat_api_call.return_value = {"id": "chatcmpl-char-template"}

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    messages = [
        ChatCompletionSystemMessageParam(role="system", content="Player hint: he likes gold."),
        ChatCompletionUserMessageParam(role="user", content="Where's the treasure?")
    ]
    request_obj = ChatCompletionRequest(
        model="test-model", messages=messages,
        prompt_template_name=template_name, character_id=char_id_to_test
    )

    response = client.post("/api/v1/chat/completions", json=request_obj.model_dump(),
                           headers={"token": valid_auth_token})

    assert response.status_code == status.HTTP_200_OK
    mock_load_template.assert_called_once_with(template_name)
    mock_chat_db.get_character_card_by_id.assert_called_once_with(char_id_to_test)
    mock_chat_api_call.assert_called_once()
    called_kwargs = mock_chat_api_call.call_args.kwargs
    assert called_kwargs[
               "system_message"] == "System for TestChar: Be a pirate!. User says: Player hint: he likes gold."
    assert called_kwargs["messages_payload"][0]["content"] == "TestChar is asked: Where's the treasure?"
    app.dependency_overrides = {}


@pytest.mark.unit
@patch("tldw_Server_API.app.api.v1.endpoints.chat.chat_api_call")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.apply_template_to_string")
def test_template_applied_to_multimodal_user_message(
        mock_apply_template, mock_load_template, mock_chat_api_call,
        client, valid_auth_token, mock_media_db, mock_chat_db
):
    template_name = "multimodal_template"
    mock_template = PromptTemplate(
        name=template_name,
        system_message_template="System: {original_system_message_from_request}",
        user_message_content_template="User content with template: {message_content}"
    )
    mock_load_template.return_value = mock_template
    mock_apply_template.side_effect = lambda template_str, data: template_str.replace("{message_content}",
                                                                                      data.get("message_content", "")) \
        .replace("{original_system_message_from_request}", data.get("original_system_message_from_request", ""))
    mock_chat_api_call.return_value = {"id": "res"}
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    user_content_parts_schema = [
        ChatCompletionRequestMessageContentPartText(type="text", text="Describe this image."),
        ChatCompletionRequestMessageContentPartImage(type="image_url",
                                                     image_url=ChatCompletionRequestMessageContentPartImageURL(
                                                         url="data:image/png;base64,fakedata"))
    ]
    messages = [ChatCompletionUserMessageParam(role="user", content=user_content_parts_schema)]
    request_obj = ChatCompletionRequest(model="test-vision-model", messages=messages,
                                        prompt_template_name=template_name)

    client.post("/api/v1/chat/completions", json=request_obj.model_dump(exclude_none=True),
                headers={"token": valid_auth_token})

    mock_chat_api_call.assert_called_once()
    called_kwargs = mock_chat_api_call.call_args.kwargs
    templated_payload = called_kwargs["messages_payload"]
    assert len(templated_payload) == 1
    user_msg_content_final = templated_payload[0]["content"]  # This will be a list of dicts now
    assert isinstance(user_msg_content_final, list)
    assert len(user_msg_content_final) == 2

    # Check templated text part
    text_part_found = False
    for part in user_msg_content_final:
        if part["type"] == "text":
            assert part["text"] == "User content with template: Describe this image."
            text_part_found = True
    assert text_part_found
    # Check image part remains untouched
    assert {"type": "image_url",
            "image_url": {"url": "data:image/png;base64,fakedata", "detail": "auto"}} in user_msg_content_final
    app.dependency_overrides = {}


# Keep other unit tests (streaming, error handling, API keys)
# Ensure they use the new schema for messages if they construct request_data
# and use the mock_db_dependencies

# Example adaptation for error handling test
@pytest.mark.unit
@patch("tldw_Server_API.app.api.v1.endpoints.chat.chat_api_call")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.apply_template_to_string")
@pytest.mark.parametrize("error_type, expected_status, error_message_detail", [
    (ChatAuthenticationError(provider="test", message="Auth failed"), status.HTTP_401_UNAUTHORIZED, "Auth failed"),
    (ChatRateLimitError(provider="test", message="Rate limit"), status.HTTP_429_TOO_MANY_REQUESTS, "Rate limit"),
    (ChatBadRequestError(provider="test", message="Bad request"), status.HTTP_400_BAD_REQUEST, "Bad request"),
    (ChatConfigurationError(provider="test", message="Config error"), status.HTTP_500_INTERNAL_SERVER_ERROR,
     "Config error"),
    (ChatProviderError(provider="test", message="Provider issue", status_code=503), status.HTTP_503_SERVICE_UNAVAILABLE,
     "Provider issue"),
])
def test_chat_api_call_exception_handling_adapted(
        mock_apply_template, mock_load_template, mock_chat_api_call,
        client, valid_auth_token, mock_media_db, mock_chat_db,  # Use DB mocks
        default_chat_request_data,  # Use fixture for request data
        error_type, expected_status, error_message_detail
):
    mock_load_template.return_value = DEFAULT_RAW_PASSTHROUGH_TEMPLATE
    mock_apply_template.side_effect = lambda template_str, data: data.get("message_content", data.get(
        "original_system_message_from_request", ""))
    mock_chat_api_call.side_effect = error_type

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    response = client.post(
        "/api/v1/chat/completions",
        json=default_chat_request_data.model_dump(),
        headers={"token": valid_auth_token}
    )
    assert response.status_code == expected_status
    response_detail = response.json().get("detail", "")
    assert error_message_detail.lower() in response_detail.lower()
    app.dependency_overrides = {}