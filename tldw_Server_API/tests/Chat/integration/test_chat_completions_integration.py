# tests/integration/api/v1/test_chat_completions_integration.py
import pytest

pytestmark = pytest.mark.unit
import os
import json
from fastapi import status
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv
from pathlib import Path

from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAuthenticationError

# Load environment variables from the canonical Config_Files/.env
try:
    _tests_file = Path(__file__).resolve()
    _project_root = _tests_file.parents[3]  # repo_root/tldw_Server_API
    _env_path = _project_root / "Config_Files" / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=str(_env_path), override=False)
except Exception:
    _ = None

# Import your FastAPI app instance
from tldw_Server_API.app.main import app

# Import your actual schema definitions for constructing test requests
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
    # If you test with multimodal content in integration tests:
    # ChatCompletionRequestMessageContentPartText,
    # ChatCompletionRequestMessageContentPartImage,
    # ChatCompletionRequestMessageContentPartImageURL
)

# For mocking DB dependencies if your endpoint uses them
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.core.Chat.prompt_template_manager import PromptTemplate  # For templating test
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
from tldw_Server_API.app.core.Usage.pricing_catalog import (
    list_provider_models,
    get_pricing_catalog,
)


# --- Fixtures defined locally in this file ---
API_BEARER = os.getenv("API_BEARER", "test-api-key-12345")


@pytest.fixture
def mock_user():
    """Create a mock user for testing."""
    import datetime

    return User(
        id=1,
        username="test_user",
        email="test@example.com",
        is_active=True,
        is_admin=False,
        created_at=datetime.datetime.now(),
        updated_at=datetime.datetime.now(),
    )


@pytest.fixture(autouse=True)
def setup_auth_override(mock_user):
    """Automatically override authentication for all tests based on AUTH_MODE."""
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings

    settings = get_settings()

    # Only override authentication in multi-user mode
    if settings.AUTH_MODE == "multi_user":
        # In multi-user mode, we need to mock the entire authentication flow
        # The simplest approach is to override get_request_user to return mock user directly
        async def mock_get_request_user(api_key=None, token=None):
            return mock_user

        app.dependency_overrides[get_request_user] = mock_get_request_user
        yield
        # Clean up after test
        app.dependency_overrides.pop(get_request_user, None)
    else:
        # In single-user mode, no override needed
        yield


# Helper function to make requests with CSRF token
def make_request_with_csrf(client, method, url, headers=None, **kwargs):
    """Helper to make requests with CSRF token included"""
    if headers is None:
        headers = {}

    # Get CSRF token from cookies if not already set
    if not hasattr(client, "csrf_token"):
        # Make a GET request to get CSRF token
        response = client.get("/api/v1/health")
        csrf_token = response.cookies.get("csrf_token", "")
        client.csrf_token = csrf_token

    # Add CSRF token to headers
    headers = dict(headers)
    # In single-user mode, ensure X-API-KEY is present so AuthNZ deps pass
    try:
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings

        settings = get_settings()
        if settings.AUTH_MODE == "single_user" and "X-API-KEY" not in headers:
            api_key = settings.SINGLE_USER_API_KEY or os.getenv("SINGLE_USER_TEST_API_KEY", "test-api-key-12345")
            if api_key:
                headers["X-API-KEY"] = api_key
    except Exception:
        _ = None

    headers["X-CSRF-Token"] = getattr(client, "csrf_token", "")

    # Make the request
    method_func = getattr(client, method.lower())
    return method_func(url, headers=headers, **kwargs)


@pytest.fixture(scope="function")
def client():
    """Yields a TestClient instance for making requests to the app."""
    with TestClient(app) as c:
        # Get a CSRF token by making a GET request first
        response = c.get("/api/v1/health")
        csrf_token = response.cookies.get("csrf_token", "")

        # Store the token in the client for use in tests
        c.csrf_token = csrf_token
        c.cookies = {"csrf_token": csrf_token}

        # Add helper method to client
        c.post_with_csrf = lambda url, **kwargs: make_request_with_csrf(c, "POST", url, **kwargs)

        yield c


@pytest.fixture
def valid_auth_token() -> str:
    """Generate appropriate auth token based on current AUTH_MODE."""
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings

    settings = get_settings()

    if settings.AUTH_MODE == "multi_user":
        # In multi-user mode, we need a proper JWT token
        # For testing, we'll create a mock JWT token using the actual JWT secret
        import jwt
        import datetime

        # Use the actual JWT secret from settings
        secret_key = settings.JWT_SECRET_KEY or os.getenv("JWT_SECRET_KEY", "test-secret-key")

        payload = {
            "sub": "1",  # User ID as string
            "username": "test_user",
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1),
            "iat": datetime.datetime.utcnow(),
            "type": "access",
        }
        test_token = jwt.encode(payload, secret_key, algorithm=settings.JWT_ALGORITHM)
        return f"Bearer {test_token}"
    else:
        # In single-user mode use the configured API key and return it with Bearer prefix
        raw_key = settings.SINGLE_USER_API_KEY or os.getenv("SINGLE_USER_API_KEY")
        if not raw_key:
            raw_key = os.getenv("API_BEARER")
            if raw_key and raw_key.lower().startswith("bearer "):
                return raw_key.strip()
        if not raw_key:
            raw_key = API_BEARER
        raw_key = raw_key.strip()
        if raw_key.lower().startswith("bearer "):
            return raw_key
        return f"Bearer {raw_key}"


# --- Provider lists and helpers defined locally for this test file ---
try:
    # These are used to determine which providers are configured and have keys.
    # The actual APP_API_KEYS should be loaded by your application's schema/config logic.
    from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import API_KEYS as APP_API_KEYS_FROM_SCHEMA
    from tldw_Server_API.app.core.LLM_Calls.adapter_registry import get_registry

    ALL_CONFIGURED_PROVIDERS_FROM_APP = get_registry().list_providers()
except ImportError:
    APP_API_KEYS_FROM_SCHEMA = {}
    ALL_CONFIGURED_PROVIDERS_FROM_APP = []
    print(
        "Warning: Could not import APP_API_KEYS_FROM_SCHEMA or adapter registry for integration test parametrization."
    )


def get_commercial_providers_with_keys_integration():
    """
    Returns a list of commercial providers for which API keys are actually set
    and non-empty, as understood by the application's schema.

    Only gate by presence of provider API keys. No other qualifiers.
    """
    potentially_commercial = [
        "openai",
        "anthropic",
        "cohere",
        "groq",
        "openrouter",
        "deepseek",
        "mistral",
        "google",
        "huggingface",
        "qwen",
        # Add others that are external and need keys from your config
    ]
    # Check against keys actually loaded by the app's config (via schemas)
    providers_with_keys = []
    for p in potentially_commercial:
        # Only require the presence of a non-empty key
        api_key = APP_API_KEYS_FROM_SCHEMA.get(p)
        if api_key:
            providers_with_keys.append(p)
    return providers_with_keys


def get_local_providers_integration():
    """
    Returns a list of providers considered "local" or that might use URLs/config
    instead of globally managed API keys.
    """
    local_provider_names = [
        "llama.cpp",
        "kobold",
        "ooba",
        "tabbyapi",
        "vllm",
        "local-llm",
        "ollama",
        "aphrodite",
        "custom-openai-api",
        "custom-openai-api-2",
        # Add other local providers from your config
    ]
    return [p for p in local_provider_names if p in ALL_CONFIGURED_PROVIDERS_FROM_APP]


# Test data using your actual Pydantic schema models
INTEGRATION_MESSAGES_NO_SYS_SCHEMA = [
    ChatCompletionUserMessageParam(role="user", content="Explain the theory of relativity simply in one sentence.")
]
INTEGRATION_MESSAGES_WITH_SYS_SCHEMA = [
    ChatCompletionSystemMessageParam(
        role="system", content="You are Albert Einstein. Explain things from your perspective, but keep it brief."
    ),
    ChatCompletionUserMessageParam(role="user", content="Explain the theory of relativity simply."),
]
STREAM_INTEGRATION_MESSAGES_SCHEMA = [  # For streaming tests
    ChatCompletionUserMessageParam(role="user", content="Stream a very short poem about a star. Max 3 lines.")
]

COMMERCIAL_PROVIDERS_FOR_TEST = get_commercial_providers_with_keys_integration()


# --- Model resolution via pricing catalog ---
def _env_override_for_provider(provider_name: str) -> str:
    """Return env override value if set, e.g., OPENAI_TEST_MODEL for 'openai'."""
    if not provider_name:
        return ""
    # Common explicit envs used in CI/setup docs
    explicit = {
        "openai": "OPENAI_TEST_MODEL",
        "anthropic": "ANTHROPIC_TEST_MODEL",
        "cohere": "COHERE_TEST_MODEL",
        "groq": "GROQ_TEST_MODEL",
        "openrouter": "OPENROUTER_TEST_MODEL",
        "deepseek": "DEEPSEEK_TEST_MODEL",
        "mistral": "MISTRAL_TEST_MODEL",
        "google": "GOOGLE_TEST_MODEL",
        "huggingface": "HF_TEST_MODEL",
        "qwen": "QWEN_TEST_MODEL",
    }
    key = explicit.get(provider_name)
    if key and os.getenv(key):
        return os.getenv(key, "")
    # Fallback generic pattern: PROVIDERNAME_TEST_MODEL (alnum+underscore)
    generic = f"{''.join(ch if ch.isalnum() else '_' for ch in provider_name).upper()}_TEST_MODEL"
    return os.getenv(generic, "")


def _chat_default_model_for_provider(provider_name: str) -> str:
    if not provider_name:
        return ""
    normalized = provider_name.replace(".", "_").replace("-", "_")
    env_key = f"DEFAULT_MODEL_{normalized.upper()}"
    env_val = os.getenv(env_key, "")
    if env_val:
        return env_val
    try:
        from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint

        cfg = getattr(chat_endpoint, "_chat_config", {}) or {}
        cfg_val = cfg.get(f"default_model_{normalized.lower()}")
        return cfg_val or ""
    except Exception:
        return ""


def resolve_test_model_from_catalog(provider_name: str) -> str:
    """Pick a valid chat model for the provider from model_pricing.json.

    Selection strategy:
    - If env override like OPENAI_TEST_MODEL is set, use it.
    - Otherwise, list models from the pricing catalog and filter out embeddings.
    - Choose the cheapest model by (prompt+completion) price.
    - If none available, return a generic fallback string.
    """
    override = _env_override_for_provider(provider_name)
    if override:
        return override
    default_model = _chat_default_model_for_provider(provider_name)
    if default_model:
        return default_model

    if provider_name == "huggingface":
        return "ServiceNow-AI/Apriel-1.6-15b-Thinker:together"

    # Load from pricing catalog
    try:
        models = list_provider_models(provider_name) or []
    except Exception:
        models = []

    # Filter out non-chat models (embeddings, obvious non-chat ids)
    def is_chat_model(name: str) -> bool:
        n = (name or "").lower()
        if "embed" in n or "embedding" in n:
            return False
        if "/embeddings" in n:
            return False
        return True

    chat_models = [m for m in models if is_chat_model(m)]
    if not chat_models:
        return ""

    # Special-case: huggingface catalog may publish a placeholder 'default' only.
    if provider_name == "huggingface" and chat_models == ["default"]:
        # Return empty to let server-side config decide (or require env override)
        return ""

    # Prefer a stable chat model for DeepSeek regardless of pricing sort.
    # DeepSeek's coder model can behave differently; use chat for general tests.
    if provider_name == "deepseek":
        for m in chat_models:
            if (m or "").lower().strip() == "deepseek-chat":
                return "deepseek-chat"

    # Prefer concrete ids (-latest, dated suffixes, context sizes) when available
    def _looks_concrete(mid: str) -> bool:
        ml = (mid or "").lower()
        return any(
            tok in ml
            for tok in ("-latest", "2023", "2024", "2025", "-8192", "-instruct", "-versatile", "-instant", ":")
        )

    concrete = [m for m in chat_models if _looks_concrete(m)] or chat_models

    # Provider-specific canonicalization of popular aliases -> API ids
    def _canonicalize(p: str, m: str) -> str:
        pl = (p or "").lower()
        ml = (m or "").strip()
        if pl == "anthropic":
            mll = ml.lower()
            if "sonnet" in mll:
                return "claude-3-5-sonnet-latest"
            if "haiku" in mll:
                return "claude-3-haiku-20240307"
            if "opus" in mll:
                return "claude-3-opus-20240229"
        if pl == "groq":
            mll = ml.lower()
            # Prefer current Groq ids
            if mll in {"llama3-8b", "llama-3-8b"}:
                return "llama-3.1-8b-instant"
            if mll in {"llama3-70b", "llama-3-70b"}:
                return "llama-3.1-70b-versatile"
        if pl == "openrouter":
            # Prefer a broadly-supported OpenRouter id when not overridden
            override = os.getenv("OPENROUTER_TEST_MODEL")
            if override:
                return override
            if ml.lower().startswith("gpt-"):
                return ml
            return "gpt-4o"
        return ml

    # Choose the cheapest (prompt+completion) using catalog rates, after canonicalization
    try:
        catalog = get_pricing_catalog()

        def total_cost(mid: str) -> float:
            # Compare prices using catalog keys as-is
            pr, cr, _ = catalog.get_rates(provider_name, mid)
            return float(pr or 0.0) + float(cr or 0.0)

        concrete.sort(key=lambda mm: (total_cost(mm), mm))
        return _canonicalize(provider_name, concrete[0])
    except Exception:
        # Fallback to first model if rates unavailable
        return _canonicalize(provider_name, concrete[0])


def _configured_default_model(provider_name: str) -> str:
    return _chat_default_model_for_provider(provider_name)


def _provider_base_url_override(provider_name: str) -> str:
    try:
        from tldw_Server_API.app.core.LLM_Calls.adapter_utils import ensure_app_config, resolve_provider_section

        cfg = ensure_app_config(None)
        section = resolve_provider_section(provider_name)
        if section:
            base = (cfg.get(section) or {}).get("api_base_url")
            if isinstance(base, str) and base.strip():
                return base.strip()
    except Exception:
        _ = None
    env_map = {
        "anthropic": "ANTHROPIC_BASE_URL",
        "google": "GOOGLE_GEMINI_BASE_URL",
    }
    env_key = env_map.get(provider_name)
    if env_key:
        return os.getenv(env_key, "") or ""
    return ""


# Fixture to mock DB dependencies for integration tests if the endpoint uses them
# In tldw_Server_API/tests/Chat/test_chat_completions_integration.py
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import DEFAULT_CHARACTER_NAME  # Import


@pytest.fixture
def mock_db_dependencies_for_integration():
    mock_media_db_inst = MagicMock()
    mock_chat_db_inst = MagicMock(spec=CharactersRAGDB)

    # --- Configure mock_chat_db_inst ---
    # For default character loading by name:
    default_char_card_data = {
        "id": 9999,  # Use an int to mirror real DB rows and avoid type quirks
        "name": DEFAULT_CHARACTER_NAME,
        "system_prompt": "This is a mock default system prompt for integration tests.",
        # Add other fields the endpoint might access from the default character
        # e.g., client_id if it's directly on the card (though less likely)
    }

    def mock_get_character_card_by_name(name_or_id):
        if name_or_id == DEFAULT_CHARACTER_NAME:
            return default_char_card_data
        return None  # For any other name

    mock_chat_db_inst.get_character_card_by_name.side_effect = mock_get_character_card_by_name
    mock_chat_db_inst.get_character_card_by_id.return_value = None  # Keep this for by_id lookups unless a test needs it

    # For new conversation creation
    mock_chat_db_inst.add_conversation.return_value = "integration_mock_conv_id"

    # For saving messages
    mock_chat_db_inst.add_message.return_value = "integration_mock_msg_id"

    # <<< Crucial: Add client_id for integration tests too! >>>
    mock_chat_db_inst.client_id = "test_client_integration"
    # <<< Crucial: Ensure 'get_conversation_by_id' returns something sensible or None based on test needs >>>
    # For cases where a conversation ID *is* provided in the request and is expected to exist or not.
    # For these "no template" tests, usually no conv_id is provided initially.
    mock_chat_db_inst.get_conversation_by_id.return_value = None  # Default to not found

    # --- Dependency Override ---
    original_media_db_dep = app.dependency_overrides.get(get_media_db_for_user)
    original_chat_db_dep = app.dependency_overrides.get(get_chacha_db_for_user)

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db_inst
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db_inst

    yield mock_media_db_inst, mock_chat_db_inst

    # --- Restore ---
    if original_media_db_dep:
        app.dependency_overrides[get_media_db_for_user] = original_media_db_dep
    elif get_media_db_for_user in app.dependency_overrides:
        del app.dependency_overrides[get_media_db_for_user]

    if original_chat_db_dep:
        app.dependency_overrides[get_chacha_db_for_user] = original_chat_db_dep
    elif get_chacha_db_for_user in app.dependency_overrides:
        del app.dependency_overrides[get_chacha_db_for_user]


# --- Commercial Provider Tests ---
@pytest.mark.external_api  # Custom marker (register in pytest.ini or pyproject.toml)
@pytest.mark.integration
@pytest.mark.parametrize("provider_name", COMMERCIAL_PROVIDERS_FOR_TEST)
@pytest.mark.skipif(
    not COMMERCIAL_PROVIDERS_FOR_TEST, reason="No commercial providers with API keys configured for integration tests."
)
def test_commercial_provider_non_streaming_no_template(
    client, provider_name, valid_auth_token, mock_db_dependencies_for_integration
):
    # This test uses the DEFAULT_RAW_PASSTHROUGH_TEMPLATE because prompt_template_name is None

    selected_model = resolve_test_model_from_catalog(provider_name)
    if not selected_model and not _configured_default_model(provider_name):
        pytest.skip(
            f"No default model configured for {provider_name}; set {provider_name.upper()}_TEST_MODEL or config."
        )
    base_url_override = _provider_base_url_override(provider_name)
    if provider_name != "openai" and "api.openai.com" in base_url_override:
        pytest.skip(f"{provider_name} base_url overridden to OpenAI; skipping external provider test.")
    request_body = {
        "api_provider": provider_name,
        "messages": [msg.model_dump(exclude_none=True) for msg in INTEGRATION_MESSAGES_NO_SYS_SCHEMA],
        "temperature": 0.7,
        "stream": False,
        "prompt_template_name": None,  # Explicitly no template, should use default passthrough
    }
    if selected_model:
        request_body["model"] = selected_model
    if provider_name == "anthropic":  # Anthropic (Claude) often requires max_tokens
        request_body["max_tokens"] = 200  # Adjusted for potentially longer explanations

    print(
        f"\nTesting NON-STREAMING (no template) with {provider_name} using model {request_body.get('model', '<server-default>')}"
    )
    response = client.post_with_csrf("/api/v1/chat/completions", json=request_body, headers={"Authorization": valid_auth_token})
    # XFAIL policy for known upstream/provider issues (stabilize CI)
    if provider_name == "cohere" and (response.status_code in (400, 404) or response.status_code >= 500):
        pytest.xfail(f"Cohere upstream not stable for /v1/chat (status {response.status_code}). {response.text[:180]}")
    if (
        provider_name in {"deepseek", "anthropic", "openrouter", "google", "huggingface"}
        and response.status_code >= 500
    ):
        pytest.xfail(f"{provider_name} returned 5xx on non-stream call. {response.text[:180]}")

    assert response.status_code == status.HTTP_200_OK, f"Provider {provider_name} failed: {response.text}"
    data = response.json()

    assert data.get("id") is not None, f"Missing 'id' for {provider_name}"
    assert (
        data.get("choices") and isinstance(data["choices"], list) and len(data["choices"]) > 0
    ), f"Missing 'choices' for {provider_name}"
    message = data["choices"][0].get("message", {})
    assert message.get("role") == "assistant", f"Incorrect role for {provider_name}"
    content = message.get("content")
    assert isinstance(content, str) and len(content) > 5, f"Invalid content for {provider_name}: '{content}'"
    print(f"Response from {provider_name} (no template): {content[:80]}...")


@pytest.mark.external_api
@pytest.mark.integration
@pytest.mark.parametrize("provider_name", COMMERCIAL_PROVIDERS_FOR_TEST)
@pytest.mark.skipif(
    not COMMERCIAL_PROVIDERS_FOR_TEST, reason="No commercial providers with API keys configured for streaming tests."
)
def test_commercial_provider_streaming_no_template(
    client, provider_name, valid_auth_token, mock_db_dependencies_for_integration
):
    selected_model = resolve_test_model_from_catalog(provider_name)
    if not selected_model and not _configured_default_model(provider_name):
        pytest.skip(
            f"No default model configured for {provider_name}; set {provider_name.upper()}_TEST_MODEL or config."
        )
    base_url_override = _provider_base_url_override(provider_name)
    if provider_name != "openai" and "api.openai.com" in base_url_override:
        pytest.skip(f"{provider_name} base_url overridden to OpenAI; skipping external provider test.")
    request_body = {
        "api_provider": provider_name,
        "messages": [msg.model_dump(exclude_none=True) for msg in STREAM_INTEGRATION_MESSAGES_SCHEMA],
        "temperature": 0.7,
        "stream": True,
        "prompt_template_name": None,
    }
    if selected_model:
        request_body["model"] = selected_model
    if provider_name == "anthropic":
        request_body["max_tokens"] = 300

    print(
        f"\nTesting STREAMING (no template) with {provider_name} using model {request_body.get('model', '<server-default>')}"
    )

    # Use streaming context to avoid buffering entire body and potential hangs
    headers = {"Authorization": valid_auth_token, "X-CSRF-Token": getattr(client, "csrf_token", "")}
    full_content = ""
    received_done = False
    raw_stream_text_for_debug = ""

    with client.stream("POST", "/api/v1/chat/completions", json=request_body, headers=headers) as response:
        if response.status_code != status.HTTP_200_OK:
            response.read()
            pytest.fail(f"Provider {provider_name} streaming pre-check failed: {response.text}")
        assert response.status_code == status.HTTP_200_OK
        assert "text/event-stream" in response.headers.get("content-type", "").lower()

        try:
            for line in response.iter_lines():
                if not line:
                    continue
                s = line.strip()
                raw_stream_text_for_debug += s + "\n"
                if s.lower() == "data: [done]":
                    received_done = True
                    break
                if not s.startswith("data:"):
                    continue
                payload = s[len("data:") :].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    j = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = j.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    # Prefer content, but tolerate tool_calls-only deltas by counting them
                    if delta.get("content"):
                        full_content += delta["content"]
                    elif delta.get("tool_calls"):
                        # Treat presence of tool_calls as non-empty content for assertion purposes
                        try:
                            tc = delta.get("tool_calls") or []
                            full_content += f"[tool_calls:{len(tc)}]"
                        except Exception:
                            # Ignore formatting errors
                            _ = None
                    if choices[0].get("finish_reason") == "stop":
                        # Wait for [DONE], but we already have content
                        pass
        except Exception as e:
            print(f"Raw stream for {provider_name} before error:\n{raw_stream_text_for_debug}")
            pytest.fail(f"Error consuming stream for {provider_name}: {e}")

    # XFAIL policy for known upstream/provider stream issues
    if provider_name in {"cohere", "deepseek", "anthropic", "openrouter", "google", "huggingface"}:
        if not received_done or len(full_content) == 0:
            pytest.xfail(
                f"{provider_name} streaming unstable for this account/env (received_done={received_done}, len={len(full_content)})."
            )

    assert (
        received_done
    ), f"Stream for {provider_name} did not finish correctly. Last 500 chars:\n{raw_stream_text_for_debug[-500:]}"
    assert (
        len(full_content) > 0
    ), f"Streamed content for {provider_name} was too short or empty. Received: '{full_content}'"
    print(f"Streamed response from {provider_name} (no template): {full_content[:80]}...")


# --- Integration Test with Templating (Simplified - Mocks DB interaction) ---
@pytest.mark.external_api
@pytest.mark.integration
def test_commercial_provider_with_template_and_char_data_openai_integration(
    client, valid_auth_token, mock_db_dependencies_for_integration
):
    provider_name = "openai"
    if provider_name not in COMMERCIAL_PROVIDERS_FOR_TEST:
        pytest.skip(f"{provider_name} not configured with API key for this templating test.")

    # Unpack the chat_db mock to configure its return values
    _, mock_chat_db_inst = mock_db_dependencies_for_integration

    test_template_name = "pirate_speak_template"
    # This template would normally be loaded from a file by `load_template`
    # For this test, we mock `load_template` to return this specific PromptTemplate object.
    test_pirate_template_obj = PromptTemplate(
        name=test_template_name,
        system_message_template="Ye be talkin' to Cap'n {char_name}! {character_system_prompt}. The request mentioned: {original_system_message_from_request}",
        user_message_content_template="Arr, the landlubber {char_name} be askin': {message_content}",
        assistant_message_content_template=None,  # Not used in request path
    )
    test_char_id_for_template = "pirate_blackheart"
    mock_character_data_for_template = {
        "id": test_char_id_for_template,
        "name": "Blackheart",
        "system_prompt": "I only speak in pirate tongues and seek treasure!",
        "personality": "Gruff but fair",
        "description": "A legendary pirate captain.",
        # Add other fields your template might use
    }
    # Configure the mock_chat_db_inst that the endpoint will use
    mock_chat_db_inst.get_character_card_by_id.return_value = None  # Or specific if ID is int

    # Ensure get_character_card_by_name returns the pirate when called with "pirate_blackheart"
    def specific_char_by_name_lookup(name_or_id):
        if name_or_id == test_char_id_for_template:  # "pirate_blackheart"
            return mock_character_data_for_template
        if name_or_id == DEFAULT_CHARACTER_NAME:  # Still handle default if needed elsewhere
            return {"id": 10000, "name": DEFAULT_CHARACTER_NAME, "system_prompt": "Default"}
        return None

    mock_chat_db_inst.get_character_card_by_name.side_effect = specific_char_by_name_lookup

    # Patch `load_template` within the endpoint's module scope
    with patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template", return_value=test_pirate_template_obj):
        chosen_model = resolve_test_model_from_catalog(provider_name)
        request_body = {
            "api_provider": provider_name,
            "messages": [msg.model_dump(exclude_none=True) for msg in INTEGRATION_MESSAGES_WITH_SYS_SCHEMA],
            "prompt_template_name": test_template_name,
            "character_id": test_char_id_for_template,
            "temperature": 0.5,  # Give it some creativity
            "stream": False,
        }
        if chosen_model:
            request_body["model"] = chosen_model

        print(f"\nTesting TEMPLATING with {provider_name} model {request_body.get('model', '<server-default>')}")
        response = client.post_with_csrf(
            "/api/v1/chat/completions", json=request_body, headers={"Authorization": valid_auth_token}
        )

        assert response.status_code == status.HTTP_200_OK, f"{provider_name} with template failed: {response.text}"
        data = response.json()

        assert data.get("id") is not None
        content = data["choices"][0]["message"]["content"]
        assert isinstance(content, str) and len(content) > 5
        # This is a loose check. A better check would be if you mocked chat_api_call
        # and verified the exact templated prompt, but this is an integration test for the LLM response.
        # Note: The mock server doesn't actually process the pirate template, so we skip this assertion for mock
        # In a real test with actual LLM, this would verify the pirate-themed response
        if "mock" not in response.text.lower():
            assert (
                "arr" in content.lower()
                or "matey" in content.lower()
                or "treasure" in content.lower()
                or "cap'n" in content.lower()
            ), f"Response from {provider_name} with pirate template didn't sound pirate-y enough! Got: '{content}'"
        else:
            # For mock server, just verify we got a response
            assert len(content) > 0, f"Empty response from {provider_name}"
        print(f"Templated response from {provider_name} (integration): {content[:100]}...")

        mock_chat_db_inst.get_character_card_by_name.assert_called_once_with(test_char_id_for_template)
        mock_chat_db_inst.get_character_card_by_id.assert_not_called()


# --- Local Provider Tests ---
LOCAL_PROVIDERS_FOR_TEST = get_local_providers_integration()


@pytest.mark.local_llm_service  # Custom marker
@pytest.mark.integration
@pytest.mark.parametrize("provider_name", LOCAL_PROVIDERS_FOR_TEST)
@pytest.mark.skipif(not LOCAL_PROVIDERS_FOR_TEST, reason="No local providers configured for integration tests.")
def test_local_provider_non_streaming_no_template(
    client,
    provider_name,
    valid_auth_token,
    mock_db_dependencies_for_integration,
    request,
    # request is a pytest fixture
):
    # Configuration for local provider URLs (examples, adjust to your env var names)
    config_var_map = {
        "ollama": "OLLAMA_HOST",
        "llama.cpp": "LLAMA_CPP_URL",
        "ooba": "OOBA_URL",
        "vllm": "VLLM_URL",
        "tabbyapi": "TABBYAPI_URL",
        "local-llm": "LOCAL_LLM_API_IP",
        "aphrodite": "APHRODITE_API_IP",
        "kobold": "KOBOLD_API_IP",
        "custom-openai-api": "CUSTOM_OPENAI_API_IP_1",
        "custom-openai-api-2": "CUSTOM_OPENAI_API_IP_2",
    }
    required_env_var = config_var_map.get(provider_name)
    # Skip if the URL env var is not set AND the provider isn't one that might have its URL in APP_API_KEYS_FROM_SCHEMA (less common)
    if required_env_var and not os.getenv(required_env_var):
        # Some "local" providers might have their full URL defined in the API_KEYS/config if they are hosted
        # For truly local dev instances, checking the env var is more typical.
        # This skip logic might need refinement based on how you manage local endpoint URLs.
        is_url_in_app_config = APP_API_KEYS_FROM_SCHEMA.get(provider_name, "").startswith("http")
        if not is_url_in_app_config:
            pytest.skip(
                f"Environment variable like '{required_env_var}' for {provider_name} URL not set, and not found in app config. Skipping."
            )

    model_name = "test-local-model"  # Generic, may be ignored by server if model is pre-loaded
    if provider_name == "ollama":
        model_name = os.getenv("OLLAMA_TEST_MODEL", "phi3:mini")  # Example: use a small model
    if provider_name == "vllm":
        model_name = os.getenv("VLLM_TEST_MODEL", "mistralai/Mistral-7B-Instruct-v0.1")

    request_body = {
        "api_provider": provider_name,
        "model": model_name,
        "messages": [msg.model_dump(exclude_none=True) for msg in INTEGRATION_MESSAGES_NO_SYS_SCHEMA],
        "stream": False,
        "prompt_template_name": None,  # Ensure default passthrough template
    }

    print(f"\nTesting LOCAL NON-STREAMING (no template) with {provider_name} using model {request_body['model']}")
    response = client.post_with_csrf("/api/v1/chat/completions", json=request_body, headers={"Authorization": valid_auth_token})

    assert response.status_code == status.HTTP_200_OK, f"Local provider {provider_name} failed: {response.text}"
    data = response.json()

    assert data.get("id") is not None, f"Missing 'id' for local {provider_name}"
    assert (
        data.get("choices") and isinstance(data["choices"], list) and len(data["choices"]) > 0
    ), f"Missing 'choices' for local {provider_name}"
    message = data["choices"][0].get("message", {})
    assert message.get("role") == "assistant", f"Incorrect role for local {provider_name}"
    content = message.get("content")
    # For local models, especially small ones, the content length can be very short.
    assert isinstance(content, str) and len(content) >= 1, f"Invalid content for local {provider_name}: '{content}'"
    print(f"Response from local {provider_name} (no template): {content[:80]}...")


# --- Invalid Key Test (for a commercial provider that needs a key) ---
@patch.dict("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "invalid_key"})
@patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call")  # Mock the shim
def test_chat_integration_invalid_key_for_commercial_provider_standalone(
    mock_chat_api_call_shim, client, valid_auth_token, mock_db_dependencies_for_integration
):
    provider_to_test_invalid_key = "openai"
    # Simulate that the call to the provider resulted in an auth error
    mock_chat_api_call_shim.side_effect = ChatAuthenticationError(
        provider=provider_to_test_invalid_key, message="Simulated auth error: Invalid API Key"
    )

    request_body = {  # ... minimal request body ...
        "api_provider": provider_to_test_invalid_key,
        "model": "gpt-4o-mini",
        "messages": [msg.model_dump(exclude_none=True) for msg in INTEGRATION_MESSAGES_NO_SYS_SCHEMA],
    }
    response = client.post_with_csrf("/api/v1/chat/completions", json=request_body, headers={"Authorization": valid_auth_token})
    print(f"CI DEBUG: Status Code: {response.status_code}")
    print(f"CI DEBUG: Response Headers: {response.headers}")
    print(f"CI DEBUG: Response Text: {response.text}")
    assert response.status_code == status.HTTP_502_BAD_GATEWAY, (
        f"Expected sanitized upstream auth failure for {provider_to_test_invalid_key}, "
        f"got {response.status_code}. Response: {response.text}"
    )

    detail = response.json()["detail"]
    assert detail["error_code"] == "provider_authentication_failed"
    assert "simulated" not in str(detail).lower()
    print(f"\nInvalid Key Response Detail ({provider_to_test_invalid_key}): {response.json().get('detail')}")


# --- Bad Request Test (e.g., missing messages) ---
@pytest.mark.integration
def test_chat_integration_bad_request_missing_messages_standalone(
    client, valid_auth_token, mock_db_dependencies_for_integration  # Add DB mock for consistency
):
    request_body = {
        "api_provider": "openai",  # Could be any provider
        "model": "test-model",
        # "messages" field is intentionally missing
    }
    response = client.post_with_csrf("/api/v1/chat/completions", json=request_body, headers={"Authorization": valid_auth_token})
    # This is a Pydantic validation error from FastAPI itself before hitting your logic.
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    errors = response.json().get("detail")
    assert isinstance(errors, list)
    assert any("messages" in e.get("loc", []) and "field required" in e.get("msg", "").lower() for e in errors)


@pytest.mark.integration
@patch.dict("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "sk-test"}, clear=False)
@patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call")
def test_chat_completions_response_shape_unchanged(
    mock_chat_api_call_shim, client, valid_auth_token, mock_db_dependencies_for_integration
):
    mock_chat_api_call_shim.return_value = {
        "id": "chatcmpl-test-shape",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "shape-ok"},
                "finish_reason": "stop",
            }
        ],
    }

    request_body = {
        "api_provider": "openai",
        "model": "gpt-4o-mini",
        "messages": [msg.model_dump(exclude_none=True) for msg in INTEGRATION_MESSAGES_NO_SYS_SCHEMA],
        "stream": False,
    }
    response = client.post_with_csrf(
        "/api/v1/chat/completions",
        json=request_body,
        headers={"Authorization": valid_auth_token},
    )
    assert response.status_code == status.HTTP_200_OK, response.text
    body = response.json()
    assert "choices" in body
    assert "model" in body
