import os
import pytest
from fastapi import status
from starlette.testclient import TestClient

# Import the FastAPI app the project exposes
from tldw_Server_API.app.main import app  # assumes this is the canonical import path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_all_commercial_providers():
    """Return the list of commercial providers the backend is configured for.

    We rely on the API_KEYS mapping used in the endpoint layer; each entry that
    has a non‑empty value is treated as an enabled provider.  The mapping is
    read at **runtime**, so adding a new provider automatically extends the
    parametrised test matrix.
    """
    # Import lazily to avoid circular import at collection time.
    from tldw_Server_API.app.api.v1.endpoints.chat import API_KEYS

    # Commercial providers are the keys that have *some* key set in env or file
    return [provider for provider, key in API_KEYS.items() if key]


def is_api_key_set(provider: str) -> bool:
    """True if the API key for *provider* is visible to the test suite."""
    from tldw_Server_API.app.api.v1.endpoints.chat import API_KEYS
    return bool(API_KEYS.get(provider))


# ---------------------------------------------------------------------------
# Parametrised integration tests – one entry per provider
# ---------------------------------------------------------------------------

PROVIDERS = get_all_commercial_providers()


@pytest.mark.integration
@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.skipif(not PROVIDERS, reason="No commercial providers configured in this environment")
@pytest.mark.asyncio
async def test_chat_integration_roundtrip(provider):
    """Simple happy‑path test: each provider should answer a one‑shot request."""

    client = TestClient(app)

    request_body = {
        "api_provider": provider,
        # use a minimal model name – most back‑ends fall back to default if the
        # exact model is not available; adjust per provider if needed.
        "model": "gpt-4o-mini" if provider == "openai" else "claude-3-haiku-20240307" if provider == "anthropic" else "command-r7b-12-2024" if provider == "cohere" else "mistral-large-latest" if provider == "mistral" else "llama3-8b-8192",
        "messages": [{"role": "user", "content": "Say 'Hello {0}!'".format(provider.capitalize())}],
        "stream": False,
    }

    # If the key is not set we *skip* rather than fail – this keeps CI green when
    # secrets are missing.
    if not is_api_key_set(provider):
        pytest.skip(f"API key for {provider} is not configured – skipping test")

    response = client.post("/api/v1/chat/completions", json=request_body, headers={"token": "test-token"})

    assert response.status_code == status.HTTP_200_OK, response.text


# ---------------------------------------------------------------------------
# Invalid‑key negative tests – make sure a 4xx is surfaced, not a 500
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.skipif(not PROVIDERS, reason="No commercial providers configured in this environment")
@pytest.mark.asyncio
async def test_chat_integration_invalid_key(provider, monkeypatch):
    """Override the provider's key with an obviously wrong value.

    The backend should propagate the provider's 401/400 and *not* crash with a
    500.  This mirrors the expectations of the existing invalid‑key tests for
    OpenAI.
    """

    # Patch the API_KEYS mapping used by the endpoint
    from tldw_Server_API.app.api.v1.endpoints.chat import API_KEYS
    monkeypatch.setitem(API_KEYS, provider, "sk-wrong-key-for-tests")

    client = TestClient(app)

    request_body = {
        "api_provider": provider,
        "model": "gpt-4o-mini" if provider == "openai" else "claude-3-haiku-20240307" if provider == "anthropic" else "command-r7b-12-2024" if provider == "cohere" else "mistral-large-latest" if provider == "mistral" else "llama3-8b-8192",
        "messages": [{"role": "user", "content": "Trigger invalid key path"}],
    }

    response = client.post("/api/v1/chat/completions", json=request_body, headers={"token": "test-token"})

    assert response.status_code in (status.HTTP_400_BAD_REQUEST, status.HTTP_401_UNAUTHORIZED), (
        f"{provider} should return 400/401 for invalid key, got {response.status_code} – body: {response.text}"
    )
