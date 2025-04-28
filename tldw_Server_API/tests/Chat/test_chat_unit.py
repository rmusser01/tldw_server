from unittest.mock import patch

import pytest
import os
import json
from dotenv import load_dotenv
from fastapi import status
from fastapi.testclient import TestClient

# Adjust import paths based on your project structure
from tldw_Server_API.app.main import app # Import your main FastAPI app instance

# Load environment variables from .env file (optional)
load_dotenv()

# Use TestClient for sending requests to your app
# It's often better to create a new client per test or module in integration tests
# to ensure isolation, especially if tests modify app state (though less likely here).
# client = TestClient(app)

# --- Helper to check if API key env var is set ---
def is_api_key_set(provider_name):
    env_var_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "groq": "GROQ_API_KEY",
        # Add mappings for other providers you test
    }
    var_name = env_var_map.get(provider_name.lower())
    return bool(var_name and os.getenv(var_name))

INTEGRATION_MESSAGES = [{"role": "user", "content": "Briefly explain what Large Language Models are in one sentence."}]


# --- Test Cases ---

@pytest.mark.integration
@pytest.mark.skipif(not is_api_key_set("openai"), reason="OPENAI_API_KEY not set in environment")
@pytest.mark.asyncio
async def test_chat_integration_openai_non_streaming():
    """Integration test for non-streaming OpenAI call."""
    client = TestClient(app) # Create client within test
    request_body = {
        "api_provider": "openai",
        "model": "gpt-4o-mini", # Use a cost-effective model for testing
        "messages": [{"role": "user", "content": "Say 'Hello Test!'"}]
    }
    response = client.post("/api/v1/chat/completions", json=request_body, headers={"token": "real-user-token"}) # Use a token valid for your test setup

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["id"].startswith("chatcmpl-")
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) > 0
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(data["choices"][0]["message"]["content"], str)
    assert "Hello Test!" in data["choices"][0]["message"]["content"] # Check if the specific phrase is likely there
    assert "usage" in data

@pytest.mark.integration
@pytest.mark.skipif(not is_api_key_set("openai"), reason="OPENAI_API_KEY not set in environment")
@pytest.mark.asyncio
async def test_chat_integration_openai_streaming():
    """Integration test for streaming OpenAI call."""
    client = TestClient(app)
    request_body = {
        "api_provider": "openai",
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Stream 'Hello Stream!'" }],
        "stream": True
    }
    response = client.post("/api/v1/chat/completions", json=request_body, headers={"token": "real-user-token"})

    assert response.status_code == status.HTTP_200_OK
    assert response.headers['content-type'] == 'text/event-stream; charset=utf-8'

    # Consume the stream
    full_content = ""
    received_done = False
    lines = []
    try:
        for line in response.iter_lines():
             lines.append(line) # Store for debugging if needed
             # print(f"LINE: {line}") # Uncomment for debug
             if line.startswith("data:"):
                 data_content = line[len("data:"):].strip()
                 if data_content == "[DONE]":
                     received_done = True
                     break
                 try:
                     chunk = json.loads(data_content)
                     if chunk.get("choices") and len(chunk["choices"]) > 0:
                         delta = chunk["choices"][0].get("delta", {})
                         content_part = delta.get("content")
                         if content_part:
                             full_content += content_part
                 except json.JSONDecodeError:
                     pytest.fail(f"Failed to decode JSON stream chunk: {data_content}")
    except Exception as e:
        print("Collected Lines:", "\n".join(lines)) # Print collected lines on error
        pytest.fail(f"Error consuming stream: {e}")


    assert received_done, "Stream did not finish with [DONE]"
    assert len(
        full_content) > 10, f"Expected some streamed content, but got none or very little: '{full_content}'"  # Check for reasonable length
    print(f"\nOpenAI Stream Unit Test: {full_content[:100]}...")  # Print received content

@pytest.mark.integration
@pytest.mark.skipif(not is_api_key_set("anthropic"), reason="ANTHROPIC_API_KEY not set in environment")
@pytest.mark.asyncio
async def test_chat_integration_anthropic_non_streaming():
    """Integration test for non-streaming Anthropic call."""
    client = TestClient(app)
    request_body = {
        "api_provider": "anthropic",
        "model": "claude-3-haiku-20240307", # Use a cost-effective model
        "messages": [{"role": "user", "content": "Say 'Hello Anthropic Test!'"}]
    }
    response = client.post("/api/v1/chat/completions", json=request_body, headers={"token": "real-user-token"})

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    # Basic checks - structure might differ based on how chat_api_call normalizes
    assert data.get("id") is not None
    assert data.get("type") == "message"
    assert isinstance(data.get("content"), list)
    assert len(data["content"]) > 0
    assert data["content"][0]["type"] == "text"
    assert "Hello Anthropic Test!" in data["content"][0]["text"]
    assert "usage" in data

# --- Add more integration tests for other providers as needed ---
# Example for Groq:
# @pytest.mark.integration
# @pytest.mark.skipif(not is_api_key_set("groq"), reason="GROQ_API_KEY not set in environment")
# @pytest.mark.asyncio
# async def test_chat_integration_groq_non_streaming():
#    client = TestClient(app)
#    # ... request body for Groq ...
#    response = client.post(...)
#    assert response.status_code == status.HTTP_200_OK
#    # ... assertions specific to Groq response structure (or normalized structure) ...

# --- Integration Error Test ---
@pytest.mark.integration
@pytest.mark.skipif(not is_api_key_set("openai"), reason="OPENAI_API_KEY not set in environment")
@pytest.mark.asyncio
async def test_chat_integration_openai_invalid_key():
    """Integration test with an invalid OpenAI key."""
    client = TestClient(app)
    # Temporarily override the key used *by the endpoint logic* for this test
    with patch('tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS', {"openai": "sk-invalidkey123"}):
        request_body = {
            "api_provider": "openai",
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Test invalid key"}]
        }
        response = client.post("/api/v1/chat/completions", json=request_body, headers={"token": "real-user-token"})

        # Expecting the backend call to fail, which chat_api_call should report as an error string,
        # leading to a 400 from the endpoint. Could also be 401 if the shim specifically handles it.
        assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_401_UNAUTHORIZED]
        # Check detail message if possible (depends on chat_api_call's error reporting)
        assert "error occurred" in response.json()["detail"].lower() or "authentication" in response.json()["detail"].lower()