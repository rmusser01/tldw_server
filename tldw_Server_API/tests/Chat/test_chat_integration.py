# tests/v1/integration/test_chat_integration.py
from unittest.mock import patch

import pytest
import os
import json
import time
from dotenv import load_dotenv
from fastapi import status
from fastapi.testclient import TestClient

# Adjust import paths based on your project structure
from tldw_Server_API.app.main import app # Your main FastAPI app instance

# Load environment variables from .env file
load_dotenv()

# --- Test Client ---
# Using a fixture to create a client for each test function for better isolation
@pytest.fixture(scope="function")
def test_client():
    return TestClient(app)

# --- Helper Functions ---
def is_api_key_set(provider_name):
    """Checks if the API key environment variable is set for a given provider."""
    env_var_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "cohere": "COHERE_API_KEY",
        "groq": "GROQ_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "google": "GOOGLE_API_KEY",
        "huggingface": "HUGGINGFACE_API_KEY",
        "kobold": "KOBOLD_API_KEY", # Check if needed, might be URL based
        "custom-openai-api": "CUSTOM_OPENAI_API_KEY",
        "custom-openai-api-2": "CUSTOM_OPENAI_API_KEY_2",
        "aphrodite": "APHRODITE_API_KEY", # Check if needed
        # Add other providers requiring specific keys
    }
    var_name = env_var_map.get(provider_name.lower())
    key_exists = bool(var_name and os.getenv(var_name))
    # print(f"Checking key for {provider_name}: {'Set' if key_exists else 'Not Set'}") # Debug log
    return key_exists

def is_local_service_set(service_name):
    """Checks if a local service URL environment variable is set."""
    env_var_map = {
        "ollama": "OLLAMA_HOST",         # Example name
        "llama.cpp": "LLAMA_CPP_URL",   # Example name
        "ooba": "OOBA_URL",             # Example name
        "tabbyapi": "TABBYAPI_URL",     # Example name
        "vllm": "VLLM_URL",             # Example name
        "local-llm": "LOCAL_LLM_CONFIG", # Could be a path or URL
        # Add others as needed by your chat_api_call logic
    }
    var_name = env_var_map.get(service_name.lower())
    service_configured = bool(var_name and os.getenv(var_name))
    # print(f"Checking service config for {service_name}: {'Set' if service_configured else 'Not Set'}") # Debug log
    return service_configured

# --- Constants ---
INTEGRATION_MESSAGES = [{"role": "user", "content": "Explain Large Language Models in one simple sentence."}]
# Use a token that is valid in your test environment or adapt auth mocking
VALID_TEST_TOKEN = os.getenv("TEST_AUTH_TOKEN", "real-user-token")

# --- Test Cases ---

# === OpenAI ===
@pytest.mark.integration
@pytest.mark.skipif(not is_api_key_set("openai"), reason="OPENAI_API_KEY not set")
@pytest.mark.asyncio
async def test_integration_openai_non_streaming(test_client):
    request_body = {
        "api_provider": "openai", "model": "gpt-4o-mini",
        "messages": INTEGRATION_MESSAGES, "temperature": 0.7
    }
    response = test_client.post("/api/v1/chat/completions", json=request_body, headers={"token": VALID_TEST_TOKEN})
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data.get("id") and data["id"].startswith("chatcmpl-")
    assert data.get("choices") and data["choices"][0]["message"]["role"] == "assistant"
    content = data["choices"][0]["message"]["content"]
    assert isinstance(content, str) and len(content) > 5
    print(f"\nOpenAI: {content[:100]}...")

@pytest.mark.integration
@pytest.mark.skipif(not is_api_key_set("openai"), reason="OPENAI_API_KEY not set")
@pytest.mark.asyncio
async def test_integration_openai_streaming(test_client):
    request_body = {
        "api_provider": "openai", "model": "gpt-4o-mini",
        "messages": INTEGRATION_MESSAGES, "stream": True
    }
    response = test_client.post("/api/v1/chat/completions", json=request_body, headers={"token": VALID_TEST_TOKEN})
    assert response.status_code == status.HTTP_200_OK
    assert 'text/event-stream' in response.headers['content-type']
    # Consume stream and check basic properties
    full_content = ""
    received_done = False
    for line in response.iter_lines():
        if line.startswith("data:") and "[DONE]" in line: received_done = True; break
        if line.startswith("data:"):
             try: chunk = json.loads(line[len("data:"):].strip())
             except: continue
             content_part = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
             if content_part: full_content += content_part
    assert received_done
    assert len(full_content) > 5
    print(f"\nOpenAI Stream: {full_content[:100]}...")

# === Anthropic ===
@pytest.mark.integration
@pytest.mark.skipif(not is_api_key_set("anthropic"), reason="ANTHROPIC_API_KEY not set")
@pytest.mark.asyncio
async def test_integration_anthropic_non_streaming(test_client):
    request_body = {
        "api_provider": "anthropic", "model": "claude-3-haiku-20240307",
        "messages": INTEGRATION_MESSAGES, "max_tokens": 100 # Required by Anthropic
    }
    response = test_client.post("/api/v1/chat/completions", json=request_body, headers={"token": VALID_TEST_TOKEN})
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data.get("id") and data["id"].startswith("msg_")
    assert data.get("type") == "message"
    content = data.get("content", [{}])[0].get("text", "")
    assert isinstance(content, str) and len(content) > 5
    print(f"\nAnthropic: {content[:100]}...")

# === Cohere ===
@pytest.mark.integration
@pytest.mark.skipif(not is_api_key_set("cohere"), reason="COHERE_API_KEY not set")
@pytest.mark.asyncio
async def test_integration_cohere_non_streaming(test_client):
    request_body = {
        "api_provider": "cohere", "model": "command-light",
        "messages": INTEGRATION_MESSAGES # Shim must handle conversion if needed
    }
    response = test_client.post("/api/v1/chat/completions", json=request_body, headers={"token": VALID_TEST_TOKEN})
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    content = data.get("text") # Cohere often returns 'text' directly
    assert isinstance(content, str) and len(content) > 5
    print(f"\nCohere: {content[:100]}...")

# === Groq ===
@pytest.mark.integration
@pytest.mark.skipif(not is_api_key_set("groq"), reason="GROQ_API_KEY not set")
@pytest.mark.asyncio
async def test_integration_groq_non_streaming(test_client):
    request_body = {
        "api_provider": "groq", "model": "llama3-8b-8192",
        "messages": INTEGRATION_MESSAGES
    }
    response = test_client.post("/api/v1/chat/completions", json=request_body, headers={"token": VALID_TEST_TOKEN})
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data.get("id") and data["id"].startswith("chatcmpl-")
    assert data.get("choices") and data["choices"][0]["message"]["role"] == "assistant"
    content = data["choices"][0]["message"]["content"]
    assert isinstance(content, str) and len(content) > 5
    print(f"\nGroq: {content[:100]}...")

# === OpenRouter ===
@pytest.mark.integration
@pytest.mark.skipif(not is_api_key_set("openrouter"), reason="OPENROUTER_API_KEY not set")
@pytest.mark.asyncio
async def test_integration_openrouter_non_streaming(test_client):
    request_body = {
        "api_provider": "openrouter", "model": "mistralai/mistral-7b-instruct",
        "messages": INTEGRATION_MESSAGES
    }
    response = test_client.post("/api/v1/chat/completions", json=request_body, headers={"token": VALID_TEST_TOKEN})
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data.get("id")
    assert data.get("choices") and data["choices"][0]["message"]["role"] == "assistant"
    content = data["choices"][0]["message"]["content"]
    assert isinstance(content, str) and len(content) > 5
    print(f"\nOpenRouter: {content[:100]}...")

# === DeepSeek ===
@pytest.mark.integration
@pytest.mark.skipif(not is_api_key_set("deepseek"), reason="DEEPSEEK_API_KEY not set")
@pytest.mark.asyncio
async def test_integration_deepseek_non_streaming(test_client):
    request_body = {
        "api_provider": "deepseek", "model": "deepseek-chat",
        "messages": INTEGRATION_MESSAGES
    }
    response = test_client.post("/api/v1/chat/completions", json=request_body, headers={"token": VALID_TEST_TOKEN})
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data.get("id") and data["id"].startswith("chatcmpl-")
    assert data.get("choices") and data["choices"][0]["message"]["role"] == "assistant"
    content = data["choices"][0]["message"]["content"]
    assert isinstance(content, str) and len(content) > 5
    print(f"\nDeepSeek: {content[:100]}...")

# === Mistral ===
@pytest.mark.integration
@pytest.mark.skipif(not is_api_key_set("mistral"), reason="MISTRAL_API_KEY not set")
@pytest.mark.asyncio
async def test_integration_mistral_non_streaming(test_client):
    request_body = {
        "api_provider": "mistral", "model": "mistral-tiny", # Use tiny for cost
        "messages": INTEGRATION_MESSAGES
    }
    response = test_client.post("/api/v1/chat/completions", json=request_body, headers={"token": VALID_TEST_TOKEN})
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data.get("id") is not None
    assert data.get("object") == "chat.completion"
    assert data.get("choices") and data["choices"][0]["message"]["role"] == "assistant"
    content = data["choices"][0]["message"]["content"]
    assert isinstance(content, str) and len(content) > 5
    print(f"\nMistral: {content[:100]}...")

# === Google ===
@pytest.mark.integration
@pytest.mark.skipif(not is_api_key_set("google"), reason="GOOGLE_API_KEY not set")
@pytest.mark.asyncio
async def test_integration_google_non_streaming(test_client):
    request_body = {
        "api_provider": "google", "model": "gemini-1.5-flash-latest",
        "messages": INTEGRATION_MESSAGES # Shim must handle message format conversion
    }
    response = test_client.post("/api/v1/chat/completions", json=request_body, headers={"token": VALID_TEST_TOKEN})
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    # Check based on typical Gemini structure or shim normalization
    content = ""
    if data.get("candidates"):
         content = data["candidates"][0].get("content", {}).get("parts", [{}])[0].get("text", "")
    elif data.get("choices"): # If normalized like OpenAI
        content = data["choices"][0].get("message", {}).get("content", "")
    assert len(content) > 5
    print(f"\nGoogle: {content[:100]}...")

# === HuggingFace ===
@pytest.mark.integration
@pytest.mark.skipif(not is_api_key_set("huggingface"), reason="HUGGINGFACE_API_KEY not set")
@pytest.mark.asyncio
async def test_integration_huggingface_non_streaming(test_client):
    request_body = {
        "api_provider": "huggingface",
        # This needs to be the specific model identifier your chat_api_call expects
        # Could be repo_id, could be full inference endpoint URL
        "model": "mistralai/Mistral-7B-Instruct-v0.1", # EXAMPLE - ADJUST
        "messages": INTEGRATION_MESSAGES
    }
    response = test_client.post("/api/v1/chat/completions", json=request_body, headers={"token": VALID_TEST_TOKEN})
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    # Assertion depends heavily on which HF API (inf endpoint vs lib) and how shim handles it
    # Example for basic Inference API endpoint often returning list with 'generated_text'
    assert isinstance(data, list) and len(data) > 0
    assert "generated_text" in data[0]
    assert len(data[0]["generated_text"]) > 5
    print(f"\nHuggingFace: {data[0]['generated_text'][:100]}...")

# === Llama.cpp ===
@pytest.mark.integration
@pytest.mark.skipif(not is_local_service_set("llama.cpp"), reason="Llama.cpp service not configured")
@pytest.mark.asyncio
async def test_integration_llama_cpp_non_streaming(test_client):
    request_body = {
        "api_provider": "llama.cpp",
        "model": "local-model", # Usually ignored by llama.cpp server, but required by Pydantic
        "messages": INTEGRATION_MESSAGES
    }
    response = test_client.post("/api/v1/chat/completions", json=request_body, headers={"token": VALID_TEST_TOKEN})
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    # llama.cpp server has OpenAI compatible endpoint format
    assert data.get("id") is not None
    assert data.get("choices") and data["choices"][0]["message"]["role"] == "assistant"
    content = data["choices"][0]["message"]["content"]
    assert isinstance(content, str) and len(content) > 5
    print(f"\nLlama.cpp: {content[:100]}...")

# === KoboldCpp ===
@pytest.mark.integration
@pytest.mark.skipif(not is_local_service_set("kobold"), reason="Kobold service not configured") # Assuming URL based check
@pytest.mark.asyncio
async def test_integration_kobold_non_streaming(test_client):
    request_body = {
        "api_provider": "kobold",
        "model": "local-model", # Likely ignored, but required by Pydantic
        "messages": INTEGRATION_MESSAGES # Shim must handle conversion
    }
    response = test_client.post("/api/v1/chat/completions", json=request_body, headers={"token": VALID_TEST_TOKEN})
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    # Kobold format varies, check based on shim normalization or raw output
    # Example if shim normalizes to OpenAI-like structure's content:
    content = data.get("choices", [{}])[0].get("message", {}).get("content")
    if not content and data.get("results"): # Raw KoboldAI Lite format
        content = data["results"][0].get("text")
    assert isinstance(content, str) and len(content) > 5
    print(f"\nKobold: {content[:100]}...")

# === Oobabooga ===
@pytest.mark.integration
@pytest.mark.skipif(not is_local_service_set("ooba"), reason="Oobabooga service not configured")
@pytest.mark.asyncio
async def test_integration_ooba_non_streaming(test_client):
    request_body = {
        "api_provider": "ooba",
        "model": "local-model", # Ignored if server has model loaded
        "messages": INTEGRATION_MESSAGES # Shim must handle conversion
    }
    response = test_client.post("/api/v1/chat/completions", json=request_body, headers={"token": VALID_TEST_TOKEN})
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    # Ooba API format varies, check based on shim normalization or raw output
    # Example if shim normalizes or hits OpenAI compatible endpoint:
    content = data.get("choices", [{}])[0].get("message", {}).get("content")
     # Example for older API /api/v1/generate
    if not content and data.get("results"):
        content = data["results"][0].get("text")
    assert isinstance(content, str) and len(content) > 5
    print(f"\nOoba: {content[:100]}...")


# === TabbyAPI ===
@pytest.mark.integration
@pytest.mark.skipif(not is_local_service_set("tabbyapi"), reason="TabbyAPI service not configured")
@pytest.mark.asyncio
async def test_integration_tabbyapi_non_streaming(test_client):
    request_body = {
        "api_provider": "tabbyapi",
        "model": "local-model", # Often specific model served by TabbyAPI
        "messages": INTEGRATION_MESSAGES
    }
    response = test_client.post("/api/v1/chat/completions", json=request_body, headers={"token": VALID_TEST_TOKEN})
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    # TabbyAPI typically has OpenAI compatible completion endpoint
    assert data.get("id") is not None
    assert data.get("choices") and data["choices"][0]["message"]["role"] == "assistant"
    content = data["choices"][0]["message"]["content"]
    assert isinstance(content, str) and len(content) > 5
    print(f"\nTabbyAPI: {content[:100]}...")

# === vLLM ===
@pytest.mark.integration
@pytest.mark.skipif(not is_local_service_set("vllm"), reason="vLLM service not configured")
@pytest.mark.asyncio
async def test_integration_vllm_non_streaming(test_client):
    request_body = {
        "api_provider": "vllm",
        "model": "mistralai/Mistral-7B-Instruct-v0.1", # Model served by vLLM
        "messages": INTEGRATION_MESSAGES
    }
    response = test_client.post("/api/v1/chat/completions", json=request_body, headers={"token": VALID_TEST_TOKEN})
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    # vLLM OpenAI compatible endpoint
    assert data.get("id") and data["id"].startswith("cmpl-")
    assert data.get("choices") and data["choices"][0]["message"]["role"] == "assistant"
    content = data["choices"][0]["message"]["content"]
    assert isinstance(content, str) and len(content) > 5
    print(f"\nvLLM: {content[:100]}...")

# === Local LLM (Generic) ===
@pytest.mark.integration
@pytest.mark.skipif(not is_local_service_set("local-llm"), reason="'local-llm' service not configured")
@pytest.mark.asyncio
async def test_integration_local_llm_non_streaming(test_client):
    request_body = {
        "api_provider": "local-llm",
        "model": "some-local-model-ref", # Identifier used by your local setup
        "messages": INTEGRATION_MESSAGES
    }
    response = test_client.post("/api/v1/chat/completions", json=request_body, headers={"token": VALID_TEST_TOKEN})
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    # Assert based on whatever format your 'local-llm' shim wrapper returns
    # Example assuming OpenAI format normalization:
    assert data.get("id") is not None
    assert data.get("choices") and data["choices"][0]["message"]["role"] == "assistant"
    content = data["choices"][0]["message"]["content"]
    assert isinstance(content, str) and len(content) > 5
    print(f"\nLocal LLM: {content[:100]}...")

# === Aphrodite ===
@pytest.mark.integration
@pytest.mark.skipif(not is_local_service_set("aphrodite"), reason="Aphrodite service not configured") # Or key check if needed
@pytest.mark.asyncio
async def test_integration_aphrodite_non_streaming(test_client):
    request_body = {
        "api_provider": "aphrodite",
        "model": "model-served-by-aphrodite", # Specific model name
        "messages": INTEGRATION_MESSAGES
    }
    response = test_client.post("/api/v1/chat/completions", json=request_body, headers={"token": VALID_TEST_TOKEN})
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    # Aphrodite Engine typically provides OpenAI compatible endpoint
    assert data.get("id") and data["id"].startswith("chatcmpl-")
    assert data.get("choices") and data["choices"][0]["message"]["role"] == "assistant"
    content = data["choices"][0]["message"]["content"]
    assert isinstance(content, str) and len(content) > 5
    print(f"\nAphrodite: {content[:100]}...")

# === Custom OpenAI API 1 ===
@pytest.mark.integration
@pytest.mark.skipif(not is_api_key_set("custom-openai-api"), reason="CUSTOM_OPENAI_API_KEY not set")
@pytest.mark.asyncio
async def test_integration_custom_openai_api_non_streaming(test_client):
    request_body = {
        "api_provider": "custom-openai-api",
        "model": "custom-model-1",
        "messages": INTEGRATION_MESSAGES
    }
    response = test_client.post("/api/v1/chat/completions", json=request_body, headers={"token": VALID_TEST_TOKEN})
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data.get("id") is not None # Check based on custom API's response
    assert data.get("choices") and data["choices"][0]["message"]["role"] == "assistant"
    content = data["choices"][0]["message"]["content"]
    assert isinstance(content, str) and len(content) > 5
    print(f"\nCustom OpenAI 1: {content[:100]}...")

# === Custom OpenAI API 2 ===
@pytest.mark.integration
@pytest.mark.skipif(not is_api_key_set("custom-openai-api-2"), reason="CUSTOM_OPENAI_API_KEY_2 not set")
@pytest.mark.asyncio
async def test_integration_custom_openai_api_2_non_streaming(test_client):
    request_body = {
        "api_provider": "custom-openai-api-2",
        "model": "custom-model-2",
        "messages": INTEGRATION_MESSAGES
    }
    response = test_client.post("/api/v1/chat/completions", json=request_body, headers={"token": VALID_TEST_TOKEN})
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data.get("id") is not None # Check based on custom API's response
    assert data.get("choices") and data["choices"][0]["message"]["role"] == "assistant"
    content = data["choices"][0]["message"]["content"]
    assert isinstance(content, str) and len(content) > 5
    print(f"\nCustom OpenAI 2: {content[:100]}...")


# --- (Keep the Invalid Key Integration Test) ---
@pytest.mark.integration
@pytest.mark.skipif(not is_api_key_set("openai"), reason="OPENAI_API_KEY not set") # Base skip on one known provider
@pytest.mark.asyncio
async def test_chat_integration_invalid_key_provider(test_client):
    """Integration test with an invalid key for a provider that requires one."""
    provider_to_test = "openai" # Can change this to test others like Anthropic etc.
    with patch('tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS', {provider_to_test: "sk-invalidkey123"}):
        request_body = {
            "api_provider": provider_to_test,
            "model": "gpt-4o-mini",
            "messages": INTEGRATION_MESSAGES
        }
        response = test_client.post("/api/v1/chat/completions", json=request_body, headers={"token": VALID_TEST_TOKEN})
        assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_401_UNAUTHORIZED]
        detail = response.json().get("detail", "").lower()
        assert "error occurred" in detail or "authentication" in detail or "invalid api key" in detail
        print(f"\nInvalid Key Response Detail ({provider_to_test}): {response.json().get('detail')}")