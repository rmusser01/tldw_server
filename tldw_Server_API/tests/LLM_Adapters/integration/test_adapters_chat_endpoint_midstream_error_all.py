"""
Endpoint SSE mid-stream error tests for all adapter-backed providers.

Simulate a provider that emits some normal SSE chunks then fails mid-stream.
Verify the endpoint returns exactly one structured SSE error frame and one
terminal [DONE], with earlier normal chunks preserved.
"""

from __future__ import annotations

import pytest

# Ensure chat fixtures (client/auth) are registered
from tldw_Server_API.tests._plugins import chat_fixtures as _chat_pl  # noqa: F401


@pytest.fixture(autouse=True)
def _enable(monkeypatch):
    monkeypatch.setenv("STREAMS_UNIFIED", "1")
    monkeypatch.setenv("LOGURU_LEVEL", "ERROR")
    # Disable moderation holdback buffering so early chunks are observable.
    monkeypatch.setenv("MODERATION_STREAM_BUFFER_CHARS", "0")
    monkeypatch.delenv("TEST_MODE", raising=False)
    yield


_CASES: tuple[tuple[str, str, str, str], ...] = (
    ("openai", "tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter", "OpenAIAdapter", "sk-openai-test"),
    ("anthropic", "tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter", "AnthropicAdapter", "sk-ant-test"),
    ("groq", "tldw_Server_API.app.core.LLM_Calls.providers.groq_adapter", "GroqAdapter", "sk-groq-test"),
    ("openrouter", "tldw_Server_API.app.core.LLM_Calls.providers.openrouter_adapter", "OpenRouterAdapter", "sk-or-test"),
    ("google", "tldw_Server_API.app.core.LLM_Calls.providers.google_adapter", "GoogleAdapter", "sk-gemini-test"),
    ("mistral", "tldw_Server_API.app.core.LLM_Calls.providers.mistral_adapter", "MistralAdapter", "sk-mist-test"),
    ("qwen", "tldw_Server_API.app.core.LLM_Calls.providers.qwen_adapter", "QwenAdapter", "sk-qwen-test"),
    ("deepseek", "tldw_Server_API.app.core.LLM_Calls.providers.deepseek_adapter", "DeepSeekAdapter", "sk-deepseek-test"),
    ("huggingface", "tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter", "HuggingFaceAdapter", "sk-hf-test"),
    ("custom-openai-api", "tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter", "CustomOpenAIAdapter", "sk-custom1-test"),
)

_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "google": "GOOGLE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "qwen": "QWEN_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "huggingface": "HUGGINGFACE_API_KEY",
}


def _payload(provider: str) -> dict:
    model_map = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-sonnet",
        "groq": "llama-3.1-8b-instant",
        "openrouter": "openrouter/auto",
        "google": "gemini-2.5-pro",
        "mistral": "mistral-large-latest",
        "qwen": "qwen2.5:7b",
        "deepseek": "deepseek-chat",
        "huggingface": "meta-llama/Meta-Llama-3-8B-Instruct",
        "custom-openai-api": "my-openai-compatible",
    }
    return {
        "api_provider": provider,
        "model": model_map.get(provider, "dummy"),
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    }


@pytest.mark.integration
@pytest.mark.parametrize("provider, modname, cls_name, key_value", _CASES)
def test_endpoint_midstream_error_single_sse_and_done(monkeypatch, authenticated_client, provider: str, modname: str, cls_name: str, key_value: str):
    if provider_key_env := _API_KEY_ENV.get(provider):
        monkeypatch.setenv(provider_key_env, key_value)

    # Patch adapter.stream to yield some chunks, then raise a provider error
    mod = __import__(modname, fromlist=[cls_name])
    Adapter = getattr(mod, cls_name)
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatProviderError

    def _stream_miderror(*args, **kwargs):
        def _gen():
            yield "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n"
            yield "data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n"
            raise ChatProviderError(provider=provider, message="boom")
        return _gen()

    monkeypatch.setattr(Adapter, "stream", _stream_miderror, raising=True)

    client = authenticated_client
    with client.stream("POST", "/api/v1/chat/completions", json=_payload(provider)) as resp:
        assert resp.status_code == 200
        assert resp.headers.get("content-type", "").lower().startswith("text/event-stream")
        lines = [line for line in resp.iter_lines() if line.strip()]
        content_index = next(index for index, line in enumerate(lines) if '"hello"' in line)
        continuation_index = next(
            index for index, line in enumerate(lines) if '" world"' in line
        )
        error_indexes = [index for index, line in enumerate(lines) if '"error"' in line]
        done_indexes = [
            index
            for index, line in enumerate(lines)
            if line.strip().lower() == "data: [done]"
        ]
        assert len(error_indexes) == 1
        assert len(done_indexes) == 1
        assert content_index < continuation_index < error_indexes[0] < done_indexes[0]
        assert done_indexes[0] == len(lines) - 1
