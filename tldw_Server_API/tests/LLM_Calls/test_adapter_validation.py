from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Chat.Chat_Deps import ChatBadRequestError
from tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter import AnthropicAdapter
from tldw_Server_API.app.core.LLM_Calls.providers.google_adapter import GoogleAdapter
from tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter import OpenAIAdapter
from tldw_Server_API.app.core.LLM_Calls.providers.openrouter_adapter import OpenRouterAdapter
from tldw_Server_API.app.core.LLM_Calls.providers.mistral_adapter import MistralAdapter
from tldw_Server_API.app.core.LLM_Calls.providers.cohere_adapter import CohereAdapter


def test_openai_adapter_rejects_provider_unsupported_fields(monkeypatch):
    def _fail_factory(*_args, **_kwargs):
        raise AssertionError("http_client_factory should not be called on validation errors")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter.http_client_factory",
        _fail_factory,
    )

    adapter = OpenAIAdapter()
    with pytest.raises(ChatBadRequestError) as exc:
        adapter.chat({"messages": [], "model": "gpt-3.5-turbo", "top_k": 5})
    assert "top_k" in str(exc.value)


def test_openrouter_adapter_normalizes_topk_alias(monkeypatch):
    captured = {}

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": []}

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return FakeResp()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.openrouter_adapter.http_client_factory",
        lambda *a, **k: FakeClient(),
    )

    adapter = OpenRouterAdapter()
    adapter.chat(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "model": "openrouter/test-model",
            "topk": 7,
        }
    )

    payload = captured["json"]
    assert payload["top_k"] == 7
    assert "topk" not in payload


def test_mistral_adapter_rejects_min_p(monkeypatch):
    def _fail_factory(*_args, **_kwargs):
        raise AssertionError("http_client_factory should not be called on validation errors")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.mistral_adapter.http_client_factory",
        _fail_factory,
    )

    adapter = MistralAdapter()
    with pytest.raises(ChatBadRequestError) as exc:
        adapter.chat({"messages": [], "model": "mistral-small", "min_p": 0.2})
    assert "min_p" in str(exc.value)


def test_cohere_adapter_rejects_tool_choice(monkeypatch):
    def _fail_handler(*_args, **_kwargs):
        raise AssertionError("_to_handler_args should not be called on validation errors")

    monkeypatch.setattr(CohereAdapter, "_to_handler_args", _fail_handler, raising=True)

    adapter = CohereAdapter()
    with pytest.raises(ChatBadRequestError) as exc:
        adapter.chat({"messages": [], "model": "command-r", "tool_choice": "auto"})
    assert "tool_choice" in str(exc.value)


def test_anthropic_adapter_normalizes_stop_sequences(monkeypatch):
    captured = {}

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"type": "message", "content": []}

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["json"] = json
            return FakeResp()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter.http_client_factory",
        lambda *a, **k: FakeClient(),
    )

    adapter = AnthropicAdapter()
    adapter.chat(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "claude-test",
            "api_key": "test-key",
            "stop": "END",
        }
    )

    assert captured["json"]["stop_sequences"] == ["END"]


def test_google_adapter_normalizes_stop_sequences(monkeypatch):
    captured = {}

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"candidates": []}

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["json"] = json
            return FakeResp()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.google_adapter.http_client_factory",
        lambda *a, **k: FakeClient(),
    )

    adapter = GoogleAdapter()
    adapter.chat(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "gemini-test",
            "api_key": "test-key",
            "stop": "END",
        }
    )

    payload = captured["json"]
    assert payload["generationConfig"]["stopSequences"] == ["END"]


def test_google_adapter_rejects_tools_without_mapping(monkeypatch):
    def _fail_factory(*_args, **_kwargs):
        raise AssertionError("http_client_factory should not be called on validation errors")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.google_adapter.http_client_factory",
        _fail_factory,
    )
    monkeypatch.delenv("LLM_ADAPTERS_GEMINI_TOOLS_BETA", raising=False)

    adapter = GoogleAdapter()
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    with pytest.raises(ChatBadRequestError) as exc:
        adapter.chat(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "model": "gemini-test",
                "api_key": "test-key",
                "tools": tools,
            }
        )
    assert "Gemini tools" in str(exc.value)


def test_google_adapter_tools_beta_false_rejects_openai_tools(monkeypatch):
    def _fail_factory(*_args, **_kwargs):
        raise AssertionError("http_client_factory should not be called on validation errors")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.google_adapter.http_client_factory",
        _fail_factory,
    )
    monkeypatch.setenv("LLM_ADAPTERS_GEMINI_TOOLS_BETA", "0")

    adapter = GoogleAdapter()
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    with pytest.raises(ChatBadRequestError) as exc:
        adapter.chat(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "model": "gemini-test",
                "api_key": "test-key",
                "tools": tools,
            }
        )
    assert "Gemini tools" in str(exc.value)


def test_openai_adapter_validates_config_defaults(monkeypatch):
    def _fail_factory(*_args, **_kwargs):
        raise AssertionError("http_client_factory should not be called on validation errors")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter.http_client_factory",
        _fail_factory,
    )

    adapter = OpenAIAdapter()
    with pytest.raises(ChatBadRequestError) as exc:
        adapter.chat(
            {
                "messages": [],
                "model": "gpt-3.5-turbo",
                "app_config": {"openai_api": {"response_format": "json_object"}},
            }
        )
    assert "response_format" in str(exc.value)


def test_google_adapter_validates_config_defaults(monkeypatch):
    def _fail_factory(*_args, **_kwargs):
        raise AssertionError("http_client_factory should not be called on validation errors")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.google_adapter.http_client_factory",
        _fail_factory,
    )

    adapter = GoogleAdapter()
    with pytest.raises(ChatBadRequestError) as exc:
        adapter.chat(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "model": "gemini-test",
                "api_key": "test-key",
                "app_config": {"google_api": {"response_format": "json_object"}},
            }
        )
    assert "response_format" in str(exc.value)


def test_ordered_leading_system_prefix_keeps_later_placement_and_original_input():
    from copy import deepcopy

    from tldw_Server_API.app.core.LLM_Calls.adapter_utils import split_system_message

    messages = [
        {"role": "system", "content": "Saved note"},
        {"role": "system", "content": "Saved character"},
        {"role": "user", "content": "Continue."},
        {"role": "system", "content": "Depth note"},
    ]
    original = deepcopy(messages)
    assert split_system_message(messages, leading_only=True) == (
        "Saved note\n\nSaved character",
        messages[2:],
    )
    assert messages == original


def test_default_system_extraction_preserves_existing_nonleading_contract():
    from tldw_Server_API.app.core.LLM_Calls.adapter_utils import split_system_message

    messages = [
        {"role": "user", "content": "Earlier"},
        {"role": "system", "content": "Legacy system"},
        {"role": "system", "content": "Later system"},
    ]
    assert split_system_message(messages) == (
        "Legacy system",
        [messages[0], messages[2]],
    )
