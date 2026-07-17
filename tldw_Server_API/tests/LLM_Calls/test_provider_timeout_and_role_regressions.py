from __future__ import annotations

import pytest

from tldw_Server_API.app.core.LLM_Calls.providers.cohere_adapter import CohereAdapter
from tldw_Server_API.app.core.LLM_Calls.providers.google_adapter import GoogleAdapter
from tldw_Server_API.app.core.LLM_Calls.providers.local_adapters import (
    _chat_with_openai_compatible_local_server,
)
from tldw_Server_API.app.core.LLM_Calls.providers.mlx_provider import MLXChatAdapter
from tldw_Server_API.app.core.LLM_Calls.providers.moonshot_adapter import MoonshotAdapter
from tldw_Server_API.app.core.LLM_Calls.providers.zai_adapter import ZaiAdapter


@pytest.mark.unit
def test_mlx_generate_kwargs_preserves_explicit_zero_values():
    adapter = MLXChatAdapter()
    out = adapter._generate_kwargs(
        {
            "max_tokens": 64,
            "temperature": 0.0,
            "top_p": 0.0,
            "top_k": 0,
        }
    )

    assert out["max_tokens"] == 64
    assert out["temp"] == 0.0
    assert out["top_p"] == 0.0
    assert out["top_k"] == 0


@pytest.mark.unit
def test_local_non_streaming_prod_fetch_receives_timeout(monkeypatch):
    # Force the production non-stream branch during pytest.
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "")

    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": []}

        def close(self):
            return None

    class FakeClient:
        def close(self):
            return None

    def fake_fetch(**kwargs):
        captured["timeout"] = kwargs.get("timeout")
        return FakeResponse()

    result = _chat_with_openai_compatible_local_server(
        api_base_url="http://localhost:1234/v1",
        model_name="local-model",
        input_data=[{"role": "user", "content": "ping"}],
        streaming=False,
        timeout=17,
        http_client_factory=lambda timeout: FakeClient(),
        http_fetcher=fake_fetch,
    )

    assert result == {"choices": []}
    assert captured["timeout"] == 17


@pytest.mark.unit
def test_moonshot_adapter_chat_timeout_override_is_used(monkeypatch):
    captured: dict[str, object] = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": []}

        def close(self):
            return None

    class FakeSession:
        def post(self, url, headers=None, json=None, timeout=None, stream=False):
            captured["timeout"] = timeout
            return FakeResponse()

        def close(self):
            return None

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.chat_calls.create_session_with_retries",
        lambda **kwargs: FakeSession(),
    )

    adapter = MoonshotAdapter()
    adapter.chat(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "model": "moonshot-v1-8k",
            "api_key": "test-key",
            "app_config": {"moonshot_api": {"api_timeout": 33}},
        },
        timeout=7,
    )

    assert captured["timeout"] == 7


@pytest.mark.unit
def test_zai_adapter_chat_timeout_override_is_used(monkeypatch):
    captured: dict[str, object] = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": []}

        def close(self):
            return None

    class FakeSession:
        def post(self, url, headers=None, json=None, timeout=None, stream=False):
            captured["timeout"] = timeout
            return FakeResponse()

        def close(self):
            return None

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.chat_calls.create_session_with_retries",
        lambda **kwargs: FakeSession(),
    )

    adapter = ZaiAdapter()
    adapter.chat(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "model": "glm-4.5",
            "api_key": "test-key",
            "app_config": {"zai_api": {"api_timeout": 44}},
        },
        timeout=9,
    )

    assert captured["timeout"] == 9


@pytest.mark.unit
def test_google_gemini_contents_maps_tool_role_to_user():
    contents = GoogleAdapter._to_gemini_contents(
        [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "assistant text"},
            {"role": "tool", "content": "tool output"},
            {"role": "user", "content": "user text"},
        ]
    )

    assert [entry["role"] for entry in contents] == ["model", "user", "user"]


@pytest.mark.unit
def test_cohere_adapter_chat_timeout_override_is_used(monkeypatch):
    captured: dict[str, object] = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"text": "ok"}

        def close(self):
            return None

    class FakeSession:
        def post(self, url, headers=None, json=None, timeout=None, stream=False):
            captured["timeout"] = timeout
            return FakeResponse()

        def close(self):
            return None

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.chat_calls.create_session_with_retries",
        lambda **kwargs: FakeSession(),
    )

    adapter = CohereAdapter()
    adapter.chat(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "model": "command-r",
            "api_key": "test-key",
            "app_config": {"cohere_api": {"api_timeout": 120}},
        },
        timeout=11,
    )

    assert captured["timeout"] == 11
