import pytest

from tldw_Server_API.app.api.v1.endpoints import messages as messages_endpoint
from tldw_Server_API.app.core.AuthNZ import provider_credential_runtime as runtime_module


def _build_openai_response(text: str) -> dict:
    return {
        "id": "chatcmpl-1",
        "model": "gpt-4",
        "choices": [
            {
                "message": {"content": text, "tool_calls": []},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def _build_native_response(text: str = "ok") -> dict:
    return {
        "id": "msg-1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": "test-model",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


class _FakeResponse:
    def __init__(self, json_data: dict):
        self._json_data = json_data

    def raise_for_status(self):
        return None

    def json(self):
        return self._json_data


class _FakeClient:
    def __init__(self, json_data: dict, capture: dict | None = None):
        self._json_data = json_data
        self._capture = capture

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, headers=None, json=None):
        if self._capture is not None:
            self._capture["url"] = url
            self._capture["headers"] = headers
            self._capture["json"] = json
        return _FakeResponse(self._json_data)


class _AsyncFakeClient:
    def __init__(self, json_data: dict, capture: dict | None = None):
        self._json_data = json_data
        self._capture = capture

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, headers=None, json=None):
        if self._capture is not None:
            self._capture["url"] = url
            self._capture["headers"] = headers
            self._capture["json"] = json
        return _FakeResponse(self._json_data)


@pytest.mark.integration
def test_messages_endpoint_non_streaming(client_user_only, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    async def _fake_call(**kwargs):
        return _build_openai_response("hello")

    monkeypatch.setattr(messages_endpoint, "perform_chat_api_call_async", _fake_call)

    payload = {
        "model": "openai/gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10,
    }

    response = client_user_only.post("/api/v1/messages", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["content"][0]["type"] == "text"
    assert body["content"][0]["text"] == "hello"


@pytest.mark.integration
def test_messages_endpoint_streaming(client_user_only, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    async def _fake_call(**kwargs):
        async def _stream():
            yield 'data: {"choices": [{"delta": {"content": "Hello"}, "finish_reason": null}]}\n\n'
            yield 'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n'
        return _stream()

    monkeypatch.setattr(messages_endpoint, "perform_chat_api_call_async", _fake_call)

    payload = {
        "model": "openai/gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }

    with client_user_only.stream("POST", "/api/v1/messages", json=payload) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "").lower()
        lines = list(response.iter_lines())

    joined = "\n".join(line for line in lines if line)
    assert "event: message_start" in joined
    assert "event: content_block_delta" in joined
    assert "event: message_stop" in joined


@pytest.mark.integration
def test_messages_count_tokens_unsupported_provider(client_user_only, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    payload = {
        "model": "openai/gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
    }

    response = client_user_only.post("/api/v1/messages/count_tokens", json=payload)
    assert response.status_code == 400
    assert "count_tokens" in response.text


@pytest.mark.integration
def test_messages_llamacpp_base_url_normalized(client_user_only, monkeypatch):
    capture = {}
    monkeypatch.setattr(
        runtime_module,
        "load_server_config_snapshot",
        lambda: {"llama_api": {"api_ip": "http://localhost:8080/v1/chat/completions"}},
    )

    async def native_post(url, headers, payload, **_kwargs):
        capture.update(url=url, headers=headers, json=payload)
        return _build_native_response()

    monkeypatch.setattr(messages_endpoint, "_native_post_json", native_post)

    payload = {
        "model": "llama.cpp/test-model",
        "messages": [{"role": "user", "content": "hi"}],
    }

    response = client_user_only.post("/api/v1/messages", json=payload)
    assert response.status_code == 200
    assert capture["url"].endswith("/v1/messages")
    assert "/chat/completions" not in capture["url"]


@pytest.mark.integration
def test_messages_llamacpp_base_url_normalized_completions(client_user_only, monkeypatch):
    capture = {}
    monkeypatch.setattr(
        runtime_module,
        "load_server_config_snapshot",
        lambda: {"llama_api": {"api_ip": "http://localhost:8080/v1/completions"}},
    )

    async def native_post(url, headers, payload, **_kwargs):
        capture.update(url=url, headers=headers, json=payload)
        return _build_native_response()

    monkeypatch.setattr(messages_endpoint, "_native_post_json", native_post)

    payload = {
        "model": "llama.cpp/test-model",
        "messages": [{"role": "user", "content": "hi"}],
    }

    response = client_user_only.post("/api/v1/messages", json=payload)
    assert response.status_code == 200
    assert capture["url"].endswith("/v1/messages")
    assert "/completions" not in capture["url"]


@pytest.mark.integration
def test_messages_count_tokens_anthropic(client_user_only, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        messages_endpoint,
        "http_client_factory",
        lambda *a, **k: _FakeClient({"input_tokens": 3}),
    )
    monkeypatch.setattr(
        messages_endpoint,
        "async_http_client_factory",
        lambda *a, **k: _AsyncFakeClient({"input_tokens": 3}),
    )

    payload = {
        "model": "anthropic/claude-3-sonnet-20240229",
        "messages": [{"role": "user", "content": "hi"}],
    }

    response = client_user_only.post("/api/v1/messages/count_tokens", json=payload)
    assert response.status_code == 200
    assert response.json()["input_tokens"] == 3
