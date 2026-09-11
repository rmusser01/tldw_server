from collections.abc import Iterable
from unittest.mock import Mock

import httpx
import pytest

from tldw_Server_API.app.core.Chat.chat_service import (
    perform_chat_api_call,
    perform_chat_api_call_async,
)
from tldw_Server_API.app.core.LLM_Calls.chat_calls import (
    get_openai_embeddings,
    get_openai_embeddings_batch,
)
from tldw_Server_API.app.core.LLM_Calls.sse import sse_done

# --- Adapter-oriented test helpers (OpenAI) ---

class _FakeResp:
    def __init__(self, status_code=200, json_obj=None, text="", lines=None):
        self.status_code = status_code
        self._json_obj = json_obj if json_obj is not None else {}
        self.text = text
        self._lines = list(lines or [])

    def json(self):
        return self._json_obj

    def raise_for_status(self):
        import requests

        if self.status_code and int(self.status_code) >= 400:
            err = requests.exceptions.HTTPError("HTTP error")
            err.response = self  # attach minimal response interface
            raise err
        return None

    # streaming context manager shape
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_lines(self):
        yield from self._lines


class _FakeClient:
    def __init__(self, *, post_resp: _FakeResp | None = None, stream_lines=None):
        self._post_resp = post_resp
        self._stream_lines = list(stream_lines or [])
        self.last_json = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, *, headers=None, json=None):
        self.last_json = json
        return self._post_resp or _FakeResp(status_code=200, json_obj={"ok": True})

    def stream(self, method, url, *, headers=None, json=None):
        self.last_json = json
        return _FakeResp(status_code=200, lines=self._stream_lines)

from tldw_Server_API.app.core.Chat.Chat_Deps import ChatBadRequestError, ChatProviderError


class _DummyStream:
    """Minimal async stream stub that emits predefined lines."""

    def __init__(self, lines: Iterable[str]):
        self._lines = list(lines)
        self.status_code = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    def aiter_lines(self):
        async def _gen():
            for line in self._lines:
                yield line

        return _gen()


class _DummyAsyncClient:
    """AsyncClient drop-in used to intercept streaming calls."""

    def __init__(self, stream: _DummyStream):
        self._stream = stream

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def stream(self, *args, **kwargs):
        return self._stream

    async def post(self, *args, **kwargs):
        raise AssertionError("Non-streaming POST should not be invoked in these tests.")


def _mock_session_with_response(vector):
    session = Mock()
    response = Mock()
    response.status_code = 200
    response.json.return_value = {"data": vector}
    response.raise_for_status = Mock()
    session.post.return_value = response
    session.close = Mock()
    return session


def test_get_openai_embeddings_uses_timeout_and_closes(monkeypatch):
    session = _mock_session_with_response([{"embedding": [0.1, 0.2]}])

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.chat_calls.create_session_with_retries",
        lambda **kwargs: session,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.chat_calls.load_and_log_configs",
        lambda: {
            "openai_api": {
                "api_key": "test-key",
                "api_timeout": 12,
                "api_retries": 2,
                "api_retry_delay": 0.5,
            }
        },
    )

    result = get_openai_embeddings("hello", "text-embedding-3-small")

    session.post.assert_called_once()
    assert session.post.call_args[1]["timeout"] == 12
    session.close.assert_called_once()
    assert result == [0.1, 0.2]


def test_get_openai_embeddings_includes_dimensions(monkeypatch):
    session = _mock_session_with_response([{"embedding": [0.1, 0.2]}])

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.chat_calls.create_session_with_retries",
        lambda **kwargs: session,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.chat_calls.load_and_log_configs",
        lambda: {
            "openai_api": {
                "api_key": "test-key",
            }
        },
    )

    get_openai_embeddings("hello", "text-embedding-3-small", dimensions=128)

    payload = session.post.call_args[1]["json"]
    assert payload.get("dimensions") == 128


def test_get_openai_embeddings_forwards_configured_org_and_project_headers(monkeypatch):
    session = _mock_session_with_response([{"embedding": [0.1, 0.2]}])
    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.chat_calls.create_session_with_retries",
        lambda **kwargs: session,
    )

    get_openai_embeddings(
        "hello",
        "text-embedding-3-small",
        app_config={
            "openai_api": {
                "api_key": "test-key",
                "org_id": "org-single",
                "project_id": "project-single",
            }
        },
    )

    headers = session.post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["OpenAI-Organization"] == "org-single"
    assert headers["OpenAI-Project"] == "project-single"


def test_get_openai_embeddings_batch_uses_timeout_and_closes(monkeypatch):
    session = _mock_session_with_response(
        [
            {"embedding": [0.1, 0.2]},
            {"embedding": [0.3, 0.4]},
        ]
    )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.chat_calls.create_session_with_retries",
        lambda **kwargs: session,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.chat_calls.load_and_log_configs",
        lambda: {
            "openai_api": {
                "api_key": "test-key",
                "api_timeout": 30,
                "api_retries": 1,
                "api_retry_delay": 0.1,
            }
        },
    )

    result = get_openai_embeddings_batch(["a", "b"], "text-embedding-3-small")

    session.post.assert_called_once()
    assert session.post.call_args[1]["timeout"] == 30
    session.close.assert_called_once()
    assert result == [[0.1, 0.2], [0.3, 0.4]]


def test_get_openai_embeddings_batch_includes_dimensions(monkeypatch):
    session = _mock_session_with_response(
        [
            {"embedding": [0.1, 0.2]},
            {"embedding": [0.3, 0.4]},
        ]
    )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.chat_calls.create_session_with_retries",
        lambda **kwargs: session,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.chat_calls.load_and_log_configs",
        lambda: {
            "openai_api": {
                "api_key": "test-key",
            }
        },
    )

    get_openai_embeddings_batch(["a", "b"], "text-embedding-3-small", dimensions=256)

    payload = session.post.call_args[1]["json"]
    assert payload.get("dimensions") == 256


def test_get_openai_embeddings_batch_forwards_configured_org_and_project_headers(monkeypatch):
    session = _mock_session_with_response(
        [
            {"embedding": [0.1, 0.2]},
            {"embedding": [0.3, 0.4]},
        ]
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.chat_calls.create_session_with_retries",
        lambda **kwargs: session,
    )

    get_openai_embeddings_batch(
        ["first", "second"],
        "text-embedding-3-small",
        app_config={
            "openai_api": {
                "api_key": "test-key",
                "organization": "org-batch",
                "project": "project-batch",
            }
        },
    )

    headers = session.post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["OpenAI-Organization"] == "org-batch"
    assert headers["OpenAI-Project"] == "project-batch"


def test_get_openai_embeddings_respects_api_base(monkeypatch):
    session = _mock_session_with_response([{"embedding": [0.5, 0.6]}])

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.chat_calls.create_session_with_retries",
        lambda **kwargs: session,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.chat_calls.load_and_log_configs",
        lambda: {
            "openai_api": {
                "api_key": "test-key",
                "api_base_url": "https://custom.openai.local/v1",
            }
        },
    )

    get_openai_embeddings("hello", "text-embedding-3-small")

    session.post.assert_called_once()
    called_url = session.post.call_args[0][0]
    assert called_url == "https://custom.openai.local/v1/embeddings"


def test_get_openai_embeddings_batch_respects_api_base(monkeypatch):
    session = _mock_session_with_response(
        [
            {"embedding": [0.1]},
            {"embedding": [0.2]},
        ]
    )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.chat_calls.create_session_with_retries",
        lambda **kwargs: session,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.chat_calls.load_and_log_configs",
        lambda: {
            "openai_api": {
                "api_key": "test-key",
                "api_base_url": "https://custom.openai.local/api/v1",
            }
        },
    )

    get_openai_embeddings_batch(["a", "b"], "text-embedding-3-small")

    session.post.assert_called_once()
    called_url = session.post.call_args[0][0]
    assert called_url == "https://custom.openai.local/api/v1/embeddings"


def test_chat_with_openai_logs_payload_metadata(monkeypatch):
    # Patch OpenAI adapter's HTTP client to capture payload and return 400
    fake_client = _FakeClient(
        post_resp=_FakeResp(status_code=400, json_obj={}, text="bad request")
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter.http_client_factory",
        lambda *args, **kwargs: fake_client,
    )

    with pytest.raises(ChatBadRequestError):
        perform_chat_api_call(
            api_provider="openai",
            messages=[{"role": "user", "content": "hi"}],
            api_key="key",
            model="gpt-4o-mini",
            streaming=False,
        )

    # Ensure payload included stream flag and did not hit network
    assert isinstance(fake_client.last_json, dict)
    assert fake_client.last_json.get("stream") is False


def _patch_async_client(monkeypatch, lines):
    stream = _DummyStream(lines)

    monkeypatch.setattr(
        "tldw_Server_API.app.core.http_client.create_async_client",
        lambda *args, **kwargs: _DummyAsyncClient(stream),
    )


@pytest.mark.asyncio
async def test_chat_with_openai_async_no_duplicate_done(monkeypatch):
    # Patch OpenAI adapter to stream two lines including [DONE]
    fake_client = _FakeClient(
        stream_lines=[
            'data: {"choices":[{"delta":{"content":"hi"}}]}',
            "data: [DONE]",
        ]
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter.http_client_factory",
        lambda *args, **kwargs: fake_client,
    )

    stream = await perform_chat_api_call_async(
        api_provider="openai",
        messages=[{"role": "user", "content": "hi"}], api_key="key", model="gpt-4o-mini", streaming=True
    )
    chunks = [chunk async for chunk in stream]

    assert chunks.count(sse_done()) == 1
    assert len(chunks) == 2


@pytest.mark.asyncio
async def test_chat_with_groq_async_no_duplicate_done(monkeypatch):
    fake_client = _FakeClient(
        stream_lines=[
            'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
            "data: [DONE]\n\n",
        ]
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.groq_adapter.http_client_factory",
        lambda *args, **kwargs: fake_client,
    )

    stream = await perform_chat_api_call_async(
        api_provider="groq",
        messages=[{"role": "user", "content": "hi"}], api_key="key", model="llama-3.3-70b-versatile", streaming=True
    )
    chunks = [chunk async for chunk in stream]

    assert chunks.count(sse_done()) == 1
    assert len(chunks) == 2


@pytest.mark.asyncio
async def test_chat_with_openrouter_async_no_duplicate_done(monkeypatch):
    fake_client = _FakeClient(
        stream_lines=[
            'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
            "data: [DONE]\n\n",
        ]
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.openrouter_adapter.http_client_factory",
        lambda *args, **kwargs: fake_client,
    )

    stream = await perform_chat_api_call_async(
        api_provider="openrouter",
        messages=[{"role": "user", "content": "hi"}], api_key="key", model="mistralai/mistral-7b-instruct:free", streaming=True
    )
    chunks = [chunk async for chunk in stream]

    assert chunks.count(sse_done()) == 1
    assert len(chunks) == 2


@pytest.mark.asyncio
async def test_chat_with_openai_async_retries_request_error(monkeypatch):
    # With adapter path, retries are handled in http layer; simulate a fatal error and assert mapping
    class _ErrClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def stream(self, *args, **kwargs):
            raise httpx.RequestError(
                "boom", request=httpx.Request("POST", "https://retry.test")
            )

        def post(self, *args, **kwargs):
            raise AssertionError(
                "Non-streaming POST should not be invoked in this test."
            )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter.http_client_factory",
        lambda *args, **kwargs: _ErrClient(),
    )

    with pytest.raises(ChatProviderError):
        stream = await perform_chat_api_call_async(
            api_provider="openai",
            messages=[{"role": "user", "content": "hi"}],
            api_key="key",
            model="gpt-4o-mini",
            streaming=True,
        )
        # Exhaust the iterator to trigger the exception path (should raise immediately)
        _ = [chunk async for chunk in stream]


@pytest.mark.asyncio
async def test_chat_with_openai_async_non_streaming_exhausts_retries(monkeypatch):
    class _FailPostClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, *args, **kwargs):
            raise httpx.RequestError(
                "boom-1", request=httpx.Request("POST", "https://retry.test")
            )

        def stream(self, *args, **kwargs):
            raise AssertionError("Stream should not be used in this test.")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter.http_client_factory",
        lambda *args, **kwargs: _FailPostClient(),
    )

    with pytest.raises(ChatProviderError):
        await perform_chat_api_call_async(
            api_provider="openai",
            messages=[{"role": "user", "content": "hi"}],
            api_key="key",
            model="gpt-4o-mini",
            streaming=False,
        )


@pytest.mark.asyncio
async def test_chat_with_anthropic_async_stream_tool_calls(monkeypatch):
    # Patch Anthropic adapter client to emit tool_use events
    fake_client = _FakeClient(
        stream_lines=[
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tool_1","name":"lookup","input":{}}}',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"city\\":\\"Paris\\"}"}}',
            'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"}}',
        ]
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter.http_client_factory",
        lambda *args, **kwargs: fake_client,
    )

    stream = await perform_chat_api_call_async(
        api_provider="anthropic",
        messages=[{"role": "user", "content": "Hi"}], api_key="key", model="claude-3-5-sonnet-latest", streaming=True
    )
    chunks = [chunk async for chunk in stream]

    assert any('"tool_calls"' in chunk for chunk in chunks)
    assert chunks.count(sse_done()) == 1
