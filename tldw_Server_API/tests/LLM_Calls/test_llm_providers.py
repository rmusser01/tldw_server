"""
Tests for newly added LLM providers (Moonshot AI, Z.AI, HuggingFace API).

Updated to use the canonical ``perform_chat_api_call`` / ``perform_chat_api_call_async``
entry-points from ``chat_service`` and to mock each provider adapter's
``http_client_factory`` instead of the deleted ``chat_with_*`` wrappers.
"""

import pytest
import asyncio
import json
from typing import Optional, Dict, Any
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
import httpx
import requests
from requests.structures import CaseInsensitiveDict

# Import the modules to test
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatRateLimitError,
    ChatProviderError,
)
from tldw_Server_API.app.core.Chat.chat_service import (
    perform_chat_api_call,
    perform_chat_api_call_async,
)
from tldw_Server_API.app.core.LLM_Calls.huggingface_api import (
    HuggingFaceAPI,
    find_best_gguf_model,
    download_gguf_model
)


# ---------------------------------------------------------------------------
# Shared fake helpers for mocking http_client_factory (httpx-based adapters)
# ---------------------------------------------------------------------------

class _FakeResp:
    """Mimics a httpx.Response returned by client.post(...)."""

    def __init__(self, status_code: int, body: Any = None, *, text: str = ""):
        self.status_code = status_code
        self._body = body
        self._text = text

    # Non-streaming interface
    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            req = httpx.Request("POST", "https://example.com/api")
            resp = httpx.Response(
                self.status_code, request=req,
                content=json.dumps(self._body or {}).encode(),
            )
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=req, response=resp,
            )

    @property
    def text(self):
        return self._text or json.dumps(self._body or {})

    def close(self):
        return None

    # Context-manager support (used in streaming paths)
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    # Streaming helpers
    def iter_lines(self):
        return iter([])


class _FakeClient:
    """Mimics the httpx.Client context-manager returned by http_client_factory."""

    def __init__(self, response: _FakeResp):
        self._response = response
        self.last_json: Optional[dict] = None
        self.post_called = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, url, *, headers=None, json=None):
        self.last_json = json
        self.post_called += 1
        return self._response


class _FakeStreamResp:
    """Stream-response context-manager with iter_lines support."""

    def __init__(self, lines):
        self._lines = lines
        self.status_code = 200

    def raise_for_status(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_lines(self):
        return iter(self._lines)

    def close(self):
        return None


class _FakeStreamClient:
    """Client context-manager whose .stream() returns _FakeStreamResp."""

    def __init__(self, lines):
        self._lines = lines
        self.last_json: Optional[dict] = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, url, *, headers=None, json=None):
        self.last_json = json
        resp = _FakeStreamResp(self._lines)
        return resp

    def stream(self, method, url, *, headers=None, json=None):
        self.last_json = json
        return _FakeStreamResp(self._lines)


# ---------------------------------------------------------------------------
# Mock target constants -- moonshot/zai/cohere still use requests-based
# create_session_with_retries, while other adapters use httpx http_client_factory.
# ---------------------------------------------------------------------------
_MOCK_CREATE_SESSION = "tldw_Server_API.app.core.LLM_Calls.chat_calls.create_session_with_retries"


def make_response(status_code: int, body: str = "", headers: Optional[dict] = None) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = body.encode("utf-8")
    response.headers = CaseInsensitiveDict(headers or {})
    response.url = "https://example.com/api"
    return response


def _mock_async_client() -> AsyncMock:
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = None
    return client


# ---------------------------------------------------------------------------
# Moonshot provider tests
# ---------------------------------------------------------------------------

class TestMoonshotProvider:
    """Tests for Moonshot AI provider."""

    @pytest.fixture
    def mock_response(self):
        """Mock response for Moonshot API."""
        return {
            "id": "cmpl-test123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "moonshot-v1-8k",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello from Moonshot AI!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }

    def test_moonshot_basic_chat(self, mock_response, monkeypatch):
        """Test basic chat functionality."""
        fake_session = Mock()
        mock_response_obj = Mock()
        mock_response_obj.status_code = 200
        mock_response_obj.json.return_value = mock_response
        mock_response_obj.raise_for_status = Mock()
        mock_response_obj.close = Mock()
        fake_session.post.return_value = mock_response_obj
        fake_session.close = Mock()
        monkeypatch.setattr(_MOCK_CREATE_SESSION, lambda **kw: fake_session)

        result = perform_chat_api_call(
            api_provider="moonshot",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="test_key",
            model="moonshot-v1-8k",
        )

        assert result["choices"][0]["message"]["content"] == "Hello from Moonshot AI!"
        fake_session.post.assert_called_once()

        # Check request payload
        call_args = fake_session.post.call_args
        payload = call_args[1]['json']
        assert payload['model'] == "moonshot-v1-8k"
        assert len(payload['messages']) == 1

    def test_moonshot_with_system_message(self, mock_response, monkeypatch):
        """Test chat with system message."""
        fake_session = Mock()
        mock_response_obj = Mock()
        mock_response_obj.status_code = 200
        mock_response_obj.json.return_value = mock_response
        mock_response_obj.raise_for_status = Mock()
        mock_response_obj.close = Mock()
        fake_session.post.return_value = mock_response_obj
        fake_session.close = Mock()
        monkeypatch.setattr(_MOCK_CREATE_SESSION, lambda **kw: fake_session)

        result = perform_chat_api_call(
            api_provider="moonshot",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="test_key",
            system_message="You are a helpful assistant.",
            model="moonshot-v1-8k",
        )

        call_args = fake_session.post.call_args
        payload = call_args[1]['json']
        assert payload['messages'][0]['role'] == "system"
        assert payload['messages'][0]['content'] == "You are a helpful assistant."

    def test_moonshot_vision_model(self, mock_response, monkeypatch):
        """Test vision model with image content."""
        mock_response['model'] = "moonshot-v1-8k-vision-preview"
        fake_session = Mock()
        mock_response_obj = Mock()
        mock_response_obj.status_code = 200
        mock_response_obj.json.return_value = mock_response
        mock_response_obj.raise_for_status = Mock()
        mock_response_obj.close = Mock()
        fake_session.post.return_value = mock_response_obj
        fake_session.close = Mock()
        monkeypatch.setattr(_MOCK_CREATE_SESSION, lambda **kw: fake_session)

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
            ]
        }]

        result = perform_chat_api_call(
            api_provider="moonshot",
            messages=messages,
            api_key="test_key",
            model="moonshot-v1-8k-vision-preview",
        )

        assert result["choices"][0]["message"]["content"] == "Hello from Moonshot AI!"
        call_args = fake_session.post.call_args
        payload = call_args[1]['json']
        assert payload['model'] == "moonshot-v1-8k-vision-preview"

    def test_moonshot_streaming(self, monkeypatch):
        """Test streaming response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'text/event-stream'}
        mock_response.raise_for_status = Mock()
        mock_response.iter_lines = Mock(return_value=[
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" from"}}]}',
            'data: {"choices":[{"delta":{"content":" Moonshot!"}}]}',
            'data: [DONE]',
        ])
        mock_response.close = Mock()
        fake_session = Mock()
        fake_session.post.return_value = mock_response
        fake_session.close = Mock()
        monkeypatch.setattr(_MOCK_CREATE_SESSION, lambda **kw: fake_session)

        chunks = []
        result_gen = perform_chat_api_call(
            api_provider="moonshot",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="test_key",
            streaming=True,
            model="moonshot-v1-8k",
        )

        for chunk in result_gen:
            chunks.append(chunk)

        assert len(chunks) == 4  # 3 content chunks + [DONE]
        assert "[DONE]" in chunks[-1]

    def test_moonshot_streaming_session_lifecycle(self, monkeypatch):
        """Ensure streaming keeps the session open until iteration finishes."""
        session_state = {"closed": False}
        response_state = {"closed": False}

        session_instance = MagicMock()

        def close_session():
            session_state["closed"] = True

        session_instance.close.side_effect = close_session

        response = MagicMock()
        response.status_code = 200
        response.raise_for_status = Mock()

        def close_response():
            response_state["closed"] = True

        response.close.side_effect = close_response

        def iter_lines(decode_unicode=False):
            def generator():
                if session_state["closed"]:
                    raise AssertionError("Session closed before yielding first chunk")
                yield 'data: {"choices":[{"delta":{"content":"Hello"}}]}'
                if session_state["closed"]:
                    raise AssertionError("Session closed before completion")
                yield 'data: [DONE]'
            return generator()

        response.iter_lines.side_effect = iter_lines
        session_instance.post.return_value = response
        monkeypatch.setattr(_MOCK_CREATE_SESSION, lambda **kw: session_instance)

        generator = perform_chat_api_call(
            api_provider="moonshot",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="test_key",
            streaming=True,
            model="moonshot-v1-8k",
        )

        first_chunk = next(generator)
        assert "Hello" in first_chunk
        assert session_state["closed"] is False
        assert response_state["closed"] is False

        remaining_chunks = list(generator)
        assert any("[DONE]" in chunk for chunk in remaining_chunks)
        assert session_state["closed"] is True
        assert response_state["closed"] is True

    def test_moonshot_error_handling(self, monkeypatch):
        """Test error handling."""
        fake_session = Mock()
        response = make_response(401, '{"error": {"message": "Unauthorized"}}')
        fake_session.post.return_value = response
        fake_session.close = Mock()
        monkeypatch.setattr(_MOCK_CREATE_SESSION, lambda **kw: fake_session)

        with pytest.raises(ChatAuthenticationError) as exc_info:
            _ = perform_chat_api_call(
                api_provider="moonshot",
                messages=[{"role": "user", "content": "Hello"}],
                api_key="invalid_key",
                model="moonshot-v1-8k",
            )

        assert "Unauthorized" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Z.AI provider tests
# ---------------------------------------------------------------------------

class TestZAIProvider:
    """Tests for Z.AI provider."""

    @pytest.fixture
    def mock_response(self):
        """Mock response for Z.AI API."""
        return {
            "id": "chat-test123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "glm-4.5",
            "request_id": "req_test123",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello from Z.AI GLM!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }

    def test_zai_basic_chat(self, mock_response, monkeypatch):
        """Test basic chat functionality."""
        fake_session = Mock()
        mock_response_obj = Mock()
        mock_response_obj.status_code = 200
        mock_response_obj.json.return_value = mock_response
        mock_response_obj.raise_for_status = Mock()
        mock_response_obj.close = Mock()
        fake_session.post.return_value = mock_response_obj
        fake_session.close = Mock()
        monkeypatch.setattr(_MOCK_CREATE_SESSION, lambda **kw: fake_session)

        result = perform_chat_api_call(
            api_provider="zai",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="test_key",
            model="glm-4.5",
        )

        assert result["choices"][0]["message"]["content"] == "Hello from Z.AI GLM!"
        # Ensure our session post was invoked once
        fake_session.post.assert_called_once()

        # Check request payload
        call_args = fake_session.post.call_args
        payload = call_args[1]['json']
        assert payload['model'] == "glm-4.5"

    def test_zai_with_request_id(self, mock_response, monkeypatch):
        """Test chat with request_id passed through to the adapter."""
        fake_session = Mock()
        mock_response_obj = Mock()
        mock_response_obj.status_code = 200
        mock_response_obj.json.return_value = mock_response
        mock_response_obj.raise_for_status = Mock()
        mock_response_obj.close = Mock()
        fake_session.post.return_value = mock_response_obj
        fake_session.close = Mock()
        monkeypatch.setattr(_MOCK_CREATE_SESSION, lambda **kw: fake_session)

        result = perform_chat_api_call(
            api_provider="zai",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="test_key",
            request_id="custom_req_123",
            model="glm-4.5",
        )

        call_args = fake_session.post.call_args
        payload = call_args[1]['json']
        assert payload.get('request_id') == "custom_req_123"

    def test_zai_model_variants(self, mock_response, monkeypatch):
        """Test different model variants."""
        models = ["glm-4.5", "glm-4.5-air", "glm-4.5-flash", "glm-4-32b-0414-128k"]

        for model in models:
            mock_response['model'] = model
            fake_session = Mock()
            mock_response_obj = Mock()
            mock_response_obj.status_code = 200
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status = Mock()
            mock_response_obj.close = Mock()
            fake_session.post.return_value = mock_response_obj
            fake_session.close = Mock()
            monkeypatch.setattr(_MOCK_CREATE_SESSION, lambda **kw: fake_session)

            result = perform_chat_api_call(
                api_provider="zai",
                messages=[{"role": "user", "content": "Test"}],
                api_key="test_key",
                model=model,
            )

            call_args = fake_session.post.call_args
            payload = call_args[1]['json']
            assert payload['model'] == model

    def test_zai_streaming(self, monkeypatch):
        """Test streaming response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'text/event-stream'}
        mock_response.raise_for_status = Mock()
        mock_response.iter_lines = Mock(return_value=[
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" GLM"}}]}',
            'data: [DONE]',
        ])
        mock_response.close = Mock()
        fake_session = Mock()
        fake_session.post.return_value = mock_response
        fake_session.close = Mock()
        monkeypatch.setattr(_MOCK_CREATE_SESSION, lambda **kw: fake_session)

        chunks = []
        result_gen = perform_chat_api_call(
            api_provider="zai",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="test_key",
            streaming=True,
            model="glm-4.5",
        )

        for chunk in result_gen:
            chunks.append(chunk)

        assert len(chunks) == 3
        assert "[DONE]" in chunks[-1]


# ---------------------------------------------------------------------------
# Parametrized HTTP error mapping across providers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "provider, kwargs, adapter_module, status_code, expected_exception",
    [
        (
            "openrouter",
            {
                "messages": [{"role": "user", "content": "Hi"}],
                "api_key": "test_key",
                "model": "mistralai/mistral-7b-instruct:free",
            },
            "openrouter_adapter",
            429,
            ChatRateLimitError,
        ),
        (
            "mistral",
            {"messages": [{"role": "user", "content": "Hi"}], "api_key": "test_key", "model": "mistral-small"},
            "mistral_adapter",
            401,
            ChatAuthenticationError,
        ),
        (
            "deepseek",
            {"messages": [{"role": "user", "content": "Hi"}], "api_key": "test_key", "model": "deepseek-chat"},
            "deepseek_adapter",
            400,
            ChatBadRequestError,
        ),
        (
            "google",
            {"messages": [{"role": "user", "content": "Hi"}], "api_key": "test_key", "model": "gemini-2.5-flash"},
            "google_adapter",
            503,
            ChatProviderError,
        ),
    ],
)
def test_provider_http_error_mapping(
    provider, kwargs, adapter_module, status_code, expected_exception, monkeypatch,
):
    mock_target = f"tldw_Server_API.app.core.LLM_Calls.providers.{adapter_module}.http_client_factory"

    mock_client = MagicMock()
    mock_client.__enter__ = Mock(return_value=mock_client)
    mock_client.__exit__ = Mock(return_value=None)
    request = httpx.Request("POST", "https://example.com/api")
    http_response = httpx.Response(
        status_code=status_code,
        request=request,
        content=b'{"error": {"message": "boom"}}',
    )
    mock_client.post.return_value = http_response
    monkeypatch.setattr(mock_target, lambda *a, **k: mock_client)

    with pytest.raises(expected_exception):
        perform_chat_api_call(api_provider=provider, **kwargs)


# ---------------------------------------------------------------------------
# HuggingFace API tests (unchanged - these don't use chat_with_* wrappers)
# ---------------------------------------------------------------------------

class TestHuggingFaceAPI:
    """Tests for HuggingFace API client."""

    @pytest.fixture
    def api_client(self):
        """Create HuggingFace API client."""
        return HuggingFaceAPI(token="test_token")

    @pytest.fixture
    def mock_model_response(self):
        """Mock model search response."""
        return [
            {
                "modelId": "TheBloke/Llama-2-7B-GGUF",
                "author": "TheBloke",
                "downloads": 100000,
                "likes": 500,
                "tags": ["gguf", "llama"]
            },
            {
                "modelId": "TheBloke/Mistral-7B-GGUF",
                "author": "TheBloke",
                "downloads": 50000,
                "likes": 300,
                "tags": ["gguf", "mistral"]
            }
        ]

    @pytest.mark.asyncio
    async def test_search_models(self, api_client, mock_model_response):
        """Test model search functionality."""
        mock_client = _mock_async_client()
        mock_response = Mock()
        mock_response.json = Mock(return_value=mock_model_response)
        mock_response.raise_for_status = Mock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch('tldw_Server_API.app.core.LLM_Calls.huggingface_api.create_async_client', return_value=mock_client):
            results = await api_client.search_models(
                query="llama",
                filter_tags=["gguf"],
                limit=10
            )

            assert len(results) == 2
            assert results[0]["modelId"] == "TheBloke/Llama-2-7B-GGUF"

    @pytest.mark.asyncio
    async def test_get_model_info(self, api_client):
        """Test getting model information."""
        mock_info = {
            "modelId": "TheBloke/Llama-2-7B-GGUF",
            "author": "TheBloke",
            "downloads": 100000,
            "description": "Llama 2 7B model in GGUF format"
        }

        mock_client = _mock_async_client()
        mock_response = Mock()
        mock_response.json = Mock(return_value=mock_info)
        mock_response.raise_for_status = Mock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch('tldw_Server_API.app.core.LLM_Calls.huggingface_api.create_async_client', return_value=mock_client):
            info = await api_client.get_model_info("TheBloke/Llama-2-7B-GGUF")

            assert info["modelId"] == "TheBloke/Llama-2-7B-GGUF"
            assert info["downloads"] == 100000

    @pytest.mark.asyncio
    async def test_list_model_files(self, api_client):
        """Test listing GGUF files in a repository."""
        mock_files = [
            {"path": "llama-2-7b.Q4_K_M.gguf", "size": 3825000000},
            {"path": "llama-2-7b.Q5_K_S.gguf", "size": 4650000000},
            {"path": "README.md", "size": 5000}
        ]

        mock_client = _mock_async_client()
        mock_response = Mock()
        mock_response.json = Mock(return_value=mock_files)
        mock_response.raise_for_status = Mock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch('tldw_Server_API.app.core.LLM_Calls.huggingface_api.create_async_client', return_value=mock_client):
            files = await api_client.list_model_files("TheBloke/Llama-2-7B-GGUF")

            # Should only return GGUF files
            assert len(files) == 2
            assert all(f["path"].endswith(".gguf") for f in files)

    @pytest.mark.asyncio
    async def test_download_file(self, api_client, tmp_path):
        """Test file download functionality."""
        test_content = b"This is a test GGUF file content"

        mock_client = _mock_async_client()
        # Mock HEAD request for file size
        mock_head_response = Mock()
        mock_head_response.headers = {"content-length": str(len(test_content))}
        mock_client.head = AsyncMock(return_value=mock_head_response)

        # Mock streaming download
        mock_stream_response = Mock()
        mock_stream_response.raise_for_status = Mock()
        async def _aiter_bytes(chunk_size=65536):
            yield test_content
        mock_stream_response.aiter_bytes = Mock(return_value=_aiter_bytes())
        # Configure async context manager behavior
        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_stream_response)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=None)
        mock_client.stream = Mock(return_value=mock_stream_cm)

        with patch('tldw_Server_API.app.core.LLM_Calls.huggingface_api.create_async_client', return_value=mock_client):
            destination = tmp_path / "test_model.gguf"
            success = await api_client.download_file(
                repo_id="TheBloke/Test-GGUF",
                filename="test.gguf",
                destination=destination
            )

            assert success
            assert destination.exists()
            assert destination.read_bytes() == test_content

    @pytest.mark.asyncio
    async def test_download_with_progress(self, api_client, tmp_path):
        """Test download with progress callback."""
        test_content = b"0" * 1000
        progress_calls = []

        def progress_callback(downloaded, total):
            progress_calls.append((downloaded, total))

        mock_client = _mock_async_client()
        mock_head_response = Mock()
        mock_head_response.headers = {"content-length": "1000"}
        mock_client.head = AsyncMock(return_value=mock_head_response)

        mock_stream_response = Mock()
        mock_stream_response.raise_for_status = Mock()
        # Simulate chunked download
        async def _aiter_bytes2(chunk_size=65536):
            yield test_content[:500]
            yield test_content[500:]
        mock_stream_response.aiter_bytes = Mock(return_value=_aiter_bytes2())
        # Configure async context manager behavior
        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_stream_response)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=None)
        mock_client.stream = Mock(return_value=mock_stream_cm)

        with patch('tldw_Server_API.app.core.LLM_Calls.huggingface_api.create_async_client', return_value=mock_client):
            destination = tmp_path / "test_model.gguf"
            success = await api_client.download_file(
                repo_id="TheBloke/Test-GGUF",
                filename="test.gguf",
                destination=destination,
                progress_callback=progress_callback
            )

            assert success
            assert len(progress_calls) == 2
            assert progress_calls[-1] == (1000, 1000)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "filename",
        [
            "../evil.gguf",
            "nested/model.gguf",
            "model.bin",
        ],
    )
    async def test_download_gguf_model_rejects_invalid_filename(self, monkeypatch, tmp_path, filename):
        calls = []

        async def fake_download_file(self, **kwargs):
            calls.append(kwargs)
            return True

        monkeypatch.setattr(HuggingFaceAPI, "download_file", fake_download_file)

        with pytest.raises(ValueError):
            await download_gguf_model("TheBloke/Test-GGUF", filename, tmp_path)

        assert calls == []

    @pytest.mark.asyncio
    async def test_download_gguf_model_accepts_simple_gguf_filename(self, monkeypatch, tmp_path):
        calls = []

        async def fake_download_file(self, **kwargs):
            calls.append(kwargs)
            return True

        monkeypatch.setattr(HuggingFaceAPI, "download_file", fake_download_file)

        assert await download_gguf_model("TheBloke/Test-GGUF", "model.Q4_K_M.gguf", tmp_path)
        assert calls[0]["filename"] == "model.Q4_K_M.gguf"
        assert calls[0]["destination"] == tmp_path / "model.Q4_K_M.gguf"

    @pytest.mark.asyncio
    async def test_find_best_gguf_model(self):
        """Test finding best GGUF model utility."""
        mock_models = [
            {
                "modelId": "TheBloke/Llama-2-7B-GGUF",
                "downloads": 100000
            }
        ]

        with patch('tldw_Server_API.app.core.LLM_Calls.huggingface_api.HuggingFaceAPI.search_gguf_models') as mock_search:
            async def mock_search_async(*args, **kwargs):
                return mock_models
            mock_search.side_effect = mock_search_async

            best_model = await find_best_gguf_model(
                model_name="llama-2",
                max_size_gb=10.0,
                preferred_quant="Q4_K_M"
            )

            assert best_model["modelId"] == "TheBloke/Llama-2-7B-GGUF"

    @pytest.mark.asyncio
    async def test_error_handling(self, api_client):
        """Test error handling in API calls."""
        mock_client = _mock_async_client()
        mock_client.get = AsyncMock(side_effect=httpx.HTTPError("Connection error"))
        with patch('tldw_Server_API.app.core.LLM_Calls.huggingface_api.create_async_client', return_value=mock_client):
            results = await api_client.search_models(query="test")
            assert results == []  # Should return empty list on error

            info = await api_client.get_model_info("test/model")
            assert info is None  # Should return None on error


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:
    """Integration tests for provider interactions."""

    @pytest.mark.asyncio
    async def test_provider_switching(self, monkeypatch):
        """Test switching between different providers."""
        moonshot_response = {
            "choices": [{"message": {"content": "Moonshot response"}}]
        }
        zai_response = {
            "choices": [{"message": {"content": "Z.AI response"}}]
        }

        # Both moonshot and zai use create_session_with_retries; we swap
        # the mock return between calls.
        fake_session = Mock()
        mock_response_obj = Mock()
        mock_response_obj.status_code = 200
        mock_response_obj.raise_for_status = Mock()
        mock_response_obj.close = Mock()
        fake_session.post.return_value = mock_response_obj
        fake_session.close = Mock()
        monkeypatch.setattr(_MOCK_CREATE_SESSION, lambda **kw: fake_session)

        # Test Moonshot
        mock_response_obj.json.return_value = moonshot_response
        result1 = perform_chat_api_call(
            api_provider="moonshot",
            messages=[{"role": "user", "content": "Test"}],
            api_key="key1",
            model="moonshot-v1-8k",
        )
        assert result1["choices"][0]["message"]["content"] == "Moonshot response"

        # Test Z.AI
        mock_response_obj.json.return_value = zai_response
        result2 = perform_chat_api_call(
            api_provider="zai",
            messages=[{"role": "user", "content": "Test"}],
            api_key="key2",
            model="glm-4.5",
        )
        assert result2["choices"][0]["message"]["content"] == "Z.AI response"

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, monkeypatch):
        """Test concurrent requests to multiple providers."""
        mock_response_obj = Mock()
        mock_response_obj.status_code = 200
        mock_response_obj.json.return_value = {
            "choices": [{"message": {"content": "Response"}}]
        }
        mock_response_obj.raise_for_status = Mock()
        mock_response_obj.close = Mock()

        def _make_session(**kw):
            s = Mock()
            s.post.return_value = mock_response_obj
            s.close = Mock()
            return s

        monkeypatch.setattr(_MOCK_CREATE_SESSION, _make_session)

        # Simulate concurrent requests
        tasks = [
            asyncio.create_task(
                asyncio.to_thread(
                    perform_chat_api_call,
                    api_provider="moonshot",
                    messages=[{"role": "user", "content": f"Test {i}"}],
                    api_key="key",
                    model="moonshot-v1-8k",
                )
            )
            for i in range(5)
        ]
        results = await asyncio.gather(*tasks)
        assert len(results) == 5
        assert all(r["choices"][0]["message"]["content"] == "Response" for r in results)


# ---------------------------------------------------------------------------
# SSE normalization tests across providers
# ---------------------------------------------------------------------------

class TestSSENormalization:
    """Tests for SSE normalization across providers (Cohere, Qwen, Groq)."""

    def test_cohere_stream_normalized(self, monkeypatch):
        stream_lines = [
            b'data: {"event_type":"stream-start","generation_id":"gen-1"}',
            b'data: {"event_type":"text-generation","text":"Hello"}',
            b'data: {"event_type":"text-generation","text":" world"}',
            b'data: {"event_type":"stream-end","finish_reason":"end_turn"}',
        ]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.iter_lines = Mock(return_value=stream_lines)
        mock_response.close = Mock()

        session = Mock()
        session.post.return_value = mock_response
        session.close = Mock()
        monkeypatch.setattr(_MOCK_CREATE_SESSION, lambda **kw: session)

        gen = perform_chat_api_call(
            api_provider="cohere",
            messages=[{"role": "user", "content": "Hi"}],
            api_key="test", model="command-r", streaming=True,
        )
        chunks = list(gen)
        # Expect two content chunks + DONE
        assert len(chunks) == 3
        assert chunks[0].startswith('data: ')
        assert chunks[0].endswith('\n\n')
        assert chunks[1].startswith('data: ')
        assert chunks[1].endswith('\n\n')
        assert '[DONE]' in chunks[-1]

    def test_cohere_stream_session_lifecycle(self, monkeypatch):
        stream_lines = [
            b'data: {"event_type":"text-generation","text":"Hello"}',
            b'data: {"event_type":"stream-end","finish_reason":"end_turn"}',
        ]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.iter_lines = Mock(return_value=stream_lines)
        mock_response.close = Mock()

        session = Mock()
        session.post.return_value = mock_response
        session.close = Mock()

        monkeypatch.setattr(_MOCK_CREATE_SESSION, lambda **kw: session)

        gen = perform_chat_api_call(
            api_provider="cohere",
            messages=[{"role": "user", "content": "Hi"}],
            api_key="test",
            model="command-r",
            streaming=True,
        )

        session.close.assert_not_called()
        first_chunk = next(gen)
        session.close.assert_not_called()

        remaining = list(gen)
        session.close.assert_called_once()
        assert mock_response.close.called
        assert first_chunk.startswith("data: ")

    def test_qwen_stream_normalized(self, monkeypatch):

        class _Client:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def stream(self, *args, **kwargs):
                class _Resp:
                    status_code = 200
                    def raise_for_status(self): return None
                    def __enter__(self): return self
                    def __exit__(self, exc_type, exc, tb): return False
                    def iter_lines(self):
                        return iter([
                            'data: {"choices":[{"delta":{"content":"Hi"}}]}',
                            'data: {"choices":[{"delta":{"content":"!"}}]}',
                        ])
                return _Resp()
        monkeypatch.setattr(
            "tldw_Server_API.app.core.LLM_Calls.providers.qwen_adapter.http_client_factory",
            lambda *a, **k: _Client(),
        )

        gen = perform_chat_api_call(
            api_provider="qwen",
            messages=[{"role": "user", "content": "Hi"}],
            api_key="test", model="qwen-plus", streaming=True,
        )
        chunks = list(gen)
        # 2 content chunks + normalized DONE
        assert len(chunks) == 3
        assert all(c.startswith('data: ') for c in chunks)
        assert all(c.endswith('\n\n') for c in chunks[:-1])
        assert '[DONE]' in chunks[-1]

    def test_groq_stream_normalized(self, monkeypatch):

        class _Client:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def stream(self, *args, **kwargs):
                class _Resp:
                    status_code = 200
                    def raise_for_status(self):
                        return None
                    def __enter__(self):
                        return self
                    def __exit__(self, exc_type, exc, tb):
                        return False
                    def iter_lines(self):
                        return iter([
                            'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
                            'data: {"choices":[{"delta":{"content":" Groq"}}]}\n\n',
                            'data: [DONE]\n\n',
                        ])
                return _Resp()
        monkeypatch.setattr(
            "tldw_Server_API.app.core.LLM_Calls.providers.groq_adapter.http_client_factory",
            lambda *args, **kwargs: _Client(),
        )

        gen = perform_chat_api_call(
            api_provider="groq",
            messages=[{"role": "user", "content": "Hi"}],
            api_key="test", model="llama-3.3-70b-versatile", streaming=True,
        )
        chunks = list(gen)
        assert len(chunks) == 3
        assert chunks[0].startswith('data: ')
        assert chunks[-1].strip() == 'data: [DONE]'

    def test_google_gemini_stream_normalized(self, monkeypatch):

        class _Client:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def stream(self, method, url, *, headers=None, json=None):
                class _Resp:
                    status_code = 200
                    def raise_for_status(self):
                        return None
                    def __enter__(self):
                        return self
                    def __exit__(self, exc_type, exc, tb):
                        return False
                    def iter_lines(self):
                        return iter([
                            b'data: {"candidates":[{"content":{"parts":[{"text":"Hello"}]}}]}',
                            b'data: {"candidates":[{"content":{"parts":[{"text":" Gemini"}]}}]}',
                        ])
                    def close(self):
                        return None
                return _Resp()
        monkeypatch.setattr(
            "tldw_Server_API.app.core.LLM_Calls.providers.google_adapter.http_client_factory",
            lambda *a, **k: _Client(),
        )

        gen = perform_chat_api_call(
            api_provider="google",
            messages=[{"role": "user", "content": "Hi"}],
            api_key="test",
            model="gemini-2.5-flash",
            streaming=True,
        )
        chunks = list(gen)
        # Expect two content chunks + DONE
        assert len(chunks) == 3
        assert chunks[0].startswith('data: ')
        assert chunks[0].endswith('\n\n')
        assert '[DONE]' in chunks[-1]

    def test_google_gemini_stream_tool_calls(self, monkeypatch):

        tool_chunk = (
            b'data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"lookup","args":{"query":"mars"}}}]},'
            b'"finishReason":"FUNCTION_CALL"}]}'
        )
        class _Client:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def stream(self, method, url, *, headers=None, json=None):
                class _Resp:
                    status_code = 200
                    def raise_for_status(self):
                        return None
                    def __enter__(self):
                        return self
                    def __exit__(self, exc_type, exc, tb):
                        return False
                    def iter_lines(self):
                        return iter([tool_chunk])
                    def close(self):
                        return None
                return _Resp()
        monkeypatch.setattr(
            "tldw_Server_API.app.core.LLM_Calls.providers.google_adapter.http_client_factory",
            lambda *a, **k: _Client(),
        )

        gen = perform_chat_api_call(
            api_provider="google",
            messages=[{"role": "user", "content": "Hi"}],
            api_key="test",
            model="gemini-2.5-flash",
            streaming=True,
        )
        chunks = list(gen)
        assert len(chunks) == 2  # tool-call chunk + DONE
        first_payload = chunks[0].split("data: ", 1)[1]
        payload_json = json.loads(first_payload)
        delta = payload_json["choices"][0]["delta"]
        assert "tool_calls" in delta and delta["tool_calls"]
        tool_call = delta["tool_calls"][0]
        assert tool_call["function"]["name"] == "lookup"
        assert json.loads(tool_call["function"]["arguments"]) == {"query": "mars"}
        assert chunks[-1].strip() == "data: [DONE]"

    def test_google_gemini_stream_tool_call_indices_are_monotonic(self, monkeypatch):

        tool_chunk_1 = (
            b'data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"lookup","args":{"query":"mars"}}}]}}]}'
        )
        tool_chunk_2 = (
            b'data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"lookup2","args":{"query":"moon"}}}]}}]}'
        )

        class _Client:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def stream(self, method, url, *, headers=None, json=None):
                class _Resp:
                    status_code = 200
                    def raise_for_status(self):
                        return None
                    def __enter__(self):
                        return self
                    def __exit__(self, exc_type, exc, tb):
                        return False
                    def iter_lines(self):
                        return iter([tool_chunk_1, tool_chunk_2])
                    def close(self):
                        return None
                return _Resp()

        monkeypatch.setattr(
            "tldw_Server_API.app.core.LLM_Calls.providers.google_adapter.http_client_factory",
            lambda *a, **k: _Client(),
        )

        gen = perform_chat_api_call(
            api_provider="google",
            messages=[{"role": "user", "content": "Hi"}],
            api_key="test",
            model="gemini-2.5-flash",
            streaming=True,
        )
        chunks = list(gen)
        assert len(chunks) == 3  # two tool-call chunks + DONE
        first_payload = json.loads(chunks[0].split("data: ", 1)[1])
        second_payload = json.loads(chunks[1].split("data: ", 1)[1])
        first_idx = first_payload["choices"][0]["delta"]["tool_calls"][0]["index"]
        second_idx = second_payload["choices"][0]["delta"]["tool_calls"][0]["index"]
        assert first_idx == 0
        assert second_idx == 1

    def test_bedrock_stream_normalized(self, monkeypatch):

        class _Client:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def stream(self, method, url, *, headers=None, json=None):
                class _Resp:
                    status_code = 200
                    def raise_for_status(self):
                        return None
                    def __enter__(self):
                        return self
                    def __exit__(self, exc_type, exc, tb):
                        return False
                    def iter_lines(self):
                        return iter([
                            b'data: {"choices":[{"delta":{"content":"Hi"}}]}',
                            b'data: {"choices":[{"delta":{"content":" Bedrock"}}]}',
                        ])
                return _Resp()
        monkeypatch.setattr(
            "tldw_Server_API.app.core.LLM_Calls.providers.bedrock_adapter.http_client_factory",
            lambda *a, **k: _Client(),
        )

        gen = perform_chat_api_call(
            api_provider="bedrock",
            messages=[{"role": "user", "content": "Hi"}],
            api_key="key", model="meta.llama3-8b-instruct", streaming=True,
        )
        chunks = list(gen)
        assert len(chunks) == 3
        assert chunks[0].startswith('data: ')
        assert chunks[0].endswith('\n\n')
        assert '[DONE]' in chunks[-1]

    def test_bedrock_stream_error_chunked(self, monkeypatch):

        class _Client:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def stream(self, method, url, *, headers=None, json=None):
                class _Resp:
                    status_code = 200
                    def raise_for_status(self):
                        return None
                    def __enter__(self):
                        return self
                    def __exit__(self, exc_type, exc, tb):
                        return False
                    def iter_lines(self):
                        raise RuntimeError('boom')
                return _Resp()
        monkeypatch.setattr(
            "tldw_Server_API.app.core.LLM_Calls.providers.bedrock_adapter.http_client_factory",
            lambda *a, **k: _Client(),
        )

        with pytest.raises(ChatProviderError):
            list(perform_chat_api_call(
                api_provider="bedrock",
                messages=[{"role": "user", "content": "Hi"}],
                api_key="key", model="meta.llama3-8b-instruct", streaming=True,
            ))

    def test_gemini_stream_error_chunked(self, monkeypatch):

        class _Client:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def stream(self, method, url, *, headers=None, json=None):
                class _Resp:
                    status_code = 200
                    def raise_for_status(self):
                        return None
                    def __enter__(self):
                        return self
                    def __exit__(self, exc_type, exc, tb):
                        return False
                    def iter_lines(self):
                        raise RuntimeError('boom')
                    def close(self):
                        return None
                return _Resp()
        monkeypatch.setattr(
            "tldw_Server_API.app.core.LLM_Calls.providers.google_adapter.http_client_factory",
            lambda *a, **k: _Client(),
        )

        with pytest.raises(ChatProviderError):
            list(perform_chat_api_call(
                api_provider="google",
                messages=[{"role": "user", "content": "Hi"}],
                api_key="test",
                model="gemini-2.5-flash",
                streaming=True,
            ))

    def test_gemini_stream_finish_reason(self, monkeypatch):

        class _Client:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def stream(self, method, url, *, headers=None, json=None):
                class _Resp:
                    status_code = 200
                    def raise_for_status(self):
                        return None
                    def __enter__(self):
                        return self
                    def __exit__(self, exc_type, exc, tb):
                        return False
                    def iter_lines(self):
                        return iter([
                            b'data: {"candidates":[{"content":{"parts":[{"text":"Hello"}]}}]}',
                            b'data: {"candidates":[{"finishReason":"STOP"}]}',
                        ])
                    def close(self):
                        return None
                return _Resp()
        monkeypatch.setattr(
            "tldw_Server_API.app.core.LLM_Calls.providers.google_adapter.http_client_factory",
            lambda *a, **k: _Client(),
        )

        gen = perform_chat_api_call(
            api_provider="google",
            messages=[{"role": "user", "content": "Hi"}],
            api_key="test",
            model="gemini-2.5-flash",
            streaming=True,
        )
        chunks = list(gen)
        assert any('"finish_reason": "stop"' in c for c in chunks)
        assert '[DONE]' in chunks[-1]

    def test_anthropic_stream_finish_reason(self, monkeypatch):

        class _Client:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def stream(self, *args, **kwargs):
                class _Resp:
                    status_code = 200
                    def raise_for_status(self):
                        return None
                    def __enter__(self):
                        return self
                    def __exit__(self, exc_type, exc, tb):
                        return False
                    def iter_lines(self):
                        return iter([
                            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}',
                            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}',
                        ])
                return _Resp()
        monkeypatch.setattr(
            "tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter.http_client_factory",
            lambda *a, **k: _Client(),
        )

        gen = perform_chat_api_call(
            api_provider="anthropic",
            messages=[{"role": "user", "content": "Hi"}],
            api_key="key", model="claude-3-5-sonnet-latest", streaming=True,
        )
        chunks = list(gen)
        assert any('"finish_reason": "stop"' in c for c in chunks)
        assert any('[DONE]' in c for c in chunks)

    def test_mistral_stream_normalized(self, monkeypatch):

        class _Client:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def stream(self, method, url, *, headers=None, json=None):
                class _Resp:
                    status_code = 200
                    def raise_for_status(self):
                        return None
                    def __enter__(self):
                        return self
                    def __exit__(self, exc_type, exc, tb):
                        return False
                    def iter_lines(self):
                        return iter([
                            'data: {"choices":[{"delta":{"content":"Hi"}}]}',
                            'data: {"choices":[{"delta":{"content":", Mistral"}}]}',
                        ])
                    def close(self):
                        return None
                return _Resp()
        monkeypatch.setattr(
            "tldw_Server_API.app.core.LLM_Calls.providers.mistral_adapter.http_client_factory",
            lambda *a, **k: _Client(),
        )

        gen = perform_chat_api_call(
            api_provider="mistral",
            messages=[{"role": "user", "content": "Hi"}],
            api_key="test",
            model="mistral-small",
            streaming=True,
        )
        chunks = list(gen)
        assert len(chunks) == 3
        assert chunks[0].startswith('data: ')
        assert chunks[0].endswith('\n\n')
        assert '[DONE]' in chunks[-1]

    def test_openrouter_stream_normalized(self, monkeypatch):

        class _Client:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def stream(self, *args, **kwargs):
                class _Resp:
                    status_code = 200
                    def raise_for_status(self):
                        return None
                    def __enter__(self):
                        return self
                    def __exit__(self, exc_type, exc, tb):
                        return False
                    def iter_lines(self):
                        return iter([
                            'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
                            'data: {"choices":[{"delta":{"content":" OpenRouter"}}]}\n\n',
                            'data: [DONE]\n\n',
                        ])
                return _Resp()
        monkeypatch.setattr(
            "tldw_Server_API.app.core.LLM_Calls.providers.openrouter_adapter.http_client_factory",
            lambda *args, **kwargs: _Client(),
        )

        gen = perform_chat_api_call(
            api_provider="openrouter",
            messages=[{"role": "user", "content": "Hi"}],
            api_key="test", model="mistralai/mistral-7b-instruct:free", streaming=True,
        )
        chunks = list(gen)
        assert len(chunks) == 3
        assert chunks[0].startswith('data: ')
        assert chunks[-1].strip() == 'data: [DONE]'

    def test_deepseek_stream_normalized(self, monkeypatch):

        class _Client:
            def __enter__(self): return self
            def __exit__(self, exc_type, exc, tb): return False
            def stream(self, *args, **kwargs):
                class _Resp:
                    status_code = 200
                    def raise_for_status(self): return None
                    def __enter__(self): return self
                    def __exit__(self, exc_type, exc, tb): return False
                    def iter_lines(self):
                        return iter([
                            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
                            'data: {"choices":[{"delta":{"content":" DeepSeek"}}]}',
                        ])
                return _Resp()
        monkeypatch.setattr(
            "tldw_Server_API.app.core.LLM_Calls.providers.deepseek_adapter.http_client_factory",
            lambda *a, **k: _Client(),
        )

        gen = perform_chat_api_call(
            api_provider="deepseek",
            messages=[{"role": "user", "content": "Hi"}],
            api_key="test", model="deepseek-chat", streaming=True,
        )
        chunks = list(gen)
        assert len(chunks) == 3
        assert chunks[0].startswith('data: ')
        assert chunks[0].endswith('\n\n')
        assert '[DONE]' in chunks[-1]

    def test_huggingface_stream_normalized(self, monkeypatch):

        class _Client:
            def __enter__(self): return self
            def __exit__(self, exc_type, exc, tb): return False
            def stream(self, *args, **kwargs):
                class _Resp:
                    status_code = 200
                    def raise_for_status(self): return None
                    def __enter__(self): return self
                    def __exit__(self, exc_type, exc, tb): return False
                    def iter_lines(self):
                        return iter([
                            b'data: {"choices":[{"delta":{"content":"Hi"}}]}',
                            b'data: {"choices":[{"delta":{"content":" HF"}}]}',
                        ])
                return _Resp()
        monkeypatch.setattr(
            "tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter.http_client_factory",
            lambda *a, **k: _Client(),
        )

        gen = perform_chat_api_call(
            api_provider="huggingface",
            messages=[{"role": "user", "content": "Hi"}],
            model="org/model", api_key="key", streaming=True,
        )
        chunks = list(gen)
        assert len(chunks) == 3
        assert chunks[0].startswith('data: ')
        assert chunks[0].endswith('\n\n')
        assert '[DONE]' in chunks[-1]

    def test_anthropic_stream_includes_done(self, monkeypatch):

        class _Client:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def stream(self, *args, **kwargs):
                class _Resp:
                    status_code = 200
                    def raise_for_status(self):
                        return None
                    def __enter__(self):
                        return self
                    def __exit__(self, exc_type, exc, tb):
                        return False
                    def iter_lines(self):
                        return iter([
                            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}',
                            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}',
                        ])
                return _Resp()
        monkeypatch.setattr(
            "tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter.http_client_factory",
            lambda *a, **k: _Client(),
        )
        gen = perform_chat_api_call(
            api_provider="anthropic",
            messages=[{"role": "user", "content": "Hi"}],
            api_key="key", model="claude-3-5-sonnet-latest", streaming=True,
        )
        chunks = list(gen)
        assert any('[DONE]' in c for c in chunks)

    def test_anthropic_stream_emits_tool_calls(self, monkeypatch):

        class _Client:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def stream(self, *args, **kwargs):
                class _Resp:
                    status_code = 200
                    def raise_for_status(self):
                        return None
                    def __enter__(self):
                        return self
                    def __exit__(self, exc_type, exc, tb):
                        return False
                    def iter_lines(self):
                        return iter([
                            'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tool_1","name":"lookup","input":{}}}',
                            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"city\\":\\"Paris\\"}"}}',
                            'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"}}',
                        ])
                return _Resp()
        monkeypatch.setattr(
            "tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter.http_client_factory",
            lambda *a, **k: _Client(),
        )
        gen = perform_chat_api_call(
            api_provider="anthropic",
            messages=[{"role": "user", "content": "Hi"}],
            api_key="key", model="claude-3-5-sonnet-latest", streaming=True,
        )
        chunks = list(gen)
        assert any('"tool_calls"' in c for c in chunks)
        assert any('[DONE]' in c for c in chunks)

    def test_anthropic_stream_error_chunked(self, monkeypatch):

        # Simulate a midstream error: adapter.normalize_error should raise a Chat*Error
        class _ErrClient:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def stream(self, *args, **kwargs):
                class _Resp:
                    def raise_for_status(self):
                        import httpx
                        req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
                        resp = httpx.Response(400, request=req, content=b'{"error":{"message":"bad"}}')
                        raise httpx.HTTPStatusError("err", request=req, response=resp)
                    def __enter__(self):
                        return self
                    def __exit__(self, exc_type, exc, tb):
                        return False
                    def iter_lines(self):
                        return iter([])
                return _Resp()
        monkeypatch.setattr(
            "tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter.http_client_factory",
            lambda *a, **k: _ErrClient(),
        )
        with pytest.raises(ChatBadRequestError):
            _ = list(perform_chat_api_call(
                api_provider="anthropic",
                messages=[{"role": "user", "content": "Hi"}],
                api_key="key", model="claude-3-5-sonnet-latest", streaming=True,
            ))

    def test_anthropic_payload_includes_image_url(self, monkeypatch):

        captured = {"json": None}
        class _Client:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def post(self, url, headers=None, json=None):
                captured["json"] = json
                class R:
                    status_code = 200
                    def raise_for_status(self):
                        return None
                    def json(self):
                        return {"id": "ok", "type": "message", "usage": {"input_tokens": 1, "output_tokens": 1}}
                return R()
        monkeypatch.setattr(
            "tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter.http_client_factory",
            lambda *a, **k: _Client(),
        )
        perform_chat_api_call(
            api_provider="anthropic",
            messages=[{
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}],
            }],
            api_key="key",
            model="claude-3-5-sonnet-latest",
            streaming=False,
        )
        payload = captured["json"]
        image_source = payload['messages'][0]['content'][0]['source']
        assert image_source['type'] == 'url'
        assert image_source['url'] == 'https://example.com/cat.png'

    def test_anthropic_payload_includes_base64_image(self, monkeypatch):

        captured = {"json": None}
        class _Client:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def post(self, url, headers=None, json=None):
                captured["json"] = json
                class R:
                    status_code = 200
                    def raise_for_status(self):
                        return None
                    def json(self):
                        return {"id": "ok", "type": "message", "usage": {"input_tokens": 1, "output_tokens": 1}}
                return R()
        monkeypatch.setattr(
            "tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter.http_client_factory",
            lambda *a, **k: _Client(),
        )
        perform_chat_api_call(
            api_provider="anthropic",
            messages=[{
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}],
            }],
            api_key="key",
            model="claude-3-5-sonnet-latest",
            streaming=False,
        )
        payload = captured["json"]
        image_source = payload['messages'][0]['content'][0]['source']
        assert image_source['type'] == 'base64'
        assert image_source['media_type'] == 'image/png'
        assert image_source['data'] == 'QUJD'

    def test_mistral_stream_error_chunked(self, monkeypatch):

        class _Client:
            def __enter__(self): return self
            def __exit__(self, exc_type, exc, tb): return False
            def stream(self, *args, **kwargs):
                class _Resp:
                    status_code = 200
                    def raise_for_status(self): return None
                    def __enter__(self): return self
                    def __exit__(self, exc_type, exc, tb): return False
                    def iter_lines(self):
                        raise RuntimeError('boom')
                return _Resp()
        monkeypatch.setattr(
            "tldw_Server_API.app.core.LLM_Calls.providers.mistral_adapter.http_client_factory",
            lambda *a, **k: _Client(),
        )
        with pytest.raises(ChatProviderError):
            list(perform_chat_api_call(
                api_provider="mistral",
                messages=[{"role": "user", "content": "Hi"}],
                api_key="key", model="mistral-small", streaming=True,
            ))

    def test_openrouter_stream_error_chunked(self, monkeypatch):

        class _Client:
            def __enter__(self): return self
            def __exit__(self, exc_type, exc, tb): return False
            def stream(self, *args, **kwargs):
                class _Resp:
                    status_code = 200
                    def raise_for_status(self): return None
                    def __enter__(self): return self
                    def __exit__(self, exc_type, exc, tb): return False
                    def iter_lines(self):
                        raise RuntimeError('boom')
                return _Resp()
        monkeypatch.setattr(
            "tldw_Server_API.app.core.LLM_Calls.providers.openrouter_adapter.http_client_factory",
            lambda *a, **k: _Client(),
        )
        with pytest.raises(ChatProviderError):
            list(perform_chat_api_call(
                api_provider="openrouter",
                messages=[{"role": "user", "content": "Hi"}],
                api_key="key", model="mistralai/mistral-7b-instruct:free", streaming=True,
            ))

    @pytest.mark.asyncio
    async def test_anthropic_async_matches_sync_normalization(self, monkeypatch):
        import copy

        response_payload = {
            "id": "msg_42",
            "model": "claude-haiku-4.5",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "tool_use", "id": "tool_99", "name": "lookup", "input": {"city": "Paris"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }

        def _make_response():

            class R:
                status_code = 200
                def raise_for_status(self):
                    return None
                def json(self):
                    return copy.deepcopy(response_payload)
            return R()

        class _Client:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def post(self, *args, **kwargs):
                return _make_response()
            def stream(self, *args, **kwargs):
                raise AssertionError("Streaming not expected in this test.")
        monkeypatch.setattr(
            "tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter.http_client_factory",
            lambda *a, **k: _Client(),
        )

        sync_result = perform_chat_api_call(
            api_provider="anthropic",
            messages=[{"role": "user", "content": "Hi"}],
            api_key="key",
            model="claude-haiku-4.5",
            streaming=False,
        )
        async_result = await perform_chat_api_call_async(
            api_provider="anthropic",
            messages=[{"role": "user", "content": "Hi"}],
            api_key="key",
            model="claude-haiku-4.5",
            streaming=False,
        )
        from tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter import AnthropicAdapter
        expected = AnthropicAdapter()._normalize_to_openai_shape(response_payload)
        assert sync_result == expected
        assert async_result == expected


# ---------------------------------------------------------------------------
# OpenAI adapter tests
# ---------------------------------------------------------------------------

def test_openai_defaults_with_blank_config(monkeypatch):

    captured = {}

    class FakeResp:
        status_code = 200
        def raise_for_status(self):
            return None
        def json(self):
            return {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    class FakeClient:
        def __init__(self, *_, **kwargs):
            captured["timeout"] = kwargs.get("timeout")
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
        "tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter.http_client_factory",
        lambda *a, **k: FakeClient(**k),
    )

    result = perform_chat_api_call(
        api_provider="openai",
        messages=[{"role": "user", "content": "hello"}],
        api_key="test-key",
        model="gpt-4o-mini",
        temp=0.7,
        maxp=0.95,
        app_config={"openai_api": {"api_base_url": "https://mock.openai.local/v1", "api_timeout": 90}},
    )

    assert result["choices"][0]["message"]["content"] == "ok"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["temperature"] == 0.7
    assert captured["json"]["top_p"] == 0.95
    assert captured["timeout"] == 90.0


def test_openai_sync_gpt5_payload(monkeypatch):

    captured = {}

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": []}

    class FakeClient:
        def __init__(self, *_, **kwargs):
            captured["timeout"] = kwargs.get("timeout")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return FakeResp()

    monkeypatch.setenv("LLM_ADAPTERS_NATIVE_HTTP_OPENAI", "1")
    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter.http_client_factory",
        lambda *a, **k: FakeClient(**k),
    )

    result = perform_chat_api_call(
        api_provider="openai",
        messages=[{"role": "user", "content": "hello"}],
        api_key="test-key",
        model="gpt-5-mini",
        max_tokens=128,
        maxp=0.8,
        temp=1.0,
    )

    payload = captured["json"]
    assert "top_p" not in payload
    assert "max_tokens" not in payload
    assert payload["max_completion_tokens"] == 128
    assert payload.get("temperature") == 1.0
    assert result["choices"] == []


@pytest.mark.asyncio
async def test_openai_async_streaming_normalized(monkeypatch):
    captured: Dict[str, Any] = {}

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def iter_lines(self):
            yield 'event: completion.delta'
            yield 'data: {"choices":[{"delta":{"content":"Hello"}}]}'
            yield 'id: chunk-1'
            yield 'data: {"choices":[{"delta":{"content":" async"}}]}'
            yield 'retry: 1000'
            yield 'data: [DONE]'

    class FakeStreamCtx:
        def __enter__(self):
            return FakeResp()

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeClient:
        def __init__(self, *_, **kwargs):
            self.timeout = kwargs.get("timeout")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, *, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["payload"] = json
            return FakeStreamCtx()

    def fake_factory(*args, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        return FakeClient(**kwargs)

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter.http_client_factory",
        fake_factory,
    )

    gen = await perform_chat_api_call_async(
        api_provider="openai",
        messages=[{"role": "user", "content": "hi"}],
        api_key="test",
        streaming=True,
        model="gpt-4o-mini",
    )
    out = []
    async for chunk in gen:
        out.append(chunk)
    assert len(out) == 3
    assert out[0].startswith('data: ')
    assert out[0].endswith('\n\n')
    assert '[DONE]' in out[-1]


@pytest.mark.asyncio
async def test_openai_async_non_streaming_preserves_payload(monkeypatch):
    expected_response = {
        "id": "chatcmpl-123",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Async hello"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
    }
    captured: Dict[str, Any] = {}

    def fake_config():
        return {"openai_api": {"api_key": "cfg-key", "model": "gpt-4o-mini"}}

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return expected_response

    class FakeClient:
        def __init__(self, *_, **kwargs):
            self.timeout = kwargs.get("timeout")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["payload"] = json
            return FakeResp()

    def fake_factory(*args, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        return FakeClient(**kwargs)

    monkeypatch.setattr(
        'tldw_Server_API.app.core.LLM_Calls.chat_calls.load_and_log_configs',
        fake_config,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter.http_client_factory",
        fake_factory,
    )

    result = await perform_chat_api_call_async(
        api_provider="openai",
        messages=[{"role": "user", "content": "hi async"}],
        api_key="cfg-key",
        model="gpt-4o-mini",
        streaming=False,
    )

    assert result == expected_response
    assert captured["url"].endswith("/chat/completions")
    assert captured["payload"]["messages"][-1]["content"] == "hi async"
    assert captured["timeout"] == 90.0
    assert captured["payload"]["stream"] is False


@pytest.mark.asyncio
async def test_openai_async_gpt5_payload(monkeypatch):
    captured: Dict[str, Any] = {}

    def fake_config():
        return {
            "openai_api": {
                "api_key": "cfg-key",
                "model": "gpt-5-mini",
                "max_tokens": 128,
                "temperature": 1.0,
                "top_p": 0.8,
            }
        }

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": []}

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["payload"] = json
            return FakeResp()

    def fake_factory(*args, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        return FakeClient()

    monkeypatch.setattr(
        'tldw_Server_API.app.core.LLM_Calls.chat_calls.load_and_log_configs',
        fake_config,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter.http_client_factory",
        fake_factory,
    )

    result = await perform_chat_api_call_async(
        api_provider="openai",
        messages=[{"role": "user", "content": "hi there"}],
        streaming=False,
        model="gpt-5-mini",
        app_config=fake_config(),
    )

    payload = captured["payload"]
    assert "top_p" not in payload
    assert "max_tokens" not in payload
    assert payload["max_completion_tokens"] == 128
    assert payload.get("temperature") == 1.0
    assert result["choices"] == []


@pytest.mark.asyncio
async def test_groq_async_non_streaming_preserves_payload(monkeypatch):
    expected_response = {
        "id": "chatcmpl-groq",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Groq hello"}}],
    }
    captured: Dict[str, Any] = {}

    def fake_config():
        return {"groq_api": {"api_key": "groq-key", "model": "llama-3.3-70b-versatile"}}

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return expected_response

    class FakeClient:
        def __init__(self, *_, **kwargs):
            self.timeout = kwargs.get("timeout")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["payload"] = json
            return FakeResp()

    def fake_factory(*args, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        return FakeClient(**kwargs)

    monkeypatch.setattr(
        'tldw_Server_API.app.core.LLM_Calls.chat_calls.load_and_log_configs',
        fake_config,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.groq_adapter.http_client_factory",
        fake_factory,
    )

    result = await perform_chat_api_call_async(
        api_provider="groq",
        messages=[{"role": "user", "content": "groq async"}],
        api_key="groq-key",
        model="llama-3.3-70b-versatile",
        streaming=False,
    )

    assert result == expected_response
    assert captured["url"].endswith("/chat/completions")
    assert captured["payload"]["messages"][-1]["content"] == "groq async"
    assert captured["timeout"] == 90.0


@pytest.mark.asyncio
async def test_openrouter_async_streaming_filters_control_lines(monkeypatch):
    def fake_config():
        return {
            "openrouter_api": {
                "api_key": "router-key",
                "model": "mistralai/mistral-7b-instruct:free",
            }
        }

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def iter_lines(self):
            yield "event: ping"
            yield 'data: {"choices":[{"delta":{"content":"chunk"}}]}'
            yield "id: 123"
            yield "data: [DONE]"

    class FakeStreamCtx:
        def __enter__(self):
            return FakeResp()

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeClient:
        def __init__(self, *_, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, *, headers=None, json=None):
            return FakeStreamCtx()

    def fake_factory(*args, **kwargs):
        return FakeClient(**kwargs)

    monkeypatch.setattr(
        'tldw_Server_API.app.core.LLM_Calls.chat_calls.load_and_log_configs',
        fake_config,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.openrouter_adapter.http_client_factory",
        fake_factory,
    )

    gen = await perform_chat_api_call_async(
        api_provider="openrouter",
        messages=[{"role": "user", "content": "hello"}],
        api_key="router-key",
        model="mistralai/mistral-7b-instruct:free",
        streaming=True,
    )

    chunks = []
    async for chunk in gen:
        chunks.append(chunk)

    assert len(chunks) == 2  # content + sentinel
    assert "event:" not in "".join(chunks)
    assert "id:" not in "".join(chunks)
    assert chunks[-1].strip().lower() == "data: [done]"


def test_openai_non_streaming_session_closed(monkeypatch):

    closed = {"v": False}

    class FakeResp:
        status_code = 200
        def raise_for_status(self):
            return None
        def json(self):
            return {"choices": [], "id": "test"}
        def close(self):
            return None

    class FakeClient:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            closed["v"] = True
            return True
        def post(self, *a, **k):
            return FakeResp()
        def close(self):
            return None

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter.http_client_factory",
        lambda *a, **k: FakeClient(),
    )

    perform_chat_api_call(
        api_provider="openai",
        messages=[{"role": "user", "content": "hello"}],
        streaming=False,
        api_key="cfg-key",
        model="gpt-4o-mini",
        app_config={"openai_api": {"api_timeout": 90}},
    )

    assert closed["v"] is True


def test_cohere_config_fallbacks(monkeypatch):

    def fake_config():
        return {
            "cohere_api": {
                "api_key": "cohere-key",
                "model": "command-r",
                "temperature": 0.42,
                "top_p": 0.85,
                "top_k": 12,
                "max_tokens": 256,
                "stop_sequences": ["END"],
                "seed": 123,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.05,
                "tools": [{"type": "function", "function": {"name": "lookup"}}],
                "num_generations": 2,
            }
        }

    response = Mock()
    response.status_code = 200
    response.raise_for_status = Mock()
    response.json.return_value = {
        "text": "Configured response",
        "finish_reason": "stop",
        "meta": {"billed_units": {"input_tokens": 5, "output_tokens": 7}},
    }

    session = Mock()
    session.post.return_value = response
    session.close = Mock()

    monkeypatch.setattr(
        'tldw_Server_API.app.core.LLM_Calls.chat_calls.load_and_log_configs',
        fake_config,
    )
    monkeypatch.setattr(
        'tldw_Server_API.app.core.LLM_Calls.chat_calls.create_session_with_retries',
        lambda *args, **kwargs: session,
    )

    result = perform_chat_api_call(
        api_provider="cohere",
        messages=[{"role": "user", "content": "Hi there"}],
        model="command-r",
        streaming=False,
        app_config=fake_config(),
    )

    payload = session.post.call_args.kwargs["json"]
    assert payload["temperature"] == 0.42
    assert payload["p"] == 0.85
    assert payload["k"] == 12
    assert payload["max_tokens"] == 256
    assert payload["stop_sequences"] == ["END"]
    assert payload["seed"] == 123
    assert payload["frequency_penalty"] == 0.1
    assert payload["presence_penalty"] == 0.05
    assert payload["tools"] == [{"type": "function", "function": {"name": "lookup"}}]
    assert payload["stream"] is False
    assert payload["num_generations"] == 2

    assert result["usage"]["prompt_tokens"] == 5
    session.close.assert_called_once()


def test_google_config_fallbacks(monkeypatch):

    def fake_config():
        return {
            "google_api": {
                "api_key": "google-key",
                "model": "gemini-test",
                "temperature": 0.33,
                "top_p": 0.77,
                "top_k": 40,
                "max_output_tokens": 512,
                "stop_sequences": ["STOP"],
                "candidate_count": 2,
                "tools": [{"function_declarations": [{"name": "lookup"}]}],
                "response_format": {"type": "json_object"},
            }
        }

    response = Mock()
    response.status_code = 200
    response.raise_for_status = Mock()
    response.json.return_value = {
        "candidates": [
            {
                "content": {"parts": [{"text": "Hello Gemini"}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 3,
            "candidatesTokenCount": 4,
            "totalTokenCount": 7,
        },
    }

    captured = {}
    class _Client:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def post(self, url, headers=None, json=None):
            captured["json"] = json
            return response

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.google_adapter.http_client_factory",
        lambda *a, **k: _Client(),
    )

    result = perform_chat_api_call(
        api_provider="google",
        messages=[{"role": "user", "content": "Hi"}],
        model="gemini-test",
        streaming=False,
        app_config=fake_config(),
    )

    payload = captured["json"]
    gen_cfg = payload["generationConfig"]
    assert gen_cfg["temperature"] == 0.33
    assert gen_cfg["topP"] == 0.77
    assert gen_cfg["topK"] == 40
    assert gen_cfg["maxOutputTokens"] == 512
    assert gen_cfg["stopSequences"] == ["STOP"]
    assert gen_cfg["candidateCount"] == 2
    assert gen_cfg["responseMimeType"] == "application/json"
    assert payload["tools"] == [{"function_declarations": [{"name": "lookup"}]}]

    assert result["choices"][0]["message"]["content"] == "Hello Gemini"


def test_mistral_stream_session_closed(monkeypatch):

    closed = {"client": False, "response": False}

    class _Client:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            closed["client"] = True
            return False
        def stream(self, method, url, *, headers=None, json=None):
            class _Resp:
                status_code = 200
                def raise_for_status(self):
                    return None
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc, tb):
                    closed["response"] = True
                    return False
                def iter_lines(self):
                    return iter([
                        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
                        'data: [DONE]',
                    ])
                def close(self):
                    closed["response"] = True
                    return None
            return _Resp()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.mistral_adapter.http_client_factory",
        lambda *a, **k: _Client(),
    )

    gen = perform_chat_api_call(
        api_provider="mistral",
        messages=[{"role": "user", "content": "hi"}],
        api_key="m-key",
        model="mistral-small",
        streaming=True,
    )
    chunks = list(gen)
    assert chunks[-1].strip().lower() == 'data: [done]'
    assert closed["client"] is True
    assert closed["response"] is True


def test_zai_http_error_normalized(monkeypatch):
    """Test that ZAI 429 errors are properly mapped to ChatRateLimitError."""
    session = MagicMock()
    response = MagicMock()
    response.status_code = 429
    response.text = '{"error":{"message":"Rate limit"}}'
    response.json.return_value = {"error": {"message": "Rate limit"}}
    http_error = requests.HTTPError("rate limit", response=response)
    response.raise_for_status.side_effect = http_error
    session.post.return_value = response
    session.close = Mock()

    monkeypatch.setattr(
        'tldw_Server_API.app.core.LLM_Calls.chat_calls.load_and_log_configs',
        lambda: {'zai_api': {'model': 'glm-4.5'}},
    )
    monkeypatch.setattr(_MOCK_CREATE_SESSION, lambda **kw: session)

    with pytest.raises(ChatRateLimitError):
        perform_chat_api_call(
            api_provider="zai",
            messages=[{"role": "user", "content": "hello"}],
            api_key="z-key",
            streaming=False,
            model="glm-4.5",
        )
    assert session.close.call_count >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
