from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import pytest
import requests
from loguru import logger

from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatProviderError,
    ChatRateLimitError,
)
from tldw_Server_API.app.core.LLM_Calls.error_utils import (
    build_sanitized_chat_error,
    raise_chat_error_from_http,
)
from tldw_Server_API.app.core.LLM_Calls.providers.base import ChatProvider
from tldw_Server_API.app.core.LLM_Calls.providers.cohere_adapter import CohereAdapter
from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter import (
    HuggingFaceAdapter,
)
from tldw_Server_API.app.core.LLM_Calls.providers.moonshot_adapter import (
    MoonshotAdapter,
)
from tldw_Server_API.app.core.LLM_Calls.providers.zai_adapter import ZaiAdapter

_PUBLIC_UNAVAILABLE_MESSAGE = "Error received from the chat provider API."


class _DummyProvider(ChatProvider):
    name = "dummy"

    def capabilities(self) -> dict[str, Any]:
        return {}

    def chat(
        self,
        request: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        del request, timeout
        return {}

    def stream(
        self,
        request: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> list[str]:
        del request, timeout
        return []


def _request(provider: str, url_sentinel: str, *, stream: bool = False) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": "safe request"}],
        "model": "safe-model",
        "api_key": "safe-key",
        "base_url": f"https://{provider}-{url_sentinel}.example/private",
        "stream": stream,
        "app_config": {},
    }


def _requests_error_response(
    *,
    status_code: int,
    url: str,
    body_sentinel: str,
    exception_sentinel: str,
) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = json.dumps(
        {"error": {"message": body_sentinel, "type": "private_error"}}
    ).encode()
    request = requests.Request("POST", url).prepare()
    response.request = request
    error = requests.HTTPError(
        f"{exception_sentinel} at {url}",
        request=request,
        response=response,
    )

    def _raise_for_status() -> None:
        raise error

    response.raise_for_status = _raise_for_status  # type: ignore[method-assign]
    return response


class _RequestsSession:
    def __init__(self, response: Any = None, *, error: Exception | None = None) -> None:
        self._response = response
        self._error = error

    def post(self, *_args: Any, **_kwargs: Any) -> Any:
        if self._error is not None:
            raise self._error
        return self._response

    def close(self) -> None:
        return None


def _assert_private_values_absent(
    value: BaseException,
    logs: list[str],
    *private_values: str,
) -> None:
    rendered = "\n".join(logs + [str(value), repr(value)])
    for private_value in private_values:
        assert private_value not in rendered
    assert value.__cause__ is None
    assert value.__context__ is None


@pytest.mark.parametrize(
    ("status_code", "expected_type", "expected_status"),
    [
        (400, "ChatBadRequestError", 400),
        (403, "ChatAuthenticationError", 403),
        (429, "ChatRateLimitError", 429),
        (503, "ChatProviderError", 503),
    ],
)
def test_base_error_normalization_preserves_taxonomy_without_reflecting_upstream_data(
    status_code: int,
    expected_type: str,
    expected_status: int,
) -> None:
    url_sentinel = f"private-url-{status_code}"
    body_sentinel = f"private-body-{status_code}"
    request = httpx.Request("POST", f"https://{url_sentinel}.example/chat")
    response = httpx.Response(
        status_code,
        request=request,
        json={"error": {"message": body_sentinel}},
    )
    error = httpx.HTTPStatusError(
        f"private-exception-{status_code}",
        request=request,
        response=response,
    )

    normalized = _DummyProvider().normalize_error(error)

    assert type(normalized).__name__ == expected_type
    assert normalized.status_code == expected_status
    assert url_sentinel not in str(normalized)
    assert body_sentinel not in str(normalized)
    assert f"private-exception-{status_code}" not in str(normalized)


_SPENT_QUOTA_BODY = {
    "error": {"message": "private-quota-body", "type": "insufficient_quota", "code": "insufficient_quota"}
}


def _httpx_429(content: bytes, *, read: bool = True) -> httpx.HTTPStatusError:
    """A 429 as httpx raises it; ``read=False`` leaves the body unread, as a streamed reply does."""
    request = httpx.Request("POST", "https://api.openai.example/v1/chat/completions")
    if read:
        response = httpx.Response(429, request=request, content=content)
    else:
        response = httpx.Response(429, request=request, stream=httpx.ByteStream(content))
    return httpx.HTTPStatusError("private-exception", request=request, response=response)


@pytest.mark.parametrize(
    ("content", "read", "expected"),
    [
        (json.dumps(_SPENT_QUOTA_BODY).encode(), True, True),
        (json.dumps({"error": {"type": "tokens", "code": "rate_limit_exceeded"}}).encode(), True, False),
        (json.dumps({"error": {"code": ["insufficient_quota"]}}).encode(), True, False),
        (json.dumps({"error": "insufficient_quota"}).encode(), True, False),
        (b"insufficient_quota", True, False),
        (json.dumps(_SPENT_QUOTA_BODY).encode(), False, False),
    ],
    ids=["insufficient-quota", "rate-limit", "list-code", "string-error", "not-json", "unread-body"],
)
def test_a_rate_limit_says_whether_the_billing_quota_is_spent(
    content: bytes,
    read: bool,
    expected: bool,
) -> None:
    normalized = _DummyProvider().normalize_error(_httpx_429(content, read=read))

    assert isinstance(normalized, ChatRateLimitError)
    assert normalized.status_code == 429
    assert normalized.quota_exhausted is expected
    assert "private-quota-body" not in repr(vars(normalized))


def test_requests_adapters_keep_the_spent_quota_signal() -> None:
    response = requests.Response()
    response.status_code = 429
    response._content = json.dumps(_SPENT_QUOTA_BODY).encode()
    error = requests.HTTPError("private-exception", response=response)

    with pytest.raises(ChatRateLimitError) as exc_info:
        raise_chat_error_from_http("moonshot", error)

    assert exc_info.value.quota_exhausted is True


def test_a_rate_limit_without_an_upstream_reply_is_not_a_spent_quota() -> None:
    error = build_sanitized_chat_error("openai", status_code=429)

    assert isinstance(error, ChatRateLimitError)
    assert error.quota_exhausted is False


def test_base_generic_error_normalization_never_reflects_exception_text() -> None:
    sentinel = "private-generic-exception-/srv/provider?token=secret"

    normalized = _DummyProvider().normalize_error(RuntimeError(sentinel))

    assert isinstance(normalized, ChatProviderError)
    assert str(normalized) == _PUBLIC_UNAVAILABLE_MESSAGE
    assert sentinel not in repr(normalized)


@pytest.mark.parametrize(
    ("provider", "adapter_type"),
    [
        ("cohere", CohereAdapter),
        ("moonshot", MoonshotAdapter),
        ("zai", ZaiAdapter),
    ],
)
def test_requests_adapters_sanitize_http_url_body_and_exception_at_boundary(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    adapter_type: type[ChatProvider],
) -> None:
    from tldw_Server_API.app.core.LLM_Calls import chat_calls

    url_sentinel = f"private-url-{provider}"
    body_sentinel = f"private-body-{provider}"
    exception_sentinel = f"private-exception-{provider}"
    response = _requests_error_response(
        status_code=403,
        url=f"https://{url_sentinel}.example/private",
        body_sentinel=body_sentinel,
        exception_sentinel=exception_sentinel,
    )
    monkeypatch.setattr(
        chat_calls,
        "create_session_with_retries",
        lambda **_kwargs: _RequestsSession(response),
    )
    logs: list[str] = []
    sink_id = logger.add(logs.append, level="DEBUG", format="{message}")
    try:
        with pytest.raises(ChatAuthenticationError) as exc_info:
            adapter_type().chat(_request(provider, url_sentinel))
    finally:
        logger.remove(sink_id)

    assert exc_info.value.status_code == 403
    _assert_private_values_absent(
        exc_info.value,
        logs,
        url_sentinel,
        body_sentinel,
        exception_sentinel,
    )


@pytest.mark.parametrize(
    ("provider", "adapter_type"),
    [
        ("cohere", CohereAdapter),
        ("moonshot", MoonshotAdapter),
        ("zai", ZaiAdapter),
    ],
)
def test_requests_adapters_sanitize_network_exception_at_boundary(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    adapter_type: type[ChatProvider],
) -> None:
    from tldw_Server_API.app.core.LLM_Calls import chat_calls

    url_sentinel = f"private-network-url-{provider}"
    exception_sentinel = f"private-network-exception-{provider}"
    upstream = requests.ConnectionError(
        f"{exception_sentinel} at https://{url_sentinel}.example/private"
    )
    monkeypatch.setattr(
        chat_calls,
        "create_session_with_retries",
        lambda **_kwargs: _RequestsSession(error=upstream),
    )
    logs: list[str] = []
    sink_id = logger.add(logs.append, level="DEBUG", format="{message}")
    try:
        with pytest.raises(ChatProviderError) as exc_info:
            adapter_type().chat(_request(provider, url_sentinel))
    finally:
        logger.remove(sink_id)

    assert exc_info.value.status_code == 504
    _assert_private_values_absent(
        exc_info.value,
        logs,
        url_sentinel,
        exception_sentinel,
    )


class _HuggingFaceErrorResponse:
    status_code = 403

    def __init__(
        self,
        *,
        url_sentinel: str,
        body_sentinel: str,
        exception_sentinel: str,
    ) -> None:
        self.request = httpx.Request(
            "POST",
            f"https://{url_sentinel}.example/private",
        )
        self.text = json.dumps({"error": {"message": body_sentinel}})
        self._body = {"error": {"message": body_sentinel, "type": "private"}}
        self._exception_sentinel = exception_sentinel

    def json(self) -> dict[str, Any]:
        return self._body

    def raise_for_status(self) -> None:
        raise httpx.HTTPStatusError(
            self._exception_sentinel,
            request=self.request,
            response=self,  # type: ignore[arg-type]
        )

    def iter_lines(self) -> list[str]:
        return []


class _HuggingFaceClient:
    def __init__(self, response: _HuggingFaceErrorResponse) -> None:
        self._response = response

    def __enter__(self) -> _HuggingFaceClient:
        return self

    def __exit__(self, *_args: Any) -> bool:
        return False

    def post(self, *_args: Any, **_kwargs: Any) -> _HuggingFaceErrorResponse:
        return self._response

    def stream(self, *_args: Any, **_kwargs: Any) -> _HuggingFaceStreamContext:
        return _HuggingFaceStreamContext(self._response)


class _HuggingFaceStreamContext:
    def __init__(self, response: _HuggingFaceErrorResponse) -> None:
        self._response = response

    def __enter__(self) -> _HuggingFaceErrorResponse:
        return self._response

    def __exit__(self, *_args: Any) -> bool:
        return False


@pytest.mark.parametrize("operation", ["chat", "stream"])
def test_huggingface_sanitizes_http_url_body_and_exception_at_boundary(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    from tldw_Server_API.app.core.LLM_Calls.providers import huggingface_adapter

    url_sentinel = f"private-hf-url-{operation}"
    body_sentinel = f"private-hf-body-{operation}"
    exception_sentinel = f"private-hf-exception-{operation}"
    response = _HuggingFaceErrorResponse(
        url_sentinel=url_sentinel,
        body_sentinel=body_sentinel,
        exception_sentinel=exception_sentinel,
    )
    monkeypatch.setattr(
        huggingface_adapter,
        "http_client_factory",
        lambda **_kwargs: _HuggingFaceClient(response),
    )
    logs: list[str] = []
    sink_id = logger.add(logs.append, level="DEBUG", format="{message}")
    try:
        with pytest.raises(ChatAuthenticationError) as exc_info:
            result = getattr(HuggingFaceAdapter(), operation)(
                _request("huggingface", url_sentinel, stream=operation == "stream")
            )
            if operation == "stream":
                list(result)
    finally:
        logger.remove(sink_id)

    assert exc_info.value.status_code == 403
    _assert_private_values_absent(
        exc_info.value,
        logs,
        url_sentinel,
        body_sentinel,
        exception_sentinel,
    )


class _FailingStreamResponse:
    status_code = 200

    def __init__(
        self,
        *,
        provider: str,
        sentinel: str,
        barrier: threading.Barrier,
    ) -> None:
        self._provider = provider
        self._sentinel = sentinel
        self._barrier = barrier

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self, **_kwargs: Any):
        if self._provider == "cohere":
            yield b'data: {"event_type":"text-generation","text":"safe"}'
        else:
            yield 'data: {"choices":[{"delta":{"content":"safe"}}]}'
        self._barrier.wait(timeout=5)
        raise RuntimeError(self._sentinel)

    def close(self) -> None:
        return None


class _RoutingStreamSession:
    def __init__(self, *, barrier: threading.Barrier, sentinels: dict[str, str]) -> None:
        self._barrier = barrier
        self._sentinels = sentinels

    def post(self, url: str, **_kwargs: Any) -> _FailingStreamResponse:
        provider = "cohere" if "cohere" in url else "zai"
        return _FailingStreamResponse(
            provider=provider,
            sentinel=self._sentinels[provider],
            barrier=self._barrier,
        )

    def close(self) -> None:
        return None


@pytest.mark.concurrent
def test_concurrent_requests_streams_emit_canonical_errors_without_cross_leaking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.LLM_Calls import chat_calls

    sentinels = {
        "cohere": "private-cohere-stream-/srv/alpha?token=one",
        "zai": "private-zai-stream-/srv/beta?token=two",
    }
    barrier = threading.Barrier(2)
    monkeypatch.setattr(
        chat_calls,
        "create_session_with_retries",
        lambda **_kwargs: _RoutingStreamSession(
            barrier=barrier,
            sentinels=sentinels,
        ),
    )
    logs: list[str] = []
    log_lock = threading.Lock()

    def _capture(message: Any) -> None:
        with log_lock:
            logs.append(str(message))

    sink_id = logger.add(_capture, level="DEBUG", format="{message}")
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                provider: executor.submit(
                    list,
                    adapter.stream(_request(provider, "safe-stream-url", stream=True)),
                )
                for provider, adapter in (
                    ("cohere", CohereAdapter()),
                    ("zai", ZaiAdapter()),
                )
            }
            chunks = {
                provider: future.result(timeout=10)
                for provider, future in futures.items()
            }
    finally:
        logger.remove(sink_id)

    rendered = "\n".join(logs + [chunk for values in chunks.values() for chunk in values])
    for sentinel in sentinels.values():
        assert sentinel not in rendered
    for provider, values in chunks.items():
        error_payloads = [
            json.loads(value.removeprefix("data:").strip())
            for value in values
            if '"error"' in value
        ]
        assert error_payloads == [
            {
                "error": {
                    "code": "provider_unavailable",
                    "message": "The chat service provider is currently unavailable.",
                    "type": f"{provider}_stream_error",
                }
            }
        ]
