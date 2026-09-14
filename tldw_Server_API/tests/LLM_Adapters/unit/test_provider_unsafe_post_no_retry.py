from __future__ import annotations

from typing import Any

import httpx
import pytest

from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatProviderError,
    ChatRateLimitError,
)
from tldw_Server_API.app.core.LLM_Calls.providers.base import ChatProvider
from tldw_Server_API.app.core.LLM_Calls.providers.cohere_adapter import CohereAdapter
from tldw_Server_API.app.core.LLM_Calls.providers.moonshot_adapter import (
    MoonshotAdapter,
)
from tldw_Server_API.app.core.LLM_Calls.providers.zai_adapter import ZaiAdapter


@pytest.mark.unit
@pytest.mark.parametrize(
    ("provider", "adapter_type"),
    [
        ("cohere", CohereAdapter),
        ("moonshot", MoonshotAdapter),
        ("zai", ZaiAdapter),
    ],
)
@pytest.mark.parametrize("streaming", [False, True], ids=["chat", "stream"])
@pytest.mark.parametrize(
    ("failure", "expected_error", "expected_status"),
    [
        ("network", ChatProviderError, 504),
        ("status", ChatRateLimitError, 429),
    ],
)
def test_provider_post_is_not_replayed_after_ambiguous_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    adapter_type: type[ChatProvider],
    streaming: bool,
    failure: str,
    expected_error: type[Exception],
    expected_status: int,
) -> None:
    """Provider POSTs stay single-attempt for both request facade branches."""
    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.app.core.LLM_Calls import chat_calls, http_helpers

    calls = {"n": 0}
    sentinel = f"private-{provider}-{failure}-failure"
    clients: list[_TrackingClient] = []

    class _TrackingStreamContext:
        def __init__(self, inner: Any, owner: _TrackingClient) -> None:
            self._inner = inner
            self._owner = owner

        def __enter__(self) -> Any:
            return self._inner.__enter__()

        def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> Any:
            self._owner.stream_context_exits += 1
            return self._inner.__exit__(exc_type, exc, traceback)

    class _TrackingClient(httpx.Client):
        def __init__(self) -> None:
            super().__init__(transport=httpx.MockTransport(handler))
            self.close_calls = 0
            self.stream_context_exits = 0

        def stream(self, *args: Any, **kwargs: Any) -> _TrackingStreamContext:
            return _TrackingStreamContext(super().stream(*args, **kwargs), self)

        def close(self) -> None:
            self.close_calls += 1
            super().close()

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if failure == "network":
            raise httpx.ConnectError(sentinel, request=request)
        return httpx.Response(
            429,
            request=request,
            json={"error": {"message": sentinel}},
        )

    def _create_client() -> _TrackingClient:
        client = _TrackingClient()
        clients.append(client)
        return client

    monkeypatch.setattr(http_helpers, "_hc_create_client", _create_client)
    monkeypatch.setattr(
        chat_calls,
        "create_session_with_retries",
        http_helpers.create_session_with_retries,
    )
    monkeypatch.setattr(
        http_client,
        "_validate_egress_or_raise",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(http_client.time, "sleep", lambda _delay: None)

    request: dict[str, Any] = {
        "messages": [{"role": "user", "content": "safe request"}],
        "model": "safe-model",
        "api_key": "safe-key",
        "base_url": f"https://{provider}.example.test/v1",
        "stream": streaming,
        "app_config": {},
    }
    operation = adapter_type().stream if streaming else adapter_type().chat

    with pytest.raises(expected_error) as exc_info:
        operation(request)

    rendered_error = f"{exc_info.value!s}\n{exc_info.value!r}"
    client = clients[0]
    assert (
        calls["n"],
        getattr(exc_info.value, "status_code", None),
        sentinel not in rendered_error,
        exc_info.value.__cause__,
        exc_info.value.__context__,
        client.stream_context_exits,
        client.close_calls,
    ) == (1, expected_status, True, None, None, int(streaming), 1)
