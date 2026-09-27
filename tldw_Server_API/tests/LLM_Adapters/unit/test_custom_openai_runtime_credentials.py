from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any

import httpx
import pytest
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.byok_config import build_app_config_overrides
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionStatus,
    ResolvedByokCredentials,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
    ProviderCallCredentials,
    ProviderCredentialRuntime,
)
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatConfigurationError,
    ChatProviderError,
)
from tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter import (
    CustomOpenAIAdapter2,
)


class _Response:
    status_code = 200

    def __init__(
        self,
        *,
        payload: dict[str, Any] | None = None,
        lines: list[str] | None = None,
    ) -> None:
        self._payload = payload or {
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        }
        self._lines = lines or []
        self.close_count = 0

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload

    def iter_lines(self):
        yield from self._lines

    def close(self) -> None:
        self.close_count += 1


@contextmanager
def _stream_context(response: _Response):
    try:
        yield response
    finally:
        response.close()


def _issue_credentials(
    *,
    provider: str = "custom-openai-api-2",
    endpoint: str | None,
    api_key: str | None,
) -> ProviderCallCredentials:
    """Issue one authentic runtime handle for an exact custom endpoint/key pair."""

    app_config = build_app_config_overrides(
        provider,
        {"base_url": endpoint} if endpoint is not None else {},
    )

    async def resolver(
        normalized_provider: str,
        **_kwargs: Any,
    ) -> ResolvedByokCredentials:
        return ResolvedByokCredentials(
            provider=normalized_provider,
            api_key=api_key,
            app_config=app_config,
            credential_fields={"base_url": endpoint} if endpoint is not None else {},
            source="user",
            allowlisted=True,
            status=(
                ByokResolutionStatus.RESOLVED
                if api_key is not None
                else ByokResolutionStatus.ABSENT
            ),
            auth_source="api_key" if api_key is not None else None,
        )

    async def issue() -> ProviderCallCredentials:
        runtime = ProviderCredentialRuntime(
            user_id=41,
            team_ids=(),
            org_ids=(),
            trusted_base_url_override=True,
            server_config_snapshot={},
            resolver=resolver,
        )
        try:
            return await runtime.resolve(provider)
        finally:
            await runtime.close()

    return asyncio.run(issue())


def _request(handle: ProviderCallCredentials) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "snapshot-model",
        "api_key": "loose-attacker-key",
        "app_config": {
            "custom_openai_api_2": {
                "api_base_url": "https://loose-attacker.example/v1"
            }
        },
        "credentials_resolved": True,
        PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: handle,
    }


@pytest.mark.unit
def test_chat_translation_forwards_authentic_handle_to_configured_custom_adapter() -> None:
    handle = _issue_credentials(
        endpoint="https://snapshot.example/v1",
        api_key="snapshot-key",
    )

    provider, request, internal = chat_service._build_adapter_request_from_chat_args(
        {
            "api_provider": "custom-openai-api-2",
            **_request(handle),
        }
    )

    assert provider == "custom-openai-api-2"
    assert request[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] is handle
    assert request["api_key"] == "snapshot-key"
    assert request["app_config"] == handle.app_config
    assert internal == {}


@pytest.mark.unit
def test_custom_adapter_rejects_forged_runtime_handle_before_dispatch() -> None:
    forged = ProviderCallCredentials(
        provider="custom-openai-api-2",
        api_key="forged-key",
        app_config={
            "custom_openai_api_2": {
                "api_base_url": "https://forged.example/v1"
            }
        },
        auth_source="user",
        runtime_generation=0,
        runtime_identity=object(),
        credential_identity=object(),
    )
    adapter = CustomOpenAIAdapter2()
    adapter.http_fetcher = lambda **_kwargs: pytest.fail(
        "forged credentials must not reach HTTP dispatch"
    )

    with pytest.raises(ChatConfigurationError, match="active runtime"):
        adapter.chat(_request(forged))


@pytest.mark.unit
def test_custom_adapter_fails_closed_for_authentic_snapshot_without_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = _issue_credentials(endpoint=None, api_key="snapshot-key")
    monkeypatch.setenv(
        "CUSTOM_OPENAI_API_IP_2",
        "https://late-environment.example/v1",
    )
    adapter = CustomOpenAIAdapter2()
    adapter.http_fetcher = lambda **_kwargs: pytest.fail(
        "an empty authentic snapshot must not dispatch"
    )

    with pytest.raises(ChatConfigurationError, match="endpoint is not configured"):
        adapter.chat(_request(handle))


@pytest.mark.unit
@pytest.mark.concurrent
def test_concurrent_custom_adapter_calls_keep_runtime_key_endpoint_pairs_atomic() -> None:
    handles = (
        _issue_credentials(
            endpoint="https://snapshot-a.example/v1",
            api_key="snapshot-key-a",
        ),
        _issue_credentials(
            endpoint="https://snapshot-b.example/v1",
            api_key="snapshot-key-b",
        ),
    )
    gate = threading.Barrier(2)
    capture_lock = threading.Lock()
    captured: list[tuple[str, str | None, bool]] = []

    def fetch(**kwargs: Any) -> _Response:
        gate.wait(timeout=5)
        scope = kwargs["configured_endpoint"]
        with capture_lock:
            captured.append(
                (
                    kwargs["url"],
                    kwargs["headers"].get("Authorization"),
                    scope.matches(kwargs["url"]),
                )
            )
        return _Response()

    adapter = CustomOpenAIAdapter2()
    adapter.http_fetcher = fetch
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(adapter.chat, _request(handle)) for handle in handles]
        for future in futures:
            future.result(timeout=10)

    assert set(captured) == {
        (
            "https://snapshot-a.example/v1/chat/completions",
            "Bearer snapshot-key-a",
            True,
        ),
        (
            "https://snapshot-b.example/v1/chat/completions",
            "Bearer snapshot-key-b",
            True,
        ),
    }


@pytest.mark.unit
def test_custom_adapter_replaces_nonstream_in_band_error_without_leaking() -> None:
    sentinel = "CUSTOM-NONSTREAM-PRIVATE-URL-AND-KEY"
    handle = _issue_credentials(
        endpoint="https://snapshot.example/v1",
        api_key="snapshot-key",
    )
    adapter = CustomOpenAIAdapter2()
    adapter.http_fetcher = lambda **_kwargs: _Response(
        payload={"error": {"message": sentinel, "type": "private"}}
    )
    logs: list[str] = []
    sink_id = logger.add(logs.append, level="DEBUG", format="{message}")
    try:
        with pytest.raises(ChatProviderError) as exc_info:
            adapter.chat(_request(handle))
    finally:
        logger.remove(sink_id)

    assert sentinel not in str(exc_info.value)
    assert sentinel not in "\n".join(logs)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.unit
def test_custom_adapter_sanitizes_http_error_body_url_logs_and_cause() -> None:
    sentinel = "CUSTOM-HTTP-PRIVATE-URL-BODY-AND-KEY"
    handle = _issue_credentials(
        endpoint="https://snapshot.example/v1",
        api_key="snapshot-key",
    )
    request = httpx.Request(
        "POST",
        f"https://{sentinel.lower()}.example/private",
    )
    response = httpx.Response(
        403,
        request=request,
        json={"error": {"message": sentinel, "type": "private"}},
    )
    upstream = httpx.HTTPStatusError(
        sentinel,
        request=request,
        response=response,
    )
    adapter = CustomOpenAIAdapter2()
    adapter.http_fetcher = lambda **_kwargs: (_ for _ in ()).throw(upstream)
    logs: list[str] = []
    sink_id = logger.add(logs.append, level="DEBUG", format="{message}")
    try:
        with pytest.raises(ChatAuthenticationError) as exc_info:
            adapter.chat(_request(handle))
    finally:
        logger.remove(sink_id)

    rendered = "\n".join([str(exc_info.value), *logs])
    assert sentinel not in rendered
    assert sentinel.lower() not in rendered
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.unit
def test_custom_adapter_replaces_stream_in_band_error_and_stops_forwarding() -> None:
    sentinel = "CUSTOM-STREAM-PRIVATE-URL-AND-KEY"
    handle = _issue_credentials(
        endpoint="https://snapshot.example/v1",
        api_key="snapshot-key",
    )
    response = _Response(
        lines=[
            'data: {"choices":[{"delta":{"content":"safe"}}]}',
            "data: "
            + json.dumps(
                {"type": "error", "error": {"message": sentinel}},
                separators=(",", ":"),
            ),
            'data: {"choices":[{"delta":{"content":"must-not-pass"}}]}',
        ]
    )
    adapter = CustomOpenAIAdapter2()
    adapter.http_streamer = lambda **_kwargs: _stream_context(response)

    wire = "".join(adapter.stream(_request(handle)))

    assert sentinel not in wire
    assert "must-not-pass" not in wire
    assert wire.replace(" ", "").count('"code":"provider_unavailable"') == 1
    assert wire.count("data: [DONE]") == 1
    assert response.close_count == 1


@pytest.mark.unit
def test_custom_adapter_defers_done_until_stream_context_exits_cleanly() -> None:
    sentinel = "CUSTOM-STREAM-CONTEXT-EXIT-PRIVATE"
    handle = _issue_credentials(
        endpoint="https://snapshot.example/v1",
        api_key="snapshot-key",
    )
    response = _Response(
        lines=['data: {"choices":[{"delta":{"content":"safe"}}]}']
    )

    @contextmanager
    def failing_context():
        try:
            yield response
        finally:
            response.close()
            raise RuntimeError(sentinel)

    adapter = CustomOpenAIAdapter2()
    adapter.http_streamer = lambda **_kwargs: failing_context()
    chunks: list[str] = []
    logs: list[str] = []
    sink_id = logger.add(logs.append, level="DEBUG", format="{message}")
    try:
        with pytest.raises(ChatProviderError):
            for chunk in adapter.stream(_request(handle)):
                chunks.append(chunk)
    finally:
        logger.remove(sink_id)

    assert all("[DONE]" not in chunk for chunk in chunks)
    assert sentinel not in "".join(chunks)
    assert sentinel not in "\n".join(logs)
    assert response.close_count == 1
