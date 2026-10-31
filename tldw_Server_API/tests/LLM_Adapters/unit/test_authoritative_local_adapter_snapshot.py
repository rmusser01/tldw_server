from __future__ import annotations

import asyncio
import gc
import json
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import pytest

from tldw_Server_API.app.core.AuthNZ import byok_helpers, byok_runtime
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
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatConfigurationError,
    ChatProviderError,
)
from tldw_Server_API.app.core.LLM_Calls.providers import local_adapters
from tldw_Server_API.app.core.LLM_Calls.providers.local_adapters import (
    AphroditeAdapter,
    KoboldAdapter,
    LlamaCppAdapter,
    LocalLLMAdapter,
    OllamaAdapter,
    OobaAdapter,
    TabbyAPIAdapter,
    VLLMAdapter,
)

_LOCAL_ADAPTERS = (
    LocalLLMAdapter,
    LlamaCppAdapter,
    KoboldAdapter,
    OobaAdapter,
    TabbyAPIAdapter,
    VLLMAdapter,
    OllamaAdapter,
    AphroditeAdapter,
)
_LOCAL_HANDLERS = (
    local_adapters._local_llm_request,
    local_adapters._llama_request,
    local_adapters._kobold_request,
    local_adapters._ooba_request,
    local_adapters._tabbyapi_request,
    local_adapters._vllm_request,
    local_adapters._ollama_request,
    local_adapters._aphrodite_request,
)
_LOCAL_PROVIDER_CASES = (
    (LocalLLMAdapter, "local-llm", "local_llm", "api_ip"),
    (LlamaCppAdapter, "llama.cpp", "llama_api", "api_ip"),
    (KoboldAdapter, "kobold", "kobold_api", "api_ip"),
    (OobaAdapter, "ooba", "ooba_api", "api_ip"),
    (TabbyAPIAdapter, "tabbyapi", "tabby_api", "api_ip"),
    (VLLMAdapter, "vllm", "vllm_api", "api_ip"),
    (OllamaAdapter, "ollama", "ollama_api", "api_url"),
    (AphroditeAdapter, "aphrodite", "aphrodite_api", "api_ip"),
)
_OPENAI_COMPATIBLE_LOCAL_PROVIDER_CASES = tuple(
    case for case in _LOCAL_PROVIDER_CASES if case[1] != "kobold"
)
_LOCAL_LLM_ENV_KEYS = (
    "LOCAL_LLM_API_KEY",
    "LOCAL_LLM_API_URL",
    "LOCAL_LLM_API_BASE",
    "LOCAL_LLM_API_IP",
    "LOCAL_LLM_BASE_URL",
    "LOCAL_LLM_MODEL",
    "LOCAL_LLM_TEMPERATURE",
    "LOCAL_LLM_STREAMING",
    "LOCAL_LLM_TOP_P",
    "LOCAL_LLM_TOP_K",
    "LOCAL_LLM_MIN_P",
    "LOCAL_LLM_MAX_TOKENS",
    "LOCAL_LLM_SEED",
    "LOCAL_LLM_STOP",
    "LOCAL_LLM_RESPONSE_FORMAT",
    "LOCAL_LLM_N",
    "LOCAL_LLM_PRESENCE_PENALTY",
    "LOCAL_LLM_FREQUENCY_PENALTY",
    "LOCAL_LLM_LOGPROBS",
    "LOCAL_LLM_TOP_LOGPROBS",
    "LOCAL_LLM_API_TIMEOUT",
    "LOCAL_LLM_API_RETRIES",
    "LOCAL_LLM_API_RETRY_DELAY",
    "LOCAL_LLM_STRICT_OPENAI_COMPAT",
)


def _request(app_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "snapshot-model",
        "app_config": app_config,
    }


def _response() -> dict[str, Any]:
    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ]
    }


def _resolved_request(
    provider: str,
    endpoint: str,
    api_key: str,
) -> dict[str, Any]:
    handle = _issue_local_credentials(provider, endpoint, api_key)
    request = _request({})
    request[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] = handle
    return request


def _issue_local_credentials(
    provider: str,
    endpoint: str | None,
    api_key: str | None,
    *,
    app_config: dict[str, Any] | None = None,
) -> ProviderCallCredentials:
    """Issue one real runtime handle for an exact local key/endpoint pair."""

    captured_config = (
        app_config
        if app_config is not None
        else build_app_config_overrides(
            provider,
            {"base_url": endpoint} if endpoint else {},
        )
    )

    async def resolver(
        normalized_provider: str,
        **_kwargs: Any,
    ) -> ResolvedByokCredentials:
        return ResolvedByokCredentials(
            provider=normalized_provider,
            api_key=api_key,
            app_config=captured_config,
            credential_fields={"base_url": endpoint} if endpoint else {},
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


@pytest.mark.unit
def test_runtime_issued_local_credentials_freeze_key_endpoint_and_scope() -> None:
    handle = _issue_local_credentials(
        "local-llm",
        "https://snapshot-a.example/v1",
        "snapshot-key-a",
    )

    assert handle.api_key == "snapshot-key-a"
    assert handle.trusted_endpoint is not None
    assert handle.trusted_endpoint.base_url == "https://snapshot-a.example/v1"
    assert handle.trusted_endpoint.scope.matches(
        "https://snapshot-a.example/v1/chat/completions"
    )
    with pytest.raises(AttributeError):
        handle.api_key = "late-key-b"
    with pytest.raises(AttributeError):
        handle.trusted_endpoint = None


@pytest.mark.unit
def test_local_adapter_rejects_forged_resolved_marker_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = LocalLLMAdapter()
    adapter._handler = lambda **_kwargs: pytest.fail(
        "forged runtime credentials must not reach the handler"
    )
    monkeypatch.setattr(
        local_adapters,
        "resolve_trusted_provider_endpoint",
        lambda _provider: pytest.fail(
            "forged runtime credentials must not fall back to live config"
        ),
    )

    with pytest.raises(ChatConfigurationError, match="active runtime capability"):
        adapter.chat(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "model": "snapshot-model",
                "api_key": "forged-key",
                "app_config": {
                    "local_llm": {"api_ip": "https://forged.example/v1"}
                },
                "credentials_resolved": True,
            }
        )


@pytest.mark.unit
@pytest.mark.parametrize("adapter_type", _LOCAL_ADAPTERS)
@pytest.mark.parametrize("method_name", ["chat", "stream"])
def test_each_local_adapter_forwards_explicit_timeout_to_handler(
    adapter_type: type[Any],
    method_name: str,
) -> None:
    captured: list[dict[str, Any]] = []
    adapter = adapter_type()
    adapter._handler = lambda **kwargs: captured.append(kwargs) or _response()

    result = getattr(adapter, method_name)(_request({}), timeout=1.25)
    if method_name == "stream":
        list(result)

    assert captured[0]["timeout"] == 1.25


@pytest.mark.unit
@pytest.mark.parametrize(
    "adapter_type,provider,section,endpoint_field",
    _LOCAL_PROVIDER_CASES,
    ids=(
        "local-llm",
        "llama-cpp",
        "kobold",
        "ooba",
        "tabbyapi",
        "vllm",
        "ollama",
        "aphrodite",
    ),
)
def test_explicit_timeout_overrides_config_at_local_http_boundary(
    monkeypatch: pytest.MonkeyPatch,
    adapter_type: type[Any],
    provider: str,
    section: str,
    endpoint_field: str,
) -> None:
    captured: list[dict[str, Any]] = []
    kobold_responses: list[_KoboldResponse] = []
    app_config = {
        section: {
            endpoint_field: "https://example.com/selected-local/v1",
            "model": "snapshot-model",
            "api_timeout": 999,
        }
    }
    request = _request(app_config)
    adapter = adapter_type()

    if provider == "kobold":
        def capture_kobold_fetch(**kwargs: Any) -> _KoboldResponse:
            captured.append(kwargs)
            response = _KoboldResponse()
            kobold_responses.append(response)
            return response

        adapter.http_fetcher = capture_kobold_fetch
    else:
        monkeypatch.setattr(
            local_adapters,
            "_chat_with_openai_compatible_local_server",
            lambda **kwargs: captured.append(kwargs) or _response(),
        )

    adapter.chat(request, timeout=1.25)

    assert captured[0]["timeout"] == 1.25
    if kobold_responses:
        assert kobold_responses[0].status_checked is True
        assert kobold_responses[0].close_count == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "_adapter_type,provider,section,endpoint_field",
    _LOCAL_PROVIDER_CASES,
    ids=(
        "local-llm",
        "llama-cpp",
        "kobold",
        "ooba",
        "tabbyapi",
        "vllm",
        "ollama",
        "aphrodite",
    ),
)
def test_local_credential_base_url_projects_to_handler_endpoint_field(
    _adapter_type: type[Any],
    provider: str,
    section: str,
    endpoint_field: str,
) -> None:
    app_config = build_app_config_overrides(
        provider,
        {"base_url": "https://example.com/selected-local/v1"},
    )

    assert app_config == {
        section: {endpoint_field: "https://example.com/selected-local/v1"}
    }


@pytest.mark.unit
@pytest.mark.concurrent
@pytest.mark.parametrize(
    "adapter_type,provider,_section,_endpoint_field",
    _OPENAI_COMPATIBLE_LOCAL_PROVIDER_CASES,
    ids=(
        "local-llm",
        "llama-cpp",
        "ooba",
        "tabbyapi",
        "vllm",
        "ollama",
        "aphrodite",
    ),
)
def test_concurrent_local_full_dispatch_keeps_endpoint_and_key_pairs_isolated(
    monkeypatch: pytest.MonkeyPatch,
    adapter_type: type[Any],
    provider: str,
    _section: str,
    _endpoint_field: str,
) -> None:
    captured: list[tuple[str, str | None]] = []
    captured_guard = threading.Lock()
    gate = threading.Barrier(2)

    def capture(**kwargs: Any) -> dict[str, Any]:
        gate.wait(timeout=5)
        with captured_guard:
            captured.append((kwargs["api_base_url"], kwargs["api_key"]))
        return _response()

    monkeypatch.setattr(
        local_adapters,
        "_chat_with_openai_compatible_local_server",
        capture,
    )
    adapter = adapter_type()
    first_request = _resolved_request(
        provider,
        "https://snapshot-a.example/v1",
        "snapshot-key-a",
    )
    second_request = _resolved_request(
        provider,
        "https://snapshot-b.example/v1",
        "snapshot-key-b",
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(adapter.chat, first_request)
        second = executor.submit(adapter.chat, second_request)
        first.result(timeout=10)
        second.result(timeout=10)

    assert set(captured) == {
        ("https://snapshot-a.example/v1", "snapshot-key-a"),
        ("https://snapshot-b.example/v1", "snapshot-key-b"),
    }


class _KoboldResponse:
    def __init__(self) -> None:
        self.status_checked = False
        self.close_count = 0

    def raise_for_status(self) -> None:
        self.status_checked = True

    def json(self) -> dict[str, Any]:
        return {"results": [{"text": "ok"}]}

    def close(self) -> None:
        self.close_count += 1


class _OpaqueLocalBoundaryFailure(Exception):
    """Failure shape deliberately absent from the adapter's compatibility tuple."""


class _OpenAICompatibleResponse:
    def __init__(
        self,
        *,
        status_error: BaseException | None = None,
        stream_error: BaseException | None = None,
        exit_error: BaseException | None = None,
        stream_lines: list[bytes] | None = None,
    ) -> None:
        self._status_error = status_error
        self._stream_error = stream_error
        self._exit_error = exit_error
        self._stream_lines = stream_lines

    def __enter__(self) -> _OpenAICompatibleResponse:
        return self

    def __exit__(self, *_args: Any) -> None:
        if self._exit_error is not None:
            raise self._exit_error
        return None

    def raise_for_status(self) -> None:
        if self._status_error is not None:
            raise self._status_error

    def json(self) -> dict[str, Any]:
        return _response()

    def iter_lines(self):
        yield from self._stream_lines or [
            b'data: {"choices":[{"delta":{"content":"ok"}}]}'
        ]
        if self._stream_error is not None:
            raise self._stream_error

    def close(self) -> None:
        return None


class _OpenAICompatibleClient:
    def __init__(
        self,
        captured: list[float],
        *,
        request_error: BaseException | None = None,
        status_error: BaseException | None = None,
        stream_error: BaseException | None = None,
        exit_error: BaseException | None = None,
        stream_lines: list[bytes] | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self._captured = captured
        self._request_error = request_error
        self._status_error = status_error
        self._stream_error = stream_error
        self._exit_error = exit_error
        self._stream_lines = stream_lines
        self._close_error = close_error
        self.close_count = 0

    def post(self, *_args: Any, timeout: float, **_kwargs: Any) -> _OpenAICompatibleResponse:
        self._captured.append(timeout)
        if self._request_error is not None:
            raise self._request_error
        return _OpenAICompatibleResponse()

    def stream(
        self,
        *_args: Any,
        timeout: float,
        **_kwargs: Any,
    ) -> _OpenAICompatibleResponse:
        self._captured.append(timeout)
        return _OpenAICompatibleResponse(
            status_error=self._status_error,
            stream_error=self._stream_error,
            exit_error=self._exit_error,
            stream_lines=self._stream_lines,
        )

    def close(self) -> None:
        self.close_count += 1
        if self._close_error is not None:
            raise self._close_error


@pytest.mark.unit
@pytest.mark.parametrize("streaming", [False, True], ids=("chat", "stream"))
def test_openai_compatible_local_http_boundary_preserves_explicit_timeout(
    streaming: bool,
) -> None:
    request_timeouts: list[float] = []
    factory_timeouts: list[float] = []
    client = _OpenAICompatibleClient(request_timeouts)

    def factory(timeout: float) -> _OpenAICompatibleClient:
        factory_timeouts.append(timeout)
        return client

    def fetcher(**kwargs: Any) -> _OpenAICompatibleResponse:
        request_timeouts.append(kwargs["timeout"])
        return _OpenAICompatibleResponse()

    result = local_adapters._chat_with_openai_compatible_local_server(
        api_base_url="https://example.com/selected-local/v1",
        model_name="snapshot-model",
        input_data=[{"role": "user", "content": "hello"}],
        streaming=streaming,
        timeout=1.25,
        http_client_factory=factory,
        http_fetcher=None if streaming else fetcher,
    )
    if streaming:
        list(result)

    assert factory_timeouts == [1.25]
    assert request_timeouts == [1.25]
    assert client.close_count == 1


@pytest.mark.unit
def test_openai_compatible_local_nonstream_error_is_detached_and_sanitized(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel = "LOCAL-NONSTREAM-PRIVATE-URL-AND-KEY"
    caplog.set_level("DEBUG")

    with pytest.raises(ChatProviderError) as exc_info:
        local_adapters._chat_with_openai_compatible_local_server(
            api_base_url="https://example.com/selected-local/v1",
            model_name="snapshot-model",
            input_data=[{"role": "user", "content": "hello"}],
            timeout=1.25,
            http_client_factory=lambda _timeout: _OpenAICompatibleClient(
                [],
                request_error=RuntimeError(sentinel),
            ),
        )

    assert sentinel not in str(exc_info.value)
    assert sentinel not in caplog.text
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.unit
def test_openai_compatible_local_stream_error_has_canonical_wire_and_safe_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel = "LOCAL-STREAM-PRIVATE-URL-AND-KEY"
    caplog.set_level("DEBUG")

    chunks = list(
        local_adapters._chat_with_openai_compatible_local_server(
            api_base_url="https://example.com/selected-local/v1",
            model_name="snapshot-model",
            input_data=[{"role": "user", "content": "hello"}],
            streaming=True,
            timeout=1.25,
            http_client_factory=lambda _timeout: _OpenAICompatibleClient(
                [],
                stream_error=RuntimeError(sentinel),
            ),
        )
    )

    wire = "".join(chunks)
    assert sentinel not in wire
    assert sentinel not in caplog.text
    assert '"code":"provider_unavailable"' in wire.replace(" ", "")
    assert wire.replace(" ", "").count('"code":"provider_unavailable"') == 1
    assert wire.count("data: [DONE]") == 1


@pytest.mark.unit
def test_openai_compatible_local_in_band_error_is_replaced_and_terminal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel = "LOCAL-IN-BAND-PRIVATE-URL-AND-KEY"
    caplog.set_level("DEBUG")
    upstream_error = json.dumps(
        {"type": "error", "error": {"message": sentinel}},
        separators=(",", ":"),
    ).encode()

    chunks = list(
        local_adapters._chat_with_openai_compatible_local_server(
            api_base_url="https://example.com/selected-local/v1",
            model_name="snapshot-model",
            input_data=[{"role": "user", "content": "hello"}],
            streaming=True,
            timeout=1.25,
            http_client_factory=lambda _timeout: _OpenAICompatibleClient(
                [],
                stream_lines=[
                    b'data: {"choices":[{"delta":{"content":"ok"}}]}',
                    b"data: " + upstream_error,
                    b'data: {"choices":[{"delta":{"content":"must-not-pass"}}]}',
                ],
            ),
        )
    )

    wire = "".join(chunks)
    assert sentinel not in wire
    assert sentinel not in caplog.text
    assert "must-not-pass" not in wire
    assert wire.replace(" ", "").count('"code":"provider_unavailable"') == 1
    assert wire.count("data: [DONE]") == 1
    assert wire.index('"code"') < wire.index("data: [DONE]")


@pytest.mark.unit
def test_openai_compatible_local_context_exit_failure_has_one_terminal_sequence(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel = "LOCAL-CONTEXT-EXIT-PRIVATE-URL-AND-KEY"
    caplog.set_level("DEBUG")

    chunks = list(
        local_adapters._chat_with_openai_compatible_local_server(
            api_base_url="https://example.com/selected-local/v1",
            model_name="snapshot-model",
            input_data=[{"role": "user", "content": "hello"}],
            streaming=True,
            timeout=1.25,
            http_client_factory=lambda _timeout: _OpenAICompatibleClient(
                [],
                exit_error=_OpaqueLocalBoundaryFailure(sentinel),
            ),
        )
    )

    wire = "".join(chunks)
    assert sentinel not in wire
    assert sentinel not in caplog.text
    assert wire.replace(" ", "").count('"code":"provider_unavailable"') == 1
    assert wire.count("data: [DONE]") == 1
    assert wire.index('"code"') < wire.index("data: [DONE]")


@pytest.mark.unit
def test_openai_compatible_local_client_close_failure_cannot_break_terminal_wire(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel = "LOCAL-CLIENT-CLOSE-PRIVATE-URL-AND-KEY"
    caplog.set_level("DEBUG")

    chunks = list(
        local_adapters._chat_with_openai_compatible_local_server(
            api_base_url="https://example.com/selected-local/v1",
            model_name="snapshot-model",
            input_data=[{"role": "user", "content": "hello"}],
            streaming=True,
            timeout=1.25,
            http_client_factory=lambda _timeout: _OpenAICompatibleClient(
                [],
                close_error=_OpaqueLocalBoundaryFailure(sentinel),
            ),
        )
    )

    wire = "".join(chunks)
    assert sentinel not in wire
    assert sentinel not in caplog.text
    assert wire.count("data: [DONE]") == 1


@pytest.mark.unit
@pytest.mark.parametrize("streaming", [False, True], ids=("chat", "stream"))
def test_openai_compatible_local_http_error_is_sanitized_at_adapter_boundary(
    caplog: pytest.LogCaptureFixture,
    streaming: bool,
) -> None:
    sentinel = "LOCAL-HTTP-PRIVATE-URL-AND-BODY"
    request = httpx.Request(
        "POST",
        f"https://example.com/selected-local/{sentinel}/v1/chat/completions",
    )
    response = httpx.Response(
        502,
        request=request,
        content=f'{{"error":{{"message":"{sentinel}"}}}}'.encode(),
    )
    upstream_error = httpx.HTTPStatusError(
        sentinel,
        request=request,
        response=response,
    )
    caplog.set_level("DEBUG")
    client = _OpenAICompatibleClient(
        [],
        request_error=None if streaming else upstream_error,
        status_error=upstream_error if streaming else None,
    )

    def invoke():
        return local_adapters._chat_with_openai_compatible_local_server(
            api_base_url=f"https://example.com/selected-local/{sentinel}/v1",
            model_name="snapshot-model",
            input_data=[{"role": "user", "content": "hello"}],
            streaming=streaming,
            timeout=1.25,
            http_client_factory=lambda _timeout: client,
        )

    if streaming:
        wire = "".join(invoke())
        assert sentinel not in wire
        assert wire.replace(" ", "").count('"code":"provider_unavailable"') == 1
        assert wire.count("data: [DONE]") == 1
    else:
        with pytest.raises(ChatProviderError) as exc_info:
            invoke()
        assert sentinel not in str(exc_info.value)
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None

    assert sentinel not in caplog.text
    assert client.close_count == 1


@pytest.mark.unit
def test_unstarted_openai_compatible_local_stream_never_creates_client() -> None:
    created: list[_OpenAICompatibleClient] = []

    def factory(_timeout: float) -> _OpenAICompatibleClient:
        client = _OpenAICompatibleClient([])
        created.append(client)
        return client

    stream = local_adapters._chat_with_openai_compatible_local_server(
        api_base_url="https://example.com/selected-local/v1",
        model_name="snapshot-model",
        input_data=[{"role": "user", "content": "hello"}],
        streaming=True,
        timeout=1.25,
        http_client_factory=factory,
    )
    stream.close()

    assert created == []


@pytest.mark.unit
def test_started_openai_compatible_local_stream_closes_once_and_releases_key() -> None:
    class _Credential(str):
        pass

    credential = _Credential("LOCAL-STREAM-CREDENTIAL-MARKER")
    credential_ref = weakref.ref(credential)
    client = _OpenAICompatibleClient([])
    stream = local_adapters._chat_with_openai_compatible_local_server(
        api_base_url="https://example.com/selected-local/v1",
        model_name="snapshot-model",
        input_data=[{"role": "user", "content": "hello"}],
        api_key=credential,
        streaming=True,
        timeout=1.25,
        http_client_factory=lambda _timeout: client,
    )

    assert next(stream).startswith("data: ")
    stream.close()
    del credential
    gc.collect()

    assert client.close_count == 1
    assert credential_ref() is None


@pytest.mark.unit
@pytest.mark.concurrent
def test_concurrent_kobold_full_dispatch_keeps_endpoint_and_key_pairs_isolated(
) -> None:
    captured: list[tuple[str, str | None]] = []
    responses: list[_KoboldResponse] = []
    captured_guard = threading.Lock()
    gate = threading.Barrier(2)

    def capture_fetch(**kwargs: Any) -> _KoboldResponse:
        gate.wait(timeout=5)
        headers = kwargs["headers"]
        response = _KoboldResponse()
        with captured_guard:
            captured.append((kwargs["url"], headers.get("X-Api-Key")))
            responses.append(response)
        return response

    adapter = KoboldAdapter()
    adapter.http_fetcher = capture_fetch
    first_request = _resolved_request(
        "kobold",
        "https://snapshot-a.example",
        "snapshot-key-a",
    )
    second_request = _resolved_request(
        "kobold",
        "https://snapshot-b.example",
        "snapshot-key-b",
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(adapter.chat, first_request)
        second = executor.submit(adapter.chat, second_request)
        first.result(timeout=10)
        second.result(timeout=10)

    assert set(captured) == {
        ("https://snapshot-a.example", "snapshot-key-a"),
        ("https://snapshot-b.example", "snapshot-key-b"),
    }
    assert len(responses) == 2
    assert all(response.status_checked for response in responses)
    assert all(response.close_count == 1 for response in responses)


@pytest.mark.unit
def test_local_llm_static_snapshot_freezes_key_endpoint_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(byok_helpers, "load_and_log_configs", lambda **_kwargs: {})
    for name in _LOCAL_LLM_ENV_KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "snapshot-key-a")
    monkeypatch.setenv("LOCAL_LLM_API_URL", "https://snapshot-a.example/v1")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "snapshot-model-a")
    snapshot = byok_helpers.load_server_config_snapshot()

    monkeypatch.setenv("LOCAL_LLM_API_KEY", "late-key-b")
    monkeypatch.setenv("LOCAL_LLM_API_URL", "https://late-b.example/v1")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "late-model-b")
    monkeypatch.setattr(
        byok_runtime,
        "load_server_config_snapshot",
        lambda: snapshot,
    )
    fallback = byok_runtime.resolve_static_server_fallback("local-llm")
    captured: list[dict[str, Any]] = []
    monkeypatch.setattr(
        local_adapters,
        "_chat_with_openai_compatible_local_server",
        lambda **kwargs: captured.append(kwargs) or _response(),
    )
    handle = _issue_local_credentials(
        "local-llm",
        "https://snapshot-a.example/v1",
        fallback.api_key,
        app_config=dict(fallback.app_config or {}),
    )

    LocalLLMAdapter().chat(
        {
            "messages": [{"role": "user", "content": "hello"}],
            PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: handle,
        }
    )

    assert (
        captured[0]["api_base_url"],
        captured[0]["api_key"],
        captured[0]["model_name"],
    ) == (
        "https://snapshot-a.example/v1",
        "snapshot-key-a",
        "snapshot-model-a",
    )


@pytest.mark.unit
@pytest.mark.concurrent
def test_local_llm_loader_freezes_supported_behavior_through_adapter_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader_entered = threading.Event()
    release_loader = threading.Event()

    def gated_loader(**_kwargs: Any) -> dict[str, Any]:
        loader_entered.set()
        if not release_loader.wait(5):
            raise TimeoutError("local-LLM snapshot loader was not released")
        return {}

    generation_a = {
        "LOCAL_LLM_API_KEY": "snapshot-key-a",
        "LOCAL_LLM_API_URL": "https://snapshot-a.example/v1",
        "LOCAL_LLM_MODEL": "snapshot-model-a",
        "LOCAL_LLM_TEMPERATURE": "0.21",
        "LOCAL_LLM_STREAMING": "false",
        "LOCAL_LLM_TOP_P": "0.82",
        "LOCAL_LLM_TOP_K": "23",
        "LOCAL_LLM_MIN_P": "0.04",
        "LOCAL_LLM_MAX_TOKENS": "321",
        "LOCAL_LLM_SEED": "17",
        "LOCAL_LLM_STOP": '["SNAPSHOT-A"]',
        "LOCAL_LLM_RESPONSE_FORMAT": '{"type":"json_object"}',
        "LOCAL_LLM_N": "2",
        "LOCAL_LLM_PRESENCE_PENALTY": "0.13",
        "LOCAL_LLM_FREQUENCY_PENALTY": "0.14",
        "LOCAL_LLM_LOGPROBS": "true",
        "LOCAL_LLM_TOP_LOGPROBS": "4",
        "LOCAL_LLM_API_TIMEOUT": "17",
        "LOCAL_LLM_API_RETRIES": "3",
        "LOCAL_LLM_API_RETRY_DELAY": "2",
        "LOCAL_LLM_STRICT_OPENAI_COMPAT": "false",
    }
    generation_b = {
        name: "true" if name.endswith(("STREAMING", "LOGPROBS", "COMPAT")) else "999"
        for name in generation_a
    }
    generation_b.update(
        {
            "LOCAL_LLM_API_KEY": "late-key-b",
            "LOCAL_LLM_API_URL": "https://late-b.example/v1",
            "LOCAL_LLM_MODEL": "late-model-b",
            "LOCAL_LLM_STOP": '["LATE-B"]',
            "LOCAL_LLM_RESPONSE_FORMAT": '{"type":"text"}',
        }
    )
    monkeypatch.setattr(byok_helpers, "load_and_log_configs", gated_loader)
    for name in _LOCAL_LLM_ENV_KEYS:
        monkeypatch.delenv(name, raising=False)
    for name, value in generation_a.items():
        monkeypatch.setenv(name, value)

    with ThreadPoolExecutor(max_workers=1) as executor:
        snapshot_future = executor.submit(byok_helpers.load_server_config_snapshot)
        assert loader_entered.wait(5)
        for name, value in generation_b.items():
            monkeypatch.setenv(name, value)
        release_loader.set()
        snapshot = snapshot_future.result(timeout=5)

    fallback = byok_runtime.resolve_static_server_fallback_from_snapshot(
        "local-llm",
        snapshot,
    )
    captured: list[dict[str, Any]] = []
    monkeypatch.setattr(
        local_adapters,
        "_chat_with_openai_compatible_local_server",
        lambda **kwargs: captured.append(kwargs) or _response(),
    )
    handle = _issue_local_credentials(
        "local-llm",
        "https://snapshot-a.example/v1",
        fallback.api_key,
        app_config=dict(fallback.app_config or {}),
    )

    LocalLLMAdapter().chat(
        {
            "messages": [{"role": "user", "content": "hello"}],
            PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: handle,
        }
    )

    assert fallback.app_config["local_llm"]["streaming"] is False
    expected = {
        "api_base_url": "https://snapshot-a.example/v1",
        "api_key": "snapshot-key-a",
        "model_name": "snapshot-model-a",
        "temp": 0.21,
        "top_p": 0.82,
        "top_k": 23,
        "min_p": 0.04,
        "max_tokens": 321,
        "seed": 17,
        "stop": ["SNAPSHOT-A"],
        "response_format": {"type": "json_object"},
        "n": 2,
        "presence_penalty": 0.13,
        "frequency_penalty": 0.14,
        "logprobs": True,
        "top_logprobs": 4,
        "timeout": 17,
        "api_retries": 3,
        "api_retry_delay": 2,
        "filter_unknown_params": False,
    }
    assert {key: captured[0][key] for key in expected} == expected


@pytest.mark.unit
def test_empty_local_llm_static_snapshot_ignores_environment_added_later(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(byok_helpers, "load_and_log_configs", lambda **_kwargs: {})
    for name in _LOCAL_LLM_ENV_KEYS:
        monkeypatch.delenv(name, raising=False)
    snapshot = byok_helpers.load_server_config_snapshot()

    monkeypatch.setenv("LOCAL_LLM_API_KEY", "late-key-b")
    monkeypatch.setenv("LOCAL_LLM_API_URL", "https://late-b.example/v1")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "late-model-b")
    monkeypatch.setattr(
        byok_runtime,
        "load_server_config_snapshot",
        lambda: snapshot,
    )
    fallback = byok_runtime.resolve_static_server_fallback("local-llm")
    monkeypatch.setattr(
        local_adapters,
        "resolve_trusted_provider_endpoint",
        lambda _provider: pytest.fail(
            "an authentic empty snapshot must not adopt late live config"
        ),
    )
    handle = _issue_local_credentials(
        "local-llm",
        None,
        fallback.api_key,
        app_config=dict(fallback.app_config or {}),
    )

    with pytest.raises(ChatConfigurationError, match="endpoint is not configured"):
        LocalLLMAdapter().chat(
            {
                "messages": [{"role": "user", "content": "hello"}],
                PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: handle,
            }
        )

    assert fallback.api_key is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "adapter_type,provider,_section,_endpoint_field",
    _LOCAL_PROVIDER_CASES,
)
def test_resolved_local_adapter_forwards_authoritative_runtime_snapshot(
    adapter_type: type[Any],
    provider: str,
    _section: str,
    _endpoint_field: str,
) -> None:
    captured: list[dict[str, Any]] = []
    snapshot = build_app_config_overrides(
        provider,
        {"base_url": "https://snapshot.example/v1"},
    )
    snapshot["marker"] = "issued"
    handle = _issue_local_credentials(
        provider,
        "https://snapshot.example/v1",
        "snapshot-key",
        app_config=snapshot,
    )
    adapter = adapter_type()
    adapter._handler = lambda **kwargs: captured.append(kwargs) or _response()

    adapter.chat(
        {
            "messages": [{"role": "user", "content": "hello"}],
            PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: handle,
        }
    )

    assert captured[0]["app_config"] == snapshot
    assert captured[0]["credentials_resolved"] is True
    assert captured[0]["configured_endpoint_base_url"] == (
        "https://snapshot.example/v1"
    )
    assert captured[0]["configured_endpoint_scope"].matches(
        "https://snapshot.example/v1/chat/completions"
    )


@pytest.mark.unit
@pytest.mark.concurrent
@pytest.mark.parametrize(
    "adapter_type,provider,_section,_endpoint_field",
    _LOCAL_PROVIDER_CASES,
)
def test_concurrent_resolved_local_adapters_keep_snapshots_isolated(
    adapter_type: type[Any],
    provider: str,
    _section: str,
    _endpoint_field: str,
) -> None:
    first_snapshot = build_app_config_overrides(
        provider,
        {"base_url": "https://snapshot-a.example/v1"},
    )
    first_snapshot["marker"] = "first"
    second_snapshot = build_app_config_overrides(
        provider,
        {"base_url": "https://snapshot-b.example/v1"},
    )
    second_snapshot["marker"] = "second"
    first_handle = _issue_local_credentials(
        provider,
        "https://snapshot-a.example/v1",
        "snapshot-key-a",
        app_config=first_snapshot,
    )
    second_handle = _issue_local_credentials(
        provider,
        "https://snapshot-b.example/v1",
        "snapshot-key-b",
        app_config=second_snapshot,
    )
    captured: list[tuple[dict[str, Any], bool]] = []
    captured_guard = threading.Lock()
    gate = threading.Barrier(2)

    def capture(**kwargs: Any) -> dict[str, Any]:
        gate.wait(timeout=5)
        with captured_guard:
            captured.append(
                (kwargs["app_config"], kwargs["credentials_resolved"])
            )
        return _response()

    adapter = adapter_type()
    adapter._handler = capture
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            adapter.chat,
            {
                "messages": [{"role": "user", "content": "hello"}],
                PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: first_handle,
            },
        )
        second = executor.submit(
            adapter.chat,
            {
                "messages": [{"role": "user", "content": "hello"}],
                PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: second_handle,
            },
        )
        first.result(timeout=10)
        second.result(timeout=10)

    assert {snapshot["marker"] for snapshot, _resolved in captured} == {
        "first",
        "second",
    }
    assert all(resolved is True for _snapshot, resolved in captured)


@pytest.mark.unit
def test_authoritative_empty_local_snapshot_does_not_reload_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        local_adapters,
        "load_settings",
        lambda: pytest.fail("authoritative empty snapshot must not reload settings"),
    )

    selector = local_adapters._select_local_app_config

    assert selector({}, credentials_resolved=True) == {}


@pytest.mark.unit
@pytest.mark.parametrize("handler", _LOCAL_HANDLERS)
def test_each_local_handler_honors_authoritative_empty_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    handler: Any,
) -> None:
    monkeypatch.setattr(
        local_adapters,
        "load_settings",
        lambda: pytest.fail("authoritative local handler must not reload settings"),
    )
    monkeypatch.setattr(
        local_adapters,
        "_chat_with_openai_compatible_local_server",
        lambda **_kwargs: _response(),
    )

    try:
        handler(
            input_data=[{"role": "user", "content": "hello"}],
            model="snapshot-model",
            app_config={},
            credentials_resolved=True,
        )
    except ChatConfigurationError:
        # Most local handlers require an endpoint in the captured snapshot.
        # Reaching that bounded error proves they did not adopt later settings.
        pass


@pytest.mark.unit
def test_unmarked_local_snapshot_preserves_legacy_settings_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_config = {"local_llm": {"api_ip": "http://legacy.example"}}
    monkeypatch.setattr(local_adapters, "load_settings", lambda: legacy_config)

    selector = local_adapters._select_local_app_config

    assert selector({}, credentials_resolved=False) is legacy_config
