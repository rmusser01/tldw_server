from __future__ import annotations

import asyncio
import importlib
import inspect
import io
import json
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from dataclasses import replace
from hashlib import sha256
from types import ModuleType
from typing import Any

import httpx
import pytest
from loguru import logger

from tldw_Server_API.app.core.Slides.standalone_html_config import (
    CLOSED_ADAPTER_CATALOG,
    ResolvedExecutionTarget,
    ResolvedPrompt,
    SlidesStandaloneHtmlConfig,
    StandaloneHtmlInputLimits,
    StandaloneHtmlOutputLimits,
    StandaloneHtmlProviderLimits,
)

pytestmark = pytest.mark.unit


SYSTEM_PROMPT = "application-owned system prompt"
USER_CONTENT = "untrusted source-secret user content"
PROVIDER_SECRET = "provider-secret"
_TEST_TRANSPORT: ContextVar[httpx.AsyncBaseTransport | None] = ContextVar(
    "standalone_html_test_transport",
    default=None,
)
DOCUMENT = (
    '<!doctype html><html><head><meta charset="utf-8"><title>Deck</title></head>'
    '<body><section class="slide">Deck</section><script></script></body></html>'
)
EXPECTED_CLOSED_ADAPTER_MANIFEST = (
    (
        "openai_official_chat_v1",
        "openai",
        "https://api.openai.com:443/v1/chat/completions",
    ),
    (
        "anthropic_official_messages_v1",
        "anthropic",
        "https://api.anthropic.com:443/v1/messages",
    ),
    (
        "llamacpp_loopback_chat_v1_ipv4",
        "llama.cpp",
        "http://127.0.0.1:8080/v1/chat/completions",
    ),
    (
        "llamacpp_loopback_chat_v1_ipv6",
        "llama.cpp",
        "http://[::1]:8080/v1/chat/completions",
    ),
    (
        "ollama_loopback_chat_v1_ipv4",
        "ollama",
        "http://127.0.0.1:11434/v1/chat/completions",
    ),
    (
        "ollama_loopback_chat_v1_ipv6",
        "ollama",
        "http://[::1]:11434/v1/chat/completions",
    ),
)


@pytest.fixture
def provider_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    try:
        module = importlib.import_module("tldw_Server_API.app.core.Slides.standalone_html_provider")
    except ModuleNotFoundError:
        missing = ModuleType("standalone_html_provider_missing")

        class MissingProviderError(RuntimeError):
            pass

        async def fail_missing(**_kwargs: Any) -> bytes:
            pytest.fail("standalone_html_provider is not implemented")

        missing.StandaloneHtmlProviderError = MissingProviderError
        missing.generate_standalone_html = fail_missing
        missing.httpx = httpx
        return missing

    def isolated_test_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        assert "transport" not in kwargs
        transport = _TEST_TRANSPORT.get()
        if transport is not None:
            kwargs["transport"] = transport
        return httpx.AsyncClient(*args, **kwargs)

    monkeypatch.setattr(module, "_AsyncClient", isolated_test_client)
    return module


class _CountingStream(httpx.AsyncByteStream):
    def __init__(
        self,
        *chunks: bytes,
        block: bool = False,
    ) -> None:
        self.chunks = chunks
        self.block = block
        self.iterations = 0
        self.yielded: list[bytes] = []
        self.closed = False
        self.started = asyncio.Event()

    async def __aiter__(self):
        self.iterations += 1
        self.started.set()
        if self.block:
            await asyncio.Event().wait()
        for chunk in self.chunks:
            self.yielded.append(chunk)
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


def _target(adapter_id: str, *, model: str = "CaseSensitive-Model") -> ResolvedExecutionTarget:
    adapter = next(item for item in CLOSED_ADAPTER_CATALOG if item.adapter_id == adapter_id)
    return ResolvedExecutionTarget(
        provider=adapter.provider,
        model=model,
        adapter_id=adapter.adapter_id,
        endpoint_identity=adapter.endpoint_identity,
    )


def _config(
    target: ResolvedExecutionTarget,
    *,
    provider_response_bytes: int = 8_388_608,
    document_bytes: int = 1_048_576,
    overall_timeout: float = 180.0,
) -> SlidesStandaloneHtmlConfig:
    prompt_bytes = SYSTEM_PROMPT.encode("utf-8")
    return SlidesStandaloneHtmlConfig(
        feature_enabled=True,
        egress_enabled=True,
        enabled=True,
        disabled_reason=None,
        target=target,
        prompt=ResolvedPrompt(
            text=SYSTEM_PROMPT,
            sha256=sha256(prompt_bytes).hexdigest(),
            contract_version="slides.standalone_html.v1",
            byte_count=len(prompt_bytes),
        ),
        allowed_targets=(target,),
        input_limits=StandaloneHtmlInputLimits(
            max_request_bytes=4_194_304,
            max_source_chars=200_000,
            max_source_tokens=50_000,
            max_audience_chars=500,
            max_source_identifier_bytes=256,
            max_note_ids=100,
            max_rag_query_chars=20_000,
            max_rag_top_k=100,
        ),
        output_limits=StandaloneHtmlOutputLimits(
            max_provider_response_bytes=provider_response_bytes,
            max_document_bytes=document_bytes,
        ),
        provider_limits=StandaloneHtmlProviderLimits(
            connect_timeout_seconds=10.0,
            read_timeout_seconds=120.0,
            overall_timeout_seconds=overall_timeout,
            max_output_tokens=16_384,
        ),
        generation_config_revision="sha256:" + ("0" * 64),
        _revision_manifest="{}",
    )


def _response_body(provider: str, text: str = DOCUMENT) -> bytes:
    if provider == "anthropic":
        payload: dict[str, Any] = {
            "content": [{"type": "text", "text": text}],
        }
    else:
        payload = {"choices": [{"message": {"content": text}}]}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _response(
    body: bytes,
    *,
    status_code: int = 200,
    headers: list[tuple[bytes, bytes]] | None = None,
    stream: _CountingStream | None = None,
) -> tuple[httpx.Response, _CountingStream]:
    body_stream = stream or _CountingStream(body)
    return (
        httpx.Response(
            status_code,
            headers=headers,
            stream=body_stream,
        ),
        body_stream,
    )


async def _call(
    provider: ModuleType,
    config: SlidesStandaloneHtmlConfig,
    transport: httpx.AsyncBaseTransport,
    *,
    stored_target: ResolvedExecutionTarget | None = None,
    current_config_loader=None,
    provider_api_key: str | None = PROVIDER_SECRET,
    system_prompt: str = SYSTEM_PROMPT,
    user_content: str = USER_CONTENT,
) -> bytes:
    token = _TEST_TRANSPORT.set(transport)
    try:
        return await provider.generate_standalone_html(
            stored_target=stored_target or config.target,
            system_prompt=system_prompt,
            user_content=user_content,
            provider_api_key=provider_api_key,
            current_config_loader=current_config_loader or (lambda: config),
        )
    finally:
        _TEST_TRANSPORT.reset(token)


@pytest.fixture
def captured_logs(
    caplog: pytest.LogCaptureFixture,
) -> Iterator[Callable[[], str]]:
    log_capture = io.StringIO()
    sink_id = logger.add(log_capture)
    try:
        yield lambda: f"{caplog.text} {log_capture.getvalue()}"
    finally:
        logger.remove(sink_id)


def _assert_redacted(error: BaseException, logs: str) -> None:
    observed = f"{error!s} {error!r} {logs}"
    assert "provider-body-secret" not in observed
    assert "source-secret" not in observed
    assert PROVIDER_SECRET not in observed
    assert error.__context__ is None


def _request_identity(request: httpx.Request) -> str:
    host = request.url.host
    if ":" in host:
        host = f"[{host}]"
    port = request.url.port or {"http": 80, "https": 443}[request.url.scheme]
    return f"{request.url.scheme}://{host}:{port}{request.url.path}"


def test_closed_adapter_catalog_matches_the_literal_v1_manifest() -> None:
    actual = tuple(
        (adapter.adapter_id, adapter.provider, adapter.endpoint_identity) for adapter in CLOSED_ADAPTER_CATALOG
    )

    assert actual == EXPECTED_CLOSED_ADAPTER_MANIFEST


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", CLOSED_ADAPTER_CATALOG, ids=lambda item: item.adapter_id)
async def test_each_catalog_adapter_uses_exact_endpoint_payload_and_credentials(
    provider_module: ModuleType,
    adapter,
) -> None:
    target = _target(adapter.adapter_id)
    config = _config(target)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        body = _response_body(target.provider)
        response, _ = _response(
            body,
            headers=[(b"Content-Length", str(len(body)).encode("ascii"))],
        )
        return response

    result = await _call(provider_module, config, httpx.MockTransport(handler))

    assert result == DOCUMENT.encode("utf-8")
    assert len(calls) == 1
    request = calls[0]
    assert _request_identity(request) == target.endpoint_identity
    assert request.method == "POST"
    assert request.url.query == b""
    assert request.url.userinfo == b""
    assert request.headers["accept-encoding"] == "identity"
    assert request.headers["accept"] == "application/json"
    application_headers = set(request.headers) - {
        "connection",
        "content-length",
        "host",
        "user-agent",
    }
    expected_headers = {"accept", "accept-encoding", "content-type"}
    payload = json.loads(request.content)
    if target.provider == "anthropic":
        expected_headers.update({"anthropic-version", "x-api-key"})
        assert payload == {
            "model": target.model,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": USER_CONTENT}],
            "max_tokens": 16_384,
            "stream": False,
        }
        assert request.headers["x-api-key"] == PROVIDER_SECRET
        assert request.headers["anthropic-version"] == "2023-06-01"
        assert "authorization" not in request.headers
    else:
        assert payload == {
            "model": target.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_CONTENT},
            ],
            "max_tokens": 16_384,
            "stream": False,
        }
        if target.provider == "openai":
            expected_headers.add("authorization")
            assert request.headers["authorization"] == f"Bearer {PROVIDER_SECRET}"
            assert "x-api-key" not in request.headers
        else:
            assert "authorization" not in request.headers
            assert "x-api-key" not in request.headers
    assert application_headers == expected_headers


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "token_field"),
    [("gpt-4o-mini", "max_tokens"), ("gpt-5", "max_completion_tokens")],
)
async def test_openai_uses_the_model_specific_fixed_token_field(
    provider_module: ModuleType,
    model: str,
    token_field: str,
) -> None:
    target = _target("openai_official_chat_v1", model=model)
    config = _config(target)
    payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        response, _ = _response(_response_body(target.provider))
        return response

    await _call(provider_module, config, httpx.MockTransport(handler))

    assert payloads[0][token_field] == 16_384
    assert ({"max_tokens", "max_completion_tokens"} & payloads[0].keys()) == {token_field}


@pytest.mark.asyncio
async def test_client_disables_environment_redirects_and_shared_hooks(
    provider_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target("openai_official_chat_v1")
    config = _config(target)
    body = _response_body(target.provider)
    real_client = httpx.AsyncClient
    seen: dict[str, Any] = {}

    def handler(_request: httpx.Request) -> httpx.Response:
        response, _ = _response(body)
        return response

    transport = httpx.MockTransport(handler)

    def client_factory(*args: Any, **kwargs: Any):
        seen.update(kwargs)
        kwargs["transport"] = _TEST_TRANSPORT.get()
        return real_client(*args, **kwargs)

    monkeypatch.setattr(provider_module, "_AsyncClient", client_factory)

    await _call(provider_module, config, transport)

    assert seen["trust_env"] is False
    assert seen["follow_redirects"] is False
    assert "transport" not in seen
    assert "event_hooks" not in seen
    assert "proxy" not in seen
    assert seen.get("verify", True) is not False
    assert seen["timeout"].connect == 10.0
    assert seen["timeout"].read == 120.0


def test_public_generation_api_has_no_transport_override(
    provider_module: ModuleType,
) -> None:
    signature = inspect.signature(provider_module.generate_standalone_html)

    assert "transport" not in signature.parameters


@pytest.mark.asyncio
async def test_redirect_is_not_followed(provider_module: ModuleType) -> None:
    target = _target("openai_official_chat_v1")
    config = _config(target)
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        response, _ = _response(
            b"redirect-body-secret",
            status_code=307,
            headers=[(b"Location", b"https://attacker.invalid/collect")],
        )
        return response

    with pytest.raises(provider_module.StandaloneHtmlProviderError) as exc_info:
        await _call(provider_module, config, httpx.MockTransport(handler))

    assert exc_info.value.code == "standalone_html_provider_http_error"
    assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "override",
    [
        "base_url",
        "endpoint",
        "proxy",
        "fallback_provider",
        "fallback_model",
        "router",
        "extra_headers",
        "extra_body",
    ],
)
async def test_request_overrides_are_rejected_before_network(
    provider_module: ModuleType,
    override: str,
) -> None:
    target = _target("openai_official_chat_v1")
    config = _config(target)
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError("network must not be reached")

    transport = httpx.MockTransport(handler)
    kwargs: dict[str, Any] = {
        "stored_target": target,
        "system_prompt": SYSTEM_PROMPT,
        "user_content": USER_CONTENT,
        "provider_api_key": PROVIDER_SECRET,
        "current_config_loader": lambda: config,
        override: "https://attacker.invalid/override",
    }

    token = _TEST_TRANSPORT.set(transport)
    try:
        with pytest.raises(TypeError):
            await provider_module.generate_standalone_html(**kwargs)
    finally:
        _TEST_TRANSPORT.reset(token)

    assert calls == 0


@pytest.mark.asyncio
async def test_environment_endpoint_and_proxy_overrides_are_ignored(
    provider_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target("openai_official_chat_v1")
    config = _config(target)
    calls: list[httpx.Request] = []
    for name in (
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "ANTHROPIC_BASE_URL",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
    ):
        monkeypatch.setenv(name, "https://attacker.invalid:443/collect")

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        response, _ = _response(_response_body(target.provider))
        return response

    await _call(provider_module, config, httpx.MockTransport(handler))

    assert len(calls) == 1
    assert _request_identity(calls[0]) == target.endpoint_identity
    assert calls[0].url.query == b""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config_update", "stored_update", "expected_code"),
    [
        ({"feature_enabled": False, "enabled": False}, {}, "standalone_html_egress_disabled"),
        ({"egress_enabled": False, "enabled": False}, {}, "standalone_html_egress_disabled"),
        (
            {},
            {"endpoint_identity": "https://attacker.invalid:443/v1/chat/completions"},
            "standalone_html_endpoint_not_allowed",
        ),
        ({}, {"model": "Substituted-Model"}, "standalone_html_model_not_allowed"),
        ({}, {"provider": "anthropic"}, "standalone_html_endpoint_not_allowed"),
        (
            {},
            {"adapter_id": "anthropic_official_messages_v1"},
            "standalone_html_endpoint_not_allowed",
        ),
    ],
)
async def test_kills_and_target_changes_fail_before_network(
    provider_module: ModuleType,
    config_update: dict[str, Any],
    stored_update: dict[str, Any],
    expected_code: str,
) -> None:
    target = _target("openai_official_chat_v1")
    config = replace(_config(target), **config_update)
    stored_target = replace(target, **stored_update)
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError("network must not be reached")

    with pytest.raises(provider_module.StandaloneHtmlProviderError) as exc_info:
        await _call(
            provider_module,
            config,
            httpx.MockTransport(handler),
            stored_target=stored_target,
        )

    assert exc_info.value.code == expected_code
    assert calls == 0


@pytest.mark.asyncio
async def test_current_allowlist_removal_fails_before_network(
    provider_module: ModuleType,
) -> None:
    target = _target("openai_official_chat_v1")
    config = replace(_config(target), allowed_targets=())
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError("network must not be reached")

    with pytest.raises(provider_module.StandaloneHtmlProviderError) as exc_info:
        await _call(provider_module, config, httpx.MockTransport(handler))

    assert exc_info.value.code == "standalone_html_endpoint_not_allowed"
    assert calls == 0


@pytest.mark.asyncio
async def test_unrelated_aggregate_unavailability_does_not_disable_stored_target(
    provider_module: ModuleType,
) -> None:
    target = _target("openai_official_chat_v1")
    config = replace(
        _config(target),
        enabled=False,
        disabled_reason="validator_unavailable",
        target=None,
        prompt=None,
    )
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        response, _ = _response(_response_body(target.provider))
        return response

    result = await _call(
        provider_module,
        config,
        httpx.MockTransport(handler),
        stored_target=target,
    )

    assert result == DOCUMENT.encode()
    assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "adapter_id",
    ["openai_official_chat_v1", "anthropic_official_messages_v1"],
)
async def test_remote_adapter_requires_provider_credential_before_network(
    provider_module: ModuleType,
    adapter_id: str,
) -> None:
    target = _target(adapter_id)
    config = _config(target)
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError("network must not be reached")

    with pytest.raises(provider_module.StandaloneHtmlProviderError) as exc_info:
        await _call(
            provider_module,
            config,
            httpx.MockTransport(handler),
            provider_api_key=None,
        )

    assert exc_info.value.code == "standalone_html_provider_credentials_unavailable"
    assert calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "credential",
    ["provider-\N{SNOWMAN}-secret", "provider\tsecret", "x" * 4_097],
    ids=["non-ascii", "control", "oversize"],
)
async def test_remote_credential_must_be_bounded_header_safe_ascii(
    provider_module: ModuleType,
    credential: str,
) -> None:
    target = _target("openai_official_chat_v1")
    config = _config(target)
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError("network must not be reached")

    with pytest.raises(provider_module.StandaloneHtmlProviderError) as exc_info:
        await _call(
            provider_module,
            config,
            httpx.MockTransport(handler),
            provider_api_key=credential,
        )

    assert exc_info.value.code == "standalone_html_provider_credentials_unavailable"
    assert exc_info.value.__context__ is None
    assert credential not in repr(exc_info.value)
    assert calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_provider_response_bytes", float("nan")),
        ("max_provider_response_bytes", True),
        ("max_provider_response_bytes", 0),
        ("max_document_bytes", float("nan")),
        ("max_document_bytes", True),
        ("max_document_bytes", 0),
    ],
)
async def test_malformed_output_limit_snapshot_fails_before_network(
    provider_module: ModuleType,
    field: str,
    value: object,
) -> None:
    target = _target("openai_official_chat_v1")
    original = _config(target)
    output_limits = replace(original.output_limits, **{field: value})
    config = replace(original, output_limits=output_limits)
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError("network must not be reached")

    with pytest.raises(provider_module.StandaloneHtmlProviderError) as exc_info:
        await _call(provider_module, config, httpx.MockTransport(handler))

    assert exc_info.value.code == "standalone_html_provider_request_invalid"
    assert exc_info.value.__context__ is None
    assert calls == 0


@pytest.mark.asyncio
async def test_config_loader_failure_has_no_retained_secret_context(
    provider_module: ModuleType,
    captured_logs: Callable[[], str],
) -> None:
    target = _target("openai_official_chat_v1")
    config = _config(target)

    def fail_loader() -> SlidesStandaloneHtmlConfig:
        raise RuntimeError("source-secret provider-body-secret")

    with pytest.raises(provider_module.StandaloneHtmlProviderError) as exc_info:
        await _call(
            provider_module,
            config,
            httpx.MockTransport(lambda _request: pytest.fail("network reached")),
            current_config_loader=fail_loader,
        )

    assert exc_info.value.code == "standalone_html_endpoint_not_allowed"
    _assert_redacted(exc_info.value, captured_logs())


@pytest.mark.asyncio
async def test_current_configuration_is_rechecked_immediately_before_request(
    provider_module: ModuleType,
) -> None:
    target = _target("openai_official_chat_v1")
    config = _config(target)
    order: list[str] = []

    def load_config() -> SlidesStandaloneHtmlConfig:
        order.append("config")
        return config

    def handler(_request: httpx.Request) -> httpx.Response:
        order.append("network")
        body = _response_body(target.provider)
        response, _ = _response(body)
        return response

    await _call(
        provider_module,
        config,
        httpx.MockTransport(handler),
        current_config_loader=load_config,
    )

    assert order == ["config", "config", "network"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fresh_update", "expected_code"),
    [
        ({"feature_enabled": False}, "standalone_html_egress_disabled"),
        ({"egress_enabled": False}, "standalone_html_egress_disabled"),
        ({"allowed_targets": ()}, "standalone_html_endpoint_not_allowed"),
    ],
    ids=["feature-disabled", "egress-disabled", "target-removed"],
)
async def test_dispatch_rechecks_config_after_client_entry_before_network(
    provider_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    fresh_update: dict[str, Any],
    expected_code: str,
) -> None:
    target = _target("openai_official_chat_v1")
    initial = _config(target)
    current = [initial]
    loader_calls = 0
    network_calls = 0
    client_entered = asyncio.Event()
    config_flipped = asyncio.Event()

    def load_config() -> SlidesStandaloneHtmlConfig:
        nonlocal loader_calls
        loader_calls += 1
        return current[0]

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal network_calls
        network_calls += 1
        response, _ = _response(_response_body(target.provider))
        return response

    async def flip_config() -> None:
        await client_entered.wait()
        current[0] = replace(initial, **fresh_update)
        config_flipped.set()

    class SchedulingGapClient(httpx.AsyncClient):
        async def __aenter__(self):
            client_entered.set()
            await config_flipped.wait()
            return await super().__aenter__()

    def client_factory(*args: Any, **kwargs: Any) -> SchedulingGapClient:
        assert "transport" not in kwargs
        return SchedulingGapClient(
            *args,
            transport=httpx.MockTransport(handler),
            **kwargs,
        )

    monkeypatch.setattr(provider_module, "_AsyncClient", client_factory)
    flipper = asyncio.create_task(flip_config())
    try:
        with pytest.raises(provider_module.StandaloneHtmlProviderError) as exc_info:
            await provider_module.generate_standalone_html(
                stored_target=target,
                system_prompt=SYSTEM_PROMPT,
                user_content=USER_CONTENT,
                provider_api_key=PROVIDER_SECRET,
                current_config_loader=load_config,
            )
    finally:
        await flipper

    assert exc_info.value.code == expected_code
    assert exc_info.value.__context__ is None
    assert loader_calls == 2
    assert network_calls == 0


@pytest.mark.asyncio
async def test_post_client_entry_loader_failure_is_redacted_before_network(
    provider_module: ModuleType,
    captured_logs: Callable[[], str],
) -> None:
    target = _target("openai_official_chat_v1")
    config = _config(target)
    loader_calls = 0
    network_calls = 0

    def load_config() -> SlidesStandaloneHtmlConfig:
        nonlocal loader_calls
        loader_calls += 1
        if loader_calls == 1:
            return config
        raise RuntimeError("source-secret provider-body-secret")

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal network_calls
        network_calls += 1
        response, _ = _response(_response_body(target.provider))
        return response

    with pytest.raises(provider_module.StandaloneHtmlProviderError) as exc_info:
        await _call(
            provider_module,
            config,
            httpx.MockTransport(handler),
            current_config_loader=load_config,
        )

    assert exc_info.value.code == "standalone_html_endpoint_not_allowed"
    assert loader_calls == 2
    assert network_calls == 0
    _assert_redacted(exc_info.value, captured_logs())


@pytest.mark.asyncio
async def test_dispatch_does_not_yield_after_fresh_recheck_before_transport(
    provider_module: ModuleType,
) -> None:
    target = _target("openai_official_chat_v1")
    config = _config(target)
    loader_calls = 0
    order: list[str] = []
    config_flipped = asyncio.Event()

    def flip_config() -> None:
        order.append("config-flipped")
        config_flipped.set()

    def load_config() -> SlidesStandaloneHtmlConfig:
        nonlocal loader_calls
        loader_calls += 1
        if loader_calls == 2:
            asyncio.get_running_loop().call_soon(flip_config)
        return config

    def handler(_request: httpx.Request) -> httpx.Response:
        order.append("network")
        response, _ = _response(_response_body(target.provider))
        return response

    result = await _call(
        provider_module,
        config,
        httpx.MockTransport(handler),
        current_config_loader=load_config,
    )
    await config_flipped.wait()

    assert result == DOCUMENT.encode()
    assert loader_calls == 2
    assert order == ["network", "config-flipped"]


@pytest.mark.asyncio
async def test_fresh_attempt_snapshot_solely_controls_overall_timeout(
    provider_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target("openai_official_chat_v1")
    initial = _config(target, overall_timeout=0.01)
    fresh = _config(target, overall_timeout=1.0)
    snapshots = iter((initial, fresh))
    loader_calls = 0
    network_calls = 0

    def load_config() -> SlidesStandaloneHtmlConfig:
        nonlocal loader_calls
        loader_calls += 1
        return next(snapshots)

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal network_calls
        network_calls += 1
        response, _ = _response(_response_body(target.provider))
        return response

    class DelayedEntryClient(httpx.AsyncClient):
        async def __aenter__(self):
            await asyncio.sleep(0.02)
            return await super().__aenter__()

    def client_factory(*args: Any, **kwargs: Any) -> DelayedEntryClient:
        assert "transport" not in kwargs
        return DelayedEntryClient(
            *args,
            transport=httpx.MockTransport(handler),
            **kwargs,
        )

    monkeypatch.setattr(provider_module, "_AsyncClient", client_factory)
    result = await provider_module.generate_standalone_html(
        stored_target=target,
        system_prompt=SYSTEM_PROMPT,
        user_content=USER_CONTENT,
        provider_api_key=PROVIDER_SECRET,
        current_config_loader=load_config,
    )

    assert result == DOCUMENT.encode()
    assert loader_calls == 2
    assert network_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [200, 429])
@pytest.mark.parametrize(
    "headers",
    [
        [(b"Content-Encoding", b"gzip")],
        [(b"Content-Encoding", b"br")],
        [(b"Content-Encoding", b"deflate")],
        [(b"Content-Encoding", b"zstd")],
        [(b"Content-Encoding", b"compress")],
        [(b"Content-Encoding", b"identity, gzip")],
        [(b"Content-Encoding", b"identity"), (b"Content-Encoding", b"identity")],
        [(b"Content-Encoding", b"\xff")],
    ],
)
async def test_nonidentity_or_conflicting_content_encoding_is_rejected_before_body_read(
    provider_module: ModuleType,
    headers: list[tuple[bytes, bytes]],
    status_code: int,
) -> None:
    target = _target("openai_official_chat_v1")
    config = _config(target)
    body = _response_body(target.provider)
    response, stream = _response(body, headers=headers, status_code=status_code)

    with pytest.raises(provider_module.StandaloneHtmlProviderError) as exc_info:
        await _call(
            provider_module,
            config,
            httpx.MockTransport(lambda _request: response),
        )

    assert exc_info.value.code == "standalone_html_provider_response_invalid"
    assert exc_info.value.__context__ is None
    assert stream.iterations == 0
    assert stream.closed is True


@pytest.mark.asyncio
async def test_single_identity_content_encoding_is_accepted(
    provider_module: ModuleType,
) -> None:
    target = _target("openai_official_chat_v1")
    config = _config(target)
    response, _ = _response(
        _response_body(target.provider),
        headers=[(b"Content-Encoding", b"identity")],
    )

    result = await _call(
        provider_module,
        config,
        httpx.MockTransport(lambda _request: response),
    )

    assert result == DOCUMENT.encode()


@pytest.mark.asyncio
async def test_declared_oversize_response_is_rejected_before_body_read(
    provider_module: ModuleType,
) -> None:
    target = _target("openai_official_chat_v1")
    config = _config(target, provider_response_bytes=64)
    response, stream = _response(
        b"provider-body-secret",
        headers=[(b"Content-Length", b"65")],
    )

    with pytest.raises(provider_module.StandaloneHtmlProviderError) as exc_info:
        await _call(
            provider_module,
            config,
            httpx.MockTransport(lambda _request: response),
        )

    assert exc_info.value.code == "standalone_html_provider_response_too_large"
    assert stream.iterations == 0


@pytest.mark.asyncio
async def test_extreme_decimal_content_length_is_fixed_oversize_error_before_body_read(
    provider_module: ModuleType,
) -> None:
    target = _target("openai_official_chat_v1")
    config = _config(target)
    response, stream = _response(
        b"provider-body-secret",
        headers=[(b"Content-Length", b"9" * 5_000)],
    )

    with pytest.raises(provider_module.StandaloneHtmlProviderError) as exc_info:
        await _call(
            provider_module,
            config,
            httpx.MockTransport(lambda _request: response),
        )

    assert exc_info.value.code == "standalone_html_provider_response_too_large"
    assert stream.iterations == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        [],
        [(b"Transfer-Encoding", b"chunked")],
        [(b"Content-Length", b"1")],
    ],
)
async def test_raw_stream_cap_cannot_be_bypassed_by_missing_chunked_or_dishonest_length(
    provider_module: ModuleType,
    headers: list[tuple[bytes, bytes]],
    captured_logs: Callable[[], str],
) -> None:
    target = _target("openai_official_chat_v1")
    config = _config(target, provider_response_bytes=64)
    stream = _CountingStream(b"x" * 32, b"y" * 33, b"must-not-be-read")
    response, _ = _response(b"", headers=headers, stream=stream)

    with pytest.raises(provider_module.StandaloneHtmlProviderError) as exc_info:
        await _call(
            provider_module,
            config,
            httpx.MockTransport(lambda _request: response),
        )

    assert exc_info.value.code == "standalone_html_provider_response_too_large"
    assert stream.closed is True
    assert stream.yielded == [b"x" * 32, b"y" * 33]
    _assert_redacted(exc_info.value, captured_logs())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        [(b"Content-Length", b"65")],
        [],
        [(b"Transfer-Encoding", b"chunked")],
        [(b"Content-Length", b"1")],
    ],
    ids=["declared", "missing", "chunked", "dishonest"],
)
async def test_non_2xx_body_uses_the_same_raw_cap_and_stops_early(
    provider_module: ModuleType,
    headers: list[tuple[bytes, bytes]],
) -> None:
    target = _target("openai_official_chat_v1")
    config = _config(target, provider_response_bytes=64)
    stream = _CountingStream(
        b"provider-body-secret".ljust(32, b"x"),
        b"source-secret".ljust(33, b"y"),
        b"must-not-be-read",
    )
    response, _ = _response(
        b"",
        status_code=429,
        headers=headers,
        stream=stream,
    )

    with pytest.raises(provider_module.StandaloneHtmlProviderError) as exc_info:
        await _call(
            provider_module,
            config,
            httpx.MockTransport(lambda _request: response),
        )

    assert exc_info.value.code == "standalone_html_provider_response_too_large"
    if headers == [(b"Content-Length", b"65")]:
        assert stream.yielded == []
    else:
        assert stream.yielded == list(stream.chunks[:2])
    assert stream.closed is True


@pytest.mark.asyncio
async def test_under_limit_dishonest_content_length_is_invalid(
    provider_module: ModuleType,
) -> None:
    target = _target("openai_official_chat_v1")
    config = _config(target)
    body = _response_body(target.provider)
    response, _ = _response(
        body,
        headers=[(b"Content-Length", str(len(body) - 1).encode("ascii"))],
    )

    with pytest.raises(provider_module.StandaloneHtmlProviderError) as exc_info:
        await _call(
            provider_module,
            config,
            httpx.MockTransport(lambda _request: response),
        )

    assert exc_info.value.code == "standalone_html_provider_response_invalid"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        (b"[" * 65) + b"0" + (b"]" * 65),
        json.dumps({"containers": [[] for _ in range(25_001)]}, separators=(",", ":")).encode(),
        json.dumps({"items": [0 for _ in range(100_001)]}, separators=(",", ":")).encode(),
        json.dumps(
            {
                **{f"k{index}": 0 for index in range(50_001)},
                "choices": [{"message": {"content": DOCUMENT}}],
            },
            separators=(",", ":"),
        ).encode(),
        b'{"choices":[{"message":{"content":"' + (b"x" * (7 * 1024 * 1024 + 1)) + b'"}}]}',
    ],
    ids=["depth", "containers", "items", "tokens", "encoded-string"],
)
async def test_provider_success_json_lexical_budgets_fail_closed(
    provider_module: ModuleType,
    body: bytes,
    captured_logs: Callable[[], str],
) -> None:
    target = _target("openai_official_chat_v1")
    config = _config(target)
    response, _ = _response(body)

    with pytest.raises(provider_module.StandaloneHtmlProviderError) as exc_info:
        await _call(
            provider_module,
            config,
            httpx.MockTransport(lambda _request: response),
        )

    assert exc_info.value.code == "standalone_html_provider_response_invalid"
    _assert_redacted(exc_info.value, captured_logs())


def test_json_preflight_accepts_depth_64_and_rejects_depth_65(
    provider_module: ModuleType,
) -> None:
    at_limit = provider_module._ProviderJsonPreflight()
    at_limit.feed((b"[" * 64) + b"0" + (b"]" * 64))
    at_limit.finish()

    over_limit = provider_module._ProviderJsonPreflight()
    with pytest.raises(provider_module.StandaloneHtmlProviderError):
        over_limit.feed((b"[" * 65) + b"0" + (b"]" * 65))


def test_json_preflight_accepts_container_limit_and_rejects_next(
    provider_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provider_module, "_PROVIDER_JSON_MAX_DEPTH", 30_000)
    monkeypatch.setattr(provider_module, "_PROVIDER_JSON_MAX_TOKENS", 1_000_000)

    at_limit = provider_module._ProviderJsonPreflight()
    at_limit.feed((b"[" * 25_000) + (b"]" * 25_000))
    at_limit.finish()

    over_limit = provider_module._ProviderJsonPreflight()
    with pytest.raises(provider_module.StandaloneHtmlProviderError):
        over_limit.feed((b"[" * 25_001) + (b"]" * 25_001))


def test_json_preflight_accepts_nearest_token_limit_and_rejects_next_document(
    provider_module: ModuleType,
) -> None:
    at_limit = provider_module._ProviderJsonPreflight()
    at_limit.feed(b"[" + (b"0," * 99_998) + b"0]")
    at_limit.finish()

    over_limit = provider_module._ProviderJsonPreflight()
    with pytest.raises(provider_module.StandaloneHtmlProviderError):
        over_limit.feed(b"[" + (b"0," * 99_999) + b"0]")


def test_json_preflight_counts_mapping_members_at_the_fixed_limit(
    provider_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provider_module, "_PROVIDER_JSON_MAX_TOKENS", 1_000_000)

    at_limit_body = b"{" + b",".join(f'"k{index}":0'.encode() for index in range(100_000)) + b"}"
    at_limit = provider_module._ProviderJsonPreflight()
    at_limit.feed(at_limit_body)
    at_limit.finish()

    over_limit = provider_module._ProviderJsonPreflight()
    with pytest.raises(provider_module.StandaloneHtmlProviderError):
        over_limit.feed(at_limit_body[:-1] + b',"overflow":0}')


def test_json_preflight_accepts_encoded_string_limit_and_rejects_next_byte(
    provider_module: ModuleType,
) -> None:
    at_limit = provider_module._ProviderJsonPreflight()
    at_limit.feed(b'"' + (b"x" * (7 * 1024 * 1024)) + b'"')
    at_limit.finish()

    over_limit = provider_module._ProviderJsonPreflight()
    with pytest.raises(provider_module.StandaloneHtmlProviderError):
        over_limit.feed(b'"' + (b"x" * (7 * 1024 * 1024 + 1)) + b'"')


@pytest.mark.asyncio
async def test_json_preflight_rejects_split_oversize_string_before_materialization(
    provider_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target("openai_official_chat_v1")
    config = _config(target)
    decoder_called = False

    def forbidden_decode(_raw: bytes) -> object:
        nonlocal decoder_called
        decoder_called = True
        raise AssertionError("materializing decoder must not run")

    monkeypatch.setattr(provider_module, "_strict_json_loads", forbidden_decode)
    prefix = b'{"choices":[{"message":{"content":"'
    first = prefix + (b"x" * (7 * 1024 * 1024))
    second = b"x"
    stream = _CountingStream(first, second, b'"}}]}')
    response, _ = _response(b"", stream=stream)

    with pytest.raises(provider_module.StandaloneHtmlProviderError) as exc_info:
        await _call(
            provider_module,
            config,
            httpx.MockTransport(lambda _request: response),
        )

    assert exc_info.value.code == "standalone_html_provider_response_invalid"
    assert decoder_called is False
    assert stream.yielded == [first, second]


@pytest.mark.asyncio
@pytest.mark.parametrize("ensure_ascii", [True, False], ids=["escape", "utf8-codepoint"])
async def test_json_preflight_preserves_string_state_across_chunk_boundaries(
    provider_module: ModuleType,
    ensure_ascii: bool,
) -> None:
    target = _target("openai_official_chat_v1")
    config = _config(target)
    document = DOCUMENT.replace("Deck", "Snowman \N{SNOWMAN}")
    payload = {"choices": [{"message": {"content": document}}]}
    body = json.dumps(payload, ensure_ascii=ensure_ascii, separators=(",", ":")).encode()
    marker = b"\\u2603" if ensure_ascii else "\N{SNOWMAN}".encode()
    split_at = body.index(marker) + 1
    stream = _CountingStream(body[:split_at], body[split_at:])
    response, _ = _response(b"", stream=stream)

    result = await _call(
        provider_module,
        config,
        httpx.MockTransport(lambda _request: response),
    )

    assert result == document.encode()
    assert stream.yielded == list(stream.chunks)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        b'{"choices":[{"message":{"content":"<!doctype html><html></html>",'
        b'"content":"<!doctype html><html></html>"}}]}',
        b'{"choices":[{"message":{"content":"<!doctype html><html></html>"}}],"x":NaN}',
        b'{"choices":[{"message":{"content":"<!doctype html><html></html>"}}],"x":Infinity}',
        b'{"choices":[{"message":{"content":"<!doctype html><html></html>"}}],"x":-Infinity}',
        b'{"choices":[{"message":{"content":"\\ud800"}}]}',
        b'{"choices":[{"message":{"content":"\\udc00"}}]}',
        b'{"choices":[{"message":{"content":"\xed\xa0\x80"}}]}',
        b'{"choices":[{"message":{"content":"<!doctype html><html></html>"}}],"x":"\xff"}',
    ],
    ids=[
        "duplicate",
        "nan",
        "infinity",
        "negative-infinity",
        "high-lone-surrogate",
        "low-lone-surrogate",
        "surrogate-utf8",
        "invalid-utf8",
    ],
)
async def test_provider_json_is_strict_and_scalar_unicode_only(
    provider_module: ModuleType,
    body: bytes,
    captured_logs: Callable[[], str],
) -> None:
    target = _target("openai_official_chat_v1")
    config = _config(target)
    response, _ = _response(body)

    with pytest.raises(provider_module.StandaloneHtmlProviderError) as exc_info:
        await _call(
            provider_module,
            config,
            httpx.MockTransport(lambda _request: response),
        )

    assert exc_info.value.code == "standalone_html_provider_response_invalid"
    _assert_redacted(exc_info.value, captured_logs())


@pytest.mark.asyncio
async def test_provider_json_accepts_a_valid_surrogate_pair(
    provider_module: ModuleType,
) -> None:
    target = _target("openai_official_chat_v1")
    config = _config(target)
    document = DOCUMENT.replace("Deck", "Deck \N{GRINNING FACE}")
    body = json.dumps(
        {"choices": [{"message": {"content": document}}]},
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode()
    assert b"\\ud83d\\ude00" in body
    response, _ = _response(body)

    result = await _call(
        provider_module,
        config,
        httpx.MockTransport(lambda _request: response),
    )

    assert result == document.encode()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        [],
        [{"type": "tool_use", "text": DOCUMENT}],
        [{"type": "text"}],
        [{"type": "text", "text": DOCUMENT}, {"type": "text", "text": DOCUMENT}],
        [{"type": "text", "text": DOCUMENT, "unexpected": True}],
    ],
)
async def test_anthropic_codec_rejects_malformed_or_ambiguous_content_blocks(
    provider_module: ModuleType,
    content: list[dict[str, Any]],
) -> None:
    target = _target("anthropic_official_messages_v1")
    config = _config(target)
    body = json.dumps({"content": content}, separators=(",", ":")).encode()
    response, _ = _response(body)

    with pytest.raises(provider_module.StandaloneHtmlProviderError) as exc_info:
        await _call(
            provider_module,
            config,
            httpx.MockTransport(lambda _request: response),
        )

    assert exc_info.value.code == "standalone_html_provider_response_invalid"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "accepted"),
    [
        (DOCUMENT, True),
        (f"  \n{DOCUMENT}\n  ", True),
        (f"```html\n{DOCUMENT}\n```", True),
        (f"```HTML\r\n{DOCUMENT}\r\n```", True),
        (f"```\n{DOCUMENT}\n```", True),
        (f"prose before\n{DOCUMENT}", False),
        (f"{DOCUMENT}\nprose after", False),
        (f"prose\n```html\n{DOCUMENT}\n```", False),
        (f"```html\n{DOCUMENT}\n```\nprose", False),
        (f"```python\n{DOCUMENT}\n```", False),
        (f"```html\n{DOCUMENT}", False),
        (DOCUMENT.replace("</body>", "```\n</body>"), False),
    ],
)
async def test_only_one_complete_outer_markdown_fence_is_tolerated(
    provider_module: ModuleType,
    text: str,
    accepted: bool,
) -> None:
    target = _target("openai_official_chat_v1")
    config = _config(target)
    body = _response_body(target.provider, text)
    response, _ = _response(body)
    transport = httpx.MockTransport(lambda _request: response)

    if accepted:
        assert await _call(provider_module, config, transport) == DOCUMENT.encode("utf-8")
    else:
        with pytest.raises(provider_module.StandaloneHtmlProviderError) as exc_info:
            await _call(provider_module, config, transport)
        assert exc_info.value.code == "standalone_html_provider_response_invalid"


@pytest.mark.asyncio
async def test_non_2xx_body_is_bounded_discarded_and_never_echoed(
    provider_module: ModuleType,
    captured_logs: Callable[[], str],
) -> None:
    target = _target("openai_official_chat_v1")
    config = _config(target)
    body = b"provider-body-secret source-secret"
    response, stream = _response(body, status_code=429)
    with pytest.raises(provider_module.StandaloneHtmlProviderError) as exc_info:
        await _call(
            provider_module,
            config,
            httpx.MockTransport(lambda _request: response),
        )

    error = exc_info.value
    assert error.code == "standalone_html_provider_http_error"
    assert error.status_code == 429
    assert stream.iterations == 1
    _assert_redacted(error, captured_logs())


@pytest.mark.asyncio
async def test_timeout_is_bounded_and_closes_response(
    provider_module: ModuleType,
    captured_logs: Callable[[], str],
) -> None:
    target = _target("openai_official_chat_v1")
    config = _config(target, overall_timeout=0.01)
    stream = _CountingStream(block=True)
    response, _ = _response(b"", stream=stream)

    with pytest.raises(provider_module.StandaloneHtmlProviderError) as exc_info:
        await _call(
            provider_module,
            config,
            httpx.MockTransport(lambda _request: response),
        )

    assert exc_info.value.code == "standalone_html_provider_timeout"
    assert stream.closed is True
    _assert_redacted(exc_info.value, captured_logs())


@pytest.mark.asyncio
async def test_connection_failure_is_redacted(
    provider_module: ModuleType,
    captured_logs: Callable[[], str],
) -> None:
    target = _target("openai_official_chat_v1")
    config = _config(target)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            "provider-body-secret source-secret",
            request=request,
        )

    with pytest.raises(provider_module.StandaloneHtmlProviderError) as exc_info:
        await _call(provider_module, config, httpx.MockTransport(handler))

    assert exc_info.value.code == "standalone_html_provider_unavailable"
    _assert_redacted(exc_info.value, captured_logs())


@pytest.mark.asyncio
async def test_caller_cancellation_propagates_and_closes_response(
    provider_module: ModuleType,
    captured_logs: Callable[[], str],
) -> None:
    target = _target("openai_official_chat_v1")
    config = _config(target)
    stream = _CountingStream(block=True)
    response, _ = _response(b"", stream=stream)
    task = asyncio.create_task(
        _call(
            provider_module,
            config,
            httpx.MockTransport(lambda _request: response),
        )
    )
    await asyncio.wait_for(stream.started.wait(), timeout=1.0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert stream.closed is True
    observed = captured_logs()
    assert "source-secret" not in observed
    assert PROVIDER_SECRET not in observed


@pytest.mark.asyncio
async def test_extracted_document_has_independent_one_mib_limit(
    provider_module: ModuleType,
    captured_logs: Callable[[], str],
) -> None:
    target = _target("openai_official_chat_v1")
    config = _config(target)
    oversized = "<!doctype html><html>" + ("x" * 1_048_576) + "</html>"
    body = _response_body(target.provider, oversized)
    response, _ = _response(body)

    with pytest.raises(provider_module.StandaloneHtmlProviderError) as exc_info:
        await _call(
            provider_module,
            config,
            httpx.MockTransport(lambda _request: response),
        )

    assert exc_info.value.code == "standalone_html_provider_response_too_large"
    _assert_redacted(exc_info.value, captured_logs())


@pytest.mark.asyncio
async def test_provider_response_limit_is_clamped_to_fixed_eight_mib(
    provider_module: ModuleType,
) -> None:
    target = _target("openai_official_chat_v1")
    config = _config(target, provider_response_bytes=8_388_609)
    response, stream = _response(
        b"provider-body-secret",
        headers=[(b"Content-Length", b"8388609")],
    )

    with pytest.raises(provider_module.StandaloneHtmlProviderError) as exc_info:
        await _call(
            provider_module,
            config,
            httpx.MockTransport(lambda _request: response),
        )

    assert exc_info.value.code == "standalone_html_provider_response_too_large"
    assert stream.yielded == []


@pytest.mark.asyncio
async def test_document_limit_is_utf8_bytes_and_clamped_to_fixed_one_mib(
    provider_module: ModuleType,
) -> None:
    target = _target("openai_official_chat_v1")
    config = _config(target, document_bytes=2_097_152)
    oversized = "<!doctype html><html><body>" + ("é" * 524_288) + "</body></html>"
    assert len(oversized) < 1_048_576
    assert len(oversized.encode()) > 1_048_576
    body = _response_body(target.provider, oversized)
    response, _ = _response(body)

    with pytest.raises(provider_module.StandaloneHtmlProviderError) as exc_info:
        await _call(
            provider_module,
            config,
            httpx.MockTransport(lambda _request: response),
        )

    assert exc_info.value.code == "standalone_html_provider_response_too_large"
