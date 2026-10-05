"""Certification tests for the single-attempt native OpenAI transport."""

from __future__ import annotations

import asyncio
import gzip
import importlib
import importlib.util
import json
import logging
import os
from dataclasses import FrozenInstanceError, fields, replace
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from loguru import logger

from tldw_Server_API.app.core import http_client as hc
from tldw_Server_API.app.core.AuthNZ import byok_runtime
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    ProviderCallCredentials,
    ProviderCredentialRuntime,
    is_runtime_issued_provider_call_credentials,
)
from tldw_Server_API.app.core.AuthNZ.repos.provider_scope_result import (
    ProviderScopeResult,
    ProviderScopeStatus,
)
from tldw_Server_API.app.core.LLM_Calls.provider_config_resolution import TrustedProviderEndpoint
from tldw_Server_API.app.core.MCP_unified.adapters.model_completion.normalization import NormalizedModelCompletion
from tldw_Server_API.app.core.MCP_unified.interfaces.model_completion import (
    ModelCompletionFailure,
    ModelCompletionRequest,
    ModelFailureDomain,
)
from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope

MODULE = "tldw_Server_API.app.core.MCP_unified.adapters.model_completion.transport"
BASE = "https://fixed.example/v1"
URL = BASE + "/chat/completions"
SECRET = "sk-transport-sentinel"
PROMPT = "private-prompt-sentinel"
PRIVATE = f"{SECRET} {PROMPT} {URL} private-cause-sentinel"
pytestmark = pytest.mark.unit


def _api():
    assert importlib.util.find_spec(MODULE) is not None, "certified transport is not implemented"
    return importlib.import_module(MODULE)


def _endpoint(base=BASE):
    return TrustedProviderEndpoint(base_url=base, scope=ConfiguredEndpointScope.from_url(base))


def _policy(**overrides):
    values = {"provider": "openai", "model": "gpt-fixed", "endpoint": _endpoint(), "timeout_seconds": 7}
    values.update(overrides)
    return _api().OpenAITransportPolicy(**values)


def _request(**overrides):
    values = {
        "system_prompt": "system-sentinel",
        "user_prompt": PROMPT,
        "max_output_tokens": 32,
        "max_output_chars": 100,
        "max_output_bytes": 400,
        "max_provider_response_bytes": 1024,
    }
    values.update(overrides)
    return ModelCompletionRequest(**values)


def _envelope(content=" answer\r\nline "):
    return {
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 17, "completion_tokens": 3, "raw_private": PRIVATE},
        "private": PRIVATE,
    }


def _assert_failure(error, code, domain):
    assert type(error) is ModelCompletionFailure
    assert (error.code, error.domain, str(error), error.args) == (code, domain, code, (code,))
    assert error.__cause__ is None
    assert error.__context__ is None
    assert error.__suppress_context__ is True
    assert all(value not in repr(error) for value in (SECRET, PROMPT, URL, "private-cause-sentinel"))


@pytest.fixture(autouse=True)
def isolate_network(monkeypatch):
    """Mock DNS/egress only; requests and bounded streaming remain native HTTPX."""
    validations = []
    monkeypatch.delenv("HTTP_CERT_PINS", raising=False)

    async def validate(url, **kwargs):
        validations.append((url, kwargs))

    monkeypatch.setattr(hc, "_avalidate_egress_or_raise", validate)
    return validations


@pytest_asyncio.fixture
async def credentials(monkeypatch):
    """Issue genuine handles through the existing authoritative runtime."""
    runtimes = []

    class Repo:
        async def resolve_authorized_secret(self, user_id, provider):
            return ProviderScopeResult(ProviderScopeStatus.AUTHORIZED_ABSENT)

    async def repo():
        return Repo()

    monkeypatch.setattr(byok_runtime, "_get_user_repo", repo)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)

    async def issue(base=BASE, provider="openai"):
        runtime = ProviderCredentialRuntime(
            user_id=7,
            team_ids=[],
            org_ids=[],
            trusted_base_url_override=False,
            authoritative_scope=byok_runtime.AuthoritativeProviderScope(7),
            server_config_snapshot={f"{provider}_api": {"api_key": SECRET, "api_base_url": base}},
        )
        runtimes.append(runtime)
        handle = await runtime.resolve(provider)
        assert is_runtime_issued_provider_call_credentials(handle, provider=provider)
        return handle

    yield issue
    for runtime in runtimes:
        await runtime.close()


@pytest_asyncio.fixture
async def asyncio_diagnostics():
    """Observe loop errors and their unsuppressed default asyncio log output."""
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    contexts = []
    messages = []

    class CaptureHandler(logging.Handler):
        def emit(self, record):
            messages.append(self.format(record))

    def report(context_loop, context):
        contexts.append(context)
        context_loop.default_exception_handler(context)

    handler = CaptureHandler(level=logging.ERROR)
    asyncio_logger = logging.getLogger("asyncio")
    asyncio_logger.addHandler(handler)
    loop.set_exception_handler(report)
    try:
        yield contexts, messages
    finally:
        loop.set_exception_handler(previous_handler)
        asyncio_logger.removeHandler(handler)
        handler.close()


class Body(httpx.AsyncByteStream):
    """Observable native response stream, including failures and cancellation."""

    def __init__(self, chunks=(), *, error=None, entered=None, cleanup_gate=None):
        self.chunks = chunks
        self.error = error
        self.entered = entered
        self.cleanup_gate = cleanup_gate
        self.close_started = asyncio.Event()
        self.closed = False
        self.reads = 0

    async def __aiter__(self):
        for chunk in self.chunks:
            self.reads += 1
            yield chunk
        if self.entered is not None:
            self.entered.set()
            await asyncio.Event().wait()
        if self.error is not None:
            raise self.error

    async def aclose(self):
        self.close_started.set()
        if self.cleanup_gate is not None:
            await self.cleanup_gate.wait()
        self.closed = True


class Harness:
    """Trusted client-factory seam with real shared HTTP and MockTransport I/O."""

    def __init__(self, *, status=200, body=None, headers=None, error=None):
        self.status = status
        self.body = body or Body([json.dumps(_envelope()).encode()])
        self.headers = {"content-type": "application/json", **(headers or {})}
        self.error = error
        self.requests = []
        self.clients = []
        self.options = []
        self.fetches = []

    async def handle(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return httpx.Response(self.status, headers=self.headers, stream=self.body, request=request)

    def client_factory(self, **kwargs):
        self.options.append(kwargs)
        client = hc.create_async_client(transport=httpx.MockTransport(self.handle), **kwargs)
        self.clients.append(client)
        return client

    async def fetch_json(self, **kwargs):
        self.fetches.append(kwargs)
        return await hc.afetch_json(**kwargs)

    def transport(self, policy=None):
        return _api().OpenAICompletionTransport(
            policy or _policy(), fetch_json=self.fetch_json, client_factory=self.client_factory
        )


def test_policy_is_frozen_slotted_and_canonicalizes_trailing_slash():
    policy = _policy()
    assert policy.endpoint == _endpoint(BASE + "/")
    assert policy.output_token_field == "max_completion_tokens"
    assert not hasattr(policy, "__dict__")
    with pytest.raises(FrozenInstanceError):
        policy.model = "changed"


@pytest.mark.parametrize("provider", ["oai", "OpenAI", " openai", "openai ", "anthropic", "custom-openai-api", None, 1])
def test_policy_rejects_noncanonical_provider(provider):
    values = {"provider": provider, "model": "gpt-fixed", "endpoint": _endpoint(), "timeout_seconds": 7}
    with pytest.raises(ModelCompletionFailure) as caught:
        _api().OpenAITransportPolicy(**values)
    _assert_failure(caught.value, "model_provider_unsupported", ModelFailureDomain.SHARED_INFRASTRUCTURE)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("model", None),
        ("model", ""),
        ("model", " "),
        ("model", " gpt-fixed"),
        ("model", "gpt-fixed "),
        ("model", "gpt\n" + PRIVATE),
        ("model", "gpt\x7f"),
        ("model", "gpt\x85"),
        ("model", "gpt\ud800"),
        ("model", "x" * 257),
        ("output_token_field", "auto"),
        ("output_token_field", "MAX_TOKENS"),
        ("output_token_field", None),
        ("timeout_seconds", 0),
        ("timeout_seconds", -1),
        ("timeout_seconds", True),
        ("timeout_seconds", 1.5),
        ("timeout_seconds", "7"),
        ("timeout_seconds", 10**400),
        ("endpoint", None),
        ("endpoint", SimpleNamespace(base_url=BASE, scope=ConfiguredEndpointScope.from_url(BASE))),
    ],
)
def test_policy_rejects_invalid_operator_state(name, value):
    values = {"provider": "openai", "model": "gpt-fixed", "endpoint": _endpoint(), "timeout_seconds": 7}
    values[name] = value
    with pytest.raises(ModelCompletionFailure) as caught:
        _api().OpenAITransportPolicy(**values)
    _assert_failure(caught.value, "model_transport_policy_invalid", ModelFailureDomain.SHARED_INFRASTRUCTURE)


@pytest.mark.parametrize(
    "base",
    [
        "ftp://fixed.example/v1",
        BASE + "?secret=" + SECRET,
        BASE + "?",
        BASE + "#private",
        BASE + "#",
        "https://user:password@fixed.example/v1",
        "https://user@fixed.example/v1",
        " " + BASE,
        BASE + "\n",
        "https://fixed.example/v1\\private",
    ],
)
def test_policy_rejects_unsafe_endpoint_without_leaking_url(base):
    endpoint = TrustedProviderEndpoint(base_url=base, scope=ConfiguredEndpointScope.from_url(BASE))
    with pytest.raises(ModelCompletionFailure) as caught:
        _policy(endpoint=endpoint)
    _assert_failure(caught.value, "model_transport_policy_invalid", ModelFailureDomain.SHARED_INFRASTRUCTURE)


def test_policy_rejects_scope_mismatch():
    endpoint = TrustedProviderEndpoint(base_url=BASE, scope=ConfiguredEndpointScope.from_url("https://other.example"))
    with pytest.raises(ModelCompletionFailure) as caught:
        _policy(endpoint=endpoint)
    _assert_failure(caught.value, "model_transport_policy_invalid", ModelFailureDomain.SHARED_INFRASTRUCTURE)


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_policy_accepts_egress_supported_schemes(scheme):
    assert _policy(endpoint=_endpoint(f"{scheme}://fixed.example/v1")).endpoint.base_url.startswith(scheme + "://")


def test_capabilities_are_static_frozen_five_true_flags_without_io():
    harness = Harness()
    transport = harness.transport()
    caps = transport.capabilities
    assert len(fields(caps)) == 5
    assert all(getattr(caps, field.name) is True for field in fields(caps))
    assert transport.capabilities is caps
    assert transport.policy == _policy()
    assert harness.options == harness.fetches == harness.requests == []
    with pytest.raises(FrozenInstanceError):
        caps.tool_suppression = False
    with pytest.raises(AttributeError):
        transport.policy = _policy()
    with pytest.raises(AttributeError):
        transport.capabilities = caps


@pytest.mark.asyncio
@pytest.mark.parametrize("token_field", ["max_tokens", "max_completion_tokens"])
async def test_exact_bounded_payload_headers_scope_and_normalized_result(credentials, isolate_network, token_field):
    harness = Harness()
    credential = await credentials()
    result = await harness.transport(_policy(output_token_field=token_field)).complete(_request(), credential)
    assert result == NormalizedModelCompletion(" answer\nline ", 17, 3)
    assert not hasattr(result, "usage")
    assert PRIVATE not in repr(result)
    assert len(harness.requests) == len(harness.fetches) == len(harness.clients) == 1
    request = harness.requests[0]
    assert (request.method, str(request.url)) == ("POST", URL)
    assert json.loads(request.content) == {
        "model": "gpt-fixed",
        "messages": [{"role": "system", "content": "system-sentinel"}, {"role": "user", "content": PROMPT}],
        "stream": False,
        "n": 1,
        "tools": None,
        token_field: 32,
    }
    assert request.headers["authorization"] == "Bearer " + SECRET
    assert request.headers["content-type"] == request.headers["accept"] == "application/json"
    assert "cookie" not in request.headers
    assert "openai-organization" not in request.headers
    call = harness.fetches[0]
    assert call["headers"] == {
        "Authorization": "Bearer " + SECRET,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    assert call["client"] is harness.clients[0]
    assert call["max_bytes"] == 1024
    assert call["require_json_ct"] is True
    assert call["allow_redirects"] is False
    assert call["retry"].attempts == 1
    assert call["sensitive_observability"] is True
    assert call["configured_endpoint"] == credential.trusted_endpoint.scope
    assert "backend" not in call and "follow_redirects" not in call
    assert harness.options == [{"trust_env": False, "timeout": 7.0}]
    assert all(kwargs["configured_endpoint"] == credential.trusted_endpoint.scope for _, kwargs in isolate_network)
    assert harness.body.closed and harness.clients[0].is_closed


@pytest.mark.asyncio
async def test_explicit_native_client_overrides_environment_and_default_backend(credentials, monkeypatch):
    monkeypatch.setenv("HTTP_CLIENT_BACKEND", hc.AiohttpAdapter.name)
    monkeypatch.setenv("HTTP_BACKEND", "curl")
    monkeypatch.setenv("HTTP_PROXY", "http://untrusted.example:1234")
    monkeypatch.setenv("HTTPS_PROXY", "http://untrusted.example:1234")
    adapter = hc._get_transport_adapter
    selected = []

    def select(name):
        selected.append(name)
        assert name == "httpx", "ambient aiohttp/default backend must not be selected"
        return adapter(name)

    monkeypatch.setattr(hc, "_get_transport_adapter", select)
    harness = Harness()
    await harness.transport().complete(_request(), await credentials())
    assert selected == ["httpx"]
    assert len(harness.requests) == 1
    assert harness.clients[0]._trust_env is False


@pytest.mark.asyncio
async def test_fresh_clients_do_not_carry_response_cookies_between_calls(credentials):
    harness = Harness(headers={"set-cookie": "ambient=private; Path=/"})
    transport = harness.transport()
    credential = await credentials()
    await transport.complete(_request(), credential)
    harness.body = Body([json.dumps(_envelope()).encode()])
    await transport.complete(_request(), credential)
    assert len(harness.clients) == 2
    assert harness.clients[0] is not harness.clients[1]
    assert all(client.is_closed for client in harness.clients)
    assert all("cookie" not in request.headers for request in harness.requests)


@pytest.mark.asyncio
async def test_default_factory_uses_zero_retry_native_httpx(credentials, monkeypatch):
    native_factory = hc.create_async_client
    clients = []

    def factory(**kwargs):
        client = native_factory(**kwargs)
        clients.append(client)
        return client

    async def fetch(**kwargs):
        await kwargs["on_response"](200, {"content-type": "application/json"})
        assert type(kwargs["client"]) is httpx.AsyncClient
        assert kwargs["client"]._transport._pool._retries == 0
        assert not list(kwargs["client"].cookies.jar)
        return _envelope()

    monkeypatch.setattr(hc, "create_async_client", factory)
    await _api().OpenAICompletionTransport(_policy(), fetch_json=fetch).complete(_request(), await credentials())
    assert len(clients) == 1 and clients[0].is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("credential", [None, object(), SimpleNamespace(provider="openai", api_key=SECRET)])
async def test_missing_or_unissued_credentials_rejected_before_client(credential):
    harness = Harness()
    with pytest.raises(ModelCompletionFailure) as caught:
        await harness.transport().complete(_request(), credential)
    _assert_failure(caught.value, "model_credentials_invalid", ModelFailureDomain.CREDENTIAL_SCOPE)
    assert harness.options == harness.fetches == harness.requests == []


@pytest.mark.asyncio
async def test_constructed_credential_is_not_a_runtime_issued_handle():
    credential = ProviderCallCredentials(
        provider="openai",
        api_key=SECRET,
        app_config={},
        auth_source="api_key",
        runtime_generation=1,
        runtime_identity=object(),
        credential_identity=object(),
        trusted_endpoint=_endpoint(),
    )
    harness = Harness()
    with pytest.raises(ModelCompletionFailure) as caught:
        await harness.transport().complete(_request(), credential)
    _assert_failure(caught.value, "model_credentials_invalid", ModelFailureDomain.CREDENTIAL_SCOPE)
    assert harness.options == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("api_key", None),
        ("api_key", ""),
        ("api_key", " "),
        ("api_key", SECRET + "\r\nX-Private: " + PROMPT),
        ("api_key", SECRET + "\x00"),
        ("api_key", SECRET + "\t"),
        ("api_key", SECRET + "\x7f"),
        ("api_key", SECRET + "\x85"),
        ("api_key", SECRET + "\ud800"),
        ("api_key", 7),
        ("provider", "anthropic"),
        ("provider", "oai"),
        ("endpoint_provenance", "byok"),
        ("endpoint_provenance", None),
        ("trusted_endpoint", None),
        ("trusted_endpoint", _endpoint("https://other.example/v1")),
        ("trusted_endpoint", _endpoint(BASE + "/other")),
        (
            "trusted_endpoint",
            TrustedProviderEndpoint(base_url=BASE, scope=ConfiguredEndpointScope.from_url("http://fixed.example")),
        ),
    ],
)
async def test_corrupted_runtime_issued_state_fails_closed(credentials, field, value):
    credential = await credentials()
    # Corrupt a genuine handle's frozen snapshot; never forge the runtime issuance token.
    object.__setattr__(credential, "_transport_snapshot", replace(credential._transport_snapshot, **{field: value}))
    harness = Harness()
    with pytest.raises(ModelCompletionFailure) as caught:
        await harness.transport().complete(_request(), credential)
    _assert_failure(caught.value, "model_credentials_invalid", ModelFailureDomain.CREDENTIAL_SCOPE)
    assert harness.options == harness.fetches == harness.requests == []


@pytest.mark.asyncio
async def test_genuine_other_endpoint_credential_rejected(credentials):
    harness = Harness()
    with pytest.raises(ModelCompletionFailure) as caught:
        await harness.transport().complete(_request(), await credentials(BASE + "/other"))
    _assert_failure(caught.value, "model_credentials_invalid", ModelFailureDomain.CREDENTIAL_SCOPE)
    assert harness.options == []


@pytest.mark.asyncio
@pytest.mark.parametrize("request_value", [None, {}, SimpleNamespace(system_prompt=PROMPT)])
async def test_only_exact_frozen_request_is_accepted(credentials, request_value):
    harness = Harness()
    with pytest.raises(ModelCompletionFailure) as caught:
        await harness.transport().complete(request_value, await credentials())
    _assert_failure(caught.value, "model_request_invalid", ModelFailureDomain.REQUEST)
    assert harness.options == harness.fetches == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("system_prompt", 1),
        ("user_prompt", None),
        ("system_prompt", "\ud800"),
        ("user_prompt", "\udfff"),
        ("system_prompt", "x" * 100001),
        ("user_prompt", "x" * 100001),
        ("max_output_tokens", 8193),
        ("max_output_chars", 100001),
        ("max_output_bytes", 400001),
        ("max_provider_response_bytes", 2000001),
        ("max_output_tokens", True),
        ("max_output_chars", 0),
        ("max_output_bytes", -1),
        ("max_provider_response_bytes", 1.5),
    ],
    ids=lambda value: f"text-{len(value)}" if isinstance(value, str) and len(value) > 1000 else None,
)
async def test_request_bounds_and_strict_utf8_revalidated_before_dispatch(credentials, field, value):
    request = _request()
    object.__setattr__(request, field, value)
    harness = Harness()
    with pytest.raises(ModelCompletionFailure) as caught:
        await harness.transport().complete(request, await credentials())
    _assert_failure(caught.value, "model_request_invalid", ModelFailureDomain.REQUEST)
    assert harness.options == harness.fetches == harness.requests == []


@pytest.mark.asyncio
async def test_combined_prompt_ceiling_is_enforced(credentials):
    harness = Harness()
    with pytest.raises(ModelCompletionFailure) as caught:
        await harness.transport().complete(
            _request(system_prompt="x" * 60000, user_prompt="y" * 60000), await credentials()
        )
    _assert_failure(caught.value, "model_request_invalid", ModelFailureDomain.REQUEST)
    assert harness.options == []


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [100, 301, 302, 303, 307, 308, 400, 401, 403, 404, 429, 500, 503])
async def test_any_non2xx_response_is_neutral_and_never_retried_or_redirected(credentials, status):
    body = Body([PRIVATE.encode()])
    harness = Harness(
        status=status, body=body, headers={"location": "https://untrusted.example/" + SECRET, "retry-after": "0"}
    )
    with pytest.raises(ModelCompletionFailure) as caught:
        await harness.transport().complete(_request(), await credentials())
    _assert_failure(caught.value, "model_response_rejected", ModelFailureDomain.REQUEST)
    assert len(harness.requests) == len(harness.fetches) == 1
    assert body.reads == 0
    assert body.closed and harness.clients[0].is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["malformed_json", "wrong_content_type", "overflow", "advertised_overflow", "gzip_overflow"]
)
async def test_bounded_decode_failures_are_neutral_single_attempt_and_close_stream(credentials, mode):
    headers = {}
    body = Body([b'{"private":"' + PRIVATE.encode()])
    if mode == "wrong_content_type":
        headers = {"content-type": "text/html"}
    elif mode == "overflow":
        body = Body([b"x" * 2048, b"unread"])
    elif mode == "advertised_overflow":
        headers = {"content-length": "2048"}
    elif mode == "gzip_overflow":
        payload = gzip.compress(b"x" * 8192)
        body = Body([payload, b"unread"])
        headers = {"content-encoding": "gzip", "content-length": str(len(payload))}
    harness = Harness(body=body, headers=headers)
    with pytest.raises(ModelCompletionFailure) as caught:
        await harness.transport().complete(_request(), await credentials())
    _assert_failure(caught.value, "model_response_invalid", ModelFailureDomain.REQUEST)
    assert len(harness.requests) == len(harness.fetches) == 1
    assert body.closed and harness.clients[0].is_closed
    if mode in ("wrong_content_type", "advertised_overflow"):
        assert body.reads == 0
    elif mode in ("overflow", "gzip_overflow"):
        assert body.reads == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ReadTimeout, RuntimeError, ValueError])
@pytest.mark.parametrize("response_received", [False, True])
async def test_failure_provenance_uses_response_phase_not_exception_class(credentials, error_type, response_received):
    error = error_type(PRIVATE)
    body = Body(error=error)
    harness = Harness(body=body, error=None if response_received else error)
    with pytest.raises(ModelCompletionFailure) as caught:
        await harness.transport().complete(_request(), await credentials())
    domain = ModelFailureDomain.REQUEST if response_received else ModelFailureDomain.SHARED_INFRASTRUCTURE
    code = "model_response_invalid" if response_received else "model_transport_unavailable"
    _assert_failure(caught.value, code, domain)
    assert len(harness.requests) == len(harness.fetches) == 1
    assert harness.clients[0].is_closed
    assert body.closed is response_received


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["", "  ", "\ud800", "x" * 101])
async def test_normalization_failure_remains_request_local(credentials, content):
    harness = Harness(body=Body([json.dumps(_envelope(content)).encode()]))
    with pytest.raises(ModelCompletionFailure) as caught:
        await harness.transport().complete(_request(), await credentials())
    _assert_failure(caught.value, "invalid_model_output", ModelFailureDomain.REQUEST)
    assert len(harness.requests) == 1
    assert harness.body.closed and harness.clients[0].is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("before_response", [True, False])
async def test_cancellation_propagates_untouched_after_native_cleanup(credentials, before_response):
    entered = asyncio.Event()
    body = Body(entered=entered)
    harness = Harness(body=body)
    cancellation = asyncio.CancelledError("private-cancellation-sentinel")

    if before_response:

        async def handler(request):
            harness.requests.append(request)
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise cancellation from None

        harness.handle = handler
    transport = harness.transport()
    credential = await credentials()
    task = asyncio.create_task(transport.complete(_request(), credential))
    await asyncio.wait_for(entered.wait(), timeout=2)
    task.cancel("caller-cancel")
    with pytest.raises(asyncio.CancelledError) as caught:
        await task
    if before_response:
        assert caught.value is cancellation
    else:
        assert caught.value.args == ("caller-cancel",)
        assert body.closed
    assert len(harness.requests) == len(harness.fetches) == 1
    assert harness.clients[0].is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RuntimeError, httpx.ReadError])
async def test_client_cleanup_error_preserves_validated_result_without_logs(
    credentials, asyncio_diagnostics, error_type
):
    close_tasks = []
    messages = []

    class BrokenCloseClient(httpx.AsyncClient):
        async def aclose(self):
            close_tasks.append(asyncio.current_task())
            await super().aclose()
            raise error_type(PRIVATE) from ValueError(PRIVATE)

    client = BrokenCloseClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=_envelope())), trust_env=False
    )
    transport = _api().OpenAICompletionTransport(_policy(), client_factory=lambda **_kwargs: client)
    initial_caps = transport.capabilities
    sink = logger.add(lambda message: messages.append(str(message)), filter=lambda record: record["name"] == MODULE)
    try:
        result = await transport.complete(_request(), await credentials())
    finally:
        logger.remove(sink)
    assert result == NormalizedModelCompletion(" answer\nline ", 17, 3)
    assert client.is_closed
    assert transport.capabilities is not initial_caps
    assert transport.capabilities.native_async_cancellation is False
    latched_caps = transport.capabilities
    with pytest.raises(ModelCompletionFailure) as caught:
        await transport.complete(_request(), await credentials())
    _assert_failure(caught.value, "model_transport_uncertified", ModelFailureDomain.SHARED_INFRASTRUCTURE)
    assert transport.capabilities is latched_caps
    assert len(close_tasks) == 1
    assert close_tasks[0].done() and not close_tasks[0]._log_traceback
    assert messages == []
    assert asyncio_diagnostics == ([], [])


@pytest.mark.asyncio
async def test_receipt_notification_precedes_owned_client_cleanup(credentials):
    close_entered, release_close = asyncio.Event(), asyncio.Event()
    receipts = []

    class GatedClient(httpx.AsyncClient):
        async def aclose(self):
            close_entered.set()
            await release_close.wait()
            await super().aclose()

    client = GatedClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=_envelope())))
    transport = _api().OpenAICompletionTransport(_policy(), client_factory=lambda **_kwargs: client)
    call = None
    try:
        call = asyncio.create_task(transport.complete(_request(), await credentials(), on_receipt=receipts.append))
        await asyncio.wait_for(close_entered.wait(), 2)
        assert receipts == [" answer\nline "]
        assert not call.done()
        release_close.set()
        assert await call == NormalizedModelCompletion(receipts[0], 17, 3)
        assert client.is_closed
    finally:
        release_close.set()
        if call is not None:
            await asyncio.gather(call, return_exceptions=True)
        else:
            await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["connection", "invalid"])
async def test_no_receipt_notification_before_validated_output(credentials, outcome):
    receipts = []
    harness = Harness(
        body=Body([json.dumps(_envelope(" \n ")).encode()]),
        error=httpx.ConnectError(PRIVATE) if outcome == "connection" else None,
    )
    with pytest.raises(ModelCompletionFailure) as caught:
        await harness.transport().complete(_request(), await credentials(), on_receipt=receipts.append)
    _assert_failure(
        caught.value,
        "model_transport_unavailable" if outcome == "connection" else "invalid_model_output",
        ModelFailureDomain.SHARED_INFRASTRUCTURE if outcome == "connection" else ModelFailureDomain.REQUEST,
    )
    assert receipts == []
    assert len(harness.requests) == 1
    assert harness.clients[0].is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_kind", ["task_cancel", "task_cancel_return", "raised_cancel"])
async def test_native_client_close_self_cancellation_latches_uncertified(credentials, asyncio_diagnostics, cancel_kind):
    receipts, close_tasks = [], []

    class SelfCancellingClient(httpx.AsyncClient):
        async def aclose(self):
            close_tasks.append(asyncio.current_task())
            if cancel_kind != "raised_cancel":
                asyncio.current_task().cancel(PRIVATE)
                if cancel_kind == "task_cancel":
                    await asyncio.sleep(0)
            else:
                raise asyncio.CancelledError(PRIVATE)

    client = SelfCancellingClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=_envelope())))
    transport = _api().OpenAICompletionTransport(_policy(), client_factory=lambda **_kwargs: client)
    try:
        with pytest.raises(asyncio.CancelledError):
            await transport.complete(_request(), await credentials(), on_receipt=receipts.append)
        assert receipts == [" answer\nline "]
        assert not client.is_closed
        assert transport.capabilities.native_async_cancellation is False
        assert all(task.done() and task.cancelled() for task in close_tasks)
        assert asyncio_diagnostics == ([], [])
    finally:
        await httpx.AsyncClient.aclose(client)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_kind", ["task_cancel", "task_cancel_return", "raised_cancel"])
@pytest.mark.parametrize(
    "outcome,code,domain",
    [
        ("connection", "model_transport_unavailable", ModelFailureDomain.SHARED_INFRASTRUCTURE),
        ("rejected", "model_response_rejected", ModelFailureDomain.REQUEST),
        ("invalid", "invalid_model_output", ModelFailureDomain.REQUEST),
    ],
)
async def test_cleanup_self_cancellation_cannot_replace_known_failure(
    credentials, asyncio_diagnostics, cancel_kind, outcome, code, domain
):
    requests, close_tasks, receipts = [], [], []

    class SelfCancellingClient(httpx.AsyncClient):
        async def aclose(self):
            close_tasks.append(asyncio.current_task())
            if cancel_kind == "raised_cancel":
                raise asyncio.CancelledError(PRIVATE)
            asyncio.current_task().cancel(PRIVATE)
            if cancel_kind == "task_cancel":
                await asyncio.sleep(0)

    def handle(incoming):
        requests.append(incoming)
        if outcome == "connection":
            raise httpx.ConnectError(PRIVATE)
        return httpx.Response(
            429 if outcome == "rejected" else 200,
            json=_envelope(" \r\n "),
        )

    client = SelfCancellingClient(transport=httpx.MockTransport(handle), trust_env=False)
    transport = _api().OpenAICompletionTransport(_policy(), client_factory=lambda **_kwargs: client)
    try:
        with pytest.raises(ModelCompletionFailure) as caught:
            await transport.complete(_request(), await credentials(), on_receipt=receipts.append)
        _assert_failure(caught.value, code, domain)
        assert not asyncio.current_task().cancelling()
        assert receipts == []
        assert len(requests) == len(close_tasks) == 1
        assert close_tasks[0].done() and not close_tasks[0]._log_traceback
        assert transport.capabilities.native_async_cancellation is False
        assert asyncio_diagnostics == ([], [])
    finally:
        await httpx.AsyncClient.aclose(client)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["connection", "rejected", "invalid"])
async def test_caller_cancellation_wins_known_failure_and_cleanup_self_cancellation(credentials, outcome):
    entered, release = asyncio.Event(), asyncio.Event()

    class SelfCancellingClient(httpx.AsyncClient):
        async def aclose(self):
            entered.set()
            await release.wait()
            asyncio.current_task().cancel(PRIVATE)

    def handle(incoming):
        if outcome == "connection":
            raise httpx.ConnectError(PRIVATE)
        return httpx.Response(429 if outcome == "rejected" else 200, json=_envelope(" \r\n "))

    client = SelfCancellingClient(transport=httpx.MockTransport(handle), trust_env=False)
    transport = _api().OpenAICompletionTransport(_policy(), client_factory=lambda **_kwargs: client)
    call = asyncio.create_task(transport.complete(_request(), await credentials()))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        call.cancel("original-caller-cancel")
        release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await call
        assert caught.value.args == ("original-caller-cancel",)
        assert transport.capabilities.native_async_cancellation is False
    finally:
        release.set()
        await asyncio.gather(call, return_exceptions=True)
        await httpx.AsyncClient.aclose(client)


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_owned_client_close(credentials):
    entered = asyncio.Event()
    close_entered = asyncio.Event()
    release_close = asyncio.Event()

    class SlowCloseClient(httpx.AsyncClient):
        async def aclose(self):
            close_entered.set()
            await release_close.wait()
            await super().aclose()

    body = Body(entered=entered)
    client = SlowCloseClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=body, headers={"content-type": "application/json"})
        ),
        trust_env=False,
    )
    transport = _api().OpenAICompletionTransport(_policy(), client_factory=lambda **_kwargs: client)
    task = asyncio.create_task(transport.complete(_request(), await credentials()))
    await asyncio.wait_for(entered.wait(), timeout=2)
    task.cancel("original-cancel")
    await asyncio.wait_for(close_entered.wait(), timeout=2)
    task.cancel("second-cancel")
    await asyncio.sleep(0)
    assert not task.done()
    release_close.set()
    with pytest.raises(asyncio.CancelledError) as caught:
        await task
    assert caught.value.args == ("original-cancel",)
    assert body.closed and client.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_origin", ["stream", "client"])
@pytest.mark.parametrize("error_type", [RuntimeError, httpx.ReadError])
async def test_cancelled_cleanup_failure_never_reaches_asyncio_diagnostics(
    credentials, asyncio_diagnostics, cleanup_origin, error_type
):
    entered = asyncio.Event()
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    owned_operations = []

    class FailingCloseBody(Body):
        async def aclose(self):
            owned_operations.append(asyncio.current_task())
            close_started.set()
            await release_close.wait()
            self.closed = True
            raise error_type(PRIVATE)

    class FailingCloseClient(httpx.AsyncClient):
        async def aclose(self):
            owned_operations.append(asyncio.current_task())
            close_started.set()
            await release_close.wait()
            await super().aclose()
            raise error_type(PRIVATE)

    body = FailingCloseBody(entered=entered) if cleanup_origin == "stream" else Body([json.dumps(_envelope()).encode()])
    harness = Harness(body=body)
    if cleanup_origin == "client":

        def client_factory(**kwargs):
            harness.options.append(kwargs)
            client = FailingCloseClient(transport=httpx.MockTransport(harness.handle), **kwargs)
            harness.clients.append(client)
            return client

        harness.client_factory = client_factory

    transport = harness.transport()
    task = asyncio.create_task(transport.complete(_request(), await credentials()))
    if cleanup_origin == "stream":
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel("original-cancel")
        await asyncio.wait_for(close_started.wait(), timeout=2)
        task.cancel("repeated-cancel")
    else:
        await asyncio.wait_for(close_started.wait(), timeout=2)
        task.cancel("original-cancel")
        await asyncio.sleep(0)
        task.cancel("repeated-cancel")
    await asyncio.sleep(0)
    assert not task.done()
    release_close.set()
    with pytest.raises(asyncio.CancelledError) as caught:
        await task
    await asyncio.sleep(0)
    assert caught.value.args == ("original-cancel",)
    assert body.closed and harness.clients[0].is_closed
    assert len(harness.requests) == len(harness.fetches) == 1
    assert all(operation.done() and not operation._log_traceback for operation in owned_operations)
    if cleanup_origin == "client":
        assert transport.capabilities.native_async_cancellation is False
    contexts, messages = asyncio_diagnostics
    assert (contexts, messages) == ([], [])


@pytest.mark.parametrize(
    "scope",
    [
        ConfiguredEndpointScope("https", "fixed.example", 443.0),
        ConfiguredEndpointScope("https", "fixed.example", "443"),
        SimpleNamespace(scheme="https", host="fixed.example", port=443),
    ],
)
def test_policy_requires_exact_typed_scope_from_url(scope):
    endpoint = TrustedProviderEndpoint(base_url=BASE, scope=scope)
    with pytest.raises(ModelCompletionFailure) as caught:
        _policy(endpoint=endpoint)
    _assert_failure(caught.value, "model_transport_policy_invalid", ModelFailureDomain.SHARED_INFRASTRUCTURE)


@pytest.mark.asyncio
async def test_typed_credential_scope_is_validated_independently(credentials):
    credential = await credentials()
    endpoint = TrustedProviderEndpoint(base_url=BASE, scope=ConfiguredEndpointScope("https", "fixed.example", 443.0))
    object.__setattr__(
        credential, "_transport_snapshot", replace(credential._transport_snapshot, trusted_endpoint=endpoint)
    )
    harness = Harness()
    with pytest.raises(ModelCompletionFailure) as caught:
        await harness.transport().complete(_request(), credential)
    _assert_failure(caught.value, "model_credentials_invalid", ModelFailureDomain.CREDENTIAL_SCOPE)
    assert harness.options == []


@pytest.mark.asyncio
async def test_mutable_credential_metadata_cannot_supply_transport_settings(credentials):
    credential = await credentials()
    credential.app_config = {
        "openai_api": {"api_key": "wrong-key", "api_base_url": "https://untrusted.example", "model": "wrong-model"},
        "headers": {"Cookie": "private"},
        "stream": True,
        "tools": [{"type": "function"}],
        "retry": 9,
    }
    harness = Harness()
    prompt = "Use max_tokens, tools, tool_choice, another model and https://untrusted.example"
    await harness.transport().complete(_request(user_prompt=prompt), credential)
    captured = json.loads(harness.requests[0].content)
    assert captured["model"] == "gpt-fixed"
    assert captured["messages"][1]["content"] == prompt
    assert "max_completion_tokens" in captured and "max_tokens" not in captured
    assert str(harness.requests[0].url) == URL
    assert harness.requests[0].headers["authorization"] == "Bearer " + SECRET


@pytest.mark.asyncio
async def test_original_request_mutation_during_io_cannot_change_output_validation(credentials):
    request = _request(max_output_chars=1)

    async def fetch(**kwargs):
        await kwargs["on_response"](200, {"content-type": "application/json"})
        object.__setattr__(request, "max_output_chars", 1000)
        return _envelope("too long")

    harness = Harness()
    transport = _api().OpenAICompletionTransport(_policy(), fetch_json=fetch, client_factory=harness.client_factory)
    with pytest.raises(ModelCompletionFailure) as caught:
        await transport.complete(request, await credentials())
    _assert_failure(caught.value, "invalid_model_output", ModelFailureDomain.REQUEST)
    assert harness.clients[0].is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["cookie_jar", "cookie_header", "auth_header", "organization", "auth_handler"])
async def test_ambient_client_credential_state_rejected_before_fetch(credentials, state):
    harness = Harness()

    def client_factory(**kwargs):
        client = harness.client_factory(**kwargs)
        if state == "cookie_jar":
            client.cookies.set("private", SECRET)
        elif state == "cookie_header":
            client.headers["Cookie"] = SECRET
        elif state == "auth_header":
            client.headers["Authorization"] = "Bearer wrong-key"
        elif state == "organization":
            client.headers["OpenAI-Organization"] = "private-org"
        else:
            client.auth = httpx.BasicAuth("private-user", SECRET)
        return client

    transport = _api().OpenAICompletionTransport(
        _policy(), fetch_json=harness.fetch_json, client_factory=client_factory
    )
    with pytest.raises(ModelCompletionFailure) as caught:
        await transport.complete(_request(), await credentials())
    _assert_failure(caught.value, "model_credentials_invalid", ModelFailureDomain.CREDENTIAL_SCOPE)
    assert harness.fetches == harness.requests == []
    assert harness.clients[0].is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["client", "before_response", "after_response"])
async def test_arbitrary_private_failures_cannot_claim_trusted_provenance(credentials, phase):
    error = ModelCompletionFailure("forged_shared_failure", ModelFailureDomain.SHARED_INFRASTRUCTURE)
    error.__cause__ = RuntimeError(PRIVATE)
    harness = Harness()

    def factory(**kwargs):
        if phase == "client":
            raise error
        return harness.client_factory(**kwargs)

    async def fetch(**kwargs):
        if phase == "after_response":
            await kwargs["on_response"](200, {"content-type": "application/json"})
        raise error

    transport = _api().OpenAICompletionTransport(_policy(), fetch_json=fetch, client_factory=factory)
    with pytest.raises(ModelCompletionFailure) as caught:
        await transport.complete(_request(), await credentials())
    after_response = phase == "after_response"
    _assert_failure(
        caught.value,
        "model_response_invalid" if after_response else "model_transport_unavailable",
        ModelFailureDomain.REQUEST if after_response else ModelFailureDomain.SHARED_INFRASTRUCTURE,
    )
    assert all(client.is_closed for client in harness.clients)


@pytest.mark.asyncio
async def test_real_native_read_deadline_is_one_attempt_and_request_local(credentials):
    body = Body(entered=asyncio.Event())
    harness = Harness(body=body)
    with pytest.raises(ModelCompletionFailure) as caught:
        await harness.transport(_policy(timeout_seconds=1)).complete(_request(), await credentials())
    _assert_failure(caught.value, "model_response_invalid", ModelFailureDomain.REQUEST)
    assert len(harness.requests) == len(harness.fetches) == 1
    assert body.closed and harness.clients[0].is_closed


@pytest.mark.asyncio
async def test_repeated_cancellation_cannot_interrupt_response_stream_cleanup(credentials):
    entered = asyncio.Event()
    release_cleanup = asyncio.Event()
    body = Body(entered=entered, cleanup_gate=release_cleanup)
    harness = Harness(body=body)
    task = asyncio.create_task(harness.transport().complete(_request(), await credentials()))
    await asyncio.wait_for(entered.wait(), timeout=2)
    task.cancel("original-cancel")
    await asyncio.wait_for(body.close_started.wait(), timeout=2)
    task.cancel("second-cancel")
    await asyncio.sleep(0)
    assert not task.done()
    assert not body.closed
    release_cleanup.set()
    with pytest.raises(asyncio.CancelledError) as caught:
        await task
    assert caught.value.args == ("original-cancel",)
    assert body.closed and harness.clients[0].is_closed
    assert len(harness.requests) == len(harness.fetches) == 1


@pytest.mark.asyncio
async def test_sensitive_native_failure_logs_no_secret_prompt_or_endpoint(credentials):
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)), level="DEBUG")
    harness = Harness(error=httpx.ConnectError(PRIVATE))
    try:
        with pytest.raises(ModelCompletionFailure):
            await harness.transport().complete(_request(), await credentials())
    finally:
        logger.remove(sink)
    assert all(
        value not in "".join(messages) for value in (SECRET, PROMPT, URL, "fixed.example", "private-cause-sentinel")
    )


@pytest.fixture
def tls_probes(monkeypatch):
    probes = []

    def forbidden_probe(*args, **kwargs):
        probes.append((args, kwargs))
        raise AssertionError("Blocking certificate probe must never run in this transport")

    monkeypatch.setattr(hc, "_check_cert_pinning", forbidden_probe)
    return probes


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["fixed.example", "FIXED.EXAMPLE", "fixed.example."])
async def test_initial_selected_host_pins_fail_certification_without_fetch_or_tls_probe(
    credentials, monkeypatch, tls_probes, host
):
    pins = host + "=" + "a" * 64
    monkeypatch.setenv("HTTP_CERT_PINS", pins)
    harness = Harness()
    transport = harness.transport()
    caps = transport.capabilities
    assert caps.native_async_cancellation is False
    assert all(getattr(caps, field.name) is True for field in fields(caps) if field.name != "native_async_cancellation")
    assert transport.capabilities is caps
    with pytest.raises(FrozenInstanceError):
        caps.native_async_cancellation = True
    with pytest.raises(ModelCompletionFailure) as caught:
        await transport.complete(_request(), await credentials())
    _assert_failure(caught.value, "model_transport_uncertified", ModelFailureDomain.SHARED_INFRASTRUCTURE)
    assert harness.options == harness.fetches == harness.requests == tls_probes == []
    assert hc._parse_pins_from_env() == {"fixed.example": {"a" * 64}}
    assert os.environ["HTTP_CERT_PINS"] == pins


@pytest.mark.asyncio
async def test_late_environment_pins_latch_uncertified_and_close_fresh_client(credentials, monkeypatch, tls_probes):
    harness = Harness()
    transport = harness.transport()
    initial_caps = transport.capabilities
    assert initial_caps.native_async_cancellation is True
    monkeypatch.setenv("HTTP_CERT_PINS", "fixed.example=" + "b" * 64)
    with pytest.raises(ModelCompletionFailure) as caught:
        await transport.complete(_request(), await credentials())
    _assert_failure(caught.value, "model_transport_uncertified", ModelFailureDomain.SHARED_INFRASTRUCTURE)
    assert harness.clients[0].is_closed
    assert harness.clients[0]._tldw_cert_pinning == {"fixed.example": {"b" * 64}}
    assert harness.fetches == harness.requests == tls_probes == []
    latched_caps = transport.capabilities
    assert latched_caps is not initial_caps
    assert latched_caps.native_async_cancellation is False
    monkeypatch.delenv("HTTP_CERT_PINS")
    with pytest.raises(ModelCompletionFailure) as caught:
        await transport.complete(_request(), await credentials())
    _assert_failure(caught.value, "model_transport_uncertified", ModelFailureDomain.SHARED_INFRASTRUCTURE)
    assert transport.capabilities is latched_caps
    assert len(harness.clients) == 1


@pytest.mark.asyncio
async def test_actual_client_selected_host_pins_latch_uncertified(credentials, tls_probes):
    harness = Harness()

    def factory(**kwargs):
        client = harness.client_factory(**kwargs)
        client._tldw_cert_pinning = {"FIXED.EXAMPLE.": {"c" * 64}}
        return client

    transport = _api().OpenAICompletionTransport(_policy(), fetch_json=harness.fetch_json, client_factory=factory)
    with pytest.raises(ModelCompletionFailure) as caught:
        await transport.complete(_request(), await credentials())
    _assert_failure(caught.value, "model_transport_uncertified", ModelFailureDomain.SHARED_INFRASTRUCTURE)
    assert transport.capabilities.native_async_cancellation is False
    assert harness.fetches == harness.requests == tls_probes == []
    assert harness.clients[0].is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["other.example", "fixed.example.evil", "sub.fixed.example"])
async def test_other_host_pins_preserved_without_disabling_selected_path(credentials, monkeypatch, tls_probes, host):
    monkeypatch.setenv("HTTP_CERT_PINS", host + "=" + "d" * 64)
    harness = Harness()
    transport = harness.transport()
    caps = transport.capabilities
    await transport.complete(_request(), await credentials())
    assert transport.capabilities is caps
    assert all(getattr(caps, field.name) is True for field in fields(caps))
    assert harness.clients[0]._tldw_cert_pinning == {host: {"d" * 64}}
    assert len(harness.requests) == len(harness.fetches) == 1
    assert tls_probes == []


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "error", "cancelled"])
@pytest.mark.parametrize("cancel_on_cancellation", [True, False])
async def test_completed_owned_future_never_replaces_concurrent_caller_cancellation(outcome, cancel_on_cancellation):
    future = asyncio.get_running_loop().create_future()
    task = asyncio.create_task(_api()._await_owned_operation(future, cancel_on_cancellation=cancel_on_cancellation))
    await asyncio.sleep(0)
    if outcome == "success":
        future.set_result("late-success")
    elif outcome == "error":
        future.set_exception(RuntimeError(PRIVATE))
    else:
        future.cancel("inner-cancel")
    task.cancel("caller-cancel")
    with pytest.raises(asyncio.CancelledError) as caught:
        await task
    assert caught.value.args == ("caller-cancel",)
    assert future.done()
    assert not future._log_traceback, "owned failures must be consumed, never abandoned"


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "error"])
async def test_fetch_completing_at_caller_cancellation_emits_no_success_or_failure(credentials, outcome):
    harness = Harness()
    caller = None

    async def fetch(**kwargs):
        await kwargs["on_response"](200, {"content-type": "application/json"})
        caller.cancel("caller-cancel")
        if outcome == "error":
            raise RuntimeError(PRIVATE)
        return _envelope()

    transport = _api().OpenAICompletionTransport(_policy(), fetch_json=fetch, client_factory=harness.client_factory)
    caller = asyncio.create_task(transport.complete(_request(), await credentials()))
    with pytest.raises(asyncio.CancelledError) as caught:
        await caller
    assert caught.value.args == ("caller-cancel",)
    assert harness.clients[0].is_closed


def test_public_snapshot_helper_returns_identical_strict_independent_frozen_request():
    api = _api()
    assert hasattr(api, "snapshot_model_completion_request"), "lifecycle snapshot API is not exposed"
    request = _request()
    snapshot = api.snapshot_model_completion_request(request)
    assert snapshot == request
    assert snapshot is not request
    assert not hasattr(snapshot, "__dict__")
    object.__setattr__(request, "max_output_tokens", 10000)
    assert snapshot.max_output_tokens == 32
    with pytest.raises(FrozenInstanceError):
        snapshot.max_output_tokens = 10000
    with pytest.raises(ModelCompletionFailure) as caught:
        api.snapshot_model_completion_request(request)
    _assert_failure(caught.value, "model_request_invalid", ModelFailureDomain.REQUEST)


@pytest.mark.asyncio
@pytest.mark.parametrize("denylist_name", ["WORKFLOWS_EGRESS_DENYLIST", "EGRESS_DENYLIST"])
async def test_real_egress_denylist_is_request_local_and_configured_scope_cannot_bypass(
    credentials, monkeypatch, denylist_name
):
    credential = await credentials()
    # Issue the genuine handle first, then undo the autouse network mock and use
    # a separate context for real policy configuration and normal allowed ports.
    monkeypatch.undo()
    with monkeypatch.context() as policy_env:
        policy_env.delenv("HTTP_CERT_PINS", raising=False)
        policy_env.delenv("WORKFLOWS_EGRESS_DENYLIST", raising=False)
        policy_env.delenv("EGRESS_DENYLIST", raising=False)
        policy_env.setenv(denylist_name, "fixed.example")
        policy_env.setenv("WORKFLOWS_EGRESS_ALLOWLIST", "fixed.example")
        policy_env.setenv("EGRESS_ALLOWLIST", "fixed.example")
        policy_env.setenv("WORKFLOWS_EGRESS_ALLOWED_PORTS", "80,443,8080")
        assert hc._avalidate_egress_or_raise.__module__ == hc.__name__
        assert hc._avalidate_egress_or_raise.__name__ == "_avalidate_egress_or_raise"
        harness = Harness()
        with pytest.raises(ModelCompletionFailure) as caught:
            await harness.transport().complete(_request(), credential)
        _assert_failure(caught.value, "model_egress_denied", ModelFailureDomain.REQUEST)
        assert harness.requests == []
        assert len(harness.fetches) == len(harness.clients) == 1
        assert harness.fetches[0]["configured_endpoint"] == credential.trusted_endpoint.scope
        assert credential.trusted_endpoint.scope.matches(URL)
        assert harness.clients[0].is_closed
        assert harness.body.reads == 0
        assert os.environ["WORKFLOWS_EGRESS_ALLOWED_PORTS"] == "80,443,8080"
