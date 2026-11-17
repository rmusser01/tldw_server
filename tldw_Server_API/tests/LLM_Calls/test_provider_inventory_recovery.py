"""Inventory recovery through the real HTTP layer, without provider requests."""

from __future__ import annotations

import asyncio
import inspect
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx
import pytest

from tldw_Server_API.app.core import http_client
from tldw_Server_API.app.core.exceptions import ChatAuthenticationError, ChatConfigurationError
from tldw_Server_API.app.core.LLM_Calls import gateway
from tldw_Server_API.app.core.LLM_Calls import provider_model_inventory as inventory
from tldw_Server_API.app.core.LLM_Calls.adapter_registry import AuditedCallPolicyTransport
from tldw_Server_API.app.core.LLM_Calls.capability_registry import ProviderCallPolicy
from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope

pytestmark = [pytest.mark.unit, pytest.mark.provider_inventory]
BASE_URL = "http://93.184.216.34/v1"


@pytest.fixture
def transport(monkeypatch):
    """Replace network I/O while retaining discovery, egress, retries and deadlines."""
    inventory._CACHE.clear()
    inventory._FAILED.clear()
    monkeypatch.setenv("WORKFLOWS_EGRESS_ALLOWLIST", "93.184.216.34")
    create_client = http_client.create_async_client

    def install(handler):
        state = SimpleNamespace(requests=[], clients=[], fetches=[])

        async def dispatch(request):
            assert str(request.url) == BASE_URL + "/models"
            state.requests.append(request)
            result = handler(request)
            return await result if inspect.isawaitable(result) else result

        def owned_client(**kwargs):
            client = create_client(transport=httpx.MockTransport(dispatch), **kwargs)
            state.clients.append(client)
            return client

        def fetch(**kwargs):
            state.fetches.append(kwargs)
            return http_client.fetch(**kwargs)

        monkeypatch.setattr(http_client, "create_async_client", owned_client)
        monkeypatch.setattr(inventory, "_http_fetch", fetch)
        return state

    yield install
    inventory._CACHE.clear()
    inventory._FAILED.clear()


def ready_response():
    """An unread stream, matching an actual HTTP response's bounded-read path."""
    return httpx.Response(200, stream=httpx.ByteStream(b'{"data":[{"id":"current-model"}]}'))


def discover(**kwargs):
    """Use one synthetic credential and public literal endpoint, without DNS."""
    return inventory.discover_provider_models("openai", "synthetic-key", base_url=BASE_URL, **kwargs)


def test_model_list_timeout_recovers_before_publishing_inventory_failure(transport):
    def handler(request):
        if len(state.requests) == 1:
            raise httpx.ReadTimeout("synthetic timeout", request=request)
        return ready_response()

    state = transport(handler)

    assert discover().models == ("current-model",)
    assert len(state.requests) == 2
    assert all(request.extensions["timeout"]["read"] <= 5 for request in state.requests)
    assert state.clients and all(client.is_closed for client in state.clients)


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_transient_inventory_status_recovers_through_shared_http_retry(transport, status):
    state = transport(lambda _request: httpx.Response(status) if len(state.requests) == 1 else ready_response())

    assert discover().models == ("current-model",)
    assert len(state.requests) == 2


@pytest.mark.parametrize(
    "status, expected", [(401, "auth_failed"), (403, "auth_failed"), (404, "unsupported"), (408, "unsupported")]
)
def test_nontransient_inventory_response_is_not_retried(transport, status, expected):
    state = transport(lambda _request: httpx.Response(status))

    assert discover().status == expected
    assert len(state.requests) == 1


def test_malformed_inventory_is_not_retried(transport):
    state = transport(lambda _request: httpx.Response(200, stream=httpx.ByteStream(b"invalid json")))

    assert discover().status == "unsupported"
    assert len(state.requests) == 1


def test_egress_refusal_never_reaches_transport(transport, monkeypatch):
    monkeypatch.setenv("WORKFLOWS_EGRESS_DENYLIST", "93.184.216.34")
    state = transport(lambda _request: ready_response())

    assert discover().status != "ready"
    assert state.requests == []


def test_dns_failure_is_not_retried(transport):
    def handler(request):
        raise httpx.ConnectError("synthetic DNS failure", request=request) from socket.gaierror()

    state = transport(handler)

    assert discover().status == "unreachable"
    assert len(state.requests) == 1


def test_oversized_inventory_is_not_retried(transport, monkeypatch):
    monkeypatch.setattr(inventory, "_MAX_RESPONSE_BYTES", 4)
    state = transport(lambda _request: ready_response())

    assert discover().status == "unreachable"
    assert len(state.requests) == 1


def test_one_attempt_policy_preserves_original_failure_bound(transport):
    def handler(request):
        raise httpx.ReadTimeout("synthetic timeout", request=request)

    state = transport(handler)
    before = time.monotonic()

    assert discover(call_policy=ProviderCallPolicy(max_transport_attempts=1)).status == "unreachable"
    assert len(state.requests) == 1
    assert state.fetches[0]["deadline"] - before <= 5.05


def test_short_deadline_expires_during_retry_after_without_another_request(transport):
    state = transport(lambda _request: httpx.Response(429, headers={"Retry-After": "4"}))
    before = time.monotonic()

    assert discover(call_policy=ProviderCallPolicy(maximum_timeout_seconds=0.1)).status == "unreachable"
    assert time.monotonic() - before < 2
    assert len(state.requests) == 1
    assert state.requests[0].extensions["timeout"]["read"] <= 0.1


def test_slow_transport_cannot_outlive_the_aggregate_discovery_budget(transport, monkeypatch):
    monkeypatch.setattr(inventory, "_TIMEOUT_SECONDS", 0.05)

    async def handler(_request):
        await asyncio.sleep(4)
        return ready_response()

    state = transport(handler)
    before = time.monotonic()

    assert discover().status == "unreachable"
    assert time.monotonic() - before < 2
    assert len(state.requests) == 1
    assert state.clients and all(client.is_closed for client in state.clients)


def test_exhausted_recovery_is_negative_cached_without_stale_models(transport):
    def handler(request):
        if len(state.requests) == 1:
            return ready_response()
        raise httpx.ReadTimeout("synthetic timeout", request=request)

    state = transport(handler)
    assert discover().models == ("current-model",)
    assert discover(force_refresh=True).status == "unreachable"
    assert discover().models == ()
    assert len(state.requests) == 3  # One earlier success, then two exhausted attempts.


def test_concurrent_misses_share_recovery_without_per_caller_retries(transport, monkeypatch):
    joined = threading.Event()
    lock = threading.Lock()
    entries = 0

    class ObservedLock:
        def __enter__(self):
            nonlocal entries
            lock.acquire()
            entries += 1
            if entries == 4:
                joined.set()

        def __exit__(self, *_args):
            lock.release()

    monkeypatch.setattr(inventory, "_CACHE_LOCK", ObservedLock())

    def handler(request):
        if len(state.requests) == 1:
            assert joined.wait(3)
            raise httpx.ReadTimeout("synthetic timeout", request=request)
        return ready_response()

    state = transport(handler)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = [future.result(timeout=4) for future in [pool.submit(discover) for _ in range(4)]]

    assert [result.models for result in results] == [("current-model",)] * 4
    assert len(state.requests) == 2


@pytest.fixture
def generation(monkeypatch):
    """Record generation dispatch while keeping real gateway validation."""
    calls = []

    class Adapter:
        async_chat_is_native = True
        strict_async_chat_is_native = True

        def chat(self, request, **_kwargs):
            calls.append(request)
            return {"choices": [{"message": {"content": "accepted"}}]}

        async def achat(self, request, **kwargs):
            return self.chat(request, **kwargs)

    registry = SimpleNamespace(
        get_adapter=lambda _provider: Adapter(),
        get_audited_call_policy_transport=lambda _provider: AuditedCallPolicyTransport(
            maximum_transport_attempts=2,
            enforces_configured_endpoint_scope=True,
            enforces_maximum_timeout=True,
        ),
    )
    monkeypatch.setattr(gateway, "_get_llm_registry", lambda: registry)
    return SimpleNamespace(
        calls=calls,
        kwargs={
            "api_endpoint": "openai",
            "api_key": "synthetic-key",
            "model": "current-model",
            "app_config": {"openai_api": {"api_base_url": BASE_URL}},
            "messages_payload": [{"role": "user", "content": "synthetic input"}],
            "streaming": False,
        },
    )


@pytest.mark.parametrize("async_call", [False, True])
def test_recovered_inventory_dispatches_generation_exactly_once(transport, generation, async_call):
    def handler(request):
        if len(state.requests) == 1:
            raise httpx.ReadTimeout("synthetic timeout", request=request)
        return ready_response()

    state = transport(handler)
    policy = ProviderCallPolicy(
        max_transport_attempts=2,
        maximum_timeout_seconds=2,
        required_endpoint_scope=ConfiguredEndpointScope.from_url(BASE_URL),
    )
    kwargs = {**generation.kwargs, "call_policy": policy}
    if async_call:
        result = asyncio.run(gateway.perform_chat_api_call_async(**kwargs))
    else:
        result = gateway.perform_chat_api_call(**kwargs)

    assert result["choices"][0]["message"]["content"] == "accepted"
    assert len(generation.calls) == 1
    assert len(state.requests) == 2
    assert 0 < generation.calls[0]["call_policy"].maximum_timeout_seconds < 2


@pytest.mark.parametrize(
    "status, error, attempts", [(503, ChatConfigurationError, 2), (401, ChatAuthenticationError, 1)]
)
def test_failed_inventory_never_dispatches_generation(transport, generation, status, error, attempts):
    state = transport(lambda _request: httpx.Response(status))

    with pytest.raises(error):
        asyncio.run(gateway.perform_chat_api_call_async(**generation.kwargs))

    assert generation.calls == []
    assert len(state.requests) == attempts


@pytest.mark.asyncio
async def test_cancellation_drains_recovery_without_dispatching_generation(transport, generation):
    retry_started = threading.Event()
    release = threading.Event()

    async def handler(request):
        if len(state.requests) == 1:
            raise httpx.ReadTimeout("synthetic timeout", request=request)
        retry_started.set()
        while not release.is_set():
            await asyncio.sleep(0.001)
        return ready_response()

    state = transport(handler)
    task = asyncio.create_task(gateway.perform_chat_api_call_async(**generation.kwargs))
    try:
        assert await asyncio.to_thread(retry_started.wait, 3)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert generation.calls == []
    assert state.clients and all(client.is_closed for client in state.clients)
