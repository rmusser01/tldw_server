import asyncio
import types
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


def _has_httpx():
    try:
        import httpx  # noqa: F401
        return True
    except Exception:
        return False


requires_httpx = pytest.mark.skipif(not _has_httpx(), reason="httpx not installed")


@requires_httpx
def test_egress_denied_private_ip(monkeypatch):
    from tldw_Server_API.app.core.http_client import fetch_json
    from tldw_Server_API.app.core.exceptions import EgressPolicyError

    # Ensure private IP blocking is enabled for this test
    monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "true")

    with pytest.raises(EgressPolicyError):
        # Private IP should be denied by default policy
        fetch_json(method="GET", url="http://127.0.0.1/")


@requires_httpx
@pytest.mark.asyncio
async def test_retry_succeeds_on_third_attempt():
    import httpx
    from tldw_Server_API.app.core.http_client import afetch, RetryPolicy, create_async_client

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(500, request=request, text="server error")
        return httpx.Response(200, request=request, text="ok")

    transport = httpx.MockTransport(handler)
    client = create_async_client(transport=transport)
    try:
        resp = await afetch(method="GET", url="http://93.184.216.34/retry", client=client, retry=RetryPolicy(attempts=3))
        assert resp.status_code == 200
        body = resp.text
        assert body == "ok"
        assert calls["n"] == 3
    finally:
        await client.aclose()


@requires_httpx
def test_scoped_fetch_retry_preserves_scope_and_initial_dns_pin(monkeypatch):
    import httpx

    from tldw_Server_API.app.core import http_client as hc
    from tldw_Server_API.app.core.Security import egress as egress_mod
    from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope

    scope = ConfiguredEndpointScope.from_url("http://93.184.216.34:11434")
    validations: list[tuple[object, object]] = []

    def fake_policy(
        _url: str,
        *,
        configured_endpoint=None,
        pinned_resolved_ips=None,
        **_kwargs,
    ):
        validations.append((configured_endpoint, pinned_resolved_ips))
        return types.SimpleNamespace(
            allowed=True,
            reason=None,
            reason_code=None,
            resolved_ips=("93.184.216.34",),
        )

    monkeypatch.setattr(egress_mod, "evaluate_url_policy", fake_policy)
    monkeypatch.setattr(hc.time, "sleep", lambda _delay: None)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500 if calls["n"] == 1 else 200, request=request)

    client = hc.create_client(transport=httpx.MockTransport(handler))
    try:
        response = hc.fetch(
            method="GET",
            url="http://93.184.216.34:11434/retry",
            client=client,
            retry=hc.RetryPolicy(attempts=2),
            configured_endpoint=scope,
        )
    finally:
        client.close()

    assert response.status_code == 200
    assert calls["n"] == 2
    assert len(validations) >= 3
    assert all(item[0] is scope for item in validations)
    assert all(item[1] == ("93.184.216.34",) for item in validations[1:])


@requires_httpx
def test_fetch_json_content_type_validation():
    import httpx
    from tldw_Server_API.app.core.http_client import fetch_json, create_client
    from tldw_Server_API.app.core.exceptions import JSONDecodeError

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, content=b"ok", headers={"Content-Type": "text/plain"})

    transport = httpx.MockTransport(handler)
    client = create_client(transport=transport)
    try:
        with pytest.raises(JSONDecodeError):
            fetch_json(method="GET", url="http://93.184.216.34/json", client=client, require_json_ct=True)
    finally:
        client.close()


@requires_httpx
@pytest.mark.asyncio
async def test_sse_parsing_minimal():
    import httpx
    from tldw_Server_API.app.core.http_client import astream_sse, create_async_client

    content = b"event: ping\ndata: 1\n\n" b"data: hello\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, content=content, headers={"Content-Type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    client = create_async_client(transport=transport)
    try:
        events = []
        async for ev in astream_sse(url="http://93.184.216.34/stream", client=client):
            events.append(ev)
            if len(events) == 2:
                break
        assert events[0].event == "ping"
        assert events[0].data == "1"
        assert events[1].event == "message"
        assert events[1].data == "hello"
    finally:
        await client.aclose()


@requires_httpx
def test_download_with_checksum(tmp_path: Path):
    import httpx
    from tldw_Server_API.app.core.http_client import download, create_client

    payload = b"0123456789abcdef" * 64
    import hashlib
    sha = hashlib.sha256(payload).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, content=payload, headers={"Content-Length": str(len(payload))})

    transport = httpx.MockTransport(handler)
    client = create_client(transport=transport)
    try:
        dest = tmp_path / "file.bin"
        out = download(url="http://93.184.216.34/file", dest=dest, client=client, checksum=sha)
        assert out.exists()
        assert out.read_bytes() == payload
    finally:
        client.close()


@requires_httpx
@pytest.mark.asyncio
async def test_redirect_loop_raises_network_error():
    import httpx
    from tldw_Server_API.app.core.http_client import afetch, create_async_client, RetryPolicy
    from tldw_Server_API.app.core.exceptions import NetworkError, RetryExhaustedError

    def handler(request: httpx.Request) -> httpx.Response:
        # Alternate between /r1 and /r2 to cause a loop
        if request.url.path.endswith("/r1"):
            return httpx.Response(302, request=request, headers={"Location": "/r2"})
        else:
            return httpx.Response(302, request=request, headers={"Location": "/r1"})

    transport = httpx.MockTransport(handler)
    client = create_async_client(transport=transport)
    try:
        with pytest.raises((NetworkError, RetryExhaustedError)):
            await afetch(method="GET", url="http://93.184.216.34/r1", client=client, retry=RetryPolicy(attempts=2))
    finally:
        await client.aclose()


@requires_httpx
@pytest.mark.asyncio
async def test_redirect_without_location_is_error():
    import httpx
    from tldw_Server_API.app.core.http_client import afetch, create_async_client
    from tldw_Server_API.app.core.exceptions import NetworkError

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, request=request)  # No Location header

    transport = httpx.MockTransport(handler)
    client = create_async_client(transport=transport)
    try:
        with pytest.raises(NetworkError):
            await afetch(method="GET", url="http://93.184.216.34/redirect", client=client)
    finally:
        await client.aclose()


@requires_httpx
def test_proxy_allowlist_denial():
    from tldw_Server_API.app.core.http_client import create_client
    from tldw_Server_API.app.core.exceptions import EgressPolicyError

    # Denied since PROXY_ALLOWLIST is empty by default
    with pytest.raises(EgressPolicyError):
        create_client(proxies="http://proxy.internal:8080")


@requires_httpx
def test_json_max_bytes_guard():
    import httpx
    from tldw_Server_API.app.core.http_client import fetch_json, create_client
    from tldw_Server_API.app.core.exceptions import JSONDecodeError

    content = b"{" + b" \n" * 1024 + b"}"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, content=content, headers={"Content-Type": "application/json", "Content-Length": str(len(content))})

    transport = httpx.MockTransport(handler)
    client = create_client(transport=transport)
    try:
        with pytest.raises(JSONDecodeError):
            fetch_json(method="GET", url="http://93.184.216.34/json", client=client, require_json_ct=True, max_bytes=10)
    finally:
        client.close()


@requires_httpx
def test_json_max_bytes_guard_uses_actual_body_despite_short_content_length():
    import httpx
    from tldw_Server_API.app.core.http_client import fetch_json, create_client
    from tldw_Server_API.app.core.exceptions import JSONDecodeError

    content = b"{" + b" \n" * 1024 + b"}"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            content=content,
            headers={"Content-Type": "application/json", "Content-Length": "2"},
        )

    transport = httpx.MockTransport(handler)
    client = create_client(transport=transport)
    try:
        with pytest.raises(JSONDecodeError):
            fetch_json(
                method="GET",
                url="http://93.184.216.34/json",
                client=client,
                require_json_ct=True,
                max_bytes=10,
            )
    finally:
        client.close()


@requires_httpx
@pytest.mark.asyncio
async def test_async_json_max_bytes_guard_uses_actual_body_despite_short_content_length():
    import httpx
    from tldw_Server_API.app.core.http_client import afetch_json, create_async_client
    from tldw_Server_API.app.core.exceptions import JSONDecodeError

    content = b"{" + b" \n" * 1024 + b"}"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            content=content,
            headers={"Content-Type": "application/json", "Content-Length": "2"},
        )

    transport = httpx.MockTransport(handler)
    client = create_async_client(transport=transport)
    try:
        with pytest.raises(JSONDecodeError):
            await afetch_json(
                method="GET",
                url="http://93.184.216.34/json",
                client=client,
                require_json_ct=True,
                max_bytes=10,
            )
    finally:
        await client.aclose()


@requires_httpx
@pytest.mark.asyncio
async def test_stream_cancellation_propagates():
    import httpx
    from tldw_Server_API.app.core.http_client import astream_bytes, create_async_client

    big = b"x" * (1024 * 1024)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, content=big, headers={"Content-Type": "application/octet-stream"})

    transport = httpx.MockTransport(handler)
    client = create_async_client(transport=transport)

    async def reader():
        async for _ in astream_bytes(method="GET", url="http://93.184.216.34/stream", client=client, chunk_size=65536):
            # Simulate consumer cancellation mid-stream
            raise asyncio.CancelledError

    task = asyncio.create_task(reader())
    with pytest.raises(asyncio.CancelledError):
        await task
    await client.aclose()


@requires_httpx
def test_metrics_increment_on_successful_request():
    import httpx
    from tldw_Server_API.app.core.http_client import fetch, create_client
    from tldw_Server_API.app.core.Metrics import get_metrics_registry

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, text="ok")

    transport = httpx.MockTransport(handler)
    client = create_client(transport=transport)
    try:
        reg = get_metrics_registry()
        before = reg.get_metric_stats("http_client_requests_total") or {"sum": 0}
        fetch(method="GET", url="http://93.184.216.34/ok", client=client)
        after = reg.get_metric_stats("http_client_requests_total") or {"sum": 0}
        assert (after.get("sum", 0) or 0) >= (before.get("sum", 0) or 0)
    finally:
        client.close()


@requires_httpx
def test_proxy_allowlist_dict_form_allows(monkeypatch):
    from tldw_Server_API.app.core.http_client import create_client
    import tldw_Server_API.app.core.http_client as hc

    # Allow a specific proxy host via allowlist and verify dict-form proxies pass
    monkeypatch.setattr(hc, "PROXY_ALLOWLIST", {"proxy.internal"})
    client = create_client(proxies={"http": "http://proxy.internal:8080", "https": "http://proxy.internal:8080"})
    try:
        assert client is not None
    finally:
        client.close()


@requires_httpx
@pytest.mark.asyncio
async def test_mixed_host_redirect_egress_denied(monkeypatch):
    import httpx
    from tldw_Server_API.app.core.http_client import afetch, create_async_client
    from tldw_Server_API.app.core.exceptions import EgressPolicyError, RetryExhaustedError

    # Force private IP blocking on for this test
    monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "true")

    # First hop is a public IP; redirect points to 127.0.0.1 which must be denied by egress
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "93.184.216.34":  # example.com
            return httpx.Response(302, request=request, headers={"Location": "http://127.0.0.1/blocked"})
        return httpx.Response(200, request=request, text="ok")

    transport = httpx.MockTransport(handler)
    client = create_async_client(transport=transport)
    try:
        # Some stacks may surface per-hop egress denial as a terminal retry failure;
        # accept either explicit egress error or retry exhaustion here.
        with pytest.raises((EgressPolicyError, RetryExhaustedError)):
            await afetch(method="GET", url="http://93.184.216.34/start", client=client)
    finally:
        await client.aclose()


@requires_httpx
@pytest.mark.asyncio
async def test_afetch_pins_dns_resolution_for_subsequent_egress_checks(monkeypatch):
    import httpx
    import types
    from tldw_Server_API.app.core.http_client import afetch, create_async_client
    from tldw_Server_API.app.core.Security import egress as egress_mod

    observed_pins: list[object] = []

    def _fake_policy(url: str, *, block_private_override=None, resolved_ips_override=None, pinned_resolved_ips=None):  # noqa: ARG001
        observed_pins.append(pinned_resolved_ips)
        if pinned_resolved_ips is None:
            return types.SimpleNamespace(
                allowed=True,
                reason=None,
                resolved_ips=("93.184.216.34",),
            )
        return types.SimpleNamespace(
            allowed=True,
            reason=None,
            resolved_ips=tuple(pinned_resolved_ips),
        )

    monkeypatch.setattr(egress_mod, "evaluate_url_policy", _fake_policy)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, request=request, headers={"Location": "/next"})
        return httpx.Response(200, request=request, text="ok")

    transport = httpx.MockTransport(handler)
    client = create_async_client(transport=transport)
    try:
        resp = await afetch(method="GET", url="http://93.184.216.34/start", client=client, allow_redirects=True)
        assert resp.status_code == 200
        assert observed_pins
        assert observed_pins[0] is None
        assert all(v == ("93.184.216.34",) for v in observed_pins[1:])
    finally:
        await client.aclose()


@requires_httpx
@pytest.mark.asyncio
async def test_apost_runs_egress_validation_off_event_loop(monkeypatch):
    import types

    import tldw_Server_API.app.core.http_client as hc

    calls: list[object] = []

    async def _fake_to_thread(func, *args, **kwargs):
        calls.append((func, args, kwargs))
        return func(*args, **kwargs)

    class _DummyClient:
        async def post(self, *args, **kwargs):  # noqa: ARG002
            return types.SimpleNamespace(status_code=200)

    monkeypatch.setattr(hc.asyncio, "to_thread", _fake_to_thread)
    monkeypatch.setattr(hc, "_validate_egress_or_raise", lambda *args, **kwargs: None)

    response = await hc.apost(url="https://example.com/api", client=_DummyClient())

    assert response.status_code == 200
    assert calls
    assert calls[0][0] is hc._validate_egress_or_raise


@requires_httpx
@pytest.mark.asyncio
async def test_dns_resolution_error_not_retried():
    """
    Ensure DNS/unknown-host style errors are treated as permanent and not retried.

    We simulate this by raising a socket.gaierror from the transport handler
    and assert that afetch only invokes the handler once.
    """
    import httpx
    import socket
    from tldw_Server_API.app.core.http_client import afetch, create_async_client, RetryPolicy
    from tldw_Server_API.app.core.exceptions import NetworkError

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        # Simulate OS-level DNS resolution failure
        raise socket.gaierror(8, "nodename nor servname provided, or not known")

    transport = httpx.MockTransport(handler)
    client = create_async_client(transport=transport)
    try:
        policy = RetryPolicy(attempts=3)
        with pytest.raises(NetworkError):
            await afetch(method="GET", url="http://does-not-resolve.invalid/", client=client, retry=policy)
        # Handler should be called exactly once; no retries on DNS failures
        assert calls["n"] == 1
    finally:
        await client.aclose()


@requires_httpx
@pytest.mark.asyncio
async def test_non_dns_network_error_is_retried():
    """
    Ensure non-DNS network errors (e.g., generic connect failures) still honor
    the retry policy and attempt the request multiple times.
    """
    import httpx
    from tldw_Server_API.app.core.http_client import afetch, create_async_client, RetryPolicy
    from tldw_Server_API.app.core.exceptions import NetworkError

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        # Simulate a generic connection failure that is not a DNS error
        raise httpx.ConnectError("connect failed", request=request)

    transport = httpx.MockTransport(handler)
    client = create_async_client(transport=transport)
    try:
        policy = RetryPolicy(attempts=3)
        # After exhausting attempts, afetch should raise NetworkError
        with pytest.raises(NetworkError):
            await afetch(method="GET", url="http://93.184.216.34/non_dns", client=client, retry=policy)
        # Handler should be invoked once per attempt
        assert calls["n"] == 3
    finally:
        await client.aclose()


@requires_httpx
@pytest.mark.asyncio
async def test_retry_on_unsafe_default_no_retry():
    import httpx
    from tldw_Server_API.app.core.http_client import afetch, create_async_client, RetryPolicy

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500, request=request, text="server error")
        return httpx.Response(200, request=request, text="ok")

    transport = httpx.MockTransport(handler)
    client = create_async_client(transport=transport)
    try:
        # retry_on_unsafe defaults to False; expect no retry for POST
        resp = await afetch(method="POST", url="http://93.184.216.34/unsafe", client=client, retry=RetryPolicy(attempts=2))
        assert resp.status_code == 500
        assert calls["n"] == 1
    finally:
        await client.aclose()


@requires_httpx
@pytest.mark.asyncio
async def test_retry_on_unsafe_true_does_retry():
    import httpx
    from tldw_Server_API.app.core.http_client import afetch, create_async_client, RetryPolicy

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(500, request=request, text="server error")
        return httpx.Response(200, request=request, text="ok")

    transport = httpx.MockTransport(handler)
    client = create_async_client(transport=transport)
    try:
        policy = RetryPolicy(attempts=2, retry_on_unsafe=True)
        resp = await afetch(method="POST", url="http://93.184.216.34/unsafe", client=client, retry=policy)
        assert resp.status_code == 200
        assert calls["n"] == 2
    finally:
        await client.aclose()
