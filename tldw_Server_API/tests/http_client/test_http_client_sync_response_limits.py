"""Synchronous response byte limits through the real central HTTP transport."""

import asyncio
import socket
import threading
import time
import types
from contextlib import contextmanager

import httpx
import pytest

from tldw_Server_API.app.core import http_client as hc

pytestmark = pytest.mark.unit
URL = "http://93.184.216.34/models"


@contextmanager
def trickling_server(phase):
    stop = threading.Event()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(3)
        url = f"http://127.0.0.1:{listener.getsockname()[1]}"

        def serve():
            try:
                connection, _ = listener.accept()
                with connection:
                    connection.settimeout(3)
                    connection.recv(65536)
                    body = b'{"data":[{"id":"current-model"}]}'
                    headers = f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\n\r\n".encode()
                    if phase == "headers":
                        trickle, tail = headers, body
                    elif phase == "body":
                        connection.sendall(headers)
                        trickle, tail = body, b""
                    else:
                        connection.sendall(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
                        trickle = b"0" * 40
                        tail = f"{len(body):x}\r\n".encode() + body + b"\r\n0\r\n\r\n"
                    for byte in trickle:
                        if stop.wait(0.03):
                            return
                        connection.sendall(bytes([byte]))
                    connection.sendall(tail)
            except OSError:
                pass  # The timed-out client closes its socket.

        worker = threading.Thread(target=serve)
        worker.start()
        try:
            yield url
        finally:
            stop.set()
            worker.join(4)
            assert not worker.is_alive()


@pytest.mark.parametrize("phase", ["headers", "body", "chunk_frame"])
@pytest.mark.parametrize("running_loop", [False, True])
def test_discovery_deadline_interrupts_trickling_network_reads(monkeypatch, phase, running_loop):
    from tldw_Server_API.app.core.LLM_Calls import provider_model_inventory as inventory
    from tldw_Server_API.app.core.Security import egress

    monkeypatch.setattr(inventory, "_TIMEOUT_SECONDS", 0.25)
    clients = []
    create_async_client = hc.create_async_client

    def owned_client(**kwargs):
        assert hc._SENSITIVE_HTTP_LOG_CONTEXT.get()
        client = create_async_client(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(hc, "create_async_client", owned_client)
    monkeypatch.setattr(
        egress,
        "evaluate_url_policy",
        lambda *_args, **_kwargs: egress.URLPolicyResult(True, resolved_ips=("127.0.0.1",)),
    )
    with trickling_server(phase) as base:
        def fetch_local(**kwargs):
            kwargs["url"] = base + "/models"
            return hc.fetch(**kwargs)

        monkeypatch.setattr(inventory, "_http_fetch", fetch_local)
        def discover():
            return inventory.discover_provider_models(
                "deepseek", "synthetic-key", base_url=base.replace("http:", "https:"), force_refresh=True
            )

        async def on_loop():
            return discover()

        threads_before = set(threading.enumerate())
        started = time.monotonic()
        result = asyncio.run(on_loop()) if running_loop else discover()
        elapsed = time.monotonic() - started
        assert not any(
            thread.name.startswith("ThreadPoolExecutor")
            for thread in set(threading.enumerate()) - threads_before
        )
    assert result.status == "unreachable"
    assert elapsed < 0.8
    assert clients and all(client.is_closed for client in clients)


class CountingStream(httpx.SyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.yielded = 0
        self.closed = False

    def __iter__(self):
        for chunk in self.chunks:
            self.yielded += 1
            yield chunk

    def close(self):
        self.closed = True


@pytest.mark.parametrize(("chunks", "limit"), [([b"sa", b"fe"], 5), ([b"safe"], 4), ([], 0)])
@pytest.mark.parametrize("method", ["GET", "POST"])
def test_sync_fetch_bounds_raw_body_and_forces_identity(chunks, limit, method):
    stream = CountingStream(chunks)
    seen = []
    headers = {"aCcEpT-EnCoDiNg": "gzip, br", "X-Test": "kept"}

    def handler(request):
        seen.append(request)
        return httpx.Response(200, stream=stream, headers={"X-Result": "kept"})

    with hc.create_client(transport=httpx.MockTransport(handler)) as client:
        response = hc.fetch(
            method=method,
            url=URL,
            client=client,
            headers=headers,
            max_response_bytes=limit,
            retry=hc.RetryPolicy(attempts=1),
        )
    assert response.content == b"".join(chunks)
    assert response.headers["x-result"] == "kept"
    assert seen[0].headers["accept-encoding"] == "identity"
    assert seen[0].headers["x-test"] == "kept"
    assert seen[0].extensions[hc._BOUNDED_RESPONSE_EXTENSION] is True
    assert headers["aCcEpT-EnCoDiNg"] == "gzip, br"
    assert stream.closed


@pytest.mark.parametrize("chunks", [[b"x" * 6, b"sentinel"], [b"x"] * 6 + [b"sentinel"]])
def test_sync_fetch_aborts_at_first_oversized_raw_chunk(chunks):
    stream = CountingStream(chunks)

    class RawOnlyResponse(httpx.Response):
        def iter_bytes(self, chunk_size=None):
            raise AssertionError("Bounded reads must not decode the body")

    with hc.create_client(transport=httpx.MockTransport(lambda request: RawOnlyResponse(200, stream=stream))) as client:
        with pytest.raises(hc.NetworkError):
            hc.fetch(method="GET", url=URL, client=client, max_response_bytes=5, retry=hc.RetryPolicy(attempts=1))
    assert stream.yielded == (1 if len(chunks[0]) > 5 else 6)
    assert stream.closed


@pytest.mark.parametrize("status", [200, 206])
def test_sync_fetch_rejects_compressed_success_before_consuming_body(status):
    stream = CountingStream([b"must-not-read"])

    def handler(request):
        return httpx.Response(status, stream=stream, headers={"Content-Encoding": "gzip"})

    with hc.create_client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(hc.NetworkError):
            hc.fetch(method="GET", url=URL, client=client, max_response_bytes=5, retry=hc.RetryPolicy(attempts=1))
    assert stream.yielded == 0
    assert stream.closed


@pytest.mark.parametrize("status", [302, 401, 422, 500])
def test_sync_fetch_skips_non_success_body_even_with_error_capture_hook(status):
    stream = CountingStream([b"must-not-read"])

    def handler(request):
        return httpx.Response(status, stream=stream, headers={"Content-Encoding": "gzip", "X-Error": "kept"})

    with hc.create_client(transport=httpx.MockTransport(handler)) as client:
        response = hc.fetch(
            method="GET",
            url=URL,
            client=client,
            max_response_bytes=5,
            allow_redirects=False,
            retry=hc.RetryPolicy(attempts=1),
        )
    assert response.status_code == status
    assert response.content == b""
    assert response.text == ""
    assert response.headers["content-encoding"] == "gzip"
    assert response.headers["x-error"] == "kept"
    assert stream.yielded == 0
    assert stream.closed


@pytest.mark.parametrize("limit", [-1, True, 1.5, "5"])
def test_sync_io_rejects_invalid_limit_without_dispatch(limit):
    def handler(request):
        raise AssertionError("Invalid limit must not dispatch")

    with hc.create_client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="^max_response_bytes must be a non-negative integer$"):
            hc._httpx_request_io(client=client, method="GET", url=URL, max_response_bytes=limit)


@pytest.mark.parametrize("deadline_enabled", [False, True])
def test_sync_bounded_fetch_preserves_accepted_dns_pin_host_sni_and_public_response_url(monkeypatch, deadline_enabled):
    from tldw_Server_API.app.core.Security import egress as egress_mod

    original = "https://models.example/v1/models"
    validations = []
    seen = []
    stream = httpx.ByteStream(b"safe") if deadline_enabled else CountingStream([b"safe"])

    def allow(url, **kwargs):
        validations.append(kwargs.get("pinned_resolved_ips"))
        return types.SimpleNamespace(allowed=True, reason=None, reason_code=None, resolved_ips=("93.184.216.34",))

    def handler(request):
        seen.append((str(request.url), request.headers["host"], request.extensions["sni_hostname"]))
        return httpx.Response(200, stream=stream)

    monkeypatch.setattr(egress_mod, "evaluate_url_policy", allow)
    create_async_client = hc.create_async_client
    monkeypatch.setattr(
        hc, "create_async_client", lambda **kwargs: create_async_client(transport=httpx.MockTransport(handler), **kwargs)
    )
    with hc.create_client(transport=httpx.MockTransport(handler)) as client:
        options = {"deadline": time.monotonic() + 1} if deadline_enabled else {"client": client}
        response = hc.fetch(method="GET", url=original, max_response_bytes=4, **options)
    assert seen == [("https://93.184.216.34/v1/models", "models.example", "models.example")]
    assert str(response.url) == original
    assert response.content == b"safe"
    assert ("93.184.216.34",) in validations


@pytest.mark.parametrize("deadline", [True, "5", float("nan"), float("inf")])
def test_sync_bounded_fetch_rejects_invalid_deadline_before_dispatch(deadline):
    with pytest.raises(ValueError, match="deadline must be a finite monotonic timestamp"):
        hc.fetch(method="GET", url=URL, max_response_bytes=4, deadline=deadline)


def test_sync_bounded_fetch_expired_deadline_does_not_dispatch(monkeypatch):
    def no_client(**kwargs):
        raise AssertionError("Expired request must not create a network client")

    monkeypatch.setattr(hc, "create_async_client", no_client)
    with pytest.raises(hc.NetworkError, match="TimeoutError"):
        hc.fetch(
            method="GET", url=URL, max_response_bytes=4, deadline=time.monotonic() - 1,
            retry=hc.RetryPolicy(attempts=1),
        )


def test_sync_bounded_redirect_enforces_limit_on_next_hop():
    streams = [CountingStream([b"redirect-body-must-not-read"]), CountingStream([b"123456", b"sentinel"])]
    seen = []

    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(302, stream=streams[0], headers={"Location": "http://93.184.216.35/final"})
        return httpx.Response(200, stream=streams[1])

    with hc.create_client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(hc.NetworkError):
            hc.fetch(
                method="GET",
                url=URL,
                client=client,
                max_response_bytes=5,
                headers={"Authorization": "synthetic-secret"},
                retry=hc.RetryPolicy(attempts=1),
            )
    assert len(seen) == 2
    assert "authorization" not in seen[1].headers
    assert all(request.headers["accept-encoding"] == "identity" for request in seen)
    assert [stream.yielded for stream in streams] == [0, 1]
    assert all(stream.closed for stream in streams)


def test_sync_bounded_fetch_keeps_egress_guard_before_transport(monkeypatch):
    from tldw_Server_API.app.core.exceptions import EgressPolicyError

    monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "true")

    def handler(request):
        raise AssertionError("Denied URL must not dispatch")

    with hc.create_client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(EgressPolicyError):
            hc.fetch(method="GET", url="http://127.0.0.1/models", client=client, max_response_bytes=5)


def test_unbounded_sync_fetch_does_not_add_keyword_to_legacy_io(monkeypatch):
    def legacy_io(
        *,
        client,
        method,
        url,
        headers,
        cookies,
        params,
        json,
        data,
        files,
        timeout,
        follow_redirects,
        accepted_resolved_ips,
    ):
        return httpx.Response(200, content=b"legacy", request=httpx.Request(method, url))

    monkeypatch.setattr(hc, "_httpx_request_io", legacy_io)
    with hc.create_client(transport=httpx.MockTransport(lambda request: None)) as client:
        response = hc.fetch(method="GET", url=URL, client=client, retry=hc.RetryPolicy(attempts=1))
    assert response.content == b"legacy"
