"""Exercise the policy and exact three transports flagged on PR 2761."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest

from tldw_Server_API.app.core import http_client
from tldw_Server_API.app.core.exceptions import EgressPolicyError
from tldw_Server_API.app.core.Security import egress

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
def production_egress(monkeypatch):
    """Use the real production policy; replace DNS and transport I/O only."""
    monkeypatch.setattr(http_client, "is_explicit_pytest_runtime", lambda: False)
    monkeypatch.setattr(http_client, "is_test_mode", lambda: False)
    monkeypatch.setattr(http_client, "env_flag_enabled", lambda _name: False)
    monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "true")
    monkeypatch.delenv("WORKFLOWS_EGRESS_ALLOWED_DOMAINS", raising=False)
    monkeypatch.delenv("WORKFLOWS_EGRESS_DENIED_DOMAINS", raising=False)
    monkeypatch.setattr(egress, "_resolve_host_ips", lambda _host: ["93.184.216.34"])


class _Body(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b"ok"


class _AiohttpSession:
    def __init__(self, seen, redirect):
        self.seen = seen
        self.redirect = redirect
        self.cookie_jar = SimpleNamespace(clear=lambda: None)

    @asynccontextmanager
    async def request(self, method, url, **kwargs):
        self.seen.append({"url": url, **kwargs})
        status = 302 if self.redirect else 200
        headers = {"location": self.redirect} if self.redirect else {}
        remaining = bytearray(b"ok")

        async def read(size=None):
            body = bytes(remaining if size is None else remaining[:size])
            del remaining[: len(body)]
            return body

        yield SimpleNamespace(
            status=status,
            headers=headers,
            url=url,
            read=read,
            content=SimpleNamespace(read=read),
        )


async def _request(backend, url, seen, *, max_response_bytes, redirect=None):
    kwargs = {
        "method": "GET",
        "url": url,
        "headers": {"Host": "attacker.example"},
        "max_response_bytes": max_response_bytes,
        "retry": http_client.RetryPolicy(attempts=1),
    }
    if backend == "aiohttp":
        return await http_client._afetch_aiohttp(
            client=_AiohttpSession(seen, redirect), **kwargs
        )

    def handle(request):
        seen.append(
            {
                "url": str(request.url),
                "headers": dict(request.headers),
                "extensions": request.extensions,
            }
        )
        return httpx.Response(
            302 if redirect else 200,
            headers={"location": redirect} if redirect else {},
            stream=_Body(),
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        return await http_client._afetch_httpx(client=client, **kwargs)


@pytest.mark.parametrize("backend", ["httpx", "aiohttp"])
@pytest.mark.parametrize("max_response_bytes", [None, 2])
async def test_vetted_dns_address_reaches_transport_with_original_identity(
    production_egress, backend, max_response_bytes
):
    seen = []
    response = await _request(
        backend,
        "https://public.example/resource",
        seen,
        max_response_bytes=max_response_bytes,
    )
    assert response.content == b"ok"
    assert str(response.url) == "https://public.example/resource"
    assert [item["url"] for item in seen] == ["https://93.184.216.34/resource"]
    assert {key.lower(): value for key, value in seen[0]["headers"].items()}[
        "host"
    ] == "public.example"
    if backend == "aiohttp":
        assert seen[0]["server_hostname"] == "public.example"
        assert seen[0]["allow_redirects"] is False
    else:
        assert seen[0]["extensions"]["sni_hostname"] == "public.example"


@pytest.mark.parametrize("backend", ["httpx", "aiohttp"])
@pytest.mark.parametrize("max_response_bytes", [None, 2])
@pytest.mark.parametrize("url", ["http://127.0.0.1/private", "http://169.254.169.254/meta-data"])
async def test_private_destination_never_reaches_transport(
    production_egress, backend, max_response_bytes, url
):
    seen = []
    with pytest.raises(EgressPolicyError):
        await _request(backend, url, seen, max_response_bytes=max_response_bytes)
    assert seen == []


@pytest.mark.parametrize("backend", ["httpx", "aiohttp"])
@pytest.mark.parametrize("max_response_bytes", [None, 2])
async def test_public_redirect_to_private_host_never_dispatches_second_request(
    production_egress, backend, max_response_bytes
):
    seen = []
    with pytest.raises(EgressPolicyError):
        await _request(
            backend,
            "https://public.example/resource",
            seen,
            max_response_bytes=max_response_bytes,
            redirect="http://127.0.0.1/private",
        )
    assert [item["url"] for item in seen] == ["https://93.184.216.34/resource"]


@pytest.mark.parametrize("backend", ["httpx", "aiohttp"])
@pytest.mark.parametrize("max_response_bytes", [None, 2])
async def test_rebound_dns_is_denied_before_first_transport_request(
    production_egress, monkeypatch, backend, max_response_bytes
):
    answers = iter([["93.184.216.34"], ["127.0.0.1"]])
    monkeypatch.setattr(egress, "_resolve_host_ips", lambda _host: next(answers))
    seen = []
    with pytest.raises(EgressPolicyError):
        await _request(
            backend,
            "https://public.example/resource",
            seen,
            max_response_bytes=max_response_bytes,
        )
    assert seen == []
