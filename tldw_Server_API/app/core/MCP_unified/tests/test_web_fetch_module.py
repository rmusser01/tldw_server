from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.exceptions import EgressPolicyError, NetworkError
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations import (
    web_fetch_module as wfm,
)
from tldw_Server_API.app.core.MCP_unified.modules.implementations.web_fetch_module import (
    HttpxWebFetchClient,
    WebFetchModule,
    WebFetchResponse,
)
from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext

pytestmark = pytest.mark.asyncio


class _FakeClient:
    """Injectable fetcher so unit tests never touch the network."""

    def __init__(self, response: WebFetchResponse | None = None, exc: BaseException | None = None) -> None:
        self.response = response
        self.exc = exc
        self.calls: list[dict[str, Any]] = []

    async def fetch(
        self,
        url: str,
        *,
        timeout_seconds: float,
        max_bytes: int,
        user_agent: str,
    ) -> WebFetchResponse:
        self.calls.append(
            {
                "url": url,
                "timeout_seconds": timeout_seconds,
                "max_bytes": max_bytes,
                "user_agent": user_agent,
            }
        )
        if self.exc is not None:
            raise self.exc
        assert self.response is not None  # nosec B101
        return self.response


def _module(client: wfm.WebFetchHttpClient, *, rate_limiter: Any | None = None) -> WebFetchModule:
    return WebFetchModule(ModuleConfig(name="WebFetch"), client=client, rate_limiter=rate_limiter)


def _allow_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _decide(url: str, **kwargs: Any) -> Any:
        return SimpleNamespace(
            allowed=True,
            reason="allowed",
            mode="compat",
            stage=kwargs.get("stage", ""),
            source=kwargs.get("source", ""),
            details=None,
        )

    monkeypatch.setattr(wfm, "decide_web_outbound_policy", _decide)


def _deny_policy(monkeypatch: pytest.MonkeyPatch, *, reason: str = "ssrf_private_ip") -> list[str]:
    seen: list[str] = []

    async def _decide(url: str, **kwargs: Any) -> Any:
        seen.append(url)
        return SimpleNamespace(
            allowed=False,
            reason=reason,
            mode="strict",
            stage=kwargs.get("stage", ""),
            source=kwargs.get("source", ""),
            details=None,
        )

    monkeypatch.setattr(wfm, "decide_web_outbound_policy", _decide)
    return seen


def _html_response(
    body: bytes = b"",
    *,
    final_url: str = "https://example.com/post",
    status_code: int = 200,
    content_type: str = "text/html; charset=utf-8",
    truncated: bool = False,
) -> WebFetchResponse:
    return WebFetchResponse(
        final_url=final_url,
        status_code=status_code,
        content_type=content_type,
        body=body,
        truncated=truncated,
    )


def _redirect_response(*, location: str | None, final_url: str = "https://example.com/start") -> WebFetchResponse:
    return WebFetchResponse(
        final_url=final_url,
        status_code=302,
        content_type="text/html",
        body=b"",
        truncated=False,
        location=location,
    )


class _SequenceClient:
    """Fake client that returns one response per call, recording fetched URLs."""

    def __init__(self, responses: list[WebFetchResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    async def fetch(
        self,
        url: str,
        *,
        timeout_seconds: float,
        max_bytes: int,
        user_agent: str,
    ) -> WebFetchResponse:
        self.calls.append(url)
        if not self._responses:
            raise AssertionError(f"unexpected extra fetch: {url}")
        return self._responses.pop(0)


def _policy_allowing(monkeypatch: pytest.MonkeyPatch, allowed_hosts: set[str]) -> list[str]:
    """Policy mock that allows only the given hosts; records every checked URL."""
    from urllib.parse import urlsplit

    checked: list[str] = []

    async def _decide(url: str, **kwargs: Any) -> Any:
        checked.append(url)
        host = urlsplit(url).hostname or ""
        return SimpleNamespace(
            allowed=host in allowed_hosts,
            reason="allowed" if host in allowed_hosts else "ssrf_private_ip",
            mode="strict",
            stage=kwargs.get("stage", ""),
            source=kwargs.get("source", ""),
            details=None,
        )

    monkeypatch.setattr(wfm, "decide_web_outbound_policy", _decide)
    return checked


_RICH_HTML = (
    b"<html><head><title>Example Title</title></head>"
    b"<body><article><h1>Heading</h1>"
    b"<p>Example article body paragraph one with enough text to extract.</p>"
    b"<p>Example article body paragraph two continuing the readable content.</p>"
    b"</article></body></html>"
)


async def test_get_tools_exposes_web_fetch_contract() -> None:
    module = _module(_FakeClient())
    tools = await module.get_tools()
    assert [tool["name"] for tool in tools] == ["web.fetch"]  # nosec B101
    tool = tools[0]
    assert tool["inputSchema"]["required"] == ["url"]  # nosec B101
    assert tool["inputSchema"]["additionalProperties"] is False  # nosec B101
    assert tool["metadata"]["readOnlyHint"] is True  # nosec B101


async def test_invalid_scheme_returns_invalid_url(monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_policy(monkeypatch)
    client = _FakeClient()
    module = _module(client)
    result = await module.execute_tool("web.fetch", {"url": "ftp://example.com/x"})
    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "invalid_url"  # nosec B101
    assert client.calls == []  # nosec B101


async def test_outbound_policy_denial_blocks_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _deny_policy(monkeypatch)
    client = _FakeClient()
    module = _module(client)
    result = await module.execute_tool("web.fetch", {"url": "https://169.254.169.254/latest"})
    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "outbound_policy_denied"  # nosec B101
    assert seen == ["https://169.254.169.254/latest"]  # nosec B101
    assert client.calls == []  # nosec B101


async def test_happy_path_html_extracted_to_markdown(monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_policy(monkeypatch)
    client = _FakeClient(_html_response(_RICH_HTML))
    module = _module(client)
    result = await module.execute_tool(
        "web.fetch",
        {"url": "https://example.com/post", "format": "markdown"},
    )
    assert result["ok"] is True  # nosec B101
    assert result["status_code"] == 200  # nosec B101
    assert result["final_url"] == "https://example.com/post"  # nosec B101
    assert result["format"] == "markdown"  # nosec B101
    assert "Example article body" in result["content"]  # nosec B101
    assert result["title"] == "Example Title"  # nosec B101
    assert result["bytes_fetched"] == len(_RICH_HTML)  # nosec B101
    assert result["truncated"] is False  # nosec B101
    assert client.calls and client.calls[0]["url"] == "https://example.com/post"  # nosec B101


async def test_byte_cap_truncation_is_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_policy(monkeypatch)
    client = _FakeClient(_html_response(_RICH_HTML, truncated=True))
    module = _module(client)
    result = await module.execute_tool(
        "web.fetch",
        {"url": "https://example.com/post", "max_bytes": 16},
    )
    assert result["ok"] is True  # nosec B101
    assert result["truncated"] is True  # nosec B101
    assert client.calls[0]["max_bytes"] == 16  # nosec B101


async def test_text_plain_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_policy(monkeypatch)
    body = b"plain readable body content"
    client = _FakeClient(
        _html_response(body, content_type="text/plain; charset=utf-8")
    )
    module = _module(client)
    result = await module.execute_tool("web.fetch", {"url": "https://example.com/raw.txt"})
    assert result["ok"] is True  # nosec B101
    assert result["content"] == "plain readable body content"  # nosec B101


async def test_unsupported_content_type_returns_empty_content(monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_policy(monkeypatch)
    client = _FakeClient(_html_response(b"\x89PNG\r\n", content_type="image/png"))
    module = _module(client)
    result = await module.execute_tool("web.fetch", {"url": "https://example.com/logo.png"})
    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "empty_content"  # nosec B101


async def test_client_exception_maps_to_fetch_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_policy(monkeypatch)
    client = _FakeClient(exc=TimeoutError("boom"))
    module = _module(client)
    result = await module.execute_tool("web.fetch", {"url": "https://example.com/slow"})
    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "fetch_failed"  # nosec B101


async def test_http_error_status_maps_to_fetch_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_policy(monkeypatch)
    client = _FakeClient(_html_response(b"not found", status_code=404, content_type="text/html"))
    module = _module(client)
    result = await module.execute_tool("web.fetch", {"url": "https://example.com/missing"})
    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "fetch_failed"  # nosec B101
    assert result["status_code"] == 404  # nosec B101


async def test_unknown_argument_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_policy(monkeypatch)
    module = _module(_FakeClient())
    result = await module.execute_tool(
        "web.fetch", {"url": "https://example.com", "depth": 3}
    )
    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "invalid_arguments"  # nosec B101


@pytest.mark.parametrize(
    "arguments",
    [
        {"url": "https://example.com", "max_bytes": -1},
        {"url": "https://example.com", "timeout_seconds": 0},
        {"url": "https://example.com", "max_bytes": 10**9},
        {"url": "https://example.com", "format": "pdf"},
        {"url": 123},
    ],
)
async def test_bad_bounds_are_rejected(
    monkeypatch: pytest.MonkeyPatch, arguments: dict[str, Any]
) -> None:
    _allow_policy(monkeypatch)
    module = _module(_FakeClient())
    result = await module.execute_tool("web.fetch", arguments)
    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "invalid_arguments"  # nosec B101


async def test_unknown_tool_returns_structured_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_policy(monkeypatch)
    module = _module(_FakeClient())
    result = await module.execute_tool("web.other", {"url": "https://example.com"})
    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "unknown_tool"  # nosec B101


async def test_eval_metadata_records_profile_from_context(monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_policy(monkeypatch)
    client = _FakeClient(_html_response(_RICH_HTML))
    module = _module(client)
    context = RequestContext(request_id="req-web", metadata={"profile_id": "deep-researcher"})
    result = await module.execute_tool(
        "web.fetch", {"url": "https://example.com/post"}, context
    )
    assert result["ok"] is True  # nosec B101
    assert result["eval"]["profile_id"] == "deep-researcher"  # nosec B101


async def test_redirect_into_denied_host_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    checked = _policy_allowing(monkeypatch, {"example.com"})
    client = _SequenceClient(
        [_redirect_response(location="https://169.254.169.254/latest/meta-data")]
    )
    module = _module(client)
    result = await module.execute_tool("web.fetch", {"url": "https://example.com/start"})
    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "outbound_policy_denied"  # nosec B101
    # Only the first hop was fetched; the denied redirect target was never requested.
    assert client.calls == ["https://example.com/start"]  # nosec B101
    assert "https://169.254.169.254/latest/meta-data" in checked  # nosec B101


async def test_redirect_to_allowed_host_is_followed(monkeypatch: pytest.MonkeyPatch) -> None:
    _policy_allowing(monkeypatch, {"example.com", "example.org"})
    client = _SequenceClient(
        [
            _redirect_response(location="https://example.org/final"),
            _html_response(_RICH_HTML, final_url="https://example.org/final"),
        ]
    )
    module = _module(client)
    result = await module.execute_tool("web.fetch", {"url": "https://example.com/start"})
    assert result["ok"] is True  # nosec B101
    assert result["final_url"] == "https://example.org/final"  # nosec B101
    assert client.calls == ["https://example.com/start", "https://example.org/final"]  # nosec B101


async def test_redirect_without_location_is_fetch_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    _policy_allowing(monkeypatch, {"example.com"})
    client = _SequenceClient([_redirect_response(location=None)])
    module = _module(client)
    result = await module.execute_tool("web.fetch", {"url": "https://example.com/start"})
    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "fetch_failed"  # nosec B101


async def test_redirect_limit_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    _policy_allowing(monkeypatch, {"example.com"})
    client = _SequenceClient(
        [_redirect_response(location=f"https://example.com/hop{i}") for i in range(10)]
    )
    module = _module(client)
    result = await module.execute_tool("web.fetch", {"url": "https://example.com/start"})
    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "fetch_failed"  # nosec B101


async def test_url_with_sql_like_substrings_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_policy(monkeypatch)
    client = _FakeClient(_html_response(_RICH_HTML))
    module = _module(client)
    # URLs commonly contain '--' and '/*'; sanitize_input must not reject them.
    result = await module.execute_tool(
        "web.fetch", {"url": "https://example.com/a--b/c?x=1/*y*/&z=--q"}
    )
    assert result["ok"] is True  # nosec B101


async def test_sanitize_input_allows_sql_like_url() -> None:
    # The MCP protocol layer runs module.sanitize_input before execution; the
    # override must not reject URLs containing '--' or '/*'.
    module = _module(_FakeClient())
    cleaned = module.sanitize_input({"url": "https://example.com/a--b?x=/*y*/"})
    assert cleaned["url"] == "https://example.com/a--b?x=/*y*/"  # nosec B101


async def test_sanitize_input_strips_control_characters() -> None:
    module = _module(_FakeClient())
    cleaned = module.sanitize_input({"url": "https://example.com/a\x00b"})
    assert cleaned["url"] == "https://example.com/ab"  # nosec B101


async def test_non_utf8_charset_is_decoded(monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_policy(monkeypatch)
    body = "café déjà vu".encode("latin-1")
    client = _FakeClient(
        _html_response(body, content_type="text/plain; charset=ISO-8859-1")
    )
    module = _module(client)
    result = await module.execute_tool("web.fetch", {"url": "https://example.com/raw"})
    assert result["ok"] is True  # nosec B101
    assert "café" in result["content"]  # nosec B101


async def test_missing_content_type_falls_back_to_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_policy(monkeypatch)
    client = _FakeClient(_html_response(b"just some text", content_type=""))
    module = _module(client)
    result = await module.execute_tool("web.fetch", {"url": "https://example.com/raw"})
    assert result["ok"] is True  # nosec B101
    assert result["content"] == "just some text"  # nosec B101


async def test_html_entities_are_unescaped(monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_policy(monkeypatch)
    body = (
        b"<html><head><title>Tom &amp; Jerry</title></head>"
        b"<body><p>Cats &amp; dogs &lt;3 &gt; rocks</p></body></html>"
    )
    client = _FakeClient(_html_response(body))
    module = _module(client)
    result = await module.execute_tool(
        "web.fetch", {"url": "https://example.com/post", "format": "text"}
    )
    assert result["ok"] is True  # nosec B101
    assert result["title"] == "Tom & Jerry"  # nosec B101
    assert "&amp;" not in result["content"]  # nosec B101


async def test_httpx_client_does_not_auto_follow_redirects() -> None:
    import httpx

    requested: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(302, headers={"Location": "/next"})

    client = HttpxWebFetchClient(transport=httpx.MockTransport(_handler))
    response = await client.fetch(
        "https://example.com/start",
        timeout_seconds=5,
        max_bytes=1000,
        user_agent="test",
    )
    assert response.status_code == 302  # nosec B101
    assert response.location == "https://example.com/next"  # nosec B101
    # The redirect target was never requested by the client.
    assert requested == ["https://example.com/start"]  # nosec B101


async def test_httpx_client_skips_body_for_unsupported_content_type() -> None:
    import httpx

    body_chunks_sent = {"n": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        body_chunks_sent["n"] += 1
        return httpx.Response(200, headers={"Content-Type": "image/png"}, content=b"\x89PNG" * 100)

    client = HttpxWebFetchClient(transport=httpx.MockTransport(_handler))
    response = await client.fetch(
        "https://example.com/logo.png",
        timeout_seconds=5,
        max_bytes=1000,
        user_agent="test",
    )
    assert response.status_code == 200  # nosec B101
    assert response.content_type == "image/png"  # nosec B101
    assert response.body == b""  # nosec B101 - unsupported type not downloaded


async def test_per_domain_rate_limit_blocks_second_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.MCP_unified.modules.implementations.web_rate_limit import (
        DomainRateLimiter,
    )

    _allow_policy(monkeypatch)
    client = _FakeClient(_html_response(_RICH_HTML))
    module = _module(client, rate_limiter=DomainRateLimiter(max_requests=1, window_seconds=60))

    first = await module.execute_tool("web.fetch", {"url": "https://example.com/a"})
    assert first["ok"] is True  # nosec B101

    second = await module.execute_tool("web.fetch", {"url": "https://example.com/b"})
    assert second["ok"] is False  # nosec B101
    assert second["reason_code"] == "rate_limited"  # nosec B101
    # The blocked request never reached the client (one successful fetch only).
    assert len(client.calls) == 1  # nosec B101


async def test_rate_limit_is_per_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.MCP_unified.modules.implementations.web_rate_limit import (
        DomainRateLimiter,
    )

    _allow_policy(monkeypatch)
    client = _FakeClient(_html_response(_RICH_HTML))
    module = _module(client, rate_limiter=DomainRateLimiter(max_requests=1, window_seconds=60))

    a = await module.execute_tool("web.fetch", {"url": "https://example.com/x"})
    b = await module.execute_tool("web.fetch", {"url": "https://example.org/y"})
    assert a["ok"] is True  # nosec B101
    assert b["ok"] is True  # nosec B101 - different domain, independent budget


async def test_rate_limit_disabled_allows_many(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.MCP_unified.modules.implementations.web_rate_limit import (
        DomainRateLimiter,
    )

    _allow_policy(monkeypatch)
    client = _FakeClient(_html_response(_RICH_HTML))
    module = _module(client, rate_limiter=DomainRateLimiter(max_requests=0))
    for _ in range(5):
        result = await module.execute_tool("web.fetch", {"url": "https://example.com/a"})
        assert result["ok"] is True  # nosec B101


async def test_response_cache_serves_second_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.MCP_unified.modules.implementations.web_cache import (
        ResponseCache,
    )

    _allow_policy(monkeypatch)
    client = _FakeClient(_html_response(_RICH_HTML))
    module = WebFetchModule(
        ModuleConfig(name="WebFetch"), client=client, response_cache=ResponseCache(ttl_seconds=300)
    )

    first = await module.execute_tool("web.fetch", {"url": "https://example.com/post"})
    assert first["ok"] is True  # nosec B101
    assert first["cached"] is False  # nosec B101
    assert len(client.calls) == 1  # nosec B101

    second = await module.execute_tool("web.fetch", {"url": "https://example.com/post"})
    assert second["ok"] is True  # nosec B101
    assert second["cached"] is True  # nosec B101
    assert second["content"] == first["content"]  # nosec B101
    # Served from cache: the client was not called again.
    assert len(client.calls) == 1  # nosec B101


async def test_response_cache_key_includes_format(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.MCP_unified.modules.implementations.web_cache import (
        ResponseCache,
    )

    _allow_policy(monkeypatch)
    client = _FakeClient(_html_response(_RICH_HTML))
    module = WebFetchModule(
        ModuleConfig(name="WebFetch"), client=client, response_cache=ResponseCache(ttl_seconds=300)
    )
    await module.execute_tool("web.fetch", {"url": "https://example.com/post", "format": "markdown"})
    # A different format is a cache miss -> the client is called again.
    result = await module.execute_tool("web.fetch", {"url": "https://example.com/post", "format": "text"})
    assert result["cached"] is False  # nosec B101
    assert len(client.calls) == 2  # nosec B101


async def test_response_cache_does_not_store_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.MCP_unified.modules.implementations.web_cache import (
        ResponseCache,
    )

    _allow_policy(monkeypatch)
    client = _FakeClient(_html_response(b"missing", status_code=404, content_type="text/html"))
    module = WebFetchModule(
        ModuleConfig(name="WebFetch"), client=client, response_cache=ResponseCache(ttl_seconds=300)
    )
    first = await module.execute_tool("web.fetch", {"url": "https://example.com/missing"})
    assert first["ok"] is False  # nosec B101
    second = await module.execute_tool("web.fetch", {"url": "https://example.com/missing"})
    # Errors are not cached -> the client is retried.
    assert second["ok"] is False  # nosec B101
    assert len(client.calls) == 2  # nosec B101


async def test_cache_hit_uses_current_context_for_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.MCP_unified.modules.implementations.web_cache import (
        ResponseCache,
    )

    _allow_policy(monkeypatch)
    client = _FakeClient(_html_response(_RICH_HTML))
    module = WebFetchModule(
        ModuleConfig(name="WebFetch"), client=client, response_cache=ResponseCache(ttl_seconds=300)
    )
    ctx_a = RequestContext(request_id="r1", metadata={"profile_id": "alpha"})
    ctx_b = RequestContext(request_id="r2", metadata={"profile_id": "beta"})

    first = await module.execute_tool("web.fetch", {"url": "https://example.com/post"}, ctx_a)
    assert first["eval"]["profile_id"] == "alpha"  # nosec B101

    second = await module.execute_tool("web.fetch", {"url": "https://example.com/post"}, ctx_b)
    assert second["cached"] is True  # nosec B101
    # eval is rebuilt from the current request context, not the cached one.
    assert second["eval"]["profile_id"] == "beta"  # nosec B101


async def test_cache_hit_re_enforces_outbound_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.MCP_unified.modules.implementations.web_cache import (
        ResponseCache,
    )

    _allow_policy(monkeypatch)
    client = _FakeClient(_html_response(_RICH_HTML))
    module = WebFetchModule(
        ModuleConfig(name="WebFetch"), client=client, response_cache=ResponseCache(ttl_seconds=300)
    )
    first = await module.execute_tool("web.fetch", {"url": "https://example.com/post"})
    assert first["ok"] is True  # nosec B101

    # The domain becomes denied; a cache hit must NOT serve the stored content.
    _deny_policy(monkeypatch)
    second = await module.execute_tool("web.fetch", {"url": "https://example.com/post"})
    assert second["ok"] is False  # nosec B101
    assert second["reason_code"] == "outbound_policy_denied"  # nosec B101


# ---- Egress: the default client goes through core.http_client (TASK-13950.11) ----

_PUBLIC_URL = "http://93.184.216.34/page"


@pytest.fixture
def _http_egress(monkeypatch: pytest.MonkeyPatch) -> None:
    """Block private and link-local addresses.

    A refused request is proven by the MockTransport's record of requests. httpx
    connects through anyio, not ``socket.create_connection``, so the patch below
    only catches a stdlib or urllib fetch.
    """
    import socket

    def _no_real_network(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("web.fetch bypassed core.http_client")

    monkeypatch.setattr(socket, "create_connection", _no_real_network)
    monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "true")


def _recording_transport(requested: list[str], handler: Any | None = None) -> Any:
    """MockTransport that records each URL; by default it serves _RICH_HTML."""
    import httpx

    def _handle(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        if handler is not None:
            return handler(request)
        return httpx.Response(200, request=request, headers={"Content-Type": "text/html"}, content=_RICH_HTML)

    return httpx.MockTransport(_handle)


@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1/admin", "http://169.254.169.254/latest/meta-data/", "http://10.0.0.5/"],
)
async def test_httpx_client_refuses_private_targets_itself(_http_egress: None, url: str) -> None:
    """The default client is egress-checked even without the module's own policy check."""
    requested: list[str] = []
    client = HttpxWebFetchClient(transport=_recording_transport(requested))

    with pytest.raises(EgressPolicyError):
        await client.fetch(url, timeout_seconds=5, max_bytes=1000, user_agent="test")

    assert requested == []  # nosec B101


async def test_policy_race_still_cannot_reach_private_address(
    monkeypatch: pytest.MonkeyPatch, _http_egress: None
) -> None:
    """If the module's check passes (a DNS answer that later changes), http_client refuses the connection."""
    _allow_policy(monkeypatch)
    requested: list[str] = []
    module = _module(HttpxWebFetchClient(transport=_recording_transport(requested)))

    result = await module.execute_tool("web.fetch", {"url": "http://169.254.169.254/latest/meta-data/"})

    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "outbound_policy_denied"  # nosec B101
    assert requested == []  # nosec B101


async def test_real_policy_refuses_redirect_to_private_address_at_the_hop(_http_egress: None) -> None:
    """A public page that redirects to cloud metadata is refused before the hop is requested."""
    import httpx

    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, request=request, headers={"Location": "http://169.254.169.254/latest/meta-data/"})

    module = _module(HttpxWebFetchClient(transport=_recording_transport(requested, handler)))

    result = await module.execute_tool("web.fetch", {"url": _PUBLIC_URL})

    assert result["reason_code"] == "outbound_policy_denied"  # nosec B101
    assert requested == [_PUBLIC_URL]  # nosec B101


async def test_real_policy_public_page_fetches_through_http_client(_http_egress: None) -> None:
    """A public page still comes back unchanged through the real policy and client."""
    import httpx

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200, request=request, headers={"Content-Type": "text/html; charset=utf-8"}, content=_RICH_HTML
        )

    module = _module(HttpxWebFetchClient(transport=httpx.MockTransport(handler)))

    result = await module.execute_tool("web.fetch", {"url": _PUBLIC_URL, "format": "text"})

    assert result["ok"] is True  # nosec B101
    assert result["final_url"] == _PUBLIC_URL  # nosec B101
    assert result["status_code"] == 200  # nosec B101
    assert result["content_type"] == "text/html; charset=utf-8"  # nosec B101
    assert result["title"] == "Example Title"  # nosec B101
    assert result["bytes_fetched"] == len(_RICH_HTML)  # nosec B101
    assert result["truncated"] is False  # nosec B101
    assert [str(request.url) for request in seen] == [_PUBLIC_URL]  # nosec B101
    assert seen[0].headers["user-agent"] == "tldw-mcp-web-fetch/1.0"  # nosec B101


async def test_httpx_client_byte_cap_stops_the_download(_http_egress: None) -> None:
    """max_bytes truncates the body and stops reading the stream."""
    import httpx

    chunks = [b"x" * 65536] * 20
    yielded: list[int] = []

    class _CountingStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for chunk in chunks:
                yielded.append(len(chunk))
                yield chunk

    client = HttpxWebFetchClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, request=request, headers={"Content-Type": "text/plain"}, stream=_CountingStream()
            )
        )
    )

    response = await client.fetch(_PUBLIC_URL, timeout_seconds=5, max_bytes=1000, user_agent="test")

    assert response.truncated is True  # nosec B101
    assert response.body == b"x" * 1000  # nosec B101
    assert len(yielded) < len(chunks)  # nosec B101


async def test_httpx_client_timeout_bounds_a_silent_server(_http_egress: None) -> None:
    """timeout_seconds bounds the wait for body bytes (http_client's stream timeouts)."""
    import asyncio

    import httpx

    class _SilentStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.Event().wait()
            yield b""

    client = HttpxWebFetchClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, request=request, headers={"Content-Type": "text/html"}, stream=_SilentStream()
            )
        )
    )

    with pytest.raises(NetworkError, match="StreamTimeout"):
        await asyncio.wait_for(
            client.fetch(_PUBLIC_URL, timeout_seconds=0.2, max_bytes=1000, user_agent="test"),
            timeout=2,
        )
