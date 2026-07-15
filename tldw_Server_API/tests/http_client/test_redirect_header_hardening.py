"""Cross-origin redirect credential hardening for the central HTTP client.

Regression coverage for the PR #2604 review finding (qodo "Option B"): the
manual redirect loops in fetch/afetch (and the SSE streaming loops) reused
the caller's headers across hops, so a cross-origin redirect would resend
Authorization / x-api-key / cookies to the redirect target. The client now
strips sensitive headers, drops explicit cookies, and clears ambient client
cookie-jar state whenever a hop leaves the original origin.
"""

from typing import Callable

import httpx
import pytest
from hypothesis import given, settings as hyp_settings, strategies as st

from tldw_Server_API.app.core.http_client import (
    SENSITIVE_REDIRECT_HEADERS,
    _clear_client_cookie_jar,
    _cookies_for_hop,
    _is_cross_origin,
    _strip_sensitive_headers_for_cross_origin,
    _url_origin,
)

pytestmark = pytest.mark.unit

# Public hosts already allowlisted in CI; MockTransport intercepts before any
# real socket is opened.
ORIGIN_A = "http://93.184.216.34"
ORIGIN_B = "http://example.com"

SECRET_HEADERS = {
    "Authorization": "Bearer secret-token",
    "X-Api-Key": "sk-secret",
    "Cookie": "session=abc",
    "X-Custom": "keep-me",
}


class TestOriginHelpers:
    """Unit coverage for the origin parsing/comparison helpers."""

    def test_default_ports_normalized(self) -> None:
        """Explicit default ports compare equal to implicit ones."""
        assert _url_origin("http://a.example") == ("http", "a.example", 80)
        assert _url_origin("http://a.example:80/x") == ("http", "a.example", 80)
        assert _url_origin("https://a.example") == ("https", "a.example", 443)
        assert not _is_cross_origin("http://a.example", "http://a.example:80/path")
        assert not _is_cross_origin("https://a.example/x", "https://a.example:443/y")

    def test_scheme_host_port_changes_are_cross_origin(self) -> None:
        """Any change to scheme, host (incl. subdomain), or port is cross-origin."""
        assert _is_cross_origin("https://a.example", "http://a.example")
        assert _is_cross_origin("https://a.example", "https://b.example")
        assert _is_cross_origin("https://a.example", "https://a.example:8443")
        assert _is_cross_origin("https://a.example", "https://sub.a.example")

    def test_host_comparison_is_case_insensitive(self) -> None:
        """Host names are compared lowercased."""
        assert not _is_cross_origin("https://A.Example/x", "https://a.example/y")

    def test_unparseable_urls_fail_closed(self) -> None:
        """Junk URLs are treated as cross-origin, never same-origin."""
        assert _is_cross_origin("https://a.example", "not a url")
        assert _is_cross_origin("", "https://a.example")

    def test_invalid_ports_and_malformed_hosts_fail_closed(self) -> None:
        """Malformed ports / IPv6 brackets must not defeat the stripping.

        urlparse raises ValueError lazily when ``.port`` is accessed on
        these; ``_url_origin`` must swallow that and report ``None`` →
        cross-origin.
        """
        assert _url_origin("http://a.example:invalid-port/") is None
        assert _is_cross_origin("http://a.example:invalid-port/", "http://a.example/")
        assert _is_cross_origin("http://a.example/", "http://a.example:99999/")
        # malformed IPv6 bracket
        assert _is_cross_origin("http://a.example/", "http://[::1/x")
        # sensitive headers must be stripped for such targets, not sent
        result = _strip_sensitive_headers_for_cross_origin(
            dict(SECRET_HEADERS),
            original_url="http://a.example/",
            target_url="http://a.example:invalid-port/",
        )
        assert result == {"X-Custom": "keep-me"}

    @hyp_settings(max_examples=50, deadline=None)
    @given(
        scheme=st.sampled_from(["http", "https"]),
        host=st.from_regex(r"[a-z][a-z0-9-]{0,10}\.example", fullmatch=True),
        port=st.one_of(st.none(), st.integers(min_value=1, max_value=65535)),
        path=st.from_regex(r"/[a-z0-9/]{0,12}", fullmatch=True),
    )
    def test_url_is_never_cross_origin_with_itself(
        self, scheme: str, host: str, port: int | None, path: str
    ) -> None:
        """Property: a well-formed URL is always same-origin with itself."""
        netloc = f"{host}:{port}" if port is not None else host
        url = f"{scheme}://{netloc}{path}"
        assert not _is_cross_origin(url, url)


class TestHeaderStripping:
    """Unit coverage for header/cookie stripping decisions."""

    def test_same_origin_keeps_everything(self) -> None:
        """Same-origin hops pass headers through untouched."""
        result = _strip_sensitive_headers_for_cross_origin(
            dict(SECRET_HEADERS), original_url=f"{ORIGIN_A}/a", target_url=f"{ORIGIN_A}/b"
        )
        assert result == SECRET_HEADERS

    def test_cross_origin_strips_sensitive_case_insensitively(self) -> None:
        """Header-name matching ignores case."""
        headers = {"AUTHORIZATION": "x", "x-API-key": "y", "cookie": "z", "X-Custom": "keep"}
        result = _strip_sensitive_headers_for_cross_origin(
            headers, original_url=ORIGIN_A, target_url=ORIGIN_B
        )
        assert result == {"X-Custom": "keep"}

    def test_every_documented_sensitive_header_is_stripped(self) -> None:
        """The full SENSITIVE_REDIRECT_HEADERS set is enforced."""
        headers = {name: "secret" for name in SENSITIVE_REDIRECT_HEADERS}
        headers["x-trace"] = "keep"
        result = _strip_sensitive_headers_for_cross_origin(
            headers, original_url=ORIGIN_A, target_url=ORIGIN_B
        )
        assert result == {"x-trace": "keep"}

    def test_none_headers_pass_through(self) -> None:
        """``None`` headers stay ``None``."""
        assert (
            _strip_sensitive_headers_for_cross_origin(
                None, original_url=ORIGIN_A, target_url=ORIGIN_B
            )
            is None
        )

    def test_cookies_dropped_only_cross_origin(self) -> None:
        """Explicit cookies survive same-origin hops and die cross-origin."""
        cookies = {"session": "abc"}
        assert (
            _cookies_for_hop(cookies, original_url=ORIGIN_A, target_url=f"{ORIGIN_A}/x")
            == cookies
        )
        assert _cookies_for_hop(cookies, original_url=ORIGIN_A, target_url=ORIGIN_B) is None
        assert _cookies_for_hop(None, original_url=ORIGIN_A, target_url=ORIGIN_B) is None

    def test_clear_client_cookie_jar_handles_httpx_clients(self) -> None:
        """The jar-clear helper empties an httpx client's cookie store."""
        client = httpx.Client()
        try:
            client.cookies.set("session", "abc", domain="93.184.216.34")
            assert len(client.cookies) == 1
            _clear_client_cookie_jar(client)
            assert len(client.cookies) == 0
        finally:
            client.close()


class TestRedirectFlows:
    """End-to-end through the real redirect loops via httpx.MockTransport."""

    @staticmethod
    def _handler(
        seen: dict[str, dict[str, str]],
    ) -> Callable[[httpx.Request], httpx.Response]:
        """Build a MockTransport handler recording per-host request headers."""

        def handler(request: httpx.Request) -> httpx.Response:
            host = request.url.host
            seen[host] = dict(request.headers)
            if host == "93.184.216.34":
                if request.url.path == "/same-origin":
                    return httpx.Response(
                        302, request=request, headers={"Location": f"{ORIGIN_A}/landed"}
                    )
                if request.url.path == "/cross-origin":
                    return httpx.Response(
                        302, request=request, headers={"Location": f"{ORIGIN_B}/landed"}
                    )
            return httpx.Response(200, request=request, json={"host": host})

        return handler

    async def test_afetch_strips_credentials_on_cross_origin_redirect(self) -> None:
        """The async loop must not resend credentials to a different origin."""
        from tldw_Server_API.app.core.http_client import afetch, create_async_client

        seen: dict[str, dict[str, str]] = {}
        client = create_async_client(transport=httpx.MockTransport(self._handler(seen)))
        try:
            resp = await afetch(
                method="GET",
                url=f"{ORIGIN_A}/cross-origin",
                client=client,
                headers=dict(SECRET_HEADERS),
            )
            assert resp.status_code == 200
        finally:
            await client.aclose()

        origin_headers = seen["93.184.216.34"]
        target_headers = seen["example.com"]
        # original host got the credentials
        assert origin_headers.get("authorization") == "Bearer secret-token"
        assert origin_headers.get("x-api-key") == "sk-secret"
        # redirect target must not
        for name in ("authorization", "x-api-key", "cookie"):
            assert name not in target_headers, f"{name} leaked to redirect target"
        # non-sensitive headers still follow the redirect
        assert target_headers.get("x-custom") == "keep-me"

    async def test_afetch_keeps_credentials_on_same_origin_redirect(self) -> None:
        """Same-origin redirects keep the caller's credentials intact."""
        from tldw_Server_API.app.core.http_client import afetch, create_async_client

        seen: dict[str, dict[str, str]] = {}
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            seen[request.url.path] = dict(request.headers)
            if request.url.path == "/same-origin":
                return httpx.Response(
                    302, request=request, headers={"Location": f"{ORIGIN_A}/landed"}
                )
            return httpx.Response(200, request=request, json={"ok": True})

        client = create_async_client(transport=httpx.MockTransport(handler))
        try:
            resp = await afetch(
                method="GET",
                url=f"{ORIGIN_A}/same-origin",
                client=client,
                headers=dict(SECRET_HEADERS),
            )
            assert resp.status_code == 200
        finally:
            await client.aclose()

        assert len(calls) == 2
        assert seen["/landed"].get("authorization") == "Bearer secret-token"
        assert seen["/landed"].get("x-api-key") == "sk-secret"

    async def test_scoped_afetch_revalidates_same_origin_redirect(self, monkeypatch) -> None:
        """A configured scope remains attached to each accepted path redirect."""
        from types import SimpleNamespace

        from tldw_Server_API.app.core import http_client as hc
        from tldw_Server_API.app.core.Security import egress as egress_mod
        from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope

        scope = ConfiguredEndpointScope.from_url(f"{ORIGIN_A}:11434")
        validations: list[tuple[str, object, object]] = []

        def fake_policy(url, *, configured_endpoint=None, pinned_resolved_ips=None, **_kwargs):
            validations.append((url, configured_endpoint, pinned_resolved_ips))
            return SimpleNamespace(
                allowed=configured_endpoint is scope,
                reason=None,
                resolved_ips=("93.184.216.34",),
                reason_code=None,
            )

        monkeypatch.setattr(egress_mod, "evaluate_url_policy", fake_policy)

        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            if request.url.path == "/start":
                return httpx.Response(302, request=request, headers={"Location": "/next"})
            return httpx.Response(200, request=request, text="ok")

        client = hc.create_async_client(transport=httpx.MockTransport(handler))
        try:
            response = await hc.afetch(
                method="GET",
                url=f"{ORIGIN_A}:11434/start",
                client=client,
                configured_endpoint=scope,
            )
        finally:
            await client.aclose()

        assert response.status_code == 200
        assert calls == [f"{ORIGIN_A}:11434/start", f"{ORIGIN_A}:11434/next"]
        assert len(validations) >= 3
        assert all(item[1] is scope for item in validations)
        assert all(item[2] == ("93.184.216.34",) for item in validations[1:])

    @pytest.mark.parametrize(
        "location",
        [
            "http://93.184.216.35:11434/blocked",
            "https://93.184.216.34:11434/blocked",
            "http://93.184.216.34:11435/blocked",
        ],
    )
    def test_scoped_fetch_denies_cross_origin_redirect_before_io(
        self, monkeypatch, location: str
    ) -> None:
        """A configured request never sends a redirected hop outside its exact origin."""
        from tldw_Server_API.app.core import http_client as hc
        from tldw_Server_API.app.core.exceptions import EgressPolicyError
        from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope

        scope = ConfiguredEndpointScope.from_url(f"{ORIGIN_A}:11434")
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            return httpx.Response(
                302,
                request=request,
                headers={"Location": location},
            )

        client = hc.create_client(transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(EgressPolicyError) as exc:
                hc.fetch(
                    method="GET",
                    url=f"{ORIGIN_A}:11434/start",
                    client=client,
                    configured_endpoint=scope,
                )
        finally:
            client.close()

        assert exc.value.reason_code == "origin_mismatch"
        assert calls == [f"{ORIGIN_A}:11434/start"]

    def test_sync_fetch_strips_credentials_on_cross_origin_redirect(self) -> None:
        """The sync loop applies the same stripping as the async loop."""
        from tldw_Server_API.app.core.http_client import create_client, fetch

        seen: dict[str, dict[str, str]] = {}
        client = create_client(transport=httpx.MockTransport(self._handler(seen)))
        try:
            resp = fetch(
                method="GET",
                url=f"{ORIGIN_A}/cross-origin",
                client=client,
                headers=dict(SECRET_HEADERS),
            )
            assert resp.status_code == 200
        finally:
            client.close()

        target_headers = seen["example.com"]
        for name in ("authorization", "x-api-key", "cookie"):
            assert name not in target_headers, f"{name} leaked to redirect target"
        assert target_headers.get("x-custom") == "keep-me"

    async def test_afetch_clears_ambient_cookie_jar_on_cross_origin_redirect(self) -> None:
        """Cookies accumulated in the client's jar are wiped at the origin boundary."""
        from tldw_Server_API.app.core.http_client import afetch, create_async_client

        seen: dict[str, dict[str, str]] = {}
        client = create_async_client(transport=httpx.MockTransport(self._handler(seen)))
        try:
            client.cookies.set("ambient", "jar-secret", domain="93.184.216.34")
            assert len(client.cookies) == 1
            resp = await afetch(
                method="GET",
                url=f"{ORIGIN_A}/cross-origin",
                client=client,
                headers={"X-Custom": "keep-me"},
            )
            assert resp.status_code == 200
            assert len(client.cookies) == 0, "cookie jar not cleared at origin boundary"
        finally:
            await client.aclose()
        assert "cookie" not in seen["example.com"]

    async def test_sse_stream_strips_credentials_on_cross_origin_redirect(self) -> None:
        """The SSE streaming redirect loop applies the same stripping."""
        from tldw_Server_API.app.core.http_client import _astream_sse_httpx, create_async_client

        seen: dict[str, dict[str, str]] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            host = request.url.host
            seen[host] = dict(request.headers)
            if host == "93.184.216.34":
                return httpx.Response(
                    302, request=request, headers={"Location": f"{ORIGIN_B}/stream"}
                )
            return httpx.Response(
                200,
                request=request,
                headers={"Content-Type": "text/event-stream"},
                text="data: hello\n\n",
            )

        client = create_async_client(transport=httpx.MockTransport(handler))
        try:
            events = [
                event
                async for event in _astream_sse_httpx(
                    url=f"{ORIGIN_A}/sse",
                    client=client,
                    headers=dict(SECRET_HEADERS),
                )
            ]
        finally:
            await client.aclose()

        assert events, "expected at least one SSE event through the redirect"
        assert seen["93.184.216.34"].get("authorization") == "Bearer secret-token"
        target_headers = seen["example.com"]
        for name in ("authorization", "x-api-key", "cookie"):
            assert name not in target_headers, f"{name} leaked to SSE redirect target"
        assert target_headers.get("x-custom") == "keep-me"
