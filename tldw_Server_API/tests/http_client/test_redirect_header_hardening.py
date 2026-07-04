"""Cross-origin redirect credential hardening for the central HTTP client.

Regression coverage for the PR #2604 review finding (qodo "Option B"): the
manual redirect loops in fetch/afetch reused the caller's headers across
hops, so a cross-origin redirect would resend Authorization / x-api-key /
cookies to the redirect target. The client now strips sensitive headers and
drops cookies whenever a hop leaves the original origin.
"""

import pytest
from hypothesis import given, settings as hyp_settings, strategies as st

from tldw_Server_API.app.core.http_client import (
    SENSITIVE_REDIRECT_HEADERS,
    _cookies_for_hop,
    _is_cross_origin,
    _strip_sensitive_headers_for_cross_origin,
    _url_origin,
)

pytestmark = pytest.mark.unit


def _has_httpx() -> bool:
    try:
        import httpx  # noqa: F401

        return True
    except Exception:
        return False


requires_httpx = pytest.mark.skipif(not _has_httpx(), reason="httpx not installed")

# TEST-NET literal IPs: public-range, pass egress checks without DNS, and
# MockTransport intercepts before any real socket is opened.
ORIGIN_A = "http://93.184.216.34"
ORIGIN_B = "http://203.0.113.7"

SECRET_HEADERS = {
    "Authorization": "Bearer secret-token",
    "X-Api-Key": "sk-secret",
    "Cookie": "session=abc",
    "X-Custom": "keep-me",
}


class TestOriginHelpers:
    def test_default_ports_normalized(self) -> None:
        assert _url_origin("http://a.example") == ("http", "a.example", 80)
        assert _url_origin("http://a.example:80/x") == ("http", "a.example", 80)
        assert _url_origin("https://a.example") == ("https", "a.example", 443)
        assert not _is_cross_origin("http://a.example", "http://a.example:80/path")
        assert not _is_cross_origin("https://a.example/x", "https://a.example:443/y")

    def test_scheme_host_port_changes_are_cross_origin(self) -> None:
        assert _is_cross_origin("https://a.example", "http://a.example")
        assert _is_cross_origin("https://a.example", "https://b.example")
        assert _is_cross_origin("https://a.example", "https://a.example:8443")
        assert _is_cross_origin("https://a.example", "https://sub.a.example")

    def test_host_comparison_is_case_insensitive(self) -> None:
        assert not _is_cross_origin("https://A.Example/x", "https://a.example/y")

    def test_unparseable_urls_fail_closed(self) -> None:
        assert _is_cross_origin("https://a.example", "not a url")
        assert _is_cross_origin("", "https://a.example")

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
        netloc = f"{host}:{port}" if port is not None else host
        url = f"{scheme}://{netloc}{path}"
        assert not _is_cross_origin(url, url)


class TestHeaderStripping:
    def test_same_origin_keeps_everything(self) -> None:
        result = _strip_sensitive_headers_for_cross_origin(
            dict(SECRET_HEADERS), original_url=f"{ORIGIN_A}/a", target_url=f"{ORIGIN_A}/b"
        )
        assert result == SECRET_HEADERS

    def test_cross_origin_strips_sensitive_case_insensitively(self) -> None:
        headers = {"AUTHORIZATION": "x", "x-API-key": "y", "cookie": "z", "X-Custom": "keep"}
        result = _strip_sensitive_headers_for_cross_origin(
            headers, original_url=ORIGIN_A, target_url=ORIGIN_B
        )
        assert result == {"X-Custom": "keep"}

    def test_every_documented_sensitive_header_is_stripped(self) -> None:
        headers = {name: "secret" for name in SENSITIVE_REDIRECT_HEADERS}
        headers["x-trace"] = "keep"
        result = _strip_sensitive_headers_for_cross_origin(
            headers, original_url=ORIGIN_A, target_url=ORIGIN_B
        )
        assert result == {"x-trace": "keep"}

    def test_none_headers_pass_through(self) -> None:
        assert (
            _strip_sensitive_headers_for_cross_origin(
                None, original_url=ORIGIN_A, target_url=ORIGIN_B
            )
            is None
        )

    def test_cookies_dropped_only_cross_origin(self) -> None:
        cookies = {"session": "abc"}
        assert (
            _cookies_for_hop(cookies, original_url=ORIGIN_A, target_url=f"{ORIGIN_A}/x")
            == cookies
        )
        assert _cookies_for_hop(cookies, original_url=ORIGIN_A, target_url=ORIGIN_B) is None
        assert _cookies_for_hop(None, original_url=ORIGIN_A, target_url=ORIGIN_B) is None


@requires_httpx
class TestRedirectFlows:
    """End-to-end through the real redirect loops via httpx.MockTransport."""

    @staticmethod
    def _handler(seen: dict[str, dict[str, str]]):
        import httpx

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

    @pytest.mark.asyncio
    async def test_afetch_strips_credentials_on_cross_origin_redirect(self) -> None:
        import httpx

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
        target_headers = seen["203.0.113.7"]
        # original host got the credentials
        assert origin_headers.get("authorization") == "Bearer secret-token"
        assert origin_headers.get("x-api-key") == "sk-secret"
        # redirect target must not
        for name in ("authorization", "x-api-key", "cookie"):
            assert name not in target_headers, f"{name} leaked to redirect target"
        # non-sensitive headers still follow the redirect
        assert target_headers.get("x-custom") == "keep-me"

    @pytest.mark.asyncio
    async def test_afetch_keeps_credentials_on_same_origin_redirect(self) -> None:
        import httpx

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

    def test_sync_fetch_strips_credentials_on_cross_origin_redirect(self) -> None:
        import httpx

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

        target_headers = seen["203.0.113.7"]
        for name in ("authorization", "x-api-key", "cookie"):
            assert name not in target_headers, f"{name} leaked to redirect target"
        assert target_headers.get("x-custom") == "keep-me"
