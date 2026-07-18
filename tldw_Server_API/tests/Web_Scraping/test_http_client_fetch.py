import asyncio
import types
from contextlib import asynccontextmanager

import pytest

import tldw_Server_API.app.core.http_client as hc


class DummyResp:
    def __init__(
        self,
        url: str,
        headers: dict,
        *,
        status_code: int = 200,
        text: str = "<html><body>ok</body></html>",
    ):
        self.status_code = status_code
        self.headers = headers
        self.text = text
        self.url = url


class DummyClient:
    def __init__(self, timeout=None, trust_env=None, proxies=None):
        self.last_headers = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def request(self, method, url, headers=None, cookies=None, follow_redirects=None):
        self.last_headers = headers or {}
        # echo headers back in response for assertions
        return DummyResp(url, self.last_headers)


_DUMMY_HTTP_BACKEND = types.SimpleNamespace(Client=DummyClient)


async def _allow_async_egress(*_args, **_kwargs):
    return None


class _CapturedTracingManager:
    def __init__(self):
        self.spans = []
        self.attributes = []
        self.events = []

    @asynccontextmanager
    async def async_span(self, name, *, attributes=None, **_kwargs):
        self.spans.append((name, dict(attributes or {})))
        yield None

    def set_attributes(self, attributes):
        self.attributes.append(dict(attributes))

    def add_event(self, name, attributes=None):
        self.events.append((name, dict(attributes or {})))


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["httpx", "aiohttp"])
async def test_afetch_timeout_errors_have_stable_classification(monkeypatch, backend):
    monkeypatch.setattr(hc, "_avalidate_egress_or_raise", _allow_async_egress)
    monkeypatch.setattr(hc, "_is_aiohttp_client", lambda _client: backend == "aiohttp")

    async def raise_timeout(**_kwargs):
        if backend == "httpx":
            raise hc.httpx.ReadTimeout(
                "httpx-timeout-secret",
                request=hc.httpx.Request("GET", "https://example.com/"),
            )
        raise hc.aiohttp.ServerTimeoutError("aiohttp-timeout-secret")

    io_name = "_httpx_arequest_io" if backend == "httpx" else "_aiohttp_request_io"
    monkeypatch.setattr(hc, io_name, raise_timeout)

    with pytest.raises(hc.NetworkError) as raised:
        await hc.afetch(
            method="GET",
            url="https://example.com/timeout",
            client=object(),
            retry=hc.RetryPolicy(attempts=1),
        )

    assert raised.value.classification == "timeout"
    assert "secret" not in str(raised.value)


@pytest.mark.asyncio
async def test_afetch_probe_sensitive_observability_excludes_all_secrets(monkeypatch):
    tracing = _CapturedTracingManager()
    logs = []
    monkeypatch.setattr(hc, "_avalidate_egress_or_raise", _allow_async_egress)
    monkeypatch.setattr(hc, "_is_aiohttp_client", lambda _client: False)
    monkeypatch.setattr(hc, "get_tracing_manager", lambda: tracing)

    async def raise_timeout(**_kwargs):
        raise hc.httpx.ReadTimeout(
            "exception-secret-2f",
            request=hc.httpx.Request("GET", "https://example.com/"),
        )

    monkeypatch.setattr(hc, "_httpx_arequest_io", raise_timeout)
    sink_id = hc.logger.add(
        lambda message: logs.append(str(message)),
        level="DEBUG",
        format="{message}|{extra}",
    )
    try:
        with pytest.raises(hc.NetworkError):
            await hc.afetch(
                method="GET",
                url=(
                    "https://user-secret-9c:password-secret-0d@example.com/"
                    "path-secret-7a?token=query-secret-8b#fragment-secret-3g"
                ),
                client=object(),
                headers={"Authorization": "header-secret-1e"},
                retry=hc.RetryPolicy(attempts=1),
                sensitive_observability=True,
            )
    finally:
        hc.logger.remove(sink_id)

    captured = repr((logs, tracing.spans, tracing.attributes, tracing.events))
    for secret in (
        "path-secret-7a",
        "query-secret-8b",
        "user-secret-9c",
        "password-secret-0d",
        "header-secret-1e",
        "exception-secret-2f",
        "fragment-secret-3g",
    ):
        assert secret not in captured
    assert "sensitive-endpoint.invalid" in captured


@pytest.mark.asyncio
async def test_afetch_sensitive_observability_scope_is_concurrent_and_resets(monkeypatch):
    entered = 0
    release = asyncio.Event()

    class RecordingAdapter:
        async def arequest(self, *, url, **_kwargs):
            nonlocal entered
            entered += 1
            if entered == 2:
                release.set()
            await release.wait()
            return hc._sanitize_url_for_logs(url)

    monkeypatch.setattr(hc, "_get_transport_adapter", lambda _name: RecordingAdapter())

    safe_result, default_result = await asyncio.gather(
        hc.afetch(
            method="GET",
            url="https://example.com/safe-path-secret",
            sensitive_observability=True,
        ),
        hc.afetch(
            method="GET",
            url="https://example.com/default-path-visible",
        ),
    )

    assert safe_result == "https://example.com"
    assert default_result == "https://example.com/default-path-visible"
    assert hc._sanitize_url_for_logs("https://example.com/after-scope") == ("https://example.com/after-scope")


@pytest.mark.asyncio
async def test_afetch_sensitive_observability_parse_failure_never_returns_raw_url(monkeypatch):
    class RecordingAdapter:
        async def arequest(self, *, url, **_kwargs):
            return hc._sanitize_url_for_logs(url)

    monkeypatch.setattr(hc, "_get_transport_adapter", lambda _name: RecordingAdapter())

    result = await hc.afetch(
        method="GET",
        url="malformed-path-secret?query-secret",
        sensitive_observability=True,
    )

    assert result == ""


def test_httpx_fetch_sanitizes_accept_encoding_and_backend_label(monkeypatch):
    # allow egress
    monkeypatch.setattr(hc, "_is_url_allowed", lambda url: True)
    # swap client constructor via resolver
    monkeypatch.setattr(hc, "_resolve_httpx", lambda: _DUMMY_HTTP_BACKEND)

    headers = {"Accept-Encoding": "gzip, deflate, br, zstd"}
    resp = hc.fetch("https://example.com/", headers=headers, backend="httpx")
    assert resp["backend"] == "httpx"  # nosec B101
    # zstd should be removed for httpx path
    assert "zstd" not in resp["headers"].get("Accept-Encoding", "").lower()  # nosec B101


def test_httpx_fetch_accept_encoding_case_and_params(monkeypatch):
    # allow egress
    monkeypatch.setattr(hc, "_is_url_allowed", lambda url: True)
    # swap client constructor via resolver
    monkeypatch.setattr(hc, "_resolve_httpx", lambda: _DUMMY_HTTP_BACKEND)

    # Lower-case header key with parameterized zstd; should be dropped and canonicalized
    headers = {"accept-encoding": "gzip, zstd;q=0.9, br"}
    resp = hc.fetch("https://example.com/", headers=headers, backend="httpx")

    # Original exact lower-case key should not remain; canonical should be present
    assert "accept-encoding" not in resp["headers"]  # nosec B101
    enc = resp["headers"].get("Accept-Encoding", "")
    assert "zstd" not in enc.lower()  # nosec B101
    assert "gzip" in enc.lower() and "br" in enc.lower()  # nosec B101


def test_httpx_fetch_accept_encoding_all_removed(monkeypatch):
    # allow egress
    monkeypatch.setattr(hc, "_is_url_allowed", lambda url: True)
    # swap client constructor via resolver
    monkeypatch.setattr(hc, "_resolve_httpx", lambda: _DUMMY_HTTP_BACKEND)

    headers = {"Accept-Encoding": "zstd;q=0.9, zstd"}
    resp = hc.fetch("https://example.com/", headers=headers, backend="httpx")

    # Accept-Encoding should be removed entirely if no tokens remain
    keys_lower = {k.lower() for k in resp["headers"]}
    assert "accept-encoding" not in keys_lower  # nosec B101


def test_requests_fetch_sanitizes_accept_encoding(monkeypatch):
    # allow egress
    monkeypatch.setattr(hc, "_is_url_allowed", lambda url: True)
    # swap client constructor via resolver
    monkeypatch.setattr(hc, "_resolve_httpx", lambda: _DUMMY_HTTP_BACKEND)

    headers = {"Accept-Encoding": "gzip, deflate, br, zstd"}
    resp = hc.fetch("https://example.com/", headers=headers, backend="requests")

    enc = resp["headers"].get("Accept-Encoding", "").lower()
    assert "zstd" not in enc  # nosec B101
    assert "br" not in enc  # nosec B101


def test_fetch_egress_denied_raises(monkeypatch):
    monkeypatch.setattr(hc, "_is_url_allowed", lambda url: False)
    with pytest.raises(ValueError):
        hc.fetch("https://example.com/")


def test_curl_fetch_uses_curl_session_and_preserves_encodings(monkeypatch):
    monkeypatch.setattr(hc, "_is_url_allowed", lambda url: True)
    monkeypatch.setattr(hc, "_validate_proxies_or_raise", lambda proxies: None)

    calls = {}

    class DummyCurlSession:
        def __init__(self, impersonate=None):
            calls["impersonate"] = impersonate

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, **kwargs):
            calls["url"] = url
            calls["kwargs"] = kwargs
            return DummyResp(url, kwargs.get("headers", {}))

    monkeypatch.setattr(hc, "_resolve_curl_session", lambda: DummyCurlSession)

    headers = {"Accept-Encoding": "gzip, deflate, br, zstd"}
    resp = hc.fetch(
        "https://example.com/",
        headers=headers,
        backend="curl",
        impersonate="chrome120",
    )

    assert resp["backend"] == "curl"  # nosec B101
    assert calls["impersonate"] == "chrome120"  # nosec B101
    assert "zstd" in resp["headers"].get("Accept-Encoding", "").lower()  # nosec B101


def test_curl_fetch_follows_same_host_redirects_under_policy(monkeypatch):
    monkeypatch.setattr(hc, "_is_url_allowed", lambda url: True)
    monkeypatch.setattr(hc, "_validate_proxies_or_raise", lambda proxies: None)

    calls = []

    class DummyCurlSession:
        def __init__(self, impersonate=None):
            self.impersonate = impersonate

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, **kwargs):
            calls.append((url, kwargs))
            if url == "https://example.com/start":
                return DummyResp(
                    url,
                    {"Location": "/final"},
                    status_code=302,
                    text="",
                )
            return DummyResp("https://example.com/final", {}, status_code=200, text="<html>final</html>")

    monkeypatch.setattr(hc, "_resolve_curl_session", lambda: DummyCurlSession)

    resp = hc.fetch(
        "https://example.com/start",
        headers={"Accept-Encoding": "gzip, br, zstd"},
        backend="curl",
        follow_redirects=True,
    )

    assert [url for url, _ in calls] == [  # nosec B101
        "https://example.com/start",
        "https://example.com/final",
    ]
    assert calls[0][1]["allow_redirects"] is False  # nosec B101
    assert calls[1][1]["allow_redirects"] is False  # nosec B101
    assert resp["status"] == 200  # nosec B101
    assert resp["url"] == "https://example.com/final"  # nosec B101


def test_curl_fetch_follows_same_host_redirects_without_httpx(monkeypatch):
    monkeypatch.setattr(hc, "_is_url_allowed", lambda url: True)
    monkeypatch.setattr(hc, "_validate_proxies_or_raise", lambda proxies: None)
    monkeypatch.setattr(hc, "_resolve_httpx", lambda: None)

    calls = []

    class DummyCurlSession:
        def __init__(self, impersonate=None):
            self.impersonate = impersonate

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, **kwargs):
            calls.append((url, kwargs))
            if url == "https://example.com/start":
                return DummyResp(
                    url,
                    {"Location": "/final"},
                    status_code=302,
                    text="",
                )
            return DummyResp("https://example.com/final", {}, status_code=200, text="<html>final</html>")

    monkeypatch.setattr(hc, "_resolve_curl_session", lambda: DummyCurlSession)

    resp = hc.fetch(
        "https://example.com/start",
        backend="curl",
        follow_redirects=True,
    )

    assert [url for url, _ in calls] == [  # nosec B101
        "https://example.com/start",
        "https://example.com/final",
    ]
    assert calls[0][1]["allow_redirects"] is False  # nosec B101
    assert calls[1][1]["allow_redirects"] is False  # nosec B101
    assert resp["status"] == 200  # nosec B101
    assert resp["url"] == "https://example.com/final"  # nosec B101


def test_curl_fetch_cross_host_redirect_strips_origin_bound_state(monkeypatch):
    monkeypatch.setattr(hc, "_is_url_allowed", lambda url: True)
    monkeypatch.setattr(hc, "_validate_proxies_or_raise", lambda proxies: None)
    monkeypatch.setenv("HTTP_ALLOW_CROSS_HOST_REDIRECTS", "true")

    calls = []

    class DummyCurlSession:
        def __init__(self, impersonate=None):
            self.impersonate = impersonate

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, **kwargs):
            calls.append(
                {
                    "url": url,
                    "headers": dict(kwargs.get("headers") or {}),
                    "cookies": dict(kwargs.get("cookies") or {}),
                }
            )
            if url == "https://source.example/start":
                return DummyResp(
                    url,
                    {"Location": "https://target.example/final"},
                    status_code=302,
                    text="",
                )
            return DummyResp(url, {}, status_code=200, text="<html>final</html>")

    monkeypatch.setattr(hc, "_resolve_curl_session", lambda: DummyCurlSession)

    resp = hc.fetch(
        "https://source.example/start",
        backend="curl",
        follow_redirects=True,
        headers={
            "Authorization": "Bearer top-secret",
            "Proxy-Authorization": "Basic proxy-secret",
            "X-Custom-Trace": "origin-bound",
            "User-Agent": "tldw-test-agent/1.0",
            "Accept-Encoding": "gzip, br",
        },
        cookies={"session": "origin-cookie"},
    )

    assert [c["url"] for c in calls] == [  # nosec B101
        "https://source.example/start",
        "https://target.example/final",
    ]
    second_hop_headers = {k.lower(): v for k, v in calls[1]["headers"].items()}
    assert "authorization" not in second_hop_headers  # nosec B101
    assert "proxy-authorization" not in second_hop_headers  # nosec B101
    assert "x-custom-trace" not in second_hop_headers  # nosec B101
    assert calls[1]["cookies"] == {}  # nosec B101
    assert second_hop_headers.get("user-agent") == "tldw-test-agent/1.0"  # nosec B101
    assert second_hop_headers.get("accept-encoding") == "gzip, br"  # nosec B101
    assert resp["status"] == 200  # nosec B101
    assert resp["url"] == "https://target.example/final"  # nosec B101


def test_httpx_fetch_cross_host_redirect_strips_origin_bound_state(monkeypatch):
    monkeypatch.setattr(hc, "_is_url_allowed", lambda url: True)
    monkeypatch.setattr(hc, "_validate_proxies_or_raise", lambda proxies: None)
    monkeypatch.setenv("HTTP_ALLOW_CROSS_HOST_REDIRECTS", "true")

    calls = []

    class DummyRedirectClient:
        def __init__(self, timeout=None, trust_env=None, proxies=None):
            self.timeout = timeout
            self.trust_env = trust_env
            self.proxies = proxies

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def request(self, method, url, headers=None, cookies=None, follow_redirects=None):
            calls.append(
                {
                    "url": url,
                    "headers": dict(headers or {}),
                    "cookies": dict(cookies or {}),
                    "follow_redirects": follow_redirects,
                }
            )
            if url == "https://source.example/start":
                return DummyResp(
                    url,
                    {"Location": "https://target.example/final"},
                    status_code=302,
                    text="",
                )
            return DummyResp(url, {}, status_code=200, text="<html>final</html>")

    monkeypatch.setattr(hc, "_resolve_httpx", lambda: types.SimpleNamespace(Client=DummyRedirectClient))

    resp = hc.fetch(
        "https://source.example/start",
        backend="httpx",
        follow_redirects=True,
        headers={
            "Authorization": "Bearer top-secret",
            "Proxy-Authorization": "Basic proxy-secret",
            "X-Custom-Trace": "origin-bound",
            "User-Agent": "tldw-test-agent/1.0",
            "Accept-Encoding": "gzip, br",
        },
        cookies={"session": "origin-cookie"},
    )

    assert [c["url"] for c in calls] == [  # nosec B101
        "https://source.example/start",
        "https://target.example/final",
    ]
    second_hop_headers = {k.lower(): v for k, v in calls[1]["headers"].items()}
    assert "authorization" not in second_hop_headers  # nosec B101
    assert "proxy-authorization" not in second_hop_headers  # nosec B101
    assert "x-custom-trace" not in second_hop_headers  # nosec B101
    assert calls[1]["cookies"] == {}  # nosec B101
    assert second_hop_headers.get("user-agent") == "tldw-test-agent/1.0"  # nosec B101
    assert second_hop_headers.get("accept-encoding") == "gzip, br"  # nosec B101
    assert calls[0]["follow_redirects"] is False  # nosec B101
    assert calls[1]["follow_redirects"] is False  # nosec B101
    assert resp["status"] == 200  # nosec B101
    assert resp["url"] == "https://target.example/final"  # nosec B101


def test_curl_fetch_same_host_downgrade_strips_state_and_clears_cookie_jar(monkeypatch):
    monkeypatch.setattr(hc, "_is_url_allowed", lambda url: True)
    monkeypatch.setattr(hc, "_validate_proxies_or_raise", lambda proxies: None)
    monkeypatch.setenv("HTTP_ALLOW_SCHEME_DOWNGRADE", "true")

    calls = []

    class DummyCurlSession:
        def __init__(self, impersonate=None):
            self.impersonate = impersonate
            self.cookie_jar = {}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, **kwargs):
            request_cookies = dict(kwargs.get("cookies") or {})
            merged_cookies = {**request_cookies, **self.cookie_jar}
            calls.append(
                {
                    "url": url,
                    "headers": dict(kwargs.get("headers") or {}),
                    "cookies": merged_cookies,
                }
            )
            if url == "https://example.com/start":
                self.cookie_jar["challenge"] = "passed"
                return DummyResp(
                    url,
                    {"Location": "http://example.com/final"},
                    status_code=302,
                    text="",
                )
            return DummyResp(url, {}, status_code=200, text="<html>final</html>")

    monkeypatch.setattr(hc, "_resolve_curl_session", lambda: DummyCurlSession)

    resp = hc.fetch(
        "https://example.com/start",
        backend="curl",
        follow_redirects=True,
        headers={
            "Authorization": "Bearer top-secret",
            "X-Custom-Trace": "origin-bound",
            "User-Agent": "tldw-test-agent/1.0",
            "Accept-Encoding": "gzip, br",
        },
        cookies={"session": "origin-cookie"},
    )

    assert [c["url"] for c in calls] == [  # nosec B101
        "https://example.com/start",
        "http://example.com/final",
    ]
    second_hop_headers = {k.lower(): v for k, v in calls[1]["headers"].items()}
    assert "authorization" not in second_hop_headers  # nosec B101
    assert "x-custom-trace" not in second_hop_headers  # nosec B101
    assert second_hop_headers.get("user-agent") == "tldw-test-agent/1.0"  # nosec B101
    assert second_hop_headers.get("accept-encoding") == "gzip, br"  # nosec B101
    assert calls[1]["cookies"] == {}  # nosec B101
    assert resp["status"] == 200  # nosec B101
    assert resp["url"] == "http://example.com/final"  # nosec B101


def test_httpx_fetch_port_change_strips_state_and_clears_cookie_jar(monkeypatch):
    monkeypatch.setattr(hc, "_is_url_allowed", lambda url: True)
    monkeypatch.setattr(hc, "_validate_proxies_or_raise", lambda proxies: None)
    monkeypatch.setenv("HTTP_ALLOW_CROSS_HOST_REDIRECTS", "true")

    calls = []

    class DummyRedirectClient:
        def __init__(self, timeout=None, trust_env=None, proxies=None):
            self.timeout = timeout
            self.trust_env = trust_env
            self.proxies = proxies
            self.cookies = {}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def request(self, method, url, headers=None, cookies=None, follow_redirects=None):
            request_cookies = dict(cookies or {})
            merged_cookies = {**request_cookies, **self.cookies}
            calls.append(
                {
                    "url": url,
                    "headers": dict(headers or {}),
                    "cookies": merged_cookies,
                    "follow_redirects": follow_redirects,
                }
            )
            if url == "https://example.com/start":
                self.cookies["challenge"] = "passed"
                return DummyResp(
                    url,
                    {"Location": "https://example.com:8443/final"},
                    status_code=302,
                    text="",
                )
            return DummyResp(url, {}, status_code=200, text="<html>final</html>")

    monkeypatch.setattr(hc, "_resolve_httpx", lambda: types.SimpleNamespace(Client=DummyRedirectClient))

    resp = hc.fetch(
        "https://example.com/start",
        backend="httpx",
        follow_redirects=True,
        headers={
            "Authorization": "Bearer top-secret",
            "Proxy-Authorization": "Basic proxy-secret",
            "X-Custom-Trace": "origin-bound",
            "User-Agent": "tldw-test-agent/1.0",
            "Accept-Encoding": "gzip, br",
        },
        cookies={"session": "origin-cookie"},
    )

    assert [c["url"] for c in calls] == [  # nosec B101
        "https://example.com/start",
        "https://example.com:8443/final",
    ]
    second_hop_headers = {k.lower(): v for k, v in calls[1]["headers"].items()}
    assert "authorization" not in second_hop_headers  # nosec B101
    assert "proxy-authorization" not in second_hop_headers  # nosec B101
    assert "x-custom-trace" not in second_hop_headers  # nosec B101
    assert second_hop_headers.get("user-agent") == "tldw-test-agent/1.0"  # nosec B101
    assert second_hop_headers.get("accept-encoding") == "gzip, br"  # nosec B101
    assert calls[1]["cookies"] == {}  # nosec B101
    assert resp["status"] == 200  # nosec B101
    assert resp["url"] == "https://example.com:8443/final"  # nosec B101


def test_curl_fetch_denies_blocked_redirect_hop(monkeypatch):
    allowed = {
        "https://example.com/start": True,
        "https://example.com/final": False,
    }
    monkeypatch.setattr(hc, "_is_url_allowed", lambda url: allowed.get(url, True))
    monkeypatch.setattr(hc, "_validate_proxies_or_raise", lambda proxies: None)

    calls = []

    class DummyCurlSession:
        def __init__(self, impersonate=None):
            self.impersonate = impersonate

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, **kwargs):
            calls.append((url, kwargs))
            return DummyResp(
                url,
                {"Location": "/final"},
                status_code=302,
                text="",
            )

    monkeypatch.setattr(hc, "_resolve_curl_session", lambda: DummyCurlSession)

    with pytest.raises(ValueError, match="Egress denied for URL"):
        hc.fetch("https://example.com/start", backend="curl", follow_redirects=True)

    assert [url for url, _ in calls] == ["https://example.com/start"]  # nosec B101
    assert calls[0][1]["allow_redirects"] is False  # nosec B101


def test_curl_fetch_preserves_redirect_established_cookie(monkeypatch):
    monkeypatch.setattr(hc, "_is_url_allowed", lambda url: True)
    monkeypatch.setattr(hc, "_validate_proxies_or_raise", lambda proxies: None)

    calls = []

    class DummyCurlSession:
        def __init__(self, impersonate=None):
            self.impersonate = impersonate
            self.cookie_jar = {}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, **kwargs):
            request_cookies = dict(kwargs.get("cookies") or {})
            merged_cookies = {**request_cookies, **self.cookie_jar}
            calls.append((url, merged_cookies))

            if url == "https://example.com/start":
                # Simulate session cookie set during redirect response.
                self.cookie_jar["challenge"] = "passed"
                return DummyResp(
                    url,
                    {"Location": "/final", "Set-Cookie": "challenge=passed; Path=/"},
                    status_code=302,
                    text="",
                )

            if url == "https://example.com/final" and merged_cookies.get("challenge") == "passed":
                return DummyResp(url, {}, status_code=200, text="<html>ok</html>")

            return DummyResp(url, {}, status_code=403, text="<html>missing-cookie</html>")

    monkeypatch.setattr(hc, "_resolve_curl_session", lambda: DummyCurlSession)

    resp = hc.fetch(
        "https://example.com/start",
        backend="curl",
        follow_redirects=True,
    )

    assert [url for url, _ in calls] == [  # nosec B101
        "https://example.com/start",
        "https://example.com/final",
    ]
    assert calls[1][1].get("challenge") == "passed"  # nosec B101
    assert resp["status"] == 200  # nosec B101
    assert resp["url"] == "https://example.com/final"  # nosec B101
