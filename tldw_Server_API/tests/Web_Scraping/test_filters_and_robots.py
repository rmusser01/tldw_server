from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Web_Scraping.filters import (
    ContentTypeFilter,
    DomainFilter,
    RobotsFilter,
    URLPatternFilter,
)


@pytest.mark.unit
def test_domain_and_content_filters():
    df = DomainFilter(allowed={"example.com"}, blocked={"blocked.com"})
    cf = ContentTypeFilter()
    pf = URLPatternFilter(include_patterns=["/docs/"], exclude_patterns=["/admin/"])

    assert df.apply("https://example.com/docs/index.html") is True
    assert df.apply("https://sub.example.com/docs/") is True
    assert df.apply("https://blocked.com/") is False

    assert cf.apply("https://example.com/index.html") is True
    assert cf.apply("https://example.com/file.pdf") is False

    assert pf.apply("https://example.com/docs/page") is True
    assert pf.apply("https://example.com/admin/panel") is False
    # include gating if include patterns present
    assert pf.apply("https://example.com/other/path") is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_robots_filter_mocked(monkeypatch):
    # Force egress allow
    from tldw_Server_API.app.core.Web_Scraping import filters as filt_mod

    monkeypatch.setattr(
        filt_mod,
        "evaluate_url_policy",
        lambda url: SimpleNamespace(allowed=True),
        raising=False,
    )

    # Provide a deterministic robots.txt that disallows everything
    def fake_http_fetch(url, method="GET", backend="httpx", timeout=5.0, allow_redirects=True):
        return {
            "status": 200,
            "text": "User-agent: *\nDisallow: /\n",
        }

    monkeypatch.setattr(filt_mod, "http_fetch", fake_http_fetch, raising=False)

    rf = RobotsFilter(user_agent="TestBot/1.0", ttl_seconds=1)
    allowed = await rf.allowed("https://example.com/private")
    assert allowed is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_robots_filter_reports_egress_error_when_policy_evaluation_fails(monkeypatch):
    from tldw_Server_API.app.core.Security import egress as eg

    def raise_eval_error(_url):
        raise RuntimeError("boom")

    monkeypatch.setattr(eg, "evaluate_url_policy", raise_eval_error, raising=False)

    rf = RobotsFilter(user_agent="TestBot/1.0", ttl_seconds=1)
    result = await rf.check("https://example.com/private")

    assert result.allowed is False
    assert result.status == "egress_error"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_public_robots_redirect_denied_before_network(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc
    from tldw_Server_API.app.core.Security import egress
    from tldw_Server_API.app.core.Web_Scraping import filters
    from tldw_Server_API.tests.http_client.test_http_client_simple_response_limits import (
        _StreamingHTTPXClient,
        _StreamingResponse,
    )

    monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "false")
    monkeypatch.setenv("HTTP_ALLOW_CROSS_HOST_REDIRECTS", "true")
    monkeypatch.setattr(egress, "_resolve_host_ips", lambda *_args, **_kwargs: ["93.184.216.34"])
    _StreamingHTTPXClient.instances = []
    _StreamingHTTPXClient.responses = [
        _StreamingResponse(
            "https://example.com/robots.txt",
            [],
            status_code=302,
            headers={"Location": "https://user:secret@127.0.0.1/robots.txt"},
        )
    ]
    monkeypatch.setattr(hc, "_resolve_httpx", lambda: SimpleNamespace(Client=_StreamingHTTPXClient))
    monkeypatch.setattr(filters, "http_fetch", hc.fetch)
    with egress.public_url_policy_scope():
        result = await RobotsFilter(user_agent="test", credential_free=True).check(
            "https://example.com/article", fail_open=False
        )
    assert result.allowed is False
    assert len(_StreamingHTTPXClient.instances) == 1
    assert [call["url"] for call in _StreamingHTTPXClient.instances[0].stream_calls] == [
        "https://93.184.216.34/robots.txt"
    ]
    request = _StreamingHTTPXClient.instances[0].stream_calls[0]
    assert request["headers"]["Host"] == "example.com"
    assert request["extensions"]["sni_hostname"] == "example.com"
    assert request["cookies"] is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_public_robots_uses_fresh_bounded_cookie_free_negotiation(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping import filters

    calls = []

    def fetch(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": 200, "text": "User-agent: *\nAllow: /\n"}

    monkeypatch.setattr(filters, "http_fetch", fetch)
    result = await RobotsFilter(user_agent="test", credential_free=True).check(
        "https://example.com/article", skip_egress_check=True, fail_open=False
    )
    assert result.allowed is True
    assert calls[0][0] == ("https://example.com/robots.txt",)
    assert calls[0][1]["trust_env"] is False
    assert calls[0][1]["max_response_bytes"] == 1_000_000
    assert "cookies" not in calls[0][1]
    assert "method" not in calls[0][1]
    assert calls[0][1]["headers"]["Accept-Language"] == "en-US,en;q=0.9"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_public_robots_pins_every_redirect_and_preserves_logical_host(monkeypatch):
    import httpx

    from tldw_Server_API.app.core import http_client as hc
    from tldw_Server_API.app.core.Security import egress
    from tldw_Server_API.app.core.Web_Scraping import filters

    requests = []
    dispatched_hosts = []
    monkeypatch.setenv("HTTP_ALLOW_CROSS_HOST_REDIRECTS", "true")
    monkeypatch.setattr(
        egress, "_resolve_host_ips", lambda host, **kwargs: ["93.184.216.34"] if host == "example.com" else ["8.8.8.8"]
    )

    def send(request):
        requests.append(request)
        dispatched_hosts.append(request.url.host)
        if len(requests) == 1:
            return httpx.Response(
                302, headers={"Location": "https://robots.example/allow.txt", "Set-Cookie": "secret=server-session"}
            )
        return httpx.Response(200, stream=httpx.ByteStream(b"User-agent: *\nAllow: /\n"))

    def client_factory(**kwargs):
        return httpx.Client(transport=httpx.MockTransport(send), **kwargs)

    monkeypatch.setattr(hc, "_resolve_httpx", lambda: SimpleNamespace(Client=client_factory))
    monkeypatch.setattr(filters, "http_fetch", hc.fetch)
    with egress.public_url_policy_scope():
        result = await RobotsFilter(user_agent="test", credential_free=True).check(
            "https://example.com/article", fail_open=False
        )
    assert result.allowed is True
    assert dispatched_hosts == ["93.184.216.34", "8.8.8.8"]
    assert [request.headers["Host"] for request in requests] == ["example.com", "robots.example"]
    assert [request.extensions["sni_hostname"] for request in requests] == ["example.com", "robots.example"]
    assert all("Cookie" not in request.headers for request in requests)
