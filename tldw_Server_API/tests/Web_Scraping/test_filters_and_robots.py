import asyncio
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Web_Scraping.filters import (
    DomainFilter,
    ContentTypeFilter,
    URLPatternFilter,
    RobotsFilter,
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
