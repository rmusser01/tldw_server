import types

import pytest

from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as AEL


class DummyResp:
    def __init__(self, text: str):
        self.data = {"status": 200, "text": text, "url": "https://example.com", "headers": {}, "backend": "httpx"}
    def __getitem__(self, k):
        return self.data[k]


class DummyAsyncPlaywright:
    async def __aenter__(self):
        # Raise so we don't actually try to launch browsers in tests
        raise RuntimeError("playwright disabled in test")
    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_js_required_emits_fallback_metric(monkeypatch):
    from tldw_Server_API.app.core.Security import egress as egress_module

    monkeypatch.setattr(
        egress_module,
        "evaluate_url_policy",
        lambda url, **_kwargs: types.SimpleNamespace(allowed=True),
    )

    async def allow_robots(*_args, **_kwargs):
        return True

    async def allow_policy(*_args, **_kwargs):
        return types.SimpleNamespace(
            allowed=True,
            reason="allowed",
            mode="test",
            stage="pre_fetch",
            source="article_extract",
        )

    # stub robots allow
    monkeypatch.setattr(AEL, "is_allowed_by_robots_async", allow_robots)
    monkeypatch.setattr(AEL, "_decide_article_pre_fetch_policy", allow_policy)

    # stub the lightweight fetch boundary to return JS-required HTML
    def fake_fetch_article_lightweight(*_args, **_kwargs):
        return DummyResp("Please enable JavaScript to continue"), "httpx"

    monkeypatch.setattr(AEL, "_fetch_article_lightweight", fake_fetch_article_lightweight)

    # stub async_playwright to avoid launching
    monkeypatch.setattr(AEL, "async_playwright", lambda: DummyAsyncPlaywright())

    # capture metrics
    calls = []
    def _increment_counter(name, value=1, labels=None):
        calls.append((name, dict(labels or {})))
    monkeypatch.setattr(AEL, "increment_counter", _increment_counter)

    # run
    res = await AEL.scrape_article("https://example.com")
    assert res["extraction_successful"] is False
    # Ensure js_required metric was emitted at least once
    js_fallbacks = [c for c in calls if c[0] == "scrape_playwright_fallback_total" and c[1].get("reason") == "js_required"]
    assert js_fallbacks, f"expected js_required fallback metric, got: {calls}"
