from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import EnhancedWebScraper
from tldw_Server_API.app.core.Web_Scraping.outbound_policy import (
    WebOutboundPolicyDecision,
)


def _allow_egress(monkeypatch):
    import tldw_Server_API.app.core.Security.egress as egress

    monkeypatch.setattr(egress, "evaluate_url_policy", lambda url: SimpleNamespace(allowed=True))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_best_first_metadata_exposes_relevance_and_ordering_scores(monkeypatch):
    _allow_egress(monkeypatch)
    scraper = EnhancedWebScraper(config={})

    base = "http://example.com"

    async def fake_scrape_multiple(urls, method="trafilatura", **kwargs):
        return [
            {
                "url": u,
                "content": "<html><body>ok</body></html>",
                "extraction_successful": True,
                "method": method,
            }
            for u in urls
        ]

    async def fake_extract_links(url, content):
        return ["/a"] if url == base else []

    monkeypatch.setattr(scraper, "scrape_multiple", fake_scrape_multiple)
    monkeypatch.setattr(scraper, "_extract_links", fake_extract_links)

    res = await scraper.recursive_scrape(base_url=base, max_pages=2, max_depth=2)
    assert len(res) == 2

    md = res[1].get("metadata", {})
    assert md.get("crawl_strategy") == "best_first"
    assert md.get("score_type") == "composite_relevance"
    assert md.get("ordering_score_type") == "path_depth"
    assert isinstance(md.get("score"), float)
    assert isinstance(md.get("ordering_score"), float)
    assert md.get("score") != md.get("ordering_score")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fifo_metadata_is_strategy_tagged_and_keeps_score_fields(monkeypatch):
    _allow_egress(monkeypatch)
    scraper = EnhancedWebScraper(config={})

    base = "http://example.com/base"
    b1 = "http://example.com/base/a"
    b2 = "http://example.com/base/b"

    async def fake_scrape_multiple(urls, method="trafilatura", **kwargs):
        return [
            {
                "url": u,
                "content": "<html><body>ok</body></html>",
                "extraction_successful": True,
                "method": method,
            }
            for u in urls
        ]

    async def fake_extract_links(url, content):
        return [b1, b2] if url == base else []

    monkeypatch.setattr(scraper, "scrape_multiple", fake_scrape_multiple)
    monkeypatch.setattr(scraper, "_extract_links", fake_extract_links)

    res = await scraper.recursive_scrape(
        base_url=base,
        max_pages=3,
        max_depth=2,
        crawl_strategy="default",
    )
    assert len(res) == 3

    md = res[1].get("metadata", {})
    assert md.get("crawl_strategy") == "default"
    assert md.get("score_type") == "composite_relevance"
    assert md.get("ordering_score_type") == "path_depth"
    assert isinstance(md.get("score"), float)
    assert isinstance(md.get("ordering_score"), float)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_best_first_path_depth_guard_is_relative_to_base_path(monkeypatch):
    _allow_egress(monkeypatch)
    scraper = EnhancedWebScraper(config={})

    base = "http://example.com/docs/v1"
    allowed_child = "http://example.com/docs/v1/topic"
    blocked_child = "http://example.com/docs/v1/topic/deep"

    async def fake_scrape_multiple(urls, method="trafilatura", **kwargs):
        return [
            {
                "url": u,
                "content": "<html><body>ok</body></html>",
                "extraction_successful": True,
                "method": method,
            }
            for u in urls
        ]

    async def fake_extract_links(url, content):
        if url == base:
            return [allowed_child, blocked_child]
        return []

    monkeypatch.setattr(scraper, "scrape_multiple", fake_scrape_multiple)
    monkeypatch.setattr(scraper, "_extract_links", fake_extract_links)

    res = await scraper.recursive_scrape(base_url=base, max_pages=5, max_depth=1)
    urls = [r.get("url") for r in res]

    assert base in urls
    assert allowed_child in urls
    assert blocked_child not in urls


@pytest.mark.integration
@pytest.mark.asyncio
async def test_best_first_strict_policy_skips_candidate_on_robots_error(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping import enhanced_web_scraping as scraper_mod

    _allow_egress(monkeypatch)
    scraper = EnhancedWebScraper(config={"web_scraper_respect_robots": True})

    base = "http://example.com"
    blocked_child = "http://example.com/blocked"

    async def fake_scrape_multiple(urls, method="trafilatura", **kwargs):
        return [
            {
                "url": u,
                "content": "<html><body>ok</body></html>",
                "extraction_successful": True,
                "method": method,
            }
            for u in urls
        ]

    async def fake_extract_links(url, content):
        if url == base:
            return [blocked_child]
        return []

    def fake_sync_policy(*args, **kwargs):
        return WebOutboundPolicyDecision(
            allowed=True,
            mode="strict",
            reason="allowed",
            stage=kwargs.get("stage", "pre_fetch"),
            source=kwargs.get("source", "recursive_scrape"),
        )

    async def fake_policy(url, **kwargs):
        if url == blocked_child:
            return WebOutboundPolicyDecision(
                allowed=False,
                mode="strict",
                reason="robots_unreachable",
                stage=kwargs.get("stage", "discovery"),
                source=kwargs.get("source", "recursive_scrape_candidate"),
            )
        return WebOutboundPolicyDecision(
            allowed=True,
            mode="strict",
            reason="allowed",
            stage=kwargs.get("stage", "discovery"),
            source=kwargs.get("source", "recursive_scrape_candidate"),
        )

    monkeypatch.setattr(
        scraper_mod,
        "decide_web_outbound_policy_sync",
        fake_sync_policy,
        raising=False,
    )
    monkeypatch.setattr(
        scraper_mod,
        "decide_web_outbound_policy",
        fake_policy,
        raising=False,
    )
    monkeypatch.setattr(scraper, "scrape_multiple", fake_scrape_multiple)
    monkeypatch.setattr(scraper, "_extract_links", fake_extract_links)
    monkeypatch.setattr(
        scraper_mod.RobotsFilter,
        "allowed",
        lambda self, url: True,
        raising=False,
    )

    res = await scraper.recursive_scrape(base_url=base, max_pages=5, max_depth=2)
    urls = [r.get("url") for r in res]

    assert base in urls
    assert blocked_child not in urls
