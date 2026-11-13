from tldw_Server_API.tests.Web_Scraping.phase5_fakes import make_harness


def test_scrape_from_sitemap_uses_canonical_sync_adapter():
    fixture = make_harness(
        profile="legacy_sitemap",
        outcomes=[
            {"url": "https://example.com/a"},
            {"url": "https://example.com/b"},
        ],
    )
    results = fixture.run_sync()
    assert len(results) == 2
    assert {(call["url"], call["kwargs"]["allow_llm_extraction"]) for call in fixture.calls} == {
        ("https://example.com/a", True),
        ("https://example.com/b", True),
    }


def test_scrape_by_url_level_uses_canonical_sync_adapter():
    fixture = make_harness(
        profile="legacy_url_level",
        outcomes=[
            {"url": "https://example.com/one", "links": ["/two/deep"]},
            {"url": "https://example.com/two/deep"},
        ],
    )
    results = fixture.run_sync(level=1)
    assert len(results) == 1
    assert [(call["url"], call["kwargs"]["allow_llm_extraction"]) for call in fixture.calls] == [
        ("https://example.com/one", True)
    ]


def test_scrape_from_sitemap_blocks_on_shared_policy_denial():
    fixture = make_harness(profile="legacy_sitemap", outcomes=[{"url": "https://example.com/a"}])
    fixture.denied_urls.add(fixture.sitemap_url)
    assert fixture.run_sync() == []
    assert fixture.fetch_calls == []
