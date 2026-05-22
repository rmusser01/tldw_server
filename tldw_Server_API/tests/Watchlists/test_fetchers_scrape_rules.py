import pytest

from tldw_Server_API.app.core.Watchlists.fetchers import (
    fetch_site_items_with_rules,
    parse_scraped_items,
)


def test_parse_scraped_items_xpath():


    html_text = """
    <html>
      <body>
        <section class="entries">
          <article>
            <h2><a href="/post-1">Post 1</a></h2>
            <p class="summary">Alpha Summary</p>
            <p class="summary">More details.</p>
            <time datetime="2024-08-01">August 1, 2024</time>
          </article>
          <article>
            <h2><a href="/post-2">Post 2</a></h2>
            <p class="summary">Beta Summary</p>
            <time datetime="2024-08-02">August 2, 2024</time>
          </article>
        </section>
      </body>
    </html>
    """
    rules = {
        "entry_xpath": "//section[@class='entries']/article",
        "title_xpath": ".//h2/a/text()",
        "link_xpath": ".//h2/a/@href",
        "summary_xpath": ".//p[@class='summary']/text()",
        "published_xpath": ".//time/@datetime",
        "published_format": "%Y-%m-%d",
    }
    result = parse_scraped_items(html_text, "https://example.com/blog", rules)
    assert len(result["items"]) == 2
    first = result["items"][0]
    assert first["url"] == "https://example.com/post-1"
    assert first["title"] == "Post 1"
    assert first["summary"] == "Alpha Summary More details."
    assert first["published"].startswith("2024-08-01")


def test_parse_scraped_items_alternates():


    html_text = """
    <html>
      <body>
        <div class="main">
          <article>
            <a class="link" href="/alpha">Alpha</a>
          </article>
        </div>
        <section class="fallback">
          <div class="entry">
            <a class="head" href="/beta">Beta</a>
          </div>
        </section>
      </body>
    </html>
    """
    rules = {
        "entry_xpath": "//div[@class='main']//article",
        "link_xpath": ".//a[@class='link']/@href",
        "title_xpath": ".//a[@class='link']/text()",
        "alternates": [
            {
                "entry_xpath": "//section[@class='fallback']//div[@class='entry']",
                "link_xpath": ".//a[@class='head']/@href",
                "title_xpath": ".//a[@class='head']/text()",
            }
        ],
    }
    result = parse_scraped_items(html_text, "https://example.com/blog", rules)
    urls = sorted(item["url"] for item in result["items"])
    assert urls == ["https://example.com/alpha", "https://example.com/beta"]


def test_parse_scraped_items_skips_unsafe_xpath_selectors():
    html_text = """
    <html>
      <body>
        <div class="entry">
          <a href="/unsafe">Unsafe</a>
        </div>
      </body>
    </html>
    """
    rules = {
        "entry_xpath": "//div[@class='entry']",
        # Wildcard descendant selectors are too broad and should be rejected.
        "link_xpath": "//*",
        "title_xpath": ".//a/text()",
    }

    result = parse_scraped_items(html_text, "https://example.com/blog", rules)
    assert result["items"] == []


@pytest.mark.asyncio
async def test_fetch_site_items_with_rules_test_mode(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "1")
    rules = {"list_url": "https://example.com/blog", "limit": 2}
    items = await fetch_site_items_with_rules("https://example.com/blog", rules)
    assert len(items) == 2
    assert items[0]["url"] == "https://example.com/blog/test-scrape-1"
    assert items[0]["title"] == "Test scraped item 1"


@pytest.mark.asyncio
async def test_fetch_site_items_with_rules_test_mode_y(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "y")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")
    rules = {"list_url": "https://example.com/blog", "limit": 1}
    items = await fetch_site_items_with_rules("https://example.com/blog", rules)
    assert len(items) == 1
    assert items[0]["url"] == "https://example.com/blog/test-scrape-1"


@pytest.mark.asyncio
async def test_fetch_site_items_with_rules_pagination(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "0")

    page_one = """
    <html>
      <body>
        <div class="entry"><a href="/post-1">First</a><p>One</p></div>
        <div class="entry"><a href="/post-2">Second</a><p>Two</p></div>
        <a class="next" href="/blog?page=2">Next</a>
      </body>
    </html>
    """
    page_two = """
    <html>
      <body>
        <div class="entry"><a href="/post-3">Third</a><p>Three</p></div>
      </body>
    </html>
    """

    responses = {
        "https://example.com/blog": page_one,
        "https://example.com/blog?page=2": page_two,
    }

    class FakeResponse:
        def __init__(self, url: str):
            self.status_code = 200
            self.headers = {}
            self.text = responses[url]

    async def fake_afetch(*, method: str, url: str, client=None, headers=None, timeout=None, **kwargs):
        # mimic http_client.afetch signature; ignore method and headers
        if url not in responses:
            raise ValueError(f"unexpected URL {url}")
        return FakeResponse(url)

    # fetch_site_items_with_rules relies on http_client.afetch
    # Patch it in the http_client module so no real network calls occur
    monkeypatch.setattr(
        "tldw_Server_API.app.core.http_client.afetch",
        fake_afetch,
        raising=True,
    )
    monkeypatch.setattr("tldw_Server_API.app.core.Watchlists.fetchers.is_url_allowed_for_tenant", lambda url, tenant_id: True)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Watchlists.fetchers.is_url_allowed",
        lambda url: True,
    )

    rules = {
        "list_url": "https://example.com/blog",
        "entry_xpath": "//div[@class='entry']",
        "link_xpath": ".//a/@href",
        "title_xpath": ".//a/text()",
        "summary_xpath": ".//p/text()",
        "pagination": {
            "next_xpath": "//a[@class='next']/@href",
            "max_pages": 2,
        },
    }

    items = await fetch_site_items_with_rules("https://example.com/blog", rules)
    assert len(items) == 3
    assert items[0]["title"] == "First"
    assert items[-1]["url"] == "https://example.com/post-3"


@pytest.mark.asyncio
async def test_fetch_site_items_with_rules_reports_fetch_status(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "0")

    class FakeResponse:
        status_code = 503
        headers = {}
        text = "Service unavailable"

    async def fake_afetch(*, method: str, url: str, client=None, headers=None, timeout=None, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.http_client.afetch",
        fake_afetch,
        raising=True,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Watchlists.fetchers.is_url_allowed_for_tenant",
        lambda url, tenant_id: True,
    )
    monkeypatch.setattr("tldw_Server_API.app.core.Watchlists.fetchers.is_url_allowed", lambda url: True)

    events: list[dict[str, object]] = []
    items = await fetch_site_items_with_rules(
        "https://example.com/blog",
        {
            "entry_xpath": "//article",
            "link_xpath": ".//a/@href",
        },
        fetch_diagnostics=events.append,
    )

    assert items == []
    assert events == [
        {
            "url": "https://example.com/blog",
            "status": 503,
        }
    ]
