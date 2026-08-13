from tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib import (
    clear_extraction_caches,
    extract_article_with_pipeline,
    get_extraction_cache_stats,
)
from tldw_Server_API.app.core.Web_Scraping.extraction import throttles


def test_schema_result_cache_and_selector_cache():
    html = """
    <html>
      <body>
        <article>
          <h1>Title</h1>
          <div class="content">Body</div>
        </article>
      </body>
    </html>
    """
    schema_rules = {
        "baseSelector": "//article",
        "fields": [
            {"name": "title", "selector": "//article/h1", "type": "text"},
            {"name": "content", "selector": "//article/div[@class='content']", "type": "text"},
        ],
    }

    clear_extraction_caches()
    result = extract_article_with_pipeline(
        html,
        "https://example.com",
        strategy_order=["schema"],
        schema_rules=schema_rules,
    )

    assert result["extraction_successful"] is True

    stats = get_extraction_cache_stats()
    assert stats.get("schema_result_cache_size", 0) == 1
    assert stats.get("selector_xpath_cache_size", 0) > 0

    result_cached = extract_article_with_pipeline(
        html,
        "https://example.com",
        strategy_order=["schema"],
        schema_rules=schema_rules,
    )
    assert result_cached.get("schema_cache_hit") is True

    clear_extraction_caches()
    stats = get_extraction_cache_stats()
    assert stats.get("schema_result_cache_size", 0) == 0
    assert stats.get("selector_xpath_cache_size", 0) == 0


def test_extraction_cache_clear_requires_explicit_operation(monkeypatch):
    html = """
    <html>
      <body>
        <article>
          <h1>Title</h1>
          <div class="content">Body</div>
        </article>
      </body>
    </html>
    """
    schema_rules = {
        "baseSelector": "//article",
        "fields": [
            {"name": "title", "selector": "//article/h1", "type": "text"},
            {"name": "content", "selector": "//article/div[@class='content']", "type": "text"},
        ],
    }

    clear_extraction_caches()
    monkeypatch.setenv("EXTRACTOR_CLEAR_CACHES", "end")
    result = extract_article_with_pipeline(
        html,
        "https://example.com",
        strategy_order=["schema"],
        schema_rules=schema_rules,
    )

    assert result["extraction_successful"] is True

    stats = get_extraction_cache_stats()
    assert stats.get("schema_result_cache_size", 0) == 1
    assert stats.get("selector_xpath_cache_size", 0) > 0

    clear_extraction_caches()
    stats = get_extraction_cache_stats()
    assert stats.get("schema_result_cache_size", 0) == 0
    assert stats.get("selector_xpath_cache_size", 0) == 0


def test_strategy_limit_cache_stats(monkeypatch):
    html = """
    <html>
      <body>
        <p>No metadata here, just plain content.</p>
      </body>
    </html>
    """

    clear_extraction_caches()
    monkeypatch.setenv("EXTRACTOR_MAX_WORKERS", "2")
    extract_article_with_pipeline(
        html,
        "https://example.com",
        strategy_order=["jsonld", "regex"],
    )

    stats = get_extraction_cache_stats()
    assert stats.get("strategy_limit_count", 0) == 2

    clear_extraction_caches()
    throttles.clear_throttle_state()
    stats = get_extraction_cache_stats()
    assert stats.get("strategy_limit_count", 0) == 0
