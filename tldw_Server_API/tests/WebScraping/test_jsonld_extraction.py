import pytest

from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as extractor_lib
from tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib import (
    extract_article_with_pipeline,
    extract_jsonld_entities,
)

DESCRIPTION_ONLY_JSONLD = """
<html>
  <head>
    <script type="application/ld+json">
      {
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": "Structured title",
        "description": "Structured summary"
      }
    </script>
  </head>
  <body></body>
</html>
"""


@pytest.mark.asyncio
async def test_legacy_scrape_preserves_structured_summary_when_summarization_disabled(monkeypatch):
    async def fake_scrape_article(
        _url: str,
        custom_cookies=None,
        *,
        allow_llm_extraction: bool = True,
    ):
        _ = allow_llm_extraction
        return {
            "url": "https://example.com",
            "title": "Structured title",
            "content": "Fallback body",
            "summary": "Structured summary",
            "extraction_successful": True,
        }

    monkeypatch.setattr(extractor_lib, "scrape_article", fake_scrape_article)
    monkeypatch.setattr(extractor_lib, "RateLimiter", lambda: None)

    result = await extractor_lib.scrape_and_summarize_multiple(
        urls="https://example.com",
        custom_prompt_arg=None,
        api_name="",
        api_key=None,
        keywords="",
        custom_article_titles=None,
        summarize_checkbox=False,
    )

    assert result[0]["summary"] == "Structured summary"


def test_jsonld_description_only_is_not_successful():
    result = extract_jsonld_entities(DESCRIPTION_ONLY_JSONLD, "https://example.com")

    assert result["extraction_successful"] is False
    assert result["content"] == ""
    assert result["summary"] == "Structured summary"


def test_pipeline_retains_jsonld_summary_after_fallback():
    def fallback_extractor(_html: str, url: str) -> dict[str, str | bool]:
        return {
            "url": url,
            "title": "Fallback title",
            "author": "N/A",
            "date": "N/A",
            "content": "Fallback body",
            "summary": "   ",
            "extraction_successful": True,
        }

    result = extract_article_with_pipeline(
        DESCRIPTION_ONLY_JSONLD,
        "https://example.com",
        strategy_order=["jsonld", "trafilatura"],
        fallback_extractor=fallback_extractor,
    )

    assert result["content"] == "Fallback body"
    assert result["extraction_strategy"] == "trafilatura"
    assert result["summary"] == "Structured summary"


def test_pipeline_keeps_fallback_summary_over_jsonld_summary():
    def fallback_extractor(_html: str, url: str) -> dict[str, str | bool]:
        return {
            "url": url,
            "title": "Fallback title",
            "author": "N/A",
            "date": "N/A",
            "content": "Fallback body",
            "summary": "Fallback summary",
            "extraction_successful": True,
        }

    result = extract_article_with_pipeline(
        DESCRIPTION_ONLY_JSONLD,
        "https://example.com",
        strategy_order=["jsonld", "trafilatura"],
        fallback_extractor=fallback_extractor,
    )

    assert result["content"] == "Fallback body"
    assert result["extraction_strategy"] == "trafilatura"
    assert result["summary"] == "Fallback summary"


def test_pipeline_does_not_carry_jsonld_summary_when_fallback_fails():
    def fallback_extractor(_html: str, url: str) -> dict[str, str | bool]:
        return {
            "url": url,
            "title": "Fallback title",
            "author": "N/A",
            "date": "N/A",
            "content": "",
            "extraction_successful": False,
        }

    result = extract_article_with_pipeline(
        DESCRIPTION_ONLY_JSONLD,
        "https://example.com",
        strategy_order=["jsonld", "trafilatura"],
        fallback_extractor=fallback_extractor,
    )

    assert result["extraction_successful"] is False
    assert result.get("summary") is None


def test_jsonld_extraction_basic():
    html = """
    <html>
      <head>
        <script type="application/ld+json">
          {
            "@context": "https://schema.org",
            "@type": "NewsArticle",
            "headline": "JSON-LD Title",
            "author": {"@type": "Person", "name": "Jane Doe"},
            "datePublished": "2024-05-01",
            "articleBody": "JSON-LD body text."
          }
        </script>
      </head>
      <body></body>
    </html>
    """
    result = extract_jsonld_entities(html, "https://example.com")

    assert result["extraction_successful"] is True
    assert result["title"] == "JSON-LD Title"
    assert result["author"] == "Jane Doe"
    assert result["date"] == "2024-05-01"
    assert "JSON-LD body text." in result["content"]


def test_jsonld_extraction_multiple_blocks_prefers_article():
    html = """
    <html>
      <head>
        <script type="application/ld+json">
          {"@context": "https://schema.org", "@type": "WebPage", "name": "Page Title", "description": "Short desc"}
        </script>
        <script type="application/ld+json">
          {"@context": "https://schema.org", "@type": "Article", "headline": "Article Title", "articleBody": "Article body"}
        </script>
      </head>
      <body></body>
    </html>
    """
    result = extract_jsonld_entities(html, "https://example.com")

    assert result["extraction_successful"] is True
    assert result["title"] == "Article Title"
    assert result["content"] == "Article body"


def test_jsonld_extraction_invalid_json():
    html = """
    <html>
      <head>
        <script type="application/ld+json">
          {"@context": "https://schema.org", "@type": "Article", "headline": "Bad JSON",
        </script>
      </head>
      <body></body>
    </html>
    """
    result = extract_jsonld_entities(html, "https://example.com")

    assert result["extraction_successful"] is False
    assert result.get("jsonld_error")


def test_jsonld_pipeline_short_circuits():
    html = """
    <html>
      <head>
        <script type="application/ld+json">
          {"@context": "https://schema.org", "@type": "Article", "headline": "Pipe Title", "articleBody": "Pipe body"}
        </script>
      </head>
      <body></body>
    </html>
    """
    result = extract_article_with_pipeline(html, "https://example.com")

    assert result["extraction_successful"] is True
    assert result["extraction_strategy"] == "jsonld"
    assert result["content"] == "Pipe body"


def test_microdata_extraction_basic():
    html = """
    <html>
      <body>
        <div itemscope itemtype="https://schema.org/Article">
          <span itemprop="headline">Micro Title</span>
          <span itemprop="author">Micro Author</span>
          <time itemprop="datePublished" datetime="2024-06-01">June 1</time>
          <div itemprop="articleBody">Micro body text</div>
        </div>
      </body>
    </html>
    """
    result = extract_jsonld_entities(html, "https://example.com")

    assert result["extraction_successful"] is True
    assert result["title"] == "Micro Title"
    assert result["author"] == "Micro Author"
    assert result["date"] == "2024-06-01"
    assert result["content"] == "Micro body text"
