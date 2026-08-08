import dataclasses

from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as ael
from tldw_Server_API.app.core.Web_Scraping.extraction import pipeline
from tldw_Server_API.app.core.Web_Scraping.extraction.dependencies import build_default_dependencies


def test_extraction_metrics_emitted_for_regex(monkeypatch):
    calls = []

    def fake_histogram(name, value, labels=None):
        calls.append((name, labels or {}, value))

    monkeypatch.setattr(
        pipeline,
        "build_default_dependencies",
        lambda: dataclasses.replace(build_default_dependencies(), observe_histogram=fake_histogram),
    )

    html = "<html><body>Email: demo@example.com</body></html>"
    result = ael.extract_article_with_pipeline(
        html,
        "https://example.com",
        strategy_order=["regex"],
    )

    assert result["extraction_successful"] is True

    assert any(
        name == "extraction_strategy_duration_seconds"
        and labels.get("strategy") == "regex"
        and labels.get("status") == "success"
        for name, labels, _value in calls
    )
    assert any(
        name == "extraction_content_length_bytes"
        and labels.get("strategy") == "regex"
        for name, labels, _value in calls
    )


def test_schema_observability_fields_present():
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

    result = ael.extract_article_with_pipeline(
        html,
        "https://example.com",
        strategy_order=["schema"],
        schema_rules=schema_rules,
    )

    assert result["extraction_successful"] is True
    selector_counts = result.get("schema_selector_counts") or {}
    assert selector_counts.get("baseSelector") == 1
    assert selector_counts.get("title") == 1
    assert selector_counts.get("content") == 1
    rule_keys = result.get("schema_rule_keys") or []
    assert "baseSelector" in rule_keys
    assert "title" in rule_keys
    assert "content" in rule_keys
