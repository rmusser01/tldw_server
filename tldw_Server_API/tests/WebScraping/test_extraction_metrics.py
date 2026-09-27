import dataclasses

from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as ael
from tldw_Server_API.app.core.Web_Scraping.extraction import pipeline
from tldw_Server_API.app.core.Web_Scraping.extraction.dependencies import build_default_dependencies


def test_extraction_metrics_emits_success_status(monkeypatch):
    calls = []

    def fake_log_counter(name, labels=None):
        calls.append((name, labels or {}))

    monkeypatch.setattr(
        pipeline,
        "build_default_dependencies",
        lambda: dataclasses.replace(build_default_dependencies(), log_counter=fake_log_counter),
    )

    html = "<html><body>Email: demo@example.com</body></html>"
    result = ael.extract_article_with_pipeline(
        html,
        "https://example.com",
        strategy_order=["regex", "cluster"],
    )

    assert result["extraction_strategy"] == "regex"

    labels = [
        entry for name, entry in calls
        if name == "extraction_strategy_total"
    ]
    assert {"strategy": "regex", "status": "success"} in labels
    assert not any(entry.get("strategy") == "cluster" for entry in labels)


def test_extraction_metrics_emits_failed_status(monkeypatch):
    calls = []

    def fake_log_counter(name, labels=None):
        calls.append((name, labels or {}))

    monkeypatch.setattr(
        pipeline,
        "build_default_dependencies",
        lambda: dataclasses.replace(build_default_dependencies(), log_counter=fake_log_counter),
    )

    result = ael.extract_article_with_pipeline(
        "",
        "https://example.com",
        strategy_order=["regex", "cluster"],
    )

    assert result["extraction_successful"] is False

    labels = [
        entry for name, entry in calls
        if name == "extraction_strategy_total"
    ]
    assert {"strategy": "regex", "status": "failed"} in labels
    assert {"strategy": "cluster", "status": "failed"} in labels
