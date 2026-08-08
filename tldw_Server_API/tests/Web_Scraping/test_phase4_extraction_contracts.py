from __future__ import annotations

import ast
import asyncio
import dataclasses
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as legacy
from tldw_Server_API.app.core.Web_Scraping import extraction
from tldw_Server_API.app.core.Web_Scraping.extraction import caches
from tldw_Server_API.app.core.Web_Scraping.extraction.dependencies import (
    ExtractionDependencies,
    build_default_dependencies,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
EXTRACTION_ROOT = REPO_ROOT / "tldw_Server_API" / "app" / "core" / "Web_Scraping" / "extraction"


def test_extraction_facade_has_the_phase4b_public_contract() -> None:
    assert extraction.__all__ == [
        "ExtractionDependencies",
        "build_default_dependencies",
        "clear_extraction_caches",
        "get_extraction_cache_stats",
        "extract_jsonld_entities",
        "extract_regex_entities",
    ]
    assert legacy.clear_extraction_caches is extraction.clear_extraction_caches
    assert legacy.get_extraction_cache_stats is extraction.get_extraction_cache_stats


def test_legacy_jsonld_and_regex_exports_are_canonical() -> None:
    assert legacy.extract_jsonld_entities is extraction.extract_jsonld_entities
    assert legacy.extract_regex_entities is extraction.extract_regex_entities


def test_extraction_package_does_not_import_the_legacy_wrapper() -> None:
    violations: list[str] = []
    for path in EXTRACTION_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports = (alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports = (node.module or "",)
            else:
                continue
            if any("Article_Extractor_Lib" in imported for imported in imports):
                violations.append(str(path.relative_to(EXTRACTION_ROOT)))

    assert violations == []


def test_default_dependencies_are_immutable_and_created_at_call_time() -> None:
    first = build_default_dependencies()
    second = build_default_dependencies()

    assert isinstance(first, ExtractionDependencies)
    assert first is not second
    assert dataclasses.is_dataclass(first)
    assert first.validate_selector_rules is not None
    assert first.extract_schema_fields is not None
    assert first.perform_chat_api_call is not None
    assert first.cancellation_checkpoint() is None

    try:
        first.sleep = lambda _seconds: None
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("ExtractionDependencies must be frozen")


def test_schema_and_cluster_cache_reads_and_writes_are_isolated() -> None:
    caches.clear_extraction_caches()
    schema_value = {"extraction_successful": True, "nested": {"values": ["original"]}}
    cluster_value = [1.0, 2.0]

    caches._schema_cache_put("schema", schema_value)
    caches._cluster_cache_put("cluster", cluster_value)
    schema_value["nested"]["values"].append("caller-mutation")
    cluster_value.append(3.0)

    cached_schema = caches._schema_cache_get("schema")
    cached_cluster = caches._cluster_cache_get("cluster")
    assert cached_schema == {"extraction_successful": True, "nested": {"values": ["original"]}}
    assert cached_cluster == [1.0, 2.0]

    assert cached_schema is not None
    assert cached_cluster is not None
    cached_schema["nested"]["values"].append("cache-read-mutation")
    cached_cluster.append(4.0)

    assert caches._schema_cache_get("schema") == {
        "extraction_successful": True,
        "nested": {"values": ["original"]},
    }
    assert caches._cluster_cache_get("cluster") == [1.0, 2.0]


def test_schema_cache_stores_successes_only_and_reports_selector_keys() -> None:
    caches.clear_extraction_caches()
    caches._schema_cache_put("failed", {"extraction_successful": False})
    caches._schema_cache_put("success", {"extraction_successful": True, "content": "value"})

    assert caches._schema_cache_get("failed") is None
    assert caches._schema_cache_get("success") == {
        "extraction_successful": True,
        "content": "value",
    }
    assert extraction.get_extraction_cache_stats() == {
        "cluster_embedding_cache_size": 0,
        "schema_result_cache_size": 1,
        "llm_provider_limit_count": 0,
        "llm_provider_last_call_count": 0,
        "strategy_limit_count": 0,
        "selector_xpath_cache_size": 0,
        "selector_css_cache_size": 0,
    }


def test_partial_schema_result_with_no_matches_warning_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    html = """
    <html><body><article><h1>Available title</h1></article></body></html>
    """
    schema_rules = {
        "baseSelector": "//article",
        "fields": [
            {"name": "title", "selector": "//article/h1", "type": "text"},
            {"name": "content", "selector": "//article/div[@class='missing']", "type": "text"},
        ],
    }
    original_extract_schema_fields = legacy.extract_schema_fields
    extraction_calls = 0

    def track_schema_extraction(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal extraction_calls
        extraction_calls += 1
        return original_extract_schema_fields(*args, **kwargs)

    monkeypatch.setattr(legacy, "extract_schema_fields", track_schema_extraction)
    extraction.clear_extraction_caches()

    first_result = legacy.extract_article_with_pipeline(
        html,
        "https://example.com/partial",
        strategy_order=["schema"],
        schema_rules=schema_rules,
    )
    second_result = legacy.extract_article_with_pipeline(
        html,
        "https://example.com/partial",
        strategy_order=["schema"],
        schema_rules=schema_rules,
    )

    assert first_result["extraction_successful"] is True
    assert first_result["schema_selector_warnings"] == [
        {
            "key": "fields.content",
            "selector": "//article/div[@class='missing']",
            "warning": "no_matches",
        }
    ]
    assert extraction.get_extraction_cache_stats()["schema_result_cache_size"] == 0
    assert second_result.get("schema_cache_hit") is None
    assert extraction_calls == 2


def test_selector_cache_lifecycle_failures_are_silent_and_omit_selector_stats(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "https://user:secret@example.test/selector"

    def raise_sensitive_error() -> None:
        raise ValueError(secret)

    monkeypatch.setattr(caches, "get_selector_cache_stats", raise_sensitive_error)
    monkeypatch.setattr(caches, "clear_selector_caches", raise_sensitive_error)

    stats = caches.get_extraction_cache_stats()
    caches.clear_extraction_caches()

    assert "selector_xpath_cache_size" not in stats
    assert "selector_css_cache_size" not in stats
    assert secret not in caplog.text


def test_selector_cache_lifecycle_cancellation_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_cancellation() -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(caches, "get_selector_cache_stats", raise_cancellation)
    with pytest.raises(asyncio.CancelledError):
        caches.get_extraction_cache_stats()

    monkeypatch.setattr(caches, "clear_selector_caches", raise_cancellation)
    with pytest.raises(asyncio.CancelledError):
        caches.clear_extraction_caches()
