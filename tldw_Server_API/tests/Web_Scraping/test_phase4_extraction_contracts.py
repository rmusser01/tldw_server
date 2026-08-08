from __future__ import annotations

import ast
import asyncio
import dataclasses
import inspect
from pathlib import Path
from typing import Any, Optional

import pytest

from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as legacy
from tldw_Server_API.app.core.Web_Scraping import extraction
from tldw_Server_API.app.core.Web_Scraping.extraction import caches
from tldw_Server_API.app.core.Web_Scraping.extraction.dependencies import (
    ExtractionDependencies,
    build_default_dependencies,
)
from tldw_Server_API.app.core.Web_Scraping.extraction.strategies import (
    jsonld as jsonld_strategy,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
EXTRACTION_ROOT = REPO_ROOT / "tldw_Server_API" / "app" / "core" / "Web_Scraping" / "extraction"
FORBIDDEN_EXTRACTION_IMPORT_PARTS = {
    "Article_Extractor_Lib",
    "Watchlists",
    "WebSearch",
    "WebSearch_APIs",
    "enhanced_web_scraping",
    "orchestration",
    "playwright",
    "policy",
    "preflight",
    "routing",
    "scraper_router",
}


@pytest.mark.parametrize(
    ("source", "expected_import"),
    [
        (
            "from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib",
            "tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib",
        ),
        ("from .. import policy", "policy"),
    ],
)
def test_extraction_dependency_guard_rejects_absolute_and_relative_imports(
    source: str,
    expected_import: str,
) -> None:
    assert _forbidden_extraction_imports(ast.parse(source)) == [expected_import]


def _forbidden_extraction_imports(tree: ast.AST) -> list[str]:
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports = [f"{module}.{alias.name}" if module else alias.name for alias in node.names]
        else:
            continue
        for imported in imports:
            if set(imported.split(".")) & FORBIDDEN_EXTRACTION_IMPORT_PARTS:
                violations.append(imported)
    return violations


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


def test_canonical_and_legacy_strategy_signatures_match_predecessor() -> None:
    expected_jsonld = inspect.Signature(
        parameters=[
            inspect.Parameter("html_text", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
            inspect.Parameter("url", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
        ],
        return_annotation=dict[str, Any],
    )
    expected_regex = inspect.Signature(
        parameters=[
            inspect.Parameter("html_text", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
            inspect.Parameter("url", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
            inspect.Parameter(
                "mask_pii",
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=Optional[bool],
            ),
        ],
        return_annotation=dict[str, Any],
    )

    assert inspect.signature(extraction.extract_jsonld_entities) == expected_jsonld
    assert inspect.signature(legacy.extract_jsonld_entities) == expected_jsonld
    assert inspect.signature(extraction.extract_regex_entities) == expected_regex
    assert inspect.signature(legacy.extract_regex_entities) == expected_regex


def test_extraction_package_does_not_import_forbidden_upward_layers() -> None:
    violations: list[tuple[str, str]] = []
    for path in EXTRACTION_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        violations.extend(
            (str(path.relative_to(EXTRACTION_ROOT)), imported) for imported in _forbidden_extraction_imports(tree)
        )

    assert violations == []


def test_jsonld_extracts_article_from_graph() -> None:
    html = """
    <script type="application/ld+json">
      {"@graph": [
        {"@type": "WebPage", "name": "Landing page"},
        {"@type": "Article", "headline": "Graph title", "articleBody": "Graph body"}
      ]}
    </script>
    """

    result = extraction.extract_jsonld_entities(html, "https://example.com/graph")

    assert result["title"] == "Graph title"
    assert result["content"] == "Graph body"
    assert result["jsonld_types"] == ["article"]
    assert result["extraction_successful"] is True


@pytest.mark.parametrize("reference_key", ["mainEntity", "mainEntityOfPage"])
def test_jsonld_resolves_article_references_through_id_map(
    monkeypatch: pytest.MonkeyPatch,
    reference_key: str,
) -> None:
    resolved_targets: list[tuple[dict[str, Any], dict[str, Any]]] = []
    canonical_resolver = jsonld_strategy._resolve_jsonld_refs

    def observe_resolution(value: Any, id_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        resolved = canonical_resolver(value, id_map)
        if isinstance(value, dict) and value.get("@id") == "#article":
            assert len(resolved) == 1
            resolved_targets.append((resolved[0], id_map["#article"]))
        return resolved

    monkeypatch.setattr(jsonld_strategy, "_resolve_jsonld_refs", observe_resolution)

    html = f"""
    <script type="application/ld+json">
      {{"@graph": [
        {{"@type": "WebPage", "{reference_key}": {{"@id": "#article"}}}},
        {{"@id": "#article", "@type": "Article", "headline": "Referenced title", "articleBody": "Referenced body"}}
      ]}}
    </script>
    """

    result = extraction.extract_jsonld_entities(html, "https://example.com/reference")

    assert len(resolved_targets) == 1
    resolved_target, mapped_target = resolved_targets[0]
    assert resolved_target is mapped_target
    assert result["title"] == "Referenced title"
    assert result["content"] == "Referenced body"
    assert result["extraction_successful"] is True


def test_jsonld_decodes_concatenated_objects() -> None:
    html = """
    <script type="application/ld+json">
      {"@type": "WebPage", "name": "Landing"}
      {"@type": "Article", "headline": "Second object", "articleBody": "Second body"}
    </script>
    """

    result = extraction.extract_jsonld_entities(html, "https://example.com/objects")

    assert result["title"] == "Second object"
    assert result["content"] == "Second body"
    assert result["extraction_successful"] is True


def test_jsonld_parse_failure_is_sanitized_and_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "private parser detail"

    def raise_sensitive_error(_payload: str) -> object:
        raise ValueError(secret)

    monkeypatch.setattr(jsonld_strategy.json, "loads", raise_sensitive_error)

    result = extraction.extract_jsonld_entities(
        '<script type="application/ld+json">not-json</script>',
        "https://example.com/private",
    )

    assert result["jsonld_error"] == "jsonld_parse_failed"
    assert secret not in str(result)


def test_jsonld_cancellation_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_cancellation(_payload: str) -> object:
        raise asyncio.CancelledError

    monkeypatch.setattr(jsonld_strategy.json, "loads", raise_cancellation)

    with pytest.raises(asyncio.CancelledError):
        extraction.extract_jsonld_entities(
            '<script type="application/ld+json">not-json</script>',
            "https://example.com/cancel",
        )


def test_regex_total_match_cap_is_200() -> None:
    html = " ".join(f"user{index}@example.com" for index in range(250))

    result = extraction.extract_regex_entities(html, "https://example.com/total", mask_pii=False)

    assert len(result["regex_matches"]) == 200
    assert {match["label"] for match in result["regex_matches"]} == {"email"}


def test_regex_number_match_cap_is_50() -> None:
    html = ";".join(str(index) for index in range(1, 76))

    result = extraction.extract_regex_entities(html, "https://example.com/numbers", mask_pii=False)
    number_matches = [match for match in result["regex_matches"] if match["label"] == "number"]

    assert len(number_matches) == 50


def test_regex_suppresses_number_overlaps() -> None:
    result = extraction.extract_regex_entities(
        "Visit https://example.com/12345 today",
        "https://example.com/overlap",
        mask_pii=False,
    )

    assert not [match for match in result["regex_matches"] if match["label"] == "number"]


def test_regex_rejects_invalid_ip_addresses() -> None:
    result = extraction.extract_regex_entities(
        "Invalid address: 999.999.999.999",
        "https://example.com/ip",
        mask_pii=False,
    )

    assert not [match for match in result["regex_matches"] if match["label"] in {"ipv4", "ipv6"}]


def test_regex_applies_luhn_filtering() -> None:
    result = extraction.extract_regex_entities(
        "Valid 4111 1111 1111 1111 invalid 4111 1111 1111 1112",
        "https://example.com/cards",
        mask_pii=False,
    )
    cards = [match["value"] for match in result["regex_matches"] if match["label"] == "credit_card"]

    assert cards == ["4111 1111 1111 1111"]


@pytest.mark.parametrize(
    ("environment_value", "mask_pii", "expected_value"),
    [("true", False, "demo@example.com"), ("false", True, "d***o@example.com")],
)
def test_explicit_regex_mask_setting_precedes_environment(
    monkeypatch: pytest.MonkeyPatch,
    environment_value: str,
    mask_pii: bool,
    expected_value: str,
) -> None:
    monkeypatch.setenv("REGEX_PII_MASK", environment_value)

    result = extraction.extract_regex_entities(
        "Email demo@example.com",
        "https://example.com/masking",
        mask_pii=mask_pii,
    )
    email = next(match for match in result["regex_matches"] if match["label"] == "email")

    assert email["value"] == expected_value


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
