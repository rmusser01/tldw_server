from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

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
    ]
    assert legacy.clear_extraction_caches is extraction.clear_extraction_caches
    assert legacy.get_extraction_cache_stats is extraction.get_extraction_cache_stats


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
