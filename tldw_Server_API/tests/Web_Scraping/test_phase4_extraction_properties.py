from __future__ import annotations

import dataclasses
import inspect
from concurrent.futures import ThreadPoolExecutor

from hypothesis import given, settings
from hypothesis import strategies as st

from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as legacy
from tldw_Server_API.app.core.Web_Scraping import extraction
from tldw_Server_API.app.core.Web_Scraping.extraction import caches
from tldw_Server_API.app.core.Web_Scraping.extraction.dependencies import (
    build_default_dependencies,
)
from tldw_Server_API.app.core.Web_Scraping.extraction.strategies import (
    cluster as cluster_strategy,
)

_CLUSTER_HTML = """
<html><head><title>Cluster title</title></head><body>
<p>Alpha research system improves energy studies and accuracy.</p>
<p>Alpha research system reports energy savings and accuracy.</p>
<p>Subscribe for unrelated newsletters and marketing promotions.</p>
</body></html>
"""


def test_cluster_strategy_legacy_and_facade_exports_are_direct_aliases() -> None:
    expected_signature = (
        "(html_text: 'str', url: 'str', *, cluster_settings: 'Optional[dict[str, Any]]' = None) -> 'dict[str, Any]'"
    )

    assert extraction.extract_cluster_entities is cluster_strategy.extract_cluster_entities
    assert legacy.extract_cluster_entities is cluster_strategy.extract_cluster_entities
    assert str(inspect.signature(cluster_strategy.extract_cluster_entities)) == expected_signature
    assert str(inspect.signature(extraction.extract_cluster_entities)) == expected_signature
    assert str(inspect.signature(legacy.extract_cluster_entities)) == expected_signature


def test_cluster_settings_precede_environment_thresholds(monkeypatch) -> None:
    monkeypatch.setenv("SIM_THRESHOLD", "0.99")
    monkeypatch.setenv("WORD_COUNT_THRESHOLD", "99")
    monkeypatch.setenv("CLUSTER_LINKAGE", "complete")

    result = extraction.extract_cluster_entities(
        _CLUSTER_HTML,
        "https://example.com/thresholds",
        cluster_settings={
            "method": "hierarchical",
            "similarity_threshold": 0.1,
            "min_word_count": 1,
            "cluster_linkage": "single",
            "min_block_chars": 1,
        },
    )

    assert result["cluster_similarity_threshold"] == 0.1
    assert result["cluster_word_threshold"] == 1
    assert result["cluster_linkage"] == "single"


def test_cluster_uses_call_time_dependencies_for_metrics_and_cancellation(monkeypatch) -> None:
    counters: list[tuple[str, dict[str, str]]] = []
    checkpoints: list[None] = []
    default_dependencies = build_default_dependencies()
    dependencies = dataclasses.replace(
        default_dependencies,
        increment_counter=lambda name, *, labels: counters.append((name, labels)),
        cancellation_checkpoint=lambda: checkpoints.append(None),
    )
    calls = 0

    def build_dependencies_at_call_time():
        nonlocal calls
        calls += 1
        return dependencies

    monkeypatch.setattr(cluster_strategy, "build_default_dependencies", build_dependencies_at_call_time)

    result = extraction.extract_cluster_entities(
        _CLUSTER_HTML,
        "https://example.com/dependencies",
        cluster_settings={"min_block_chars": 1, "min_word_count": 1},
    )

    assert result["extraction_successful"] is True
    assert calls == 1
    assert checkpoints
    status_counters = [entry for entry in counters if entry[0] == "extraction_cluster_total"]
    assert status_counters == [
        ("extraction_cluster_total", {"status": "started"}),
        ("extraction_cluster_total", {"status": "success"}),
    ]


def test_cluster_error_values_are_stable_for_each_failure_state(monkeypatch) -> None:
    assert extraction.extract_cluster_entities("", "https://example.com/empty")["cluster_error"] == "cluster_empty_html"

    monkeypatch.setattr(cluster_strategy, "_extract_cluster_blocks", lambda *_args, **_kwargs: [])
    assert (
        extraction.extract_cluster_entities("<p>x</p>", "https://example.com/no-blocks")["cluster_error"]
        == "cluster_no_blocks"
    )

    monkeypatch.setattr(cluster_strategy, "_extract_cluster_blocks", lambda *_args, **_kwargs: ["content"])
    monkeypatch.setattr(cluster_strategy, "_cluster_embedding", lambda *_args, **_kwargs: [1.0])
    monkeypatch.setattr(cluster_strategy, "_cluster_blocks_greedy", lambda *_args, **_kwargs: [])
    assert (
        extraction.extract_cluster_entities("<p>x</p>", "https://example.com/no-clusters")["cluster_error"]
        == "cluster_no_clusters"
    )

    monkeypatch.setattr(
        cluster_strategy,
        "_cluster_blocks_greedy",
        lambda *_args, **_kwargs: [{"members": [(0, "", 1.0)], "sum_vec": [1.0], "centroid": [1.0], "total_chars": 0}],
    )
    assert (
        extraction.extract_cluster_entities("<p>x</p>", "https://example.com/empty-content")["cluster_error"]
        == "cluster_empty_content"
    )


def test_cluster_cache_lru_eviction_preserves_recently_read_entry(monkeypatch) -> None:
    monkeypatch.setattr(caches, "_CLUSTER_EMBED_CACHE_MAX", 2)
    caches.clear_extraction_caches()

    caches._cluster_cache_put("old", [1.0])
    caches._cluster_cache_put("recent", [2.0])
    assert caches._cluster_cache_get("old") == [1.0]
    caches._cluster_cache_put("new", [3.0])

    assert caches._cluster_cache_get("recent") is None
    assert caches._cluster_cache_get("old") == [1.0]
    assert caches._cluster_cache_get("new") == [3.0]


def test_cluster_cache_is_safe_for_concurrent_reads_and_writes(monkeypatch) -> None:
    monkeypatch.setattr(caches, "_CLUSTER_EMBED_CACHE_MAX", 16)
    caches.clear_extraction_caches()

    def write_and_read(index: int) -> list[float] | None:
        key = f"key-{index % 24}"
        vector = [float(index), float(index + 1)]
        caches._cluster_cache_put(key, vector)
        vector.append(-1.0)
        return caches._cluster_cache_get(key)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(write_and_read, range(128)))

    assert all(result is None or len(result) == 2 for result in results)
    assert caches.get_extraction_cache_stats()["cluster_embedding_cache_size"] <= 16


def test_cluster_cache_returns_vector_copies() -> None:
    caches.clear_extraction_caches()
    original = [1.0, 2.0]
    caches._cluster_cache_put("vector", original)
    original.append(3.0)

    cached = caches._cluster_cache_get("vector")
    assert cached == [1.0, 2.0]
    assert cached is not None
    cached.append(4.0)
    assert caches._cluster_cache_get("vector") == [1.0, 2.0]


@settings(max_examples=25, deadline=None)
@given(st.lists(st.text(min_size=1, max_size=16), min_size=1, max_size=48, unique=True))
def test_cluster_cache_size_never_exceeds_configured_maximum(keys: list[str]) -> None:
    maximum = 7
    original_maximum = caches._CLUSTER_EMBED_CACHE_MAX
    try:
        caches._CLUSTER_EMBED_CACHE_MAX = maximum
        caches.clear_extraction_caches()

        for index, key in enumerate(keys):
            caches._cluster_cache_put(key, [float(index)])
            assert caches.get_extraction_cache_stats()["cluster_embedding_cache_size"] <= maximum
    finally:
        caches._CLUSTER_EMBED_CACHE_MAX = original_maximum
        caches.clear_extraction_caches()
