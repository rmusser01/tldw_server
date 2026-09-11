import pytest

from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as ael
from tldw_Server_API.app.core.Web_Scraping.cluster_settings import CLUSTER_MIN_WORDS
from tldw_Server_API.app.core.Web_Scraping.extraction import caches
from tldw_Server_API.app.core.Web_Scraping.extraction.strategies import (
    cluster as cluster_strategy,
)


def test_cluster_hierarchical_prefers_largest_cluster(monkeypatch):
    html = """
    <html><body>
      <p>Alpha system improves research accuracy for energy studies.</p>
      <p>Alpha system results show better energy savings.</p>
      <p>Completely different topic unrelated to energy.</p>
    </body></html>
    """

    def fake_assignments(_vectors, similarity_threshold, linkage):
        assert linkage == "single"
        return [0, 0, 1]

    monkeypatch.setattr(cluster_strategy, "_cluster_assignments_hierarchical", fake_assignments)

    result = ael.extract_cluster_entities(
        html,
        "https://example.com",
        cluster_settings={
            "method": "hierarchical",
            "linkage": "single",
            "prefilter_threshold": 0.0,
            "min_block_chars": 10,
            "min_word_count": 1,
        },
    )

    assert result["extraction_successful"] is True
    assert result.get("cluster_method") == "hierarchical"
    assert result.get("cluster_block_count") == 2
    content = result.get("content") or ""
    assert "Alpha system improves" in content
    assert "Alpha system results" in content
    assert "Completely different" not in content


def test_cluster_embedding_cache_separates_dimensions() -> None:
    caches.clear_extraction_caches()

    four_dimensions = cluster_strategy._cluster_embedding("same text", 4)
    nine_dimensions = cluster_strategy._cluster_embedding("same text", 9)

    assert len(four_dimensions) == 4
    assert len(nine_dimensions) == 9
    assert caches.get_extraction_cache_stats()["cluster_embedding_cache_size"] == 2


def test_cluster_threshold_preserves_explicit_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIM_THRESHOLD", "0.8")
    html = """
    <html><body>
      <p>Alpha system improves research accuracy for energy studies.</p>
      <p>Completely different words describe an unrelated subject here.</p>
    </body></html>
    """

    result = ael.extract_cluster_entities(
        html,
        "https://example.com",
        cluster_settings={
            "cluster_threshold": 0.0,
            "prefilter_threshold": 0.0,
            "min_block_chars": 1,
            "min_word_count": 1,
        },
    )

    assert result["extraction_successful"] is True
    assert result["cluster_similarity_threshold"] == 0.0


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("min_block_chars", "invalid"),
        ("max_blocks", float("inf")),
        ("prefilter_threshold", "invalid"),
        ("cluster_threshold", float("nan")),
        ("embed_dims", 0),
    ],
)
def test_invalid_cluster_numeric_settings_fall_back_without_escaping(
    setting: str,
    value: object,
) -> None:
    html = "<html><body><p>Enough article words to produce one useful extraction block.</p></body></html>"

    result = ael.extract_cluster_entities(
        html,
        "https://example.com",
        cluster_settings={setting: value},
    )

    assert result["extraction_successful"] is True


def test_zero_min_word_count_preserves_legacy_default_threshold() -> None:
    html = "<html><body><p>Enough article words to produce one useful extraction block.</p></body></html>"

    result = ael.extract_cluster_entities(
        html,
        "https://example.com",
        cluster_settings={"min_block_chars": 1, "min_word_count": 0},
    )

    assert result["extraction_successful"] is True
    assert result["cluster_word_threshold"] == CLUSTER_MIN_WORDS


def test_unsupported_hierarchical_linkage_uses_greedy_fallback() -> None:
    html = """
    <html><body>
      <p>Alpha system improves research accuracy for energy studies.</p>
      <p>Alpha system results show better energy savings for researchers.</p>
    </body></html>
    """

    result = ael.extract_cluster_entities(
        html,
        "https://example.com",
        cluster_settings={
            "method": "hierarchical",
            "linkage": "ward",
            "prefilter_threshold": 0.0,
            "min_block_chars": 1,
            "min_word_count": 1,
        },
    )

    assert result["extraction_successful"] is True
    assert result["cluster_method"] == "greedy_fallback"


def test_hierarchical_fit_failure_uses_greedy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    sklearn_cluster = pytest.importorskip("sklearn.cluster")

    class FailingClusterer:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def fit_predict(self, _distances: list[list[float]]) -> list[int]:
            raise ValueError("invalid clustering input")

    monkeypatch.setattr(sklearn_cluster, "AgglomerativeClustering", FailingClusterer)
    html = """
    <html><body>
      <p>Alpha system improves research accuracy for energy studies.</p>
      <p>Alpha system results show better energy savings for researchers.</p>
    </body></html>
    """

    result = ael.extract_cluster_entities(
        html,
        "https://example.com",
        cluster_settings={
            "method": "hierarchical",
            "prefilter_threshold": 0.0,
            "min_block_chars": 1,
            "min_word_count": 1,
        },
    )

    assert result["extraction_successful"] is True
    assert result["cluster_method"] == "greedy_fallback"


def test_cluster_max_blocks_is_bounded() -> None:
    paragraphs = "".join(
        f"<p>Paragraph {index} contains enough words for bounded hierarchical clustering.</p>"
        for index in range(cluster_strategy._CLUSTER_MAX_BLOCKS + 20)
    )

    result = ael.extract_cluster_entities(
        f"<html><body>{paragraphs}</body></html>",
        "https://example.com",
        cluster_settings={
            "max_blocks": cluster_strategy._CLUSTER_MAX_BLOCKS * 10,
            "min_block_chars": 1,
            "min_word_count": 1,
        },
    )

    assert result["extraction_successful"] is True
    assert result["cluster_total_blocks"] <= cluster_strategy._CLUSTER_MAX_BLOCKS
