from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as ael
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
