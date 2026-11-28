from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as extractor_lib
from tldw_Server_API.app.core.Web_Scraping import extraction
from tldw_Server_API.app.core.Web_Scraping.extraction import pipeline


def test_cluster_env_overrides(monkeypatch):
    html = """
    <html>
      <body>
        <p>First block of meaningful text for clustering.</p>
        <p>Second block with enough words to pass filters.</p>
      </body>
    </html>
    """
    monkeypatch.setenv("SIM_THRESHOLD", "0.15")
    monkeypatch.setenv("WORD_COUNT_THRESHOLD", "2")
    monkeypatch.setenv("CLUSTER_LINKAGE", "complete")

    result = extractor_lib.extract_cluster_entities(
        html,
        "https://example.com",
        cluster_settings={"method": "hierarchical"},
    )

    assert result["extraction_successful"] is True
    assert result["cluster_similarity_threshold"] == 0.15
    assert result["cluster_word_threshold"] == 2
    assert result["cluster_linkage"] == "complete"


def test_extractor_max_workers_env(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping.orchestration.executor import _load_worker_count

    monkeypatch.setenv("EXTRACTOR_MAX_WORKERS", "3")

    async def recursive(*args, **kwargs):
        return ["ok"]

    monkeypatch.setattr(extractor_lib, "recursive_scrape", recursive)
    assert extractor_lib._extractor_max_workers() == 3
    assert _load_worker_count() == "3"
    assert extractor_lib.sync_recursive_scrape("https://example.com", 1, 1) == ["ok"]


def test_regex_pii_mask_flag_accepts_y(monkeypatch):
    monkeypatch.setenv("REGEX_PII_MASK", "y")
    result = extraction.extract_regex_entities(
        "Contact demo@example.com",
        "https://example.com/contact",
    )
    email = next(match for match in result["regex_matches"] if match["label"] == "email")

    assert email["value"] == "d***o@example.com"


def test_clear_caches_flag_does_not_restore_per_request_global_resets(monkeypatch):
    monkeypatch.setenv("EXTRACTOR_CLEAR_CACHES", "y")
    assert not hasattr(pipeline, "_should_clear_caches")
