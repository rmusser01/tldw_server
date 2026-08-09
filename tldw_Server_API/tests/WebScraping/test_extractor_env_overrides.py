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
    monkeypatch.setenv("EXTRACTOR_MAX_WORKERS", "3")

    async def _fake_recursive(*_args, **_kwargs):
        return ["ok"]

    monkeypatch.setattr(extractor_lib, "recursive_scrape", _fake_recursive)

    captured: dict[str, int | None] = {}

    class _DummyFuture:
        def __init__(self, fn):
            self._fn = fn

        def result(self):
            return self._fn()

    class _DummyExecutor:
        def __init__(self, max_workers=None):
            captured["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn):
            return _DummyFuture(fn)

    monkeypatch.setattr(extractor_lib, "ThreadPoolExecutor", _DummyExecutor)

    result = extractor_lib.sync_recursive_scrape("https://example.com", 1, 1)

    assert captured["max_workers"] == 3
    assert result == ["ok"]


def test_regex_pii_mask_flag_accepts_y(monkeypatch):
    monkeypatch.setenv("REGEX_PII_MASK", "y")
    result = extraction.extract_regex_entities(
        "Contact demo@example.com",
        "https://example.com/contact",
    )
    email = next(match for match in result["regex_matches"] if match["label"] == "email")

    assert email["value"] == "d***o@example.com"


def test_clear_caches_flag_accepts_y_end_only(monkeypatch):
    monkeypatch.setenv("EXTRACTOR_CLEAR_CACHES", "y")
    assert pipeline._should_clear_caches("start") is False
    assert pipeline._should_clear_caches("end") is True
