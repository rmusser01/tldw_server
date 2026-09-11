import json
from contextlib import contextmanager

import pytest

import tldw_Server_API.app.services.enhanced_web_scraping_service as enhanced_svc_mod
from tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib import ContentMetadataHandler
from tldw_Server_API.app.services.enhanced_web_scraping_service import WebScrapingService


class _MetricsStub:
    def set_gauge(self, *args, **kwargs):
        return None

    def observe(self, *args, **kwargs):
        return None

    def increment(self, *args, **kwargs):
        return None


class _FakeDB:
    def __init__(self):
        self.calls: list[dict] = []
        self.closed = False

    def add_media_with_keywords(self, **kwargs):
        self.calls.append(kwargs)
        idx = len(self.calls)
        return idx, f"uuid-{idx}", "ok"

    def close_connection(self):
        self.closed = True


class _LoggerStub:
    def __init__(self):
        self.warnings: list[str] = []

    def warning(self, message, *args, **kwargs):  # noqa: ARG002
        self.warnings.append(str(message).format(*args))

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


@pytest.mark.unit
def test_content_metadata_handler_falls_back_for_deeply_nested_envelope():
    nested_value = "[" * 2000 + "0" + "]" * 2000
    content = f'[METADATA]{{"value":{nested_value}}}[/METADATA]'

    assert ContentMetadataHandler.has_metadata(content) is False
    assert ContentMetadataHandler.strip_metadata(content) == content


@pytest.mark.unit
@pytest.mark.parametrize(
    ("nested_arrays", "expected_metadata"),
    [
        (63, True),
        (64, False),
    ],
)
def test_content_metadata_handler_enforces_fixed_nesting_limit_across_public_methods(
    nested_arrays,
    expected_metadata,
):
    nested_value = "[" * nested_arrays + "0" + "]" * nested_arrays
    content = f'[METADATA]{{"value":{nested_value}}}[/METADATA]\nArticle body'

    metadata, clean_content = ContentMetadataHandler.extract_metadata(content)

    assert ContentMetadataHandler.has_metadata(content) is expected_metadata
    if expected_metadata:
        assert metadata
        assert clean_content == "Article body"
        assert ContentMetadataHandler.strip_metadata(content) == "Article body"
    else:
        assert metadata == {}
        assert clean_content == content
        assert ContentMetadataHandler.strip_metadata(content) == content


@pytest.mark.unit
def test_content_metadata_handler_ignores_json_delimiters_inside_escaped_strings():
    metadata = {
        "literal": 'brackets [{]} and an escaped quote " plus a backslash \\',
    }
    content = f"[METADATA]{json.dumps(metadata)}[/METADATA]\nArticle body"

    extracted, clean_content = ContentMetadataHandler.extract_metadata(content)

    assert ContentMetadataHandler.has_metadata(content) is True
    assert extracted == metadata
    assert clean_content == "Article body"
    assert ContentMetadataHandler.strip_metadata(content) == "Article body"


@pytest.mark.unit
def test_content_metadata_handler_ignores_non_string_input():
    assert ContentMetadataHandler.has_metadata(None) is False
    assert ContentMetadataHandler.strip_metadata(None) is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_store_persistent_persists_crawl_metadata_when_available(monkeypatch):
    service = WebScrapingService()
    fake_db = _FakeDB()

    @contextmanager
    def _fake_managed_media_database(*args, **kwargs):  # noqa: ARG001
        try:
            yield fake_db
        finally:
            fake_db.close_connection()

    monkeypatch.setattr(
        enhanced_svc_mod,
        "get_user_media_db_path",
        # Test-only path; the managed database is mocked, so no filesystem write occurs.
        lambda _: "/tmp/test-media.db",  # nosec B108
    )
    monkeypatch.setattr(
        enhanced_svc_mod,
        "managed_media_database",
        _fake_managed_media_database,
    )
    monkeypatch.setattr(
        enhanced_svc_mod,
        "create_media_database",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("legacy raw factory should not be used")),
        raising=False,
    )
    monkeypatch.setattr(
        enhanced_svc_mod,
        "get_metrics_registry",
        lambda: _MetricsStub(),
    )

    result = {
        "method": "Recursive Scraping",
        "articles": [
            {
                "url": "https://example.com/article",
                "title": "Example",
                "author": "Author",
                "content": "<html><body>content</body></html>",
                "summary": "summary",
                "extraction_successful": True,
                "metadata": {
                    "crawl_depth": "2",
                    "crawl_parent_url": "https://example.com",
                    "crawl_score": "0.75",
                },
            }
        ],
    }

    persisted = await service._store_persistent(
        result=result,
        keywords="k1,k2",
        user_id=7,
        perform_chunking=False,
        chunking_mode=None,
        auto_chunking_goal="balanced",
        auto_chunking_use_llm=False,
    )

    assert persisted["status"] == "persist-ok"
    assert persisted["stored_articles"] == 1
    assert len(fake_db.calls) == 1
    assert fake_db.closed is True

    safe_metadata_raw = fake_db.calls[0]["safe_metadata"]
    safe_metadata = json.loads(safe_metadata_raw)
    assert safe_metadata["crawl_depth"] == 2
    assert safe_metadata["crawl_parent_url"] == "https://example.com"
    assert safe_metadata["crawl_score"] == pytest.approx(0.75)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_store_persistent_skips_articles_without_body_content(monkeypatch):
    service = WebScrapingService()
    fake_db = _FakeDB()
    logger_stub = _LoggerStub()

    @contextmanager
    def _fake_managed_media_database(*args, **kwargs):  # noqa: ARG001
        try:
            yield fake_db
        finally:
            fake_db.close_connection()

    monkeypatch.setattr(
        enhanced_svc_mod,
        "get_user_media_db_path",
        # Test-only path; the managed database is mocked, so no filesystem write occurs.
        lambda _: "/tmp/test-media.db",  # nosec B108
    )
    monkeypatch.setattr(
        enhanced_svc_mod,
        "managed_media_database",
        _fake_managed_media_database,
    )
    monkeypatch.setattr(
        enhanced_svc_mod,
        "get_metrics_registry",
        lambda: _MetricsStub(),
    )
    monkeypatch.setattr(enhanced_svc_mod, "logger", logger_stub)

    result = {
        "method": "Individual URLs",
        "articles": [
            {
                "url": "https://example.com/valid",
                "content": "Actual body content",
                "extraction_successful": True,
            },
            {
                "url": "https://example.com/literal-markers",
                "content": "Documentation mentions [METADATA] and [/METADATA]",
                "extraction_successful": True,
            },
            {
                "url": "https://example.com/wrapped",
                "content": '[METADATA]{"source":"old"}[/METADATA]\nWrapped body',
                "extraction_successful": True,
            },
            {
                "url": "https://example.com/missing",
                "content": None,
                "extraction_successful": True,
            },
            {
                "url": "https://example.com/non-string",
                "content": 42,
                "extraction_successful": True,
            },
            {
                "url": "https://user:password@example.com/blank?token=secret#fragment",
                "content": "  \n",
                "extraction_successful": True,
            },
            {
                "url": "https://example.com/envelope",
                "content": '[METADATA]{"url":"https://example.com/envelope"}[/METADATA]\n  ',
                "extraction_successful": True,
            },
            {
                "url": "https://example.com/crafted-envelope",
                "content": '[METADATA]{"note":"[/METADATA]"}[/METADATA]\n ',
                "extraction_successful": True,
            },
        ],
    }

    persisted = await service._store_persistent(
        result=result,
        keywords="",
        user_id=7,
        perform_chunking=False,
        chunking_mode=None,
        auto_chunking_goal="balanced",
        auto_chunking_use_llm=False,
    )

    assert persisted["status"] == "persist-ok"
    assert persisted["stored_articles"] == 3
    assert persisted["media_ids"] == [1, 2, 3]
    assert len(fake_db.calls) == 3
    assert "Actual body content" in fake_db.calls[0]["content"]
    assert "Documentation mentions [METADATA] and [/METADATA]" in fake_db.calls[1]["content"]
    assert fake_db.calls[2]["content"].count("[METADATA]") == 1
    assert "Wrapped body" in fake_db.calls[2]["content"]
    assert '"source":"old"' not in fake_db.calls[2]["content"]
    assert persisted["errors"] == [
        "No extracted content: https://example.com/missing",
        "No extracted content: https://example.com/non-string",
        "No extracted content: https://user:password@example.com/blank?token=secret#fragment",
        "No extracted content: https://example.com/envelope",
        "No extracted content: https://example.com/crafted-envelope",
    ]
    warning_text = "\n".join(logger_stub.warnings)
    assert "password" not in warning_text
    assert "token=secret" not in warning_text
    assert "#fragment" not in warning_text
