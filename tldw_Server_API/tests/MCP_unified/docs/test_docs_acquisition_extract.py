from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from mcp_unified.docs.acquisition.extract import available_extractors, extract_fetched_document


def test_static_html_fallback_extracts_title_sections_and_text(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    parsed = extract_fetched_document(
        url="https://example.com/docs",
        content_type="text/html",
        body=b"<html><body><h1>Guide</h1><p>SQLite FTS5 content.</p><script>skip()</script></body></html>",
    )

    assert parsed.title == "Guide"  # nosec B101
    assert parsed.document_type == "html"  # nosec B101
    assert parsed.source_path is None  # nosec B101
    assert parsed.source_url == "https://example.com/docs"  # nosec B101
    assert parsed.canonical_uri == "https://example.com/docs"  # nosec B101
    assert parsed.extraction_method == "static_html"  # nosec B101
    assert "rich_extractors_unavailable" in parsed.warnings  # nosec B101
    assert "SQLite FTS5 content." in parsed.text  # nosec B101
    assert "skip()" not in parsed.text  # nosec B101


def test_plain_text_extraction() -> None:
    parsed = extract_fetched_document(
        url="https://example.com/readme.txt",
        content_type="text/plain",
        body=b"Line one\nLine two",
    )

    assert parsed.title == "readme.txt"  # nosec B101
    assert parsed.document_type == "text"  # nosec B101
    assert parsed.source_path is None  # nosec B101
    assert parsed.source_url == "https://example.com/readme.txt"  # nosec B101
    assert parsed.extraction_method == "text"  # nosec B101
    assert parsed.text == "Line one\nLine two"  # nosec B101


def test_trafilatura_is_used_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    module = SimpleNamespace(extract=lambda text, include_comments=False, include_tables=True: "Rich body")

    def fake_import(name: str) -> object:
        if name == "trafilatura":
            return module
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    parsed = extract_fetched_document(
        url="https://example.com/docs",
        content_type="text/html",
        body=b"<h1>Guide</h1><p>Rich body</p>",
    )

    assert parsed.title == "Guide"  # nosec B101
    assert parsed.extraction_method == "trafilatura"  # nosec B101
    assert parsed.text == "Rich body"  # nosec B101
    assert parsed.source_url == "https://example.com/docs"  # nosec B101


def test_available_extractors_uses_lazy_import_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import(name: str) -> object:
        if name == "bs4":
            return SimpleNamespace(BeautifulSoup=object)
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    assert available_extractors() == ["beautifulsoup", "static_html", "text"]  # nosec B101
