import pytest

from tldw_Server_API.app.core.Third_Party import Figshare as figshare


pytestmark = pytest.mark.unit


def test_search_articles_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise RuntimeError("figshare token at /private/figshare.key")

    monkeypatch.setattr(figshare, "fetch_json", fail_fetch_json)

    items, total, error = figshare.search_articles("climate", 1, 10)

    assert items is None
    assert total == 0
    assert error == "Figshare request failed."
    assert "figshare token" not in error
    assert "/private/figshare.key" not in error


def test_search_articles_preserves_timeout_classification(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise TimeoutError("timed out at /private/figshare-timeout.key")

    monkeypatch.setattr(figshare, "fetch_json", fail_fetch_json)

    items, total, error = figshare.search_articles("climate", 1, 10)

    assert items is None
    assert total == 0
    assert error == "Figshare request timed out."
    assert "timed out at" not in error
    assert "/private/figshare-timeout.key" not in error


def test_get_article_by_id_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch(**_kwargs):
        raise RuntimeError("figshare article token at /private/figshare-id.key")

    monkeypatch.setattr(figshare, "fetch", fail_fetch)

    item, error = figshare.get_article_by_id("42")

    assert item is None
    assert error == "Figshare article request failed."
    assert "figshare article token" not in error
    assert "/private/figshare-id.key" not in error
    assert "42" not in error


def test_get_article_by_id_preserves_timeout_classification(monkeypatch):
    def fail_fetch(**_kwargs):
        raise TimeoutError("timed out at /private/figshare-id-timeout.key")

    monkeypatch.setattr(figshare, "fetch", fail_fetch)

    item, error = figshare.get_article_by_id("42")

    assert item is None
    assert error == "Figshare article request timed out."
    assert "timed out at" not in error
    assert "/private/figshare-id-timeout.key" not in error


def test_get_article_raw_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise RuntimeError("figshare raw token at /private/figshare-raw.key")

    monkeypatch.setattr(figshare, "fetch_json", fail_fetch_json)

    item, error = figshare.get_article_raw("42")

    assert item is None
    assert error == "Figshare raw article request failed."
    assert "figshare raw token" not in error
    assert "/private/figshare-raw.key" not in error
    assert "42" not in error


def test_get_article_raw_preserves_timeout_classification(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise TimeoutError("timed out at /private/figshare-raw-timeout.key")

    monkeypatch.setattr(figshare, "fetch_json", fail_fetch_json)

    item, error = figshare.get_article_raw("42")

    assert item is None
    assert error == "Figshare raw article request timed out."
    assert "timed out at" not in error
    assert "/private/figshare-raw-timeout.key" not in error


def test_get_article_files_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise RuntimeError("figshare files token at /private/figshare-files.key")

    monkeypatch.setattr(figshare, "fetch_json", fail_fetch_json)

    items, error = figshare.get_article_files("42")

    assert items is None
    assert error == "Figshare file request failed."
    assert "figshare files token" not in error
    assert "/private/figshare-files.key" not in error
    assert "42" not in error


def test_get_article_files_preserves_timeout_classification(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise TimeoutError("timed out at /private/figshare-files-timeout.key")

    monkeypatch.setattr(figshare, "fetch_json", fail_fetch_json)

    items, error = figshare.get_article_files("42")

    assert items is None
    assert error == "Figshare file request timed out."
    assert "timed out at" not in error
    assert "/private/figshare-files-timeout.key" not in error


def test_get_article_by_doi_sanitizes_lookup_failures(monkeypatch):
    def fail_search_articles(*_args, **_kwargs):
        raise RuntimeError("figshare doi token at /private/figshare-doi.key")

    monkeypatch.setattr(figshare, "search_articles", fail_search_articles)

    item, error = figshare.get_article_by_doi("10.6084/m9.figshare.123")

    assert item is None
    assert error == "Figshare DOI request failed."
    assert "figshare doi token" not in error
    assert "/private/figshare-doi.key" not in error
    assert "10.6084/m9.figshare.123" not in error


def test_get_article_by_doi_preserves_timeout_classification(monkeypatch):
    def fail_search_articles(*_args, **_kwargs):
        raise TimeoutError("timed out at /private/figshare-doi-timeout.key")

    monkeypatch.setattr(figshare, "search_articles", fail_search_articles)

    item, error = figshare.get_article_by_doi("10.6084/m9.figshare.123")

    assert item is None
    assert error == "Figshare DOI request timed out."
    assert "timed out at" not in error
    assert "/private/figshare-doi-timeout.key" not in error


def test_oai_raw_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch(**_kwargs):
        raise RuntimeError("figshare oai token at /private/figshare-oai.key")

    monkeypatch.setattr(figshare, "fetch", fail_fetch)

    content, content_type, error = figshare.oai_raw({"verb": "Identify"})

    assert content is None
    assert content_type is None
    assert error == "Figshare OAI-PMH request failed."
    assert "figshare oai token" not in error
    assert "/private/figshare-oai.key" not in error


def test_oai_raw_preserves_timeout_classification(monkeypatch):
    def fail_fetch(**_kwargs):
        raise TimeoutError("timed out at /private/figshare-oai-timeout.key")

    monkeypatch.setattr(figshare, "fetch", fail_fetch)

    content, content_type, error = figshare.oai_raw({"verb": "Identify"})

    assert content is None
    assert content_type is None
    assert error == "Figshare OAI-PMH request timed out."
    assert "timed out at" not in error
    assert "/private/figshare-oai-timeout.key" not in error
