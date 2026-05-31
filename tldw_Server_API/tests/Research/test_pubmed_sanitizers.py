import pytest

from tldw_Server_API.app.core.Third_Party import PubMed as pubmed


pytestmark = pytest.mark.unit


def test_search_pubmed_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise RuntimeError("pubmed token at /private/pubmed.key")

    monkeypatch.setattr(pubmed, "fetch_json", fail_fetch_json)

    items, total, error = pubmed.search_pubmed("cancer", 0, 10)

    assert items is None
    assert total == 0
    assert error == "PubMed request failed."
    assert "pubmed token" not in error
    assert "/private/pubmed.key" not in error


def test_search_pubmed_preserves_timeout_classification(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise TimeoutError("timed out at /private/pubmed-timeout.key")

    monkeypatch.setattr(pubmed, "fetch_json", fail_fetch_json)

    items, total, error = pubmed.search_pubmed("cancer", 0, 10)

    assert items is None
    assert total == 0
    assert error == "PubMed request timed out."
    assert "timed out at" not in error
    assert "/private/pubmed-timeout.key" not in error


def test_get_pubmed_by_id_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise RuntimeError("pubmed by-id token at /private/pubmed-id.key")

    monkeypatch.setattr(pubmed, "fetch_json", fail_fetch_json)

    item, error = pubmed.get_pubmed_by_id("123456")

    assert item is None
    assert error == "PubMed by-id request failed."
    assert "pubmed by-id token" not in error
    assert "/private/pubmed-id.key" not in error


def test_get_pubmed_by_id_preserves_timeout_classification(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise TimeoutError("timed out at /private/pubmed-id-timeout.key")

    monkeypatch.setattr(pubmed, "fetch_json", fail_fetch_json)

    item, error = pubmed.get_pubmed_by_id("123456")

    assert item is None
    assert error == "PubMed by-id request timed out."
    assert "timed out at" not in error
    assert "/private/pubmed-id-timeout.key" not in error
