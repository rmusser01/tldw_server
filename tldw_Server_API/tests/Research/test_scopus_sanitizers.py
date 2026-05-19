import pytest

from tldw_Server_API.app.core.Third_Party import Elsevier_Scopus as scopus


pytestmark = pytest.mark.unit


def test_search_scopus_sanitizes_fetch_failures(monkeypatch):
    monkeypatch.setenv("ELSEVIER_API_KEY", "test-key")

    def fail_fetch_json(**_kwargs):
        raise RuntimeError("scopus token at /private/scopus.key")

    monkeypatch.setattr(scopus, "fetch_json", fail_fetch_json)

    items, total, error = scopus.search_scopus("topic", 0, 10)

    assert items is None
    assert total == 0
    assert error == "Scopus request failed."
    assert "scopus token" not in error
    assert "/private/scopus.key" not in error


def test_search_scopus_preserves_timeout_classification(monkeypatch):
    monkeypatch.setenv("ELSEVIER_API_KEY", "test-key")

    def fail_fetch_json(**_kwargs):
        raise TimeoutError("timed out at /private/scopus-timeout.key")

    monkeypatch.setattr(scopus, "fetch_json", fail_fetch_json)

    items, total, error = scopus.search_scopus("topic", 0, 10)

    assert items is None
    assert total == 0
    assert error == "Scopus request timed out."
    assert "timed out at" not in error
    assert "/private/scopus-timeout.key" not in error


def test_get_scopus_by_doi_sanitizes_fetch_failures(monkeypatch):
    monkeypatch.setenv("ELSEVIER_API_KEY", "test-key")

    def fail_fetch(**_kwargs):
        raise RuntimeError("scopus doi token at /private/scopus-doi.key")

    monkeypatch.setattr(scopus, "fetch", fail_fetch)

    item, error = scopus.get_scopus_by_doi("10.123/example")

    assert item is None
    assert error == "Scopus request failed."
    assert "scopus doi token" not in error
    assert "/private/scopus-doi.key" not in error


def test_get_scopus_by_doi_preserves_timeout_classification(monkeypatch):
    monkeypatch.setenv("ELSEVIER_API_KEY", "test-key")

    def fail_fetch(**_kwargs):
        raise TimeoutError("timed out at /private/scopus-doi-timeout.key")

    monkeypatch.setattr(scopus, "fetch", fail_fetch)

    item, error = scopus.get_scopus_by_doi("10.123/example")

    assert item is None
    assert error == "Scopus request timed out."
    assert "timed out at" not in error
    assert "/private/scopus-doi-timeout.key" not in error
