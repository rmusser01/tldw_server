import pytest

from tldw_Server_API.app.core.Third_Party import Crossref as crossref


pytestmark = pytest.mark.unit


def test_search_crossref_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise RuntimeError("crossref token at /private/crossref-search.key")

    monkeypatch.setattr(crossref, "fetch_json", fail_fetch_json)

    items, total, error = crossref.search_crossref("retrieval", offset=0, limit=1)

    assert items is None
    assert total == 0
    assert error == "Crossref request failed."
    assert "crossref token" not in error
    assert "/private/crossref-search.key" not in error


def test_search_crossref_preserves_timeout_classification(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise TimeoutError("timed out at /private/crossref-timeout.key")

    monkeypatch.setattr(crossref, "fetch_json", fail_fetch_json)

    items, total, error = crossref.search_crossref("retrieval", offset=0, limit=1)

    assert items is None
    assert total == 0
    assert error == "Crossref request timed out."
    assert "timed out at" not in error
    assert "/private/crossref-timeout.key" not in error


def test_get_crossref_by_doi_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch(**_kwargs):
        raise RuntimeError("crossref DOI token at /private/crossref-doi.key")

    monkeypatch.setattr(crossref, "fetch", fail_fetch)

    item, error = crossref.get_crossref_by_doi("10.private/request")

    assert item is None
    assert error == "Crossref request failed."
    assert "crossref DOI token" not in error
    assert "/private/crossref-doi.key" not in error
    assert "10.private/request" not in error


def test_get_crossref_by_doi_preserves_timeout_classification(monkeypatch):
    def fail_fetch(**_kwargs):
        raise TimeoutError("timed out at /private/crossref-doi-timeout.key")

    monkeypatch.setattr(crossref, "fetch", fail_fetch)

    item, error = crossref.get_crossref_by_doi("10.private/request")

    assert item is None
    assert error == "Crossref request timed out."
    assert "timed out at" not in error
    assert "/private/crossref-doi-timeout.key" not in error
    assert "10.private/request" not in error
