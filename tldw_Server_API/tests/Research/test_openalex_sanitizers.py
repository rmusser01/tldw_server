import pytest

from tldw_Server_API.app.core.Third_Party import OpenAlex as openalex


pytestmark = pytest.mark.unit


def test_search_openalex_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise RuntimeError("openalex token at /private/openalex.key")

    monkeypatch.setattr(openalex, "fetch_json", fail_fetch_json)

    items, total, error = openalex.search_openalex("topic", 0, 10)

    assert items is None
    assert total == 0
    assert error == "OpenAlex request failed."
    assert "openalex token" not in error
    assert "/private/openalex.key" not in error


def test_search_openalex_preserves_timeout_classification(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise TimeoutError("timed out at /private/openalex-timeout.key")

    monkeypatch.setattr(openalex, "fetch_json", fail_fetch_json)

    items, total, error = openalex.search_openalex("topic", 0, 10)

    assert items is None
    assert total == 0
    assert error == "OpenAlex request timed out."
    assert "timed out at" not in error
    assert "/private/openalex-timeout.key" not in error


def test_get_openalex_by_doi_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch(**_kwargs):
        raise RuntimeError("openalex doi token at /private/openalex-doi.key")

    monkeypatch.setattr(openalex, "fetch", fail_fetch)

    item, error = openalex.get_openalex_by_doi("10.123/example")

    assert item is None
    assert error == "OpenAlex request failed."
    assert "openalex doi token" not in error
    assert "/private/openalex-doi.key" not in error


def test_get_openalex_by_doi_preserves_timeout_classification(monkeypatch):
    def fail_fetch(**_kwargs):
        raise TimeoutError("timed out at /private/openalex-doi-timeout.key")

    monkeypatch.setattr(openalex, "fetch", fail_fetch)

    item, error = openalex.get_openalex_by_doi("10.123/example")

    assert item is None
    assert error == "OpenAlex request timed out."
    assert "timed out at" not in error
    assert "/private/openalex-doi-timeout.key" not in error
