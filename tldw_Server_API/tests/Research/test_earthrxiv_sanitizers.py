import pytest

from tldw_Server_API.app.core.Third_Party import EarthRxiv as earthrxiv


pytestmark = pytest.mark.unit


def test_search_items_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise RuntimeError("earthrxiv token at /private/earthrxiv.key")

    monkeypatch.setattr(earthrxiv, "fetch_json", fail_fetch_json)

    items, total, error = earthrxiv.search_items("topic", 1, 10)

    assert items is None
    assert total == 0
    assert error == "EarthArXiv request failed."
    assert "earthrxiv token" not in error
    assert "/private/earthrxiv.key" not in error


def test_search_items_preserves_timeout_classification(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise TimeoutError("timed out at /private/earthrxiv-timeout.key")

    monkeypatch.setattr(earthrxiv, "fetch_json", fail_fetch_json)

    items, total, error = earthrxiv.search_items("topic", 1, 10)

    assert items is None
    assert total == 0
    assert error == "EarthArXiv request timed out."
    assert "timed out at" not in error
    assert "/private/earthrxiv-timeout.key" not in error


def test_get_item_by_id_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise RuntimeError("earthrxiv id token at /private/earthrxiv-id.key")

    monkeypatch.setattr(earthrxiv, "fetch_json", fail_fetch_json)

    item, error = earthrxiv.get_item_by_id("abc123")

    assert item is None
    assert error == "EarthArXiv request failed."
    assert "earthrxiv id token" not in error
    assert "/private/earthrxiv-id.key" not in error


def test_get_item_by_id_preserves_timeout_classification(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise TimeoutError("timed out at /private/earthrxiv-id-timeout.key")

    monkeypatch.setattr(earthrxiv, "fetch_json", fail_fetch_json)

    item, error = earthrxiv.get_item_by_id("abc123")

    assert item is None
    assert error == "EarthArXiv request timed out."
    assert "timed out at" not in error
    assert "/private/earthrxiv-id-timeout.key" not in error


def test_get_item_by_doi_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch(**_kwargs):
        raise RuntimeError("earthrxiv doi token at /private/earthrxiv-doi.key")

    monkeypatch.setattr(earthrxiv, "fetch", fail_fetch)

    item, error = earthrxiv.get_item_by_doi("10.123/example")

    assert item is None
    assert error == "EarthArXiv request failed."
    assert "earthrxiv doi token" not in error
    assert "/private/earthrxiv-doi.key" not in error


def test_get_item_by_doi_preserves_timeout_classification(monkeypatch):
    def fail_fetch(**_kwargs):
        raise TimeoutError("timed out at /private/earthrxiv-doi-timeout.key")

    monkeypatch.setattr(earthrxiv, "fetch", fail_fetch)

    item, error = earthrxiv.get_item_by_doi("10.123/example")

    assert item is None
    assert error == "EarthArXiv request timed out."
    assert "timed out at" not in error
    assert "/private/earthrxiv-doi-timeout.key" not in error
