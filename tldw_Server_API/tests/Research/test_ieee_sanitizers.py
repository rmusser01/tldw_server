import pytest

from tldw_Server_API.app.core.Third_Party import IEEE_Xplore as ieee


pytestmark = pytest.mark.unit


def test_search_ieee_sanitizes_fetch_failures(monkeypatch):
    monkeypatch.setenv("IEEE_API_KEY", "test-key")

    def fail_fetch_json(**_kwargs):
        raise RuntimeError("ieee token at /private/ieee.key")

    monkeypatch.setattr(ieee, "fetch_json", fail_fetch_json)

    items, total, error = ieee.search_ieee("topic", 0, 10)

    assert items is None
    assert total == 0
    assert error == "IEEE Xplore request failed."
    assert "ieee token" not in error
    assert "/private/ieee.key" not in error


def test_search_ieee_preserves_timeout_classification(monkeypatch):
    monkeypatch.setenv("IEEE_API_KEY", "test-key")

    def fail_fetch_json(**_kwargs):
        raise TimeoutError("timed out at /private/ieee-timeout.key")

    monkeypatch.setattr(ieee, "fetch_json", fail_fetch_json)

    items, total, error = ieee.search_ieee("topic", 0, 10)

    assert items is None
    assert total == 0
    assert error == "IEEE Xplore request timed out."
    assert "timed out at" not in error
    assert "/private/ieee-timeout.key" not in error


def test_get_ieee_by_doi_sanitizes_fetch_failures(monkeypatch):
    monkeypatch.setenv("IEEE_API_KEY", "test-key")

    def fail_fetch_json(**_kwargs):
        raise RuntimeError("ieee doi token at /private/ieee-doi.key")

    monkeypatch.setattr(ieee, "fetch_json", fail_fetch_json)

    item, error = ieee.get_ieee_by_doi("10.123/example")

    assert item is None
    assert error == "IEEE Xplore request failed."
    assert "ieee doi token" not in error
    assert "/private/ieee-doi.key" not in error


def test_get_ieee_by_doi_preserves_timeout_classification(monkeypatch):
    monkeypatch.setenv("IEEE_API_KEY", "test-key")

    def fail_fetch_json(**_kwargs):
        raise TimeoutError("timed out at /private/ieee-doi-timeout.key")

    monkeypatch.setattr(ieee, "fetch_json", fail_fetch_json)

    item, error = ieee.get_ieee_by_doi("10.123/example")

    assert item is None
    assert error == "IEEE Xplore request timed out."
    assert "timed out at" not in error
    assert "/private/ieee-doi-timeout.key" not in error


def test_get_ieee_by_id_sanitizes_fetch_failures(monkeypatch):
    monkeypatch.setenv("IEEE_API_KEY", "test-key")

    def fail_fetch_json(**_kwargs):
        raise RuntimeError("ieee id token at /private/ieee-id.key")

    monkeypatch.setattr(ieee, "fetch_json", fail_fetch_json)

    item, error = ieee.get_ieee_by_id("123456")

    assert item is None
    assert error == "IEEE Xplore request failed."
    assert "ieee id token" not in error
    assert "/private/ieee-id.key" not in error


def test_get_ieee_by_id_preserves_timeout_classification(monkeypatch):
    monkeypatch.setenv("IEEE_API_KEY", "test-key")

    def fail_fetch_json(**_kwargs):
        raise TimeoutError("timed out at /private/ieee-id-timeout.key")

    monkeypatch.setattr(ieee, "fetch_json", fail_fetch_json)

    item, error = ieee.get_ieee_by_id("123456")

    assert item is None
    assert error == "IEEE Xplore request timed out."
    assert "timed out at" not in error
    assert "/private/ieee-id-timeout.key" not in error
