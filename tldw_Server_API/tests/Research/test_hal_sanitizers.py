import pytest

from tldw_Server_API.app.core.Third_Party import HAL as hal


pytestmark = pytest.mark.unit


def test_search_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise RuntimeError("hal token at /private/hal.key")

    monkeypatch.setattr(hal, "fetch_json", fail_fetch_json)

    items, total, error = hal.search("topic", 0, 10)

    assert items is None
    assert total == 0
    assert error == "HAL request failed."
    assert "hal token" not in error
    assert "/private/hal.key" not in error


def test_search_preserves_timeout_classification(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise TimeoutError("timed out at /private/hal-timeout.key")

    monkeypatch.setattr(hal, "fetch_json", fail_fetch_json)

    items, total, error = hal.search("topic", 0, 10)

    assert items is None
    assert total == 0
    assert error == "HAL request timed out."
    assert "timed out at" not in error
    assert "/private/hal-timeout.key" not in error


def test_by_docid_sanitizes_search_failures(monkeypatch):
    def fail_search(**_kwargs):
        raise RuntimeError("hal doc token at /private/hal-doc.key")

    monkeypatch.setattr(hal, "search", fail_search)

    item, error = hal.by_docid("12345")

    assert item is None
    assert error == "HAL request failed."
    assert "hal doc token" not in error
    assert "/private/hal-doc.key" not in error


def test_by_docid_preserves_timeout_classification(monkeypatch):
    def fail_search(**_kwargs):
        raise TimeoutError("timed out at /private/hal-doc-timeout.key")

    monkeypatch.setattr(hal, "search", fail_search)

    item, error = hal.by_docid("12345")

    assert item is None
    assert error == "HAL request timed out."
    assert "timed out at" not in error
    assert "/private/hal-doc-timeout.key" not in error


def test_raw_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch(**_kwargs):
        raise RuntimeError("hal raw token at /private/hal-raw.key")

    monkeypatch.setattr(hal, "fetch", fail_fetch)

    content, media_type, error = hal.raw({"wt": "xml"})

    assert content is None
    assert media_type is None
    assert error == "HAL request failed."
    assert "hal raw token" not in error
    assert "/private/hal-raw.key" not in error


def test_raw_preserves_timeout_classification(monkeypatch):
    def fail_fetch(**_kwargs):
        raise TimeoutError("timed out at /private/hal-raw-timeout.key")

    monkeypatch.setattr(hal, "fetch", fail_fetch)

    content, media_type, error = hal.raw({"wt": "xml"})

    assert content is None
    assert media_type is None
    assert error == "HAL request timed out."
    assert "timed out at" not in error
    assert "/private/hal-raw-timeout.key" not in error
