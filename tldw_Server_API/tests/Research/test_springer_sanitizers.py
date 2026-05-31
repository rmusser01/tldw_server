import pytest

from tldw_Server_API.app.core.Third_Party import Springer_Nature as springer


pytestmark = pytest.mark.unit


def test_search_springer_sanitizes_fetch_failures(monkeypatch):
    monkeypatch.setenv("SPRINGER_NATURE_API_KEY", "test-key")

    def fail_fetch_json(**_kwargs):
        raise RuntimeError("springer token at /private/springer.key")

    monkeypatch.setattr(springer, "fetch_json", fail_fetch_json)

    items, total, error = springer.search_springer("topic", 0, 10)

    assert items is None
    assert total == 0
    assert error == "Springer request failed."
    assert "springer token" not in error
    assert "/private/springer.key" not in error


def test_search_springer_preserves_timeout_classification(monkeypatch):
    monkeypatch.setenv("SPRINGER_NATURE_API_KEY", "test-key")

    def fail_fetch_json(**_kwargs):
        raise TimeoutError("timed out at /private/springer-timeout.key")

    monkeypatch.setattr(springer, "fetch_json", fail_fetch_json)

    items, total, error = springer.search_springer("topic", 0, 10)

    assert items is None
    assert total == 0
    assert error == "Springer request timed out."
    assert "timed out at" not in error
    assert "/private/springer-timeout.key" not in error


def test_get_springer_by_doi_sanitizes_fetch_failures(monkeypatch):
    monkeypatch.setenv("SPRINGER_NATURE_API_KEY", "test-key")

    def fail_fetch_json(**_kwargs):
        raise RuntimeError("springer doi token at /private/springer-doi.key")

    monkeypatch.setattr(springer, "fetch_json", fail_fetch_json)

    item, error = springer.get_springer_by_doi("10.123/example")

    assert item is None
    assert error == "Springer request failed."
    assert "springer doi token" not in error
    assert "/private/springer-doi.key" not in error


def test_get_springer_by_doi_preserves_timeout_classification(monkeypatch):
    monkeypatch.setenv("SPRINGER_NATURE_API_KEY", "test-key")

    def fail_fetch_json(**_kwargs):
        raise TimeoutError("timed out at /private/springer-doi-timeout.key")

    monkeypatch.setattr(springer, "fetch_json", fail_fetch_json)

    item, error = springer.get_springer_by_doi("10.123/example")

    assert item is None
    assert error == "Springer request timed out."
    assert "timed out at" not in error
    assert "/private/springer-doi-timeout.key" not in error
