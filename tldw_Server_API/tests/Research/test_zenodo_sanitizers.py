import pytest

from tldw_Server_API.app.core.Third_Party import Zenodo as zenodo


pytestmark = pytest.mark.unit


def test_search_records_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise RuntimeError("zenodo token at /private/zenodo.key")

    monkeypatch.setattr(zenodo, "fetch_json", fail_fetch_json)

    items, total, error = zenodo.search_records("climate", 1, 10)

    assert items is None
    assert total == 0
    assert error == "Zenodo request failed."
    assert "zenodo token" not in error
    assert "/private/zenodo.key" not in error


def test_search_records_preserves_timeout_classification(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise TimeoutError("timed out at /private/zenodo-timeout.key")

    monkeypatch.setattr(zenodo, "fetch_json", fail_fetch_json)

    items, total, error = zenodo.search_records("climate", 1, 10)

    assert items is None
    assert total == 0
    assert error == "Zenodo request timed out."
    assert "timed out at" not in error
    assert "/private/zenodo-timeout.key" not in error


def test_get_record_by_id_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise RuntimeError("zenodo by-id token at /private/zenodo-id.key")

    monkeypatch.setattr(zenodo, "fetch_json", fail_fetch_json)

    item, error = zenodo.get_record_by_id("123456")

    assert item is None
    assert error == "Zenodo record request failed."
    assert "zenodo by-id token" not in error
    assert "/private/zenodo-id.key" not in error
    assert "123456" not in error


def test_get_record_by_doi_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise RuntimeError("zenodo doi token at /private/zenodo-doi.key")

    monkeypatch.setattr(zenodo, "fetch_json", fail_fetch_json)

    item, error = zenodo.get_record_by_doi("10.5281/zenodo.123456")

    assert item is None
    assert error == "Zenodo DOI request failed."
    assert "zenodo doi token" not in error
    assert "/private/zenodo-doi.key" not in error
    assert "10.5281/zenodo.123456" not in error


def test_oai_raw_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch(**_kwargs):
        raise RuntimeError("zenodo oai token at /private/zenodo-oai.key")

    monkeypatch.setattr(zenodo, "fetch", fail_fetch)

    content, content_type, error = zenodo.oai_raw({"verb": "Identify"})

    assert content is None
    assert content_type is None
    assert error == "Zenodo OAI-PMH request failed."
    assert "zenodo oai token" not in error
    assert "/private/zenodo-oai.key" not in error


def test_get_record_raw_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise RuntimeError("zenodo raw token at /private/zenodo-raw.key")

    monkeypatch.setattr(zenodo, "fetch_json", fail_fetch_json)

    item, error = zenodo.get_record_raw("123456")

    assert item is None
    assert error == "Zenodo raw record request failed."
    assert "zenodo raw token" not in error
    assert "/private/zenodo-raw.key" not in error
    assert "123456" not in error
