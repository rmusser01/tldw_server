import pytest

from tldw_Server_API.app.core.Third_Party import ChemRxiv as chemrxiv


pytestmark = pytest.mark.unit


def test_search_items_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise RuntimeError("chemrxiv token at /private/chemrxiv.key")

    monkeypatch.setattr(chemrxiv, "fetch_json", fail_fetch_json)

    items, total, error = chemrxiv.search_items("catalyst", 0, 10)

    assert items is None
    assert total == 0
    assert error == "ChemRxiv request failed."
    assert "chemrxiv token" not in error
    assert "/private/chemrxiv.key" not in error


def test_search_items_preserves_timeout_classification(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise TimeoutError("timed out at /private/chemrxiv-timeout.key")

    monkeypatch.setattr(chemrxiv, "fetch_json", fail_fetch_json)

    items, total, error = chemrxiv.search_items("catalyst", 0, 10)

    assert items is None
    assert total == 0
    assert error == "ChemRxiv request timed out."
    assert "timed out at" not in error
    assert "/private/chemrxiv-timeout.key" not in error


def test_get_item_by_id_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch(**_kwargs):
        raise RuntimeError("chemrxiv item token at /private/chemrxiv-id.key")

    monkeypatch.setattr(chemrxiv, "fetch", fail_fetch)

    item, error = chemrxiv.get_item_by_id("item-123")

    assert item is None
    assert error == "ChemRxiv item request failed."
    assert "chemrxiv item token" not in error
    assert "/private/chemrxiv-id.key" not in error
    assert "item-123" not in error


def test_get_item_by_id_preserves_timeout_classification(monkeypatch):
    def fail_fetch(**_kwargs):
        raise TimeoutError("timed out at /private/chemrxiv-id-timeout.key")

    monkeypatch.setattr(chemrxiv, "fetch", fail_fetch)

    item, error = chemrxiv.get_item_by_id("item-123")

    assert item is None
    assert error == "ChemRxiv item request timed out."
    assert "timed out at" not in error
    assert "/private/chemrxiv-id-timeout.key" not in error


def test_get_item_by_doi_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch(**_kwargs):
        raise RuntimeError("chemrxiv doi token at /private/chemrxiv-doi.key")

    monkeypatch.setattr(chemrxiv, "fetch", fail_fetch)

    item, error = chemrxiv.get_item_by_doi("10.26434/private")

    assert item is None
    assert error == "ChemRxiv DOI request failed."
    assert "chemrxiv doi token" not in error
    assert "/private/chemrxiv-doi.key" not in error
    assert "10.26434/private" not in error


def test_get_item_by_doi_preserves_timeout_classification(monkeypatch):
    def fail_fetch(**_kwargs):
        raise TimeoutError("timed out at /private/chemrxiv-doi-timeout.key")

    monkeypatch.setattr(chemrxiv, "fetch", fail_fetch)

    item, error = chemrxiv.get_item_by_doi("10.26434/private")

    assert item is None
    assert error == "ChemRxiv DOI request timed out."
    assert "timed out at" not in error
    assert "/private/chemrxiv-doi-timeout.key" not in error


def test_get_categories_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise RuntimeError("chemrxiv categories token at /private/chemrxiv-categories.key")

    monkeypatch.setattr(chemrxiv, "fetch_json", fail_fetch_json)

    data, error = chemrxiv.get_categories()

    assert data is None
    assert error == "ChemRxiv categories request failed."
    assert "chemrxiv categories token" not in error
    assert "/private/chemrxiv-categories.key" not in error


def test_get_categories_preserves_timeout_classification(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise TimeoutError("timed out at /private/chemrxiv-categories-timeout.key")

    monkeypatch.setattr(chemrxiv, "fetch_json", fail_fetch_json)

    data, error = chemrxiv.get_categories()

    assert data is None
    assert error == "ChemRxiv categories request timed out."
    assert "timed out at" not in error
    assert "/private/chemrxiv-categories-timeout.key" not in error


def test_get_licenses_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise RuntimeError("chemrxiv licenses token at /private/chemrxiv-licenses.key")

    monkeypatch.setattr(chemrxiv, "fetch_json", fail_fetch_json)

    data, error = chemrxiv.get_licenses()

    assert data is None
    assert error == "ChemRxiv licenses request failed."
    assert "chemrxiv licenses token" not in error
    assert "/private/chemrxiv-licenses.key" not in error


def test_get_licenses_preserves_timeout_classification(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise TimeoutError("timed out at /private/chemrxiv-licenses-timeout.key")

    monkeypatch.setattr(chemrxiv, "fetch_json", fail_fetch_json)

    data, error = chemrxiv.get_licenses()

    assert data is None
    assert error == "ChemRxiv licenses request timed out."
    assert "timed out at" not in error
    assert "/private/chemrxiv-licenses-timeout.key" not in error


def test_get_version_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise RuntimeError("chemrxiv version token at /private/chemrxiv-version.key")

    monkeypatch.setattr(chemrxiv, "fetch_json", fail_fetch_json)

    data, error = chemrxiv.get_version()

    assert data is None
    assert error == "ChemRxiv version request failed."
    assert "chemrxiv version token" not in error
    assert "/private/chemrxiv-version.key" not in error


def test_get_version_preserves_timeout_classification(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise TimeoutError("timed out at /private/chemrxiv-version-timeout.key")

    monkeypatch.setattr(chemrxiv, "fetch_json", fail_fetch_json)

    data, error = chemrxiv.get_version()

    assert data is None
    assert error == "ChemRxiv version request timed out."
    assert "timed out at" not in error
    assert "/private/chemrxiv-version-timeout.key" not in error


def test_oai_raw_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch(**_kwargs):
        raise RuntimeError("chemrxiv oai token at /private/chemrxiv-oai.key")

    monkeypatch.setattr(chemrxiv, "fetch", fail_fetch)

    content, content_type, error = chemrxiv.oai_raw({"verb": "Identify"})

    assert content is None
    assert content_type is None
    assert error == "ChemRxiv OAI-PMH request failed."
    assert "chemrxiv oai token" not in error
    assert "/private/chemrxiv-oai.key" not in error


def test_oai_raw_preserves_timeout_classification(monkeypatch):
    def fail_fetch(**_kwargs):
        raise TimeoutError("timed out at /private/chemrxiv-oai-timeout.key")

    monkeypatch.setattr(chemrxiv, "fetch", fail_fetch)

    content, content_type, error = chemrxiv.oai_raw({"verb": "Identify"})

    assert content is None
    assert content_type is None
    assert error == "ChemRxiv OAI-PMH request timed out."
    assert "timed out at" not in error
    assert "/private/chemrxiv-oai-timeout.key" not in error
