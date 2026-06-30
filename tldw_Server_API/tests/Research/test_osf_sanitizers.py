import pytest

from tldw_Server_API.app.core.Third_Party import OSF as osf


pytestmark = pytest.mark.unit


def test_search_preprints_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise RuntimeError("osf token at /private/osf.key")

    monkeypatch.setattr(osf, "fetch_json", fail_fetch_json)

    items, total, error = osf.search_preprints("climate", 1, 10)

    assert items is None
    assert total == 0
    assert error == "OSF request failed."
    assert "osf token" not in error
    assert "/private/osf.key" not in error


def test_search_preprints_preserves_timeout_classification(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise TimeoutError("timed out at /private/osf-timeout.key")

    monkeypatch.setattr(osf, "fetch_json", fail_fetch_json)

    items, total, error = osf.search_preprints("climate", 1, 10)

    assert items is None
    assert total == 0
    assert error == "OSF request timed out."
    assert "timed out at" not in error
    assert "/private/osf-timeout.key" not in error


def test_get_preprint_by_id_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch(**_kwargs):
        raise RuntimeError("osf by-id token at /private/osf-id.key")

    monkeypatch.setattr(osf, "fetch", fail_fetch)

    item, error = osf.get_preprint_by_id("abc123")

    assert item is None
    assert error == "OSF preprint request failed."
    assert "osf by-id token" not in error
    assert "/private/osf-id.key" not in error
    assert "abc123" not in error


def test_get_preprint_by_id_preserves_timeout_classification(monkeypatch):
    def fail_fetch(**_kwargs):
        raise TimeoutError("timed out at /private/osf-id-timeout.key")

    monkeypatch.setattr(osf, "fetch", fail_fetch)

    item, error = osf.get_preprint_by_id("abc123")

    assert item is None
    assert error == "OSF preprint request timed out."
    assert "timed out at" not in error
    assert "/private/osf-id-timeout.key" not in error


def test_get_preprint_by_doi_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch(**_kwargs):
        raise RuntimeError("osf doi token at /private/osf-doi.key")

    monkeypatch.setattr(osf, "fetch", fail_fetch)

    item, error = osf.get_preprint_by_doi("10.31219/osf.io/abc12")

    assert item is None
    assert error == "OSF DOI request failed."
    assert "osf doi token" not in error
    assert "/private/osf-doi.key" not in error
    assert "10.31219/osf.io/abc12" not in error


def test_get_preprint_by_doi_preserves_timeout_classification(monkeypatch):
    def fail_fetch(**_kwargs):
        raise TimeoutError("timed out at /private/osf-doi-timeout.key")

    monkeypatch.setattr(osf, "fetch", fail_fetch)

    item, error = osf.get_preprint_by_doi("10.31219/osf.io/abc12")

    assert item is None
    assert error == "OSF DOI request timed out."
    assert "timed out at" not in error
    assert "/private/osf-doi-timeout.key" not in error


def test_get_primary_file_download_url_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise RuntimeError("osf primary token at /private/osf-primary.key")

    monkeypatch.setattr(osf, "fetch_json", fail_fetch_json)

    download_url, error = osf.get_primary_file_download_url("abc123")

    assert download_url is None
    assert error == "OSF primary file request failed."
    assert "osf primary token" not in error
    assert "/private/osf-primary.key" not in error
    assert "abc123" not in error


def test_get_primary_file_download_url_preserves_timeout_classification(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise TimeoutError("timed out at /private/osf-primary-timeout.key")

    monkeypatch.setattr(osf, "fetch_json", fail_fetch_json)

    download_url, error = osf.get_primary_file_download_url("abc123")

    assert download_url is None
    assert error == "OSF primary file request timed out."
    assert "timed out at" not in error
    assert "/private/osf-primary-timeout.key" not in error


def test_raw_preprints_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch(**_kwargs):
        raise RuntimeError("osf raw token at /private/osf-raw.key")

    monkeypatch.setattr(osf, "fetch", fail_fetch)

    content, content_type, error = osf.raw_preprints({"q": "climate"})

    assert content is None
    assert content_type is None
    assert error == "OSF raw preprints request failed."
    assert "osf raw token" not in error
    assert "/private/osf-raw.key" not in error


def test_raw_preprints_preserves_timeout_classification(monkeypatch):
    def fail_fetch(**_kwargs):
        raise TimeoutError("timed out at /private/osf-raw-timeout.key")

    monkeypatch.setattr(osf, "fetch", fail_fetch)

    content, content_type, error = osf.raw_preprints({"q": "climate"})

    assert content is None
    assert content_type is None
    assert error == "OSF raw preprints request timed out."
    assert "timed out at" not in error
    assert "/private/osf-raw-timeout.key" not in error


def test_raw_by_id_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch(**_kwargs):
        raise RuntimeError("osf raw by-id token at /private/osf-raw-id.key")

    monkeypatch.setattr(osf, "fetch", fail_fetch)

    content, content_type, error = osf.raw_by_id("abc123")

    assert content is None
    assert content_type is None
    assert error == "OSF raw preprint request failed."
    assert "osf raw by-id token" not in error
    assert "/private/osf-raw-id.key" not in error
    assert "abc123" not in error


def test_raw_by_id_preserves_timeout_classification(monkeypatch):
    def fail_fetch(**_kwargs):
        raise TimeoutError("timed out at /private/osf-raw-id-timeout.key")

    monkeypatch.setattr(osf, "fetch", fail_fetch)

    content, content_type, error = osf.raw_by_id("abc123")

    assert content is None
    assert content_type is None
    assert error == "OSF raw preprint request timed out."
    assert "timed out at" not in error
    assert "/private/osf-raw-id-timeout.key" not in error
