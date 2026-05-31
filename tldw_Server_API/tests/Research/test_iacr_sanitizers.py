import pytest

from tldw_Server_API.app.core.Third_Party import IACR as iacr


pytestmark = pytest.mark.unit


def test_fetch_conference_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise RuntimeError("iacr token at /private/iacr.key")

    monkeypatch.setattr(iacr, "fetch_json", fail_fetch_json)

    data, error = iacr.fetch_conference("crypto", 2017)

    assert data is None
    assert error == "IACR request failed."
    assert "iacr token" not in error
    assert "/private/iacr.key" not in error


def test_fetch_conference_preserves_timeout_classification(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise TimeoutError("timed out at /private/iacr-timeout.key")

    monkeypatch.setattr(iacr, "fetch_json", fail_fetch_json)

    data, error = iacr.fetch_conference("crypto", 2017)

    assert data is None
    assert error == "IACR request timed out."
    assert "timed out at" not in error
    assert "/private/iacr-timeout.key" not in error


def test_fetch_conference_raw_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch(**_kwargs):
        raise RuntimeError("iacr raw token at /private/iacr-raw.key")

    monkeypatch.setattr(iacr, "fetch", fail_fetch)

    content, media_type, error = iacr.fetch_conference_raw("crypto", 2017)

    assert content is None
    assert media_type is None
    assert error == "IACR request failed."
    assert "iacr raw token" not in error
    assert "/private/iacr-raw.key" not in error


def test_fetch_conference_raw_preserves_timeout_classification(monkeypatch):
    def fail_fetch(**_kwargs):
        raise TimeoutError("timed out at /private/iacr-raw-timeout.key")

    monkeypatch.setattr(iacr, "fetch", fail_fetch)

    content, media_type, error = iacr.fetch_conference_raw("crypto", 2017)

    assert content is None
    assert media_type is None
    assert error == "IACR request timed out."
    assert "timed out at" not in error
    assert "/private/iacr-raw-timeout.key" not in error
