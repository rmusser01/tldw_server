import pytest

from tldw_Server_API.app.core.Third_Party import Unpaywall as unpaywall


pytestmark = pytest.mark.unit


def test_resolve_oa_pdf_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch(**_kwargs):
        raise RuntimeError("unpaywall token at /private/unpaywall.key")

    monkeypatch.setenv("UNPAYWALL_EMAIL", "research@example.com")
    monkeypatch.setattr(unpaywall, "fetch", fail_fetch)

    pdf_url, error = unpaywall.resolve_oa_pdf("10.private/request")

    assert pdf_url is None
    assert error == "Unpaywall request failed."
    assert "unpaywall token" not in error
    assert "/private/unpaywall.key" not in error
    assert "10.private/request" not in error


def test_resolve_oa_pdf_preserves_timeout_classification(monkeypatch):
    def fail_fetch(**_kwargs):
        raise TimeoutError("timed out at /private/unpaywall-timeout.key")

    monkeypatch.setenv("UNPAYWALL_EMAIL", "research@example.com")
    monkeypatch.setattr(unpaywall, "fetch", fail_fetch)

    pdf_url, error = unpaywall.resolve_oa_pdf("10.private/request")

    assert pdf_url is None
    assert error == "Unpaywall request timed out."
    assert "timed out at" not in error
    assert "/private/unpaywall-timeout.key" not in error
    assert "10.private/request" not in error
