import pytest

from tldw_Server_API.app.core.Third_Party import PMC_OA as pmc_oa


pytestmark = pytest.mark.unit


def test_pmc_oa_identify_sanitizes_xml_failures(monkeypatch):
    def fail_get_xml(_params):
        raise RuntimeError("pmc oa token at /private/pmc-oa.key")

    monkeypatch.setattr(pmc_oa, "_get_xml", fail_get_xml)

    info, error = pmc_oa.pmc_oa_identify()

    assert info is None
    assert error == "PMC OA Identify request failed."
    assert "pmc oa token" not in error
    assert "/private/pmc-oa.key" not in error


def test_pmc_oa_identify_preserves_timeout_classification(monkeypatch):
    def fail_get_xml(_params):
        raise TimeoutError("timed out at /private/pmc-oa-timeout.key")

    monkeypatch.setattr(pmc_oa, "_get_xml", fail_get_xml)

    info, error = pmc_oa.pmc_oa_identify()

    assert info is None
    assert error == "PMC OA Identify request timed out."
    assert "timed out at" not in error
    assert "/private/pmc-oa-timeout.key" not in error


def test_pmc_oa_query_sanitizes_xml_failures(monkeypatch):
    def fail_get_xml(_params):
        raise RuntimeError("pmc oa query token at /private/pmc-oa-query.key")

    monkeypatch.setattr(pmc_oa, "_get_xml", fail_get_xml)

    items, token, error = pmc_oa.pmc_oa_query(id_param="PMC123456")

    assert items is None
    assert token is None
    assert error == "PMC OA query request failed."
    assert "pmc oa query token" not in error
    assert "/private/pmc-oa-query.key" not in error
    assert "PMC123456" not in error


def test_pmc_oa_query_preserves_timeout_classification(monkeypatch):
    def fail_get_xml(_params):
        raise TimeoutError("timed out at /private/pmc-oa-query-timeout.key")

    monkeypatch.setattr(pmc_oa, "_get_xml", fail_get_xml)

    items, token, error = pmc_oa.pmc_oa_query(id_param="PMC123456")

    assert items is None
    assert token is None
    assert error == "PMC OA query request timed out."
    assert "timed out at" not in error
    assert "/private/pmc-oa-query-timeout.key" not in error


def test_download_pmc_pdf_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch(**_kwargs):
        raise RuntimeError("pmc pdf token at /private/pmc-pdf.key")

    monkeypatch.setattr(pmc_oa, "fetch", fail_fetch)

    content, filename, error = pmc_oa.download_pmc_pdf("PMC123456")

    assert content is None
    assert filename is None
    assert error == "PMC PDF download failed."
    assert "pmc pdf token" not in error
    assert "/private/pmc-pdf.key" not in error
    assert "PMC123456" not in error


def test_download_pmc_pdf_preserves_timeout_classification(monkeypatch):
    def fail_fetch(**_kwargs):
        raise TimeoutError("timed out at /private/pmc-pdf-timeout.key")

    monkeypatch.setattr(pmc_oa, "fetch", fail_fetch)

    content, filename, error = pmc_oa.download_pmc_pdf("PMC123456")

    assert content is None
    assert filename is None
    assert error == "PMC PDF download timed out."
    assert "timed out at" not in error
    assert "/private/pmc-pdf-timeout.key" not in error
