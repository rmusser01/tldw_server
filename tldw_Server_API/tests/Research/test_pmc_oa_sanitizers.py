from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.Third_Party import PMC_OA as pmc_oa


pytestmark = pytest.mark.unit


def test_pmc_oa_identify_sanitizes_xml_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_get_xml(_params: dict[str, Any]) -> None:
        raise RuntimeError("pmc oa token at /private/pmc-oa.key")

    monkeypatch.setattr(pmc_oa, "_get_xml", fail_get_xml)

    info, error = pmc_oa.pmc_oa_identify()

    assert info is None
    assert error == "PMC OA Identify request failed."
    assert "pmc oa token" not in error
    assert "/private/pmc-oa.key" not in error


def test_pmc_oa_identify_preserves_timeout_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_get_xml(_params: dict[str, Any]) -> None:
        raise TimeoutError("timed out at /private/pmc-oa-timeout.key")

    monkeypatch.setattr(pmc_oa, "_get_xml", fail_get_xml)

    info, error = pmc_oa.pmc_oa_identify()

    assert info is None
    assert error == "PMC OA Identify request timed out."
    assert "timed out at" not in error
    assert "/private/pmc-oa-timeout.key" not in error


def test_pmc_oa_query_sanitizes_xml_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_get_xml(_params: dict[str, Any]) -> None:
        raise RuntimeError("pmc oa query token at /private/pmc-oa-query.key")

    monkeypatch.setattr(pmc_oa, "_get_xml", fail_get_xml)

    items, token, error = pmc_oa.pmc_oa_query(id_param="PMC123456")

    assert items is None
    assert token is None
    assert error == "PMC OA query request failed."
    assert "pmc oa query token" not in error
    assert "/private/pmc-oa-query.key" not in error
    assert "PMC123456" not in error


def test_pmc_oa_query_preserves_timeout_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_get_xml(_params: dict[str, Any]) -> None:
        raise TimeoutError("timed out at /private/pmc-oa-query-timeout.key")

    monkeypatch.setattr(pmc_oa, "_get_xml", fail_get_xml)

    items, token, error = pmc_oa.pmc_oa_query(id_param="PMC123456")

    assert items is None
    assert token is None
    assert error == "PMC OA query request timed out."
    assert "timed out at" not in error
    assert "/private/pmc-oa-query-timeout.key" not in error


def test_download_pmc_pdf_sanitizes_download_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_download(**_kwargs: Any) -> None:
        raise RuntimeError("pmc pdf token at /private/pmc-pdf.key")

    monkeypatch.setattr(pmc_oa, "download", fail_download)

    content, filename, error = pmc_oa.download_pmc_pdf("PMC123456")

    assert content is None
    assert filename is None
    assert error == "PMC PDF download failed."
    assert "pmc pdf token" not in error
    assert "/private/pmc-pdf.key" not in error
    assert "PMC123456" not in error


def test_download_pmc_pdf_preserves_timeout_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_download(**_kwargs: Any) -> None:
        raise TimeoutError("timed out at /private/pmc-pdf-timeout.key")

    monkeypatch.setattr(pmc_oa, "download", fail_download)

    content, filename, error = pmc_oa.download_pmc_pdf("PMC123456")

    assert content is None
    assert filename is None
    assert error == "PMC PDF download timed out."
    assert "timed out at" not in error
    assert "/private/pmc-pdf-timeout.key" not in error


def test_download_pmc_pdf_rejects_invalid_pmcid_without_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_fetch(**_kwargs: Any) -> None:
        raise AssertionError("fetch should not be called for an invalid PMCID")

    monkeypatch.setattr(pmc_oa, "fetch", fail_fetch)

    content, filename, error = pmc_oa.download_pmc_pdf("PMCX123")

    assert content is None
    assert filename is None
    assert error == "Invalid PMCID."


def test_download_pmc_pdf_uses_bounded_download_and_validates_pdf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, Any] = {}

    def fake_download(**kwargs: Any) -> Path:
        calls.update(kwargs)
        dest = tmp_path / "pmc.pdf"
        dest.write_bytes(b"%PDF-1.7\nbody")
        return dest

    monkeypatch.setattr(pmc_oa, "download", fake_download)

    content, filename, error = pmc_oa.download_pmc_pdf("pmc123456")

    assert error is None
    assert filename == "PMC123456.pdf"
    assert content == b"%PDF-1.7\nbody"
    assert calls["url"] == "https://pmc.ncbi.nlm.nih.gov/PMC123456/pdf"
    assert calls["max_bytes_total"] == pmc_oa.PMC_PDF_MAX_BYTES
    assert calls["require_content_type"] == "application/pdf"
    assert calls["dest"].name == "PMC123456.pdf"


def test_download_pmc_pdf_rejects_non_pdf_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_download(**kwargs: Any) -> Path:
        del kwargs
        dest = tmp_path / "not-pdf.pdf"
        dest.write_bytes(b"<html>not found</html>")
        return dest

    monkeypatch.setattr(pmc_oa, "download", fake_download)

    content, filename, error = pmc_oa.download_pmc_pdf("PMC123456")

    assert content is None
    assert filename is None
    assert error == "PMC PDF download did not return a valid PDF."
