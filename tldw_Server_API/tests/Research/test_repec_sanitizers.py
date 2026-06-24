from typing import Any

import pytest

from tldw_Server_API.app.core.Third_Party import RePEc as repec


pytestmark = pytest.mark.unit


def test_get_ref_by_handle_sanitizes_fetch_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPEC_API_CODE", "test-code")

    def fail_fetch(**_kwargs: Any) -> None:
        raise RuntimeError("repec token at /private/repec.key")

    monkeypatch.setattr(repec, "fetch", fail_fetch)

    item, error = repec.get_ref_by_handle("RePEc:abc:def:123")

    assert item is None
    assert error == "RePEc getref request failed."
    assert "repec token" not in error
    assert "/private/repec.key" not in error


def test_get_ref_by_handle_preserves_timeout_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPEC_API_CODE", "test-code")

    def fail_fetch(**_kwargs: Any) -> None:
        raise TimeoutError("timed out at /private/repec-timeout.key")

    monkeypatch.setattr(repec, "fetch", fail_fetch)

    item, error = repec.get_ref_by_handle("RePEc:abc:def:123")

    assert item is None
    assert error == "RePEc getref request timed out."
    assert "timed out at" not in error
    assert "/private/repec-timeout.key" not in error


def test_get_citations_plain_sanitizes_fetch_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_fetch(**_kwargs: Any) -> None:
        raise RuntimeError("citec token at /private/citec.key")

    monkeypatch.setattr(repec, "fetch", fail_fetch)

    item, error = repec.get_citations_plain("RePEc:abc:def:123")

    assert item is None
    assert error == "CitEc request failed."
    assert "citec token" not in error
    assert "/private/citec.key" not in error


def test_get_citations_plain_preserves_timeout_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_fetch(**_kwargs: Any) -> None:
        raise TimeoutError("timed out at /private/citec-timeout.key")

    monkeypatch.setattr(repec, "fetch", fail_fetch)

    item, error = repec.get_citations_plain("RePEc:abc:def:123")

    assert item is None
    assert error == "CitEc request timed out."
    assert "timed out at" not in error
    assert "/private/citec-timeout.key" not in error


def test_get_citations_amf_raw_sanitizes_fetch_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_fetch(**_kwargs: Any) -> None:
        raise RuntimeError("citec amf token at /private/citec-amf.key")

    monkeypatch.setattr(repec, "fetch", fail_fetch)

    xml_text, error = repec.get_citations_amf_raw("RePEc:abc:def:123")

    assert xml_text is None
    assert error == "CitEc request failed."
    assert "citec amf token" not in error
    assert "/private/citec-amf.key" not in error


def test_get_citations_amf_raw_preserves_timeout_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_fetch(**_kwargs: Any) -> None:
        raise TimeoutError("timed out at /private/citec-amf-timeout.key")

    monkeypatch.setattr(repec, "fetch", fail_fetch)

    xml_text, error = repec.get_citations_amf_raw("RePEc:abc:def:123")

    assert xml_text is None
    assert error == "CitEc request timed out."
    assert "timed out at" not in error
    assert "/private/citec-amf-timeout.key" not in error


def test_citec_base_uses_https() -> None:
    assert repec._CITEC_BASE.startswith("https://")


def test_get_citations_plain_uses_https_citec_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    class FakeResponse:
        status_code = 200
        text = "<citationData id=\"RePEc:abc:def:123\"><citedBy>1</citedBy><cites>2</cites></citationData>"

    def fake_fetch(**kwargs: Any) -> FakeResponse:
        captured["url"] = kwargs["url"]
        return FakeResponse()

    monkeypatch.setattr(repec, "fetch", fake_fetch)

    item, error = repec.get_citations_plain("RePEc:abc:def:123")

    assert error is None
    assert item is not None
    assert captured["url"] == "https://citec.repec.org/api/plain/RePEc:abc:def:123"
