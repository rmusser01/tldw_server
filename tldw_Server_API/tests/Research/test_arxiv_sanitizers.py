from typing import Any

import pytest

from tldw_Server_API.app.core.Third_Party import Arxiv as arxiv


pytestmark = pytest.mark.unit


def test_fetch_arxiv_pdf_url_sanitizes_fetch_logs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_fetch(**_kwargs: Any) -> None:
        raise RuntimeError("arxiv pdf token at /private/arxiv-pdf.key")

    monkeypatch.setattr(arxiv, "fetch", fail_fetch)

    result = arxiv.fetch_arxiv_pdf_url("paper-id-from-/private/request")

    assert result is None
    output = capsys.readouterr().out
    assert "Error fetching arXiv PDF URL." in output
    assert "arxiv pdf token" not in output
    assert "/private/arxiv-pdf.key" not in output
    assert "paper-id-from-/private/request" not in output


def test_search_arxiv_custom_api_sanitizes_fetch_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_fetch(**_kwargs: Any) -> None:
        raise RuntimeError("arxiv search token at /private/arxiv-search.key")

    monkeypatch.setattr(arxiv, "fetch", fail_fetch)

    papers, total, error = arxiv.search_arxiv_custom_api(
        "retrieval",
        author=None,
        year=None,
        start_index=0,
        page_size=1,
    )

    assert papers is None
    assert total == 0
    assert error == "arXiv API request failed."
    assert "arxiv search token" not in error
    assert "/private/arxiv-search.key" not in error


def test_search_arxiv_custom_api_preserves_timeout_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_fetch(**_kwargs: Any) -> None:
        raise TimeoutError("timed out at /private/arxiv-timeout.key")

    monkeypatch.setattr(arxiv, "fetch", fail_fetch)

    papers, total, error = arxiv.search_arxiv_custom_api(
        "retrieval",
        author=None,
        year=None,
        start_index=0,
        page_size=1,
    )

    assert papers is None
    assert total == 0
    assert error == "arXiv API request timed out."
    assert "timed out at" not in error
    assert "/private/arxiv-timeout.key" not in error


def test_fetch_arxiv_xml_sanitizes_fetch_logs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_fetch(**_kwargs: Any) -> None:
        raise RuntimeError("arxiv xml token at /private/arxiv.xml")

    monkeypatch.setattr(arxiv, "fetch", fail_fetch)

    result = arxiv.fetch_arxiv_xml("paper-id-from-/private/request")

    assert result is None
    output = capsys.readouterr().out
    assert "Error fetching arXiv XML." in output
    assert "arxiv xml token" not in output
    assert "/private/arxiv.xml" not in output
    assert "paper-id-from-/private/request" not in output


def test_parse_arxiv_feed_sanitizes_parser_fallback_warning(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class EmptySoup:
        def find_all(self, _name: str) -> list[Any]:
            return []

    def fake_soup(_xml_content: bytes, parser: str) -> EmptySoup:
        if parser == "lxml-xml":
            raise arxiv.FeatureNotFound("lxml load failed at /private/lxml-plugin.so")
        return EmptySoup()

    monkeypatch.setattr(arxiv, "BeautifulSoup", fake_soup)

    entries = arxiv.parse_arxiv_feed(b"<feed />")

    assert entries == []
    output = capsys.readouterr().out
    assert "Failed to use 'lxml-xml' parser." in output
    assert "lxml load failed" not in output
    assert "/private/lxml-plugin.so" not in output


def test_get_arxiv_by_id_sanitizes_parse_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_parse(_xml_content: str) -> None:
        raise ValueError("arxiv parse path at /private/arxiv-feed.xml")

    monkeypatch.setattr(arxiv, "fetch_arxiv_xml", lambda _paper_id: "<feed />")
    monkeypatch.setattr(arxiv, "parse_arxiv_feed", fail_parse)

    item, error = arxiv.get_arxiv_by_id("2401.00001")

    assert item is None
    assert error == "Failed to parse arXiv XML."
    assert "arxiv parse path" not in error
    assert "/private/arxiv-feed.xml" not in error


def test_get_arxiv_by_id_sanitizes_fetch_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_fetch_xml(_paper_id: str) -> None:
        raise RuntimeError("arxiv by-id token at /private/arxiv-by-id.xml")

    monkeypatch.setattr(arxiv, "fetch_arxiv_xml", fail_fetch_xml)

    item, error = arxiv.get_arxiv_by_id("paper-id-from-/private/request")

    assert item is None
    assert error == "Failed to fetch arXiv paper."
    assert "arxiv by-id token" not in error
    assert "/private/arxiv-by-id.xml" not in error
    assert "paper-id-from-/private/request" not in error


def test_arxiv_query_builder_uses_https_export_endpoint() -> None:
    url = arxiv.build_query_url("retrieval", author=None, year=None, start=0, max_results=1)

    assert url.startswith("https://export.arxiv.org/api/query?")


def test_fetch_arxiv_xml_uses_https_export_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    class FakeResponse:
        status_code = 200
        text = "<feed />"

    def fake_fetch(**kwargs: Any) -> FakeResponse:
        captured["url"] = kwargs["url"]
        return FakeResponse()

    monkeypatch.setattr(arxiv, "fetch", fake_fetch)
    monkeypatch.setattr(arxiv.time, "sleep", lambda _seconds: None)

    xml = arxiv.fetch_arxiv_xml("2401.00001")

    assert xml == "<feed />"
    assert captured["url"] == "https://export.arxiv.org/api/query?id_list=2401.00001"
