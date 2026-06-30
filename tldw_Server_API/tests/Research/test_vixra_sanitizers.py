from typing import Any

import pytest

from tldw_Server_API.app.core.Third_Party import Vixra as vixra


pytestmark = pytest.mark.unit


def test_get_vixra_by_id_sanitizes_resolution_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_try_pdf(_url: str) -> None:
        raise RuntimeError("vixra token at /private/vixra.key")

    monkeypatch.setattr(vixra, "_try_pdf", fail_try_pdf)

    item, error = vixra.get_vixra_by_id("1901.0001")

    assert item is None
    assert error == "viXra request failed."
    assert "vixra token" not in error
    assert "/private/vixra.key" not in error


def test_get_vixra_by_id_preserves_timeout_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_try_pdf(_url: str) -> None:
        raise TimeoutError("timed out at /private/vixra-timeout.key")

    monkeypatch.setattr(vixra, "_try_pdf", fail_try_pdf)

    item, error = vixra.get_vixra_by_id("1901.0001")

    assert item is None
    assert error == "viXra request timed out."
    assert "timed out at" not in error
    assert "/private/vixra-timeout.key" not in error


def test_search_sanitizes_setup_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_quote(_value: str) -> None:
        raise RuntimeError("vixra search token at /private/vixra-search.key")

    monkeypatch.setattr(vixra, "urlquote", fail_quote)

    items, total, error = vixra.search("quantum", 1, 10)

    assert items is None
    assert total == 0
    assert error == "viXra search request failed."
    assert "vixra search token" not in error
    assert "/private/vixra-search.key" not in error


def test_search_preserves_timeout_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_quote(_value: str) -> None:
        raise TimeoutError("timed out at /private/vixra-search-timeout.key")

    monkeypatch.setattr(vixra, "urlquote", fail_quote)

    items, total, error = vixra.search("quantum", 1, 10)

    assert items is None
    assert total == 0
    assert error == "viXra search request timed out."
    assert "timed out at" not in error
    assert "/private/vixra-search-timeout.key" not in error


def test_search_applies_page_before_enriching_results(monkeypatch: pytest.MonkeyPatch) -> None:
    search_html = """
        <a href="/abs/1001.0001">First result</a>
        <a href="/abs/1001.0002">Second result</a>
        <a href="/abs/1001.0003">Third result</a>
    """
    abs_requests: list[str] = []
    responses: list["FakeResponse"] = []

    class FakeResponse:
        def __init__(self, text: str) -> None:
            self.status_code = 200
            self.text = text
            self.headers: dict[str, str] = {}
            self.closed = False

        def close(self) -> None:
            self.closed = True

    def fake_fetch(**kwargs: Any) -> FakeResponse:
        url = kwargs["url"]
        if "/find/" in url or "?search=" in url or "?find=" in url:
            response = FakeResponse(search_html)
            responses.append(response)
            return response
        if "/abs/" in url:
            abs_requests.append(url)
            vid = url.rsplit("/", 1)[-1]
            response = FakeResponse(
                f'<meta name="citation_title" content="Better {vid}">'
                f'<meta name="citation_author" content="Author {vid}">'
                '<meta name="citation_date" content="2024-01-01">'
            )
            responses.append(response)
            return response
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(vixra, "fetch", fake_fetch)

    items, total, error = vixra.search("quantum", page=2, results_per_page=1)

    assert error is None
    assert total == 3
    assert [item["id"] for item in items or []] == ["1001.0002"]
    assert [item["title"] for item in items or []] == ["Better 1001.0002"]
    assert abs_requests == ["https://vixra.org/abs/1001.0002"]
    assert responses
    assert all(response.closed for response in responses)
