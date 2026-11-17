"""Current SearxNG articles and the legacy Searx HTML contract."""

import httpx
import pytest
from bs4 import BeautifulSoup

from tldw_Server_API.app.core.Web_Scraping import WebSearch_APIs as ws

pytestmark = pytest.mark.unit
MODERN = """<article class="result default-category">
<a class="url_header" href="https://example.com/">example.com</a>
<h3><a href="https://example.com/">Example Domain</a></h3>
<p class="content">Example domain for documentation.</p>
</article>"""
LEGACY = """<div class="result">
<h3>Legacy <em>heading</em></h3><a class="url" href="https://legacy.example/">legacy</a>
<p class="content">Legacy text</p><span class="published_date">2026-10-29</span>
</div>"""
MODERN_ROW = {
    "title": "Example Domain",
    "url": "https://example.com/",
    "content": "Example domain for documentation.",
    "publishedDate": "",
}
LEGACY_ROW = {
    "title": "Legacy heading",
    "url": "https://legacy.example/",
    "content": "Legacy text",
    "publishedDate": "2026-10-29",
}


@pytest.mark.parametrize("markup, expected", [(MODERN, MODERN_ROW), (LEGACY, LEGACY_ROW)])
def test_html_result_container_and_link_contract(markup, expected):
    actual = ws.parse_html_search_results_generic(BeautifulSoup(markup, "html.parser"))
    assert actual == [expected]  # nosec B101 # Public controlled parser contract.


def test_mixed_result_markup_preserves_document_order():
    actual = ws.parse_html_search_results_generic(BeautifulSoup(MODERN + LEGACY, "html.parser"))
    assert actual == [MODERN_ROW, LEGACY_ROW]  # nosec B101 # Preserve ordering across supported containers.


def test_navigation_links_are_not_search_results():
    soup = BeautifulSoup(
        '<nav><a class="url" href="https://navigation.example/">Menu</a></nav><article><h3><a href="https://other.example/">Other</a></h3></article>',
        "html.parser",
    )
    assert ws.parse_html_search_results_generic(soup) == []  # nosec B101 # Result class is the parsing boundary.


def test_modern_result_without_a_link_is_ignored():
    soup = BeautifulSoup(
        '<article class="result"><h3>No link</h3><p class="content">incomplete</p></article>', "html.parser"
    )
    assert ws.parse_html_search_results_generic(soup) == []  # nosec B101 # Never return an unusable result URL.


@pytest.mark.parametrize("markup, expected", [(MODERN, MODERN_ROW), (LEGACY, LEGACY_ROW)])
def test_searx_adapter_normalizes_supported_html(monkeypatch, markup, expected):
    monkeypatch.setattr(
        ws,
        "load_and_log_configs",
        lambda: {"search_engines": {"searx_search_api_url": "http://127.0.0.1:18990/search"}},
    )
    monkeypatch.setattr(ws.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        ws,
        "fetch",
        lambda **_kwargs: httpx.Response(200, text=markup, headers={"Content-Type": "text/html; charset=utf-8"}),
    )
    actual = ws.search_web_searx("controlled example")
    assert actual == {
        "results": [
            {
                "title": expected["title"],
                "link": expected["url"],
                "snippet": expected["content"],
                "publishedDate": expected["publishedDate"] or None,
            }
        ]
    }  # nosec B101 # Real adapter normalization with mocked network only.
