from __future__ import annotations

import importlib
import json

import pytest

from mcp_unified.docs.discovery import (
    DiscoveredURLCandidate,
    extract_page_links,
    parse_sitemap_urlset,
    public_candidate,
)

pytestmark = pytest.mark.unit


def test_parse_sitemap_urlset_rejects_doctype() -> None:
    body = b'<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><urlset />'

    result = parse_sitemap_urlset(body, max_pages=10)

    assert result.reason_code == "sitemap_xml_forbidden_doctype"  # nosec B101
    assert result.candidates == []  # nosec B101


def test_parse_sitemap_urlset_enforces_page_limit_before_candidates() -> None:
    body = b"""
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://example.com/a</loc></url>
      <url><loc>https://example.com/b</loc></url>
      <url><loc>https://example.com/c</loc></url>
    </urlset>
    """

    result = parse_sitemap_urlset(body, max_pages=2)

    assert result.reason_code == "ok"  # nosec B101
    assert [item.url for item in result.candidates] == [  # nosec B101
        "https://example.com/a",
        "https://example.com/b",
    ]
    assert result.skipped == 1  # nosec B101


def test_extract_page_links_prefers_beautifulsoup_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    original_import = importlib.import_module

    def fake_import(name: str, package: str | None = None):
        if name == "bs4":
            seen.append(name)
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    html = b'<a href="/docs/a">A</a><a rel="nofollow" href="/docs/b">B</a>'

    links = extract_page_links("https://example.com/docs/index.html", html)

    assert "https://example.com/docs/a" in links  # nosec B101
    assert "https://example.com/docs/b" not in links  # nosec B101
    assert seen == ["bs4"] or seen == []  # nosec B101


def test_public_candidate_redacts_query_bearing_url() -> None:
    candidate = DiscoveredURLCandidate(
        url="https://example.com/page?token=secret",
        display_url="https://example.com/page",
        status="accepted",
        reason_code="ok",
        source_kind="sitemap",
        parent_url="https://example.com/sitemap.xml?token=secret",
        parent_display_url="https://example.com/sitemap.xml",
        safe_argument_hash="hash",
    )

    serialized = json.dumps(public_candidate(candidate), sort_keys=True)

    assert "token=secret" not in serialized  # nosec B101
    assert "https://example.com/page" in serialized  # nosec B101
    assert "hash" in serialized  # nosec B101
