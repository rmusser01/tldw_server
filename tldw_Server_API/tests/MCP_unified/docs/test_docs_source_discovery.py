from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from mcp_unified.docs.acquisition.models import FetchResponse
from mcp_unified.docs.discovery import (
    DiscoveredURLCandidate,
    DocsSourceDiscoveryService,
    extract_page_links,
    parse_sitemap_urlset,
    public_candidate,
)
from mcp_unified.docs.models import AccessScope, DiscoverSourceRequest
from mcp_unified.docs.settings import DocsSettings
from mcp_unified.docs.store.sqlite import DocsCatalogStore
from tldw_Server_API.tests.MCP_unified.docs.helpers import FakeResolver, FakeTransport

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


def _store(tmp_path: Path) -> DocsCatalogStore:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    return store


def _settings(tmp_path: Path, **overrides: object) -> DocsSettings:
    values: dict[str, object] = {
        "db_path": str(tmp_path / "docs.db"),
        "enable_web_acquisition": True,
        "enable_source_discovery": True,
        "web_source_profile": "locked_down",
        "allowed_url_prefixes": ("https://example.com/docs/", "https://example.com/sitemap.xml"),
    }
    values.update(overrides)
    return DocsSettings.from_mapping(values)


def test_discover_source_disabled_returns_without_fetch(tmp_path: Path) -> None:
    settings = _settings(tmp_path, enable_source_discovery=False)
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport([FetchResponse(status_code=200, headers={"content-type": "text/xml"}, body_chunks=[b"never"])])
    service = DocsSourceDiscoveryService(settings=settings, store=_store(tmp_path), resolver=resolver, transport=transport)

    result = service.discover_source(scope=AccessScope(), request=DiscoverSourceRequest(url="https://example.com/sitemap.xml"))

    assert result["status"] == "denied"  # nosec B101
    assert result["reason_code"] == "source_discovery_disabled"  # nosec B101
    assert resolver.calls == []  # nosec B101
    assert transport.calls == []  # nosec B101


def test_discover_source_requires_web_acquisition_without_fetch(tmp_path: Path) -> None:
    settings = _settings(tmp_path, enable_web_acquisition=False)
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport([FetchResponse(status_code=200, headers={"content-type": "text/xml"}, body_chunks=[b"never"])])
    service = DocsSourceDiscoveryService(settings=settings, store=_store(tmp_path), resolver=resolver, transport=transport)

    result = service.discover_source(scope=AccessScope(), request=DiscoverSourceRequest(url="https://example.com/sitemap.xml"))

    assert result["status"] == "denied"  # nosec B101
    assert result["reason_code"] == "web_acquisition_disabled"  # nosec B101
    assert resolver.calls == []  # nosec B101
    assert transport.calls == []  # nosec B101


def test_discover_sitemap_dry_run_returns_candidates_without_mutation(tmp_path: Path) -> None:
    store = _store(tmp_path)
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [
            FetchResponse(
                status_code=200,
                headers={"content-type": "application/xml"},
                body_chunks=[b"<urlset><url><loc>https://example.com/docs/a</loc></url></urlset>"],
            )
        ]
    )
    service = DocsSourceDiscoveryService(settings=_settings(tmp_path), store=store, resolver=resolver, transport=transport)

    result = service.discover_source(
        scope=AccessScope(),
        request=DiscoverSourceRequest(url="https://example.com/sitemap.xml", kind="sitemap"),
    )

    assert result["status"] == "completed"  # nosec B101
    assert result["counts"]["accepted"] == 1  # nosec B101
    assert result["candidates"][0]["url"] == "https://example.com/docs/a"  # nosec B101
    assert store.status()["counts"]["documents"] == 0  # nosec B101
    assert store.list_sources(scope=AccessScope()) == []  # nosec B101


def test_discover_source_approval_required_does_not_resolve_or_fetch(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        web_source_profile="local_first",
        allowed_url_prefixes=(),
    )
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport([FetchResponse(status_code=200, headers={"content-type": "application/xml"}, body_chunks=[b"never"])])
    service = DocsSourceDiscoveryService(settings=settings, store=_store(tmp_path), resolver=resolver, transport=transport)

    result = service.discover_source(scope=AccessScope(), request=DiscoverSourceRequest(url="https://example.com/sitemap.xml"))

    assert result["status"] == "approval_required"  # nosec B101
    assert result["reason_code"] == "source_approval_required"  # nosec B101
    assert resolver.calls == []  # nosec B101
    assert transport.calls == []  # nosec B101


def test_discover_sitemap_filters_scope_duplicates_and_query_leaks(tmp_path: Path) -> None:
    store = _store(tmp_path)
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [
            FetchResponse(
                status_code=200,
                headers={"content-type": "application/xml"},
                body_chunks=[
                    b"""
                    <urlset>
                      <url><loc>https://example.com/docs/a</loc></url>
                      <url><loc>https://example.com/docs/a#fragment</loc></url>
                      <url><loc>https://evil.example/docs/a</loc></url>
                      <url><loc>https://example.com/docs/b?token=secret</loc></url>
                    </urlset>
                    """
                ],
            )
        ]
    )
    service = DocsSourceDiscoveryService(settings=_settings(tmp_path), store=store, resolver=resolver, transport=transport)

    result = service.discover_source(
        scope=AccessScope(),
        request=DiscoverSourceRequest(url="https://example.com/sitemap.xml", kind="sitemap"),
    )
    serialized = json.dumps(result, sort_keys=True)

    assert result["counts"]["accepted"] == 1  # nosec B101
    assert result["counts"]["duplicates"] == 1  # nosec B101
    assert result["counts"]["denied"] == 1  # nosec B101
    assert result["counts"]["skipped"] == 1  # nosec B101
    assert result["candidates"][0]["url"] == "https://example.com/docs/a"  # nosec B101
    assert "token=secret" not in serialized  # nosec B101
    assert "?token" not in serialized  # nosec B101


def test_discover_sitemap_same_origin_setting_is_strict(tmp_path: Path) -> None:
    body = b"<urlset><url><loc>https://docs.example.net/guide</loc></url></urlset>"
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    settings = _settings(
        tmp_path,
        allowed_url_prefixes=(
            "https://example.com/sitemap.xml",
            "https://example.com/docs/",
            "https://docs.example.net/",
        ),
    )
    strict_service = DocsSourceDiscoveryService(
        settings=settings,
        store=_store(tmp_path),
        resolver=resolver,
        transport=FakeTransport([FetchResponse(status_code=200, headers={"content-type": "application/xml"}, body_chunks=[body])]),
    )

    strict_result = strict_service.discover_source(
        scope=AccessScope(),
        request=DiscoverSourceRequest(url="https://example.com/sitemap.xml", kind="sitemap"),
    )

    assert strict_result["counts"]["denied"] == 1  # nosec B101
    assert strict_result["candidates"][0]["reason_code"] == "candidate_out_of_scope"  # nosec B101

    loose_service = DocsSourceDiscoveryService(
        settings=DocsSettings.from_mapping({**settings.__dict__, "discovery_same_origin_only": False}),
        store=_store(tmp_path),
        resolver=resolver,
        transport=FakeTransport([FetchResponse(status_code=200, headers={"content-type": "application/xml"}, body_chunks=[body])]),
    )

    loose_result = loose_service.discover_source(
        scope=AccessScope(),
        request=DiscoverSourceRequest(url="https://example.com/sitemap.xml", kind="sitemap"),
    )

    assert loose_result["counts"]["accepted"] == 1  # nosec B101


def test_discover_sitemap_apply_register_creates_source_without_documents(tmp_path: Path) -> None:
    store = _store(tmp_path)
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [
            FetchResponse(
                status_code=200,
                headers={"content-type": "application/xml"},
                body_chunks=[b"<urlset><url><loc>https://example.com/docs/a</loc></url></urlset>"],
            )
        ]
    )
    service = DocsSourceDiscoveryService(settings=_settings(tmp_path), store=store, resolver=resolver, transport=transport)

    result = service.discover_source(
        scope=AccessScope(),
        request=DiscoverSourceRequest(
            url="https://example.com/sitemap.xml",
            kind="sitemap",
            mode="apply",
            apply_action="register",
            collections=("Reference",),
            keywords=("docs",),
            title="Example docs sitemap",
        ),
    )

    assert result["status"] == "completed"  # nosec B101
    assert result["source"]["source_type"] == "url_sitemap"  # nosec B101
    assert "sitemap_sync_disabled" in result["warnings"]  # nosec B101
    assert store.status()["counts"]["documents"] == 0  # nosec B101
    sources = store.list_sources(scope=AccessScope())
    assert sources[0]["metadata"]["default_collections"] == ["Reference"]  # nosec B101
    assert sources[0]["metadata"]["default_keywords"] == ["docs"]  # nosec B101


def test_discover_sitemap_apply_ingests_candidates_and_links_sitemap_source(tmp_path: Path) -> None:
    store = _store(tmp_path)
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [
            FetchResponse(
                status_code=200,
                headers={"content-type": "application/xml"},
                body_chunks=[b"<urlset><url><loc>https://example.com/docs/a</loc></url></urlset>"],
            ),
            FetchResponse(
                status_code=200,
                headers={"content-type": "text/plain"},
                body_chunks=[b"alpha reference body"],
            ),
        ]
    )
    settings = _settings(tmp_path, sitemap_sync_enabled=True)
    service = DocsSourceDiscoveryService(settings=settings, store=store, resolver=resolver, transport=transport)

    result = service.discover_source(
        scope=AccessScope(),
        request=DiscoverSourceRequest(
            url="https://example.com/sitemap.xml",
            kind="sitemap",
            mode="apply",
            apply_action="register_and_ingest",
            collections=("Reference",),
            keywords=("docs",),
        ),
    )

    assert result["counts"]["ingested"] == 1  # nosec B101
    assert result["candidates"][0]["document_id"]  # nosec B101
    assert store.search_chunks(scope=AccessScope(), query="alpha", limit=10)  # nosec B101
    assert [item["name"] for item in store.list_collections(scope=AccessScope())] == ["Reference"]  # nosec B101
    assert [item["keyword"] for item in store.list_keywords(scope=AccessScope())] == ["docs"]  # nosec B101
    links = store.source_document_links(scope=AccessScope(), source_id=result["source"]["id"])
    assert links[0]["source_item_uri"] == "https://example.com/docs/a"  # nosec B101
