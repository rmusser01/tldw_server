from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from mcp_unified.docs.acquisition.models import FetchResponse, ResolvedAddress, URLRequest
from mcp_unified.docs.acquisition.service import DocsAcquisitionService
from mcp_unified.docs.models import AccessScope, ContextRequest, SearchFilters, SearchRequest
from mcp_unified.docs.retrieval.context import DocsContextBuilder
from mcp_unified.docs.retrieval.search import DocsRetrievalService
from mcp_unified.docs.settings import DocsSettings
from mcp_unified.docs.store.sqlite import DocsCatalogStore


class FakeResolver:
    def __init__(self, addresses: dict[str, list[str]]) -> None:
        self.addresses = addresses
        self.calls: list[tuple[str, int]] = []

    def resolve(self, host: str, port: int) -> Iterable[ResolvedAddress]:
        self.calls.append((host, port))
        return [ResolvedAddress(host=host, ip=ip, port=port) for ip in self.addresses[host]]


class FakeTransport:
    dials_validated_address = True

    def __init__(self, responses: list[FetchResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[ResolvedAddress, URLRequest, float]] = []

    def request(
        self,
        *,
        address: ResolvedAddress,
        request: URLRequest,
        timeout_seconds: float,
    ) -> FetchResponse:
        self.calls.append((address, request, timeout_seconds))
        return self.responses.pop(0)


def _store(tmp_path: Path) -> DocsCatalogStore:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    return store


def test_service_returns_capability_disabled_before_policy_or_fetch(tmp_path: Path) -> None:
    settings = DocsSettings.from_mapping(
        {
            "enable_web_acquisition": False,
            "web_source_profile": "online_capable",
            "allow_arbitrary_public_domains": True,
        }
    )
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [FetchResponse(status_code=200, headers={"content-type": "text/html"}, body_chunks=[b"never"])]
    )
    service = DocsAcquisitionService(settings=settings, store=_store(tmp_path), resolver=resolver, transport=transport)

    result = service.ingest_url(scope=AccessScope(), url="https://example.com/docs")

    assert result["status"] == "failed"  # nosec B101
    assert result["reason_code"] == "capability_disabled"  # nosec B101
    assert resolver.calls == []  # nosec B101
    assert transport.calls == []  # nosec B101


def test_service_returns_approval_required_without_fetch(tmp_path: Path) -> None:
    settings = DocsSettings.from_mapping({"enable_web_acquisition": True, "web_source_profile": "local_first"})
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [FetchResponse(status_code=200, headers={"content-type": "text/html"}, body_chunks=[b"never"])]
    )
    service = DocsAcquisitionService(settings=settings, store=_store(tmp_path), resolver=resolver, transport=transport)

    result = service.ingest_url(scope=AccessScope(), url="https://example.com/docs")

    assert result["status"] == "approval_required"  # nosec B101
    assert result["reason_code"] == "source_approval_required"  # nosec B101
    assert resolver.calls == []  # nosec B101
    assert transport.calls == []  # nosec B101


def test_service_fails_closed_when_robots_is_enabled(tmp_path: Path) -> None:
    settings = DocsSettings.from_mapping(
        {
            "enable_web_acquisition": True,
            "web_source_profile": "online_capable",
            "allow_arbitrary_public_domains": True,
            "respect_robots": True,
        }
    )
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [FetchResponse(status_code=200, headers={"content-type": "text/html"}, body_chunks=[b"never"])]
    )
    service = DocsAcquisitionService(settings=settings, store=_store(tmp_path), resolver=resolver, transport=transport)

    result = service.ingest_url(scope=AccessScope(), url="https://example.com/docs")

    assert result["status"] == "denied"  # nosec B101
    assert result["reason_code"] == "robots_unavailable"  # nosec B101
    assert resolver.calls == []  # nosec B101
    assert transport.calls == []  # nosec B101


def test_service_ingests_approved_page_into_search_and_context(tmp_path: Path) -> None:
    settings = DocsSettings.from_mapping(
        {
            "enable_web_acquisition": True,
            "web_source_profile": "locked_down",
            "allowed_url_prefixes": ("https://example.com/docs/",),
        }
    )
    store = _store(tmp_path)
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [
            FetchResponse(
                status_code=200,
                headers={"content-type": "text/html"},
                body_chunks=[b"<h1>SQLite Guide</h1><p>FTS5 indexing details.</p>"],
            )
        ]
    )
    service = DocsAcquisitionService(settings=settings, store=store, resolver=resolver, transport=transport)
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")

    result = service.ingest_url(
        scope=scope,
        url="https://example.com/docs/sqlite.html",
        keywords=("sqlite", "fts5"),
        collection_names=("Reference",),
    )
    search = DocsRetrievalService(store).search(
        scope=scope,
        request=SearchRequest(query="FTS5", filters=SearchFilters(collection="Reference", keywords=("fts5",))),
    )
    context = DocsContextBuilder(DocsRetrievalService(store)).build(scope=scope, request=ContextRequest(query="FTS5"))

    assert result["status"] == "created"  # nosec B101
    assert result["document"]["source_url"] == "https://example.com/docs/sqlite.html"  # nosec B101
    assert result["document"]["chunks"] >= 1  # nosec B101
    assert search["results"][0]["title"] == "SQLite Guide"  # nosec B101
    assert context["chunks"]  # nosec B101


def test_service_reports_unchanged_for_same_content(tmp_path: Path) -> None:
    settings = DocsSettings.from_mapping(
        {
            "enable_web_acquisition": True,
            "web_source_profile": "online_capable",
            "allow_arbitrary_public_domains": True,
        }
    )
    store = _store(tmp_path)
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [
            FetchResponse(status_code=200, headers={"content-type": "text/plain"}, body_chunks=[b"same body"]),
            FetchResponse(status_code=200, headers={"content-type": "text/plain"}, body_chunks=[b"same body"]),
        ]
    )
    service = DocsAcquisitionService(settings=settings, store=store, resolver=resolver, transport=transport)

    first = service.ingest_url(scope=AccessScope(), url="https://example.com/readme.txt")
    second = service.ingest_url(scope=AccessScope(), url="https://example.com/readme.txt")

    assert first["status"] == "created"  # nosec B101
    assert second["status"] == "unchanged"  # nosec B101
