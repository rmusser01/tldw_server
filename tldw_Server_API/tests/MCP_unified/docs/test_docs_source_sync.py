from __future__ import annotations

from contextlib import closing
from pathlib import Path

import pytest

from mcp_unified.docs.acquisition.models import FetchResponse
from mcp_unified.docs.acquisition.service import DocsAcquisitionService
from mcp_unified.docs.errors import DocsError
from mcp_unified.docs.mcp_module import DocsMCPToolProvider
from mcp_unified.docs.models import AccessScope, SyncSourceRequest
from mcp_unified.docs.settings import DocsSettings
from mcp_unified.docs.sync import DocsSourceSyncService
from mcp_unified.docs.store.sqlite import DocsCatalogStore
from tldw_Server_API.tests.MCP_unified.docs.helpers import FakeResolver, FakeTransport

pytestmark = pytest.mark.unit

ZERO_SYNC_COUNTS = {
    "created": 0,
    "updated": 0,
    "unchanged": 0,
    "missing": 0,
    "tombstoned": 0,
    "failed": 0,
    "skipped": 0,
}


def _document_count(items: list[dict], name: str) -> int:
    for item in items:
        if item.get("name") == name or item.get("keyword") == name:
            return int(item["document_count"])
    raise AssertionError(f"{name!r} not found")


def test_url_page_sync_apply_refreshes_existing_source_with_fake_transport(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    settings = DocsSettings.from_mapping(
        {
            "db_path": str(tmp_path / "docs.db"),
            "enable_web_acquisition": True,
            "web_source_profile": "online_capable",
            "allow_arbitrary_public_domains": True,
        }
    )
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [
            FetchResponse(status_code=200, headers={"content-type": "text/plain"}, body_chunks=[b"old sqlite body"]),
            FetchResponse(status_code=200, headers={"content-type": "text/plain"}, body_chunks=[b"new sqlite body"]),
        ]
    )
    acquisition = DocsAcquisitionService(settings=settings, store=store, resolver=resolver, transport=transport)
    scope = AccessScope()
    ingested = acquisition.ingest_url(scope=scope, url="https://example.com/page")
    service = DocsSourceSyncService(settings=settings, store=store, resolver=resolver, transport=transport)

    result = service.sync_source(scope=scope, request=SyncSourceRequest(source_id=ingested["source"]["id"], mode="apply"))
    search = store.search_chunks(scope=scope, query="new", limit=10)

    assert result["counts"]["updated"] == 1  # nosec B101
    assert search[0]["title"]  # nosec B101


def test_url_page_sync_does_not_fetch_when_policy_requires_approval(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    scope = AccessScope()
    source_id = store.upsert_source(
        scope=scope,
        source_type="url_page",
        canonical_uri="https://example.com/page",
        display_name="page",
        source_path=None,
        source_url="https://example.com/page",
        redacted_source_url="https://example.com/page",
        policy_profile="local_first",
        sync_enabled=True,
        metadata={},
    )
    settings = DocsSettings.from_mapping(
        {
            "db_path": str(tmp_path / "docs.db"),
            "enable_web_acquisition": True,
            "web_source_profile": "local_first",
        }
    )
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport([FetchResponse(status_code=200, body_chunks=[b"never"])])
    service = DocsSourceSyncService(settings=settings, store=store, resolver=resolver, transport=transport)

    result = service.sync_source(scope=scope, request=SyncSourceRequest(source_id=source_id, mode="apply"))

    assert result["status"] == "denied"  # nosec B101
    assert result["reason_code"] == "approval_required"  # nosec B101
    assert resolver.calls == []  # nosec B101
    assert transport.calls == []  # nosec B101


def test_url_page_sync_dry_run_does_not_mutate_content_links_or_runs(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    settings = DocsSettings.from_mapping(
        {
            "db_path": str(tmp_path / "docs.db"),
            "enable_web_acquisition": True,
            "web_source_profile": "online_capable",
            "allow_arbitrary_public_domains": True,
        }
    )
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [
            FetchResponse(status_code=200, headers={"content-type": "text/plain"}, body_chunks=[b"old sqlite body"]),
            FetchResponse(status_code=200, headers={"content-type": "text/plain"}, body_chunks=[b"new sqlite body"]),
        ]
    )
    acquisition = DocsAcquisitionService(settings=settings, store=store, resolver=resolver, transport=transport)
    scope = AccessScope()
    ingested = acquisition.ingest_url(scope=scope, url="https://example.com/page")
    source_id = ingested["source"]["id"]
    source_before = store.get_source(scope=scope, source_id=source_id)
    links_before = store.source_document_links(scope=scope, source_id=source_id)
    service = DocsSourceSyncService(settings=settings, store=store, resolver=resolver, transport=transport)

    result = service.sync_source(
        scope=scope,
        request=SyncSourceRequest(source_id=source_id, mode="dry_run"),
    )
    search_new = store.search_chunks(scope=scope, query="new", limit=10)
    search_old = store.search_chunks(scope=scope, query="old", limit=10)
    status = store.status()

    assert result["counts"]["updated"] == 1  # nosec B101
    assert search_new == []  # nosec B101
    assert search_old  # nosec B101
    assert store.get_source(scope=scope, source_id=source_id) == source_before  # nosec B101
    assert store.source_document_links(scope=scope, source_id=source_id) == links_before  # nosec B101
    assert status["counts"]["sync_runs"] == 0  # nosec B101


def test_url_page_sync_redacts_query_bearing_source_summary(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    settings = DocsSettings.from_mapping(
        {
            "db_path": str(tmp_path / "docs.db"),
            "enable_web_acquisition": True,
            "web_source_profile": "online_capable",
            "allow_arbitrary_public_domains": True,
            "persist_url_query_strings": True,
        }
    )
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [
            FetchResponse(status_code=200, headers={"content-type": "text/plain"}, body_chunks=[b"old sqlite body"]),
            FetchResponse(status_code=200, headers={"content-type": "text/plain"}, body_chunks=[b"new sqlite body"]),
        ]
    )
    acquisition = DocsAcquisitionService(settings=settings, store=store, resolver=resolver, transport=transport)
    scope = AccessScope()
    ingested = acquisition.ingest_url(scope=scope, url="https://example.com/page?token=secret")
    service = DocsSourceSyncService(settings=settings, store=store, resolver=resolver, transport=transport)

    result = service.sync_source(
        scope=scope,
        request=SyncSourceRequest(source_id=ingested["source"]["id"], mode="apply"),
    )

    assert "token=secret" not in repr(result["source"])  # nosec B101
    assert result["source"]["canonical_uri"] == "https://example.com/page"  # nosec B101
    assert result["source"]["display_uri"] == "https://example.com/page"  # nosec B101
    assert result["source"]["redacted_source_url"] == "https://example.com/page"  # nosec B101


def test_url_page_sync_private_address_denial_preserves_old_content_without_run(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    settings = DocsSettings.from_mapping(
        {
            "db_path": str(tmp_path / "docs.db"),
            "enable_web_acquisition": True,
            "web_source_profile": "online_capable",
            "allow_arbitrary_public_domains": True,
        }
    )
    scope = AccessScope()
    ingest_transport = FakeTransport(
        [FetchResponse(status_code=200, headers={"content-type": "text/plain"}, body_chunks=[b"old sqlite body"])]
    )
    acquisition = DocsAcquisitionService(
        settings=settings,
        store=store,
        resolver=FakeResolver({"example.com": ["93.184.216.34"]}),
        transport=ingest_transport,
    )
    ingested = acquisition.ingest_url(scope=scope, url="https://example.com/page")
    source_id = ingested["source"]["id"]
    sync_resolver = FakeResolver({"example.com": ["127.0.0.1"]})
    sync_transport = FakeTransport([FetchResponse(status_code=200, body_chunks=[b"new sqlite body"])])
    service = DocsSourceSyncService(settings=settings, store=store, resolver=sync_resolver, transport=sync_transport)

    result = service.sync_source(scope=scope, request=SyncSourceRequest(source_id=source_id, mode="apply"))
    search_old = store.search_chunks(scope=scope, query="old", limit=10)
    search_new = store.search_chunks(scope=scope, query="new", limit=10)
    status = store.status()

    assert result["status"] == "denied"  # nosec B101
    assert result["reason_code"] == "egress_private_address_denied"  # nosec B101
    assert search_old  # nosec B101
    assert search_new == []  # nosec B101
    assert sync_transport.calls == []  # nosec B101
    assert status["counts"]["sync_runs"] == 0  # nosec B101


def test_url_page_sync_retires_previous_redirect_target_when_canonical_changes_with_new_body(
    tmp_path: Path,
) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    settings = DocsSettings.from_mapping(
        {
            "db_path": str(tmp_path / "docs.db"),
            "enable_web_acquisition": True,
            "web_source_profile": "online_capable",
            "allow_arbitrary_public_domains": True,
        }
    )
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [
            FetchResponse(status_code=302, headers={"location": "https://example.com/b"}),
            FetchResponse(status_code=200, headers={"content-type": "text/plain"}, body_chunks=[b"oldbmarker body"]),
            FetchResponse(status_code=302, headers={"location": "https://example.com/c?token=secret"}),
            FetchResponse(status_code=200, headers={"content-type": "text/plain"}, body_chunks=[b"newcmarker body"]),
        ]
    )
    acquisition = DocsAcquisitionService(settings=settings, store=store, resolver=resolver, transport=transport)
    scope = AccessScope()
    ingested = acquisition.ingest_url(scope=scope, url="https://example.com/a")
    source_id = ingested["source"]["id"]
    service = DocsSourceSyncService(settings=settings, store=store, resolver=resolver, transport=transport)

    result = service.sync_source(scope=scope, request=SyncSourceRequest(source_id=source_id, mode="apply"))
    listed = DocsMCPToolProvider(settings=settings, store=store).execute("docs.list", {"kind": "sources"}, scope=scope)
    stored_source = store.get_source(scope=scope, source_id=source_id)
    links = store.source_document_links(scope=scope, source_id=source_id)
    active_links = [link for link in links if link["status"] == "active"]
    old_search = store.search_chunks(scope=scope, query="oldbmarker", limit=10)
    new_search = store.search_chunks(scope=scope, query="newcmarker", limit=10)
    future_transport = FakeTransport(
        [FetchResponse(status_code=200, headers={"content-type": "text/plain"}, body_chunks=[b"newcmarker body"])]
    )
    future_service = DocsSourceSyncService(settings=settings, store=store, resolver=resolver, transport=future_transport)
    future_service.sync_source(scope=scope, request=SyncSourceRequest(source_id=source_id, mode="dry_run"))

    assert result["counts"]["updated"] == 1  # nosec B101
    assert "token=secret" not in repr(result["source"])  # nosec B101
    assert result["source"]["canonical_uri"] == "https://example.com/c"  # nosec B101
    assert result["source"]["document_count"] == 1  # nosec B101
    assert listed["sources"][0]["canonical_uri"] == "https://example.com/c"  # nosec B101
    assert listed["sources"][0]["document_count"] == 1  # nosec B101
    assert stored_source["id"] == source_id  # nosec B101
    assert stored_source["source_url"] == "https://example.com/c?token=secret"  # nosec B101
    assert [link["source_item_uri"] for link in active_links] == ["https://example.com/c?token=secret"]  # nosec B101
    assert future_transport.calls[0][1].target == "/c?token=secret"  # nosec B101
    assert old_search == []  # nosec B101
    assert new_search  # nosec B101


def test_url_page_sync_relinks_same_body_when_canonical_changes(
    tmp_path: Path,
) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    settings = DocsSettings.from_mapping(
        {
            "db_path": str(tmp_path / "docs.db"),
            "enable_web_acquisition": True,
            "web_source_profile": "online_capable",
            "allow_arbitrary_public_domains": True,
        }
    )
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [
            FetchResponse(status_code=302, headers={"location": "https://example.com/b"}),
            FetchResponse(status_code=200, headers={"content-type": "text/plain"}, body_chunks=[b"samebodymarker"]),
            FetchResponse(status_code=302, headers={"location": "https://example.com/c"}),
            FetchResponse(status_code=200, headers={"content-type": "text/plain"}, body_chunks=[b"samebodymarker"]),
        ]
    )
    acquisition = DocsAcquisitionService(settings=settings, store=store, resolver=resolver, transport=transport)
    scope = AccessScope()
    ingested = acquisition.ingest_url(scope=scope, url="https://example.com/a")
    source_id = ingested["source"]["id"]
    service = DocsSourceSyncService(settings=settings, store=store, resolver=resolver, transport=transport)

    result = service.sync_source(scope=scope, request=SyncSourceRequest(source_id=source_id, mode="apply"))
    listed = DocsMCPToolProvider(settings=settings, store=store).execute("docs.list", {"kind": "sources"}, scope=scope)
    links = store.source_document_links(scope=scope, source_id=source_id)
    active_links = [link for link in links if link["status"] == "active"]

    assert result["counts"]["updated"] == 1  # nosec B101
    assert result["counts"]["unchanged"] == 0  # nosec B101
    assert result["source"]["canonical_uri"] == "https://example.com/c"  # nosec B101
    assert result["source"]["document_count"] == 1  # nosec B101
    assert listed["sources"][0]["canonical_uri"] == "https://example.com/c"  # nosec B101
    assert listed["sources"][0]["document_count"] == 1  # nosec B101
    assert [link["source_item_uri"] for link in active_links] == ["https://example.com/c"]  # nosec B101


def test_provider_advertises_sync_source_when_enabled(tmp_path: Path) -> None:
    provider = DocsMCPToolProvider(settings=DocsSettings(db_path=tmp_path / "docs.db", trusted_roots=(tmp_path,)))

    tools = {tool["name"]: tool for tool in provider.tool_definitions()}

    assert "docs.sync_source" in tools  # nosec B101
    assert tools["docs.sync_source"]["metadata"]["category"] == "ingestion"  # nosec B101
    assert tools["docs.sync_source"]["metadata"]["readOnlyHint"] is False  # nosec B101


def test_provider_omits_sync_source_when_disabled(tmp_path: Path) -> None:
    provider = DocsMCPToolProvider(
        settings=DocsSettings(db_path=tmp_path / "docs.db", trusted_roots=(tmp_path,), enable_source_sync=False)
    )

    tools = {tool["name"]: tool for tool in provider.tool_definitions()}

    assert "docs.sync_source" not in tools  # nosec B101


def test_provider_list_sources_returns_real_sources(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")
    store.upsert_source(
        scope=scope,
        source_type="local_file",
        canonical_uri="file:///docs/guide.md",
        display_name="guide.md",
        source_path="/docs/guide.md",
        source_url=None,
        redacted_source_url=None,
        policy_profile="locked_down",
        sync_enabled=True,
        metadata={},
    )
    provider = DocsMCPToolProvider(settings=DocsSettings(db_path=tmp_path / "docs.db"), store=store)

    result = provider.execute("docs.list", {"kind": "sources"}, scope=scope)

    assert result["sources"][0]["canonical_uri"] == "file:///docs/guide.md"  # nosec B101
    assert result["warnings"] == []  # nosec B101


def test_provider_list_sources_redacts_query_bearing_url_sources(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")
    store.upsert_source(
        scope=scope,
        source_type="url_page",
        canonical_uri="https://example.com/page?token=secret",
        display_name="Example page",
        source_path=None,
        source_url="https://example.com/page?token=secret",
        redacted_source_url="https://example.com/page",
        policy_profile="locked_down",
        sync_enabled=True,
        metadata={},
    )
    provider = DocsMCPToolProvider(
        settings=DocsSettings(db_path=tmp_path / "docs.db", persist_url_query_strings=True),
        store=store,
    )

    result = provider.execute("docs.list", {"kind": "sources"}, scope=scope)

    source = result["sources"][0]
    assert "token=secret" not in repr(source)  # nosec B101
    assert "source_url" not in source  # nosec B101
    assert source["canonical_uri"] == "https://example.com/page"  # nosec B101
    assert source["display_uri"] == "https://example.com/page"  # nosec B101
    assert source["redacted_source_url"] == "https://example.com/page"  # nosec B101


def test_provider_sync_source_denies_stale_call_when_disabled(tmp_path: Path) -> None:
    provider = DocsMCPToolProvider(
        settings=DocsSettings(db_path=tmp_path / "docs.db", trusted_roots=(tmp_path,), enable_source_sync=False)
    )

    result = provider.execute("docs.sync_source", {"source_uri": "file:///docs/guide.md"}, scope=AccessScope())

    assert result == {"status": "denied", "reason_code": "source_sync_disabled"}  # nosec B101


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"source_id": 1, "source_uri": "file:///docs/guide.md"},
    ],
)
def test_provider_sync_source_validates_exactly_one_selector_without_mutation(
    tmp_path: Path,
    arguments: dict[str, object],
) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")
    source_id = store.upsert_source(
        scope=scope,
        source_type="local_file",
        canonical_uri="file:///docs/guide.md",
        display_name="guide.md",
        source_path="/docs/guide.md",
        source_url=None,
        redacted_source_url=None,
        policy_profile="locked_down",
        sync_enabled=True,
        metadata={},
    )
    before = store.get_source(scope=scope, source_id=source_id)
    provider = DocsMCPToolProvider(settings=DocsSettings(db_path=tmp_path / "docs.db"), store=store)

    result = provider.execute("docs.sync_source", arguments, scope=scope)

    assert result["status"] == "denied"  # nosec B101
    assert result["reason_code"] == "source_selector_invalid"  # nosec B101
    assert store.get_source(scope=scope, source_id=source_id) == before  # nosec B101


def test_provider_sync_source_denies_missing_source(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    provider = DocsMCPToolProvider(settings=DocsSettings(db_path=tmp_path / "docs.db"), store=store)

    result = provider.execute("docs.sync_source", {"source_id": 404}, scope=AccessScope())

    assert result["status"] == "denied"  # nosec B101
    assert result["reason_code"] == "source_not_found"  # nosec B101


def test_provider_sync_source_redacts_query_bearing_url_summary(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")
    source_id = store.upsert_source(
        scope=scope,
        source_type="url_page",
        canonical_uri="https://example.com/page?token=secret",
        display_name="Example page",
        source_path=None,
        source_url="https://example.com/page?token=secret",
        redacted_source_url="https://example.com/page",
        policy_profile="locked_down",
        sync_enabled=True,
        metadata={},
    )
    provider = DocsMCPToolProvider(
        settings=DocsSettings(db_path=tmp_path / "docs.db", persist_url_query_strings=True),
        store=store,
    )

    result = provider.execute("docs.sync_source", {"source_id": source_id}, scope=scope)

    source = result["source"]
    assert "token=secret" not in repr(source)  # nosec B101
    assert "source_url" not in source  # nosec B101
    assert source["canonical_uri"] == "https://example.com/page"  # nosec B101
    assert source["display_uri"] == "https://example.com/page"  # nosec B101
    assert source["redacted_source_url"] == "https://example.com/page"  # nosec B101
    assert result["counts"] == ZERO_SYNC_COUNTS  # nosec B101


def test_provider_sync_source_parses_force_false_string(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")
    source_id = store.upsert_source(
        scope=scope,
        source_type="local_file",
        canonical_uri="file:///docs/guide.md",
        display_name="guide.md",
        source_path="/docs/guide.md",
        source_url=None,
        redacted_source_url=None,
        policy_profile="locked_down",
        sync_enabled=True,
        metadata={},
    )
    provider = DocsMCPToolProvider(settings=DocsSettings(db_path=tmp_path / "docs.db"), store=store)

    result = provider.execute("docs.sync_source", {"source_id": source_id, "force": "false"}, scope=scope)

    assert result["force"] is False  # nosec B101


@pytest.mark.parametrize(
    ("arguments", "field"),
    [
        ({"source_id": True}, "source_id"),
        ({"source_id": 1, "max_documents": True}, "max_documents"),
        ({"source_id": 1, "max_pages": "not-an-int"}, "max_pages"),
        ({"source_id": 1, "max_documents": 0}, "max_documents"),
    ],
)
def test_provider_sync_source_returns_stable_invalid_response_for_malformed_args(
    tmp_path: Path,
    arguments: dict[str, object],
    field: str,
) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")
    store.upsert_source(
        scope=scope,
        source_type="local_file",
        canonical_uri="file:///docs/guide.md",
        display_name="guide.md",
        source_path="/docs/guide.md",
        source_url=None,
        redacted_source_url=None,
        policy_profile="locked_down",
        sync_enabled=True,
        metadata={},
    )
    provider = DocsMCPToolProvider(settings=DocsSettings(db_path=tmp_path / "docs.db"), store=store)

    result = provider.execute("docs.sync_source", arguments, scope=scope)

    assert result == {  # nosec B101
        "status": "denied",
        "reason_code": "source_sync_request_invalid",
        "field": field,
    }


def test_provider_sync_source_uses_custom_default_stale_policy(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")
    source_id = store.upsert_source(
        scope=scope,
        source_type="local_file",
        canonical_uri="file:///docs/guide.md",
        display_name="guide.md",
        source_path="/docs/guide.md",
        source_url=None,
        redacted_source_url=None,
        policy_profile="locked_down",
        sync_enabled=True,
        metadata={},
    )
    provider = DocsMCPToolProvider(
        settings=DocsSettings(db_path=tmp_path / "docs.db", default_stale_policy="tombstone"),
        store=store,
    )

    result = provider.execute("docs.sync_source", {"source_id": source_id}, scope=scope)

    assert result["stale_policy"] == "tombstone"  # nosec B101


def test_provider_sync_source_returns_stable_unsupported_response_for_sitemap(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")
    source_id = store.upsert_source(
        scope=scope,
        source_type="url_sitemap",
        canonical_uri="https://example.com/sitemap.xml",
        display_name="Example sitemap",
        source_path=None,
        source_url="https://example.com/sitemap.xml",
        redacted_source_url="https://example.com/sitemap.xml",
        policy_profile="locked_down",
        sync_enabled=True,
        metadata={},
    )
    provider = DocsMCPToolProvider(settings=DocsSettings(db_path=tmp_path / "docs.db"), store=store)

    result = provider.execute("docs.sync_source", {"source_id": source_id}, scope=scope)

    assert result["status"] == "denied"  # nosec B101
    assert result["reason_code"] == "sitemap_sync_disabled"  # nosec B101
    assert result["source"]["id"] == source_id  # nosec B101
    assert result["source"]["canonical_uri"] == "https://example.com/sitemap.xml"  # nosec B101
    assert result["counts"] == ZERO_SYNC_COUNTS  # nosec B101


def test_local_file_sync_dry_run_does_not_mutate_document_or_run_rows(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    guide = root / "guide.md"
    guide.write_text("# Guide\n\nOld sqlite content.\n", encoding="utf-8")
    provider = DocsMCPToolProvider(settings=DocsSettings(db_path=tmp_path / "docs.db", trusted_roots=(root,)))
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")
    imported = provider.execute("docs.import_path", {"path": str(guide)}, scope=scope)
    source_id = imported["source"]["id"]
    guide.write_text("# Guide\n\nNew sqlite content.\n", encoding="utf-8")

    result = provider.execute("docs.sync_source", {"source_id": source_id, "mode": "dry_run"}, scope=scope)
    search = provider.execute("docs.search", {"query": "New"}, scope=scope)
    status = provider.store.status()

    assert result["mode"] == "dry_run"  # nosec B101
    assert result["counts"]["updated"] == 1  # nosec B101
    assert search["results"] == []  # nosec B101
    assert status["counts"]["sync_runs"] == 0  # nosec B101


def test_local_file_sync_apply_updates_content_and_preserves_user_metadata(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    guide = root / "guide.md"
    guide.write_text("# Guide\n\nOld sqlite content.\n", encoding="utf-8")
    provider = DocsMCPToolProvider(settings=DocsSettings(db_path=tmp_path / "docs.db", trusted_roots=(root,)))
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")
    imported = provider.execute(
        "docs.import_path",
        {"path": str(guide), "keywords": ["source"], "collections": ["Source"]},
        scope=scope,
    )
    document_id = imported["documents"][0]["id"]
    source_id = imported["source"]["id"]
    provider.execute("docs.keywords.apply", {"document_id": document_id, "keywords": ["manual"]}, scope=scope)
    provider.execute(
        "docs.collections.set_membership",
        {"collection": "Manual", "document_id": document_id, "action": "add"},
        scope=scope,
    )
    guide.write_text("# Guide\n\nNew sqlite content.\n", encoding="utf-8")

    result = provider.execute("docs.sync_source", {"source_id": source_id, "mode": "apply"}, scope=scope)
    search = provider.execute(
        "docs.search",
        {"query": "New", "filters": {"keywords": ["manual"], "collection": "Manual"}},
        scope=scope,
    )

    assert result["counts"]["updated"] == 1  # nosec B101
    assert search["results"][0]["document_id"] == document_id  # nosec B101


def test_local_directory_sync_report_missing_does_not_hide_document(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    guide = root / "guide.md"
    guide.write_text("# Guide\n\nSQLite content.\n", encoding="utf-8")
    provider = DocsMCPToolProvider(settings=DocsSettings(db_path=tmp_path / "docs.db", trusted_roots=(root,)))
    scope = AccessScope()
    imported = provider.execute("docs.import_path", {"path": str(root)}, scope=scope)
    source_id = imported["source"]["id"]
    guide.unlink()

    result = provider.execute(
        "docs.sync_source",
        {"source_id": source_id, "mode": "apply", "stale_policy": "report"},
        scope=scope,
    )
    search = provider.execute("docs.search", {"query": "SQLite"}, scope=scope)

    assert result["counts"]["missing"] == 1  # nosec B101
    assert result["counts"]["tombstoned"] == 0  # nosec B101
    assert search["results"]  # nosec B101


def test_local_directory_sync_tombstone_hides_document_from_default_search(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    guide = root / "guide.md"
    guide.write_text("# Guide\n\nSQLite content.\n", encoding="utf-8")
    provider = DocsMCPToolProvider(settings=DocsSettings(db_path=tmp_path / "docs.db", trusted_roots=(root,)))
    scope = AccessScope()
    imported = provider.execute("docs.import_path", {"path": str(root)}, scope=scope)
    source_id = imported["source"]["id"]
    guide.unlink()

    result = provider.execute(
        "docs.sync_source",
        {"source_id": source_id, "mode": "apply", "stale_policy": "tombstone"},
        scope=scope,
    )
    search = provider.execute("docs.search", {"query": "SQLite"}, scope=scope)

    assert result["counts"]["tombstoned"] == 1  # nosec B101
    assert search["results"] == []  # nosec B101


def test_local_directory_sync_tombstone_hides_document_from_lookup_and_facets(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    guide = root / "guide.md"
    guide.write_text("# Tombstone Guide\n\nTombstone visibility content.\n", encoding="utf-8")
    provider = DocsMCPToolProvider(settings=DocsSettings(db_path=tmp_path / "docs.db", trusted_roots=(root,)))
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")
    imported = provider.execute(
        "docs.import_path",
        {"path": str(root), "keywords": ["obsolete"], "collections": ["Archive"]},
        scope=scope,
    )
    document_id = imported["documents"][0]["id"]
    source_id = imported["source"]["id"]
    with closing(provider.store.connect()) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO docs_aliases (owner_scope, profile_scope, name, document_id)
                VALUES (?, ?, ?, ?)
                """,
                (scope.owner_scope, scope.profile_scope, "old-guide", document_id),
            )
    guide.unlink()

    result = provider.execute(
        "docs.sync_source",
        {"source_id": source_id, "mode": "apply", "stale_policy": "tombstone"},
        scope=scope,
    )
    search = provider.execute("docs.search", {"query": "visibility"}, scope=scope)
    resolved = provider.execute("docs.resolve", {"name": "Tombstone Guide"}, scope=scope)
    collections = provider.execute("docs.list", {"kind": "collections"}, scope=scope)
    keywords = provider.execute("docs.list", {"kind": "keywords"}, scope=scope)

    assert result["counts"]["tombstoned"] == 1  # nosec B101
    assert search["results"] == []  # nosec B101
    with pytest.raises(DocsError) as direct_get:
        provider.execute("docs.get", {"id": str(document_id)}, scope=scope)
    with pytest.raises(DocsError) as alias_get:
        provider.execute("docs.get", {"id": "old-guide"}, scope=scope)
    assert direct_get.value.code == "document_not_found"  # nosec B101
    assert alias_get.value.code == "document_not_found"  # nosec B101
    assert resolved["matches"] == []  # nosec B101
    assert _document_count(collections["collections"], "Archive") == 0  # nosec B101
    assert _document_count(keywords["keywords"], "obsolete") == 0  # nosec B101


def test_local_directory_sync_counts_stale_items_toward_limit_before_tombstoning(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "a.md").write_text("# A\n\nSQLite A.\n", encoding="utf-8")
    (root / "b.md").write_text("# B\n\nSQLite B.\n", encoding="utf-8")
    provider = DocsMCPToolProvider(settings=DocsSettings(db_path=tmp_path / "docs.db", trusted_roots=(root,)))
    scope = AccessScope()
    imported = provider.execute("docs.import_path", {"path": str(root)}, scope=scope)
    source_id = imported["source"]["id"]
    for file_path in root.iterdir():
        file_path.unlink()

    result = provider.execute(
        "docs.sync_source",
        {"source_id": source_id, "mode": "apply", "stale_policy": "tombstone", "max_documents": 1},
        scope=scope,
    )
    search = provider.execute("docs.search", {"query": "SQLite"}, scope=scope)

    assert result["status"] == "denied"  # nosec B101
    assert result["reason_code"] == "source_sync_limit_exceeded"  # nosec B101
    assert result["counts"]["tombstoned"] == 0  # nosec B101
    assert sorted(match["title"] for match in search["results"]) == ["A", "B"]  # nosec B101


def test_local_directory_sync_limit_exceeded_before_parsing_candidates(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    guide = root / "guide.md"
    guide.write_text("# Guide\n\nSQLite content.\n", encoding="utf-8")
    provider = DocsMCPToolProvider(
        settings=DocsSettings(db_path=tmp_path / "docs.db", trusted_roots=(root,), max_import_file_bytes=64)
    )
    scope = AccessScope()
    imported = provider.execute("docs.import_path", {"path": str(root)}, scope=scope)
    source_id = imported["source"]["id"]
    large = root / "large.md"
    large.write_text("# Large\n\n" + ("too large " * 20), encoding="utf-8")

    result = provider.execute(
        "docs.sync_source",
        {"source_id": source_id, "mode": "apply", "max_documents": 1},
        scope=scope,
    )
    search = provider.execute("docs.search", {"query": "large"}, scope=scope)
    status = provider.store.status()

    assert result["status"] == "denied"  # nosec B101
    assert result["reason_code"] == "source_sync_limit_exceeded"  # nosec B101
    assert search["results"] == []  # nosec B101
    assert status["counts"]["sync_runs"] == 0  # nosec B101


def test_local_directory_sync_skips_symlink_escaping_source_directory(tmp_path: Path) -> None:
    trusted_root = tmp_path
    root = trusted_root / "workspace"
    root.mkdir()
    (root / "public.md").write_text("# Public\n\nSQLite public.\n", encoding="utf-8")
    secret = trusted_root / "secret.md"
    secret.write_text("# Secret\n\nClassified sibling content.\n", encoding="utf-8")
    provider = DocsMCPToolProvider(
        settings=DocsSettings(db_path=tmp_path / "docs.db", trusted_roots=(trusted_root,))
    )
    scope = AccessScope()
    imported = provider.execute("docs.import_path", {"path": str(root)}, scope=scope)
    source_id = imported["source"]["id"]
    link = root / "secret.md"
    try:
        link.symlink_to(secret)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable in this environment: {exc}")

    result = provider.execute("docs.sync_source", {"source_id": source_id, "mode": "apply"}, scope=scope)
    search = provider.execute("docs.search", {"query": "Classified"}, scope=scope)

    assert result["counts"]["created"] == 0  # nosec B101
    assert search["results"] == []  # nosec B101


def test_local_file_sync_apply_unchanged_does_not_rewrite_without_force(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    guide = root / "guide.md"
    guide.write_text("# Guide\n\nSQLite content.\n", encoding="utf-8")
    provider = DocsMCPToolProvider(settings=DocsSettings(db_path=tmp_path / "docs.db", trusted_roots=(root,)))
    scope = AccessScope()
    imported = provider.execute("docs.import_path", {"path": str(guide)}, scope=scope)
    document_id = imported["documents"][0]["id"]
    source_id = imported["source"]["id"]
    before = provider.execute("docs.get", {"id": str(document_id), "mode": "chunk"}, scope=scope)
    before_chunk_ids = [chunk["id"] for chunk in before["chunks"]]

    unchanged = provider.execute("docs.sync_source", {"source_id": source_id, "mode": "apply"}, scope=scope)
    after_unchanged = provider.execute("docs.get", {"id": str(document_id), "mode": "chunk"}, scope=scope)
    forced = provider.execute(
        "docs.sync_source",
        {"source_id": source_id, "mode": "apply", "force": True},
        scope=scope,
    )
    after_forced = provider.execute("docs.get", {"id": str(document_id), "mode": "chunk"}, scope=scope)

    assert unchanged["counts"]["unchanged"] == 1  # nosec B101
    assert after_unchanged["chunks"] and [chunk["id"] for chunk in after_unchanged["chunks"]] == before_chunk_ids  # nosec B101
    assert forced["counts"]["updated"] == 1  # nosec B101
    assert [chunk["id"] for chunk in after_forced["chunks"]] != before_chunk_ids  # nosec B101


def test_local_file_sync_report_missing_deleted_source_file_keeps_document_visible(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    guide = root / "guide.md"
    guide.write_text("# Guide\n\nSQLite content.\n", encoding="utf-8")
    provider = DocsMCPToolProvider(settings=DocsSettings(db_path=tmp_path / "docs.db", trusted_roots=(root,)))
    scope = AccessScope()
    imported = provider.execute("docs.import_path", {"path": str(guide)}, scope=scope)
    source_id = imported["source"]["id"]
    guide.unlink()

    result = provider.execute(
        "docs.sync_source",
        {"source_id": source_id, "mode": "apply", "stale_policy": "report"},
        scope=scope,
    )
    search = provider.execute("docs.search", {"query": "SQLite"}, scope=scope)

    assert result["counts"]["missing"] == 1  # nosec B101
    assert result["counts"]["tombstoned"] == 0  # nosec B101
    assert search["results"]  # nosec B101
