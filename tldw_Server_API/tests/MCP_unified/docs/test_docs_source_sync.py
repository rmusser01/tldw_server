from __future__ import annotations

from pathlib import Path

import pytest

from mcp_unified.docs.mcp_module import DocsMCPToolProvider
from mcp_unified.docs.models import AccessScope
from mcp_unified.docs.settings import DocsSettings
from mcp_unified.docs.store.sqlite import DocsCatalogStore

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
