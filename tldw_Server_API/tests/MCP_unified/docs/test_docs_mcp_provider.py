from __future__ import annotations

from pathlib import Path

import pytest

from mcp_unified.docs.mcp_module import DocsMCPToolProvider
from mcp_unified.docs.models import AccessScope
from mcp_unified.docs.settings import DocsSettings
from mcp_unified.docs.store.sqlite import DocsCatalogStore

pytestmark = pytest.mark.unit


def _provider(
    tmp_path: Path,
    settings: DocsSettings | None = None,
) -> tuple[DocsMCPToolProvider, AccessScope, int]:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")
    document_id = store.upsert_document(
        scope=scope,
        title="SQLite Reference",
        document_type="markdown",
        canonical_uri="file:///docs/sqlite.md",
        source_path="/docs/sqlite.md",
        source_url=None,
        text="SQLite FTS5 reference for agents.",
        sections=[],
        chunks=[{"text": "SQLite FTS5 reference for agents.", "citation": "sqlite.md:1"}],
        keywords=("database",),
        collection_names=("sqlite",),
        metadata={"package": "sqlite", "version": "3"},
    )
    settings = settings or DocsSettings(db_path=tmp_path / "docs.db", trusted_roots=(tmp_path,))
    return DocsMCPToolProvider(settings=settings, store=store), scope, document_id


def test_context7_resolve_library_id_prefers_package_like_collection(tmp_path: Path) -> None:
    provider, scope, _document_id = _provider(tmp_path)

    result = provider.execute("resolve-library-id", {"libraryName": "sqlite"}, scope=scope)

    assert result["canonical_tool"] == "docs.resolve"  # nosec B101
    assert result["matches"][0]["id"] == "sqlite"  # nosec B101
    assert result["matches"][0]["canonical_tool"] == "docs.resolve"  # nosec B101


def test_context7_get_library_docs_routes_to_context(tmp_path: Path) -> None:
    provider, scope, _document_id = _provider(tmp_path)

    result = provider.execute(
        "get-library-docs",
        {"context7CompatibleLibraryID": "sqlite", "topic": "FTS5", "tokens": 1},
        scope=scope,
    )

    assert result["canonical_tool"] == "docs.context"  # nosec B101
    assert result["chunks"]  # nosec B101
    assert result["budget"]["used_characters"] <= 4  # nosec B101


def test_general_resolve_does_not_force_library_version_semantics(tmp_path: Path) -> None:
    provider, scope, _document_id = _provider(tmp_path)

    result = provider.execute("docs.resolve", {"name": "database"}, scope=scope)

    assert result["matches"]  # nosec B101
    assert result["matches"][0]["target_type"] == "keyword"  # nosec B101
    assert result["query"] == "database"  # nosec B101


def test_provider_advertises_stage1_tools_without_ingest_url(tmp_path: Path) -> None:
    provider, _scope, _document_id = _provider(tmp_path)

    tools = {tool["name"]: tool for tool in provider.tool_definitions()}
    names = set(tools)

    assert "docs.search" in names  # nosec B101
    assert "docs.context" in names  # nosec B101
    assert "docs.import_path" in names  # nosec B101
    assert "resolve-library-id" in names  # nosec B101
    assert "get-library-docs" in names  # nosec B101
    assert "tokens" in tools["get-library-docs"]["inputSchema"]["properties"]  # nosec B101
    assert "docs.ingest_url" not in names  # nosec B101


def test_provider_stale_ingest_url_call_is_disabled_when_not_advertised(tmp_path: Path) -> None:
    provider, scope, _document_id = _provider(tmp_path)

    result = provider.execute("docs.ingest_url", {"url": "https://example.com/docs"}, scope=scope)

    assert result["status"] == "capability_disabled"  # nosec B101
    assert result["reason_code"] == "web_acquisition_disabled"  # nosec B101


def test_provider_advertises_ingest_url_when_enabled(tmp_path: Path) -> None:
    settings = DocsSettings.from_mapping(
        {
            "db_path": str(tmp_path / "docs.db"),
            "enable_web_acquisition": True,
            "web_source_profile": "locked_down",
            "allowed_url_prefixes": ("https://example.com/docs/",),
        }
    )
    provider = DocsMCPToolProvider(settings=settings)

    tools = {tool["name"]: tool for tool in provider.tool_definitions()}

    assert "docs.ingest_url" in tools  # nosec B101
    assert tools["docs.ingest_url"]["metadata"]["category"] == "ingestion"  # nosec B101
    assert tools["docs.ingest_url"]["metadata"]["readOnlyHint"] is False  # nosec B101


def test_provider_ingest_url_delegates_to_acquisition_service(tmp_path: Path) -> None:
    class FakeAcquisition:
        def __init__(self) -> None:
            self.calls = []

        def ingest_url(self, **kwargs):
            self.calls.append(kwargs)
            return {"status": "created", "reason_code": "ok"}

    settings = DocsSettings.from_mapping({"db_path": str(tmp_path / "docs.db"), "enable_web_acquisition": True})
    provider = DocsMCPToolProvider(settings=settings)
    acquisition = FakeAcquisition()
    provider.acquisition = acquisition
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")

    result = provider.execute(
        "docs.ingest_url",
        {"url": "https://example.com/docs", "keywords": ["sqlite"], "collections": ["Reference"], "title": "Guide"},
        scope=scope,
    )

    assert result["status"] == "created"  # nosec B101
    assert acquisition.calls == [  # nosec B101
        {
            "scope": scope,
            "url": "https://example.com/docs",
            "keywords": ("sqlite",),
            "collection_names": ("Reference",),
            "title_override": "Guide",
        }
    ]


def test_provider_rejects_empty_ingest_url_when_enabled(tmp_path: Path) -> None:
    settings = DocsSettings.from_mapping({"db_path": str(tmp_path / "docs.db"), "enable_web_acquisition": True})
    provider = DocsMCPToolProvider(settings=settings)

    try:
        provider.execute("docs.ingest_url", {"url": "   "}, scope=AccessScope())
    except ValueError as exc:
        assert "url is required" in str(exc)  # nosec B101
    else:
        raise AssertionError("Expected empty docs.ingest_url url to be rejected")


@pytest.mark.parametrize(
    ("tool_name", "arguments", "message"),
    [
        ("docs.search", {}, "query is required"),
        ("docs.get", {}, "id is required"),
        ("docs.list", {}, "kind is required"),
        ("docs.resolve", {}, "name is required"),
        ("resolve-library-id", {}, "libraryName is required"),
        ("get-library-docs", {}, "context7CompatibleLibraryID is required"),
    ],
)
def test_provider_reports_missing_required_fields(
    tmp_path: Path,
    tool_name: str,
    arguments: dict,
    message: str,
) -> None:
    provider, scope, _document_id = _provider(tmp_path)

    with pytest.raises(ValueError, match=message):
        provider.execute(tool_name, arguments, scope=scope)


def test_provider_marks_write_tools_with_ingestion_or_management_category(tmp_path: Path) -> None:
    provider, _scope, _document_id = _provider(tmp_path)
    write_names = {
        "docs.import_path",
        "docs.collections.create",
        "docs.collections.update",
        "docs.collections.set_membership",
        "docs.keywords.apply",
    }

    tools = {tool["name"]: tool for tool in provider.tool_definitions()}

    for name in write_names:
        assert tools[name]["metadata"]["category"] in {"ingestion", "management"}  # nosec B101
        assert tools[name]["metadata"]["readOnlyHint"] is False  # nosec B101


def test_provider_management_tools_update_collections_and_keywords(tmp_path: Path) -> None:
    provider, scope, document_id = _provider(tmp_path)

    created = provider.execute(
        "docs.collections.create",
        {"name": "Manual", "description": "Human curated docs"},
        scope=scope,
    )
    updated = provider.execute(
        "docs.collections.update",
        {"name": "Manual", "description": "Updated docs"},
        scope=scope,
    )
    membership = provider.execute(
        "docs.collections.set_membership",
        {"collection": "Manual", "document_id": document_id, "action": "add"},
        scope=scope,
    )
    keywords = provider.execute(
        "docs.keywords.apply",
        {"document_id": document_id, "keywords": ["agent", "sqlite"]},
        scope=scope,
    )
    search = provider.execute(
        "docs.search",
        {"query": "FTS5", "filters": {"collection": "Manual", "keywords": ["agent"]}},
        scope=scope,
    )

    assert created["status"] == "created"  # nosec B101
    assert updated["status"] == "updated"  # nosec B101
    assert membership["status"] == "added"  # nosec B101
    assert keywords["keywords"] == ["agent", "sqlite"]  # nosec B101
    assert search["results"][0]["title"] == "SQLite Reference"  # nosec B101


def test_provider_status_reports_web_acquisition_disabled(tmp_path: Path) -> None:
    provider, scope, _document_id = _provider(tmp_path)

    status = provider.execute("docs.status", {}, scope=scope)

    assert status["web_acquisition_enabled"] is False  # nosec B101
    assert status["web_acquisition_available"] is False  # nosec B101
    assert status["web_extractors"] == []  # nosec B101
    assert status["web_source_profile"] == "locked_down"  # nosec B101
    assert status["web_acquisition_unavailable_reason"] == "web_acquisition_disabled"  # nosec B101
    assert status["web_policy"]["allow_arbitrary_public_domains"] is False  # nosec B101
    assert status["web_policy"]["preapproved_domains"] == []  # nosec B101
    assert status["web_policy"]["allowed_url_prefixes"] == []  # nosec B101


def test_provider_status_reports_custom_web_policy(tmp_path: Path) -> None:
    settings = DocsSettings(
        db_path=tmp_path / "docs.db",
        trusted_roots=(tmp_path,),
        enable_web_acquisition=True,
        web_source_profile="online_capable",
        preapproved_domains=("docs.python.org",),
        allowed_url_prefixes=("https://docs.python.org/3/",),
        denied_domains=("blocked.example",),
        max_url_redirects=5,
        max_url_body_bytes=4096,
        url_request_timeout_seconds=2.5,
        allowed_content_types=("text/plain",),
        url_user_agent="tldw-docs-test/1",
        respect_robots=True,
        allow_arbitrary_public_domains=True,
    )
    provider, scope, _document_id = _provider(tmp_path, settings=settings)

    status = provider.execute("docs.status", {}, scope=scope)

    assert status["web_source_profile"] == "online_capable"  # nosec B101
    assert status["web_policy"] == {  # nosec B101
        "allow_arbitrary_public_domains": True,
        "preapproved_domains": ["docs.python.org"],
        "allowed_url_prefixes": ["https://docs.python.org/3/"],
        "denied_domains": ["blocked.example"],
        "max_url_redirects": 5,
        "max_url_body_bytes": 4096,
        "url_request_timeout_seconds": 2.5,
        "allowed_content_types": ["text/plain"],
        "url_user_agent": "tldw-docs-test/1",
        "respect_robots": True,
    }


def test_provider_status_reports_enabled_static_extractors(tmp_path: Path) -> None:
    settings = DocsSettings.from_mapping(
        {
            "db_path": str(tmp_path / "docs.db"),
            "enable_web_acquisition": True,
            "web_source_profile": "locked_down",
            "allowed_url_prefixes": ("https://example.com/docs/",),
        }
    )
    provider = DocsMCPToolProvider(settings=settings)

    status = provider.execute("docs.status", {}, scope=AccessScope())

    assert status["web_acquisition_enabled"] is True  # nosec B101
    assert status["web_acquisition_available"] is True  # nosec B101
    assert status["web_acquisition_unavailable_reason"] is None  # nosec B101
    assert "static_html" in status["web_extractors"]  # nosec B101
