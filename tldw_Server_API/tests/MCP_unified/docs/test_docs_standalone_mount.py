from __future__ import annotations

from pathlib import Path

from mcp_unified.docs import AccessScope
from mcp_unified.docs.standalone import (
    StandaloneDocsProfile,
    create_standalone_docs_mount,
    standalone_docs_settings_for_profile,
)


def test_standalone_mount_defaults_to_docs_with_local_sqlite(tmp_path: Path) -> None:
    mount = create_standalone_docs_mount({"db_path": str(tmp_path / "docs.db"), "trusted_roots": [str(tmp_path)]})

    names = {tool["name"] for tool in mount.tool_definitions()}
    status = mount.execute_tool("docs.status", {}, scope=AccessScope())

    assert mount.module_id == "docs"  # nosec B101
    assert mount.name == "Docs Corpus"  # nosec B101
    assert mount.settings.db_path == tmp_path / "docs.db"  # nosec B101
    assert "docs.search" in names  # nosec B101
    assert "docs.import_path" in names  # nosec B101
    assert "docs.ingest_url" not in names  # nosec B101
    assert status["web_acquisition_enabled"] is False  # nosec B101


def test_standalone_mount_can_import_search_and_context(tmp_path: Path) -> None:
    guide = tmp_path / "guide.md"
    guide.write_text("# Guide\n\nSQLite FTS5 context for local agents.\n", encoding="utf-8")
    mount = create_standalone_docs_mount({"db_path": str(tmp_path / "docs.db"), "trusted_roots": [str(tmp_path)]})
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")

    imported = mount.execute_tool("docs.import_path", {"path": str(guide)}, scope=scope)
    search = mount.execute_tool("docs.search", {"query": "FTS5"}, scope=scope)
    context = mount.execute_tool("docs.context", {"query": "local agents"}, scope=scope)

    assert imported["status"] in {"created", "updated"}  # nosec B101
    assert search["results"]  # nosec B101
    assert context["chunks"]  # nosec B101


def test_standalone_profile_defaults_are_downgradeable(tmp_path: Path) -> None:
    locked = standalone_docs_settings_for_profile(
        StandaloneDocsProfile.LOCKED_DOWN,
        overrides={"db_path": str(tmp_path / "locked.db")},
    )
    local = standalone_docs_settings_for_profile(
        StandaloneDocsProfile.LOCAL_FIRST,
        overrides={"db_path": str(tmp_path / "local.db")},
    )
    online = standalone_docs_settings_for_profile(
        StandaloneDocsProfile.ONLINE_CAPABLE,
        overrides={
            "db_path": str(tmp_path / "online.db"),
            "allowed_url_prefixes": ["https://example.com/docs/"],
        },
    )

    assert locked.web_source_profile == "locked_down"  # nosec B101
    assert locked.enable_web_acquisition is False  # nosec B101
    assert locked.allow_arbitrary_public_domains is False  # nosec B101
    assert local.web_source_profile == "local_first"  # nosec B101
    assert local.enable_web_acquisition is True  # nosec B101
    assert local.allow_arbitrary_public_domains is False  # nosec B101
    assert online.web_source_profile == "online_capable"  # nosec B101
    assert online.enable_web_acquisition is True  # nosec B101
    assert online.allowed_url_prefixes == ("https://example.com/docs/",)  # nosec B101


def test_standalone_mount_online_profile_advertises_url_ingest_when_enabled(tmp_path: Path) -> None:
    mount = create_standalone_docs_mount(
        profile=StandaloneDocsProfile.ONLINE_CAPABLE,
        settings={
            "db_path": str(tmp_path / "docs.db"),
            "allowed_url_prefixes": ["https://example.com/docs/"],
        },
    )

    names = {tool["name"] for tool in mount.tool_definitions()}

    assert "docs.ingest_url" in names  # nosec B101
