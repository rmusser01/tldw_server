from __future__ import annotations

from pathlib import Path
import threading
from typing import Any

import pytest
import yaml

from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.docs_module import DocsModule
from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_docs_module_advertises_provider_tools(tmp_path: Path) -> None:
    module = DocsModule(
        ModuleConfig(
            name="docs",
            settings={
                "db_path": str(tmp_path / "docs.db"),
                "trusted_roots": [str(tmp_path)],
                "enable_web_acquisition": False,
            },
        )
    )
    await module.on_initialize()

    names = {tool["name"] for tool in await module.get_tools()}

    assert "docs.search" in names  # nosec B101
    assert "docs.import_path" in names  # nosec B101
    assert "docs.ingest_url" not in names  # nosec B101


@pytest.mark.asyncio
async def test_docs_module_exposes_sync_source_without_media_or_rag(tmp_path: Path) -> None:
    module = DocsModule(
        ModuleConfig(
            name="docs",
            settings={"db_path": str(tmp_path / "docs.db"), "trusted_roots": [str(tmp_path)]},
        )
    )
    await module.on_initialize()

    tools = {tool["name"]: tool for tool in await module.get_tools()}

    assert "docs.sync_source" in tools  # nosec B101
    assert tools["docs.sync_source"]["metadata"]["category"] == "ingestion"  # nosec B101


@pytest.mark.asyncio
async def test_docs_module_executes_with_context_scope(tmp_path: Path) -> None:
    doc_path = tmp_path / "guide.md"
    doc_path.write_text("# Guide\n\nSQLite local docs.\n", encoding="utf-8")
    module = DocsModule(
        ModuleConfig(
            name="docs",
            settings={
                "db_path": str(tmp_path / "docs.db"),
                "trusted_roots": [str(tmp_path)],
                "enable_web_acquisition": False,
            },
        )
    )
    await module.on_initialize()
    ctx = RequestContext(
        request_id="docs-test",
        user_id="user-1",
        client_id="unit",
        metadata={"profile_scope": "profile-1"},
    )

    await module.execute_tool("docs.import_path", {"path": str(doc_path), "keywords": ["sqlite"]}, context=ctx)
    result = await module.execute_tool("docs.search", {"query": "SQLite"}, context=ctx)

    assert result["results"]  # nosec B101


@pytest.mark.asyncio
async def test_docs_module_rejects_empty_ingest_url(tmp_path: Path) -> None:
    module = DocsModule(
        ModuleConfig(
            name="docs",
            settings={
                "db_path": str(tmp_path / "docs.db"),
                "enable_web_acquisition": True,
                "web_source_profile": "locked_down",
                "allowed_url_prefixes": ["https://example.com/docs/"],
            },
        )
    )
    await module.on_initialize()

    with pytest.raises(ValueError, match="url is required"):
        await module.execute_tool("docs.ingest_url", {"url": "   "}, context=None)


@pytest.mark.asyncio
async def test_docs_module_allows_docs_queries_that_look_like_cli_flags(tmp_path: Path) -> None:
    class FakeProvider:
        def __init__(self) -> None:
            self.thread_ident: int | None = None
            self.arguments: dict[str, Any] | None = None

        def execute(self, tool_name: str, arguments: dict[str, Any], *, scope: Any) -> dict[str, Any]:
            self.thread_ident = threading.current_thread().ident
            self.arguments = arguments
            return {"tool_name": tool_name, "owner_scope": scope.owner_scope}

    module = DocsModule(ModuleConfig(name="docs", settings={"db_path": str(tmp_path / "docs.db")}))
    fake_provider = FakeProvider()
    module._provider = fake_provider
    main_thread_ident = threading.current_thread().ident

    result = await module.execute_tool("docs.search", {"query": "--flag"}, context=None)

    assert result == {"tool_name": "docs.search", "owner_scope": None}  # nosec B101
    assert fake_provider.arguments == {"query": "--flag"}  # nosec B101
    assert fake_provider.thread_ident != main_thread_ident  # nosec B101


@pytest.mark.asyncio
async def test_docs_module_reports_missing_required_fields(tmp_path: Path) -> None:
    module = DocsModule(ModuleConfig(name="docs", settings={"db_path": str(tmp_path / "docs.db")}))

    with pytest.raises(ValueError, match="id is required"):
        await module.execute_tool("docs.get", {}, context=None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        (
            "docs.collections.set_membership",
            {"collection": "Reference", "document_id": "   ", "action": "add"},
        ),
        ("docs.keywords.apply", {"document_id": "   ", "keywords": ["sqlite"]}),
    ],
)
async def test_docs_module_rejects_whitespace_document_ids(
    tmp_path: Path,
    tool_name: str,
    arguments: dict[str, Any],
) -> None:
    module = DocsModule(ModuleConfig(name="docs", settings={"db_path": str(tmp_path / "docs.db")}))

    with pytest.raises(ValueError, match="document_id is required"):
        await module.execute_tool(tool_name, arguments, context=None)


def test_repo_docs_mcp_config_keeps_web_acquisition_disabled() -> None:
    config_path = Path("tldw_Server_API/Config_Files/mcp_modules.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    docs_modules = [module for module in config["modules"] if module["id"] == "docs"]

    assert len(docs_modules) == 1  # nosec B101
    settings = docs_modules[0]["settings"]
    assert settings["enable_web_acquisition"] is False  # nosec B101
    assert settings["web_source_profile"] == "locked_down"  # nosec B101
    assert settings["allow_arbitrary_public_domains"] is False  # nosec B101
    assert settings["preapproved_domains"] == []  # nosec B101
    assert settings["allowed_url_prefixes"] == []  # nosec B101
    assert settings["denied_domains"] == []  # nosec B101
    assert settings["max_url_redirects"] == 3  # nosec B101
    assert settings["max_url_body_bytes"] == 2_000_000  # nosec B101
    assert settings["url_request_timeout_seconds"] == 10.0  # nosec B101
    assert settings["allowed_content_types"] == [  # nosec B101
        "text/html",
        "application/xhtml+xml",
        "text/plain",
        "text/markdown",
    ]
    assert settings["respect_robots"] is False  # nosec B101
