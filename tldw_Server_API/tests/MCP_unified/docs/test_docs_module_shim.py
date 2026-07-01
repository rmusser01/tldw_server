from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.docs_module import DocsModule
from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext


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
