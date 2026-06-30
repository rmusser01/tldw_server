from __future__ import annotations

from pathlib import Path

import pytest

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
