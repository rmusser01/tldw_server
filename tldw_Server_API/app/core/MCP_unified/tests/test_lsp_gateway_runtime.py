"""Tests for the standalone LSP gateway runtime."""

from __future__ import annotations

from pathlib import Path

import pytest
from mcp_unified.gateway.runtime import GatewayPolicyDenied, GatewayRequestContext
from mcp_unified.lsp import (
    FakeLspBackend,
    LspCodeIntelligenceService,
    LspLocation,
    LspLocationsResult,
    LspPosition,
    LspRange,
)

pytestmark = pytest.mark.integration


def _fake_service() -> LspCodeIntelligenceService:
    return LspCodeIntelligenceService.from_backends(
        ruff=FakeLspBackend("ruff"),
        pylsp=FakeLspBackend("pylsp"),
    )


def _sample_range() -> LspRange:
    return LspRange(start=LspPosition(0, 0), end=LspPosition(0, 4))


async def test_lsp_gateway_runtime_lists_lsp_tools(tmp_path: Path) -> None:
    from mcp_unified.lsp.gateway_runtime import LspGatewayRuntime

    runtime = LspGatewayRuntime(workspace_root=tmp_path, service=_fake_service())

    tools = await runtime.list_tools(GatewayRequestContext(request_id="r1"))

    tool_names = {tool["name"] for tool in tools}
    assert tool_names == {  # nosec B101
        "lsp.status",
        "lsp.diagnostics",
        "lsp.document_symbols",
        "lsp.workspace_symbols",
        "lsp.definition",
        "lsp.references",
        "lsp.hover",
        "lsp.signature_help",
        "lsp.format_preview",
        "lsp.code_actions",
    }
    assert all(tool["inputSchema"]["additionalProperties"] is False for tool in tools)  # nosec B101


async def test_lsp_gateway_runtime_calls_status(tmp_path: Path) -> None:
    from mcp_unified.lsp.gateway_runtime import LspGatewayRuntime

    runtime = LspGatewayRuntime(workspace_root=tmp_path, service=_fake_service())

    result = await runtime.call_tool(
        "lsp.status",
        {},
        GatewayRequestContext(request_id="r1"),
    )

    assert result["isError"] is False  # nosec B101
    assert result["structuredContent"]["status"] == "healthy"  # nosec B101
    assert result["structuredContent"]["capabilities"]["available"]  # nosec B101
    assert result["content"][0]["type"] == "text"  # nosec B101


async def test_lsp_gateway_runtime_filters_denied_result_paths(tmp_path: Path) -> None:
    from mcp_unified.lsp.gateway_runtime import LspGatewayRuntime

    class _Service:
        async def definition(self, **_: object) -> LspLocationsResult:
            return LspLocationsResult(
                locations=(
                    LspLocation(path="src/app.py", range=_sample_range()),
                    LspLocation(path="private/secret.py", range=_sample_range()),
                )
            )

    runtime = LspGatewayRuntime(
        workspace_root=tmp_path,
        service=_Service(),
        path_allow_predicate=lambda path: path.startswith("src/"),
    )

    result = await runtime.call_tool(
        "lsp.definition",
        {"path": "src/app.py", "position": {"line": 0, "character": 1}},
        GatewayRequestContext(request_id="r1"),
    )

    assert result["structuredContent"]["locations"] == [  # nosec B101
        {"path": "src/app.py", "range": _sample_range().to_dict()}
    ]
    assert result["structuredContent"]["filtered_count"] == 1  # nosec B101


async def test_lsp_gateway_runtime_rejects_missing_workspace() -> None:
    from mcp_unified.lsp.gateway_runtime import LspGatewayRuntime

    runtime = LspGatewayRuntime(service=_fake_service())

    with pytest.raises(GatewayPolicyDenied) as exc:
        await runtime.call_tool(
            "lsp.status",
            {},
            GatewayRequestContext(request_id="r1"),
        )

    assert exc.value.reason_code == "workspace_not_supported"  # nosec B101
