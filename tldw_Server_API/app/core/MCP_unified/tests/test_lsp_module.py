from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from mcp_unified.interfaces.path_scope import PathScopeCandidate
from mcp_unified.lsp import (
    LspBackendStatus,
    LspDiagnosticsResult,
    LspLocation,
    LspLocationsResult,
    LspPosition,
    LspPreview,
    LspRange,
    LspSymbolsResult,
    LspToolError,
)

from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.lsp_module import LSPModule
from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext


class _FakeWorkspaceRootResolver:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root

    async def resolve_for_context(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "workspace_root": str(self.workspace_root),
            "workspace_id": "workspace-1",
            "source": "test",
            "reason": None,
        }


class _FakeLspService:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def status(self, **kwargs: Any) -> dict[str, object]:
        self.calls.append(("status", kwargs))
        return {
            "backends": [LspBackendStatus(name="ruff", healthy=True, capabilities=["lsp.diagnostics"]).to_dict()],
            "degraded": False,
        }

    async def diagnostics(self, **kwargs: Any) -> LspDiagnosticsResult:
        self.calls.append(("diagnostics", kwargs))
        return self.result  # type: ignore[return-value]

    async def document_symbols(self, **kwargs: Any) -> LspSymbolsResult:
        self.calls.append(("document_symbols", kwargs))
        return self.result  # type: ignore[return-value]

    async def workspace_symbols(self, **kwargs: Any) -> LspSymbolsResult:
        self.calls.append(("workspace_symbols", kwargs))
        return self.result  # type: ignore[return-value]

    async def definition(self, **kwargs: Any) -> LspLocationsResult:
        self.calls.append(("definition", kwargs))
        return self.result  # type: ignore[return-value]

    async def references(self, **kwargs: Any) -> LspLocationsResult:
        self.calls.append(("references", kwargs))
        return self.result  # type: ignore[return-value]

    async def hover(self, **kwargs: Any) -> object:
        self.calls.append(("hover", kwargs))
        return self.result

    async def signature_help(self, **kwargs: Any) -> object:
        self.calls.append(("signature_help", kwargs))
        return self.result

    async def format_preview(self, **kwargs: Any) -> LspPreview:
        self.calls.append(("format_preview", kwargs))
        return self.result  # type: ignore[return-value]

    async def code_actions(self, **kwargs: Any) -> object:
        self.calls.append(("code_actions", kwargs))
        return self.result


def _sample_range() -> LspRange:
    return LspRange(start=LspPosition(0, 0), end=LspPosition(0, 4))


def _module(tmp_path: Path, result: object) -> LSPModule:
    return LSPModule(
        ModuleConfig(name="LSP"),
        service=_FakeLspService(result),
        workspace_root_resolver=_FakeWorkspaceRootResolver(tmp_path),
    )


def _context() -> RequestContext:
    return RequestContext(
        request_id="req-lsp",
        user_id="7",
        session_id="sess-1",
        metadata={"workspace_id": "workspace-1"},
    )


@pytest.mark.asyncio
async def test_lsp_module_exposes_tool_definitions(tmp_path: Path) -> None:
    module = _module(tmp_path, LspLocationsResult(locations=()))

    tools = await module.get_tools()
    by_name = {tool["name"]: tool for tool in tools}

    assert set(by_name) == {  # nosec B101
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
    for tool in by_name.values():
        metadata = tool["metadata"]
        assert metadata["readOnlyHint"] is True  # nosec B101
        assert metadata["uses_filesystem"] is True  # nosec B101
        assert metadata["path_boundable"] is True  # nosec B101
        assert tool["inputSchema"]["additionalProperties"] is False  # nosec B101

    assert by_name["lsp.status"]["metadata"]["category"] == "retrieval"  # nosec B101
    assert by_name["lsp.workspace_symbols"]["metadata"]["path_scope_candidate_source"] == "module"  # nosec B101
    assert by_name["lsp.definition"]["metadata"]["path_scope_action"] == "read"  # nosec B101
    assert by_name["lsp.format_preview"]["metadata"]["category"] == "analysis"  # nosec B101


@pytest.mark.asyncio
async def test_lsp_definition_extracts_read_candidate_for_file(tmp_path: Path) -> None:
    module = _module(tmp_path, LspLocationsResult(locations=()))

    candidates = await module.extract_path_scope_candidates(
        "lsp.definition",
        {"path": "src/app.py", "position": {"line": 1, "character": 2}},
    )

    assert candidates == [  # nosec B101
        PathScopeCandidate(
            path="src/app.py",
            action="read",
            source="lsp.definition",
            requires_existing_file=True,
        )
    ]


@pytest.mark.asyncio
async def test_lsp_workspace_symbols_requires_workspace_root_read_candidate(tmp_path: Path) -> None:
    module = _module(tmp_path, LspSymbolsResult(symbols=()))

    candidates = await module.extract_path_scope_candidates("lsp.workspace_symbols", {"query": "Widget"})

    assert candidates == [PathScopeCandidate(path=".", action="read", source="lsp.workspace_symbols")]  # nosec B101


@pytest.mark.asyncio
async def test_lsp_module_filters_denied_definition_results(tmp_path: Path) -> None:
    result = LspLocationsResult(
        locations=(
            LspLocation(path="src/app.py", range=_sample_range()),
            LspLocation(path="private/secret.py", range=_sample_range()),
        )
    )
    module = _module(tmp_path, result)

    payload = await module.execute_tool(
        "lsp.definition",
        {"path": "src/app.py", "position": {"line": 0, "character": 1}},
        context=_context(),
    )

    assert [item["path"] for item in payload["locations"]] == ["src/app.py"]  # nosec B101
    assert payload["filtered_count"] == 1  # nosec B101


@pytest.mark.asyncio
async def test_lsp_module_rejects_preview_with_denied_affected_path(tmp_path: Path) -> None:
    result = LspPreview(path="private/secret.py", preview="--- private/secret.py\n+++ private/secret.py\n")
    module = _module(tmp_path, result)

    with pytest.raises(PermissionError, match="path_denied"):
        await module.execute_tool(
            "lsp.format_preview",
            {"path": "src/app.py"},
            context=_context(),
        )


@pytest.mark.asyncio
async def test_lsp_module_converts_lsp_errors_to_permission_errors(tmp_path: Path) -> None:
    module = _module(tmp_path, LspToolError("path_denied", "blocked"))

    with pytest.raises(PermissionError, match="path_denied"):
        await module.execute_tool(
            "lsp.definition",
            {"path": "src/app.py", "position": {"line": 0, "character": 1}},
            context=_context(),
        )


@pytest.mark.parametrize(
    ("tool_name", "arguments", "message"),
    [
        ("lsp.definition", {"path": "src/app.py", "position": {"line": -1, "character": 0}}, "line"),
        ("lsp.definition", {"path": "src/app.js", "position": {"line": 0, "character": 0}}, "python"),
        ("lsp.format_preview", {"path": "src/app.py", "include_text_edits": "yes"}, "boolean"),
        ("lsp.references", {"path": "src/app.py", "position": {"line": 0, "character": 0}, "limit": 0}, "positive"),
    ],
)
def test_lsp_module_validates_arguments(
    tmp_path: Path,
    tool_name: str,
    arguments: dict[str, Any],
    message: str,
) -> None:
    module = _module(tmp_path, LspLocationsResult(locations=()))

    with pytest.raises(ValueError, match=message):
        module.validate_tool_arguments(tool_name, arguments)


def test_lsp_module_accepts_server_resolved_numeric_settings(tmp_path: Path) -> None:
    module = LSPModule(
        ModuleConfig(
            name="LSP",
            settings={
                "request_timeout_seconds": "2.5",
                "startup_timeout_seconds": "7",
                "idle_ttl_seconds": "120",
            },
        ),
        service=_FakeLspService(LspLocationsResult(locations=())),
        workspace_root_resolver=_FakeWorkspaceRootResolver(tmp_path),
    )

    assert module._runtime_config.request_timeout_seconds == 2.5  # nosec B101, SLF001
    assert module._runtime_config.startup_timeout_seconds == 7.0  # nosec B101, SLF001
    assert module._runtime_config.idle_ttl_seconds == 120  # nosec B101, SLF001
