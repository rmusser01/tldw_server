from pathlib import Path

import pytest
from mcp_unified.lsp import LspPosition, LspRange, LspToolError
from mcp_unified.lsp.backends import FakeLspBackend
from mcp_unified.lsp.router import LSP_TOOL_NAMES, LspCapabilityRouter
from mcp_unified.lsp.service import LspCodeIntelligenceService


def _router_with_fakes() -> LspCapabilityRouter:
    return LspCapabilityRouter(ruff=FakeLspBackend("ruff"), pylsp=FakeLspBackend("pylsp"))


async def _call_router_tool(tool_name: str, router: LspCapabilityRouter) -> dict[str, object]:
    position = LspPosition(0, 1)
    lsp_range = LspRange(start=position, end=LspPosition(0, 4))
    if tool_name == "lsp.diagnostics":
        return await router.diagnostics(file_path="pkg/app.py")
    if tool_name == "lsp.format_preview":
        return await router.format_preview(file_path="pkg/app.py")
    if tool_name == "lsp.code_actions":
        return await router.code_actions(file_path="pkg/app.py", range=lsp_range)
    if tool_name == "lsp.document_symbols":
        return await router.document_symbols(file_path="pkg/app.py")
    if tool_name == "lsp.workspace_symbols":
        return await router.workspace_symbols(query="app")
    if tool_name == "lsp.definition":
        return await router.definition(file_path="pkg/app.py", position=position)
    if tool_name == "lsp.references":
        return await router.references(file_path="pkg/app.py", position=position)
    if tool_name == "lsp.hover":
        return await router.hover(file_path="pkg/app.py", position=position)
    if tool_name == "lsp.signature_help":
        return await router.signature_help(file_path="pkg/app.py", position=position)
    raise AssertionError(f"unhandled tool in test helper: {tool_name}")


async def test_router_routes_diagnostics_to_ruff_backend():
    result = await _router_with_fakes().diagnostics(file_path="pkg/app.py")

    assert result["backend"] == "ruff"
    assert result["tool"] == "lsp.diagnostics"


async def test_router_routes_definition_to_pylsp_backend():
    result = await _router_with_fakes().definition(file_path="pkg/app.py", position=LspPosition(0, 1))

    assert result["backend"] == "pylsp"
    assert result["tool"] == "lsp.definition"


async def test_missing_backend_returns_backend_missing():
    router = LspCapabilityRouter(ruff=None, pylsp=None)

    with pytest.raises(LspToolError) as exc:
        await router.definition(file_path="pkg/app.py", position=LspPosition(0, 1))

    assert exc.value.reason_code == "backend_missing"


@pytest.mark.parametrize(
    ("tool_name", "expected_backend"),
    [
        ("lsp.diagnostics", "ruff"),
        ("lsp.format_preview", "ruff"),
        ("lsp.code_actions", "ruff"),
        ("lsp.document_symbols", "pylsp"),
        ("lsp.workspace_symbols", "pylsp"),
        ("lsp.definition", "pylsp"),
        ("lsp.references", "pylsp"),
        ("lsp.hover", "pylsp"),
        ("lsp.signature_help", "pylsp"),
    ],
)
async def test_router_covers_every_lsp_tool(tool_name: str, expected_backend: str):
    result = await _call_router_tool(tool_name, _router_with_fakes())

    assert tool_name in LSP_TOOL_NAMES
    assert result["backend"] == expected_backend
    assert result["tool"] == tool_name
    assert result["result"]


async def test_router_reports_capability_unavailable_when_backend_lacks_tool():
    router = LspCapabilityRouter(ruff=FakeLspBackend("ruff", capabilities={"lsp.diagnostics"}), pylsp=None)

    with pytest.raises(LspToolError) as exc:
        await router.format_preview(file_path="pkg/app.py")

    assert exc.value.reason_code == "capability_unavailable"


async def test_router_reports_backend_unhealthy_before_calling_backend():
    backend = FakeLspBackend("ruff", healthy=False)
    router = LspCapabilityRouter(ruff=backend, pylsp=None)

    with pytest.raises(LspToolError) as exc:
        await router.diagnostics(file_path="pkg/app.py")

    assert exc.value.reason_code == "backend_unhealthy"
    assert backend.calls == [("status", None)]


async def test_router_converts_backend_crashes_to_structured_errors():
    router = LspCapabilityRouter(ruff=FakeLspBackend("ruff", crash_methods={"diagnostics"}), pylsp=None)

    with pytest.raises(LspToolError) as exc:
        await router.diagnostics(file_path="pkg/app.py")

    assert exc.value.reason_code == "backend_unhealthy"
    assert "RuntimeError" in (exc.value.detail or "")


async def test_service_status_reports_partial_availability_and_redacts_paths(tmp_path: Path):
    ruff = FakeLspBackend("ruff", version="0.14.0", detail=f"using {tmp_path}/.venv/bin/ruff")
    service = LspCodeIntelligenceService(LspCapabilityRouter(ruff=ruff, pylsp=None))

    status = await service.status(workspace_root=tmp_path)

    assert status["supported_languages"] == ["python"]
    assert status["backends"]["ruff"]["healthy"] is True
    assert status["backends"]["ruff"]["version"] == "0.14.0"
    assert status["backends"]["pylsp"]["reason_code"] == "backend_missing"
    assert "pip install" in status["backends"]["pylsp"]["install_hint"]
    assert str(tmp_path) not in str(status)
