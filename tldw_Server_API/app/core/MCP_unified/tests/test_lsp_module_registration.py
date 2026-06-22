from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


async def _capture_default_registrations(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict[str, Any]]:
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    server = MCPServer()
    registrations: list[dict[str, Any]] = []

    async def _capture_registration(module_id: str, module_type: type[Any], config: Any) -> None:
        registrations.append({"module_id": module_id, "module_type": module_type, "config": config})

    monkeypatch.setattr(server.module_registry, "register_module", _capture_registration)
    monkeypatch.setenv("MCP_MODULES_CONFIG", str(tmp_path / "missing-modules.yaml"))
    monkeypatch.setenv("MCP_MODULES", "")
    monkeypatch.setenv("MCP_ENABLE_MEDIA_MODULE", "0")
    monkeypatch.setenv("MCP_ENABLE_FILESYSTEM_MODULE", "0")
    monkeypatch.setenv("MCP_ENABLE_GIT_MODULE", "0")
    monkeypatch.setenv("MCP_ENABLE_BROWSER_CDP_MODULE", "0")
    monkeypatch.delenv("MCP_BROWSER_CDP_URL", raising=False)

    await server._register_default_modules()
    return registrations


@pytest.mark.asyncio
async def test_server_registers_lsp_module_when_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MCP_ENABLE_LSP_MODULE", "1")
    monkeypatch.setenv("MCP_LSP_REQUEST_TIMEOUT_SECONDS", "2.5")

    registrations = await _capture_default_registrations(monkeypatch, tmp_path)

    lsp_registration = next(item for item in registrations if item["module_id"] == "lsp")
    assert lsp_registration["module_type"].__name__ == "LSPModule"  # nosec B101
    assert lsp_registration["config"].name == "LSP Code Intelligence"  # nosec B101
    assert lsp_registration["config"].department == "code"  # nosec B101
    assert lsp_registration["config"].settings["request_timeout_seconds"] == "2.5"  # nosec B101


@pytest.mark.asyncio
async def test_server_does_not_register_lsp_module_when_flag_is_unset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("MCP_ENABLE_LSP_MODULE", raising=False)

    registrations = await _capture_default_registrations(monkeypatch, tmp_path)

    assert "lsp" not in {item["module_id"] for item in registrations}  # nosec B101
