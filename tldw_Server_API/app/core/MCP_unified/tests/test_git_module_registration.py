from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


async def _capture_default_registrations(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict[str, Any]]:
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    server = MCPServer()
    registrations: list[dict[str, Any]] = []

    async def _capture_registration(module_id: str, module_type: type[Any], config: Any) -> None:
        registrations.append(
            {
                "module_id": module_id,
                "module_type": module_type,
                "config": config,
            }
        )

    monkeypatch.setattr(server.module_registry, "register_module", _capture_registration)
    monkeypatch.setenv("MCP_MODULES_CONFIG", str(tmp_path / "missing-modules.yaml"))
    monkeypatch.delenv("MCP_MODULES", raising=False)
    monkeypatch.setenv("MCP_ENABLE_MEDIA_MODULE", "0")
    monkeypatch.setenv("MCP_ENABLE_FILESYSTEM_MODULE", "0")
    monkeypatch.setenv("MCP_ENABLE_BROWSER_CDP_MODULE", "0")
    monkeypatch.delenv("MCP_BROWSER_CDP_URL", raising=False)

    await server._register_default_modules()
    return registrations


@pytest.mark.asyncio
async def test_server_registers_git_module_when_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MCP_ENABLE_GIT_MODULE", "1")

    registrations = await _capture_default_registrations(monkeypatch, tmp_path)

    git_registration = next(item for item in registrations if item["module_id"] == "git")
    assert git_registration["module_type"].__name__ == "GitModule"  # nosec B101
    assert git_registration["config"].name == "Git"  # nosec B101
    assert git_registration["config"].version == "1.0.0"  # nosec B101
    assert git_registration["config"].department == "management"  # nosec B101
    assert git_registration["config"].settings == {}  # nosec B101


@pytest.mark.asyncio
async def test_server_does_not_register_git_module_when_flag_is_unset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("MCP_ENABLE_GIT_MODULE", raising=False)

    registrations = await _capture_default_registrations(monkeypatch, tmp_path)

    assert "git" not in {item["module_id"] for item in registrations}  # nosec B101


@pytest.mark.asyncio
async def test_server_does_not_register_git_module_when_flag_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MCP_ENABLE_GIT_MODULE", "0")

    registrations = await _capture_default_registrations(monkeypatch, tmp_path)

    assert "git" not in {item["module_id"] for item in registrations}  # nosec B101


@pytest.mark.asyncio
async def test_filesystem_media_and_browser_env_vars_do_not_enable_git_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("MCP_ENABLE_GIT_MODULE", raising=False)
    monkeypatch.setenv("MCP_ENABLE_MEDIA_MODULE", "1")
    monkeypatch.setenv("MCP_ENABLE_FILESYSTEM_MODULE", "1")
    monkeypatch.setenv("MCP_ENABLE_BROWSER_CDP_MODULE", "1")
    monkeypatch.setenv("MCP_BROWSER_CDP_URL", "http://127.0.0.1:9222")

    registrations = await _capture_default_registrations(monkeypatch, tmp_path)

    assert "git" not in {item["module_id"] for item in registrations}  # nosec B101
