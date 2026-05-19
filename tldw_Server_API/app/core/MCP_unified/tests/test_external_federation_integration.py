from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.MCP_unified.external_servers.config_schema import (
    parse_external_server_registry,
)
from tldw_Server_API.app.core.MCP_unified.external_servers.manager import ExternalServerManager
from tldw_Server_API.app.core.MCP_unified.external_servers import manager as manager_mod
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.external_federation_module import (
    ExternalFederationModule,
)

websockets = pytest.importorskip("websockets")


@dataclass
class _StubExternalMCPServer:
    port: int
    calls: list[tuple[str, dict[str, Any]]]
    _server: Any

    @classmethod
    async def start(cls) -> "_StubExternalMCPServer":
        calls: list[tuple[str, dict[str, Any]]] = []

        async def _handler(websocket, *_args):  # websockets version compatibility
            async for raw in websocket:
                payload = json.loads(raw)
                request_id = payload.get("id")
                method = payload.get("method")
                params = payload.get("params") or {}

                if method == "initialize":
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "serverInfo": {"name": "stub-external", "version": "0.1.0"},
                        },
                    }
                    await websocket.send(json.dumps(response))
                    continue

                if method == "tools/list":
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "tools": [
                                {
                                    "name": "docs.search",
                                    "description": "Search docs",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {"q": {"type": "string"}},
                                    },
                                    "metadata": {"category": "read"},
                                },
                                {
                                    "name": "docs.fail",
                                    "description": "Return upstream error",
                                    "inputSchema": {"type": "object"},
                                    "metadata": {"category": "read"},
                                },
                                {
                                    "name": "docs.slow",
                                    "description": "Slow tool",
                                    "inputSchema": {"type": "object"},
                                    "metadata": {"category": "read"},
                                },
                            ]
                        },
                    }
                    await websocket.send(json.dumps(response))
                    continue

                if method == "tools/call":
                    tool_name = str(params.get("name") or "")
                    arguments = params.get("arguments") or {}
                    if isinstance(arguments, dict):
                        calls.append((tool_name, dict(arguments)))
                    else:
                        calls.append((tool_name, {}))

                    if tool_name == "docs.search":
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "result": {
                                "content": [{"type": "text", "text": "stub search result"}],
                                "isError": False,
                            },
                        }
                        await websocket.send(json.dumps(response))
                        continue

                    if tool_name == "docs.fail":
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {"code": -32042, "message": "upstream failure"},
                        }
                        await websocket.send(json.dumps(response))
                        continue

                    if tool_name == "docs.slow":
                        await asyncio.sleep(0.35)
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "result": {
                                "content": [{"type": "text", "text": "slow result"}],
                                "isError": False,
                            },
                        }
                        await websocket.send(json.dumps(response))
                        continue

                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32601, "message": f"method not found for {tool_name}"},
                    }
                    await websocket.send(json.dumps(response))
                    continue

                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"unknown method: {method}"},
                }
                await websocket.send(json.dumps(response))

        try:
            server = await websockets.serve(_handler, "127.0.0.1", 0)
        except PermissionError as exc:
            pytest.skip(f"local websocket bind not permitted in this environment: {exc}")
        port = server.sockets[0].getsockname()[1]
        return cls(port=port, calls=calls, _server=server)

    async def stop(self) -> None:
        self._server.close()
        await self._server.wait_closed()


def _federation_registry_payload(port: int, *, request_seconds: float) -> dict[str, Any]:
    return {
        "servers": [
            {
                "id": "docs",
                "name": "Docs Stub",
                "transport": "websocket",
                "websocket": {"url": f"ws://127.0.0.1:{port}"},
                "policy": {
                    "allow_tool_patterns": ["docs.*"],
                    "deny_tool_patterns": [],
                    "allow_writes": False,
                    "require_write_confirmation": True,
                },
                "timeouts": {
                    "connect_seconds": 1.0,
                    "request_seconds": request_seconds,
                },
            }
        ]
    }


@pytest.mark.asyncio
async def test_external_manager_integration_discovery_execute_error_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stub = await _StubExternalMCPServer.start()
    try:
        cfg = parse_external_server_registry(
            _federation_registry_payload(stub.port, request_seconds=0.1)
        )
        external_config = tmp_path / "external_servers.yaml"
        cfg_payload = cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()
        external_config.write_text(json.dumps(cfg_payload), encoding="utf-8")
        monkeypatch.setattr(manager_mod, "load_external_server_registry", lambda _path=None: cfg)

        manager = ExternalServerManager(config_path=str(external_config))
        try:
            await manager.initialize()
            virtual_names = [tool.virtual_name for tool in manager.list_virtual_tools()]
            assert "ext.docs.docs.search" in virtual_names  # nosec B101
            assert "ext.docs.docs.fail" in virtual_names  # nosec B101
            assert "ext.docs.docs.slow" in virtual_names  # nosec B101

            ok = await manager.execute_virtual_tool("ext.docs.docs.search", {"q": "hello"})
            assert ok["is_error"] is False  # nosec B101
            assert ok["server_id"] == "docs"  # nosec B101
            assert ok["upstream_tool"] == "docs.search"  # nosec B101
            assert ok["content"] == [{"type": "text", "text": "stub search result"}]  # nosec B101

            err = await manager.execute_virtual_tool("ext.docs.docs.fail", {})
            assert err["is_error"] is True  # nosec B101
            assert err["metadata"]["upstream_error"]["code"] == -32042  # nosec B101

            with pytest.raises(TimeoutError, match="timed out"):
                await manager.execute_virtual_tool("ext.docs.docs.slow", {})
        finally:
            await manager.shutdown()
    finally:
        await stub.stop()


@pytest.mark.asyncio
async def test_external_federation_module_integration_exposes_and_executes_virtual_tools(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stub = await _StubExternalMCPServer.start()
    try:
        cfg = parse_external_server_registry(
            _federation_registry_payload(stub.port, request_seconds=0.4)
        )
        external_config = tmp_path / "external_servers.yaml"
        cfg_payload = cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()
        external_config.write_text(json.dumps(cfg_payload), encoding="utf-8")
        monkeypatch.setattr(manager_mod, "load_external_server_registry", lambda _path=None: cfg)

        module = ExternalFederationModule(
            ModuleConfig(
                name="external_federation",
                settings={
                    "external_servers_config_path": str(external_config),
                    "external_server_loader": lambda: list(cfg.servers),
                },
            )
        )
        try:
            await module.initialize()
            tools = await module.get_tools()
            names = {tool["name"] for tool in tools if isinstance(tool, dict) and "name" in tool}
            assert "external.servers.list" in names  # nosec B101
            assert "external.tools.refresh" in names  # nosec B101
            assert "ext.docs.docs.search" in names  # nosec B101

            result = await module.execute_tool("ext.docs.docs.search", {"q": "from-module"})
            assert result["is_error"] is False  # nosec B101
            assert result["content"] == [{"type": "text", "text": "stub search result"}]  # nosec B101

            health = await module.check_health()
            assert health["manager_initialized"] is True  # nosec B101
            assert health["servers_configured"] is True  # nosec B101
        finally:
            await module.shutdown()
    finally:
        await stub.stop()
