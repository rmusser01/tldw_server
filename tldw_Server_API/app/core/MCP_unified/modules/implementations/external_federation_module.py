"""External federation module scaffold for MCP Unified.

This module exposes external MCP tools as namespaced virtual tools
(`ext.<server_id>.<tool_name>`) and routes execution through transport
adapters managed by `ExternalServerManager`.

The implementation is intentionally minimal and safe-by-default; transport
adapters are stubs in this phase.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from loguru import logger

from tldw_Server_API.app.core.MCP_unified.external_servers import ExternalServerManager
from tldw_Server_API.app.services.mcp_hub_external_registry_service import (
    get_mcp_hub_external_registry_service,
)
from tldw_Server_API.app.services.mcp_credential_broker_service import (
    get_mcp_credential_broker_service,
)

from ..base import BaseModule, create_tool_definition


class ExternalFederationModule(BaseModule):
    """Federates approved external MCP servers into MCP Unified tool catalogs."""

    def __init__(self, config):
        super().__init__(config)
        self._manager: Optional[ExternalServerManager] = None

    async def on_initialize(self) -> None:
        config_path = str(
            self.config.settings.get("external_servers_config_path")
            or os.getenv("MCP_EXTERNAL_SERVERS_CONFIG", "tldw_Server_API/Config_Files/mcp_external_servers.yaml")
        )

        manager = ExternalServerManager(config_path=config_path)
        custom_loader = self.config.settings.get("external_server_loader")
        custom_broker = self.config.settings.get("external_credential_broker")
        if callable(custom_loader):
            manager = manager.with_server_loader(custom_loader)
        else:
            registry_service = await get_mcp_hub_external_registry_service()
            manager = manager.with_server_loader(registry_service.list_runtime_servers)
        if callable(custom_broker):
            manager = manager.with_credential_broker(custom_broker)
        else:
            broker_service = await get_mcp_credential_broker_service()
            manager = manager.with_credential_broker(broker_service.broker_external_tool_call)
        self._manager = manager
        await self._manager.initialize()
        logger.info(f"External federation module initialized with managed registry; legacy inventory path: {config_path}")

    async def on_shutdown(self) -> None:
        if self._manager is not None:
            await self._manager.shutdown()

    async def check_health(self) -> dict[str, bool]:
        if self._manager is None:
            return {"manager_initialized": False}

        rows = await self._manager.list_servers()
        if not rows:
            return {
                "manager_initialized": True,
                "servers_configured": True,
                "servers_connected": True,
                "servers_discovery_ok": True,
                "any_server_healthy": True,
            }

        connected = []
        discovery_ok = []
        healthy = []
        for row in rows:
            checks = row.get("checks") or {}
            connected.append(bool(checks.get("connected")))
            discovery_ok.append(bool(row.get("discovery_ok", False)))
            healthy.append(str(row.get("status", "")).lower() == "healthy")

        return {
            "manager_initialized": True,
            "servers_configured": True,
            "servers_connected": all(connected),
            "servers_discovery_ok": all(discovery_ok),
            "any_server_healthy": any(healthy),
        }

    async def get_tools(self) -> list[dict[str, Any]]:
        tools = [
            create_tool_definition(
                name="external.servers.list",
                description="List configured external MCP federation servers and status.",
                parameters={"properties": {}, "required": []},
                metadata={"category": "discovery", "catalog_exempt": True},
            ),
            create_tool_definition(
                name="external.tools.refresh",
                description="Refresh external MCP tool discovery cache.",
                parameters={
                    "properties": {
                        "server_id": {
                            "type": "string",
                            "description": "Optional server id; if omitted all servers are refreshed.",
                        }
                    },
                    "required": [],
                },
                metadata={"category": "management"},
            ),
        ]

        if self._manager is None:
            return tools

        for virtual_tool in self._manager.list_virtual_tools():
            tools.append(
                {
                    "name": virtual_tool.virtual_name,
                    "description": virtual_tool.description,
                    "inputSchema": virtual_tool.input_schema,
                    "metadata": {
                        "category": "external",
                        "federated": True,
                        "server_id": virtual_tool.server_id,
                        "upstream_tool": virtual_tool.upstream_tool_name,
                        **(virtual_tool.metadata or {}),
                    },
                }
            )

        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: Optional[Any] = None,
    ) -> Any:
        args = self.sanitize_input(arguments or {})

        if self._manager is None:
            raise RuntimeError("External federation manager is not initialized")

        if tool_name == "external.servers.list":
            servers = await self._manager.list_servers()
            return {
                "servers": servers,
                "count": len(servers),
            }

        if tool_name == "external.tools.refresh":
            server_id = args.get("server_id")
            if server_id is not None and not isinstance(server_id, str):
                raise ValueError("server_id must be a string when provided")
            try:
                return await self._manager.refresh_discovery(server_id=server_id)
            finally:
                self.invalidate_capability_caches()

        if tool_name.startswith("ext."):
            return await self._manager.execute_virtual_tool(
                virtual_tool_name=tool_name,
                arguments=args,
                context=context,
            )

        raise ValueError(f"Unknown tool: {tool_name}")
