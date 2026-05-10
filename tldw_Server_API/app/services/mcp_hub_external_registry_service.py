from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.repos.mcp_hub_repo import McpHubRepo
from tldw_Server_API.app.core.MCP_unified.external_servers.config_schema import (
    ExternalMCPServerConfig,
    ExternalServerRegistryPartialLoadError,
    parse_external_server_registry,
)


_EXTERNAL_SERVER_CONFIG_INVALID = "external_server_config_invalid"


@dataclass
class McpHubExternalRegistryService:
    """Build executable external server configs from managed MCP Hub state."""

    repo: McpHubRepo

    async def list_runtime_servers(self) -> list[ExternalMCPServerConfig]:
        rows = await self.repo.list_external_servers()
        runtime_servers: list[ExternalMCPServerConfig] = []
        errors: dict[str, str] = {}
        for row in rows:
            try:
                payload = await self._build_runtime_payload(row)
                if payload is None:
                    continue
                registry = parse_external_server_registry({"servers": [payload]})
                runtime_servers.extend(registry.servers)
            except Exception as exc:
                server_id = str(row.get("id") or "").strip()
                if server_id:
                    errors[server_id] = _EXTERNAL_SERVER_CONFIG_INVALID
                logger.warning(
                    "Skipping managed external server '{}' during runtime registry load: {}",
                    row.get("id"),
                    exc,
                )
        if errors:
            raise ExternalServerRegistryPartialLoadError(
                "One or more managed external servers could not be loaded",
                servers=runtime_servers,
                errors=errors,
            )
        return runtime_servers

    async def _build_runtime_payload(self, row: dict[str, Any]) -> dict[str, Any] | None:
        if str(row.get("server_source") or "managed") != "managed":
            return None
        if row.get("superseded_by_server_id"):
            return None
        if not bool(row.get("enabled")):
            return None

        config = dict(row.get("config") or {})
        auth = dict(config.get("auth") or {})
        mode = str(auth.get("mode") or "none").strip().lower()

        if mode not in {"", "none"}:
            # Managed runtime auth is brokered per execution; registry payloads stay auth-neutral.
            config["auth"] = {"mode": "none"}

        payload = {
            "id": str(row.get("id") or ""),
            "name": str(row.get("name") or ""),
            "enabled": bool(row.get("enabled")),
            "transport": str(row.get("transport") or ""),
            **config,
        }
        return payload


async def get_mcp_hub_external_registry_service() -> McpHubExternalRegistryService:
    """Resolve the managed external runtime registry service from the active AuthNZ DB."""
    pool = await get_db_pool()
    repo = McpHubRepo(pool)
    await repo.ensure_tables()
    return McpHubExternalRegistryService(repo=repo)
