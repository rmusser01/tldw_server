"""Storage protocols for standalone MCP profile and registry stores."""

from __future__ import annotations

from typing import Any, Protocol


class ProfileStore(Protocol):
    """Store for named MCP tool and permission profiles."""

    async def get_profile(self, profile_id: str) -> dict[str, Any] | None: ...


class ExternalRegistryStore(Protocol):
    """Store for external MCP server registry entries."""

    async def list_servers(self) -> list[dict[str, Any]]: ...


class AuditStore(Protocol):
    """Append-only audit sink for MCP policy and tool events."""

    async def append_event(self, event: dict[str, Any]) -> None: ...
