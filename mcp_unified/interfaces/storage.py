"""Storage protocols for standalone MCP profile and registry stores."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from mcp_unified.profiles.models import MCPProfile


class ProfileStore(Protocol):
    """Store for named MCP tool and permission profiles.

    Implementations return caller-owned ``MCPProfile`` instances so callers may
    inspect or mutate returned models without changing persisted state.
    """

    async def get_profile(self, profile_id: str) -> MCPProfile | None: ...

    async def list_profiles(self) -> list[MCPProfile]: ...

    async def upsert_profile(
        self,
        profile: MCPProfile,
    ) -> MCPProfile: ...

    async def delete_profile(self, profile_id: str) -> bool: ...


class ExternalRegistryStore(Protocol):
    """Store for external MCP server registry entries."""

    async def list_servers(self) -> list[dict[str, Any]]: ...


class AuditStore(Protocol):
    """Append-only audit sink for MCP policy and tool events."""

    async def append_event(self, event: dict[str, Any]) -> None: ...
