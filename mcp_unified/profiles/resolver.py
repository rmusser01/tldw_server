"""Profile resolution contracts for MCP Unified hosts and gateways."""

from __future__ import annotations

from typing import Protocol

from .models import MCPProfile


class ProfileResolver(Protocol):
    """Resolve an effective MCP profile for a request principal."""

    async def resolve_profile(
        self,
        profile_id: str | None,
        *,
        user_id: str | None = None,
    ) -> MCPProfile | None: ...
