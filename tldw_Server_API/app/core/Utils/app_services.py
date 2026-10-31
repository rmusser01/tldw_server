"""App-scoped service container.

Holds the process's long-lived services in explicit slots. ``main.lifespan``
publishes it as ``app.state.services``; module getters such as
``get_db_pool()`` are the fallback for code with no request or constructor
argument and read the same slots.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
    from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService
    from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager
    from tldw_Server_API.app.core.DB_Management.chacha.runtime import ChaChaRuntimeManager
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer


@dataclass
class AppServices:
    """One slot per service; each slot is ``None`` until its getter fills it."""

    db_pool: DatabasePool | None = None
    session_manager: SessionManager | None = None
    jwt_service: JWTService | None = None
    mcp_server: MCPServer | None = None
    chacha_runtime: ChaChaRuntimeManager | None = None
    # Serializes first-time session-manager init; a fresh container gets a fresh lock.
    session_manager_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


_SERVICES = AppServices()


def app_services() -> AppServices:
    """Return the process-wide service container."""
    return _SERVICES
