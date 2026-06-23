"""External MCP server install/update contracts for standalone runtimes."""

from __future__ import annotations

from typing import Any, Protocol

from mcp_unified.storage import ExternalServerDefinition


class ExternalServerInstaller(Protocol):
    """Adapter contract for optional external server install/update workflows."""

    async def install_server(
        self,
        server: ExternalServerDefinition,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        """Install one configured external server, when supported."""
        ...

    async def update_server(
        self,
        server: ExternalServerDefinition,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        """Update one configured external server, when supported."""
        ...

    async def get_status(
        self,
        server: ExternalServerDefinition,
    ) -> dict[str, Any]:
        """Return installer availability/status for one server."""
        ...


class NullExternalServerInstaller:
    """Disabled-by-default installer that performs no external side effects."""

    async def install_server(
        self,
        server: ExternalServerDefinition,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        """Return a deterministic not-configured install response."""
        del context
        return {
            "ok": False,
            "available": False,
            "reason_code": "external_server_install_not_configured",
            "server_id": server.id,
        }

    async def update_server(
        self,
        server: ExternalServerDefinition,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        """Return a deterministic not-configured update response."""
        del context
        return {
            "ok": False,
            "available": False,
            "reason_code": "external_server_update_not_configured",
            "server_id": server.id,
        }

    async def get_status(
        self,
        server: ExternalServerDefinition,
    ) -> dict[str, Any]:
        """Return unavailable installer status for one server."""
        return {
            "available": False,
            "server_id": server.id,
        }
