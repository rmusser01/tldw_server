"""In-process external MCP server runtime management for the standalone gateway."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from mcp_unified.federation.models import (
    ExternalToolDefinition,
    VirtualExternalTool,
)
from mcp_unified.federation.transports import ExternalFederationTransport
from mcp_unified.interfaces.storage import AuditStore, ExternalRegistryStore
from mcp_unified.storage import AuditEvent, ExternalServerDefinition


class GatewayExternalRuntimeError(RuntimeError):
    """Raised when an external runtime operation cannot be completed."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str,
        server_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.server_id = server_id

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable error payload for gateway callers."""
        payload: dict[str, Any] = {
            "ok": False,
            "error": str(self),
            "reason_code": self.reason_code,
        }
        if self.server_id is not None:
            payload["server_id"] = self.server_id
        return payload


class GatewayExternalRuntimeManager:
    """Manage active external server transports for one in-process gateway."""

    def __init__(
        self,
        *,
        external_registry_store: ExternalRegistryStore,
        transport_factory: Callable[[ExternalServerDefinition], ExternalFederationTransport],
        audit_store: AuditStore | None = None,
    ) -> None:
        self._external_registry_store = external_registry_store
        self._transport_factory = transport_factory
        self._audit_store = audit_store
        self._servers: dict[str, ExternalServerDefinition] = {}
        self._transports: dict[str, ExternalFederationTransport] = {}
        self._virtual_tools: dict[str, VirtualExternalTool] = {}
        self._last_errors: dict[str, str | None] = {}
        self._lock = asyncio.Lock()

    async def start_server(self, server_id: str) -> dict[str, Any]:
        """Start one configured external server and discover its virtual tools."""
        async with self._lock:
            server = await self._load_server(server_id)
            self._require_enabled(server)
            if server.id in self._transports:
                await self._stop_server_unlocked(server.id)

            transport = self._transport_factory(server.model_copy(deep=True))
            try:
                await transport.connect()
                tools = await transport.list_tools()
            except Exception as exc:  # noqa: BLE001 - transport adapters define their own errors.
                self._last_errors[server.id] = self._exception_summary(exc)
                await self._close_best_effort(transport)
                await self._audit_best_effort(
                    "external_server.lifecycle",
                    payload={
                        "reason_code": "external_server_start_failed",
                        "server_id": server.id,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                    target_type="external_server",
                    target_id=server.id,
                )
                raise

            self._servers[server.id] = server.model_copy(deep=True)
            self._transports[server.id] = transport
            self._last_errors[server.id] = None
            self._replace_server_tools(server.id, tools)
            await self._audit(
                "external_server.lifecycle",
                payload={
                    "reason_code": "external_server_started",
                    "server_id": server.id,
                    "transport": server.transport,
                },
                target_type="external_server",
                target_id=server.id,
            )
            await self._audit(
                "external_server.discovery",
                payload={
                    "reason_code": "external_server_discovered",
                    "server_id": server.id,
                    "tool_count": len(tools),
                },
                target_type="external_server",
                target_id=server.id,
            )
            return {
                "ok": True,
                "reason_code": "external_server_started",
                "server_id": server.id,
                "tool_count": len(tools),
            }

    async def stop_server(self, server_id: str) -> dict[str, Any]:
        """Stop one external server and clear its discovered virtual tools."""
        async with self._lock:
            server = await self._find_server(server_id)
            if server is None:
                raise GatewayExternalRuntimeError(
                    f"External server '{server_id}' was not found",
                    reason_code="external_server_not_found",
                    server_id=server_id,
                )
            if server_id not in self._transports:
                self._clear_server_tools(server_id)
                self._servers.pop(server_id, None)
                self._last_errors.pop(server_id, None)
                return {
                    "ok": True,
                    "reason_code": "external_server_already_stopped",
                    "server_id": server_id,
                }
            await self._stop_server_unlocked(server_id)
            return {
                "ok": True,
                "reason_code": "external_server_stopped",
                "server_id": server_id,
            }

    async def list_runtime_servers(self) -> dict[str, Any]:
        """Return runtime status rows for configured external servers."""
        async with self._lock:
            configured = {
                server.id: server
                for server in await self._list_server_definitions()
            }
            for server_id, server in self._servers.items():
                configured.setdefault(server_id, server.model_copy(deep=True))
            snapshots = [
                (
                    server_id,
                    server.model_copy(deep=True),
                    self._transports.get(server_id),
                    self._last_errors.get(server_id),
                    self._count_tools_for_server(server_id),
                )
                for server_id, server in sorted(configured.items())
            ]

        rows: list[dict[str, Any]] = []
        for server_id, server, transport, last_error, tool_count in snapshots:
            checks: dict[str, Any] = {
                "configured": True,
                "connected": False,
                "initialized": False,
            }
            if transport is not None:
                checks.update(await transport.health_check())
            rows.append(
                {
                    "id": server_id,
                    "name": server.name,
                    "transport": server.transport,
                    "enabled": server.enabled,
                    "status": self._server_status(
                        active=transport is not None,
                        connected=bool(checks.get("connected")),
                        enabled=server.enabled,
                        last_error=last_error,
                    ),
                    "tool_count": tool_count,
                    "checks": dict(checks),
                    "last_error": last_error,
                }
            )
        return {"servers": rows, "total_servers": len(rows)}

    async def list_virtual_tools(self) -> list[VirtualExternalTool]:
        """Return caller-owned discovered virtual tools sorted by name."""
        async with self._lock:
            return [
                self._virtual_tools[name].copy()
                for name in sorted(self._virtual_tools)
            ]

    async def _stop_server_unlocked(self, server_id: str) -> None:
        transport = self._transports.pop(server_id, None)
        close_error: str | None = None
        if transport is not None:
            try:
                await transport.close()
            except Exception as exc:  # noqa: BLE001 - stop must still clear state.
                close_error = self._exception_summary(exc)
        self._servers.pop(server_id, None)
        self._last_errors.pop(server_id, None)
        self._clear_server_tools(server_id)
        payload: dict[str, Any] = {
            "reason_code": "external_server_stopped",
            "server_id": server_id,
        }
        if close_error is not None:
            payload["close_error"] = close_error
        await self._audit_best_effort(
            "external_server.lifecycle",
            payload=payload,
            target_type="external_server",
            target_id=server_id,
        )

    async def _load_server(self, server_id: str) -> ExternalServerDefinition:
        server = await self._find_server(server_id)
        if server is None:
            raise GatewayExternalRuntimeError(
                f"External server '{server_id}' was not found",
                reason_code="external_server_not_found",
                server_id=server_id,
            )
        return server

    async def _find_server(self, server_id: str) -> ExternalServerDefinition | None:
        row = await self._external_registry_store.get_server(server_id)
        if row is None:
            row = self._servers.get(server_id)
        if row is None:
            return None
        return self._coerce_server_definition(row)

    async def _list_server_definitions(self) -> list[ExternalServerDefinition]:
        if hasattr(self._external_registry_store, "list_server_definitions"):
            rows = await self._external_registry_store.list_server_definitions()
        else:
            rows = await self._external_registry_store.list_servers()
        return [self._coerce_server_definition(row) for row in rows]

    @staticmethod
    def _coerce_server_definition(row: Any) -> ExternalServerDefinition:
        if isinstance(row, ExternalServerDefinition):
            return row.model_copy(deep=True)
        if isinstance(row, dict):
            return ExternalServerDefinition(**deepcopy(row))
        raise TypeError("external registry rows must be ExternalServerDefinition or dict")

    @staticmethod
    def _require_enabled(server: ExternalServerDefinition) -> None:
        if server.enabled:
            return
        raise GatewayExternalRuntimeError(
            f"External server '{server.id}' is disabled",
            reason_code="external_server_disabled",
            server_id=server.id,
        )

    def _replace_server_tools(
        self,
        server_id: str,
        tools: list[ExternalToolDefinition],
    ) -> None:
        self._clear_server_tools(server_id)
        for tool in tools:
            tool_copy = tool.copy()
            metadata = deepcopy(tool_copy.metadata or {})
            metadata.setdefault("external_server_id", server_id)
            metadata.setdefault("upstream_tool_name", tool_copy.name)
            virtual_name = self._virtual_tool_name(server_id, tool_copy.name)
            self._virtual_tools[virtual_name] = VirtualExternalTool(
                virtual_name=virtual_name,
                server_id=server_id,
                upstream_tool_name=tool_copy.name,
                description=tool_copy.description,
                input_schema=deepcopy(tool_copy.input_schema),
                metadata=metadata,
                is_write=self._is_write_tool(tool_copy.name, metadata),
            )

    def _clear_server_tools(self, server_id: str) -> None:
        self._virtual_tools = {
            name: tool
            for name, tool in self._virtual_tools.items()
            if tool.server_id != server_id
        }

    def _count_tools_for_server(self, server_id: str) -> int:
        return sum(1 for tool in self._virtual_tools.values() if tool.server_id == server_id)

    async def _audit(
        self,
        event_type: str,
        *,
        payload: dict[str, Any],
        actor_id: str | None = None,
        profile_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
    ) -> None:
        if self._audit_store is None:
            return
        event = AuditEvent(
            id=f"mcp-ext-runtime-{uuid4().hex}",
            event_type=event_type,
            actor_id=actor_id,
            profile_id=profile_id,
            target_type=target_type,
            target_id=target_id,
            payload=deepcopy(payload),
            created_at=datetime.now(timezone.utc),
        )
        await self._audit_store.append_event(event)

    async def _audit_best_effort(
        self,
        event_type: str,
        *,
        payload: dict[str, Any],
        actor_id: str | None = None,
        profile_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
    ) -> None:
        try:
            await self._audit(
                event_type,
                payload=payload,
                actor_id=actor_id,
                profile_id=profile_id,
                target_type=target_type,
                target_id=target_id,
            )
        except Exception:
            return

    @staticmethod
    async def _close_best_effort(transport: ExternalFederationTransport) -> None:
        try:
            await transport.close()
        except Exception:
            return

    @staticmethod
    def _server_status(
        *,
        active: bool,
        connected: bool,
        enabled: bool,
        last_error: str | None,
    ) -> str:
        if not enabled:
            return "disabled"
        if not active:
            return "stopped" if last_error is None else "unhealthy"
        if connected and last_error is None:
            return "healthy"
        if connected:
            return "degraded"
        return "unhealthy"

    @staticmethod
    def _exception_summary(exc: BaseException) -> str:
        message = str(exc).strip()
        error_type = type(exc).__name__
        return f"{error_type}: {message}" if message else error_type

    @staticmethod
    def _virtual_tool_name(server_id: str, tool_name: str) -> str:
        return f"ext.{server_id}.{tool_name}"

    @staticmethod
    def _is_write_tool(tool_name: str, metadata: dict[str, Any]) -> bool:
        annotations = metadata.get("annotations")
        if isinstance(annotations, dict) and isinstance(annotations.get("readOnlyHint"), bool):
            return not annotations["readOnlyHint"]
        for key in ("read_only", "readOnly", "is_read_only"):
            value = metadata.get(key)
            if isinstance(value, bool):
                return not value
        lowered = tool_name.lower()
        return any(
            token in lowered
            for token in ("create", "update", "delete", "write", "patch", "import")
        )
