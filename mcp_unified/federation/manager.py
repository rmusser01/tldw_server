"""Registry-backed non-spawning external federation manager."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from mcp_unified.storage import AuditEvent, ExternalServerDefinition

from .models import (
    ExternalToolCallResult,
    FederatedToolResult,
    FederationPolicyDenied,
    VirtualExternalTool,
)
from .transports import ExternalFederationTransport


class ExternalFederationManager:
    """Standalone external federation shell with no process-spawning transports."""

    def __init__(
        self,
        *,
        registry_store: Any,
        transport_factory: Callable[[ExternalServerDefinition], ExternalFederationTransport],
        audit_store: Any | None = None,
    ) -> None:
        self._registry_store = registry_store
        self._transport_factory = transport_factory
        self._audit_store = audit_store
        self._servers: dict[str, ExternalServerDefinition] = {}
        self._transports: dict[str, ExternalFederationTransport] = {}
        self._virtual_tools: dict[str, VirtualExternalTool] = {}
        self._last_errors: dict[str, str | None] = {}
        self._started = False
        self._lock = asyncio.Lock()

    @property
    def started(self) -> bool:
        """Return whether the manager has an active logical lifecycle."""
        return self._started

    async def start(self) -> None:
        """Load enabled registry definitions and connect fake transports."""
        async with self._lock:
            await self._stop_unlocked()
            servers = await self._load_enabled_servers()
            for server in servers:
                transport = self._transport_factory(server)
                self._servers[server.id] = server
                self._transports[server.id] = transport
                self._last_errors[server.id] = None
                await transport.connect()
                await self._audit(
                    "external_server.lifecycle",
                    payload={
                        "reason_code": "started",
                        "server_id": server.id,
                        "transport": server.transport,
                        "spawns_process": False,
                    },
                    target_type="external_server",
                    target_id=server.id,
                )
                await self._refresh_server_tools_unlocked(server.id)
            self._started = True

    async def stop(self) -> None:
        """Close active fake transports and clear runtime state."""
        async with self._lock:
            await self._stop_unlocked()

    async def refresh(self, server_id: str | None = None) -> dict[str, Any]:
        """Refresh virtual tool metadata for one server or all loaded servers."""
        async with self._lock:
            target_ids = [server_id] if server_id else sorted(self._servers)
            refreshed = 0
            errors: dict[str, str] = {}
            for target_id in target_ids:
                if target_id not in self._servers:
                    errors[target_id] = "unknown_server"
                    continue
                try:
                    await self._refresh_server_tools_unlocked(target_id)
                    refreshed += 1
                except (OSError, RuntimeError, TimeoutError, ValueError, TypeError):
                    errors[target_id] = "external_server_discovery_failed"
                    self._last_errors[target_id] = "external_server_discovery_failed"
                    self._clear_server_tools(target_id)
            return {
                "refreshed_servers": refreshed,
                "total_servers": len(target_ids),
                "virtual_tools": len(self._virtual_tools),
                "errors": errors,
            }

    async def list_servers(self) -> list[dict[str, Any]]:
        """Return summarized health for loaded external server definitions."""
        rows: list[dict[str, Any]] = []
        for server_id in sorted(self._servers):
            server = self._servers[server_id]
            transport = self._transports.get(server_id)
            checks = {"configured": True, "connected": False, "spawns_process": False}
            if transport is not None:
                checks = await transport.health_check()
            connected = bool(checks.get("connected"))
            status = self._server_status(server_id=server_id, connected=connected)
            rows.append(
                {
                    "id": server.id,
                    "name": server.name,
                    "transport": server.transport,
                    "tool_count": self._count_tools_for_server(server.id),
                    "status": status,
                    "checks": dict(checks),
                    "last_error": self._last_errors.get(server.id),
                }
            )
        return rows

    def list_virtual_tools(self) -> list[VirtualExternalTool]:
        """Return discovered virtual tools sorted by namespaced tool name."""
        return [
            self._virtual_tools[name]
            for name in sorted(self._virtual_tools)
        ]

    async def execute_virtual_tool(
        self,
        virtual_tool_name: str,
        arguments: dict[str, Any] | None,
        *,
        effective_policy: Any,
        actor_id: str | None = None,
        context: Any = None,
    ) -> FederatedToolResult:
        """Execute a virtual external tool after effective-policy checks pass."""
        async with self._lock:
            virtual_tool = self._virtual_tools.get(virtual_tool_name)
            if virtual_tool is None:
                raise ValueError(f"Unknown external virtual tool '{virtual_tool_name}'")

            server = self._servers[virtual_tool.server_id]
            profile_id = self._policy_profile_id(effective_policy)
            deny_reason = self._deny_reason(
                server=server,
                virtual_tool=virtual_tool,
                effective_policy=effective_policy,
            )
            if deny_reason is not None:
                await self._audit_execution(
                    event_type="external_tool.denied",
                    reason_code=deny_reason,
                    virtual_tool=virtual_tool,
                    actor_id=actor_id,
                    profile_id=profile_id,
                )
                raise FederationPolicyDenied(
                    deny_reason,
                    f"External tool '{virtual_tool_name}' denied: {deny_reason}",
                    payload={
                        "reason_code": deny_reason,
                        "server_id": virtual_tool.server_id,
                        "virtual_tool_name": virtual_tool_name,
                    },
                )

            transport = self._transports[virtual_tool.server_id]
            result = await transport.call_tool(
                virtual_tool.upstream_tool_name,
                deepcopy(arguments or {}),
                context=context,
            )
            await self._audit_execution(
                event_type="external_tool.allowed",
                reason_code="allowed",
                virtual_tool=virtual_tool,
                actor_id=actor_id,
                profile_id=profile_id,
            )
            return self._federated_result(virtual_tool, result)

    async def _load_enabled_servers(self) -> list[ExternalServerDefinition]:
        if hasattr(self._registry_store, "list_server_definitions"):
            rows = await self._registry_store.list_server_definitions(enabled=True)
        else:
            rows = await self._registry_store.list_servers()
        servers: list[ExternalServerDefinition] = []
        for row in rows:
            server = self._coerce_server_definition(row)
            if server.enabled:
                servers.append(server)
        return servers

    @staticmethod
    def _coerce_server_definition(row: Any) -> ExternalServerDefinition:
        if isinstance(row, ExternalServerDefinition):
            return row.model_copy(deep=True)
        if isinstance(row, dict):
            return ExternalServerDefinition(**deepcopy(row))
        raise TypeError("external registry rows must be ExternalServerDefinition or dict")

    async def _stop_unlocked(self) -> None:
        for server_id, transport in list(self._transports.items()):
            await transport.close()
            await self._audit(
                "external_server.lifecycle",
                payload={
                    "reason_code": "stopped",
                    "server_id": server_id,
                    "spawns_process": False,
                },
                target_type="external_server",
                target_id=server_id,
            )
        self._servers = {}
        self._transports = {}
        self._virtual_tools = {}
        self._last_errors = {}
        self._started = False

    async def _refresh_server_tools_unlocked(self, server_id: str) -> None:
        transport = self._transports[server_id]
        tools = await transport.list_tools()
        self._clear_server_tools(server_id)
        for tool in tools:
            virtual_name = self._virtual_tool_name(server_id, tool.name)
            metadata = deepcopy(tool.metadata or {})
            metadata.setdefault("external_server_id", server_id)
            metadata.setdefault("upstream_tool_name", tool.name)
            self._virtual_tools[virtual_name] = VirtualExternalTool(
                virtual_name=virtual_name,
                server_id=server_id,
                upstream_tool_name=tool.name,
                description=tool.description,
                input_schema=deepcopy(tool.input_schema),
                metadata=metadata,
                is_write=self._is_write_tool(tool.name, metadata),
            )
        self._last_errors[server_id] = None
        await self._audit(
            "external_server.discovery",
            payload={
                "reason_code": "discovered",
                "server_id": server_id,
                "tool_count": len(tools),
            },
            target_type="external_server",
            target_id=server_id,
        )

    def _deny_reason(
        self,
        *,
        server: ExternalServerDefinition,
        virtual_tool: VirtualExternalTool,
        effective_policy: Any,
    ) -> str | None:
        if self._tool_matches_any(
            virtual_tool,
            self._policy_list(effective_policy, "denied_tools"),
        ):
            return "tool_denied"
        if not self._has_server_grant(effective_policy, virtual_tool):
            return "external_server_not_granted"
        allowed_tools = self._policy_list(effective_policy, "allowed_tools")
        if allowed_tools and not self._tool_matches_any(virtual_tool, allowed_tools):
            return "tool_not_allowed"
        if self._missing_credential_slots(server, effective_policy):
            return "required_credential_grant_missing"
        return None

    def _has_server_grant(
        self,
        effective_policy: Any,
        virtual_tool: VirtualExternalTool,
    ) -> bool:
        for grant in self._policy_dicts(effective_policy, "external_server_grants"):
            if not self._grant_matches_server(grant, virtual_tool.server_id):
                continue
            tool_patterns = self._grant_tool_patterns(grant)
            if tool_patterns and not self._tool_matches_any(virtual_tool, tool_patterns):
                continue
            return True
        return False

    def _missing_credential_slots(
        self,
        server: ExternalServerDefinition,
        effective_policy: Any,
    ) -> list[str]:
        required_slots = {
            str(slot).strip()
            for slot in server.credential_slots
            if str(slot).strip()
        }
        if not required_slots:
            return []
        granted_slots: set[str] = set()
        for grant in self._policy_dicts(effective_policy, "credential_grants"):
            if not self._grant_matches_server(grant, server.id):
                continue
            granted_slots.update(self._credential_slots_for_grant(grant))
        return sorted(required_slots - granted_slots)

    @staticmethod
    def _credential_slots_for_grant(grant: dict[str, Any]) -> set[str]:
        slots: set[str] = set()
        for key in ("credential_slot", "slot"):
            value = str(grant.get(key) or "").strip()
            if value:
                slots.add(value)
        for key in ("credential_slots", "slots"):
            raw = grant.get(key)
            if isinstance(raw, str):
                values: Iterable[Any] = [raw]
            elif isinstance(raw, Iterable) and not isinstance(raw, (bytes, dict)):
                values = raw
            else:
                values = []
            slots.update(str(value).strip() for value in values if str(value).strip())
        return slots

    @staticmethod
    def _grant_matches_server(grant: dict[str, Any], server_id: str) -> bool:
        if grant.get("enabled") is False:
            return False
        raw_server_id = (
            grant.get("server_id")
            or grant.get("external_server_id")
            or grant.get("id")
        )
        return str(raw_server_id or "").strip() == server_id

    @staticmethod
    def _grant_tool_patterns(grant: dict[str, Any]) -> list[str]:
        patterns: list[str] = []
        for key in ("tools", "tool_names", "allowed_tools"):
            raw = grant.get(key)
            if isinstance(raw, str):
                patterns.append(raw)
            elif isinstance(raw, Iterable) and not isinstance(raw, (bytes, dict)):
                patterns.extend(str(item) for item in raw)
        return [pattern for pattern in patterns if pattern.strip()]

    @classmethod
    def _tool_matches_any(
        cls,
        virtual_tool: VirtualExternalTool,
        patterns: list[str],
    ) -> bool:
        return any(cls._tool_matches_pattern(virtual_tool, pattern) for pattern in patterns)

    @staticmethod
    def _tool_matches_pattern(
        virtual_tool: VirtualExternalTool,
        pattern: str,
    ) -> bool:
        candidate = str(pattern or "").strip()
        if candidate in {"*", "ext.*"}:
            return True
        if candidate.endswith("*"):
            prefix = candidate[:-1]
            return (
                virtual_tool.virtual_name.startswith(prefix)
                or virtual_tool.upstream_tool_name.startswith(prefix)
            )
        return candidate in {
            virtual_tool.virtual_name,
            virtual_tool.upstream_tool_name,
        }

    @staticmethod
    def _policy_list(effective_policy: Any, field_name: str) -> list[str]:
        values = ExternalFederationManager._policy_value(effective_policy, field_name, [])
        if isinstance(values, str):
            return [values]
        if isinstance(values, Iterable) and not isinstance(values, (bytes, dict)):
            return [str(value) for value in values if str(value).strip()]
        return []

    @staticmethod
    def _policy_dicts(effective_policy: Any, field_name: str) -> list[dict[str, Any]]:
        values = ExternalFederationManager._policy_value(effective_policy, field_name, [])
        if isinstance(values, dict):
            return [deepcopy(values)]
        if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
            return []
        return [
            deepcopy(value)
            for value in values
            if isinstance(value, dict)
        ]

    @staticmethod
    def _policy_profile_id(effective_policy: Any) -> str | None:
        value = ExternalFederationManager._policy_value(effective_policy, "profile_id", None)
        return str(value) if value is not None else None

    @staticmethod
    def _policy_value(effective_policy: Any, field_name: str, default: Any) -> Any:
        if isinstance(effective_policy, dict):
            return effective_policy.get(field_name, default)
        return getattr(effective_policy, field_name, default)

    async def _audit_execution(
        self,
        *,
        event_type: str,
        reason_code: str,
        virtual_tool: VirtualExternalTool,
        actor_id: str | None,
        profile_id: str | None,
    ) -> None:
        await self._audit(
            event_type,
            actor_id=actor_id,
            profile_id=profile_id,
            target_type="external_tool",
            target_id=virtual_tool.virtual_name,
            payload={
                "reason_code": reason_code,
                "server_id": virtual_tool.server_id,
                "virtual_tool_name": virtual_tool.virtual_name,
                "upstream_tool_name": virtual_tool.upstream_tool_name,
            },
        )

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
            id=f"mcp-fed-{uuid4().hex}",
            event_type=event_type,
            actor_id=actor_id,
            profile_id=profile_id,
            target_type=target_type,
            target_id=target_id,
            payload=deepcopy(payload),
            created_at=datetime.now(timezone.utc),
        )
        await self._audit_store.append_event(event)

    def _server_status(self, *, server_id: str, connected: bool) -> str:
        if connected and self._last_errors.get(server_id) is None:
            return "healthy"
        if connected or self._last_errors.get(server_id) is None:
            return "degraded"
        return "unhealthy"

    @staticmethod
    def _virtual_tool_name(server_id: str, tool_name: str) -> str:
        return f"ext.{server_id}.{tool_name}"

    def _clear_server_tools(self, server_id: str) -> None:
        self._virtual_tools = {
            name: tool
            for name, tool in self._virtual_tools.items()
            if tool.server_id != server_id
        }

    def _count_tools_for_server(self, server_id: str) -> int:
        return sum(1 for tool in self._virtual_tools.values() if tool.server_id == server_id)

    @staticmethod
    def _federated_result(
        virtual_tool: VirtualExternalTool,
        result: ExternalToolCallResult,
    ) -> FederatedToolResult:
        return FederatedToolResult(
            content=deepcopy(result.content),
            is_error=result.is_error,
            metadata=deepcopy(result.metadata),
            server_id=virtual_tool.server_id,
            upstream_tool_name=virtual_tool.upstream_tool_name,
            virtual_tool_name=virtual_tool.virtual_name,
        )

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
