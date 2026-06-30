"""In-process external MCP server runtime management for the standalone gateway."""

from __future__ import annotations

import asyncio
import inspect
import traceback
from collections.abc import Callable, Iterable
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import uuid4

from loguru import logger

from mcp_unified.federation.installers import (
    ExternalServerInstaller,
    NullExternalServerInstaller,
)
from mcp_unified.federation.models import (
    BrokeredExternalCredential,
    ExternalToolCallResult,
    ExternalToolDefinition,
    FederatedToolResult,
    FederationPolicyDenied,
    VirtualExternalTool,
)
from mcp_unified.federation.transports import ExternalFederationTransport
from mcp_unified.interfaces.storage import AuditStore, ExternalRegistryStore
from mcp_unified.storage import AuditEvent, ExternalServerDefinition

_INSTALLER_PUBLIC_KEYS = frozenset(
    {
        "ok",
        "available",
        "reason_code",
        "server_id",
        "installer",
        "version",
        "installed_version",
        "latest_version",
        "message",
        "details",
        "required_fields",
        "warnings",
        "error_type",
    }
)
_INSTALLER_SENSITIVE_KEY_TOKENS = (
    "secret",
    "token",
    "password",
    "credential",
    "authorization",
    "api_key",
    "apikey",
    "headers",
    "header",
    "env",
    "command",
    "args",
    "argv",
)
_UNSAFE_INSTALLER_VALUE = object()
_DEFAULT_INSTALLER_STATUS_TIMEOUT_SECONDS = 2.0


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


class ExternalCredentialBroker(Protocol):
    """Resolve ephemeral credential material for one external tool call."""

    async def resolve_external_credential(
        self,
        *,
        server: ExternalServerDefinition,
        virtual_tool: VirtualExternalTool,
        credential_slots: list[str],
        effective_policy: Any = None,
        actor_id: str | None = None,
        context: Any = None,
    ) -> BrokeredExternalCredential | dict[str, Any] | None:
        """Return per-call credential material or None when no grant applies."""
        ...


class GatewayExternalRuntimeManager:
    """Manage active external server transports for one in-process gateway."""

    def __init__(
        self,
        *,
        external_registry_store: ExternalRegistryStore,
        transport_factory: Callable[[ExternalServerDefinition], ExternalFederationTransport],
        audit_store: AuditStore | None = None,
        credential_broker: ExternalCredentialBroker | Callable[..., Any] | None = None,
        installer: ExternalServerInstaller | None = None,
        installer_status_timeout_seconds: float = _DEFAULT_INSTALLER_STATUS_TIMEOUT_SECONDS,
    ) -> None:
        self._external_registry_store = external_registry_store
        self._transport_factory = transport_factory
        self._audit_store = audit_store
        self._credential_broker = credential_broker
        self._installer = installer or NullExternalServerInstaller()
        self._installer_status_timeout_seconds = installer_status_timeout_seconds
        self._servers: dict[str, ExternalServerDefinition] = {}
        self._transports: dict[str, ExternalFederationTransport] = {}
        self._virtual_tools: dict[str, VirtualExternalTool] = {}
        self._last_errors: dict[str, str | None] = {}
        self._lock = asyncio.Lock()

    async def start_server(self, server_id: str) -> dict[str, Any]:
        """Start one configured external server and discover its virtual tools."""
        server = await self._load_server(server_id)
        self._require_enabled(server)

        async with self._lock:
            previous = self._pop_runtime_unlocked(server.id)
        if previous is not None:
            await self._close_stopped_transport(server.id, previous)

        transport, tools = await self._connect_and_discover_server(server)
        async with self._lock:
            replaced = self._pop_runtime_unlocked(server.id)
            self._commit_started_runtime_unlocked(server, transport, tools)
        if replaced is not None:
            await self._close_stopped_transport(server.id, replaced)

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
            known_active = server_id in self._transports or server_id in self._servers
            transport = self._pop_runtime_unlocked(server_id)
        if not known_active and await self._find_server(server_id) is None:
            raise GatewayExternalRuntimeError(
                f"External server '{server_id}' was not found",
                reason_code="external_server_not_found",
                server_id=server_id,
            )
        if transport is None:
            return {
                "ok": True,
                "reason_code": "external_server_already_stopped",
                "server_id": server_id,
            }
        await self._close_stopped_transport(server_id, transport)
        return {
            "ok": True,
            "reason_code": "external_server_stopped",
            "server_id": server_id,
        }

    async def stop_all(self) -> dict[str, Any]:
        """Stop every active external server transport."""
        async with self._lock:
            target_ids = sorted(self._transports)

        stopped = 0
        errors: dict[str, Any] = {}
        for target_id in target_ids:
            try:
                payload = await self.stop_server(target_id)
            except GatewayExternalRuntimeError as exc:
                errors[target_id] = exc.reason_code
            except Exception as exc:  # noqa: BLE001 - shutdown must remain best-effort.
                logger.opt(exception=True).error(
                    "External runtime stop failed server_id={!r} error_type={!r}",
                    target_id,
                    type(exc).__name__,
                )
                errors[target_id] = {
                    "reason_code": "external_server_stop_failed",
                    "error_type": type(exc).__name__,
                    "error_summary": self._exception_summary(exc),
                }
            else:
                if payload["reason_code"] == "external_server_stopped":
                    stopped += 1

        return {
            "ok": not errors,
            "reason_code": "external_runtime_stopped",
            "stopped_servers": stopped,
            "total_servers": len(target_ids),
            "errors": errors,
        }

    async def restart_server(self, server_id: str) -> dict[str, Any]:
        """Reload one server definition and replace its active transport."""
        stopped = await self.stop_server(server_id)
        server = await self._load_server(server_id)
        self._require_enabled(server)
        started = await self.start_server(server.id)
        return {
            "ok": True,
            "reason_code": "external_server_restarted",
            "server_id": server.id,
            "stop_reason_code": stopped["reason_code"],
            "start_reason_code": started["reason_code"],
            "tool_count": started["tool_count"],
        }

    async def refresh_server(self, server_id: str | None = None) -> dict[str, Any]:
        """Refresh discovered tools for one active server or every active server."""
        async with self._lock:
            if server_id is not None:
                running = server_id in self._transports
                target_ids = [server_id]
            else:
                target_ids = sorted(self._transports)
                running = True

        if server_id is not None and not running:
            await self._load_server(server_id)

        refreshed = 0
        errors: dict[str, str] = {}
        for target_id in target_ids:
            async with self._lock:
                transport = self._transports.get(target_id)
            if transport is None:
                errors[target_id] = "external_server_not_running"
                continue
            ok = await self._refresh_server_tools(target_id, transport)
            if ok:
                refreshed += 1
            else:
                errors[target_id] = "external_server_discovery_failed"

        async with self._lock:
            virtual_tool_count = len(self._virtual_tools)
        payload: dict[str, Any] = {
            "ok": not errors,
            "reason_code": "external_server_refreshed",
            "refreshed_servers": refreshed,
            "total_servers": len(target_ids),
            "virtual_tools": virtual_tool_count,
            "errors": errors,
        }
        if server_id is not None:
            payload["server_id"] = server_id
        return payload

    async def reconcile(self, server_id: str | None = None) -> dict[str, Any]:
        """Reconcile active transports against current registry definitions."""
        definitions = {
            server.id: server
            for server in await self._list_server_definitions()
        }
        async with self._lock:
            active_ids = set(self._transports)
            current_definitions = {
                target_id: server.model_copy(deep=True)
                for target_id, server in self._servers.items()
            }
        if server_id is not None:
            if server_id not in definitions and server_id not in active_ids:
                raise GatewayExternalRuntimeError(
                    f"External server '{server_id}' was not found",
                    reason_code="external_server_not_found",
                    server_id=server_id,
                )
            target_ids = [server_id]
        else:
            target_ids = sorted(set(definitions) | active_ids)

        started = 0
        stopped = 0
        restarted = 0
        refreshed = 0
        errors: dict[str, str] = {}

        for target_id in target_ids:
            server = definitions.get(target_id)
            async with self._lock:
                is_active = target_id in self._transports
                current = current_definitions.get(target_id)
            if server is None:
                if is_active:
                    await self.stop_server(target_id)
                    stopped += 1
                continue
            if not server.enabled:
                if is_active:
                    await self.stop_server(target_id)
                    stopped += 1
                continue
            if not is_active:
                if server.auto_start:
                    try:
                        await self.start_server(target_id)
                    except GatewayExternalRuntimeError as exc:
                        errors[target_id] = exc.reason_code
                    else:
                        started += 1
                continue

            if current is None or self._definition_changed(current, server):
                try:
                    await self.restart_server(target_id)
                except GatewayExternalRuntimeError as exc:
                    errors[target_id] = exc.reason_code
                else:
                    restarted += 1
                continue

            payload = await self.refresh_server(target_id)
            if payload["ok"]:
                refreshed += 1
            else:
                errors[target_id] = "external_server_discovery_failed"

        return {
            "ok": not errors,
            "reason_code": "external_server_reconciled",
            "server_id": server_id,
            "started_servers": started,
            "stopped_servers": stopped,
            "restarted_servers": restarted,
            "refreshed_servers": refreshed,
            "total_servers": len(target_ids),
            "errors": errors,
        }

    async def list_runtime_servers(self) -> dict[str, Any]:
        """Return runtime status rows for configured external servers."""
        definitions = await self._list_server_definitions()
        async with self._lock:
            configured = {server.id: server for server in definitions}
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
                try:
                    checks.update(await transport.health_check())
                except Exception as exc:  # noqa: BLE001 - status must not break.
                    last_error = self._exception_summary(exc)
                    checks.update(
                        {
                            "connected": False,
                            "initialized": False,
                            "error": True,
                            "error_type": type(exc).__name__,
                        }
                    )
                    async with self._lock:
                        if self._transports.get(server_id) is transport:
                            self._last_errors[server_id] = last_error
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
                    "installer": await self._installer_status(server),
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

    async def has_virtual_tool(self, virtual_tool_name: str) -> bool:
        """Return whether an active virtual tool exists without copying the catalog."""

        async with self._lock:
            return virtual_tool_name in self._virtual_tools

    async def install_server(
        self,
        server_id: str,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        """Delegate optional install work to the configured installer adapter."""
        server = await self._load_server(server_id)
        self._require_enabled(server)
        return await self._installer_operation_payload(
            "install",
            server=server,
            callback=lambda: self._installer.install_server(
                server.model_copy(deep=True),
                context=context,
            ),
            fallback_reason_code="external_server_install_not_configured",
            failure_reason_code="external_server_install_failed",
        )

    async def update_server(
        self,
        server_id: str,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        """Delegate optional update work to the configured installer adapter."""
        server = await self._load_server(server_id)
        self._require_enabled(server)
        return await self._installer_operation_payload(
            "update",
            server=server,
            callback=lambda: self._installer.update_server(
                server.model_copy(deep=True),
                context=context,
            ),
            fallback_reason_code="external_server_update_not_configured",
            failure_reason_code="external_server_update_failed",
        )

    async def execute_virtual_tool(
        self,
        virtual_tool_name: str,
        arguments: dict[str, Any] | None,
        *,
        effective_policy: Any = None,
        actor_id: str | None = None,
        context: Any = None,
    ) -> FederatedToolResult:
        """Execute one virtual external tool with per-call credential brokering."""
        async with self._lock:
            virtual_tool = self._virtual_tools.get(virtual_tool_name)
            if virtual_tool is None:
                raise GatewayExternalRuntimeError(
                    f"External virtual tool '{virtual_tool_name}' was not found",
                    reason_code="external_virtual_tool_not_found",
                )
            virtual_tool = virtual_tool.copy()
            server = self._servers.get(virtual_tool.server_id)
            transport = self._transports.get(virtual_tool.server_id)
            if server is None or transport is None:
                raise GatewayExternalRuntimeError(
                    f"External server '{virtual_tool.server_id}' is not active",
                    reason_code="external_server_transport_unavailable",
                    server_id=virtual_tool.server_id,
                )
            server = server.model_copy(deep=True)

        deny_reason = self._deny_reason(
            server=server,
            virtual_tool=virtual_tool,
            effective_policy=effective_policy,
        )
        if deny_reason is not None:
            profile_id = self._policy_profile_id(effective_policy)
            await self._audit_execution(
                event_type="external_tool.denied",
                reason_code=deny_reason,
                virtual_tool=virtual_tool,
                actor_id=actor_id,
                profile_id=profile_id,
            )
            raise FederationPolicyDenied(
                deny_reason,
                f"External virtual tool '{virtual_tool.virtual_name}' is not allowed",
                payload={
                    "reason_code": deny_reason,
                    "server_id": virtual_tool.server_id,
                    "virtual_tool_name": virtual_tool.virtual_name,
                },
            )

        runtime_auth = await self._resolve_runtime_auth(
            server=server,
            virtual_tool=virtual_tool,
            effective_policy=effective_policy,
            actor_id=actor_id,
            context=context,
        )
        try:
            result = await transport.call_tool(
                virtual_tool.upstream_tool_name,
                deepcopy(arguments or {}),
                context=context,
                runtime_auth=runtime_auth,
            )
        except Exception as exc:  # noqa: BLE001 - transport adapters define call errors.
            await self._audit_execution(
                event_type="external_tool.failed",
                reason_code="external_tool_call_failed",
                virtual_tool=virtual_tool,
                actor_id=actor_id,
                profile_id=self._policy_profile_id(effective_policy),
                extra_payload={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            raise GatewayExternalRuntimeError(
                f"External tool '{virtual_tool.virtual_name}' failed during execution",
                reason_code="external_tool_call_failed",
                server_id=virtual_tool.server_id,
            ) from exc

        if runtime_auth is not None:
            metadata = deepcopy(result.metadata or {})
            metadata.update(self._public_runtime_auth_metadata(runtime_auth))
            metadata["credential_injection"] = self._summarize_runtime_auth(runtime_auth)
            result = ExternalToolCallResult(
                content=deepcopy(result.content),
                is_error=result.is_error,
                metadata=metadata,
            )
        await self._audit_execution(
            event_type="external_tool.allowed",
            reason_code="allowed",
            virtual_tool=virtual_tool,
            actor_id=actor_id,
            profile_id=self._policy_profile_id(effective_policy),
            credential_injection=(
                self._summarize_runtime_auth(runtime_auth)
                if runtime_auth is not None
                else None
            ),
        )
        return self._federated_result(virtual_tool, result)

    def _pop_runtime_unlocked(
        self,
        server_id: str,
    ) -> ExternalFederationTransport | None:
        transport = self._transports.pop(server_id, None)
        self._servers.pop(server_id, None)
        self._last_errors.pop(server_id, None)
        self._clear_server_tools(server_id)
        return transport

    async def _close_stopped_transport(
        self,
        server_id: str,
        transport: ExternalFederationTransport,
    ) -> None:
        close_error: str | None = None
        try:
            await transport.close()
        except Exception as exc:  # noqa: BLE001 - stop must still clear state.
            close_error = self._exception_summary(exc)
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

    async def _connect_and_discover_server(
        self,
        server: ExternalServerDefinition,
    ) -> tuple[ExternalFederationTransport, list[ExternalToolDefinition]]:
        transport: ExternalFederationTransport | None = None
        try:
            transport = self._transport_factory(server.model_copy(deep=True))
            await transport.connect()
            tools = await transport.list_tools()
        except Exception as exc:  # noqa: BLE001 - transport adapters define their own errors.
            async with self._lock:
                self._last_errors[server.id] = self._exception_summary(exc)
            if transport is not None:
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
            raise GatewayExternalRuntimeError(
                f"External server '{server.id}' failed to start",
                reason_code="external_server_start_failed",
                server_id=server.id,
            ) from exc
        return transport, tools

    def _commit_started_runtime_unlocked(
        self,
        server: ExternalServerDefinition,
        transport: ExternalFederationTransport,
        tools: list[ExternalToolDefinition],
    ) -> None:
        self._servers[server.id] = server.model_copy(deep=True)
        self._transports[server.id] = transport
        self._last_errors[server.id] = None
        self._replace_server_tools(server.id, tools)

    async def _refresh_server_tools(
        self,
        server_id: str,
        transport: ExternalFederationTransport,
    ) -> bool:
        try:
            tools = await transport.list_tools()
        except Exception as exc:  # noqa: BLE001 - discovery adapters define their own errors.
            async with self._lock:
                if self._transports.get(server_id) is transport:
                    self._last_errors[server_id] = self._exception_summary(exc)
                    self._clear_server_tools(server_id)
            await self._audit_best_effort(
                "external_server.discovery",
                payload={
                    "reason_code": "external_server_discovery_failed",
                    "server_id": server_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
                target_type="external_server",
                target_id=server_id,
            )
            return False

        async with self._lock:
            if self._transports.get(server_id) is not transport:
                return True
            self._last_errors[server_id] = None
            self._replace_server_tools(server_id, tools)
        await self._audit(
            "external_server.discovery",
            payload={
                "reason_code": "external_server_discovered",
                "server_id": server_id,
                "tool_count": len(tools),
            },
            target_type="external_server",
            target_id=server_id,
        )
        return True

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

    async def _installer_status(
        self,
        server: ExternalServerDefinition,
    ) -> dict[str, Any]:
        """Return sanitized best-effort installer status for one server row."""

        try:
            payload = await asyncio.wait_for(
                self._installer.get_status(server.model_copy(deep=True)),
                timeout=self._installer_status_timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.opt(exception=True).error(
                "External installer status timed out server_id={!r}",
                server.id,
            )
            return {
                "available": False,
                "reason_code": "external_server_installer_status_timeout",
                "server_id": server.id,
                "error_type": "TimeoutError",
            }
        except Exception as exc:  # noqa: BLE001 - installer adapters are optional.
            traceback_frames = self._traceback_frames(exc)
            logger.error(
                "External installer status failed server_id={!r} error_type={!r} traceback_frames={!r}",
                server.id,
                type(exc).__name__,
                traceback_frames,
            )
            return {
                "available": False,
                "reason_code": "external_server_installer_status_unavailable",
                "server_id": server.id,
                "error_type": type(exc).__name__,
            }
        return self._installer_payload(
            payload,
            server=server,
            fallback_reason_code="external_server_installer_not_configured",
            include_ok=False,
        )

    async def _installer_operation_payload(
        self,
        operation: str,
        *,
        server: ExternalServerDefinition,
        callback: Callable[[], Any],
        fallback_reason_code: str,
        failure_reason_code: str,
    ) -> dict[str, Any]:
        """Run one installer operation and expose only sanitized public metadata."""

        try:
            payload = await callback()
        except Exception as exc:  # noqa: BLE001 - installer adapters define failures.
            traceback_frames = self._traceback_frames(exc)
            logger.error(
                "External installer operation failed operation={!r} server_id={!r} error_type={!r} traceback_frames={!r}",
                operation,
                server.id,
                type(exc).__name__,
                traceback_frames,
            )
            raise GatewayExternalRuntimeError(
                f"External server {operation} failed",
                reason_code=failure_reason_code,
                server_id=server.id,
            ) from self._sanitized_exception_cause(exc)
        return self._installer_payload(
            payload,
            server=server,
            fallback_reason_code=fallback_reason_code,
        )

    @classmethod
    def _installer_payload(
        cls,
        payload: dict[str, Any],
        *,
        server: ExternalServerDefinition,
        fallback_reason_code: str,
        include_ok: bool = True,
    ) -> dict[str, Any]:
        """Normalize an installer adapter payload into the public gateway shape."""

        result = cls._sanitize_installer_payload(payload or {})
        if include_ok:
            result.setdefault("ok", False)
        else:
            result.pop("ok", None)
        result.setdefault("available", False)
        result.setdefault("reason_code", fallback_reason_code)
        result["server_id"] = server.id
        return result

    @classmethod
    def _sanitize_installer_payload(cls, payload: dict[str, Any]) -> dict[str, Any]:
        """Return only allowlisted installer fields with nested sensitive keys removed."""

        sanitized: dict[str, Any] = {}
        for key, value in payload.items():
            normalized_key = str(key)
            if normalized_key not in _INSTALLER_PUBLIC_KEYS:
                continue
            if cls._is_sensitive_installer_key(normalized_key):
                continue
            sanitized_value = cls._sanitize_installer_value(value)
            if sanitized_value is not _UNSAFE_INSTALLER_VALUE:
                sanitized[normalized_key] = sanitized_value
        return sanitized

    @classmethod
    def _sanitize_installer_value(cls, value: Any) -> Any:
        """Recursively sanitize installer payload values into JSON-safe public data."""

        if isinstance(value, dict):
            sanitized: dict[str, Any] = {}
            for key, child in value.items():
                normalized_key = str(key)
                if cls._is_sensitive_installer_key(normalized_key):
                    continue
                sanitized_child = cls._sanitize_installer_value(child)
                if sanitized_child is not _UNSAFE_INSTALLER_VALUE:
                    sanitized[normalized_key] = sanitized_child
            return sanitized
        if isinstance(value, list):
            sanitized_items = [
                cls._sanitize_installer_value(item)
                for item in value
            ]
            return [
                item
                for item in sanitized_items
                if item is not _UNSAFE_INSTALLER_VALUE
            ]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return _UNSAFE_INSTALLER_VALUE

    @staticmethod
    def _is_sensitive_installer_key(key: str) -> bool:
        """Return true when a key name suggests credentials or command material."""

        lowered = key.strip().lower()
        return any(token in lowered for token in _INSTALLER_SENSITIVE_KEY_TOKENS)

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
        if self._missing_credential_slots(
            server=server,
            effective_policy=effective_policy,
        ):
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

    async def _resolve_runtime_auth(
        self,
        *,
        server: ExternalServerDefinition,
        virtual_tool: VirtualExternalTool,
        effective_policy: Any,
        actor_id: str | None,
        context: Any,
    ) -> BrokeredExternalCredential | None:
        required_slots = self._required_credential_slots(server)
        if not required_slots:
            return None
        if self._credential_broker is None:
            raise GatewayExternalRuntimeError(
                f"Credential broker is unavailable for external server '{server.id}'",
                reason_code="credential_broker_unavailable",
                server_id=server.id,
            )
        missing_slots = self._missing_credential_slots(
            server=server,
            effective_policy=effective_policy,
        )
        if missing_slots:
            raise FederationPolicyDenied(
                "required_credential_grant_missing",
                (
                    f"External server '{server.id}' requires credential grants for "
                    f"{', '.join(missing_slots)}"
                ),
                payload={
                    "reason_code": "required_credential_grant_missing",
                    "server_id": server.id,
                    "credential_slots": list(missing_slots),
                },
            )
        broker_result = await self._call_credential_broker(
            server=server,
            virtual_tool=virtual_tool,
            credential_slots=required_slots,
            effective_policy=effective_policy,
            actor_id=actor_id,
            context=context,
        )
        if broker_result is None:
            raise FederationPolicyDenied(
                "required_credential_grant_missing",
                f"Credential broker returned no grant for external server '{server.id}'",
                payload={
                    "reason_code": "required_credential_grant_missing",
                    "server_id": server.id,
                    "credential_slots": list(required_slots),
                },
            )
        return broker_result

    async def _call_credential_broker(
        self,
        *,
        server: ExternalServerDefinition,
        virtual_tool: VirtualExternalTool,
        credential_slots: list[str],
        effective_policy: Any,
        actor_id: str | None,
        context: Any,
    ) -> BrokeredExternalCredential | None:
        broker = self._credential_broker
        if broker is None:
            return None
        kwargs = {
            "server": server.model_copy(deep=True),
            "virtual_tool": virtual_tool.copy(),
            "credential_slots": list(credential_slots),
            "effective_policy": effective_policy,
            "actor_id": actor_id,
            "context": context,
        }
        if hasattr(broker, "resolve_external_credential"):
            result = broker.resolve_external_credential(**kwargs)
        else:
            result = broker(**kwargs)
        resolved = await result if inspect.isawaitable(result) else result
        if resolved is None:
            return None
        if isinstance(resolved, BrokeredExternalCredential):
            return resolved.copy()
        if isinstance(resolved, dict):
            return BrokeredExternalCredential(
                headers=dict(resolved.get("headers") or {}),
                env=dict(resolved.get("env") or {}),
                metadata=deepcopy(resolved.get("metadata") or {}),
            )
        raise TypeError("credential broker must return BrokeredExternalCredential, dict, or None")

    @staticmethod
    def _required_credential_slots(server: ExternalServerDefinition) -> list[str]:
        return sorted(
            {
                str(slot).strip()
                for slot in server.credential_slots
                if str(slot).strip()
            }
        )

    def _missing_credential_slots(
        self,
        *,
        server: ExternalServerDefinition,
        effective_policy: Any,
    ) -> list[str]:
        required_slots = set(self._required_credential_slots(server))
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
    def _policy_dicts(effective_policy: Any, field_name: str) -> list[dict[str, Any]]:
        values = GatewayExternalRuntimeManager._policy_value(effective_policy, field_name, [])
        if isinstance(values, dict):
            return [dict(values)]
        if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
            return []
        return [
            dict(value)
            for value in values
            if isinstance(value, dict)
        ]

    @staticmethod
    def _policy_list(effective_policy: Any, field_name: str) -> list[str]:
        values = GatewayExternalRuntimeManager._policy_value(effective_policy, field_name, [])
        if isinstance(values, str):
            return [values] if values.strip() else []
        if not isinstance(values, Iterable) or isinstance(values, (bytes, dict)):
            return []
        return [
            str(value)
            for value in values
            if str(value).strip()
        ]

    @staticmethod
    def _policy_profile_id(effective_policy: Any) -> str | None:
        value = GatewayExternalRuntimeManager._policy_value(
            effective_policy,
            "profile_id",
            None,
        )
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
        credential_injection: dict[str, Any] | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "reason_code": reason_code,
            "server_id": virtual_tool.server_id,
            "virtual_tool_name": virtual_tool.virtual_name,
            "upstream_tool_name": virtual_tool.upstream_tool_name,
        }
        if credential_injection is not None:
            payload["credential_injection"] = deepcopy(credential_injection)
        if extra_payload is not None:
            payload.update(deepcopy(extra_payload))
        await self._audit(
            event_type,
            actor_id=actor_id,
            profile_id=profile_id,
            target_type="external_tool",
            target_id=virtual_tool.virtual_name,
            payload=payload,
        )

    @staticmethod
    def _public_runtime_auth_metadata(
        runtime_auth: BrokeredExternalCredential,
    ) -> dict[str, Any]:
        metadata = runtime_auth.metadata or {}
        if not isinstance(metadata, dict):
            return {}
        public: dict[str, Any] = {}
        for key in ("credential_mode", "credential_source"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                public[key] = value
        return public

    @staticmethod
    def _summarize_runtime_auth(runtime_auth: BrokeredExternalCredential) -> dict[str, Any]:
        return {
            "headers": sorted(str(name) for name in (runtime_auth.headers or {})),
            "env": sorted(str(name) for name in (runtime_auth.env or {})),
        }

    @staticmethod
    def _federated_result(
        virtual_tool: VirtualExternalTool,
        result: ExternalToolCallResult,
    ) -> FederatedToolResult:
        return FederatedToolResult(
            content=deepcopy(result.content),
            server_id=virtual_tool.server_id,
            upstream_tool_name=virtual_tool.upstream_tool_name,
            virtual_tool_name=virtual_tool.virtual_name,
            is_error=result.is_error,
            metadata=deepcopy(result.metadata),
        )

    @classmethod
    def _definition_changed(
        cls,
        current: ExternalServerDefinition,
        latest: ExternalServerDefinition,
    ) -> bool:
        return cls._runtime_signature(current) != cls._runtime_signature(latest)

    @staticmethod
    def _runtime_signature(server: ExternalServerDefinition) -> tuple[Any, ...]:
        return (
            server.transport,
            tuple(server.command),
            server.url,
            server.cwd,
            tuple(server.env_allowlist),
            tuple(server.credential_slots),
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
        except Exception:  # noqa: BLE001 - audit failures must not block cleanup.
            return

    @staticmethod
    async def _close_best_effort(transport: ExternalFederationTransport) -> None:
        try:
            await transport.close()
        except Exception:  # noqa: BLE001 - adapter close failures are best-effort here.
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
    def _traceback_frames(exc: BaseException) -> list[dict[str, Any]]:
        """Return traceback frame locations without exception messages or source text."""

        return [
            {
                "function": frame.name,
                "line": frame.lineno,
                "path": frame.filename,
            }
            for frame in traceback.extract_tb(exc.__traceback__)
        ]

    @staticmethod
    def _sanitized_exception_cause(exc: BaseException) -> RuntimeError:
        """Return a chainable cause that preserves error type without secret text."""

        return RuntimeError(type(exc).__name__)

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
