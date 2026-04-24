"""Manager scaffold for external MCP server federation."""

from __future__ import annotations

import inspect
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from loguru import logger

from .config_schema import ExternalMCPServerConfig, load_external_server_registry
from .transports import (
    BrokeredExternalCredential,
    ExternalMCPTransportAdapter,
    ExternalToolCallResult,
    adapter_supports_runtime_auth,
    build_transport_adapter,
)


@dataclass(slots=True)
class VirtualExternalTool:
    """External tool exposed through a namespaced virtual name."""

    virtual_name: str
    server_id: str
    upstream_tool_name: str
    description: str
    input_schema: dict[str, Any]
    metadata: dict[str, Any]
    is_write: bool = False


@dataclass(slots=True)
class ExternalServerTelemetry:
    """Per-server operational counters and latency snapshots."""

    connect_attempts: int = 0
    connect_successes: int = 0
    connect_failures: int = 0
    discovery_attempts: int = 0
    discovery_successes: int = 0
    discovery_failures: int = 0
    call_attempts: int = 0
    call_successes: int = 0
    call_failures: int = 0
    call_timeouts: int = 0
    call_upstream_errors: int = 0
    policy_denials: int = 0
    last_discovered_tool_count: int = 0
    total_connect_latency_ms: float = 0.0
    total_discovery_latency_ms: float = 0.0
    total_call_latency_ms: float = 0.0
    last_connect_latency_ms: Optional[float] = None
    last_discovery_latency_ms: Optional[float] = None
    last_call_latency_ms: Optional[float] = None
    last_error: Optional[str] = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "connect_attempts": self.connect_attempts,
            "connect_successes": self.connect_successes,
            "connect_failures": self.connect_failures,
            "discovery_attempts": self.discovery_attempts,
            "discovery_successes": self.discovery_successes,
            "discovery_failures": self.discovery_failures,
            "call_attempts": self.call_attempts,
            "call_successes": self.call_successes,
            "call_failures": self.call_failures,
            "call_timeouts": self.call_timeouts,
            "call_upstream_errors": self.call_upstream_errors,
            "policy_denials": self.policy_denials,
            "last_discovered_tool_count": self.last_discovered_tool_count,
            "last_connect_latency_ms": self.last_connect_latency_ms,
            "last_discovery_latency_ms": self.last_discovery_latency_ms,
            "last_call_latency_ms": self.last_call_latency_ms,
            "avg_connect_latency_ms": (
                round(self.total_connect_latency_ms / self.connect_attempts, 3)
                if self.connect_attempts
                else None
            ),
            "avg_discovery_latency_ms": (
                round(self.total_discovery_latency_ms / self.discovery_attempts, 3)
                if self.discovery_attempts
                else None
            ),
            "avg_call_latency_ms": (
                round(self.total_call_latency_ms / self.call_attempts, 3)
                if self.call_attempts
                else None
            ),
            "last_error": self.last_error,
        }


class ExternalServerManager:
    """Lifecycle and routing manager for external MCP federation.

    This manager is intentionally conservative: discovery failures are isolated to
    the impacted external server and do not crash MCP Unified startup.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config_path = config_path
        self._server_loader: Callable[[], Awaitable[list[ExternalMCPServerConfig]] | list[ExternalMCPServerConfig]] | None = None
        self._servers: dict[str, ExternalMCPServerConfig] = {}
        self._adapters: dict[str, ExternalMCPTransportAdapter] = {}
        self._virtual_tools: dict[str, VirtualExternalTool] = {}
        self._discovery_errors: dict[str, str] = {}
        self._telemetry: dict[str, ExternalServerTelemetry] = {}
        self._initialized = False
        self._credential_broker: Callable[..., Awaitable[BrokeredExternalCredential | None] | BrokeredExternalCredential | None] | None = None

    def with_server_loader(
        self,
        server_loader: Callable[[], Awaitable[list[ExternalMCPServerConfig]] | list[ExternalMCPServerConfig]],
    ) -> "ExternalServerManager":
        self._server_loader = server_loader
        return self

    def with_credential_broker(
        self,
        credential_broker: Callable[..., Awaitable[BrokeredExternalCredential | None] | BrokeredExternalCredential | None],
    ) -> "ExternalServerManager":
        self._credential_broker = credential_broker
        return self

    @property
    def initialized(self) -> bool:
        return self._initialized

    async def initialize(self) -> None:
        """Load config, construct adapters, and attempt initial discovery."""

        if self._server_loader is not None:
            loaded = self._server_loader()
            servers = await loaded if inspect.isawaitable(loaded) else loaded
        else:
            cfg = load_external_server_registry(self.config_path)
            servers = list(cfg.servers)
        self._servers = {s.id: s for s in servers if s.enabled}
        self._adapters = {}
        self._virtual_tools = {}
        self._discovery_errors = {}
        self._telemetry = {
            server.id: ExternalServerTelemetry()
            for server in self._servers.values()
        }

        for server in self._servers.values():
            adapter = build_transport_adapter(server)
            self._adapters[server.id] = adapter
            try:
                await self._connect_server(server.id)
                await self._refresh_server_tools(server.id)
                self._discovery_errors.pop(server.id, None)
            except Exception as exc:
                self._discovery_errors[server.id] = str(exc)
                self._clear_server_tools(server.id)
                logger.warning(
                    "External MCP server '{}' failed initialization/discovery: {}",
                    server.id,
                    exc,
                )

        self._initialized = True

    async def shutdown(self) -> None:
        """Close all external transport adapters."""

        for server_id, adapter in list(self._adapters.items()):
            try:
                await adapter.close()
            except Exception as exc:
                logger.warning(f"External MCP adapter close failed for {server_id}: {exc}")
        self._adapters = {}
        self._virtual_tools = {}
        self._discovery_errors = {}
        self._telemetry = {}
        self._initialized = False

    async def refresh_discovery(self, server_id: Optional[str] = None) -> dict[str, Any]:
        """Refresh virtual tool cache for one server or all configured servers."""

        target_ids = [server_id] if server_id else sorted(self._adapters.keys())
        refreshed = 0
        errors: dict[str, str] = {}

        for sid in target_ids:
            if sid not in self._adapters:
                errors[sid] = "unknown_server"
                continue
            try:
                await self._refresh_server_tools(sid)
                refreshed += 1
                self._discovery_errors.pop(sid, None)
            except Exception as exc:
                errors[sid] = str(exc)
                self._discovery_errors[sid] = str(exc)
                self._clear_server_tools(sid)

        return {
            "refreshed_servers": refreshed,
            "total_servers": len(target_ids),
            "virtual_tools": len(self._virtual_tools),
            "errors": errors,
        }

    async def list_servers(self) -> list[dict[str, Any]]:
        """Return summarized status for configured external servers."""

        rows: list[dict[str, Any]] = []
        for server_id in sorted(self._servers.keys()):
            server = self._servers[server_id]
            adapter = self._adapters.get(server_id)
            checks = {"configured": True, "connected": False}
            if adapter is not None:
                try:
                    checks = await adapter.health_check()
                except Exception as exc:
                    checks = {"configured": True, "connected": False, "error": True}
                    self._discovery_errors[server_id] = str(exc)

            connected = bool(checks.get("connected"))
            discovery_ok = server_id not in self._discovery_errors
            if connected and discovery_ok:
                status = "healthy"
            elif connected or discovery_ok:
                status = "degraded"
            else:
                status = "unhealthy"

            rows.append(
                {
                    "id": server.id,
                    "name": server.name,
                    "transport": server.transport.value,
                    "tool_count": self._count_tools_for_server(server.id),
                    "status": status,
                    "discovery_ok": discovery_ok,
                    "checks": checks,
                    "last_error": self._discovery_errors.get(server.id),
                    "telemetry": self._snapshot_telemetry(server.id),
                }
            )
        return rows

    def list_virtual_tools(self) -> list[VirtualExternalTool]:
        """Return all currently discovered virtual tools."""

        return [self._virtual_tools[name] for name in sorted(self._virtual_tools.keys())]

    async def execute_virtual_tool(
        self,
        virtual_tool_name: str,
        arguments: dict[str, Any],
        context: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Route a namespaced virtual tool execution to its external adapter."""

        virtual_tool = self._virtual_tools.get(virtual_tool_name)
        if virtual_tool is None:
            raise ValueError(f"Unknown external virtual tool '{virtual_tool_name}'")

        server_id = virtual_tool.server_id
        upstream_tool_name = virtual_tool.upstream_tool_name
        adapter = self._adapters.get(server_id)
        if adapter is None:
            raise ValueError(f"Unknown external server '{server_id}'")

        server_cfg = self._servers[server_id]
        if not server_cfg.policy.allows_tool(upstream_tool_name):
            self._mark_policy_denial(
                server_id,
                (
                    f"External tool '{upstream_tool_name}' is blocked by local policy "
                    f"for server '{server_id}'"
                ),
            )
            raise PermissionError(
                f"External tool '{upstream_tool_name}' is blocked by local policy for server '{server_id}'"
            )

        call_args = dict(arguments or {})
        if virtual_tool.is_write:
            if not server_cfg.policy.allow_writes:
                self._mark_policy_denial(
                    server_id,
                    (
                        f"External write tool '{upstream_tool_name}' is disabled by local policy "
                        f"for server '{server_id}'"
                    ),
                )
                raise PermissionError(
                    f"External write tool '{upstream_tool_name}' is disabled by local policy for server '{server_id}'"
                )
            if server_cfg.policy.require_write_confirmation and not bool(call_args.get("__confirm_write")):
                self._mark_policy_denial(
                    server_id,
                    "Write confirmation required. Re-run with '__confirm_write': true.",
                )
                raise PermissionError(
                    "Write confirmation required. Re-run with '__confirm_write': true."
                )
            call_args.pop("__confirm_write", None)

        result = await self._call_external_tool(
            server_id=server_id,
            adapter=adapter,
            upstream_tool_name=upstream_tool_name,
            call_args=call_args,
            context=context,
            runtime_auth=await self._resolve_runtime_auth(
                server_id=server_id,
                upstream_tool_name=upstream_tool_name,
                call_args=call_args,
                context=context,
            ),
        )
        metadata = dict(result.metadata or {})
        return {
            "content": result.content,
            "is_error": result.is_error,
            "server_id": server_id,
            "upstream_tool": upstream_tool_name,
            "metadata": metadata,
        }

    @staticmethod
    def parse_virtual_tool_name(virtual_tool_name: str) -> tuple[str, str]:
        """Parse `ext.<server_id>.<tool_name>` into `(server_id, tool_name)` parts."""

        if not virtual_tool_name.startswith("ext."):
            raise ValueError("External tool names must start with 'ext.'")

        parts = virtual_tool_name.split(".", 2)
        if len(parts) != 3 or not parts[1] or not parts[2]:
            raise ValueError("External tool name must match 'ext.<server_id>.<tool_name>'")

        return parts[1], parts[2]

    async def _refresh_server_tools(self, server_id: str) -> None:
        """Refresh discovery cache for a single server."""

        telemetry = self._get_telemetry(server_id)
        telemetry.discovery_attempts += 1
        started_at = time.perf_counter()
        adapter = self._adapters[server_id]
        try:
            tools = await adapter.list_tools()
            server_cfg = self._servers[server_id]

            server_tools: dict[str, VirtualExternalTool] = {}

            for tool in tools:
                if not server_cfg.policy.allows_tool(tool.name):
                    continue
                virtual_name = self._virtual_tool_name(server_id, tool.name)
                tool_metadata = dict(tool.metadata or {})
                server_tools[virtual_name] = VirtualExternalTool(
                    virtual_name=virtual_name,
                    server_id=server_id,
                    upstream_tool_name=tool.name,
                    description=tool.description,
                    input_schema=tool.input_schema,
                    metadata=tool_metadata,
                    is_write=self._is_write_tool(tool.name, tool_metadata),
                )

            # Replace only this server's tools while preserving other caches.
            self._clear_server_tools(server_id)
            self._virtual_tools.update(server_tools)
            telemetry.discovery_successes += 1
            telemetry.last_discovered_tool_count = len(server_tools)
        except Exception as exc:
            telemetry.discovery_failures += 1
            telemetry.last_error = str(exc)
            raise
        finally:
            latency_ms = self._elapsed_ms(started_at)
            telemetry.last_discovery_latency_ms = latency_ms
            telemetry.total_discovery_latency_ms += latency_ms

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

    def _get_telemetry(self, server_id: str) -> ExternalServerTelemetry:
        telemetry = self._telemetry.get(server_id)
        if telemetry is None:
            telemetry = ExternalServerTelemetry()
            self._telemetry[server_id] = telemetry
        return telemetry

    def _snapshot_telemetry(self, server_id: str) -> dict[str, Any]:
        return self._get_telemetry(server_id).snapshot()

    async def _connect_server(self, server_id: str) -> None:
        telemetry = self._get_telemetry(server_id)
        telemetry.connect_attempts += 1
        started_at = time.perf_counter()
        try:
            await self._adapters[server_id].connect()
            telemetry.connect_successes += 1
        except Exception as exc:
            telemetry.connect_failures += 1
            telemetry.last_error = str(exc)
            raise
        finally:
            latency_ms = self._elapsed_ms(started_at)
            telemetry.last_connect_latency_ms = latency_ms
            telemetry.total_connect_latency_ms += latency_ms

    async def _call_external_tool(
        self,
        *,
        server_id: str,
        adapter: ExternalMCPTransportAdapter,
        upstream_tool_name: str,
        call_args: dict[str, Any],
        context: Optional[Any],
        runtime_auth: BrokeredExternalCredential | None,
    ) -> ExternalToolCallResult:
        telemetry = self._get_telemetry(server_id)
        telemetry.call_attempts += 1
        started_at = time.perf_counter()
        try:
            call_kwargs: dict[str, Any] = {"context": context}
            supports_runtime_auth = adapter_supports_runtime_auth(adapter)
            if supports_runtime_auth:
                call_kwargs["runtime_auth"] = runtime_auth
            result = await adapter.call_tool(
                upstream_tool_name,
                call_args,
                **call_kwargs,
            )
            if runtime_auth is not None and supports_runtime_auth:
                metadata = dict(result.metadata or {})
                metadata.update(self._public_runtime_auth_metadata(runtime_auth))
                metadata["credential_injection"] = self._summarize_runtime_auth(runtime_auth)
                result.metadata = metadata
            telemetry.call_successes += 1
            if result.is_error:
                telemetry.call_upstream_errors += 1
                error_text = self._extract_error_text(result)
                if error_text:
                    telemetry.last_error = error_text
            return result
        except TimeoutError as exc:
            telemetry.call_failures += 1
            telemetry.call_timeouts += 1
            telemetry.last_error = str(exc)
            raise
        except Exception as exc:
            telemetry.call_failures += 1
            telemetry.last_error = str(exc)
            raise
        finally:
            latency_ms = self._elapsed_ms(started_at)
            telemetry.last_call_latency_ms = latency_ms
            telemetry.total_call_latency_ms += latency_ms

    def _mark_policy_denial(self, server_id: str, message: str) -> None:
        telemetry = self._get_telemetry(server_id)
        telemetry.policy_denials += 1
        telemetry.last_error = message

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
    def _extract_error_text(result: ExternalToolCallResult) -> Optional[str]:
        content = result.content
        if isinstance(content, str):
            text = content.strip()
            return text or None
        if isinstance(content, dict):
            text = content.get("text")
            if isinstance(text, str):
                stripped = text.strip()
                if stripped:
                    return stripped
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if isinstance(text, str):
                    stripped = text.strip()
                    if stripped:
                        return stripped
        return None

    async def _resolve_runtime_auth(
        self,
        *,
        server_id: str,
        upstream_tool_name: str,
        call_args: dict[str, Any],
        context: Optional[Any],
    ) -> BrokeredExternalCredential | None:
        if self._credential_broker is None:
            return None
        broker_result = self._credential_broker(
            server_id=server_id,
            tool_name=upstream_tool_name,
            arguments=dict(call_args or {}),
            context=context,
        )
        resolved = await broker_result if inspect.isawaitable(broker_result) else broker_result
        if resolved is None:
            return None
        if isinstance(resolved, BrokeredExternalCredential):
            return resolved
        if isinstance(resolved, dict):
            return BrokeredExternalCredential(
                headers=dict(resolved.get("headers") or {}),
                env=dict(resolved.get("env") or {}),
                metadata=dict(resolved.get("metadata") or {}),
            )
        raise TypeError("credential broker must return BrokeredExternalCredential, dict, or None")

    @staticmethod
    def _summarize_runtime_auth(runtime_auth: BrokeredExternalCredential) -> dict[str, Any]:
        return {
            "headers": sorted(str(name) for name in runtime_auth.headers.keys()),
            "env": sorted(str(name) for name in runtime_auth.env.keys()),
        }

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        return round((time.perf_counter() - started_at) * 1000.0, 3)

    @staticmethod
    def _is_write_tool(tool_name: str, metadata: dict[str, Any]) -> bool:
        """Best-effort write classification for external tools."""

        annotations = metadata.get("annotations")
        if isinstance(annotations, dict):
            read_only_hint = annotations.get("readOnlyHint")
            if isinstance(read_only_hint, bool):
                return not read_only_hint

        for key in ("read_only", "readOnly", "is_read_only"):
            value = metadata.get(key)
            if isinstance(value, bool):
                return not value

        category = str(metadata.get("category") or "").strip().lower()
        if category in {"read", "discovery", "search"}:
            return False
        if category in {"ingestion", "management", "write", "mutation", "admin"}:
            return True

        lowered = str(tool_name).lower()
        tokens = [token for token in re.split(r"[^a-z0-9]+", lowered) if token]
        write_tokens = {
            "create",
            "update",
            "delete",
            "remove",
            "write",
            "set",
            "insert",
            "upsert",
            "patch",
            "put",
            "post",
            "ingest",
            "import",
            "exec",
            "execute",
            "run",
        }
        return any(token in write_tokens for token in tokens)


__all__ = ["ExternalServerManager", "VirtualExternalTool"]
