"""Upstream stdio transport for standalone external MCP servers."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from mcp_unified.federation.models import (
    BrokeredExternalCredential,
    ExternalToolCallResult,
    ExternalToolDefinition,
)
from mcp_unified.storage.models import ExternalServerDefinition

_MCP_PROTOCOL_VERSION = "2024-11-05"
_CLIENT_INFO = {"name": "mcp_unified_external_federation", "version": "0.1.0"}
_DEFAULT_CONNECT_TIMEOUT_S = 30.0
_DEFAULT_REQUEST_TIMEOUT_S = 30.0
_DEFAULT_CLOSE_TIMEOUT_S = 5.0
_DEFAULT_HEALTH_TIMEOUT_S = 1.0


class StdioExternalTransportError(RuntimeError):
    """Raised when a stdio transport operation fails without exposing secrets."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str,
        server_id: str | None = None,
        method: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{message} (reason_code={reason_code})")
        self.reason_code = reason_code
        self.server_id = server_id
        self.method = method
        self.details = deepcopy(details or {})


class StdioExternalTransport:
    """JSON-RPC over stdio transport for one configured upstream MCP server."""

    transport_name = "stdio"

    def __init__(
        self,
        server: ExternalServerDefinition,
        *,
        connect_timeout_s: float = _DEFAULT_CONNECT_TIMEOUT_S,
        request_timeout_s: float = _DEFAULT_REQUEST_TIMEOUT_S,
        close_timeout_s: float = _DEFAULT_CLOSE_TIMEOUT_S,
        health_timeout_s: float = _DEFAULT_HEALTH_TIMEOUT_S,
    ) -> None:
        self._server = server.model_copy(deep=True)
        self.server_id = self._server.id
        self._command = self._validate_command(self._server)
        self._cwd = self._resolve_cwd(self._server.cwd)
        self._connect_timeout_s = self._positive_timeout(connect_timeout_s, "connect_timeout_s")
        self._request_timeout_s = self._positive_timeout(request_timeout_s, "request_timeout_s")
        self._close_timeout_s = self._positive_timeout(close_timeout_s, "close_timeout_s")
        self._health_timeout_s = self._positive_timeout(health_timeout_s, "health_timeout_s")
        self._request_lock = asyncio.Lock()
        self._connect_lock = asyncio.Lock()
        self._proc: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_bytes = 0
        self._next_request_id = 1
        self._initialized = False

    async def connect(self) -> None:
        """Launch the stdio process and initialize the MCP session."""
        if self._initialized and self._is_running():
            return

        async with self._connect_lock:
            if self._initialized and self._is_running():
                return
            if self._proc is not None:
                await self._close_process_unlocked()

            try:
                self._proc = await asyncio.wait_for(
                    asyncio.create_subprocess_exec(
                        *self._command,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=self._cwd,
                        env=self._build_child_env(),
                    ),
                    timeout=self._connect_timeout_s,
                )
            except StdioExternalTransportError:
                raise
            except asyncio.TimeoutError as exc:
                raise self._error(
                    "External stdio process launch timed out",
                    reason_code="connect_timeout",
                ) from exc
            except Exception as exc:  # noqa: BLE001 - subprocess launch errors vary by platform.
                raise self._error(
                    "External stdio process launch failed",
                    reason_code="process_launch_failed",
                    details={"error_type": type(exc).__name__},
                ) from exc

            self._stderr_task = asyncio.create_task(self._drain_stderr())
            try:
                await self._request(
                    "initialize",
                    {
                        "protocolVersion": _MCP_PROTOCOL_VERSION,
                        "capabilities": {},
                        "clientInfo": _CLIENT_INFO,
                    },
                    timeout_s=self._connect_timeout_s,
                )
                await self._notify("notifications/initialized", {})
            except Exception:
                await self._close_process_unlocked()
                raise
            self._initialized = True

    async def close(self) -> None:
        """Terminate the subprocess and release stdio resources."""
        async with self._connect_lock:
            await self._close_process_unlocked()

    async def health_check(self) -> dict[str, bool]:
        """Return quick process and initialization health indicators."""
        checks = {
            "configured": True,
            "connected": self._is_running(),
            "initialized": self._initialized and self._is_running(),
            "spawns_process": True,
        }
        if not checks["connected"]:
            self._initialized = False
            checks["initialized"] = False
            return checks
        if not self._initialized:
            return checks

        try:
            await self._request("ping", {}, timeout_s=self._health_timeout_s)
        except StdioExternalTransportError:
            checks["connected"] = self._is_running()
            checks["initialized"] = False
            self._initialized = False
        return checks

    async def list_tools(self) -> list[ExternalToolDefinition]:
        """Discover and normalize upstream MCP tool definitions."""
        await self._ensure_connected()
        response = await self._request("tools/list", {})
        result = response.get("result") or {}
        if isinstance(result, dict):
            raw_tools = result.get("tools") or []
        elif isinstance(result, list):
            raw_tools = result
        else:
            raw_tools = []

        tools: list[ExternalToolDefinition] = []
        for item in raw_tools:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            description = item.get("description")
            if not isinstance(description, str):
                description = ""
            input_schema = item.get("inputSchema")
            if not isinstance(input_schema, dict):
                input_schema = {"type": "object"}
            metadata = item.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            tools.append(
                ExternalToolDefinition(
                    name=name,
                    description=description,
                    input_schema=deepcopy(input_schema),
                    metadata=deepcopy(metadata),
                )
            )
        return tools

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        context: Any = None,
        runtime_auth: BrokeredExternalCredential | None = None,
    ) -> ExternalToolCallResult:
        """Execute one upstream MCP tool call."""
        del context
        await self._ensure_connected()
        params: dict[str, Any] = {
            "name": tool_name,
            "arguments": deepcopy(arguments or {}),
        }
        runtime_auth_meta = self._runtime_auth_meta(runtime_auth)
        if runtime_auth_meta:
            params["_meta"] = {"mcp_unified_runtime_auth": runtime_auth_meta}

        response = await self._request("tools/call", params, raise_on_error=False)
        error = response.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if not isinstance(message, str) or not message:
                message = "External MCP tool call failed"
            return ExternalToolCallResult(
                content=[{"type": "text", "text": message}],
                is_error=True,
                metadata={
                    "server_id": self.server_id,
                    "tool_name": tool_name,
                    "reason_code": "upstream_error",
                },
            )

        result = response.get("result")
        if isinstance(result, dict):
            content = result.get("content") if "content" in result else result
            is_error = bool(result.get("isError"))
        else:
            content = result
            is_error = False
        return ExternalToolCallResult(
            content=deepcopy(content),
            is_error=is_error,
            metadata={"server_id": self.server_id, "tool_name": tool_name},
        )

    async def _ensure_connected(self) -> None:
        if not self._initialized or not self._is_running():
            await self.connect()

    async def _request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout_s: float | None = None,
        raise_on_error: bool = True,
    ) -> dict[str, Any]:
        async with self._request_lock:
            proc = self._proc
            if proc is None or proc.stdin is None or proc.stdout is None:
                raise self._error(
                    "External stdio process is not connected",
                    reason_code="not_connected",
                    method=method,
                )

            request_id = self._next_request_id
            self._next_request_id += 1
            payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": deepcopy(params or {}),
            }
            try:
                encoded = json.dumps(payload, separators=(",", ":"))
            except (TypeError, ValueError) as exc:
                raise self._error(
                    "External stdio request is not JSON serializable",
                    reason_code="invalid_request",
                    method=method,
                ) from exc
            if "\n" in encoded:
                raise self._error(
                    "External stdio request contains an invalid newline",
                    reason_code="invalid_request",
                    method=method,
                )

            try:
                proc.stdin.write((encoded + "\n").encode("utf-8"))
                await proc.stdin.drain()
                response = await asyncio.wait_for(
                    self._read_response(proc, request_id),
                    timeout=timeout_s or self._request_timeout_s,
                )
            except asyncio.TimeoutError as exc:
                self._schedule_process_cleanup(proc)
                raise self._error(
                    f"External stdio request timed out for method '{method}'",
                    reason_code="request_timeout",
                    method=method,
                ) from exc
            except (BrokenPipeError, ConnectionError) as exc:
                self._schedule_process_cleanup(proc)
                raise self._error(
                    "External stdio process connection closed",
                    reason_code="connection_closed",
                    method=method,
                ) from exc

        if response.get("error") and raise_on_error:
            raise self._error(
                f"External stdio request failed for method '{method}'",
                reason_code="upstream_error",
                method=method,
            )
        return response

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        async with self._request_lock:
            proc = self._proc
            if proc is None or proc.stdin is None:
                raise self._error(
                    "External stdio process is not connected",
                    reason_code="not_connected",
                    method=method,
                )
            payload = {
                "jsonrpc": "2.0",
                "method": method,
                "params": deepcopy(params or {}),
            }
            try:
                encoded = json.dumps(payload, separators=(",", ":"))
            except (TypeError, ValueError) as exc:
                raise self._error(
                    "External stdio notification is not JSON serializable",
                    reason_code="invalid_request",
                    method=method,
                ) from exc
            if "\n" in encoded:
                raise self._error(
                    "External stdio notification contains an invalid newline",
                    reason_code="invalid_request",
                    method=method,
                )
            try:
                proc.stdin.write((encoded + "\n").encode("utf-8"))
                await asyncio.wait_for(
                    proc.stdin.drain(),
                    timeout=self._request_timeout_s,
                )
            except asyncio.TimeoutError as exc:
                self._schedule_process_cleanup(proc)
                raise self._error(
                    f"External stdio notification timed out for method '{method}'",
                    reason_code="request_timeout",
                    method=method,
                ) from exc
            except (BrokenPipeError, ConnectionError) as exc:
                self._schedule_process_cleanup(proc)
                raise self._error(
                    "External stdio process connection closed",
                    reason_code="connection_closed",
                    method=method,
                ) from exc

    async def _read_response(
        self,
        proc: asyncio.subprocess.Process,
        request_id: int,
    ) -> dict[str, Any]:
        if proc.stdout is None:
            raise self._error(
                "External stdio process stdout is not available",
                reason_code="not_connected",
            )

        while True:
            line = await proc.stdout.readline()
            if not line:
                if self._proc is proc:
                    self._initialized = False
                raise self._error(
                    "External stdio process connection closed",
                    reason_code="connection_closed",
                )
            try:
                payload = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("id") != request_id:
                continue
            return payload

    async def _close_process_unlocked(self) -> None:
        proc = self._proc
        self._proc = None
        self._initialized = False

        stderr_task = self._stderr_task
        self._stderr_task = None
        await self._terminate_process(proc, stderr_task)

    def _schedule_process_cleanup(self, proc: asyncio.subprocess.Process) -> None:
        stderr_task = self._stderr_task if self._proc is proc else None
        if self._proc is proc:
            self._proc = None
            self._stderr_task = None
        self._initialized = False
        task = asyncio.create_task(self._terminate_process(proc, stderr_task))
        task.add_done_callback(self._discard_cleanup_result)

    async def _terminate_process(
        self,
        proc: asyncio.subprocess.Process | None,
        stderr_task: asyncio.Task[None] | None,
    ) -> None:
        if stderr_task is not None:
            stderr_task.cancel()

        if proc is not None:
            stdin = proc.stdin
            if stdin is not None:
                with contextlib.suppress(Exception):
                    stdin.close()
            if proc.returncode is None:
                try:
                    if proc.returncode is None:
                        proc.terminate()
                except ProcessLookupError:
                    pass
                try:
                    if proc.returncode is None:
                        await asyncio.wait_for(proc.wait(), timeout=self._close_timeout_s)
                except asyncio.TimeoutError:
                    try:
                        if proc.returncode is None:
                            proc.kill()
                    except ProcessLookupError:
                        pass
                    with contextlib.suppress(ProcessLookupError):
                        await proc.wait()
                except ProcessLookupError:
                    pass

        if stderr_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await stderr_task

    @staticmethod
    def _discard_cleanup_result(task: asyncio.Task[None]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception:  # noqa: BLE001 - background cleanup must not leak task warnings.
            return

    async def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        while True:
            line = await proc.stderr.readline()
            if not line:
                return
            self._stderr_bytes += len(line)

    def _build_child_env(self) -> dict[str, str]:
        child_env: dict[str, str] = {}
        for name in self._server.env_allowlist:
            env_name = str(name).strip()
            if env_name and env_name in os.environ:
                child_env[env_name] = os.environ[env_name]
        return child_env

    def _is_running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    def _error(
        self,
        message: str,
        *,
        reason_code: str,
        method: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> StdioExternalTransportError:
        return StdioExternalTransportError(
            message,
            reason_code=reason_code,
            server_id=self.server_id,
            method=method,
            details=details,
        )

    @classmethod
    def _validate_command(cls, server: ExternalServerDefinition) -> tuple[str, ...]:
        if server.transport != "stdio":
            raise StdioExternalTransportError(
                "External stdio transport received an unsupported server transport",
                reason_code="unsupported_transport",
                server_id=server.id,
            )
        command = tuple(str(part).strip() for part in server.command if str(part).strip())
        if not command:
            raise StdioExternalTransportError(
                "External stdio transport requires a non-empty command",
                reason_code="missing_command",
                server_id=server.id,
            )
        env_names = {str(name).strip() for name in server.env_allowlist}
        if cls._requires_path_lookup(command[0]) and "PATH" not in env_names:
            raise StdioExternalTransportError(
                "Command requires PATH; allowlist PATH or use an absolute executable path",
                reason_code="invalid_command",
                server_id=server.id,
            )
        return command

    @classmethod
    def _resolve_cwd(cls, cwd: str | None) -> str | None:
        if cwd is None:
            return None
        path = Path(cwd).expanduser()
        if not path.exists() or not path.is_dir():
            raise StdioExternalTransportError(
                "External stdio transport cwd is not an existing directory",
                reason_code="invalid_cwd",
            )
        return str(path)

    @classmethod
    def _positive_timeout(cls, value: float, field_name: str) -> float:
        try:
            timeout = float(value)
        except (TypeError, ValueError) as exc:
            raise StdioExternalTransportError(
                f"External stdio transport {field_name} must be a finite positive number",
                reason_code="invalid_timeout",
            ) from exc
        if timeout <= 0 or not math.isfinite(timeout):
            raise StdioExternalTransportError(
                f"External stdio transport {field_name} must be a finite positive number",
                reason_code="invalid_timeout",
            )
        return timeout

    @staticmethod
    def _requires_path_lookup(executable: str) -> bool:
        if Path(executable).is_absolute():
            return False
        return not any(separator and separator in executable for separator in (os.sep, os.altsep))

    @staticmethod
    def _runtime_auth_meta(
        runtime_auth: BrokeredExternalCredential | None,
    ) -> dict[str, Any]:
        if runtime_auth is None:
            return {}
        payload: dict[str, Any] = {}
        if runtime_auth.headers:
            payload["headers"] = dict(runtime_auth.headers)
        if runtime_auth.env:
            payload["env"] = dict(runtime_auth.env)
        if runtime_auth.metadata:
            payload["metadata"] = deepcopy(runtime_auth.metadata)
        return payload


def create_external_transport(server: ExternalServerDefinition) -> StdioExternalTransport:
    """Create a package-owned external transport for a supported server definition."""
    if server.transport == "stdio":
        return StdioExternalTransport(server)
    raise StdioExternalTransportError(
        "External server transport is not supported by the package factory",
        reason_code="unsupported_transport",
        server_id=server.id,
    )


__all__ = [
    "StdioExternalTransport",
    "StdioExternalTransportError",
    "create_external_transport",
]
