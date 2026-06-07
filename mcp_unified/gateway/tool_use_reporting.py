"""Gateway runtime wrapper that records metadata-only tool-use events."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from mcp_unified.tool_use_reporting import (
    NoopToolUseRecorder,
    ToolUseEvent,
    ToolUseRecorder,
    ToolUseStatus,
    classify_tool_use_exception,
    extract_safe_context_dimensions,
    record_tool_use_safely,
    sanitize_safe_id,
)

from .runtime import GatewayPolicyDenied, GatewayRequestContext, GatewayRuntime

_OBSERVED_METADATA_KEY = "mcp_tool_use_observed"
_OUTER_SURFACE_METADATA_KEY = "mcp_tool_use_outer_surface"
_BRIDGE_TOOL_NAME_METADATA_KEY = "mcp_tool_use_bridge_tool_name"
_REQUESTED_TOOL_ID_METADATA_KEY = "mcp_tool_use_requested_tool_id"
_EFFECTIVE_TOOL_NAME_METADATA_KEY = "mcp_tool_use_effective_tool_name"
_SOURCE_KIND_METADATA_KEY = "mcp_tool_use_source_kind"
_SUPPORTED_SOURCE_KINDS = {"local", "external", "federated", "bridge"}
_TOOL_USE_RESULT_METADATA_KEYS = (
    "mcp_tool_use",
    "tool_use_reporting",
    "toolUseReporting",
)


class ToolUseReportingGatewayRuntime:
    """Wrap a gateway runtime and record metadata-only tool-use attempts."""

    def __init__(
        self,
        runtime: GatewayRuntime,
        *,
        recorder: ToolUseRecorder | None = None,
        write_timeout_seconds: float | None = 2.0,
    ) -> None:
        self._runtime = runtime
        self._recorder = recorder if recorder is not None else NoopToolUseRecorder()
        self._write_timeout_seconds = write_timeout_seconds

    @property
    def recorder(self) -> ToolUseRecorder:
        """Return the configured recorder for tests and host inspection."""

        return self._recorder

    @property
    def name(self) -> str:
        """Return the wrapped runtime name."""

        return str(getattr(self._runtime, "name", "mcp-unified-gateway"))

    @property
    def version(self) -> str:
        """Return the wrapped runtime version."""

        return str(getattr(self._runtime, "version", "0.1.0"))

    async def list_tools(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        """Delegate tool discovery unchanged."""

        return await self._runtime.list_tools(context)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        """Execute a tool call and record a sanitized metadata event."""

        if _metadata(context).get(_OBSERVED_METADATA_KEY) is True:
            return await self._runtime.call_tool(name, arguments, context)

        observed_context = _context_with_observed_marker(context)
        started_at = time.perf_counter()
        try:
            result = await self._runtime.call_tool(name, arguments, observed_context)
        except GatewayPolicyDenied as exc:
            await self._record_event(
                name,
                observed_context,
                status="denied",
                reason_code=exc.reason_code,
                execution_origin="denied",
                duration_ms=_duration_ms(started_at),
            )
            raise
        except Exception as exc:
            status, reason_code = classify_tool_use_exception(exc)
            await self._record_event(
                name,
                observed_context,
                status=status,
                reason_code=reason_code,
                execution_origin=_execution_origin_for_error(status),
                duration_ms=_duration_ms(started_at),
            )
            raise

        await self._record_event(
            name,
            observed_context,
            status="success",
            reason_code=None,
            execution_origin="executed",
            duration_ms=_duration_ms(started_at),
            result=result,
        )
        return result

    async def list_resources(
        self,
        context: GatewayRequestContext,
    ) -> list[dict[str, Any]]:
        """Delegate resource discovery unchanged."""

        return await self._runtime.list_resources(context)

    async def read_resource(
        self,
        uri: str,
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        """Delegate resource reads unchanged."""

        return await self._runtime.read_resource(uri, context)

    async def list_prompts(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        """Delegate prompt discovery unchanged."""

        return await self._runtime.list_prompts(context)

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        """Delegate prompt reads unchanged."""

        return await self._runtime.get_prompt(name, arguments, context)

    async def list_modules(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        """Delegate module discovery unchanged."""

        return await self._runtime.list_modules(context)

    async def get_modules_health(self, context: GatewayRequestContext) -> dict[str, Any]:
        """Delegate module health unchanged."""

        return await self._runtime.get_modules_health(context)

    async def _record_event(
        self,
        name: str,
        context: GatewayRequestContext,
        *,
        status: ToolUseStatus,
        reason_code: str | None,
        execution_origin: str,
        duration_ms: float,
        result: Mapping[str, Any] | None = None,
    ) -> None:
        metadata = _metadata(context)
        result_metadata = _result_tool_use_metadata(result)
        event = ToolUseEvent(
            runtime_surface="gateway",
            execution_origin=execution_origin,
            requested_tool_name=_requested_tool_name(name, metadata, result_metadata),
            effective_tool_name=_effective_tool_name(name, metadata, result_metadata),
            status=status,
            reason_code=reason_code,
            duration_ms=duration_ms,
            source_kind=_source_kind(metadata, result_metadata),
            capture_ref=_requested_bridge_tool_id(metadata, result_metadata),
            **extract_safe_context_dimensions(metadata),
        )
        await record_tool_use_safely(
            self._recorder,
            event,
            timeout_seconds=self._write_timeout_seconds,
        )


def _metadata(context: GatewayRequestContext) -> dict[str, Any]:
    """Return context metadata when present, otherwise an empty mapping."""

    return context.metadata if isinstance(context.metadata, dict) else {}


def _context_with_observed_marker(
    context: GatewayRequestContext,
) -> GatewayRequestContext:
    """Return a caller-owned context copy marked as already observed."""

    metadata = dict(_metadata(context))
    metadata[_OBSERVED_METADATA_KEY] = True
    metadata[_OUTER_SURFACE_METADATA_KEY] = "gateway"
    return GatewayRequestContext(
        request_id=context.request_id,
        client_id=context.client_id,
        user_id=context.user_id,
        metadata=metadata,
    )


def _duration_ms(started_at: float) -> float:
    """Return elapsed monotonic time in milliseconds."""

    return max(0.0, (time.perf_counter() - started_at) * 1000.0)


def _execution_origin_for_error(status: ToolUseStatus) -> str:
    """Return a conservative execution-origin label for failed gateway calls."""

    if status in {"denied", "approval_required"}:
        return "denied"
    if status in {"invalid_params", "rate_limited", "unavailable"}:
        return "failed_before_execution"
    return "executed"


def _result_tool_use_metadata(result: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return safe tool-use metadata embedded in a result payload, if present."""

    if not isinstance(result, Mapping):
        return {}
    for key in _TOOL_USE_RESULT_METADATA_KEYS:
        value = result.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    metadata = result.get("metadata")
    if isinstance(metadata, Mapping):
        nested = metadata.get("mcp_tool_use")
        if isinstance(nested, Mapping):
            return dict(nested)
    return {}


def _requested_tool_name(
    name: str,
    metadata: Mapping[str, Any],
    result_metadata: Mapping[str, Any],
) -> str:
    """Return the requested surface name for a tool-use event."""

    return (
        _safe_metadata_id(metadata, _BRIDGE_TOOL_NAME_METADATA_KEY)
        or _safe_metadata_id(result_metadata, _BRIDGE_TOOL_NAME_METADATA_KEY)
        or name
    )


def _effective_tool_name(
    name: str,
    metadata: Mapping[str, Any],
    result_metadata: Mapping[str, Any],
) -> str:
    """Return the effective backend tool name for a tool-use event."""

    return (
        _safe_metadata_id(metadata, _EFFECTIVE_TOOL_NAME_METADATA_KEY)
        or _safe_metadata_id(result_metadata, _EFFECTIVE_TOOL_NAME_METADATA_KEY)
        or _safe_metadata_id(result_metadata, "effective_tool_name")
        or name
    )


def _source_kind(
    metadata: Mapping[str, Any],
    result_metadata: Mapping[str, Any],
) -> str | None:
    """Return a supported source kind from safe side-channel metadata."""

    value = (
        _safe_metadata_id(metadata, _SOURCE_KIND_METADATA_KEY)
        or _safe_metadata_id(result_metadata, _SOURCE_KIND_METADATA_KEY)
        or _safe_metadata_id(result_metadata, "source_kind")
    )
    if value in _SUPPORTED_SOURCE_KINDS:
        return value
    return None


def _requested_bridge_tool_id(
    metadata: Mapping[str, Any],
    result_metadata: Mapping[str, Any],
) -> str | None:
    """Return the safe requested bridge tool id when a bridge call supplied one."""

    return (
        _safe_metadata_id(metadata, _REQUESTED_TOOL_ID_METADATA_KEY)
        or _safe_metadata_id(result_metadata, _REQUESTED_TOOL_ID_METADATA_KEY)
        or _safe_metadata_id(result_metadata, "requested_tool_id")
    )


def _safe_metadata_id(metadata: Mapping[str, Any], key: str) -> str | None:
    """Return a safe identifier from metadata."""

    return sanitize_safe_id(metadata.get(key), field=key)


__all__ = ["ToolUseReportingGatewayRuntime"]
