"""Gateway runtime adapter for active external MCP runtime tools."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from mcp_unified.federation.models import (
    FederatedToolResult,
    FederationPolicyDenied,
    VirtualExternalTool,
)

from .external_runtime import GatewayExternalRuntimeManager
from .profile_runtime import EFFECTIVE_POLICY_METADATA_KEY
from .runtime import GatewayPolicyDenied, GatewayRequestContext, GatewayRuntime


class ExternalRuntimeGatewayRuntime:
    """Expose active external runtime virtual tools through `GatewayRuntime`."""

    def __init__(
        self,
        *,
        external_runtime_manager: GatewayExternalRuntimeManager,
        base_runtime: GatewayRuntime | None = None,
        name: str | None = None,
        version: str | None = None,
    ) -> None:
        """Store runtime dependencies and resolve public identity metadata."""

        self._external_runtime_manager = external_runtime_manager
        self._base_runtime = base_runtime
        self.name = (
            name
            or str(getattr(base_runtime, "name", "") or "")
            or "mcp-unified-gateway"
        )
        self.version = (
            version
            or str(getattr(base_runtime, "version", "") or "")
            or "0.1.0"
        )

    async def list_tools(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        """Return local tools followed by active external virtual tools."""

        tools: list[dict[str, Any]] = []
        if self._base_runtime is not None:
            tools.extend(await self._base_runtime.list_tools(context))
        tools.extend(
            _virtual_tool_descriptor(virtual_tool)
            for virtual_tool in await self._external_runtime_manager.list_virtual_tools()
        )
        return tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        """Execute an external virtual tool or delegate to the base runtime."""

        if await self._has_external_tool(name):
            try:
                result = await self._external_runtime_manager.execute_virtual_tool(
                    name,
                    deepcopy(arguments or {}),
                    effective_policy=_effective_policy_from_context(context),
                    actor_id=context.user_id,
                    context=context,
                )
            except FederationPolicyDenied as exc:
                raise GatewayPolicyDenied(
                    str(exc),
                    reason_code=exc.reason_code,
                    provenance=deepcopy(exc.payload),
                ) from exc
            return _federated_result_payload(result)

        if self._base_runtime is not None:
            return await self._base_runtime.call_tool(name, arguments, context)

        raise ValueError(f"Unknown gateway tool: {name}")

    async def list_resources(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        """Delegate resource listing to the base runtime when available."""

        if self._base_runtime is None:
            return []
        return await self._base_runtime.list_resources(context)

    async def read_resource(
        self,
        uri: str,
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        """Delegate resource reads to the base runtime when available."""

        if self._base_runtime is None:
            raise ValueError(f"Unknown gateway resource: {uri}")
        return await self._base_runtime.read_resource(uri, context)

    async def list_prompts(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        """Delegate prompt listing to the base runtime when available."""

        if self._base_runtime is None:
            return []
        return await self._base_runtime.list_prompts(context)

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        """Delegate prompt reads to the base runtime when available."""

        if self._base_runtime is None:
            raise ValueError(f"Unknown gateway prompt: {name}")
        return await self._base_runtime.get_prompt(name, arguments, context)

    async def list_modules(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        """Delegate module listing to the base runtime when available."""

        if self._base_runtime is None:
            return []
        return await self._base_runtime.list_modules(context)

    async def get_modules_health(self, context: GatewayRequestContext) -> dict[str, Any]:
        """Delegate module health to the base runtime when available."""

        if self._base_runtime is None:
            return {"modules": []}
        return await self._base_runtime.get_modules_health(context)

    async def _has_external_tool(self, name: str) -> bool:
        """Return whether the active external runtime owns this tool name."""

        return any(
            virtual_tool.virtual_name == name
            for virtual_tool in await self._external_runtime_manager.list_virtual_tools()
        )


def _virtual_tool_descriptor(virtual_tool: VirtualExternalTool) -> dict[str, Any]:
    """Convert one virtual external tool into a gateway tool descriptor."""

    metadata = deepcopy(virtual_tool.metadata or {})
    metadata.update(
        {
            "external_server_id": virtual_tool.server_id,
            "is_write": virtual_tool.is_write,
            "source": "external_runtime",
            "upstream_tool_name": virtual_tool.upstream_tool_name,
        }
    )
    return {
        "name": virtual_tool.virtual_name,
        "description": virtual_tool.description,
        "inputSchema": deepcopy(virtual_tool.input_schema),
        "metadata": metadata,
    }


def _effective_policy_from_context(context: GatewayRequestContext) -> Any:
    """Return the profile-derived effective policy stored in request metadata."""

    value = context.metadata.get(EFFECTIVE_POLICY_METADATA_KEY)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    return deepcopy(value)


def _federated_result_payload(result: FederatedToolResult) -> dict[str, Any]:
    """Convert a federated external result into gateway tool-call JSON."""

    metadata = deepcopy(result.metadata or {})
    metadata.update(
        {
            "server_id": result.server_id,
            "upstream_tool_name": result.upstream_tool_name,
            "virtual_tool_name": result.virtual_tool_name,
        }
    )
    return {
        "content": deepcopy(result.content),
        "isError": result.is_error,
        "metadata": metadata,
    }


__all__ = [
    "ExternalRuntimeGatewayRuntime",
]
