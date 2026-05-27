"""Default policy adapters that delegate MCP decisions to tldw_server services."""

from __future__ import annotations

from typing import Any


class TldwEffectivePolicyResolver:
    """Resolve the effective MCP Hub policy for a request context."""

    async def resolve_for_context(
        self,
        *,
        user_id: str | None,
        metadata: dict[str, Any],
    ) -> dict[str, Any] | None:
        from tldw_Server_API.app.services.mcp_hub_policy_resolver import (
            get_mcp_hub_policy_resolver,
        )

        resolver = await get_mcp_hub_policy_resolver()
        return await resolver.resolve_for_context(user_id=user_id, metadata=metadata)


class TldwApprovalEvaluator:
    """Evaluate MCP tool-call approval requirements through the host service."""

    async def evaluate_tool_call(
        self,
        *,
        effective_policy: dict[str, Any],
        tool_name: str,
        tool_args: Any,
        context: Any,
        tool_def: dict[str, Any] | None,
        is_write: bool | None,
        within_effective_policy: bool,
        force_approval: bool = False,
        approval_reason: str | None = None,
        scope_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from tldw_Server_API.app.services.mcp_hub_approval_service import (
            get_mcp_hub_approval_service,
        )

        service = await get_mcp_hub_approval_service()
        return await service.evaluate_tool_call(
            effective_policy=effective_policy,
            tool_name=tool_name,
            tool_args=tool_args,
            context=context,
            tool_def=tool_def,
            is_write=is_write,
            within_effective_policy=within_effective_policy,
            force_approval=force_approval,
            approval_reason=approval_reason,
            scope_payload=scope_payload,
        )


class TldwPathScopeEnforcer:
    """Apply tldw_server path-scope policy checks to MCP tool calls."""

    async def evaluate_tool_call(
        self,
        *,
        effective_policy: dict[str, Any] | None,
        context: Any,
        tool_name: str,
        tool_args: Any,
        tool_def: dict[str, Any] | None,
    ) -> dict[str, Any]:
        from tldw_Server_API.app.services.mcp_hub_path_enforcement_service import (
            get_mcp_hub_path_enforcement_service,
        )

        service = await get_mcp_hub_path_enforcement_service()
        return await service.evaluate_tool_call(
            effective_policy=effective_policy,
            context=context,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_def=tool_def,
        )


class TldwExternalAccessEvaluator:
    """Resolve external access policy for MCP federated source metadata."""

    async def resolve_for_sources(
        self,
        *,
        sources: list[dict[str, Any]],
        effective_policy: dict[str, Any] | None,
    ) -> dict[str, Any]:
        from tldw_Server_API.app.services.mcp_hub_external_access_resolver import (
            get_mcp_hub_external_access_resolver,
        )

        resolver = await get_mcp_hub_external_access_resolver()
        return await resolver.resolve_for_sources(
            sources=sources,
            effective_policy=dict(effective_policy or {}),
        )
