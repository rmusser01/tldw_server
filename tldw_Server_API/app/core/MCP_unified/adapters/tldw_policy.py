from __future__ import annotations

from typing import Any


class TldwEffectivePolicyResolver:
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
    async def evaluate_tool_call(self, **kwargs: Any) -> dict[str, Any]:
        from tldw_Server_API.app.services.mcp_hub_approval_service import (
            get_mcp_hub_approval_service,
        )

        service = await get_mcp_hub_approval_service()
        return await service.evaluate_tool_call(**kwargs)


class TldwPathScopeEnforcer:
    async def evaluate_tool_call(self, **kwargs: Any) -> dict[str, Any]:
        from tldw_Server_API.app.services.mcp_hub_path_enforcement_service import (
            get_mcp_hub_path_enforcement_service,
        )

        service = await get_mcp_hub_path_enforcement_service()
        return await service.evaluate_tool_call(**kwargs)


class TldwExternalAccessEvaluator:
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
