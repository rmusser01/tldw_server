"""Default policy adapters that delegate MCP decisions to tldw_server services."""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import sys
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import ModuleType
from typing import Any

from mcp_unified.interfaces.path_scope import PathScopeCandidate

_MCP_HUB_POLICY_RESOLVER_MODULE = "tldw_Server_API.app.services.mcp_hub_policy_resolver"
_ACP_PACKAGE_MODULE = "tldw_Server_API.app.core.Agent_Client_Protocol"
_ACP_MERGE_UTILS_MODULE = f"{_ACP_PACKAGE_MODULE}.merge_utils"
_ACP_IMPORT_LOCK = threading.RLock()

_ResolverFactory = Callable[[], Awaitable[Any]]


def _load_merge_utils_without_acp_package_init() -> tuple[ModuleType, ModuleType]:
    """Load ACP merge helpers without triggering ACP package runtime imports."""
    package_dir = Path(__file__).resolve().parents[2] / "Agent_Client_Protocol"
    merge_utils_path = package_dir / "merge_utils.py"

    package_module = ModuleType(_ACP_PACKAGE_MODULE)
    package_module.__package__ = _ACP_PACKAGE_MODULE
    package_module.__path__ = [str(package_dir)]  # type: ignore[attr-defined]
    package_spec = importlib.machinery.ModuleSpec(
        _ACP_PACKAGE_MODULE,
        loader=None,
        is_package=True,
    )
    package_spec.submodule_search_locations = [str(package_dir)]
    package_module.__spec__ = package_spec

    merge_spec = importlib.util.spec_from_file_location(
        _ACP_MERGE_UTILS_MODULE,
        merge_utils_path,
    )
    if merge_spec is None or merge_spec.loader is None:
        raise ImportError("Unable to load Agent_Client_Protocol.merge_utils")
    merge_module = importlib.util.module_from_spec(merge_spec)
    sys.modules[_ACP_PACKAGE_MODULE] = package_module
    sys.modules[_ACP_MERGE_UTILS_MODULE] = merge_module
    package_module.merge_utils = merge_module
    merge_spec.loader.exec_module(merge_module)
    return package_module, merge_module


def _load_mcp_hub_policy_resolver_module() -> ModuleType:
    """Import the MCP Hub policy resolver without tripping the ACP import cycle."""
    with _ACP_IMPORT_LOCK:
        existing = sys.modules.get(_MCP_HUB_POLICY_RESOLVER_MODULE)
        if isinstance(existing, ModuleType) and hasattr(existing, "get_mcp_hub_policy_resolver"):
            return existing

        if _ACP_PACKAGE_MODULE in sys.modules:
            return importlib.import_module(_MCP_HUB_POLICY_RESOLVER_MODULE)

        previous_package = sys.modules.get(_ACP_PACKAGE_MODULE)
        previous_merge_utils = sys.modules.get(_ACP_MERGE_UTILS_MODULE)
        try:
            _load_merge_utils_without_acp_package_init()
            return importlib.import_module(_MCP_HUB_POLICY_RESOLVER_MODULE)
        finally:
            if previous_merge_utils is None:
                sys.modules.pop(_ACP_MERGE_UTILS_MODULE, None)
            else:
                sys.modules[_ACP_MERGE_UTILS_MODULE] = previous_merge_utils
            if previous_package is None:
                sys.modules.pop(_ACP_PACKAGE_MODULE, None)
            else:
                sys.modules[_ACP_PACKAGE_MODULE] = previous_package


async def _get_mcp_hub_policy_resolver() -> Any:
    """Create the host MCP Hub policy resolver through a cycle-safe import."""
    module = _load_mcp_hub_policy_resolver_module()
    return await module.get_mcp_hub_policy_resolver()


class TldwEffectivePolicyResolver:
    """Resolve the effective MCP Hub policy for a request context."""

    def __init__(self, resolver_factory: _ResolverFactory | None = None) -> None:
        self._resolver_factory = resolver_factory or _get_mcp_hub_policy_resolver

    async def resolve_for_context(
        self,
        *,
        user_id: str | None,
        metadata: dict[str, Any],
    ) -> dict[str, Any] | None:
        resolver = await self._resolver_factory()
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
        path_scope_candidates: list[PathScopeCandidate] | None = None,
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
            path_scope_candidates=path_scope_candidates,
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
