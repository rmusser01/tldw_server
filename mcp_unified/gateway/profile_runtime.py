"""Profile-aware gateway runtime wrapper for standalone MCP gateways."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from mcp_unified.interfaces.storage import ProfileStore
from mcp_unified.profiles.models import MCPProfile
from mcp_unified.profiles.resolution import (
    EffectivePolicyResult,
    ProfileResolutionResult,
    build_effective_policy_result,
)
from mcp_unified.profiles.resolver import StoreBackedProfileResolver

from .runtime import GatewayPolicyDenied, GatewayRequestContext, GatewayRuntime

EFFECTIVE_POLICY_METADATA_KEY = "_gateway_effective_policy"


class GatewayProfileResolver(Protocol):
    """Structured profile resolver needed by the profile-aware gateway runtime."""

    async def resolve_profile_result(
        self,
        profile_id: str | None,
        *,
        user_id: str | None = None,
    ) -> ProfileResolutionResult:
        """Resolve a profile into a structured outcome."""
        ...


class ProfileAwareGatewayRuntime:
    """Apply standalone profile policy before delegating tool calls."""

    def __init__(
        self,
        backend: GatewayRuntime,
        *,
        profile_store: ProfileStore | None = None,
        profile_resolver: GatewayProfileResolver | None = None,
        default_profile_id: str | None = None,
    ) -> None:
        if profile_resolver is None:
            if profile_store is None:
                raise ValueError("profile_store or profile_resolver is required")
            profile_resolver = StoreBackedProfileResolver(
                profile_store,
                default_profile_id=default_profile_id,
            )
        self._backend = backend
        self._profile_resolver = profile_resolver

    @property
    def name(self) -> str:
        """Return the wrapped runtime name."""

        return str(getattr(self._backend, "name", "mcp-unified-gateway"))

    @property
    def version(self) -> str:
        """Return the wrapped runtime version."""

        return str(getattr(self._backend, "version", "0.1.0"))

    async def list_tools(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        """Return backend tools allowed by the resolved profile policy."""

        profile_result = await self._resolve_profile(context)
        if profile_result.status != "resolved" or profile_result.profile is None:
            return []

        tools = await self._backend.list_tools(context)
        if not isinstance(tools, list):
            return []

        allowed_tools: list[dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if _tool_name(tool) is not None and _tool_allowed_by_profile(
                profile_result.profile,
                tool,
            ):
                allowed_tools.append(tool)
        return allowed_tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        """Execute an allowed backend tool or raise a structured policy denial."""

        profile = await self._require_profile(context)
        policy_result = _allowed_policy_result_for_tool(profile, name, None)
        if _policy_needs_backend_tool_metadata(profile, policy_result):
            try:
                tool = await self._find_backend_tool(name, context)
            except Exception as exc:
                raise GatewayPolicyDenied(
                    "Gateway profile could not inspect backend tool metadata",
                    reason_code="tool_metadata_unavailable",
                    provenance={
                        "tool_name": name,
                        "backend_error": type(exc).__name__,
                    },
                ) from exc
            policy_result = _allowed_policy_result_for_tool(profile, name, tool)
        if policy_result.status != "resolved":
            raise _policy_denied(
                policy_result,
                message=f"Gateway profile denied tool execution: {policy_result.reason_code}",
            )
        return await self._backend.call_tool(
            name,
            arguments,
            _context_with_effective_policy(context, policy_result.policy),
        )

    async def list_resources(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        """Delegate resource discovery unchanged for this profile slice."""

        return await self._backend.list_resources(context)

    async def read_resource(
        self,
        uri: str,
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        """Delegate resource reads unchanged for this profile slice."""

        return await self._backend.read_resource(uri, context)

    async def list_prompts(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        """Delegate prompt discovery unchanged for this profile slice."""

        return await self._backend.list_prompts(context)

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        """Delegate prompt reads unchanged for this profile slice."""

        return await self._backend.get_prompt(name, arguments, context)

    async def list_modules(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        """Delegate module discovery unchanged for this profile slice."""

        return await self._backend.list_modules(context)

    async def get_modules_health(self, context: GatewayRequestContext) -> dict[str, Any]:
        """Delegate module health unchanged for this profile slice."""

        return await self._backend.get_modules_health(context)

    async def _resolve_profile(
        self,
        context: GatewayRequestContext,
    ) -> ProfileResolutionResult:
        profile_id = _context_profile_id(context)
        return await self._profile_resolver.resolve_profile_result(
            profile_id,
            user_id=context.user_id,
        )

    async def _require_profile(self, context: GatewayRequestContext) -> MCPProfile:
        profile_result = await self._resolve_profile(context)
        if profile_result.status != "resolved" or profile_result.profile is None:
            raise GatewayPolicyDenied(
                f"Gateway profile resolution failed: {profile_result.reason_code}",
                status=profile_result.status,
                reason_code=profile_result.reason_code,
                provenance=profile_result.provenance,
                warnings=profile_result.warnings,
            )
        return profile_result.profile

    async def _find_backend_tool(
        self,
        name: str,
        context: GatewayRequestContext,
    ) -> dict[str, Any] | None:
        tools = await self._backend.list_tools(context)
        if not isinstance(tools, list):
            return None
        return next(
            (
                tool
                for tool in tools
                if isinstance(tool, dict) and _tool_name(tool) == name
            ),
            None,
        )


def _context_profile_id(context: GatewayRequestContext) -> str | None:
    """Return an explicit profile id from transport metadata, if present."""

    for key in ("profile_id", "profileId", "mcp_profile", "mcp-profile"):
        value = context.metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _context_with_effective_policy(
    context: GatewayRequestContext,
    policy: Any,
) -> GatewayRequestContext:
    """Return a copy of the context carrying resolved effective policy data."""

    if policy is None:
        return context
    metadata = dict(context.metadata)
    metadata[EFFECTIVE_POLICY_METADATA_KEY] = _json_safe_model(policy)
    return GatewayRequestContext(
        request_id=context.request_id,
        client_id=context.client_id,
        user_id=context.user_id,
        metadata=metadata,
    )


def _json_safe_model(value: Any) -> Any:
    """Dump pydantic models to JSON-safe mappings with v1/v2 compatibility."""

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    return value


def _tool_name(tool: Any) -> str | None:
    """Return a valid tool name from a backend tool descriptor."""

    if not isinstance(tool, dict):
        return None
    name = tool.get("name")
    return name.strip() if isinstance(name, str) and name.strip() else None


def _tool_allowed_by_profile(profile: MCPProfile, tool: Any) -> bool:
    """Return whether a backend tool descriptor is visible for a profile."""

    name = _tool_name(tool)
    if name is None:
        return False
    return _allowed_policy_result_for_tool(profile, name, tool).status == "resolved"


def _allowed_policy_result_for_tool(
    profile: MCPProfile,
    tool_name: str,
    tool: Any,
) -> EffectivePolicyResult:
    """Return the first allow result, preserving exact tool-deny precedence."""

    name_only_result = _effective_policy_for_tool(profile, tool_name, None)
    if (
        name_only_result.status == "resolved"
        or name_only_result.reason_code != "tool_not_allowed"
    ):
        return name_only_result

    for capability in _tool_capabilities(tool):
        if capability is None:
            continue
        capability_result = _effective_policy_for_tool(
            profile,
            tool_name,
            tool,
            capability=capability,
        )
        if capability_result.status == "resolved":
            return capability_result
    return name_only_result


def _policy_needs_backend_tool_metadata(
    profile: MCPProfile,
    policy_result: EffectivePolicyResult,
) -> bool:
    """Return whether policy needs tool metadata before final denial."""

    if policy_result.status == "resolved" or policy_result.reason_code != "tool_not_allowed":
        return False
    policy_document = profile.policy_document
    if policy_document.allowed_tools:
        return False
    return bool(policy_document.capabilities or policy_document.denied_capabilities)


def _effective_policy_for_tool(
    profile: MCPProfile,
    tool_name: str,
    tool: Any,
    *,
    capability: str | None = None,
) -> EffectivePolicyResult:
    """Build the effective policy outcome for one tool descriptor."""

    return build_effective_policy_result(
        profile,
        tool_name=tool_name,
        capability=capability if capability is not None else _primary_capability(tool),
    )


def _primary_capability(tool: Any) -> str | None:
    """Return the first advertised tool capability, if any."""

    capabilities = _tool_capabilities(tool)
    return capabilities[0] if capabilities else None


def _tool_capabilities(tool: Any) -> list[str | None]:
    """Return advertised capabilities plus a no-capability fallback."""

    if not isinstance(tool, dict):
        return [None]
    metadata = tool.get("metadata")
    capability_values: list[Any] = []
    if isinstance(metadata, dict):
        capability_values.extend(_as_sequence(metadata.get("capabilities")))
        capability_values.extend(_as_sequence(metadata.get("capability")))
    capability_values.extend(_as_sequence(tool.get("capabilities")))
    capabilities = [
        value.strip()
        for value in capability_values
        if isinstance(value, str) and value.strip()
    ]
    return [*capabilities, None]


def _as_sequence(value: Any) -> Sequence[Any]:
    """Normalize scalar-or-sequence metadata values into a sequence."""

    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return value
    return ()


def _policy_denied(
    policy_result: EffectivePolicyResult,
    *,
    message: str,
) -> GatewayPolicyDenied:
    """Build a structured denial from an effective policy result."""

    return GatewayPolicyDenied(
        message,
        status=policy_result.status,
        reason_code=policy_result.reason_code,
        provenance=policy_result.provenance,
        warnings=policy_result.warnings,
    )


__all__ = ["GatewayProfileResolver", "ProfileAwareGatewayRuntime"]
