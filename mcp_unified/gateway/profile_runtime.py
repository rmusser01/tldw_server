"""Profile-aware gateway runtime wrapper for standalone MCP gateways."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Protocol

from mcp_unified.interfaces.storage import ProfileStore
from mcp_unified.profiles.models import MCPProfile
from mcp_unified.profiles.resolution import (
    EffectivePolicyResult,
    ProfileResolutionResult,
    build_effective_policy_result,
)
from mcp_unified.profiles.resolver import StoreBackedProfileResolver
from mcp_unified.tool_use_reporting.sanitization import sanitize_safe_id

from .runtime import GatewayPolicyDenied, GatewayRequestContext, GatewayRuntime
from .tool_discovery import (
    describe_profile_tool,
    list_direct_profile_backend_tools,
    list_profile_tools,
    profile_has_deferred_tools,
    resolve_profile_tool_call,
    search_profile_tools,
)

EFFECTIVE_POLICY_METADATA_KEY = "_gateway_effective_policy"
TOOL_USE_BRIDGE_TOOL_NAME_METADATA_KEY = "mcp_tool_use_bridge_tool_name"
TOOL_USE_REQUESTED_TOOL_ID_METADATA_KEY = "mcp_tool_use_requested_tool_id"
TOOL_USE_EFFECTIVE_TOOL_NAME_METADATA_KEY = "mcp_tool_use_effective_tool_name"
TOOL_USE_SOURCE_KIND_METADATA_KEY = "mcp_tool_use_source_kind"
_TOOL_DISCOVERY_CATEGORY = "tool_discovery"
_TOOL_DISCOVERY_READ_CAPABILITY = "tool_discovery.read"
_TOOL_DISCOVERY_CALL_CAPABILITY = "tool_discovery.call"
_TOOL_CATEGORIES_LIST = "tool_categories.list"
_PROFILE_TOOLS_LIST = "profile.tools.list"
_TOOL_SEARCH = "tool_search"
_TOOL_DESCRIBE = "tool_describe"
_TOOL_CALL = "tool_call"
_READ_ONLY_BRIDGE_TOOL_NAMES = frozenset(
    {
        _TOOL_CATEGORIES_LIST,
        _PROFILE_TOOLS_LIST,
        _TOOL_SEARCH,
        _TOOL_DESCRIBE,
    }
)
_BRIDGE_TOOL_NAMES = frozenset({*_READ_ONLY_BRIDGE_TOOL_NAMES, _TOOL_CALL})
_TOOL_CALL_ARGUMENT_KEYS = frozenset({"tool_id", "arguments"})
_TOOL_SEARCH_ARGUMENT_KEYS = frozenset({"query", "category", "limit"})
_TOOL_DESCRIBE_ARGUMENT_KEYS = frozenset({"tool_id"})
_EMPTY_ARGUMENT_KEYS = frozenset()
_BRIDGE_TOOL_DESCRIPTORS: dict[str, dict[str, Any]] = {
    _TOOL_CATEGORIES_LIST: {
        "name": _TOOL_CATEGORIES_LIST,
        "description": "List tool categories visible to the active profile.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        "metadata": {
            "category": _TOOL_DISCOVERY_CATEGORY,
            "capabilities": [_TOOL_DISCOVERY_READ_CAPABILITY],
        },
    },
    _PROFILE_TOOLS_LIST: {
        "name": _PROFILE_TOOLS_LIST,
        "description": "List the tool catalog visible to the active profile.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        "metadata": {
            "category": _TOOL_DISCOVERY_CATEGORY,
            "capabilities": [_TOOL_DISCOVERY_READ_CAPABILITY],
        },
    },
    _TOOL_SEARCH: {
        "name": _TOOL_SEARCH,
        "description": "Search tools visible to the active profile.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "default": ""},
                "category": {"type": ["string", "null"], "default": None},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 20,
                },
            },
            "additionalProperties": False,
        },
        "metadata": {
            "category": _TOOL_DISCOVERY_CATEGORY,
            "capabilities": [_TOOL_DISCOVERY_READ_CAPABILITY],
        },
    },
    _TOOL_DESCRIBE: {
        "name": _TOOL_DESCRIBE,
        "description": "Describe one tool visible to the active profile.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tool_id": {"type": "string"},
            },
            "required": ["tool_id"],
            "additionalProperties": False,
        },
        "metadata": {
            "category": _TOOL_DISCOVERY_CATEGORY,
            "capabilities": [_TOOL_DISCOVERY_READ_CAPABILITY],
        },
    },
    _TOOL_CALL: {
        "name": _TOOL_CALL,
        "description": "Call an installed tool by profile-scoped tool id.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tool_id": {"type": "string"},
                "arguments": {"type": "object"},
            },
            "required": ["tool_id", "arguments"],
            "additionalProperties": False,
        },
        "metadata": {
            "category": _TOOL_DISCOVERY_CATEGORY,
            "capabilities": [
                _TOOL_DISCOVERY_READ_CAPABILITY,
                _TOOL_DISCOVERY_CALL_CAPABILITY,
            ],
        },
    },
}


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

        tools = await self._safe_backend_tools(context)
        direct_tools = list_direct_profile_backend_tools(
            profile_result.profile,
            tools,
        )
        allowed_tool_names = {
            name
            for tool in direct_tools
            if (name := _tool_name(tool)) is not None
        }
        include_tool_call = (
            _profile_has_deferred_categories(profile_result.profile)
            or profile_has_deferred_tools(profile_result.profile, tools)
        )
        return [
            *direct_tools,
            *_profile_bridge_tool_descriptors(
                profile_result.profile,
                suppressed_names=allowed_tool_names,
                include_tool_call=include_tool_call,
            ),
        ]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        """Execute an allowed backend tool or raise a structured policy denial."""

        profile = await self._require_profile(context)
        if _is_bridge_tool_name(name):
            try:
                backend_tools = await self._safe_backend_tools(context)
            except Exception:
                _validate_synthetic_bridge_arguments(name, arguments)
                raise
            collision_tool = _allowed_backend_tool_by_name(profile, backend_tools, name)
            if collision_tool is not None:
                return await self._call_backend_tool_through_policy(
                    profile,
                    name,
                    arguments,
                    context,
                    tool=collision_tool,
                )
            return await self._call_bridge_tool(
                profile,
                name,
                arguments,
                context,
                backend_tools=backend_tools,
            )
        return await self._call_backend_tool_through_policy(
            profile,
            name,
            arguments,
            context,
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

    async def _safe_backend_tools(
        self,
        context: GatewayRequestContext,
    ) -> list[Any]:
        """Return backend discovery data for bridge catalog operations."""

        tools = await self._backend.list_tools(context)
        return tools if isinstance(tools, list) else []

    async def _call_bridge_tool(
        self,
        profile: MCPProfile,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
        *,
        backend_tools: list[Any],
    ) -> dict[str, Any]:
        """Execute one synthetic profile-scoped tool discovery helper."""

        if name == _TOOL_CATEGORIES_LIST:
            _validate_bridge_argument_keys(
                name,
                arguments,
                _EMPTY_ARGUMENT_KEYS,
                reason_code="invalid_tool_categories_arguments",
            )
            catalog = list_profile_tools(profile, backend_tools)
            return {
                "profile_id": catalog["profile_id"],
                "categories": catalog["categories"],
                "progressive_disclosure": catalog["progressive_disclosure"],
            }
        if name == _PROFILE_TOOLS_LIST:
            _validate_bridge_argument_keys(
                name,
                arguments,
                _EMPTY_ARGUMENT_KEYS,
                reason_code="invalid_profile_tools_list_arguments",
            )
            return list_profile_tools(profile, backend_tools)
        if name == _TOOL_SEARCH:
            query, category, limit = _validated_tool_search_arguments(arguments)
            return {
                "tools": search_profile_tools(
                    profile,
                    backend_tools,
                    query=query,
                    category=category,
                    limit=limit,
                )
            }
        if name == _TOOL_DESCRIBE:
            tool_id = _validated_tool_describe_arguments(arguments)
            description = describe_profile_tool(profile, backend_tools, tool_id)
            if description is None:
                return _bridge_tool_error_payload(
                    "tool_not_found",
                    status="not_found",
                    tool_id=tool_id,
                )
            return description
        if name == _TOOL_CALL:
            tool_id, delegated_arguments = _validated_tool_call_arguments(arguments)
            if not _profile_has_deferred_categories(profile):
                raise GatewayPolicyDenied(
                    "Gateway profile denied tool execution: tool_not_allowed",
                    reason_code="tool_not_allowed",
                    provenance={"profile_id": profile.id, "tool_name": name},
                )
            resolution = resolve_profile_tool_call(profile, backend_tools, tool_id)
            if resolution.get("status") == "not_found":
                return _bridge_tool_error_payload(
                    "tool_not_found",
                    status="not_found",
                    tool_id=str(resolution.get("tool_id", tool_id)),
                )
            if resolution.get("status") == "unavailable":
                return _bridge_tool_error_payload(
                    "tool_not_enabled",
                    status="unavailable",
                    tool_id=str(resolution.get("tool_id", tool_id)),
                    installation_status=resolution.get("installation_status"),
                    activation=resolution.get("activation"),
                    unavailable_reason=resolution.get("unavailable_reason"),
                )
            resolved_name = resolution.get("tool_name")
            if not isinstance(resolved_name, str) or not resolved_name.strip():
                return _bridge_tool_error_payload(
                    "tool_not_found",
                    status="not_found",
                    tool_id=tool_id,
                )
            delegated_context = _context_with_bridge_tool_use_metadata(
                context,
                bridge_tool_name=name,
                requested_tool_id=tool_id,
                effective_tool_name=resolved_name.strip(),
            )
            result = await self._call_backend_tool_through_policy(
                profile,
                resolved_name.strip(),
                delegated_arguments,
                delegated_context,
                tool=resolution.get("tool"),
            )
            return _result_with_bridge_tool_use_metadata(
                result,
                _bridge_tool_use_metadata(delegated_context),
            )

        raise GatewayPolicyDenied(
            "Gateway profile denied tool execution: tool_not_allowed",
            reason_code="tool_not_allowed",
            provenance={"profile_id": profile.id, "tool_name": name},
        )

    async def _call_backend_tool_through_policy(
        self,
        profile: MCPProfile,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
        *,
        tool: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run profile policy for a real backend tool before delegation."""

        policy_result = _allowed_policy_result_for_tool(profile, name, tool)
        if tool is None and _policy_needs_backend_tool_metadata(profile, policy_result):
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


def _validate_synthetic_bridge_arguments(name: str, arguments: Any) -> None:
    """Validate bridge arguments without requiring backend discovery."""

    if name == _TOOL_CATEGORIES_LIST:
        _validate_bridge_argument_keys(
            name,
            arguments,
            _EMPTY_ARGUMENT_KEYS,
            reason_code="invalid_tool_categories_arguments",
        )
        return
    if name == _PROFILE_TOOLS_LIST:
        _validate_bridge_argument_keys(
            name,
            arguments,
            _EMPTY_ARGUMENT_KEYS,
            reason_code="invalid_profile_tools_list_arguments",
        )
        return
    if name == _TOOL_SEARCH:
        _validated_tool_search_arguments(arguments)
        return
    if name == _TOOL_DESCRIBE:
        _validated_tool_describe_arguments(arguments)
        return
    if name == _TOOL_CALL:
        _validated_tool_call_arguments(arguments)
        return


def _is_bridge_tool_name(name: str) -> bool:
    """Return whether a tool name is reserved for profile discovery bridging."""

    return name in _BRIDGE_TOOL_NAMES


def _allowed_backend_tool_by_name(
    profile: MCPProfile,
    backend_tools: list[Any],
    name: str,
) -> dict[str, Any] | None:
    """Return an allowed backend descriptor matching a bridge-reserved name."""

    for tool in list_direct_profile_backend_tools(profile, backend_tools):
        if (
            isinstance(tool, dict)
            and _tool_name(tool) == name
        ):
            return tool
    return None


def _profile_bridge_tool_descriptors(
    profile: MCPProfile,
    *,
    suppressed_names: set[str],
    include_tool_call: bool | None = None,
) -> list[dict[str, Any]]:
    """Return caller-owned synthetic discovery tool descriptors for a profile."""

    names = [
        _TOOL_CATEGORIES_LIST,
        _PROFILE_TOOLS_LIST,
        _TOOL_SEARCH,
        _TOOL_DESCRIBE,
    ]
    should_include_tool_call = (
        _profile_has_deferred_categories(profile)
        if include_tool_call is None
        else include_tool_call
    )
    if should_include_tool_call:
        names.append(_TOOL_CALL)
    return [
        deepcopy(_BRIDGE_TOOL_DESCRIPTORS[name])
        for name in names
        if name not in suppressed_names
    ]


def _profile_has_deferred_categories(profile: MCPProfile) -> bool:
    """Return whether profile tooling metadata defers any categories."""

    metadata = profile.metadata if isinstance(profile.metadata, dict) else {}
    tooling = metadata.get("tooling")
    if not isinstance(tooling, dict):
        return False
    progressive = tooling.get("progressive_disclosure")
    if not isinstance(progressive, dict):
        return False
    return any(
        isinstance(category, str) and bool(category.strip())
        for category in _as_sequence(progressive.get("deferred_categories"))
    )


def _validated_tool_search_arguments(
    arguments: Any,
) -> tuple[str, str | None, int]:
    """Return normalized tool search arguments or raise policy-denied validation."""

    _validate_bridge_argument_keys(
        _TOOL_SEARCH,
        arguments,
        _TOOL_SEARCH_ARGUMENT_KEYS,
        reason_code="invalid_tool_search_arguments",
    )
    query = arguments.get("query", "")
    if not isinstance(query, str):
        raise _invalid_bridge_arguments(
            _TOOL_SEARCH,
            reason_code="invalid_tool_search_arguments",
            field="query",
        )
    category = arguments.get("category")
    if category is not None and not isinstance(category, str):
        raise _invalid_bridge_arguments(
            _TOOL_SEARCH,
            reason_code="invalid_tool_search_arguments",
            field="category",
        )
    limit = arguments.get("limit", 20)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1 or limit > 100:
        raise _invalid_bridge_arguments(
            _TOOL_SEARCH,
            reason_code="invalid_tool_search_arguments",
            field="limit",
        )
    return query, category, limit


def _validated_tool_describe_arguments(arguments: Any) -> str:
    """Return a normalized tool id for tool_describe."""

    _validate_bridge_argument_keys(
        _TOOL_DESCRIBE,
        arguments,
        _TOOL_DESCRIBE_ARGUMENT_KEYS,
        reason_code="invalid_tool_describe_arguments",
        required_keys=_TOOL_DESCRIBE_ARGUMENT_KEYS,
    )
    return _required_bridge_tool_id(
        _TOOL_DESCRIBE,
        arguments.get("tool_id"),
        reason_code="invalid_tool_describe_arguments",
    )


def _validated_tool_call_arguments(arguments: Any) -> tuple[str, dict[str, Any]]:
    """Return normalized installed-tool call arguments."""

    _validate_bridge_argument_keys(
        _TOOL_CALL,
        arguments,
        _TOOL_CALL_ARGUMENT_KEYS,
        reason_code="invalid_tool_call_arguments",
        required_keys=_TOOL_CALL_ARGUMENT_KEYS,
    )
    tool_id = _required_bridge_tool_id(
        _TOOL_CALL,
        arguments.get("tool_id"),
        reason_code="invalid_tool_call_arguments",
    )
    delegated_arguments = arguments.get("arguments")
    if not isinstance(delegated_arguments, dict):
        raise _invalid_bridge_arguments(
            _TOOL_CALL,
            reason_code="invalid_tool_call_arguments",
            field="arguments",
        )
    return tool_id, deepcopy(delegated_arguments)


def _validate_bridge_argument_keys(
    tool_name: str,
    arguments: Any,
    allowed_keys: frozenset[str],
    *,
    reason_code: str,
    required_keys: frozenset[str] = frozenset(),
) -> None:
    """Validate bridge argument object shape and reject unknown fields."""

    if not isinstance(arguments, dict):
        raise _invalid_bridge_arguments(tool_name, reason_code=reason_code)
    missing = sorted(key for key in required_keys if key not in arguments)
    unknown = [
        _bridge_argument_key_label(key)
        for key in arguments
        if key not in allowed_keys
    ]
    if missing or unknown:
        raise _invalid_bridge_arguments(
            tool_name,
            reason_code=reason_code,
            missing=missing,
            unknown=unknown,
        )


def _bridge_argument_key_label(key: Any) -> str:
    """Return a stable validation label for arbitrary mapping keys."""

    return key if isinstance(key, str) else repr(key)


def _required_bridge_tool_id(
    tool_name: str,
    value: Any,
    *,
    reason_code: str,
) -> str:
    """Return a non-empty tool id string for a bridge helper."""

    if not isinstance(value, str) or not value.strip():
        raise _invalid_bridge_arguments(
            tool_name,
            reason_code=reason_code,
            field="tool_id",
        )
    return value.strip()


def _invalid_bridge_arguments(
    tool_name: str,
    *,
    reason_code: str,
    field: str | None = None,
    missing: list[str] | None = None,
    unknown: list[str] | None = None,
) -> GatewayPolicyDenied:
    """Build a structured policy denial for invalid synthetic tool arguments."""

    provenance: dict[str, Any] = {"tool_name": tool_name}
    if field is not None:
        provenance["field"] = field
    if missing:
        provenance["missing_fields"] = missing
    if unknown:
        provenance["unknown_fields"] = unknown
    return GatewayPolicyDenied(
        f"Invalid arguments for profile discovery bridge tool: {tool_name}",
        reason_code=reason_code,
        provenance=provenance,
    )


def _bridge_tool_error_payload(
    reason_code: str,
    *,
    status: str,
    tool_id: str,
    **details: Any,
) -> dict[str, Any]:
    """Return a normal tool result payload for profile bridge lookup failures."""

    error = {
        "status": status,
        "reason_code": reason_code,
        "tool_id": tool_id,
    }
    for key, value in details.items():
        if value is not None:
            error[key] = value
    return {
        "ok": False,
        "status": status,
        "reason_code": reason_code,
        "tool_id": tool_id,
        "error": error,
    }


def _context_profile_id(context: GatewayRequestContext) -> str | None:
    """Return an explicit profile id from transport metadata, if present."""

    metadata = context.metadata or {}
    for key in ("profile_id", "profileId", "mcp_profile", "mcp-profile"):
        value = metadata.get(key)
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
    metadata = dict(context.metadata or {})
    metadata[EFFECTIVE_POLICY_METADATA_KEY] = _json_safe_model(policy)
    return GatewayRequestContext(
        request_id=context.request_id,
        client_id=context.client_id,
        user_id=context.user_id,
        metadata=metadata,
    )


def _context_with_bridge_tool_use_metadata(
    context: GatewayRequestContext,
    *,
    bridge_tool_name: str,
    requested_tool_id: str,
    effective_tool_name: str,
) -> GatewayRequestContext:
    """Return a context copy carrying safe bridge tool-use metadata."""

    side_channel: dict[str, Any] = {TOOL_USE_SOURCE_KIND_METADATA_KEY: "bridge"}
    for metadata_key, value in (
        (TOOL_USE_BRIDGE_TOOL_NAME_METADATA_KEY, bridge_tool_name),
        (TOOL_USE_REQUESTED_TOOL_ID_METADATA_KEY, requested_tool_id),
        (TOOL_USE_EFFECTIVE_TOOL_NAME_METADATA_KEY, effective_tool_name),
    ):
        safe_value = sanitize_safe_id(value, field=metadata_key)
        if safe_value is not None:
            side_channel[metadata_key] = safe_value

    metadata = dict(context.metadata or {})
    metadata.update(side_channel)
    return GatewayRequestContext(
        request_id=context.request_id,
        client_id=context.client_id,
        user_id=context.user_id,
        metadata=metadata,
    )


def _bridge_tool_use_metadata(context: GatewayRequestContext) -> dict[str, Any]:
    """Return only bridge tool-use metadata from a context copy."""

    bridge_keys = {
        TOOL_USE_BRIDGE_TOOL_NAME_METADATA_KEY,
        TOOL_USE_REQUESTED_TOOL_ID_METADATA_KEY,
        TOOL_USE_EFFECTIVE_TOOL_NAME_METADATA_KEY,
        TOOL_USE_SOURCE_KIND_METADATA_KEY,
    }
    return {
        key: value
        for key, value in context.metadata.items()
        if key in bridge_keys
    }


def _result_with_bridge_tool_use_metadata(
    result: dict[str, Any],
    bridge_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Return a result copy with bridge metadata for outer reporting wrappers."""

    if not bridge_metadata:
        return result
    copied = dict(result)
    metadata = copied.get("metadata")
    metadata_copy = dict(metadata) if isinstance(metadata, dict) else {}
    tool_use = metadata_copy.get("mcp_tool_use")
    tool_use_copy = dict(tool_use) if isinstance(tool_use, dict) else {}
    tool_use_copy.update(bridge_metadata)
    metadata_copy["mcp_tool_use"] = tool_use_copy
    copied["metadata"] = metadata_copy
    return copied


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
