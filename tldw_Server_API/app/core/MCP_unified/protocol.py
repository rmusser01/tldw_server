"""
MCP Protocol implementation for unified module

Implements JSON-RPC 2.0 with enhanced error handling and request routing.
"""

import asyncio
import contextlib
import inspect
import json
import re
import secrets
import time
import uuid
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Callable, Literal, Optional, Union

from pydantic import BaseModel, Field

try:
    from pydantic import field_validator, model_validator  # v2
    _PYDANTIC_V2 = True
except ImportError:  # Fallback for v1
    from pydantic import validator as field_validator  # type: ignore
    _PYDANTIC_V2 = False
    try:
        from pydantic import root_validator as model_validator  # type: ignore
    except ImportError:
        model_validator = None  # type: ignore

from loguru import logger
from mcp_unified.interfaces.path_scope import PathScopeCandidate
from mcp_unified.tool_use_reporting.models import ToolUseEvent, ToolUseStatus

from ..exception_types import PromptCatalogError
from .auth.authnz_rbac import Action, Resource
from .auth.rate_limiter import RateLimitExceeded
from .config import get_config
from .interfaces.runtime import (
    MCPRuntimeDependencies,
    NoopToolCallHookManager,
    NoopToolUseRecorder,
    TelemetryProvider,
    ToolHookAction,
    ToolHookCallContext,
    ToolHookDecision,
)
from .modules.base import BaseModule
from .modules.implementations.prompts_catalog import (
    CONFIG_PROMPT_PREFIX,
    LIBRARY_PROMPT_PREFIX,
    decode_prompt_cursor,
)
from .protocol_types import (
    _TRUSTED_COMPAT_AUTH_VIA as _TRUSTED_COMPAT_AUTH_VIA,
)
from .protocol_types import (
    _TRUSTED_COMPAT_CLAIMS_SENTINEL as _TRUSTED_COMPAT_CLAIMS_SENTINEL,
)
from .protocol_types import (
    _TRUSTED_COMPAT_CLAIMS_SENTINEL_KEY as _TRUSTED_COMPAT_CLAIMS_SENTINEL_KEY,
)
from .protocol_types import (
    _TRUSTED_COMPAT_CLAIMS_SOURCES as _TRUSTED_COMPAT_CLAIMS_SOURCES,
)
from .protocol_types import (
    ApprovalRequiredError,
    GovernanceDeniedError,
    InvalidParamsException,
    PreparedToolCall,
    RequestContext,
    _has_trusted_compat_claims,
)
from .protocol_types import (
    AuthenticatedExecutionScope as AuthenticatedExecutionScope,
)
from .protocol_types import (
    _metadata_claim_values as _metadata_claim_values,
)
from .protocol_types import (
    _metadata_has_admin_claims as _metadata_has_admin_claims,
)
from .protocol_types import (
    _trusted_compat_claims_metadata as _trusted_compat_claims_metadata,
)
from .protocol_types import (
    _TrustedCompatClaimsSentinel as _TrustedCompatClaimsSentinel,
)
from .tool_execution import ToolExecutionCoordinator, ToolExecutionDependencies, ToolExecutionReporter
from .tool_execution.hooks import ToolExecutionHooks
from .tool_execution.idempotency import IdempotencyManager, RedisError
from .tool_execution.models import PreparedExecutionPolicy
from .tool_execution.runtime import ToolExecutionRuntime
from .tool_execution.security import ToolExecutionSecurity
from .tool_observability import ensure_tool_definition_eval_metadata
from .transport.guarded_slides_websocket import (
    is_guarded_slides_websocket_metadata,
)


# JSON-RPC 2.0 Error Codes
class ErrorCode(IntEnum):
    """Standard JSON-RPC 2.0 error codes"""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # Custom error codes (must be -32000 to -32099)
    AUTHENTICATION_ERROR = -32000
    AUTHORIZATION_ERROR = -32001
    RATE_LIMIT_ERROR = -32002
    MODULE_ERROR = -32003
    TIMEOUT_ERROR = -32004


_MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    asyncio.TimeoutError,
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    json.JSONDecodeError,
    RedisError,
    RateLimitExceeded,
    InvalidParamsException,
)

_MCP_TOOL_EXECUTION_ERROR = "tool_execution_error"
_TRUTHY_VALUES = {"1", "true", "yes", "y", "on"}


def _is_truthy(value: Any) -> bool:
    """Parse host-neutral truthy flags without importing tldw_server helpers."""
    try:
        return str(value or "").strip().lower() in _TRUTHY_VALUES
    except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
        return False


def _is_unexpected_keyword_type_error(exc: TypeError, keyword: str) -> bool:
    """Return True when a TypeError reports an unsupported keyword argument."""

    message = str(exc)
    return (
        f"unexpected keyword argument '{keyword}'" in message
        or f'unexpected keyword argument "{keyword}"' in message
    )


def _jsonrpc_id_is_valid(value: Any) -> bool:
    """Return True when a value is a valid JSON-RPC request id."""

    return value is None or (not isinstance(value, bool) and isinstance(value, (str, int)))


def _safe_jsonrpc_id(value: Any) -> str | int | None:
    """Return a response-safe JSON-RPC id, normalizing invalid ids to null."""

    if _jsonrpc_id_is_valid(value):
        return value
    return None


def _validate_jsonrpc_id(value: Any) -> Any:
    """Reject JSON-RPC ids before Pydantic can coerce booleans to integers."""

    if not _jsonrpc_id_is_valid(value):
        raise ValueError("JSON-RPC id must be a string, integer, or null")
    return value


class MCPRequest(BaseModel):
    """MCP request following JSON-RPC 2.0 specification"""
    jsonrpc: Literal["2.0"] = Field(default="2.0")
    method: str = Field(..., min_length=1, max_length=100)
    params: Optional[dict[str, Any]] = None
    id: Optional[Union[str, int]] = None

    if _PYDANTIC_V2:
        @field_validator("id", mode="before")
        @classmethod
        def validate_id(cls, v):
            """Validate JSON-RPC id before type coercion."""
            return _validate_jsonrpc_id(v)
    else:
        @field_validator("id", pre=True)
        @classmethod
        def validate_id(cls, v):
            """Validate JSON-RPC id before type coercion."""
            return _validate_jsonrpc_id(v)

    @field_validator("method")
    @classmethod
    def validate_method(cls, v):
        """Validate method name"""
        # Prevent potential injection attacks
        if any(char in v for char in ["'", '"', ';', '--', '/*', '*/']):
            raise ValueError("Invalid characters in method name")
        return v

    @field_validator("params")
    @classmethod
    def validate_params(cls, v):
        """Validate and sanitize parameters"""
        if v is not None and not isinstance(v, dict):
            raise ValueError("Params must be a dictionary")
        return v


def _mcp_request_has_id(request: Any) -> bool:
    """Return whether a raw or modeled MCP request explicitly included an id."""

    if isinstance(request, dict):
        return "id" in request
    if isinstance(request, MCPRequest):
        fields_set = getattr(request, "model_fields_set", None)
        if fields_set is None:
            fields_set = getattr(request, "__fields_set__", set())
        return "id" in fields_set
    return False


class MCPError(BaseModel):
    """MCP error structure"""
    code: int
    message: str
    data: Optional[Any] = None


class MCPResponse(BaseModel):
    """MCP response following JSON-RPC 2.0 specification"""
    jsonrpc: Literal["2.0"] = Field(default="2.0")
    result: Optional[Any] = None
    error: Optional[MCPError] = None
    id: Optional[Union[str, int]] = None

    if model_validator is not None:
        @model_validator(mode="after")
        def _validate_error_result(self):
            """Ensure either result or error is set, not both"""
            if self.error is not None and self.result is not None:
                raise ValueError("Response cannot have both result and error")
            return self


class MCPProtocol:
    """
    MCP Protocol handler with enhanced security and error handling.

    Features:
    - JSON-RPC 2.0 compliance
    - Request validation and sanitization
    - Authentication and authorization
    - Rate limiting
    - Request routing
    - Error handling with proper codes
    - Request tracing
    """

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        if name == "prepare_tool_call" and callable(value) and hasattr(self, "_tool_execution"):
            if (
                getattr(value, "__self__", None) is self
                and getattr(value, "__func__", None) is type(self).prepare_tool_call
            ):
                self._tool_execution.prepare_tool_call_impl = self._prepare_tool_call_inline
            else:
                self._tool_execution.prepare_tool_call_impl = value
        elif name == "execute_prepared_tool_call" and callable(value) and hasattr(self, "_tool_execution"):
            if (
                getattr(value, "__self__", None) is self
                and getattr(value, "__func__", None) is type(self).execute_prepared_tool_call
            ):
                self._tool_execution.execute_prepared_tool_call_impl = self._execute_prepared_tool_call_inline
            else:
                self._tool_execution.execute_prepared_tool_call_impl = value
        elif name == "_tool_call_hook_manager" and hasattr(self, "_tool_execution_hooks"):
            self._tool_execution_hooks._tool_call_hook_manager = value
            if hasattr(self, "_tool_execution_security"):
                self._sync_tool_execution_dependencies()
        elif (
            name in {
                "module_registry",
                "rbac_policy",
                "rate_limiter",
                "metrics",
                "_tool_use_recorder",
                "_idempotency",
                "_tool_name_re",
            }
            and hasattr(self, "_tool_execution_security")
        ):
            self._sync_tool_execution_dependencies()
        elif name == "_build_tool_use_event" and hasattr(self, "_tool_execution_reporter"):
            if (
                getattr(value, "__self__", None) is self
                and getattr(value, "__func__", None) is type(self)._build_tool_use_event
            ):
                self._tool_execution_reporter.build_event = self._tool_execution_reporter.build_tool_use_event
            else:
                self._tool_execution_reporter.build_event = value
        elif name == "_record_tool_use_event" and hasattr(self, "_tool_execution_reporter"):
            self._tool_execution_reporter.record_event = value
        elif name == "_should_record_tool_use" and hasattr(self, "_tool_execution_reporter"):
            self._tool_execution_reporter.should_record = value

    def __init__(self, dependencies: MCPRuntimeDependencies | None = None):
        if dependencies is None:
            from .adapters.tldw_runtime import build_default_runtime_dependencies

            dependencies = build_default_runtime_dependencies()
        self.dependencies = dependencies
        self.module_registry = self.dependencies.module_registry
        self.rbac_policy = self.dependencies.rbac_policy
        self.rate_limiter = self.dependencies.rate_limiter
        self.tool_catalog_provider = self.dependencies.tool_catalog_provider
        self._tool_use_recorder = (
            getattr(
                self.dependencies,
                "tool_use_recorder",
                NoopToolUseRecorder(),
            )
            or NoopToolUseRecorder()
        )
        self._tool_call_hook_manager = (
            getattr(
                self.dependencies,
                "tool_call_hook_manager",
                NoopToolCallHookManager(),
            )
            or NoopToolCallHookManager()
        )
        self.protocol_version = "2024-11-05"
        self.metrics = self.dependencies.metrics_collector
        # Strict tool name validation regex
        self._tool_name_re = re.compile(r'^[A-Za-z0-9_.:-]{1,100}$')
        # Idempotency manager for write-capable tools
        self._idempotency = IdempotencyManager(
            redis_client_factory=self.dependencies.redis_client_factory,
            on_degraded=lambda stage, _error_type: self.metrics.record_idempotency_degraded(
                stage
            ),
        )
        # Integrity secret for prepared tool call execution
        self._prepared_call_secret = secrets.token_bytes(32)
        self._tool_execution_reporter = ToolExecutionReporter(
            recorder=self._tool_use_recorder,
            metrics=self.metrics,
            tool_name_re=self._tool_name_re,
            noncritical_exceptions=_MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS,
        )
        self._tool_execution_hooks = ToolExecutionHooks(
            hook_manager=self._tool_call_hook_manager,
            reporter=self._tool_execution_reporter,
            noncritical_exceptions=_MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS,
        )
        self._tool_execution_dependencies = ToolExecutionDependencies(
            module_registry=self.module_registry,
            rbac_policy=self.rbac_policy,
            rate_limiter=self.rate_limiter,
            metrics=self.metrics,
            telemetry=self.telemetry,
            hook_manager=self._tool_call_hook_manager,
            tool_use_recorder=self._tool_use_recorder,
            idempotency=self._idempotency,
            config_provider=lambda: get_config(),
            effective_policy_resolver=self.dependencies.effective_policy_resolver,
            path_scope_enforcer=self.dependencies.path_scope_enforcer,
            approval_evaluator=self.dependencies.approval_evaluator,
            external_access_evaluator=self.dependencies.external_access_evaluator,
            reporter=self._tool_execution_reporter,
            api_key_scope_normalizer=getattr(self.dependencies, "api_key_scope_normalizer", None),
        )
        self._tool_execution_security = ToolExecutionSecurity(
            dependencies=self._tool_execution_dependencies,
            tool_name_re=self._tool_name_re,
            prepared_call_secret=self._prepared_call_secret,
            noncritical_exceptions=_MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS,
        )
        self._tool_execution_security.configure_prepare_compatibility_callbacks(
            module_registry=lambda: self.module_registry,
            is_tool_allowed_by_context=(
                lambda tool_name, tool_args, context: self._is_tool_allowed_by_context(tool_name, tool_args, context)
            ),
            resolve_effective_tool_policy=lambda context: self._resolve_effective_tool_policy(context),
            is_tool_allowed_by_effective_policy=(
                lambda tool_name, tool_args, policy: self._is_tool_allowed_by_effective_policy(
                    tool_name,
                    tool_args,
                    policy,
                )
            ),
            evaluate_external_access=lambda **kwargs: self._evaluate_external_access(**kwargs),
            has_module_permission=lambda context, module_id: self._has_module_permission(context, module_id),
            has_tool_permission=lambda context, tool_name, **kwargs: self._has_tool_permission(
                context,
                tool_name,
                **kwargs,
            ),
            make_idempotency_cache_key=lambda context, module_name, tool_name, idempotency_key: (
                self._make_idempotency_cache_key(context, module_name, tool_name, idempotency_key)
            ),
            extract_path_scope_candidates=lambda **kwargs: self._extract_path_scope_candidates(**kwargs),
            evaluate_path_scope=lambda **kwargs: self._evaluate_path_scope(**kwargs),
            evaluate_runtime_approval=lambda **kwargs: self._evaluate_runtime_approval(**kwargs),
            run_governance_preflight=lambda **kwargs: self._run_governance_preflight(**kwargs),
        )

        async def _prepare_with_hooks(
            *,
            params: dict[str, Any],
            context: RequestContext,
            idempotency_key: str | None = None,
        ) -> PreparedToolCall:
            self._sync_tool_execution_dependencies()
            return await self._tool_execution_security.prepare_tool_call(
                params=params,
                context=context,
                idempotency_key=idempotency_key,
                hooks=self._tool_execution_hooks,
            )

        self._prepare_with_hooks = _prepare_with_hooks
        self._tool_execution_runtime = ToolExecutionRuntime(
            dependencies=self._tool_execution_dependencies,
            security=self._tool_execution_security,
            hooks=self._tool_execution_hooks,
            noncritical_exceptions=_MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS,
            tool_execution_error=_MCP_TOOL_EXECUTION_ERROR,
            generic_exception_like=self._generic_exception_like,
            run_post_tool_hooks=lambda **kwargs: self._run_post_tool_hooks(**kwargs),
        )
        self._tool_execution = ToolExecutionCoordinator(
            prepare_tool_call_impl=_prepare_with_hooks,
            execute_prepared_tool_call_impl=self._tool_execution_runtime.execute_prepared_tool_call,
            reporter=self._tool_execution_reporter,
        )

        # Method handlers — telemetry is accessed via a property so that
        # a shutdown/re-init cycle is picked up automatically.
        self.handlers: dict[str, Callable] = {
            "initialize": self._handle_initialize,
            "notifications/initialized": self._handle_initialized_notification,
            "ping": self._handle_ping,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "resources/list": self._handle_resources_list,
            "resources/read": self._handle_resources_read,
            "prompts/list": self._handle_prompts_list,
            "prompts/get": self._handle_prompts_get,
            "modules/list": self._handle_modules_list,
            "modules/health": self._handle_modules_health,
        }

        logger.info("MCP Protocol handler initialized")

    def _sync_tool_execution_dependencies(self) -> None:
        """Keep extracted tool-execution helpers aligned with mutable protocol test seams."""

        if not hasattr(self, "_tool_execution_security"):
            return
        deps = self._tool_execution_security.dependencies
        deps.module_registry = self.module_registry
        deps.rbac_policy = self.rbac_policy
        deps.rate_limiter = self.rate_limiter
        deps.metrics = self.metrics
        deps.telemetry = self.telemetry
        deps.hook_manager = self._tool_call_hook_manager
        deps.tool_use_recorder = self._tool_use_recorder
        deps.idempotency = self._idempotency
        deps.config_provider = lambda: get_config()
        deps.effective_policy_resolver = self.dependencies.effective_policy_resolver
        deps.path_scope_enforcer = self.dependencies.path_scope_enforcer
        deps.approval_evaluator = self.dependencies.approval_evaluator
        deps.external_access_evaluator = self.dependencies.external_access_evaluator
        self._tool_execution_reporter._tool_use_recorder = self._tool_use_recorder
        self._tool_execution_reporter.metrics = self.metrics
        self._tool_execution_reporter._tool_name_re = self._tool_name_re
        deps.reporter = self._tool_execution_reporter
        self._tool_execution_security.module_registry = self.module_registry
        self._tool_execution_security.rbac_policy = self.rbac_policy
        self._tool_execution_security.metrics = self.metrics
        self._tool_execution_security._tool_name_re = self._tool_name_re
        if hasattr(self, "_tool_execution_runtime"):
            self._tool_execution_runtime.dependencies = deps
            self._tool_execution_runtime.security = self._tool_execution_security
            self._tool_execution_runtime.hooks = self._tool_execution_hooks
            self._tool_execution_runtime.sync_from_dependencies()

    @staticmethod
    def _module_declares_prompts(module: BaseModule) -> bool:
        """Return whether a module can expose MCP prompts without querying catalogs.

        Args:
            module: Registered MCP module instance to inspect.

        Returns:
            True when the module overrides prompt listing hooks and can back
            MCP ``prompts/list`` calls, otherwise False.
        """

        module_type = type(module)
        return any(
            getattr(module_type, method_name, None) is not getattr(BaseModule, method_name, None)
            for method_name in ("get_prompts", "get_prompts_for_context")
        )

    @property
    def telemetry(self) -> TelemetryProvider:
        """Return the injected telemetry provider."""
        return self.dependencies.telemetry_provider

    def _should_record_tool_use(self, context: RequestContext) -> bool:
        """Return whether this protocol path should record tool-use metadata."""
        return self._tool_execution_reporter.should_record_tool_use(context)

    async def _record_tool_use_event(self, event: ToolUseEvent) -> None:
        """Record a tool-use event through the configured recorder."""
        await self._tool_execution_reporter.record_tool_use_event(event)

    def _safe_tool_use_name(self, value: Any) -> str:
        """Return a safe tool name or the unknown sentinel."""
        return self._tool_execution_reporter.safe_tool_use_name(value)

    @staticmethod
    def _tool_use_duration_ms(start_ts: float) -> float:
        """Return elapsed milliseconds from a monotonic-ish wall clock sample."""
        return ToolExecutionReporter.tool_use_duration_ms(start_ts)

    @staticmethod
    def _tool_use_execution_origin_for_failure(status: ToolUseStatus) -> str:
        """Return execution-origin metadata for a failed tool path."""
        return ToolExecutionReporter.tool_use_execution_origin_for_failure(status)

    @staticmethod
    def _tool_use_eval_metadata(
        *,
        payload: dict[str, Any] | None = None,
        tool_def: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Extract safe eval metadata from response payload or tool definition."""
        return ToolExecutionReporter.tool_use_eval_metadata(payload=payload, tool_def=tool_def)

    @staticmethod
    def _tool_use_file_policy_decisions(scope_payload: dict[str, Any] | None) -> list[dict[str, Any]]:
        """Extract redacted file-policy path decisions from a scope payload."""
        return ToolExecutionReporter.tool_use_file_policy_decisions(scope_payload)

    @staticmethod
    def _tool_use_hook_results(metadata: dict[str, Any] | None) -> list[dict[str, Any]]:
        """Consume bounded tool-hook result metadata from request metadata."""
        return ToolExecutionReporter.tool_use_hook_results(metadata)

    @staticmethod
    def _tool_hook_summary_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Return sanitized hook summary rows from a protocol hook payload."""
        return ToolExecutionReporter.tool_hook_summary_items(payload)

    @staticmethod
    def _append_tool_hook_summary(context: RequestContext, payload: dict[str, Any]) -> None:
        """Append safe hook metadata for tool-use reporting."""
        ToolExecutionReporter.append_tool_hook_summary(context, payload)

    @staticmethod
    def _tool_use_decision_grant_outcome(file_policy_decisions: list[dict[str, Any]]) -> str | None:
        """Summarize path-decision grant outcomes with denial precedence."""
        return ToolExecutionReporter.tool_use_decision_grant_outcome(file_policy_decisions)

    @staticmethod
    def _tool_use_value_present(value: Any) -> bool:
        """Return whether a sensitive value marker contains actual data."""
        return ToolExecutionReporter.tool_use_value_present(value)

    @staticmethod
    def _tool_use_contains_key(
        value: Any,
        keys: set[str],
        *,
        _depth: int = 0,
    ) -> bool:
        """Return whether a nested tool payload/args object contains any key."""
        return ToolExecutionReporter.tool_use_contains_key(value, keys, _depth=_depth)

    @staticmethod
    def _tool_use_category(tool_def: dict[str, Any] | None) -> str | None:
        """Return the metadata category from a tool definition when present."""
        return ToolExecutionReporter.tool_use_category(tool_def)

    def _build_tool_use_event(
        self,
        *,
        context: RequestContext,
        requested_tool_name: Any,
        status: ToolUseStatus,
        execution_origin: str,
        duration_ms: float,
        effective_tool_name: Any | None = None,
        module_id: str | None = None,
        tool_def: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        tool_args: Any | None = None,
        scope_payload: dict[str, Any] | None = None,
        is_write: bool | None = None,
        reason_code: str | None = None,
        idempotency_replay: bool = False,
    ) -> ToolUseEvent:
        """Build a metadata-only tool-use event."""
        return self._tool_execution_reporter.build_event(
            context=context,
            requested_tool_name=requested_tool_name,
            status=status,
            execution_origin=execution_origin,
            duration_ms=duration_ms,
            effective_tool_name=effective_tool_name,
            module_id=module_id,
            tool_def=tool_def,
            payload=payload,
            tool_args=tool_args,
            scope_payload=scope_payload,
            is_write=is_write,
            reason_code=reason_code,
            idempotency_replay=idempotency_replay,
        )

    async def _record_process_request_tool_use_failure(
        self,
        *,
        request: MCPRequest,
        context: RequestContext,
        status: ToolUseStatus,
        reason_code: str,
        start_ts: float,
        requested_tool_name: Any = None,
    ) -> None:
        """Record a tools/call failure that occurs before handler dispatch."""
        await self._tool_execution_reporter.record_process_request_failure(
            request=request,
            context=context,
            status=status,
            reason_code=reason_code,
            start_ts=start_ts,
            requested_tool_name=requested_tool_name,
            should_record=self._should_record_tool_use,
            build_event=self._build_tool_use_event,
            record_event=self._record_tool_use_event,
            duration_ms=self._tool_use_duration_ms,
            execution_origin_for_failure=self._tool_use_execution_origin_for_failure,
        )

    async def _rbac_check(self, user_id: Optional[str], resource: Resource, action: Action, resource_id: Optional[str] = None) -> bool:
        return await self._tool_execution_security.rbac_check(
            user_id,
            resource,
            action,
            resource_id,
            rbac_policy=self.rbac_policy,
        )

    def _scoped_permissions(self, context: RequestContext) -> list[str]:
        return self._tool_execution_security.scoped_permissions(context)

    def _mcp_scopes(self, context: RequestContext) -> list[str]:
        return self._tool_execution_security.mcp_scopes(
            context,
            scoped_permissions=self._scoped_permissions,
        )

    def _api_key_scopes(self, context: RequestContext) -> Optional[set[str]]:
        """Return normalized API key scopes when present on the request context."""
        return self._tool_execution_security.api_key_scopes(context)

    def _resolve_user_db_paths(self, user_id: Optional[str]) -> dict[str, str]:
        try:
            return self.dependencies.database_path_resolver.resolve_user_db_paths(user_id)
        except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
            return {}

    def _api_key_scope_level(self, context: RequestContext) -> Optional[str]:
        return self._tool_execution_security.api_key_scope_level(
            context,
            api_key_scopes=self._api_key_scopes,
        )

    def _api_key_allows(self, context: RequestContext, *, is_write: Optional[bool] = None) -> bool:
        """Gate MCP operations by API key scopes when present."""
        return self._tool_execution_security.api_key_allows(
            context,
            is_write=is_write,
            api_key_scope_level=self._api_key_scope_level,
        )

    def _scope_matches(self, scope: str, resource_kind: str, identifier: Optional[str]) -> bool:
        return self._tool_execution_security.scope_matches(scope, resource_kind, identifier)

    def _scope_allows(self, context: RequestContext, resource_kind: str, identifier: Optional[str]) -> bool:
        return self._tool_execution_security.scope_allows(
            context,
            resource_kind,
            identifier,
            mcp_scopes=self._mcp_scopes,
            scope_matches=self._scope_matches,
        )

    async def _has_module_permission(self, context: RequestContext, module_id: Optional[str]) -> bool:
        return await self._tool_execution_security.has_module_permission(
            context,
            module_id,
            rbac_check=self._rbac_check,
            scope_allows=self._scope_allows,
        )

    async def _has_tool_permission(self, context: RequestContext, tool_name: str, *, is_write: Optional[bool] = None) -> bool:
        return await self._tool_execution_security.has_tool_permission(
            context,
            tool_name,
            is_write=is_write,
            rbac_check=self._rbac_check,
            scope_allows=self._scope_allows,
            api_key_allows=self._api_key_allows,
            tool_authorization_names=self._tool_authorization_names,
        )

    async def _has_resource_permission(self, context: RequestContext, resource_uri: str, module_id: Optional[str]) -> bool:
        if await self._rbac_check(context.user_id, Resource.RESOURCE, Action.READ, resource_uri):
            return self._scope_allows(context, Resource.RESOURCE.value, resource_uri)
        if await self._has_module_permission(context, module_id):
            return self._scope_allows(context, Resource.RESOURCE.value, resource_uri)
        return False

    async def _has_prompt_permission(self, context: RequestContext, prompt_name: str, module_id: Optional[str]) -> bool:
        if await self._rbac_check(context.user_id, Resource.PROMPT, Action.READ, prompt_name):
            return self._scope_allows(context, Resource.PROMPT.value, prompt_name)
        if await self._has_module_permission(context, module_id):
            return self._scope_allows(context, Resource.PROMPT.value, prompt_name)
        return False

    async def _has_namespaced_prompt_permission(self, context: RequestContext, prompt_name: str) -> bool:
        """Check prompt catalog namespace access without falling back to module permission."""
        if not await self._rbac_check(context.user_id, Resource.PROMPT, Action.READ, prompt_name):
            return False
        if not self._scope_allows(context, Resource.PROMPT.value, prompt_name):
            return False
        return self._api_key_allows(context, is_write=None)

    @staticmethod
    def _prompt_warning_name(warning: dict[str, Any]) -> Optional[str]:
        """Resolve an optional prompt name from catalog warning metadata."""

        explicit_name = warning.get("_prompt_name") or warning.get("prompt_name")
        if isinstance(explicit_name, str) and explicit_name:
            return explicit_name

        source = warning.get("source")
        if source == "library":
            prompt_uuid = warning.get("prompt_uuid")
            if isinstance(prompt_uuid, str) and prompt_uuid:
                return f"{LIBRARY_PROMPT_PREFIX}{prompt_uuid}"
        if source == "config":
            entry_id = warning.get("id")
            if isinstance(entry_id, str) and entry_id:
                return f"{CONFIG_PROMPT_PREFIX}{entry_id}"
        return None

    async def _visible_prompt_warning(
        self,
        context: RequestContext,
        warning: dict[str, Any],
        module_id: Optional[str],
    ) -> Optional[dict[str, Any]]:
        """Return sanitized prompt warning metadata when visible to the caller."""

        prompt_name = self._prompt_warning_name(warning)
        if prompt_name:
            if self._is_namespaced_prompt_name(prompt_name):
                if not await self._has_namespaced_prompt_permission(context, prompt_name):
                    return None
            elif not await self._has_prompt_permission(context, prompt_name, module_id):
                return None

        sanitized = warning.copy()
        for key in ("_prompt_name", "prompt_name", "prompt_uuid", "prompt_id", "id"):
            sanitized.pop(key, None)
        return sanitized

    @staticmethod
    def _is_namespaced_prompt_name(prompt_name: Any) -> bool:
        return (
            isinstance(prompt_name, str)
            and (
                prompt_name.startswith(LIBRARY_PROMPT_PREFIX)
                or prompt_name.startswith(CONFIG_PROMPT_PREFIX)
            )
        )

    def _has_restrictive_prompt_scope(self, context: RequestContext) -> bool:
        """Return whether MCP scopes restrict prompt visibility for this request."""

        scopes = self._mcp_scopes(context)
        if not scopes:
            return False
        for scope in scopes:
            try:
                parts = scope.strip().lower().split(":")
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                continue
            if len(parts) == 2 and parts[0] == "mcp" and parts[1] == "*":
                return False
            if len(parts) >= 3 and parts[0] == "mcp":
                kind = parts[1]
                value = ":".join(parts[2:])
                if kind in {"*", "prompt"} and value in {"", "*"}:
                    return False
        return True

    @staticmethod
    def _prompt_cursor_has_identifier_fields(cursor: Any) -> bool:
        """Return whether a prompt catalog cursor carries prompt identifiers."""

        if not isinstance(cursor, str) or not cursor:
            return False
        try:
            decoded = decode_prompt_cursor(cursor)
        except PromptCatalogError:
            return False
        return decoded.library_after_name is not None or decoded.library_after_uuid is not None

    @staticmethod
    def _hash_arguments(arguments: dict[str, Any]) -> str | None:
        return ToolExecutionSecurity.hash_arguments_with_exceptions(
            arguments,
            noncritical_exceptions=_MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS,
        )

    async def _resolve_tool_definition(
        self,
        module: BaseModule,
        tool_name: str,
    ) -> dict[str, Any] | None:
        return await self._tool_execution_security.resolve_tool_definition(module, tool_name)

    def _classify_write_tool_call(
        self,
        module: BaseModule,
        tool_name: str,
        tool_args: Any,
        tool_def: dict[str, Any] | None,
    ) -> bool | None:
        return self._tool_execution_security.classify_write_tool_call(module, tool_name, tool_args, tool_def)

    def _resolve_write_classification(
        self,
        module: BaseModule,
        tool_name: str,
        tool_args: Any,
        tool_def: dict[str, Any] | None,
        *,
        fallback_to_name_heuristic: bool,
    ) -> bool:
        return self._tool_execution_security.resolve_write_classification(
            module,
            tool_name,
            tool_args,
            tool_def,
            fallback_to_name_heuristic=fallback_to_name_heuristic,
        )

    @staticmethod
    def _strip_forbidden_tool_argument_overrides(tool_args: dict[str, Any]) -> dict[str, Any]:
        return ToolExecutionSecurity.strip_forbidden_tool_argument_overrides(tool_args)

    def _harden_and_sanitize_tool_arguments(
        self,
        module: BaseModule,
        tool_args: Any,
    ) -> Any:
        return self._tool_execution_security.harden_and_sanitize_tool_arguments(module, tool_args)

    def _prepared_tool_call_payload(
        self,
        *,
        tool_name: str,
        module_id: str | None,
        policy: PreparedExecutionPolicy,
        idempotency_cache_key: str | None,
        normalized_idempotency_key_digest: str,
        arguments_hash: str | None,
        context_fingerprint: str,
        idempotency_scope_fingerprint: str,
        tool_definition_sha256: str,
        scope_reporting_sha256: str,
    ) -> bytes:
        return ToolExecutionSecurity.prepared_tool_call_payload(
            tool_name=tool_name,
            module_id=module_id,
            policy=policy,
            idempotency_cache_key=idempotency_cache_key,
            normalized_idempotency_key_digest=normalized_idempotency_key_digest,
            arguments_hash=arguments_hash,
            context_fingerprint=context_fingerprint,
            idempotency_scope_fingerprint=idempotency_scope_fingerprint,
            tool_definition_sha256=tool_definition_sha256,
            scope_reporting_sha256=scope_reporting_sha256,
        )

    def _build_prepared_tool_call_integrity_tag(
        self,
        *,
        tool_name: str,
        module_id: str | None,
        policy: PreparedExecutionPolicy,
        idempotency_cache_key: str | None,
        normalized_idempotency_key_digest: str,
        arguments_hash: str | None,
        context_fingerprint: str,
        idempotency_scope_fingerprint: str,
        tool_definition_sha256: str,
        scope_reporting_sha256: str,
    ) -> str:
        return self._tool_execution_security.build_prepared_tool_call_integrity_tag(
            tool_name=tool_name,
            module_id=module_id,
            policy=policy,
            idempotency_cache_key=idempotency_cache_key,
            normalized_idempotency_key_digest=normalized_idempotency_key_digest,
            arguments_hash=arguments_hash,
            context_fingerprint=context_fingerprint,
            idempotency_scope_fingerprint=idempotency_scope_fingerprint,
            tool_definition_sha256=tool_definition_sha256,
            scope_reporting_sha256=scope_reporting_sha256,
        )

    def _verify_prepared_tool_call_integrity(
        self,
        prepared: PreparedToolCall,
    ) -> None:
        self._tool_execution_security.verify_prepared_tool_call_integrity(prepared)

    def _fingerprint_request_context(self, context: RequestContext) -> str:
        return self._tool_execution_security.fingerprint_request_context(context)

    def _context_json_safe(self, value: Any) -> Any:
        return self._tool_execution_security.context_json_safe(value)

    @staticmethod
    def _normalize_idempotency_key(
        params: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> str | None:
        return ToolExecutionSecurity.normalize_idempotency_key(params, idempotency_key=idempotency_key)

    def _audit_tool_event(
        self,
        context: RequestContext,
        tool_name: str,
        module_id: Optional[str],
        status: str,
        duration_ms: float,
        arguments_hash: Optional[str],
        error: Optional[Exception] = None,
    ) -> None:
        self._tool_execution_reporter.audit_tool_event(
            context,
            tool_name,
            module_id,
            status,
            duration_ms,
            arguments_hash,
            error=error,
        )

    @staticmethod
    def _governance_preflight_bypassed(tool_name: str, context: RequestContext) -> bool:
        return ToolExecutionSecurity._governance_preflight_bypassed(tool_name, context)

    def _governance_summary(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        return self._tool_execution_security._governance_summary(tool_name, tool_args)

    @staticmethod
    def _resolve_governance_category(tool_name: str, tool_def: Optional[dict[str, Any]]) -> str:
        return ToolExecutionSecurity._resolve_governance_category(tool_name, tool_def)

    def _resolve_governance_rollout_mode(self, metadata: Optional[dict[str, Any]] = None) -> str:
        """Resolve governance rollout mode from metadata override and server config."""
        self._sync_tool_execution_dependencies()
        return self._tool_execution_security._resolve_governance_rollout_mode(metadata)

    def _record_governance_check(
        self,
        *,
        surface: str,
        category: str,
        status: str,
        rollout_mode: str,
    ) -> None:
        """Emit one governance check metric entry, failing open on metric errors."""
        self._sync_tool_execution_dependencies()
        self._tool_execution_security._record_governance_check(
            surface=surface,
            category=category,
            status=status,
            rollout_mode=rollout_mode,
        )

    @classmethod
    def _serialize_governance_decision(cls, decision: Any) -> dict[str, Any]:
        return ToolExecutionSecurity._serialize_governance_decision(decision)

    async def _ensure_governance_service(self) -> Any | None:
        self._sync_tool_execution_dependencies()
        return await self._tool_execution_security._ensure_governance_service()

    async def _run_governance_preflight(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_def: Optional[dict[str, Any]],
        context: RequestContext,
    ) -> Optional[dict[str, Any]]:
        self._sync_tool_execution_dependencies()
        return await self._tool_execution_security._run_governance_preflight(
            tool_name=tool_name,
            tool_args=tool_args,
            tool_def=tool_def,
            context=context,
            governance_preflight_bypassed=self._governance_preflight_bypassed,
            resolve_governance_rollout_mode=self._resolve_governance_rollout_mode,
            resolve_governance_category=self._resolve_governance_category,
            record_governance_check=self._record_governance_check,
            governance_summary=self._governance_summary,
            serialize_governance_decision=self._serialize_governance_decision,
            ensure_governance_service=self._ensure_governance_service,
        )

    @staticmethod
    def _hook_safe_copy(value: Any) -> Any:
        """Return a detached copy of hook-visible metadata without failing tool preparation."""
        return ToolExecutionHooks._hook_safe_copy(value)

    @staticmethod
    def _hook_safe_metadata(context: RequestContext) -> dict[str, Any]:
        """Return request metadata safe for local lifecycle hook decisions."""
        return ToolExecutionHooks._hook_safe_metadata(context)

    @staticmethod
    def _hook_safe_tool_args(tool_args: Any, *, tool_name: str | None = None) -> dict[str, Any] | None:
        """Return detached sanitized tool arguments for hook evaluation."""
        return ToolExecutionHooks._hook_safe_tool_args(tool_args, tool_name=tool_name)

    @staticmethod
    def _redact_hook_visible_tool_args(tool_args: dict[str, Any], *, tool_name: str | None = None) -> dict[str, Any]:
        """Redact secret-bearing argument values from hook-visible metadata."""

        return ToolExecutionHooks._redact_hook_visible_tool_args(tool_args, tool_name=tool_name)

    @staticmethod
    def _hook_safe_scope_payload(scope_payload: dict[str, Any] | None) -> dict[str, Any] | None:
        """Return detached path/external scope metadata for hooks."""
        return ToolExecutionHooks._hook_safe_scope_payload(scope_payload)

    def _build_tool_hook_context(
        self,
        *,
        phase: str,
        tool_name: str,
        module_id: str | None,
        tool_def: dict[str, Any] | None,
        tool_args: Any,
        is_write: bool | None,
        arguments_hash: str | None,
        context: RequestContext,
        scope_payload: dict[str, Any] | None = None,
        status: str | None = None,
        duration_ms: float | None = None,
        error: Exception | None = None,
    ) -> ToolHookCallContext:
        """Build a bounded, detached context object for lifecycle hook evaluation."""
        return self._tool_execution_hooks._build_tool_hook_context(
            phase=phase,
            tool_name=tool_name,
            module_id=module_id,
            tool_def=tool_def,
            tool_args=tool_args,
            is_write=is_write,
            arguments_hash=arguments_hash,
            context=context,
            scope_payload=scope_payload,
            status=status,
            duration_ms=duration_ms,
            error=error,
        )

    @staticmethod
    def _coerce_tool_hook_action(action: Any) -> ToolHookAction | None:
        """Normalize a runtime hook action into the public literal contract."""
        return ToolExecutionHooks._coerce_tool_hook_action(action)

    @staticmethod
    def _coerce_tool_hook_decision(
        decision: ToolHookDecision | dict[str, Any] | None,
    ) -> ToolHookDecision:
        """Normalize hook decision values from typed or dict-based embedders."""
        return ToolExecutionHooks._coerce_tool_hook_decision(decision)

    @staticmethod
    def _tool_hook_payload(
        decision: ToolHookDecision,
        *,
        phase: str,
        fallback_reason_code: str | None = None,
    ) -> dict[str, Any]:
        """Serialize a normalized hook decision into response-safe metadata."""
        return ToolExecutionHooks._tool_hook_payload(
            decision,
            phase=phase,
            fallback_reason_code=fallback_reason_code,
        )

    async def _run_pre_tool_hooks(
        self,
        *,
        tool_name: str,
        tool_args: Any,
        module_id: str | None,
        tool_def: dict[str, Any] | None,
        is_write: bool | None,
        arguments_hash: str | None,
        context: RequestContext,
        scope_payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Run pre-tool hooks and map enforcement decisions to protocol errors."""
        return await self._tool_execution_hooks._run_pre_tool_hooks(
            tool_name=tool_name,
            module_id=module_id,
            tool_def=tool_def,
            tool_args=tool_args,
            is_write=is_write,
            arguments_hash=arguments_hash,
            context=context,
            scope_payload=scope_payload,
        )

    async def _run_post_tool_hooks(
        self,
        *,
        tool_name: str,
        tool_args: Any,
        module_id: str | None,
        tool_def: dict[str, Any] | None,
        is_write: bool | None,
        arguments_hash: str | None,
        context: RequestContext,
        scope_payload: dict[str, Any] | None,
        status: str,
        duration_ms: float,
        error: Exception | None = None,
    ) -> None:
        """Notify post-tool hooks while preserving the original tool outcome."""
        await self._tool_execution_hooks._run_post_tool_hooks(
            tool_name=tool_name,
            tool_args=tool_args,
            module_id=module_id,
            tool_def=tool_def,
            is_write=is_write,
            arguments_hash=arguments_hash,
            context=context,
            scope_payload=scope_payload,
            status=status,
            duration_ms=duration_ms,
            error=error,
        )

    async def process_request(
        self,
        request: Union[dict[str, Any], list[dict[str, Any]], MCPRequest],
        context: Optional[RequestContext] = None
    ) -> Union[MCPResponse, list[MCPResponse], None]:
        """
        Process an MCP request and return response.

        Args:
            request: MCP request (dict or MCPRequest object)
            context: Request context with user/session info

        Returns:
            MCP response
        """
        # Support batch requests
        if isinstance(request, list):
            if not request:
                return self._error_response(
                    ErrorCode.INVALID_REQUEST,
                    "Invalid request: empty batch",
                    None,
                )
            responses: list[MCPResponse] = []
            for item in request:
                try:
                    resp = await self.process_request(item, context)
                    # Notifications return None; do not include in batch response
                    if isinstance(resp, MCPResponse):
                        responses.append(resp)
                except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as e:
                    # If parsing fails at top-level, try to include an error response for that item
                    try:
                        req_id = _safe_jsonrpc_id(item.get("id")) if isinstance(item, dict) else None
                    except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                        req_id = None
                    responses.append(self._error_response(ErrorCode.INVALID_REQUEST, str(e), req_id))
            # Per JSON-RPC, if the batch is empty or only notifications, return no response
            return responses if responses else None

        raw_request_has_id = _mcp_request_has_id(request)

        # Parse single request if dict
        if isinstance(request, dict):
            try:
                request = MCPRequest(**request)
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as e:
                req_id = _safe_jsonrpc_id(request.get("id")) if isinstance(request, dict) else None
                return self._error_response(
                    ErrorCode.INVALID_REQUEST,
                    f"Invalid request format: {str(e)}",
                    req_id
                )

        is_notification = isinstance(request, MCPRequest) and request.id is None and not raw_request_has_id

        def pre_dispatch_error(
            code: ErrorCode,
            message: str,
            request_id: Optional[Union[str, int]],
            data: Optional[Any] = None,
        ) -> MCPResponse | None:
            if is_notification:
                return None
            return self._error_response(code, message, request_id, data=data)

        # Create context if not provided
        if context is None:
            context = RequestContext(
                request_id=str(uuid.uuid4()),
                client_id="unknown",
                db_paths=self._resolve_user_db_paths(None),
            )

        # Bound logger for this request
        log = context.logger
        # Log request (without params) and ensure secrets get redacted in any error paths
        log.info(
            f"MCP request: method={request.method}, user={context.user_id}, client={context.client_id}",
            extra={"audit": True}
        )

        start_ts = time.time()
        handler_started = False
        try:
            # Check rate limit (skip when ingress RG already enforced)
            skip_rate_limit = False
            try:
                if context.metadata and context.metadata.get("rg_ingress_enforced"):
                    skip_rate_limit = True
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as exc:
                log.debug(
                    "Failed to read rg_ingress_enforced from metadata; rate limit will be enforced: {error_type}",
                    error_type=type(exc).__name__,
                )
                skip_rate_limit = False
            if not skip_rate_limit:
                if context.user_id:
                    await self.rate_limiter.check_rate_limit(f"user:{context.user_id}")
                elif context.client_id:
                    await self.rate_limiter.check_rate_limit(f"client:{context.client_id}")

            # Validate JSON-RPC version
            if request.jsonrpc != "2.0":
                return pre_dispatch_error(
                    ErrorCode.INVALID_REQUEST,
                    "Invalid JSON-RPC version",
                    request.id
                )

            # If this is a tools/call, validate tool name early (before RBAC)
            try:
                if request.method == "tools/call":
                    _p = request.params or {}
                    _name = _p.get("name") if isinstance(_p, dict) else None
                    if not _name:
                        await self._record_process_request_tool_use_failure(
                            request=request,
                            context=context,
                            status="invalid_params",
                            reason_code="tool_name_required",
                            start_ts=start_ts,
                            requested_tool_name=_name,
                        )
                        return pre_dispatch_error(
                            ErrorCode.INVALID_PARAMS,
                            "Tool name is required",
                            request.id,
                        )
                    if not isinstance(_name, str):
                        # Non-string name → invalid params
                        await self._record_process_request_tool_use_failure(
                            request=request,
                            context=context,
                            status="invalid_params",
                            reason_code="invalid_tool_name",
                            start_ts=start_ts,
                            requested_tool_name=_name,
                        )
                        return pre_dispatch_error(
                            ErrorCode.INVALID_PARAMS,
                            "Invalid tool name",
                            request.id,
                        )
                    if not self._tool_name_re.match(_name):
                        await self._record_process_request_tool_use_failure(
                            request=request,
                            context=context,
                            status="invalid_params",
                            reason_code="invalid_tool_name",
                            start_ts=start_ts,
                            requested_tool_name=_name,
                        )
                        return pre_dispatch_error(
                            ErrorCode.INVALID_PARAMS,
                            "Invalid tool name",
                            request.id,
                        )
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                # Uniformly surface as INVALID_PARAMS for caller clarity
                await self._record_process_request_tool_use_failure(
                    request=request,
                    context=context,
                    status="invalid_params",
                    reason_code="invalid_tool_name",
                    start_ts=start_ts,
                )
                return pre_dispatch_error(ErrorCode.INVALID_PARAMS, "Invalid tool name", request.id)

            # Find handler
            handler = self.handlers.get(request.method)
            if not handler:
                return pre_dispatch_error(
                    ErrorCode.METHOD_NOT_FOUND,
                    f"Method not found: {request.method}",
                    request.id
                )

            # Check authorization
            if not await self._check_authorization(request, context):
                # Provide a short hint for common denied operations
                hint_data = None
                try:
                    if request.method == "tools/call":
                        tool = (request.params or {}).get("name")
                        if tool:
                            hint_data = {
                                "hint": (
                                    f"Permission denied. Ask an admin to grant tools.execute:{tool} "
                                    f"or tools.execute:* to your role (Admin → Access Control)."
                                )
                            }
                except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                    hint_data = None

                await self._record_process_request_tool_use_failure(
                    request=request,
                    context=context,
                    status="denied",
                    reason_code="permission_denied",
                    start_ts=start_ts,
                )
                return pre_dispatch_error(
                    ErrorCode.AUTHORIZATION_ERROR,
                    "Insufficient permissions",
                    request.id,
                    data=hint_data
                )

            # Execute handler within OTEL span
            start_exec = time.time()
            with self.telemetry.trace_context(
                "mcp.request",
                {
                    "mcp.method": request.method,
                    "mcp.request_id": str(request.id) if request.id is not None else ("notification" if is_notification else "null"),
                    "mcp.user_id": str(context.user_id or ""),
                    "mcp.client_id": str(context.client_id or ""),
                    "mcp.session_id": str(context.session_id or ""),
                },
            ) as span:
                try:
                    handler_started = True
                    result = await handler(request.params or {}, context)
                    span.set_attribute("mcp.status", "success")
                except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as _span_e:
                    sanitized = self._sanitize_exception_for_telemetry(_span_e)
                    span.set_attribute("mcp.status", "failure")
                    span.set_attribute("mcp.error_type", sanitized.__class__.__name__)
                    span.set_attribute("mcp.error_message", str(sanitized)[:200])
                    raise
                except Exception as _span_e:
                    sanitized = self._sanitize_exception_for_telemetry(_span_e)
                    span.set_attribute("mcp.status", "failure")
                    span.set_attribute("mcp.error_type", sanitized.__class__.__name__)
                    span.set_attribute("mcp.error_message", str(sanitized)[:200])
                    raise
                finally:
                    span.set_attribute("mcp.duration_ms", max(0.0, (time.time() - start_exec) * 1000.0))

            # Log success and record metrics
            elapsed = (datetime.now(timezone.utc) - context.start_time).total_seconds()
            log.info(
                f"MCP request completed: method={request.method}, "
                f"elapsed={elapsed:.3f}s",
                extra={"audit": True}
            )
            with contextlib.suppress(_MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS):
                self.metrics.record_request(method=request.method, duration=elapsed, status="success")

            # Notification: do not return a response
            if is_notification:
                return None
            # Return success response for standard requests
            return MCPResponse(result=result, id=request.id)

        except RateLimitExceeded:
            # Record rate limit hit and re-raise for caller-specific mapping
            try:
                key_type = "user" if context.user_id else ("client" if context.client_id else "anonymous")
                self.metrics.record_rate_limit_hit(key_type=key_type)
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                pass
            if isinstance(request, MCPRequest) and not handler_started:
                await self._record_process_request_tool_use_failure(
                    request=request,
                    context=context,
                    status="rate_limited",
                    reason_code="rate_limited",
                    start_ts=start_ts,
                )
            raise
        except InvalidParamsException as ive:
            # Notification: do not return a response
            if is_notification:
                return None
            return self._error_response(ErrorCode.INVALID_PARAMS, str(ive), request.id if isinstance(request, MCPRequest) else None)
        except PermissionError as perr:
            # Map policy/permission errors to AUTHORIZATION_ERROR
            if is_notification:
                return None
            # Redact any secrets in message (defensive)
            msg = self._mask_secrets(str(perr))
            error_data = None
            if isinstance(perr, GovernanceDeniedError):
                error_data = {"governance": dict(perr.governance or {})}
            elif isinstance(perr, ApprovalRequiredError):
                error_data = {"approval": dict(perr.approval or {})}
            return self._error_response(
                ErrorCode.AUTHORIZATION_ERROR,
                msg,
                request.id if isinstance(request, MCPRequest) else None,
                data=error_data,
            )
        except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as e:
            # Log error
            masked = self._mask_secrets(str(e))
            secret_redacted = bool(getattr(e, "_mcp_masked_secret", False)) or masked != str(e)
            if secret_redacted:
                log.error(  # noqa: TRY400 - avoid logging sanitized exception traceback.
                    f"MCP request failed: method={request.method}, error={masked}",
                    extra={"audit": True},
                )
            else:
                log.exception(
                    f"MCP request failed: method={request.method}, error={masked}",
                    extra={"audit": True},
                )
            try:
                elapsed = max(0.0, time.time() - start_ts)
                self.metrics.record_request(method=request.method, duration=elapsed, status="failure")
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                pass

            # Notification: do not return a response
            if is_notification:
                return None
            # Return error response with reduced leakage when not in debug mode
            try:
                cfg = get_config()
                debug_mode = bool(getattr(cfg, "debug_mode", False))
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                debug_mode = False
            msg = masked if debug_mode and not secret_redacted else "Internal error"
            return self._error_response(
                ErrorCode.INTERNAL_ERROR,
                msg,
                request.id if isinstance(request, MCPRequest) else None,
            )
        except Exception as e:
            masked = self._mask_secrets(str(e))
            secret_redacted = bool(getattr(e, "_mcp_masked_secret", False)) or masked != str(e)
            if secret_redacted:
                log.error(  # noqa: TRY400 - avoid logging sanitized exception traceback.
                    f"MCP request failed: method={request.method}, error={masked}",
                    extra={"audit": True},
                )
            else:
                log.exception(
                    f"MCP request failed: method={request.method}, error={masked}",
                    extra={"audit": True},
                )
            with contextlib.suppress(_MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS):
                elapsed = max(0.0, time.time() - start_ts)
                self.metrics.record_request(method=request.method, duration=elapsed, status="failure")
            if is_notification:
                return None
            return self._error_response(
                ErrorCode.INTERNAL_ERROR,
                "Internal error",
                request.id if isinstance(request, MCPRequest) else None,
            )

    def _mask_secrets(self, text: str) -> str:
        """Best-effort masking of bearer/API keys in strings."""
        try:
            if not text:
                return text
            import re as _re
            # Mask Bearer tokens
            text = _re.sub(r"(Bearer)\s+[A-Za-z0-9._\-~+/=]+", r"\1 ****", text, flags=_re.IGNORECASE)
            # Mask common token fields
            patterns = [
                r"(api[_-]?key)\s*[:=]\s*([^\s,;]+)",
                r"(token)\s*[:=]\s*([^\s,;]+)",
                r"(access[_-]?token)\s*[:=]\s*([^\s,;]+)",
                r"(refresh[_-]?token)\s*[:=]\s*([^\s,;]+)",
            ]
            for p in patterns:
                text = _re.sub(p, lambda m: f"{m.group(1)}=****", text, flags=_re.IGNORECASE)
            return text
        except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
            return text

    def _sanitize_exception_for_telemetry(self, exc: Exception) -> Exception:
        """Redact secret-bearing exception messages before tracing records them."""
        original = str(exc)
        masked = self._mask_secrets(original)
        if masked == original:
            with contextlib.suppress(Exception):
                exc._mcp_masked_secret = False
            return exc
        try:
            sanitized = exc.__class__(masked)
        except Exception:
            sanitized = RuntimeError(masked)
        with contextlib.suppress(Exception):
            sanitized._mcp_masked_secret = True
        if hasattr(sanitized, "__dict__") and hasattr(exc, "__dict__"):
            with contextlib.suppress(Exception):
                for attr in ("errno", "code", "name", "lineno"):
                    if hasattr(exc, attr):
                        setattr(sanitized, attr, getattr(exc, attr))
        with contextlib.suppress(Exception):
            sanitized.args = (masked,)
        with contextlib.suppress(Exception):
            sanitized._mcp_masked_secret = True
        return sanitized

    @staticmethod
    def _generic_exception_like(exc: Exception, message: str) -> Exception:
        """Return an exception of the same class when possible, with safe text only."""
        try:
            sanitized = exc.__class__(message)
        except Exception:
            sanitized = RuntimeError(message)
        with contextlib.suppress(Exception):
            sanitized._mcp_sanitized_error = True
        return sanitized

    def _error_response(
        self,
        code: ErrorCode,
        message: str,
        request_id: Optional[Union[str, int]] = None,
        data: Optional[Any] = None
    ) -> MCPResponse:
        """Create an error response"""
        data = self._attach_error_hint(code, message, data)
        return MCPResponse(
            error=MCPError(
                code=code,
                message=message,
                data=data
            ),
            id=request_id
        )

    def _attach_error_hint(
        self,
        code: ErrorCode,
        message: str,
        data: Optional[Any]
    ) -> Optional[Any]:
        """Attach a structured hint for common error scenarios."""
        metadata = self._error_recovery_metadata(code, message)
        if data is not None:
            if isinstance(data, dict) and metadata and not any(key in data for key in ("governance", "approval")):
                merged = dict(data)
                for key, value in metadata.items():
                    merged.setdefault(key, value)
                return merged
            return data

        hint: Optional[str] = None
        lowered = message.lower()

        if code == ErrorCode.INVALID_PARAMS:
            prefix = "missing required parameter:"
            if lowered.startswith(prefix):
                # Extract the parameter name from original message
                try:
                    missing = message.split(":", 1)[1].strip().strip("'\"")
                except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                    missing = None
                if missing:
                    hint = f"Add '{missing}' to the tool arguments payload before retrying."
            elif "invalid parameters for tool" in lowered:
                hint = "Verify the tool arguments match the schema published by /mcp/tools."
        elif code == ErrorCode.AUTHORIZATION_ERROR and "write tools are disabled" in lowered:
            hint = "Enable write tools (set MCP_DISABLE_WRITE_TOOLS=0) or switch to a read-only operation."

        recovery: dict[str, Any] = dict(metadata or {})
        if hint:
            recovery["hint"] = hint
        return recovery or None

    @staticmethod
    def _error_recovery_metadata(code: ErrorCode, message: str) -> dict[str, str] | None:
        """Return additive recovery metadata for known JSON-RPC errors."""
        lowered = message.lower()
        if code == ErrorCode.INVALID_PARAMS:
            return {
                "reason_code": "invalid_params",
                "next_action": "Check the method parameters or tool input schema from tools/list before retrying.",
            }
        if code == ErrorCode.AUTHENTICATION_ERROR:
            return {
                "reason_code": "authentication_required",
                "next_action": "Send Authorization: Bearer <token> or X-API-KEY with the request.",
            }
        if code == ErrorCode.AUTHORIZATION_ERROR:
            if "write tools are disabled" in lowered:
                return {
                    "reason_code": "write_tools_disabled",
                    "next_action": (
                        "Enable write tools (set MCP_DISABLE_WRITE_TOOLS=0) "
                        "or switch to a read-only operation."
                    ),
                }
            return {
                "reason_code": "permission_denied",
                "next_action": "Use a token or API key with the required MCP permission.",
            }
        if code == ErrorCode.MODULE_ERROR:
            return {
                "reason_code": "module_unavailable",
                "next_action": "Check /api/v1/mcp/status for problem_modules.",
            }
        if code == ErrorCode.TIMEOUT_ERROR:
            return {
                "reason_code": "upstream_unavailable",
                "next_action": "Retry after checking the upstream module or external service.",
            }
        return None

    async def _check_authorization(
        self,
        request: MCPRequest,
        context: RequestContext
    ) -> bool:
        """Check if user is authorized for method"""
        # Public methods that don't require auth
        public_methods = ["initialize", "notifications/initialized", "ping"]
        method = request.method
        if method in public_methods:
            return True

        # Admin override (e.g., endpoint-level admin guard) for certain methods
        try:
            if isinstance(getattr(context, "metadata", None), dict):
                if context.metadata.get("admin_override") is True and request.method in {"modules/health"}:
                    return True
        except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
            pass
        # No user context means no auth
        if not context.user_id:
            return False

        # tools/list: allow any authenticated user (deny if unauthenticated)
        if method == "tools/list":
            if not context.user_id:
                return False
            if not self._scope_allows(context, Resource.TOOL.value, None):
                return False
            return self._api_key_allows(context, is_write=None)

        # Map methods to resources and actions
        method_permissions = {
            # tools/list handled above
            "tools/call": (Resource.TOOL, Action.EXECUTE),
            "resources/list": (Resource.RESOURCE, Action.READ),
            "resources/read": (Resource.RESOURCE, Action.READ),
            "prompts/list": (Resource.PROMPT, Action.READ),
            "prompts/get": (Resource.PROMPT, Action.READ),
            "modules/list": (Resource.MODULE, Action.READ),
            "modules/health": (Resource.MODULE, Action.READ),
        }

        if method in method_permissions:
            resource, action = method_permissions[method]
            fn = getattr(self.rbac_policy, 'check_permission', None)
            if fn is None:
                return False
            # Provide resource_id (e.g., tool name) when applicable
            resource_id = None
            try:
                if resource == Resource.TOOL and action == Action.EXECUTE:
                    params = request.params or {}
                    name = params.get("name") if isinstance(params, dict) else None
                    if isinstance(name, str) and name:
                        resource_id = name
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                resource_id = None
            if resource == Resource.TOOL and action == Action.EXECUTE:
                if not isinstance(resource_id, str):
                    return False
                tool_def = None
                module = None
                is_write = None
                try:
                    tool_args = params.get("arguments", {}) if isinstance(params, dict) else {}
                    module = await self.module_registry.find_module_for_tool(resource_id)
                    if module is not None:
                        tool_def = await self._resolve_tool_definition(module, resource_id)
                        tool_args = self._harden_and_sanitize_tool_arguments(module, tool_args)
                        is_write = self._resolve_write_classification(
                            module,
                            resource_id,
                            tool_args,
                            tool_def,
                            fallback_to_name_heuristic=True,
                        )
                    else:
                        is_write = bool(re.search(r"(ingest|update|delete|create|import)", resource_id.lower()))
                except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                    is_write = None
                return await self._has_tool_permission(context, resource_id, is_write=is_write)
            trusted_compat_allowed = False
            if _has_trusted_compat_claims(context):
                if resource == Resource.MODULE:
                    trusted_compat_allowed = await self._has_module_permission(
                        context,
                        resource_id if isinstance(resource_id, str) else None,
                    )
                elif resource in {Resource.RESOURCE, Resource.PROMPT}:
                    trusted_compat_allowed = self._scope_allows(
                        context,
                        resource.value,
                        resource_id if isinstance(resource_id, str) else None,
                    )
            if trusted_compat_allowed:
                allowed = True
            elif inspect.iscoroutinefunction(fn):
                allowed = await fn(context.user_id, resource, action, resource_id)
            else:
                allowed = fn(context.user_id, resource, action, resource_id)
            if not allowed:
                return False
            if not self._scope_allows(context, resource.value, resource_id):
                return False
            # Apply API key scope gating for read-style methods
            return self._api_key_allows(context, is_write=None)

        # Unknown method - deny by default
        return False

    # Protocol method handlers

    async def _handle_initialize(
        self,
        params: dict[str, Any],
        context: RequestContext
    ) -> dict[str, Any]:
        """Handle initialize request"""
        client_info = params.get("clientInfo", {})

        logger.info(f"Client initializing: {client_info}")

        # Get server capabilities
        modules = await self.module_registry.get_all_modules()
        has_prompt_module = any(self._module_declares_prompts(module) for module in modules.values())

        capabilities = {
            "tools": {"available": bool(modules)},
            "resources": {"available": bool(modules)},
            "prompts": {"available": has_prompt_module, "listChanged": False}
        }

        return {
            "protocolVersion": self.protocol_version,
            "capabilities": capabilities,
            "serverInfo": {
                "name": "tldw-mcp-unified",
                "version": "3.0.0"
            }
        }

    async def _handle_initialized_notification(
        self,
        params: dict[str, Any],
        context: RequestContext,
    ) -> None:
        """Accept the MCP initialized notification without side effects."""

        return None

    async def _handle_ping(
        self,
        params: dict[str, Any],
        context: RequestContext
    ) -> dict[str, Any]:
        """Handle ping request"""
        return {"pong": True, "timestamp": datetime.now(timezone.utc).isoformat()}

    async def _resolve_catalog_tool_names(
        self,
        params: dict[str, Any],
        context: RequestContext
    ) -> Optional[set[str]]:
        """Resolve catalog parameter into a set of tool names for filtering."""
        strict = False
        fail_open = False
        if isinstance(params, dict):
            raw_strict = params.get("catalog_strict")
            if isinstance(raw_strict, bool):
                strict = raw_strict
            elif isinstance(raw_strict, (int, float)):
                strict = bool(raw_strict)
            elif isinstance(raw_strict, str):
                strict = _is_truthy(raw_strict)
            raw_fail_open = params.get("catalog_fail_open")
            if isinstance(raw_fail_open, bool):
                fail_open = raw_fail_open
            elif isinstance(raw_fail_open, (int, float)):
                fail_open = bool(raw_fail_open)
            elif isinstance(raw_fail_open, str):
                fail_open = _is_truthy(raw_fail_open)
        catalog_name = None
        catalog_id = None
        if isinstance(params, dict):
            catalog_name = params.get("catalog")
            catalog_id = params.get("catalog_id")
        if catalog_name is None and catalog_id is None:
            return None
        try:
            metadata = context.metadata if isinstance(getattr(context, "metadata", None), dict) else {}
            effective_fail_open = fail_open and not strict
            resolved = await self.tool_catalog_provider.resolve_tool_names(
                catalog_name=catalog_name if isinstance(catalog_name, str) else None,
                catalog_id=catalog_id,
                metadata=metadata,
                strict=strict,
            )
            if resolved is None and not effective_fail_open:
                return set()
            return resolved
        except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as exc:
            context.logger.debug(
                "Catalog lookup unavailable; returning fallback: {}",
                exc.__class__.__name__,
            )
            return None if fail_open and not strict else set()

    def _catalog_filter_requested(self, params: dict[str, Any]) -> bool:
        """Return true when the caller requested catalog-scoped discovery."""
        if not isinstance(params, dict):
            return False
        return params.get("catalog") is not None or params.get("catalog_id") is not None

    def _catalog_filter_metadata(
        self,
        params: dict[str, Any],
        catalog_filter: Optional[set[str]],
    ) -> dict[str, Any] | None:
        """Describe catalog filter resolution for clients."""
        if not self._catalog_filter_requested(params):
            return None
        if catalog_filter is None:
            return {
                "status": "fail_open",
                "filtered": False,
                "hint": "Catalog lookup was bypassed by catalog_fail_open=true.",
            }
        if not catalog_filter:
            return {
                "status": "unresolved",
                "filtered": True,
                "toolCount": 0,
                "hint": "Check catalog name/id or remove the catalog filter.",
            }
        return {
            "status": "resolved",
            "filtered": True,
            "toolCount": len(catalog_filter),
        }

    async def _handle_tools_list(
        self,
        params: dict[str, Any],
        context: RequestContext
    ) -> dict[str, Any]:
        """List available tools"""
        tools = []
        catalog_filter = await self._resolve_catalog_tool_names(params, context)
        catalog_meta = self._catalog_filter_metadata(params, catalog_filter)
        modules = await self.module_registry.get_all_modules()
        module_filter = None
        if isinstance(params, dict):
            module_filter = params.get("module")
        allowed_modules: Optional[set[str]] = None
        if isinstance(module_filter, str) and module_filter.strip():
            allowed_modules = {module_filter.strip()}
        elif isinstance(module_filter, list):
            allowed_modules = {str(m).strip() for m in module_filter if str(m).strip()}

        for module_id, module in modules.items():
            if module_id.lower() == "slides" and not self._slides_tools_allowed(context):
                continue
            if allowed_modules is not None and module_id not in allowed_modules:
                continue
            if catalog_filter is not None:
                context.logger.info(
                    "Catalog filter applied",
                    catalog=catalog_filter,
                    module_count=len(modules),
                )
            try:
                if not await self._has_module_permission(context, module_id):
                    continue
                module_tools = await module.get_tools()

                for tool in module_tools:
                    tool_copy = tool.copy()
                    name = tool_copy.get("name")
                    if isinstance(name, str) and name.startswith("slides.") and not self._slides_tools_allowed(context):
                        continue
                    if isinstance(name, str) and name.strip():
                        try:
                            tool_copy = ensure_tool_definition_eval_metadata(tool_copy)
                        except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as exc:
                            context.logger.opt(exception=exc).debug(
                                "Failed to attach eval metadata to listed tool: module_id={module_id} "
                                "tool_name={tool_name} error_type={error_type}",
                                module_id=module_id,
                                tool_name=name,
                                error_type=exc.__class__.__name__,
                            )
                    tool_copy["module"] = module_id
                    # Scoped tool permissions: when scopes are present, list only matching tools
                    if self._mcp_scopes(context) and isinstance(name, str):
                        if not self._scope_allows_tool_name(context, name):
                            continue
                    # Catalog filter: include only when in selected catalog
                    if catalog_filter is not None and isinstance(name, str):
                        meta = tool_copy.get("metadata") if isinstance(tool_copy, dict) else None
                        exempt = isinstance(meta, dict) and bool(meta.get("catalog_exempt"))
                        if name not in catalog_filter and not exempt:
                            continue
                    is_write = None
                    try:
                        if isinstance(tool_copy, dict):
                            is_write = module.is_write_tool_def(tool_copy)
                    except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                        is_write = None
                    can_execute = await self._has_tool_permission(context, name, is_write=is_write) if name else False
                    tool_copy["canExecute"] = can_execute
                    tools.append(tool_copy)
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as e:
                context.logger.exception(f"Error getting tools from module {module_id}: {e}")

        response: dict[str, Any] = {"tools": tools}
        if catalog_meta is not None:
            response["_meta"] = {"catalog": catalog_meta}
        return response

    def _extract_allowed_tools(self, context: RequestContext) -> list[str] | None:
        """Extract allowed-tools list from request context metadata."""
        return self._tool_execution_security.extract_allowed_tools(context)

    def _extract_eval_profile_id(self, context: RequestContext) -> str | None:
        """Extract a non-sensitive profile identifier for execution eval metadata."""
        return self._tool_execution_runtime.extract_eval_profile_id(context)

    def _extract_tool_command(self, tool_args: Any) -> str | None:
        """Extract command-like string from tool arguments for pattern matching."""
        return self._tool_execution_security.extract_tool_command(tool_args)

    def _matches_allowed_tool_pattern(self, tool_name: str, tool_args: Any, pattern: str) -> bool:
        """Check if tool invocation matches an allowed-tools pattern."""
        return self._tool_execution_security.matches_allowed_tool_pattern(
            tool_name,
            tool_args,
            pattern,
            extract_tool_command=self._extract_tool_command,
        )

    @staticmethod
    def _tool_authorization_names(tool_name: str) -> tuple[str, ...]:
        """Return invoked and canonical names that may authorize a tool call."""

        return ToolExecutionSecurity.tool_authorization_names(tool_name)

    def _matches_tool_authorization_pattern(self, tool_name: str, tool_args: Any, pattern: str) -> bool:
        """Match a policy pattern against the invoked name and any canonical alias."""

        return self._tool_execution_security.matches_tool_authorization_pattern(
            tool_name,
            tool_args,
            pattern,
            matches_allowed_tool_pattern=self._matches_allowed_tool_pattern,
            tool_authorization_names=self._tool_authorization_names,
        )

    def _scope_allows_tool_name(self, context: RequestContext, tool_name: str) -> bool:
        """Return True when scopes allow the invoked tool name or canonical alias."""

        return self._tool_execution_security.scope_allows_tool_name(
            context,
            tool_name,
            scope_allows=self._scope_allows,
            tool_authorization_names=self._tool_authorization_names,
        )

    def _is_tool_allowed_by_context(self, tool_name: str, tool_args: Any, context: RequestContext) -> bool:
        """Return True when tool usage is allowed by context metadata."""
        return self._tool_execution_security.is_tool_allowed_by_context(
            tool_name,
            tool_args,
            context,
            extract_allowed_tools=self._extract_allowed_tools,
            matches_tool_authorization_pattern=self._matches_tool_authorization_pattern,
        )

    async def _resolve_effective_tool_policy(self, context: RequestContext) -> dict[str, Any] | None:
        self._sync_tool_execution_dependencies()
        return await self._tool_execution_security._resolve_effective_tool_policy(context)

    def _is_tool_allowed_by_effective_policy(
        self,
        tool_name: str,
        tool_args: Any,
        policy: dict[str, Any] | None,
    ) -> bool:
        return self._tool_execution_security._is_tool_allowed_by_effective_policy(
            tool_name,
            tool_args,
            policy,
            matches_tool_authorization_pattern=self._matches_tool_authorization_pattern,
        )

    async def _evaluate_runtime_approval(
        self,
        *,
        effective_policy: dict[str, Any] | None,
        tool_name: str,
        tool_args: Any,
        context: RequestContext,
        tool_def: dict[str, Any] | None,
        is_write: bool | None,
        within_effective_policy: bool,
        force_approval: bool = False,
        approval_reason: str | None = None,
        scope_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._sync_tool_execution_dependencies()
        return await self._tool_execution_security._evaluate_runtime_approval(
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

    async def _evaluate_path_scope(
        self,
        *,
        effective_policy: dict[str, Any] | None,
        tool_name: str,
        tool_args: Any,
        context: RequestContext,
        tool_def: dict[str, Any] | None,
        path_scope_candidates: list[PathScopeCandidate] | None = None,
    ) -> dict[str, Any]:
        self._sync_tool_execution_dependencies()
        return await self._tool_execution_security._evaluate_path_scope(
            effective_policy=effective_policy,
            tool_name=tool_name,
            tool_args=tool_args,
            context=context,
            tool_def=tool_def,
            path_scope_candidates=path_scope_candidates,
        )

    async def _extract_path_scope_candidates(
        self,
        *,
        module: BaseModule,
        tool_name: str,
        tool_args: Any,
        context: RequestContext,
        tool_def: dict[str, Any] | None,
    ) -> list[PathScopeCandidate] | None:
        self._sync_tool_execution_dependencies()
        return await self._tool_execution_security._extract_path_scope_candidates(
            module=module,
            tool_name=tool_name,
            tool_args=tool_args,
            context=context,
            tool_def=tool_def,
        )

    async def _evaluate_external_access(
        self,
        *,
        effective_policy: dict[str, Any] | None,
        tool_name: str,
        context: RequestContext,
    ) -> dict[str, Any]:
        self._sync_tool_execution_dependencies()
        return await self._tool_execution_security._evaluate_external_access(
            effective_policy=effective_policy,
            tool_name=tool_name,
            context=context,
        )

    async def _handle_tools_call(
        self,
        params: dict[str, Any],
        context: RequestContext
    ) -> dict[str, Any]:
        """Execute a tool."""
        tool_name = params.get("name") if isinstance(params, dict) else None
        if isinstance(tool_name, str) and tool_name.startswith("slides.") and not self._slides_tools_allowed(context):
            return {
                "success": False,
                "error": {
                    "code": "slides_websocket_guard_required",
                    "operation": tool_name,
                },
            }
        return await self._tool_execution.handle_tools_call(params, context)

    @staticmethod
    def _slides_tools_allowed(context: RequestContext) -> bool:
        """Allow Slides normally except on WebSockets lacking the trusted guard."""

        metadata = context.metadata if isinstance(context.metadata, dict) else {}
        if metadata.get("mcp_transport") != "websocket":
            return True
        return is_guarded_slides_websocket_metadata(metadata)

    async def prepare_tool_call(
        self,
        params: dict[str, Any],
        context: RequestContext,
        idempotency_key: str | None = None,
    ) -> PreparedToolCall:
        """Prepare a tool invocation through protocol policy, validation, and governance checks."""
        return await self._tool_execution.prepare_tool_call(
            params=params,
            context=context,
            idempotency_key=idempotency_key,
        )

    async def _prepare_tool_call_inline(
        self,
        params: dict[str, Any],
        context: RequestContext,
        idempotency_key: str | None = None,
    ) -> PreparedToolCall:
        """Prepare a tool invocation through protocol policy, validation, and governance checks."""
        self._sync_tool_execution_dependencies()
        return await self._tool_execution_security.prepare_tool_call(
            params=params,
            context=context,
            idempotency_key=idempotency_key,
            hooks=self._tool_execution_hooks,
        )

    async def execute_prepared_tool_call(self, prepared: PreparedToolCall) -> dict[str, Any]:
        """Execute a previously prepared tool invocation."""
        self._sync_tool_execution_dependencies()
        return await self._tool_execution.execute_prepared_tool_call(prepared)

    async def _execute_prepared_tool_call_inline(self, prepared: PreparedToolCall) -> dict[str, Any]:
        """Execute a previously prepared tool invocation."""
        self._sync_tool_execution_dependencies()
        return await self._tool_execution_runtime.execute_prepared_tool_call(prepared)

    async def shutdown(self) -> None:
        """Terminally drain protocol-owned tool execution state."""

        await self._idempotency.shutdown()

    @property
    def has_pending_shutdown_work(self) -> bool:
        """Whether protocol-owned execution cleanup remains pending."""

        return self._idempotency.has_pending_shutdown_work

    async def wait_for_shutdown_completion(self) -> None:
        """Wait for retained protocol-owned execution cleanup."""

        await self._idempotency.wait_for_shutdown_completion()

    # -------------------------
    # Idempotency cache helpers
    # -------------------------
    def _make_idempotency_cache_key(self, context: RequestContext, module_name: str, tool_name: str, idempotency_key: str) -> str:
        """Compatibility delegate to ToolExecutionSecurity's authoritative builder."""

        return self._tool_execution_security.make_idempotency_cache_key(
            context,
            module_name,
            tool_name,
            idempotency_key,
        )

    def _validate_input_schema(self, schema: dict[str, Any], args: dict[str, Any]) -> None:
        self._tool_execution_security.validate_input_schema(schema, args)

    async def _handle_resources_list(
        self,
        params: dict[str, Any],
        context: RequestContext
    ) -> dict[str, Any]:
        """List available resources"""
        resources = []
        modules = await self.module_registry.get_all_modules()
        catalog_filter = await self._resolve_catalog_tool_names(params, context)
        catalog_meta = self._catalog_filter_metadata(params, catalog_filter)
        module_tool_names: dict[str, set[str]] = {}

        for module_id, module in modules.items():
            try:
                if catalog_filter is not None:
                    context.logger.info(f"Catalog filter applied: {sorted(catalog_filter)}")
                if not await self._has_module_permission(context, module_id):
                    continue
                if catalog_filter is not None:
                    cached_names = module_tool_names.get(module_id)
                    if cached_names is None:
                        try:
                            module_tools = await module.get_tools()
                            cached_names = {
                                str(tool.get("name"))
                                for tool in module_tools
                                if isinstance(tool, dict) and isinstance(tool.get("name"), str)
                            }
                        except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                            cached_names = set()
                        module_tool_names[module_id] = cached_names
                    if not cached_names.intersection(catalog_filter):
                        continue
                module_resources = await module.get_resources()

                for resource in module_resources:
                    uri = resource.get("uri") if isinstance(resource, dict) else None
                    if uri and not await self._has_resource_permission(context, uri, module_id):
                        continue
                    resource_copy = resource.copy() if isinstance(resource, dict) else resource
                    if isinstance(resource_copy, dict):
                        resource_copy["module"] = module_id
                    resources.append(resource_copy)

            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as e:
                context.logger.exception(f"Error getting resources from module {module_id}: {e}")

        response: dict[str, Any] = {"resources": resources}
        if catalog_meta is not None:
            response["_meta"] = {"catalog": catalog_meta}
        return response

    async def _handle_resources_read(
        self,
        params: dict[str, Any],
        context: RequestContext
    ) -> dict[str, Any]:
        """Read a resource"""
        uri = params.get("uri")
        if not uri:
            raise InvalidParamsException("Resource URI is required")

        # Find module for resource
        module = await self.module_registry.find_module_for_resource(uri)
        if not module:
            raise InvalidParamsException(f"Resource not found: {uri}")
        module_id = self.module_registry.get_module_id_for_resource(uri) or getattr(module, "name", None)

        if not await self._has_resource_permission(context, uri, module_id):
            raise PermissionError(f"Permission denied for resource: {uri}")

        # Read resource (pass context when supported)
        read_fn = module.read_resource
        try:
            params = inspect.signature(read_fn).parameters
        except (TypeError, ValueError):
            params = {}
        if "context" in params:
            content = await read_fn(uri, context=context)
        else:
            content = await read_fn(uri)

        return {"contents": [content]}

    async def _handle_prompts_list(
        self,
        params: dict[str, Any],
        context: RequestContext
    ) -> dict[str, Any]:
        """List available prompts"""
        prompts = []
        warnings: list[dict[str, Any]] = []
        next_cursor: Any = None
        modules = await self.module_registry.get_all_modules()

        for module_id, module in modules.items():
            try:
                if module_id != "prompts" and not await self._has_module_permission(context, module_id):
                    continue
                module_warnings: list[dict[str, Any]] = []
                module_result = await module.get_prompts_for_context(context, params or {})
                if not isinstance(module_result, dict):
                    module_prompts = []
                else:
                    module_prompts = module_result.get("prompts", [])
                    if "nextCursor" in module_result:
                        next_cursor = module_result.get("nextCursor")
                    meta = module_result.get("_meta")
                    if isinstance(meta, dict):
                        tldw_meta = meta.get("tldw")
                        if isinstance(tldw_meta, dict):
                            module_warnings = tldw_meta.get("warnings")
                            if isinstance(module_warnings, list):
                                module_warnings = [
                                    warning for warning in module_warnings if isinstance(warning, dict)
                                ]

                for prompt in module_prompts:
                    name = prompt.get("name") if isinstance(prompt, dict) else None
                    if self._is_namespaced_prompt_name(name):
                        if not await self._has_namespaced_prompt_permission(context, name):
                            continue
                    elif name and not await self._has_prompt_permission(context, name, module_id):
                        continue
                    prompt_copy = prompt.copy() if isinstance(prompt, dict) else prompt
                    if isinstance(prompt_copy, dict):
                        prompt_copy["module"] = module_id
                    prompts.append(prompt_copy)

                for warning in module_warnings:
                    visible_warning = await self._visible_prompt_warning(context, warning, module_id)
                    if visible_warning:
                        warnings.append(visible_warning)

            except PromptCatalogError as e:
                if e.internal:
                    context.logger.error(
                        "Internal prompt catalog list error from module {}: {}",
                        module_id,
                        e.code,
                    )
                    raise RuntimeError("Failed to list prompts") from None
                raise InvalidParamsException(str(e)) from e
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as e:
                context.logger.exception(f"Error getting prompts from module {module_id}: {e}")

        if (
            self._has_restrictive_prompt_scope(context)
            and self._prompt_cursor_has_identifier_fields(next_cursor)
        ):
            next_cursor = None

        result: dict[str, Any] = {"prompts": prompts}
        if next_cursor is not None:
            result["nextCursor"] = next_cursor
        if warnings:
            result["_meta"] = {"tldw": {"warnings": warnings}}
        return result

    async def _handle_prompts_get(
        self,
        params: dict[str, Any],
        context: RequestContext
    ) -> dict[str, Any]:
        """Get a specific prompt"""
        name = params.get("name")
        if not isinstance(name, str) or not name:
            raise InvalidParamsException("Prompt name is required")

        arguments = params.get("arguments", {})
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise InvalidParamsException("Prompt arguments must be an object")

        if self._is_namespaced_prompt_name(name):
            module = await self.module_registry.get_module("prompts")
            if not module:
                raise InvalidParamsException(f"Prompt not found: {name}")
            if not await self._has_namespaced_prompt_permission(context, name):
                raise PermissionError(f"Permission denied for prompt: {name}")
            try:
                return await module.get_prompt_for_context(name, arguments, context)
            except PromptCatalogError as e:
                if e.code == "permission_denied":
                    raise PermissionError(f"Permission denied for prompt: {name}") from e
                if e.internal:
                    if name.startswith(LIBRARY_PROMPT_PREFIX):
                        prompt_namespace = "library"
                    elif name.startswith(CONFIG_PROMPT_PREFIX):
                        prompt_namespace = "config"
                    else:
                        prompt_namespace = "unknown"
                    context.logger.error(
                        "Internal prompt catalog get error for namespace {}: {}",
                        prompt_namespace,
                        e.code,
                    )
                    raise RuntimeError("Failed to get prompt") from None
                raise InvalidParamsException(str(e)) from e

        # Find module for prompt
        module = await self.module_registry.find_module_for_prompt(name)
        if not module:
            raise InvalidParamsException(f"Prompt not found: {name}")
        module_id = self.module_registry.get_module_id_for_prompt(name) or getattr(module, "name", None)

        if not await self._has_prompt_permission(context, name, module_id):
            raise PermissionError(f"Permission denied for prompt: {name}")

        # Get prompt
        prompt = await module.get_prompt_for_context(name, arguments, context)

        return prompt

    async def _handle_modules_list(
        self,
        params: dict[str, Any],
        context: RequestContext
    ) -> dict[str, Any]:
        """List registered modules"""
        registrations = await self.module_registry.list_registrations()
        filtered: list[dict[str, Any]] = []
        for entry in registrations:
            module_id = entry.get("module_id") if isinstance(entry, dict) else None
            try:
                if await self._has_module_permission(context, module_id):
                    filtered.append(entry)
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                continue
        return {"modules": filtered}

    async def _handle_modules_health(
        self,
        params: dict[str, Any],
        context: RequestContext
    ) -> dict[str, Any]:
        """Get module health status"""
        health_results = await self.module_registry.check_all_health()

        # Convert to serializable format
        health_data = {}
        for module_id, health in health_results.items():
            last_check_iso = None
            try:
                if getattr(health, "last_check", None):
                    last_check_iso = health.last_check.isoformat()
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                last_check_iso = None
            health_data[module_id] = {
                "status": health.status.value if getattr(health, "status", None) else "unknown",
                "message": getattr(health, "message", ""),
                "checks": getattr(health, "checks", {}),
                "last_check": last_check_iso,
            }

        return {"health": health_data}


# Convenience function
async def process_mcp_request(
    request: Union[dict[str, Any], MCPRequest],
    context: Optional[RequestContext] = None
) -> MCPResponse:
    """Process an MCP request"""
    protocol = MCPProtocol()
    return await protocol.process_request(request, context)
