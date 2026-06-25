"""Lifecycle hook helpers for MCP tool execution."""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any, cast

from loguru import logger
from mcp_unified.tool_use_reporting.models import MAX_TOOL_HOOK_RESULTS
from mcp_unified.tool_use_reporting.sanitization import (
    sanitize_reason_code,
    sanitize_safe_id,
)

from ..auth.rate_limiter import RateLimitExceeded
from ..interfaces.runtime import ToolHookAction, ToolHookCallContext, ToolHookDecision
from ..protocol_types import (
    ApprovalRequiredError,
    GovernanceDeniedError,
    InvalidParamsException,
    RequestContext,
)

try:  # pragma: no cover - optional dependency
    from redis.exceptions import RedisError as _RedisError
except ImportError:  # pragma: no cover - redis not installed
    _RedisError = RuntimeError


_HOOK_HELPER_NONCRITICAL_EXCEPTIONS = (
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
    _RedisError,
    RateLimitExceeded,
    InvalidParamsException,
)

_TOOL_HOOK_SUMMARY_ID_FIELDS = (
    "phase",
    "hook_id",
    "action",
    "status",
    "error_type",
)


class ToolExecutionHooks:
    def __init__(
        self,
        *,
        hook_manager: Any,
        reporter: Any,
        noncritical_exceptions: tuple[type[BaseException], ...],
    ) -> None:
        self._tool_call_hook_manager = hook_manager
        self._reporter = reporter
        self._noncritical_exceptions = noncritical_exceptions

    @staticmethod
    def _tool_use_category(tool_def: dict[str, Any] | None) -> str | None:
        """Return the metadata category from a tool definition when present."""
        metadata = (tool_def or {}).get("metadata") if isinstance(tool_def, dict) else None
        if not isinstance(metadata, dict):
            return None
        category = metadata.get("category")
        return str(category) if isinstance(category, str) and category.strip() else None

    @staticmethod
    def _tool_hook_summary_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Return sanitized hook summary rows from a protocol hook payload."""

        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            hook_results = metadata.get("hook_results")
            if isinstance(hook_results, list):
                return [
                    item
                    for item in (
                        ToolExecutionHooks._sanitize_tool_hook_summary_item(raw_item)
                        for raw_item in hook_results[:MAX_TOOL_HOOK_RESULTS]
                    )
                    if item
                ]

        row = ToolExecutionHooks._sanitize_tool_hook_summary_item(
            {
                "phase": payload.get("phase"),
                "action": payload.get("action"),
                "status": payload.get("status"),
                "reason_code": payload.get("reason_code"),
                "error_type": payload.get("error_type"),
                "hook_id": metadata.get("hook_id") if isinstance(metadata, dict) else payload.get("hook_id"),
                "hook_order": (
                    metadata.get("hook_order") if isinstance(metadata, dict) else payload.get("hook_order")
                ),
            }
        )
        return [row] if row else []

    @staticmethod
    def _sanitize_tool_hook_summary_item(item: Any) -> dict[str, Any]:
        """Return allowlisted, sanitized hook result metadata for request context storage."""

        if not isinstance(item, dict):
            return {}

        row: dict[str, Any] = {}
        for field in _TOOL_HOOK_SUMMARY_ID_FIELDS:
            safe_value = sanitize_safe_id(item.get(field), field=field)
            if safe_value is not None:
                row[field] = safe_value

        reason_code = sanitize_reason_code(item.get("reason_code"))
        if reason_code is not None:
            row["reason_code"] = reason_code

        hook_order = item.get("hook_order")
        if hook_order is not None:
            try:
                row["hook_order"] = int(hook_order)
            except (TypeError, ValueError):
                pass

        if "redacted" in item:
            row["redacted"] = bool(item.get("redacted"))

        return row

    @staticmethod
    def _append_tool_hook_summary(context: RequestContext, payload: dict[str, Any]) -> None:
        """Append safe hook metadata for tool-use reporting."""

        metadata = getattr(context, "metadata", None)
        if not isinstance(metadata, dict):
            return
        existing = metadata.get("mcp_tool_hook_results")
        if not isinstance(existing, list):
            existing = []
            metadata["mcp_tool_hook_results"] = existing
        remaining = max(0, MAX_TOOL_HOOK_RESULTS - len(existing))
        if remaining <= 0:
            return
        existing.extend(ToolExecutionHooks._tool_hook_summary_items(payload)[:remaining])

    @staticmethod
    def _hook_safe_copy(value: Any) -> Any:
        """Return a detached copy of hook-visible metadata without failing tool preparation."""
        try:
            return copy.deepcopy(value)
        except _HOOK_HELPER_NONCRITICAL_EXCEPTIONS:
            try:
                return json.loads(json.dumps(value, default=str))
            except _HOOK_HELPER_NONCRITICAL_EXCEPTIONS:
                return str(value)

    @staticmethod
    def _hook_safe_metadata(context: RequestContext) -> dict[str, Any]:
        """Return request metadata safe for local lifecycle hook decisions."""
        metadata = getattr(context, "metadata", None)
        if not isinstance(metadata, dict):
            return {}
        return {
            str(key): ToolExecutionHooks._hook_safe_copy(value)
            for key, value in metadata.items()
            if isinstance(key, str) and not key.startswith("_") and key not in {"governance_preflight"}
        }

    @staticmethod
    def _hook_safe_tool_args(tool_args: Any, *, tool_name: str | None = None) -> dict[str, Any] | None:
        """Return detached sanitized tool arguments for hook evaluation."""
        if not isinstance(tool_args, dict):
            return None
        copied = ToolExecutionHooks._hook_safe_copy(tool_args)
        if not isinstance(copied, dict):
            return None
        return ToolExecutionHooks._redact_hook_visible_tool_args(copied, tool_name=tool_name)

    @staticmethod
    def _redact_hook_visible_tool_args(tool_args: dict[str, Any], *, tool_name: str | None = None) -> dict[str, Any]:
        """Redact secret-bearing argument values from hook-visible metadata."""

        if str(tool_name or "") != "sandbox.run":
            return tool_args
        env = tool_args.get("env")
        if not isinstance(env, dict):
            return tool_args
        redacted = dict(tool_args)
        redacted["env"] = {str(key): "[redacted]" for key in env}
        return redacted

    @staticmethod
    def _hook_safe_scope_payload(scope_payload: dict[str, Any] | None) -> dict[str, Any] | None:
        """Return detached path/external scope metadata for hooks."""
        if not isinstance(scope_payload, dict):
            return None
        copied = ToolExecutionHooks._hook_safe_copy(scope_payload)
        return copied if isinstance(copied, dict) else None

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
        return ToolHookCallContext(
            phase="post" if phase == "post" else "pre",
            tool_name=str(tool_name),
            module_id=module_id,
            is_write=is_write,
            tool_category=self._tool_use_category(tool_def),
            arguments_hash=arguments_hash,
            request_id=str(context.request_id),
            user_id=context.user_id,
            client_id=context.client_id,
            session_id=context.session_id,
            metadata=self._hook_safe_metadata(context),
            tool_args=self._hook_safe_tool_args(tool_args, tool_name=tool_name),
            status=status,
            duration_ms=duration_ms,
            error_type=error.__class__.__name__ if error is not None else None,
            scope_payload=self._hook_safe_scope_payload(scope_payload),
        )

    @staticmethod
    def _coerce_tool_hook_action(action: Any) -> ToolHookAction | None:
        """Normalize a runtime hook action into the public literal contract."""
        normalized = str(action or "allow").strip().lower()
        if normalized in {"allow", "deny", "ask", "approval_required"}:
            return cast(ToolHookAction, normalized)
        return None

    @staticmethod
    def _coerce_tool_hook_decision(
        decision: ToolHookDecision | dict[str, Any] | None,
    ) -> ToolHookDecision:
        """Normalize hook decision values from typed or dict-based embedders."""
        if decision is None:
            return ToolHookDecision(action="allow")
        if isinstance(decision, ToolHookDecision):
            return decision
        if isinstance(decision, dict):
            metadata = decision.get("metadata")
            action = ToolExecutionHooks._coerce_tool_hook_action(
                decision.get("action") or decision.get("status") or "allow"
            )
            if action is None:
                return ToolHookDecision(
                    action="deny",
                    reason_code="invalid_tool_hook_action",
                    message="Invalid MCP tool hook action",
                )
            return ToolHookDecision(
                action=action,
                reason_code=(
                    str(decision.get("reason_code"))
                    if decision.get("reason_code") is not None
                    else None
                ),
                message=str(decision.get("message")) if decision.get("message") is not None else None,
                metadata=dict(metadata) if isinstance(metadata, dict) else {},
            )
        return ToolHookDecision(
            action="deny",
            reason_code="invalid_tool_hook_decision",
            message="Invalid MCP tool hook decision",
        )

    @staticmethod
    def _tool_hook_payload(
        decision: ToolHookDecision,
        *,
        phase: str,
        fallback_reason_code: str | None = None,
    ) -> dict[str, Any]:
        """Serialize a normalized hook decision into response-safe metadata."""
        action = str(decision.action or "allow").strip().lower()
        if action == "approval_required":
            action = "ask"
        payload: dict[str, Any] = {
            "phase": phase,
            "action": action,
            "status": action,
        }
        reason_code = sanitize_reason_code(decision.reason_code or fallback_reason_code)
        if reason_code:
            payload["reason_code"] = reason_code
        if isinstance(decision.metadata, dict) and decision.metadata:
            metadata_payload = {
                key: decision.metadata[key]
                for key in ("hook_id", "hook_order", "redacted")
                if key in decision.metadata
            }
            payload.update(ToolExecutionHooks._sanitize_tool_hook_summary_item(metadata_payload))
        return payload

    @staticmethod
    def _tool_hook_reporting_payload(
        decision: ToolHookDecision,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Return payload used only for sanitized reporting summaries."""

        summary_payload = dict(payload)
        if isinstance(decision.metadata, dict):
            hook_results = decision.metadata.get("hook_results")
            if isinstance(hook_results, list):
                summary_payload["metadata"] = {"hook_results": hook_results}
        return summary_payload

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
        hook_context = self._build_tool_hook_context(
            phase="pre",
            tool_name=tool_name,
            module_id=module_id,
            tool_def=tool_def,
            tool_args=tool_args,
            is_write=is_write,
            arguments_hash=arguments_hash,
            context=context,
            scope_payload=scope_payload,
        )
        try:
            raw_decision = await self._tool_call_hook_manager.before_tool_call(hook_context)
        except asyncio.CancelledError:
            raise
        except self._noncritical_exceptions as exc:
            logger.exception(
                "MCP pre-tool hook failed closed: tool_name={} module_id={} request_id={} error_type={}",
                tool_name,
                module_id,
                context.request_id,
                exc.__class__.__name__,
            )
            payload = {
                "phase": "pre",
                "action": "deny",
                "status": "deny",
                "reason_code": "tool_hook_unavailable",
                "error_type": exc.__class__.__name__,
            }
            hook_id = getattr(exc, "hook_id", None)
            if hook_id is not None:
                payload["hook_id"] = hook_id
            hook_order = getattr(exc, "hook_order", None)
            if hook_order is not None:
                payload["hook_order"] = hook_order
            self._append_tool_hook_summary(context, payload)
            raise GovernanceDeniedError(
                "Permission denied by MCP tool hook",
                governance={
                    "action": "deny",
                    "status": "deny",
                    "reason_code": "tool_hook_unavailable",
                    "hook": payload,
                },
            ) from exc

        decision = self._coerce_tool_hook_decision(raw_decision)
        payload = self._tool_hook_payload(decision, phase="pre")
        if raw_decision is not None:
            self._append_tool_hook_summary(context, self._tool_hook_reporting_payload(decision, payload))
        action = str(payload.get("action") or "allow").strip().lower()
        if action == "allow":
            return payload
        if action == "ask":
            raise ApprovalRequiredError(
                "Approval required by MCP tool hook",
                approval={
                    "source": "tool_hook",
                    "reason_code": payload.get("reason_code") or "tool_hook_approval_required",
                    "message": payload.get("message"),
                    "hook": payload,
                },
            )
        if action != "deny":
            payload = self._tool_hook_payload(
                ToolHookDecision(
                    action="deny",
                    reason_code="invalid_tool_hook_action",
                    message=f"Unsupported MCP tool hook action: {action}",
                    metadata={"requested_action": action},
                ),
                phase="pre",
            )
        raise GovernanceDeniedError(
            "Permission denied by MCP tool hook",
            governance={
                "action": "deny",
                "status": "deny",
                "reason_code": payload.get("reason_code") or "tool_hook_denied",
                "hook": payload,
            },
        )

    async def run_pre_tool_hooks(
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
        """Run pre-tool hooks through the public tool-execution security API."""

        return await self._run_pre_tool_hooks(
            tool_name=tool_name,
            tool_args=tool_args,
            module_id=module_id,
            tool_def=tool_def,
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
        hook_context = self._build_tool_hook_context(
            phase="post",
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
        try:
            raw_decision = await self._tool_call_hook_manager.after_tool_call(hook_context)
        except asyncio.CancelledError:
            raise
        except self._noncritical_exceptions as exc:
            logger.exception(
                "MCP post-tool hook failed; suppressed to preserve tool outcome: "
                "tool_name={} module_id={} request_id={} status={} error_type={}",
                tool_name,
                module_id,
                context.request_id,
                status,
                exc.__class__.__name__,
            )
            payload = {
                "phase": "post",
                "action": "deny",
                "status": "error",
                "reason_code": "tool_hook_unavailable",
                "error_type": exc.__class__.__name__,
            }
            hook_id = getattr(exc, "hook_id", None)
            if hook_id is not None:
                payload["hook_id"] = hook_id
            hook_order = getattr(exc, "hook_order", None)
            if hook_order is not None:
                payload["hook_order"] = hook_order
            self._append_tool_hook_summary(context, payload)
            return
        if raw_decision is not None:
            decision = self._coerce_tool_hook_decision(raw_decision)
            payload = self._tool_hook_payload(decision, phase="post")
            self._append_tool_hook_summary(context, self._tool_hook_reporting_payload(decision, payload))

    async def run_post_tool_hooks(
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
        """Run post-tool hooks through the public tool-execution runtime API."""

        await self._run_post_tool_hooks(
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
