"""Tool-use reporting helpers for MCP tool execution."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger
from mcp_unified.tool_use_reporting.builders import (
    classify_tool_use_exception,
    extract_safe_context_dimensions,
)
from mcp_unified.tool_use_reporting.models import (
    MAX_FILE_POLICY_DECISIONS,
    MAX_TOOL_HOOK_RESULTS,
    ToolUseEvent,
    ToolUseStatus,
)
from mcp_unified.tool_use_reporting.recorder import record_tool_use_safely
from mcp_unified.tool_use_reporting.sanitization import sanitize_reason_code

from ..protocol_types import GovernanceDeniedError, RequestContext
from .hooks import ToolExecutionHooks


def _safe_exception_family(exc: BaseException) -> str:
    """Return a bounded exception family without reading exception text."""

    try:
        name = type(exc).__name__
        if (
            type(name) is str
            and 1 <= len(name) <= 64
            and name.isascii()
            and (name[0].isalpha() or name[0] == "_")
            and all(character.isalnum() or character == "_" for character in name)
        ):
            return name
    except asyncio.CancelledError:
        raise
    except BaseException:  # noqa: BLE001 - hostile exception types degrade safely.
        return "Exception"
    return "Exception"


class ToolExecutionReporter:
    """Build and emit MCP tool-use reporting, metrics, and audit records."""

    def __init__(
        self,
        *,
        recorder: Any,
        metrics: Any,
        tool_name_re: Any,
        noncritical_exceptions: tuple[type[BaseException], ...],
    ) -> None:
        """Store recorder, metrics, and validation dependencies for reporting."""

        self._tool_use_recorder = recorder
        self.metrics = metrics
        self._tool_name_re = tool_name_re
        self._noncritical_exceptions = noncritical_exceptions

    def should_record_tool_use(self, context: RequestContext) -> bool:
        """Return whether this protocol path should record tool-use metadata."""
        metadata = getattr(context, "metadata", {})
        if not isinstance(metadata, dict):
            return True
        return metadata.get("mcp_tool_use_observed") is not True

    should_record = should_record_tool_use

    async def record_tool_use_event(self, event: ToolUseEvent) -> None:
        """Record a tool-use event through the configured recorder."""
        await record_tool_use_safely(self._tool_use_recorder, event)

    record_event = record_tool_use_event

    def safe_tool_use_name(self, value: Any) -> str:
        """Return a safe tool name or the unknown sentinel."""
        if isinstance(value, str) and self._tool_name_re.match(value):
            return value
        return "unknown"

    _safe_tool_use_name = safe_tool_use_name

    @staticmethod
    def tool_use_duration_ms(start_ts: float) -> float:
        """Return elapsed milliseconds from a monotonic-ish wall clock sample."""
        return max(0.0, (time.time() - start_ts) * 1000.0)

    duration_ms = tool_use_duration_ms
    _tool_use_duration_ms = tool_use_duration_ms

    @staticmethod
    def tool_use_execution_origin_for_failure(status: ToolUseStatus) -> str:
        """Return execution-origin metadata for a failed tool path."""
        if status == "unavailable":
            return "unavailable"
        return "failed_before_execution"

    execution_origin_for_failure = tool_use_execution_origin_for_failure
    _tool_use_execution_origin_for_failure = tool_use_execution_origin_for_failure

    @staticmethod
    def tool_use_eval_metadata(
        *,
        payload: dict[str, Any] | None = None,
        tool_def: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Extract safe eval metadata from response payload or tool definition."""
        if isinstance(payload, dict) and isinstance(payload.get("eval"), dict):
            return dict(payload["eval"])
        metadata = (tool_def or {}).get("metadata") if isinstance(tool_def, dict) else None
        if isinstance(metadata, dict) and isinstance(metadata.get("eval"), dict):
            return dict(metadata["eval"])
        return {}

    _tool_use_eval_metadata = tool_use_eval_metadata

    @staticmethod
    def tool_use_file_policy_decisions(scope_payload: dict[str, Any] | None) -> list[dict[str, Any]]:
        """Extract redacted file-policy path decisions from a scope payload."""

        if not isinstance(scope_payload, dict):
            return []
        raw_decisions = scope_payload.get("path_decisions")
        if not isinstance(raw_decisions, list):
            return []
        bounded_decisions = raw_decisions[:MAX_FILE_POLICY_DECISIONS]
        return [dict(decision) for decision in bounded_decisions if isinstance(decision, dict)]

    _tool_use_file_policy_decisions = tool_use_file_policy_decisions

    @staticmethod
    def tool_use_hook_results(metadata: dict[str, Any] | None) -> list[dict[str, Any]]:
        """Consume bounded tool-hook result metadata from request metadata."""

        if not isinstance(metadata, dict):
            return []
        raw_results = metadata.pop("mcp_tool_hook_results", None)
        if not isinstance(raw_results, list):
            return []
        bounded_results = raw_results[:MAX_TOOL_HOOK_RESULTS]
        return [dict(result) for result in bounded_results if isinstance(result, dict)]

    _tool_use_hook_results = tool_use_hook_results

    @staticmethod
    def tool_hook_summary_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Return sanitized hook summary rows from a protocol hook payload."""

        return ToolExecutionHooks._tool_hook_summary_items(payload)

    _tool_hook_summary_items = tool_hook_summary_items

    @staticmethod
    def append_tool_hook_summary(context: RequestContext, payload: dict[str, Any]) -> None:
        """Append safe hook metadata for tool-use reporting."""

        ToolExecutionHooks._append_tool_hook_summary(context, payload)

    _append_tool_hook_summary = append_tool_hook_summary

    @staticmethod
    def tool_use_decision_grant_outcome(file_policy_decisions: list[dict[str, Any]]) -> str | None:
        """Summarize path-decision grant outcomes with denial precedence."""

        outcomes = [
            outcome.strip()
            for decision in file_policy_decisions
            if isinstance(decision, dict)
            and isinstance(outcome := decision.get("grant_outcome"), str)
            and outcome.strip()
        ]
        if "denied" in outcomes:
            return "denied"
        if "not_granted" in outcomes:
            return "not_granted"
        if outcomes:
            return "allowed"
        return None

    _tool_use_decision_grant_outcome = tool_use_decision_grant_outcome

    @staticmethod
    def tool_use_value_present(value: Any) -> bool:
        """Return whether a sensitive value marker contains actual data."""

        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (dict, list, tuple, set)):
            return bool(value)
        return True

    _tool_use_value_present = tool_use_value_present

    @staticmethod
    def tool_use_contains_key(
        value: Any,
        keys: set[str],
        *,
        _depth: int = 0,
    ) -> bool:
        """Return whether a nested tool payload/args object contains any key."""

        if _depth > 4:
            return False
        if isinstance(value, dict):
            for key, nested in value.items():
                if key in keys and ToolExecutionReporter.tool_use_value_present(nested):
                    return True
                if isinstance(nested, (dict, list)) and ToolExecutionReporter.tool_use_contains_key(
                    nested,
                    keys,
                    _depth=_depth + 1,
                ):
                    return True
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, (dict, list)) and ToolExecutionReporter.tool_use_contains_key(
                    item,
                    keys,
                    _depth=_depth + 1,
                ):
                    return True
        return False

    _tool_use_contains_key = tool_use_contains_key

    @staticmethod
    def tool_use_category(tool_def: dict[str, Any] | None) -> str | None:
        """Return the metadata category from a tool definition when present."""
        metadata = (tool_def or {}).get("metadata") if isinstance(tool_def, dict) else None
        if not isinstance(metadata, dict):
            return None
        category = metadata.get("category")
        return str(category) if isinstance(category, str) and category.strip() else None

    _tool_use_category = tool_use_category

    def build_tool_use_event(
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
        metadata = getattr(context, "metadata", {})
        dimensions = extract_safe_context_dimensions(metadata if isinstance(metadata, dict) else None)
        nested = isinstance(metadata, dict) and metadata.get("mcp_tool_use_nested") is True
        hook_results = self.tool_use_hook_results(metadata if isinstance(metadata, dict) else None)
        eval_metadata = self.tool_use_eval_metadata(payload=payload, tool_def=tool_def)
        safe_requested_name = self.safe_tool_use_name(requested_tool_name)
        safe_effective_name = self.safe_tool_use_name(effective_tool_name or safe_requested_name)
        file_policy_decisions = self.tool_use_file_policy_decisions(scope_payload)
        file_policy_related = bool(file_policy_decisions) or safe_effective_name.startswith("fs.")
        sha256_before_keys = {"sha256_before", "expected_sha256", "expected_sha256_by_path"}
        file_policy_sha256_before_present = (
            (
                self.tool_use_contains_key(payload, sha256_before_keys)
                or self.tool_use_contains_key(tool_args, sha256_before_keys)
            )
            if file_policy_related
            else None
        )
        file_policy_sha256_after_present = (
            self.tool_use_contains_key(payload, {"sha256_after"}) if file_policy_related else None
        )
        file_policy_lock_lease_present = (
            (
                self.tool_use_contains_key(payload, {"lock_lease_id", "lock_lease_id_by_path"})
                or self.tool_use_contains_key(tool_args, {"lock_lease_id", "lock_lease_id_by_path"})
            )
            if file_policy_related
            else None
        )
        decision_grant_outcome = self.tool_use_decision_grant_outcome(file_policy_decisions)
        return ToolUseEvent(
            runtime_surface="protocol",
            execution_origin=execution_origin,  # type: ignore[arg-type]
            nested=nested,
            requested_tool_name=safe_requested_name,
            effective_tool_name=safe_effective_name,
            module_id=module_id,
            category=self.tool_use_category(tool_def),
            read_only=(not is_write) if is_write is not None else None,
            is_write=is_write,
            source_kind="local",
            status=status,
            reason_code=reason_code,
            duration_ms=duration_ms,
            idempotency_replay=idempotency_replay,
            tool_prompt_id=eval_metadata.get("tool_prompt_id"),
            tool_prompt_version=eval_metadata.get("tool_prompt_version"),
            prompt_variant=eval_metadata.get("prompt_variant"),
            action_family=eval_metadata.get("action_family"),
            result_kind=eval_metadata.get("result_kind") or eval_metadata.get("expected_result_kind"),
            path_filter_used=eval_metadata.get("path_filter_used"),
            grant_outcome=eval_metadata.get("grant_outcome") or decision_grant_outcome,
            truncated=eval_metadata.get("truncated"),
            file_policy_decisions=file_policy_decisions,
            tool_hook_results=hook_results,
            file_policy_sha256_before_present=file_policy_sha256_before_present,
            file_policy_sha256_after_present=file_policy_sha256_after_present,
            file_policy_lock_lease_present=file_policy_lock_lease_present,
            **dimensions,
        )

    build_event = build_tool_use_event
    _build_tool_use_event = build_tool_use_event

    async def record_process_request_failure(
        self,
        *,
        request: Any,
        context: RequestContext,
        status: ToolUseStatus,
        reason_code: str,
        start_ts: float,
        requested_tool_name: Any = None,
        should_record: Callable[[RequestContext], bool] | None = None,
        build_event: Callable[..., ToolUseEvent] | None = None,
        record_event: Callable[[ToolUseEvent], Awaitable[None]] | None = None,
        duration_ms: Callable[[float], float] | None = None,
        execution_origin_for_failure: Callable[[ToolUseStatus], str] | None = None,
    ) -> None:
        """Record a tools/call failure that occurs before handler dispatch."""
        should_record = should_record or self.should_record
        if getattr(request, "method", None) != "tools/call" or not should_record(context):
            return
        params = request.params if isinstance(getattr(request, "params", None), dict) else {}
        requested = requested_tool_name if requested_tool_name is not None else params.get("name")
        build_event = build_event or self.build_event
        record_event = record_event or self.record_event
        duration_ms = duration_ms or self.duration_ms
        execution_origin_for_failure = execution_origin_for_failure or self.execution_origin_for_failure
        try:
            event = build_event(
                context=context,
                requested_tool_name=requested,
                status=status,
                execution_origin=execution_origin_for_failure(status),
                duration_ms=duration_ms(start_ts),
                reason_code=reason_code,
            )
            await record_event(event)
        except Exception as exc:  # noqa: BLE001 - reporting must not affect requests.
            logger.warning(
                "Failed to build or record process-request tool-use event: {}",
                exc.__class__.__name__,
            )

    _record_process_request_tool_use_failure = record_process_request_failure

    async def record_prepare_failure(
        self,
        *,
        context: RequestContext,
        params: dict[str, Any],
        exc: Exception,
        start_ts: float,
    ) -> None:
        if not self.should_record(context):
            return
        status, reason_code = classify_tool_use_exception(exc)
        scope_payload = None
        if isinstance(exc, GovernanceDeniedError) and isinstance(exc.governance, dict):
            path_scope = exc.governance.get("path_scope")
            if isinstance(path_scope, dict):
                scope_payload = path_scope
        event = self.build_event(
            context=context,
            requested_tool_name=params.get("name") if isinstance(params, dict) else None,
            status=status,
            execution_origin="failed_before_execution",
            duration_ms=self.duration_ms(start_ts),
            reason_code=reason_code,
            tool_args=params.get("arguments") if isinstance(params, dict) else None,
            scope_payload=scope_payload,
        )
        await self.record_event(event)

    def audit_tool_event(
        self,
        context: RequestContext,
        tool_name: str,
        module_id: str | None,
        status: str,
        duration_ms: float,
        arguments_hash: str | None,
        error: Exception | None = None,
        reason_code: str | None = None,
    ) -> None:
        """Emit a best-effort structured audit log for a completed tool execution."""

        try:
            error_type = _safe_exception_family(error) if error is not None else None
            safe_reason_code = sanitize_reason_code(reason_code)
            if error is not None and safe_reason_code is None:
                _status, classified_reason_code = classify_tool_use_exception(error)
                safe_reason_code = sanitize_reason_code(classified_reason_code)
            log = logger.bind(
                audit=True,
                request_id=context.request_id,
                user_id=context.user_id,
                client_id=context.client_id,
                session_id=context.session_id,
                tool=tool_name,
                module=module_id or "unknown",
                duration_ms=round(duration_ms, 2),
                arguments_hash=arguments_hash,
                status=status,
                error_type=error_type,
                reason_code=safe_reason_code,
            )
            if error:
                log.error(
                    "MCP tool execution failed: {error_type}",
                    error_type=error_type,
                )
            else:
                log.info("MCP tool executed")
        except asyncio.CancelledError:
            raise
        except self._noncritical_exceptions as exc:
            logger.debug(
                "MCP audit log emission skipped after noncritical failure: {error_type}",
                error_type=_safe_exception_family(exc),
            )

    _audit_tool_event = audit_tool_event
