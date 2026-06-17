"""Configurable host-neutral MCP tool-call hook manager."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Iterable
from typing import Any, cast

from loguru import logger

from mcp_unified.interfaces.runtime import (
    ToolHookAction,
    ToolHookCallContext,
    ToolHookDecision,
    ToolHookPhase,
)

from .models import (
    ToolHookCallback,
    ToolHookExecutionError,
    ToolHookRegistration,
    ToolHookResult,
)

_VALID_HOOK_ACTIONS = {"allow", "deny", "ask", "approval_required"}


class ConfiguredToolCallHookManager:
    """Run ordered tool-call hooks with protocol-compatible decisions."""

    def __init__(self, registrations: Iterable[ToolHookRegistration] = ()) -> None:
        self._registrations = tuple(
            sorted(
                registrations,
                key=lambda registration: (registration.order, registration.hook_id),
            )
        )

    @property
    def registrations(self) -> tuple[ToolHookRegistration, ...]:
        """Return configured registrations in execution order."""

        return self._registrations

    async def before_tool_call(
        self,
        context: ToolHookCallContext,
    ) -> ToolHookDecision | None:
        """Run pre-tool hooks and return the first blocking decision."""

        results: list[dict[str, Any]] = []
        for registration in self._iter_phase("pre"):
            callback = self._before_callback(registration)
            if callback is None:
                continue
            try:
                raw_decision = await _maybe_await(callback(context))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise ToolHookExecutionError(
                    hook_id=registration.hook_id,
                    phase="pre",
                    error_type=exc.__class__.__name__,
                ) from exc

            decision = _coerce_decision(raw_decision)
            result = _hook_result_payload(
                decision,
                registration=registration,
                phase="pre",
            )
            results.append(result)
            if _normalized_action(decision.action) != "allow":
                return _decision_with_manager_metadata(
                    decision,
                    registration=registration,
                    results=results,
                )

        if results:
            return ToolHookDecision(action="allow", metadata={"hook_results": results})
        return None

    async def after_tool_call(
        self,
        context: ToolHookCallContext,
    ) -> ToolHookDecision | None:
        """Run post-tool hooks while preserving the original tool outcome."""

        results: list[dict[str, Any]] = []
        for registration in self._iter_phase("post"):
            callback = self._after_callback(registration)
            if callback is None:
                continue
            try:
                raw_decision = await _maybe_await(callback(context))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "MCP post-tool hook failed; continuing. hook_id={} error_type={}",
                    registration.hook_id,
                    exc.__class__.__name__,
                )
                results.append(
                    _hook_error_payload(
                        registration=registration,
                        phase="post",
                        error_type=exc.__class__.__name__,
                    )
                )
                continue

            decision = _coerce_decision(raw_decision)
            results.append(
                _hook_result_payload(
                    decision,
                    registration=registration,
                    phase="post",
                )
            )

        if results:
            return ToolHookDecision(action="allow", metadata={"hook_results": results})
        return None

    def _iter_phase(self, phase: ToolHookPhase) -> Iterable[ToolHookRegistration]:
        """Yield enabled registrations that apply to one phase."""

        for registration in self._registrations:
            if registration.enabled and phase in registration.phases:
                yield registration

    @staticmethod
    def _before_callback(registration: ToolHookRegistration) -> ToolHookCallback | None:
        """Return the callback for a pre-hook registration."""

        if registration.before is not None:
            return registration.before
        before = getattr(registration.hook, "before_tool_call", None)
        return cast(ToolHookCallback | None, before) if callable(before) else None

    @staticmethod
    def _after_callback(registration: ToolHookRegistration) -> ToolHookCallback | None:
        """Return the callback for a post-hook registration."""

        if registration.after is not None:
            return registration.after
        after = getattr(registration.hook, "after_tool_call", None)
        return cast(ToolHookCallback | None, after) if callable(after) else None


async def _maybe_await(value: ToolHookResult | Any) -> ToolHookResult:
    """Await callback results when a hook returned an awaitable."""

    if inspect.isawaitable(value):
        return await value
    return cast(ToolHookResult, value)


def _normalized_action(action: Any) -> ToolHookAction:
    """Return a supported hook action, defaulting invalid values to deny."""

    normalized = str(action or "allow").strip().lower()
    if normalized in _VALID_HOOK_ACTIONS:
        return cast(ToolHookAction, normalized)
    return "deny"


def _coerce_decision(decision: ToolHookResult) -> ToolHookDecision:
    """Normalize callback return values into a typed decision."""

    if isinstance(decision, ToolHookDecision):
        return decision
    if decision is None:
        return ToolHookDecision(action="allow")
    if isinstance(decision, dict):
        metadata = decision.get("metadata")
        action = _normalized_action(decision.get("action") or decision.get("status"))
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
    )


def _hook_result_payload(
    decision: ToolHookDecision,
    *,
    registration: ToolHookRegistration,
    phase: ToolHookPhase,
) -> dict[str, Any]:
    """Return safe metadata for one hook result."""

    action = _normalized_action(decision.action)
    status = "ask" if action == "approval_required" else action
    payload: dict[str, Any] = {
        "phase": phase,
        "hook_id": registration.hook_id,
        "hook_order": registration.order,
        "action": action,
        "status": status,
    }
    if decision.reason_code:
        payload["reason_code"] = str(decision.reason_code)
    return payload


def _hook_error_payload(
    *,
    registration: ToolHookRegistration,
    phase: ToolHookPhase,
    error_type: str,
) -> dict[str, Any]:
    """Return safe metadata for one hook execution failure."""

    return {
        "phase": phase,
        "hook_id": registration.hook_id,
        "hook_order": registration.order,
        "action": "deny",
        "status": "error",
        "reason_code": "tool_hook_unavailable",
        "error_type": error_type,
    }


def _decision_with_manager_metadata(
    decision: ToolHookDecision,
    *,
    registration: ToolHookRegistration,
    results: list[dict[str, Any]],
) -> ToolHookDecision:
    """Attach manager provenance to a blocking pre-hook decision."""

    metadata = dict(decision.metadata) if isinstance(decision.metadata, dict) else {}
    metadata.setdefault("hook_id", registration.hook_id)
    metadata.setdefault("hook_order", registration.order)
    metadata["hook_results"] = list(results)
    return ToolHookDecision(
        action=decision.action,
        reason_code=decision.reason_code,
        message=decision.message,
        metadata=metadata,
    )


__all__ = ["ConfiguredToolCallHookManager"]
