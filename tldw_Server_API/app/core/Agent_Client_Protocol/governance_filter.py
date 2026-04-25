"""GovernanceFilter -- pipeline stage between adapter and event bus.

Intercepts TOOL_CALL events and applies permission tier logic:
- ``auto`` tier tools pass through immediately.
- ``batch`` / ``individual`` tier tools are held pending human approval.

Unanswered permission requests are automatically denied after
``default_timeout_sec`` seconds.
"""
from __future__ import annotations

import asyncio
import fnmatch
import time
import uuid
from contextlib import suppress
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Agent_Client_Protocol.event_bus import SessionEventBus
from tldw_Server_API.app.core.Agent_Client_Protocol.events import AgentEvent, AgentEventKind
from tldw_Server_API.app.core.Agent_Client_Protocol.policy_conditions import (
    PolicyConditions,
    evaluate_conditions,
)
from tldw_Server_API.app.core.Agent_Client_Protocol.permission_tiers import determine_permission_tier
from tldw_Server_API.app.core.Agent_Client_Protocol.tool_gate import ToolGate, ToolGateResult


class _PendingEntry:
    """A held tool_call with its creation time for timeout enforcement."""

    __slots__ = ("event", "created_at", "timeout_task", "approval_future", "internal_only")

    def __init__(
        self,
        event: AgentEvent,
        timeout_task: asyncio.Task[None] | None = None,
        approval_future: asyncio.Future | None = None,
        internal_only: bool = False,
    ) -> None:
        self.event = event
        self.created_at = time.monotonic()
        self.timeout_task = timeout_task
        self.approval_future = approval_future
        self.internal_only = internal_only


class GovernanceFilter:
    """Pipeline stage that gates tool calls based on permission tiers.

    The adapter's ``event_callback`` should point at :meth:`process`.  Non-tool
    events are forwarded to the bus immediately.  Tool calls are classified via
    :func:`determine_permission_tier` and either forwarded (``auto``) or held
    until a human decision arrives via :meth:`on_permission_response`.

    If no response arrives within ``default_timeout_sec``, the held tool call
    is automatically denied.
    """

    def __init__(
        self,
        bus: SessionEventBus,
        default_timeout_sec: int = 300,
        policy_snapshot: Any | None = None,
        session_metadata: dict[str, Any] | None = None,
        permission_decision_service: Any | None = None,
    ) -> None:
        self._bus = bus
        self._default_timeout_sec = default_timeout_sec
        self._pending: dict[str, _PendingEntry] = {}
        self._snapshot = policy_snapshot
        self._session_metadata = session_metadata or {}
        self._perm_service = permission_decision_service

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def pending_count(self) -> int:
        """Number of tool calls awaiting a permission decision."""
        return len(self._pending)

    # ------------------------------------------------------------------
    # MCPHub snapshot policy check
    # ------------------------------------------------------------------

    def _check_snapshot_policy(self, tool_name: str) -> str | None:
        """Check the MCPHub policy snapshot for a tool-level decision.

        Returns:
            ``"_deny"`` if the tool is explicitly denied,
            ``"auto"`` if the tool is explicitly allowed,
            the tier string if a ``tool_tier_overrides`` pattern matches,
            or ``None`` if the snapshot has no opinion (fall through to
            existing tier heuristics).
        """
        if self._snapshot is None:
            return None

        doc = getattr(self._snapshot, "resolved_policy_document", None)
        if not doc or not isinstance(doc, dict):
            return None

        conditions = PolicyConditions.from_dict(doc.get("conditions"))
        if not conditions.is_empty():
            resource_labels = self._session_metadata.get("labels", {})
            ancestry_chain = self._session_metadata.get("ancestry_chain", [])
            source_ip = self._session_metadata.get("client_ip")
            if not evaluate_conditions(
                conditions,
                resource_labels=resource_labels,
                ancestry_chain=ancestry_chain,
                source_ip=source_ip,
            ):
                return None  # Conditions failed, policy doesn't apply

        # 1. Check denied_tools (highest priority)
        for pattern in doc.get("denied_tools", []):
            if fnmatch.fnmatch(tool_name, pattern):
                return "_deny"

        # 2. Check allowed_tools
        for pattern in doc.get("allowed_tools", []):
            if fnmatch.fnmatch(tool_name, pattern):
                return "auto"

        # 3. Check tool_tier_overrides (pattern -> tier mapping)
        for pattern, tier in doc.get("tool_tier_overrides", {}).items():
            if fnmatch.fnmatch(tool_name, pattern):
                return tier

        return None

    # ------------------------------------------------------------------
    # Pipeline entry point
    # ------------------------------------------------------------------

    async def process(self, event: AgentEvent) -> None:
        """Route *event* through governance logic.

        Non-TOOL_CALL events are forwarded to the bus immediately.
        TOOL_CALL events are classified by permission tier.
        """
        if event.kind != AgentEventKind.TOOL_CALL:
            await self._bus.publish(event)
            return

        if event.metadata.get("_already_approved"):
            await self._bus.publish(event)
            return

        tool_name: str = event.payload.get("tool_name", "")
        approval_future = event.metadata.get("_approval_future")

        # --- MCPHub snapshot check (unified hierarchy) ---
        snapshot_tier = self._check_snapshot_policy(tool_name)
        if snapshot_tier == "_deny":
            deny_event = AgentEvent(
                session_id=event.session_id,
                kind=AgentEventKind.TOOL_RESULT,
                payload={
                    "tool_call_id": event.payload.get("tool_call_id", ""),
                    "error": f"Tool '{tool_name}' denied by policy",
                },
                metadata={"governance_action": "denied_by_snapshot"},
            )
            await self._bus.publish(deny_event)
            if approval_future is not None and not approval_future.done():
                approval_future.set_result(("deny", f"Tool '{tool_name}' denied by policy"))
            return

        # --- Check persisted permission decisions (step 4 in hierarchy) ---
        if self._perm_service is not None and snapshot_tier is None:
            session_id = event.session_id
            user_id = self._session_metadata.get("user_id")
            if user_id is not None:
                persisted = self._perm_service.check(user_id, tool_name, session_id)
                if persisted == "allow":
                    logger.info(
                        "Governance: persisted decision auto-approved tool '{}'",
                        tool_name,
                    )
                    if approval_future is not None and not approval_future.done():
                        approval_future.set_result(("approve", None))
                    if approval_future is not None:
                        return
                    await self._bus.publish(event)
                    return
                if persisted == "deny":
                    logger.info(
                        "Governance: persisted decision denied tool '{}'",
                        tool_name,
                    )
                    deny_event = AgentEvent(
                        session_id=event.session_id,
                        kind=AgentEventKind.TOOL_RESULT,
                        payload={
                            "tool_call_id": event.payload.get("tool_call_id", ""),
                            "error": f"Tool '{tool_name}' denied by remembered decision",
                        },
                        metadata={"governance_action": "denied_by_persisted_decision"},
                    )
                    await self._bus.publish(deny_event)
                    if approval_future is not None and not approval_future.done():
                        approval_future.set_result(
                            ("deny", f"Tool '{tool_name}' denied by remembered decision")
                        )
                    return

        if snapshot_tier is not None:
            tier = snapshot_tier
        else:
            # Use the adapter-provided tier if present; fall back to re-resolving
            tier = event.payload.get("permission_tier") or determine_permission_tier(tool_name)
        internal_only = approval_future is not None

        if tier == "auto":
            if approval_future is not None and not approval_future.done():
                approval_future.set_result(("approve", None))
            if internal_only:
                return
            await self._bus.publish(event)
            return

        # Hold the event and publish a permission request
        request_id = str(uuid.uuid4())
        timeout_task = asyncio.create_task(self._timeout_pending(request_id, self._default_timeout_sec))
        self._pending[request_id] = _PendingEntry(
            event=event,
            timeout_task=timeout_task,
            approval_future=approval_future,
            internal_only=internal_only,
        )

        perm_request = AgentEvent(
            session_id=event.session_id,
            kind=AgentEventKind.PERMISSION_REQUEST,
            payload={
                "request_id": request_id,
                "tool_name": tool_name,
                "arguments": event.payload.get("arguments", {}),
                "tier": tier,
                "timeout_sec": self._default_timeout_sec,
            },
        )
        await self._bus.publish(perm_request)
        logger.info(
            "Governance: held tool_call '{}' (tier={}) as request_id={}",
            tool_name,
            tier,
            request_id,
        )

    # ------------------------------------------------------------------
    # Permission responses
    # ------------------------------------------------------------------

    async def on_permission_response(
        self,
        request_id: str,
        decision: str,
        reason: str | None = None,
        *,
        remember: bool = False,
        scope: str = "session",
    ) -> None:
        """Handle a human decision for a held tool call.

        Parameters
        ----------
        request_id:
            The id from the PERMISSION_REQUEST payload.
        decision:
            ``"approve"`` or ``"deny"``.
        reason:
            Optional human-supplied reason (used in deny error message).
        remember:
            If ``True`` and a :class:`PermissionDecisionService` is attached,
            persist the decision for future auto-application.
        scope:
            ``"session"`` (default) or ``"global"``.  Only used when
            *remember* is ``True``.
        """
        entry = self._pending.pop(request_id, None)
        if entry is None:
            logger.warning(
                "Governance: permission response for unknown request_id={}",
                request_id,
            )
            return

        # Cancel the timeout task since a decision arrived
        if entry.timeout_task is not None:
            entry.timeout_task.cancel()

        held_event = entry.event

        # Persist the decision when "remember" is requested
        if remember and self._perm_service is not None:
            user_id = self._session_metadata.get("user_id")
            tool_name = held_event.payload.get("tool_name", "")
            if user_id is not None and tool_name:
                self._perm_service.persist(
                    user_id=user_id,
                    tool_pattern=tool_name,
                    decision="allow" if decision == "approve" else "deny",
                    scope=scope,
                    session_id=held_event.session_id,
                    reason=reason,
                )
                logger.info(
                    "Governance: persisted remembered decision for tool '{}' "
                    "(decision={}, scope={})",
                    tool_name,
                    decision,
                    scope,
                )

        # Resolve the approval future if one was attached
        if entry.approval_future is not None and not entry.approval_future.done():
            entry.approval_future.set_result((decision, reason))

        if entry.internal_only:
            logger.info(
                "Governance: resolved internal approval request_id={} decision={}",
                request_id,
                decision,
            )
            return

        if decision == "approve":
            await self._bus.publish(held_event)
            logger.info("Governance: approved request_id={}", request_id)
        else:
            # Emit an error TOOL_RESULT so the agent knows the call was denied
            tool_name = held_event.payload.get("tool_name", "unknown")
            error_msg = f"Permission denied for tool '{tool_name}'"
            if reason:
                error_msg += f": {reason}"

            error_event = AgentEvent(
                session_id=held_event.session_id,
                kind=AgentEventKind.TOOL_RESULT,
                payload={
                    "tool_id": held_event.payload.get("tool_id", ""),
                    "tool_name": tool_name,
                    "output": error_msg,
                    "is_error": True,
                    "duration_ms": 0,
                },
            )
            await self._bus.publish(error_event)
            logger.info(
                "Governance: denied request_id={} reason={}",
                request_id,
                reason,
            )

    # ------------------------------------------------------------------
    # Timeout enforcement
    # ------------------------------------------------------------------

    async def _timeout_pending(self, request_id: str, timeout_sec: int) -> None:
        """Auto-deny a held tool call after *timeout_sec* seconds."""
        try:
            await asyncio.sleep(timeout_sec)
        except asyncio.CancelledError:
            return  # decision arrived before timeout
        if request_id in self._pending:
            logger.info(
                "Governance: permission timeout for request_id={} after {}s",
                request_id,
                timeout_sec,
            )
            await self.on_permission_response(request_id, "deny", reason="timeout")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def cancel_all_pending(self, session_id: str | None = None) -> None:
        """Cancel pending permission requests, optionally scoped to one session."""
        for request_id, entry in list(self._pending.items()):
            if session_id is not None and entry.event.session_id != session_id:
                continue
            await self.on_permission_response(
                request_id,
                decision="deny",
                reason="session cancelled",
            )


class GovernanceToolGate(ToolGate):
    """Concrete ToolGate that delegates to GovernanceFilter.

    Creates an ``asyncio.Future`` per tool call, attaches it to the event
    metadata, and awaits the governance decision.
    """

    def __init__(self, governance_filter: GovernanceFilter) -> None:
        self._filter = governance_filter

    async def request_approval(
        self,
        session_id: str,
        tool_name: str,
        arguments: dict,
        *,
        cancel_event: asyncio.Event | None = None,
    ) -> ToolGateResult:
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        event = AgentEvent(
            session_id=session_id,
            kind=AgentEventKind.TOOL_CALL,
            payload={
                "tool_id": str(uuid.uuid4()),
                "tool_name": tool_name,
                "arguments": arguments,
                "permission_tier": determine_permission_tier(tool_name),
            },
            metadata={"_approval_future": future},
        )
        await self._filter.process(event)
        if cancel_event is None:
            decision, reason = await future
            return ToolGateResult(approved=(decision == "approve"), reason=reason)

        cancel_task = asyncio.create_task(cancel_event.wait())
        try:
            done, _pending = await asyncio.wait(
                {future, cancel_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancel_task in done and cancel_event.is_set():
                await self._filter.cancel_all_pending(session_id=session_id)
            decision, reason = await future
        finally:
            cancel_task.cancel()
            with suppress(asyncio.CancelledError):
                await cancel_task

        return ToolGateResult(approved=(decision == "approve"), reason=reason)
