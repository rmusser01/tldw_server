"""ACP branch capability and fallback decisions for chat macros."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .context_snapshot import MacroContextSnapshot

BranchStrategy = Literal["auto", "chat_native", "acp_fork"]
ResolvedBranchStrategy = Literal["chat_native", "acp_fork", "failed"]


class AcpBranchCapability(BaseModel):
    """ACP fork capability derived from a macro context snapshot."""

    model_config = ConfigDict(extra="forbid")

    available: bool
    resumable: bool
    session_id: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BranchStrategyDecision(BaseModel):
    """Execution strategy selected for one branch prompt."""

    model_config = ConfigDict(extra="forbid")

    strategy: ResolvedBranchStrategy
    fallback: bool = False
    required_failed: bool = False
    error_code: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def resolve_acp_branch_capability(snapshot: MacroContextSnapshot) -> AcpBranchCapability:
    """Return whether the stored context can support an ACP fork branch."""
    acp = dict(snapshot.acp or {})
    forkable = bool(acp.get("forkable"))
    resumable = bool(acp.get("resumable"))
    if snapshot.acp_session_id and forkable and resumable:
        metadata = {key: value for key, value in acp.items() if key not in {"forkable", "resumable"}}
        return AcpBranchCapability(
            available=True,
            resumable=True,
            session_id=snapshot.acp_session_id,
            metadata=metadata,
        )
    reason = "acp_unavailable"
    if snapshot.acp_session_id and not resumable:
        reason = "acp_not_resumable"
    return AcpBranchCapability(
        available=False,
        resumable=resumable,
        session_id=snapshot.acp_session_id,
        reason=reason,
        metadata={key: value for key, value in acp.items() if key not in {"forkable", "resumable"}},
    )


def select_branch_strategy(
    *,
    step_strategy: str | None,
    macro_strategy: str | None,
    capability: AcpBranchCapability,
) -> BranchStrategyDecision:
    """Select chat-native, ACP fork, or required-ACP failure for one branch."""
    requested = (step_strategy or macro_strategy or "auto").strip().lower()
    if requested == "chat_native":
        return BranchStrategyDecision(
            strategy="chat_native",
            metadata={"requested": requested, "reason": "chat_native_requested"},
        )
    if requested == "acp_fork":
        if capability.available:
            return BranchStrategyDecision(
                strategy="acp_fork",
                metadata={"requested": requested, "session_id": capability.session_id},
            )
        return BranchStrategyDecision(
            strategy="failed",
            required_failed=True,
            error_code="acp_unavailable",
            metadata={
                "requested": requested,
                "reason": capability.reason or "acp_unavailable",
                "session_id": capability.session_id,
            },
        )
    return BranchStrategyDecision(
        strategy="chat_native",
        fallback=not capability.available,
        metadata={
            "requested": requested or "auto",
            "reason": capability.reason or "chat_native_default",
            "session_id": capability.session_id,
        },
    )
