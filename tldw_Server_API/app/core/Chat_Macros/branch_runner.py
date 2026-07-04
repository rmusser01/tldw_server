"""Branch prompt runner seam for chat macro execution."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .context_snapshot import MacroContextSnapshot


class BranchPromptResult(BaseModel):
    """Normalized result from one branch prompt execution."""

    model_config = ConfigDict(extra="forbid")

    text: str = ""
    status: str = "completed"
    citations: list[Any] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    acp_child_session_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None


class BranchPromptRunner(Protocol):
    """Protocol for fakeable branch prompt execution."""

    async def run_branch(
        self,
        *,
        prompt: str,
        snapshot: MacroContextSnapshot,
        model_selection: dict[str, Any],
    ) -> BranchPromptResult:
        """Run one branch prompt against a stable context snapshot."""
