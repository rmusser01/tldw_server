from __future__ import annotations

from typing import Any, Literal, TypedDict

WorkspaceProfile = Literal["research", "project"]
WorkspaceKind = Literal["research_workspace", "project_workspace"]
ProjectRootBackend = Literal["host_local", "sandbox_volume"]
ProjectRootState = Literal[
    "not_configured",
    "attached",
    "missing",
    "detached",
    "failed",
    "archived",
]
ResolutionStatus = Literal["complete", "partial", "failed"]


class AllowedAction(TypedDict):
    allowed: bool
    reason_code: str | None


def normalize_workspace_profile(value: Any) -> WorkspaceProfile:
    return "project" if str(value or "").strip().lower() == "project" else "research"


def workspace_kind_for_profile(profile: WorkspaceProfile) -> WorkspaceKind:
    return "project_workspace" if profile == "project" else "research_workspace"


def normalize_project_root_state(value: Any) -> ProjectRootState:
    normalized = str(value or "").strip().lower()
    if normalized in {"not_configured", "attached", "missing", "detached", "failed", "archived"}:
        return normalized  # type: ignore[return-value]
    return "failed"


def allowed_action() -> AllowedAction:
    return {"allowed": True, "reason_code": None}


def fail_closed_action(reason_code: str) -> AllowedAction:
    return {"allowed": False, "reason_code": reason_code}
