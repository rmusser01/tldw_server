"""Shared MCP file-policy action taxonomy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FilePolicyAction = Literal[
    "read",
    "edit",
    "write",
    "delete",
    "rename",
    "move",
    "share",
    "export",
    "chmod",
    "admin",
    "lock",
]

FILE_POLICY_ACTIONS = frozenset(
    {
        "read",
        "edit",
        "write",
        "delete",
        "rename",
        "move",
        "share",
        "export",
        "chmod",
        "admin",
        "lock",
    }
)
FILE_POLICY_EXISTING_TOOL_ACTIONS = frozenset({"read", "edit", "write"})
FILE_POLICY_DESTRUCTIVE_ACTIONS = frozenset({"delete", "rename", "move"})
FILE_POLICY_EXFILTRATION_ACTIONS = frozenset({"share", "export"})
FILE_POLICY_ADMIN_ACTIONS = frozenset({"chmod", "admin"})
FILE_POLICY_LOCK_ACTIONS = frozenset({"lock"})


@dataclass(frozen=True, slots=True)
class FilePolicyActionMetadata:
    """Operator-facing metadata for one file-policy action."""

    action: FilePolicyAction
    family: str
    description: str
    implemented: bool
    risk: str

    def as_dict(self) -> dict[str, str | bool]:
        """Return a JSON-serializable metadata payload."""

        return {
            "action": self.action,
            "family": self.family,
            "description": self.description,
            "implemented": self.implemented,
            "risk": self.risk,
        }


_ACTION_METADATA: dict[str, FilePolicyActionMetadata] = {
    "read": FilePolicyActionMetadata(
        action="read",
        family="read",
        description="Inspect file or directory content and metadata inside an allowed workspace path.",
        implemented=True,
        risk="read",
    ),
    "edit": FilePolicyActionMetadata(
        action="edit",
        family="bounded_edit",
        description="Modify existing file content through bounded edit tools with preimage checks.",
        implemented=True,
        risk="mutation",
    ),
    "write": FilePolicyActionMetadata(
        action="write",
        family="whole_write",
        description="Create or replace complete file content inside an allowed workspace path.",
        implemented=True,
        risk="mutation",
    ),
    "delete": FilePolicyActionMetadata(
        action="delete",
        family="destructive",
        description="Delete file or directory content inside an allowed workspace path.",
        implemented=False,
        risk="destructive",
    ),
    "rename": FilePolicyActionMetadata(
        action="rename",
        family="destructive",
        description="Rename a file or directory inside an allowed workspace path.",
        implemented=False,
        risk="destructive",
    ),
    "move": FilePolicyActionMetadata(
        action="move",
        family="destructive",
        description="Move a file or directory between allowed workspace paths.",
        implemented=False,
        risk="destructive",
    ),
    "share": FilePolicyActionMetadata(
        action="share",
        family="exfiltration",
        description="Share or publish file content outside the active workspace boundary.",
        implemented=False,
        risk="exfiltration",
    ),
    "export": FilePolicyActionMetadata(
        action="export",
        family="exfiltration",
        description="Export file content outside the active workspace boundary.",
        implemented=False,
        risk="exfiltration",
    ),
    "chmod": FilePolicyActionMetadata(
        action="chmod",
        family="admin",
        description="Change filesystem permission bits inside an allowed workspace path.",
        implemented=False,
        risk="admin",
    ),
    "admin": FilePolicyActionMetadata(
        action="admin",
        family="admin",
        description="Perform administrative filesystem operations inside an allowed workspace path.",
        implemented=False,
        risk="admin",
    ),
    "lock": FilePolicyActionMetadata(
        action="lock",
        family="lock",
        description="Acquire or release coordination locks for allowed workspace paths.",
        implemented=False,
        risk="coordination",
    ),
}


def normalize_file_policy_action(value: object) -> FilePolicyAction | None:
    """Normalize an arbitrary value into a known file-policy action."""

    action = str(value or "").strip().lower()
    if action not in FILE_POLICY_ACTIONS:
        return None
    return action  # type: ignore[return-value]


def get_file_policy_action_metadata(action: object) -> FilePolicyActionMetadata:
    """Return metadata for a known file-policy action."""

    normalized = normalize_file_policy_action(action)
    if normalized is None:
        raise ValueError("unknown file policy action")
    metadata = _ACTION_METADATA.get(normalized)
    if metadata is None:
        raise ValueError(f"file policy action metadata is missing for {normalized!r}")
    return metadata


def format_file_policy_action_list(actions: frozenset[str] = FILE_POLICY_ACTIONS) -> str:
    """Return a human-readable list of file-policy actions."""

    ordered = [action for action in _ACTION_METADATA if action in actions]
    if len(ordered) <= 1:
        return "".join(ordered)
    return ", ".join(ordered[:-1]) + f", or {ordered[-1]}"


__all__ = [
    "FILE_POLICY_ACTIONS",
    "FILE_POLICY_ADMIN_ACTIONS",
    "FILE_POLICY_DESTRUCTIVE_ACTIONS",
    "FILE_POLICY_EXISTING_TOOL_ACTIONS",
    "FILE_POLICY_EXFILTRATION_ACTIONS",
    "FILE_POLICY_LOCK_ACTIONS",
    "FilePolicyAction",
    "FilePolicyActionMetadata",
    "format_file_policy_action_list",
    "get_file_policy_action_metadata",
    "normalize_file_policy_action",
]
