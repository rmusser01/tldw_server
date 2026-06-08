from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


WORKSPACE_MEMBERSHIP_RESOURCE_TYPES = frozenset(
    {"workspace_note", "media", "workspace_source", "workspace_artifact", "chat"}
)
WORKSPACE_MEMBERSHIP_FUTURE_RESOURCE_TYPES = frozenset(
    {
        "note",
        "prompt",
        "workflow",
        "watchlist",
        "acp_session",
        "sandbox_session",
        "project_file",
        "study_deck",
        "quiz",
        "study_pack",
    }
)
WORKSPACE_MEMBERSHIP_ROLES = frozenset(
    {"member", "source", "artifact", "conversation", "runtime", "reference"}
)
WORKSPACE_MEMBERSHIP_TRANSFER_POLICIES = frozenset({"link", "copy", "promote", "import"})
WORKSPACE_MEMBERSHIP_CURSOR_MAX_BYTES = 2048

_CURSOR_VERSION = 1
_MEMBERSHIP_CURSOR_KEYS = frozenset({"v", "updated_at", "resource_type", "resource_id"})
_RESOURCE_MEMBERSHIP_CURSOR_KEYS = frozenset({"v", "updated_at", "workspace_id"})


@dataclass(frozen=True)
class WorkspaceMembershipCursor:
    updated_at: str
    resource_type: str
    resource_id: str


@dataclass(frozen=True)
class WorkspaceResourceMembershipCursor:
    updated_at: str
    workspace_id: str


@dataclass(frozen=True)
class WorkspaceResourceRef:
    resource_type: str
    resource_id: str
    title: str | None = None
    subtitle: str | None = None
    href: str | None = None
    updated_at: str | None = None
    state: str = "available"
    metadata: Mapping[str, Any] = field(default_factory=dict)


def encode_membership_cursor(cursor: WorkspaceMembershipCursor) -> str:
    if cursor.resource_type not in WORKSPACE_MEMBERSHIP_RESOURCE_TYPES:
        raise ValueError("Workspace membership cursor resource_type is invalid.")
    _require_non_empty_string(cursor.updated_at, "updated_at")
    _require_non_empty_string(cursor.resource_id, "resource_id")
    return _encode_cursor_payload(
        {
            "v": _CURSOR_VERSION,
            "updated_at": cursor.updated_at,
            "resource_type": cursor.resource_type,
            "resource_id": cursor.resource_id,
        }
    )


def decode_membership_cursor(value: str) -> WorkspaceMembershipCursor:
    payload = _decode_cursor_payload(value)
    if set(payload) != _MEMBERSHIP_CURSOR_KEYS or not _cursor_version_is_valid(payload.get("v")):
        raise ValueError("Workspace membership cursor is invalid.")

    updated_at = _require_non_empty_string(payload.get("updated_at"), "updated_at")
    resource_type = _require_non_empty_string(payload.get("resource_type"), "resource_type")
    resource_id = _require_non_empty_string(payload.get("resource_id"), "resource_id")
    if resource_type not in WORKSPACE_MEMBERSHIP_RESOURCE_TYPES:
        raise ValueError("Workspace membership cursor resource_type is invalid.")
    return WorkspaceMembershipCursor(
        updated_at=updated_at,
        resource_type=resource_type,
        resource_id=resource_id,
    )


def encode_resource_membership_cursor(cursor: WorkspaceResourceMembershipCursor) -> str:
    _require_non_empty_string(cursor.updated_at, "updated_at")
    _require_non_empty_string(cursor.workspace_id, "workspace_id")
    return _encode_cursor_payload(
        {
            "v": _CURSOR_VERSION,
            "updated_at": cursor.updated_at,
            "workspace_id": cursor.workspace_id,
        }
    )


def decode_resource_membership_cursor(value: str) -> WorkspaceResourceMembershipCursor:
    payload = _decode_cursor_payload(value)
    if set(payload) != _RESOURCE_MEMBERSHIP_CURSOR_KEYS or not _cursor_version_is_valid(payload.get("v")):
        raise ValueError("Workspace resource membership cursor is invalid.")

    return WorkspaceResourceMembershipCursor(
        updated_at=_require_non_empty_string(payload.get("updated_at"), "updated_at"),
        workspace_id=_require_non_empty_string(payload.get("workspace_id"), "workspace_id"),
    )


def _encode_cursor_payload(payload: Mapping[str, Any]) -> str:
    try:
        raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Workspace membership cursor is invalid.") from exc
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_cursor_payload(value: str) -> Mapping[str, Any]:
    raw = _require_non_empty_string(value, "cursor")
    if len(raw.encode("utf-8")) > WORKSPACE_MEMBERSHIP_CURSOR_MAX_BYTES:
        raise ValueError("Workspace membership cursor is invalid.")
    padded = raw + ("=" * (-len(raw) % 4))
    try:
        decoded = base64.b64decode(padded, altchars=b"-_", validate=True)
        payload = json.loads(decoded.decode("utf-8"))
    except (binascii.Error, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Workspace membership cursor is invalid.") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Workspace membership cursor is invalid.")
    return payload


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Workspace membership cursor requires string {field_name}.")
    return value


def _cursor_version_is_valid(value: Any) -> bool:
    return type(value) is int and value == _CURSOR_VERSION
