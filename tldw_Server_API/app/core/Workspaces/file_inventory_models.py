from __future__ import annotations

import base64
import binascii
import json
import re
from collections.abc import Iterable, Mapping
from typing import Any, Literal, TypedDict, cast

WorkspaceFileInventoryDurableState = Literal[
    "queued",
    "scanning",
    "current",
    "partial",
    "failed",
    "disabled",
]
WorkspaceFileInventoryState = Literal[
    "not_started",
    "queued",
    "scanning",
    "current",
    "partial",
    "stale",
    "failed",
    "disabled",
]


class WorkspaceFileInventoryCounts(TypedDict):
    files: int
    directories: int
    symlinks: int
    ignored: int
    indexing_candidates: int
    diagnostics: int
    total_entries: int


class WorkspaceFileInventoryDiagnostic(TypedDict, total=False):
    code: str
    path_hint: str
    message: str


DURABLE_INVENTORY_STATES: frozenset[str] = frozenset(
    {"queued", "scanning", "current", "partial", "failed", "disabled"}
)
PROJECTED_INVENTORY_STATES: frozenset[str] = DURABLE_INVENTORY_STATES | frozenset(
    {"not_started", "stale"}
)
INVENTORY_COUNT_KEYS: tuple[str, ...] = (
    "files",
    "directories",
    "symlinks",
    "ignored",
    "indexing_candidates",
    "diagnostics",
    "total_entries",
)
MAX_INVENTORY_DIAGNOSTICS = 50
MAX_INVENTORY_PATH_HINT_LENGTH = 240
MAX_INVENTORY_DIAGNOSTIC_MESSAGE_LENGTH = 200

_CURSOR_VERSION = 1
_DEFAULT_DIAGNOSTIC_CODE = "scan_diagnostic"
_DEFAULT_DIAGNOSTIC_MESSAGE = "A path could not be inspected."
_ABSOLUTE_PATH_PATTERN = re.compile(r"(?:[A-Za-z]:[\\/][^\s]+|/[^\s]+)")
_SAFE_CODE_PATTERN = re.compile(r"[^a-zA-Z0-9_.-]+")
_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:/")


def normalize_inventory_state(value: Any) -> WorkspaceFileInventoryState:
    normalized = str(value or "").strip().lower()
    if normalized in PROJECTED_INVENTORY_STATES:
        return cast(WorkspaceFileInventoryState, normalized)
    return "failed"


def normalize_durable_inventory_state(value: Any) -> WorkspaceFileInventoryDurableState:
    normalized = str(value or "").strip().lower()
    if normalized in DURABLE_INVENTORY_STATES:
        return cast(WorkspaceFileInventoryDurableState, normalized)
    return "failed"


def normalize_inventory_counts(value: Any) -> WorkspaceFileInventoryCounts:
    source = value if isinstance(value, Mapping) else {}
    return cast(
        WorkspaceFileInventoryCounts,
        {key: _non_negative_int(source.get(key)) for key in INVENTORY_COUNT_KEYS},
    )


def bounded_inventory_diagnostics(
    value: Any,
    *,
    root_relative_only: bool = True,
    limit: int = MAX_INVENTORY_DIAGNOSTICS,
) -> list[WorkspaceFileInventoryDiagnostic]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        return []

    max_items = max(0, min(limit, MAX_INVENTORY_DIAGNOSTICS))
    diagnostics: list[WorkspaceFileInventoryDiagnostic] = []
    for item in value:
        if len(diagnostics) >= max_items:
            break
        if not isinstance(item, Mapping):
            continue

        diagnostic: WorkspaceFileInventoryDiagnostic = {
            "code": _sanitize_diagnostic_code(item.get("code")),
            "message": _sanitize_diagnostic_message(item.get("message")),
        }
        path_hint = redact_inventory_path_hint(item.get("path_hint"))
        if path_hint:
            diagnostic["path_hint"] = path_hint
        diagnostics.append(diagnostic)

    return diagnostics


def redact_inventory_path_hint(value: Any) -> str | None:
    raw = _string_or_none(value)
    if raw is None or "\x00" in raw:
        return None

    normalized = raw.replace("\\", "/")
    if _looks_absolute_or_home_path(normalized):
        path_hint = normalized.rstrip("/").rsplit("/", 1)[-1]
    else:
        path_hint = "/".join(
            part for part in normalized.split("/") if part and part not in {".", ".."}
        )

    if not path_hint:
        return None
    if len(path_hint) > MAX_INVENTORY_PATH_HINT_LENGTH:
        return path_hint[-MAX_INVENTORY_PATH_HINT_LENGTH:]
    return path_hint


def encode_inventory_cursor(relative_path: str) -> str:
    normalized = _normalize_inventory_relative_path(relative_path)
    if normalized is None:
        raise ValueError("Inventory cursors require a safe relative path.")

    payload = json.dumps(
        {"v": _CURSOR_VERSION, "relative_path": normalized},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def decode_inventory_cursor(cursor: str) -> str:
    raw = _string_or_none(cursor)
    if raw is None:
        raise ValueError("Inventory cursor is required.")

    padded = raw + ("=" * (-len(raw) % 4))
    try:
        decoded = base64.b64decode(padded, altchars=b"-_", validate=True)
        payload = json.loads(decoded.decode("utf-8"))
    except (binascii.Error, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Inventory cursor is invalid.") from exc

    if not isinstance(payload, Mapping) or payload.get("v") != _CURSOR_VERSION:
        raise ValueError("Inventory cursor is invalid.")

    normalized = _normalize_inventory_relative_path(payload.get("relative_path"))
    if normalized is None:
        raise ValueError("Inventory cursor path is invalid.")
    return normalized


def sort_inventory_relative_paths(paths: Iterable[str]) -> list[str]:
    normalized_paths: list[str] = []
    for path in paths:
        normalized = _normalize_inventory_relative_path(path)
        if normalized is None:
            raise ValueError("Inventory paths must be safe relative paths.")
        normalized_paths.append(normalized)
    return sorted(normalized_paths)


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value if value >= 0 else 0
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdecimal():
            return int(stripped)
    return 0


def _sanitize_diagnostic_code(value: Any) -> str:
    raw = _string_or_none(value)
    if raw is None:
        return _DEFAULT_DIAGNOSTIC_CODE
    cleaned = _SAFE_CODE_PATTERN.sub("_", raw.strip())[:64].strip("_")
    return cleaned or _DEFAULT_DIAGNOSTIC_CODE


def _sanitize_diagnostic_message(value: Any) -> str:
    raw = _string_or_none(value)
    if raw is None:
        return _DEFAULT_DIAGNOSTIC_MESSAGE
    redacted = _ABSOLUTE_PATH_PATTERN.sub("[redacted-path]", raw)
    collapsed = " ".join(redacted.split())
    if not collapsed:
        return _DEFAULT_DIAGNOSTIC_MESSAGE
    if len(collapsed) > MAX_INVENTORY_DIAGNOSTIC_MESSAGE_LENGTH:
        return collapsed[: MAX_INVENTORY_DIAGNOSTIC_MESSAGE_LENGTH - 3].rstrip() + "..."
    return collapsed


def _normalize_inventory_relative_path(value: Any) -> str | None:
    raw = _string_or_none(value)
    if raw is None or "\x00" in raw:
        return None

    normalized = raw.replace("\\", "/")
    if _looks_absolute_or_home_path(normalized):
        return None

    parts: list[str] = []
    for part in normalized.split("/"):
        if not part or part == ".":
            continue
        if part == "..":
            return None
        parts.append(part)

    if not parts:
        return None
    return "/".join(parts)


def _looks_absolute_or_home_path(value: str) -> bool:
    return (
        value.startswith("/")
        or value.startswith("~")
        or value.startswith("//")
        or _WINDOWS_DRIVE_PATTERN.match(value) is not None
    )


def _string_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


__all__ = [
    "DURABLE_INVENTORY_STATES",
    "INVENTORY_COUNT_KEYS",
    "MAX_INVENTORY_DIAGNOSTICS",
    "MAX_INVENTORY_DIAGNOSTIC_MESSAGE_LENGTH",
    "MAX_INVENTORY_PATH_HINT_LENGTH",
    "PROJECTED_INVENTORY_STATES",
    "WorkspaceFileInventoryCounts",
    "WorkspaceFileInventoryDiagnostic",
    "WorkspaceFileInventoryDurableState",
    "WorkspaceFileInventoryState",
    "bounded_inventory_diagnostics",
    "decode_inventory_cursor",
    "encode_inventory_cursor",
    "normalize_durable_inventory_state",
    "normalize_inventory_counts",
    "normalize_inventory_state",
    "redact_inventory_path_hint",
    "sort_inventory_relative_paths",
]
