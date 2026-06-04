from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from stat import S_IMODE
from typing import Any, Literal

from tldw_Server_API.app.core.Workspaces.file_inventory_ignore import (
    InventoryIgnorePolicy,
    should_ignore_inventory_path,
)
from tldw_Server_API.app.core.Workspaces.file_inventory_models import (
    WorkspaceFileInventoryCounts,
    WorkspaceFileInventoryDiagnostic,
    bounded_inventory_diagnostics,
    normalize_inventory_counts,
)

DEFAULT_MAX_FILES = 25_000
DEFAULT_MAX_DIRECTORIES = 10_000
DEFAULT_MAX_DEPTH = 32
DEFAULT_MAX_PATH_LENGTH = 512
DEFAULT_MAX_DIAGNOSTICS = 50
DEFAULT_MAX_SCAN_SECONDS = 120.0

_TEXT_LIKE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".c",
        ".cc",
        ".cfg",
        ".cpp",
        ".css",
        ".csv",
        ".go",
        ".h",
        ".hpp",
        ".html",
        ".ini",
        ".java",
        ".js",
        ".json",
        ".jsx",
        ".log",
        ".md",
        ".py",
        ".rs",
        ".sh",
        ".sql",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }
)
_STATIC_MIME_HINTS: dict[str, str] = {
    ".css": "text/css",
    ".csv": "text/csv",
    ".html": "text/html",
    ".js": "text/javascript",
    ".json": "application/json",
    ".md": "text/markdown",
    ".py": "text/x-python",
    ".sh": "text/x-shellscript",
    ".toml": "application/toml",
    ".ts": "text/typescript",
    ".tsx": "text/typescript",
    ".txt": "text/plain",
    ".xml": "application/xml",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
}

InventoryEntryKind = Literal["file", "directory", "symlink"]


@dataclass(frozen=True)
class InventoryScanBounds:
    max_files: int = DEFAULT_MAX_FILES
    max_directories: int = DEFAULT_MAX_DIRECTORIES
    max_depth: int = DEFAULT_MAX_DEPTH
    max_path_length: int = DEFAULT_MAX_PATH_LENGTH
    max_diagnostics: int = DEFAULT_MAX_DIAGNOSTICS
    max_seconds: float = DEFAULT_MAX_SCAN_SECONDS


@dataclass(frozen=True)
class InventoryScanResult:
    items: tuple[dict[str, Any], ...]
    counts: WorkspaceFileInventoryCounts
    diagnostics: tuple[WorkspaceFileInventoryDiagnostic, ...]
    coverage_complete: bool


def scan_workspace_file_inventory(
    root: Path,
    *,
    policy: InventoryIgnorePolicy,
    bounds: InventoryScanBounds,
) -> InventoryScanResult:
    """Traverse a Workspace root and return metadata-only inventory items."""

    root_path = Path(root)
    normalized_bounds = _normalize_bounds(bounds)
    items: list[dict[str, Any]] = []
    diagnostics: list[WorkspaceFileInventoryDiagnostic] = []
    files_recorded = 0
    directories_recorded = 0
    ignored_count = 0
    coverage_complete = True
    started_at = time.monotonic()

    if root_path.is_symlink():
        _add_diagnostic(
            diagnostics,
            normalized_bounds,
            code="root_symlink_not_supported",
            path_hint=root_path.name,
            message="Workspace file inventory root cannot be a symlink.",
        )
        return _result(items, ignored_count, diagnostics, coverage_complete=False)
    if not root_path.is_dir():
        _add_diagnostic(
            diagnostics,
            normalized_bounds,
            code="root_not_directory",
            path_hint=root_path.name,
            message="Workspace file inventory root is not a directory.",
        )
        return _result(items, ignored_count, diagnostics, coverage_complete=False)

    stack: list[tuple[Path, str, int]] = [(root_path, "", 0)]
    stop_scan = False
    while stack and not stop_scan:
        directory, parent_relative_path, depth = stack.pop()
        if _scan_timed_out(started_at, normalized_bounds):
            coverage_complete = False
            _add_diagnostic(
                diagnostics,
                normalized_bounds,
                code="scan_timeout",
                path_hint=parent_relative_path,
                message="Workspace file inventory scan time limit was reached.",
            )
            break

        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if _scan_timed_out(started_at, normalized_bounds):
                        coverage_complete = False
                        _add_diagnostic(
                            diagnostics,
                            normalized_bounds,
                            code="scan_timeout",
                            path_hint=parent_relative_path,
                            message="Workspace file inventory scan time limit was reached.",
                        )
                        stop_scan = True
                        break

                    relative_path = _join_relative_path(parent_relative_path, entry.name)
                    try:
                        entry_is_dir = entry.is_dir(follow_symlinks=False)
                    except OSError:
                        entry_is_dir = False
                    decision = should_ignore_inventory_path(
                        relative_path,
                        is_dir=entry_is_dir,
                        policy=policy,
                    )
                    if decision.ignored:
                        ignored_count += 1
                        continue

                    if len(relative_path) > normalized_bounds.max_path_length:
                        coverage_complete = False
                        _add_diagnostic(
                            diagnostics,
                            normalized_bounds,
                            code="path_too_long",
                            path_hint=relative_path,
                            message="A path exceeded the inventory path length limit.",
                        )
                        continue

                    entry_depth = depth + 1
                    if entry_depth > normalized_bounds.max_depth:
                        coverage_complete = False
                        _add_diagnostic(
                            diagnostics,
                            normalized_bounds,
                            code="scan_limit_reached",
                            path_hint=relative_path,
                            message="Workspace file inventory depth limit was reached.",
                        )
                        continue

                    metadata = _entry_metadata(entry, relative_path)
                    if metadata is None:
                        coverage_complete = False
                        _add_diagnostic(
                            diagnostics,
                            normalized_bounds,
                            code="stat_failed",
                            path_hint=relative_path,
                            message="A path could not be inspected.",
                        )
                        continue

                    if metadata["entry_kind"] == "file" and files_recorded >= normalized_bounds.max_files:
                        coverage_complete = False
                        _add_diagnostic(
                            diagnostics,
                            normalized_bounds,
                            code="scan_limit_reached",
                            path_hint=relative_path,
                            message="Workspace file inventory file limit was reached.",
                        )
                        stop_scan = True
                        break
                    if (
                        metadata["entry_kind"] == "directory"
                        and directories_recorded >= normalized_bounds.max_directories
                    ):
                        coverage_complete = False
                        _add_diagnostic(
                            diagnostics,
                            normalized_bounds,
                            code="scan_limit_reached",
                            path_hint=relative_path,
                            message="Workspace file inventory directory limit was reached.",
                        )
                        stop_scan = True
                        break

                    items.append(metadata)
                    if metadata["entry_kind"] == "directory":
                        directories_recorded += 1
                        stack.append((Path(entry.path), relative_path, entry_depth))
                    elif metadata["entry_kind"] == "file":
                        files_recorded += 1
        except PermissionError:
            coverage_complete = False
            _add_diagnostic(
                diagnostics,
                normalized_bounds,
                code="permission_denied",
                path_hint=parent_relative_path,
                message="A directory could not be inspected.",
            )
            continue
        except OSError:
            coverage_complete = False
            _add_diagnostic(
                diagnostics,
                normalized_bounds,
                code="directory_scan_failed",
                path_hint=parent_relative_path,
                message="A directory could not be inspected.",
            )
            continue

    return _result(items, ignored_count, diagnostics, coverage_complete=coverage_complete)


def _entry_metadata(entry: os.DirEntry[str], relative_path: str) -> dict[str, Any] | None:
    try:
        is_symlink = entry.is_symlink()
        is_directory = False if is_symlink else entry.is_dir(follow_symlinks=False)
        stat_result = entry.stat(follow_symlinks=False)
    except OSError:
        return None

    entry_kind: InventoryEntryKind
    if is_symlink:
        entry_kind = "symlink"
    elif is_directory:
        entry_kind = "directory"
    else:
        entry_kind = "file"

    suffix = Path(entry.name).suffix.lower()
    mime_hint = _STATIC_MIME_HINTS.get(suffix)
    indexing_candidate = entry_kind == "file" and _is_indexing_candidate(suffix, mime_hint)
    return {
        "relative_path": relative_path,
        "entry_kind": entry_kind,
        "size_bytes": stat_result.st_size if entry_kind != "directory" else None,
        "mtime_ns": stat_result.st_mtime_ns,
        "mode_bits": S_IMODE(stat_result.st_mode),
        "extension": suffix or None,
        "mime_hint": mime_hint,
        "language_hint": None,
        "ignored": False,
        "ignore_reason": None,
        "indexing_candidate": indexing_candidate,
        "metadata": {},
    }


def _result(
    items: list[dict[str, Any]],
    ignored_count: int,
    diagnostics: list[WorkspaceFileInventoryDiagnostic],
    *,
    coverage_complete: bool,
) -> InventoryScanResult:
    sorted_items = tuple(sorted(items, key=lambda item: str(item["relative_path"])))
    bounded_diagnostics = tuple(bounded_inventory_diagnostics(diagnostics))
    counts = normalize_inventory_counts(
        {
            "files": _count_entries(sorted_items, "file"),
            "directories": _count_entries(sorted_items, "directory"),
            "symlinks": _count_entries(sorted_items, "symlink"),
            "ignored": ignored_count,
            "indexing_candidates": sum(1 for item in sorted_items if item.get("indexing_candidate")),
            "diagnostics": len(bounded_diagnostics),
            "total_entries": len(sorted_items),
        }
    )
    return InventoryScanResult(
        items=sorted_items,
        counts=counts,
        diagnostics=bounded_diagnostics,
        coverage_complete=coverage_complete and not bounded_diagnostics,
    )


def _normalize_bounds(bounds: InventoryScanBounds) -> InventoryScanBounds:
    return InventoryScanBounds(
        max_files=max(0, bounds.max_files),
        max_directories=max(0, bounds.max_directories),
        max_depth=max(0, bounds.max_depth),
        max_path_length=max(1, bounds.max_path_length),
        max_diagnostics=max(0, min(bounds.max_diagnostics, DEFAULT_MAX_DIAGNOSTICS)),
        max_seconds=max(0.0, bounds.max_seconds),
    )


def _add_diagnostic(
    diagnostics: list[WorkspaceFileInventoryDiagnostic],
    bounds: InventoryScanBounds,
    *,
    code: str,
    path_hint: str,
    message: str,
) -> None:
    if len(diagnostics) >= bounds.max_diagnostics:
        return
    diagnostics.append({"code": code, "path_hint": path_hint, "message": message})


def _scan_timed_out(started_at: float, bounds: InventoryScanBounds) -> bool:
    return bounds.max_seconds > 0.0 and time.monotonic() - started_at > bounds.max_seconds


def _join_relative_path(parent: str, name: str) -> str:
    return f"{parent}/{name}" if parent else name


def _count_entries(items: tuple[dict[str, Any], ...] | list[dict[str, Any]], entry_kind: str) -> int:
    return sum(1 for item in items if item.get("entry_kind") == entry_kind)


def _is_indexing_candidate(extension: str, mime_hint: str | None) -> bool:
    if extension in _TEXT_LIKE_EXTENSIONS:
        return True
    return bool(mime_hint and mime_hint.startswith("text/"))


__all__ = [
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_DIAGNOSTICS",
    "DEFAULT_MAX_DIRECTORIES",
    "DEFAULT_MAX_FILES",
    "DEFAULT_MAX_PATH_LENGTH",
    "DEFAULT_MAX_SCAN_SECONDS",
    "InventoryScanBounds",
    "InventoryScanResult",
    "scan_workspace_file_inventory",
]
