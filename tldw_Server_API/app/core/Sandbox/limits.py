"""Shared output and artifact limit helpers for sandbox runtimes.

This module keeps byte-cap behavior consistent across VM/helper integrations and
derives path-minimized audit metadata from the counters those helpers return.
"""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class OutputLimitResult:
    """Bounded stdout/stderr bytes plus integer counters describing truncation."""

    stdout: bytes
    stderr: bytes
    counters: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ArtifactLimitResult:
    """Captured artifact bytes plus integer counters describing skipped files."""

    artifacts: dict[str, bytes]
    counters: dict[str, int] = field(default_factory=dict)


_OUTPUT_LIMIT_COUNTER_KEYS = (
    "output_limit_bytes",
    "stdout_bytes_original",
    "stderr_bytes_original",
    "stdout_bytes_returned",
    "stderr_bytes_returned",
    "stdout_truncated",
    "stderr_truncated",
)
_LOG_LIMIT_COUNTER_KEYS = (
    "log_limit_bytes",
    "log_truncated",
)
_ARTIFACT_LIMIT_COUNTER_KEYS = (
    "artifact_limit_file_bytes",
    "artifact_limit_total_bytes",
    "artifact_files_collected",
    "artifact_files_skipped",
    "artifact_files_excluded",
    "artifact_bytes_collected",
    "artifact_skip_file_limit",
    "artifact_skip_total_limit",
    "artifact_skip_symlink",
    "artifact_skip_invalid",
    "artifact_skip_read_error",
)
_ARTIFACT_SKIP_REASON_KEYS = (
    ("artifact_skip_file_limit", "file_limit"),
    ("artifact_skip_total_limit", "total_limit"),
    ("artifact_skip_symlink", "symlink"),
    ("artifact_skip_invalid", "invalid"),
    ("artifact_skip_read_error", "read_error"),
)


def cap_output_streams(
    stdout: bytes,
    stderr: bytes,
    *,
    max_output_bytes: int | None,
) -> OutputLimitResult:
    """Apply a combined stdout/stderr cap while preserving stderr diagnostics."""
    stdout_bytes = bytes(stdout or b"")
    stderr_bytes = bytes(stderr or b"")
    original_stdout = len(stdout_bytes)
    original_stderr = len(stderr_bytes)
    if max_output_bytes is None or int(max_output_bytes) <= 0:
        return OutputLimitResult(
            stdout=stdout_bytes,
            stderr=stderr_bytes,
            counters={
                "stdout_bytes_original": original_stdout,
                "stderr_bytes_original": original_stderr,
                "stdout_bytes_returned": original_stdout,
                "stderr_bytes_returned": original_stderr,
                "stdout_truncated": 0,
                "stderr_truncated": 0,
            },
        )

    cap = int(max_output_bytes)
    if original_stdout + original_stderr <= cap:
        return OutputLimitResult(
            stdout=stdout_bytes,
            stderr=stderr_bytes,
            counters={
                "output_limit_bytes": cap,
                "stdout_bytes_original": original_stdout,
                "stderr_bytes_original": original_stderr,
                "stdout_bytes_returned": original_stdout,
                "stderr_bytes_returned": original_stderr,
                "stdout_truncated": 0,
                "stderr_truncated": 0,
            },
        )

    if stdout_bytes and stderr_bytes and cap >= 2:
        stdout_budget = min(original_stdout, max(1, cap // 2))
        stderr_budget = min(original_stderr, max(1, cap - stdout_budget))
        unused = cap - stdout_budget - stderr_budget
        if unused > 0 and original_stdout > stdout_budget:
            extra = min(unused, original_stdout - stdout_budget)
            stdout_budget += extra
            unused -= extra
        if unused > 0 and original_stderr > stderr_budget:
            extra = min(unused, original_stderr - stderr_budget)
            stderr_budget += extra
    elif stdout_bytes:
        stdout_budget = min(original_stdout, cap)
        stderr_budget = 0
    else:
        stdout_budget = 0
        stderr_budget = min(original_stderr, cap)

    returned_stdout = stdout_bytes[:stdout_budget]
    returned_stderr = stderr_bytes[:stderr_budget]
    return OutputLimitResult(
        stdout=returned_stdout,
        stderr=returned_stderr,
        counters={
            "output_limit_bytes": cap,
            "stdout_bytes_original": original_stdout,
            "stderr_bytes_original": original_stderr,
            "stdout_bytes_returned": len(returned_stdout),
            "stderr_bytes_returned": len(returned_stderr),
            "stdout_truncated": int(len(returned_stdout) < original_stdout),
            "stderr_truncated": int(len(returned_stderr) < original_stderr),
        },
    )


def collect_limited_artifacts(
    workspace: str | os.PathLike[str],
    capture_patterns: list[str] | None,
    *,
    max_file_bytes: int | None,
    max_total_bytes: int | None,
    exclude_names: Iterable[str] = (),
    exclude_hidden: bool = False,
) -> ArtifactLimitResult:
    """Collect matching artifacts without reading files that exceed byte caps."""
    counters = artifact_limit_counter_defaults(max_file_bytes, max_total_bytes)
    patterns = [str(pattern) for pattern in (capture_patterns or []) if str(pattern or "").strip()]
    if not patterns:
        return ArtifactLimitResult(artifacts={}, counters=counters)
    excludes = {str(name).strip() for name in exclude_names if str(name).strip()}

    workspace_path = Path(workspace)
    if workspace_path.is_symlink():
        counters["artifact_files_skipped"] += 1
        counters["artifact_skip_symlink"] += 1
        return ArtifactLimitResult(artifacts={}, counters=counters)
    workspace_root = workspace_path.resolve(strict=False)

    max_file = int(max_file_bytes or 0)
    max_total = int(max_total_bytes or 0)
    artifacts: dict[str, bytes] = {}

    try:
        for root, dirs, files in os.walk(workspace_root):
            dirs[:] = sorted(dirs)
            root_path = Path(root)
            if not _path_within_root(workspace_root, root_path.resolve(strict=False)):
                continue

            for file_name in sorted(files):
                full_path = root_path / file_name
                rel_posix = full_path.relative_to(workspace_root).as_posix()
                if not any(fnmatch.fnmatchcase(rel_posix, pattern) for pattern in patterns):
                    continue
                if not _artifact_allowed(rel_posix, exclude_names=excludes, exclude_hidden=exclude_hidden):
                    counters["artifact_files_excluded"] += 1
                    continue
                if full_path.is_symlink():
                    _increment_artifact_skip(counters, "artifact_skip_symlink")
                    continue
                resolved_path = full_path.resolve(strict=False)
                if not _path_within_root(workspace_root, resolved_path):
                    _increment_artifact_skip(counters, "artifact_skip_invalid")
                    continue
                try:
                    size = int(full_path.stat().st_size)
                except OSError:
                    _increment_artifact_skip(counters, "artifact_skip_read_error")
                    continue
                if max_file > 0 and size > max_file:
                    _increment_artifact_skip(counters, "artifact_skip_file_limit")
                    continue
                if max_total > 0 and counters["artifact_bytes_collected"] + size > max_total:
                    _increment_artifact_skip(counters, "artifact_skip_total_limit")
                    continue
                read_limit, read_limit_reason = _artifact_read_limit(
                    max_file=max_file,
                    max_total=max_total,
                    bytes_collected=counters["artifact_bytes_collected"],
                )
                if read_limit is not None and read_limit < 0:
                    _increment_artifact_skip(counters, "artifact_skip_total_limit")
                    continue
                try:
                    data = _read_artifact_bytes(full_path, read_limit)
                except OSError:
                    _increment_artifact_skip(counters, "artifact_skip_read_error")
                    continue
                if read_limit is not None and len(data) > read_limit:
                    _increment_artifact_skip(
                        counters,
                        read_limit_reason or "artifact_skip_file_limit",
                    )
                    continue
                artifacts[rel_posix] = data
                counters["artifact_files_collected"] += 1
                counters["artifact_bytes_collected"] += len(data)
    except OSError:
        counters["artifact_files_skipped"] += 1
        counters["artifact_skip_read_error"] += 1

    return ArtifactLimitResult(artifacts=artifacts, counters=counters)


def build_limit_audit_metadata(resource_usage: Mapping[str, object] | None) -> dict[str, object]:
    """Derive aggregate audit metadata from integer run counters without paths."""
    if not isinstance(resource_usage, Mapping):
        return {}

    metadata: dict[str, object] = {}
    for key in _OUTPUT_LIMIT_COUNTER_KEYS + _LOG_LIMIT_COUNTER_KEYS + _ARTIFACT_LIMIT_COUNTER_KEYS:
        value = _counter_value(resource_usage.get(key))
        if value is not None:
            metadata[key] = value

    stdout_truncated = int(metadata.get("stdout_truncated", 0) or 0)
    stderr_truncated = int(metadata.get("stderr_truncated", 0) or 0)
    output_truncated = stdout_truncated > 0 or stderr_truncated > 0
    if output_truncated or any(key in metadata for key in _OUTPUT_LIMIT_COUNTER_KEYS):
        metadata["output_truncated"] = output_truncated

    skip_reasons = [
        reason
        for key, reason in _ARTIFACT_SKIP_REASON_KEYS
        if int(metadata.get(key, 0) or 0) > 0
    ]
    artifact_files_skipped = int(metadata.get("artifact_files_skipped", 0) or 0)
    artifacts_limited = artifact_files_skipped > 0 or bool(skip_reasons)
    if artifacts_limited or any(key in metadata for key in _ARTIFACT_LIMIT_COUNTER_KEYS):
        metadata["artifacts_limited"] = artifacts_limited
        metadata["artifact_skip_reasons"] = skip_reasons

    return metadata


def limit_event_actions(resource_usage: Mapping[str, object] | None) -> list[str]:
    """Return aggregate audit action names for limit outcomes that affected a run."""
    metadata = build_limit_audit_metadata(resource_usage)
    actions: list[str] = []
    if bool(metadata.get("output_truncated")):
        actions.append("output_truncated")
    if int(metadata.get("log_truncated", 0) or 0) > 0:
        actions.append("log_truncated")
    if bool(metadata.get("artifacts_limited")):
        actions.append("artifacts_limited")
    return actions


def artifact_limit_counter_defaults(
    max_file_bytes: int | None,
    max_total_bytes: int | None,
) -> dict[str, int]:
    """Return the shared artifact counter schema with limit values and zero counts."""
    return {
        "artifact_limit_file_bytes": int(max_file_bytes or 0),
        "artifact_limit_total_bytes": int(max_total_bytes or 0),
        "artifact_files_collected": 0,
        "artifact_files_skipped": 0,
        "artifact_files_excluded": 0,
        "artifact_bytes_collected": 0,
        "artifact_skip_file_limit": 0,
        "artifact_skip_total_limit": 0,
        "artifact_skip_symlink": 0,
        "artifact_skip_invalid": 0,
        "artifact_skip_read_error": 0,
    }


def _increment_artifact_skip(counters: dict[str, int], reason_key: str) -> None:
    counters["artifact_files_skipped"] += 1
    counters[reason_key] += 1


def _artifact_read_limit(*, max_file: int, max_total: int, bytes_collected: int) -> tuple[int | None, str | None]:
    limit: int | None = None
    reason: str | None = None
    if max_file > 0:
        limit = max_file
        reason = "artifact_skip_file_limit"
    if max_total > 0:
        remaining = max_total - bytes_collected
        if remaining < 0:
            return -1, "artifact_skip_total_limit"
        if limit is None or remaining < limit:
            limit = remaining
            reason = "artifact_skip_total_limit"
    return limit, reason


def _read_artifact_bytes(path: Path, read_limit: int | None) -> bytes:
    if read_limit is None:
        return path.read_bytes()
    with path.open("rb") as handle:
        return handle.read(read_limit + 1)


def _counter_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on"}:
            return 1
        if normalized in {"false", "no", "off", ""}:
            return 0
        try:
            parsed = int(normalized)
        except ValueError:
            return None
        return parsed if parsed >= 0 else None
    return None


def _path_within_root(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _artifact_allowed(path: str, *, exclude_names: set[str], exclude_hidden: bool) -> bool:
    rel = Path(path)
    if exclude_hidden and any(part.startswith(".") for part in rel.parts):
        return False
    return rel.as_posix() not in exclude_names
