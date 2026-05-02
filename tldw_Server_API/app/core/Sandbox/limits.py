from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class OutputLimitResult:
    stdout: bytes
    stderr: bytes
    counters: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ArtifactLimitResult:
    artifacts: dict[str, bytes]
    counters: dict[str, int] = field(default_factory=dict)


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
) -> ArtifactLimitResult:
    """Collect matching artifacts without reading files that exceed byte caps."""
    counters = {
        "artifact_limit_file_bytes": int(max_file_bytes or 0),
        "artifact_limit_total_bytes": int(max_total_bytes or 0),
        "artifact_files_collected": 0,
        "artifact_files_skipped": 0,
        "artifact_bytes_collected": 0,
        "artifact_skip_file_limit": 0,
        "artifact_skip_total_limit": 0,
        "artifact_skip_symlink": 0,
        "artifact_skip_invalid": 0,
        "artifact_skip_read_error": 0,
    }
    patterns = [str(pattern) for pattern in (capture_patterns or []) if str(pattern or "").strip()]
    if not patterns:
        return ArtifactLimitResult(artifacts={}, counters=counters)

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
                try:
                    data = full_path.read_bytes()
                except OSError:
                    _increment_artifact_skip(counters, "artifact_skip_read_error")
                    continue
                artifacts[rel_posix] = data
                counters["artifact_files_collected"] += 1
                counters["artifact_bytes_collected"] += len(data)
    except OSError:
        counters["artifact_files_skipped"] += 1
        counters["artifact_skip_read_error"] += 1

    return ArtifactLimitResult(artifacts=artifacts, counters=counters)


def _increment_artifact_skip(counters: dict[str, int], reason_key: str) -> None:
    counters["artifact_files_skipped"] += 1
    counters[reason_key] += 1


def _path_within_root(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
