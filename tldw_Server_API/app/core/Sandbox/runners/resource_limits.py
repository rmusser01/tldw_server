from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..limits import ArtifactLimitResult, collect_limited_artifacts
from ..policy import SandboxPolicyConfig

_RESOURCE_LIMIT_EXCEPTIONS = (
    AttributeError,
    OSError,
    PermissionError,
    RuntimeError,
    TypeError,
    ValueError,
)


def sandbox_policy_config() -> SandboxPolicyConfig:
    try:
        return SandboxPolicyConfig.from_settings()
    except _RESOURCE_LIMIT_EXCEPTIONS:
        return SandboxPolicyConfig()


def artifact_limit_values() -> tuple[int, int]:
    cfg = sandbox_policy_config()
    return (
        _positive_int(getattr(cfg, "max_artifact_file_bytes", None), 64 * 1024 * 1024),
        _positive_int(getattr(cfg, "max_artifact_total_bytes", None), 256 * 1024 * 1024),
    )


def collect_runner_artifacts(
    workspace: str,
    capture_patterns: list[str] | None,
    *,
    exclude_names: Iterable[str] = (),
    exclude_hidden: bool = False,
) -> ArtifactLimitResult:
    max_file_bytes, max_total_bytes = artifact_limit_values()
    try:
        result = collect_limited_artifacts(
            workspace,
            capture_patterns,
            max_file_bytes=max_file_bytes,
            max_total_bytes=max_total_bytes,
        )
    except _RESOURCE_LIMIT_EXCEPTIONS:
        return ArtifactLimitResult(artifacts={}, counters={})

    excludes = {str(name).strip() for name in exclude_names if str(name).strip()}
    if not excludes and not exclude_hidden:
        return result

    artifacts = {
        path: data
        for path, data in result.artifacts.items()
        if _artifact_allowed(path, exclude_names=excludes, exclude_hidden=exclude_hidden)
    }
    if len(artifacts) == len(result.artifacts):
        return result
    counters = dict(result.counters)
    counters["artifact_files_collected"] = len(artifacts)
    counters["artifact_bytes_collected"] = sum(len(value) for value in artifacts.values())
    return ArtifactLimitResult(artifacts=artifacts, counters=counters)


def log_limit_counters(hub: Any, run_id: str, max_log_bytes: int) -> dict[str, int]:
    try:
        is_truncated = bool(hub.is_log_truncated(run_id))
    except _RESOURCE_LIMIT_EXCEPTIONS:
        is_truncated = False
    if not is_truncated:
        return {}
    return {
        "log_limit_bytes": int(max_log_bytes),
        "log_truncated": 1,
    }


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except _RESOURCE_LIMIT_EXCEPTIONS:
        return default
    return parsed if parsed > 0 else default


def _artifact_allowed(path: str, *, exclude_names: set[str], exclude_hidden: bool) -> bool:
    rel = Path(path)
    if exclude_hidden and any(part.startswith(".") for part in rel.parts):
        return False
    return rel.as_posix() not in exclude_names
