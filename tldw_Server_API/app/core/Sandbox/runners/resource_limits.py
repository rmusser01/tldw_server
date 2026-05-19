"""Runner-facing artifact and log limit helpers.

This module keeps runtime implementations on one shared reporting contract while
letting each runner keep its own execution lifecycle. Helpers intentionally
return aggregate counters only; they do not expose artifact paths in limit/audit
metadata.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..limits import ArtifactLimitResult, artifact_limit_counter_defaults, collect_limited_artifacts
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
    """Load sandbox policy settings with safe defaults for runner cleanup paths."""
    try:
        return SandboxPolicyConfig.from_settings()
    except _RESOURCE_LIMIT_EXCEPTIONS:
        return SandboxPolicyConfig()


def artifact_limit_values() -> tuple[int, int]:
    """Return positive per-file and total artifact byte caps for runner collection."""
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
    """Collect runner artifacts with shared caps and optional internal exclusions."""
    max_file_bytes, max_total_bytes = artifact_limit_values()
    try:
        result = collect_limited_artifacts(
            workspace,
            capture_patterns,
            max_file_bytes=max_file_bytes,
            max_total_bytes=max_total_bytes,
            exclude_names=exclude_names,
            exclude_hidden=exclude_hidden,
        )
    except _RESOURCE_LIMIT_EXCEPTIONS:
        counters = artifact_limit_counter_defaults(max_file_bytes, max_total_bytes)
        counters["artifact_files_skipped"] = 1
        counters["artifact_skip_read_error"] = 1
        return ArtifactLimitResult(artifacts={}, counters=counters)
    return result


def log_limit_counters(hub: Any, run_id: str, max_log_bytes: int) -> dict[str, int]:
    """Return resource_usage counters when the stream hub observed log truncation."""
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
