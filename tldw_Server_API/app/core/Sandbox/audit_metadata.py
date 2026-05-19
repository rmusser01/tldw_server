"""Path-minimized audit metadata helpers for sandbox lifecycle events.

The sandbox security policy matrix requires audit rows to explain runtime,
policy, lifecycle, and artifact-limit decisions without leaking host paths,
environment variables, or raw artifact paths. Keep that contract centralized
here so endpoint and background execution paths do not drift.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

from .limits import build_limit_audit_metadata
from .models import RunPhase, RunStatus
from .run_status_taxonomy import normalize_run_status_reason


def _enum_value(value: object) -> str | None:
    """Return a stripped string value for enums and scalar metadata fields."""

    if value is None:
        return None
    raw = getattr(value, "value", value)
    text = str(raw).strip()
    return text or None


def _run_completion_outcome(status: RunStatus) -> str:
    """Map a run status into the stable audit outcome vocabulary."""

    phase = status.phase
    phase_value = phase.value if isinstance(phase, RunPhase) else str(phase)
    if phase == RunPhase.completed and (status.exit_code or 0) == 0:
        return "success"
    if phase == RunPhase.timed_out:
        return "timeout"
    if phase == RunPhase.killed:
        return "killed"
    if phase == RunPhase.failed:
        return "failed"
    return phase_value


def _pathlike_base_image(base_image: str) -> bool:
    """Detect base image strings that look like local host filesystem paths."""

    text = base_image.strip()
    if not text:
        return False
    if text.startswith(("/", "~", "./", "../")):
        return True
    windows_path = PureWindowsPath(text)
    if windows_path.drive or "\\" in text:
        return True
    if windows_path.is_absolute() or PurePosixPath(text).is_absolute():
        return True
    return False


def _safe_base_image(base_image: str | None) -> tuple[str | None, str | None]:
    """Return a redacted-safe base image reference and its audit-visible kind."""

    text = str(base_image or "").strip()
    if not text:
        return None, None
    if _pathlike_base_image(text):
        return None, "host_path"
    return text, "image_ref"


def _capture_pattern_count(capture_patterns: Iterable[object] | None) -> int | None:
    """Count non-empty capture patterns without recording their raw values."""

    if capture_patterns is None:
        return None
    count = 0
    for pattern in capture_patterns:
        if str(pattern or "").strip():
            count += 1
    return count


def build_run_completion_audit_metadata(
    *,
    status: RunStatus,
    spec_version: str | None,
    requested_runtime: object = None,
    trust_level: object = None,
    network_policy: object = None,
    capture_patterns: Iterable[object] | None = None,
) -> dict[str, Any]:
    """Build the shared safe metadata contract for sandbox run completion.

    Raw host paths and artifact paths are intentionally omitted. If a base image
    looks like a host filesystem path, only its kind is recorded.
    """

    effective_runtime = _enum_value(status.runtime)
    requested_runtime_value = _enum_value(requested_runtime)
    base_image, base_image_kind = _safe_base_image(status.base_image)
    outcome = _run_completion_outcome(status)
    status_reason_code = normalize_run_status_reason(
        phase=status.phase,
        message=status.message,
        exit_code=status.exit_code,
        resource_usage=(
            status.resource_usage if isinstance(status.resource_usage, Mapping) else None
        ),
    )
    reason_code = None
    if outcome in {"timeout", "failed"}:
        reason_code = status.message or None

    metadata: dict[str, Any] = {
        "runtime": effective_runtime,
        "effective_runtime": effective_runtime,
        "requested_runtime": requested_runtime_value,
        "base_image": base_image,
        "base_image_kind": base_image_kind,
        "image_digest": status.image_digest,
        "policy_hash": status.policy_hash,
        "exit_code": status.exit_code,
        "outcome": outcome,
        "status_reason_code": status_reason_code,
        "spec_version": spec_version or status.spec_version,
        "reason_code": reason_code,
    }

    runtime_version = _enum_value(status.runtime_version)
    if runtime_version is not None:
        metadata["runtime_version"] = runtime_version
    trust_level_value = _enum_value(trust_level)
    if trust_level_value is not None:
        metadata["trust_level"] = trust_level_value
    network_policy_value = _enum_value(network_policy)
    if network_policy_value is not None:
        metadata["network_policy"] = network_policy_value
    capture_count = _capture_pattern_count(capture_patterns)
    if capture_count is not None:
        metadata["capture_pattern_count"] = capture_count

    metadata.update(build_limit_audit_metadata(status.resource_usage))
    return metadata
