from __future__ import annotations

from typing import Any, Literal, Mapping

from .models import RunPhase

RunStatusReasonCode = Literal[
    "queued",
    "starting",
    "running",
    "completed",
    "limits_applied",
    "nonzero_exit",
    "policy_failed",
    "runtime_unavailable",
    "startup_timeout",
    "execution_timeout",
    "canceled_by_user",
    "killed",
    "queue_ttl_expired",
    "runtime_error",
    "unknown",
]

_LIMIT_COUNTER_KEYS = (
    "stdout_truncated",
    "stderr_truncated",
    "guest_output_limit_exceeded",
    "artifact_files_skipped",
    "artifact_skip_file_limit",
    "artifact_skip_total_limit",
    "artifact_skip_symlink",
    "artifact_skip_invalid",
    "artifact_skip_read_error",
)


def _phase_text(phase: RunPhase | str | None) -> str:
    if isinstance(phase, RunPhase):
        return phase.value
    return str(phase or "").strip().lower()


def _message_text(message: str | None) -> str:
    return str(message or "").strip().lower()


def _has_limit_signal(resource_usage: Mapping[str, Any] | None) -> bool:
    if not isinstance(resource_usage, Mapping):
        return False
    for key in _LIMIT_COUNTER_KEYS:
        try:
            if int(resource_usage.get(key) or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _is_nonzero_exit(exit_code: int | str | None) -> bool:
    if exit_code is None:
        return False
    try:
        return int(exit_code) != 0
    except (TypeError, ValueError):
        return False


def normalize_run_status_reason(
    *,
    phase: RunPhase | str | None,
    message: str | None,
    exit_code: int | str | None,
    resource_usage: Mapping[str, Any] | None,
) -> RunStatusReasonCode:
    """Derive a stable client-facing run reason code from existing status data."""

    phase_value = _phase_text(phase)
    message_value = _message_text(message)

    if phase_value in {"queued", "starting", "running"}:
        return phase_value  # type: ignore[return-value]

    if phase_value == "completed":
        if _has_limit_signal(resource_usage):
            return "limits_applied"
        return "completed"

    if phase_value == "timed_out":
        if "startup" in message_value:
            return "startup_timeout"
        return "execution_timeout"

    if phase_value == "killed":
        if "cancel" in message_value:
            return "canceled_by_user"
        return "killed"

    if phase_value == "failed":
        if message_value == "queue_ttl_expired":
            return "queue_ttl_expired"
        if "policy_failed" in message_value or (
            "policy" in message_value and "failed" in message_value
        ):
            return "policy_failed"
        if (
            "runtime_unavailable" in message_value
            or "unavailable" in message_value
            or "not found" in message_value
            or "missing" in message_value
        ):
            return "runtime_unavailable"
        if "startup_timeout" in message_value:
            return "startup_timeout"
        if "execution_timeout" in message_value or message_value == "timeout":
            return "execution_timeout"
        if _is_nonzero_exit(exit_code):
            return "nonzero_exit"
        return "runtime_error"

    return "unknown"
