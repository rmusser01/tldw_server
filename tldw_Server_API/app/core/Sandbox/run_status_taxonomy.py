"""Normalize sandbox run status details into stable client-facing reason codes.

This module derives an additive `status_reason_code` from existing run status
fields. The vocabulary is intentionally small and stable for API clients, while
the raw phase, message, and exit code remain available for operator debugging.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

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

_RUNTIME_UNAVAILABLE_MESSAGES = frozenset({
    "runtime_unavailable",
    "runtime unavailable",
    "docker_unavailable",
    "firecracker_unavailable",
    "vz_linux_unavailable",
    "vz_macos_unavailable",
    "seatbelt_unavailable",
    "worktree_unavailable",
    "vz_linux_policy_failed",
    "vz_macos_policy_failed",
})

_RUNTIME_CONTEXT_TERMS = ("runtime", "provision")
_RUNTIME_UNAVAILABLE_TERMS = ("unavailable", "not found", "missing")


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


def _is_runtime_unavailable_message(message_value: str) -> bool:
    if message_value in _RUNTIME_UNAVAILABLE_MESSAGES:
        return True
    if "runtime_unavailable" in message_value:
        return True
    if message_value.startswith("runtime_provisioning"):
        return True
    has_runtime_context = any(term in message_value for term in _RUNTIME_CONTEXT_TERMS)
    has_unavailable_signal = any(term in message_value for term in _RUNTIME_UNAVAILABLE_TERMS)
    return has_runtime_context and has_unavailable_signal


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

    if phase_value in {"completed", "failed", "timed_out"} and _has_limit_signal(
        resource_usage
    ):
        return "limits_applied"

    if phase_value == "completed":
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
        if _is_runtime_unavailable_message(message_value):
            return "runtime_unavailable"
        if "policy_failed" in message_value or (
            "policy" in message_value and "failed" in message_value
        ):
            return "policy_failed"
        if "startup_timeout" in message_value:
            return "startup_timeout"
        if "execution_timeout" in message_value or message_value == "timeout":
            return "execution_timeout"
        if _is_nonzero_exit(exit_code):
            return "nonzero_exit"
        return "runtime_error"

    return "unknown"
