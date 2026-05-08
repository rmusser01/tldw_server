"""Normalize sandbox run status details into stable client-facing reason codes.

This module derives an additive `status_reason_code` from existing run status
fields. The vocabulary is intentionally small and stable for API clients, while
the raw phase, message, and exit code remain available for operator debugging.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, get_args

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
RunStatusReasonCategory = Literal[
    "lifecycle",
    "success",
    "limits",
    "policy",
    "runtime",
    "timeout",
    "cancellation",
    "execution",
    "unknown",
]
RunStatusReasonSeverity = Literal["info", "warning", "error"]
RunStatusOperatorAction = Literal[
    "none",
    "inspect_logs",
    "review_limits",
    "review_policy",
    "check_runtime_readiness",
    "retry_later",
    "review_exit_code",
    "unknown",
]


@dataclass(frozen=True)
class RunStatusReasonDetails:
    """Structured client/operator metadata for a normalized run status reason."""

    code: RunStatusReasonCode
    category: RunStatusReasonCategory
    severity: RunStatusReasonSeverity
    terminal: bool
    retryable: bool
    operator_action: RunStatusOperatorAction
    user_message_key: str


RUN_STATUS_REASON_METADATA: Mapping[str, RunStatusReasonDetails] = {
    "queued": RunStatusReasonDetails(
        code="queued",
        category="lifecycle",
        severity="info",
        terminal=False,
        retryable=False,
        operator_action="none",
        user_message_key="sandbox.status.queued",
    ),
    "starting": RunStatusReasonDetails(
        code="starting",
        category="lifecycle",
        severity="info",
        terminal=False,
        retryable=False,
        operator_action="none",
        user_message_key="sandbox.status.starting",
    ),
    "running": RunStatusReasonDetails(
        code="running",
        category="lifecycle",
        severity="info",
        terminal=False,
        retryable=False,
        operator_action="none",
        user_message_key="sandbox.status.running",
    ),
    "completed": RunStatusReasonDetails(
        code="completed",
        category="success",
        severity="info",
        terminal=True,
        retryable=False,
        operator_action="none",
        user_message_key="sandbox.status.completed",
    ),
    "limits_applied": RunStatusReasonDetails(
        code="limits_applied",
        category="limits",
        severity="warning",
        terminal=True,
        retryable=False,
        operator_action="review_limits",
        user_message_key="sandbox.status.limits_applied",
    ),
    "nonzero_exit": RunStatusReasonDetails(
        code="nonzero_exit",
        category="execution",
        severity="error",
        terminal=True,
        retryable=True,
        operator_action="review_exit_code",
        user_message_key="sandbox.status.nonzero_exit",
    ),
    "policy_failed": RunStatusReasonDetails(
        code="policy_failed",
        category="policy",
        severity="error",
        terminal=True,
        retryable=False,
        operator_action="review_policy",
        user_message_key="sandbox.status.policy_failed",
    ),
    "runtime_unavailable": RunStatusReasonDetails(
        code="runtime_unavailable",
        category="runtime",
        severity="error",
        terminal=True,
        retryable=True,
        operator_action="check_runtime_readiness",
        user_message_key="sandbox.status.runtime_unavailable",
    ),
    "startup_timeout": RunStatusReasonDetails(
        code="startup_timeout",
        category="timeout",
        severity="error",
        terminal=True,
        retryable=True,
        operator_action="check_runtime_readiness",
        user_message_key="sandbox.status.startup_timeout",
    ),
    "execution_timeout": RunStatusReasonDetails(
        code="execution_timeout",
        category="timeout",
        severity="error",
        terminal=True,
        retryable=True,
        operator_action="retry_later",
        user_message_key="sandbox.status.execution_timeout",
    ),
    "canceled_by_user": RunStatusReasonDetails(
        code="canceled_by_user",
        category="cancellation",
        severity="warning",
        terminal=True,
        retryable=False,
        operator_action="none",
        user_message_key="sandbox.status.canceled_by_user",
    ),
    "killed": RunStatusReasonDetails(
        code="killed",
        category="cancellation",
        severity="error",
        terminal=True,
        retryable=True,
        operator_action="inspect_logs",
        user_message_key="sandbox.status.killed",
    ),
    "queue_ttl_expired": RunStatusReasonDetails(
        code="queue_ttl_expired",
        category="lifecycle",
        severity="warning",
        terminal=True,
        retryable=True,
        operator_action="retry_later",
        user_message_key="sandbox.status.queue_ttl_expired",
    ),
    "runtime_error": RunStatusReasonDetails(
        code="runtime_error",
        category="execution",
        severity="error",
        terminal=True,
        retryable=True,
        operator_action="inspect_logs",
        user_message_key="sandbox.status.runtime_error",
    ),
    "unknown": RunStatusReasonDetails(
        code="unknown",
        category="unknown",
        severity="warning",
        terminal=False,
        retryable=False,
        operator_action="unknown",
        user_message_key="sandbox.status.unknown",
    ),
}

_LIMIT_COUNTER_KEYS = (
    "stdout_truncated",
    "stderr_truncated",
    "log_truncated",
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
    "limactl_missing",
    "vz_linux_unavailable",
    "vz_macos_unavailable",
    "seatbelt_unavailable",
    "worktree_unavailable",
})

_POLICY_FAILED_MESSAGES = frozenset({
    "lima_policy_failed",
    "vz_linux_policy_failed",
    "vz_macos_policy_failed",
    "seatbelt_policy_failed",
    "worktree_policy_failed",
})

_RUNTIME_CONTEXT_TERMS = ("runtime", "provision")
_RUNTIME_UNAVAILABLE_TERMS = ("unavailable", "not found", "missing")


def _validate_run_status_reason_metadata() -> None:
    expected = set(get_args(RunStatusReasonCode))
    actual = set(RUN_STATUS_REASON_METADATA)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise RuntimeError(
            "Run status reason metadata map is incomplete: "
            f"missing={missing}, extra={extra}"
        )
    mismatched = sorted(
        (key, metadata.code)
        for key, metadata in RUN_STATUS_REASON_METADATA.items()
        if metadata.code != key
    )
    if mismatched:
        raise RuntimeError(
            "Run status reason metadata code mismatch: "
            f"mismatched={mismatched}"
        )


_validate_run_status_reason_metadata()


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


def _is_policy_failed_message(message_value: str) -> bool:
    if message_value in _POLICY_FAILED_MESSAGES:
        return True
    return "policy_failed" in message_value or (
        "policy" in message_value and "failed" in message_value
    )


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
        if _is_policy_failed_message(message_value):
            return "policy_failed"
        if _is_runtime_unavailable_message(message_value):
            return "runtime_unavailable"
        if "startup_timeout" in message_value:
            return "startup_timeout"
        if "execution_timeout" in message_value or message_value == "timeout":
            return "execution_timeout"
        if _is_nonzero_exit(exit_code):
            return "nonzero_exit"
        return "runtime_error"

    return "unknown"


def run_status_reason_details(
    code: RunStatusReasonCode | str | None,
) -> RunStatusReasonDetails:
    """Return structured metadata for a run status reason code."""

    code_value = str(code or "unknown").strip()
    metadata = RUN_STATUS_REASON_METADATA.get(code_value)
    if metadata is None:
        return RUN_STATUS_REASON_METADATA["unknown"]
    return metadata


def normalize_run_status_reason_details(
    *,
    phase: RunPhase | str | None,
    message: str | None,
    exit_code: int | str | None,
    resource_usage: Mapping[str, Any] | None,
) -> RunStatusReasonDetails:
    """Derive structured reason metadata from existing status data."""

    code = normalize_run_status_reason(
        phase=phase,
        message=message,
        exit_code=exit_code,
        resource_usage=resource_usage,
    )
    return run_status_reason_details(code)
