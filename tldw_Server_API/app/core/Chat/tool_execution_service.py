"""Request guardrails for local chat tool auto-execution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from fastapi import HTTPException, status

_ABSENT_CHOICE_STRINGS = {"", "none", "false", "null"}


class UnsupportedMultiChoiceToolAutoexecError(HTTPException):
    """Canonical local request error for unsupported tool auto-execution."""

    error_code = "unsupported_multi_choice_tool_autoexec"
    public_message = "Local tool auto-execution supports one assistant choice per request."

    def __init__(self) -> None:
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": self.error_code,
                "message": self.public_message,
            },
        )


def request_choice_count(cleaned_args: dict[str, Any] | None) -> int:
    """Return the requested assistant choice count, defaulting invalid input to one."""

    if not isinstance(cleaned_args, dict):
        return 1
    raw_n = cleaned_args.get("n", 1)
    try:
        return max(1, int(raw_n))
    except (TypeError, ValueError):
        return 1


def _non_empty_request_value(value: Any) -> bool:
    """Return whether a request option declares meaningful user intent."""

    if value is None or value is False:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in _ABSENT_CHOICE_STRINGS
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return bool(value)
    return bool(value)


def request_declares_local_tool_use(cleaned_args: dict[str, Any] | None) -> bool:
    """Return whether the request includes local tools or legacy functions."""

    if not isinstance(cleaned_args, dict):
        return False
    return _non_empty_request_value(cleaned_args.get("tools")) or _non_empty_request_value(
        cleaned_args.get("functions")
    )


def ensure_tool_autoexec_supports_request(
    *,
    cleaned_args: dict[str, Any] | None,
    should_run_tool_autoexec: Callable[[dict[str, Any] | None], bool],
) -> None:
    """Reject ambiguous multi-choice requests when local tool auto-execution is active."""

    if (
        should_run_tool_autoexec(cleaned_args)
        and request_declares_local_tool_use(cleaned_args)
        and request_choice_count(cleaned_args) > 1
    ):
        raise UnsupportedMultiChoiceToolAutoexecError()
