from __future__ import annotations

from typing import Any, Callable

from fastapi import HTTPException, status


def request_choice_count(cleaned_args: dict[str, Any] | None) -> int:
    if not isinstance(cleaned_args, dict):
        return 1
    raw_n = cleaned_args.get("n", 1)
    try:
        return max(1, int(raw_n))
    except (TypeError, ValueError):
        return 1


def ensure_tool_autoexec_supports_request(
    *,
    cleaned_args: dict[str, Any] | None,
    should_run_tool_autoexec: Callable[[dict[str, Any] | None], bool],
) -> None:
    if should_run_tool_autoexec(cleaned_args) and request_choice_count(cleaned_args) > 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "unsupported_multi_choice_tool_autoexec",
                "message": "Local tool auto-execution supports one assistant choice per request.",
            },
        )
