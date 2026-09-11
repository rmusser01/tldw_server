"""Bounded expected-failure outcomes for MCP tool execution."""

from __future__ import annotations

from enum import Enum


class BreakerAction(str, Enum):
    """Server-defined circuit-breaker accounting for an expected failure."""

    IGNORE = "ignore"
    RECORD_FAILURE = "record_failure"


class ExpectedToolFailureReason(Enum):
    """Fixed public reasons that a tool execution may return as an error result."""

    RATE_LIMIT_UNAVAILABLE = (
        "rate_limit_unavailable",
        "Rate-limit admission is temporarily unavailable.",
        BreakerAction.IGNORE,
    )
    IDEMPOTENCY_IN_PROGRESS = (
        "idempotency_in_progress",
        "A request with this idempotency key is still in progress.",
        BreakerAction.IGNORE,
    )
    IDEMPOTENCY_UNAVAILABLE = (
        "idempotency_unavailable",
        "Idempotent execution is temporarily unavailable.",
        BreakerAction.IGNORE,
    )
    STALE_PREPARED_CALL = (
        "stale_prepared_call",
        "The prepared tool call is no longer valid.",
        BreakerAction.IGNORE,
    )
    DEPENDENCY_UNAVAILABLE = (
        "dependency_unavailable",
        "A required tool dependency is temporarily unavailable.",
        BreakerAction.RECORD_FAILURE,
    )

    @property
    def reason_code(self) -> str:
        return self.value[0]

    @property
    def public_message(self) -> str:
        return self.value[1]

    @property
    def breaker_action(self) -> BreakerAction:
        return self.value[2]


class ExpectedToolFailure(Exception):
    """Expected tool failure whose public contract is selected by the server."""

    __slots__ = ("_reason",)

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("ExpectedToolFailure cannot be subclassed")

    def __init__(self, reason: ExpectedToolFailureReason) -> None:
        if not isinstance(reason, ExpectedToolFailureReason):
            raise TypeError("reason must be an ExpectedToolFailureReason")
        self._reason = reason
        super().__init__(reason.reason_code)

    @property
    def reason(self) -> ExpectedToolFailureReason:
        return self._reason

    @property
    def reason_code(self) -> str:
        return self._reason.reason_code

    @property
    def public_message(self) -> str:
        return self._reason.public_message

    @property
    def breaker_action(self) -> BreakerAction:
        return self._reason.breaker_action


def get_expected_tool_failure_reason(
    failure: BaseException,
) -> ExpectedToolFailureReason | None:
    """Return the exact stored catalog reason for a valid expected failure."""

    if type(failure) is not ExpectedToolFailure:
        return None
    try:
        reason = object.__getattribute__(failure, "_reason")
    except (AttributeError, TypeError):
        return None
    return reason if type(reason) is ExpectedToolFailureReason else None
