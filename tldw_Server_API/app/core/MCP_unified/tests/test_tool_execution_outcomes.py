"""Contracts for bounded expected tool failures and safe module logging."""

from __future__ import annotations

import importlib
import re
from typing import Any

import pytest
from loguru import logger

from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule, ModuleConfig


def _outcomes_module() -> Any:
    return importlib.import_module("tldw_Server_API.app.core.MCP_unified.execution_outcomes")


def test_expected_tool_failure_reason_catalog_is_fixed_and_bounded() -> None:
    outcomes = _outcomes_module()

    assert issubclass(outcomes.BreakerAction, str)
    assert issubclass(outcomes.BreakerAction, __import__("enum").Enum)
    assert [(item.name, item.value) for item in outcomes.BreakerAction] == [
        ("IGNORE", "ignore"),
        ("RECORD_FAILURE", "record_failure"),
    ]

    expected_values = {
        "RATE_LIMIT_UNAVAILABLE": (
            "rate_limit_unavailable",
            "Rate-limit admission is temporarily unavailable.",
            outcomes.BreakerAction.IGNORE,
        ),
        "IDEMPOTENCY_IN_PROGRESS": (
            "idempotency_in_progress",
            "A request with this idempotency key is still in progress.",
            outcomes.BreakerAction.IGNORE,
        ),
        "IDEMPOTENCY_UNAVAILABLE": (
            "idempotency_unavailable",
            "Idempotent execution is temporarily unavailable.",
            outcomes.BreakerAction.IGNORE,
        ),
        "STALE_PREPARED_CALL": (
            "stale_prepared_call",
            "The prepared tool call is no longer valid.",
            outcomes.BreakerAction.IGNORE,
        ),
        "DEPENDENCY_UNAVAILABLE": (
            "dependency_unavailable",
            "A required tool dependency is temporarily unavailable.",
            outcomes.BreakerAction.RECORD_FAILURE,
        ),
    }

    assert {
        name: member.value for name, member in outcomes.ExpectedToolFailureReason.__members__.items()
    } == expected_values

    values = [member.value for member in outcomes.ExpectedToolFailureReason.__members__.values()]
    assert len(values) == len(set(values))
    for reason_code, public_message, breaker_action in values:
        assert re.fullmatch(r"^[a-z][a-z0-9_]{0,63}$", reason_code)
        assert public_message.strip()
        assert len(public_message) <= 200
        assert isinstance(breaker_action, outcomes.BreakerAction)


def test_expected_tool_failure_accepts_only_a_catalog_reason() -> None:
    outcomes = _outcomes_module()
    reason = outcomes.ExpectedToolFailureReason.DEPENDENCY_UNAVAILABLE

    failure = outcomes.ExpectedToolFailure(reason)

    assert failure.reason is reason
    assert failure.reason_code == reason.reason_code
    assert failure.public_message == reason.public_message
    assert failure.breaker_action is reason.breaker_action
    with pytest.raises(TypeError):
        outcomes.ExpectedToolFailure("dependency_unavailable")
    with pytest.raises(TypeError):
        outcomes.ExpectedToolFailure(reason, public_message="caller text")
    with pytest.raises(AttributeError):
        failure.public_message = "caller text"
    with pytest.raises(AttributeError):
        failure.breaker_action = outcomes.BreakerAction.IGNORE


def test_expected_tool_failure_cannot_be_subclassed() -> None:
    outcomes = _outcomes_module()

    with pytest.raises(TypeError, match="cannot be subclassed"):

        class _InjectingExpectedFailure(outcomes.ExpectedToolFailure):
            @property
            def reason_code(self) -> str:
                return "SENTINEL_EXPECTED_SUBCLASS_SECRET"


def test_module_outcome_wrappers_do_not_render_the_original_exception() -> None:
    base = importlib.import_module("tldw_Server_API.app.core.MCP_unified.modules.base")
    original = RuntimeError("SENTINEL_WRAPPER_SECRET")

    ignored = base._IgnoredModuleOutcome(original)
    counted = base._CountedModuleOutcome(original)

    assert ignored.original is original
    assert counted.original is original
    rendered = " ".join((str(ignored), repr(ignored), str(counted), repr(counted)))
    assert "SENTINEL_WRAPPER_SECRET" not in rendered


class _SentinelModuleError(RuntimeError):
    pass


class _BrokenToolList:
    def __iter__(self):
        raise _SentinelModuleError("SENTINEL_MODULE_SECRET:tool-cache")


class _LoggingModule(BaseModule):
    def __init__(self, phase: str) -> None:
        self.phase = phase
        super().__init__(
            ModuleConfig(
                name=f"safe_logs_{phase}",
                circuit_breaker_threshold=5,
            )
        )

    async def on_initialize(self) -> None:
        if self.phase == "initialize":
            raise _SentinelModuleError("SENTINEL_MODULE_SECRET:initialize")

    async def on_shutdown(self) -> None:
        if self.phase == "shutdown":
            raise _SentinelModuleError("SENTINEL_MODULE_SECRET:shutdown")

    async def check_health(self) -> dict[str, bool]:
        if self.phase == "health":
            raise _SentinelModuleError("SENTINEL_MODULE_SECRET:health")
        return {"ok": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        if self.phase == "tool-cache":
            return _BrokenToolList()  # type: ignore[return-value]
        return []

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: Any = None,
    ) -> Any:
        del tool_name, arguments, context
        return None


@pytest.mark.asyncio
async def test_base_module_failure_logs_are_structured_and_message_safe() -> None:
    captured: list[Any] = []
    sink_id = logger.add(lambda message: captured.append(message.record), level="DEBUG")
    try:
        with pytest.raises(_SentinelModuleError):
            await _LoggingModule("initialize").initialize()

        await _LoggingModule("shutdown").shutdown()
        await _LoggingModule("health").health_check()
        await _LoggingModule("tool-cache").get_tool_def("tool.safe")

        execution_module = _LoggingModule("execution")

        async def fail_execution() -> None:
            raise _SentinelModuleError("SENTINEL_MODULE_SECRET:execution")

        with pytest.raises(_SentinelModuleError):
            await execution_module.execute_with_circuit_breaker(fail_execution)

        outcomes = _outcomes_module()

        async def fail_expected() -> None:
            raise outcomes.ExpectedToolFailure(outcomes.ExpectedToolFailureReason.DEPENDENCY_UNAVAILABLE)

        with pytest.raises(outcomes.ExpectedToolFailure):
            await execution_module.execute_with_circuit_breaker(fail_expected)
    finally:
        logger.remove(sink_id)

    rendered = "\n".join(f"{record['message']} {record['extra']!r} {record['exception']!r}" for record in captured)
    assert "SENTINEL_MODULE_SECRET" not in rendered
    assert "_SentinelModuleError" in rendered
    assert "ExpectedToolFailure" in rendered
    assert "dependency_unavailable" in rendered
    for reason_code in (
        "module_initialization_failed",
        "module_shutdown_failed",
        "module_health_check_failed",
        "module_tool_cache_lookup_failed",
        "module_operation_failed",
    ):
        assert reason_code in rendered
