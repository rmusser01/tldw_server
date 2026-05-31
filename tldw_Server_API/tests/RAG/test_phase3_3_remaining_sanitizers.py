"""Phase 3.3 conservative sanitizer coverage for remaining RAG tail logs."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from typing import Any

import pytest

from tldw_Server_API.app.core.RAG.rag_service import guardrails, research_agent
from tldw_Server_API.app.core.RAG.rag_service.query_classifier import QueryClassification


pytestmark = pytest.mark.unit


_SECRET_PATH = "/private/rag-tail/tenant-token.db"
_SECRET_TOKEN = "rag-tail-secret-token"
_SENSITIVE_MARKERS = (
    "backend exploded",
    _SECRET_PATH,
    "tenant-token.db",
    _SECRET_TOKEN,
)


def _secret_error() -> RuntimeError:
    return RuntimeError(f"backend exploded at {_SECRET_PATH}?token={_SECRET_TOKEN}")


def _raise_secret_error() -> None:
    raise _secret_error()


def _capture_records(logger: Any, level: str = "DEBUG") -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []

    def _sink(message: Any) -> None:
        records.append(
            {
                "message": str(message.record.get("message") or ""),
                "extra": dict(message.record.get("extra") or {}),
                "exception": message.record.get("exception"),
            }
        )

    sink_id = logger.add(_sink, level=level)
    return records, sink_id


def _assert_sanitized_records(records: list[dict[str, Any]], expected_messages: list[str]) -> None:
    assert [record["message"] for record in records] == expected_messages
    rendered = "\n".join(str(record) for record in records)
    for marker in _SENSITIVE_MARKERS:
        assert marker not in rendered
    assert all("exc_info" not in record["extra"] for record in records)
    assert all(record["exception"] is None for record in records)


def test_action_capability_failure_log_is_sanitized() -> None:
    registry = research_agent.ActionRegistry()
    registry.register(
        research_agent.ResearchAction(
            name="bad",
            description="bad action",
            schema={},
            enabled=lambda _classification: _raise_secret_error(),
            execute=lambda _params: [],
        )
    )
    registry.register(
        research_agent.ResearchAction(
            name="good",
            description="good action",
            schema={},
            enabled=lambda _classification: True,
            execute=lambda _params: [],
        )
    )

    records, sink_id = _capture_records(research_agent.logger)
    try:
        available = registry.get_available(QueryClassification())
    finally:
        research_agent.logger.remove(sink_id)

    assert [action.name for action in available] == ["good"]
    _assert_sanitized_records(
        records,
        ["Research action capability check failed; action skipped"],
    )


@pytest.mark.asyncio
async def test_reasoning_preamble_progress_failure_log_is_sanitized() -> None:
    action = research_agent._create_reasoning_preamble_action(
        on_progress=lambda _event: _raise_secret_error()
    )

    records, sink_id = _capture_records(research_agent.logger)
    try:
        result = action.execute({"reasoning": "safe reasoning", "plan": "safe plan"})
        if asyncio.iscoroutine(result):
            result = await result
    finally:
        research_agent.logger.remove(sink_id)

    assert result.success is True
    assert result.metadata == {"reasoning": "safe reasoning\nsafe plan"}
    _assert_sanitized_records(records, ["Research agent progress callback failed"])


@pytest.mark.asyncio
async def test_research_loop_progress_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_chat_call(**_kwargs: Any) -> str:
        return json.dumps(
            {
                "reasoning": "safe reasoning",
                "action": "done",
                "params": {"reason": "safe complete"},
            }
        )

    fake_chat_service = types.ModuleType("tldw_Server_API.app.core.Chat.chat_service")
    fake_chat_service.perform_chat_api_call_async = fake_chat_call
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.Chat.chat_service",
        fake_chat_service,
    )

    records, sink_id = _capture_records(research_agent.logger)
    try:
        output = await research_agent.research_loop(
            query="safe query",
            classification=QueryClassification(standalone_query="safe query"),
            mode="speed",
            max_iterations=1,
            on_progress=lambda _event: _raise_secret_error(),
            registry=research_agent.ActionRegistry(),
        )
    finally:
        research_agent.logger.remove(sink_id)

    assert output.completed is True
    assert output.final_reasoning == "safe complete"
    progress_records = [
        record
        for record in records
        if record["message"] == "Research iteration progress callback failed"
    ]
    _assert_sanitized_records(
        progress_records,
        [
            "Research iteration progress callback failed",
            "Research iteration progress callback failed",
        ],
    )


def test_guardrail_numeric_setup_failure_logs_are_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenNumericToken(str):
        def __getitem__(self, _key: Any) -> str:
            raise ValueError(f"backend exploded at {_SECRET_PATH}?token={_SECRET_TOKEN}")

    monkeypatch.setattr(
        guardrails,
        "_normalize_number_token",
        lambda _raw: BrokenNumericToken("1234"),
    )

    records, sink_id = _capture_records(guardrails.logger)
    try:
        tokens = guardrails._extract_numeric_tokens("1234")
    finally:
        guardrails.logger.remove(sink_id)

    assert tokens == {"1234"}
    _assert_sanitized_records(
        records,
        [
            "Guardrail numeric canonicalization failed",
            "Guardrail numeric expansion setup failed",
        ],
    )
