from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.RAG.rag_service import evidence_accumulator
from tldw_Server_API.app.core.RAG.rag_service.evidence_accumulator import EvidenceAccumulator
from tldw_Server_API.app.core.RAG.rag_service.types import Document


pytestmark = pytest.mark.unit


_SECRET_PATH = "/private/evidence/tenant-token.db"
_SECRET_TOKEN = "evidence-secret-token"
_SENSITIVE_MARKERS = (
    "backend exploded",
    _SECRET_PATH,
    "tenant-token.db",
    _SECRET_TOKEN,
    "secret query",
    "secret evidence",
)


def _capture_records(level: str = "DEBUG") -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []

    def _sink(message: Any) -> None:
        records.append(
            {
                "message": str(message.record.get("message") or ""),
                "extra": dict(message.record.get("extra") or {}),
                "exception": message.record.get("exception"),
            }
        )

    sink_id = evidence_accumulator.logger.add(_sink, level=level)
    return records, sink_id


def _assert_sanitized(records: list[dict[str, Any]], expected_messages: list[str]) -> None:
    assert [record["message"] for record in records] == expected_messages
    rendered = "\n".join(str(record) for record in records)
    for marker in _SENSITIVE_MARKERS:
        assert marker not in rendered
    assert all("exc_info" not in record["extra"] for record in records)
    assert all(record["exception"] is None for record in records)


def _secret_exception() -> RuntimeError:
    return RuntimeError(f"backend exploded at {_SECRET_PATH}?token={_SECRET_TOKEN}")


@pytest.mark.asyncio
async def test_retrieval_round_failure_log_is_sanitized() -> None:
    accumulator = EvidenceAccumulator(
        max_rounds=2,
        min_docs_per_round=1,
        enable_gap_assessment=False,
    )
    initial_docs = [
        Document(
            id="initial",
            content="unrelated content",
            metadata={},
            score=0.1,
        )
    ]

    async def _failing_retrieval(_query: str, _exclude_ids: set[str]) -> list[Document]:
        raise _secret_exception()

    records, sink_id = _capture_records("WARNING")
    try:
        result = await accumulator.accumulate(
            query=f"secret query {_SECRET_TOKEN}",
            initial_results=initial_docs,
            retrieval_fn=_failing_retrieval,
        )
    finally:
        evidence_accumulator.logger.remove(sink_id)

    assert result.documents == initial_docs
    _assert_sanitized(records, ["Retrieval error during evidence accumulation round"])


@pytest.mark.asyncio
async def test_gap_assessment_fallback_log_is_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    accumulator = EvidenceAccumulator(enable_gap_assessment=True)

    async def _failing_assessment(_query: str, _evidence: str) -> str:
        raise _secret_exception()

    monkeypatch.setattr(accumulator, "_call_llm_for_assessment", _failing_assessment)
    records, sink_id = _capture_records("WARNING")
    try:
        is_sufficient, reason, gap_queries = await accumulator._assess_evidence(
            f"secret query {_SECRET_TOKEN}",
            [
                Document(
                    id="doc-1",
                    content=f"secret evidence {_SECRET_TOKEN}",
                    metadata={},
                    score=0.1,
                )
            ],
        )
    finally:
        evidence_accumulator.logger.remove(sink_id)

    assert is_sufficient is False
    assert "Coverage" in reason
    assert gap_queries
    _assert_sanitized(records, ["LLM gap assessment failed, using heuristic"])


@pytest.mark.asyncio
async def test_llm_assessment_call_failure_log_is_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FailingAnswerGenerator:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        async def generate(self, *_args: Any, **_kwargs: Any) -> str:
            raise _secret_exception()

    from tldw_Server_API.app.core.RAG.rag_service import generation

    monkeypatch.setattr(generation, "AnswerGenerator", _FailingAnswerGenerator)
    accumulator = EvidenceAccumulator(enable_gap_assessment=True)
    records, sink_id = _capture_records("WARNING")
    try:
        with pytest.raises(RuntimeError):
            await accumulator._call_llm_for_assessment(
                f"secret query {_SECRET_TOKEN}",
                f"secret evidence {_SECRET_TOKEN}",
            )
    finally:
        evidence_accumulator.logger.remove(sink_id)

    _assert_sanitized(records, ["LLM assessment call failed"])
