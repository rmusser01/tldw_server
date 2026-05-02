"""
Unit tests for the post-generation verification interface.

Verifies that unsupported claims trigger adaptive retry metrics and that
central metrics registry counters/histograms are updated.
"""

import asyncio
import pytest

from tldw_Server_API.app.core.Claims_Extraction import monitoring as claims_monitoring
from tldw_Server_API.app.core.RAG.rag_service import post_generation_verifier as verifier_module
from tldw_Server_API.app.core.RAG.rag_service.post_generation_verifier import PostGenerationVerifier
from tldw_Server_API.app.core.RAG.rag_service.types import Document, DataSource
from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry


class _CapturingLogger:
    def __init__(self):
        self.messages: list[str] = []
        self.kwargs: list[dict] = []

    def warning(self, message, *args, **kwargs):
        self.messages.append(str(message))
        self.kwargs.append(kwargs)

    def debug(self, message, *args, **kwargs):
        self.messages.append(str(message))
        self.kwargs.append(kwargs)


def _assert_no_sensitive_log_leak(messages: list[str], kwargs: list[dict] | None = None):
    combined = "\n".join(messages)
    if kwargs is not None:
        combined = "\n".join([combined, "\n".join(str(item) for item in kwargs)])
    assert "sk-live-secret" not in combined
    assert "/Users/alice/private/db.sqlite" not in combined
    assert "C:\\Users\\alice\\private\\db.sqlite" not in combined


def _base_docs() -> list[Document]:
    return [
        Document(id="1", content="A", metadata={"source": DataSource.MEDIA_DB}),
        Document(id="2", content="B", metadata={"source": DataSource.MEDIA_DB}),
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_post_verifier_metrics_increment_on_unsupported(monkeypatch):
    # Stub claims runner returning unsupported ratio above threshold
    async def _fake_runner(**kwargs):
        return {
            "claims": [{"id": "c1"}, {"id": "c2"}, {"id": "c3"}, {"id": "c4"}],
            "summary": {
                "supported": 1,
                "refuted": 2,
                "nei": 1,
                "precision": 0.25,
                "coverage": 0.75,
                "claim_faithfulness": 0.25,
            },
        }

    verifier = PostGenerationVerifier(
        claims_runner=_fake_runner,
        max_retries=1,
        unsupported_threshold=0.10,  # low threshold to trigger repair
        max_claims=10,
        time_budget_sec=0.25,
    )

    # Minimal docs
    docs = [
        Document(id="1", content="A", metadata={"source": DataSource.MEDIA_DB}),
        Document(id="2", content="B", metadata={"source": DataSource.MEDIA_DB}),
    ]

    out = await verifier.verify_and_maybe_fix(
        query="What is RAG?",
        answer="RAG is X.",
        base_documents=docs,
        user_id="u1",
        media_db_path=None,  # no retrieval in this test
        generation_model=None,
        top_k=5,
    )

    # Unsupported ratio should reflect fake summary (3/4)
    assert out.unsupported_ratio > 0.5
    assert out.total_claims == 4
    assert out.unsupported_count == 3

    # Verify central metrics registry recorded increments
    registry = get_metrics_registry()

    def _sum_values(name: str) -> float:
        vals = registry.values.get(name)
        return sum(v.value for v in (vals or []))

    # Unsupported claims total should be incremented by 3
    assert _sum_values("rag_unsupported_claims_total") >= 3
    # One adaptive retry attempted
    assert _sum_values("rag_adaptive_retries_total") >= 1
    # Postcheck duration histogram should have recorded at least one observation
    assert registry.values.get("rag_postcheck_duration_seconds") is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_claims_verification_fallback_log_redacts_exception_details(monkeypatch):
    logger = _CapturingLogger()
    monkeypatch.setattr(verifier_module, "logger", logger)

    async def _leaking_runner(**kwargs):
        raise RuntimeError("token=sk-live-secret path=/Users/alice/private/db.sqlite")

    verifier = PostGenerationVerifier(
        claims_runner=_leaking_runner,
        max_retries=1,
        unsupported_threshold=0.10,
    )

    out = await verifier.verify_and_maybe_fix(
        query="What is RAG?",
        answer="RAG is X.",
        base_documents=_base_docs(),
    )

    assert out.reason == "threshold_not_exceeded"
    assert out.unsupported_ratio == 0.0
    assert out.total_claims == 0
    _assert_no_sensitive_log_leak(logger.messages)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_metrics_recording_fallback_log_redacts_exception_details(monkeypatch):
    logger = _CapturingLogger()
    monkeypatch.setattr(verifier_module, "logger", logger)

    def _failing_record_postcheck_metrics(total_claims, unsupported_claims):
        raise RuntimeError("token=sk-live-secret path=/Users/alice/private/db.sqlite")

    monkeypatch.setattr(claims_monitoring, "record_postcheck_metrics", _failing_record_postcheck_metrics)

    async def _supported_runner(**kwargs):
        return {
            "claims": [{"id": "c1"}],
            "summary": {"supported": 1, "refuted": 0, "nei": 0},
        }

    verifier = PostGenerationVerifier(
        claims_runner=_supported_runner,
        max_retries=1,
        unsupported_threshold=0.10,
    )

    out = await verifier.verify_and_maybe_fix(
        query="What is RAG?",
        answer="RAG is X.",
        base_documents=_base_docs(),
    )

    assert out.reason == "threshold_not_exceeded"
    assert out.unsupported_ratio == 0.0
    assert out.total_claims == 1
    assert out.fixed is False
    assert "Post-generation verifier metrics recording failed" in logger.messages
    _assert_no_sensitive_log_leak(logger.messages, logger.kwargs)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_query_expansion_fallback_log_redacts_exception_details(monkeypatch):
    logger = _CapturingLogger()
    monkeypatch.setattr(verifier_module, "logger", logger)
    monkeypatch.setattr(verifier_module, "AnswerGenerator", None)
    monkeypatch.setattr(verifier_module, "generate_hypothetical_answer", None)
    monkeypatch.setattr(verifier_module, "hyde_embed_text", None)

    async def _failing_expansion(*args, **kwargs):
        raise RuntimeError("token=sk-live-secret path=/Users/alice/private/db.sqlite")

    monkeypatch.setattr(verifier_module, "multi_strategy_expansion", _failing_expansion)

    class _Retriever:
        async def retrieve(self, **kwargs):
            return _base_docs()

    class _MultiDatabaseRetriever:
        def __init__(self, *args, **kwargs):
            self.retrievers = {verifier_module.DataSource.MEDIA_DB: _Retriever()}

    monkeypatch.setattr(verifier_module, "MultiDatabaseRetriever", _MultiDatabaseRetriever)

    async def _unsupported_runner(**kwargs):
        return {
            "claims": [{"id": "c1"}, {"id": "c2"}],
            "summary": {"supported": 0, "refuted": 1, "nei": 1},
        }

    verifier = PostGenerationVerifier(
        claims_runner=_unsupported_runner,
        max_retries=1,
        unsupported_threshold=0.10,
        use_advanced_rewrites=True,
    )

    out = await verifier.verify_and_maybe_fix(
        query="What is RAG?",
        answer="RAG is X.",
        base_documents=_base_docs(),
        media_db_path="media.db",
    )

    assert out.reason == "regen_failed"
    assert out.unsupported_ratio == 1.0
    assert out.total_claims == 2
    assert out.fixed is False
    assert "Adaptive fix query expansion failed; continuing" in logger.messages
    _assert_no_sensitive_log_leak(logger.messages, logger.kwargs)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_adaptive_fix_success_metric_fallback_log_redacts_exception_details(monkeypatch):
    logger = _CapturingLogger()
    monkeypatch.setattr(verifier_module, "logger", logger)
    monkeypatch.setattr(verifier_module, "MultiDatabaseRetriever", None)

    def _increment_counter(metric_name, value=1.0, labels=None):
        if metric_name == "rag_adaptive_fix_success_total":
            raise RuntimeError("token=sk-live-secret path=/Users/alice/private/db.sqlite")
        return None

    monkeypatch.setattr(verifier_module, "increment_counter", _increment_counter)

    class _Generator:
        def __init__(self, *args, **kwargs):
            pass

        async def generate(self, **kwargs):
            return {"answer": "A repaired answer."}

    monkeypatch.setattr(verifier_module, "AnswerGenerator", _Generator)

    calls = 0

    async def _runner(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {
                "claims": [{"id": "c1"}, {"id": "c2"}],
                "summary": {"supported": 0, "refuted": 1, "nei": 1},
            }
        return {
            "claims": [{"id": "c1"}],
            "summary": {"supported": 1, "refuted": 0, "nei": 0},
        }

    verifier = PostGenerationVerifier(
        claims_runner=_runner,
        max_retries=1,
        unsupported_threshold=0.10,
    )

    out = await verifier.verify_and_maybe_fix(
        query="What is RAG?",
        answer="RAG is X.",
        base_documents=_base_docs(),
    )

    assert calls == 2
    assert out.fixed is True
    assert out.new_answer == "A repaired answer."
    assert out.reason == ""
    assert "Post-generation adaptive-fix success metric failed" in logger.messages
    _assert_no_sensitive_log_leak(logger.messages, logger.kwargs)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_adaptive_retrieval_fallback_log_redacts_exception_details(monkeypatch):
    logger = _CapturingLogger()
    monkeypatch.setattr(verifier_module, "logger", logger)
    monkeypatch.setattr(verifier_module, "AnswerGenerator", None)

    class _FailingRetriever:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("token=sk-live-secret path=/Users/alice/private/db.sqlite")

    monkeypatch.setattr(verifier_module, "MultiDatabaseRetriever", _FailingRetriever)

    async def _unsupported_runner(**kwargs):
        return {
            "claims": [{"id": "c1"}, {"id": "c2"}],
            "summary": {"supported": 0, "refuted": 1, "nei": 1},
        }

    verifier = PostGenerationVerifier(
        claims_runner=_unsupported_runner,
        max_retries=1,
        unsupported_threshold=0.10,
    )

    out = await verifier.verify_and_maybe_fix(
        query="What is RAG?",
        answer="RAG is X.",
        base_documents=_base_docs(),
        media_db_path="media.db",
    )

    assert out.reason == "regen_failed"
    assert out.unsupported_ratio == 1.0
    assert out.total_claims == 2
    assert out.fixed is False
    _assert_no_sensitive_log_leak(logger.messages)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_adaptive_regeneration_fallback_log_redacts_exception_details(monkeypatch):
    logger = _CapturingLogger()
    monkeypatch.setattr(verifier_module, "logger", logger)
    monkeypatch.setattr(verifier_module, "MultiDatabaseRetriever", None)

    class _FailingGenerator:
        def __init__(self, *args, **kwargs):
            pass

        async def generate(self, **kwargs):
            raise RuntimeError("token=sk-live-secret path=C:\\Users\\alice\\private\\db.sqlite")

    monkeypatch.setattr(verifier_module, "AnswerGenerator", _FailingGenerator)

    async def _unsupported_runner(**kwargs):
        return {
            "claims": [{"id": "c1"}, {"id": "c2"}],
            "summary": {"supported": 0, "refuted": 1, "nei": 1},
        }

    verifier = PostGenerationVerifier(
        claims_runner=_unsupported_runner,
        max_retries=1,
        unsupported_threshold=0.10,
    )

    out = await verifier.verify_and_maybe_fix(
        query="What is RAG?",
        answer="RAG is X.",
        base_documents=_base_docs(),
    )

    assert out.reason == "regen_failed"
    assert out.unsupported_ratio == 1.0
    assert out.total_claims == 2
    assert out.fixed is False
    _assert_no_sensitive_log_leak(logger.messages)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_adaptive_recheck_fallback_log_redacts_exception_details(monkeypatch):
    logger = _CapturingLogger()
    monkeypatch.setattr(verifier_module, "logger", logger)
    monkeypatch.setattr(verifier_module, "MultiDatabaseRetriever", None)

    class _Generator:
        def __init__(self, *args, **kwargs):
            pass

        async def generate(self, **kwargs):
            return {"answer": "A repaired answer."}

    monkeypatch.setattr(verifier_module, "AnswerGenerator", _Generator)

    calls = 0

    async def _runner(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {
                "claims": [{"id": "c1"}, {"id": "c2"}],
                "summary": {"supported": 0, "refuted": 1, "nei": 1},
            }
        raise RuntimeError("token=sk-live-secret path=/Users/alice/private/db.sqlite")

    verifier = PostGenerationVerifier(
        claims_runner=_runner,
        max_retries=1,
        unsupported_threshold=0.10,
    )

    out = await verifier.verify_and_maybe_fix(
        query="What is RAG?",
        answer="RAG is X.",
        base_documents=_base_docs(),
    )

    assert calls == 2
    assert out.fixed is True
    assert out.new_answer == "A repaired answer."
    assert out.unsupported_ratio == 1.0
    _assert_no_sensitive_log_leak(logger.messages)
