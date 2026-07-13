"""
Unit tests for the post-generation verification interface.

Verifies that unsupported claims trigger adaptive retry metrics and that
central metrics registry counters/histograms are updated.
"""

from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionError
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatConfigurationError,
)
from tldw_Server_API.app.core.Claims_Extraction import monitoring as claims_monitoring
from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry
from tldw_Server_API.app.core.RAG.rag_service import post_generation_verifier as verifier_module
from tldw_Server_API.app.core.RAG.rag_service.post_generation_verifier import PostGenerationVerifier
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document


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


def test_embedding_degradation_preserves_missing_credentials_code() -> None:
    outcome = verifier_module.VerificationOutcome(
        unsupported_ratio=1.0,
        total_claims=1,
        unsupported_count=1,
        fixed=False,
    )

    verifier_module._record_embedding_degradation(
        outcome,
        ChatConfigurationError(
            "secret configuration details",
            provider="openai",
            error_code="missing_provider_credentials",
        ),
    )

    assert outcome.embedding_coverage == "degraded"
    assert outcome.embedding_failure_code == "missing_provider_credentials"


def test_embedding_degradation_preserves_invalid_configuration_code() -> None:
    outcome = verifier_module.VerificationOutcome(
        unsupported_ratio=1.0,
        total_claims=1,
        unsupported_count=1,
        fixed=False,
    )

    verifier_module._record_embedding_degradation(
        outcome,
        ChatConfigurationError(
            "secret endpoint configuration details",
            provider="local_api",
            error_code="provider_configuration_invalid",
        ),
    )

    assert outcome.embedding_coverage == "degraded"
    assert outcome.embedding_failure_code == "provider_configuration_invalid"


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


@pytest.mark.unit
@pytest.mark.asyncio
async def test_adaptive_retrieval_threads_runtime_to_retriever_and_hyde_embedding(monkeypatch):
    runtime = object()
    captured: dict[str, Any] = {}

    class _MediaRetriever:
        async def retrieve_hybrid(self, **kwargs):
            return _base_docs()

    class _MultiDatabaseRetriever:
        def __init__(self, *args, **kwargs):
            captured["retrieval_runtime"] = kwargs.get("credential_runtime")
            self.retrievers = {verifier_module.DataSource.MEDIA_DB: _MediaRetriever()}

    async def fake_hyde_embedding(text: str, **kwargs):
        captured["hyde_runtime"] = kwargs.get("credential_runtime")
        captured["hyde_metadata"] = kwargs.get("stage_metadata")
        kwargs["stage_metadata"].update(
            embedding_coverage="degraded",
            failure_code="credential_store_unavailable",
        )
        return [0.1, 0.2]

    async def _unsupported_runner(**kwargs):
        return {
            "claims": [{"id": "c1"}, {"id": "c2"}],
            "summary": {"supported": 0, "refuted": 1, "nei": 1},
        }

    monkeypatch.setattr(verifier_module, "MultiDatabaseRetriever", _MultiDatabaseRetriever)
    monkeypatch.setattr(verifier_module, "generate_hypothetical_answer", lambda *args: "hypothesis")
    monkeypatch.setattr(verifier_module, "hyde_embed_text", fake_hyde_embedding)
    monkeypatch.setattr(verifier_module, "multi_strategy_expansion", None)
    monkeypatch.setattr(verifier_module, "AnswerGenerator", None)

    verifier = PostGenerationVerifier(
        claims_runner=_unsupported_runner,
        max_retries=1,
        unsupported_threshold=0.10,
        use_advanced_rewrites=True,
        credential_runtime=runtime,
    )
    outcome = await verifier.verify_and_maybe_fix(
        query="What is RAG?",
        answer="RAG is X.",
        base_documents=_base_docs(),
        media_db_path="media.db",
    )

    assert captured["retrieval_runtime"] is runtime
    assert captured["hyde_runtime"] is runtime
    assert isinstance(captured["hyde_metadata"], dict)
    assert outcome.embedding_coverage == "degraded"
    assert outcome.embedding_failure_code == "credential_store_unavailable"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_claim_retrieval_reports_bounded_embedding_credential_degradation(monkeypatch):
    class _MediaRetriever:
        async def retrieve_hybrid(self, **kwargs):
            raise ByokResolutionError("credential_scope_revoked", "openai")

    class _MultiDatabaseRetriever:
        def __init__(self, *args, **kwargs):
            self.retrievers = {verifier_module.DataSource.MEDIA_DB: _MediaRetriever()}

    class _ClaimsEngine:
        async def run(self, **kwargs):
            await kwargs["retrieve_fn"]("claim")
            return {
                "claims": [{"id": "c1"}],
                "summary": {"supported": 1, "refuted": 0, "nei": 0},
            }

    verifier = PostGenerationVerifier(credential_runtime=object())

    async def build_claims_engine():
        return _ClaimsEngine(), None, {"used": False}

    monkeypatch.setattr(verifier_module, "ClaimsEngine", object)
    monkeypatch.setattr(verifier_module, "MultiDatabaseRetriever", _MultiDatabaseRetriever)
    monkeypatch.setattr(verifier, "_build_claims_engine", build_claims_engine)

    outcome = await verifier.verify_and_maybe_fix(
        query="What is RAG?",
        answer="RAG is X.",
        base_documents=_base_docs(),
        media_db_path="media.db",
    )

    assert outcome.embedding_coverage == "degraded"
    assert outcome.embedding_failure_code == "credential_scope_revoked"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_claim_retrieval_latches_first_provider_failure(monkeypatch):
    dispatches = 0
    callback_results: list[list[Document]] = []

    class _MediaRetriever:
        async def retrieve_hybrid(self, **_kwargs):
            nonlocal dispatches
            dispatches += 1
            raise ChatConfigurationError(
                "secret endpoint configuration detail",
                provider="local_api",
                error_code="provider_configuration_invalid",
            )

    class _MultiDatabaseRetriever:
        def __init__(self, *args, **kwargs):
            self.retrievers = {verifier_module.DataSource.MEDIA_DB: _MediaRetriever()}

    class _ClaimsEngine:
        async def run(self, **kwargs):
            callback_results.append(await kwargs["retrieve_fn"]("first claim"))
            callback_results.append(await kwargs["retrieve_fn"]("second claim"))
            return {
                "claims": [{"id": "c1"}],
                "summary": {"supported": 1, "refuted": 0, "nei": 0},
            }

    verifier = PostGenerationVerifier(credential_runtime=object())

    async def build_claims_engine():
        return _ClaimsEngine(), None, {"used": False}

    monkeypatch.setattr(verifier_module, "ClaimsEngine", object)
    monkeypatch.setattr(verifier_module, "MultiDatabaseRetriever", _MultiDatabaseRetriever)
    monkeypatch.setattr(verifier, "_build_claims_engine", build_claims_engine)

    outcome = await verifier.verify_and_maybe_fix(
        query="What is RAG?",
        answer="RAG is X.",
        base_documents=_base_docs(),
        media_db_path="media.db",
    )

    assert dispatches == 1
    assert callback_results == [[], []]
    assert outcome.total_claims == 1
    assert outcome.unsupported_ratio == 0.0
    assert outcome.embedding_coverage == "degraded"
    assert outcome.embedding_failure_code == "provider_configuration_invalid"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_claim_provider_failure_latches_adaptive_retrieval(monkeypatch):
    dispatches = 0
    generated_contexts: list[str] = []

    class _MediaRetriever:
        async def retrieve_hybrid(self, **_kwargs):
            nonlocal dispatches
            dispatches += 1
            raise ChatConfigurationError(
                "secret endpoint configuration detail",
                provider="local_api",
                error_code="provider_configuration_invalid",
            )

    class _MultiDatabaseRetriever:
        def __init__(self, *args, **kwargs):
            self.retrievers = {verifier_module.DataSource.MEDIA_DB: _MediaRetriever()}

    class _ClaimsEngine:
        async def run(self, **kwargs):
            assert await kwargs["retrieve_fn"]("unsupported claim") == []
            return {
                "claims": [{"id": "c1"}],
                "summary": {"supported": 0, "refuted": 1, "nei": 0},
            }

    class _AnswerGenerator:
        def __init__(self, **_kwargs):
            pass

        async def generate(self, **kwargs):
            generated_contexts.append(kwargs["context"])
            return {"answer": ""}

    verifier = PostGenerationVerifier(
        max_retries=1,
        unsupported_threshold=0.1,
        use_advanced_rewrites=False,
        credential_runtime=object(),
    )

    async def build_claims_engine():
        return _ClaimsEngine(), None, {"used": False}

    monkeypatch.setattr(verifier_module, "ClaimsEngine", object)
    monkeypatch.setattr(verifier_module, "MultiDatabaseRetriever", _MultiDatabaseRetriever)
    monkeypatch.setattr(verifier_module, "AnswerGenerator", _AnswerGenerator)
    monkeypatch.setattr(verifier, "_build_claims_engine", build_claims_engine)

    base_documents = _base_docs()
    outcome = await verifier.verify_and_maybe_fix(
        query="What is RAG?",
        answer="Original answer.",
        base_documents=base_documents,
        media_db_path="media.db",
    )

    assert dispatches == 1
    assert generated_contexts == ["A\n\nB"]
    assert base_documents == _base_docs()
    assert outcome.new_answer is None
    assert outcome.fixed is False
    assert outcome.total_claims == 1
    assert outcome.unsupported_ratio == 1.0
    assert outcome.embedding_coverage == "degraded"
    assert outcome.embedding_failure_code == "provider_configuration_invalid"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_adaptive_retrieval_reports_bounded_embedding_auth_degradation(monkeypatch):
    class _MediaRetriever:
        async def retrieve_hybrid(self, **kwargs):
            raise ChatAuthenticationError("secret detail", provider="openai")

    class _MultiDatabaseRetriever:
        def __init__(self, *args, **kwargs):
            self.retrievers = {verifier_module.DataSource.MEDIA_DB: _MediaRetriever()}

    async def unsupported_runner(**kwargs):
        return {
            "claims": [{"id": "c1"}],
            "summary": {"supported": 0, "refuted": 1, "nei": 0},
        }

    monkeypatch.setattr(verifier_module, "MultiDatabaseRetriever", _MultiDatabaseRetriever)
    monkeypatch.setattr(verifier_module, "generate_hypothetical_answer", None)
    monkeypatch.setattr(verifier_module, "hyde_embed_text", None)
    monkeypatch.setattr(verifier_module, "multi_strategy_expansion", None)
    monkeypatch.setattr(verifier_module, "AnswerGenerator", None)

    verifier = PostGenerationVerifier(
        claims_runner=unsupported_runner,
        max_retries=1,
        unsupported_threshold=0.1,
        credential_runtime=object(),
    )
    outcome = await verifier.verify_and_maybe_fix(
        query="What is RAG?",
        answer="RAG is X.",
        base_documents=_base_docs(),
        media_db_path="media.db",
    )

    assert outcome.embedding_coverage == "degraded"
    assert outcome.embedding_failure_code == "invalid_provider_credentials"
