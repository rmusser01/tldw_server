"""
Unit tests for the post-generation verification interface.

Verifies that unsupported claims trigger adaptive retry metrics and that
central metrics registry counters/histograms are updated.
"""

import asyncio
import os
import subprocess
import sys
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionError
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatConfigurationError,
)
from tldw_Server_API.app.core.Claims_Extraction import monitoring as claims_monitoring
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import (
    SummaryProviderError,
)
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


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_outcome", "should_mark", "should_log_usage"),
    [
        ("valid", True, True),
        ("valid_stream", True, True),
        ("empty", False, True),
        ("malformed", False, True),
        ("error_shaped", False, True),
        ("error", False, False),
        ("partial_error_stream", False, False),
    ],
)
async def test_runtime_bound_real_claims_engine_starts_direct_and_closes_after_exit(
    monkeypatch: pytest.MonkeyPatch,
    provider_outcome: str,
    should_mark: bool,
    should_log_usage: bool,
) -> None:
    """Only structurally valid completed claims output is marked after exit."""
    import tldw_Server_API.app.core.Claims_Extraction.claims_engine as claims_module
    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool

    lifecycle: list[str] = []
    provider_entered = threading.Event()
    provider_release = threading.Event()
    default_entered = threading.Event()
    default_release = threading.Event()
    usage_entered = asyncio.Event()
    usage_release = asyncio.Event()

    class TrackingPool(BoundedDaemonPool):
        def _release_capacity(self) -> None:
            lifecycle.append("capacity-release")
            super()._release_capacity()

    handle = SimpleNamespace(
        provider="openai",
        api_key="runtime-secret",
        app_config={"openai_api": {"model": "model-a"}},
    )

    class Runtime:
        async def resolve(self, provider: str, *, model: str | None = None):
            assert provider == "openai"
            assert model == "model-a"
            return handle

        async def mark_used(self, resolved_handle: object) -> None:
            assert resolved_handle is handle
            lifecycle.append("mark")

        async def close(self) -> None:
            lifecycle.append("runtime-close")

    def block_default_executor() -> None:
        default_entered.set()
        default_release.wait(timeout=2.0)

    def blocking_analyze(*_args: Any, **_kwargs: Any) -> Any:
        lifecycle.append("provider-start")
        provider_entered.set()
        provider_release.wait(timeout=2.0)
        lifecycle.append("provider-exit")
        if provider_outcome == "error":
            raise SummaryProviderError(code="provider_failure", provider="openai")
        if provider_outcome == "valid_stream":
            return iter(['{"claims": [{"text": "Supported runtime claim."}]}'])
        if provider_outcome == "partial_error_stream":
            return iter(
                [
                    '{"claims": [{"text": "Incomplete runtime claim."}]}',
                    "Error: hostile provider trailer",
                ]
            )
        if provider_outcome == "empty":
            return ""
        if provider_outcome == "malformed":
            return "not valid claims json"
        if provider_outcome == "error_shaped":
            return '{"error": {"message": "provider failed"}}'
        return '{"claims": [{"text": "Supported runtime claim."}]}'

    async def controlled_usage_log(**_kwargs: Any) -> None:
        lifecycle.append("usage-start")
        usage_entered.set()
        await usage_release.wait()
        lifecycle.append("usage-exit")

    pool = TrackingPool(1)
    runtime = Runtime()
    verifier = PostGenerationVerifier(credential_runtime=runtime)
    monkeypatch.setattr(claims_module, "SYNC_ADAPTER_CALL_POOL", pool, raising=False)
    monkeypatch.setattr(verifier_module, "ClaimsEngine", claims_module.ClaimsEngine)
    monkeypatch.setattr(
        claims_module,
        "CLAIMS_PROVIDER_CALL_TIMEOUT_SECONDS",
        1.0,
        raising=False,
    )
    monkeypatch.setattr(
        claims_module,
        "_resolve_claims_llm_config",
        lambda: ("openai", "model-a", 0.1),
    )
    monkeypatch.setattr(claims_module, "_claims_local_nli_enabled", lambda: False)
    monkeypatch.setattr(claims_module, "_log_claims_llm_usage", controlled_usage_log)
    monkeypatch.setattr(sgl, "analyze", blocking_analyze)

    engine, credential_handle, state = await verifier._build_claims_engine()

    async def endpoint_scope() -> dict[str, Any]:
        try:
            return await verifier._run_claims_engine_owned(
                engine,
                credential_handle,
                state,
                answer="Supported runtime claim.",
                query="query",
                documents=_base_docs(),
                claim_extractor="llm",
                claim_verifier="nli",
                claims_max=2,
                nli_model=None,
            )
        finally:
            await runtime.close()

    loop = asyncio.get_running_loop()
    previous_executor = getattr(loop, "_default_executor", None)
    default_executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(default_executor)
    blocker = loop.run_in_executor(None, block_default_executor)
    task: asyncio.Task[dict[str, Any]] | None = None
    try:
        for _ in range(1000):
            if default_entered.is_set():
                break
            await asyncio.sleep(0.001)
        assert default_entered.is_set()

        task = asyncio.create_task(endpoint_scope())
        for _ in range(100):
            if provider_entered.is_set():
                break
            await asyncio.sleep(0.001)
        started_before_default_release = provider_entered.is_set()

        if not provider_entered.is_set():
            default_release.set()
            await asyncio.gather(blocker, return_exceptions=True)
            for _ in range(1000):
                if provider_entered.is_set():
                    break
                await asyncio.sleep(0.001)
        assert provider_entered.is_set()

        task.cancel()
        await asyncio.sleep(0.03)
        assert "runtime-close" not in lifecycle

        provider_release.set()
        if should_log_usage:
            for _ in range(1000):
                if usage_entered.is_set():
                    break
                await asyncio.sleep(0.001)
            assert usage_entered.is_set()
            assert "runtime-close" not in lifecycle
            usage_release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        provider_release.set()
        usage_release.set()
        default_release.set()
        await asyncio.gather(blocker, return_exceptions=True)
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        loop.set_default_executor(previous_executor or ThreadPoolExecutor())
        default_executor.shutdown(wait=True, cancel_futures=True)

    assert started_before_default_release is True
    expected = [
        "provider-start",
        "provider-exit",
        "capacity-release",
    ]
    if should_log_usage:
        expected.extend(["usage-start", "usage-exit"])
    if should_mark:
        expected.append("mark")
    expected.append("runtime-close")
    assert lifecycle == expected
    assert lifecycle.count("mark") == int(should_mark)
    expected_usage_calls = int(should_log_usage)
    assert lifecycle.count("usage-start") == expected_usage_calls
    assert lifecycle.count("usage-exit") == expected_usage_calls
    assert pool.active_count == 0


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_outcome", "should_mark"),
    [
        ("valid", True),
        ("empty", False),
        ("malformed", False),
        ("error_shaped", False),
        ("partial_error_stream", False),
    ],
)
async def test_runtime_bound_verifier_output_marks_only_structurally_valid_json(
    monkeypatch: pytest.MonkeyPatch,
    provider_outcome: str,
    should_mark: bool,
) -> None:
    """Verifier usage requires a completed valid verdict, never partial content."""
    import tldw_Server_API.app.core.Claims_Extraction.claims_engine as claims_module
    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool

    lifecycle: list[str] = []
    handle = SimpleNamespace(
        provider="openai",
        api_key="runtime-secret",
        app_config={"openai_api": {"model": "model-a"}},
    )

    class TrackingPool(BoundedDaemonPool):
        def _release_capacity(self) -> None:
            super()._release_capacity()
            lifecycle.append("capacity-release")

    class Runtime:
        async def resolve(self, provider: str, *, model: str | None = None) -> object:
            assert (provider, model) == ("openai", "model-a")
            return handle

        async def mark_used(self, resolved_handle: object) -> None:
            assert resolved_handle is handle
            lifecycle.append("mark")

    def analyze_verdict(*_args: Any, **_kwargs: Any) -> Any:
        lifecycle.append("provider-start")
        if provider_outcome == "empty":
            response: Any = ""
        elif provider_outcome == "malformed":
            response = "not valid verifier json"
        elif provider_outcome == "error_shaped":
            response = '{"error": {"message": "provider failed"}}'
        elif provider_outcome == "partial_error_stream":
            response = iter(
                [
                    '{"label": "supported", "confidence": 0.9}',
                    "Error: hostile provider trailer",
                ]
            )
        else:
            response = (
                '{"label": "supported", "confidence": 0.9, '
                '"rationale": "complete"}'
            )
        lifecycle.append("provider-exit")
        return response

    async def ignore_usage_log(**_kwargs: Any) -> None:
        return None

    pool = TrackingPool(1)
    runtime = Runtime()
    verifier = PostGenerationVerifier(credential_runtime=runtime)
    monkeypatch.setattr(claims_module, "SYNC_ADAPTER_CALL_POOL", pool, raising=False)
    monkeypatch.setattr(verifier_module, "ClaimsEngine", claims_module.ClaimsEngine)
    monkeypatch.setattr(
        claims_module,
        "_resolve_claims_llm_config",
        lambda: ("openai", "model-a", 0.1),
    )
    monkeypatch.setattr(claims_module, "_claims_local_nli_enabled", lambda: False)
    monkeypatch.setattr(claims_module, "_log_claims_llm_usage", ignore_usage_log)
    monkeypatch.setattr(sgl, "analyze", analyze_verdict)
    engine, credential_handle, state = await verifier._build_claims_engine()

    operation = verifier._run_claims_engine_owned(
        engine,
        credential_handle,
        state,
        answer="Acme was founded in 2000.",
        query="When was Acme founded?",
        documents=[
            Document(
                id="evidence",
                content="Acme was founded in 2000.",
                metadata={"source": DataSource.MEDIA_DB},
            )
        ],
        claim_extractor="heuristic",
        claim_verifier="llm",
        claims_max=1,
        claims_concurrency=1,
    )
    if provider_outcome == "partial_error_stream":
        with pytest.raises(SummaryProviderError):
            await operation
    else:
        await operation

    expected = ["provider-start", "provider-exit", "capacity-release"]
    if should_mark:
        expected.append("mark")
    assert lifecycle == expected
    assert lifecycle.count("mark") == int(should_mark)
    assert pool.active_count == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_production_claims_import_surface_binds_real_engine_to_verifier_and_pipeline() -> None:
    """Production imports cannot silently downgrade verification to available/zero."""
    from tldw_Server_API.app.core.Claims_Extraction.claims_engine import (
        ClaimsEngine as RealClaimsEngine,
    )
    from tldw_Server_API.app.core.RAG.rag_service import claims as claims_surface
    from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline as pipeline_module

    assert claims_surface.ClaimsEngine is RealClaimsEngine
    assert pipeline_module.PostGenerationVerifier is PostGenerationVerifier

    engine, credential_handle, state = await PostGenerationVerifier()._build_claims_engine()
    assert isinstance(engine, RealClaimsEngine)
    assert verifier_module.ClaimsEngine is RealClaimsEngine
    assert credential_handle is None
    assert state == {"used": False}


@pytest.mark.unit
def test_evaluator_first_import_lazily_recovers_all_claims_engine_bindings() -> None:
    """Evaluation-first imports cannot permanently disable RAG claims checks."""
    script = textwrap.dedent(
        """
        import asyncio

        from tldw_Server_API.app.core.Evaluations import rag_evaluator
        from tldw_Server_API.app.core.Claims_Extraction.budget_guard import (
            ClaimsJobContext as RealClaimsJobContext,
            resolve_claims_job_budget as real_resolve_claims_job_budget,
        )
        from tldw_Server_API.app.core.Claims_Extraction.claims_engine import (
            ClaimsEngine as RealClaimsEngine,
        )
        from tldw_Server_API.app.core.RAG.rag_service import generation
        from tldw_Server_API.app.core.RAG.rag_service import post_generation_verifier
        from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline

        assert rag_evaluator.ClaimsEngine is RealClaimsEngine
        assert unified_pipeline.ClaimsJobContext is RealClaimsJobContext
        assert (
            unified_pipeline.resolve_claims_job_budget
            is real_resolve_claims_job_budget
        )

        def analyze(*_args, **_kwargs):
            return '{"claims": []}'

        for module in (generation, unified_pipeline):
            resolved = module._resolve_claims_engine()
            assert resolved is RealClaimsEngine
            assert module.ClaimsEngine is RealClaimsEngine
            assert isinstance(resolved(analyze), RealClaimsEngine)

        verifier = post_generation_verifier.PostGenerationVerifier()
        engine, credential_handle, state = asyncio.run(verifier._build_claims_engine())
        assert isinstance(engine, RealClaimsEngine)
        assert post_generation_verifier.ClaimsEngine is RealClaimsEngine
        assert credential_handle is None
        assert state == {"used": False}

        class PatchedClaimsEngine:
            pass

        for module in (generation, post_generation_verifier, unified_pipeline):
            module.ClaimsEngine = PatchedClaimsEngine
            assert module._resolve_claims_engine() is PatchedClaimsEngine
        """
    )
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(  # noqa: S603 - fixed interpreter and in-repo script
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[4],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


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

    async def fake_hyde_generation(query: str, provider=None, model=None, **kwargs):
        captured["hyde_generation_provider"] = provider
        captured["hyde_generation_model"] = model
        captured["hyde_generation_runtime"] = kwargs.get("credential_runtime")
        captured["hyde_generation_metadata"] = kwargs.get("stage_metadata")
        return "hypothesis"

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
    monkeypatch.setattr(
        verifier_module,
        "generate_hypothetical_answer",
        lambda *args: pytest.fail("runtime HyDE must not use the sync helper"),
    )
    monkeypatch.setattr(
        verifier_module,
        "generate_hypothetical_answer_async",
        fake_hyde_generation,
        raising=False,
    )
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
        generation_provider="anthropic",
        generation_model="claude-3-7-sonnet",
    )

    assert captured["retrieval_runtime"] is runtime
    assert captured["hyde_generation_provider"] == "anthropic"
    assert captured["hyde_generation_model"] == "claude-3-7-sonnet"
    assert captured["hyde_generation_runtime"] is runtime
    assert isinstance(captured["hyde_generation_metadata"], dict)
    assert captured["hyde_runtime"] is runtime
    assert isinstance(captured["hyde_metadata"], dict)
    assert captured["hyde_generation_metadata"] is captured["hyde_metadata"]
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
    generator_calls = 0
    recheck_calls = 0

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
            nonlocal recheck_calls
            if kwargs["retrieve_fn"] is None:
                recheck_calls += 1
                return {
                    "claims": [{"id": "repaired"}],
                    "summary": {"supported": 1, "refuted": 0, "nei": 0},
                }
            assert await kwargs["retrieve_fn"]("unsupported claim") == []
            return {
                "claims": [{"id": "c1"}],
                "summary": {"supported": 0, "refuted": 1, "nei": 0},
            }

    class _AnswerGenerator:
        def __init__(self, **_kwargs):
            pass

        async def generate(self, **kwargs):
            nonlocal generator_calls
            generator_calls += 1
            return {"answer": "Replacement answer."}

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
    assert generator_calls == 0
    assert recheck_calls == 0
    assert base_documents == _base_docs()
    assert outcome.new_answer is None
    assert outcome.fixed is False
    assert outcome.total_claims == 1
    assert outcome.unsupported_ratio == 1.0
    assert outcome.embedding_coverage == "degraded"
    assert outcome.embedding_failure_code == "provider_configuration_invalid"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_adaptive_provider_failure_aborts_repair_before_empty_union(monkeypatch):
    dispatches = 0
    generator_calls = 0
    runner_calls = 0
    diversity_calls = 0

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

    class _AnswerGenerator:
        def __init__(self, **_kwargs):
            pass

        async def generate(self, **_kwargs):
            nonlocal generator_calls
            generator_calls += 1
            return {"answer": "Replacement answer."}

    async def claims_runner(**_kwargs):
        nonlocal runner_calls
        runner_calls += 1
        if runner_calls == 1:
            return {
                "claims": [{"id": "c1"}],
                "summary": {"supported": 0, "refuted": 1, "nei": 0},
            }
        return {
            "claims": [{"id": "repaired"}],
            "summary": {"supported": 1, "refuted": 0, "nei": 0},
        }

    def select_diverse(docs, **_kwargs):
        nonlocal diversity_calls
        diversity_calls += 1
        return docs

    monkeypatch.setattr(verifier_module, "MultiDatabaseRetriever", _MultiDatabaseRetriever)
    monkeypatch.setattr(verifier_module, "generate_hypothetical_answer", None)
    monkeypatch.setattr(verifier_module, "hyde_embed_text", None)
    monkeypatch.setattr(verifier_module, "multi_strategy_expansion", None)
    monkeypatch.setattr(verifier_module, "AnswerGenerator", _AnswerGenerator)
    monkeypatch.setattr(verifier_module, "_select_diverse", select_diverse)

    verifier = PostGenerationVerifier(
        claims_runner=claims_runner,
        max_retries=1,
        unsupported_threshold=0.1,
        use_advanced_rewrites=True,
        credential_runtime=object(),
    )
    base_documents = _base_docs()
    outcome = await verifier.verify_and_maybe_fix(
        query="What is RAG?",
        answer="Original answer.",
        base_documents=base_documents,
        media_db_path="media.db",
    )

    assert dispatches == 1
    assert diversity_calls == 0
    assert generator_calls == 0
    assert runner_calls == 1
    assert base_documents == _base_docs()
    assert outcome.new_answer is None
    assert outcome.fixed is False
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
