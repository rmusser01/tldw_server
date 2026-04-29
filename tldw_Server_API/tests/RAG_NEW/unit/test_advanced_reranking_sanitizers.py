import asyncio
import sys
import types

import pytest

from tldw_Server_API.app.core.RAG.rag_service import advanced_reranking as ar
from tldw_Server_API.app.core.RAG.rag_service.types import Document


class _RecordingLogger:
    def __init__(self):
        self.debug_calls = []
        self.error_calls = []
        self.warning_calls = []

    def debug(self, *args, **kwargs):
        self.debug_calls.append((args, kwargs))

    def error(self, *args, **kwargs):
        self.error_calls.append((args, kwargs))

    def warning(self, *args, **kwargs):
        self.warning_calls.append((args, kwargs))

    def info(self, *args, **kwargs):
        pass


def _documents():
    return [
        Document(id="doc-1", content="alpha", metadata={}, score=0.7),
        Document(id="doc-2", content="beta", metadata={}, score=0.5),
    ]


class _SensitiveError(RuntimeError):
    def __str__(self):
        return "RAW_DEBUG_MARKER path=/private/rag-cache token=secret-token"


def _assert_debug_log_sanitized(logger, expected_message):
    rendered_calls = repr(logger.debug_calls)
    assert expected_message in rendered_calls
    assert "exc_info" not in rendered_calls
    assert "RAW_DEBUG_MARKER" not in rendered_calls
    assert "/private/rag-cache" not in rendered_calls
    assert "token=secret-token" not in rendered_calls


@pytest.mark.unit
def test_flashrank_rerank_fallback_log_is_sanitized(monkeypatch):
    class _RerankRequest:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _SensitiveRanker:
        def rerank(self, _request):
            raise RuntimeError("flashrank failed at /private/models token=secret")

    logger = _RecordingLogger()
    monkeypatch.setattr(ar, "logger", logger)
    monkeypatch.setitem(
        sys.modules,
        "flashrank",
        types.SimpleNamespace(RerankRequest=_RerankRequest),
    )

    reranker = ar.FlashRankReranker.__new__(ar.FlashRankReranker)
    reranker.config = ar.RerankingConfig(top_k=1)
    reranker._ranker = _SensitiveRanker()

    result = asyncio.run(reranker.rerank("query", _documents()))

    assert [item.document.id for item in result] == ["doc-1"]
    rendered_calls = repr(logger.error_calls)
    assert "FlashRank reranking failed" in rendered_calls
    assert "/private/models" not in rendered_calls
    assert "token=secret" not in rendered_calls


@pytest.mark.unit
def test_transformers_rerank_fallback_log_is_sanitized(monkeypatch):
    class _SensitiveCrossEncoder:
        def predict(self, *_args, **_kwargs):
            raise RuntimeError("cross encoder read /private/hf-cache token=secret")

    logger = _RecordingLogger()
    monkeypatch.setattr(ar, "logger", logger)

    reranker = ar.TransformersCrossEncoderReranker.__new__(ar.TransformersCrossEncoderReranker)
    reranker.config = ar.RerankingConfig(top_k=1, batch_size=2)
    reranker._ce = _SensitiveCrossEncoder()
    reranker._using_st = True

    result = asyncio.run(reranker.rerank("query", _documents()))

    assert [item.document.id for item in result] == ["doc-1"]
    rendered_calls = repr(logger.error_calls)
    assert "Transformers cross-encoder reranking failed" in rendered_calls
    assert "/private/hf-cache" not in rendered_calls
    assert "token=secret" not in rendered_calls


@pytest.mark.unit
def test_transformers_model_load_fallback_warning_is_sanitized(monkeypatch):
    sensitive_model_id = "/private/hf-cache/acme-reranker?token=model-secret"

    class _FailingCrossEncoder:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("sentence-transformers read /private/st-cache token=st-secret")

    class _FailingTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            raise RuntimeError("tokenizer read /private/hf-cache token=loader-secret")

    class _UnusedModel:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):  # pragma: no cover - tokenizer fails first
            raise AssertionError("model loader should not be reached")

    logger = _RecordingLogger()
    monkeypatch.setattr(ar, "logger", logger)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.config.load_and_log_configs",
        lambda: {"TRUSTED_HF_REMOTE_CODE_MODELS": []},
    )
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(CrossEncoder=_FailingCrossEncoder),
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoModelForSequenceClassification=_UnusedModel,
            AutoTokenizer=_FailingTokenizer,
        ),
    )

    reranker = ar.TransformersCrossEncoderReranker(
        ar.RerankingConfig(model_name=sensitive_model_id, top_k=1)
    )
    result = asyncio.run(reranker.rerank("query", _documents()))

    assert reranker._ce is None
    assert not hasattr(reranker, "_model")
    assert [item.document.id for item in result] == ["doc-1"]
    rendered_calls = repr(logger.warning_calls)
    assert "Failed to load transformers reranker model" in rendered_calls
    assert sensitive_model_id not in rendered_calls
    assert "/private/hf-cache" not in rendered_calls
    assert "token=model-secret" not in rendered_calls
    assert "token=loader-secret" not in rendered_calls
    assert "sentence-transformers read" not in rendered_calls


@pytest.mark.unit
def test_llamacpp_rerank_fallback_log_is_sanitized(monkeypatch):
    async def _raise_sensitive_subprocess_error(*_args, **_kwargs):
        raise RuntimeError("llama backend read /private/llama.gguf token=secret")

    logger = _RecordingLogger()
    monkeypatch.setattr(ar, "logger", logger)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _raise_sensitive_subprocess_error)

    reranker = ar.LlamaCppReranker.__new__(ar.LlamaCppReranker)
    reranker.config = ar.RerankingConfig(top_k=1)
    reranker.binary = "llama-embedding"
    reranker.model_path = "model.gguf"
    reranker.sep = "<#sep#>"
    reranker.embd_format = "json+"
    reranker.pooling = "mean"
    reranker.normalize = -1
    reranker.ngl = 0
    reranker.max_doc_chars = 2000
    reranker.query_prefix = None
    reranker.doc_prefix = None

    result = asyncio.run(reranker.rerank("query", _documents()))

    assert [item.document.id for item in result] == ["doc-1"]
    rendered_calls = repr(logger.error_calls)
    assert "LlamaCppReranker failed" in rendered_calls
    assert "/private/llama.gguf" not in rendered_calls
    assert "token=secret" not in rendered_calls


@pytest.mark.unit
def test_qwen3_device_move_debug_fallback_log_is_sanitized(monkeypatch):
    class _FakeTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return _FakeTokenizer()

        def convert_tokens_to_ids(self, token):
            return {"no": 1, "yes": 2}[token]

        def encode(self, *_args, **_kwargs):
            return [1, 2]

    class _FakeModel:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return _FakeModel()

        def eval(self):
            return self

        def to(self, _device):
            raise _SensitiveError()

    logger = _RecordingLogger()
    monkeypatch.setattr(ar, "logger", logger)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoModelForCausalLM=_FakeModel,
            AutoTokenizer=_FakeTokenizer,
        ),
    )
    monkeypatch.setattr(ar, "load_prompt", lambda *_args, **_kwargs: None)

    reranker = ar.Qwen3CausalLMReranker(
        ar.RerankingConfig(transformers_device="cuda", top_k=1)
    )

    assert reranker._device == "cuda"
    assert reranker.token_true_id == 2
    _assert_debug_log_sanitized(
        logger,
        "LLM scoring reranker device move failed; keeping default device",
    )


@pytest.mark.unit
def test_llm_scoring_metrics_increment_debug_fallback_logs_are_sanitized(monkeypatch):
    class _FailingMetricsCollector:
        def increment(self, *_args, **_kwargs):
            raise _SensitiveError()

    class _StaticLLM:
        def analyze(self, _prompt):
            return "0.75"

    def _fail_increment_counter(*_args, **_kwargs):
        raise _SensitiveError()

    logger = _RecordingLogger()
    monkeypatch.setattr(ar, "logger", logger)
    monkeypatch.setattr(ar, "load_prompt", lambda *_args, **_kwargs: "score")
    monkeypatch.setenv("RAG_LLM_RERANK_TIMEOUT_SEC", "1")
    monkeypatch.setenv("RAG_LLM_RERANK_TOTAL_BUDGET_SEC", "10")
    monkeypatch.setenv("RAG_LLM_RERANK_MAX_DOCS", "2")
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.RAG.rag_service.metrics_collector",
        types.SimpleNamespace(get_metrics_collector=lambda: _FailingMetricsCollector()),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Metrics.metrics_manager.increment_counter",
        _fail_increment_counter,
    )

    reranker = ar.LLMReranker(ar.RerankingConfig(top_k=2), llm_client=_StaticLLM())
    scores = asyncio.run(reranker._score_batch("query", _documents()))

    assert scores == [0.75, 0.75]
    _assert_debug_log_sanitized(logger, "LLM reranker local metrics increment failed")
    _assert_debug_log_sanitized(logger, "LLM reranker central metrics increment failed")


@pytest.mark.unit
def test_llm_scoring_docs_scored_metric_debug_fallback_log_is_sanitized(monkeypatch):
    class _FailingMetricsCollector:
        def increment(self, *_args, **_kwargs):
            raise _SensitiveError()

    class _StaticLLM:
        def analyze(self, _prompt):
            return "0.5"

    class _LoggerRaisingOnFirstMetricsDebug(_RecordingLogger):
        def __init__(self):
            super().__init__()
            self._raised = False

        def debug(self, *args, **kwargs):
            if (
                args
                and str(args[0]).startswith("LLM reranker local metrics increment failed")
                and not self._raised
            ):
                self._raised = True
                raise _SensitiveError()
            super().debug(*args, **kwargs)

    logger = _LoggerRaisingOnFirstMetricsDebug()
    monkeypatch.setattr(ar, "logger", logger)
    monkeypatch.setattr(ar, "load_prompt", lambda *_args, **_kwargs: "score")
    monkeypatch.setenv("RAG_LLM_RERANK_TIMEOUT_SEC", "1")
    monkeypatch.setenv("RAG_LLM_RERANK_TOTAL_BUDGET_SEC", "10")
    monkeypatch.setenv("RAG_LLM_RERANK_MAX_DOCS", "1")
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.RAG.rag_service.metrics_collector",
        types.SimpleNamespace(get_metrics_collector=lambda: _FailingMetricsCollector()),
    )

    reranker = ar.LLMReranker(ar.RerankingConfig(top_k=1), llm_client=_StaticLLM())
    scores = asyncio.run(reranker._score_batch("query", _documents()))

    assert scores == [0.5]
    _assert_debug_log_sanitized(logger, "LLM reranker docs_scored metric failed")


@pytest.mark.unit
def test_two_tier_duration_metric_debug_fallback_logs_are_sanitized(monkeypatch):
    class _StaticCrossReranker:
        async def rerank(self, _query, documents, original_scores=None):
            return [
                ar.ScoredDocument(
                    document=doc,
                    original_score=doc.score,
                    rerank_score=doc.score,
                    relevance_score=doc.score,
                )
                for doc in documents
            ]

    class _StaticLLMReranker:
        async def rerank(self, _query, documents, original_scores=None):
            return [
                ar.ScoredDocument(
                    document=doc,
                    original_score=doc.score,
                    rerank_score=0.25 if doc.id == "sentinel:irrelevant" else doc.score,
                    relevance_score=0.25 if doc.id == "sentinel:irrelevant" else doc.score,
                )
                for doc in documents
            ]

    def _fail_observe_histogram(*_args, **_kwargs):
        raise _SensitiveError()

    logger = _RecordingLogger()
    monkeypatch.setattr(ar, "logger", logger)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Metrics.metrics_manager.observe_histogram",
        _fail_observe_histogram,
    )

    reranker = ar.TwoTierReranker(
        ar.RerankingConfig(top_k=2),
        cross_reranker=_StaticCrossReranker(),
        llm_reranker=_StaticLLMReranker(),
    )
    result = asyncio.run(reranker.rerank("query", _documents()))

    assert [item.document.id for item in result] == ["doc-1", "doc-2"]
    _assert_debug_log_sanitized(
        logger,
        "Two-tier reranker cross-encoder duration metric failed",
    )
    _assert_debug_log_sanitized(
        logger,
        "Two-tier reranker llm duration metric failed",
    )
