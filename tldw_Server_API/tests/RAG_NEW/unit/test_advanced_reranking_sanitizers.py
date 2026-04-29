import asyncio
import sys
import types

import pytest

from tldw_Server_API.app.core.RAG.rag_service import advanced_reranking as ar
from tldw_Server_API.app.core.RAG.rag_service.types import Document


class _RecordingLogger:
    def __init__(self):
        self.error_calls = []
        self.warning_calls = []

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
