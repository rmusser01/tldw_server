import asyncio
import importlib
import shutil
import sys
import types
from pathlib import Path

import pytest

from tldw_Server_API.app.core.RAG.exceptions import RAGConfigurationError, RAGError
from tldw_Server_API.app.core.RAG.rag_service import advanced_reranking as ar
from tldw_Server_API.app.core.RAG.rag_service.types import Document

_FLASHRANK_MODEL = "ms-marco-TinyBERT-L-2-v2"
_FLASHRANK_MODEL_FILE = "flashrank-TinyBERT-L-2-v2.onnx"
_FLASHRANK_REQUIRED_FILES = {
    _FLASHRANK_MODEL_FILE,
    "config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
}


def _write_flashrank_bundle(model_dir: Path) -> None:
    model_dir.mkdir(parents=True)
    for filename in _FLASHRANK_REQUIRED_FILES:
        (model_dir / filename).write_bytes(b"{}")


def _patch_flashrank_modules(monkeypatch, ranker_type) -> None:
    real_import_module = importlib.import_module

    def _import_module(name, package=None):
        if name == "flashrank.Ranker":
            return types.SimpleNamespace(
                model_file_map={_FLASHRANK_MODEL: _FLASHRANK_MODEL_FILE},
                listwise_rankers=set(),
            )
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _import_module)
    monkeypatch.setitem(sys.modules, "flashrank", types.SimpleNamespace(Ranker=ranker_type))


def _documents() -> list[Document]:
    return [
        Document(id="doc-1", content="alpha", metadata={}, score=0.7),
        Document(id="doc-2", content="beta", metadata={}, score=0.5),
    ]


@pytest.mark.unit
def test_preinstalled_flashrank_uses_one_resolved_config_and_cannot_download(
    monkeypatch,
    tmp_path,
):
    model_dir = tmp_path / _FLASHRANK_MODEL
    _write_flashrank_bundle(model_dir)
    calls = {"config": 0, "download": 0}

    class _DownloadCapableRanker:
        def __init__(self, model_name, cache_dir):
            calls["model_name"] = model_name
            calls["cache_dir"] = cache_dir
            self.cache_dir = Path(cache_dir)
            self.model_dir = self.cache_dir / model_name
            self._prepare_model_dir(model_name)

        def _prepare_model_dir(self, _model_name):
            self._download_model_files(_model_name)

        def _download_model_files(self, _model_name):
            calls["download"] += 1
            raise AssertionError("download must be unreachable")

    def _load_config():
        calls["config"] += 1
        return _FLASHRANK_MODEL, str(tmp_path)

    _patch_flashrank_modules(monkeypatch, _DownloadCapableRanker)
    monkeypatch.setattr(ar, "_load_flashrank_defaults_from_config", _load_config)

    reranker = ar.create_preinstalled_local_reranker("flashrank", top_k=3)

    assert reranker is not None
    assert calls["config"] == 1
    assert calls["download"] == 0
    assert calls["model_name"] == _FLASHRANK_MODEL
    assert Path(calls["cache_dir"]).resolve() == tmp_path.resolve()
    assert reranker._ranker.model_dir.resolve() == model_dir.resolve()
    assert reranker.config.fail_closed_on_error is True


@pytest.mark.unit
def test_preinstalled_flashrank_fails_if_verified_bundle_disappears_without_downloading(
    monkeypatch,
    tmp_path,
):
    model_dir = tmp_path / _FLASHRANK_MODEL
    _write_flashrank_bundle(model_dir)
    calls = {"download": 0}

    class _RacingRanker:
        def __init__(self, model_name, cache_dir):
            self.cache_dir = Path(cache_dir)
            self.model_dir = self.cache_dir / model_name
            shutil.rmtree(self.model_dir)
            self._prepare_model_dir(model_name)

        def _prepare_model_dir(self, _model_name):
            self._download_model_files(_model_name)

        def _download_model_files(self, _model_name):
            calls["download"] += 1
            raise AssertionError("download must be unreachable")

    _patch_flashrank_modules(monkeypatch, _RacingRanker)
    monkeypatch.setattr(
        ar,
        "_load_flashrank_defaults_from_config",
        lambda: (_FLASHRANK_MODEL, str(tmp_path)),
    )

    with pytest.raises(RAGConfigurationError, match="could not be loaded locally"):
        ar.create_preinstalled_local_reranker("flashrank", top_k=3)

    assert calls["download"] == 0


@pytest.mark.unit
def test_preinstalled_flashrank_blocks_direct_downloader_calls(monkeypatch, tmp_path):
    model_dir = tmp_path / _FLASHRANK_MODEL
    _write_flashrank_bundle(model_dir)
    calls = {"download": 0}

    class _DirectDownloadRanker:
        def __init__(self, model_name, cache_dir):
            self.cache_dir = Path(cache_dir)
            self.model_dir = self.cache_dir / model_name
            self._download_model_files(model_name)

        def _download_model_files(self, _model_name):
            calls["download"] += 1

    _patch_flashrank_modules(monkeypatch, _DirectDownloadRanker)
    monkeypatch.setattr(
        ar,
        "_load_flashrank_defaults_from_config",
        lambda: (_FLASHRANK_MODEL, str(tmp_path)),
    )

    with pytest.raises(RAGConfigurationError, match="could not be loaded locally"):
        ar.create_preinstalled_local_reranker("flashrank", top_k=3)

    assert calls["download"] == 0


@pytest.mark.unit
def test_preinstalled_flashrank_validation_errors_are_fixed_and_redacted(
    monkeypatch,
    tmp_path,
):
    model_dir = tmp_path / _FLASHRANK_MODEL
    _write_flashrank_bundle(model_dir)
    secret = "/private/model/cache/tenant-secret"

    def _raise_sensitive_error(*_args, **_kwargs):
        raise OSError(secret)

    monkeypatch.setattr(importlib, "import_module", _raise_sensitive_error)
    monkeypatch.setattr(
        ar,
        "_load_flashrank_defaults_from_config",
        lambda: (_FLASHRANK_MODEL, str(tmp_path)),
    )

    with pytest.raises(
        RAGConfigurationError,
        match="local model validation is unavailable",
    ) as exc_info:
        ar.create_preinstalled_local_reranker("flashrank", top_k=3)

    assert secret not in str(exc_info.value)
    assert exc_info.value.original_error is None
    assert exc_info.value.__cause__ is None


@pytest.mark.unit
def test_preinstalled_cross_encoder_is_configured_to_fail_closed(monkeypatch, tmp_path):
    model_dir = tmp_path / "cross-encoder"
    model_dir.mkdir()

    class _CrossEncoderReranker:
        def __init__(self, config):
            self.config = config
            self._ce = object()

    monkeypatch.setattr(ar, "_load_cross_encoder_model_from_config", lambda: str(model_dir))
    monkeypatch.setattr(ar, "TransformersCrossEncoderReranker", _CrossEncoderReranker)

    reranker = ar.create_preinstalled_local_reranker("cross_encoder", top_k=3)

    assert reranker is not None
    assert reranker.config.fail_closed_on_error is True


@pytest.mark.unit
def test_preinstalled_cross_encoder_raw_fallback_stays_local_only(monkeypatch, tmp_path):
    model_dir = tmp_path / "cross-encoder"
    model_dir.mkdir()
    calls = []

    class _UnavailableCrossEncoder:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("sentence-transformers unavailable")

    class _Tokenizer:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("tokenizer", model_id, kwargs))
            return cls()

    class _Model:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("model", model_id, kwargs))
            return cls()

        def eval(self):
            return self

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(CrossEncoder=_UnavailableCrossEncoder),
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoModelForSequenceClassification=_Model,
            AutoTokenizer=_Tokenizer,
        ),
    )
    monkeypatch.setattr(ar, "_load_cross_encoder_model_from_config", lambda: str(model_dir))

    reranker = ar.create_preinstalled_local_reranker("cross_encoder", top_k=3)

    assert reranker is not None
    assert [name for name, _, _ in calls] == ["tokenizer", "model"]
    for _, model_id, kwargs in calls:
        assert model_id == str(model_dir)
        assert kwargs["local_files_only"] is True
        assert kwargs["trust_remote_code"] is False


@pytest.mark.unit
@pytest.mark.parametrize("reranker_type", ["flashrank", "cross_encoder"])
def test_preinstalled_reranker_inference_errors_fail_closed(monkeypatch, reranker_type):
    class _FailingRanker:
        def rerank(self, _request):
            raise RuntimeError("secret local model path")

    class _FailingCrossEncoder:
        def predict(self, *_args, **_kwargs):
            raise RuntimeError("secret local model path")

    config = ar.RerankingConfig(top_k=1)
    config.fail_closed_on_error = True
    if reranker_type == "flashrank":

        class _RerankRequest:
            def __init__(self, **_kwargs):
                pass

        monkeypatch.setitem(
            sys.modules,
            "flashrank",
            types.SimpleNamespace(RerankRequest=_RerankRequest),
        )
        reranker = ar.FlashRankReranker.__new__(ar.FlashRankReranker)
        reranker.config = config
        reranker._ranker = _FailingRanker()
    else:
        reranker = ar.TransformersCrossEncoderReranker.__new__(ar.TransformersCrossEncoderReranker)
        reranker.config = config
        reranker._ce = _FailingCrossEncoder()
        reranker._using_st = True

    with pytest.raises(RAGError, match="Local reranking failed") as exc_info:
        asyncio.run(reranker.rerank("query", _documents()))

    assert "secret local model path" not in str(exc_info.value)


@pytest.mark.unit
def test_preinstalled_cross_encoder_rejects_nonfinite_normalized_scores():
    class _ExtremeCrossEncoder:
        @staticmethod
        def predict(*_args, **_kwargs):
            return [-1e308, 1e308]

    config = ar.RerankingConfig(top_k=2, fail_closed_on_error=True)
    reranker = ar.TransformersCrossEncoderReranker.__new__(ar.TransformersCrossEncoderReranker)
    reranker.config = config
    reranker._ce = _ExtremeCrossEncoder()
    reranker._using_st = True

    documents = [
        Document(id="doc-1", content="alpha", metadata={}, score=0.7),
        Document(id="doc-2", content="beta", metadata={}, score=0.6),
    ]
    with pytest.raises(RAGError, match="Local reranking failed"):
        asyncio.run(reranker.rerank("query", documents))


@pytest.mark.unit
@pytest.mark.parametrize(
    "results",
    [
        [{"id": 0, "score": 0.9}],
        [{"id": 0, "score": 0.9}, {"id": 0, "score": 0.8}],
        [{"id": 0, "score": 0.9}, {"id": 1, "score": float("nan")}],
        [{"id": 0, "score": 0.9}, {"id": 1, "score": float("inf")}],
    ],
)
def test_preinstalled_flashrank_rejects_incomplete_or_nonfinite_results(
    monkeypatch,
    results,
):
    class _RerankRequest:
        def __init__(self, **_kwargs):
            pass

    class _Ranker:
        def rerank(self, _request):
            return results

    monkeypatch.setitem(
        sys.modules,
        "flashrank",
        types.SimpleNamespace(RerankRequest=_RerankRequest),
    )
    reranker = ar.FlashRankReranker.__new__(ar.FlashRankReranker)
    reranker.config = ar.RerankingConfig(top_k=2, fail_closed_on_error=True)
    reranker._ranker = _Ranker()

    with pytest.raises(RAGError, match="Local reranking failed"):
        asyncio.run(reranker.rerank("query", _documents()))


@pytest.mark.unit
@pytest.mark.parametrize(
    "scores",
    [
        [0.9],
        [0.9, float("nan")],
        [0.9, float("inf")],
    ],
)
def test_preinstalled_cross_encoder_rejects_incomplete_or_nonfinite_scores(scores):
    class _CrossEncoder:
        def predict(self, *_args, **_kwargs):
            return scores

    reranker = ar.TransformersCrossEncoderReranker.__new__(ar.TransformersCrossEncoderReranker)
    reranker.config = ar.RerankingConfig(
        top_k=2,
        batch_size=2,
        fail_closed_on_error=True,
    )
    reranker._ce = _CrossEncoder()
    reranker._using_st = True

    with pytest.raises(RAGError, match="Local reranking failed"):
        asyncio.run(reranker.rerank("query", _documents()))
