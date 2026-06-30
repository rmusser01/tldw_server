import sys
import types
from pathlib import Path

import pytest

from tldw_Server_API.app.core.RAG.rag_service import advanced_reranking as ar


class _RecordingLogger:
    def __init__(self):
        self.debug_calls = []

    def debug(self, *args, **kwargs):
        self.debug_calls.append((args, kwargs))


@pytest.mark.unit
def test_flashrank_cache_dir_defaults_to_repo_models_dir():
    resolved = Path(ar._resolve_flashrank_cache_dir(None))
    assert resolved == ar._repo_root_dir() / "models" / "flashrank"


@pytest.mark.unit
def test_flashrank_config_lookup_failure_logs_sanitized_debug(monkeypatch):
    recording_logger = _RecordingLogger()

    def _raise_sensitive_config_error():
        raise RuntimeError("config failed for /tmp/source token=secret")

    monkeypatch.delenv("RAG_FLASHRANK_MODEL_NAME", raising=False)
    monkeypatch.delenv("RAG_FLASHRANK_CACHE_DIR", raising=False)
    monkeypatch.setattr(ar, "logger", recording_logger)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.config.load_comprehensive_config",
        _raise_sensitive_config_error,
    )

    defaults = ar._load_flashrank_defaults_from_config()

    assert defaults == ("ms-marco-TinyBERT-L-2-v2", None)
    assert recording_logger.debug_calls
    assert all(not kwargs.get("exc_info") for _args, kwargs in recording_logger.debug_calls)

    rendered_calls = repr(recording_logger.debug_calls)
    assert "/tmp/source" not in rendered_calls
    assert "token=secret" not in rendered_calls


@pytest.mark.unit
def test_flashrank_reranker_handles_ranker_init_failure(monkeypatch):
    class _BrokenRanker:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("missing model bundle")

    monkeypatch.setitem(sys.modules, "flashrank", types.SimpleNamespace(Ranker=_BrokenRanker))
    class _Cfg:
        def has_section(self, section):
            return section == "RAG"

        def get(self, section, key, fallback=None):
            if section != "RAG":
                return fallback
            if key == "flashrank_model_name":
                return "ms-marco-TinyBERT-L-2-v2"
            if key == "flashrank_cache_dir":
                return "models/flashrank"
            return fallback

    monkeypatch.setattr("tldw_Server_API.app.core.config.load_comprehensive_config", lambda: _Cfg())

    reranker = ar.FlashRankReranker(ar.RerankingConfig())

    # Missing local assets should not fail request construction.
    assert reranker._ranker is None


@pytest.mark.unit
def test_flashrank_reranker_uses_env_model_and_cache(monkeypatch, tmp_path):
    calls = {}

    class _DummyRanker:
        def __init__(self, model_name, cache_dir):
            calls["model_name"] = model_name
            calls["cache_dir"] = cache_dir

    monkeypatch.setitem(sys.modules, "flashrank", types.SimpleNamespace(Ranker=_DummyRanker))
    monkeypatch.setattr("tldw_Server_API.app.core.config.load_comprehensive_config", lambda: None)
    monkeypatch.setenv("RAG_FLASHRANK_MODEL_NAME", "custom-model")
    monkeypatch.setenv("RAG_FLASHRANK_CACHE_DIR", str(tmp_path / "flashrank_cache"))

    reranker = ar.FlashRankReranker(ar.RerankingConfig())

    assert calls["model_name"] == "custom-model"
    assert Path(calls["cache_dir"]) == tmp_path / "flashrank_cache"
    assert reranker._ranker is not None
    assert reranker._flashrank_model_name == "custom-model"
    assert Path(reranker._flashrank_cache_dir) == tmp_path / "flashrank_cache"
