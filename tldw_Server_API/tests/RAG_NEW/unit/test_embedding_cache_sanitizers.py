import numpy as np
import pytest

from tldw_Server_API.app.core.RAG.rag_service import embedding_cache
from tldw_Server_API.app.core.RAG.rag_service.embedding_cache import EmbeddingCache


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(str(message))


def test_save_cache_failure_log_omits_exception_details_and_preserves_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(embedding_cache, "logger", logger_stub)

    def fail_dump(*_args, **_kwargs) -> None:
        raise RuntimeError("save failed for /private/rag/cache.json?token=secret-token")

    monkeypatch.setattr(embedding_cache.json, "dump", fail_dump)

    cache = EmbeddingCache(persist_path=None)
    cache.put("sensitive text", np.array([1.0, 2.0, 3.0]), model_name="model")
    cache.persist_path = str(tmp_path / "embedding_cache.json")

    cache._save_cache()

    cache.persist_path = None
    assert cache.get("sensitive text", model_name="model").tolist() == [1.0, 2.0, 3.0]
    assert logger_stub.errors == ["Failed to save cache: RuntimeError"]
    joined = "\n".join(logger_stub.errors)
    assert "/private/" not in joined
    assert "secret-token" not in joined
    assert "save failed" not in joined


def test_load_cache_failure_log_omits_exception_details_and_preserves_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(embedding_cache, "logger", logger_stub)

    def fail_load(*_args, **_kwargs):
        raise ValueError("load failed for /private/rag/cache.json?token=secret-token")

    monkeypatch.setattr(embedding_cache.json, "load", fail_load)

    cache_path = tmp_path / "embedding_cache.json"
    cache_path.write_text("{}", encoding="utf-8")
    cache = EmbeddingCache(persist_path=None)
    cache.put("existing text", np.array([4.0, 5.0, 6.0]), model_name="model")
    cache.persist_path = str(cache_path)

    cache._load_cache()

    cache.persist_path = None
    assert cache.get("existing text", model_name="model").tolist() == [4.0, 5.0, 6.0]
    assert logger_stub.errors == ["Failed to load cache: ValueError"]
    joined = "\n".join(logger_stub.errors)
    assert "/private/" not in joined
    assert "secret-token" not in joined
    assert "load failed" not in joined
