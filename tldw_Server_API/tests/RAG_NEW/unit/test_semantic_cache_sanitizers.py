import pytest

from tldw_Server_API.app.core.RAG.rag_service import semantic_cache
from tldw_Server_API.app.core.RAG.rag_service.semantic_cache import SemanticCache

pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(str(message))


class _FailingEmbedder:
    async def embed(self, _text: str):
        raise RuntimeError("embedding failed for /private/rag/semantic.db?token=secret-token")


@pytest.mark.asyncio
async def test_embedding_failure_log_omits_exception_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(semantic_cache, "logger", logger_stub)
    cache = SemanticCache(embedding_model=_FailingEmbedder())

    embedding = await cache.get_embedding("sensitive query")

    assert embedding is None
    assert logger_stub.errors == ["Failed to generate embedding"]
    joined = "\n".join(logger_stub.errors)
    assert "/private/" not in joined
    assert "secret-token" not in joined
    assert "embedding failed" not in joined


def test_save_failure_log_omits_exception_details(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(semantic_cache, "logger", logger_stub)

    def fail_dump(*_args, **_kwargs) -> None:
        raise RuntimeError("save failed for /private/rag/cache.json?token=secret-token")

    monkeypatch.setattr(semantic_cache.json, "dump", fail_dump)

    cache = SemanticCache()
    cache.persist_path = str(tmp_path / "semantic_cache.json")

    cache.save()

    assert logger_stub.errors == ["Failed to save semantic cache: RuntimeError"]
    joined = "\n".join(logger_stub.errors)
    assert "/private/" not in joined
    assert "secret-token" not in joined
    assert "save failed" not in joined


def test_load_failure_log_omits_exception_details_and_preserves_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(semantic_cache, "logger", logger_stub)

    def fail_load(*_args, **_kwargs):
        raise ValueError("load failed for /private/rag/cache.json?token=secret-token")

    monkeypatch.setattr(semantic_cache.json, "load", fail_load)

    cache_path = tmp_path / "semantic_cache.json"
    cache_path.write_text("{}", encoding="utf-8")
    cache = SemanticCache()
    cache.persist_path = str(cache_path)
    cache._hits = 3

    cache.load()

    assert cache._hits == 3
    assert logger_stub.errors == ["Failed to load semantic cache: ValueError"]
    joined = "\n".join(logger_stub.errors)
    assert "/private/" not in joined
    assert "secret-token" not in joined
    assert "load failed" not in joined
