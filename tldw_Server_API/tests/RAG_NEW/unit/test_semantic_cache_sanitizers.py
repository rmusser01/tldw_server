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
