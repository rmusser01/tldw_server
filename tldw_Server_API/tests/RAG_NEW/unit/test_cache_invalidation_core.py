import pytest

from tldw_Server_API.app.core.RAG.rag_service.cache_invalidation import (
    collect_cache_namespaces,
    invalidate_rag_caches,
)


pytestmark = pytest.mark.unit


def test_collect_cache_namespaces_includes_user_id_username_and_explicit_namespaces() -> None:
    namespaces = collect_cache_namespaces(
        type("User", (), {"id": 4, "username": "alice"})(),
        namespaces=[" tenant-a "],
    )

    assert namespaces == {"4", "alice", "tenant-a"}


def test_invalidate_rag_caches_clears_shared_and_agentic_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_calls: list[tuple[str | None, ...]] = []
    agentic_calls: list[str] = []

    def _clear_shared_caches(*, namespace: str | None = None) -> int:
        clear_calls.append((namespace,))
        return 1

    def _invalidate_intra_doc_vectors(media_id: str) -> int:
        agentic_calls.append(media_id)
        return 1

    monkeypatch.setattr(
        "tldw_Server_API.app.core.RAG.rag_service.semantic_cache.clear_shared_caches",
        _clear_shared_caches,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.RAG.rag_service.agentic_chunker.invalidate_intra_doc_vectors",
        _invalidate_intra_doc_vectors,
    )

    invalidate_rag_caches(
        type("User", (), {"id": 4, "username": "alice"})(),
        namespaces=["tenant-a"],
        media_id=99,
    )

    assert len(clear_calls) == 3
    assert {namespace for (namespace,) in clear_calls} == {"4", "alice", "tenant-a"}
    assert agentic_calls == ["99"]
