from __future__ import annotations

from types import SimpleNamespace

import pytest
from loguru import logger

from tldw_Server_API.app.api.v1.endpoints import rag_unified
from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGRequest


pytestmark = pytest.mark.unit


class _LogCapture:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self._handler_id: int | None = None

    def __enter__(self) -> "_LogCapture":
        self._handler_id = logger.add(
            lambda message: self.messages.append(str(message)),
            level="INFO",
            filter=lambda record: record["level"].name == "INFO",
            format="{message}",
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handler_id is not None:
            logger.remove(self._handler_id)


@pytest.mark.asyncio
async def test_unified_search_info_log_omits_raw_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel_query = "private customer token sk-live-rag-secret"

    monkeypatch.setattr(
        rag_unified,
        "_apply_media_collection_scope",
        lambda request, _collections_db: request,
    )
    monkeypatch.setattr(
        rag_unified,
        "_build_standard_request_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stop after logging")),
    )

    request_raw = SimpleNamespace(state=SimpleNamespace())
    current_user = SimpleNamespace(username="alice", id=1)

    with _LogCapture() as logs:
        with pytest.raises(Exception, match="Search failed"):
            await rag_unified.unified_search_endpoint(
                request_raw=request_raw,
                request=UnifiedRAGRequest(query=sentinel_query),
                background_tasks=SimpleNamespace(),
                current_user=current_user,
                media_db=None,
                chacha_db=None,
                prompts_db=None,
                collections_db=None,
            )

    joined = "\n".join(logs.messages)
    assert sentinel_query not in joined
    assert "query_hash=" in joined
    assert f"len={len(sentinel_query)}" in joined


@pytest.mark.asyncio
async def test_advanced_search_info_log_omits_raw_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel_query = "confidential acquisition plan RAG query"

    async def _stop_after_logging(*args, **kwargs):
        raise RuntimeError("stop after logging")

    monkeypatch.setattr(rag_unified, "advanced_search", _stop_after_logging)

    current_user = SimpleNamespace(username="alice", id=1)

    with _LogCapture() as logs:
        with pytest.raises(Exception, match="Search failed"):
            await rag_unified.advanced_search_endpoint(
                request=SimpleNamespace(state=SimpleNamespace()),
                query=sentinel_query,
                with_citations=True,
                with_answer=True,
                current_user=current_user,
                media_db=None,
                chacha_db=None,
            )

    joined = "\n".join(logs.messages)
    assert sentinel_query not in joined
    assert "query_hash=" in joined
    assert f"len={len(sentinel_query)}" in joined
