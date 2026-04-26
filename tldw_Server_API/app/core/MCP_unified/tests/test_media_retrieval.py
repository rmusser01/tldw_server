"""Unit tests for media module helpers and retrieval behaviours."""

from datetime import datetime
from types import MethodType, SimpleNamespace
from typing import Any, Dict, List

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.implementations import media_module as media_module_impl
from tldw_Server_API.app.core.MCP_unified.modules.implementations.media_module import (
    MediaModule,
    UnsupportedMediaQueueBackendError,
)
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings


@pytest.fixture(autouse=True)
def _single_user_test_key(monkeypatch):
    monkeypatch.setenv("SINGLE_USER_TEST_API_KEY", "test-api-key-1234567890")
    reset_settings()
    yield
    reset_settings()


class FakeMediaDB:
    def __init__(self) -> None:
        # Prepare deterministic 5 chunks of 10 chars each
        self._chunks = [
            {"chunk_index": 0, "uuid": "u0", "chunk_text": "A" * 10},
            {"chunk_index": 1, "uuid": "u1", "chunk_text": "B" * 10},
            {"chunk_index": 2, "uuid": "u2", "chunk_text": "C" * 10},
            {"chunk_index": 3, "uuid": "u3", "chunk_text": "D" * 10},
            {"chunk_index": 4, "uuid": "u4", "chunk_text": "E" * 10},
        ]

    def get_media_by_id(self, media_id: int, include_deleted: bool = False, include_trash: bool = False) -> Dict[str, Any]:
        full = "".join(c["chunk_text"] for c in self._chunks)
        return {
            "id": media_id,
            "title": "T",
            "content": full,
            "type": "html",
            "url": None,
            "ingestion_date": None,
            "last_modified": None,
            "version": 1,
        }

    def has_unvectorized_chunks(self, media_id: int) -> bool:
        return True

    def get_unvectorized_chunk_index_by_uuid(self, media_id: int, chunk_uuid: str):
        for c in self._chunks:
            if c["uuid"] == chunk_uuid:
                return c["chunk_index"]
        return None

    def get_unvectorized_anchor_index_for_offset(self, media_id: int, approx_offset: int):
        # Map 10-char chunks
        return max(0, min(4, approx_offset // 10))

    def get_unvectorized_chunks_in_range(self, media_id: int, start_index: int, end_index: int) -> List[Dict[str, Any]]:
        si = max(0, min(start_index, end_index))
        ei = min(max(start_index, end_index), len(self._chunks) - 1)
        return [self._chunks[i] for i in range(si, ei + 1)]


@pytest.mark.asyncio
async def test_media_get_chunk_with_siblings_budget():
    mod = MediaModule(ModuleConfig(name="media"))

    # Monkeypatch per-user DB open to our fake
    mod._open_media_db = lambda ctx: FakeMediaDB()  # type: ignore[attr-defined]
    context = SimpleNamespace(user_id="1", metadata={})

    # Anchor around approx_offset=12 → chunk_index 1, cpt=1 → 10 tokens per chunk
    out = await mod.execute_tool(
        "media.get",
        {
            "media_id": 42,
            "retrieval": {
                "mode": "chunk_with_siblings",
                "max_tokens": 25,  # can fit 2 chunks (10 + 10) and not a third (would be 30)
                "chars_per_token": 1,  # nosec B105
                "loc": {"approx_offset": 12},
            },
        },
        context=context,
    )

    assert isinstance(out, dict)  # nosec B101
    assert out["meta"]["loc"]["chunk_index"] == 1  # nosec B101
    body = out["content"]
    # Greedy expansion adds left(0) then right(2) or vice versa depending on order; our code checks left then right
    # Anchor chunk_index 1 → chunks 1 and 0 can fit within 25 tokens (20 total); third chunk would exceed.
    assert body in ("B" * 10 + "\n\n" + "A" * 10, "A" * 10 + "\n\n" + "B" * 10)  # nosec B101
    # Ensure total chars <= 25
    assert len(body.replace("\n", "")) <= 25  # nosec B101


class RecordingDB:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def search_media_db(
        self,
        search_query: Any = None,
        search_fields: Any = None,
        media_types: Any = None,
        date_range: Any = None,
        must_have_keywords: Any = None,
        must_not_have_keywords: Any = None,
        sort_by: Any = None,
        media_ids_filter: Any = None,
        page: int = 1,
        results_per_page: int = 20,
        include_trash: bool = False,
        include_deleted: bool = False,
    ):
        self.calls.append(
            {
                "search_query": search_query,
                "media_types": media_types,
                "sort_by": sort_by,
                "page": page,
                "results_per_page": results_per_page,
                "include_trash": include_trash,
                "include_deleted": include_deleted,
            }
        )
        row = {
            "id": len(self.calls),
            "title": f"T{len(self.calls)}",
            "type": "video",
            "ingestion_date": None,
            "last_modified": None,
            "url": None,
        }
        return [row], 5

    def get_distinct_media_types(self):
        return ["video", "pdf"]


@pytest.mark.asyncio
async def test_search_media_cache_respects_filters():
    mod = MediaModule(ModuleConfig(name="media"))
    mod.db = RecordingDB()
    mod._media_cache = {}
    mod._cache_ttl = 300

    await mod._search_media(query="foo", search_type="keyword", limit=5, offset=0, media_types=["video"])
    assert len(mod.db.calls) == 1  # nosec B101

    await mod._search_media(query="foo", search_type="keyword", limit=5, offset=0, media_types=["audio"])
    assert len(mod.db.calls) == 2  # different filter should bypass cache  # nosec B101

    await mod._search_media(query="foo", search_type="keyword", limit=5, offset=0, media_types=["audio"])
    assert len(mod.db.calls) == 2  # cached response reused  # nosec B101


def test_clear_media_cache_flushes_all_entries():
    mod = MediaModule(ModuleConfig(name="media"))
    mod._media_cache = {"k": {"time": datetime.utcnow(), "data": {}}}
    mod._clear_media_cache(1)
    assert mod._media_cache == {}  # nosec B101


@pytest.mark.asyncio
async def test_media_resources_use_search_api():
    mod = MediaModule(ModuleConfig(name="media"))
    mod.db = RecordingDB()

    recent = await mod.read_resource("media://recent")
    popular = await mod.read_resource("media://popular")
    assert len(mod.db.calls) == 2  # nosec B101
    assert mod.db.calls[0]["sort_by"] == "last_modified_desc"  # nosec B101
    assert mod.db.calls[1]["sort_by"] == "date_desc"  # nosec B101
    assert recent["items"][0]["title"].startswith("T")  # nosec B101
    types_resource = await mod.read_resource("media://types")
    assert "video" in types_resource["items"]  # nosec B101
    assert "pdf" in types_resource["items"]  # nosec B101


@pytest.mark.asyncio
async def test_search_media_semantic_path(monkeypatch):
    mod = MediaModule(ModuleConfig(name="media"))
    mod.db = RecordingDB()
    mod._media_cache = {}
    mod._semantic_retrievers = {}
    mod._cache_ttl = 300

    class StubRetriever:
        def __init__(self) -> None:
            self.config = SimpleNamespace(max_results=0)

        async def _retrieve_vector(self, query: str, **_kwargs):
            return [
                SimpleNamespace(
                    id="42",
                    content="hello world",
                    metadata={"title": "Doc", "media_type": "text", "url": "u"},
                    score=0.9,
                )
            ]

    mod._get_semantic_retriever = MethodType(lambda self, db, ctx: StubRetriever(), mod)  # type: ignore
    result = await mod._search_media(query="hello", search_type="semantic", limit=5, offset=0)
    assert result["count"] == 1  # nosec B101
    assert result["results"][0]["id"] == 42  # nosec B101
    assert result["results"][0]["semantic_score"] == pytest.approx(0.9)  # nosec B101


def test_media_open_db_requires_context():
    mod = MediaModule(ModuleConfig(name="media"))
    mod.db = SimpleNamespace(db_path_str=":memory:", close_connection=lambda: None)
    mod._module_db_owner = mod.db
    with pytest.raises(PermissionError):
        mod._open_media_db(context=None)


def test_media_open_db_allows_injected_non_owner_db_without_context():
    mod = MediaModule(ModuleConfig(name="media"))
    injected_db = SimpleNamespace(db_path_str=":memory:", close_connection=lambda: None)
    mod.db = injected_db

    assert mod._open_media_db(context=None) is injected_db  # nosec B101


@pytest.mark.asyncio
async def test_get_transcript_srt_vtt_fallback(monkeypatch):
    mod = MediaModule(ModuleConfig(name="media"))

    class DummyDB:
        def get_media_by_id(self, media_id: int, include_deleted: bool = False, include_trash: bool = False):
            return {"owner_user_id": "1"}

    mod.db = DummyDB()
    context = SimpleNamespace(user_id="1", metadata={})
    monkeypatch.setattr(media_module_impl, "get_latest_transcription", lambda _db, _media_id: "Hello world")

    srt = await mod.execute_tool("get_transcript", {"media_id": 1, "format": "srt"}, context=context)
    assert "Hello world" in srt["transcript"]  # nosec B101
    assert "-->" in srt["transcript"]  # nosec B101

    vtt = await mod.execute_tool("get_transcript", {"media_id": 1, "format": "vtt"}, context=context)
    assert "WEBVTT" in vtt["transcript"]  # nosec B101
    assert "Hello world" in vtt["transcript"]  # nosec B101


@pytest.mark.asyncio
async def test_ingest_media_falls_back_to_inline_processing_when_queue_backend_is_unimplemented(monkeypatch):
    mod = MediaModule(ModuleConfig(name="media", settings={"ingestion_queue": "rq"}))
    events = []
    original_queue = mod._queue_media_job

    async def _fake_process(job_id: str):
        events.append(("process", job_id))

    monkeypatch.setattr(mod, "_validate_url", lambda _url: True)
    monkeypatch.setattr(mod, "_process_media_job", _fake_process)

    async def _recording_queue(job_id: str):
        events.append(("queue", job_id))
        return await original_queue(job_id)

    monkeypatch.setattr(mod, "_queue_media_job", _recording_queue)

    result = await mod._ingest_media(url="https://example.com/video", priority="normal")

    assert result["status"] == "processing"  # nosec B101
    assert events == [("queue", result["job_id"]), ("process", result["job_id"])]  # nosec B101


@pytest.mark.asyncio
async def test_ingest_media_propagates_generic_queue_runtime_error(monkeypatch):
    mod = MediaModule(ModuleConfig(name="media", settings={"ingestion_queue": "rq"}))
    monkeypatch.setattr(mod, "_validate_url", lambda _url: True)

    async def _boom(_job_id: str):
        raise RuntimeError("queue worker unavailable")

    async def _unexpected_process(_job_id: str):
        raise AssertionError("inline fallback should not run for generic queue failures")

    monkeypatch.setattr(mod, "_queue_media_job", _boom)
    monkeypatch.setattr(mod, "_process_media_job", _unexpected_process)

    with pytest.raises(RuntimeError, match="queue worker unavailable"):
        await mod._ingest_media(url="https://example.com/video", priority="normal")


def test_media_access_enforces_owner_user_id():
    mod = MediaModule(ModuleConfig(name="media"))

    class StubDB:
        def get_media_by_id(self, media_id: int, include_deleted: bool = False, include_trash: bool = False) -> Dict[str, Any]:
            return {"id": media_id, "owner_user_id": 12}

    ctx = SimpleNamespace(user_id="99", metadata={})
    with pytest.raises(PermissionError):
        mod._assert_media_access(1, ctx, StubDB())


@pytest.mark.asyncio
async def test_queue_media_job_requires_backend():
    mod = MediaModule(ModuleConfig(name="media"))
    with pytest.raises(RuntimeError):
        await mod._queue_media_job("job-1")


def test_media_access_missing_owner_fails_closed_multi_user(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("JWT_SECRET_KEY", "x" * 64)
    reset_settings()
    try:
        mod = MediaModule(ModuleConfig(name="media"))

        class StubDB:
            def get_media_by_id(self, media_id: int, include_deleted: bool = False, include_trash: bool = False) -> Dict[str, Any]:
                return {"id": media_id, "title": "no-owner"}

        ctx = SimpleNamespace(user_id="99", metadata={})
        with pytest.raises(PermissionError):
            mod._assert_media_access(1, ctx, StubDB())
    finally:
        reset_settings()


def test_media_access_lookup_error_fails_closed_multi_user(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("JWT_SECRET_KEY", "x" * 64)
    reset_settings()
    try:
        mod = MediaModule(ModuleConfig(name="media"))

        class StubDB:
            def get_media_by_id(self, media_id: int, include_deleted: bool = False, include_trash: bool = False) -> Dict[str, Any]:
                raise RuntimeError("db error")

        ctx = SimpleNamespace(user_id="99", metadata={})
        with pytest.raises(PermissionError):
            mod._assert_media_access(1, ctx, StubDB())
    finally:
        reset_settings()


@pytest.mark.asyncio
async def test_get_media_metadata_sanitizes_and_adds_description(monkeypatch):
    mod = MediaModule(ModuleConfig(name="media"))

    class StubDB:
        def get_media_by_id(self, media_id: int, include_deleted: bool = False, include_trash: bool = False) -> Dict[str, Any]:
            return {
                "id": media_id,
                "title": "Title",
                "content": "secret content",
                "client_id": "client",
                "vector_embedding": b"\x00\x01",
            }

    mod._open_media_db = lambda ctx: StubDB()  # type: ignore[assignment]
    monkeypatch.setattr(
        media_module_impl,
        "get_document_version",
        lambda db_instance, media_id, version_number=None, include_content=False: {"analysis_content": "desc"},
    )

    ctx = SimpleNamespace(user_id="1", metadata={})
    result = await mod._get_media_metadata(media_id=1, include_stats=False, context=ctx)
    assert result["description"] == "desc"  # nosec B101
    assert "content" not in result  # nosec B101
    assert "client_id" not in result  # nosec B101
    assert "vector_embedding" not in result  # nosec B101


@pytest.mark.asyncio
async def test_media_get_includes_description(monkeypatch):
    mod = MediaModule(ModuleConfig(name="media"))

    class StubDB:
        def get_media_by_id(self, media_id: int, include_deleted: bool = False, include_trash: bool = False) -> Dict[str, Any]:
            return {
                "id": media_id,
                "title": "Title",
                "content": "alpha beta gamma",
                "type": "text",
                "url": None,
                "ingestion_date": None,
                "last_modified": None,
                "version": 1,
            }

    mod._open_media_db = lambda ctx: StubDB()  # type: ignore[assignment]
    monkeypatch.setattr(
        media_module_impl,
        "get_document_version",
        lambda db_instance, media_id, version_number=None, include_content=False: {"analysis_content": "desc"},
    )

    ctx = SimpleNamespace(user_id="1", metadata={})
    result = await mod._media_get_normalized(media_id=1, retrieval=None, context=ctx)
    assert result["meta"]["description"] == "desc"  # nosec B101


@pytest.mark.asyncio
async def test_delete_media_permanent_requires_admin(monkeypatch):
    mod = MediaModule(ModuleConfig(name="media"))
    mod._media_cache = {}

    class StubDB:
        def get_media_by_id(self, media_id: int, include_deleted: bool = False, include_trash: bool = False) -> Dict[str, Any]:
            return {"id": media_id}

    mod._open_media_db = lambda ctx: StubDB()  # type: ignore[assignment]

    def _should_not_call(*_args, **_kwargs):
        raise AssertionError("permanent delete should be gated by admin")

    monkeypatch.setattr(media_module_impl, "permanently_delete_item", _should_not_call)
    ctx = SimpleNamespace(user_id="1", metadata={"roles": []})

    with pytest.raises(PermissionError):
        await mod._delete_media(media_id=1, permanent=True, context=ctx)
