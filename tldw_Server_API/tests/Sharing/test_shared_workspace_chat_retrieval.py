"""Security contracts for recipient retrieval over shared workspace sources."""
from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import (
    unified_rag_pipeline,
)
from tldw_Server_API.app.core.Sharing.shared_workspace_chat_service import (
    SHARED_RETRIEVAL_POLICY,
    SharedSourceSnapshot,
    SharedSourceSnapshotItem,
    SharedWorkspaceChatService,
    SharedWorkspaceNoRelevantEvidence,
    SharedWorkspaceRetrievalUnavailable,
    SharedWorkspaceSourceChanged,
    SharedWorkspaceSourceScopeInvalid,
    SharedWorkspaceSourceSubsetRequired,
    VerifiedSharedEvidence,
)

pytestmark = pytest.mark.unit


def _source(
    source_id: str,
    media_id: int,
    *,
    title: str | None = None,
    position: int = 0,
    selected: bool = True,
) -> dict[str, Any]:
    return {
        "id": source_id,
        "workspace_id": "workspace-alpha",
        "media_id": media_id,
        "title": title or f"Title {source_id}",
        "source_type": "pdf",
        "position": position,
        "selected": selected,
        "added_at": "2026-08-21T00:00:00+00:00",
    }


def _media(
    media_id: int,
    *,
    media_uuid: str | None = None,
    content_hash: str | None = None,
    content: str = "retrieval-ready text",
    deleted: int = 0,
    is_trash: int = 0,
    vector_processing: int = 1,
    chunking_status: str = "completed",
) -> dict[str, Any]:
    return {
        "id": media_id,
        "uuid": media_uuid or f"media-uuid-{media_id}",
        "content_hash": content_hash or f"sha256-{media_id}",
        "title": f"Media {media_id}",
        "type": "pdf",
        "url": f"https://example.test/media/{media_id}",
        "content": content,
        "deleted": deleted,
        "is_trash": is_trash,
        "vector_processing": vector_processing,
        "chunking_status": chunking_status,
    }


class _OwnerChaCha:
    def __init__(self, sources: list[dict[str, Any]]) -> None:
        self.sources = sources
        self.calls: list[str] = []

    def list_workspace_sources(self, workspace_id: str) -> list[dict[str, Any]]:
        self.calls.append(workspace_id)
        return [dict(source) for source in self.sources]


class _OwnerMedia:
    def __init__(self, rows: dict[int, dict[str, Any]]) -> None:
        self.rows = rows
        self.by_id_calls: list[tuple[int, bool, bool]] = []
        self.status_calls: list[tuple[int, bool, bool]] = []

    def get_media_by_id(
        self,
        media_id: int,
        *,
        include_deleted: bool = False,
        include_trash: bool = False,
    ) -> dict[str, Any] | None:
        self.by_id_calls.append((media_id, include_deleted, include_trash))
        row = self.rows.get(media_id)
        if row is None:
            return None
        if not include_deleted and row.get("deleted"):
            return None
        if not include_trash and row.get("is_trash"):
            return None
        return dict(row)

    def get_media_status_by_id(
        self,
        media_id: int,
        *,
        include_deleted: bool = False,
        include_trash: bool = False,
    ) -> dict[str, Any] | None:
        self.status_calls.append((media_id, include_deleted, include_trash))
        return self.get_media_by_id(
            media_id,
            include_deleted=include_deleted,
            include_trash=include_trash,
        )


class _Pipeline:
    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


def _result(
    documents: list[Any],
    *,
    errors: list[str] | None = None,
    generated_answer: Any = None,
    metadata: dict[str, Any] | None = None,
    cache_hit: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        documents=documents,
        errors=errors or [],
        generated_answer=generated_answer,
        metadata=metadata or {},
        cache_hit=cache_hit,
    )


def _document(
    media_id: Any,
    *,
    document_id: str = "chunk-1",
    content: str = "bounded evidence",
    source: Any = "media_db",
    score: Any = 0.75,
    chunk_index: Any = 1,
    start_char: Any = 10,
    end_char: Any = 26,
    title: str = "Primary evidence",
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "media_id": media_id,
        "chunk_id": document_id,
        "chunk_index": chunk_index,
        "start_char": start_char,
        "end_char": end_char,
        "title": title,
    }
    if source is not None:
        metadata["source"] = source
    return {
        "id": document_id,
        "content": content,
        "score": score,
        "metadata": metadata,
    }


def _service(
    sources: list[dict[str, Any]],
    rows: dict[int, dict[str, Any]],
    *,
    pipeline: _Pipeline | None = None,
) -> tuple[SharedWorkspaceChatService, _OwnerChaCha, _OwnerMedia, _Pipeline]:
    owner_chacha = _OwnerChaCha(sources)
    owner_media = _OwnerMedia(rows)
    resolved_pipeline = pipeline or _Pipeline(_result([]))
    service = SharedWorkspaceChatService(
        owner_chacha_db=owner_chacha,
        owner_media_db=owner_media,
        owner_media_db_path="/private/owner/media.db",
        owner_user_id=7,
        workspace_id="workspace-alpha",
        rag_pipeline=resolved_pipeline,
    )
    return service, owner_chacha, owner_media, resolved_pipeline


def _snapshot_hash(items: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        items,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_snapshot_types_and_service_are_frozen() -> None:
    item = SharedSourceSnapshotItem(
        source_id="source-a",
        media_id=1,
        media_uuid="uuid-1",
        content_hash="hash-1",
        readiness_class="queryable",
    )
    snapshot = SharedSourceSnapshot(
        mode="include",
        items=(item,),
        snapshot_hash="snapshot-hash",
    )
    evidence = VerifiedSharedEvidence(
        label="E1",
        source_id="source-a",
        source_title="Evidence",
        content="text",
        score=1.0,
        chunk_index=1,
        start_char=0,
        end_char=4,
    )
    service, _chacha, _media_db, _pipeline = _service(
        [_source("source-a", 1)],
        {1: _media(1)},
    )

    for value, attribute in (
        (item, "media_id"),
        (snapshot, "snapshot_hash"),
        (evidence, "content"),
        (service, "owner_user_id"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(value, attribute, "changed")


def test_all_and_include_resolve_sorted_canonical_snapshot_and_hash_only_approved_fields() -> None:
    sources = [_source("source-b", 2, position=0), _source("source-a", 1, position=9)]
    service, _chacha, media_db, _pipeline = _service(
        sources,
        {1: _media(1), 2: _media(2)},
    )

    all_snapshot = service.resolve_source_snapshot(mode="all")
    include_snapshot = service.resolve_source_snapshot(
        mode="include",
        source_ids=("source-b", "source-a"),
    )

    assert all_snapshot.mode == "all"
    assert include_snapshot.mode == "include"
    assert all_snapshot.items == include_snapshot.items
    assert all_snapshot.source_ids == ("source-a", "source-b")
    assert all_snapshot.media_ids == (1, 2)
    approved_items = [
        {
            "content_hash": "sha256-1",
            "media_id": 1,
            "media_uuid": "media-uuid-1",
            "readiness_class": "queryable",
            "source_id": "source-a",
        },
        {
            "content_hash": "sha256-2",
            "media_id": 2,
            "media_uuid": "media-uuid-2",
            "readiness_class": "queryable",
            "source_id": "source-b",
        },
    ]
    assert all_snapshot.snapshot_hash == _snapshot_hash(approved_items)
    assert include_snapshot.snapshot_hash == all_snapshot.snapshot_hash
    assert media_db.by_id_calls == [(1, True, True), (2, True, True)] * 2
    assert media_db.status_calls == []


@pytest.mark.parametrize(
    ("source_ids", "sources", "rows"),
    [
        (("source-a", "source-a"), [_source("source-a", 1)], {1: _media(1)}),
        (("   ",), [_source("source-a", 1)], {1: _media(1)}),
        (("unknown",), [_source("source-a", 1)], {1: _media(1)}),
        (("source-a",), [_source("source-a", 1)], {1: _media(1, content="")}),
    ],
    ids=["duplicate", "empty", "unknown", "nonqueryable"],
)
def test_include_rejects_duplicate_empty_unknown_and_nonqueryable_ids(
    source_ids: tuple[str, ...],
    sources: list[dict[str, Any]],
    rows: dict[int, dict[str, Any]],
) -> None:
    service, _chacha, _media_db, _pipeline = _service(sources, rows)

    with pytest.raises(SharedWorkspaceSourceScopeInvalid) as exc_info:
        service.resolve_source_snapshot(mode="include", source_ids=source_ids)

    assert exc_info.value.code == "invalid_shared_chat_request"
    assert "source-a" not in str(exc_info.value)
    assert "unknown" not in str(exc_info.value)


def test_source_caps_reject_501_requested_or_effective_sources_without_truncation() -> None:
    sources = [_source(f"source-{index:03d}", index) for index in range(1, 502)]
    rows = {index: _media(index) for index in range(1, 502)}
    service, _chacha, _media_db, pipeline = _service(sources, rows)

    with pytest.raises(SharedWorkspaceSourceScopeInvalid):
        service.resolve_source_snapshot(
            mode="include",
            source_ids=tuple(source["id"] for source in sources),
        )
    with pytest.raises(SharedWorkspaceSourceSubsetRequired) as exc_info:
        service.resolve_source_snapshot(mode="all")

    assert exc_info.value.code == "source_subset_required"
    assert pipeline.calls == []


def test_exactly_500_effective_sources_are_allowed() -> None:
    sources = [_source(f"source-{index:03d}", index) for index in range(1, 501)]
    rows = {index: _media(index) for index in range(1, 501)}
    service, _chacha, _media_db, _pipeline = _service(sources, rows)

    snapshot = service.resolve_source_snapshot(mode="all")

    assert len(snapshot.items) == 500


@pytest.mark.parametrize(
    "mutation",
    [
        "source_removed",
        "source_remapped",
        "uuid_changed",
        "hash_changed",
        "deleted",
        "trashed",
        "readiness_lost",
    ],
)
def test_revalidation_detects_authorization_content_and_readiness_changes(
    mutation: str,
) -> None:
    service, chacha, media_db, _pipeline = _service(
        [_source("source-a", 1)],
        {1: _media(1), 2: _media(2)},
    )
    snapshot = service.resolve_source_snapshot(
        mode="include",
        source_ids=("source-a",),
    )
    if mutation == "source_removed":
        chacha.sources.clear()
    elif mutation == "source_remapped":
        chacha.sources[0]["media_id"] = 2
    elif mutation == "uuid_changed":
        media_db.rows[1]["uuid"] = "replacement-uuid"
    elif mutation == "hash_changed":
        media_db.rows[1]["content_hash"] = "replacement-hash"
    elif mutation == "deleted":
        media_db.rows[1]["deleted"] = 1
    elif mutation == "trashed":
        media_db.rows[1]["is_trash"] = 1
    elif mutation == "readiness_lost":
        media_db.rows[1]["content"] = ""

    with pytest.raises(SharedWorkspaceSourceChanged) as exc_info:
        service.revalidate_source_snapshot(snapshot=snapshot)

    assert exc_info.value.code == "shared_source_changed"
    assert "replacement" not in str(exc_info.value)


def test_include_revalidation_ignores_unrelated_and_non_authorization_source_changes() -> None:
    service, chacha, media_db, _pipeline = _service(
        [_source("source-a", 1, title="Original", position=1)],
        {1: _media(1), 2: _media(2)},
    )
    snapshot = service.resolve_source_snapshot(
        mode="include",
        source_ids=("source-a",),
    )
    chacha.sources[0].update(title="Renamed", position=999, selected=False)
    chacha.sources.append(_source("new-unrelated", 2))

    current = service.revalidate_source_snapshot(snapshot=snapshot)

    assert current == snapshot
    assert media_db.by_id_calls[-1] == (1, True, True)


def test_frozen_all_retry_reuses_original_ids_and_does_not_expand() -> None:
    service, chacha, _media_db, _pipeline = _service(
        [_source("source-a", 1)],
        {1: _media(1), 2: _media(2)},
    )
    original = service.resolve_source_snapshot(mode="all")
    chacha.sources.append(_source("source-new", 2))

    retry = service.resolve_source_snapshot(
        mode="all",
        frozen_source_ids=original.source_ids,
    )

    assert retry == original
    assert retry.source_ids == ("source-a",)


@pytest.mark.asyncio
async def test_duplicate_canonical_sources_retrieve_media_once_and_map_to_smallest_id() -> None:
    pipeline = _Pipeline(_result([_document(7, title="Shared report")]))
    service, _chacha, media_db, _pipeline = _service(
        [_source("source-z", 7), _source("source-a", 7)],
        {7: _media(7)},
        pipeline=pipeline,
    )
    snapshot = service.resolve_source_snapshot(mode="all")

    evidence = await service.retrieve_verified_evidence(
        query="What is supported?",
        snapshot=snapshot,
    )

    assert media_db.by_id_calls == [(7, True, True)]
    assert pipeline.calls[0]["include_media_ids"] == [7]
    assert evidence[0].source_id == "source-a"
    assert evidence[0].source_title == "Shared report"


@pytest.mark.asyncio
async def test_retrieval_call_is_media_only_owner_scoped_and_locked() -> None:
    pipeline = _Pipeline(_result([_document(1), _document(2, document_id="chunk-2")]))
    service, _chacha, media_db, _pipeline = _service(
        [_source("source-a", 1), _source("source-b", 2)],
        {1: _media(1), 2: _media(2), 999: _media(999, content="OWNER_SENTINEL")},
        pipeline=pipeline,
    )
    snapshot = service.resolve_source_snapshot(mode="all")

    await service.retrieve_verified_evidence(query="Question", snapshot=snapshot)

    call = pipeline.calls[0]
    assert call["query"] == "Question"
    assert call["sources"] == ["media_db"]
    assert call["include_media_ids"] == [1, 2]
    assert 999 not in call["include_media_ids"]
    assert call["index_namespace"] == "user_7_media_embeddings"
    assert call["media_db_path"] == "/private/owner/media.db"
    assert call["media_db"] is media_db
    assert call["notes_db_path"] is None
    assert call["chacha_db"] is None
    assert call["include_note_ids"] is None
    assert call["search_mode"] == "fts"
    assert call["fts_level"] == "chunk"
    assert call["top_k"] == 20
    assert call["reranking_strategy"] == "none"
    assert call["search_depth_mode"] is None
    assert call["rag_profile"] is None
    assert call["retrieval_plan"] is None
    assert call["resolved_request"] is None
    assert call["credential_runtime"] is None
    assert call["chat_history"] is None
    assert call["search_url_scraping"] is False
    assert call["fallback_on_error"] is False
    assert call["adaptive_cache"] is False
    for name, value in call.items():
        if name.startswith("enable_"):
            assert value is False, name


def test_signature_sentinel_pins_or_reviews_every_pipeline_parameter() -> None:
    parameters = inspect.signature(unified_rag_pipeline).parameters
    assert parameters["kwargs"].kind is inspect.Parameter.VAR_KEYWORD
    reviewed = (
        {"query", "kwargs"}
        | SHARED_RETRIEVAL_POLICY.pinned_parameter_names
        | SHARED_RETRIEVAL_POLICY.reviewed_inert_parameter_names
    )
    assert set(parameters) == reviewed

    security_sensitive = {
        name
        for name in parameters
        if name.startswith(("enable_", "fallback_", "adaptive_"))
        or name
        in {
            "sources",
            "media_db_path",
            "notes_db_path",
            "character_db_path",
            "kanban_db_path",
            "sql_target_id",
            "sql_retriever",
            "search_mode",
            "fts_level",
            "auto_temporal_filters",
            "expand_query",
            "spell_check",
            "include_media_ids",
            "include_note_ids",
            "reranking_strategy",
            "generation_model",
            "generation_provider",
            "generation_prompt",
            "index_namespace",
            "retrieval_plan",
            "resolved_request",
            "credential_runtime",
            "media_db",
            "chacha_db",
            "user_id",
            "session_id",
            "search_depth_mode",
            "chat_history",
            "discussion_platforms",
            "search_url_scraping",
            "research_progress_callback",
            "classifier_provider",
            "classifier_model",
            "rag_profile",
        }
    }
    assert security_sensitive <= SHARED_RETRIEVAL_POLICY.pinned_parameter_names
    assert "kwargs" not in inspect.signature(
        SharedWorkspaceChatService.retrieve_verified_evidence
    ).parameters


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline",
    [
        _Pipeline(error=RuntimeError("provider secret and owner path")),
        _Pipeline(_result([_document(1)], errors=["raw retriever error"])),
        _Pipeline(_result([_document(1)], generated_answer="unexpected answer")),
        _Pipeline(_result([_document(1)], metadata={"generation_executed": True})),
        _Pipeline(_result([_document(1)], metadata={"web_fallback": {"triggered": True}})),
        _Pipeline(_result([_document(1)], metadata={"sources_searched": ["media_db", "notes"]})),
        _Pipeline(_result([_document(1)], cache_hit=True)),
    ],
    ids=[
        "exception",
        "reported-error",
        "generated-answer",
        "generation-metadata",
        "external-metadata",
        "broadened-sources",
        "cache-hit",
    ],
)
async def test_pipeline_errors_generation_external_metadata_and_cache_fail_closed(
    pipeline: _Pipeline,
) -> None:
    service, _chacha, _media_db, _pipeline = _service(
        [_source("source-a", 1)],
        {1: _media(1)},
        pipeline=pipeline,
    )
    snapshot = service.resolve_source_snapshot(mode="all")

    with pytest.raises(SharedWorkspaceRetrievalUnavailable) as exc_info:
        await service.retrieve_verified_evidence(query="Question", snapshot=snapshot)

    assert exc_info.value.code == "retrieval_unavailable"
    assert "secret" not in str(exc_info.value)
    assert "/private" not in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "document",
    [
        _document(1, source=None),
        _document(1, source="notes"),
        _document("not-an-id"),
        _document(999),
    ],
    ids=["provenance-less", "non-media", "unparsable-media", "out-of-scope"],
)
async def test_any_unverified_document_rejects_the_complete_result(document: dict[str, Any]) -> None:
    pipeline = _Pipeline(_result([_document(1), document]))
    service, _chacha, _media_db, _pipeline = _service(
        [_source("source-a", 1)],
        {1: _media(1), 999: _media(999, content="OWNER_SENTINEL")},
        pipeline=pipeline,
    )
    snapshot = service.resolve_source_snapshot(mode="all")

    with pytest.raises(SharedWorkspaceRetrievalUnavailable):
        await service.retrieve_verified_evidence(query="Question", snapshot=snapshot)


@pytest.mark.asyncio
async def test_out_of_scope_document_after_retention_limit_still_rejects_full_result() -> None:
    documents = [
        _document(1, document_id=f"allowed-{index}", content=f"allowed {index}")
        for index in range(20)
    ]
    documents.append(_document(999, document_id="sentinel", content="OWNER_SENTINEL"))
    pipeline = _Pipeline(_result(documents))
    service, _chacha, _media_db, _pipeline = _service(
        [_source("source-a", 1)],
        {1: _media(1), 999: _media(999, content="OWNER_SENTINEL")},
        pipeline=pipeline,
    )
    snapshot = service.resolve_source_snapshot(mode="all")

    with pytest.raises(SharedWorkspaceRetrievalUnavailable):
        await service.retrieve_verified_evidence(query="Question", snapshot=snapshot)


@pytest.mark.asyncio
async def test_empty_or_contentless_retrieval_returns_stable_no_evidence_error() -> None:
    for documents in ([], [_document(1, content="   ")]):
        pipeline = _Pipeline(_result(documents))
        service, _chacha, _media_db, _pipeline = _service(
            [_source("source-a", 1)],
            {1: _media(1)},
            pipeline=pipeline,
        )
        snapshot = service.resolve_source_snapshot(mode="all")

        with pytest.raises(SharedWorkspaceNoRelevantEvidence) as exc_info:
            await service.retrieve_verified_evidence(query="Question", snapshot=snapshot)

        assert exc_info.value.code == "no_relevant_evidence"


@pytest.mark.asyncio
async def test_evidence_is_deduplicated_deterministically_labeled_and_bounded() -> None:
    documents = [
        _document(
            1,
            document_id=f"chunk-{index:02d}",
            content=(f"evidence-{index:02d} " + ("x" * 680)),
            score=1.0 - (index / 100),
            chunk_index=index,
            start_char=index * 100,
            end_char=(index * 100) + 700,
            title="T" * 700,
        )
        for index in range(25)
    ]
    documents.insert(1, dict(documents[0]))
    documents[1]["metadata"] = dict(documents[0]["metadata"])
    pipeline = _Pipeline(_result(documents))
    service, _chacha, _media_db, _pipeline = _service(
        [_source("source-a", 1)],
        {1: _media(1)},
        pipeline=pipeline,
    )
    snapshot = service.resolve_source_snapshot(mode="all")

    evidence = await service.retrieve_verified_evidence(query="Question", snapshot=snapshot)

    assert len(evidence) == 20
    assert [item.label for item in evidence] == [f"E{index}" for index in range(1, 21)]
    assert len({(item.source_id, item.chunk_index, item.content) for item in evidence}) == 20
    assert all(len(item.content) <= 1_000 for item in evidence)
    assert sum(len(item.content) for item in evidence) <= 16_000
    assert all(len(item.source_title) <= 512 for item in evidence)
    assert all(math.isfinite(item.score) for item in evidence)


@pytest.mark.asyncio
async def test_invalid_snapshot_or_nonfinite_score_fails_before_evidence_escape() -> None:
    pipeline = _Pipeline(_result([_document(1, score=math.inf)]))
    service, _chacha, _media_db, _pipeline = _service(
        [_source("source-a", 1)],
        {1: _media(1)},
        pipeline=pipeline,
    )
    snapshot = service.resolve_source_snapshot(mode="all")

    with pytest.raises(SharedWorkspaceRetrievalUnavailable):
        await service.retrieve_verified_evidence(query="Question", snapshot=snapshot)

    forged = SharedSourceSnapshot(
        mode="include",
        items=(
            SharedSourceSnapshotItem(
                source_id="",
                media_id=1,
                media_uuid="uuid",
                content_hash="hash",
                readiness_class="queryable",
            ),
        ),
        snapshot_hash="forged",
    )
    with pytest.raises(SharedWorkspaceRetrievalUnavailable):
        await service.retrieve_verified_evidence(query="Question", snapshot=forged)
