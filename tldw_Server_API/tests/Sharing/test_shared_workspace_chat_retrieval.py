"""Security contracts for recipient retrieval over shared workspace sources."""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import textwrap
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGResponse
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import (
    _serialize_result_document,
    unified_rag_pipeline,
)
from tldw_Server_API.app.core.Sharing.shared_workspace_chat_service import (
    SHARED_RETRIEVAL_POLICY,
    SharedSourceSnapshot,
    SharedSourceSnapshotItem,
    SharedWorkspaceChatService,
    SharedWorkspaceChatServiceError,
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


class _ControlledRealRetriever:
    calls: list[dict[str, Any]] = []

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.retrievers = {DataSource.MEDIA_DB: object()}

    async def retrieve_from_plan(self, _plan: Any, **kwargs: Any) -> list[Document]:
        type(self).calls.append(kwargs)
        return [
            Document(
                id="chunk-real-1",
                content="real pipeline evidence",
                source=DataSource.MEDIA_DB,
                score=0.91,
                metadata={
                    "source": "media_db",
                    "media_id": 1,
                    "chunk_id": "chunk-real-1",
                    "chunk_index": 1,
                    "start_char": 0,
                    "end_char": 22,
                },
            )
        ]


def _result(
    documents: list[Any],
    *,
    query: str = "Question",
    expanded_queries: list[str] | None = None,
    errors: list[str] | None = None,
    generated_answer: Any = None,
    metadata: dict[str, Any] | None = None,
    cache_hit: bool = False,
    citations: list[dict[str, Any]] | None = None,
    academic_citations: list[str] | None = None,
    chunk_citations: list[dict[str, Any]] | None = None,
    feedback_id: str | None = None,
    security_report: dict[str, Any] | None = None,
    claims: list[dict[str, Any]] | None = None,
    factuality: dict[str, Any] | None = None,
    verification_report: dict[str, Any] | None = None,
    retrieval_metrics: dict[str, Any] | None = None,
    faithfulness: dict[str, Any] | None = None,
    query_classification: dict[str, Any] | None = None,
    reformulated_query: str | None = None,
    research_summary: dict[str, Any] | None = None,
    suggestions: list[str] | None = None,
    images: list[dict[str, Any]] | None = None,
    videos: list[dict[str, Any]] | None = None,
) -> UnifiedRAGResponse:
    return UnifiedRAGResponse(
        documents=documents,
        query=query,
        expanded_queries=expanded_queries or [],
        timings={},
        citations=citations or [],
        academic_citations=academic_citations or [],
        chunk_citations=chunk_citations or [],
        feedback_id=feedback_id,
        errors=errors or [],
        generated_answer=generated_answer,
        metadata=metadata or {},
        cache_hit=cache_hit,
        security_report=security_report,
        claims=claims,
        factuality=factuality,
        verification_report=verification_report,
        retrieval_metrics=retrieval_metrics,
        faithfulness=faithfulness,
        query_classification=query_classification,
        reformulated_query=reformulated_query,
        research_summary=research_summary,
        suggestions=suggestions,
        images=images,
        videos=videos,
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
    document = {
        "id": document_id,
        "content": content,
        "score": score,
        "metadata": metadata,
    }
    if source is not None:
        document["source"] = source
        metadata["source"] = source
    return document


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
    "sources",
    [
        [_source(" source-a", 1)],
        [_source("source-a", 1), _source("source-a", 2)],
    ],
    ids=["noncanonical-source-id", "duplicate-source-id"],
)
def test_all_maps_malformed_authoritative_rows_to_sanitized_data_unavailable(
    sources: list[dict[str, Any]],
) -> None:
    service, _chacha, _media_db, _pipeline = _service(
        sources,
        {1: _media(1), 2: _media(2)},
    )

    with pytest.raises(SharedWorkspaceChatServiceError) as exc_info:
        service.resolve_source_snapshot(mode="all")

    assert exc_info.value.code == "shared_workspace_unavailable"
    assert type(exc_info.value) is not ValueError


@pytest.mark.parametrize("include_valid_source", [False, True], ids=["alone", "mixed"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("media_id", 0),
        ("uuid", " media-uuid-1"),
        ("uuid", "media\x00uuid"),
        ("uuid", "u" * 513),
        ("content_hash", " sha256-1"),
        ("content_hash", "sha256\x00hash"),
        ("content_hash", "h" * 513),
    ],
    ids=[
        "invalid-media-id",
        "uuid-whitespace",
        "uuid-nonprintable",
        "uuid-oversized",
        "hash-whitespace",
        "hash-nonprintable",
        "hash-oversized",
    ],
)
def test_all_rejects_malformed_authoritative_media_identity_as_storage_shape(
    field: str,
    value: Any,
    include_valid_source: bool,
) -> None:
    malformed_source = _source("source-bad", 1)
    malformed_media = _media(1)
    if field == "media_id":
        malformed_source["media_id"] = value
    else:
        malformed_media[field] = value
    sources = [malformed_source]
    rows = {1: malformed_media}
    if include_valid_source:
        sources.append(_source("source-valid", 2))
        rows[2] = _media(2)
    service, _chacha, _media_db, _pipeline = _service(sources, rows)

    with pytest.raises(SharedWorkspaceChatServiceError) as exc_info:
        service.resolve_source_snapshot(mode="all")

    assert exc_info.value.code == "shared_workspace_unavailable"
    assert type(exc_info.value) is not ValueError


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("media_id", 0),
        ("uuid", "media-uuid-1 "),
        ("content_hash", "sha256\x00hash"),
    ],
)
def test_include_maps_malformed_authoritative_media_identity_to_invalid_request(
    field: str,
    value: Any,
) -> None:
    source = _source("source-a", 1)
    media = _media(1)
    if field == "media_id":
        source["media_id"] = value
    else:
        media[field] = value
    service, _chacha, _media_db, _pipeline = _service([source], {1: media})

    with pytest.raises(SharedWorkspaceSourceScopeInvalid):
        service.resolve_source_snapshot(mode="include", source_ids=("source-a",))


@pytest.mark.parametrize("field", ["uuid", "content_hash"])
def test_initial_snapshot_rejects_noncanonical_exact_media_identity(field: str) -> None:
    media = _media(1)
    media[field] = f" {media[field]}"
    service, _chacha, _media_db, _pipeline = _service(
        [_source("source-a", 1)],
        {1: media},
    )

    with pytest.raises(SharedWorkspaceSourceScopeInvalid):
        service.resolve_source_snapshot(mode="include", source_ids=("source-a",))


def test_requested_source_ids_must_be_exact_canonical_values() -> None:
    service, _chacha, _media_db, _pipeline = _service(
        [_source("source-a", 1)],
        {1: _media(1)},
    )

    with pytest.raises(SharedWorkspaceSourceScopeInvalid):
        service.resolve_source_snapshot(mode="include", source_ids=(" source-a",))


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


@pytest.mark.parametrize("field", ["uuid", "content_hash"])
def test_revalidation_rejects_noncanonical_identity_whitespace(field: str) -> None:
    service, _chacha, media_db, _pipeline = _service(
        [_source("source-a", 1)],
        {1: _media(1)},
    )
    snapshot = service.resolve_source_snapshot(mode="all")
    media_db.rows[1][field] = f"{media_db.rows[1][field]} "

    with pytest.raises(SharedWorkspaceSourceChanged):
        service.revalidate_source_snapshot(snapshot=snapshot)


@pytest.mark.parametrize(
    ("field", "value"),
    [("media_id", 0), ("uuid", "media\x00uuid"), ("content_hash", "h" * 513)],
)
def test_revalidation_maps_malformed_identity_shape_to_source_changed(
    field: str,
    value: Any,
) -> None:
    service, chacha, media_db, _pipeline = _service(
        [_source("source-a", 1)],
        {1: _media(1)},
    )
    snapshot = service.resolve_source_snapshot(mode="all")
    if field == "media_id":
        chacha.sources[0][field] = value
    else:
        media_db.rows[1][field] = value

    with pytest.raises(SharedWorkspaceSourceChanged):
        service.revalidate_source_snapshot(snapshot=snapshot)


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
    pipeline = _Pipeline(
        _result(
            [_document(7, title="Spoofed RAG title")],
            query="What is supported?",
        )
    )
    service, _chacha, media_db, _pipeline = _service(
        [
            _source("source-z", 7, title="Alias Z"),
            _source("source-a", 7, title="Canonical A"),
        ],
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
    assert evidence[0].source_title == "Canonical A"


@pytest.mark.asyncio
async def test_evidence_title_uses_current_authoritative_source_without_changing_hash() -> None:
    pipeline = _Pipeline(_result([_document(1, title="Spoofed title")]))
    service, chacha, _media_db, _pipeline = _service(
        [_source("source-a", 1, title="Original title")],
        {1: _media(1)},
        pipeline=pipeline,
    )
    snapshot = service.resolve_source_snapshot(mode="all")
    original_hash = snapshot.snapshot_hash
    chacha.sources[0]["title"] = "Current authoritative title"

    evidence = await service.retrieve_verified_evidence(query="Question", snapshot=snapshot)

    assert snapshot.snapshot_hash == original_hash
    assert evidence[0].source_title == "Current authoritative title"


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
    assert call["user_id"] == "7"
    assert call["media_db_path"] == "/private/owner/media.db"
    assert call["media_db"] is media_db
    assert call["notes_db_path"] is None
    assert call["chacha_db"] is None
    assert call["include_note_ids"] is None
    assert call["search_mode"] == "fts"
    assert call["fts_level"] == "chunk"
    assert call["top_k"] == 20
    assert call["min_score"] == 0.0
    assert call["chunk_type_filter"] is None
    assert call["ocr_confidence_threshold"] is None
    assert call["timeout_seconds"] is None
    assert call["include_retrieval_diagnostics"] is False
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


def _literal_outer_pipeline_kwargs_reads(source: str | None = None) -> set[str]:
    tree = ast.parse(
        textwrap.dedent(source or inspect.getsource(unified_rag_pipeline))
    )
    root = tree.body[0]
    assert isinstance(root, ast.AsyncFunctionDef)
    keys: set[str] = set()
    parents = {
        child: parent
        for parent in ast.walk(root)
        for child in ast.iter_child_nodes(parent)
    }

    class KwargsReadVisitor(ast.NodeVisitor):
        @staticmethod
        def _bound_argument_names(arguments: ast.arguments) -> set[str]:
            names = {
                argument.arg
                for argument in (
                    *arguments.posonlyargs,
                    *arguments.args,
                    *arguments.kwonlyargs,
                )
            }
            if arguments.vararg is not None:
                names.add(arguments.vararg.arg)
            if arguments.kwarg is not None:
                names.add(arguments.kwarg.arg)
            return names

        def _visit_argument_header(self, arguments: ast.arguments) -> None:
            declared_arguments = [
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
            ]
            if arguments.vararg is not None:
                declared_arguments.append(arguments.vararg)
            if arguments.kwarg is not None:
                declared_arguments.append(arguments.kwarg)
            for argument in declared_arguments:
                if argument.annotation is not None:
                    self.visit(argument.annotation)
            for default in (*arguments.defaults, *arguments.kw_defaults):
                if default is not None:
                    self.visit(default)

        def _visit_function_header(
            self,
            node: ast.FunctionDef | ast.AsyncFunctionDef,
        ) -> None:
            for decorator in node.decorator_list:
                self.visit(decorator)
            for type_parameter in getattr(node, "type_params", ()):
                self.visit(type_parameter)
            self._visit_argument_header(node.args)
            if node.returns is not None:
                self.visit(node.returns)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if "kwargs" in self._bound_argument_names(node.args):
                self._visit_function_header(node)
            else:
                self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Lambda(self, node: ast.Lambda) -> None:
            if "kwargs" in self._bound_argument_names(node.args):
                self._visit_argument_header(node.args)
            else:
                self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> None:
            if (
                node.id != "kwargs"
                or not isinstance(node.ctx, ast.Load)
            ):
                return

            parent = parents.get(node)
            if (
                isinstance(parent, ast.Subscript)
                and parent.value is node
                and isinstance(parent.ctx, ast.Load)
                and isinstance(parent.slice, ast.Constant)
                and isinstance(parent.slice.value, str)
            ):
                keys.add(parent.slice.value)
                return

            grandparent = parents.get(parent) if parent is not None else None
            if (
                isinstance(parent, ast.Attribute)
                and parent.value is node
                and parent.attr in {"get", "pop", "setdefault"}
                and isinstance(grandparent, ast.Call)
                and grandparent.func is parent
                and grandparent.args
                and isinstance(grandparent.args[0], ast.Constant)
                and isinstance(grandparent.args[0].value, str)
            ):
                keys.add(grandparent.args[0].value)
                return

            raise AssertionError(
                f"unapproved outer kwargs load at line {node.lineno}"
            )

    visitor = KwargsReadVisitor()
    for statement in root.body:
        visitor.visit(statement)
    return keys


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


def test_hidden_pipeline_kwargs_reads_are_explicitly_reviewed_absent() -> None:
    expected_absent = {
        "metadata",
        "workspace_id",
        "prompts_db_path",
        "world_books_db_path",
        "chat_dictionaries_db_path",
        "include_sources",
        "include_metadata",
        "prompts_db",
        "enable_expansion",
        "claims_budget_usd",
        "claims_budget_tokens",
        "claims_budget_strict",
        "faithfulness_llm",
    }

    assert _literal_outer_pipeline_kwargs_reads() == expected_absent
    assert SHARED_RETRIEVAL_POLICY.reviewed_absent_kwarg_names == expected_absent
    assert not (
        SHARED_RETRIEVAL_POLICY.reviewed_absent_kwarg_names
        & SHARED_RETRIEVAL_POLICY.pinned_parameter_names
    )


@pytest.mark.parametrize(
    "body",
    [
        "options = kwargs",
        "key = 'metadata'\nvalue = kwargs.get(key)",
        "key = 'metadata'\nvalue = kwargs[key]",
        "values = list(kwargs)",
        "present = 'metadata' in kwargs",
        "enabled = bool(kwargs)",
        "values = {**kwargs}",
        "reader = kwargs.get\nvalue = reader('metadata')",
    ],
    ids=[
        "alias",
        "dynamic-method-key",
        "dynamic-subscript-key",
        "iteration",
        "membership",
        "truthiness",
        "unpacking",
        "indirect-method",
    ],
)
def test_outer_kwargs_analyzer_rejects_every_unapproved_load(body: str) -> None:
    source = "async def unified_rag_pipeline(**kwargs):\n" + textwrap.indent(
        body,
        "    ",
    )

    with pytest.raises(AssertionError, match="unapproved outer kwargs load"):
        _literal_outer_pipeline_kwargs_reads(source)


def test_outer_kwargs_analyzer_ignores_nested_function_kwargs() -> None:
    source = """
        async def unified_rag_pipeline(**kwargs):
            value = kwargs.get("metadata")
            def nested(**kwargs):
                options = kwargs
                return kwargs.get(dynamic_key)
            return value, nested
    """

    assert _literal_outer_pipeline_kwargs_reads(source) == {"metadata"}


@pytest.mark.parametrize(
    "nested_callable",
    [
        "def nested(value=kwargs, **kwargs):\n    return None",
        "def nested(*, value=kwargs, **kwargs):\n    return None",
        "@decorate(options=kwargs)\ndef nested(**kwargs):\n    return None",
        "def nested(value: kwargs, **kwargs):\n    return None",
        "def nested(**kwargs) -> kwargs:\n    return None",
        "nested = lambda value=kwargs, **kwargs: None",
    ],
    ids=[
        "function-positional-default",
        "function-keyword-default",
        "function-decorator",
        "function-parameter-annotation",
        "function-return-annotation",
        "lambda-default",
    ],
)
def test_outer_kwargs_analyzer_rejects_nested_callable_header_loads(
    nested_callable: str,
) -> None:
    source = "async def unified_rag_pipeline(**kwargs):\n" + textwrap.indent(
        nested_callable,
        "    ",
    )

    with pytest.raises(AssertionError, match="unapproved outer kwargs load"):
        _literal_outer_pipeline_kwargs_reads(source)


def test_outer_kwargs_analyzer_ignores_independently_bound_nested_body() -> None:
    source = """
        async def unified_rag_pipeline(**kwargs):
            def nested(**kwargs):
                options = kwargs
                return kwargs.get(dynamic_key)
            return nested
    """

    assert _literal_outer_pipeline_kwargs_reads(source) == set()


@pytest.mark.parametrize(
    "nested_callable",
    [
        'def nested(value=kwargs.get("metadata"), **kwargs):\n    return None',
        '@decorate(kwargs["metadata"])\ndef nested(**kwargs):\n    return None',
        'nested = lambda value=kwargs.get("metadata"), **kwargs: None',
    ],
    ids=["function-default", "function-decorator", "lambda-default"],
)
def test_outer_kwargs_analyzer_collects_approved_nested_header_reads(
    nested_callable: str,
) -> None:
    source = "async def unified_rag_pipeline(**kwargs):\n" + textwrap.indent(
        nested_callable,
        "    ",
    )

    assert _literal_outer_pipeline_kwargs_reads(source) == {"metadata"}


@pytest.mark.asyncio
async def test_actual_notes_document_cannot_spoof_media_provenance_through_metadata() -> None:
    serialized = _serialize_result_document(
        Document(
            id="note-chunk-1",
            content="OWNER_NOTE_SENTINEL",
            source=DataSource.NOTES,
            score=0.9,
            metadata={"source": "media_db", "media_id": 1},
        )
    )
    pipeline = _Pipeline(_result([serialized]))
    service, _chacha, _media_db, _pipeline = _service(
        [_source("source-a", 1)],
        {1: _media(1)},
        pipeline=pipeline,
    )
    snapshot = service.resolve_source_snapshot(mode="all")

    with pytest.raises(SharedWorkspaceRetrievalUnavailable):
        await service.retrieve_verified_evidence(query="Question", snapshot=snapshot)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "document",
    [
        {
            "id": "chunk-1",
            "content": "Metadata-only marker",
            "score": 0.8,
            "metadata": {"source": "media_db", "media_id": 1, "chunk_id": "chunk-1"},
        },
        {
            "id": "chunk-1",
            "content": "Conflicting marker",
            "score": 0.8,
            "source": "media_db",
            "metadata": {"source": "notes", "media_id": 1, "chunk_id": "chunk-1"},
        },
    ],
    ids=["missing-top-level-source", "conflicting-source-markers"],
)
async def test_shared_retrieval_requires_consistent_authoritative_top_level_source(
    document: dict[str, Any],
) -> None:
    pipeline = _Pipeline(_result([document]))
    service, _chacha, _media_db, _pipeline = _service(
        [_source("source-a", 1)],
        {1: _media(1)},
        pipeline=pipeline,
    )
    snapshot = service.resolve_source_snapshot(mode="all")

    with pytest.raises(SharedWorkspaceRetrievalUnavailable):
        await service.retrieve_verified_evidence(query="Question", snapshot=snapshot)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response_changes",
    [
        {"query": "transformed query"},
        {"expanded_queries": ["expanded"]},
        {"generated_answer": {}},
        {"citations": [{"id": "citation"}]},
        {"academic_citations": ["citation"]},
        {"chunk_citations": [{"id": "chunk-citation"}]},
        {"feedback_id": "feedback-1"},
        {"security_report": {"pii_detected": False}},
        {"claims": [{"claim": "provider output"}]},
        {"factuality": {"score": 1.0}},
        {"verification_report": {"verified": True}},
        {"retrieval_metrics": {"precision": 1.0}},
        {"faithfulness": {"score": 1.0}},
        {"query_classification": {"intent": "external"}},
        {"reformulated_query": "rewritten"},
        {"research_summary": {"iterations": 1}},
        {"suggestions": ["Follow up"]},
        {"images": [{"url": "https://example.test/image"}]},
        {"videos": [{"url": "https://example.test/video"}]},
        {"errors": ["raw error"]},
        {"cache_hit": True},
    ],
    ids=[
        "query-mismatch",
        "expanded-query",
        "dict-answer",
        "citation",
        "academic-citation",
        "chunk-citation",
        "feedback",
        "security-report",
        "claims",
        "factuality",
        "verification",
        "retrieval-metrics",
        "faithfulness",
        "classification",
        "reformulation",
        "research",
        "suggestions",
        "images",
        "videos",
        "errors",
        "cache",
    ],
)
async def test_actual_unified_response_rejects_every_non_retrieval_output(
    response_changes: dict[str, Any],
) -> None:
    pipeline = _Pipeline(_result([_document(1)], **response_changes))
    service, _chacha, _media_db, _pipeline = _service(
        [_source("source-a", 1)],
        {1: _media(1)},
        pipeline=pipeline,
    )
    snapshot = service.resolve_source_snapshot(mode="all")

    with pytest.raises(SharedWorkspaceRetrievalUnavailable):
        await service.retrieve_verified_evidence(query="Question", snapshot=snapshot)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata",
    [
        {"provider_output": {"model": "external"}},
        {"original_query": "different"},
        {"sources_requested": ["media_db", "notes"]},
        {"generation_executed": "false"},
    ],
    ids=["unknown-key", "query-mismatch", "broadened-source", "non-boolean-flag"],
)
async def test_metadata_is_allowlisted_and_validated_fail_closed(
    metadata: dict[str, Any],
) -> None:
    pipeline = _Pipeline(_result([_document(1)], metadata=metadata))
    service, _chacha, _media_db, _pipeline = _service(
        [_source("source-a", 1)],
        {1: _media(1)},
        pipeline=pipeline,
    )
    snapshot = service.resolve_source_snapshot(mode="all")

    with pytest.raises(SharedWorkspaceRetrievalUnavailable):
        await service.retrieve_verified_evidence(query="Question", snapshot=snapshot)


@pytest.mark.asyncio
async def test_actual_pipeline_metadata_allowlist_accepts_locked_media_only_shape() -> None:
    metadata = {
        "original_query": "Question",
        "retrieval_cache_hit": False,
        "generation_executed": False,
        "explicit_source_selection": {
            "enabled": True,
            "requested_sources": ["media_db"],
            "resolved_sources": ["media_db"],
            "include_media_ids_count": 1,
            "include_note_ids_count": 0,
            "scope_intersection_empty": False,
            "cache_disabled": False,
        },
        "sources_requested": ["media_db"],
        "sources_searched": ["media_db"],
        "documents_retrieved": 1,
        "retrieval_guidance": "Internal retrieval guidance",
        "retrieval_plan": {
            "query": "Question",
            "sources": ["media_db"],
            "search_mode": "fts",
            "top_k": 20,
            "index_namespace": "user_7_media_embeddings",
        },
    }
    pipeline = _Pipeline(_result([_document(1)], metadata=metadata))
    service, _chacha, _media_db, _pipeline = _service(
        [_source("source-a", 1, title="Canonical")],
        {1: _media(1)},
        pipeline=pipeline,
    )
    snapshot = service.resolve_source_snapshot(mode="all")

    evidence = await service.retrieve_verified_evidence(query="Question", snapshot=snapshot)

    assert evidence[0].source_title == "Canonical"


@pytest.mark.asyncio
async def test_real_pipeline_shared_retrieval_disables_normal_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline_module = inspect.getmodule(unified_rag_pipeline)
    assert pipeline_module is not None
    _ControlledRealRetriever.calls.clear()
    monkeypatch.setattr(
        pipeline_module,
        "MultiDatabaseRetriever",
        _ControlledRealRetriever,
    )
    service, _chacha, owner_media, _pipeline = _service(
        [_source("source-a", 1, title="Canonical")],
        {1: _media(1)},
        pipeline=unified_rag_pipeline,
    )
    snapshot = service.resolve_source_snapshot(mode="all")

    evidence = await service.retrieve_verified_evidence(
        query="Question",
        snapshot=snapshot,
    )

    assert evidence[0].content == "real pipeline evidence"
    assert _ControlledRealRetriever.calls[0]["allowed_media_ids"] == [1]

    ordinary_call = SHARED_RETRIEVAL_POLICY.build_call(
        media_ids=(1,),
        media_db_path="/private/owner/media.db",
        media_db=owner_media,
        owner_user_id=7,
    )
    ordinary_call.pop("include_retrieval_diagnostics", None)
    ordinary_result = await unified_rag_pipeline(query="Question", **ordinary_call)

    assert {
        "profile_resolution",
        "source_status",
        "why_these_sources",
    } <= set(ordinary_result.metadata)


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
@pytest.mark.parametrize(
    "document",
    [
        _document(1, document_id=""),
        _document(1, document_id=" chunk-1"),
        _document(1, document_id=["chunk-1"]),
        _document(1, document_id="x" * 513),
        _document(1, document_id="chunk-1", chunk_index=" 1"),
        _document(1, document_id="chunk-1", start_char=-1),
        _document(1, document_id="chunk-1", end_char=2_147_483_648),
        _document(1, document_id="chunk-1", start_char=20, end_char=10),
    ],
    ids=[
        "empty-identity",
        "noncanonical-identity",
        "nonscalar-identity",
        "oversized-identity",
        "noncanonical-chunk-index",
        "negative-start",
        "oversized-end",
        "reversed-range",
    ],
)
async def test_malformed_chunk_identity_or_locator_rejects_complete_result(
    document: dict[str, Any],
) -> None:
    pipeline = _Pipeline(_result([document]))
    service, _chacha, _media_db, _pipeline = _service(
        [_source("source-a", 1)],
        {1: _media(1)},
        pipeline=pipeline,
    )
    snapshot = service.resolve_source_snapshot(mode="all")

    with pytest.raises(SharedWorkspaceRetrievalUnavailable):
        await service.retrieve_verified_evidence(query="Question", snapshot=snapshot)


@pytest.mark.asyncio
async def test_contentless_document_still_requires_valid_locators() -> None:
    pipeline = _Pipeline(
        _result([_document(1, content="   ", start_char="invalid")])
    )
    service, _chacha, _media_db, _pipeline = _service(
        [_source("source-a", 1)],
        {1: _media(1)},
        pipeline=pipeline,
    )
    snapshot = service.resolve_source_snapshot(mode="all")

    with pytest.raises(SharedWorkspaceRetrievalUnavailable):
        await service.retrieve_verified_evidence(query="Question", snapshot=snapshot)


@pytest.mark.asyncio
async def test_conflicting_top_level_and_metadata_chunk_identity_fails_closed() -> None:
    document = _document(1, document_id="chunk-top")
    document["metadata"]["chunk_id"] = "chunk-metadata"
    pipeline = _Pipeline(_result([document]))
    service, _chacha, _media_db, _pipeline = _service(
        [_source("source-a", 1)],
        {1: _media(1)},
        pipeline=pipeline,
    )
    snapshot = service.resolve_source_snapshot(mode="all")

    with pytest.raises(SharedWorkspaceRetrievalUnavailable):
        await service.retrieve_verified_evidence(query="Question", snapshot=snapshot)


@pytest.mark.asyncio
async def test_serialized_document_chunk_identity_conflict_fails_closed() -> None:
    serialized = _serialize_result_document(
        Document(
            id="top-level-chunk",
            content="Conflicting identity",
            source=DataSource.MEDIA_DB,
            score=0.8,
            metadata={
                "source": "media_db",
                "media_id": 1,
                "chunk_id": "metadata-chunk",
                "chunk_index": 1,
                "start_char": 0,
                "end_char": 20,
            },
        )
    )
    pipeline = _Pipeline(_result([serialized]))
    service, _chacha, _media_db, _pipeline = _service(
        [_source("source-a", 1)],
        {1: _media(1)},
        pipeline=pipeline,
    )
    snapshot = service.resolve_source_snapshot(mode="all")

    with pytest.raises(SharedWorkspaceRetrievalUnavailable):
        await service.retrieve_verified_evidence(query="Question", snapshot=snapshot)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed_field",
    ["content", "content_whitespace", "score", "chunk_index", "start_char", "end_char"],
)
async def test_conflicting_duplicate_chunk_records_reject_complete_result(
    changed_field: str,
) -> None:
    original = _document(
        1,
        document_id="chunk-identity",
        content="original",
        score=0.8,
        chunk_index=1,
        start_char=10,
        end_char=18,
    )
    conflicting = {**original, "metadata": dict(original["metadata"])}
    if changed_field in {"content", "content_whitespace", "score"}:
        if changed_field == "score":
            conflicting["score"] = 0.7
        elif changed_field == "content_whitespace":
            conflicting["content"] = " original"
        else:
            conflicting["content"] = "changed"
    else:
        conflicting["metadata"][changed_field] = {
            "chunk_index": 2,
            "start_char": 11,
            "end_char": 19,
        }[changed_field]
    documents = [
        _document(1, document_id=f"allowed-{index}", content=f"allowed {index}")
        for index in range(20)
    ]
    documents.extend([original, conflicting])
    pipeline = _Pipeline(_result(documents))
    service, _chacha, _media_db, _pipeline = _service(
        [_source("source-a", 1)],
        {1: _media(1)},
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
    assert all(len(item.content) <= 4_000 for item in evidence)
    assert sum(len(item.content) for item in evidence) <= 48_000
    assert all(len(item.source_title) <= 512 for item in evidence)
    assert all(math.isfinite(item.score) for item in evidence)


@pytest.mark.asyncio
async def test_evidence_capacity_is_4000_per_item_and_48000_aggregate() -> None:
    documents = [
        _document(
            1,
            document_id=f"chunk-{index:02d}",
            content=str(index) + ("x" * 4_999),
            chunk_index=index,
            start_char=index * 5_000,
            end_char=(index + 1) * 5_000,
        )
        for index in range(13)
    ]
    pipeline = _Pipeline(_result(documents))
    service, _chacha, _media_db, _pipeline = _service(
        [_source("source-a", 1)],
        {1: _media(1)},
        pipeline=pipeline,
    )
    snapshot = service.resolve_source_snapshot(mode="all")

    evidence = await service.retrieve_verified_evidence(query="Question", snapshot=snapshot)

    assert len(evidence) == 12
    assert all(len(item.content) == 4_000 for item in evidence)
    assert sum(len(item.content) for item in evidence) == 48_000
    assert [item.label for item in evidence] == [f"E{index}" for index in range(1, 13)]


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
