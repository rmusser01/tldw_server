from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from loguru import logger

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.RAG.exceptions import RAGDatabaseError
from tldw_Server_API.app.core.RAG.rag_service import (
    database_retrievers,
    unified_pipeline,
)
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document


def _document(source: DataSource, document_id: str, score: float) -> Document:
    return Document(
        id=document_id,
        content=f"# {document_id}\n\nbody",
        metadata={"_standalone_source_preformatted": True},
        source=source,
        score=score,
    )


def _two_phase_retriever(source: DataSource, documents: list[Document]):
    candidates = []
    for document in documents:
        candidates.append(
            Document(
                id=document.id,
                content="",
                metadata={
                    **document.metadata,
                    "_standalone_source_full_chars": len(document.content),
                    "_standalone_source_projection_key": document.id,
                    "_test_full_content": document.content,
                },
                source=document.source,
                score=document.score,
            )
        )

    def project(*, projections, owner_user_id):
        assert owner_user_id == "owner-1"
        projected = []
        for candidate, char_cap in projections:
            full_content = candidate.metadata["_test_full_content"]
            metadata = {key: value for key, value in candidate.metadata.items() if not key.startswith("_test_")}
            if len(full_content) > char_cap:
                metadata["_standalone_source_projection_truncated"] = True
            projected.append(
                Document(
                    id=candidate.id,
                    content=full_content[:char_cap],
                    metadata=metadata,
                    source=source,
                    score=candidate.score,
                )
            )
        return projected

    def legacy_project(*, max_source_chars, top_k, **_):
        selected = documents[:top_k]
        if not selected:
            return []
        per_document_chars = max(1, max_source_chars // len(selected))
        projected = []
        for document in selected:
            metadata = dict(document.metadata)
            if len(document.content) > per_document_chars:
                metadata["_standalone_source_projection_truncated"] = True
            projected.append(
                Document(
                    id=document.id,
                    content=document.content[:per_document_chars],
                    metadata=metadata,
                    source=document.source,
                    score=document.score,
                )
            )
        return projected

    retriever = MagicMock()
    retriever.retrieve_slides_source_documents_v1 = AsyncMock(side_effect=legacy_project)
    retriever.retrieve_slides_source_candidates_v1 = AsyncMock(return_value=candidates)
    retriever.project_slides_source_documents_v1 = AsyncMock(side_effect=project)
    return retriever


def _patch_two_phase_retrievers(monkeypatch, source_documents):
    retrievers = {}
    for source_name, class_name, source in (
        ("media", "MediaDBRetriever", DataSource.MEDIA_DB),
        ("notes", "NotesDBRetriever", DataSource.NOTES),
        ("chats", "ChatHistoryRetriever", DataSource.CHAT_HISTORY),
    ):
        retriever = _two_phase_retriever(source, source_documents[source_name])
        retrievers[source_name] = retriever
        monkeypatch.setattr(unified_pipeline, class_name, MagicMock(return_value=retriever))
    return retrievers


class _RowCursor:
    description = None

    def __init__(self, rows):
        self._rows = iter(rows)

    def fetchone(self):
        return next(self._rows, None)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("C++", '"C"'),
        ("owner ... evidence", '"owner" "evidence"'),
        ('100% _ "quoted"', '"100" "quoted"'),
        ("% _ +", None),
    ],
)
def test_locked_sqlite_fts_query_treats_input_as_natural_language(query, expected):
    assert database_retrievers._build_slides_source_fts_query(query) == expected


@pytest.mark.unit
def test_candidate_reader_requires_explicit_invalid_text_marker():
    cursor = _RowCursor(
        [
            {
                "_standalone_source_full_chars": 10,
                "_standalone_source_projection_key": "source-1",
            }
        ]
    )

    with pytest.raises(ValueError, match="validation marker"):
        database_retrievers._read_slides_source_candidate_rows(cursor)


@pytest.mark.unit
@pytest.mark.parametrize(
    "missing_marker",
    (
        "_standalone_source_invalid_text",
        "_standalone_source_projection_truncated",
    ),
)
def test_projection_reader_requires_explicit_validation_markers(missing_marker):
    row = {
        "projection_cap": 10,
        "source_text": "bounded",
        "_standalone_source_invalid_text": False,
        "_standalone_source_projection_truncated": False,
    }
    row.pop(missing_marker)

    with pytest.raises(ValueError, match="marker"):
        database_retrievers._read_slides_source_projection_rows(
            _RowCursor([row]),
            max_materialized_chars=10,
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_locked_media_sqlite_fts_accepts_cpp_query_without_grammar_error():
    db = MediaDatabase(db_path=":memory:", client_id="owner-1")
    try:
        content = "C++ templates are local evidence."
        media_id, _, _ = db.add_media_with_keywords(
            url="local://slides-cpp",
            title="C++ guide",
            media_type="web_document",
            content=content,
            keywords=["cpp"],
            prompt=None,
            analysis_content=None,
            safe_metadata=None,
            transcription_model=None,
            author="unit",
            ingestion_date=None,
            overwrite=False,
            chunk_options=None,
            chunks=[
                {
                    "text": content,
                    "start_char": 0,
                    "end_char": len(content),
                    "chunk_type": "text",
                    "metadata": {},
                }
            ],
        )
        assert media_id is not None
        db.ensure_chunk_fts()
        db.maybe_rebuild_chunk_fts_if_empty()
        retriever = database_retrievers.MediaDBRetriever(
            None,
            config=database_retrievers.RetrievalConfig(
                max_results=2,
                use_fts=True,
                use_vector=False,
                fts_level="chunk",
            ),
            user_id="owner-1",
            media_db=db,
        )

        documents = await retriever.retrieve_slides_source_documents_v1(
            query="C++",
            owner_user_id="owner-1",
            max_source_chars=500,
            top_k=2,
        )

        assert documents
        assert "C++" in documents[0].content
        assert documents[0].metadata["_standalone_source_preformatted"] is True
    finally:
        db.close_connection()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_locked_notes_and_chat_use_sqlite_fts_for_noncontiguous_terms(tmp_path):
    db = CharactersRAGDB(
        db_path=str(tmp_path / "slides-rag.sqlite"),
        client_id="owner-1",
    )
    try:
        db.add_note(
            title="Owner note",
            content="owner filler words evidence",
            note_id="note-1",
        )
        character_id = db.add_character_card({"name": "Source character"})
        conversation_id = db.add_conversation({"character_id": character_id, "title": "Source chat"})
        db.add_message(
            {
                "id": "message-1",
                "conversation_id": conversation_id,
                "sender": "user",
                "content": "owner filler words evidence",
            }
        )
        config = database_retrievers.RetrievalConfig(
            max_results=2,
            use_fts=True,
            use_vector=False,
        )
        notes = database_retrievers.NotesDBRetriever(None, config=config, chacha_db=db)
        chats = database_retrievers.ChatHistoryRetriever(None, config=config, chacha_db=db)

        note_documents = await notes.retrieve_slides_source_documents_v1(
            query="owner evidence",
            owner_user_id="owner-1",
            max_source_chars=500,
            top_k=2,
        )
        chat_documents = await chats.retrieve_slides_source_documents_v1(
            query="owner evidence",
            owner_user_id="owner-1",
            max_source_chars=500,
            top_k=2,
        )

        assert [document.id for document in note_documents] == ["note_note-1"]
        assert [document.id for document in chat_documents] == ["chat_message-1"]
        assert all(
            document.metadata["_standalone_source_preformatted"] is True for document in note_documents + chat_documents
        )
    finally:
        db.close_connection()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_locked_chacha_fts_query_is_not_written_to_debug_logs(tmp_path):
    secret_query = "PRIVATEQUERYTOKEN"
    db = CharactersRAGDB(
        db_path=str(tmp_path / "slides-rag-logs.sqlite"),
        client_id="owner-1",
    )
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="DEBUG", format="{message}")
    try:
        retriever = database_retrievers.NotesDBRetriever(
            None,
            config=database_retrievers.RetrievalConfig(use_fts=True, use_vector=False),
            chacha_db=db,
        )
        await retriever.retrieve_slides_source_documents_v1(
            query=secret_query,
            owner_user_id="owner-1",
            max_source_chars=100,
            top_k=2,
        )
    finally:
        logger.remove(sink_id)
        db.close_connection()

    assert secret_query not in "\n".join(messages)
    assert any("Params: [redacted]" in message for message in messages)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_closed_entry_point_bounds_final_projection_and_fuses_sources(
    monkeypatch,
):
    source_documents = {
        "media": [
            _document(DataSource.MEDIA_DB, "media-1", 0.99),
            _document(DataSource.MEDIA_DB, "media-2", 0.98),
        ],
        "notes": [_document(DataSource.NOTES, "note-1", 0.01)],
        "chats": [_document(DataSource.CHAT_HISTORY, "chat-1", 0.001)],
    }
    retrievers = _patch_two_phase_retrievers(monkeypatch, source_documents)

    result = await unified_pipeline.retrieve_slides_source_documents_v1(
        query="owner evidence",
        owner_user_id="owner-1",
        top_k=3,
        max_source_chars=100,
        media_db=object(),
        chacha_db=object(),
    )

    assert all(
        retriever.retrieve_slides_source_candidates_v1.await_args.kwargs["top_k"] == 3
        for retriever in retrievers.values()
    )
    projected_caps = sum(
        char_cap
        for retriever in retrievers.values()
        for _, char_cap in (
            retriever.project_slides_source_documents_v1.await_args.kwargs["projections"]
            if retriever.project_slides_source_documents_v1.await_count
            else []
        )
    )
    assert projected_caps + 4 <= 101
    assert [document.id for document in result.documents] == [
        "media-1",
        "note-1",
        "chat-1",
    ]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("top_k", (1, 2))
async def test_closed_entry_point_searches_every_source_family_at_low_top_k(
    monkeypatch,
    top_k,
):
    retrievers = _patch_two_phase_retrievers(
        monkeypatch,
        {
            "media": [_document(DataSource.MEDIA_DB, "media-1", 0.9)],
            "notes": [_document(DataSource.NOTES, "note-1", 0.8)],
            "chats": [_document(DataSource.CHAT_HISTORY, "chat-1", 0.7)],
        },
    )

    result = await unified_pipeline.retrieve_slides_source_documents_v1(
        query="owner evidence",
        owner_user_id="owner-1",
        top_k=top_k,
        max_source_chars=100,
        media_db=object(),
        chacha_db=object(),
    )

    assert len(result.documents) == top_k
    for retriever in retrievers.values():
        retriever.retrieve_slides_source_candidates_v1.assert_awaited_once_with(
            query="owner evidence",
            owner_user_id="owner-1",
            top_k=top_k,
        )
        retriever.retrieve_slides_source_documents_v1.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_closed_entry_point_reuses_sparse_family_capacity(monkeypatch):
    retrievers = _patch_two_phase_retrievers(
        monkeypatch,
        {
            "media": [],
            "notes": [
                _document(DataSource.NOTES, "note-1", 0.9),
                _document(DataSource.NOTES, "note-2", 0.8),
            ],
            "chats": [],
        },
    )

    result = await unified_pipeline.retrieve_slides_source_documents_v1(
        query="owner evidence",
        owner_user_id="owner-1",
        top_k=2,
        max_source_chars=100,
        media_db=object(),
        chacha_db=object(),
    )

    assert [document.id for document in result.documents] == ["note-1", "note-2"]
    assert retrievers["notes"].project_slides_source_documents_v1.await_count == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_closed_entry_point_marks_budget_omitted_candidate_as_truncated(monkeypatch):
    media_document = Document(
        id="m",
        content="m",
        metadata={"_standalone_source_preformatted": True},
        source=DataSource.MEDIA_DB,
        score=0.9,
    )
    note_document = Document(
        id="n",
        content="n",
        metadata={"_standalone_source_preformatted": True},
        source=DataSource.NOTES,
        score=0.8,
    )
    _patch_two_phase_retrievers(
        monkeypatch,
        {
            "media": [media_document],
            "notes": [note_document],
            "chats": [],
        },
    )

    result = await unified_pipeline.retrieve_slides_source_documents_v1(
        query="owner evidence",
        owner_user_id="owner-1",
        top_k=2,
        max_source_chars=1,
        media_db=object(),
        chacha_db=object(),
    )

    assert result.documents
    assert any(
        document.metadata.get("_standalone_source_projection_truncated") is True for document in result.documents
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_closed_entry_point_reuses_unused_projection_chars_without_false_truncation(
    monkeypatch,
):
    full_content = "# short\n\n" + ("x" * 50)
    retrievers = _patch_two_phase_retrievers(
        monkeypatch,
        {
            "media": [],
            "notes": [
                Document(
                    id="note-1",
                    content=full_content,
                    metadata={"_standalone_source_preformatted": True},
                    source=DataSource.NOTES,
                    score=0.9,
                )
            ],
            "chats": [],
        },
    )

    result = await unified_pipeline.retrieve_slides_source_documents_v1(
        query="owner evidence",
        owner_user_id="owner-1",
        top_k=3,
        max_source_chars=100,
        media_db=object(),
        chacha_db=object(),
    )

    assert result.documents[0].content == full_content
    assert not result.documents[0].metadata.get("_standalone_source_projection_truncated")
    projection_calls = [
        retriever.project_slides_source_documents_v1.await_args
        for retriever in retrievers.values()
        if retriever.project_slides_source_documents_v1.await_count
    ]
    projected_caps = sum(char_cap for call in projection_calls for _, char_cap in call.kwargs["projections"])
    assert projected_caps + (2 * (len(result.documents) - 1)) <= 101


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("source_name", ("media", "notes", "chats"))
async def test_locked_sqlite_sources_fail_closed_on_nul_text(tmp_path, source_name):
    query = "nulneedle"
    content = f"{query}\x00hidden"
    config = database_retrievers.RetrievalConfig(
        max_results=2,
        use_fts=True,
        use_vector=False,
        fts_level="chunk",
    )

    if source_name == "media":
        db = MediaDatabase(db_path=":memory:", client_id="owner-1")
        db.add_media_with_keywords(
            url="local://slides-nul",
            title="NUL source",
            media_type="web_document",
            content=content,
            keywords=["nul"],
            prompt=None,
            analysis_content=None,
            safe_metadata=None,
            transcription_model=None,
            author="unit",
            ingestion_date=None,
            overwrite=False,
            chunk_options=None,
            chunks=[
                {
                    "text": content,
                    "start_char": 0,
                    "end_char": len(content),
                    "chunk_type": "text",
                    "metadata": {},
                }
            ],
        )
        db.ensure_chunk_fts()
        db.maybe_rebuild_chunk_fts_if_empty()
        retriever = database_retrievers.MediaDBRetriever(
            None,
            config=config,
            user_id="owner-1",
            media_db=db,
        )
    else:
        db = CharactersRAGDB(
            db_path=str(tmp_path / f"slides-{source_name}-nul.sqlite"),
            client_id="owner-1",
        )
        if source_name == "notes":
            db.add_note(title="NUL source", content=content, note_id="note-1")
            retriever = database_retrievers.NotesDBRetriever(
                None,
                config=config,
                chacha_db=db,
            )
        else:
            character_id = db.add_character_card({"name": "Source character"})
            conversation_id = db.add_conversation({"character_id": character_id, "title": "Source chat"})
            db.add_message(
                {
                    "id": "message-1",
                    "conversation_id": conversation_id,
                    "sender": "user",
                    "content": content,
                }
            )
            retriever = database_retrievers.ChatHistoryRetriever(
                None,
                config=config,
                chacha_db=db,
            )

    try:
        with pytest.raises(RAGDatabaseError):
            await retriever.retrieve_slides_source_documents_v1(
                query=query,
                owner_user_id="owner-1",
                max_source_chars=100,
                top_k=2,
            )
    finally:
        db.close_connection()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("source_name", ("media", "notes", "chats"))
async def test_direct_locked_retrievers_reject_invalid_budget_before_query(
    source_name,
):
    adapter = MagicMock()
    adapter.backend_type = database_retrievers.BackendType.POSTGRESQL
    adapter.execute_query.side_effect = AssertionError("database was queried")
    config = database_retrievers.RetrievalConfig(use_fts=True, use_vector=False)
    if source_name == "media":
        retriever = database_retrievers.MediaDBRetriever(
            None,
            config=config,
            user_id="owner-1",
            media_db=adapter,
        )
    elif source_name == "notes":
        retriever = database_retrievers.NotesDBRetriever(
            None,
            config=config,
            chacha_db=adapter,
        )
    else:
        retriever = database_retrievers.ChatHistoryRetriever(
            None,
            config=config,
            chacha_db=adapter,
        )

    with pytest.raises(ValueError, match="max_source_chars"):
        await retriever.retrieve_slides_source_documents_v1(
            query="owner evidence",
            owner_user_id="owner-1",
            max_source_chars=0,
            top_k=2,
        )

    adapter.execute_query.assert_not_called()
