from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.RAG.rag_service import database_retrievers as database_retrievers_module
from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import (
    ChatHistoryRetriever,
    MediaDBRetriever,
    NotesDBRetriever,
    RetrievalConfig,
)
from tldw_Server_API.app.core.RAG.rag_service.generation import (
    FallbackGenerator,
    GenerationConfig,
)
from tldw_Server_API.app.core.RAG.rag_service.retrieval_plan import RetrievalPlan
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document
from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline as unified_pipeline_module
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import unified_rag_pipeline
from tldw_Server_API.tests.RAG.knowledge_qa_uat_fixtures import (
    KNOWN_NOTE_SOURCE_BODY,
    KNOWN_NOTE_SOURCE_TITLE,
    SCOPED_INCLUDED_PHRASE,
    SCOPED_INCLUDED_QUERY,
)


def _document_id(document: object) -> str:
    if isinstance(document, dict):
        return str(document.get("id") or "")
    return str(getattr(document, "id", ""))


def _document_metadata(document: object) -> dict:
    if isinstance(document, dict):
        metadata = document.get("metadata")
    else:
        metadata = getattr(document, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _document_source(document: object) -> str:
    metadata = _document_metadata(document)
    if metadata.get("source") is not None:
        return str(metadata["source"])
    if isinstance(document, dict):
        return str(document.get("source") or "")
    return str(getattr(document, "source", ""))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_notes_only_scope_does_not_fall_back_to_media_db(tmp_path: Path) -> None:
    media_db_path = tmp_path / "Media_DB_v2.db"
    notes_db_path = tmp_path / "ChaChaNotes.db"
    media_db = MediaDatabase(db_path=str(media_db_path), client_id="pytest")
    chacha_db = CharactersRAGDB(db_path=str(notes_db_path), client_id="pytest")

    media_id, _, _ = media_db.add_media_with_keywords(
        title="Out of scope media",
        media_type="document",
        content="scoped note rule should never come from media fallback",
        keywords=["knowledge-qa-live-regression"],
    )

    result = await unified_rag_pipeline(
        query="scoped note rule",
        sources=["notes"],
        media_db_path=str(media_db_path),
        notes_db_path=str(notes_db_path),
        media_db=media_db,
        chacha_db=chacha_db,
        search_mode="hybrid",
        top_k=5,
        min_score=0.0,
        enable_generation=False,
        enable_reranking=False,
        enable_cache=False,
        include_note_ids=["missing-note"],
    )

    returned_ids = {
        str(_document_metadata(document).get("media_id") or _document_id(document))
        for document in result.documents
    }
    assert str(media_id) not in returned_ids
    assert all(
        _document_source(document) != "media_db"
        for document in result.documents
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_selected_note_scope_returns_selected_note_when_query_is_natural_language(
    tmp_path: Path,
) -> None:
    notes_db_path = tmp_path / "ChaChaNotes.db"
    chacha_db = CharactersRAGDB(db_path=str(notes_db_path), client_id="pytest")
    note_id = chacha_db.add_note(
        title=KNOWN_NOTE_SOURCE_TITLE,
        content=KNOWN_NOTE_SOURCE_BODY,
    )

    result = await unified_rag_pipeline(
        query=SCOPED_INCLUDED_QUERY,
        sources=["notes"],
        notes_db_path=str(notes_db_path),
        chacha_db=chacha_db,
        search_mode="hybrid",
        top_k=5,
        min_score=0.0,
        enable_generation=False,
        enable_reranking=False,
        enable_cache=False,
        include_note_ids=[note_id],
    )

    matched_documents = [
        document
        for document in result.documents
        if str(_document_metadata(document).get("source") or _document_source(document)) == "notes_db"
        and str(_document_id(document)).replace("note_", "") == str(note_id)
        and SCOPED_INCLUDED_PHRASE
        in str(document.get("content") if isinstance(document, dict) else document.content)
    ]
    assert matched_documents
    matched_metadata = _document_metadata(matched_documents[0])
    assert matched_metadata["note_id"] == str(note_id)
    assert matched_metadata["source_id"] == str(note_id)
    assert matched_metadata["source_type"] == "notes"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_selected_note_scope_survives_webui_chunk_type_filter(
    tmp_path: Path,
) -> None:
    notes_db_path = tmp_path / "ChaChaNotes.db"
    chacha_db = CharactersRAGDB(db_path=str(notes_db_path), client_id="pytest")
    note_id = chacha_db.add_note(
        title=KNOWN_NOTE_SOURCE_TITLE,
        content=KNOWN_NOTE_SOURCE_BODY,
    )

    result = await unified_rag_pipeline(
        query=SCOPED_INCLUDED_QUERY,
        sources=["media_db", "notes", "characters", "chats"],
        notes_db_path=str(notes_db_path),
        chacha_db=chacha_db,
        search_mode="hybrid",
        top_k=8,
        min_score=0.0,
        enable_generation=False,
        enable_reranking=True,
        enable_cache=False,
        include_note_ids=[note_id],
        chunk_type_filter=["text", "code", "table", "list"],
    )

    assert result.metadata.get("sources_searched") == ["notes"]
    assert result.metadata.get("chunk_type_filter_after") != 0
    assert any(
        str(_document_id(document)).replace("note_", "") == str(note_id)
        for document in result.documents
    )
    selected_document = next(
        document
        for document in result.documents
        if str(_document_id(document)).replace("note_", "") == str(note_id)
    )
    selected_metadata = _document_metadata(selected_document)
    assert selected_metadata["note_id"] == str(note_id)
    assert selected_metadata["source_id"] == str(note_id)
    assert selected_metadata["source_type"] == "notes"
    assert selected_metadata["chunk_type"] == "text"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_explicit_note_scope_does_not_collapse_when_default_sources_omit_notes(
    tmp_path: Path,
) -> None:
    notes_db_path = tmp_path / "ChaChaNotes.db"
    chacha_db = CharactersRAGDB(db_path=str(notes_db_path), client_id="pytest")
    note_id = chacha_db.add_note(
        title=KNOWN_NOTE_SOURCE_TITLE,
        content=KNOWN_NOTE_SOURCE_BODY,
    )

    result = await unified_rag_pipeline(
        query=SCOPED_INCLUDED_QUERY,
        sources=["media_db"],
        notes_db_path=str(notes_db_path),
        chacha_db=chacha_db,
        search_mode="hybrid",
        top_k=5,
        min_score=0.0,
        enable_generation=False,
        enable_reranking=False,
        enable_cache=True,
        include_note_ids=[note_id],
    )

    assert result.metadata.get("sources_searched") == ["notes"]
    assert result.metadata.get("explicit_source_selection", {}).get("scope_intersection_empty") is True
    assert result.metadata.get("cache_bypassed", {}).get("reason") == "explicit_source_selection"
    assert any(
        str(_document_id(document)).replace("note_", "") == str(note_id)
        for document in result.documents
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_explicit_note_selection_excludes_unselected_media_sources(
    tmp_path: Path,
) -> None:
    media_db_path = tmp_path / "Media_DB_v2.db"
    notes_db_path = tmp_path / "ChaChaNotes.db"
    media_db = MediaDatabase(db_path=str(media_db_path), client_id="pytest")
    chacha_db = CharactersRAGDB(db_path=str(notes_db_path), client_id="pytest")
    media_id, _, _ = media_db.add_media_with_keywords(
        title="Unselected media distractor",
        media_type="document",
        content="Excluded distractor should not appear in selected note search.",
        keywords=["knowledge-qa-live-regression"],
    )
    note_id = chacha_db.add_note(
        title=KNOWN_NOTE_SOURCE_TITLE,
        content=KNOWN_NOTE_SOURCE_BODY,
    )

    result = await unified_rag_pipeline(
        query=SCOPED_INCLUDED_QUERY,
        sources=["media_db", "notes", "characters", "chats"],
        media_db_path=str(media_db_path),
        notes_db_path=str(notes_db_path),
        media_db=media_db,
        chacha_db=chacha_db,
        search_mode="hybrid",
        top_k=5,
        min_score=0.0,
        enable_generation=False,
        enable_reranking=False,
        enable_cache=False,
        include_note_ids=[note_id],
    )

    returned_ids = {
        str(_document_metadata(document).get("media_id") or _document_id(document))
        for document in result.documents
    }
    assert str(media_id) not in returned_ids
    assert all(
        _document_source(document) != "media_db"
        for document in result.documents
    )
    assert result.metadata.get("sources_searched") == ["notes"]
    assert any(
        str(_document_id(document)).replace("note_", "") == str(note_id)
        for document in result.documents
    )


@pytest.mark.unit
def test_selected_note_sql_allowlist_is_bounded_by_max_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = NotesDBRetriever(
        str(tmp_path / "ChaChaNotes.db"),
        config=RetrievalConfig(max_results=2),
        chacha_db=None,
    )
    captured_params: tuple[object, ...] | None = None

    def fake_execute_query(sql: str, params: tuple[object, ...]) -> list[dict[str, object]]:
        nonlocal captured_params
        captured_params = params
        return []

    monkeypatch.setattr(retriever, "_execute_query", fake_execute_query)

    retriever._retrieve_allowed_notes_via_sql(["1", "2", "3", "4"], notebook_id=None)

    assert captured_params == ("1", "2", 2)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_selected_note_retrieval_uses_thread_offload_for_sync_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notes_db_path = tmp_path / "ChaChaNotes.db"
    chacha_db = CharactersRAGDB(db_path=str(notes_db_path), client_id="pytest")
    note_id = chacha_db.add_note(
        title=KNOWN_NOTE_SOURCE_TITLE,
        content=KNOWN_NOTE_SOURCE_BODY,
    )
    retriever = NotesDBRetriever(
        str(notes_db_path),
        config=RetrievalConfig(max_results=5),
        chacha_db=chacha_db,
    )
    offloaded_functions: list[str] = []

    async def fake_to_thread(function: object, *args: object, **kwargs: object) -> object:
        offloaded_functions.append(getattr(function, "__name__", "unknown"))
        return function(*args, **kwargs)  # type: ignore[misc]

    monkeypatch.setattr(database_retrievers_module.asyncio, "to_thread", fake_to_thread)

    documents = await retriever.retrieve(SCOPED_INCLUDED_QUERY, allowed_note_ids=[note_id])

    assert offloaded_functions == ["_retrieve_allowed_notes_via_chacha"]
    assert any(str(_document_id(document)).replace("note_", "") == str(note_id) for document in documents)


@pytest.mark.unit
def test_chat_history_excluded_source_lookup_is_cached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = ChatHistoryRetriever(str(tmp_path / "ChaChaNotes.db"))
    query_count = 0

    def fake_execute_query(sql: str, params: tuple[object, ...]) -> list[dict[str, object]]:
        nonlocal query_count
        query_count += 1
        return [{"source": "knowledge_qa"}]

    monkeypatch.setattr(retriever, "_execute_query", fake_execute_query)

    assert retriever._is_excluded_conversation_source("conversation-1") is True
    assert retriever._is_excluded_conversation_source("conversation-1") is True
    assert query_count == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_chat_history_retriever_excludes_knowledge_qa_self_history(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "ChaChaNotes.db"
    chacha_db = CharactersRAGDB(db_path=str(db_path), client_id="pytest")
    character_id = chacha_db.add_character_card(
        {
            "name": "Knowledge QA Test Character",
            "description": "Test helper",
            "personality": "helpful",
            "system_prompt": "You are helpful.",
            "client_id": "pytest",
        }
    )

    normal_conversation_id = chacha_db.add_conversation(
        {
            "character_id": character_id,
            "title": "Normal chat",
            "client_id": "pytest",
            "source": "chat",
        }
    )
    knowledge_conversation_id = chacha_db.add_conversation(
        {
            "character_id": character_id,
            "title": "Knowledge QA search history",
            "client_id": "pytest",
            "source": "knowledge_qa",
        }
    )
    chacha_db.add_message(
        {
            "conversation_id": normal_conversation_id,
            "sender": "user",
            "content": "basalt telemetry belongs in an ordinary chat",
        }
    )
    chacha_db.add_message(
        {
            "conversation_id": knowledge_conversation_id,
            "sender": "user",
            "content": "basalt telemetry is only a Knowledge QA search query",
        }
    )

    retriever = ChatHistoryRetriever(
        str(db_path),
        config=RetrievalConfig(max_results=10),
        chacha_db=chacha_db,
    )

    documents = await retriever.retrieve("basalt telemetry")

    conversation_ids = {
        str(document.metadata.get("conversation_id"))
        for document in documents
        if isinstance(document.metadata, dict)
    }
    assert normal_conversation_id in conversation_ids
    assert knowledge_conversation_id not in conversation_ids


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fallback_generator_uses_inline_citation_markers() -> None:
    document = Document(
        id="1",
        content=(
            "Knowledge QA UAT Grounded Checklist\n"
            "Grounded answers cite visible evidence."
        ),
        source=DataSource.MEDIA_DB,
        metadata={"title": "Knowledge QA UAT Grounded Checklist"},
        score=1.0,
    )
    generator = FallbackGenerator(GenerationConfig(provider="fallback"))

    result = await generator.generate(
        SimpleNamespace(documents=[document]),
        "What does the grounded QA checklist require?",
    )

    assert "[1]" in result.response
    assert "Grounded answers cite visible evidence" in result.response


@pytest.mark.unit
@pytest.mark.asyncio
async def test_media_retriever_natural_language_no_match_does_not_return_all_rows(
    tmp_path: Path,
) -> None:
    media_db_path = tmp_path / "Media_DB_v2.db"
    media_db = MediaDatabase(db_path=str(media_db_path), client_id="pytest")
    media_db.add_media_with_keywords(
        title="Knowledge QA UAT Grounded Checklist",
        media_type="document",
        content=(
            "Knowledge QA UAT fixture. Grounded answers cite visible evidence. "
            "Users must be able to inspect the cited source excerpt before trusting the answer."
        ),
        keywords=["knowledge-qa-live-regression"],
    )
    media_db.add_media_with_keywords(
        title="Knowledge QA UAT Distractor Checklist",
        media_type="document",
        content=(
            "Knowledge QA UAT distractor fixture. A distracting checklist also mentions citations. "
            "Excluded distractor should not appear."
        ),
        keywords=["knowledge-qa-live-regression"],
    )
    retriever = MediaDBRetriever(
        str(media_db_path),
        config=RetrievalConfig(max_results=10, use_fts=True, use_vector=False),
        user_id="0",
        media_db=media_db,
    )

    documents = await retriever.retrieve(
        "What does the library say about nonexistent basalt telemetry?"
    )

    assert documents == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_match_pipeline_does_not_generate_empty_context_answer(
    tmp_path: Path,
) -> None:
    media_db_path = tmp_path / "Media_DB_v2.db"
    media_db = MediaDatabase(db_path=str(media_db_path), client_id="pytest")
    media_db.add_media_with_keywords(
        title="Knowledge QA UAT Grounded Checklist",
        media_type="document",
        content="Knowledge QA UAT fixture. Grounded answers cite visible evidence.",
        keywords=["knowledge-qa-live-regression"],
    )

    result = await unified_rag_pipeline(
        query="What does the library say about nonexistent basalt telemetry?",
        sources=["media_db"],
        media_db_path=str(media_db_path),
        media_db=media_db,
        search_mode="hybrid",
        top_k=5,
        min_score=0.0,
        enable_generation=True,
        enable_reranking=False,
        enable_cache=False,
    )

    assert result.documents == []
    assert not result.generated_answer


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retrieval_error_fallback_uses_effective_retrieval_search_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_db_path = tmp_path / "Media_DB_v2.db"
    media_db = MediaDatabase(db_path=str(media_db_path), client_id="pytest")
    media_db.add_media_with_keywords(
        title="Knowledge QA fallback mode fixture",
        media_type="document",
        content="effective retrieval fallback needle phrase",
        keywords=["knowledge-qa-live-regression"],
    )

    async def fail_primary_retrieval(*args: object, **kwargs: object) -> object:
        raise RuntimeError("primary retrieval unavailable")

    monkeypatch.setattr(
        unified_pipeline_module,
        "execute_retrieval_phase",
        fail_primary_retrieval,
    )

    result = await unified_rag_pipeline(
        query="effective retrieval fallback needle phrase",
        sources=["media_db"],
        media_db_path=str(media_db_path),
        media_db=media_db,
        search_mode="vector",
        retrieval_plan=RetrievalPlan(
            query="effective retrieval fallback needle phrase",
            sources=("media_db",),
            search_mode="fts",
            top_k=5,
            min_score=0.0,
            index_namespace=None,
        ),
        top_k=5,
        min_score=0.0,
        enable_generation=False,
        enable_reranking=False,
        enable_cache=False,
    )

    assert any(
        "effective retrieval fallback needle phrase"
        in str(document.get("content") if isinstance(document, dict) else document.content)
        for document in result.documents
    )
