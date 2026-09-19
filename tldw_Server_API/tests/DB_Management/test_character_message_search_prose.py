"""Real database searches accept the prose used by Knowledge QA."""

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError

pytestmark = pytest.mark.integration

QUERIES = [
    "Who directs Rowan Observatory, where is it located, and when do public tours begin? Cite the source.",
    "Who directs [entity] Observatory, where is it located and when do public tours begin? [entity] the source.",
    "ORBIT-742",
]


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def search_db(request, tmp_path):
    """Use per-user SQLite or the repository's official PostgreSQL fixture."""
    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    db = CharactersRAGDB(tmp_path / "1" / "search.db", client_id="1", backend=backend)
    try:
        yield db
    finally:
        db.close_all_connections()
        if backend is not None:
            backend.get_pool().close_all()


@pytest.mark.parametrize("query", QUERIES)
def test_character_prose_search_returns_matching_card_without_unrelated_or_deleted(search_db, query):
    matching = search_db.add_character_card({"name": "Rowan guide", "description": query})
    search_db.add_character_card({"name": "Unrelated ocean diary"})
    removed = search_db.add_character_card({"name": "Removed guide", "description": query})
    search_db.soft_delete_character_card(removed, expected_version=1)

    assert [row["id"] for row in search_db.search_character_cards(query)] == [matching]


@pytest.mark.parametrize("query", QUERIES)
def test_message_prose_search_preserves_conversation_filter_and_pagination(search_db, query):
    conversation = search_db.add_conversation({"title": "Rowan tour"})
    other = search_db.add_conversation({"title": "Other tour"})
    expected = {
        search_db.add_message({"conversation_id": conversation, "sender": "user", "content": query}),
        search_db.add_message({"conversation_id": conversation, "sender": "ai", "content": query}),
    }
    search_db.add_message({"conversation_id": other, "sender": "user", "content": query})
    search_db.add_message({"conversation_id": conversation, "sender": "user", "content": "Unrelated ocean diary"})
    removed = search_db.add_message({"conversation_id": conversation, "sender": "user", "content": query})
    search_db.soft_delete_message(removed, expected_version=1)

    pages = [
        search_db.search_messages_by_content(query, conversation_id=conversation, limit=1, offset=offset)
        for offset in range(3)
    ]
    assert {row["id"] for page in pages for row in page} == expected
    assert [len(page) for page in pages] == [1, 1, 0]


def test_sqlite_search_keeps_terms_and_quoted_phrase_behavior(tmp_path):
    db = CharactersRAGDB(tmp_path / "owner" / "search.db", client_id="1")
    foreign = CharactersRAGDB(tmp_path / "other" / "search.db", client_id="2")
    try:
        near = db.add_character_card({"name": "Rowan Observatory"})
        separated = db.add_character_card({"name": "Rowan hill Observatory"})
        foreign.add_character_card({"name": "Rowan Observatory private"})
        assert {r["id"] for r in db.search_character_cards("Rowan Observatory,")} == {near, separated}
        assert [r["id"] for r in db.search_character_cards('"Rowan Observatory"')] == [near]
        assert len(db.search_character_cards("Rowan", limit=1)) == 1
        assert db.search_character_cards("NoSuchCanary") == []
        own_conversation = db.add_conversation({"title": "Own conversation"})
        own_message = db.add_message({"conversation_id": own_conversation, "sender": "user", "content": "Rowan Observatory"})
        foreign_conversation = foreign.add_conversation({"title": "Foreign conversation"})
        foreign.add_message({"conversation_id": foreign_conversation, "sender": "user", "content": "Rowan Observatory"})
        assert [r["id"] for r in db.search_messages_by_content('"Rowan Observatory"')] == [own_message]
        assert db.search_messages_by_content("   ") == []
    finally:
        db.close_all_connections()
        foreign.close_all_connections()


@pytest.mark.asyncio
async def test_real_qa_retrievers_find_character_and_chat_evidence_for_prose(search_db):
    from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import MultiDatabaseRetriever
    from tldw_Server_API.app.core.RAG.rag_service.types import DataSource

    character = search_db.add_character_card({"name": "Observatory guide Rowan"})
    conversation = search_db.add_conversation({"title": "Tour answers"})
    message = search_db.add_message({"conversation_id": conversation, "sender": "user", "content": "The Observatory is called Rowan."})
    retriever = MultiDatabaseRetriever({"character_cards_db": search_db.db_path_str}, chacha_db=search_db)
    try:
        docs = await retriever.retrieve("Rowan Observatory,", sources=[DataSource.CHARACTER_CARDS, DataSource.CHAT_HISTORY])
        assert {doc.id for doc in docs} == {f"character_{character}", f"chat_{message}"}
    finally:
        retriever.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["Rowan", "Rowan Observatory,"])
async def test_real_qa_pipeline_default_chunk_filter_keeps_owned_character_and_chat(search_db, query):
    from tldw_Server_API.app.core.RAG.rag_service.streaming_executor import _context_events
    from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import unified_rag_pipeline

    character = search_db.add_character_card({"name": "Rowan Observatory guide"})
    conversation = search_db.add_conversation({"title": "Tour answers"})
    message = search_db.add_message({"conversation_id": conversation, "sender": "user", "content": "Rowan Observatory is on Cedar Hill."})
    search_db.add_character_card({"name": "Unrelated ocean diary"})
    result = await unified_rag_pipeline(
        query=query, sources=["characters", "chats"],
        character_db_path=search_db.db_path_str, chacha_db=search_db, user_id="1",
        chunk_type_filter=["text", "code", "table", "list"],
        search_mode="fts", enable_cache=False, enable_reranking=False, enable_generation=False,
    )
    assert {doc["id"] for doc in result.documents} == {f"character_{character}", f"chat_{message}"}
    expected = {
        f"character_{character}": ("Rowan Observatory guide", "characters", str(character)),
        f"chat_{message}": ("Tour answers", "chats", conversation),
    }
    for doc in result.documents:
        metadata = doc["metadata"]
        assert (metadata["title"], metadata["source_type"], metadata["source_id"]) == expected[doc["id"]]
    contexts = _context_events(docs=result.documents, payload={}, request_defaults={})[0]["contexts"]
    assert {
        context["id"]: (context["title"], context["source_type"], context["source_id"])
        for context in contexts
    } == expected


def test_sqlite_search_keeps_grouped_fts_boolean_meaning(tmp_path):
    db = CharactersRAGDB(tmp_path / "grouped.db", client_id="1")
    try:
        matching = {
            db.add_character_card({"name": "Rowan Observatory"}),
            db.add_character_card({"name": "Vega Observatory"}),
        }
        db.add_character_card({"name": "Rowan unrelated diary"})
        conversation = db.add_conversation({"title": "Grouped query"})
        messages = {
            db.add_message({"conversation_id": conversation, "sender": "user", "content": "Rowan Observatory"}),
            db.add_message({"conversation_id": conversation, "sender": "user", "content": "Vega Observatory"}),
        }
        db.add_message({"conversation_id": conversation, "sender": "user", "content": "Rowan unrelated diary"})
        query = "(Rowan OR Vega) AND Observatory"
        assert {row["id"] for row in db.search_character_cards(query)} == matching
        assert {row["id"] for row in db.search_messages_by_content(query)} == messages
        assert len(db.search_character_cards("NEAR(Rowan Observatory, 3)")) == 1
        assert len(db.search_messages_by_content("NEAR(Rowan Observatory, 3)")) == 1
    finally:
        db.close_all_connections()


def test_sqlite_prose_fallback_preserves_caller_rollback_and_real_errors(tmp_path):
    db = CharactersRAGDB(tmp_path / "rollback.db", client_id="1")

    class Rollback(Exception):
        pass

    try:
        with pytest.raises(Rollback), db.transaction():
            character = db.add_character_card({"name": "Rowan Observatory"})
            conversation = db.add_conversation({"title": "Pending conversation"})
            message = db.add_message({"conversation_id": conversation, "sender": "user", "content": "Rowan Observatory"})
            assert [row["id"] for row in db.search_character_cards("Rowan Observatory,")] == [character]
            assert [row["id"] for row in db.search_messages_by_content("Rowan Observatory,")] == [message]
            raise Rollback
        assert db.search_character_cards("Rowan") == []
        assert db.search_messages_by_content("Rowan") == []
        db.get_connection().execute("DROP TABLE character_cards_fts")
        with pytest.raises(CharactersRAGDBError, match="no such table"):
            db.search_character_cards("Rowan")
    finally:
        db.close_all_connections()


@pytest.fixture
def private_chat_evidence(search_db, tmp_path):
    """Seed matching private evidence in the two users' actual storage layout."""
    from tldw_Server_API.app.core.DB_Management.backends.base import BackendType

    foreign = CharactersRAGDB(
        tmp_path / "2" / "search.db", client_id="2",
        backend=search_db.backend if search_db.backend_type == BackendType.POSTGRESQL else None,
    )
    records = {}
    try:
        for name, db in (("own", search_db), ("foreign", foreign), ("deleted", search_db)):
            conversation = db.add_conversation({"title": f"{name} tour"})
            message = db.add_message({
                "conversation_id": conversation, "sender": "user",
                "content": "Rowan Observatory opens Fridays.",
            })
            records[name] = (conversation, message)
        # Older/imported data may retain active child rows under a deleted parent.
        search_db.execute_query(
            "UPDATE conversations SET deleted = TRUE WHERE id = ?",
            (records["deleted"][0],), commit=True,
        )
        yield search_db, records
    finally:
        foreign.close_all_connections()


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["Rowan", "Rowan Observatory,"])
async def test_chat_retrieval_excludes_foreign_and_deleted_conversations(private_chat_evidence, query):
    from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import ChatHistoryRetriever

    db, records = private_chat_evidence
    retriever = ChatHistoryRetriever(db.db_path_str, chacha_db=db)
    documents = await retriever.retrieve(query)

    assert {doc.id for doc in documents} == {f"chat_{records['own'][1]}"}


def test_message_search_filters_owners_and_deleted_parents_before_pagination(private_chat_evidence):
    db, records = private_chat_evidence
    pages = [db.search_messages_by_content("Rowan", limit=1, offset=offset) for offset in range(3)]

    assert [row["id"] for page in pages for row in page] == [records["own"][1]]
    assert db.search_messages_by_content("Rowan", conversation_id=records["foreign"][0]) == []


@pytest.mark.asyncio
async def test_chat_metadata_excludes_foreign_and_deleted_conversations(private_chat_evidence):
    from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import ChatHistoryRetriever

    db, records = private_chat_evidence
    retriever = ChatHistoryRetriever(db.db_path_str, chacha_db=db)

    assert (await retriever.get_metadata(f"chat_{records['own'][1]}"))["id"] == records["own"][1]
    assert await retriever.get_metadata(f"chat_{records['foreign'][1]}") == {}
    assert await retriever.get_metadata(f"chat_{records['deleted'][1]}") == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("retriever_name", ["ChatHistoryRetriever", "CharacterCardsRetriever"])
@pytest.mark.parametrize("query", ["Rowan", "Rowan Observatory,"])
async def test_chat_evidence_projects_visible_text_without_rewriting_messages(search_db, retriever_name, query):
    from tldw_Server_API.app.core.RAG.rag_service import database_retrievers

    conversation = search_db.add_conversation({"title": "Public tour answer"})
    raw = "<think>Rowan Observatory internal scratchpad.</think>Rowan Observatory opens Fridays."
    answer = search_db.add_message({"conversation_id": conversation, "sender": "assistant", "content": raw})
    question = search_db.add_message({"conversation_id": conversation, "sender": "user", "content": "Rowan Observatory tour hours?"})
    search_db.add_message({"conversation_id": conversation, "sender": "assistant", "content": "<reasoning>Rowan Observatory unfinished scratchpad"})
    retriever = getattr(database_retrievers, retriever_name)(search_db.db_path_str, chacha_db=search_db)

    documents = await retriever.retrieve(query)

    assert {doc.id for doc in documents} == {f"chat_{answer}", f"chat_{question}"}
    assert all("scratchpad" not in doc.content for doc in documents)
    assert "Rowan Observatory opens Fridays." in next(doc.content for doc in documents if doc.id == f"chat_{answer}")
    assert search_db.get_message_by_id(answer)["content"] == raw
