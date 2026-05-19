import sqlite3
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import (
    MultiDatabaseRetriever,
    PromptsDBRetriever,
)
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document
from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline as up


pytestmark = pytest.mark.unit


def _create_prompts_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE Prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                author TEXT,
                details TEXT,
                system_prompt TEXT,
                user_prompt TEXT,
                uuid TEXT,
                last_modified TEXT,
                version INTEGER DEFAULT 1,
                usage_count INTEGER DEFAULT 0,
                deleted BOOLEAN DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            INSERT INTO Prompts (
                name, author, details, system_prompt, user_prompt, uuid, last_modified
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "Release note prompt",
                "tester",
                "Reusable release notes guidance",
                "You write release notes.",
                "Summarize shipped changes.",
                "prompt-uuid-1",
                "2026-05-12T00:00:00Z",
            ),
        )


def _create_chacha_source_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE character_cards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                description TEXT,
                personality TEXT,
                first_message TEXT,
                system_prompt TEXT,
                scenario TEXT,
                creator TEXT,
                version INTEGER DEFAULT 1,
                deleted BOOLEAN DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                character_id INTEGER,
                deleted BOOLEAN DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                content TEXT,
                sender TEXT,
                timestamp TEXT,
                deleted BOOLEAN DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE world_books (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                enabled BOOLEAN DEFAULT 1,
                deleted BOOLEAN DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE world_book_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                world_book_id INTEGER NOT NULL,
                keywords TEXT NOT NULL,
                content TEXT NOT NULL,
                priority INTEGER DEFAULT 0,
                enabled BOOLEAN DEFAULT 1,
                metadata TEXT DEFAULT '{}'
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE chat_dictionaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                is_active BOOLEAN DEFAULT 1,
                deleted BOOLEAN DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE dictionary_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dictionary_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                content TEXT NOT NULL,
                group_name TEXT,
                enabled BOOLEAN DEFAULT 1
            )
            """
        )
        conn.execute(
            """
            INSERT INTO character_cards (
                name, description, personality, first_message, system_prompt, scenario, creator
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "Ada",
                "Systems retrieval researcher",
                "Precise",
                "Hello",
                "Answer with evidence.",
                "Lab notebook",
                "tester",
            ),
        )
        conn.execute("INSERT INTO conversations (character_id) VALUES (1)")
        conn.execute(
            "INSERT INTO messages (conversation_id, content, sender, timestamp) VALUES (?, ?, ?, ?)",
            (1, "The experiment showed better retrieval coverage.", "user", "2026-05-12T01:00:00Z"),
        )
        conn.execute(
            "INSERT INTO world_books (name, description) VALUES (?, ?)",
            ("Lore", "Retrieval terminology"),
        )
        conn.execute(
            "INSERT INTO world_book_entries (world_book_id, keywords, content, priority) VALUES (?, ?, ?, ?)",
            (1, '["retrieval"]', "Retrieval means finding grounded evidence.", 10),
        )
        conn.execute(
            "INSERT INTO chat_dictionaries (name, description) VALUES (?, ?)",
            ("Glossary", "Search terms"),
        )
        conn.execute(
            "INSERT INTO dictionary_entries (dictionary_id, key, content, group_name) VALUES (?, ?, ?, ?)",
            (1, "RAG", "Retrieval augmented generation", "default"),
        )


@pytest.mark.asyncio
async def test_prompts_retriever_searches_current_prompts_database_schema(tmp_path: Path) -> None:
    prompts_db = tmp_path / "prompts.sqlite"
    _create_prompts_db(prompts_db)

    retriever = PromptsDBRetriever(str(prompts_db))

    documents = await retriever.retrieve("release notes")

    assert [doc.source for doc in documents] == [DataSource.PROMPTS]
    assert documents[0].metadata["source"] == "prompts"
    assert documents[0].metadata["prompt_id"] == 1
    assert "Summarize shipped changes" in documents[0].content


@pytest.mark.asyncio
async def test_multi_database_retriever_searches_all_chacha_knowledge_surfaces(tmp_path: Path) -> None:
    chacha_db = tmp_path / "chacha.sqlite"
    _create_chacha_source_db(chacha_db)

    retriever = MultiDatabaseRetriever(
        {
            "character_cards_db": str(chacha_db),
            "world_books_db": str(chacha_db),
            "chat_dictionaries_db": str(chacha_db),
        }
    )

    documents = await retriever.retrieve(
        "retrieval",
        sources=[
            DataSource.CHAT_HISTORY,
            DataSource.CHARACTER_CARDS,
            DataSource.WORLD_BOOKS,
            DataSource.DICTIONARIES,
        ],
    )

    sources = {doc.source for doc in documents}
    assert DataSource.CHAT_HISTORY in sources
    assert DataSource.CHARACTER_CARDS in sources
    assert DataSource.WORLD_BOOKS in sources
    assert DataSource.DICTIONARIES in sources


@pytest.mark.asyncio
async def test_unified_pipeline_reports_source_status_for_requested_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRetriever:
        def __init__(self, db_paths: dict[str, str], **kwargs: Any) -> None:
            self.retrievers = {
                DataSource.MEDIA_DB: object(),
                DataSource.NOTES: object(),
                DataSource.CHAT_HISTORY: object(),
                DataSource.CHARACTER_CARDS: object(),
                DataSource.KANBAN: object(),
                DataSource.PROMPTS: object(),
            }

        async def retrieve_from_plan(self, plan: Any, **kwargs: Any) -> list[Document]:
            return [
                Document(
                    id="media-1",
                    content="grounded result",
                    metadata={"source": "media_db"},
                    source=DataSource.MEDIA_DB,
                    score=0.9,
                )
            ]

    monkeypatch.setattr(up, "MultiDatabaseRetriever", FakeRetriever)

    result = await up.unified_rag_pipeline(
        query="coverage",
        sources=[
            "media_db",
            "notes",
            "chats",
            "characters",
            "kanban",
            "prompts",
            "world_books",
            "dictionaries",
        ],
        media_db_path=":memory:",
        enable_generation=False,
        enable_cache=False,
        enable_reranking=False,
        enable_security_filter=False,
    )

    source_status = result.metadata["source_status"]
    assert set(source_status) == {
        "media_db",
        "notes",
        "chats",
        "characters",
        "kanban",
        "prompts",
        "world_books",
        "dictionaries",
    }
    assert source_status["media_db"] == {"status": "searched", "count": 1}
    assert source_status["notes"] == {
        "status": "empty",
        "count": 0,
        "reason": "no_matching_entries",
    }
    assert source_status["world_books"]["status"] == "unavailable"
    assert source_status["world_books"]["reason"] == "no_retriever_configured"
    assert source_status["dictionaries"]["status"] == "unavailable"
    assert source_status["dictionaries"]["reason"] == "no_retriever_configured"


@pytest.mark.asyncio
async def test_unified_pipeline_filters_workspace_artifacts_without_workspace_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRetriever:
        def __init__(self, db_paths: dict[str, str], **kwargs: Any) -> None:
            self.retrievers = {DataSource.MEDIA_DB: object()}

        async def retrieve_from_plan(self, plan: Any, **kwargs: Any) -> list[Document]:
            return [
                Document(
                    id="normal",
                    content="normal result",
                    metadata={"source": "media_db"},
                    source=DataSource.MEDIA_DB,
                    score=0.9,
                ),
                Document(
                    id="generated",
                    content="generated workspace draft",
                    metadata={"source": "media_db", "workspace_id": "ws-1", "is_generated": True},
                    source=DataSource.MEDIA_DB,
                    score=0.8,
                ),
                Document(
                    id="test-artifact",
                    content="test fixture output",
                    metadata={"source": "media_db", "workspace_id": "ws-1", "is_test_artifact": True},
                    source=DataSource.MEDIA_DB,
                    score=0.7,
                ),
                Document(
                    id="workspace-only",
                    content="workspace note",
                    metadata={"source": "media_db", "workspace_id": "ws-1"},
                    source=DataSource.MEDIA_DB,
                    score=0.6,
                ),
            ]

    monkeypatch.setattr(up, "MultiDatabaseRetriever", FakeRetriever)

    global_result = await up.unified_rag_pipeline(
        query="workspace",
        sources=["media_db"],
        media_db_path=":memory:",
        enable_generation=False,
        enable_cache=False,
        enable_reranking=False,
        enable_security_filter=False,
    )
    scoped_result = await up.unified_rag_pipeline(
        query="workspace",
        sources=["media_db"],
        media_db_path=":memory:",
        workspace_id="ws-1",
        enable_generation=False,
        enable_cache=False,
        enable_reranking=False,
        enable_security_filter=False,
    )

    assert [doc["id"] for doc in global_result.documents] == ["normal"]
    assert [doc["id"] for doc in scoped_result.documents] == [
        "normal",
        "generated",
        "test-artifact",
        "workspace-only",
    ]
    assert global_result.metadata["source_status"]["media_db"]["filtered_artifact_count"] == 3


@pytest.mark.asyncio
async def test_unified_pipeline_filters_workspace_artifacts_added_by_prf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRetriever:
        def __init__(self, db_paths: dict[str, str], **kwargs: Any) -> None:
            self.retrievers = {DataSource.MEDIA_DB: object()}

        async def retrieve_from_plan(self, plan: Any, **kwargs: Any) -> list[Document]:
            if getattr(plan, "query", "") == "workspace expanded":
                return [
                    Document(
                        id="workspace-prf",
                        content="generated workspace evidence",
                        metadata={
                            "source": "media_db",
                            "workspace_id": "ws-1",
                            "is_generated": True,
                        },
                        source=DataSource.MEDIA_DB,
                        score=0.95,
                    )
                ]
            return [
                Document(
                    id="normal",
                    content="normal result",
                    metadata={"source": "media_db"},
                    source=DataSource.MEDIA_DB,
                    score=0.9,
                )
            ]

    async def fake_apply_prf(query: str, documents: list[Document], config: Any) -> tuple[str, dict[str, Any]]:
        return "workspace expanded", {"enabled": True}

    monkeypatch.setattr(up, "MultiDatabaseRetriever", FakeRetriever)
    monkeypatch.setattr(up, "apply_prf", fake_apply_prf)

    result = await up.unified_rag_pipeline(
        query="workspace",
        sources=["media_db"],
        media_db_path=":memory:",
        enable_prf=True,
        enable_generation=False,
        enable_cache=False,
        enable_reranking=False,
        enable_security_filter=False,
    )

    assert [doc["id"] for doc in result.documents] == ["normal"]
    assert result.metadata["source_status"]["media_db"]["filtered_artifact_count"] == 1


@pytest.mark.asyncio
async def test_unified_pipeline_cache_namespace_includes_workspace_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespaces: list[str | None] = []

    class RecordingCache:
        def get(self, query: str) -> None:
            return None

        def find_similar(self, query: str) -> None:
            return None

        def set(self, query: str, value: object, ttl: int | None = None) -> None:
            return None

    class FakeCacheClass:
        pass

    class FakeRetriever:
        def __init__(self, db_paths: dict[str, str], **kwargs: Any) -> None:
            self.retrievers = {DataSource.MEDIA_DB: object()}

        async def retrieve_from_plan(self, plan: Any, **kwargs: Any) -> list[Document]:
            return [
                Document(
                    id="cache-doc",
                    content="cacheable result",
                    metadata={"source": "media_db"},
                    source=DataSource.MEDIA_DB,
                    score=0.9,
                )
            ]

    def fake_shared_cache(**kwargs: Any) -> RecordingCache:
        namespaces.append(kwargs.get("namespace"))
        return RecordingCache()

    monkeypatch.setattr(up, "SemanticCache", FakeCacheClass)
    monkeypatch.setattr(up, "AdaptiveCache", None)
    monkeypatch.setattr(up, "get_shared_cache", fake_shared_cache)
    monkeypatch.setattr(up, "MultiDatabaseRetriever", FakeRetriever)

    common_kwargs = {
        "query": "cache namespace",
        "sources": ["media_db"],
        "media_db_path": "media.sqlite",
        "enable_generation": False,
        "enable_cache": True,
        "adaptive_cache": False,
        "enable_reranking": False,
        "enable_security_filter": False,
    }
    await up.unified_rag_pipeline(**common_kwargs)
    await up.unified_rag_pipeline(**common_kwargs, workspace_id="ws-1")

    assert len(namespaces) >= 2
    assert namespaces[0] != namespaces[1]
    assert namespaces[0] and "workspace:global" in namespaces[0]
    assert namespaces[1] and "workspace:ws-1" in namespaces[1]
