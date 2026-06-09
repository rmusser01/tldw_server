"""Tests for the database-backed retrieval helpers in the RAG service."""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import pytest

import tldw_Server_API.app.core.RAG.rag_service.database_retrievers as retr_mod
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.media_database import (
    MediaDatabase,
)
from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import (
    MediaDatabaseError,
    MediaDBRetriever,
    MultiDatabaseRetriever,
    RetrievalConfig,
    _derive_bounded_media_term_query,
)
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document


@pytest.mark.unit
class TestRetrievalConfig:
    def test_defaults_match_expected_contract(self):
        cfg = RetrievalConfig()
        assert cfg.max_results == 20
        assert cfg.min_score == 0.0
        assert cfg.use_fts is True
        assert cfg.use_vector is True
        assert cfg.include_metadata is True

    def test_custom_configuration_preserves_values(self):

        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        cfg = RetrievalConfig(
            max_results=5,
            min_score=0.25,
            use_fts=False,
            use_vector=True,
            date_filter=(start, end),
            tags_filter=["tag-a"],
        )
        assert cfg.max_results == 5
        assert cfg.min_score == 0.25
        assert cfg.use_fts is False
        assert cfg.use_vector is True
        assert cfg.date_filter == (start, end)
        assert cfg.tags_filter == ["tag-a"]


@pytest.mark.unit
class TestMediaDBRetriever:
    def test_initializes_vector_store_from_default_factory_when_type_is_unset(self, monkeypatch):
        sentinel_store = object()
        factory_calls: list[tuple[dict, str]] = []

        def fake_factory(settings_dict: dict, user_id: str):
            factory_calls.append((settings_dict, user_id))
            return sentinel_store

        monkeypatch.setattr(
            "tldw_Server_API.app.core.config.settings",
            {"RAG": {}, "USER_DB_BASE_DIR": "/tmp"},
            raising=False,
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.RAG.rag_service.database_retrievers.create_from_settings_for_user",
            fake_factory,
        )

        retriever = MediaDBRetriever(":memory:", user_id="user-7")

        assert retriever.vector_store is sentinel_store
        assert len(factory_calls) == 1
        assert factory_calls[0][1] == "user-7"

    @pytest.mark.asyncio
    async def test_vector_retrieval_uses_scoped_embedding_model_and_filter(self, monkeypatch):
        create_calls: list[dict[str, Any]] = []

        class FakeCollection:
            def __init__(self) -> None:
                self.last_get_kwargs: dict[str, Any] | None = None

            def get(self, **kwargs):
                self.last_get_kwargs = kwargs
                return {
                    "metadatas": [
                        {
                            "media_id": "10",
                            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                            "embedding_provider": "huggingface",
                        }
                    ]
                }

        class FakeManager:
            def __init__(self) -> None:
                self.collection = FakeCollection()

            def get_collection(self, _collection_name: str):
                return self.collection

        class FakeVectorStore:
            def __init__(self) -> None:
                self._initialized = True
                self.manager = FakeManager()
                self.search_calls: list[dict[str, Any]] = []

            async def initialize(self) -> None:
                self._initialized = True

            async def search(self, *, collection_name, query_vector, k, filter, include_metadata):
                self.search_calls.append(
                    {
                        "collection_name": collection_name,
                        "query_vector": query_vector,
                        "k": k,
                        "filter": filter,
                        "include_metadata": include_metadata,
                    }
                )
                return [
                    SimpleNamespace(
                        id="media_10_chunk_0",
                        content="Scoped vector match for media 10.",
                        metadata={"media_id": "10", "title": "Scoped Source"},
                        score=0.91,
                    )
                ]

        fake_vector_store = FakeVectorStore()

        def fake_factory(_settings_dict: dict, _user_id: str):
            return fake_vector_store

        def fake_create_embeddings_batch(texts, user_app_config, model_id_override=None):
            create_calls.append(
                {
                    "texts": list(texts),
                    "user_app_config": user_app_config,
                    "model_id_override": model_id_override,
                }
            )
            return [[0.12, 0.34, 0.56]]

        monkeypatch.setattr(
            "tldw_Server_API.app.core.config.settings",
            {"RAG": {}, "USER_DB_BASE_DIR": "/tmp"},
            raising=False,
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.RAG.rag_service.database_retrievers.create_from_settings_for_user",
            fake_factory,
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create.get_embedding_config",
            lambda: {"embedding_config": {"default_model_id": None, "models": {}}},
            raising=True,
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create.create_embeddings_batch",
            fake_create_embeddings_batch,
            raising=True,
        )

        retriever = MediaDBRetriever(":memory:", user_id="user-42")

        docs = await retriever._retrieve_vector(
            "what does the selected source say about evidence handling",
            allowed_media_ids=[10],
        )

        assert [doc.id for doc in docs] == ["10"]
        assert create_calls[0]["model_id_override"] == "huggingface:sentence-transformers/all-MiniLM-L6-v2"
        assert fake_vector_store.manager.collection.last_get_kwargs == {
            "where": {"media_id": "10"},
            "include": ["metadatas"],
            "limit": 5,
        }
        assert fake_vector_store.search_calls[0]["filter"] == {
            "$and": [{"media_id": "10"}, {"kind": "chunk"}]
        }

    @pytest.mark.asyncio
    async def test_vector_retrieval_falls_back_when_strict_collection_lookup_misses_scope(
        self,
        monkeypatch,
    ):
        create_calls: list[dict[str, Any]] = []

        class FakeCollection:
            def __init__(self) -> None:
                self.last_get_kwargs: dict[str, Any] | None = None

            def get(self, **kwargs):
                self.last_get_kwargs = kwargs
                return {
                    "metadatas": [
                        {
                            "media_id": "12",
                            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                            "embedding_provider": "huggingface",
                        }
                    ]
                }

        class FakeManager:
            def __init__(self) -> None:
                self.collection = FakeCollection()
                self.get_collection_calls = 0
                self.get_or_create_collection_calls = 0

            def get_collection(self, _collection_name: str):
                self.get_collection_calls += 1
                raise KeyError("Collection 'user_user-12_media_embeddings' does not exist")

            def get_or_create_collection(self, _collection_name: str):
                self.get_or_create_collection_calls += 1
                return self.collection

        class FakeVectorStore:
            def __init__(self) -> None:
                self._initialized = True
                self.manager = FakeManager()
                self.search_calls: list[dict[str, Any]] = []

            async def initialize(self) -> None:
                self._initialized = True

            async def search(self, *, collection_name, query_vector, k, filter, include_metadata):
                self.search_calls.append(
                    {
                        "collection_name": collection_name,
                        "query_vector": query_vector,
                        "k": k,
                        "filter": filter,
                        "include_metadata": include_metadata,
                    }
                )
                return [
                    SimpleNamespace(
                        id="media_12_chunk_0",
                        content="Scoped vector match for media 12.",
                        metadata={"media_id": "12", "title": "Scoped Source"},
                        score=0.88,
                    )
                ]

        fake_vector_store = FakeVectorStore()

        def fake_factory(_settings_dict: dict, _user_id: str):
            return fake_vector_store

        def fake_create_embeddings_batch(texts, user_app_config, model_id_override=None):
            create_calls.append(
                {
                    "texts": list(texts),
                    "user_app_config": user_app_config,
                    "model_id_override": model_id_override,
                }
            )
            return [[0.45, 0.67, 0.89]]

        monkeypatch.setattr(
            "tldw_Server_API.app.core.config.settings",
            {"RAG": {}, "USER_DB_BASE_DIR": "/tmp"},
            raising=False,
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.RAG.rag_service.database_retrievers.create_from_settings_for_user",
            fake_factory,
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create.get_embedding_config",
            lambda: {"embedding_config": {"default_model_id": None, "models": {}}},
            raising=True,
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create.create_embeddings_batch",
            fake_create_embeddings_batch,
            raising=True,
        )

        retriever = MediaDBRetriever(":memory:", user_id="user-12")

        docs = await retriever._retrieve_vector(
            "what does the selected source say about beta readiness",
            allowed_media_ids=[12],
        )

        assert [doc.id for doc in docs] == ["12"]
        assert create_calls[0]["model_id_override"] == "huggingface:sentence-transformers/all-MiniLM-L6-v2"
        assert fake_vector_store.manager.get_collection_calls == 1
        assert fake_vector_store.manager.get_or_create_collection_calls == 1
        assert fake_vector_store.manager.collection.last_get_kwargs == {
            "where": {"media_id": "12"},
            "include": ["metadatas"],
            "limit": 5,
        }

    @pytest.mark.asyncio
    async def test_retrieve_returns_documents_from_media_db(self, populated_media_db):
        retriever = MediaDBRetriever(str(populated_media_db.db_path))
        docs = await retriever.retrieve("retrieval")
        assert docs, "Expected at least one document for the seeded media database"
        _assert_documents(docs, expected_source=DataSource.MEDIA_DB)

    @pytest.mark.asyncio
    async def test_respects_max_results(self, populated_media_db):
        config = RetrievalConfig(max_results=1)
        retriever = MediaDBRetriever(str(populated_media_db.db_path), config=config)
        docs = await retriever.retrieve("retrieval")
        assert len(docs) == 1

    @pytest.mark.asyncio
    async def test_media_type_filter_applies(self, populated_media_db):
        retriever = MediaDBRetriever(str(populated_media_db.db_path))
        docs = await retriever.retrieve("vector", media_type="video")
        assert docs, "Fixture seeds a video entry containing the word 'vector'"
        assert all(doc.metadata.get("media_type") == "video" for doc in docs)

    @pytest.mark.asyncio
    async def test_chunk_level_retrieval_late_chunks_media_when_media_has_no_stored_chunks(
        self,
        tmp_path: Path,
    ):
        db = _create_media_db(tmp_path)
        media_id, _, _ = db.add_media_with_keywords(
            title="weakness frieza saiyans chunk doc",
            media_type="transcript",
            content=(
                "This document includes the exact fallback phrase weakness frieza saiyans."
            ),
            keywords=["frieza", "weakness", "saiyans"],
        )

        retriever = MediaDBRetriever(
            db_path=str(db.db_path),
            config=RetrievalConfig(max_results=5, use_fts=True, use_vector=False, fts_level="chunk"),
            media_db=db,
            user_id="0",
        )

        docs = await retriever.retrieve("What weakness does Frieza reveal about the Saiyans?")

        assert docs
        assert docs[0].id != str(media_id)
        assert docs[0].metadata.get("media_id") == str(media_id)
        assert docs[0].metadata.get("media_type") == "transcript"
        assert docs[0].metadata.get("chunk_index") == 0
        assert docs[0].metadata.get("retrieval_mode") == "late_chunk"

    @pytest.mark.asyncio
    async def test_media_retrieval_retries_with_bounded_terms_after_strict_query_miss(self, tmp_path: Path, monkeypatch):
        db = _create_media_db(tmp_path)
        media_id, _, _ = db.add_media_with_keywords(
            title="weakness frieza saiyans doc",
            media_type="transcript",
            content=(
                "This document includes the exact fallback phrase weakness frieza saiyans."
            ),
            keywords=["frieza", "weakness", "saiyans"],
        )
        query = "what weakness does Frieza mention about the Saiyans during the fight"
        expected_fallback_query = _derive_bounded_media_term_query(query)
        assert expected_fallback_query == "weakness OR frieza OR saiyans"

        retriever = MediaDBRetriever(
            db_path=str(db.db_path),
            config=RetrievalConfig(max_results=5, use_fts=True, use_vector=False),
            media_db=db,
            user_id="0",
        )

        calls: list[dict[str, object]] = []

        def _spy_search_media(*args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            if len(calls) == 1:
                return [], 0
            return ([{
                "id": media_id,
                "title": "Goku Lands A Devastating One Inch Punch On Frieza",
                "content": (
                    "This document includes the exact fallback phrase weakness "
                    "frieza saiyans."
                ),
                "type": "video",
                "url": None,
                "ingestion_date": None,
                "transcription_model": None,
                "last_modified": None,
                "relevance_score": 0.0,
            }], 1)

        monkeypatch.setattr(retr_mod, "search_media", _spy_search_media)

        docs = await retriever.retrieve(query)

        assert docs
        assert str(media_id) in {doc.id for doc in docs}
        assert any("weakness frieza saiyans" in doc.content.lower() for doc in docs)
        assert len(calls) == 2
        assert calls[0]["kwargs"]["search_query"] == query
        assert calls[1]["kwargs"]["search_query"] == expected_fallback_query

    @pytest.mark.asyncio
    async def test_media_retrieval_fallback_handles_question_shaped_query_against_title_terms(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        db = _create_media_db(tmp_path)
        media_id, _, _ = db.add_media_with_keywords(
            title="Goku Lands A Devastating One-Inch Punch On Frieza",
            media_type="video",
            content=(
                "Kakarot spots Frieza's weakness and lands a devastating one-inch punch."
            ),
            keywords=["goku", "frieza", "punch"],
        )
        query = "Why was goku able to land a one inch punch on frieza?"

        retriever = MediaDBRetriever(
            db_path=str(db.db_path),
            config=RetrievalConfig(max_results=5, use_fts=True, use_vector=False),
            media_db=db,
            user_id="0",
        )

        calls: list[dict[str, object]] = []
        real_search_media = retr_mod.search_media

        def _spy_search_media(*args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            if len(calls) == 1 and kwargs.get("search_query") == query:
                return [], 0
            return real_search_media(*args, **kwargs)

        monkeypatch.setattr(retr_mod, "search_media", _spy_search_media)

        docs = await retriever.retrieve(query)

        assert docs
        assert {doc.id for doc in docs} == {str(media_id)}
        first_query = calls[0]["kwargs"]["search_query"]
        if first_query == query:
            assert len(calls) == 2
            search_query = calls[1]["kwargs"]["search_query"]
        else:
            assert len(calls) == 1
            search_query = first_query
        assert isinstance(search_query, str)
        assert "goku" in search_query.lower()
        assert "frieza" in search_query.lower()

    @pytest.mark.asyncio
    async def test_media_retrieval_does_not_fallback_when_rows_exist_but_docs_are_filtered_out(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        db = _create_media_db(tmp_path)
        db.add_media_with_keywords(
            title="row-presence doc",
            media_type="transcript",
            content="This row exists but will be filtered out after retrieval.",
            keywords=["row", "presence"],
        )

        retriever = MediaDBRetriever(
            db_path=str(db.db_path),
            config=RetrievalConfig(max_results=5, use_fts=True, use_vector=False),
            media_db=db,
            user_id="0",
        )

        calls: list[dict[str, object]] = []

        def _spy_search_media(*args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            return ([{
                "id": 1,
                "title": "row-presence doc",
                "content": "This row exists but will be filtered out after retrieval.",
                "type": "transcript",
                "url": None,
                "ingestion_date": None,
                "transcription_model": None,
                "last_modified": None,
                "relevance_score": 0.0,
            }], 1)

        monkeypatch.setattr(retr_mod, "search_media", _spy_search_media)
        monkeypatch.setattr(
            MediaDBRetriever,
            "_build_media_documents",
            lambda self, results, *, backend_type: [],
        )

        docs = await retriever.retrieve("What weakness does Frieza mention about the Saiyans during the fight?")

        assert docs == []
        assert len(calls) == 1
        assert calls[0]["kwargs"]["search_query"] == "What weakness does Frieza mention about the Saiyans during the fight?"

    @pytest.mark.asyncio
    async def test_media_fallback_respects_allowed_media_ids(self, tmp_path: Path, monkeypatch):
        db = _create_media_db(tmp_path)
        allowed_media_id, allowed_uuid, _ = db.add_media_with_keywords(
            title="weakness frieza saiyans allowed doc",
            media_type="transcript",
            content=(
                "This document includes the exact fallback phrase weakness frieza saiyans."
            ),
            keywords=["frieza", "weakness", "saiyans"],
        )
        blocked_media_id, _, _ = db.add_media_with_keywords(
            title="weakness frieza saiyans blocked doc",
            media_type="transcript",
            content=(
                "This document includes the exact fallback phrase weakness frieza saiyans "
                "with blocked-only extra details."
            ),
            keywords=["frieza", "weakness", "saiyans"],
        )
        query = "What weakness does Frieza mention about the Saiyans during the fight?"
        expected_fallback_query = _derive_bounded_media_term_query(query)
        assert expected_fallback_query == "weakness OR frieza OR saiyans"

        retriever = MediaDBRetriever(
            db_path=str(db.db_path),
            config=RetrievalConfig(max_results=10, use_fts=True, use_vector=False),
            media_db=db,
            user_id="0",
        )

        calls: list[dict[str, object]] = []
        real_search_media = retr_mod.search_media

        def _spy_search_media(*args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            if len(calls) == 1:
                return [], 0
            return real_search_media(*args, **kwargs)

        monkeypatch.setattr(retr_mod, "search_media", _spy_search_media)

        docs = await retriever.retrieve(
            query,
            allowed_media_ids=[allowed_uuid],
        )

        assert {doc.id for doc in docs} == {str(allowed_media_id)}
        assert str(blocked_media_id) not in {doc.id for doc in docs}
        assert len(calls) == 2
        assert calls[0]["kwargs"]["search_query"] == query
        assert calls[1]["kwargs"]["search_query"] == expected_fallback_query
        assert calls[0]["kwargs"]["media_ids_filter"] == [allowed_uuid]
        assert calls[1]["kwargs"]["media_ids_filter"] == [allowed_uuid]

    @pytest.mark.asyncio
    async def test_chunk_level_retrieval_does_not_fall_back_when_chunk_rows_exist_but_docs_are_filtered_out(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        db = _create_media_db(tmp_path)
        db.add_media_with_keywords(
            title="Chunk row presence doc",
            media_type="transcript",
            content="Chunk row exists but will be filtered out after retrieval.",
            keywords=["chunk", "presence"],
            chunks=[
                {
                    "text": "Chunk row exists but will be filtered out after retrieval.",
                    "start_char": 0,
                    "end_char": 59,
                    "chunk_type": "text",
                    "metadata": {"speaker": "Narrator"},
                }
            ],
        )

        retriever = MediaDBRetriever(
            db_path=str(db.db_path),
            config=RetrievalConfig(max_results=5, use_fts=True, use_vector=False, fts_level="chunk"),
            media_db=db,
            user_id="0",
        )

        monkeypatch.setattr(
            MediaDBRetriever,
            "_retrieve_chunk_fts_with_stats",
            lambda self, query, media_type, **kwargs: ([], 1),
        )
        monkeypatch.setattr(
            MediaDBRetriever,
            "_retrieve_via_backend",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Media fallback should not run")),
        )

        docs = await retriever.retrieve("Chunk row exists but will be filtered out after retrieval.")

        assert docs == []

    @pytest.mark.asyncio
    async def test_chunk_level_retrieval_still_returns_chunk_documents_when_chunks_exist(self, tmp_path: Path):
        db = _create_media_db(tmp_path)
        content = "Alpha scouting report.\n\nVegeta calls out Frieza's weakness."
        start = content.index("Vegeta")
        end = start + len("Vegeta calls out Frieza's weakness.")
        media_id, _, _ = db.add_media_with_keywords(
            title="Chunked Frieza Doc",
            media_type="transcript",
            content=content,
            keywords=["frieza"],
            chunks=[
                {
                    "text": "Alpha scouting report.",
                    "start_char": 0,
                    "end_char": len("Alpha scouting report."),
                    "chunk_type": "text",
                    "metadata": {"speaker": "Narrator"},
                },
                {
                    "text": content[start:end],
                    "start_char": start,
                    "end_char": end,
                    "chunk_type": "text",
                    "metadata": {"speaker": "Vegeta"},
                }
            ],
        )
        db.ensure_chunk_fts()
        db.maybe_rebuild_chunk_fts_if_empty()

        retriever = MediaDBRetriever(
            db_path=str(db.db_path),
            config=RetrievalConfig(max_results=5, use_fts=True, use_vector=False, fts_level="chunk"),
            media_db=db,
            user_id="0",
        )

        docs = await retriever.retrieve("Vegeta", media_type="transcript")

        assert docs
        assert docs[0].metadata.get("media_id") == str(media_id)
        assert docs[0].metadata.get("chunk_index") == 1

    @pytest.mark.asyncio
    async def test_chunk_level_retrieval_prefers_transient_late_chunks_when_text_late_chunking_enabled(
        self,
        tmp_path: Path,
    ):
        db = _create_media_db(tmp_path)
        content = (
            "Goku closes the distance.\n\n"
            "Frieza drops his guard and Goku lands a one inch punch to the chest."
        )
        media_id, _, _ = db.add_media_with_keywords(
            title="Chunked Goku Frieza Doc",
            media_type="transcript",
            content=content,
            keywords=["goku", "frieza", "punch"],
            chunks=[
                {
                    "text": "Stored alpha chunk.",
                    "start_char": 0,
                    "end_char": len("Stored alpha chunk."),
                    "chunk_type": "text",
                    "metadata": {"speaker": "Narrator"},
                },
                {
                    "text": "Stored Frieza chunk that should be bypassed.",
                    "start_char": 20,
                    "end_char": 64,
                    "chunk_type": "text",
                    "metadata": {"speaker": "Narrator"},
                },
            ],
        )
        db.ensure_chunk_fts()
        db.maybe_rebuild_chunk_fts_if_empty()

        before_chunk_count = db.execute_query(
            "SELECT COUNT(*) FROM UnvectorizedMediaChunks WHERE media_id = ?",
            (media_id,),
        ).fetchone()[0]

        retriever = MediaDBRetriever(
            db_path=str(db.db_path),
            config=RetrievalConfig(
                max_results=5,
                use_fts=True,
                use_vector=False,
                fts_level="chunk",
                enable_text_late_chunking=True,
            ),
            media_db=db,
            user_id="0",
        )

        docs = await retriever.retrieve(
            "Why was goku able to land a one inch punch on frieza?",
            media_type="transcript",
        )

        after_chunk_count = db.execute_query(
            "SELECT COUNT(*) FROM UnvectorizedMediaChunks WHERE media_id = ?",
            (media_id,),
        ).fetchone()[0]

        assert docs
        assert docs[0].id.startswith(f"late_chunk:{media_id}:")
        assert docs[0].metadata.get("retrieval_mode") == "late_chunk"
        assert docs[0].metadata.get("media_id") == str(media_id)
        assert before_chunk_count == 2
        assert after_chunk_count == before_chunk_count

    @pytest.mark.asyncio
    async def test_chunk_level_retrieval_uses_custom_text_late_chunking_knobs(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        db = _create_media_db(tmp_path)
        content = (
            "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu. "
            "Nu xi omicron pi rho sigma tau upsilon phi chi psi omega."
        )
        media_id, _, _ = db.add_media_with_keywords(
            title="Chunked Knob Control Doc",
            media_type="transcript",
            content=content,
            keywords=["alpha", "omega"],
            chunks=[
                {
                    "text": "Persisted chunk that should be bypassed.",
                    "start_char": 0,
                    "end_char": 39,
                    "chunk_type": "text",
                    "metadata": {"speaker": "Narrator"},
                },
            ],
        )
        db.ensure_chunk_fts()
        db.maybe_rebuild_chunk_fts_if_empty()

        captured: dict[str, object] = {}

        from tldw_Server_API.app.core.Chunking import chunker as chunker_mod

        real_chunk_text_hierarchical_flat = chunker_mod.Chunker.chunk_text_hierarchical_flat

        def _spy_chunk_text_hierarchical_flat(self, text, *, method, max_size, overlap, language=None, template=None):
            captured.update(
                {
                    "method": method,
                    "max_size": max_size,
                    "overlap": overlap,
                    "language": language,
                }
            )
            return real_chunk_text_hierarchical_flat(
                self,
                text,
                method=method,
                max_size=max_size,
                overlap=overlap,
                language=language,
                template=template,
            )

        monkeypatch.setattr(chunker_mod.Chunker, "chunk_text_hierarchical_flat", _spy_chunk_text_hierarchical_flat)

        retriever = MediaDBRetriever(
            db_path=str(db.db_path),
            config=RetrievalConfig(
                max_results=5,
                use_fts=True,
                use_vector=False,
                fts_level="chunk",
                enable_text_late_chunking=True,
                chunk_method="words",
                chunk_size=6,
                chunk_overlap=2,
                chunk_language="en",
            ),
            media_db=db,
            user_id="0",
        )

        docs = await retriever.retrieve("alpha omega", media_type="transcript")

        assert docs
        assert docs[0].id.startswith(f"late_chunk:{media_id}:")
        assert captured == {
            "method": "words",
            "max_size": 6,
            "overlap": 2,
            "language": "en",
        }

    @pytest.mark.asyncio
    async def test_chunk_level_late_chunking_ranks_typo_query_to_matching_media_entity(
        self,
        tmp_path: Path,
    ):
        db = _create_media_db(tmp_path)
        target_media_id, _, _ = db.add_media_with_keywords(
            title="Goku Lands A Devastating One-Inch Punch On Frieza",
            media_type="video",
            content=(
                "Frieza boasts about his new golden form. "
                "Goku explains that the new form is burning through more power than "
                "Frieza can supply because he rushed into the fight before he was "
                "used to regulating it."
            ),
            keywords=["goku", "frieza", "golden", "form", "power"],
        )
        db.add_media_with_keywords(
            title="Generic issue review",
            media_type="video",
            content=(
                "This review discusses an issue with a new form and generic product feedback."
            ),
            keywords=["issue", "new", "form"],
        )
        db.add_media_with_keywords(
            title="Another form explainer",
            media_type="video",
            content=(
                "A broad explainer covering issue triage and alternate form handling."
            ),
            keywords=["issue", "form", "handling"],
        )

        retriever = MediaDBRetriever(
            db_path=str(db.db_path),
            config=RetrievalConfig(
                max_results=5,
                min_score=0.2,
                use_fts=True,
                use_vector=False,
                fts_level="chunk",
                enable_text_late_chunking=True,
            ),
            media_db=db,
            user_id="0",
        )

        docs = await retriever.retrieve("what was the issue with friezes new form")

        assert docs
        assert docs[0].metadata.get("media_id") == str(target_media_id)
        assert docs[0].metadata.get("retrieval_mode") == "late_chunk"
        assert "burning through more power" in docs[0].content.lower()

    @pytest.mark.asyncio
    async def test_chunk_level_retrieval_respects_uuid_allowed_media_ids_without_falling_back(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        db = _create_media_db(tmp_path)
        content = "Alpha scouting report.\n\nVegeta calls out Frieza's weakness."
        start = content.index("Vegeta")
        end = start + len("Vegeta calls out Frieza's weakness.")
        media_id, media_uuid, _ = db.add_media_with_keywords(
            title="Chunked Frieza UUID Doc",
            media_type="transcript",
            content=content,
            keywords=["frieza"],
            chunks=[
                {
                    "text": "Alpha scouting report.",
                    "start_char": 0,
                    "end_char": len("Alpha scouting report."),
                    "chunk_type": "text",
                    "metadata": {"speaker": "Narrator"},
                },
                {
                    "text": content[start:end],
                    "start_char": start,
                    "end_char": end,
                    "chunk_type": "text",
                    "metadata": {"speaker": "Vegeta"},
                },
            ],
        )
        db.ensure_chunk_fts()
        db.maybe_rebuild_chunk_fts_if_empty()

        retriever = MediaDBRetriever(
            db_path=str(db.db_path),
            config=RetrievalConfig(max_results=5, use_fts=True, use_vector=False, fts_level="chunk"),
            media_db=db,
            user_id="0",
        )

        monkeypatch.setattr(
            MediaDBRetriever,
            "_retrieve_via_backend",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Media fallback should not run")),
        )

        docs = await retriever.retrieve("Vegeta", allowed_media_ids=[media_uuid])

        assert docs
        assert all(doc.metadata.get("media_id") == str(media_id) for doc in docs)
        assert all(doc.metadata.get("chunk_index") == 1 for doc in docs)
        assert all(doc.id != str(media_id) for doc in docs)

    def test_attach_media_db_uses_shared_factory(self, monkeypatch, tmp_path):
        import tldw_Server_API.app.core.RAG.rag_service.database_retrievers as retr_mod

        events: list[tuple[str, object]] = []

        class _FakeDb:
            def close_connection(self):
                events.append(("close", None))

        def _fake_create_media_database(client_id, **kwargs):
            events.append(("create", client_id))
            events.append(("kwargs", kwargs))
            return _FakeDb()

        monkeypatch.setattr(retr_mod, "create_media_database", _fake_create_media_database)
        monkeypatch.setattr(retr_mod.MediaDBRetriever, "_initialize_vector_store", lambda self: None)

        retriever = retr_mod.MediaDBRetriever(str(tmp_path / "media.db"))

        try:
            assert retriever._own_media_db is True
            assert isinstance(retriever.media_db, _FakeDb)
            assert retriever._db_adapter is retriever.media_db
            assert events[:2] == [
                ("create", "rag_service"),
                ("kwargs", {"db_path": str((tmp_path / "media.db").resolve())}),
            ]
        finally:
            retriever.close()

        assert ("close", None) in events

    def test_attach_media_db_propagates_factory_configuration_errors(self, monkeypatch, tmp_path):
        import tldw_Server_API.app.core.RAG.rag_service.database_retrievers as retr_mod

        monkeypatch.setattr(
            retr_mod,
            "create_media_database",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(MediaDatabaseError("bad config")),
        )
        monkeypatch.setattr(retr_mod.MediaDBRetriever, "_initialize_vector_store", lambda self: None)

        with pytest.raises(MediaDatabaseError):
            retr_mod.MediaDBRetriever(str(tmp_path / "media.db"))


@pytest.mark.unit
class TestMultiDatabaseRetriever:
    @pytest.mark.asyncio
    async def test_retrieve_uses_media_db_when_present(self, populated_media_db):
        db_paths = {"media_db": str(populated_media_db.db_path)}
        retriever = MultiDatabaseRetriever(db_paths, user_id="test-user")
        config = RetrievalConfig(max_results=3, use_fts=True, use_vector=False, include_metadata=True)
        docs = await retriever.retrieve("retrieval", sources=[DataSource.MEDIA_DB], config=config)
        assert docs
        _assert_documents(docs, expected_source=DataSource.MEDIA_DB)
        scores = [doc.score for doc in docs]
        assert scores == sorted(scores, reverse=True)
        assert len(docs) <= 3

    @pytest.mark.asyncio
    async def test_retrieve_ignores_unknown_sources(self, populated_media_db):
        db_paths = {"media_db": str(populated_media_db.db_path)}
        retriever = MultiDatabaseRetriever(db_paths, user_id="test-user")
        docs = await retriever.retrieve("retrieval", sources=["nonexistent_source"])  # type: ignore[arg-type]
        assert docs == []

    @pytest.mark.asyncio
    async def test_retrieve_uses_media_adapter_without_sqlite_path(self, monkeypatch):
        fake_media_db = SimpleNamespace(backend_type=BackendType.POSTGRESQL)
        search_calls: list[tuple[object, dict[str, object]]] = []

        monkeypatch.setattr(retr_mod.MediaDBRetriever, "_initialize_vector_store", lambda self: None)

        def _fake_search_media(media_db, **kwargs):
            search_calls.append((media_db, kwargs))
            return (
                [
                    {
                        "id": 101,
                        "title": "Adapter-backed media",
                        "content": "retrieval through adapter only",
                        "type": "article",
                        "url": "https://example.com/item",
                        "ingestion_date": "2025-01-01T00:00:00",
                        "transcription_model": None,
                        "last_modified": "2025-01-02T00:00:00",
                        "rank": 0.9,
                    }
                ],
                1,
            )

        monkeypatch.setattr(retr_mod, "search_media", _fake_search_media)

        retriever = retr_mod.MultiDatabaseRetriever({}, user_id="test-user", media_db=fake_media_db)
        config = retr_mod.RetrievalConfig(
            max_results=3,
            use_fts=True,
            use_vector=False,
            include_metadata=True,
        )

        docs = await retriever.retrieve("adapter", sources=[DataSource.MEDIA_DB], config=config)

        assert docs
        assert docs[0].id == "101"
        assert docs[0].source == DataSource.MEDIA_DB
        assert docs[0].metadata["source"] == "media_db"
        assert search_calls and search_calls[0][0] is fake_media_db


@pytest.mark.unit
def test_claims_retriever_attach_uses_shared_factory(monkeypatch, tmp_path):
    import tldw_Server_API.app.core.RAG.rag_service.database_retrievers as retr_mod

    events: list[tuple[str, object]] = []

    class _FakeDb:
        def close_connection(self):
            events.append(("close", None))

    def _fake_create_media_database(client_id, **kwargs):
        events.append(("create", client_id))
        events.append(("kwargs", kwargs))
        return _FakeDb()

    monkeypatch.setattr(retr_mod, "create_media_database", _fake_create_media_database)

    retriever = retr_mod.ClaimsRetriever(str(tmp_path / "claims.db"))

    try:
        assert retriever._own_media_db is True
        assert isinstance(retriever.media_db, _FakeDb)
        assert retriever._db_adapter is retriever.media_db
        assert events[:2] == [
            ("create", "rag_service"),
            ("kwargs", {"db_path": str((tmp_path / "claims.db").resolve())}),
        ]
    finally:
        retriever.close()

    assert ("close", None) in events


@pytest.mark.unit
def test_claims_retriever_attach_propagates_factory_configuration_errors(monkeypatch, tmp_path):
    import tldw_Server_API.app.core.RAG.rag_service.database_retrievers as retr_mod

    monkeypatch.setattr(
        retr_mod,
        "create_media_database",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad path")),
    )

    with pytest.raises(ValueError):
        retr_mod.ClaimsRetriever(str(tmp_path / "claims.db"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_documents(docs: Iterable[Document], *, expected_source: DataSource) -> None:
    for doc in docs:
        assert isinstance(doc, Document)
        assert doc.source == expected_source
        assert isinstance(doc.content, str) and doc.content
        assert isinstance(doc.metadata, dict)
        assert isinstance(doc.score, float)


def _create_media_db(tmp_path: Path) -> MediaDatabase:
    db_path = tmp_path / "media.db"
    db = MediaDatabase(db_path=str(db_path), client_id="pytest")
    db.initialize_db()
    return db
