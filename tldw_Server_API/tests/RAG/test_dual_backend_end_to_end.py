from __future__ import annotations

import uuid

import pytest

from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import (
    ClaimsRetriever,
    MediaDBRetriever,
    NotesDBRetriever,
    RetrievalConfig,
)

from tldw_Server_API.tests.RAG.conftest import DualBackendEnv


def _seed_media(env: DualBackendEnv) -> int:
    media_id, _media_uuid, _content_hash = env.media_db.add_media_with_keywords(
        title="Dual Backend Coverage",
        media_type="text",
        content="This media item validates backend parity for claims and FTS.",
        keywords=["parity", "postgres"],
    )
    assert media_id is not None

    env.media_db.upsert_claims(
        [
            {
                "media_id": media_id,
                "chunk_index": 0,
                "claim_text": "Backend parity ensures claims retrieval works the same on Postgres and SQLite.",
                "chunk_hash": f"chunk-{uuid.uuid4()}",
                "extractor": "pytest",
                "extractor_version": "v1",
            }
        ]
    )
    # Ensure the FTS views are populated identically across backends
    env.media_db.rebuild_claims_fts()

    return media_id


def _seed_notes(env: DualBackendEnv) -> None:
    keyword_id = env.chacha_db.add_keyword("parity")
    assert keyword_id is not None

    note_id = env.chacha_db.add_note(
        title="Dual Backend Note",
        content="Notes retrieval must maintain backend parity across Postgres and SQLite deployments.",
    )
    assert note_id is not None

    env.chacha_db.link_note_to_keyword(note_id, int(keyword_id))
    env.chacha_db.rebuild_full_text_indexes()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_dual_backend_media_and_claims_retrieval(dual_backend_env: DualBackendEnv) -> None:
    env = dual_backend_env
    _seed_media(env)

    # Claims FTS/API parity
    stored_claims = env.media_db.search_claims("backend parity")
    assert stored_claims, f"Expected stored claims for backend {env.label}"

    claims_retriever = ClaimsRetriever(
        db_path=env.media_db.db_path_str,
        config=RetrievalConfig(max_results=5),
        media_db=env.media_db,
    )
    retrieved_claims = await claims_retriever.retrieve("backend parity")
    assert retrieved_claims, f"Claims retriever returned no results for {env.label}"
    assert any("parity" in doc.content.lower() for doc in retrieved_claims)

    media_retriever = MediaDBRetriever(
        db_path=env.media_db.db_path_str,
        config=RetrievalConfig(max_results=5, use_vector=False),
        media_db=env.media_db,
    )
    media_docs = await media_retriever.retrieve("backend parity")
    assert media_docs, f"Media retriever returned no results for {env.label}"
    assert any("coverage" in doc.content.lower() for doc in media_docs)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_dual_backend_notes_retrieval(dual_backend_env: DualBackendEnv) -> None:
    env = dual_backend_env
    _seed_notes(env)

    notes_retriever = NotesDBRetriever(
        db_path=env.chacha_db.db_path_str,
        config=RetrievalConfig(max_results=5),
        chacha_db=env.chacha_db,
    )

    documents = await notes_retriever.retrieve("backend parity")
    assert documents, f"Notes retriever returned no results for {env.label}"
    top_doc = documents[0]
    assert "Dual Backend Note" in top_doc.metadata.get("title", "")
    assert "parity" in top_doc.content.lower()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_dual_backend_hybrid_merge_with_controlled_vector_results(
    dual_backend_env: DualBackendEnv,
    deterministic_embeddings,
) -> None:
    """Hybrid MERGE logic over a real FTS backend + a controlled vector input.

    Scope note (audits/2026-07-04-test-suite-audit-round2.md, RA1): the real
    DB and FTS retrieval path IS exercised here (both backends via
    ``dual_backend_env``). The vector-similarity layer is deliberately
    controlled — exercising real ChromaDB similarity is a separate concern —
    so the vector store returns a known record and this test verifies the
    hybrid combiner merges FTS + vector results correctly, not vector recall.
    """
    env = dual_backend_env
    media_id = _seed_media(env)

    media_record = env.media_db.get_media_by_id(media_id)
    assert media_record is not None

    class _ControlledVectorResult:
        """A single known vector hit — stand-in for real similarity output."""

        def __init__(self, record, score: float) -> None:
            record_id = record.get("uuid") or record.get("id")
            self.id = str(record_id)
            self.content = record.get("content", "")
            self.metadata = {
                "title": record.get("title"),
                "media_type": record.get("type"),
                "url": record.get("url"),
                "media_id": record.get("id"),
                "source": "media_db",
            }
            self.score = score

    class _ControlledVectorStore:
        """Returns one known record so the MERGE, not vector recall, is tested."""

        def __init__(self, record) -> None:
            self._record = record
            self._initialized = False

        async def initialize(self) -> None:
            self._initialized = True

        async def search(self, *_, **__) -> list[_ControlledVectorResult]:
            return [_ControlledVectorResult(self._record, score=0.95)]

    media_retriever = MediaDBRetriever(
        db_path=env.media_db.db_path_str,
        config=RetrievalConfig(max_results=5, use_vector=True, use_fts=True),
        media_db=env.media_db,
    )
    media_retriever.vector_store = _ControlledVectorStore(media_record)

    vector_docs = await media_retriever._retrieve_vector("backend parity")
    assert vector_docs, f"Controlled vector input returned no results for {env.label}"
    assert vector_docs[0].metadata.get("media_id") == media_record["id"]

    # The real FTS path must contribute independently of the controlled vector hit.
    fts_docs = await media_retriever.retrieve("backend parity")
    assert fts_docs, f"Real FTS retrieval returned no results for {env.label}"

    hybrid_docs = await media_retriever.retrieve_hybrid("backend parity")
    assert hybrid_docs, f"Hybrid merge returned no results for {env.label}"
    assert any(doc.id == vector_docs[0].id for doc in hybrid_docs)
    assert all(doc.metadata.get("source") == "media_db" for doc in hybrid_docs)
