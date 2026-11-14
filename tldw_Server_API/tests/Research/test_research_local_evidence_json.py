"""Real local-corpus collection must preserve database metadata in JSON artifacts."""

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
async def test_collecting_persists_real_backend_evidence(tmp_path, monkeypatch, request, backend):
    from dataclasses import asdict
    from functools import partial

    from tldw_Server_API.app.core.DB_Management.media_db.api import create_media_database, get_media_repository
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
    from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import MultiDatabaseRetriever
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
    from tldw_Server_API.app.core.Research.broker import ResearchBroker
    from tldw_Server_API.app.core.Research.jobs import handle_research_phase_job
    from tldw_Server_API.app.core.Research.models import ResearchPlan
    from tldw_Server_API.app.core.Research.providers import local

    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")
    path = tmp_path / "media.db"
    monkeypatch.setattr(local.DatabasePaths, "get_media_db_path", lambda user: path)
    db_backend = request.getfixturevalue("pg_restricted_backend") if backend == "postgres" else None
    db = ResearchSessionsDB(tmp_path / "research.db")
    question = "Rowan Observatory: identify the telescope, Friday tour time, dome opening time and visitor entrance. Cite the local sources. /no_think"
    session = db.create_session(
        owner_user_id="2", query=question, source_policy="local_only", autonomy_mode="checkpointed", limits_json={}
    )
    db.update_phase(session.id, phase="collecting", status="queued", active_job_id="2")
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)
    plan = ResearchPlan(
        query=question,
        focus_areas=["rowan evidence review"],
        source_policy="local_only",
        autonomy_mode="checkpointed",
        stop_criteria={"min_cited_sections": 1},
    )
    for name, payload in (
        ("plan.json", asdict(plan)),
        ("provider_config.json", {"local": {"sources": ["media_db"], "top_k": 2, "use_vector": False}}),
    ):
        store.write_json(
            owner_user_id="2",
            session_id=session.id,
            artifact_name=name,
            payload=payload,
            phase="collecting",
            job_id=None,
        )
    with scoped_context(user_id=2, is_admin=False):
        media = create_media_database(client_id="2", db_path=path, backend=db_backend)
        try:
            get_media_repository(media).add_media_with_keywords(
                url="research://rowan",
                title="Rowan source",
                media_type="document",
                content="Rowan Observatory telescope Selene, Friday tours at18:00, dome17:30 and east gate.",
                keywords=[],
                overwrite=False,
            )
            provider = local.LocalResearchProvider(retriever_cls=partial(MultiDatabaseRetriever, media_db=media))
            result = await handle_research_phase_job(
                {"id": 2, "payload": {"session_id": session.id, "phase": "collecting"}},
                research_db_path=tmp_path / "research.db",
                outputs_dir=tmp_path / "outputs",
                broker=ResearchBroker(local_provider=provider),
            )
            assert result["phase"] == "awaiting_source_review"  # nosec B101 # Real collection must persist its source registry and reach review.
            sources = store.read_json(session_id=session.id, artifact_name="source_registry.json")["sources"]
            assert len(sources) == 1 and "Selene" in sources[0]["snippet"]  # nosec B101 # Persisted evidence retains the actual backend content.
        finally:
            media.close_connection()


def test_local_provider_preserves_json_metadata_values():
    import json
    from datetime import date, datetime, timezone
    from decimal import Decimal
    from uuid import UUID

    from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document
    from tldw_Server_API.app.core.Research.providers.local import LocalResearchProvider

    metadata = {
        "created_at": datetime(2026, 10, 30, 1, 2, 3, tzinfo=timezone.utc),
        "nested": {
            "day": date(2026, 10, 30),
            "amount": Decimal("1.20"),
            "id": UUID("12345678-1234-1234-1234-123456789abc"),
        },
    }
    document = Document(id="1", content="Selene", source=DataSource.MEDIA_DB, metadata=metadata, score=1.0)
    result = LocalResearchProvider._normalize_document(document)
    assert json.loads(json.dumps(result))["metadata"] == {
        "created_at": "2026-10-30T01:02:03Z",
        "nested": {"day": "2026-10-30", "amount": "1.20", "id": "12345678-1234-1234-1234-123456789abc"},
    }  # nosec B101 # Provider evidence preserves supported metadata values in its JSON contract.
    assert document.metadata is metadata and isinstance(metadata["created_at"], datetime)  # nosec B101 # Shared retrieval metadata must remain typed and unchanged.


def test_local_provider_rejects_nonserializable_metadata():
    from pydantic_core import PydanticSerializationError

    from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document
    from tldw_Server_API.app.core.Research.providers.local import LocalResearchProvider

    document = Document(
        id="1", content="Selene", source=DataSource.MEDIA_DB, metadata={"unsupported": object()}, score=1.0
    )
    with pytest.raises(PydanticSerializationError):
        LocalResearchProvider._normalize_document(document)
