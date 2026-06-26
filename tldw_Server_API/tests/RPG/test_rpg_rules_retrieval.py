from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from tldw_Server_API.app.core.RAG.rag_service.evidence_models import RetrievedEvidence
from tldw_Server_API.app.core.RAG.rag_service.types import Document
from tldw_Server_API.app.core.RPG.rules.refs import RulesPackRef, RulesPackSourceValidation
from tldw_Server_API.app.core.RPG.rules.retrieval import RulesRetrievalAdapter


def _ref(source_type: str, source_id: int, *, enabled: bool = True) -> RulesPackRef:
    now = datetime(2026, 6, 25, tzinfo=timezone.utc)
    return RulesPackRef(
        ref_id=f"{source_type}:{source_id}",
        source_type=source_type,  # type: ignore[arg-type]
        source_id=source_id,
        display_name=f"{source_type} {source_id}",
        enabled=enabled,
        created_at=now,
        updated_at=now,
        metadata={},
    )


class FakeSourceValidator:
    def __init__(self, validations: dict[str, RulesPackSourceValidation]) -> None:
        self.validations = validations
        self.calls: list[tuple[str, int, int]] = []

    async def validate_media_item(self, owner_user_id: int, media_id: int) -> RulesPackSourceValidation:
        self.calls.append(("media_item", owner_user_id, media_id))
        return self.validations[f"media_item:{media_id}"]

    async def validate_media_collection(
        self,
        owner_user_id: int,
        collection_id: int,
    ) -> RulesPackSourceValidation:
        self.calls.append(("media_collection", owner_user_id, collection_id))
        return self.validations[f"media_collection:{collection_id}"]


class FakeRetrievalExecutor:
    def __init__(self, documents: list[Document] | None = None) -> None:
        self.documents = documents or []
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> RetrievedEvidence:
        self.calls.append(kwargs)
        return RetrievedEvidence(documents=self.documents)


@pytest.mark.asyncio
async def test_retrieval_skips_disabled_refs():
    validator = FakeSourceValidator({})
    executor = FakeRetrievalExecutor()
    retriever = RulesRetrievalAdapter(
        source_validator=validator,
        rag_retriever=object(),
        retrieval_executor=executor,
    )

    result = await retriever.retrieve(
        owner_user_id=42,
        query="advantage",
        refs=[_ref("media_item", 7, enabled=False)],
        max_results=5,
    )

    assert validator.calls == []  # nosec B101
    assert executor.calls == []  # nosec B101
    assert result.ready_media_ids == []  # nosec B101
    assert result.diagnostics["broad_fallback_used"] is False  # nosec B101
    assert result.skipped_refs == [{"ref_id": "media_item:7", "reason": "disabled"}]  # nosec B101


@pytest.mark.asyncio
async def test_retrieval_resolves_media_item_to_allowed_media_ids():
    validator = FakeSourceValidator(
        {
            "media_item:11": RulesPackSourceValidation(
                ref_id="media_item:11",
                readable=True,
                display_name="Rules Notes",
                ready_media_ids=[11],
            )
        }
    )
    executor = FakeRetrievalExecutor()
    retriever = RulesRetrievalAdapter(
        source_validator=validator,
        rag_retriever=object(),
        retrieval_executor=executor,
    )

    result = await retriever.retrieve(owner_user_id=42, query="rules", refs=[_ref("media_item", 11)], max_results=3)

    assert result.ready_media_ids == [11]  # nosec B101
    assert validator.calls == [("media_item", 42, 11)]  # nosec B101


@pytest.mark.asyncio
async def test_retrieval_resolves_collection_ready_items_only():
    validator = FakeSourceValidator(
        {
            "media_collection:5": RulesPackSourceValidation(
                ref_id="media_collection:5",
                readable=True,
                display_name="Collection",
                ready_media_ids=[9, 9, 12],
            )
        }
    )
    executor = FakeRetrievalExecutor()
    retriever = RulesRetrievalAdapter(
        source_validator=validator,
        rag_retriever=object(),
        retrieval_executor=executor,
    )

    result = await retriever.retrieve(
        owner_user_id=42,
        query="rules",
        refs=[_ref("media_collection", 5)],
        max_results=3,
    )

    assert result.ready_media_ids == [9, 12]  # nosec B101
    assert validator.calls == [("media_collection", 42, 5)]  # nosec B101


@pytest.mark.asyncio
async def test_retrieval_reports_empty_collection_without_error():
    validator = FakeSourceValidator(
        {
            "media_collection:5": RulesPackSourceValidation(
                ref_id="media_collection:5",
                readable=True,
                display_name="Empty",
                ready_media_ids=[],
            )
        }
    )
    executor = FakeRetrievalExecutor()
    retriever = RulesRetrievalAdapter(
        source_validator=validator,
        rag_retriever=object(),
        retrieval_executor=executor,
    )

    result = await retriever.retrieve(
        owner_user_id=42,
        query="rules",
        refs=[_ref("media_collection", 5)],
        max_results=3,
    )

    assert result.items == []  # nosec B101
    assert result.ready_media_ids == []  # nosec B101
    assert executor.calls == []  # nosec B101
    assert result.skipped_refs == [{"ref_id": "media_collection:5", "reason": "no_ready_media"}]  # nosec B101


@pytest.mark.asyncio
async def test_retrieval_reports_no_ready_sources_without_broad_fallback():
    validator = FakeSourceValidator({})
    executor = FakeRetrievalExecutor()
    retriever = RulesRetrievalAdapter(
        source_validator=validator,
        rag_retriever=object(),
        retrieval_executor=executor,
    )

    result = await retriever.retrieve(owner_user_id=42, query="rules", refs=[], max_results=3)

    assert result.items == []  # nosec B101
    assert executor.calls == []  # nosec B101
    assert result.diagnostics["broad_fallback_used"] is False  # nosec B101
    assert result.diagnostics["no_ready_sources"] is True  # nosec B101


@pytest.mark.asyncio
async def test_retrieval_passes_allowed_media_ids_to_executor():
    validator = FakeSourceValidator(
        {
            "media_item:9": RulesPackSourceValidation(
                ref_id="media_item:9",
                readable=True,
                display_name="Rules",
                ready_media_ids=[9],
            )
        }
    )
    executor = FakeRetrievalExecutor()
    retriever = RulesRetrievalAdapter(
        source_validator=validator,
        rag_retriever=object(),
        retrieval_executor=executor,
    )

    await retriever.retrieve(owner_user_id=42, query="rules", refs=[_ref("media_item", 9)], max_results=2)

    assert len(executor.calls) == 1  # nosec B101
    assert executor.calls[0]["allowed_media_ids"] == [9]  # nosec B101
    assert executor.calls[0]["allowed_note_ids"] is None  # nosec B101
    assert executor.calls[0]["retrieval_plan"].sources == ("media_db",)  # nosec B101


@pytest.mark.asyncio
async def test_retrieval_maps_documents_to_user_provided_lookup_items():
    document = Document(
        id="doc-1",
        content="Rules snippet",
        metadata={
            "media_id": 12,
            "title": "Player Rules",
            "source_url": "https://example.test/rules",
            "content_hash": "sha256:abc",
            "snippet_id": "custom-snippet",
        },
        score=0.82,
        chunk_index=3,
    )
    validator = FakeSourceValidator(
        {
            "media_item:12": RulesPackSourceValidation(
                ref_id="media_item:12",
                readable=True,
                display_name="Player Rules",
                ready_media_ids=[12],
            )
        }
    )
    executor = FakeRetrievalExecutor([document])
    retriever = RulesRetrievalAdapter(
        source_validator=validator,
        rag_retriever=object(),
        retrieval_executor=executor,
    )

    result = await retriever.retrieve(owner_user_id=42, query="rules", refs=[_ref("media_item", 12)], max_results=2)

    item = result.items[0]
    assert item.origin == "user_provided"  # nosec B101
    assert item.text == "Rules snippet"  # nosec B101
    assert item.score == 0.82  # nosec B101
    assert item.citation.source_type == "media_item"  # nosec B101
    assert item.citation.source_id == 12  # nosec B101
    assert item.citation.source_title == "Player Rules"  # nosec B101
    assert item.citation.source_url == "https://example.test/rules"  # nosec B101
    assert item.citation.trust_level == "user_provided"  # nosec B101
    assert item.citation.content_hash == "sha256:abc"  # nosec B101
    assert item.citation.snippet_id == "custom-snippet"  # nosec B101


@pytest.mark.asyncio
async def test_retrieval_drops_documents_outside_ready_media_scope():
    documents = [
        Document(id="12", content="Allowed rules snippet", metadata={"title": "Allowed"}, score=0.9),
        Document(id="99", content="Unscoped rules snippet", metadata={"title": "Other"}, score=0.8),
    ]
    validator = FakeSourceValidator(
        {
            "media_item:12": RulesPackSourceValidation(
                ref_id="media_item:12",
                readable=True,
                display_name="Allowed",
                ready_media_ids=[12],
            )
        }
    )
    executor = FakeRetrievalExecutor(documents)
    retriever = RulesRetrievalAdapter(
        source_validator=validator,
        rag_retriever=object(),
        retrieval_executor=executor,
    )

    result = await retriever.retrieve(owner_user_id=42, query="rules", refs=[_ref("media_item", 12)], max_results=5)

    assert [item.citation.source_id for item in result.items] == [12]  # nosec B101
    assert result.items[0].text == "Allowed rules snippet"  # nosec B101


@pytest.mark.asyncio
async def test_retrieval_uses_document_id_for_media_level_results_without_media_metadata():
    document = Document(
        id="12",
        content="Media-level rules snippet",
        metadata={"title": "Player Rules"},
        score=0.5,
    )
    validator = FakeSourceValidator(
        {
            "media_item:12": RulesPackSourceValidation(
                ref_id="media_item:12",
                readable=True,
                display_name="Player Rules",
                ready_media_ids=[12],
            )
        }
    )
    executor = FakeRetrievalExecutor([document])
    retriever = RulesRetrievalAdapter(
        source_validator=validator,
        rag_retriever=object(),
        retrieval_executor=executor,
    )

    result = await retriever.retrieve(owner_user_id=42, query="rules", refs=[_ref("media_item", 12)], max_results=2)

    item = result.items[0]
    assert item.citation.source_id == 12  # nosec B101
    assert item.citation.snippet_id == "media:12:chunk:unknown"  # nosec B101


@pytest.mark.asyncio
async def test_retrieval_uses_stable_snippet_ids():
    document = Document(
        id="doc-1",
        content="Rules snippet",
        metadata={"media_id": 12, "title": "Player Rules"},
        score=0.5,
        chunk_index=4,
    )
    validator = FakeSourceValidator(
        {
            "media_item:12": RulesPackSourceValidation(
                ref_id="media_item:12",
                readable=True,
                display_name="Player Rules",
                ready_media_ids=[12],
            )
        }
    )
    executor = FakeRetrievalExecutor([document])
    retriever = RulesRetrievalAdapter(
        source_validator=validator,
        rag_retriever=object(),
        retrieval_executor=executor,
    )

    result = await retriever.retrieve(owner_user_id=42, query="rules", refs=[_ref("media_item", 12)], max_results=2)

    assert result.items[0].citation.snippet_id == "media:12:chunk:4"  # nosec B101
