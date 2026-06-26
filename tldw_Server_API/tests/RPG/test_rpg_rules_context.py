import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.RPG_DB import RPGRepository
from tldw_Server_API.app.core.RPG.context import SessionContextBuilder
from tldw_Server_API.app.core.RPG.errors import RPGValidationError
from tldw_Server_API.app.core.RPG.models import RPGSnapshotState
from tldw_Server_API.app.core.RPG.rules.content_packs import RuleLookupCitation, RuleLookupItem, RuleLookupResult
from tldw_Server_API.app.core.RPG.rules.lookup import RulesLookupService
from tldw_Server_API.app.core.RPG.rules.retrieval import RulesRetrievalResult
from tldw_Server_API.app.core.RPG.service import RPGService

pytestmark = pytest.mark.unit


def _service() -> RPGService:
    repo = RPGRepository.initialized(CharactersRAGDB(":memory:", "rpg-rules-context-test"))
    return RPGService(repo=repo, owner_user_id=42)


class FakeLookupRetriever:
    def __init__(self, result: RulesRetrievalResult) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    async def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class FailingLookupRetriever:
    def __init__(self, message: str) -> None:
        self.message = message

    async def retrieve(self, **kwargs):
        raise RuntimeError(self.message)


class FakeRulesLookupService:
    def __init__(self, result: RuleLookupResult | None = None, exc: Exception | None = None) -> None:
        self.result = result
        self.exc = exc
        self.calls: list[dict[str, object]] = []

    async def lookup(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return self.result


def _user_lookup_item(snippet_id: str = "media:42:chunk:7") -> RuleLookupItem:
    return RuleLookupItem(
        origin="user_provided",
        text="User rules say this applies.",
        citation=RuleLookupCitation(
            source_type="media_item",
            source_id=42,
            source_title="Player Rules",
            source_url=None,
            license=None,
            license_url=None,
            attribution=None,
            trust_level="user_provided",
            content_hash="sha256:abc",
            snippet_id=snippet_id,
        ),
        score=0.91,
    )


@pytest.mark.asyncio
async def test_rules_lookup_returns_citations_without_pf2e_prose():
    lookup = RulesLookupService()

    result = await lookup.lookup(
        owner_user_id=42,
        adapter_key="pf2e",
        query="dying condition",
        linked_rules_pack_refs=[],
    )

    assert result.query == "dying condition"  # nosec B101
    assert result.results  # nosec B101
    assert all(item.text == "" for item in result.results)  # nosec B101
    assert all(item.origin == "bundled_citation" for item in result.results)  # nosec B101
    assert all(item.citation.source_type == "bundled_rules_citation" for item in result.results)  # nosec B101
    assert result.diagnostics["bundled_policy"] == "citations_only"  # nosec B101
    assert result.answer_status == "not_requested"  # nosec B101


@pytest.mark.asyncio
async def test_context_builder_includes_snapshot_rule_citations_and_budget():
    lookup = RulesLookupService()
    rule_result = await lookup.lookup(owner_user_id=42, adapter_key="fate", query="stress", linked_rules_pack_refs=[])
    snapshot = RPGSnapshotState(
        scene={"summary": "Rain at the old docks"},
        npcs={"npc-1": {"npc_id": "npc-1", "name": "Ada"}},
        notes=[{"note_id": "n1", "text": "The ferry bell rings twice."}],
        unresolved_rulings={"r1": {"ruling_id": "r1", "question": "How does stress clear?"}},
    )

    context = SessionContextBuilder(max_chars=1000).build(
        adapter_key="fate",
        session_title="Opening",
        snapshot=snapshot,
        rules_results=rule_result.results,
    )

    assert "Opening" in context.text  # nosec B101
    assert "Rain at the old docks" in context.text  # nosec B101
    assert "Ada" in context.text  # nosec B101
    assert "Fate SRD" in context.text  # nosec B101
    assert context.diagnostics["truncated"] is False  # nosec B101
    assert context.diagnostics["rules_result_count"] == len(rule_result.results)  # nosec B101

    truncated = SessionContextBuilder(max_chars=80).build(
        adapter_key="fate",
        session_title="Opening",
        snapshot=snapshot,
        rules_results=rule_result.results,
    )

    assert len(truncated.text) <= 80  # nosec B101
    assert truncated.diagnostics["truncated"] is True  # nosec B101
    assert not truncated.text.endswith("https:/")  # nosec B101


@pytest.mark.asyncio
async def test_service_lookup_rules_and_context_are_owner_scoped():
    service = _service()
    campaign = service.create_campaign("Campaign", None, "pf2e", idempotency_key="campaign-rules")
    session = service.create_session(
        campaign.id,
        "Opening",
        adapter_key="pf2e",
        idempotency_key="session-rules",
    )
    service.record_events(
        session_id=session.id,
        events=[
            {
                "event_type": "scene.updated",
                "event_payload": {"scene_id": "scene-1", "summary": "Ruins under moonlight"},
            }
        ],
        source_type="user",
        expected_last_event_sequence=0,
        idempotency_key="rules-scene-1",
    )

    lookup = await service.lookup_rules(session.id, query="dying condition")
    context = await service.build_context(session.id, query="dying condition", max_chars=1000)

    assert lookup.results  # nosec B101
    assert lookup.results[0].citation.source_title == "Archives of Nethys Pathfinder 2e"  # nosec B101
    assert "Ruins under moonlight" in context.text  # nosec B101
    assert "Archives of Nethys" in context.text  # nosec B101


@pytest.mark.asyncio
async def test_service_context_clamps_tiny_budget_for_non_rest_callers():
    service = _service()
    campaign = service.create_campaign("Campaign", None, "fate", idempotency_key="campaign-clamp")
    session = service.create_session(
        campaign.id,
        "Opening",
        adapter_key="fate",
        idempotency_key="session-clamp",
    )

    context = await service.build_context(session.id, query="stress", max_chars=1)

    assert context.diagnostics["max_chars"] == 1000  # nosec B101


@pytest.mark.asyncio
async def test_service_context_includes_lookup_diagnostics_and_uses_lookup_mode():
    rule_result = RuleLookupResult(
        query="stress",
        mode="lookup",
        results=[_user_lookup_item()],
        answer=None,
        answer_status="not_requested",
        answer_citation_ids=[],
        diagnostics={"retrieval_result_count": 1, "skipped_refs": [{"ref_id": "media:9", "reason": "disabled"}]},
    )
    lookup = FakeRulesLookupService(rule_result)
    repo = RPGRepository.initialized(CharactersRAGDB(":memory:", "rpg-context-diagnostics-test"))
    service = RPGService(repo=repo, owner_user_id=42, rules_lookup_service=lookup)
    campaign = service.create_campaign("Campaign", None, "fate", idempotency_key="campaign-context-diag")
    session = service.create_session(campaign.id, "Opening", "fate", idempotency_key="session-context-diag")

    context = await service.build_context(session.id, query="stress", max_chars=1000)

    assert "User rules say this applies." in context.text  # nosec B101
    assert lookup.calls[0]["mode"] == "lookup"  # nosec B101
    assert context.diagnostics["rules_lookup"]["retrieval_result_count"] == 1  # nosec B101
    assert context.diagnostics["rules_lookup"]["skipped_refs"] == [{"ref_id": "media:9", "reason": "disabled"}]  # nosec B101


@pytest.mark.asyncio
async def test_service_context_continues_when_rules_lookup_validation_fails():
    lookup = FakeRulesLookupService(exc=RPGValidationError("rules_pack_source_unreadable"))
    repo = RPGRepository.initialized(CharactersRAGDB(":memory:", "rpg-context-lookup-error-test"))
    service = RPGService(repo=repo, owner_user_id=42, rules_lookup_service=lookup)
    campaign = service.create_campaign("Campaign", None, "fate", idempotency_key="campaign-context-error")
    session = service.create_session(campaign.id, "Opening", "fate", idempotency_key="session-context-error")
    service.record_events(
        session_id=session.id,
        events=[{"event_type": "scene.updated", "event_payload": {"scene_id": "s1", "summary": "Still playable"}}],
        source_type="user",
        expected_last_event_sequence=0,
        idempotency_key="context-error-scene",
    )

    context = await service.build_context(session.id, query="stress", max_chars=1000)

    assert "Still playable" in context.text  # nosec B101
    assert context.diagnostics["rules_lookup"]["lookup_error"] == "rules_pack_source_unreadable"  # nosec B101
    assert context.diagnostics["rules_result_count"] == 0  # nosec B101


@pytest.mark.asyncio
async def test_lookup_returns_user_results_before_bundled_citations():
    retriever = FakeLookupRetriever(
        RulesRetrievalResult(
            items=[_user_lookup_item()],
            ready_media_ids=[42],
            skipped_refs=[],
            diagnostics={"retrieval_result_count": 1},
        )
    )
    lookup = RulesLookupService(retriever=retriever)

    result = await lookup.lookup(
        owner_user_id=42,
        adapter_key="fate",
        query="stress",
        linked_rules_pack_refs=[{"source_type": "media_item", "source_id": 42}],
    )

    assert result.results[0].origin == "user_provided"  # nosec B101
    assert result.results[1].origin == "bundled_citation"  # nosec B101
    assert result.diagnostics["retrieval_result_count"] == 1  # nosec B101


@pytest.mark.asyncio
async def test_answer_mode_marks_retrieved_evidence_as_not_generated():
    retriever = FakeLookupRetriever(
        RulesRetrievalResult(
            items=[_user_lookup_item()],
            ready_media_ids=[42],
            skipped_refs=[],
            diagnostics={"retrieval_result_count": 1},
        )
    )
    lookup = RulesLookupService(retriever=retriever)

    result = await lookup.lookup(
        owner_user_id=42,
        adapter_key="fate",
        query="stress",
        linked_rules_pack_refs=[{"source_type": "media_item", "source_id": 42}],
        mode="answer",
    )

    assert result.results[0].origin == "user_provided"  # nosec B101
    assert result.answer is None  # nosec B101
    assert result.answer_status == "not_generated"  # nosec B101


@pytest.mark.asyncio
async def test_lookup_keeps_bundled_citations_score_zero():
    lookup = RulesLookupService()

    result = await lookup.lookup(owner_user_id=42, adapter_key="pf2e", query="dying", linked_rules_pack_refs=[])

    bundled = [item for item in result.results if item.origin == "bundled_citation"]
    assert bundled  # nosec B101
    assert all(item.score == 0.0 for item in bundled)  # nosec B101


@pytest.mark.asyncio
async def test_lookup_returns_diagnostics_for_skipped_refs():
    retriever = FakeLookupRetriever(
        RulesRetrievalResult(
            items=[],
            ready_media_ids=[],
            skipped_refs=[{"ref_id": "media_collection:5", "reason": "no_ready_media"}],
            diagnostics={"broad_fallback_used": False},
        )
    )
    lookup = RulesLookupService(retriever=retriever)

    result = await lookup.lookup(
        owner_user_id=42,
        adapter_key="fate",
        query="stress",
        linked_rules_pack_refs=[{"source_type": "media_collection", "source_id": 5}],
    )

    assert result.diagnostics["skipped_refs"] == [  # nosec B101
        {"ref_id": "media_collection:5", "reason": "no_ready_media"}
    ]
    assert result.diagnostics["broad_fallback_used"] is False  # nosec B101


@pytest.mark.asyncio
async def test_lookup_redacts_unexpected_retrieval_errors_from_diagnostics():
    lookup = RulesLookupService(retriever=FailingLookupRetriever("/private/db/path failed"))

    result = await lookup.lookup(
        owner_user_id=42,
        adapter_key="fate",
        query="stress",
        linked_rules_pack_refs=[{"source_type": "media_item", "source_id": 42}],
    )

    assert result.results  # nosec B101
    assert result.diagnostics["retrieval_error"] == "retrieval_failed"  # nosec B101
    assert "/private/db/path" not in str(result.diagnostics)  # nosec B101


@pytest.mark.asyncio
async def test_lookup_redacts_unexpected_retrieval_errors_from_warning_log(monkeypatch):
    from tldw_Server_API.app.core.RPG.rules import lookup as lookup_module

    logged: list[tuple[str, tuple[object, ...]]] = []

    def fake_warning(message, *args):
        logged.append((str(message), args))

    monkeypatch.setattr(lookup_module.logger, "warning", fake_warning)
    lookup = RulesLookupService(retriever=FailingLookupRetriever("/private/db/path failed"))

    await lookup.lookup(
        owner_user_id=42,
        adapter_key="fate",
        query="stress",
        linked_rules_pack_refs=[{"source_type": "media_item", "source_id": 42}],
    )

    assert logged  # nosec B101
    assert "/private/db/path" not in str(logged)  # nosec B101


@pytest.mark.asyncio
async def test_lookup_does_not_call_retriever_when_query_is_blank():
    retriever = FakeLookupRetriever(RulesRetrievalResult(items=[], ready_media_ids=[], skipped_refs=[], diagnostics={}))
    lookup = RulesLookupService(retriever=retriever)

    with pytest.raises(RPGValidationError, match="rules_query_required"):
        await lookup.lookup(
            owner_user_id=42,
            adapter_key="fate",
            query=" ",
            linked_rules_pack_refs=[{"source_type": "media_item", "source_id": 42}],
        )

    assert retriever.calls == []  # nosec B101
