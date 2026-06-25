from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.RPG_DB import RPGRepository
from tldw_Server_API.app.core.RPG.context import SessionContextBuilder
from tldw_Server_API.app.core.RPG.models import RPGSnapshotState
from tldw_Server_API.app.core.RPG.rules.lookup import RulesLookupService
from tldw_Server_API.app.core.RPG.service import RPGService


def _service() -> RPGService:
    repo = RPGRepository.initialized(CharactersRAGDB(":memory:", "rpg-rules-context-test"))
    return RPGService(repo=repo, owner_user_id=42)


def test_rules_lookup_returns_citations_without_pf2e_prose():
    lookup = RulesLookupService()

    result = lookup.lookup(adapter_key="pf2e", query="dying condition", linked_rules_pack_refs=[])

    assert result.query == "dying condition"  # nosec B101
    assert result.results  # nosec B101
    assert all(item.text == "" for item in result.results)  # nosec B101
    assert all(item.citation.adapter_key == "pf2e" for item in result.results)  # nosec B101
    assert result.diagnostics["bundled_policy"] == "citations_only"  # nosec B101


def test_context_builder_includes_snapshot_rule_citations_and_budget():
    lookup = RulesLookupService()
    rule_result = lookup.lookup(adapter_key="fate", query="stress", linked_rules_pack_refs=[])
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


def test_service_lookup_rules_and_context_are_owner_scoped():
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

    lookup = service.lookup_rules(session.id, query="dying condition")
    context = service.build_context(session.id, query="dying condition", max_chars=1000)

    assert lookup.results  # nosec B101
    assert lookup.results[0].citation.adapter_key == "pf2e"  # nosec B101
    assert "Ruins under moonlight" in context.text  # nosec B101
    assert "Archives of Nethys" in context.text  # nosec B101


def test_service_context_clamps_tiny_budget_for_non_rest_callers():
    service = _service()
    campaign = service.create_campaign("Campaign", None, "fate", idempotency_key="campaign-clamp")
    session = service.create_session(
        campaign.id,
        "Opening",
        adapter_key="fate",
        idempotency_key="session-clamp",
    )

    context = service.build_context(session.id, query="stress", max_chars=1)

    assert context.diagnostics["max_chars"] == 1000  # nosec B101
