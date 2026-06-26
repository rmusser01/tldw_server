import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.RPG_DB import RPGRepository
from tldw_Server_API.app.core.RPG.errors import RPGValidationError
from tldw_Server_API.app.core.RPG.rules.refs import RulesPackSourceValidation
from tldw_Server_API.app.core.RPG.service import RPGService


class FakeRulesSourceValidator:
    def __init__(self, readable: bool = True) -> None:
        self.readable = readable
        self.media_item_calls: list[tuple[int, int]] = []
        self.media_collection_calls: list[tuple[int, int]] = []

    async def validate_media_item(self, owner_user_id: int, media_id: int) -> RulesPackSourceValidation:
        self.media_item_calls.append((owner_user_id, media_id))
        return RulesPackSourceValidation(
            ref_id=f"media_item:{media_id}",
            readable=self.readable,
            display_name=f"Media {media_id}",
        )

    async def validate_media_collection(
        self,
        owner_user_id: int,
        collection_id: int,
    ) -> RulesPackSourceValidation:
        self.media_collection_calls.append((owner_user_id, collection_id))
        return RulesPackSourceValidation(
            ref_id=f"media_collection:{collection_id}",
            readable=self.readable,
            display_name=f"Collection {collection_id}",
            ready_media_ids=[],
        )


def _service(rules_source_validator: FakeRulesSourceValidator | None = None) -> RPGService:
    repo = RPGRepository.initialized(CharactersRAGDB(":memory:", "rpg-service-test"))
    return RPGService(repo=repo, owner_user_id=42, rules_source_validator=rules_source_validator)


def test_model_events_create_pending_proposal_by_default():
    service = _service()
    campaign = service.create_campaign("Campaign", None, "fate", idempotency_key="campaign-model")
    session = service.create_session(
        campaign.id,
        "Opening",
        adapter_key="fate",
        idempotency_key="session-model",
    )

    result = service.record_events(
        session_id=session.id,
        events=[{"event_type": "note.added", "event_payload": {"note_id": "n1", "text": "Suggested"}}],
        source_type="model",
        expected_last_event_sequence=0,
        idempotency_key="model-1",
    )

    assert result.committed_events == []  # nosec B101
    assert result.proposal is not None  # nosec B101
    assert result.proposal.status == "pending"  # nosec B101
    assert service.get_snapshot(session.id).snapshot_version == 0  # nosec B101


def test_user_events_commit_and_update_snapshot():
    service = _service()
    campaign = service.create_campaign("Campaign", None, "fate", idempotency_key="campaign-user")
    session = service.create_session(
        campaign.id,
        "Opening",
        adapter_key="fate",
        idempotency_key="session-user",
    )

    result = service.record_events(
        session_id=session.id,
        events=[{"event_type": "note.added", "event_payload": {"note_id": "n1", "text": "Observed"}}],
        source_type="user",
        expected_last_event_sequence=0,
        idempotency_key="user-1",
    )

    snapshot = service.get_snapshot(session.id)

    assert [event.sequence_number for event in result.committed_events] == [1]  # nosec B101
    assert snapshot.snapshot_version == 1  # nosec B101
    assert snapshot.last_event_sequence == 1  # nosec B101
    assert snapshot.snapshot.notes[0]["text"] == "Observed"  # nosec B101


def test_applying_proposal_commits_events_and_advances_snapshot_once():
    service = _service()
    campaign = service.create_campaign("Campaign", None, "fate", idempotency_key="campaign-proposal")
    session = service.create_session(
        campaign.id,
        "Opening",
        adapter_key="fate",
        idempotency_key="session-proposal",
    )
    proposal = service.record_events(
        session_id=session.id,
        events=[
            {"event_type": "npc.upserted", "event_payload": {"npc_id": "npc-1", "name": "Ada"}},
            {"event_type": "quest.upserted", "event_payload": {"quest_id": "q1", "title": "Find Ada"}},
        ],
        source_type="model",
        expected_last_event_sequence=0,
        idempotency_key="model-2",
    ).proposal

    assert proposal is not None  # nosec B101

    applied = service.apply_proposal(
        session_id=session.id,
        proposal_id=proposal.id,
        expected_last_event_sequence=0,
        idempotency_key="proposal-apply-1",
        review_notes="accepted",
    )

    snapshot = service.get_snapshot(session.id)

    assert [event.sequence_number for event in applied.committed_events] == [1, 2]  # nosec B101
    assert snapshot.snapshot_version == 1  # nosec B101
    assert snapshot.last_event_sequence == 2  # nosec B101
    assert snapshot.snapshot.npcs["npc-1"]["name"] == "Ada"  # nosec B101
    assert snapshot.snapshot.quests["q1"]["title"] == "Find Ada"  # nosec B101


def test_replaying_proposal_apply_returns_events_without_new_snapshot():
    service = _service()
    campaign = service.create_campaign("Campaign", None, "fate", idempotency_key="campaign-replay")
    session = service.create_session(
        campaign.id,
        "Opening",
        adapter_key="fate",
        idempotency_key="session-replay",
    )
    proposal = service.record_events(
        session_id=session.id,
        events=[{"event_type": "note.added", "event_payload": {"note_id": "n1", "text": "Accepted"}}],
        source_type="model",
        expected_last_event_sequence=0,
        idempotency_key="model-replay",
    ).proposal

    assert proposal is not None  # nosec B101

    first = service.apply_proposal(
        session_id=session.id,
        proposal_id=proposal.id,
        expected_last_event_sequence=0,
        idempotency_key="proposal-apply-replay",
    )
    second = service.apply_proposal(
        session_id=session.id,
        proposal_id=proposal.id,
        expected_last_event_sequence=0,
        idempotency_key="proposal-apply-replay",
    )
    snapshot = service.get_snapshot(session.id)

    assert [event.id for event in second.committed_events] == [
        event.id for event in first.committed_events
    ]  # nosec B101
    assert snapshot.snapshot_version == 1  # nosec B101
    assert snapshot.snapshot.notes == [{"note_id": "n1", "text": "Accepted"}]  # nosec B101


def test_replaying_proposal_reject_returns_same_rejected_proposal():
    service = _service()
    campaign = service.create_campaign("Campaign", None, "fate", idempotency_key="campaign-reject")
    session = service.create_session(
        campaign.id,
        "Opening",
        adapter_key="fate",
        idempotency_key="session-reject",
    )
    proposal = service.record_events(
        session_id=session.id,
        events=[{"event_type": "note.added", "event_payload": {"note_id": "n1", "text": "Nope"}}],
        source_type="model",
        expected_last_event_sequence=0,
        idempotency_key="model-reject",
    ).proposal

    assert proposal is not None  # nosec B101

    first = service.reject_proposal(
        session_id=session.id,
        proposal_id=proposal.id,
        idempotency_key="proposal-reject-replay",
        review_notes="not now",
    )
    second = service.reject_proposal(
        session_id=session.id,
        proposal_id=proposal.id,
        idempotency_key="proposal-reject-replay",
        review_notes="not now",
    )

    snapshot = service.get_snapshot(session.id)

    assert first.id == proposal.id  # nosec B101
    assert second.id == first.id  # nosec B101
    assert second.status == "rejected"  # nosec B101
    assert snapshot.snapshot_version == 0  # nosec B101


def test_create_session_copies_campaign_rules_refs_when_request_omits_refs():
    service = _service()
    campaign = service.create_campaign("Campaign", None, "fate", idempotency_key="campaign-copy-rules")
    service.repo.replace_campaign_rules_pack_refs(
        owner_user_id=42,
        campaign_id=campaign.id,
        rules_pack_refs=[{"source_type": "media_item", "source_id": 7, "display_name": "Rules"}],
        expected_version=campaign.version,
        idempotency_key="campaign-copy-rules-ref",
        request_payload_hash="arranged",
        source_type="user",
    )

    session = service.create_session(
        campaign.id,
        "Opening",
        adapter_key="fate",
        idempotency_key="session-copy-rules",
    )

    assert session.active_rules_pack_refs[0]["ref_id"] == "media_item:7"  # nosec B101


def test_create_session_replays_omitted_rules_refs_after_campaign_refs_change():
    service = _service()
    campaign = service.create_campaign("Campaign", None, "fate", idempotency_key="campaign-replay-omitted-rules")
    service.repo.replace_campaign_rules_pack_refs(
        owner_user_id=42,
        campaign_id=campaign.id,
        rules_pack_refs=[{"source_type": "media_item", "source_id": 7}],
        expected_version=campaign.version,
        idempotency_key="campaign-replay-omitted-rules-ref-1",
        request_payload_hash="arranged-1",
        source_type="user",
    )

    first = service.create_session(
        campaign.id,
        "Opening",
        adapter_key="fate",
        idempotency_key="session-replay-omitted-rules",
    )
    service.repo.replace_campaign_rules_pack_refs(
        owner_user_id=42,
        campaign_id=campaign.id,
        rules_pack_refs=[{"source_type": "media_item", "source_id": 8}],
        expected_version=2,
        idempotency_key="campaign-replay-omitted-rules-ref-2",
        request_payload_hash="arranged-2",
        source_type="user",
    )

    second = service.create_session(
        campaign.id,
        "Opening",
        adapter_key="fate",
        idempotency_key="session-replay-omitted-rules",
    )

    assert second.id == first.id  # nosec B101
    assert [ref["ref_id"] for ref in second.active_rules_pack_refs] == ["media_item:7"]  # nosec B101


def test_create_session_uses_explicit_empty_rules_refs():
    service = _service()
    campaign = service.create_campaign("Campaign", None, "fate", idempotency_key="campaign-empty-rules")
    service.repo.replace_campaign_rules_pack_refs(
        owner_user_id=42,
        campaign_id=campaign.id,
        rules_pack_refs=[{"source_type": "media_item", "source_id": 7, "display_name": "Rules"}],
        expected_version=campaign.version,
        idempotency_key="campaign-empty-rules-ref",
        request_payload_hash="arranged",
        source_type="user",
    )

    session = service.create_session(
        campaign.id,
        "Opening",
        adapter_key="fate",
        idempotency_key="session-empty-rules",
        active_rules_pack_refs=[],
    )

    assert session.active_rules_pack_refs == []  # nosec B101


def test_create_session_validates_explicit_rules_refs():
    validator = FakeRulesSourceValidator()
    service = _service(validator)
    campaign = service.create_campaign("Campaign", None, "fate", idempotency_key="campaign-explicit-rules")

    session = service.create_session(
        campaign.id,
        "Opening",
        adapter_key="fate",
        idempotency_key="session-explicit-rules",
        active_rules_pack_refs=[{"source_type": "media_item", "source_id": 7}],
    )

    assert validator.media_item_calls == [(42, 7)]  # nosec B101
    assert session.active_rules_pack_refs[0]["ref_id"] == "media_item:7"  # nosec B101


def test_create_session_replays_explicit_rules_refs_before_source_validation():
    validator = FakeRulesSourceValidator()
    service = _service(validator)
    campaign = service.create_campaign("Campaign", None, "fate", idempotency_key="campaign-explicit-replay")
    refs = [{"source_type": "media_item", "source_id": 7}]

    first = service.create_session(
        campaign.id,
        "Opening",
        adapter_key="fate",
        idempotency_key="session-explicit-replay",
        active_rules_pack_refs=refs,
    )
    validator.readable = False
    second = service.create_session(
        campaign.id,
        "Opening",
        adapter_key="fate",
        idempotency_key="session-explicit-replay",
        active_rules_pack_refs=refs,
    )

    assert second.id == first.id  # nosec B101
    assert [ref["ref_id"] for ref in second.active_rules_pack_refs] == ["media_item:7"]  # nosec B101
    assert validator.media_item_calls == [(42, 7)]  # nosec B101


@pytest.mark.asyncio
async def test_replace_campaign_rules_pack_refs_validates_each_enabled_source():
    validator = FakeRulesSourceValidator()
    service = _service(validator)
    campaign = service.create_campaign("Campaign", None, "fate", idempotency_key="campaign-validate-rules")

    result = await service.replace_campaign_rules_pack_refs(
        campaign.id,
        [
            {"source_type": "media_item", "source_id": 7},
            {"source_type": "media_collection", "source_id": 3},
        ],
        expected_version=campaign.version,
        idempotency_key="campaign-validate-rules-ref",
    )

    assert validator.media_item_calls == [(42, 7)]  # nosec B101
    assert validator.media_collection_calls == [(42, 3)]  # nosec B101
    assert [ref.ref_id for ref in result.refs] == ["media_item:7", "media_collection:3"]  # nosec B101


@pytest.mark.asyncio
async def test_replace_campaign_rules_pack_refs_replays_before_source_validation():
    validator = FakeRulesSourceValidator()
    service = _service(validator)
    campaign = service.create_campaign("Campaign", None, "fate", idempotency_key="campaign-replay-rules")
    refs = [{"source_type": "media_item", "source_id": 7}]

    first = await service.replace_campaign_rules_pack_refs(
        campaign.id,
        refs,
        expected_version=campaign.version,
        idempotency_key="campaign-replay-rules-ref",
    )
    validator.readable = False
    second = await service.replace_campaign_rules_pack_refs(
        campaign.id,
        refs,
        expected_version=campaign.version,
        idempotency_key="campaign-replay-rules-ref",
    )

    assert first.replayed is False  # nosec B101
    assert second.replayed is True  # nosec B101
    assert [ref.ref_id for ref in second.refs] == ["media_item:7"]  # nosec B101
    assert validator.media_item_calls == [(42, 7)]  # nosec B101


@pytest.mark.asyncio
async def test_replace_session_rules_pack_refs_validates_each_enabled_source():
    validator = FakeRulesSourceValidator()
    service = _service(validator)
    campaign = service.create_campaign("Campaign", None, "fate", idempotency_key="campaign-session-validate")
    session = service.create_session(
        campaign.id,
        "Opening",
        adapter_key="fate",
        idempotency_key="session-validate-rules",
    )

    result = await service.replace_session_rules_pack_refs(
        session.id,
        [
            {"source_type": "media_item", "source_id": 7},
            {"source_type": "media_collection", "source_id": 3},
        ],
        expected_version=session.version,
        idempotency_key="session-validate-rules-ref",
    )

    assert validator.media_item_calls == [(42, 7)]  # nosec B101
    assert validator.media_collection_calls == [(42, 3)]  # nosec B101
    assert [ref.ref_id for ref in result.refs] == ["media_item:7", "media_collection:3"]  # nosec B101


@pytest.mark.asyncio
async def test_replace_session_rules_pack_refs_replays_before_source_validation():
    validator = FakeRulesSourceValidator()
    service = _service(validator)
    campaign = service.create_campaign("Campaign", None, "fate", idempotency_key="campaign-session-replay")
    session = service.create_session(
        campaign.id,
        "Opening",
        adapter_key="fate",
        idempotency_key="session-replay-rules",
    )
    refs = [{"source_type": "media_collection", "source_id": 3}]

    first = await service.replace_session_rules_pack_refs(
        session.id,
        refs,
        expected_version=session.version,
        idempotency_key="session-replay-rules-ref",
    )
    validator.readable = False
    second = await service.replace_session_rules_pack_refs(
        session.id,
        refs,
        expected_version=session.version,
        idempotency_key="session-replay-rules-ref",
    )

    assert first.replayed is False  # nosec B101
    assert second.replayed is True  # nosec B101
    assert [ref.ref_id for ref in second.refs] == ["media_collection:3"]  # nosec B101
    assert validator.media_collection_calls == [(42, 3)]  # nosec B101


@pytest.mark.asyncio
async def test_replace_rules_pack_refs_allows_disabled_unreadable_source_without_dereference():
    validator = FakeRulesSourceValidator(readable=False)
    service = _service(validator)
    campaign = service.create_campaign("Campaign", None, "fate", idempotency_key="campaign-disabled-rules")

    result = await service.replace_campaign_rules_pack_refs(
        campaign.id,
        [{"source_type": "media_item", "source_id": 7, "enabled": False}],
        expected_version=campaign.version,
        idempotency_key="campaign-disabled-rules-ref",
    )

    assert validator.media_item_calls == []  # nosec B101
    assert result.refs[0].enabled is False  # nosec B101


@pytest.mark.asyncio
async def test_replace_rules_pack_refs_rejects_unreadable_media_item():
    validator = FakeRulesSourceValidator(readable=False)
    service = _service(validator)
    campaign = service.create_campaign("Campaign", None, "fate", idempotency_key="campaign-unreadable-rules")

    with pytest.raises(RPGValidationError, match="rules_pack_source_unreadable"):
        await service.replace_campaign_rules_pack_refs(
            campaign.id,
            [{"source_type": "media_item", "source_id": 7}],
            expected_version=campaign.version,
            idempotency_key="campaign-unreadable-rules-ref",
        )

    assert validator.media_item_calls == [(42, 7)]  # nosec B101


@pytest.mark.asyncio
async def test_replace_rules_pack_refs_allows_empty_readable_collection():
    validator = FakeRulesSourceValidator()
    service = _service(validator)
    campaign = service.create_campaign("Campaign", None, "fate", idempotency_key="campaign-empty-collection")

    result = await service.replace_campaign_rules_pack_refs(
        campaign.id,
        [{"source_type": "media_collection", "source_id": 3}],
        expected_version=campaign.version,
        idempotency_key="campaign-empty-collection-ref",
    )

    assert validator.media_collection_calls == [(42, 3)]  # nosec B101
    assert result.refs[0].ref_id == "media_collection:3"  # nosec B101


def test_list_campaign_rules_pack_refs_returns_current_version():
    service = _service()
    campaign = service.create_campaign("Campaign", None, "fate", idempotency_key="campaign-list-rules")
    service.repo.replace_campaign_rules_pack_refs(
        owner_user_id=42,
        campaign_id=campaign.id,
        rules_pack_refs=[{"source_type": "media_item", "source_id": 7}],
        expected_version=campaign.version,
        idempotency_key="campaign-list-rules-ref",
        request_payload_hash="arranged",
        source_type="user",
    )

    result = service.list_campaign_rules_pack_refs(campaign.id)

    assert result.version == 2  # nosec B101
    assert result.refs[0].ref_id == "media_item:7"  # nosec B101
    assert result.replayed is False  # nosec B101


def test_list_session_rules_pack_refs_returns_current_version():
    service = _service()
    campaign = service.create_campaign("Campaign", None, "fate", idempotency_key="campaign-list-session-rules")
    session = service.create_session(
        campaign.id,
        "Opening",
        adapter_key="fate",
        idempotency_key="session-list-rules",
    )
    service.repo.replace_session_rules_pack_refs(
        owner_user_id=42,
        session_id=session.id,
        rules_pack_refs=[{"source_type": "media_collection", "source_id": 3}],
        expected_version=session.version,
        idempotency_key="session-list-rules-ref",
        request_payload_hash="arranged",
        source_type="user",
    )

    result = service.list_session_rules_pack_refs(session.id)

    assert result.version == 2  # nosec B101
    assert result.refs[0].ref_id == "media_collection:3"  # nosec B101
    assert result.replayed is False  # nosec B101
