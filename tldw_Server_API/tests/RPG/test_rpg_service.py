from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.RPG_DB import RPGRepository
from tldw_Server_API.app.core.RPG.service import RPGService


def _service() -> RPGService:
    repo = RPGRepository.initialized(CharactersRAGDB(":memory:", "rpg-service-test"))
    return RPGService(repo=repo, owner_user_id=42)


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
