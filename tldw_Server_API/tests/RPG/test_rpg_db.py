import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.RPG_DB import RPGRepository
from tldw_Server_API.app.core.RPG.errors import RPGConflictError


def _repo() -> RPGRepository:
    return RPGRepository.initialized(CharactersRAGDB(":memory:", "rpg-test-client"))


def _campaign(repo: RPGRepository, owner_user_id: int = 42, adapter_key: str = "fate"):
    return repo.create_campaign(
        owner_user_id=owner_user_id,
        title="Campaign",
        description=None,
        default_adapter_key=adapter_key,
        default_adapter_version="1.0.0",
        settings={},
        linked_rules_pack_refs=[],
        idempotency_key=f"campaign-{adapter_key}",
        request_payload_hash=f"hash-campaign-{adapter_key}",
        source_type="user",
    )


def _session(repo: RPGRepository, campaign_id: int, owner_user_id: int = 42, adapter_key: str = "fate"):
    return repo.create_session(
        owner_user_id=owner_user_id,
        campaign_id=campaign_id,
        title="Opening",
        adapter_key=adapter_key,
        adapter_version="1.0.0",
        authority_settings={"model_auto_commit": False},
        linked_chat_id=None,
        active_rules_pack_refs=[],
        idempotency_key=f"session-{campaign_id}-{adapter_key}",
        request_payload_hash=f"hash-session-{campaign_id}-{adapter_key}",
        source_type="user",
    )


def test_repository_creates_campaign_session_and_initial_snapshot():
    repo = _repo()

    campaign = repo.create_campaign(
        owner_user_id=42,
        title="Saltmarsh",
        description="Coastal trouble",
        default_adapter_key="dnd5e_srd",
        default_adapter_version="1.0.0",
        settings={"tone": "nautical"},
        linked_rules_pack_refs=[{"media_id": 7}],
        idempotency_key="campaign-saltmarsh",
        request_payload_hash="hash-campaign-saltmarsh",
        source_type="user",
    )
    session = repo.create_session(
        owner_user_id=42,
        campaign_id=campaign.id,
        title="Session 1",
        adapter_key="dnd5e_srd",
        adapter_version="1.0.0",
        authority_settings={"model_auto_commit": False},
        linked_chat_id=None,
        active_rules_pack_refs=[{"media_id": 7}],
        idempotency_key="session-1",
        request_payload_hash="hash-session-1",
        source_type="user",
    )

    snapshot = repo.get_latest_snapshot(owner_user_id=42, session_id=session.id)

    assert campaign.settings["tone"] == "nautical"  # nosec B101
    assert session.campaign_id == campaign.id  # nosec B101
    assert session.last_event_sequence == 0  # nosec B101
    assert session.current_snapshot_version == 0  # nosec B101
    assert snapshot.snapshot_version == 0  # nosec B101
    assert snapshot.last_event_sequence == 0  # nosec B101
    assert snapshot.snapshot_json["notes"] == []  # nosec B101


def test_create_session_replays_same_idempotency_key_without_duplicate_session():
    repo = _repo()
    campaign = _campaign(repo)

    first = _session(repo, campaign.id)
    second = _session(repo, campaign.id)

    assert second.id == first.id  # nosec B101
    assert repo.get_latest_snapshot(owner_user_id=42, session_id=second.id).snapshot_version == 0  # nosec B101


def test_commit_events_assigns_sequences_and_updates_snapshot_cursor():
    repo = _repo()
    campaign = _campaign(repo)
    session = _session(repo, campaign.id)

    result = repo.commit_events_and_snapshot(
        owner_user_id=42,
        session_id=session.id,
        expected_last_event_sequence=0,
        base_snapshot_version=0,
        events=[
            {
                "event_type": "scene.updated",
                "event_payload": {"scene_id": "scene-start", "summary": "At the docks"},
                "source_type": "user",
            },
            {
                "event_type": "note.added",
                "event_payload": {"note_id": "note-1", "text": "Storm clouds gather"},
                "source_type": "user",
            },
        ],
        snapshot={
            "scene": {"scene_id": "scene-start", "summary": "At the docks"},
            "notes": [{"note_id": "note-1", "text": "Storm clouds gather"}],
        },
        diagnostics={"applied_event_count": 2},
        idempotency_key="req-1",
        request_payload_hash="hash-a",
        adapter_key="fate",
        adapter_version="1.0.0",
        proposal_id=None,
    )

    updated = repo.get_session(owner_user_id=42, session_id=session.id)
    snapshot = repo.get_latest_snapshot(owner_user_id=42, session_id=session.id)

    assert [event.sequence_number for event in result.events] == [1, 2]  # nosec B101
    assert all(event.operation_id is not None for event in result.events)  # nosec B101
    assert updated.last_event_sequence == 2  # nosec B101
    assert updated.current_snapshot_version == 1  # nosec B101
    assert snapshot.last_event_sequence == 2  # nosec B101
    assert snapshot.snapshot_json["scene"]["summary"] == "At the docks"  # nosec B101


def test_commit_events_replays_same_idempotency_key_without_new_snapshot():
    repo = _repo()
    campaign = _campaign(repo)
    session = _session(repo, campaign.id)
    payload = [{"event_type": "note.added", "event_payload": {"note_id": "n1", "text": "A"}, "source_type": "user"}]

    first = repo.commit_events_and_snapshot(
        42,
        session.id,
        0,
        0,
        payload,
        {"notes": [{"note_id": "n1", "text": "A"}]},
        {},
        "same-key",
        "hash-a",
        "fate",
        "1.0.0",
        None,
    )
    second = repo.commit_events_and_snapshot(
        42,
        session.id,
        0,
        0,
        payload,
        {"notes": [{"note_id": "n1", "text": "A"}]},
        {},
        "same-key",
        "hash-a",
        "fate",
        "1.0.0",
        None,
    )

    assert [event.id for event in second.events] == [event.id for event in first.events]  # nosec B101
    assert second.replayed is True  # nosec B101
    assert repo.get_session(owner_user_id=42, session_id=session.id).current_snapshot_version == 1  # nosec B101


def test_commit_events_rejects_same_idempotency_key_with_different_hash():
    repo = _repo()
    campaign = _campaign(repo)
    session = _session(repo, campaign.id)
    payload = [{"event_type": "note.added", "event_payload": {"note_id": "n1", "text": "A"}, "source_type": "user"}]

    repo.commit_events_and_snapshot(
        42,
        session.id,
        0,
        0,
        payload,
        {"notes": [{"note_id": "n1", "text": "A"}]},
        {},
        "same-key",
        "hash-a",
        "fate",
        "1.0.0",
        None,
    )

    with pytest.raises(RPGConflictError, match="idempotency"):
        repo.commit_events_and_snapshot(
            42,
            session.id,
            1,
            1,
            payload,
            {"notes": [{"note_id": "n1", "text": "A"}]},
            {},
            "same-key",
            "hash-b",
            "fate",
            "1.0.0",
            None,
        )


def test_commit_events_rejects_stale_expected_sequence():
    repo = _repo()
    campaign = _campaign(repo)
    session = _session(repo, campaign.id)
    payload = [{"event_type": "note.added", "event_payload": {"note_id": "n1", "text": "A"}, "source_type": "user"}]

    with pytest.raises(RPGConflictError, match="stale_event_sequence"):
        repo.commit_events_and_snapshot(42, session.id, 7, 0, payload, {"notes": []}, {}, "stale-key", "hash-a", "fate", "1.0.0", None)
