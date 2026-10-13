import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.RPG_DB import RPGRepository
from tldw_Server_API.app.core.RPG.errors import RPGConflictError, RPGNotFoundError

pytestmark = pytest.mark.integration


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


def test_get_campaign_returns_owner_scoped_campaign():
    repo = _repo()
    campaign = _campaign(repo, owner_user_id=42)

    fetched = repo.get_campaign(owner_user_id=42, campaign_id=campaign.id)

    assert fetched.id == campaign.id  # nosec B101
    assert fetched.owner_user_id == 42  # nosec B101
    with pytest.raises(RPGNotFoundError, match="rpg_campaign_not_found"):
        repo.get_campaign(owner_user_id=43, campaign_id=campaign.id)


def test_replace_campaign_rules_pack_refs_requires_expected_version():
    repo = _repo()
    campaign = _campaign(repo)

    with pytest.raises(RPGConflictError, match="stale_rules_pack_ref_version"):
        repo.replace_campaign_rules_pack_refs(
            owner_user_id=42,
            campaign_id=campaign.id,
            rules_pack_refs=[{"source_type": "media_item", "source_id": 7}],
            expected_version=campaign.version + 1,
            idempotency_key="campaign-refs",
            request_payload_hash="hash-campaign-refs",
            source_type="user",
        )


def test_replace_campaign_rules_pack_refs_increments_version():
    repo = _repo()
    campaign = _campaign(repo)

    result = repo.replace_campaign_rules_pack_refs(
        owner_user_id=42,
        campaign_id=campaign.id,
        rules_pack_refs=[{"source_type": "media_item", "source_id": 7, "display_name": "Rules"}],
        expected_version=campaign.version,
        idempotency_key="campaign-refs",
        request_payload_hash="hash-campaign-refs",
        source_type="user",
    )
    updated = repo.get_campaign(owner_user_id=42, campaign_id=campaign.id)

    assert result.version == campaign.version + 1  # nosec B101
    assert updated.version == campaign.version + 1  # nosec B101
    assert updated.linked_rules_pack_refs[0]["ref_id"] == "media_item:7"  # nosec B101
    assert result.refs[0].display_name == "Rules"  # nosec B101


def test_replace_campaign_rules_pack_refs_replays_idempotency_key():
    repo = _repo()
    campaign = _campaign(repo)
    payload = [{"source_type": "media_item", "source_id": 7}]

    first = repo.replace_campaign_rules_pack_refs(
        42,
        campaign.id,
        payload,
        campaign.version,
        "campaign-refs",
        "hash-campaign-refs",
        "user",
    )
    second = repo.replace_campaign_rules_pack_refs(
        42,
        campaign.id,
        payload,
        campaign.version,
        "campaign-refs",
        "hash-campaign-refs",
        "user",
    )

    assert second.replayed is True  # nosec B101
    assert second.version == first.version  # nosec B101
    assert [ref.ref_id for ref in second.refs] == [ref.ref_id for ref in first.refs]  # nosec B101


def test_replace_campaign_rules_pack_refs_rejects_idempotency_payload_mismatch():
    repo = _repo()
    campaign = _campaign(repo)
    payload = [{"source_type": "media_item", "source_id": 7}]

    repo.replace_campaign_rules_pack_refs(
        42,
        campaign.id,
        payload,
        campaign.version,
        "campaign-refs",
        "hash-campaign-refs",
        "user",
    )

    with pytest.raises(RPGConflictError, match="idempotency"):
        repo.replace_campaign_rules_pack_refs(
            42,
            campaign.id,
            payload,
            campaign.version,
            "campaign-refs",
            "hash-campaign-refs-changed",
            "user",
        )


def test_replace_session_rules_pack_refs_requires_expected_version():
    repo = _repo()
    campaign = _campaign(repo)
    session = _session(repo, campaign.id)

    with pytest.raises(RPGConflictError, match="stale_rules_pack_ref_version"):
        repo.replace_session_rules_pack_refs(
            owner_user_id=42,
            session_id=session.id,
            rules_pack_refs=[{"source_type": "media_collection", "source_id": 3}],
            expected_version=session.version + 1,
            idempotency_key="session-refs",
            request_payload_hash="hash-session-refs",
            source_type="user",
        )


def test_replace_session_rules_pack_refs_increments_version():
    repo = _repo()
    campaign = _campaign(repo)
    session = _session(repo, campaign.id)

    result = repo.replace_session_rules_pack_refs(
        owner_user_id=42,
        session_id=session.id,
        rules_pack_refs=[{"source_type": "media_collection", "source_id": 3}],
        expected_version=session.version,
        idempotency_key="session-refs",
        request_payload_hash="hash-session-refs",
        source_type="user",
    )
    updated = repo.get_session(owner_user_id=42, session_id=session.id)

    assert result.version == session.version + 1  # nosec B101
    assert updated.version == session.version + 1  # nosec B101
    assert updated.active_rules_pack_refs[0]["ref_id"] == "media_collection:3"  # nosec B101
    assert result.refs[0].ref_id == "media_collection:3"  # nosec B101


def test_replace_session_rules_pack_refs_replays_idempotency_key():
    repo = _repo()
    campaign = _campaign(repo)
    session = _session(repo, campaign.id)
    payload = [{"source_type": "media_collection", "source_id": 3}]

    first = repo.replace_session_rules_pack_refs(
        42,
        session.id,
        payload,
        session.version,
        "session-refs",
        "hash-session-refs",
        "user",
    )
    second = repo.replace_session_rules_pack_refs(
        42,
        session.id,
        payload,
        session.version,
        "session-refs",
        "hash-session-refs",
        "user",
    )

    assert second.replayed is True  # nosec B101
    assert second.version == first.version  # nosec B101
    assert [ref.ref_id for ref in second.refs] == [ref.ref_id for ref in first.refs]  # nosec B101


def test_replace_session_rules_pack_refs_rejects_idempotency_payload_mismatch():
    repo = _repo()
    campaign = _campaign(repo)
    session = _session(repo, campaign.id)
    payload = [{"source_type": "media_collection", "source_id": 3}]

    repo.replace_session_rules_pack_refs(
        42,
        session.id,
        payload,
        session.version,
        "session-refs",
        "hash-session-refs",
        "user",
    )

    with pytest.raises(RPGConflictError, match="idempotency"):
        repo.replace_session_rules_pack_refs(
            42,
            session.id,
            payload,
            session.version,
            "session-refs",
            "hash-session-refs-changed",
            "user",
        )


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


def test_list_campaigns_filters_status_and_paginates():
    repo = _repo()
    first = _campaign(repo)
    second = repo.create_campaign(
        owner_user_id=42, title="Second", description=None,
        default_adapter_key="fate", default_adapter_version="1.0.0",
        settings={}, linked_rules_pack_refs=[],
        idempotency_key="campaign-2", request_payload_hash="hash-2", source_type="user",
    )
    listed, total = repo.list_campaigns(owner_user_id=42)
    assert total == 2  # nosec B101
    assert [c.id for c in listed] == [second.id, first.id]  # nosec B101  (updated_at DESC, id DESC)
    empty, empty_total = repo.list_campaigns(owner_user_id=42, status="archived")
    assert empty == [] and empty_total == 0  # nosec B101
    page, page_total = repo.list_campaigns(owner_user_id=42, limit=1, offset=1)
    assert page_total == 2 and [c.id for c in page] == [first.id]  # nosec B101
    foreign, foreign_total = repo.list_campaigns(owner_user_id=7)
    assert foreign == [] and foreign_total == 0  # nosec B101
    with repo.db.transaction() as conn:
        conn.execute("UPDATE rpg_campaigns SET status = 'archived' WHERE id = ?", (second.id,))
    archived, archived_total = repo.list_campaigns(owner_user_id=42, status="archived")
    assert archived_total == 1 and archived[0].id == second.id  # nosec B101


def test_list_sessions_filters_and_counts():
    repo = _repo()
    campaign = _campaign(repo)
    session = _session(repo, campaign.id)
    listed, total = repo.list_sessions(owner_user_id=42)
    assert total == 1 and listed[0].id == session.id  # nosec B101
    assert repo.list_sessions(owner_user_id=42, campaign_id=campaign.id)[1] == 1  # nosec B101
    assert repo.list_sessions(owner_user_id=42, campaign_id=999)[1] == 0  # nosec B101
    assert repo.count_campaign_sessions(owner_user_id=42, campaign_id=campaign.id) == 1  # nosec B101
    assert repo.count_pending_proposals(owner_user_id=42, session_id=session.id) == 0  # nosec B101


def test_list_events_cursors_and_filters():
    repo = _repo()
    campaign = _campaign(repo)
    session = _session(repo, campaign.id)
    for seq, (event_type, source) in enumerate(
        [("note.added", "user"), ("note.added", "mcp"), ("scene.updated", "user")], start=1
    ):
        repo.commit_events_and_snapshot(
            owner_user_id=42, session_id=session.id,
            expected_last_event_sequence=seq - 1, base_snapshot_version=seq - 1,
            events=[{"event_type": event_type, "event_payload": {"note_id": f"n{seq}", "scene_id": f"s{seq}"}, "source_type": source}],
            snapshot={}, diagnostics={}, idempotency_key=f"evt-{seq}", request_payload_hash=f"h{seq}",
            adapter_key=session.adapter_key, adapter_version=session.adapter_version, proposal_id=None,
        )
    events = repo.list_events(owner_user_id=42, session_id=session.id)
    assert [e.sequence_number for e in events] == [1, 2, 3]  # nosec B101
    assert [e.sequence_number for e in repo.list_events(owner_user_id=42, session_id=session.id, after_sequence=1)] == [2, 3]  # nosec B101
    assert len(repo.list_events(owner_user_id=42, session_id=session.id, source_type="mcp")) == 1  # nosec B101
    assert len(repo.list_events(owner_user_id=42, session_id=session.id, event_type="scene.updated")) == 1  # nosec B101


def test_list_proposals_and_get_snapshot_by_version():
    repo = _repo()
    campaign = _campaign(repo)
    session = _session(repo, campaign.id)
    assert repo.list_proposals(owner_user_id=42, session_id=session.id) == ([], 0)  # nosec B101
    record = repo.get_snapshot_by_version(owner_user_id=42, session_id=session.id, snapshot_version=0)
    assert record.snapshot_version == 0  # nosec B101
    with pytest.raises(RPGNotFoundError):
        repo.get_snapshot_by_version(owner_user_id=42, session_id=session.id, snapshot_version=99)


def test_update_campaign_metadata_with_cas_and_replay():
    repo = _repo()
    campaign = _campaign(repo)
    result = repo.update_campaign(
        owner_user_id=42, campaign_id=campaign.id,
        title="Renamed", description="Now with words", status="archived",
        expected_version=campaign.version, idempotency_key="upd-1", request_payload_hash="h-upd-1",
        source_type="user",
    )
    assert result.replayed is False  # nosec B101
    assert result.campaign.title == "Renamed" and result.campaign.status == "archived"  # nosec B101
    assert result.campaign.version == campaign.version + 1  # nosec B101
    replay = repo.update_campaign(
        owner_user_id=42, campaign_id=campaign.id,
        title="Renamed", description="Now with words", status="archived",
        expected_version=campaign.version, idempotency_key="upd-1", request_payload_hash="h-upd-1",
        source_type="user",
    )
    assert replay.replayed is True and replay.campaign.id == campaign.id  # nosec B101
    with pytest.raises(RPGConflictError):
        repo.update_campaign(
            owner_user_id=42, campaign_id=campaign.id, title="X",
            expected_version=campaign.version, idempotency_key="upd-2",
            request_payload_hash="h-upd-2", source_type="user",
        )
    with pytest.raises(RPGConflictError):  # same key, different payload
        repo.update_campaign(
            owner_user_id=42, campaign_id=campaign.id, title="Y",
            expected_version=campaign.version, idempotency_key="upd-1",
            request_payload_hash="h-DIFFERENT", source_type="user",
        )
    with pytest.raises(RPGNotFoundError):
        repo.update_campaign(
            owner_user_id=7, campaign_id=campaign.id, title="Z",
            expected_version=1, idempotency_key="upd-3", request_payload_hash="h-3", source_type="user",
        )


def test_update_session_metadata_partial_and_cas():
    repo = _repo()
    campaign = _campaign(repo)
    session = _session(repo, campaign.id)
    result = repo.update_session(
        owner_user_id=42, session_id=session.id, title="Act II", status="archived",
        expected_version=session.version, idempotency_key="s-upd-1",
        request_payload_hash="h-s-1", source_type="user",
    )
    assert result.session.title == "Act II" and result.session.status == "archived"  # nosec B101
    untouched = repo.update_session(
        owner_user_id=42, session_id=session.id,
        expected_version=result.session.version, idempotency_key="s-upd-2",
        request_payload_hash="h-s-2", source_type="user",
    )
    assert untouched.session.title == "Act II"  # nosec B101  (COALESCE keeps omitted fields)


def test_update_session_authority_settings_cas_and_replay():
    repo = _repo()
    campaign = _campaign(repo)
    session = _session(repo, campaign.id)
    settings = {"import_auto_commit": True, "mcp_auto_commit": False, "model_auto_commit": False, "model_auto_commit_event_types": []}
    result = repo.update_session_authority_settings(
        owner_user_id=42, session_id=session.id, authority_settings=settings,
        expected_version=session.version, idempotency_key="auth-1",
        request_payload_hash="h-auth-1", source_type="user",
    )
    assert result.session.authority_settings["import_auto_commit"] is True  # nosec B101
    assert result.session.version == session.version + 1  # nosec B101
    replay = repo.update_session_authority_settings(
        owner_user_id=42, session_id=session.id, authority_settings=settings,
        expected_version=session.version, idempotency_key="auth-1",
        request_payload_hash="h-auth-1", source_type="user",
    )
    assert replay.replayed is True  # nosec B101
    with pytest.raises(RPGConflictError):
        repo.update_session_authority_settings(
            owner_user_id=42, session_id=session.id, authority_settings=settings,
            expected_version=session.version, idempotency_key="auth-2",
            request_payload_hash="h-auth-2", source_type="user",
        )
