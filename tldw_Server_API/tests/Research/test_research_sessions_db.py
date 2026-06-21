import pytest


pytestmark = pytest.mark.unit


def test_create_session_and_checkpoint_round_trip(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB

    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="7",
        query="Compare local and external evidence on quantum networking",
        source_policy="balanced",
        autonomy_mode="checkpointed",
        limits_json={"max_searches": 25},
    )

    assert session.phase == "drafting_plan"
    stored = db.get_session(session.id)
    assert stored is not None
    assert stored.query.startswith("Compare")

    checkpoint = db.create_checkpoint(
        session_id=session.id,
        checkpoint_type="plan_review",
        proposed_payload={"focus_areas": ["background", "primary sources"]},
    )
    resolved = db.resolve_checkpoint(
        checkpoint.id,
        resolution="patched",
        user_patch_payload={"focus_areas": ["background", "contradictions"]},
    )

    assert resolved.status == "resolved"
    assert resolved.user_patch_payload["focus_areas"][1] == "contradictions"


def test_research_run_events_round_trip_with_owner_scoped_cursor_reads(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB

    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="11",
        query="Map open questions about battery recycling policy",
        source_policy="balanced",
        autonomy_mode="checkpointed",
        limits_json={"max_searches": 5},
    )
    other_session = db.create_session(
        owner_user_id="22",
        query="Separate owner session",
        source_policy="local_only",
        autonomy_mode="autonomous",
        limits_json={},
    )

    first = db.record_run_event(
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        event_type="status",
        event_payload={"status": "queued", "phase": "drafting_plan"},
        phase="drafting_plan",
        job_id="101",
    )
    second = db.record_run_event(
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        event_type="progress",
        event_payload={"progress_percent": 10.0, "progress_message": "planning research"},
        phase="drafting_plan",
        job_id="101",
    )
    db.record_run_event(
        owner_user_id=other_session.owner_user_id,
        session_id=other_session.id,
        event_type="status",
        event_payload={"status": "queued", "phase": "drafting_plan"},
        phase="drafting_plan",
        job_id="202",
    )

    assert isinstance(first.id, int)
    assert second.id > first.id

    events = db.list_run_events_after(
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        after_id=first.id,
    )

    assert [event.id for event in events] == [second.id]
    assert events[0].event_type == "progress"
    assert events[0].event_payload["progress_message"] == "planning research"


def test_research_run_events_table_is_created_automatically(tmp_path):
    import sqlite3

    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB

    db_path = tmp_path / "research.db"
    ResearchSessionsDB(db_path)

    with sqlite3.connect(db_path) as conn:
        table_names = {
            str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }

    assert "research_run_events" in table_names


def test_list_sessions_is_owner_scoped_and_sorted_by_created_at_desc(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB

    db = ResearchSessionsDB(tmp_path / "research.db")
    older = db.create_session(
        owner_user_id="1",
        query="Older run",
        source_policy="balanced",
        autonomy_mode="checkpointed",
        limits_json={},
    )
    newer = db.create_session(
        owner_user_id="1",
        query="Newer run",
        source_policy="balanced",
        autonomy_mode="checkpointed",
        limits_json={},
    )
    db.create_session(
        owner_user_id="2",
        query="Other owner run",
        source_policy="local_only",
        autonomy_mode="autonomous",
        limits_json={},
    )

    sessions = db.list_sessions(owner_user_id="1", limit=10)

    assert [session.id for session in sessions] == [newer.id, older.id]
    assert [session.query for session in sessions] == ["Newer run", "Older run"]
    assert all(session.owner_user_id == "1" for session in sessions)


def test_discovery_snapshot_round_trip_is_owner_scoped(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB

    db = ResearchSessionsDB(tmp_path / "research.db")
    snapshot = db.create_discovery_snapshot(
        owner_user_id="1",
        query="open access batteries",
        request_json={"query": "open access batteries"},
        response_json={"results": [{"result_id": "res_1"}]},
        effective_config_json={"source_ids": ["openalex"]},
        catalog_version="research-discovery-v1",
        retention_hours=24,
    )

    stored = db.get_discovery_snapshot(snapshot.id, owner_user_id="1")

    assert snapshot.id.startswith("rd_")
    assert stored is not None
    assert stored.owner_user_id == "1"
    assert stored.query == "open access batteries"
    assert stored.request_json["query"] == "open access batteries"
    assert stored.response_json["results"][0]["result_id"] == "res_1"
    assert stored.effective_config_json["source_ids"] == ["openalex"]
    assert stored.catalog_version == "research-discovery-v1"
    assert db.get_discovery_snapshot(snapshot.id, owner_user_id="2") is None


def test_discovery_snapshot_rejects_expired_rows(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB

    db = ResearchSessionsDB(tmp_path / "research.db")
    snapshot = db.create_discovery_snapshot(
        owner_user_id="1",
        query="expired",
        request_json={},
        response_json={},
        effective_config_json={},
        catalog_version="research-discovery-v1",
        retention_hours=-1,
    )

    assert db.get_discovery_snapshot(snapshot.id, owner_user_id="1") is None


def test_delete_expired_discovery_snapshots_only_removes_expired_rows(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB

    db = ResearchSessionsDB(tmp_path / "research.db")
    expired = db.create_discovery_snapshot(
        owner_user_id="1",
        query="expired",
        request_json={},
        response_json={},
        effective_config_json={},
        catalog_version="research-discovery-v1",
        retention_hours=-1,
    )
    active = db.create_discovery_snapshot(
        owner_user_id="1",
        query="active",
        request_json={},
        response_json={},
        effective_config_json={},
        catalog_version="research-discovery-v1",
        retention_hours=24,
    )

    deleted_count = db.delete_expired_discovery_snapshots()

    assert deleted_count == 1
    assert db.get_discovery_snapshot(expired.id, owner_user_id="1") is None
    assert db.get_discovery_snapshot(active.id, owner_user_id="1") is not None
