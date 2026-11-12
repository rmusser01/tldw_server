import pytest


pytestmark = pytest.mark.unit


def _snapshot(
    *,
    status: str = "queued",
    phase: str = "collecting",
    control_state: str = "running",
    progress_percent: float | None = None,
    progress_message: str | None = None,
    active_job_id: str | None = "21",
    latest_checkpoint_id: str | None = None,
    completed_at: str | None = None,
    latest_event_id: int = 0,
    checkpoint: dict | None = None,
    artifacts: list[dict] | None = None,
):
    from tldw_Server_API.app.api.v1.schemas.research_runs_schemas import ResearchRunSnapshotResponse

    payload = {
        "run": {
            "id": "rs_1",
            "status": status,
            "phase": phase,
            "control_state": control_state,
            "progress_percent": progress_percent,
            "progress_message": progress_message,
            "active_job_id": active_job_id,
            "latest_checkpoint_id": latest_checkpoint_id,
            "completed_at": completed_at,
        },
        "latest_event_id": latest_event_id,
        "checkpoint": checkpoint,
        "artifacts": artifacts or [],
    }
    return ResearchRunSnapshotResponse.model_validate(payload)


def test_build_research_stream_state_overlays_job_progress_and_tracks_artifact_versions():
    from tldw_Server_API.app.core.Research.streaming import build_research_stream_state

    snapshot = _snapshot(
        progress_percent=45.0,
        progress_message="collecting sources",
        artifacts=[
            {
                "artifact_name": "plan.json",
                "artifact_version": 2,
                "content_type": "application/json",
                "phase": "drafting_plan",
                "job_id": "12",
            },
            {
                "artifact_name": "provider_config.json",
                "artifact_version": 1,
                "content_type": "application/json",
                "phase": "drafting_plan",
                "job_id": "12",
            },
        ],
    )

    state = build_research_stream_state(
        snapshot=snapshot,
        active_job={"id": 21, "progress_percent": 52.5, "progress_message": "job-level progress"},
    )

    assert state.snapshot.run.progress_percent == 52.5
    assert state.snapshot.run.progress_message == "job-level progress"
    assert state.artifact_versions == {"plan.json": 2, "provider_config.json": 1}


def test_initial_stream_events_emit_snapshot_for_non_terminal_state():
    from tldw_Server_API.app.core.Research.streaming import (
        build_research_stream_state,
        initial_stream_events,
    )

    state = build_research_stream_state(snapshot=_snapshot())

    events = initial_stream_events(state)

    assert [event.event for event in events] == ["snapshot"]
    assert events[0].data["run"]["id"] == "rs_1"


def test_diff_stream_events_emit_status_progress_and_checkpoint_changes():
    from tldw_Server_API.app.core.Research.streaming import (
        build_research_stream_state,
        diff_stream_events,
    )

    previous = build_research_stream_state(snapshot=_snapshot())
    current = build_research_stream_state(
        snapshot=_snapshot(
            status="waiting_human",
            phase="awaiting_source_review",
            progress_percent=45.0,
            progress_message="collecting sources",
            active_job_id=None,
            latest_checkpoint_id="cp_1",
            checkpoint={
                "checkpoint_id": "cp_1",
                "checkpoint_type": "sources_review",
                "status": "pending",
                "proposed_payload": {"source_count": 3},
                "resolution": None,
            },
        )
    )

    events = diff_stream_events(previous=previous, current=current)

    assert [event.event for event in events] == ["status", "progress", "checkpoint"]
    assert events[0].data["phase"] == "awaiting_source_review"
    assert events[1].data["progress_message"] == "collecting sources"
    assert events[2].data["checkpoint_id"] == "cp_1"


def test_diff_stream_events_emit_only_new_artifact_versions_after_baseline():
    from tldw_Server_API.app.core.Research.streaming import (
        build_research_stream_state,
        diff_stream_events,
    )

    previous = build_research_stream_state(
        snapshot=_snapshot(
            artifacts=[
                {
                    "artifact_name": "plan.json",
                    "artifact_version": 1,
                    "content_type": "application/json",
                    "phase": "drafting_plan",
                    "job_id": "11",
                },
                {
                    "artifact_name": "provider_config.json",
                    "artifact_version": 1,
                    "content_type": "application/json",
                    "phase": "drafting_plan",
                    "job_id": "11",
                },
            ]
        )
    )
    current = build_research_stream_state(
        snapshot=_snapshot(
            artifacts=[
                {
                    "artifact_name": "plan.json",
                    "artifact_version": 2,
                    "content_type": "application/json",
                    "phase": "drafting_plan",
                    "job_id": "12",
                },
                {
                    "artifact_name": "provider_config.json",
                    "artifact_version": 1,
                    "content_type": "application/json",
                    "phase": "drafting_plan",
                    "job_id": "11",
                },
                {
                    "artifact_name": "report_v1.md",
                    "artifact_version": 1,
                    "content_type": "text/markdown",
                    "phase": "synthesizing",
                    "job_id": "13",
                },
            ]
        )
    )

    events = diff_stream_events(previous=previous, current=current)

    assert [event.event for event in events] == ["artifact", "artifact"]
    assert [event.data["artifact_name"] for event in events] == ["plan.json", "report_v1.md"]


def test_initial_and_diff_stream_events_emit_terminal_state_once():
    from tldw_Server_API.app.core.Research.streaming import (
        build_research_stream_state,
        diff_stream_events,
        initial_stream_events,
    )

    terminal_state = build_research_stream_state(
        snapshot=_snapshot(
            status="completed",
            phase="completed",
            progress_percent=100.0,
            progress_message="packaging results",
            active_job_id=None,
            completed_at="2026-03-07T00:00:00+00:00",
        )
    )
    initial_events = initial_stream_events(terminal_state)

    previous = build_research_stream_state(snapshot=_snapshot())
    terminal_events = diff_stream_events(previous=previous, current=terminal_state)

    assert [event.event for event in initial_events] == ["snapshot", "terminal"]
    assert terminal_events[-1].event == "terminal"
    assert terminal_events[-1].data["status"] == "completed"


def test_persisted_stream_event_includes_event_id_and_replayed_flag():
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchRunEventRow
    from tldw_Server_API.app.core.Research.streaming import persisted_event_to_stream_event

    event = persisted_event_to_stream_event(
        ResearchRunEventRow(
            id=7,
            session_id="rs_1",
            owner_user_id="1",
            event_type="artifact",
            event_payload={"artifact_name": "plan.json", "artifact_version": 1},
            phase="drafting_plan",
            job_id="22",
            created_at="2026-03-07T00:00:00+00:00",
        ),
        replayed=True,
    )

    assert event.event == "artifact"
    assert event.event_id == "7"
    assert event.data["event_id"] == 7
    assert event.data["replayed"] is True


@pytest.mark.parametrize(
    ("event_type", "checkpoint_id"), [("checkpoint", "cp_1"), ("checkpoint", "cp_old"), ("status", "cp_1")]
)
def test_persisted_stream_proposal_binds_only_current_checkpoint(event_type, checkpoint_id):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchRunEventRow
    from tldw_Server_API.app.core.Research.streaming import persisted_event_to_stream_event

    payload = {"checkpoint_id": checkpoint_id, "has_proposed_payload": True}
    row = ResearchRunEventRow(
        id=7,
        session_id="rs_1",
        owner_user_id="1",
        event_type=event_type,
        event_payload=payload,
        phase="awaiting_plan_review",
        job_id=None,
        created_at="2026-10-30T00:00:00Z",
    )
    checkpoint = _snapshot(
        checkpoint={
            "checkpoint_id": "cp_1",
            "checkpoint_type": "plan_review",
            "status": "pending",
            "proposed_payload": {"focus_areas": ["Rowan"]},
            "resolution": None,
        }
    ).checkpoint
    event = persisted_event_to_stream_event(row, replayed=True, checkpoint=checkpoint)
    assert event.data.get("proposed_payload") == (  # nosec B101 # Pytest regression checks SSE delivery and checkpoint integrity.
        {"focus_areas": ["Rowan"]} if event_type == "checkpoint" and checkpoint_id == "cp_1" else None
    )
    assert payload == {"checkpoint_id": checkpoint_id, "has_proposed_payload": True}  # nosec B101 # Pytest regression checks SSE delivery and checkpoint integrity.
