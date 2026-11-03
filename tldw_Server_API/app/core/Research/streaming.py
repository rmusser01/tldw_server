"""Helpers for research-native live progress streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchRunEventRow
from tldw_Server_API.app.core.Research.research_runs_schemas import ResearchRunSnapshotResponse

_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


@dataclass(frozen=True)
class ResearchStreamEvent:
    """Normalized research-native stream event."""

    event: str
    data: dict[str, Any]
    event_id: str | None = None


@dataclass(frozen=True)
class ResearchStreamState:
    """Current stream state plus artifact baseline for change detection."""

    snapshot: ResearchRunSnapshotResponse
    artifact_versions: dict[str, int]


def _run_status_payload(snapshot: ResearchRunSnapshotResponse) -> dict[str, Any]:
    run = snapshot.run
    return {
        "id": run.id,
        "status": run.status,
        "phase": run.phase,
        "control_state": run.control_state,
        "active_job_id": run.active_job_id,
        "latest_checkpoint_id": run.latest_checkpoint_id,
        "completed_at": run.completed_at,
    }


def _run_progress_payload(snapshot: ResearchRunSnapshotResponse) -> dict[str, Any]:
    run = snapshot.run
    return {
        "id": run.id,
        "progress_percent": run.progress_percent,
        "progress_message": run.progress_message,
    }


def _checkpoint_payload(snapshot: ResearchRunSnapshotResponse) -> dict[str, Any] | None:
    checkpoint = snapshot.checkpoint
    if checkpoint is None:
        return None
    return checkpoint.model_dump(mode="json")


def _artifact_payload(snapshot: ResearchRunSnapshotResponse, artifact_name: str) -> dict[str, Any]:
    artifact = next(item for item in snapshot.artifacts if item.artifact_name == artifact_name)
    return artifact.model_dump(mode="json")


def build_research_stream_state(
    *,
    snapshot: ResearchRunSnapshotResponse,
    active_job: dict[str, Any] | None = None,
) -> ResearchStreamState:
    """Build the current stream state with optional active-job progress overlay."""

    updated_snapshot = snapshot
    if isinstance(active_job, dict):
        run_updates: dict[str, Any] = {}
        if active_job.get("progress_percent") is not None:
            run_updates["progress_percent"] = float(active_job["progress_percent"])
        if active_job.get("progress_message") is not None:
            run_updates["progress_message"] = str(active_job["progress_message"])
        if run_updates:
            updated_snapshot = snapshot.model_copy(
                update={"run": snapshot.run.model_copy(update=run_updates)}
            )
    artifact_versions = {
        artifact.artifact_name: artifact.artifact_version
        for artifact in updated_snapshot.artifacts
    }
    return ResearchStreamState(
        snapshot=updated_snapshot,
        artifact_versions=artifact_versions,
    )


def load_research_stream_state(
    *,
    service: Any,
    owner_user_id: str,
    session_id: str,
    job_manager: Any | None = None,
) -> ResearchStreamState:
    """Load current stream state from the research service and optional Jobs manager."""

    snapshot = service.get_stream_snapshot(
        owner_user_id=owner_user_id,
        session_id=session_id,
    )
    active_job: dict[str, Any] | None = None
    active_job_id = snapshot.run.active_job_id
    if (
        job_manager is not None
        and isinstance(active_job_id, str)
        and active_job_id.isdigit()
        and hasattr(job_manager, "get_job")
    ):
        loaded_job = job_manager.get_job(int(active_job_id))
        if isinstance(loaded_job, dict):
            active_job = loaded_job
    return build_research_stream_state(snapshot=snapshot, active_job=active_job)


def initial_stream_events(state: ResearchStreamState) -> list[ResearchStreamEvent]:
    """Return events to emit when a client first connects."""

    events = [
        ResearchStreamEvent(
            event="snapshot",
            data=state.snapshot.model_dump(mode="json"),
        )
    ]
    if state.snapshot.run.status in _TERMINAL_STATUSES:
        events.append(
            ResearchStreamEvent(
                event="terminal",
                data=synthetic_terminal_payload(state.snapshot),
                event_id=(
                    str(state.snapshot.latest_event_id)
                    if state.snapshot.latest_event_id > 0
                    else None
                ),
            )
        )
    return events


def persisted_event_to_stream_event(
    event_row: ResearchRunEventRow,
    *,
    replayed: bool,
) -> ResearchStreamEvent:
    """Convert a persisted run-event row into an SSE-ready event."""

    payload = dict(event_row.event_payload)
    payload["event_id"] = int(event_row.id)
    payload["replayed"] = bool(replayed)
    return ResearchStreamEvent(
        event=event_row.event_type,
        data=payload,
        event_id=str(event_row.id),
    )


def synthetic_terminal_payload(snapshot: ResearchRunSnapshotResponse) -> dict[str, Any]:
    """Build a terminal event payload from the current snapshot."""

    payload = _run_status_payload(snapshot)
    payload["event_id"] = int(snapshot.latest_event_id)
    payload["replayed"] = False
    return payload


def diff_stream_events(
    *,
    previous: ResearchStreamState,
    current: ResearchStreamState,
) -> list[ResearchStreamEvent]:
    """Return research-native events for state changes between two polls."""

    events: list[ResearchStreamEvent] = []

    previous_status = _run_status_payload(previous.snapshot)
    current_status = _run_status_payload(current.snapshot)
    if previous_status != current_status:
        events.append(ResearchStreamEvent(event="status", data=current_status))

    previous_progress = _run_progress_payload(previous.snapshot)
    current_progress = _run_progress_payload(current.snapshot)
    if previous_progress != current_progress:
        events.append(ResearchStreamEvent(event="progress", data=current_progress))

    previous_checkpoint = _checkpoint_payload(previous.snapshot)
    current_checkpoint = _checkpoint_payload(current.snapshot)
    if previous_checkpoint != current_checkpoint:
        events.append(
            ResearchStreamEvent(
                event="checkpoint",
                data=current_checkpoint or {"checkpoint_id": None},
            )
        )

    artifact_names = sorted(set(previous.artifact_versions) | set(current.artifact_versions))
    for artifact_name in artifact_names:
        previous_version = previous.artifact_versions.get(artifact_name)
        current_version = current.artifact_versions.get(artifact_name)
        if current_version is None:
            continue
        if previous_version is None or current_version > previous_version:
            events.append(
                ResearchStreamEvent(
                    event="artifact",
                    data=_artifact_payload(current.snapshot, artifact_name),
                )
            )

    previous_terminal = previous.snapshot.run.status in _TERMINAL_STATUSES
    current_terminal = current.snapshot.run.status in _TERMINAL_STATUSES
    if current_terminal and (
        not previous_terminal or previous.snapshot.run.status != current.snapshot.run.status
    ):
        events.append(ResearchStreamEvent(event="terminal", data=current_status))

    return events


__all__ = [
    "ResearchStreamEvent",
    "ResearchStreamState",
    "build_research_stream_state",
    "diff_stream_events",
    "initial_stream_events",
    "load_research_stream_state",
    "persisted_event_to_stream_event",
    "synthetic_terminal_payload",
]
