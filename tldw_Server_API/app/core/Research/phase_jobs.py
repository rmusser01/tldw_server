"""Replayable Research dispatch and terminal-state reconciliation using normal Jobs APIs."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import (
    ResearchPhaseChangedError,
    ResearchSessionRow,
    ResearchSessionsDB,
)

from .jobs import RESEARCH_DOMAIN, RESEARCH_JOB_TYPE, RESEARCH_QUEUE, enqueue_research_phase_job
from .phase_dispatch import new_phase_dispatch, read_phase_dispatch


class ResearchAdmissionRefusedError(ValueError):
    """A definite refusal needs a new user decision, never automatic recovery."""

    retryable = False


def dispatch_matches_job(dispatch: dict[str, Any], job: dict[str, Any]) -> bool:
    """Compare immutable phase authorization; lease ownership is checked separately."""
    payload = job.get("payload")
    if not isinstance(payload, dict) or not isinstance(job.get("uuid"), str) or not job["uuid"]:
        return False
    delivery = payload.get("dispatch_id")
    if delivery is None:
        delivery = str(uuid.uuid5(uuid.NAMESPACE_URL, "research-job:" + job["uuid"]))
    return bool(
        delivery == dispatch["dispatch_id"]
        and payload.get("phase") == dispatch["phase"]
        and payload.get("checkpoint_id") == dispatch["checkpoint_id"]
        and type(payload.get("policy_version")) is int
        and payload["policy_version"] == dispatch["policy_version"]
        and type(payload.get("approved_outline_locked", False)) is bool
        and payload.get("approved_outline_locked", False) == dispatch["approved_outline_locked"]
    )


def predecessor_authorizes_continuation(dispatch: dict[str, Any], job: dict[str, Any]) -> bool:
    """An old delivery controls only its own automatic, policy-preserving phase edge."""
    payload = job.get("payload")
    edges = {"drafting_plan": "collecting", "collecting": "synthesizing", "synthesizing": "packaging"}
    return bool(
        dispatch["authorization"] == "continuation"
        and dispatch["predecessor_job_uuid"] == job.get("uuid")
        and isinstance(payload, dict)
        and isinstance(payload.get("phase"), str)
        and edges.get(payload["phase"]) == dispatch["phase"]
        and payload.get("checkpoint_id") == dispatch["checkpoint_id"]
        and type(payload.get("policy_version")) is int
        and payload["policy_version"] == dispatch["policy_version"]
        and type(payload.get("approved_outline_locked", False)) is bool
        and payload.get("approved_outline_locked", False) == dispatch["approved_outline_locked"]
    )


def adopt_bound_legacy_dispatch(
    db: ResearchSessionsDB, session: ResearchSessionRow, job: dict[str, Any]
) -> ResearchSessionRow:
    """Bootstrap only a verified current binding, never a phase with lost identity."""
    if session.phase_dispatch_json:
        read_phase_dispatch(session.phase_dispatch_json)
        return session
    payload = job.get("payload")
    if (
        session.active_job_id != str(job.get("id"))
        or not isinstance(payload, dict)
        or payload.get("session_id") != session.id
        or payload.get("phase") != session.phase
        or str(job.get("owner_user_id")) != session.owner_user_id
        or not isinstance(job.get("uuid"), str)
        or not job["uuid"]
        or any(
            job.get(key) != expected
            for key, expected in (
                ("domain", RESEARCH_DOMAIN),
                ("queue", RESEARCH_QUEUE),
                ("job_type", RESEARCH_JOB_TYPE),
            )
        )
        or "checkpoint_id" not in payload
        or "policy_version" not in payload
    ):
        return session
    dispatch = new_phase_dispatch(
        session.phase,
        checkpoint_id=payload["checkpoint_id"],
        policy_version=payload["policy_version"],
        approved_outline_locked=payload.get("approved_outline_locked", False),
    )
    dispatch["dispatch_id"] = payload.get("dispatch_id") or str(
        uuid.uuid5(uuid.NAMESPACE_URL, "research-job:" + job["uuid"])
    )
    dispatch["job_uuid"] = job["uuid"]
    updated = db.compare_and_set_phase(
        session,
        phase=session.phase,
        status=session.status,
        control_state=session.control_state,
        active_job_id=session.active_job_id,
        phase_dispatch_json=read_phase_dispatch(dispatch),
        completed_at=session.completed_at,
    )
    return updated or db.get_session(session.id)


def _current_candidate_authorized(session: ResearchSessionRow | None, job: dict[str, Any]) -> bool:
    """Cleanup cannot revoke an exact candidate retained by a newer Pause or Resume."""
    if (
        session is None
        or session.status != "queued"
        or session.control_state not in {"running", "pause_requested", "paused"}
        or session.active_job_id not in {None, str(job["id"])}
        or not session.phase_dispatch_json
        or str(job.get("owner_user_id")) != session.owner_user_id
        or any(
            job.get(key) != value
            for key, value in (
                ("domain", RESEARCH_DOMAIN),
                ("queue", RESEARCH_QUEUE),
                ("job_type", RESEARCH_JOB_TYPE),
            )
        )
    ):
        return False
    dispatch = read_phase_dispatch(session.phase_dispatch_json)
    payload = job.get("payload")
    return bool(
        isinstance(payload, dict)
        and dispatch["phase"] == session.phase
        and dispatch["job_uuid"] in {None, job["uuid"]}
        and payload.get("session_id") == session.id
        and dispatch_matches_job(dispatch, job)
    )


def _settle_bound_candidate(
    jm: Any, db: ResearchSessionsDB, session: ResearchSessionRow, job: dict[str, Any]
) -> ResearchSessionRow:
    """An ended candidate is not executable merely because admission returned its key."""
    if job.get("status") in {"failed", "quarantined", "cancelled"}:
        return reconcile_terminal_job(jm, db, session, job)
    if job.get("status") == "completed":
        return db.compare_and_set_phase(
            session,
            phase=session.phase,
            status="failed",
            control_state=session.control_state,
            active_job_id=None,
            phase_dispatch_json=session.phase_dispatch_json,
            completed_at=datetime.now(UTC).isoformat(),
            progress_message="The queued research phase has already ended. Review the run before retrying.",
        ) or db.get_session(session.id)
    return session


def dispatch_pending_phase(jm: Any, db: ResearchSessionsDB, session: ResearchSessionRow) -> ResearchSessionRow:
    """Queue and bind a committed successor before its parent is acknowledged."""
    if session.status != "queued" or session.control_state != "running" or not session.phase_dispatch_json:
        return session
    dispatch = read_phase_dispatch(session.phase_dispatch_json)
    if dispatch["phase"] != session.phase or dispatch["job_uuid"] is not None or session.active_job_id is not None:
        return session
    try:
        job = enqueue_research_phase_job(
            jm=jm,
            session_id=session.id,
            phase=session.phase,
            owner_user_id=session.owner_user_id,
            phase_dispatch=dispatch,
            continuation=dispatch["predecessor_job_uuid"] is not None,
        )
    except ValueError as error:
        db.compare_and_set_phase(
            session,
            phase=session.phase,
            status="failed",
            control_state=session.control_state,
            active_job_id=None,
            phase_dispatch_json=dispatch,
            completed_at=datetime.now(UTC).isoformat(),
            progress_message="This run could not be queued. Review the server limits before retrying.",
        )
        raise ResearchAdmissionRefusedError(str(error)) from error
    lookup = getattr(jm, "get_job_or_archived_by_uuid", None)
    if lookup is not None:
        stored = lookup(str(job["uuid"]), domain=RESEARCH_DOMAIN, owner_user_id=session.owner_user_id)
        if (
            stored is None
            or str(stored.get("id")) != str(job["id"])
            or str(stored.get("owner_user_id")) != session.owner_user_id
            or any(
                stored.get(key) != value
                for key, value in (
                    ("domain", RESEARCH_DOMAIN),
                    ("queue", RESEARCH_QUEUE),
                    ("job_type", RESEARCH_JOB_TYPE),
                )
            )
            or not dispatch_matches_job(dispatch, stored)
            or stored["payload"].get("session_id") != session.id
        ):
            raise ResearchPhaseChangedError("research_candidate_identity_changed")
        job = stored
    bound = read_phase_dispatch({**dispatch, "job_uuid": str(job["uuid"])})
    updated = db.compare_and_set_phase(
        session,
        phase=session.phase,
        status=session.status,
        control_state=session.control_state,
        active_job_id=str(job["id"]),
        phase_dispatch_json=bound,
    )
    if updated is not None:
        return _settle_bound_candidate(jm, db, updated, job)
    current = db.get_session(session.id)
    if _current_candidate_authorized(current, job):
        current_dispatch = read_phase_dispatch(current.phase_dispatch_json)
        if current.active_job_id == str(job["id"]) and current_dispatch["job_uuid"] == job["uuid"]:
            return _settle_bound_candidate(jm, db, current, job)
        rebound = db.compare_and_set_phase(
            current,
            phase=current.phase,
            status=current.status,
            control_state=current.control_state,
            active_job_id=str(job["id"]),
            phase_dispatch_json={**current_dispatch, "job_uuid": str(job["uuid"])},
        )
        if rebound is not None:
            return _settle_bound_candidate(jm, db, rebound, job)
        current = db.get_session(session.id)
        if _current_candidate_authorized(current, job):
            return current
    # A stale dispatcher cannot revoke a candidate that may already have advanced the run.
    if current is None:
        raise KeyError(session.id)
    return current


def reconcile_terminal_job(
    jm: Any, db: ResearchSessionsDB, session: ResearchSessionRow, job: dict[str, Any]
) -> ResearchSessionRow:
    """Replay exact terminal failure/cancellation bookkeeping after an interrupted callback."""
    stored = jm.get_job_or_archived_by_uuid(
        str(job.get("uuid") or ""),
        domain=RESEARCH_DOMAIN,
        owner_user_id=session.owner_user_id,
    )
    if not stored or stored.get("status") not in {"failed", "quarantined", "cancelled"}:
        return session
    if any(stored.get(key) != job.get(key) for key in ("uuid", "owner_user_id", "domain", "queue", "job_type")):
        return session
    if session.status in {"completed", "failed", "cancelled"}:
        return session
    session = adopt_bound_legacy_dispatch(db, session, stored)
    if not session.phase_dispatch_json:
        return session
    dispatch = read_phase_dispatch(session.phase_dispatch_json)
    bound = (
        session.active_job_id == str(stored.get("id"))
        and dispatch["job_uuid"] == stored["uuid"]
        and dispatch["phase"] == session.phase
        and dispatch_matches_job(dispatch, stored)
    )
    pending = (
        session.active_job_id is None
        and dispatch["job_uuid"] is None
        and predecessor_authorizes_continuation(dispatch, stored)
        and stored.get("payload", {}).get("session_id") == session.id
        and dispatch["phase"] == session.phase
    )
    if not (bound or pending):
        return session
    cancelled = stored["status"] == "cancelled"
    return db.compare_and_set_phase(
        session,
        phase=session.phase,
        status="cancelled" if cancelled else "failed",
        control_state="cancelled" if cancelled else session.control_state,
        active_job_id=None,
        phase_dispatch_json={},
        completed_at=datetime.now(UTC).isoformat(),
    ) or db.get_session(session.id)


def recover_owned_phase(jm: Any, db: ResearchSessionsDB, session: ResearchSessionRow) -> ResearchSessionRow:
    """Repair owned durable dispatch/failure state on normal reload, without rerunning a phase."""
    if session.status in {"completed", "failed", "cancelled"} or not session.phase_dispatch_json:
        return session
    dispatch = read_phase_dispatch(session.phase_dispatch_json)
    authority_uuid = dispatch["job_uuid"] or (
        dispatch["predecessor_job_uuid"] if dispatch["authorization"] == "continuation" else None
    )
    if authority_uuid:
        lookup = getattr(jm, "get_job_or_archived_by_uuid", None)
        # Older injected read-only managers cannot authorize recovery writes.
        if lookup is None:
            return session
        stored = lookup(authority_uuid, domain=RESEARCH_DOMAIN, owner_user_id=session.owner_user_id)
        if stored is None:
            return session
        if (
            session.active_job_id == str(stored.get("id"))
            and dispatch["job_uuid"] == stored.get("uuid")
            and dispatch["phase"] == session.phase
            and dispatch_matches_job(dispatch, stored)
        ):
            session = _settle_bound_candidate(jm, db, session, stored)
        else:
            session = reconcile_terminal_job(jm, db, session, stored)
    try:
        return dispatch_pending_phase(jm, db, session)
    except ResearchAdmissionRefusedError:
        current = db.get_session(session.id)
        if current is not None and current.status == "failed":
            return current
        raise
