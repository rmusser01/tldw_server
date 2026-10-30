"""Worker entrypoint for deep research Jobs slices."""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
    refresh_llm_provider_overrides,
    shutdown_llm_provider_override_recovery,
    start_llm_provider_override_refresh_service,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchPhaseChangedError, ResearchSessionsDB
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.Jobs.worker_utils import coerce_int as _coerce_int
from tldw_Server_API.app.core.Jobs.worker_utils import jobs_manager_from_env as _jobs_manager

from .jobs import RESEARCH_DOMAIN, RESEARCH_QUEUE, handle_research_phase_job
from .phase_dispatch import read_phase_dispatch
from .phase_jobs import (
    adopt_bound_legacy_dispatch,
    dispatch_matches_job,
    dispatch_pending_phase,
    predecessor_authorizes_continuation,
    reconcile_terminal_job,
)


class _ResearchJobOwnerError(ValueError):
    """Reject invalid authority without retrying a malformed persisted job."""

    retryable = False


class _ResearchPhaseAuthorityError(ValueError):
    """A retired phase must not retry or publish under a different Job lease."""

    retryable = False


async def run_research_jobs_worker(
    stop_event: asyncio.Event | None = None,
    *,
    research_db_path: str | Path | None = None,
    outputs_dir: str | Path | None = None,
) -> None:
    """Run the deep research worker loop until stopped."""
    worker_id = (os.getenv("RESEARCH_JOBS_WORKER_ID") or f"research-jobs-{os.getpid()}").strip()
    lease_seconds = _coerce_int(os.getenv("RESEARCH_JOBS_LEASE_SECONDS") or os.getenv("JOBS_LEASE_SECONDS"), 60)
    renew_jitter = _coerce_int(os.getenv("RESEARCH_JOBS_RENEW_JITTER_SECONDS") or os.getenv("JOBS_LEASE_RENEW_JITTER_SECONDS"), 5)
    renew_threshold = _coerce_int(os.getenv("RESEARCH_JOBS_RENEW_THRESHOLD_SECONDS") or os.getenv("JOBS_LEASE_RENEW_THRESHOLD_SECONDS"), 10)
    cfg = WorkerConfig(
        domain=RESEARCH_DOMAIN,
        queue=RESEARCH_QUEUE,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        renew_jitter_seconds=renew_jitter,
        renew_threshold_seconds=renew_threshold,
        bind_completion_token=True,
    )
    manager = _jobs_manager()
    if not manager.should_enforce_leases():
        raise ValueError("Research requires Jobs lease enforcement")
    sdk = WorkerSDK(manager, cfg)
    _stop_watcher_task: asyncio.Task[None] | None = None

    if stop_event is not None:
        async def _watch_stop() -> None:
            await stop_event.wait()
            sdk.stop()

        _stop_watcher_task = asyncio.create_task(_watch_stop())

    research_db_path_override = research_db_path or os.getenv("RESEARCH_SESSIONS_DB_PATH")
    outputs_dir_override = outputs_dir or os.getenv("RESEARCH_OUTPUTS_DIR")
    resolved_research_db_path = Path(research_db_path_override) if research_db_path_override else None
    resolved_outputs_dir = Path(outputs_dir_override) if outputs_dir_override else None

    def _paths_for_job(job: dict[str, object]) -> tuple[int, Path, Path]:
        owner = job.get("owner_user_id")
        if (
            isinstance(owner, bool)
            or not isinstance(owner, (str, int))
            or (isinstance(owner, int) and not 0 < owner < 2**63)
            or (isinstance(owner, str) and (not re.fullmatch(r"[0-9]{1,19}", owner) or not 0 < int(owner) < 2**63))
        ):
            raise _ResearchJobOwnerError("Missing or invalid research job owner")
        owner_user_id = int(owner)
        research_path = resolved_research_db_path or DatabasePaths.get_research_sessions_db_path(str(owner_user_id))
        output_path = resolved_outputs_dir or DatabasePaths.get_user_outputs_dir(str(owner_user_id))
        return owner_user_id, research_path, output_path

    async def _cancel_check(job: dict[str, object]) -> bool:
        return bool(job.get("cancel_requested_at") or job.get("cancelled_at"))

    async def _handler(job: dict[str, object]) -> dict[str, object]:
        owner, research_path, output_path = _paths_for_job(job)
        payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
        db = ResearchSessionsDB(research_path)
        session = db.get_session(str(payload.get("session_id") or "").strip())
        if session is None or session.owner_user_id != str(owner):
            raise _ResearchJobOwnerError("Research session does not belong to the job owner")
        with scoped_context(user_id=owner, is_admin=False):

            def require_lease() -> None:
                stored = manager.get_job(int(job["id"]), owner_user_id=str(owner))
                if (
                    not stored
                    or any(
                        stored.get(key) != job.get(key)
                        for key in (
                            "id",
                            "uuid",
                            "owner_user_id",
                            "domain",
                            "queue",
                            "job_type",
                            "worker_id",
                            "lease_id",
                            "payload",
                        )
                    )
                    or not manager.has_live_processing_lease(int(job["id"]), owner_user_id=str(owner))
                    or not manager.renew_job_lease(
                        int(job["id"]),
                        seconds=lease_seconds,
                        worker_id=worker_id,
                        lease_id=str(job["lease_id"]),
                        enforce=True,
                    )
                ):
                    raise _ResearchPhaseAuthorityError("research_phase_lease_lost")

            require_lease()
            session = adopt_bound_legacy_dispatch(db, session, job)
            if not session.phase_dispatch_json:
                return {"obsolete": True}
            dispatch = read_phase_dispatch(session.phase_dispatch_json)
            if dispatch["phase"] != session.phase:
                return {"obsolete": True}
            if session.control_state in {"cancel_requested", "cancelled"} or session.status in {
                "completed",
                "failed",
                "cancelled",
            }:
                if session.status not in {"completed", "failed", "cancelled"}:
                    db.compare_and_set_phase(
                        session,
                        phase=session.phase,
                        status="cancelled",
                        control_state="cancelled",
                        active_job_id=None,
                        phase_dispatch_json={},
                        completed_at=datetime.now(UTC).isoformat(),
                    )
                manager.finalize_cancelled(
                    int(job["id"]),
                    expected_uuid=str(job["uuid"]),
                    worker_id=worker_id,
                    lease_id=str(job["lease_id"]),
                    reason="research_not_executable",
                )
                return {"obsolete": True}
            bound = session.active_job_id == str(job["id"]) and dispatch["job_uuid"] == job["uuid"]
            if not bound or not dispatch_matches_job(dispatch, job):
                if predecessor_authorizes_continuation(dispatch, job) and dispatch["job_uuid"] is None:
                    current = dispatch_pending_phase(manager, db, session)
                    return {"session_id": current.id, "phase": current.phase, "handoff_reconciled": True}
                if dispatch["job_uuid"] is None and dispatch_matches_job(dispatch, job):
                    manager.release_job(
                        int(job["id"]),
                        worker_id=worker_id,
                        lease_id=str(job["lease_id"]),
                        reason="research_dispatch_unbound",
                        enforce=True,
                    )
                    return {"deferred": True}
                return {"obsolete": True}
            if session.control_state in {"pause_requested", "paused"}:
                parked = db.compare_and_set_phase(
                    session,
                    phase=session.phase,
                    status=session.status,
                    control_state="paused",
                    active_job_id=None,
                    phase_dispatch_json=dispatch,
                )
                if parked is None:
                    manager.release_job(
                        int(job["id"]),
                        worker_id=worker_id,
                        lease_id=str(job["lease_id"]),
                        reason="research_control_changed",
                        enforce=True,
                    )
                    return {"deferred": True}
                return {"session_id": session.id, "phase": session.phase, "paused": True}
            try:
                result = await handle_research_phase_job(
                    job,
                    research_db_path=research_path,
                    outputs_dir=output_path,
                    lease_check=require_lease,
                )
            except (ResearchPhaseChangedError, _ResearchPhaseAuthorityError):
                current = db.get_session(session.id)
                stored = manager.get_job(int(job["id"]), owner_user_id=str(owner))
                if current is not None and (
                    current.control_state in {"cancel_requested", "cancelled"}
                    or (stored and stored.get("cancel_requested_at"))
                ):
                    cancelled = manager.finalize_cancelled(
                        int(job["id"]),
                        expected_uuid=str(job["uuid"]),
                        worker_id=worker_id,
                        lease_id=str(job["lease_id"]),
                        reason="research_cancelled",
                    )
                    if cancelled:
                        reconciled = reconcile_terminal_job(manager, db, current, job)
                        return {"session_id": current.id, "phase": reconciled.phase, "cancelled": True}
                raise
            current = db.get_session(session.id)
            if current is None:
                return {"obsolete": True}
            if current.status == "cancelled":
                manager.finalize_cancelled(
                    int(job["id"]),
                    expected_uuid=str(job["uuid"]),
                    worker_id=worker_id,
                    lease_id=str(job["lease_id"]),
                    reason="research_cancelled",
                )
                return result
            require_lease()
            if result.get("retry_delivery"):
                manager.release_job(
                    int(job["id"]),
                    worker_id=worker_id,
                    lease_id=str(job["lease_id"]),
                    reason="research_control_changed",
                    enforce=True,
                )
                return {"deferred": True}
            dispatch_pending_phase(manager, db, current)
            return result

    async def _on_failed(job: dict[str, object], _error: Exception) -> None:
        owner, research_path, _output_path = _paths_for_job(job)
        db = ResearchSessionsDB(research_path)
        payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
        session = db.get_session(str(payload.get("session_id") or "").strip())
        if session is not None and session.owner_user_id == str(owner):
            with scoped_context(user_id=owner, is_admin=False):
                reconcile_terminal_job(manager, db, session, job)

    logger.info("Research Jobs worker starting: queue={} worker_id={}", RESEARCH_QUEUE, worker_id)
    try:
        await sdk.run(handler=_handler, cancel_check=_cancel_check, on_failed=_on_failed)
    finally:
        if _stop_watcher_task is not None and not _stop_watcher_task.done():
            _stop_watcher_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await _stop_watcher_task


async def main(stop_event: asyncio.Event | None = None) -> None:
    """Own standalone provider policy initialization and its event-loop resources."""
    pool = await get_db_pool()
    try:
        await refresh_llm_provider_overrides(pool, force=True)
        start_llm_provider_override_refresh_service()
        await run_research_jobs_worker(stop_event)
    finally:
        try:
            await shutdown_llm_provider_override_recovery()
        finally:
            await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
