"""Shared Jobs failure lifecycle, split at transaction and disposition boundaries.

The manager facade retains its public API. These helpers preserve backend-specific
SQL and stage observers until the owning transaction has committed.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from . import manager as runtime

if TYPE_CHECKING:
    from .manager import JobManager
    from .operations.resource import ResourceMutation


@dataclass(frozen=True)
class FailureHooks:
    """Environment policy and diagnostics supplied by the manager facade."""

    initialize_diagnostics: Callable[[], None]
    log_enter: Callable[[str], None]
    retry_parameters: Callable[[int, str], tuple[int, int]]
    snapshot_retry: Callable[[Any, Any], None]
    log_retry: Callable[[Any, str, int], None]
    log_terminal: Callable[[Any, str, bool], None]


@dataclass
class FailureOperation:
    """Inputs and transaction-local staged effects for one failure attempt."""

    job_id: int
    error: str
    retryable: bool
    backoff_seconds: int
    worker_id: str | None
    lease_id: str | None
    enforce: bool
    error_code: str | None
    error_class: str | None
    error_stack: dict[str, Any] | None
    completion_token: str | None
    quarantine: bool
    expected_uuid: str | None
    streak_code: str
    outbox_enabled: bool
    hooks: FailureHooks
    post_commit_side_effects: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)
    post_commit_cancel_uuid: str | None = None
    post_commit_cancel_reason: str | None = None
    current: int = 0
    delay: int = 0
    thresh: int = 0
    transition_status: str = ""


def _update_retry_postgres(op: FailureOperation, cur: Any) -> bool:
    """Apply the fenced PostgreSQL retry/quarantine update."""
    if op.enforce:
        cur.execute(
            (
                "UPDATE jobs SET status = CASE WHEN (CASE WHEN COALESCE(failure_streak_code, '') = %s THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END) >= %s THEN 'quarantined' ELSE 'queued' END, "
                "retry_count = retry_count + 1, last_error = %s, error_message = %s, error_code = %s, error_class = %s, error_stack = %s::jsonb, "
                "failure_streak_count = CASE WHEN COALESCE(failure_streak_code, '') = %s THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END, "
                "failure_streak_code = %s, "
                "completion_token = CASE WHEN (CASE WHEN COALESCE(failure_streak_code, '') = %s THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END) >= %s THEN %s ELSE NULL END, "
                "available_at = CASE WHEN (CASE WHEN COALESCE(failure_streak_code, '') = %s THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END) >= %s THEN available_at WHEN %s <= 0 THEN NULL ELSE NOW() + (%s || ' seconds')::interval END, "
                "quarantined_at = CASE WHEN (CASE WHEN COALESCE(failure_streak_code, '') = %s THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END) >= %s THEN NOW() ELSE quarantined_at END, "
                "leased_until = NULL, worker_id = NULL, lease_id = NULL "
                "WHERE id = %s AND status = 'processing' AND retry_count < max_retries AND worker_id = %s AND lease_id = %s AND (completion_token IS NULL OR completion_token = %s)"
            ),
            (
                op.streak_code,
                int(op.thresh),
                op.streak_code,
                op.error,
                op.error_code,
                op.error_class,
                (json.dumps(op.error_stack) if op.error_stack is not None else None),
                op.streak_code,
                op.streak_code,
                op.streak_code,
                int(op.thresh),
                op.completion_token,
                op.streak_code,
                int(op.thresh),
                int(op.delay),
                int(op.delay),
                op.streak_code,
                int(op.thresh),
                int(op.job_id),
                op.worker_id,
                op.lease_id,
                op.completion_token,
            ),
        )
    else:
        cur.execute(
            (
                "UPDATE jobs SET status = CASE WHEN (CASE WHEN COALESCE(failure_streak_code, '') = %s THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END) >= %s THEN 'quarantined' ELSE 'queued' END, "
                "retry_count = retry_count + 1, last_error = %s, error_message = %s, error_code = %s, error_class = %s, error_stack = %s::jsonb, "
                "failure_streak_count = CASE WHEN COALESCE(failure_streak_code, '') = %s THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END, "
                "failure_streak_code = %s, "
                "completion_token = CASE WHEN (CASE WHEN COALESCE(failure_streak_code, '') = %s THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END) >= %s THEN %s ELSE NULL END, "
                "available_at = CASE WHEN (CASE WHEN COALESCE(failure_streak_code, '') = %s THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END) >= %s THEN available_at WHEN %s <= 0 THEN NULL ELSE NOW() + (%s || ' seconds')::interval END, "
                "quarantined_at = CASE WHEN (CASE WHEN COALESCE(failure_streak_code, '') = %s THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END) >= %s THEN NOW() ELSE quarantined_at END, "
                "leased_until = NULL, worker_id = NULL, lease_id = NULL "
                "WHERE id = %s AND status = 'processing' AND retry_count < max_retries AND (completion_token IS NULL OR completion_token = %s)"
            ),
            (
                op.streak_code,
                int(op.thresh),
                op.streak_code,
                op.error,
                op.error_code,
                op.error_class,
                (json.dumps(op.error_stack) if op.error_stack is not None else None),
                op.streak_code,
                op.streak_code,
                op.streak_code,
                int(op.thresh),
                op.completion_token,
                op.streak_code,
                int(op.thresh),
                int(op.delay),
                int(op.delay),
                op.streak_code,
                int(op.thresh),
                int(op.job_id),
                op.completion_token,
            ),
        )
    return cur.rowcount > 0


def _finish_retry_postgres(
    manager: JobManager, op: FailureOperation, conn: Any, cur: Any, mutation: ResourceMutation, elem: dict[str, Any]
) -> bool:
    """Stage retry accounting, commit, then publish effects and cancellation."""
    cur.execute(
        "SELECT status, uuid FROM jobs WHERE id = %s",
        (int(op.job_id),),
    )
    _srow = cur.fetchone()
    op.transition_status = str((_srow or {}).get("status") or "")
    if op.transition_status == "quarantined":
        op.post_commit_cancel_uuid = (_srow or {}).get("uuid")
        op.post_commit_cancel_reason = "dependency_failed"
    try:
        if elem:
            if op.transition_status == "queued":
                op.post_commit_side_effects.append(
                    (
                        runtime.increment_retries,
                        (dict(elem),),
                        {},
                    )
                )
                try:
                    from .metrics import observe_retry_after

                    op.post_commit_side_effects.append(
                        (
                            observe_retry_after,
                            (dict(elem), float(op.delay)),
                            {},
                        )
                    )
                except runtime._JOB_NONCRITICAL_EXCEPTIONS:
                    pass
            # Append to failure_timeline (retryable)
            try:
                cur.execute("SELECT failure_timeline FROM jobs WHERE id = %s", (int(op.job_id),))
                _tlrow = cur.fetchone()
                tl_val = _tlrow.get("failure_timeline") if _tlrow else None
                if isinstance(tl_val, list):
                    tl = tl_val
                elif isinstance(tl_val, str):
                    try:
                        tl = json.loads(tl_val)
                    except runtime._JOB_NONCRITICAL_EXCEPTIONS:
                        tl = []
                else:
                    tl = []
                tl.append(
                    {
                        "ts": datetime.utcnow().isoformat(),
                        "error_code": (op.error_code or op.error),
                        "retry_backoff": int(op.delay),
                    }
                )
                tl = tl[-10:]
                cur.execute(
                    "UPDATE jobs SET failure_timeline = %s::jsonb WHERE id = %s",
                    (json.dumps(tl), int(op.job_id)),
                )
            except runtime._JOB_NONCRITICAL_EXCEPTIONS:
                pass
    except runtime._JOB_NONCRITICAL_EXCEPTIONS:
        pass
    if elem and runtime.JobManager._is_truthy(os.getenv("JOBS_COUNTERS_ENABLED", "")):
        will_quarantine = op.transition_status == "quarantined"
        add_ready = int(not will_quarantine and int(op.delay) <= 0)
        add_sched = int(not will_quarantine and int(op.delay) > 0)
        add_quar = int(will_quarantine)
        cur.execute(
            (
                "INSERT INTO job_counters(domain,queue,job_type,ready_count,scheduled_count,processing_count,quarantined_count) VALUES(%s,%s,%s,%s,%s,0,%s) "
                "ON CONFLICT (domain,queue,job_type) DO UPDATE SET "
                "ready_count = job_counters.ready_count + %s, "
                "scheduled_count = job_counters.scheduled_count + %s, "
                "processing_count = GREATEST(job_counters.processing_count - 1, 0), "
                "quarantined_count = job_counters.quarantined_count + %s, "
                "updated_at = NOW()"
            ),
            (
                elem.get("domain"),
                elem.get("queue"),
                elem.get("job_type"),
                add_ready,
                add_sched,
                add_quar,
                add_ready,
                add_sched,
                add_quar,
            ),
        )
    op.hooks.log_retry(cur, "postgres", op.delay)
    if elem:
        op.post_commit_side_effects.append(
            (
                manager._update_gauges,
                (),
                {
                    "domain": elem.get("domain"),
                    "queue": elem.get("queue"),
                    "job_type": elem.get("job_type"),
                },
            )
        )
        event_type = "job.quarantined" if op.transition_status == "quarantined" else "job.retry_scheduled"
        attrs = {
            "error_code": (op.error_code or op.error),
            "retry_count": int(op.current + 1),
        }
        if op.transition_status == "queued":
            attrs["backoff_seconds"] = int(op.delay)
        event_job = {
            "id": int(op.job_id),
            "domain": elem.get("domain"),
            "queue": elem.get("queue"),
            "job_type": elem.get("job_type"),
            "owner_user_id": elem.get("owner_user_id"),
            "request_id": elem.get("request_id"),
            "trace_id": elem.get("trace_id"),
        }
        if op.outbox_enabled:
            runtime._insert_lifecycle_event(
                cur,
                backend=manager.backend,
                event_type=event_type,
                job=event_job,
                attrs=attrs,
            )
        runtime._queue_lifecycle_event_observer(
            op.post_commit_side_effects,
            event_type=event_type,
            job=event_job,
            attrs=attrs,
        )
    mutation.finish()
    runtime._commit_postgres_transaction(
        conn,
        operation="job retry scheduling",
    )
    runtime._run_post_commit_side_effects(op.post_commit_side_effects)
    op.post_commit_side_effects.clear()
    if op.post_commit_cancel_uuid:
        with contextlib.suppress(runtime._JOB_NONCRITICAL_EXCEPTIONS):
            manager._cancel_dependent_jobs(
                op.post_commit_cancel_uuid,
                reason=op.post_commit_cancel_reason or "dependency_failed",
            )
    return True


def _retry_postgres(
    manager: JobManager, op: FailureOperation, conn: Any, cur: Any, mutation: ResourceMutation, elem: dict[str, Any]
) -> bool | None:
    """Try retry scheduling or a terminal token replay; None falls through."""
    cur.execute("SELECT retry_count FROM jobs WHERE id = %s", (int(op.job_id),))
    row = cur.fetchone()
    op.current = int(row["retry_count"]) if row else 0
    op.delay, op.thresh = op.hooks.retry_parameters(op.current, manager.backend)
    retry_transition_changed = _update_retry_postgres(op, cur)
    if retry_transition_changed:
        return _finish_retry_postgres(manager, op, conn, cur, mutation, elem)
    if op.completion_token:
        cur.execute(
            "SELECT status, completion_token FROM jobs WHERE id=%s",
            (int(op.job_id),),
        )
        replay = cur.fetchone()
        if (
            replay
            and str(replay.get("status")) in {"failed", "quarantined"}
            and str(replay.get("completion_token") or "") == str(op.completion_token)
        ):
            return True
    return None


def _update_terminal_postgres(op: FailureOperation, cur: Any) -> tuple[bool, bool]:
    """Apply the processing failure or permitted queued fallback."""
    # terminal failure
    failed_from_queued = False
    if op.enforce:
        cur.execute(
            (
                "UPDATE jobs SET status = 'failed', last_error = %s, error_message = %s, error_code = %s, error_class = %s, error_stack = %s::jsonb, completion_token = %s, "
                "completed_at = NOW(), leased_until = NULL, worker_id = NULL, lease_id = NULL WHERE id = %s AND status = 'processing' AND worker_id = %s AND lease_id = %s AND (completion_token IS NULL OR completion_token = %s)"
            ),
            (
                (op.error_code or op.error),
                op.error,
                op.error_code,
                op.error_class,
                (json.dumps(op.error_stack) if op.error_stack is not None else None),
                op.completion_token,
                int(op.job_id),
                op.worker_id,
                op.lease_id,
                op.completion_token,
            ),
        )
        failed_from_processing = cur.rowcount > 0
    else:
        cur.execute(
            (
                "UPDATE jobs SET status = 'failed', last_error = %s, error_message = %s, error_code = %s, error_class = %s, error_stack = %s::jsonb, completion_token = COALESCE(completion_token, %s), "
                "completed_at = NOW(), leased_until = NULL, worker_id = NULL, lease_id = NULL WHERE id = %s AND status = 'processing' AND (completion_token IS NULL OR completion_token = %s)"
            ),
            (
                (op.error_code or op.error),
                op.error,
                op.error_code,
                op.error_class,
                (json.dumps(op.error_stack) if op.error_stack is not None else None),
                op.completion_token,
                int(op.job_id),
                op.completion_token,
            ),
        )
        failed_from_processing = cur.rowcount > 0
        if not failed_from_processing:
            try:
                allow = {
                    d.strip().lower()
                    for d in os.getenv(
                        "JOBS_ADMIN_COMPLETE_QUEUED_ALLOW_DOMAINS",
                        "chatbooks,embeddings",
                    ).split(",")
                    if d.strip()
                }
                cur.execute("SELECT domain FROM jobs WHERE id = %s", (int(op.job_id),))
                row_dom = cur.fetchone()
                dom_val = str(row_dom.get("domain") or "").lower() if row_dom else ""
            except runtime._JOB_NONCRITICAL_EXCEPTIONS:
                allow = {"chatbooks", "embeddings"}
                dom_val = ""
            if dom_val in allow:
                cur.execute(
                    (
                        "UPDATE jobs SET status = 'failed', last_error = %s, error_message = %s, error_code = %s, error_class = %s, error_stack = %s::jsonb, completion_token = COALESCE(completion_token, %s), "
                        "completed_at = NOW(), leased_until = NULL, worker_id = NULL, lease_id = NULL WHERE id = %s AND status = 'queued' AND (completion_token IS NULL OR completion_token = %s)"
                    ),
                    (
                        (op.error_code or op.error),
                        op.error,
                        op.error_code,
                        op.error_class,
                        (json.dumps(op.error_stack) if op.error_stack is not None else None),
                        op.completion_token,
                        int(op.job_id),
                        op.completion_token,
                    ),
                )
                failed_from_queued = cur.rowcount > 0
    ok = failed_from_processing or failed_from_queued
    return ok, failed_from_queued


def _stage_terminal_postgres(
    manager: JobManager, op: FailureOperation, cur: Any, elem: dict[str, Any], ok: bool, failed_from_queued: bool
) -> None:
    """Persist terminal accounting and stage observers without committing."""
    if ok and (op.error_code or op.error_class or op.error_stack is not None):
        with contextlib.suppress(runtime._JOB_NONCRITICAL_EXCEPTIONS):
            cur.execute(
                (
                    "UPDATE jobs SET error_code = COALESCE(%s, error_code), "
                    "error_class = COALESCE(%s, error_class), "
                    "error_stack = COALESCE(%s::jsonb, error_stack) "
                    "WHERE id = %s"
                ),
                (
                    op.error_code,
                    op.error_class,
                    (json.dumps(op.error_stack) if op.error_stack is not None else None),
                    int(op.job_id),
                ),
            )
    try:
        if ok and elem:
            d = dict(elem)
            op.post_commit_side_effects.append((runtime.increment_failures, (d,), {"reason": "terminal"}))
            try:
                if op.error_code:
                    from .metrics import increment_failures_by_code

                    op.post_commit_side_effects.append(
                        (
                            increment_failures_by_code,
                            (d, op.error_code),
                            {},
                        )
                    )
            except runtime._JOB_NONCRITICAL_EXCEPTIONS:
                pass
            try:
                # Append terminal failure to timeline (no backoff)
                cur.execute(
                    "UPDATE jobs SET failure_timeline = COALESCE(failure_timeline, '[]'::jsonb) || jsonb_build_array(jsonb_build_object('ts', NOW(), 'error_code', %s::text, 'retry_backoff', 0)) WHERE id = %s",
                    ((op.error_code or op.error), int(op.job_id)),
                )

                op.post_commit_side_effects.append(
                    (
                        runtime._record_job_span,
                        ("job.fail", d),
                        {
                            "attrs": {
                                "retryable": False,
                                "error_code": op.error_code,
                            }
                        },
                    )
                )
            except runtime._PG_ERRORS:
                raise
            except runtime._JOB_NONCRITICAL_EXCEPTIONS:
                pass
            op.post_commit_side_effects.append(
                (
                    manager._update_gauges,
                    (),
                    {
                        "domain": d.get("domain"),
                        "queue": d.get("queue"),
                        "job_type": d.get("job_type"),
                    },
                )
            )
            op.post_commit_cancel_uuid = d.get("uuid")
            op.post_commit_cancel_reason = "dependency_failed"
    except runtime._PG_ERRORS:
        raise
    except runtime._JOB_NONCRITICAL_EXCEPTIONS:
        pass
    if ok and elem and runtime.JobManager._is_truthy(os.getenv("JOBS_COUNTERS_ENABLED", "")):
        d_counter = dict(elem)
        if failed_from_queued:
            is_scheduled = d_counter.get("available_at") is not None
            cur.execute(
                (
                    "UPDATE job_counters SET "
                    "ready_count = GREATEST(ready_count - %s, 0), "
                    "scheduled_count = GREATEST(scheduled_count - %s, 0), "
                    "updated_at = NOW() "
                    "WHERE domain=%s AND queue=%s AND job_type=%s"
                ),
                (
                    int(not is_scheduled),
                    int(is_scheduled),
                    d_counter.get("domain"),
                    d_counter.get("queue"),
                    d_counter.get("job_type"),
                ),
            )
        else:
            cur.execute(
                "UPDATE job_counters SET processing_count = GREATEST(processing_count - 1, 0), updated_at = NOW() WHERE domain=%s AND queue=%s AND job_type=%s",
                (
                    d_counter.get("domain"),
                    d_counter.get("queue"),
                    d_counter.get("job_type"),
                ),
            )
        if cur.rowcount == 0:
            runtime._reconcile_lifecycle_counter_row(
                cur,
                backend=manager.backend,
                domain=d_counter.get("domain"),
                queue=d_counter.get("queue"),
                job_type=d_counter.get("job_type"),
            )
    if ok and elem:
        event_job = {
            "id": int(op.job_id),
            "domain": elem.get("domain"),
            "queue": elem.get("queue"),
            "job_type": elem.get("job_type"),
            "owner_user_id": elem.get("owner_user_id"),
            "request_id": elem.get("request_id"),
            "trace_id": elem.get("trace_id"),
        }
        attrs = {"error_code": (op.error_code or op.error)}
        if op.outbox_enabled:
            runtime._insert_lifecycle_event(
                cur,
                backend=manager.backend,
                event_type="job.failed",
                job=event_job,
                attrs=attrs,
            )
        runtime._queue_lifecycle_event_observer(
            op.post_commit_side_effects,
            event_type="job.failed",
            job=event_job,
            attrs=attrs,
        )


def _terminal_postgres(
    manager: JobManager, op: FailureOperation, conn: Any, cur: Any, mutation: ResourceMutation, elem: dict[str, Any]
) -> bool:
    """Finalize terminal failure and publish only after the explicit commit."""
    ok, failed_from_queued = _update_terminal_postgres(op, cur)
    if not ok and op.completion_token:
        cur.execute(
            "SELECT status, completion_token FROM jobs WHERE id=%s",
            (int(op.job_id),),
        )
        replay = cur.fetchone()
        if (
            replay
            and str(replay.get("status")) == "failed"
            and str(replay.get("completion_token") or "") == str(op.completion_token)
        ):
            return True
    _stage_terminal_postgres(manager, op, cur, elem, ok, failed_from_queued)
    op.hooks.log_terminal(cur, "postgres", ok)
    if ok:
        mutation.finish()
        runtime._commit_postgres_transaction(
            conn,
            operation="terminal job failure",
        )
        runtime._run_post_commit_side_effects(op.post_commit_side_effects)
        op.post_commit_side_effects.clear()
        if op.post_commit_cancel_uuid:
            with contextlib.suppress(runtime._JOB_NONCRITICAL_EXCEPTIONS):
                manager._cancel_dependent_jobs(
                    op.post_commit_cancel_uuid,
                    reason=op.post_commit_cancel_reason or "dependency_failed",
                )
    return ok


def _fail_postgres(manager: JobManager, op: FailureOperation, conn: Any) -> bool:
    """Own the PostgreSQL row/resource fence and failure transaction."""
    with conn:
        with manager._resource_pg_cursor(
            conn, where="id=%s", params=(int(op.job_id),), expected_uuid=op.expected_uuid
        ) as (cur, mutation):
            if op.expected_uuid is not None and int(op.job_id) not in mutation.rows:
                return False
            # For metrics and idempotency
            op.hooks.log_enter("pg")
            cur.execute(
                "SELECT status, completion_token, retry_count, failure_streak_code, failure_streak_count, "
                "domain, queue, job_type, uuid, request_id, trace_id, owner_user_id, available_at "
                f"FROM jobs WHERE id = %s AND ({mutation.where}) FOR UPDATE",  # nosec B608 - closed resource identity predicate; values bound.
                (int(op.job_id), *mutation.params),
            )
            elem = cur.fetchone()
            if elem is None or (op.expected_uuid is not None and str(elem.get("uuid") or "") != str(op.expected_uuid)):
                return False
            if not mutation.worker_allowed(int(op.job_id), op.worker_id, op.lease_id, op.expected_uuid):
                return False
            if elem:
                st = str(elem.get("status"))
                ct = elem.get("completion_token")
                if st in {"completed", "failed", "cancelled", "quarantined"}:
                    replay_states = {"failed", "quarantined"} if op.retryable else {"failed"}
                    return bool(
                        st in replay_states and op.completion_token and ct and str(ct) == str(op.completion_token)
                    )
            if op.retryable:
                retry_result = _retry_postgres(manager, op, conn, cur, mutation, elem)
                if retry_result is not None:
                    return retry_result
            return _terminal_postgres(manager, op, conn, cur, mutation, elem)


def _update_retry_sqlite(op: FailureOperation, conn: Any) -> bool:
    """Apply the fenced SQLite retry/quarantine update."""
    if op.enforce:
        retry_cursor = conn.execute(
            (
                "UPDATE jobs SET status = CASE WHEN (CASE WHEN COALESCE(failure_streak_code, '') = ? THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END) >= ? THEN 'quarantined' ELSE 'queued' END, "
                "retry_count = retry_count + 1, last_error = ?, error_message = ?, error_code = ?, error_class = ?, error_stack = ?, "
                "failure_streak_count = CASE WHEN COALESCE(failure_streak_code, '') = ? THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END, "
                "failure_streak_code = ?, "
                "completion_token = CASE WHEN (CASE WHEN COALESCE(failure_streak_code, '') = ? THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END) >= ? THEN ? ELSE NULL END, "
                "available_at = CASE WHEN (CASE WHEN COALESCE(failure_streak_code, '') = ? THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END) >= ? THEN available_at WHEN ? <= 0 THEN NULL ELSE DATETIME('now', ?) END, "
                "quarantined_at = CASE WHEN (CASE WHEN COALESCE(failure_streak_code, '') = ? THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END) >= ? THEN DATETIME('now') ELSE quarantined_at END, "
                "leased_until = NULL, worker_id = NULL, lease_id = NULL "
                "WHERE id = ? AND status = 'processing' AND retry_count < max_retries AND worker_id = ? AND lease_id = ? AND (completion_token IS NULL OR completion_token = ?)"
            ),
            (
                op.streak_code,
                int(op.thresh),
                op.streak_code,
                op.error,
                op.error_code,
                op.error_class,
                (json.dumps(op.error_stack) if op.error_stack is not None else None),
                op.streak_code,
                op.streak_code,
                op.streak_code,
                int(op.thresh),
                op.completion_token,
                op.streak_code,
                int(op.thresh),
                int(op.delay),
                f"+{op.delay} seconds",
                op.streak_code,
                int(op.thresh),
                op.job_id,
                op.worker_id,
                op.lease_id,
                op.completion_token,
            ),
        )
    else:
        retry_cursor = conn.execute(
            (
                "UPDATE jobs SET status = CASE WHEN (CASE WHEN COALESCE(failure_streak_code, '') = ? THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END) >= ? THEN 'quarantined' ELSE 'queued' END, "
                "retry_count = retry_count + 1, last_error = ?, error_message = ?, error_code = ?, error_class = ?, error_stack = ?, "
                "failure_streak_count = CASE WHEN COALESCE(failure_streak_code, '') = ? THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END, "
                "failure_streak_code = ?, "
                "completion_token = CASE WHEN (CASE WHEN COALESCE(failure_streak_code, '') = ? THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END) >= ? THEN ? ELSE NULL END, "
                "available_at = CASE WHEN (CASE WHEN COALESCE(failure_streak_code, '') = ? THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END) >= ? THEN available_at WHEN ? <= 0 THEN NULL ELSE DATETIME('now', ?) END, "
                "quarantined_at = CASE WHEN (CASE WHEN COALESCE(failure_streak_code, '') = ? THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END) >= ? THEN DATETIME('now') ELSE quarantined_at END, "
                "leased_until = NULL, worker_id = NULL, lease_id = NULL "
                "WHERE id = ? AND status = 'processing' AND retry_count < max_retries AND (completion_token IS NULL OR completion_token = ?)"
            ),
            (
                op.streak_code,
                int(op.thresh),
                op.streak_code,
                op.error,
                op.error_code,
                op.error_class,
                (json.dumps(op.error_stack) if op.error_stack is not None else None),
                op.streak_code,
                op.streak_code,
                op.streak_code,
                int(op.thresh),
                op.completion_token,
                op.streak_code,
                int(op.thresh),
                int(op.delay),
                f"+{op.delay} seconds",
                op.streak_code,
                int(op.thresh),
                op.job_id,
                op.completion_token,
            ),
        )
    return (retry_cursor.rowcount or 0) > 0


def _stage_retry_sqlite(manager: JobManager, op: FailureOperation, conn: Any, rowl: Any) -> None:
    """Persist retry accounting and stage effects inside the SQLite transaction."""
    rowq = conn.execute(
        "SELECT status, uuid FROM jobs WHERE id = ?",
        (op.job_id,),
    ).fetchone()
    op.transition_status = str(rowq[0] if rowq else "")
    if op.transition_status == "quarantined":
        op.post_commit_cancel_uuid = rowq[1]
        op.post_commit_cancel_reason = "dependency_failed"
    try:
        if rowl:
            dtmp = dict(rowl)
            if op.transition_status == "queued":
                op.post_commit_side_effects.append(
                    (
                        runtime.increment_retries,
                        (dtmp,),
                        {},
                    )
                )
                try:
                    from .metrics import observe_retry_after

                    op.post_commit_side_effects.append(
                        (
                            observe_retry_after,
                            (dtmp, float(op.delay)),
                            {},
                        )
                    )
                except runtime._JOB_NONCRITICAL_EXCEPTIONS:
                    pass
            op.post_commit_side_effects.append(
                (
                    manager._update_gauges,
                    (),
                    {
                        "domain": dtmp.get("domain"),
                        "queue": dtmp.get("queue"),
                        "job_type": dtmp.get("job_type"),
                    },
                )
            )
            # Append to failure_timeline
            try:
                row_t = conn.execute("SELECT failure_timeline FROM jobs WHERE id = ?", (op.job_id,)).fetchone()
                timeline_json = row_t[0] if row_t else None
                try:
                    tl = json.loads(timeline_json) if timeline_json else []
                except runtime._JOB_NONCRITICAL_EXCEPTIONS:
                    tl = []
                tl.append(
                    {
                        "ts": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                        "error_code": (op.error_code or op.error),
                        "retry_backoff": int(op.delay),
                    }
                )
                tl = tl[-10:]
                conn.execute(
                    "UPDATE jobs SET failure_timeline = ? WHERE id = ?",
                    (json.dumps(tl), int(op.job_id)),
                )
                # Update last-acquired snapshot for test-mode fallbacks to preserve timeline growth
                op.hooks.snapshot_retry(conn, rowl)
            except runtime._JOB_NONCRITICAL_EXCEPTIONS:
                pass
    except runtime._JOB_NONCRITICAL_EXCEPTIONS:
        pass
    if rowl and runtime.JobManager._is_truthy(os.getenv("JOBS_COUNTERS_ENABLED", "")):
        will_quarantine = op.transition_status == "quarantined"
        add_ready = int(not will_quarantine and int(op.delay) <= 0)
        add_sched = int(not will_quarantine and int(op.delay) > 0)
        add_quar = int(will_quarantine)
        conn.execute(
            (
                "INSERT INTO job_counters(domain,queue,job_type,ready_count,scheduled_count,processing_count,quarantined_count) VALUES(?,?,?, ?, ?, 0, ?) "
                "ON CONFLICT(domain,queue,job_type) DO UPDATE SET ready_count = ready_count + ?, scheduled_count = scheduled_count + ?, processing_count = CASE WHEN processing_count>0 THEN processing_count-1 ELSE 0 END, quarantined_count = quarantined_count + ?, updated_at = DATETIME('now')"
            ),
            (
                rowl[2],
                rowl[3],
                rowl[4],
                add_ready,
                add_sched,
                add_quar,
                add_ready,
                add_sched,
                add_quar,
            ),
        )
    if rowl:
        dtmp = dict(rowl)
        event_type = "job.quarantined" if op.transition_status == "quarantined" else "job.retry_scheduled"
        attrs = {
            "error_code": (op.error_code or op.error),
            "retry_count": int(op.current + 1),
        }
        if op.transition_status == "queued":
            attrs["backoff_seconds"] = int(op.delay)
        event_job = {
            "id": int(op.job_id),
            "domain": dtmp.get("domain"),
            "queue": dtmp.get("queue"),
            "job_type": dtmp.get("job_type"),
            "owner_user_id": dtmp.get("owner_user_id"),
            "request_id": dtmp.get("request_id"),
            "trace_id": dtmp.get("trace_id"),
        }
        if op.outbox_enabled:
            runtime._insert_lifecycle_event(
                conn,
                backend=manager.backend,
                event_type=event_type,
                job=event_job,
                attrs=attrs,
            )
        runtime._queue_lifecycle_event_observer(
            op.post_commit_side_effects,
            event_type=event_type,
            job=event_job,
            attrs=attrs,
        )
    op.hooks.log_retry(conn, "sqlite", op.delay)


def _retry_sqlite(manager: JobManager, op: FailureOperation, conn: Any, rowl: Any) -> tuple[bool, bool]:
    """Return retry-transition and token-replay outcomes separately."""
    # compute jittered backoff based on current retry_count
    row = conn.execute("SELECT retry_count FROM jobs WHERE id = ?", (op.job_id,)).fetchone()
    op.current = int(row[0]) if row else 0
    op.delay, op.thresh = op.hooks.retry_parameters(op.current, manager.backend)
    retry_transition_changed = _update_retry_sqlite(op, conn)
    if retry_transition_changed:
        _stage_retry_sqlite(manager, op, conn, rowl)
        return True, False
    elif op.completion_token:
        replay = conn.execute(
            "SELECT status, completion_token FROM jobs WHERE id=?",
            (int(op.job_id),),
        ).fetchone()
        if replay and str(replay[0]) in {"failed", "quarantined"} and str(replay[1] or "") == str(op.completion_token):
            return False, True
    return False, False


def _update_terminal_sqlite(op: FailureOperation, conn: Any) -> tuple[bool, bool]:
    """Apply the processing failure or permitted queued fallback."""
    # terminal failure
    failed_from_queued = False
    if op.enforce:
        processing_cursor = conn.execute(
            (
                "UPDATE jobs SET status = 'failed', last_error = ?, error_message = ?, error_code = ?, error_class = ?, error_stack = ?, completion_token = ?, "
                "completed_at = DATETIME('now'), leased_until = NULL, worker_id = NULL, lease_id = NULL WHERE id = ? AND status = 'processing' AND worker_id = ? AND lease_id = ? AND (completion_token IS NULL OR completion_token = ?)"
            ),
            (
                (op.error_code or op.error),
                op.error,
                op.error_code,
                op.error_class,
                (json.dumps(op.error_stack) if op.error_stack is not None else None),
                op.completion_token,
                op.job_id,
                op.worker_id,
                op.lease_id,
                op.completion_token,
            ),
        )
        failed_from_processing = (processing_cursor.rowcount or 0) > 0
    else:
        # Enforcement disabled: allow failing processing without matching worker/lease,
        # and fall back to failing queued jobs (admin-style terminalization) when appropriate.
        processing_cursor = conn.execute(
            (
                "UPDATE jobs SET status = 'failed', last_error = ?, error_message = ?, error_code = ?, error_class = ?, error_stack = ?, completion_token = COALESCE(completion_token, ?), "
                "completed_at = DATETIME('now'), leased_until = NULL, worker_id = NULL, lease_id = NULL WHERE id = ? AND status = 'processing' AND (completion_token IS NULL OR completion_token = ?)"
            ),
            (
                (op.error_code or op.error),
                op.error,
                op.error_code,
                op.error_class,
                (json.dumps(op.error_stack) if op.error_stack is not None else None),
                op.completion_token,
                op.job_id,
                op.completion_token,
            ),
        )
        failed_from_processing = (processing_cursor.rowcount or 0) > 0
        if not failed_from_processing:
            # Admin-style finalize: optionally allow failing queued jobs when enforcement is disabled
            # Scope via allowlist of domains (default: chatbooks) to avoid global behavior in tests
            try:
                allow = {
                    d.strip().lower()
                    for d in os.getenv(
                        "JOBS_ADMIN_COMPLETE_QUEUED_ALLOW_DOMAINS",
                        "chatbooks,embeddings",
                    ).split(",")
                    if d.strip()
                }
                row_dom = conn.execute("SELECT domain FROM jobs WHERE id = ?", (op.job_id,)).fetchone()
                dom_val = str(row_dom[0]).lower() if row_dom and row_dom[0] else ""
            except runtime._JOB_NONCRITICAL_EXCEPTIONS:
                allow = {"chatbooks", "embeddings"}
                dom_val = ""
            if dom_val in allow:
                cur2 = conn.execute(
                    (
                        "UPDATE jobs SET status = 'failed', last_error = ?, error_message = ?, error_code = ?, error_class = ?, error_stack = ?, completion_token = COALESCE(completion_token, ?), "
                        "completed_at = DATETIME('now'), leased_until = NULL, worker_id = NULL, lease_id = NULL WHERE id = ? AND status = 'queued' AND (completion_token IS NULL OR completion_token = ?)"
                    ),
                    (
                        (op.error_code or op.error),
                        op.error,
                        op.error_code,
                        op.error_class,
                        (json.dumps(op.error_stack) if op.error_stack is not None else None),
                        op.completion_token,
                        op.job_id,
                        op.completion_token,
                    ),
                )
                failed_from_queued = (cur2.rowcount or 0) > 0
    ok = failed_from_processing or failed_from_queued
    return ok, failed_from_queued


def _stage_terminal_sqlite(
    manager: JobManager, op: FailureOperation, conn: Any, rowl: Any, ok: bool, failed_from_queued: bool
) -> None:
    """Persist terminal accounting and stage SQLite observers without committing."""
    try:
        if ok and rowl:
            d = dict(rowl)
            op.post_commit_side_effects.append((runtime.increment_failures, (d,), {"reason": "terminal"}))
            try:
                if op.error_code:
                    from .metrics import increment_failures_by_code

                    op.post_commit_side_effects.append(
                        (
                            increment_failures_by_code,
                            (d, op.error_code),
                            {},
                        )
                    )
            except runtime._JOB_NONCRITICAL_EXCEPTIONS:
                pass
            # Append terminal failure to timeline (no backoff)
            try:
                row_t2 = conn.execute("SELECT failure_timeline FROM jobs WHERE id = ?", (op.job_id,)).fetchone()
                timeline_json2 = row_t2[0] if row_t2 else None
                try:
                    tl2 = json.loads(timeline_json2) if timeline_json2 else []
                except runtime._JOB_NONCRITICAL_EXCEPTIONS:
                    tl2 = []
                tl2.append(
                    {
                        "ts": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                        "error_code": (op.error_code or op.error),
                        "retry_backoff": 0,
                    }
                )
                tl2 = tl2[-10:]
                conn.execute(
                    "UPDATE jobs SET failure_timeline = ? WHERE id = ?",
                    (json.dumps(tl2), int(op.job_id)),
                )
            except runtime._JOB_NONCRITICAL_EXCEPTIONS:
                pass
            try:
                op.post_commit_side_effects.append(
                    (
                        runtime._record_job_span,
                        ("job.fail", d),
                        {
                            "attrs": {
                                "retryable": False,
                                "error_code": op.error_code,
                            }
                        },
                    )
                )
            except runtime._JOB_NONCRITICAL_EXCEPTIONS:
                pass
            op.post_commit_side_effects.append(
                (
                    manager._update_gauges,
                    (),
                    {
                        "domain": d.get("domain"),
                        "queue": d.get("queue"),
                        "job_type": d.get("job_type"),
                    },
                )
            )
            try:
                if d.get("uuid"):
                    op.post_commit_cancel_uuid = d.get("uuid")
                    op.post_commit_cancel_reason = "dependency_failed"
            except runtime._JOB_NONCRITICAL_EXCEPTIONS:
                pass
    except runtime._JOB_NONCRITICAL_EXCEPTIONS:
        pass
    if ok and rowl and runtime.JobManager._is_truthy(os.getenv("JOBS_COUNTERS_ENABLED", "")):
        d_counter = dict(rowl)
        if failed_from_queued:
            is_scheduled = d_counter.get("available_at") is not None
            counter_cursor = conn.execute(
                (
                    "UPDATE job_counters SET "
                    "ready_count = CASE WHEN ready_count > ? THEN ready_count - ? ELSE 0 END, "
                    "scheduled_count = CASE WHEN scheduled_count > ? THEN scheduled_count - ? ELSE 0 END, "
                    "updated_at = DATETIME('now') "
                    "WHERE domain=? AND queue=? AND job_type=?"
                ),
                (
                    int(not is_scheduled),
                    int(not is_scheduled),
                    int(is_scheduled),
                    int(is_scheduled),
                    d_counter.get("domain"),
                    d_counter.get("queue"),
                    d_counter.get("job_type"),
                ),
            )
        else:
            counter_cursor = conn.execute(
                "UPDATE job_counters SET processing_count = CASE WHEN processing_count>0 THEN processing_count-1 ELSE 0 END, updated_at = DATETIME('now') WHERE domain=? AND queue=? AND job_type=?",
                (
                    d_counter.get("domain"),
                    d_counter.get("queue"),
                    d_counter.get("job_type"),
                ),
            )
        if (counter_cursor.rowcount or 0) == 0:
            runtime._reconcile_lifecycle_counter_row(
                conn,
                backend=manager.backend,
                domain=d_counter.get("domain"),
                queue=d_counter.get("queue"),
                job_type=d_counter.get("job_type"),
            )
    if ok and rowl:
        d_event = dict(rowl)
        event_job = {
            "id": int(op.job_id),
            "domain": d_event.get("domain"),
            "queue": d_event.get("queue"),
            "job_type": d_event.get("job_type"),
            "owner_user_id": d_event.get("owner_user_id"),
            "request_id": d_event.get("request_id"),
            "trace_id": d_event.get("trace_id"),
        }
        attrs = {"error_code": (op.error_code or op.error)}
        if op.outbox_enabled:
            runtime._insert_lifecycle_event(
                conn,
                backend=manager.backend,
                event_type="job.failed",
                job=event_job,
                attrs=attrs,
            )
        runtime._queue_lifecycle_event_observer(
            op.post_commit_side_effects,
            event_type="job.failed",
            job=event_job,
            attrs=attrs,
        )


def _terminal_sqlite(manager: JobManager, op: FailureOperation, conn: Any, rowl: Any) -> tuple[bool, bool]:
    """Stage terminal failure or accept a matching terminal replay."""
    ok, failed_from_queued = _update_terminal_sqlite(op, conn)
    if not ok and op.completion_token:
        replay = conn.execute(
            "SELECT status, completion_token FROM jobs WHERE id=?",
            (int(op.job_id),),
        ).fetchone()
        if replay and str(replay[0]) == "failed" and str(replay[1] or "") == str(op.completion_token):
            return False, True
    _stage_terminal_sqlite(manager, op, conn, rowl, ok, failed_from_queued)
    op.hooks.log_terminal(conn, "sqlite", ok)
    return bool(ok), False


def _fail_sqlite(manager: JobManager, op: FailureOperation, conn: Any) -> bool:
    """Own SQLite fencing; publish staged effects after transaction exit."""
    result = False
    with manager._resource_sqlite_transaction(
        conn, where="id=?", params=(int(op.job_id),), expected_uuid=op.expected_uuid
    ) as mutation:
        if op.expected_uuid is not None and int(op.job_id) not in mutation.rows:
            return False
        op.hooks.log_enter("sqlite")
        # For metrics, fetch labels
        rowl = conn.execute(
            "SELECT status, completion_token, domain, queue, job_type, uuid, request_id, trace_id, owner_user_id, available_at FROM jobs WHERE id = ?",
            (op.job_id,),
        ).fetchone()
        if rowl is None or (op.expected_uuid is not None and str(rowl["uuid"] or "") != str(op.expected_uuid)):
            return False
        if not mutation.worker_allowed(int(op.job_id), op.worker_id, op.lease_id, op.expected_uuid):
            return False
        if rowl:
            st = str(rowl[0])
            ct = rowl[1]
            if st in {"completed", "failed", "cancelled", "quarantined"}:
                replay_states = {"failed", "quarantined"} if op.retryable else {"failed"}
                return bool(st in replay_states and op.completion_token and ct and str(ct) == str(op.completion_token))
        retry_scheduled = False
        if op.retryable:
            retry_scheduled, replayed = _retry_sqlite(manager, op, conn, rowl)
            if replayed:
                return True
            result = retry_scheduled
        if not retry_scheduled:
            result, replayed = _terminal_sqlite(manager, op, conn, rowl)
            if replayed:
                return True
    runtime._run_post_commit_side_effects(op.post_commit_side_effects)
    op.post_commit_side_effects.clear()
    if op.post_commit_cancel_uuid:
        with contextlib.suppress(runtime._JOB_NONCRITICAL_EXCEPTIONS):
            manager._cancel_dependent_jobs(
                op.post_commit_cancel_uuid,
                reason=op.post_commit_cancel_reason or "dependency_failed",
            )
    return bool(result)


def execute_failure(manager: JobManager, op: FailureOperation) -> bool:
    """Dispatch the failure transaction and close its connection on every exit."""
    conn = manager._connect()
    op.hooks.initialize_diagnostics()
    try:
        if manager.backend == "postgres":
            return _fail_postgres(manager, op, conn)
        return _fail_sqlite(manager, op, conn)
    finally:
        runtime._close_connection_nonfatal(conn, operation="job failure")
