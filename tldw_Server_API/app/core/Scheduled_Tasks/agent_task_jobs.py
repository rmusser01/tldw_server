"""Jobs consumer for scheduled automation definitions (TASK-13021).

Executes the ``agent_task_run`` Jobs the scheduler feed (TASK-13020)
enqueues, mirroring ``core/Reminders/reminder_jobs.py``: durable run rows
deduped on (definition, run slot), lifecycle re-checked at execution
time, and delivery of the outcome as a user notification through the same
channel reminders use.

Phase-1 boundary (tldw_chatbook ADR-077, decision 4, owner-accepted):
execution is **side-effect-free only**. ``recurring_question`` runs
generate their answer through the registered executor; ``agent_task``
runs execute in generation-only mode; tool-using configurations are NOT
executed — they resolve to an explicit ``skipped`` run with an actionable
reason until the approval-escalation design exists. The boundary is
enforced HERE, by the consumer, not assumed from the definition.

Timeout semantics (ADR-077 decision 5): a run cancelled at its execution
deadline records the distinct ``timed_out`` status — the client displays
what the server reports (its own vocabulary matches, tldw_chatbook
TASK-18939); no translation layer.

The LLM executor is an injected async callable so this module owns the
run/notification/audit machinery while the executor wiring lives where
the server's model plumbing is configured. Without a registered executor
the run fails honestly ("no executor configured") — visible, never
silent.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.exceptions import BadRequestError
from tldw_Server_API.app.core.DB_Management.Scheduled_Tasks_DB import (
    DefinitionRow,
    ScheduledTasksDatabase,
)

AUTOMATION_DOMAIN = "scheduled_tasks"
AUTOMATION_JOB_TYPE = "agent_task_run"
CONSUMER_ACTOR = "automation-consumer"

#: Notification kinds delivered for run outcomes (bounded summary in the
#: message body; the full result stays server-side per ADR-077 decision 3).
NOTIFICATION_KIND_BY_STATUS = {
    "succeeded": "automation_run_succeeded",
    "failed": "automation_run_failed",
    "timed_out": "automation_run_timed_out",
    "skipped": "automation_run_skipped",
}

#: Execution deadline (seconds) for one phase-1 run. Distinct from the
#: feed's misfire concerns: this bounds the EXECUTION, recording
#: ``timed_out`` on expiry.
RUN_EXECUTION_TIMEOUT_SECONDS = 300.0

#: Bounded result summary cap for notifications (ADR-077 decision 3).
RESULT_SUMMARY_MAX_CHARS = 1000

#: Definition families executable in phase 1 (side-effect-free only).
_PHASE1_EXECUTABLE_FAMILIES = {"recurring_question", "agent_task"}

Executor = Callable[[DefinitionRow, dict[str, Any]], Awaitable[str]]

#: Process-wide executor registry. The wiring layer (server model
#: plumbing) registers per-family executors at startup; tests inject
#: their own. Keyed by family; ``None`` value = generation-only default.
_EXECUTORS: dict[str, Executor] = {}


def register_executor(family: str, executor: Executor) -> None:
    """Register the phase-1 executor callable for one definition family."""
    _EXECUTORS[family] = executor


def _phase1_tools_requested(definition: DefinitionRow) -> bool:
    """Return True when the definition's config asks for tools (phase-2).

    Phase 1 executes generation-only work; a config that enables tools of
    any kind marks the definition out of bounds until the
    approval-escalation design exists.
    """
    config = definition.input if isinstance(definition.input, dict) else {}
    tools = config.get("tools") or config.get("allowed_tools") or config.get("enable_tools")
    if isinstance(tools, (list, tuple, set)) and tools:
        return True
    if isinstance(tools, str) and tools.strip().lower() not in ("", "false", "none", "0"):
        return True
    if isinstance(tools, bool) and tools:
        return True
    return False


def _normalize_slot_utc(value: Any) -> str:
    """Normalize a slot timestamp to second-precision UTC ISO-8601."""
    if not value:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        parsed = datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _bound_summary(text: str) -> str:
    """Bound a result summary for notification delivery."""
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= RESULT_SUMMARY_MAX_CHARS:
        return cleaned
    return cleaned[: RESULT_SUMMARY_MAX_CHARS - 20] + " … [truncated]"


def _notification_enabled(definition: DefinitionRow, status: str) -> bool:
    """Apply the definition's notification_policy to an outcome.

    Policies default to delivering everything; an explicit ``kinds``
    allowlist filters by outcome, and ``enabled: false`` silences all.
    """
    policy = (
        definition.notification_policy
        if isinstance(definition.notification_policy, dict)
        else {}
    )
    if policy.get("enabled") is False:
        return False
    kinds = policy.get("kinds")
    if isinstance(kinds, (list, tuple)) and kinds:
        return status in kinds
    return True


async def handle_agent_task_job(
    job: dict[str, Any],
    *,
    scheduled_db: ScheduledTasksDatabase | None = None,
    collections_db: CollectionsDatabase | None = None,
    execution_timeout_seconds: float = RUN_EXECUTION_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Consume one ``agent_task_run`` Job: run row, execute, notify, audit.

    Returns a result dict shaped like the reminders consumer's (``status``,
    ``definition_id``, ``run_id``, ``deduped`` for no-op redeliveries).
    ``run_id`` is ``None`` only when the owner-scoped definition is unavailable;
    that pre-run skip also includes ``reason="definition_missing"``.
    """
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    definition_id = str(payload.get("definition_id") or "").strip()
    if not definition_id:
        raise BadRequestError("missing definition_id")
    owner = job.get("owner_user_id") or payload.get("user_id")
    if owner is None or str(owner).strip() == "":
        raise BadRequestError("missing owner_user_id")
    user_id = int(owner)

    run_slot_utc = _normalize_slot_utc(payload.get("scheduled_for"))
    run_slot_key = run_slot_utc
    now_iso = datetime.now(timezone.utc).isoformat()

    sdb = scheduled_db or ScheduledTasksDatabase.for_user(user_id=user_id)

    try:
        definition = sdb.get_definition(owner_id=user_id, definition_id=definition_id)
    except KeyError:
        definition = None
    if definition is None:
        logger.warning(
            "Automation Job skipped because its definition is unavailable",
            definition_id=definition_id,
            user_id=user_id,
            job_id=job.get("id"),
        )
        return {
            "status": "skipped",
            "definition_id": definition_id,
            "run_id": None,
            "reason": "definition_missing",
        }

    cdb = collections_db or CollectionsDatabase.for_user(user_id=user_id)

    # Durable run row, deduped on (definition, slot): a redelivered Job for
    # an already-terminal (or in-flight) slot is a recorded no-op.
    run = sdb.create_scheduled_task_run(
        definition_id=definition_id,
        owner_id=user_id,
        scheduled_for=payload.get("scheduled_for"),
        job_id=str(job.get("id")) if job.get("id") is not None else None,
        run_slot_utc=run_slot_utc,
        run_slot_key=run_slot_key,
        status="running",
        started_at=now_iso,
    )
    if run["status"] in ("succeeded", "skipped", "failed", "timed_out"):
        return {
            "status": run["status"],
            "definition_id": definition_id,
            "run_id": run["id"],
            "deduped": True,
        }

    # Lifecycle re-check at execution time: arming gates on 'configured',
    # but the definition may have been paused/archived/disabled since.
    if definition.lifecycle != "configured":
        _finish(
            sdb,
            cdb,
            definition=definition,
            run_id=run["id"],
            status="skipped",
            error=f"definition_{definition.lifecycle}",
            summary=None,
            jobs_job_id=str(job.get("id")) if job.get("id") is not None else None,
            execution_timeout_seconds=execution_timeout_seconds,
        )
        return {"status": "skipped", "definition_id": definition_id, "run_id": run["id"]}

    # Phase-1 boundary, enforced by the consumer (not assumed): tools are
    # out of bounds until the approval-escalation design exists.
    if _phase1_tools_requested(definition):
        _finish(
            sdb,
            cdb,
            definition=definition,
            run_id=run["id"],
            status="skipped",
            error="tools_not_executable_in_phase1",
            summary=(
                "This definition requests tools; server-side tool use is not "
                "executable until the approval-escalation design lands."
            ),
            jobs_job_id=str(job.get("id")) if job.get("id") is not None else None,
            execution_timeout_seconds=execution_timeout_seconds,
        )
        return {"status": "skipped", "definition_id": definition_id, "run_id": run["id"]}

    if definition.family not in _PHASE1_EXECUTABLE_FAMILIES:
        _finish(
            sdb,
            cdb,
            definition=definition,
            run_id=run["id"],
            status="skipped",
            error=f"family_not_executable:{definition.family}",
            summary=None,
            jobs_job_id=str(job.get("id")) if job.get("id") is not None else None,
            execution_timeout_seconds=execution_timeout_seconds,
        )
        return {"status": "skipped", "definition_id": definition_id, "run_id": run["id"]}

    executor = _EXECUTORS.get(definition.family)
    if executor is None:
        # Not a failure: no executor is wired for this family in this
        # deployment phase (phase 1 wires recurring_question only --
        # agent_task messages are redacted at rest). The skip carries an
        # actionable reason.
        _finish(
            sdb,
            cdb,
            definition=definition,
            run_id=run["id"],
            status="skipped",
            error=f"family_not_wired_for_execution:{definition.family}",
            summary=(
                "No executor is wired for this family in this deployment "
                "phase; the run was recorded and skipped without executing."
            ),
            jobs_job_id=str(job.get("id")) if job.get("id") is not None else None,
            execution_timeout_seconds=execution_timeout_seconds,
        )
        return {"status": "skipped", "definition_id": definition_id, "run_id": run["id"]}

    timed_out = False
    result_text: str | None = None
    error_text: str | None = None
    jobs_job_id = str(job.get("id")) if job.get("id") is not None else None
    try:
        result_text = await asyncio.wait_for(
            executor(definition, payload), timeout=execution_timeout_seconds
        )
    except asyncio.TimeoutError:
        timed_out = True
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - one bad run must not kill the worker
        error_text = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "Automation run executor failed for definition {}: {}",
            definition_id,
            error_text,
            definition_id=definition_id,
            user_id=user_id,
            run_id=run["id"],
            job_id=job.get("id"),
        )

    status = "timed_out" if timed_out else ("failed" if error_text else "succeeded")
    _finish(
        sdb,
        cdb,
        definition=definition,
        run_id=run["id"],
        status=status,
        error=error_text,
        summary=_bound_summary(result_text) if result_text is not None else None,
        jobs_job_id=jobs_job_id,
        execution_timeout_seconds=execution_timeout_seconds,
    )
    return {"status": status, "definition_id": definition_id, "run_id": run["id"]}


def _finish(
    sdb: ScheduledTasksDatabase,
    cdb: CollectionsDatabase,
    *,
    definition: DefinitionRow | None,
    run_id: int,
    status: str,
    error: str | None,
    summary: str | None,
    jobs_job_id: str | None = None,
    execution_timeout_seconds: float = RUN_EXECUTION_TIMEOUT_SECONDS,
) -> None:
    """Record the terminal run state, deliver the notification, update health.

    Notification and audit failures never mask the run outcome: the run
    row is written first, and downstream failures log and continue.
    """
    completed_at = datetime.now(timezone.utc).isoformat()
    try:
        sdb.update_scheduled_task_run_status(
            run_id=run_id,
            status=status,
            error=error,
            result_summary=summary,
            completed_at=completed_at,
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "Automation run status update failed",
            run_id=run_id,
            status=status,
        )

    if definition is None:
        return

    # Definition health reflects the latest run's reality (only-on-change:
    # no version churn for repeated identical outcomes).
    health = {
        "succeeded": "ready",
        "timed_out": "degraded",
        "failed": "degraded",
        "skipped": definition.health,
    }.get(status, definition.health)
    if health != definition.health:
        try:
            sdb.update_definition(
                owner_id=definition.owner_id,
                definition_id=definition.id,
                patch={"health": health, "updated_by": CONSUMER_ACTOR},
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Automation health update failed",
                definition_id=definition.id,
                owner_id=definition.owner_id,
                health=health,
            )

    if _notification_enabled(definition, status):
        try:
            kind = NOTIFICATION_KIND_BY_STATUS.get(status, "automation_run_failed")
            if status == "succeeded" and summary:
                message = summary
            elif status == "timed_out":
                message = (
                    f"Run timed out after the execution deadline "
                    f"({int(execution_timeout_seconds)}s)."
                )
            elif status == "skipped":
                message = summary or f"Skipped: {error or 'unknown reason'}."
            else:
                message = f"Run failed: {error or 'unknown error'}."
            cdb.create_user_notification(
                kind=kind,
                title=definition.name,
                message=message,
                severity="info" if status == "succeeded" else "warning",
                source_domain=AUTOMATION_DOMAIN,
                source_job_type=AUTOMATION_JOB_TYPE,
                source_job_id=jobs_job_id,
                dedupe_key=f"automation_run:{definition.id}:{run_id}",
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Automation run notification failed",
                definition_id=definition.id,
                run_id=run_id,
                status=status,
            )

    try:
        sdb.create_audit_event(
            owner_id=definition.owner_id,
            definition_id=definition.id,
            event_type=f"run_{status}",
            actor=CONSUMER_ACTOR,
            summary=f"Run {status}" + (f": {error}" if error else "."),
            before=None,
            after={"run_id": run_id, "status": status},
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "Automation run audit failed",
            definition_id=definition.id,
            run_id=run_id,
            status=status,
        )
