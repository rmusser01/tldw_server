"""Recurring Question execution control-plane service."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel

from tldw_Server_API.app.api.v1.schemas.scheduled_tasks_automation_schemas import (
    ScheduledTaskResultListResponse,
    ScheduledTaskResultResponse,
    ScheduledTaskRunListResponse,
    ScheduledTaskRunResponse,
)
from tldw_Server_API.app.core.DB_Management.Scheduled_Tasks_DB import (
    DefinitionRow,
    ResultRow,
    RunRow,
    ScheduledTasksDatabase,
    ScheduledTasksTransaction,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Scheduled_Tasks.recurring_question_jobs import (
    RECURRING_QUESTION_JOB_TYPE,
    RECURRING_QUESTION_QUEUE,
    SCHEDULED_TASKS_DOMAIN,
    build_manual_run_idempotency_payload,
    build_recurring_question_run_job_payload,
    build_scheduled_run_idempotency_key,
)
from tldw_Server_API.app.core.Scheduled_Tasks.recurring_question_scope import normalize_recurring_question_scope
from tldw_Server_API.app.services.scheduled_task_automation_service import (
    ScheduledTaskAutomationError,
    ScheduledTaskAutomationService,
    _canonical_hash,
)


class ScheduledTaskRecurringQuestionService(ScheduledTaskAutomationService):
    """Service for Recurring Question run/result APIs."""

    def __init__(
        self,
        *,
        repository: ScheduledTasksDatabase | None = None,
        job_manager: JobManager | None = None,
    ) -> None:
        super().__init__(repository=repository)
        self._job_manager = job_manager or JobManager()

    def create_manual_run(
        self,
        *,
        owner_id: int,
        actor: str,
        definition_id: str,
        idempotency_key: str | None = None,
        request_id: str | None = None,
    ) -> ScheduledTaskRunResponse:
        payload_hash = _canonical_hash(build_manual_run_idempotency_payload(definition_id=definition_id))
        response = self._with_idempotency(
            owner_id=owner_id,
            route="scheduled_task_recurring_question.manual_run",
            key=idempotency_key,
            payload_hash=payload_hash,
            operation=lambda tx: self._create_manual_run(
                tx=tx,
                owner_id=owner_id,
                actor=actor,
                definition_id=definition_id,
                idempotency_key=idempotency_key,
                request_id=request_id,
            ),
        )
        return ScheduledTaskRunResponse.model_validate(response)

    def create_scheduled_run(
        self,
        *,
        owner_id: int,
        actor: str,
        definition_id: str,
        definition_version: int,
        schedule_slot: str | datetime,
        request_id: str | None = None,
    ) -> ScheduledTaskRunResponse:
        """Create or replay one scheduled Recurring Question run for a due slot."""

        normalized_slot = _normalize_schedule_slot(schedule_slot)
        idempotency_key = build_scheduled_run_idempotency_key(
            definition_id=definition_id,
            definition_version=definition_version,
            schedule_slot=normalized_slot,
        )
        payload_hash = _canonical_hash(
            {
                "action": "create_scheduled_run",
                "definition_id": definition_id,
                "definition_version": definition_version,
                "schedule_slot": normalized_slot,
                "trigger_reason": "scheduled",
            }
        )
        response = self._with_idempotency(
            owner_id=owner_id,
            route="scheduled_task_recurring_question.scheduled_run",
            key=idempotency_key,
            payload_hash=payload_hash,
            operation=lambda tx: self._create_scheduled_run(
                tx=tx,
                owner_id=owner_id,
                actor=actor,
                definition_id=definition_id,
                definition_version=definition_version,
                schedule_slot=normalized_slot,
                job_idempotency_key=idempotency_key,
                request_id=request_id,
            ),
        )
        return ScheduledTaskRunResponse.model_validate(response)

    def reconcile_stale_runs(
        self,
        *,
        owner_id: int,
        actor: str,
        now: datetime | None = None,
        stale_after: timedelta = timedelta(hours=2),
        limit_per_status: int = 200,
    ) -> list[str]:
        """Repair stale queued/running runs that no worker can truthfully finish."""

        now_utc = _ensure_aware_utc(now or datetime.now(timezone.utc))
        repaired: list[str] = []
        repo = self._repo(owner_id)
        for status in ("queued", "running"):
            rows, _total = repo.list_runs(
                owner_id=owner_id,
                status=status,
                limit=limit_per_status,
                offset=0,
            )
            for row in rows:
                updated_at = _parse_optional_datetime(row.updated_at)
                if updated_at is None or now_utc - updated_at < stale_after:
                    continue
                failure_reason = self._repair_failure_reason(row)
                repaired_row = repo.update_run(
                    owner_id=owner_id,
                    run_id=row.id,
                    patch={
                        "status": "failed",
                        "outcome": "degraded",
                        "run_summary": {
                            **row.run_summary,
                            "message": failure_reason["message"],
                            "repair_reason": failure_reason["code"],
                            "needs_attention": failure_reason.get("needs_attention", True),
                            "previous_status": row.status,
                        },
                        "failure_reason": failure_reason,
                        "ended_at": now_utc.isoformat(),
                    },
                )
                self._create_audit(
                    owner_id=owner_id,
                    definition_id=row.definition_id,
                    event_type="run.repaired",
                    actor=actor,
                    summary=f"Repaired stale {row.status} Recurring Question run",
                    before=self._run_response(row).model_dump(mode="json"),
                    after=self._run_response(repaired_row).model_dump(mode="json"),
                    idempotency_key=None,
                    request_id=None,
                )
                repaired.append(row.id)
        return repaired

    def prune_definition_history(
        self,
        *,
        owner_id: int,
        definition_id: str,
        now: datetime | None = None,
    ) -> dict[str, int]:
        """Apply a definition's stored run/result retention policy."""

        definition = self._get_definition_row(owner_id=owner_id, definition_id=definition_id)
        now_utc = _ensure_aware_utc(now or datetime.now(timezone.utc))
        no_match_days = _retention_ttl_days(
            definition.retention_policy,
            preferred_key="no_match_run_ttl_days",
            fallback_key="run_ttl_days",
            default_days=30,
        )
        result_days = _retention_ttl_days(
            definition.retention_policy,
            preferred_key="result_ttl_days",
            fallback_key=None,
            default_days=max(no_match_days, 180),
        )
        return self._repo(owner_id).prune_run_history(
            owner_id=owner_id,
            definition_id=definition_id,
            no_match_before=(now_utc - timedelta(days=no_match_days)).isoformat(),
            result_before=(now_utc - timedelta(days=result_days)).isoformat(),
        )

    def list_runs(
        self,
        *,
        owner_id: int,
        definition_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> ScheduledTaskRunListResponse:
        if definition_id is not None:
            self._get_definition_row(owner_id=owner_id, definition_id=definition_id)
        rows, total = self._repo(owner_id).list_runs(
            owner_id=owner_id,
            definition_id=definition_id,
            status=status,
            limit=limit,
            offset=offset,
        )
        return ScheduledTaskRunListResponse(
            items=[self._run_response(row) for row in rows],
            total=total,
            limit=limit,
            offset=offset,
            has_more=offset + len(rows) < total,
            next_offset=offset + len(rows) if offset + len(rows) < total else None,
        )

    def get_run(self, *, owner_id: int, run_id: str) -> ScheduledTaskRunResponse:
        row = self._repo(owner_id).get_run(owner_id=owner_id, run_id=run_id)
        if row is None:
            raise ScheduledTaskAutomationError("run_not_found")
        return self._run_response(row)

    def list_results(
        self,
        *,
        owner_id: int,
        definition_id: str | None = None,
        run_id: str | None = None,
        review_state: str | None = None,
        kind: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> ScheduledTaskResultListResponse:
        if definition_id is not None:
            self._get_definition_row(owner_id=owner_id, definition_id=definition_id)
        rows, total = self._repo(owner_id).list_results(
            owner_id=owner_id,
            definition_id=definition_id,
            run_id=run_id,
            review_state=review_state,
            kind=kind,
            limit=limit,
            offset=offset,
        )
        return ScheduledTaskResultListResponse(
            items=[self._result_response(row) for row in rows],
            total=total,
            limit=limit,
            offset=offset,
            has_more=offset + len(rows) < total,
            next_offset=offset + len(rows) if offset + len(rows) < total else None,
        )

    def get_result(self, *, owner_id: int, result_id: str) -> ScheduledTaskResultResponse:
        row = self._repo(owner_id).get_result(owner_id=owner_id, result_id=result_id)
        if row is None:
            raise ScheduledTaskAutomationError("result_not_found")
        return self._result_response(row)

    def update_result_review_state(
        self,
        *,
        owner_id: int,
        actor: str,
        result_id: str,
        review_state: str,
        review_note: str | None = None,
        idempotency_key: str | None = None,
        request_id: str | None = None,
    ) -> ScheduledTaskResultResponse:
        repo = self._repo(owner_id)

        def _run(tx: ScheduledTasksTransaction) -> ScheduledTaskResultResponse:
            before_row = tx.get_result(owner_id=owner_id, result_id=result_id)
            if before_row is None:
                raise ScheduledTaskAutomationError("result_not_found")
            try:
                after_row = tx.update_result_review_state(
                    owner_id=owner_id,
                    result_id=result_id,
                    review_state=review_state,
                    reviewed_by=actor,
                    review_note=review_note,
                )
            except KeyError as exc:
                raise ScheduledTaskAutomationError("result_not_found") from exc
            before = self._result_response(before_row)
            after = self._result_response(after_row)
            self._create_audit(
                tx=tx,
                owner_id=owner_id,
                definition_id=after.definition_id,
                event_type="result.reviewed",
                actor=actor,
                summary="Updated result review state",
                before=before.model_dump(mode="json"),
                after=after.model_dump(mode="json"),
                idempotency_key=idempotency_key,
                request_id=request_id,
            )
            return after

        return repo.write_transaction(_run)

    def _create_manual_run(
        self,
        *,
        tx: ScheduledTasksTransaction,
        owner_id: int,
        actor: str,
        definition_id: str,
        idempotency_key: str | None,
        request_id: str | None,
    ) -> ScheduledTaskRunResponse:
        definition = self._get_definition_row(tx=tx, owner_id=owner_id, definition_id=definition_id)
        config = self._recurring_question_config(tx=tx, definition=definition)
        scope_snapshot = dict(config.get("scope") or {})
        self._validate_manual_run_admission(tx=tx, definition=definition, scope_snapshot=scope_snapshot)
        run = tx.create_run(
            owner_id=owner_id,
            definition_id=definition.id,
            definition_version=definition.version,
            trigger_reason="manual",
            status="queued",
            outcome="none",
            scope_snapshot=scope_snapshot,
            finding_policy_snapshot=definition.finding_policy,
            rag_request_snapshot=self._rag_request_snapshot(definition=definition, config=config, scope=scope_snapshot),
            run_summary={"message": "Queued Recurring Question run.", "trigger_reason": "manual"},
        )
        job_payload = build_recurring_question_run_job_payload(run=run, owner_user_id=str(owner_id))
        job = self._create_jobs_entry(
            run=run,
            owner_user_id=str(owner_id),
            payload=job_payload,
            request_id=request_id,
            idempotency_key=idempotency_key,
        )
        job_id = str(job["id"]) if job.get("id") is not None else None
        if job_id is not None:
            run = tx.update_run(
                owner_id=owner_id,
                run_id=run.id,
                patch={
                    "job_id": job_id,
                    "run_summary": {
                        "message": "Queued Recurring Question run.",
                        "trigger_reason": "manual",
                        "job_id": job_id,
                    },
                },
            )
        response = self._run_response(run)
        self._create_audit(
            tx=tx,
            owner_id=owner_id,
            definition_id=definition.id,
            event_type="run.created",
            actor=actor,
            summary="Created manual Recurring Question run",
            before=None,
            after=response.model_dump(mode="json"),
            idempotency_key=idempotency_key,
            request_id=request_id,
        )
        return response

    def _create_scheduled_run(
        self,
        *,
        tx: ScheduledTasksTransaction,
        owner_id: int,
        actor: str,
        definition_id: str,
        definition_version: int,
        schedule_slot: str,
        job_idempotency_key: str,
        request_id: str | None,
    ) -> ScheduledTaskRunResponse:
        definition = self._get_definition_row(tx=tx, owner_id=owner_id, definition_id=definition_id)
        if definition.version != definition_version:
            raise ScheduledTaskAutomationError("definition_version_mismatch")
        config = self._recurring_question_config(tx=tx, definition=definition)
        scope_snapshot = dict(config.get("scope") or {})
        self._validate_scheduled_run_admission(tx=tx, definition=definition, scope_snapshot=scope_snapshot)
        run = tx.create_run(
            owner_id=owner_id,
            definition_id=definition.id,
            definition_version=definition.version,
            trigger_reason="scheduled",
            status="queued",
            outcome="none",
            scope_snapshot=scope_snapshot,
            finding_policy_snapshot=definition.finding_policy,
            rag_request_snapshot=self._rag_request_snapshot(definition=definition, config=config, scope=scope_snapshot),
            run_summary={
                "message": "Queued scheduled Recurring Question run.",
                "trigger_reason": "scheduled",
                "schedule_slot": schedule_slot,
            },
            schedule_slot=schedule_slot,
        )
        job_payload = build_recurring_question_run_job_payload(run=run, owner_user_id=str(owner_id))
        job = self._create_jobs_entry(
            run=run,
            owner_user_id=str(owner_id),
            payload=job_payload,
            request_id=request_id,
            idempotency_key=job_idempotency_key,
        )
        job_id = str(job["id"]) if job.get("id") is not None else None
        if job_id is not None:
            run = tx.update_run(
                owner_id=owner_id,
                run_id=run.id,
                patch={
                    "job_id": job_id,
                    "run_summary": {
                        "message": "Queued scheduled Recurring Question run.",
                        "trigger_reason": "scheduled",
                        "schedule_slot": schedule_slot,
                        "job_id": job_id,
                    },
                },
            )
        response = self._run_response(run)
        self._create_audit(
            tx=tx,
            owner_id=owner_id,
            definition_id=definition.id,
            event_type="run.created",
            actor=actor,
            summary="Created scheduled Recurring Question run",
            before=None,
            after=response.model_dump(mode="json"),
            idempotency_key=job_idempotency_key,
            request_id=request_id,
        )
        return response

    def _create_jobs_entry(
        self,
        *,
        run: RunRow,
        owner_user_id: str,
        payload: dict[str, Any],
        request_id: str | None,
        priority: int = 5,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        return self._job_manager.create_job(
            domain=SCHEDULED_TASKS_DOMAIN,
            queue=RECURRING_QUESTION_QUEUE,
            job_type=RECURRING_QUESTION_JOB_TYPE,
            payload=payload,
            owner_user_id=owner_user_id,
            priority=priority,
            idempotency_key=idempotency_key or f"scheduled-task-rq-run:{run.id}",
            request_id=request_id,
        )

    def _validate_manual_run_admission(
        self,
        *,
        tx: ScheduledTasksTransaction,
        definition: DefinitionRow,
        scope_snapshot: dict[str, Any],
    ) -> None:
        if definition.family != "recurring_question":
            raise ScheduledTaskAutomationError("definition_family_mismatch")
        if definition.lifecycle == "archived":
            raise ScheduledTaskAutomationError("definition_archived")
        if definition.lifecycle == "disabled":
            raise ScheduledTaskAutomationError("definition_disabled")
        if definition.lifecycle not in {"configured", "paused"}:
            raise ScheduledTaskAutomationError("lifecycle_transition_invalid")
        if definition.resolution_state == "solved":
            raise ScheduledTaskAutomationError("definition_solved")
        if not _scope_has_sources(scope_snapshot):
            raise ScheduledTaskAutomationError("scope_empty")
        for status in ("queued", "running"):
            active, total = tx.list_runs(
                owner_id=definition.owner_id,
                definition_id=definition.id,
                status=status,
                limit=1,
                offset=0,
            )
            if total > 0 or active:
                raise ScheduledTaskAutomationError("run_in_progress")

    def _validate_scheduled_run_admission(
        self,
        *,
        tx: ScheduledTasksTransaction,
        definition: DefinitionRow,
        scope_snapshot: dict[str, Any],
    ) -> None:
        if definition.family != "recurring_question":
            raise ScheduledTaskAutomationError("definition_family_mismatch")
        if definition.lifecycle == "archived":
            raise ScheduledTaskAutomationError("definition_archived")
        if definition.lifecycle == "disabled":
            raise ScheduledTaskAutomationError("definition_disabled")
        if definition.lifecycle != "configured":
            raise ScheduledTaskAutomationError("definition_not_scheduled")
        if definition.resolution_state == "solved":
            raise ScheduledTaskAutomationError("definition_solved")
        if not _scope_has_sources(scope_snapshot):
            raise ScheduledTaskAutomationError("scope_empty")
        for status in ("queued", "running"):
            active, total = tx.list_runs(
                owner_id=definition.owner_id,
                definition_id=definition.id,
                status=status,
                limit=1,
                offset=0,
            )
            if total > 0 or active:
                raise ScheduledTaskAutomationError("run_in_progress")

    def _repair_failure_reason(self, row: RunRow) -> dict[str, Any]:
        job_status: str | None = None
        if row.job_id:
            try:
                job = self._job_manager.get_job(int(row.job_id))
            except (TypeError, ValueError):
                job = None
            if isinstance(job, dict):
                job_status = str(job.get("status") or "")
        if job_status == "completed":
            return {
                "code": "job_completed_without_run_finalization",
                "message": "The backing job completed but the run did not finalize its result state.",
                "job_id": row.job_id,
                "job_status": job_status,
                "needs_attention": True,
            }
        if job_status in {"failed", "cancelled"}:
            return {
                "code": f"job_{job_status}_without_run_finalization",
                "message": f"The backing job is {job_status} but the run did not finalize its result state.",
                "job_id": row.job_id,
                "job_status": job_status,
                "needs_attention": True,
            }
        return {
            "code": "scheduler_repair_stale_run",
            "message": "The run exceeded the stale-run repair window without worker progress.",
            "job_id": row.job_id,
            "job_status": job_status,
            "needs_attention": True,
        }

    def _recurring_question_config(
        self,
        *,
        tx: ScheduledTasksTransaction,
        definition: DefinitionRow,
    ) -> dict[str, Any]:
        preview = tx.get_preview(owner_id=definition.owner_id, preview_id=definition.preview_id)
        if preview is None:
            return {"scope": {"mode": "all_searchable_library", "resolved_sources": ["media_db", "notes", "chats"]}}
        config = preview.normalized_config.get("config", {})
        normalized = dict(config) if isinstance(config, dict) else {}
        if "scope" not in normalized:
            scope, _errors, _warnings = normalize_recurring_question_scope(None)
            normalized["scope"] = scope
        return normalized

    @staticmethod
    def _rag_request_snapshot(
        *,
        definition: DefinitionRow,
        config: dict[str, Any],
        scope: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "query": str(definition.input.get("question") or ""),
            "scope": scope,
            "generation_mode": config.get("generation_mode", "optional"),
            "finding_policy": definition.finding_policy,
        }

    @staticmethod
    def _run_response(row: RunRow) -> ScheduledTaskRunResponse:
        return ScheduledTaskRunResponse(
            id=row.id,
            owner_id=str(row.owner_id),
            definition_id=row.definition_id,
            definition_version=row.definition_version,
            trigger_reason=row.trigger_reason,
            status=row.status,
            outcome=row.outcome,
            job_id=row.job_id,
            schedule_slot=row.schedule_slot,
            scope_snapshot=row.scope_snapshot,
            finding_policy_snapshot=row.finding_policy_snapshot,
            rag_request_snapshot=row.rag_request_snapshot,
            run_summary=row.run_summary,
            evidence_summary=row.evidence_summary,
            failure_reason=row.failure_reason,
            created_at=row.created_at,
            updated_at=row.updated_at,
            started_at=row.started_at,
            ended_at=row.ended_at,
        )

    @staticmethod
    def _result_response(row: ResultRow) -> ScheduledTaskResultResponse:
        return ScheduledTaskResultResponse(
            id=row.id,
            owner_id=str(row.owner_id),
            definition_id=row.definition_id,
            run_id=row.run_id,
            kind=row.kind,
            title=row.title,
            summary=row.summary,
            answer=row.answer,
            answer_mode=row.answer_mode,
            confidence=row.confidence,
            source_refs=row.source_refs,
            dedupe_key=row.dedupe_key,
            visibility_destination=row.visibility_destination,
            review_state=row.review_state,
            created_at=row.created_at,
            updated_at=row.updated_at,
            reviewed_at=row.reviewed_at,
            reviewed_by=row.reviewed_by,
            review_note=row.review_note,
        )

    def _load_response_ref(self, *, owner_id: int, response_ref: dict[str, Any]) -> BaseModel:
        snapshot = response_ref.get("snapshot")
        if isinstance(snapshot, dict):
            if response_ref.get("type") == "run":
                return ScheduledTaskRunResponse.model_validate(snapshot)
            if response_ref.get("type") == "result":
                return ScheduledTaskResultResponse.model_validate(snapshot)
        if response_ref.get("type") == "run":
            return self.get_run(owner_id=owner_id, run_id=str(response_ref["id"]))
        if response_ref.get("type") == "result":
            return self.get_result(owner_id=owner_id, result_id=str(response_ref["id"]))
        return super()._load_response_ref(owner_id=owner_id, response_ref=response_ref)

    def _response_ref(self, response: BaseModel) -> dict[str, Any]:
        if isinstance(response, ScheduledTaskRunResponse):
            return {
                "type": "run",
                "id": response.id,
                "snapshot": response.model_dump(mode="json"),
            }
        if isinstance(response, ScheduledTaskResultResponse):
            return {
                "type": "result",
                "id": response.id,
                "snapshot": response.model_dump(mode="json"),
            }
        return super()._response_ref(response)


def _scope_has_sources(scope: dict[str, Any]) -> bool:
    if scope.get("mode") == "all_searchable_library":
        sources = scope.get("resolved_sources")
    else:
        sources = scope.get("sources")
    return isinstance(sources, list) and any(isinstance(source, str) and source.strip() for source in sources)


def _ensure_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _retention_ttl_days(
    policy: dict[str, Any],
    *,
    preferred_key: str,
    fallback_key: str | None,
    default_days: int,
) -> int:
    candidate = policy.get(preferred_key)
    if candidate is None and fallback_key is not None:
        candidate = policy.get(fallback_key)
    try:
        days = int(candidate)
    except (TypeError, ValueError):
        return default_days
    return max(days, 0)


def _parse_optional_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return _ensure_aware_utc(parsed)


def _normalize_schedule_slot(value: str | datetime) -> str:
    if isinstance(value, datetime):
        return _ensure_aware_utc(value).isoformat()
    parsed = _parse_optional_datetime(str(value))
    if parsed is None:
        raise ScheduledTaskAutomationError("invalid_schedule_slot")
    return parsed.isoformat()
