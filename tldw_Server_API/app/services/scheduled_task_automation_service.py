"""Service layer for Scheduled Tasks automation definition foundations."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from uuid import uuid4

from pydantic import BaseModel

from tldw_Server_API.app.api.v1.schemas.scheduled_tasks_automation_schemas import (
    ScheduledTaskActionCapability,
    ScheduledTaskAuditEventResponse,
    ScheduledTaskAuditListResponse,
    ScheduledTaskAutomationCapabilitiesResponse,
    ScheduledTaskAutomationCapability,
    ScheduledTaskDefinitionCreateRequest,
    ScheduledTaskDefinitionListResponse,
    ScheduledTaskDefinitionResponse,
    ScheduledTaskDefinitionUpdateRequest,
    ScheduledTaskDuplicateRequest,
    ScheduledTaskExecutionCertificationCapability,
    ScheduledTaskPreviewCreateRequest,
    ScheduledTaskPreviewListResponse,
    ScheduledTaskPreviewResponse,
    ScheduledTaskResultResponse,
    ScheduledTaskRunNowResponse,
    ScheduledTaskRunResponse,
)
from tldw_Server_API.app.core.AuthNZ.permissions import TASKS_CONTROL
from tldw_Server_API.app.core.DB_Management.Scheduled_Tasks_DB import (
    AuditEventRow,
    DefinitionRow,
    PreviewRow,
    ScheduledTasksDatabase,
    ScheduledTasksTransaction,
)
from tldw_Server_API.app.core.Scheduled_Tasks.execution_certification import (
    AgentAutomationAdmission,
    AgentExecutionDispatchReadiness,
    ExecutionCertification,
    agent_automation_admission,
    agent_execution_dispatch_readiness,
    certification_recovery_action,
    current_agent_execution_stack_ready,
    readiness_recovery_action,
    resolve_current_agent_execution_certification,
)
from tldw_Server_API.app.core.Scheduled_Tasks.recurring_question_models import (
    FINDING_POLICY_PRESETS,
    GENERATION_MODES,
    RETENTION_POLICY_MODES,
)
from tldw_Server_API.app.core.Scheduled_Tasks.recurring_question_scope import normalize_recurring_question_scope
from tldw_Server_API.app.core.testing import env_flag_enabled

PREVIEW_TTL = timedelta(hours=24)
IDEMPOTENCY_TTL = timedelta(hours=24)
DEFAULT_DEFINITION_HEALTH = "execution_unavailable"
_SUPPORTED_SCHEDULE_KINDS = {"one_time", "interval", "daily", "weekly", "cron"}
class ScheduledTaskAutomationError(Exception):
    """Expected, user-actionable scheduled task automation failure."""

    def __init__(
        self,
        code: str,
        *,
        reason: str | None = None,
        recovery_action: str | None = None,
    ) -> None:
        """Initialize a bounded service error for endpoint translation."""

        super().__init__(code)
        self.code = code
        self.reason = reason
        self.recovery_action = recovery_action


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _canonical_hash(payload: dict[str, Any]) -> str:
    """Return a stable SHA-256 hash for JSON-compatible idempotency payloads."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _field_error(field: str, code: str, message: str) -> dict[str, Any]:
    return {"field": field, "code": code, "message": message}


def _redact_agent_message(message: str) -> dict[str, Any]:
    """Return safe Agent Task message metadata without the raw message."""
    return {
        "message_redacted": True,
        "message_ref": f"redacted:{uuid4().hex}",
        "message_preview": "[redacted]",
    }


def _validate_schedule(schedule: Any) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    if not isinstance(schedule, dict):
        return {}, [_field_error("schedule", "invalid_type", "Schedule must be an object.")], warnings

    normalized = dict(schedule)
    kind = normalized.get("kind")
    if not isinstance(kind, str) or not kind.strip():
        errors.append(_field_error("schedule.kind", "required", "Schedule kind is required."))
    elif kind not in _SUPPORTED_SCHEDULE_KINDS:
        errors.append(_field_error("schedule.kind", "unsupported", f"Unsupported schedule kind: {kind}"))
    else:
        normalized["kind"] = kind
    return normalized, errors, warnings


def _validate_recurring_question_config(
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    name = str(config.get("name") or "").strip()
    option_config = dict(config.get("config") or {})
    input_config = dict(config.get("input") or {})
    question = str(input_config.get("question") or "").strip()

    if not name:
        errors.append(_field_error("name", "required", "Name is required."))
    if not question:
        errors.append(_field_error("input.question", "required", "Question is required."))

    scope, scope_errors, scope_warnings = normalize_recurring_question_scope(option_config.get("scope"))
    finding_policy = _normalize_finding_policy(option_config.get("finding_policy"), errors)
    retention_policy = _normalize_retention_policy(option_config.get("retention_policy"), errors)
    generation_mode = str(option_config.get("generation_mode") or "optional").strip() or "optional"
    if generation_mode not in GENERATION_MODES:
        errors.append(
            _field_error(
                "config.generation_mode",
                "unsupported",
                f"Unsupported generation mode: {generation_mode}",
            )
        )

    normalized = dict(config)
    normalized["name"] = name
    normalized["input"] = {**input_config, "question": question}
    normalized["config"] = {
        **option_config,
        "scope": scope,
        "finding_policy": finding_policy,
        "retention_policy": retention_policy,
        "generation_mode": generation_mode,
    }
    errors.extend(scope_errors)
    warnings.extend(warning["code"] for warning in scope_warnings)
    return normalized, errors, warnings


def _normalize_finding_policy(value: Any, errors: list[dict[str, Any]]) -> dict[str, Any]:
    policy = dict(value) if isinstance(value, dict) else {}
    preset = str(policy.get("preset") or "balanced_findings").strip() or "balanced_findings"
    if preset not in FINDING_POLICY_PRESETS:
        errors.append(
            _field_error(
                "config.finding_policy.preset",
                "unsupported",
                f"Unsupported finding policy preset: {preset}",
            )
        )
    return {**policy, "preset": preset}


def _normalize_retention_policy(value: Any, errors: list[dict[str, Any]]) -> dict[str, Any]:
    policy = dict(value) if isinstance(value, dict) else {}
    mode = str(policy.get("mode") or "default").strip() or "default"
    if mode not in RETENTION_POLICY_MODES:
        errors.append(
            _field_error(
                "config.retention_policy.mode",
                "unsupported",
                f"Unsupported retention policy mode: {mode}",
            )
        )
    return {**policy, "mode": mode}


def _validate_agent_task_config(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    name = str(config.get("name") or "").strip()
    input_config = dict(config.get("input") or {})
    agent_ref = str(input_config.get("agent_ref") or "").strip()
    message = str(input_config.get("message") or "")

    if not name:
        errors.append(_field_error("name", "required", "Name is required."))
    if not agent_ref:
        errors.append(_field_error("input.agent_ref", "required", "Agent reference is required."))
    if not message.strip():
        errors.append(_field_error("input.message", "required", "Agent message is required."))

    safe_input = {key: value for key, value in input_config.items() if key != "message"}
    if message:
        safe_input.update(_redact_agent_message(message))
    safe_input["agent_ref"] = agent_ref

    normalized = dict(config)
    normalized["name"] = name
    normalized["input"] = safe_input
    normalized["redaction_policy"] = {"fields": ["input.message"], "mode": "metadata_only"}
    return normalized, errors, warnings


def _normalize_visibility_policy(family: str, value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        mode = value.get("mode") or value.get("visibility") or value.get("policy")
        if isinstance(mode, str) and mode.strip():
            return mode.strip()
    return "metadata_only" if family == "agent_task" else "findings_only"


class ScheduledTaskAutomationService:
    """Business service for Scheduled Tasks-owned automation definitions."""

    def __init__(
        self,
        repository: ScheduledTasksDatabase | None = None,
        *,
        execution_certification_resolver: Callable[
            [], ExecutionCertification
        ] = resolve_current_agent_execution_certification,
        execution_stack_ready_resolver: Callable[
            [], bool
        ] = current_agent_execution_stack_ready,
    ) -> None:
        """Initialize repository access and injectable readiness resolvers."""

        self._repository = repository
        self._schema_ready_keys: set[tuple[int, str]] = set()
        self._execution_certification_resolver = (
            execution_certification_resolver
        )
        self._execution_stack_ready_resolver = execution_stack_ready_resolver

    def _agent_execution_readiness(
        self,
    ) -> tuple[ExecutionCertification, AgentExecutionDispatchReadiness]:
        """Resolve certification and core dispatch readiness together."""

        certification = self._execution_certification_resolver()
        readiness = agent_execution_dispatch_readiness(
            certification,
            execution_stack_ready=self._execution_stack_ready_resolver(),
        )
        return certification, readiness

    @staticmethod
    def _certification_capability(
        certification: ExecutionCertification,
        admission: AgentAutomationAdmission,
    ) -> ScheduledTaskExecutionCertificationCapability:
        """Project one sanitized core certification into the API schema."""

        return ScheduledTaskExecutionCertificationCapability(
            outcome=admission.effective_outcome,
            deployment_class_id=certification.deployment_class_id,
            evidence_id=certification.evidence_id,
            evidence_source=certification.evidence_source,
            observed_at=certification.observed_at,
            expires_at=certification.expires_at,
            reason_codes=list(certification.reason_codes),
            recovery_action=certification_recovery_action(
                admission.effective_outcome
            ),
        )

    @staticmethod
    def _agent_gated_action(
        *,
        certification: ExecutionCertification,
        readiness: AgentExecutionDispatchReadiness,
        status: str = "disabled",
    ) -> ScheduledTaskActionCapability:
        """Project one core readiness blocker into an action capability."""

        return ScheduledTaskActionCapability(
            status=status,
            reason=readiness.reason,
            required_permissions=[TASKS_CONTROL],
            evidence_source=certification.evidence_source,
            recovery_action=readiness_recovery_action(readiness.reason),
            observed_at=certification.observed_at,
            expires_at=certification.expires_at,
        )

    def _require_agent_execution_available(self, family: str) -> None:
        """Raise the service error when core execution admission is blocked."""

        if family != "agent_task":
            return
        _certification, readiness = self._agent_execution_readiness()
        if readiness.ready:
            return
        raise ScheduledTaskAutomationError(
            "agent_execution_unavailable",
            reason=readiness.reason,
            recovery_action=readiness_recovery_action(readiness.reason),
        )

    def _require_agent_automation_supported(self, family: str) -> None:
        """Raise the service error when core authoring admission is blocked."""

        if family != "agent_task":
            return
        admission = agent_automation_admission(
            self._execution_certification_resolver()
        )
        if admission.allowed:
            return
        raise ScheduledTaskAutomationError(
            "agent_automation_unsupported",
            reason=admission.reason,
            recovery_action=admission.recovery_action,
        )

    def get_capabilities(self) -> ScheduledTaskAutomationCapabilitiesResponse:
        """Return additive per-family capability and feasibility truth."""

        def _execution_actions(
            tools_reason: str,
        ) -> dict[str, ScheduledTaskActionCapability]:
            """Build the common action map for executable task families."""

            actions = self._definition_actions()
            actions["run_now"] = ScheduledTaskActionCapability(
                status="available",
                required_permissions=[TASKS_CONTROL],
            )
            actions["execute"] = ScheduledTaskActionCapability(
                status="available",
                reason="phase1_generation_only",
                required_permissions=[TASKS_CONTROL],
            )
            actions["execute_tools"] = ScheduledTaskActionCapability(
                status="planned",
                reason=tools_reason,
                required_permissions=[TASKS_CONTROL],
            )
            return actions

        recurring_question_actions = self._recurring_question_actions()
        recurring_question_actions.update(
            _execution_actions(
                "recurring_question has no tool surface; tools are not applicable"
            )
        )
        certification, readiness = self._agent_execution_readiness()
        admission = agent_automation_admission(certification)
        agent_actions = self._definition_actions()
        agent_actions["execute"] = self._agent_gated_action(
            certification=certification,
            readiness=readiness,
        )
        agent_actions["run_now"] = self._agent_gated_action(
            certification=certification,
            readiness=readiness,
        )
        agent_actions["execute_tools"] = ScheduledTaskActionCapability(
            status="planned",
            reason="agent_tool_execution_requires_reviewed_approval_mediation",
            required_permissions=[TASKS_CONTROL],
        )
        if not admission.allowed:
            for action_name in (
                "preview",
                "create_definition",
                "update_definition",
                "duplicate",
            ):
                agent_actions[action_name] = self._agent_gated_action(
                    certification=certification,
                    readiness=readiness,
                    status="unavailable",
                )
        return ScheduledTaskAutomationCapabilitiesResponse(
            items=[
                ScheduledTaskAutomationCapability(
                    family="recurring_question",
                    family_availability="available",
                    actions=recurring_question_actions,
                    related_capabilities={
                        "rag": {"status": "not_checked"},
                        "scheduler": {
                            "status": "enabled"
                            if env_flag_enabled("SCHEDULED_TASKS_RECURRING_QUESTION_SCHEDULER_ENABLED")
                            else "disabled",
                        },
                        "worker": {
                            "status": "enabled"
                            if env_flag_enabled("SCHEDULED_TASKS_RECURRING_QUESTION_WORKER_ENABLED")
                            else "disabled",
                        },
                    },
                ),
                ScheduledTaskAutomationCapability(
                    family="agent_task",
                    family_availability=(
                        "available" if admission.allowed else "unavailable"
                    ),
                    actions=agent_actions,
                    related_capabilities={"acp": {"status": "not_checked"}},
                    reason=admission.reason,
                    execution_certification=self._certification_capability(
                        certification,
                        admission,
                    ),
                ),
            ]
        )

    def create_preview(
        self,
        *,
        owner_id: int,
        actor: str,
        payload: ScheduledTaskPreviewCreateRequest | dict[str, Any],
        idempotency_key: str | None = None,
    ) -> ScheduledTaskPreviewResponse:
        request = payload if isinstance(payload, ScheduledTaskPreviewCreateRequest) else ScheduledTaskPreviewCreateRequest(**payload)
        self._require_agent_automation_supported(request.family)
        payload_hash = _canonical_hash(self._preview_hash_payload(request))
        return self._with_idempotency(
            owner_id=owner_id,
            route="scheduled_task_automation.preview",
            key=idempotency_key,
            payload_hash=payload_hash,
            operation=lambda tx: self._create_preview(
                tx=tx,
                owner_id=owner_id,
                actor=actor,
                request=request,
                payload_hash=payload_hash,
            ),
        )

    def get_preview(self, *, owner_id: int, preview_id: str) -> ScheduledTaskPreviewResponse:
        preview = self._repo(owner_id).get_preview(owner_id=owner_id, preview_id=preview_id)
        if preview is None:
            raise ScheduledTaskAutomationError("preview_resource_not_found")
        return self._preview_response(preview)

    def list_previews(
        self,
        *,
        owner_id: int | None = None,
        limit: int,
        offset: int,
        family: str | None = None,
        mode: str | None = None,
        status: str | None = None,
        definition_id: str | None = None,
        expired: bool | None = None,
    ) -> ScheduledTaskPreviewListResponse:
        """Return a page of owner-scoped previews."""
        if owner_id is None:
            return ScheduledTaskPreviewListResponse(limit=limit, offset=offset)
        rows, total = self._repo(owner_id).list_previews(
            owner_id=owner_id,
            limit=limit,
            offset=offset,
            family=family,
            mode=mode,
            status=status,
            definition_id=definition_id,
            expired=expired,
        )
        return ScheduledTaskPreviewListResponse(
            items=[self._preview_response(row) for row in rows],
            total=total,
            limit=limit,
            offset=offset,
            has_more=offset + len(rows) < total,
            next_offset=offset + len(rows) if offset + len(rows) < total else None,
        )

    def create_definition(
        self,
        *,
        owner_id: int,
        actor: str,
        payload: ScheduledTaskDefinitionCreateRequest | dict[str, Any],
        idempotency_key: str | None = None,
        request_id: str | None = None,
    ) -> ScheduledTaskDefinitionResponse:
        request = (
            payload
            if isinstance(payload, ScheduledTaskDefinitionCreateRequest)
            else ScheduledTaskDefinitionCreateRequest(**payload)
        )
        preview = self._repo(owner_id).get_preview(
            owner_id=owner_id,
            preview_id=request.preview_id,
        )
        if preview is not None:
            self._require_agent_automation_supported(preview.family)
        payload_hash = _canonical_hash(request.model_dump(mode="json"))
        return self._with_idempotency(
            owner_id=owner_id,
            route="scheduled_task_automation.definition.create",
            key=idempotency_key,
            payload_hash=payload_hash,
            operation=lambda tx: self._create_definition(
                tx=tx,
                owner_id=owner_id,
                actor=actor,
                request=request,
                idempotency_key=idempotency_key,
                request_id=request_id,
            ),
        )

    def update_definition(
        self,
        *,
        owner_id: int,
        actor: str,
        definition_id: str,
        payload: ScheduledTaskDefinitionUpdateRequest | dict[str, Any],
        idempotency_key: str | None = None,
        request_id: str | None = None,
    ) -> ScheduledTaskDefinitionResponse:
        """Update a definition from a valid preview under current admission.

        Args:
            owner_id: Authenticated definition owner.
            actor: Sanitized audit actor identity.
            definition_id: Existing definition identifier.
            payload: Typed update request or equivalent mapping.
            idempotency_key: Optional replay key for this exact request.
            request_id: Optional request correlation identifier.

        Returns:
            The updated definition projection.

        Raises:
            ScheduledTaskAutomationError: If the definition, preview,
                lifecycle, version, or Agent admission state is invalid.
        """

        request = (
            payload
            if isinstance(payload, ScheduledTaskDefinitionUpdateRequest)
            else ScheduledTaskDefinitionUpdateRequest(**payload)
        )
        current = self._get_definition_row(
            owner_id=owner_id,
            definition_id=definition_id,
        )
        self._require_agent_automation_supported(current.family)
        payload_hash = _canonical_hash({"definition_id": definition_id, **request.model_dump(mode="json")})
        return self._with_idempotency(
            owner_id=owner_id,
            route="scheduled_task_automation.definition.update",
            key=idempotency_key,
            payload_hash=payload_hash,
            operation=lambda tx: self._update_definition(
                tx=tx,
                owner_id=owner_id,
                actor=actor,
                definition_id=definition_id,
                request=request,
                idempotency_key=idempotency_key,
                request_id=request_id,
            ),
        )

    def get_definition(self, *, owner_id: int, definition_id: str) -> ScheduledTaskDefinitionResponse:
        definition = self._get_definition_row(owner_id=owner_id, definition_id=definition_id)
        return self._definition_response(definition)

    def list_definitions(
        self,
        *,
        owner_id: int | None = None,
        limit: int,
        offset: int,
        family: str | None = None,
        lifecycle: str | None = None,
        health: str | None = None,
        visibility_policy: str | None = None,
        query: str | None = None,
        created_from: str | None = None,
        created_to: str | None = None,
    ) -> ScheduledTaskDefinitionListResponse:
        """Return a page of owner-scoped automation definitions."""
        if owner_id is None:
            return ScheduledTaskDefinitionListResponse(limit=limit, offset=offset)
        repo = self._repo(owner_id)
        rows, total = repo.list_definitions(
            owner_id=owner_id,
            limit=limit,
            offset=offset,
            family=family,
            lifecycle=lifecycle,
            health=health,
            visibility_policy=visibility_policy,
            query=query,
            created_from=created_from,
            created_to=created_to,
        )
        previews_by_id = repo.get_previews_by_ids(owner_id=owner_id, preview_ids=[row.preview_id for row in rows])
        return ScheduledTaskDefinitionListResponse(
            items=[
                self._definition_response(
                    row,
                    preview=previews_by_id.get(row.preview_id),
                    lookup_preview=False,
                )
                for row in rows
            ],
            total=total,
            limit=limit,
            offset=offset,
            has_more=offset + len(rows) < total,
            next_offset=offset + len(rows) if offset + len(rows) < total else None,
        )

    def pause_definition(
        self,
        *,
        owner_id: int,
        actor: str,
        definition_id: str,
        idempotency_key: str | None = None,
        request_id: str | None = None,
    ) -> ScheduledTaskDefinitionResponse:
        payload_hash = _canonical_hash({"definition_id": definition_id, "action": "pause"})
        return self._with_idempotency(
            owner_id=owner_id,
            route="scheduled_task_automation.definition.pause",
            key=idempotency_key,
            payload_hash=payload_hash,
            operation=lambda tx: self._transition_definition(
                tx=tx,
                owner_id=owner_id,
                actor=actor,
                definition_id=definition_id,
                target_lifecycle="paused",
                idempotent_lifecycle="paused",
                event_type="definition.paused",
                summary="Paused definition",
                idempotency_key=idempotency_key,
                request_id=request_id,
            ),
        )

    def resume_definition(
        self,
        *,
        owner_id: int,
        actor: str,
        definition_id: str,
        idempotency_key: str | None = None,
        request_id: str | None = None,
    ) -> ScheduledTaskDefinitionResponse:
        payload_hash = _canonical_hash({"definition_id": definition_id, "action": "resume"})
        return self._with_idempotency(
            owner_id=owner_id,
            route="scheduled_task_automation.definition.resume",
            key=idempotency_key,
            payload_hash=payload_hash,
            operation=lambda tx: self._transition_definition(
                tx=tx,
                owner_id=owner_id,
                actor=actor,
                definition_id=definition_id,
                target_lifecycle="configured",
                idempotent_lifecycle="configured",
                event_type="definition.resumed",
                summary="Resumed definition",
                idempotency_key=idempotency_key,
                request_id=request_id,
            ),
        )

    def archive_definition(
        self,
        *,
        owner_id: int,
        actor: str,
        definition_id: str,
        idempotency_key: str | None = None,
        request_id: str | None = None,
    ) -> ScheduledTaskDefinitionResponse:
        payload_hash = _canonical_hash({"definition_id": definition_id, "action": "archive"})
        return self._with_idempotency(
            owner_id=owner_id,
            route="scheduled_task_automation.definition.archive",
            key=idempotency_key,
            payload_hash=payload_hash,
            operation=lambda tx: self._archive_definition(
                tx=tx,
                owner_id=owner_id,
                actor=actor,
                definition_id=definition_id,
                idempotency_key=idempotency_key,
                request_id=request_id,
            ),
        )

    def run_now(
        self,
        *,
        owner_id: int,
        actor: str,
        definition_id: str,
        idempotency_key: str | None = None,
        request_id: str | None = None,
        jobs: Any | None = None,
    ) -> ScheduledTaskRunNowResponse:
        """Trigger one immediate execution through the standard Jobs path.

        A manual run is a REAL dispatch (tldw_chatbook ADR-077 decision 7 /
        TASK-13022): the same ``agent_task_run`` Jobs pipeline the feed
        enqueues into, with the same idempotency-key semantics -- a manual
        run colliding with a scheduled run of the same slot dedupes
        exactly like a redelivered Job. The manual slot is "now",
        second-truncated UTC; a repeat trigger inside the same second
        returns the existing job (``deduped=True``).

        Lifecycle refusals reuse the transition error codes: archived
        definitions refuse ``definition_archived``; admin/security-locked
        disabled definitions refuse ``definition_disabled_locked``; paused
        or unlocked-disabled definitions refuse ``definition_paused`` /
        ``definition_disabled`` (a manual trigger must not silently
        resurrect a definition the owner paused).
        """
        from datetime import datetime
        from datetime import timezone as _tz

        from tldw_Server_API.app.core.Jobs.manager import JobManager
        from tldw_Server_API.app.services.scheduled_task_automation_scheduler import (
            automation_jobs_queue,
        )

        repo = self._repo(owner_id)
        definition = repo.get_definition(owner_id=owner_id, definition_id=definition_id)
        if definition is None:
            raise ScheduledTaskAutomationError("definition_not_found")
        self._require_agent_execution_available(definition.family)
        if definition.lifecycle == "archived":
            raise ScheduledTaskAutomationError("definition_archived")
        if definition.lifecycle == "disabled" and definition.disabled_lock_kind in {
            "admin",
            "security",
        }:
            raise ScheduledTaskAutomationError("definition_disabled_locked")
        if definition.lifecycle == "paused":
            raise ScheduledTaskAutomationError("definition_paused")
        if definition.lifecycle == "disabled":
            raise ScheduledTaskAutomationError("definition_disabled")

        payload_hash = _canonical_hash(
            {"definition_id": definition_id, "action": "run_now"}
        )

        def _dispatch(_tx: Any) -> ScheduledTaskRunNowResponse:
            run_slot_utc = (
                datetime.now(_tz.utc).replace(microsecond=0).isoformat()
            )
            payload = {
                "definition_id": definition.id,
                "user_id": owner_id,
                "family": definition.family,
                "scheduled_for": run_slot_utc,
                "manual": True,
            }
            idem = f"definition:{definition.id}:{run_slot_utc}"
            jm = jobs if jobs is not None else JobManager()
            job = jm.create_job(
                domain="scheduled_tasks",
                queue=automation_jobs_queue(),
                job_type="agent_task_run",
                payload=payload,
                owner_user_id=owner_id,
                idempotency_key=idem,
            )
            deduped = bool(job.get("deduped")) if isinstance(job, dict) else False

            try:
                repo.create_audit_event(
                    owner_id=owner_id,
                    definition_id=definition.id,
                    event_type="definition.run_now",
                    actor=actor,
                    summary=f"Manual run triggered (slot {run_slot_utc})",
                    before=None,
                    after={
                        "run_slot_utc": run_slot_utc,
                        "job_id": (job or {}).get("id") if isinstance(job, dict) else None,
                    },
                    request_id=request_id,
                    idempotency_key=idempotency_key,
                )
            except Exception:  # noqa: BLE001 - audit must not fail the trigger
                from loguru import logger as _logger

                _logger.exception(
                    "run_now audit failed",
                    definition_id=definition_id,
                    owner_id=owner_id,
                )

            return ScheduledTaskRunNowResponse(
                definition_id=definition.id,
                run_slot_utc=run_slot_utc,
                job_id=(job or {}).get("id") if isinstance(job, dict) else None,
                deduped=deduped,
            )

        # Request-retry idempotency: the same (route, idempotency-key,
        # payload) replays the PRIOR response instead of enqueueing a new
        # manual slot -- the control-plane convention every other mutating
        # action follows. The JOB-layer key (definition:{id}:{slot})
        # remains the slot-collision dedupe with the scheduler feed.
        if idempotency_key is None:
            return _dispatch(None)
        return self._with_idempotency(
            owner_id=owner_id,
            route="scheduled_task_automation.definition.run_now",
            key=idempotency_key,
            payload_hash=payload_hash,
            operation=_dispatch,
        )

    def mark_solved(
        self,
        *,
        owner_id: int,
        actor: str,
        definition_id: str,
        resolved_result_id: str | None = None,
        idempotency_key: str | None = None,
        request_id: str | None = None,
    ) -> ScheduledTaskDefinitionResponse:
        payload_hash = _canonical_hash(
            {
                "definition_id": definition_id,
                "action": "mark_solved",
                "resolved_result_id": resolved_result_id,
            }
        )
        return self._with_idempotency(
            owner_id=owner_id,
            route="scheduled_task_automation.definition.mark_solved",
            key=idempotency_key,
            payload_hash=payload_hash,
            operation=lambda tx: self._mark_solved_definition(
                tx=tx,
                owner_id=owner_id,
                actor=actor,
                definition_id=definition_id,
                resolved_result_id=resolved_result_id,
                idempotency_key=idempotency_key,
                request_id=request_id,
            ),
        )

    def reopen_definition(
        self,
        *,
        owner_id: int,
        actor: str,
        definition_id: str,
        target_lifecycle: str = "paused",
        reason: str | None = None,
        idempotency_key: str | None = None,
        request_id: str | None = None,
    ) -> ScheduledTaskDefinitionResponse:
        payload_hash = _canonical_hash(
            {
                "definition_id": definition_id,
                "action": "reopen",
                "target_lifecycle": target_lifecycle,
                "reason": reason,
            }
        )
        return self._with_idempotency(
            owner_id=owner_id,
            route="scheduled_task_automation.definition.reopen",
            key=idempotency_key,
            payload_hash=payload_hash,
            operation=lambda tx: self._reopen_definition(
                tx=tx,
                owner_id=owner_id,
                actor=actor,
                definition_id=definition_id,
                target_lifecycle=target_lifecycle,
                reason=reason,
                idempotency_key=idempotency_key,
                request_id=request_id,
            ),
        )

    def duplicate_definition(
        self,
        *,
        owner_id: int,
        actor: str,
        definition_id: str,
        payload: ScheduledTaskDuplicateRequest | dict[str, Any],
        idempotency_key: str | None = None,
        request_id: str | None = None,
    ) -> ScheduledTaskDefinitionResponse:
        request = payload if isinstance(payload, ScheduledTaskDuplicateRequest) else ScheduledTaskDuplicateRequest(**payload)
        source = self._get_definition_row(
            owner_id=owner_id,
            definition_id=definition_id,
        )
        self._require_agent_automation_supported(source.family)
        payload_hash = _canonical_hash({"definition_id": definition_id, **request.model_dump(mode="json")})
        return self._with_idempotency(
            owner_id=owner_id,
            route="scheduled_task_automation.definition.duplicate",
            key=idempotency_key,
            payload_hash=payload_hash,
            operation=lambda tx: self._duplicate_definition(
                tx=tx,
                owner_id=owner_id,
                actor=actor,
                definition_id=definition_id,
                request=request,
                idempotency_key=idempotency_key,
                request_id=request_id,
            ),
        )

    def list_audit_events(
        self,
        *,
        owner_id: int | None = None,
        definition_id: str | None = None,
        limit: int,
        offset: int,
        event_type: str | None = None,
        actor: str | None = None,
        created_from: str | None = None,
        created_to: str | None = None,
        idempotency_key: str | None = None,
        request_id: str | None = None,
    ) -> ScheduledTaskAuditListResponse:
        """Return a page of owner-scoped definition audit events."""
        if owner_id is None or definition_id is None:
            return ScheduledTaskAuditListResponse(limit=limit, offset=offset)
        self._get_definition_row(owner_id=owner_id, definition_id=definition_id)
        rows, total = self._repo(owner_id).list_audit_events(
            owner_id=owner_id,
            definition_id=definition_id,
            limit=limit,
            offset=offset,
            event_type=event_type,
            actor=actor,
            created_from=created_from,
            created_to=created_to,
            idempotency_key=idempotency_key,
            request_id=request_id,
        )
        return ScheduledTaskAuditListResponse(
            items=[self._audit_response(row) for row in rows],
            total=total,
            limit=limit,
            offset=offset,
            has_more=offset + len(rows) < total,
            next_offset=offset + len(rows) if offset + len(rows) < total else None,
        )

    def _repo(self, owner_id: int) -> ScheduledTasksDatabase:
        repo = self._repository or ScheduledTasksDatabase.for_user(owner_id)
        schema_key = (owner_id, str(repo.db_path))
        if schema_key not in self._schema_ready_keys:
            repo.ensure_schema()
            self._schema_ready_keys.add(schema_key)
        return repo

    def _create_preview(
        self,
        *,
        tx: ScheduledTasksTransaction,
        owner_id: int,
        actor: str,
        request: ScheduledTaskPreviewCreateRequest,
        payload_hash: str,
    ) -> ScheduledTaskPreviewResponse:
        self._require_agent_automation_supported(request.family)
        normalized, validation_errors, warnings = self._normalize_preview(request)
        status = "invalid" if validation_errors else "valid"
        row = tx.create_preview(
            owner_id=owner_id,
            mode=request.mode,
            family=request.family,
            definition_id=request.definition_id,
            definition_version=request.definition_version,
            status=status,
            payload_hash=payload_hash,
            normalized_config=normalized,
            validation_errors=validation_errors,
            warnings=[{"message": warning} for warning in warnings],
            risk_class=None,
            visibility_policy=normalized["visibility_policy"],
            schedule_preview=normalized["schedule"],
            redaction_policy=normalized.get("redaction_policy", {"mode": "none", "fields": []}),
            expires_at=_iso(_utcnow() + PREVIEW_TTL),
            created_by=actor,
        )
        return self._preview_response(row)

    def _create_definition(
        self,
        *,
        tx: ScheduledTasksTransaction,
        owner_id: int,
        actor: str,
        request: ScheduledTaskDefinitionCreateRequest,
        idempotency_key: str | None,
        request_id: str | None,
    ) -> ScheduledTaskDefinitionResponse:
        preview = self._require_valid_preview(tx=tx, owner_id=owner_id, preview_id=request.preview_id)
        self._require_agent_automation_supported(preview.family)
        if preview.mode != "create" or preview.definition_id is not None:
            raise ScheduledTaskAutomationError("preview_mode_mismatch")
        normalized = preview.normalized_config
        normalized_config = normalized.get("config", {})
        definition = tx.create_definition(
            owner_id=owner_id,
            family=preview.family,
            name=normalized["name"],
            description=normalized.get("description"),
            lifecycle=request.initial_lifecycle,
            health=DEFAULT_DEFINITION_HEALTH,
            schedule=normalized["schedule"],
            input=normalized["input"],
            visibility_policy=preview.visibility_policy,
            notification_policy=normalized.get("notification_policy", {}),
            approval_policy=normalized.get("approval_policy", {}),
            preview_id=preview.id,
            created_by=actor,
            updated_by=actor,
            finding_policy=normalized_config.get("finding_policy"),
            retention_policy=normalized_config.get("retention_policy"),
        )
        response = self._definition_response(definition)
        self._create_audit(
            tx=tx,
            owner_id=owner_id,
            definition_id=definition.id,
            event_type="definition.created",
            actor=actor,
            summary="Created definition",
            before=None,
            after=response.model_dump(mode="json"),
            idempotency_key=idempotency_key,
            request_id=request_id,
        )
        return response

    def _update_definition(
        self,
        *,
        tx: ScheduledTasksTransaction,
        owner_id: int,
        actor: str,
        definition_id: str,
        request: ScheduledTaskDefinitionUpdateRequest,
        idempotency_key: str | None,
        request_id: str | None,
    ) -> ScheduledTaskDefinitionResponse:
        """Apply one previewed update after transactional admission checks."""

        current = self._get_definition_row(tx=tx, owner_id=owner_id, definition_id=definition_id)
        self._require_agent_automation_supported(current.family)
        if current.lifecycle == "archived":
            raise ScheduledTaskAutomationError("definition_archived")
        preview = self._require_valid_preview(tx=tx, owner_id=owner_id, preview_id=request.preview_id)
        self._require_agent_automation_supported(preview.family)
        if preview.mode != "update" or preview.definition_id != definition_id:
            raise ScheduledTaskAutomationError("preview_definition_mismatch")
        if preview.definition_version != current.version:
            raise ScheduledTaskAutomationError("definition_version_mismatch")
        normalized = preview.normalized_config
        normalized_config = normalized.get("config", {})
        updated = tx.update_definition(
            owner_id=owner_id,
            definition_id=definition_id,
            patch={
                "family": preview.family,
                "name": normalized["name"],
                "description": normalized.get("description"),
                "schedule": normalized["schedule"],
                "input": normalized["input"],
                "visibility_policy": preview.visibility_policy,
                "notification_policy": normalized.get("notification_policy", {}),
                "approval_policy": normalized.get("approval_policy", {}),
                "preview_id": preview.id,
                "updated_by": actor,
                "finding_policy": normalized_config.get("finding_policy", current.finding_policy),
                "retention_policy": normalized_config.get("retention_policy", current.retention_policy),
            },
            expected_version=current.version,
        )
        tx.mark_preview_consumed(
            owner_id=owner_id,
            preview_id=preview.id,
            created_definition_id=definition_id,
        )
        response = self._definition_response(updated)
        self._create_audit(
            tx=tx,
            owner_id=owner_id,
            definition_id=definition_id,
            event_type="definition.updated",
            actor=actor,
            summary="Updated definition",
            before=self._definition_response(current).model_dump(mode="json"),
            after=response.model_dump(mode="json"),
            idempotency_key=idempotency_key,
            request_id=request_id,
        )
        return response

    def _transition_definition(
        self,
        *,
        tx: ScheduledTasksTransaction,
        owner_id: int,
        actor: str,
        definition_id: str,
        target_lifecycle: str,
        idempotent_lifecycle: str,
        event_type: str,
        summary: str,
        idempotency_key: str | None,
        request_id: str | None,
    ) -> ScheduledTaskDefinitionResponse:
        current = self._get_definition_row(tx=tx, owner_id=owner_id, definition_id=definition_id)
        if current.lifecycle == "archived":
            raise ScheduledTaskAutomationError("definition_archived")
        if current.lifecycle == idempotent_lifecycle:
            return self._definition_response(current)
        if current.lifecycle == "disabled":
            raise ScheduledTaskAutomationError("definition_disabled")
        updated = tx.update_definition(
            owner_id=owner_id,
            definition_id=definition_id,
            patch={"lifecycle": target_lifecycle, "updated_by": actor},
            expected_version=current.version,
        )
        response = self._definition_response(updated)
        self._create_audit(
            tx=tx,
            owner_id=owner_id,
            definition_id=definition_id,
            event_type=event_type,
            actor=actor,
            summary=summary,
            before=self._definition_response(current).model_dump(mode="json"),
            after=response.model_dump(mode="json"),
            idempotency_key=idempotency_key,
            request_id=request_id,
        )
        return response

    def _archive_definition(
        self,
        *,
        tx: ScheduledTasksTransaction,
        owner_id: int,
        actor: str,
        definition_id: str,
        idempotency_key: str | None,
        request_id: str | None,
    ) -> ScheduledTaskDefinitionResponse:
        current = self._get_definition_row(tx=tx, owner_id=owner_id, definition_id=definition_id)
        if current.lifecycle == "archived":
            return self._definition_response(current)
        updated = tx.update_definition(
            owner_id=owner_id,
            definition_id=definition_id,
            patch={"lifecycle": "archived", "updated_by": actor},
            expected_version=current.version,
        )
        response = self._definition_response(updated)
        self._create_audit(
            tx=tx,
            owner_id=owner_id,
            definition_id=definition_id,
            event_type="definition.archived",
            actor=actor,
            summary="Archived definition",
            before=self._definition_response(current).model_dump(mode="json"),
            after=response.model_dump(mode="json"),
            idempotency_key=idempotency_key,
            request_id=request_id,
        )
        return response

    def _mark_solved_definition(
        self,
        *,
        tx: ScheduledTasksTransaction,
        owner_id: int,
        actor: str,
        definition_id: str,
        resolved_result_id: str | None,
        idempotency_key: str | None,
        request_id: str | None,
    ) -> ScheduledTaskDefinitionResponse:
        current = self._get_definition_row(tx=tx, owner_id=owner_id, definition_id=definition_id)
        if current.lifecycle == "archived":
            raise ScheduledTaskAutomationError("definition_archived")
        if current.lifecycle == "disabled":
            raise ScheduledTaskAutomationError("definition_disabled")
        if current.resolution_state == "solved":
            return self._definition_response(current)
        try:
            updated = tx.mark_definition_solved(
                owner_id=owner_id,
                definition_id=definition_id,
                resolved_by=actor,
                resolved_result_id=resolved_result_id,
            )
        except KeyError as exc:
            raise ScheduledTaskAutomationError("result_not_found") from exc
        except ValueError as exc:
            if str(exc) == "definition_family_mismatch":
                raise ScheduledTaskAutomationError("definition_family_mismatch") from exc
            raise
        response = self._definition_response(updated)
        self._create_audit(
            tx=tx,
            owner_id=owner_id,
            definition_id=definition_id,
            event_type="definition.marked_solved",
            actor=actor,
            summary="Marked definition solved",
            before=self._definition_response(current).model_dump(mode="json"),
            after=response.model_dump(mode="json"),
            idempotency_key=idempotency_key,
            request_id=request_id,
        )
        return response

    def _reopen_definition(
        self,
        *,
        tx: ScheduledTasksTransaction,
        owner_id: int,
        actor: str,
        definition_id: str,
        target_lifecycle: str,
        reason: str | None,
        idempotency_key: str | None,
        request_id: str | None,
    ) -> ScheduledTaskDefinitionResponse:
        current = self._get_definition_row(tx=tx, owner_id=owner_id, definition_id=definition_id)
        if current.family != "recurring_question":
            raise ScheduledTaskAutomationError("definition_family_mismatch")
        if current.lifecycle == "archived":
            raise ScheduledTaskAutomationError("definition_archived")
        if current.lifecycle == "disabled":
            raise ScheduledTaskAutomationError("definition_disabled")
        if target_lifecycle not in {"configured", "paused"}:
            raise ScheduledTaskAutomationError("definition_resolution_transition_invalid")
        if current.resolution_state != "solved":
            raise ScheduledTaskAutomationError("definition_resolution_transition_invalid")
        updated = tx.update_definition(
            owner_id=owner_id,
            definition_id=definition_id,
            patch={
                "resolution_state": "open",
                "resolved_at": None,
                "resolved_by": None,
                "resolved_result_id": None,
                "lifecycle": target_lifecycle,
                "updated_by": actor,
            },
            expected_version=current.version,
        )
        response = self._definition_response(updated)
        self._create_audit(
            tx=tx,
            owner_id=owner_id,
            definition_id=definition_id,
            event_type="definition.reopened",
            actor=actor,
            summary="Reopened definition",
            before=self._definition_response(current).model_dump(mode="json"),
            after={**response.model_dump(mode="json"), "reason": reason},
            idempotency_key=idempotency_key,
            request_id=request_id,
        )
        return response

    def _duplicate_definition(
        self,
        *,
        tx: ScheduledTasksTransaction,
        owner_id: int,
        actor: str,
        definition_id: str,
        request: ScheduledTaskDuplicateRequest,
        idempotency_key: str | None,
        request_id: str | None,
    ) -> ScheduledTaskDefinitionResponse:
        source = self._get_definition_row(tx=tx, owner_id=owner_id, definition_id=definition_id)
        self._require_agent_automation_supported(source.family)
        if source.lifecycle == "archived":
            raise ScheduledTaskAutomationError("definition_archived")
        if source.lifecycle == "disabled" and source.disabled_lock_kind in {"admin", "security"}:
            raise ScheduledTaskAutomationError("definition_disabled_locked")
        copy_name = request.name or f"{source.name} copy"
        copy_description = request.description if request.description is not None else source.description
        normalized = {
            "family": source.family,
            "name": copy_name,
            "description": copy_description,
            "schedule": source.schedule,
            "input": source.input,
            "visibility_policy": source.visibility_policy,
            "notification_policy": source.notification_policy,
            "approval_policy": source.approval_policy,
            "config": {},
            "redaction_policy": {"fields": ["input.message"], "mode": "metadata_only"}
            if source.family == "agent_task"
            else {"fields": [], "mode": "none"},
        }
        preview = tx.create_preview(
            owner_id=owner_id,
            mode="create",
            family=source.family,
            definition_id=None,
            definition_version=None,
            status="valid",
            payload_hash=_canonical_hash({"duplicate_of": definition_id, "name": copy_name}),
            normalized_config=normalized,
            validation_errors=[],
            warnings=[],
            risk_class=None,
            visibility_policy=source.visibility_policy,
            schedule_preview=source.schedule,
            redaction_policy=normalized["redaction_policy"],
            expires_at=_iso(_utcnow() + PREVIEW_TTL),
            created_by=actor,
        )
        created = tx.create_definition(
            owner_id=owner_id,
            family=source.family,
            name=copy_name,
            description=copy_description,
            lifecycle="paused",
            health=DEFAULT_DEFINITION_HEALTH,
            schedule=source.schedule,
            input=source.input,
            visibility_policy=source.visibility_policy,
            notification_policy=source.notification_policy,
            approval_policy=source.approval_policy,
            preview_id=preview.id,
            created_by=actor,
            updated_by=actor,
            disabled_lock_kind="none",
            disabled_reason=None,
            finding_policy=source.finding_policy,
            retention_policy=source.retention_policy,
        )
        response = self._definition_response(created)
        source_response = self._definition_response(source)
        self._create_audit(
            tx=tx,
            owner_id=owner_id,
            definition_id=source.id,
            event_type="definition_duplicated",
            actor=actor,
            summary="Duplicated definition",
            before=source_response.model_dump(mode="json"),
            after={"duplicate_definition_id": created.id, "name": copy_name},
            idempotency_key=idempotency_key,
            request_id=request_id,
        )
        self._create_audit(
            tx=tx,
            owner_id=owner_id,
            definition_id=created.id,
            event_type="definition_duplicate_created",
            actor=actor,
            summary="Created duplicate definition",
            before=None,
            after={**response.model_dump(mode="json"), "source_definition_id": source.id},
            idempotency_key=idempotency_key,
            request_id=request_id,
        )
        return response

    def _with_idempotency(
        self,
        *,
        owner_id: int,
        route: str,
        key: str | None,
        payload_hash: str,
        operation: Callable[[ScheduledTasksTransaction], BaseModel],
    ) -> BaseModel:
        repo = self._repo(owner_id)

        def _run(tx: ScheduledTasksTransaction) -> BaseModel:
            if key is None:
                return operation(tx)
            existing = tx.get_idempotency_record(owner_id=owner_id, route=route, key=key)
            if existing is not None:
                if existing.payload_hash != payload_hash:
                    raise ScheduledTaskAutomationError("scheduled_task_idempotency_conflict")
                return self._load_response_ref(owner_id=owner_id, response_ref=existing.response_ref)
            response = operation(tx)
            tx.create_idempotency_record(
                owner_id=owner_id,
                route=route,
                key=key,
                payload_hash=payload_hash,
                response_ref=self._response_ref(response),
                expires_at=_iso(_utcnow() + IDEMPOTENCY_TTL),
            )
            return response

        return repo.write_transaction(_run)

    def _normalize_preview(
        self,
        request: ScheduledTaskPreviewCreateRequest,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
        base = self._preview_hash_payload(request)["config"]
        schedule, schedule_errors, schedule_warnings = _validate_schedule(base["schedule"])
        base["schedule"] = schedule
        base["visibility_policy"] = _normalize_visibility_policy(request.family, base["visibility_policy"])
        mode_errors: list[dict[str, Any]] = []
        if request.mode == "update":
            if not request.definition_id:
                mode_errors.append(
                    _field_error(
                        "definition_id",
                        "required_for_update",
                        "Definition id is required for update previews.",
                    )
                )
            if request.definition_version is None:
                mode_errors.append(
                    _field_error(
                        "definition_version",
                        "required_for_update",
                        "Definition version is required for update previews.",
                    )
                )
        else:
            if request.definition_id is not None:
                mode_errors.append(
                    _field_error(
                        "definition_id",
                        "not_allowed_for_create",
                        "Definition id is not allowed for create previews.",
                    )
                )
            if request.definition_version is not None:
                mode_errors.append(
                    _field_error(
                        "definition_version",
                        "not_allowed_for_create",
                        "Definition version is not allowed for create previews.",
                    )
                )
        if request.family == "recurring_question":
            normalized, errors, warnings = _validate_recurring_question_config(base)
        else:
            normalized, errors, warnings = _validate_agent_task_config(base)
        normalized["family"] = request.family
        normalized["schedule"] = schedule
        normalized["visibility_policy"] = base["visibility_policy"]
        return normalized, [*mode_errors, *errors, *schedule_errors], [*warnings, *schedule_warnings]

    def _preview_hash_payload(self, request: ScheduledTaskPreviewCreateRequest) -> dict[str, Any]:
        payload = request.model_dump(mode="json")
        config = {
            "name": payload.get("name"),
            "description": payload.get("description"),
            "config": payload.get("config") or {},
            "input": payload.get("input") or {},
            "schedule": payload.get("schedule") or {},
            "visibility_policy": payload.get("visibility_policy") or {},
            "notification_policy": payload.get("notification_policy") or {},
            "approval_policy": payload.get("approval_policy") or {},
        }
        return {
            "mode": payload["mode"],
            "family": payload["family"],
            "definition_id": payload.get("definition_id"),
            "definition_version": payload.get("definition_version"),
            "config": config,
        }

    def _require_valid_preview(
        self,
        *,
        tx: ScheduledTasksTransaction | None = None,
        owner_id: int,
        preview_id: str,
    ) -> PreviewRow:
        preview = (
            tx.get_preview(owner_id=owner_id, preview_id=preview_id)
            if tx is not None
            else self._repo(owner_id).get_preview(owner_id=owner_id, preview_id=preview_id)
        )
        if preview is None:
            raise ScheduledTaskAutomationError("preview_not_found")
        if preview.status == "consumed":
            raise ScheduledTaskAutomationError("preview_consumed")
        if preview.status == "expired" or _parse_iso(preview.expires_at) <= _utcnow():
            raise ScheduledTaskAutomationError("preview_expired")
        if preview.status != "valid":
            raise ScheduledTaskAutomationError("preview_invalid")
        return preview

    def _get_definition_row(
        self,
        *,
        tx: ScheduledTasksTransaction | None = None,
        owner_id: int,
        definition_id: str,
    ) -> DefinitionRow:
        definition = (
            tx.get_definition(owner_id=owner_id, definition_id=definition_id)
            if tx is not None
            else self._repo(owner_id).get_definition(owner_id=owner_id, definition_id=definition_id)
        )
        if definition is None:
            raise ScheduledTaskAutomationError("definition_not_found")
        return definition

    def _create_audit(
        self,
        *,
        tx: ScheduledTasksTransaction | None = None,
        owner_id: int,
        definition_id: str,
        event_type: str,
        actor: str,
        summary: str,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
        idempotency_key: str | None,
        request_id: str | None = None,
    ) -> None:
        target = tx if tx is not None else self._repo(owner_id)
        target.create_audit_event(
            owner_id=owner_id,
            definition_id=definition_id,
            event_type=event_type,
            actor=actor,
            summary=summary,
            before=before,
            after=after,
            request_id=request_id,
            idempotency_key=idempotency_key,
        )

    def _load_response_ref(self, *, owner_id: int, response_ref: dict[str, Any]) -> BaseModel:
        snapshot = response_ref.get("snapshot")
        if isinstance(snapshot, dict):
            if response_ref.get("type") == "preview":
                return ScheduledTaskPreviewResponse.model_validate(snapshot)
            if response_ref.get("type") == "definition":
                return ScheduledTaskDefinitionResponse.model_validate(snapshot)
            if response_ref.get("type") == "run_now":
                return ScheduledTaskRunNowResponse.model_validate(snapshot)
            if response_ref.get("type") == "run":
                return ScheduledTaskRunResponse.model_validate(snapshot)
            if response_ref.get("type") == "result":
                return ScheduledTaskResultResponse.model_validate(snapshot)
        if response_ref.get("type") == "preview":
            return self.get_preview(owner_id=owner_id, preview_id=str(response_ref["id"]))
        if response_ref.get("type") == "definition":
            return self.get_definition(owner_id=owner_id, definition_id=str(response_ref["id"]))
        raise ScheduledTaskAutomationError("scheduled_task_idempotency_response_unavailable")

    @staticmethod
    def _response_ref(response: BaseModel) -> dict[str, Any]:
        if isinstance(response, ScheduledTaskPreviewResponse):
            return {
                "type": "preview",
                "id": response.id,
                "snapshot": response.model_dump(mode="json"),
            }
        if isinstance(response, ScheduledTaskDefinitionResponse):
            return {
                "type": "definition",
                "id": response.id,
                "snapshot": response.model_dump(mode="json"),
            }
        if isinstance(response, ScheduledTaskRunNowResponse):
            return {
                "type": "run_now",
                "id": response.definition_id,
                "snapshot": response.model_dump(mode="json"),
            }
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
        raise TypeError(f"unsupported idempotency response type: {type(response)!r}")

    def _preview_response(self, row: PreviewRow) -> ScheduledTaskPreviewResponse:
        return ScheduledTaskPreviewResponse(
            id=row.id,
            owner_id=str(row.owner_id),
            mode=row.mode,
            family=row.family,
            definition_id=row.definition_id,
            definition_version=row.definition_version,
            status=row.status,
            payload_hash=row.payload_hash,
            normalized_config=row.normalized_config,
            validation_errors=row.validation_errors,
            warnings=row.warnings,
            risk_class=row.risk_class,
            visibility_policy={"mode": row.visibility_policy},
            schedule_preview=row.schedule_preview,
            redaction_policy=row.redaction_policy,
            expires_at=row.expires_at,
            created_by=row.created_by,
            created_at=row.created_at,
            consumed_at=row.consumed_at,
            created_definition_id=row.created_definition_id,
        )

    def _definition_response(
        self,
        row: DefinitionRow,
        *,
        preview: PreviewRow | None = None,
        lookup_preview: bool = True,
    ) -> ScheduledTaskDefinitionResponse:
        if lookup_preview:
            preview = self._repo(row.owner_id).get_preview(owner_id=row.owner_id, preview_id=row.preview_id)
        config = preview.normalized_config.get("config", {}) if preview is not None else {}
        return ScheduledTaskDefinitionResponse(
            id=row.id,
            owner_id=str(row.owner_id),
            version=row.version,
            family=row.family,
            name=row.name,
            description=row.description,
            lifecycle=row.lifecycle,
            health=row.health,
            disabled_lock_kind=row.disabled_lock_kind,
            disabled_reason=row.disabled_reason,
            schedule=row.schedule,
            input=row.input,
            config=config,
            visibility_policy={"mode": row.visibility_policy},
            notification_policy=row.notification_policy,
            approval_policy=row.approval_policy,
            preview_id=row.preview_id,
            created_by=row.created_by,
            updated_by=row.updated_by,
            created_at=row.created_at,
            updated_at=row.updated_at,
            archived_at=row.updated_at if row.lifecycle == "archived" else None,
            resolution_state=row.resolution_state,
            resolved_at=row.resolved_at,
            resolved_by=row.resolved_by,
            resolved_result_id=row.resolved_result_id,
            finding_policy=row.finding_policy,
            retention_policy=row.retention_policy,
        )

    @staticmethod
    def _audit_response(row: AuditEventRow) -> ScheduledTaskAuditEventResponse:
        return ScheduledTaskAuditEventResponse(
            id=row.id,
            definition_id=row.definition_id,
            event_type=row.event_type,
            actor=row.actor,
            summary=row.summary,
            before=row.before,
            after=row.after,
            created_at=row.created_at,
            request_id=row.request_id,
            idempotency_key=row.idempotency_key,
        )

    @staticmethod
    def _definition_actions() -> dict[str, ScheduledTaskActionCapability]:
        return {
            "preview": ScheduledTaskActionCapability(
                status="available",
                required_permissions=[TASKS_CONTROL],
            ),
            "create_definition": ScheduledTaskActionCapability(
                status="available",
                required_permissions=[TASKS_CONTROL],
            ),
            "update_definition": ScheduledTaskActionCapability(
                status="available",
                required_permissions=[TASKS_CONTROL],
            ),
            "pause": ScheduledTaskActionCapability(
                status="available",
                required_permissions=[TASKS_CONTROL],
            ),
            "resume": ScheduledTaskActionCapability(
                status="available",
                required_permissions=[TASKS_CONTROL],
            ),
            "archive": ScheduledTaskActionCapability(
                status="available",
                required_permissions=[TASKS_CONTROL],
            ),
            "duplicate": ScheduledTaskActionCapability(
                status="available",
                required_permissions=[TASKS_CONTROL],
            ),
            "execute": ScheduledTaskActionCapability(
                status="unavailable",
                reason="execution_not_implemented",
                required_permissions=[TASKS_CONTROL],
            ),
        }

    @classmethod
    def _recurring_question_actions(cls) -> dict[str, ScheduledTaskActionCapability]:
        actions = cls._definition_actions()
        actions.update(
            {
                "create_run_manual": ScheduledTaskActionCapability(
                    status="available",
                    required_permissions=[TASKS_CONTROL],
                ),
                "execute_scheduled": ScheduledTaskActionCapability(
                    status="available",
                    required_permissions=[TASKS_CONTROL],
                ),
                "read_runs": ScheduledTaskActionCapability(status="available"),
                "read_results": ScheduledTaskActionCapability(status="available"),
                "mutate_results": ScheduledTaskActionCapability(
                    status="available",
                    required_permissions=[TASKS_CONTROL],
                ),
                "mark_solved": ScheduledTaskActionCapability(
                    status="available",
                    required_permissions=[TASKS_CONTROL],
                ),
                "reopen": ScheduledTaskActionCapability(
                    status="available",
                    required_permissions=[TASKS_CONTROL],
                ),
            }
        )
        return actions
