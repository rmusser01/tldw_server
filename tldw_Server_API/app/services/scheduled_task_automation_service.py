"""Service layer for Scheduled Tasks automation definition foundations."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

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
    ScheduledTaskPreviewCreateRequest,
    ScheduledTaskPreviewListResponse,
    ScheduledTaskPreviewResponse,
)
from tldw_Server_API.app.core.AuthNZ.permissions import TASKS_CONTROL
from tldw_Server_API.app.core.DB_Management.Scheduled_Tasks_DB import (
    AuditEventRow,
    DefinitionRow,
    PreviewRow,
    ScheduledTasksDatabase,
)

PREVIEW_TTL = timedelta(hours=24)
IDEMPOTENCY_TTL = timedelta(hours=24)
DEFAULT_DEFINITION_HEALTH = "execution_unavailable"
_SUPPORTED_SCHEDULE_KINDS = {"one_time", "interval", "daily", "weekly", "cron"}


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
    digest = hashlib.sha256(message.encode("utf-8")).hexdigest()
    return {
        "message_redacted": True,
        "message_ref": f"sha256:{digest}",
        "message_preview": "[redacted]",
        "message_length": len(message),
    }


def _validate_schedule(schedule: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    normalized = dict(schedule or {})
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
    input_config = dict(config.get("input") or {})
    question = str(input_config.get("question") or "").strip()

    if not name:
        errors.append(_field_error("name", "required", "Name is required."))
    if not question:
        errors.append(_field_error("input.question", "required", "Question is required."))

    normalized = dict(config)
    normalized["name"] = name
    normalized["input"] = {**input_config, "question": question}
    return normalized, errors, warnings


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

    def __init__(self, repository: ScheduledTasksDatabase | None = None):
        self._repository = repository

    def get_capabilities(self) -> ScheduledTaskAutomationCapabilitiesResponse:
        """Return Phase 4B capabilities without exposing execution support."""
        return ScheduledTaskAutomationCapabilitiesResponse(
            items=[
                ScheduledTaskAutomationCapability(
                    family="recurring_question",
                    family_availability="available",
                    actions=self._definition_actions(),
                    related_capabilities={"rag": {"status": "not_checked"}},
                ),
                ScheduledTaskAutomationCapability(
                    family="agent_task",
                    family_availability="available",
                    actions=self._definition_actions(),
                    related_capabilities={"acp": {"status": "not_checked"}},
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
        payload_hash = _canonical_hash(self._preview_hash_payload(request))
        return self._with_idempotency(
            owner_id=owner_id,
            route="scheduled_task_automation.preview",
            key=idempotency_key,
            payload_hash=payload_hash,
            operation=lambda: self._create_preview(owner_id=owner_id, actor=actor, request=request, payload_hash=payload_hash),
        )

    def get_preview(self, *, owner_id: int, preview_id: str) -> ScheduledTaskPreviewResponse:
        preview = self._repo(owner_id).get_preview(owner_id=owner_id, preview_id=preview_id)
        if preview is None:
            raise KeyError("preview_not_found")
        return self._preview_response(preview)

    def list_previews(
        self,
        *,
        owner_id: int | None = None,
        limit: int,
        offset: int,
    ) -> ScheduledTaskPreviewListResponse:
        """Return a page of owner-scoped previews."""
        if owner_id is None:
            return ScheduledTaskPreviewListResponse(limit=limit, offset=offset)
        rows, total = self._repo(owner_id).list_previews(owner_id=owner_id, limit=limit, offset=offset)
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
    ) -> ScheduledTaskDefinitionResponse:
        request = (
            payload
            if isinstance(payload, ScheduledTaskDefinitionCreateRequest)
            else ScheduledTaskDefinitionCreateRequest(**payload)
        )
        payload_hash = _canonical_hash(request.model_dump(mode="json"))
        return self._with_idempotency(
            owner_id=owner_id,
            route="scheduled_task_automation.definition.create",
            key=idempotency_key,
            payload_hash=payload_hash,
            operation=lambda: self._create_definition(owner_id=owner_id, actor=actor, request=request, idempotency_key=idempotency_key),
        )

    def update_definition(
        self,
        *,
        owner_id: int,
        actor: str,
        definition_id: str,
        payload: ScheduledTaskDefinitionUpdateRequest | dict[str, Any],
        idempotency_key: str | None = None,
    ) -> ScheduledTaskDefinitionResponse:
        request = (
            payload
            if isinstance(payload, ScheduledTaskDefinitionUpdateRequest)
            else ScheduledTaskDefinitionUpdateRequest(**payload)
        )
        payload_hash = _canonical_hash({"definition_id": definition_id, **request.model_dump(mode="json")})
        return self._with_idempotency(
            owner_id=owner_id,
            route="scheduled_task_automation.definition.update",
            key=idempotency_key,
            payload_hash=payload_hash,
            operation=lambda: self._update_definition(
                owner_id=owner_id,
                actor=actor,
                definition_id=definition_id,
                request=request,
                idempotency_key=idempotency_key,
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
    ) -> ScheduledTaskDefinitionListResponse:
        """Return a page of owner-scoped automation definitions."""
        if owner_id is None:
            return ScheduledTaskDefinitionListResponse(limit=limit, offset=offset)
        rows, total = self._repo(owner_id).list_definitions(owner_id=owner_id, limit=limit, offset=offset)
        return ScheduledTaskDefinitionListResponse(
            items=[self._definition_response(row) for row in rows],
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
    ) -> ScheduledTaskDefinitionResponse:
        payload_hash = _canonical_hash({"definition_id": definition_id, "action": "pause"})
        return self._with_idempotency(
            owner_id=owner_id,
            route="scheduled_task_automation.definition.pause",
            key=idempotency_key,
            payload_hash=payload_hash,
            operation=lambda: self._transition_definition(
                owner_id=owner_id,
                actor=actor,
                definition_id=definition_id,
                target_lifecycle="paused",
                idempotent_lifecycle="paused",
                event_type="definition.paused",
                summary="Paused definition",
                idempotency_key=idempotency_key,
            ),
        )

    def resume_definition(
        self,
        *,
        owner_id: int,
        actor: str,
        definition_id: str,
        idempotency_key: str | None = None,
    ) -> ScheduledTaskDefinitionResponse:
        payload_hash = _canonical_hash({"definition_id": definition_id, "action": "resume"})
        return self._with_idempotency(
            owner_id=owner_id,
            route="scheduled_task_automation.definition.resume",
            key=idempotency_key,
            payload_hash=payload_hash,
            operation=lambda: self._transition_definition(
                owner_id=owner_id,
                actor=actor,
                definition_id=definition_id,
                target_lifecycle="configured",
                idempotent_lifecycle="configured",
                event_type="definition.resumed",
                summary="Resumed definition",
                idempotency_key=idempotency_key,
            ),
        )

    def archive_definition(
        self,
        *,
        owner_id: int,
        actor: str,
        definition_id: str,
        idempotency_key: str | None = None,
    ) -> ScheduledTaskDefinitionResponse:
        payload_hash = _canonical_hash({"definition_id": definition_id, "action": "archive"})
        return self._with_idempotency(
            owner_id=owner_id,
            route="scheduled_task_automation.definition.archive",
            key=idempotency_key,
            payload_hash=payload_hash,
            operation=lambda: self._archive_definition(
                owner_id=owner_id,
                actor=actor,
                definition_id=definition_id,
                idempotency_key=idempotency_key,
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
    ) -> ScheduledTaskDefinitionResponse:
        request = payload if isinstance(payload, ScheduledTaskDuplicateRequest) else ScheduledTaskDuplicateRequest(**payload)
        payload_hash = _canonical_hash({"definition_id": definition_id, **request.model_dump(mode="json")})
        return self._with_idempotency(
            owner_id=owner_id,
            route="scheduled_task_automation.definition.duplicate",
            key=idempotency_key,
            payload_hash=payload_hash,
            operation=lambda: self._duplicate_definition(
                owner_id=owner_id,
                actor=actor,
                definition_id=definition_id,
                request=request,
                idempotency_key=idempotency_key,
            ),
        )

    def list_audit_events(
        self,
        *,
        owner_id: int | None = None,
        definition_id: str | None = None,
        limit: int,
        offset: int,
    ) -> ScheduledTaskAuditListResponse:
        """Return a page of owner-scoped definition audit events."""
        if owner_id is None or definition_id is None:
            return ScheduledTaskAuditListResponse(limit=limit, offset=offset)
        rows, total = self._repo(owner_id).list_audit_events(
            owner_id=owner_id,
            definition_id=definition_id,
            limit=limit,
            offset=offset,
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
        repo.ensure_schema()
        return repo

    def _create_preview(
        self,
        *,
        owner_id: int,
        actor: str,
        request: ScheduledTaskPreviewCreateRequest,
        payload_hash: str,
    ) -> ScheduledTaskPreviewResponse:
        normalized, validation_errors, warnings = self._normalize_preview(request)
        status = "invalid" if validation_errors else "valid"
        row = self._repo(owner_id).create_preview(
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
        owner_id: int,
        actor: str,
        request: ScheduledTaskDefinitionCreateRequest,
        idempotency_key: str | None,
    ) -> ScheduledTaskDefinitionResponse:
        preview = self._require_valid_preview(owner_id=owner_id, preview_id=request.preview_id)
        if preview.mode != "create" or preview.definition_id is not None:
            raise ValueError("preview_mode_mismatch")
        normalized = preview.normalized_config
        definition = self._repo(owner_id).create_definition(
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
        )
        response = self._definition_response(definition)
        self._create_audit(
            owner_id=owner_id,
            definition_id=definition.id,
            event_type="definition.created",
            actor=actor,
            summary="Created definition",
            before=None,
            after=response.model_dump(mode="json"),
            idempotency_key=idempotency_key,
        )
        return response

    def _update_definition(
        self,
        *,
        owner_id: int,
        actor: str,
        definition_id: str,
        request: ScheduledTaskDefinitionUpdateRequest,
        idempotency_key: str | None,
    ) -> ScheduledTaskDefinitionResponse:
        current = self._get_definition_row(owner_id=owner_id, definition_id=definition_id)
        if current.lifecycle == "archived":
            raise ValueError("definition_archived")
        preview = self._require_valid_preview(owner_id=owner_id, preview_id=request.preview_id)
        if preview.mode != "update" or preview.definition_id != definition_id:
            raise ValueError("preview_definition_mismatch")
        if preview.definition_version != current.version:
            raise ValueError("definition_version_mismatch")
        normalized = preview.normalized_config
        updated = self._repo(owner_id).update_definition(
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
            },
            expected_version=current.version,
        )
        self._repo(owner_id).mark_preview_consumed(
            owner_id=owner_id,
            preview_id=preview.id,
            created_definition_id=definition_id,
        )
        response = self._definition_response(updated)
        self._create_audit(
            owner_id=owner_id,
            definition_id=definition_id,
            event_type="definition.updated",
            actor=actor,
            summary="Updated definition",
            before=self._definition_response(current).model_dump(mode="json"),
            after=response.model_dump(mode="json"),
            idempotency_key=idempotency_key,
        )
        return response

    def _transition_definition(
        self,
        *,
        owner_id: int,
        actor: str,
        definition_id: str,
        target_lifecycle: str,
        idempotent_lifecycle: str,
        event_type: str,
        summary: str,
        idempotency_key: str | None,
    ) -> ScheduledTaskDefinitionResponse:
        current = self._get_definition_row(owner_id=owner_id, definition_id=definition_id)
        if current.lifecycle == "archived":
            raise ValueError("definition_archived")
        if current.lifecycle == idempotent_lifecycle:
            return self._definition_response(current)
        if current.lifecycle == "disabled":
            raise ValueError("definition_disabled")
        updated = self._repo(owner_id).update_definition(
            owner_id=owner_id,
            definition_id=definition_id,
            patch={"lifecycle": target_lifecycle, "updated_by": actor},
            expected_version=current.version,
        )
        response = self._definition_response(updated)
        self._create_audit(
            owner_id=owner_id,
            definition_id=definition_id,
            event_type=event_type,
            actor=actor,
            summary=summary,
            before=self._definition_response(current).model_dump(mode="json"),
            after=response.model_dump(mode="json"),
            idempotency_key=idempotency_key,
        )
        return response

    def _archive_definition(
        self,
        *,
        owner_id: int,
        actor: str,
        definition_id: str,
        idempotency_key: str | None,
    ) -> ScheduledTaskDefinitionResponse:
        current = self._get_definition_row(owner_id=owner_id, definition_id=definition_id)
        if current.lifecycle == "archived":
            return self._definition_response(current)
        updated = self._repo(owner_id).update_definition(
            owner_id=owner_id,
            definition_id=definition_id,
            patch={"lifecycle": "archived", "updated_by": actor},
            expected_version=current.version,
        )
        response = self._definition_response(updated)
        self._create_audit(
            owner_id=owner_id,
            definition_id=definition_id,
            event_type="definition.archived",
            actor=actor,
            summary="Archived definition",
            before=self._definition_response(current).model_dump(mode="json"),
            after=response.model_dump(mode="json"),
            idempotency_key=idempotency_key,
        )
        return response

    def _duplicate_definition(
        self,
        *,
        owner_id: int,
        actor: str,
        definition_id: str,
        request: ScheduledTaskDuplicateRequest,
        idempotency_key: str | None,
    ) -> ScheduledTaskDefinitionResponse:
        source = self._get_definition_row(owner_id=owner_id, definition_id=definition_id)
        if source.lifecycle == "archived":
            raise ValueError("definition_archived")
        if source.lifecycle == "disabled" and source.disabled_lock_kind in {"admin", "security"}:
            raise ValueError("definition_disabled_locked")
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
        preview = self._repo(owner_id).create_preview(
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
        created = self._repo(owner_id).create_definition(
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
        )
        response = self._definition_response(created)
        source_response = self._definition_response(source)
        self._create_audit(
            owner_id=owner_id,
            definition_id=source.id,
            event_type="definition_duplicated",
            actor=actor,
            summary="Duplicated definition",
            before=source_response.model_dump(mode="json"),
            after={"duplicate_definition_id": created.id, "name": copy_name},
            idempotency_key=idempotency_key,
        )
        self._create_audit(
            owner_id=owner_id,
            definition_id=created.id,
            event_type="definition_duplicate_created",
            actor=actor,
            summary="Created duplicate definition",
            before=None,
            after={**response.model_dump(mode="json"), "source_definition_id": source.id},
            idempotency_key=idempotency_key,
        )
        return response

    def _with_idempotency(
        self,
        *,
        owner_id: int,
        route: str,
        key: str | None,
        payload_hash: str,
        operation: Callable[[], BaseModel],
    ) -> BaseModel:
        if key is None:
            return operation()
        repo = self._repo(owner_id)
        existing = repo.get_idempotency_record(owner_id=owner_id, route=route, key=key)
        if existing is not None:
            if existing.payload_hash != payload_hash:
                raise ValueError("scheduled_task_idempotency_conflict")
            return self._load_response_ref(owner_id=owner_id, response_ref=existing.response_ref)
        response = operation()
        repo.create_idempotency_record(
            owner_id=owner_id,
            route=route,
            key=key,
            payload_hash=payload_hash,
            response_ref=self._response_ref(response),
            expires_at=_iso(_utcnow() + IDEMPOTENCY_TTL),
        )
        return response

    def _normalize_preview(
        self,
        request: ScheduledTaskPreviewCreateRequest,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
        base = self._preview_hash_payload(request)["config"]
        schedule, schedule_errors, schedule_warnings = _validate_schedule(base["schedule"])
        base["schedule"] = schedule
        base["visibility_policy"] = _normalize_visibility_policy(request.family, base["visibility_policy"])
        if request.family == "recurring_question":
            normalized, errors, warnings = _validate_recurring_question_config(base)
        else:
            normalized, errors, warnings = _validate_agent_task_config(base)
        normalized["family"] = request.family
        normalized["schedule"] = schedule
        normalized["visibility_policy"] = base["visibility_policy"]
        return normalized, [*errors, *schedule_errors], [*warnings, *schedule_warnings]

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

    def _require_valid_preview(self, *, owner_id: int, preview_id: str) -> PreviewRow:
        preview = self._repo(owner_id).get_preview(owner_id=owner_id, preview_id=preview_id)
        if preview is None:
            raise KeyError("preview_not_found")
        if preview.status == "consumed":
            raise ValueError("preview_consumed")
        if preview.status == "expired" or _parse_iso(preview.expires_at) <= _utcnow():
            raise ValueError("preview_expired")
        if preview.status != "valid":
            raise ValueError("preview_invalid")
        return preview

    def _get_definition_row(self, *, owner_id: int, definition_id: str) -> DefinitionRow:
        definition = self._repo(owner_id).get_definition(owner_id=owner_id, definition_id=definition_id)
        if definition is None:
            raise KeyError("definition_not_found")
        return definition

    def _create_audit(
        self,
        *,
        owner_id: int,
        definition_id: str,
        event_type: str,
        actor: str,
        summary: str,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
        idempotency_key: str | None,
    ) -> None:
        self._repo(owner_id).create_audit_event(
            owner_id=owner_id,
            definition_id=definition_id,
            event_type=event_type,
            actor=actor,
            summary=summary,
            before=before,
            after=after,
            request_id=None,
            idempotency_key=idempotency_key,
        )

    def _load_response_ref(self, *, owner_id: int, response_ref: dict[str, Any]) -> BaseModel:
        snapshot = response_ref.get("snapshot")
        if isinstance(snapshot, dict):
            if response_ref.get("type") == "preview":
                return ScheduledTaskPreviewResponse.model_validate(snapshot)
            if response_ref.get("type") == "definition":
                return ScheduledTaskDefinitionResponse.model_validate(snapshot)
        if response_ref.get("type") == "preview":
            return self.get_preview(owner_id=owner_id, preview_id=str(response_ref["id"]))
        if response_ref.get("type") == "definition":
            return self.get_definition(owner_id=owner_id, definition_id=str(response_ref["id"]))
        raise ValueError("scheduled_task_idempotency_response_unavailable")

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

    def _definition_response(self, row: DefinitionRow) -> ScheduledTaskDefinitionResponse:
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
