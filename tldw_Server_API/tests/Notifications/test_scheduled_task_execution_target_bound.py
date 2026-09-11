"""Authoring-time bound on automation execution targets (TASK-13234).

ADR-077 (chatbook task-18940) AC#7: per-task model selection rides the
definition payload, bounded to the providers the server can actually run.
The executor honors ``input.provider``/``input.model`` per run; these
tests pin the authoring-time bound in ``_bound_execution_target_findings``
and its wiring into preview / create / update.
"""

from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.api.v1.schemas.scheduled_tasks_automation_schemas import (
    ScheduledTaskDefinitionCreateRequest,
    ScheduledTaskDefinitionUpdateRequest,
)
from tldw_Server_API.app.core.DB_Management.Scheduled_Tasks_DB import (
    ScheduledTasksDatabase,
)
from tldw_Server_API.app.services import scheduled_task_automation_service as service_module
from tldw_Server_API.app.services.scheduled_task_automation_service import (
    ScheduledTaskAutomationError,
    ScheduledTaskAutomationService,
)

OWNER_ID = 4210
ACTOR = "target-bound-test"

_LISTING: dict[str, Any] = {
    "providers": [
        {"name": "openai", "models": ["gpt-4o", "gpt-4o-mini"], "is_enabled": True},
        {"name": "anthropic", "models": ["claude-3-5"], "is_enabled": True},
    ],
    "default_provider": "openai",
    "total_configured": 2,
}


def _service(tmp_path) -> tuple[ScheduledTaskAutomationService, ScheduledTasksDatabase]:
    repo = ScheduledTasksDatabase(tmp_path / "scheduled_tasks_bound.db")
    repo.ensure_schema()
    return ScheduledTaskAutomationService(repository=repo), repo


def _bound(monkeypatch: pytest.MonkeyPatch, listing: dict[str, Any] | None) -> None:
    monkeypatch.setattr(service_module, "_usable_provider_listing", lambda: listing)


def _no_admin_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service_module, "validate_provider_override", lambda p, m: None)


def _preview_payload(provider: str | None, model: str | None):
    input_payload: dict[str, Any] = {
        "question": "What changed in the selected sources?"
    }
    if provider is not None:
        input_payload["provider"] = provider
    if model is not None:
        input_payload["model"] = model
    return service_module.ScheduledTaskPreviewCreateRequest(
        mode="create",
        family="recurring_question",
        name="Daily research check",
        config={},
        input=input_payload,
        schedule={"kind": "daily", "time": "09:00", "timezone": "UTC"},
        visibility_policy={"mode": "findings_only"},
    )


def test_usable_provider_passes_clean(tmp_path, monkeypatch):
    _bound(monkeypatch, _LISTING)
    _no_admin_policy(monkeypatch)
    service, _repo = _service(tmp_path)

    preview = service.create_preview(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=_preview_payload("openai", "gpt-4o"),
    )

    assert preview.status == "valid"  # nosec B101
    assert preview.validation_errors == []  # nosec B101
    assert preview.warnings == []  # nosec B101


def test_unconfigured_provider_is_a_validation_error(tmp_path, monkeypatch):
    _bound(monkeypatch, _LISTING)
    _no_admin_policy(monkeypatch)
    service, _repo = _service(tmp_path)

    preview = service.create_preview(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=_preview_payload("not-a-provider", None),
    )

    assert preview.status == "invalid"  # nosec B101
    assert [(e["field"], e["code"]) for e in preview.validation_errors] == [
        ("input.provider", "unusable")
    ]  # nosec B101


def test_admin_disabled_provider_is_a_validation_error(tmp_path, monkeypatch):
    _bound(monkeypatch, _LISTING)
    monkeypatch.setattr(
        service_module,
        "validate_provider_override",
        lambda p, m: {"error_code": "provider_disabled"}
        if p == "openai"
        else None,
    )
    service, _repo = _service(tmp_path)

    preview = service.create_preview(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=_preview_payload("openai", None),
    )

    assert preview.status == "invalid"  # nosec B101
    assert [(e["field"], e["code"]) for e in preview.validation_errors] == [
        ("input.provider", "provider_disabled")
    ]  # nosec B101


def test_admin_model_not_allowed_is_a_validation_error(tmp_path, monkeypatch):
    _bound(monkeypatch, _LISTING)
    monkeypatch.setattr(
        service_module,
        "validate_provider_override",
        lambda p, m: {"error_code": "model_not_allowed"}
        if (p == "openai" and m == "gpt-4o")
        else None,
    )
    service, _repo = _service(tmp_path)

    preview = service.create_preview(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=_preview_payload("openai", "gpt-4o"),
    )

    assert preview.status == "invalid"  # nosec B101
    assert [(e["field"], e["code"]) for e in preview.validation_errors] == [
        ("input.model", "model_not_allowed")
    ]  # nosec B101


def test_unknown_model_is_a_warning_only(tmp_path, monkeypatch):
    _bound(monkeypatch, _LISTING)
    _no_admin_policy(monkeypatch)
    service, _repo = _service(tmp_path)

    preview = service.create_preview(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=_preview_payload("openai", "brand-new-model"),
    )

    # The preview stays valid; the drift is surfaced, not fatal.
    assert preview.status == "valid"  # nosec B101
    assert preview.validation_errors == []  # nosec B101
    assert len(preview.warnings) == 1  # nosec B101
    assert "unknown_model" in str(preview.warnings[0])  # nosec B101
    assert "brand-new-model" in str(preview.warnings[0])  # nosec B101


def test_blank_keys_keep_fallback_semantics(tmp_path, monkeypatch):
    _bound(monkeypatch, _LISTING)
    _no_admin_policy(monkeypatch)
    service, _repo = _service(tmp_path)

    preview = service.create_preview(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=_preview_payload("   ", ""),
    )

    assert preview.status == "valid"  # nosec B101
    assert preview.validation_errors == []  # nosec B101
    assert preview.warnings == []  # nosec B101


def test_listing_read_failure_does_not_brick_authoring(tmp_path, monkeypatch):
    _bound(monkeypatch, None)
    service, _repo = _service(tmp_path)

    preview = service.create_preview(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=_preview_payload("openai", "gpt-4o"),
    )

    assert preview.status == "valid"  # nosec B101
    assert preview.validation_errors == []  # nosec B101


def test_legacy_preview_with_unusable_provider_hard_fails_on_create(tmp_path, monkeypatch):
    _bound(monkeypatch, _LISTING)
    _no_admin_policy(monkeypatch)
    service, _repo = _service(tmp_path)

    # Preview authored while the provider looked usable...
    preview = service.create_preview(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=_preview_payload("openai", None),
    )
    # ...then the server config loses the provider before create consumes it.
    _bound(monkeypatch, {"providers": [], "default_provider": None, "total_configured": 0})

    with pytest.raises(ScheduledTaskAutomationError) as excinfo:
        service.create_definition(
            owner_id=OWNER_ID,
            actor=ACTOR,
            payload=ScheduledTaskDefinitionCreateRequest(preview_id=preview.id),
        )
    assert excinfo.value.code == "execution_target_unusable"  # nosec B101


def test_update_hard_fails_for_newly_unusable_target(tmp_path, monkeypatch):
    _bound(monkeypatch, _LISTING)
    _no_admin_policy(monkeypatch)
    service, _repo = _service(tmp_path)

    create_preview = service.create_preview(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=_preview_payload("openai", None),
    )
    definition = service.create_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=ScheduledTaskDefinitionCreateRequest(preview_id=create_preview.id),
    )

    update_preview = service.create_preview(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=service_module.ScheduledTaskPreviewCreateRequest(
            mode="update",
            family="recurring_question",
            definition_id=definition.id,
            definition_version=definition.version,
            name="Daily research check",
            config={},
            input={"question": "What changed?", "provider": "anthropic"},
            schedule={"kind": "daily", "time": "09:00", "timezone": "UTC"},
            visibility_policy={"mode": "findings_only"},
        ),
    )
    # Admin disables the provider between preview and update.
    monkeypatch.setattr(
        service_module,
        "validate_provider_override",
        lambda p, m: {"error_code": "provider_disabled"}
        if p == "anthropic"
        else None,
    )

    with pytest.raises(ScheduledTaskAutomationError) as excinfo:
        service.update_definition(
            owner_id=OWNER_ID,
            actor=ACTOR,
            definition_id=definition.id,
            payload=ScheduledTaskDefinitionUpdateRequest(preview_id=update_preview.id),
        )
    assert excinfo.value.code == "execution_target_unusable"  # nosec B101
