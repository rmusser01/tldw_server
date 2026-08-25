from __future__ import annotations

import asyncio
from dataclasses import asdict

import pytest

from tldw_Server_API.app.core.Admin_Webhooks import audit
from tldw_Server_API.app.core.Admin_Webhooks.audit import (
    AUDIT_STOP_TIMEOUT_SECONDS,
    AUDIT_WRITE_TIMEOUT_SECONDS,
    MutationAudit,
    OperationalAudit,
    WebhookOperationalReasonCode,
    emit_mandatory_webhook_audit,
    emit_mandatory_webhook_operation_audit,
    validate_actor_kind,
    validate_actor_principal_id,
    validate_actor_roles,
)
from tldw_Server_API.app.core.Admin_Webhooks.domain import WebhookErrorCode
from tldw_Server_API.app.core.Audit.unified_audit_service import (
    AuditEventCategory,
    AuditEventType,
    MandatoryAuditWriteError,
)

pytestmark = pytest.mark.asyncio


class RecordingAuditService:
    def __init__(
        self,
        *,
        fail_log: bool = False,
        fail_flush: bool = False,
        block_log: bool = False,
        block_stop: bool = False,
    ) -> None:
        self.fail_log = fail_log
        self.fail_flush = fail_flush
        self.block_log = block_log
        self.block_stop = block_stop
        self.events: list[dict[str, object]] = []
        self.flush_calls: list[bool] = []
        self.stop_calls = 0

    async def log_event(self, **kwargs: object) -> None:
        if self.block_log:
            await asyncio.Event().wait()
        if self.fail_log:
            raise RuntimeError("sensitive adapter failure")
        self.events.append(kwargs)

    async def flush(self, *, raise_on_failure: bool = False) -> None:
        self.flush_calls.append(raise_on_failure)
        if self.fail_flush:
            raise RuntimeError("sensitive flush failure")

    async def stop(self) -> None:
        self.stop_calls += 1
        if self.block_stop:
            await asyncio.Event().wait()


def _mutation_record(**overrides: object) -> MutationAudit:
    values: dict[str, object] = {
        "actor_id": 7,
        "action": "admin_webhook.create",
        "webhook_id": 41,
        "target_hostname": "hooks.example.com",
        "event_types": ("user.created",),
        "outcome": "accepted",
        "request_id": "request-0123456789",
        "reason_code": None,
    }
    values.update(overrides)
    return MutationAudit(**values)  # type: ignore[arg-type]


def _operation_record(**overrides: object) -> OperationalAudit:
    values: dict[str, object] = {
        "operator_id": 7,
        "action": "admin_webhook.key_rotation.start",
        "operation_id": "rotation-0123456789",
        "outcome": "accepted",
        "request_id": "request-0123456789",
        "reason_code": None,
    }
    values.update(overrides)
    return OperationalAudit(**values)  # type: ignore[arg-type]


@pytest.mark.unit
async def test_mutation_audit_writes_flushes_and_stops_isolated_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = RecordingAuditService()
    observed_user_ids: list[int] = []

    async def create(user_id: int) -> RecordingAuditService:
        observed_user_ids.append(user_id)
        return service

    monkeypatch.setattr(audit, "_create_isolated_audit_service", create)
    record = _mutation_record()

    await emit_mandatory_webhook_audit(
        record,
        actor_principal_id="api_key:0123456789abcdef",
        actor_kind="api_key",
        actor_roles=("admin", "platform_admin"),
    )

    assert observed_user_ids == [7]
    assert service.flush_calls == [True]
    assert service.stop_calls == 1
    assert len(service.events) == 1
    event = service.events[0]
    assert event["event_type"] is AuditEventType.DATA_WRITE
    assert event["category"] is AuditEventCategory.DATA_MODIFICATION
    assert event["resource_type"] == "admin_webhook"
    assert event["resource_id"] == "41"
    assert event["action"] == "admin_webhook.create"
    assert event["metadata"] == {
        "actor_principal_id": "api_key:0123456789abcdef",
        "actor_kind": "api_key",
        "actor_roles": ["admin", "platform_admin"],
        "target_hostname": "hooks.example.com",
        "event_types": ["user.created"],
        "outcome": "accepted",
        "request_id": "request-0123456789",
        "reason_code": None,
    }


@pytest.mark.parametrize(
    ("action", "event_type", "category"),
    [
        ("admin_webhook.create", AuditEventType.DATA_WRITE, AuditEventCategory.DATA_MODIFICATION),
        ("admin_webhook.patch", AuditEventType.DATA_UPDATE, AuditEventCategory.DATA_MODIFICATION),
        ("admin_webhook.delete", AuditEventType.DATA_DELETE, AuditEventCategory.DATA_MODIFICATION),
        ("admin_webhook.rotate_secret", AuditEventType.DATA_UPDATE, AuditEventCategory.SECURITY),
    ],
)
@pytest.mark.unit
async def test_mutation_actions_have_closed_event_mapping(
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    event_type: AuditEventType,
    category: AuditEventCategory,
) -> None:
    service = RecordingAuditService()

    async def create(_user_id: int) -> RecordingAuditService:
        return service

    monkeypatch.setattr(audit, "_create_isolated_audit_service", create)

    await emit_mandatory_webhook_audit(
        _mutation_record(action=action),
        actor_principal_id="user:0123456789abcdef",
        actor_kind="user",
        actor_roles=("platform_admin",),
    )

    assert service.events[0]["event_type"] is event_type
    assert service.events[0]["category"] is category


@pytest.mark.parametrize("failure", ["create", "log", "flush"])
@pytest.mark.unit
async def test_every_mandatory_adapter_failure_is_sanitized_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    service = RecordingAuditService(
        fail_log=failure == "log",
        fail_flush=failure == "flush",
    )

    async def create(_user_id: int) -> RecordingAuditService:
        if failure == "create":
            raise RuntimeError("sensitive create failure")
        return service

    monkeypatch.setattr(audit, "_create_isolated_audit_service", create)

    with pytest.raises(MandatoryAuditWriteError) as exc_info:
        await emit_mandatory_webhook_audit(
            _mutation_record(),
            actor_principal_id="user:0123456789abcdef",
            actor_kind="user",
            actor_roles=("platform_admin",),
        )

    assert str(exc_info.value) == "Mandatory audit persistence unavailable"
    assert "sensitive" not in str(exc_info.value)
    assert service.stop_calls == (0 if failure == "create" else 1)


@pytest.mark.unit
async def test_write_and_stop_are_independently_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = RecordingAuditService(block_log=True, block_stop=True)

    async def create(_user_id: int) -> RecordingAuditService:
        return service

    monkeypatch.setattr(audit, "_create_isolated_audit_service", create)
    monkeypatch.setattr(audit, "AUDIT_WRITE_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(audit, "AUDIT_STOP_TIMEOUT_SECONDS", 0.01)

    with pytest.raises(MandatoryAuditWriteError):
        await asyncio.wait_for(
            emit_mandatory_webhook_audit(
                _mutation_record(),
                actor_principal_id="user:0123456789abcdef",
                actor_kind="user",
                actor_roles=("platform_admin",),
            ),
            timeout=0.2,
        )

    assert service.stop_calls == 1
    assert AUDIT_WRITE_TIMEOUT_SECONDS == 5.0
    assert AUDIT_STOP_TIMEOUT_SECONDS == 1.0


@pytest.mark.unit
async def test_operational_audit_has_only_closed_bounded_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = RecordingAuditService()

    async def create(_user_id: int) -> RecordingAuditService:
        return service

    monkeypatch.setattr(audit, "_create_isolated_audit_service", create)
    record = _operation_record(
        outcome="failed",
        reason_code=WebhookOperationalReasonCode.KEY_CONFIGURATION_MISMATCH,
    )

    await emit_mandatory_webhook_operation_audit(record)

    assert service.stop_calls == 1
    event = service.events[0]
    assert event["resource_type"] == "admin_webhook_operation"
    assert event["resource_id"] == "rotation-0123456789"
    assert event["metadata"] == {
        "operator_id": 7,
        "operation_id": "rotation-0123456789",
        "outcome": "failed",
        "request_id": "request-0123456789",
        "reason_code": "admin_webhook_key_configuration_mismatch",
    }
    serialized = repr({"record": asdict(record), "event": event})
    for forbidden in ("https://", "?token=", "whsec_", "/srv/", "payload"):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    "record",
    [
        lambda: _mutation_record(actor_id=0),
        lambda: _mutation_record(action="admin_webhook.export"),
        lambda: _mutation_record(webhook_id=-1),
        lambda: _mutation_record(target_hostname="hooks.example.com/private?token=x"),
        lambda: _mutation_record(event_types=("*",)),
        lambda: _mutation_record(outcome="succeeded"),
        lambda: _mutation_record(request_id="bad request"),
        lambda: _mutation_record(reason_code="caller text"),
        lambda: _operation_record(action="admin_webhook.shell"),
        lambda: _operation_record(operation_id="/srv/private/key"),
        lambda: _operation_record(outcome="denied"),
        lambda: _operation_record(reason_code="caller text"),
    ],
)
@pytest.mark.unit
async def test_audit_record_types_reject_unbounded_or_open_values(record: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        record()  # type: ignore[operator]


@pytest.mark.unit
def test_actor_identity_validators_reject_free_text_and_bound_output() -> None:
    assert validate_actor_principal_id("api_key:0123456789abcdef") == "api_key:0123456789abcdef"
    assert validate_actor_kind("api_key") == "api_key"
    assert validate_actor_kind(None) is None
    assert validate_actor_roles(("platform_admin", "admin", "admin")) == (
        "admin",
        "platform_admin",
    )

    for invalid in ("", "user name", "user/path", "x" * 129):
        with pytest.raises(ValueError):
            validate_actor_principal_id(invalid)
    for invalid in ("root", "", "api key"):
        with pytest.raises(ValueError):
            validate_actor_kind(invalid)
    for invalid in (("role with spaces",), ("x" * 65,), tuple(f"role-{i}" for i in range(33))):
        with pytest.raises(ValueError):
            validate_actor_roles(invalid)


@pytest.mark.unit
def test_reason_codes_are_closed_enums() -> None:
    mutation = _mutation_record(
        outcome="denied",
        reason_code=WebhookErrorCode.REGISTRATION_LIMIT,
    )
    operation = _operation_record(
        outcome="failed",
        reason_code=WebhookOperationalReasonCode.DATABASE_BUSY,
    )

    assert mutation.reason_code is WebhookErrorCode.REGISTRATION_LIMIT
    assert operation.reason_code is WebhookOperationalReasonCode.DATABASE_BUSY
