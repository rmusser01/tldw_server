from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.api.v1.endpoints import admin as admin_endpoints
from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops
from tldw_Server_API.app.core.Admin_Webhooks.config import (
    AdminWebhookMode,
    AdminWebhookSettings,
)
from tldw_Server_API.app.core.Admin_Webhooks.control_plane import (
    evaluate_activation_readiness,
)
from tldw_Server_API.app.core.Admin_Webhooks.crypto import WebhookKeyLoadCode
from tldw_Server_API.app.core.Admin_Webhooks.domain import (
    AdminWebhookActivationPhase,
    AdminWebhookActivationReasonCode,
    DeliveryBacklogCounts,
    DeliveryCapabilityStatus,
    DeliveryComponentStatus,
    DeliveryRuntimeComponent,
    DeliveryRuntimeReasonCode,
    WebhookLimits,
    WebhookMigrationSummary,
    WebhookStatus,
)
from tldw_Server_API.app.services import admin_system_ops_service as system_ops

NOW = datetime(2026, 9, 1, 5, 0, tzinfo=timezone.utc)


def _component(
    component: DeliveryRuntimeComponent,
    *,
    ready: bool,
) -> DeliveryComponentStatus:
    return DeliveryComponentStatus(
        component=component,
        ready=ready,
        reason_code=(
            None if ready else DeliveryRuntimeReasonCode.HEARTBEAT_STALE
        ),
        heartbeat_age_seconds=1 if ready else 31,
    )


def _status(
    *,
    mode: str,
    runtime_ready: bool = True,
    oldest_age: int | None = None,
    registrations_over_limit: bool = False,
    active_registrations_over_limit: bool = False,
) -> WebhookStatus:
    return WebhookStatus(
        mode=mode,
        route_selection="canonical",
        schema_ready=True,
        key_state=WebhookKeyLoadCode.AVAILABLE.value,
        delivery_capability_ready=runtime_ready,
        delivery=DeliveryCapabilityStatus(
            canonical_schema_version=1,
            schema_ready=True,
            delivery_schema_ready=True,
            migration_complete=True,
            key_ready=True,
            key_primary_match=True,
            jobs_database_ready=True,
            queue_ready=True,
            job_type_ready=True,
            jobs_backend="sqlite",
            worker=_component(
                DeliveryRuntimeComponent.WORKER,
                ready=runtime_ready,
            ),
            reconciler=_component(
                DeliveryRuntimeComponent.RECONCILER,
                ready=runtime_ready,
            ),
            retention=_component(
                DeliveryRuntimeComponent.RETENTION,
                ready=runtime_ready,
            ),
            backlog=DeliveryBacklogCounts(pending=1 if oldest_age is not None else 0),
            oldest_nonterminal_age_seconds=oldest_age,
            acquisition_ready=runtime_ready,
            acquisition_reason_code=(
                None
                if runtime_ready
                else DeliveryRuntimeReasonCode.RECONCILER_UNAVAILABLE
            ),
            delivery_capability_ready=runtime_ready,
        ),
        limits=WebhookLimits(
            registrations=100,
            active_registrations=25,
            current_registrations=1,
            current_active_registrations=1,
            registrations_over_limit=registrations_over_limit,
            active_registrations_over_limit=active_registrations_over_limit,
        ),
        migration=WebhookMigrationSummary(phase="complete"),
    )


@pytest.mark.unit
def test_final_admin_router_mounts_each_canonical_webhook_route_exactly_once() -> None:
    expected = {
        ("GET", "/admin/webhooks/status"),
        ("GET", "/admin/webhooks/catalog"),
        ("GET", "/admin/webhooks"),
        ("POST", "/admin/webhooks"),
        ("GET", "/admin/webhooks/{webhook_id}"),
        ("PATCH", "/admin/webhooks/{webhook_id}"),
        ("DELETE", "/admin/webhooks/{webhook_id}"),
        ("POST", "/admin/webhooks/{webhook_id}/rotate-secret"),
        ("POST", "/admin/webhooks/{webhook_id}/test"),
        ("GET", "/admin/webhooks/{webhook_id}/deliveries"),
        (
            "POST",
            "/admin/webhooks/{webhook_id}/deliveries/{delivery_id}/redeliver",
        ),
        ("POST", "/admin/incidents/{incident_id}/notify-webhooks"),
    }
    pairs = Counter(
        (method, route.path)
        for route in admin_endpoints.router.routes
        for method in (route.methods or set())
        if route.path.startswith("/admin/webhooks")
        or route.path.endswith("/notify-webhooks")
    )

    assert set(pairs) == expected
    assert set(pairs.values()) == {1}
    assert not hasattr(admin_endpoints, "_mount_admin_webhook_routes")
    assert not hasattr(admin_ops, "legacy_webhooks_router")


@pytest.mark.unit
def test_settings_are_canonical_only_and_reject_requested_legacy_mode() -> None:
    settings = AdminWebhookSettings.from_environment(
        {"TLDW_ADMIN_WEBHOOKS_LEGACY_COMPAT": "false"}
    )

    assert settings.mode is AdminWebhookMode.OFF
    assert not hasattr(settings, "route_selection")
    assert settings.activation_max_backlog_age_seconds == 300
    with pytest.raises(ValueError, match="no longer supported"):
        AdminWebhookSettings.from_environment(
            {"TLDW_ADMIN_WEBHOOKS_LEGACY_COMPAT": "true"}
        )


@pytest.mark.unit
def test_normal_system_ops_store_never_recreates_legacy_webhook_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "system_ops.json"
    monkeypatch.setattr(system_ops, "_STORE_PATH", store_path)
    store_path.write_text('{"incidents":[]}', encoding="utf-8")

    assert "webhooks" not in system_ops._default_store()
    assert "webhook_deliveries" not in system_ops._default_store()
    loaded = system_ops._load_store()
    assert "webhooks" not in loaded
    assert "webhook_deliveries" not in loaded


@pytest.mark.unit
def test_predeploy_requires_static_dependencies_but_not_live_heartbeats() -> None:
    result = evaluate_activation_readiness(
        _status(mode="migrate", runtime_ready=False, oldest_age=10_000),
        phase=AdminWebhookActivationPhase.PREDEPLOY,
        max_backlog_age_seconds=300,
    )

    assert result.ready is True
    assert result.phase is AdminWebhookActivationPhase.PREDEPLOY
    assert result.runtime_ready is False
    assert result.backlog_age_ready is False
    assert result.reason_codes == ()


@pytest.mark.unit
def test_live_requires_on_mode_fresh_runtime_and_bounded_backlog_age() -> None:
    ready = evaluate_activation_readiness(
        _status(mode="on", oldest_age=300),
        phase=AdminWebhookActivationPhase.LIVE,
        max_backlog_age_seconds=300,
    )
    failed = evaluate_activation_readiness(
        _status(
            mode="migrate",
            runtime_ready=False,
            oldest_age=301,
            registrations_over_limit=True,
            active_registrations_over_limit=True,
        ),
        phase=AdminWebhookActivationPhase.LIVE,
        max_backlog_age_seconds=300,
    )

    assert ready.ready is True
    assert ready.reason_codes == ()
    assert failed.ready is False
    assert failed.reason_codes == (
        AdminWebhookActivationReasonCode.PHASE_MISMATCH,
        AdminWebhookActivationReasonCode.REGISTRATION_LIMIT_EXCEEDED,
        AdminWebhookActivationReasonCode.ACTIVE_LIMIT_EXCEEDED,
        AdminWebhookActivationReasonCode.WORKER_UNAVAILABLE,
        AdminWebhookActivationReasonCode.RECONCILER_UNAVAILABLE,
        AdminWebhookActivationReasonCode.RETENTION_UNAVAILABLE,
        AdminWebhookActivationReasonCode.BACKLOG_AGE_EXCEEDED,
    )


@pytest.mark.unit
def test_activation_check_rejects_invalid_phase_and_backlog_bound() -> None:
    with pytest.raises(TypeError, match="phase"):
        evaluate_activation_readiness(  # type: ignore[arg-type]
            _status(mode="migrate"),
            phase="predeploy",
            max_backlog_age_seconds=300,
        )
    for value in (0, 86_401, True):
        with pytest.raises(ValueError, match="backlog"):
            evaluate_activation_readiness(
                _status(mode="migrate"),
                phase=AdminWebhookActivationPhase.PREDEPLOY,
                max_backlog_age_seconds=value,  # type: ignore[arg-type]
            )
