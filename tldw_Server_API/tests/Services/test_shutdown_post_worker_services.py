from __future__ import annotations

import inspect
from dataclasses import fields
from types import SimpleNamespace
from typing import Any

import pytest

pytestmark = pytest.mark.unit


def _base_shutdown_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "jobs_notifications_bridge_task": None,
        "jobs_metrics_task": None,
        "jobs_metrics_stop_event": None,
        "loop_lag_task": None,
        "loop_lag_stop_event": None,
        "jobs_metrics_reconcile_task": None,
        "jobs_metrics_reconcile_stop": None,
        "jobs_crypto_rotate_task": None,
        "jobs_crypto_rotate_stop_event": None,
        "jobs_integrity_task": None,
        "jobs_integrity_stop_event": None,
        "jobs_webhooks_task": None,
        "jobs_webhooks_stop_event": None,
        "meetings_webhook_dlq_task": None,
        "meetings_webhook_dlq_stop_event": None,
        "workflows_dlq_task": None,
        "workflows_dlq_stop_event": None,
        "workflows_gc_task": None,
        "workflows_gc_stop_event": None,
        "workflows_maint_task": None,
        "workflows_maint_stop_event": None,
        "guard_exceptions": (RuntimeError,),
    }
    kwargs.update(overrides)
    return kwargs


def _patch_post_worker_helpers(
    monkeypatch: pytest.MonkeyPatch,
    shutdown_services: Any,
    calls: list[tuple[str, dict[str, object]]] | None = None,
) -> dict[str, dict[str, object]]:
    recorded: dict[str, dict[str, object]] = {}

    def _record(name: str, kwargs: dict[str, object]) -> None:
        recorded[name] = kwargs
        if calls is not None:
            calls.append((name, kwargs))

    async def _notifications(**kwargs: object) -> SimpleNamespace:
        _record("notifications", kwargs)
        return SimpleNamespace(
            jobs_notifications_bridge_task=kwargs["jobs_notifications_bridge_task"],
        )

    async def _runtime(**kwargs: object) -> SimpleNamespace:
        _record("runtime", kwargs)
        return SimpleNamespace(
            jobs_metrics_task=kwargs["jobs_metrics_task"],
            loop_lag_task=kwargs["loop_lag_task"],
        )

    async def _reconcile(**kwargs: object) -> SimpleNamespace:
        _record("reconcile", kwargs)
        return SimpleNamespace(
            jobs_metrics_reconcile_task=kwargs["jobs_metrics_reconcile_task"],
            jobs_metrics_reconcile_stop=kwargs["jobs_metrics_reconcile_stop"],
        )

    async def _personalization(**kwargs: object) -> None:
        _record("personalization", kwargs)

    async def _optional(**kwargs: object) -> SimpleNamespace:
        _record("optional", kwargs)
        return SimpleNamespace(
            jobs_crypto_rotate_task=kwargs["jobs_crypto_rotate_task"],
            jobs_integrity_task=kwargs["jobs_integrity_task"],
            jobs_webhooks_task=kwargs["jobs_webhooks_task"],
            meetings_webhook_dlq_task=kwargs["meetings_webhook_dlq_task"],
            workflows_dlq_task=kwargs["workflows_dlq_task"],
            workflows_gc_task=kwargs["workflows_gc_task"],
            workflows_maint_task=kwargs["workflows_maint_task"],
        )

    monkeypatch.setattr(
        shutdown_services,
        "_shutdown_notifications_compactor_websub_workers",
        _notifications,
    )
    monkeypatch.setattr(shutdown_services, "_shutdown_runtime_monitors", _runtime)
    monkeypatch.setattr(shutdown_services, "_shutdown_jobs_metrics_reconcile", _reconcile)
    monkeypatch.setattr(
        shutdown_services,
        "_shutdown_personalization_consolidation",
        _personalization,
    )
    monkeypatch.setattr(shutdown_services, "_shutdown_optional_workers", _optional)
    return recorded


def test_post_worker_shutdown_contract_omits_registry_owned_custom_worker_handles() -> None:
    from tldw_Server_API.app.services import shutdown_post_worker_services as shutdown_services

    obsolete_fields = {
        "claims_task",
        "embeddings_compactor_task",
        "embeddings_compactor_stop_event",
        "websub_renewal_task",
        "usage_task",
        "llm_usage_task",
        "coordinated_legacy_component_names",
    }

    assert obsolete_fields.isdisjoint(
        inspect.signature(shutdown_services.shutdown_post_worker_services).parameters
    )
    assert obsolete_fields.isdisjoint(
        inspect.signature(shutdown_services.run_shutdown_post_worker_services).parameters
    )
    assert obsolete_fields.isdisjoint(
        {field.name for field in fields(shutdown_services.PostWorkerShutdownHandles)}
    )
    assert not hasattr(shutdown_services, "_stop_usage_aggregators")


def test_post_worker_shutdown_contract_omits_registry_owned_scheduler_handles() -> None:
    from tldw_Server_API.app.services import shutdown_post_worker_services as shutdown_services

    obsolete_fields = {
        "jobs_prune_task",
        "files_export_gc_task",
        "notifications_prune_task",
        "workflows_sched_task",
        "reading_digest_sched_task",
        "admin_backup_sched_task",
        "companion_reflection_sched_task",
        "reminders_sched_task",
        "connectors_sync_sched_task",
    }

    assert obsolete_fields.isdisjoint(
        inspect.signature(shutdown_services.shutdown_post_worker_services).parameters
    )
    assert obsolete_fields.isdisjoint(
        inspect.signature(shutdown_services.run_shutdown_post_worker_services).parameters
    )
    assert obsolete_fields.isdisjoint(
        {field.name for field in fields(shutdown_services.PostWorkerShutdownHandles)}
    )
    assert not hasattr(shutdown_services, "_stop_recurring_schedulers")
    assert not hasattr(shutdown_services, "_shutdown_claims_maintenance_tasks")


def test_base_shutdown_kwargs_match_post_worker_shutdown_signatures() -> None:
    from tldw_Server_API.app.services import shutdown_post_worker_services as shutdown_services

    helper_keys = set(_base_shutdown_kwargs())

    assert helper_keys.issubset(
        inspect.signature(shutdown_services.shutdown_post_worker_services).parameters
    )
    assert helper_keys.issubset(
        inspect.signature(shutdown_services.run_shutdown_post_worker_services).parameters
    )


@pytest.mark.asyncio
async def test_shutdown_post_worker_services_runs_helpers_in_order_and_returns_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import shutdown_post_worker_services as shutdown_services

    calls: list[tuple[str, dict[str, object]]] = []
    _patch_post_worker_helpers(monkeypatch, shutdown_services, calls)

    handles = await shutdown_services.shutdown_post_worker_services(
        **_base_shutdown_kwargs(
            jobs_notifications_bridge_task="bridge-task",
            jobs_metrics_task="jobs-metrics-task",
            jobs_metrics_stop_event="jobs-metrics-stop",
            loop_lag_task="loop-lag-task",
            loop_lag_stop_event="loop-lag-stop",
            jobs_metrics_reconcile_task="jobs-metrics-reconcile-task",
            jobs_metrics_reconcile_stop="jobs-metrics-reconcile-stop",
            jobs_crypto_rotate_task="crypto-task",
            jobs_crypto_rotate_stop_event="crypto-stop",
            jobs_integrity_task="integrity-task",
            jobs_integrity_stop_event="integrity-stop",
            jobs_webhooks_task="webhooks-task",
            jobs_webhooks_stop_event="webhooks-stop",
            meetings_webhook_dlq_task="meetings-task",
            meetings_webhook_dlq_stop_event="meetings-stop",
            workflows_dlq_task="workflows-dlq-task",
            workflows_dlq_stop_event="workflows-dlq-stop",
            workflows_gc_task="workflows-gc-task",
            workflows_gc_stop_event="workflows-gc-stop",
            workflows_maint_task="workflows-maint-task",
            workflows_maint_stop_event="workflows-maint-stop",
            guard_exceptions=(RuntimeError,),
        )
    )

    assert [name for name, _ in calls] == [
        "notifications",
        "runtime",
        "reconcile",
        "personalization",
        "optional",
    ]
    assert calls[0][1]["jobs_notifications_bridge_task"] == "bridge-task"
    assert calls[1][1]["jobs_metrics_task"] == "jobs-metrics-task"
    assert calls[2][1]["jobs_metrics_reconcile_task"] == "jobs-metrics-reconcile-task"
    assert calls[3][1]["guard_exceptions"] == (RuntimeError,)
    assert calls[4][1]["jobs_crypto_rotate_task"] == "crypto-task"
    assert handles.jobs_notifications_bridge_task == "bridge-task"
    assert handles.jobs_metrics_task == "jobs-metrics-task"
    assert handles.loop_lag_task == "loop-lag-task"
    assert handles.jobs_metrics_reconcile_task == "jobs-metrics-reconcile-task"
    assert handles.jobs_metrics_reconcile_stop == "jobs-metrics-reconcile-stop"
    assert handles.jobs_crypto_rotate_task == "crypto-task"
    assert handles.jobs_integrity_task == "integrity-task"
    assert handles.jobs_webhooks_task == "webhooks-task"
    assert handles.meetings_webhook_dlq_task == "meetings-task"
    assert handles.workflows_dlq_task == "workflows-dlq-task"
    assert handles.workflows_gc_task == "workflows-gc-task"
    assert handles.workflows_maint_task == "workflows-maint-task"


@pytest.mark.asyncio
async def test_shutdown_post_worker_services_skips_loop_lag_after_background_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import shutdown_post_worker_services as shutdown_services

    recorded = _patch_post_worker_helpers(monkeypatch, shutdown_services)

    handles = await shutdown_services.shutdown_post_worker_services(
        **_base_shutdown_kwargs(
            loop_lag_task="loop-lag-task",
            loop_lag_stop_event="loop-lag-stop",
            stopped_background_worker_names={"loop_lag_task"},
        )
    )

    assert recorded["runtime"]["loop_lag_task"] is None
    assert recorded["runtime"]["loop_lag_stop_event"] is None
    assert handles.loop_lag_task is None


@pytest.mark.asyncio
async def test_run_shutdown_post_worker_services_fallback_skips_stopped_loop_lag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import shutdown_post_worker_services as shutdown_services

    async def _raise_shutdown_error(**kwargs: object) -> None:
        del kwargs
        raise RuntimeError("shutdown failed")

    monkeypatch.setattr(shutdown_services, "shutdown_post_worker_services", _raise_shutdown_error)

    handles = await shutdown_services.run_shutdown_post_worker_services(
        **_base_shutdown_kwargs(
            jobs_metrics_task="jobs-metrics-input",
            loop_lag_task="loop-lag-input",
            stopped_background_worker_names={"loop_lag_task"},
        )
    )

    assert handles.jobs_metrics_task == "jobs-metrics-input"
    assert handles.loop_lag_task is None


@pytest.mark.asyncio
async def test_shutdown_post_worker_services_skips_jobs_webhooks_after_background_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import shutdown_post_worker_services as shutdown_services

    recorded = _patch_post_worker_helpers(monkeypatch, shutdown_services)

    handles = await shutdown_services.shutdown_post_worker_services(
        **_base_shutdown_kwargs(
            jobs_crypto_rotate_task="crypto-task",
            jobs_crypto_rotate_stop_event="crypto-stop",
            jobs_integrity_task="integrity-task",
            jobs_integrity_stop_event="integrity-stop",
            jobs_webhooks_task="webhooks-task",
            jobs_webhooks_stop_event="webhooks-stop",
            meetings_webhook_dlq_task="meetings-task",
            meetings_webhook_dlq_stop_event="meetings-stop",
            stopped_background_worker_names={"jobs_webhooks_task"},
        )
    )

    assert recorded["optional"]["jobs_crypto_rotate_task"] == "crypto-task"
    assert recorded["optional"]["jobs_integrity_task"] == "integrity-task"
    assert recorded["optional"]["jobs_webhooks_task"] is None
    assert recorded["optional"]["jobs_webhooks_stop_event"] is None
    assert recorded["optional"]["meetings_webhook_dlq_task"] == "meetings-task"
    assert handles.jobs_crypto_rotate_task == "crypto-task"
    assert handles.jobs_integrity_task == "integrity-task"
    assert handles.jobs_webhooks_task is None
    assert handles.meetings_webhook_dlq_task == "meetings-task"


@pytest.mark.asyncio
async def test_shutdown_post_worker_services_skips_bridge_after_background_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import shutdown_post_worker_services as shutdown_services

    recorded = _patch_post_worker_helpers(monkeypatch, shutdown_services)

    handles = await shutdown_services.shutdown_post_worker_services(
        **_base_shutdown_kwargs(
            jobs_notifications_bridge_task="bridge-task",
            stopped_background_worker_names={"jobs_notifications_bridge_task"},
        )
    )

    assert recorded["notifications"]["jobs_notifications_bridge_task"] is None
    assert handles.jobs_notifications_bridge_task is None


@pytest.mark.asyncio
async def test_run_shutdown_post_worker_services_delegates_and_returns_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import shutdown_post_worker_services as shutdown_services

    recorded_kwargs: dict[str, object] = {}

    async def _fake_shutdown_post_worker_services(**kwargs):
        recorded_kwargs.update(kwargs)
        return shutdown_services.PostWorkerShutdownHandles(
            jobs_notifications_bridge_task="bridge-result",
            jobs_metrics_task="metrics-result",
            loop_lag_task="loop-lag-result",
            jobs_metrics_reconcile_task="reconcile-result",
            jobs_metrics_reconcile_stop="reconcile-stop-result",
            jobs_crypto_rotate_task="crypto-result",
            jobs_integrity_task="integrity-result",
            jobs_webhooks_task="webhooks-result",
            meetings_webhook_dlq_task="meetings-result",
            workflows_dlq_task="dlq-result",
            workflows_gc_task="gc-result",
            workflows_maint_task="maint-result",
        )

    monkeypatch.setattr(
        shutdown_services,
        "shutdown_post_worker_services",
        _fake_shutdown_post_worker_services,
    )

    handles = await shutdown_services.run_shutdown_post_worker_services(
        **_base_shutdown_kwargs(
            jobs_notifications_bridge_task="bridge-input",
            jobs_metrics_reconcile_task="reconcile-input",
            workflows_maint_task="maint-input",
        )
    )

    assert recorded_kwargs["jobs_notifications_bridge_task"] == "bridge-input"
    assert recorded_kwargs["jobs_metrics_reconcile_task"] == "reconcile-input"
    assert recorded_kwargs["workflows_maint_task"] == "maint-input"
    assert handles.jobs_notifications_bridge_task == "bridge-result"
    assert handles.jobs_metrics_task == "metrics-result"
    assert handles.loop_lag_task == "loop-lag-result"
    assert handles.jobs_metrics_reconcile_task == "reconcile-result"
    assert handles.jobs_metrics_reconcile_stop == "reconcile-stop-result"
    assert handles.jobs_crypto_rotate_task == "crypto-result"
    assert handles.jobs_integrity_task == "integrity-result"
    assert handles.jobs_webhooks_task == "webhooks-result"
    assert handles.meetings_webhook_dlq_task == "meetings-result"
    assert handles.workflows_dlq_task == "dlq-result"
    assert handles.workflows_gc_task == "gc-result"
    assert handles.workflows_maint_task == "maint-result"


@pytest.mark.asyncio
async def test_run_shutdown_post_worker_services_guard_failure_suppresses_stopped_background_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import shutdown_post_worker_services as shutdown_services

    async def _fail(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("post-worker boom")

    log_messages: list[str] = []

    monkeypatch.setattr(shutdown_services, "shutdown_post_worker_services", _fail)
    monkeypatch.setattr(
        shutdown_services.logger,
        "debug",
        lambda message: log_messages.append(message),
    )

    handles = await shutdown_services.run_shutdown_post_worker_services(
        **_base_shutdown_kwargs(
            jobs_notifications_bridge_task="bridge-input",
            jobs_metrics_task="metrics-input",
            loop_lag_task="loop-lag-input",
            jobs_metrics_reconcile_task="reconcile-input",
            jobs_metrics_reconcile_stop="reconcile-stop-input",
            jobs_crypto_rotate_task="crypto-input",
            jobs_integrity_task="integrity-input",
            jobs_webhooks_task="webhooks-input",
            meetings_webhook_dlq_task="meetings-input",
            workflows_dlq_task="dlq-input",
            workflows_gc_task="gc-input",
            workflows_maint_task="maint-input",
            stopped_background_worker_names={
                "jobs_notifications_bridge_task",
                "jobs_metrics_task",
                "jobs_metrics_reconcile_task",
                "jobs_crypto_rotate_task",
                "jobs_integrity_task",
                "meetings_webhook_dlq_task",
                "workflows_dlq_task",
                "workflows_gc_task",
                "workflows_maint_task",
            },
        )
    )

    assert log_messages == ["Post-worker services skipped: post-worker boom"]
    assert handles.jobs_notifications_bridge_task is None
    assert handles.jobs_metrics_task is None
    assert handles.loop_lag_task == "loop-lag-input"
    assert handles.jobs_metrics_reconcile_task is None
    assert handles.jobs_metrics_reconcile_stop is None
    assert handles.jobs_crypto_rotate_task is None
    assert handles.jobs_integrity_task is None
    assert handles.jobs_webhooks_task == "webhooks-input"
    assert handles.meetings_webhook_dlq_task is None
    assert handles.workflows_dlq_task is None
    assert handles.workflows_gc_task is None
    assert handles.workflows_maint_task is None
