from __future__ import annotations

from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_shutdown_post_worker_services_runs_helpers_in_order_and_returns_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import shutdown_post_worker_services as shutdown_services

    calls: list[tuple[str, dict[str, object]]] = []
    coordinated_legacy_component_names = {"usage_aggregator"}
    guard_exceptions = (RuntimeError,)

    async def _record_claims(**kwargs):
        calls.append(("claims", kwargs))
        return SimpleNamespace(
            claims_task=kwargs["claims_task"],
            jobs_prune_task=kwargs["jobs_prune_task"],
            files_export_gc_task=kwargs["files_export_gc_task"],
            notifications_prune_task=kwargs["notifications_prune_task"],
        )

    async def _record_notifications(**kwargs):
        calls.append(("notifications", kwargs))
        return SimpleNamespace(
            jobs_notifications_bridge_task=kwargs["jobs_notifications_bridge_task"],
            embeddings_compactor_task=kwargs["embeddings_compactor_task"],
            embeddings_compactor_stop_event=kwargs["embeddings_compactor_stop_event"],
            websub_renewal_task=kwargs["websub_renewal_task"],
        )

    async def _record_usage(**kwargs):
        calls.append(("usage", kwargs))
        return SimpleNamespace(
            usage_task=kwargs["usage_task"],
            llm_usage_task=kwargs["llm_usage_task"],
        )

    async def _record_recurring(**kwargs):
        calls.append(("recurring", kwargs))

    async def _record_runtime(**kwargs):
        calls.append(("runtime", kwargs))
        return SimpleNamespace(
            jobs_metrics_task=kwargs["jobs_metrics_task"],
            loop_lag_task=kwargs["loop_lag_task"],
        )

    async def _record_reconcile(**kwargs):
        calls.append(("reconcile", kwargs))
        return SimpleNamespace(
            jobs_metrics_reconcile_task=kwargs["jobs_metrics_reconcile_task"],
            jobs_metrics_reconcile_stop=kwargs["jobs_metrics_reconcile_stop"],
        )

    async def _record_personalization(**kwargs):
        calls.append(("personalization", kwargs))

    async def _record_optional(**kwargs):
        calls.append(("optional", kwargs))
        return SimpleNamespace(
            jobs_crypto_rotate_task=kwargs["jobs_crypto_rotate_task"],
            jobs_integrity_task=kwargs["jobs_integrity_task"],
            jobs_webhooks_task=kwargs["jobs_webhooks_task"],
            meetings_webhook_dlq_task=kwargs["meetings_webhook_dlq_task"],
            workflows_dlq_task=kwargs["workflows_dlq_task"],
            workflows_gc_task=kwargs["workflows_gc_task"],
            workflows_maint_task=kwargs["workflows_maint_task"],
        )

    monkeypatch.setattr(shutdown_services, "_shutdown_claims_maintenance_tasks", _record_claims)
    monkeypatch.setattr(
        shutdown_services,
        "_shutdown_notifications_compactor_websub_workers",
        _record_notifications,
    )
    monkeypatch.setattr(shutdown_services, "_stop_usage_aggregators", _record_usage)
    monkeypatch.setattr(shutdown_services, "_stop_recurring_schedulers", _record_recurring)
    monkeypatch.setattr(shutdown_services, "_shutdown_runtime_monitors", _record_runtime)
    monkeypatch.setattr(shutdown_services, "_shutdown_jobs_metrics_reconcile", _record_reconcile)
    monkeypatch.setattr(
        shutdown_services,
        "_shutdown_personalization_consolidation",
        _record_personalization,
    )
    monkeypatch.setattr(shutdown_services, "_shutdown_optional_workers", _record_optional)

    handles = await shutdown_services.shutdown_post_worker_services(
        claims_task="claims-task",
        jobs_prune_task="jobs-prune-task",
        files_export_gc_task="files-gc-task",
        notifications_prune_task="notifications-prune-task",
        jobs_notifications_bridge_task="bridge-task",
        embeddings_compactor_task="compactor-task",
        embeddings_compactor_stop_event="compactor-stop",
        websub_renewal_task="websub-task",
        coordinated_legacy_component_names=coordinated_legacy_component_names,
        usage_task="usage-task",
        llm_usage_task="llm-task",
        workflows_sched_task="workflows-sched-task",
        reading_digest_sched_task="reading-digest-sched-task",
        admin_backup_sched_task="admin-backup-sched-task",
        companion_reflection_sched_task="companion-reflection-sched-task",
        reminders_sched_task="reminders-sched-task",
        connectors_sync_sched_task="connectors-sync-sched-task",
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
        guard_exceptions=guard_exceptions,
    )

    assert [name for name, _ in calls] == [
        "claims",
        "notifications",
        "usage",
        "recurring",
        "runtime",
        "reconcile",
        "personalization",
        "optional",
    ]
    assert calls[0][1]["claims_task"] == "claims-task"
    assert calls[1][1]["jobs_notifications_bridge_task"] == "bridge-task"
    assert calls[2][1]["coordinated_legacy_component_names"] is coordinated_legacy_component_names
    assert calls[3][1]["workflows_sched_task"] == "workflows-sched-task"
    assert calls[4][1]["jobs_metrics_task"] == "jobs-metrics-task"
    assert calls[5][1]["jobs_metrics_reconcile_task"] == "jobs-metrics-reconcile-task"
    assert calls[6][1]["guard_exceptions"] == guard_exceptions
    assert calls[7][1]["jobs_crypto_rotate_task"] == "crypto-task"
    assert handles.claims_task == "claims-task"
    assert handles.jobs_prune_task == "jobs-prune-task"
    assert handles.files_export_gc_task == "files-gc-task"
    assert handles.notifications_prune_task == "notifications-prune-task"
    assert handles.jobs_notifications_bridge_task == "bridge-task"
    assert handles.embeddings_compactor_task == "compactor-task"
    assert handles.embeddings_compactor_stop_event == "compactor-stop"
    assert handles.websub_renewal_task == "websub-task"
    assert handles.usage_task == "usage-task"
    assert handles.llm_usage_task == "llm-task"
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
async def test_run_shutdown_post_worker_services_delegates_and_returns_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import shutdown_post_worker_services as shutdown_services

    recorded_kwargs: dict[str, object] = {}

    async def _fake_shutdown_post_worker_services(**kwargs):
        recorded_kwargs.update(kwargs)
        return shutdown_services.PostWorkerShutdownHandles(
            claims_task="claims-result",
            jobs_prune_task="prune-result",
            files_export_gc_task="files-result",
            notifications_prune_task="notifications-result",
            jobs_notifications_bridge_task="bridge-result",
            embeddings_compactor_task="compactor-result",
            embeddings_compactor_stop_event="compactor-stop-result",
            websub_renewal_task="websub-result",
            usage_task="usage-result",
            llm_usage_task="llm-result",
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
        claims_task="claims-input",
        jobs_prune_task="prune-input",
        files_export_gc_task="files-input",
        notifications_prune_task="notifications-input",
        jobs_notifications_bridge_task="bridge-input",
        embeddings_compactor_task="compactor-input",
        embeddings_compactor_stop_event="compactor-stop-input",
        websub_renewal_task="websub-input",
        coordinated_legacy_component_names={"usage_aggregator"},
        usage_task="usage-input",
        llm_usage_task="llm-input",
        workflows_sched_task="workflows-sched-input",
        reading_digest_sched_task="reading-sched-input",
        admin_backup_sched_task="admin-sched-input",
        companion_reflection_sched_task="companion-sched-input",
        reminders_sched_task="reminders-sched-input",
        connectors_sync_sched_task="connectors-sched-input",
        jobs_metrics_task="metrics-input",
        jobs_metrics_stop_event="metrics-stop-input",
        loop_lag_task="loop-lag-input",
        loop_lag_stop_event="loop-lag-stop-input",
        jobs_metrics_reconcile_task="reconcile-input",
        jobs_metrics_reconcile_stop="reconcile-stop-input",
        jobs_crypto_rotate_task="crypto-input",
        jobs_crypto_rotate_stop_event="crypto-stop-input",
        jobs_integrity_task="integrity-input",
        jobs_integrity_stop_event="integrity-stop-input",
        jobs_webhooks_task="webhooks-input",
        jobs_webhooks_stop_event="webhooks-stop-input",
        meetings_webhook_dlq_task="meetings-input",
        meetings_webhook_dlq_stop_event="meetings-stop-input",
        workflows_dlq_task="dlq-input",
        workflows_dlq_stop_event="dlq-stop-input",
        workflows_gc_task="gc-input",
        workflows_gc_stop_event="gc-stop-input",
        workflows_maint_task="maint-input",
        workflows_maint_stop_event="maint-stop-input",
        guard_exceptions=(RuntimeError,),
    )

    assert recorded_kwargs["claims_task"] == "claims-input"
    assert recorded_kwargs["jobs_notifications_bridge_task"] == "bridge-input"
    assert recorded_kwargs["coordinated_legacy_component_names"] == {"usage_aggregator"}
    assert recorded_kwargs["jobs_metrics_reconcile_task"] == "reconcile-input"
    assert recorded_kwargs["workflows_maint_task"] == "maint-input"
    assert handles.claims_task == "claims-result"
    assert handles.jobs_prune_task == "prune-result"
    assert handles.files_export_gc_task == "files-result"
    assert handles.notifications_prune_task == "notifications-result"
    assert handles.jobs_notifications_bridge_task == "bridge-result"
    assert handles.embeddings_compactor_task == "compactor-result"
    assert handles.embeddings_compactor_stop_event == "compactor-stop-result"
    assert handles.websub_renewal_task == "websub-result"
    assert handles.usage_task == "usage-result"
    assert handles.llm_usage_task == "llm-result"
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
async def test_run_shutdown_post_worker_services_logs_and_returns_original_handles_on_guard_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import shutdown_post_worker_services as shutdown_services

    def _fail(*args, **kwargs):
        raise RuntimeError("post-worker boom")

    log_messages: list[str] = []

    monkeypatch.setattr(shutdown_services, "shutdown_post_worker_services", _fail)
    monkeypatch.setattr(
        shutdown_services.logger,
        "debug",
        lambda message: log_messages.append(message),
    )

    handles = await shutdown_services.run_shutdown_post_worker_services(
        claims_task="claims-input",
        jobs_prune_task="prune-input",
        files_export_gc_task="files-input",
        notifications_prune_task="notifications-input",
        jobs_notifications_bridge_task="bridge-input",
        embeddings_compactor_task="compactor-input",
        embeddings_compactor_stop_event="compactor-stop-input",
        websub_renewal_task="websub-input",
        coordinated_legacy_component_names={"usage_aggregator"},
        usage_task="usage-input",
        llm_usage_task="llm-input",
        workflows_sched_task="workflows-sched-input",
        reading_digest_sched_task="reading-sched-input",
        admin_backup_sched_task="admin-sched-input",
        companion_reflection_sched_task="companion-sched-input",
        reminders_sched_task="reminders-sched-input",
        connectors_sync_sched_task="connectors-sched-input",
        jobs_metrics_task="metrics-input",
        jobs_metrics_stop_event="metrics-stop-input",
        loop_lag_task="loop-lag-input",
        loop_lag_stop_event="loop-lag-stop-input",
        jobs_metrics_reconcile_task="reconcile-input",
        jobs_metrics_reconcile_stop="reconcile-stop-input",
        jobs_crypto_rotate_task="crypto-input",
        jobs_crypto_rotate_stop_event="crypto-stop-input",
        jobs_integrity_task="integrity-input",
        jobs_integrity_stop_event="integrity-stop-input",
        jobs_webhooks_task="webhooks-input",
        jobs_webhooks_stop_event="webhooks-stop-input",
        meetings_webhook_dlq_task="meetings-input",
        meetings_webhook_dlq_stop_event="meetings-stop-input",
        workflows_dlq_task="dlq-input",
        workflows_dlq_stop_event="dlq-stop-input",
        workflows_gc_task="gc-input",
        workflows_gc_stop_event="gc-stop-input",
        workflows_maint_task="maint-input",
        workflows_maint_stop_event="maint-stop-input",
        guard_exceptions=(RuntimeError,),
    )

    assert log_messages == ["Post-worker services skipped: post-worker boom"]
    assert handles.claims_task == "claims-input"
    assert handles.jobs_prune_task == "prune-input"
    assert handles.files_export_gc_task == "files-input"
    assert handles.notifications_prune_task == "notifications-input"
    assert handles.jobs_notifications_bridge_task == "bridge-input"
    assert handles.embeddings_compactor_task == "compactor-input"
    assert handles.embeddings_compactor_stop_event == "compactor-stop-input"
    assert handles.websub_renewal_task == "websub-input"
    assert handles.usage_task == "usage-input"
    assert handles.llm_usage_task == "llm-input"
    assert handles.jobs_metrics_task == "metrics-input"
    assert handles.loop_lag_task == "loop-lag-input"
    assert handles.jobs_metrics_reconcile_task == "reconcile-input"
    assert handles.jobs_metrics_reconcile_stop == "reconcile-stop-input"
    assert handles.jobs_crypto_rotate_task == "crypto-input"
    assert handles.jobs_integrity_task == "integrity-input"
    assert handles.jobs_webhooks_task == "webhooks-input"
    assert handles.meetings_webhook_dlq_task == "meetings-input"
    assert handles.workflows_dlq_task == "dlq-input"
    assert handles.workflows_gc_task == "gc-input"
    assert handles.workflows_maint_task == "maint-input"
