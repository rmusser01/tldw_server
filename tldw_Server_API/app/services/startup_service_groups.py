"""
Startup service-tail orchestration extracted from the application lifespan.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class StartupServiceGroupHandles:
    """Combined startup handles produced by the service-tail startup burst."""

    jobs_metrics_stop_event: Any | None = None
    jobs_metrics_task: Any | None = None
    loop_lag_stop_event: Any | None = None
    loop_lag_task: Any | None = None
    jobs_metrics_reconcile_stop: Any | None = None
    jobs_metrics_reconcile_task: Any | None = None
    jobs_crypto_rotate_stop_event: Any | None = None
    jobs_crypto_rotate_task: Any | None = None
    jobs_webhooks_stop_event: Any | None = None
    jobs_webhooks_task: Any | None = None
    meetings_webhook_dlq_stop_event: Any | None = None
    meetings_webhook_dlq_task: Any | None = None
    workflows_dlq_stop_event: Any | None = None
    workflows_dlq_task: Any | None = None
    workflows_gc_stop_event: Any | None = None
    workflows_gc_task: Any | None = None
    workflows_maint_stop_event: Any | None = None
    workflows_maint_task: Any | None = None
    jobs_integrity_stop_event: Any | None = None
    jobs_integrity_task: Any | None = None
    claims_alerts_task: Any | None = None
    claims_review_metrics_task: Any | None = None
    tts_history_cleanup_task: Any | None = None
    tts_history_cleanup_stop_event: Any | None = None
    connectors_jobs_task: Any | None = None
    connectors_jobs_stop_event: Any | None = None


async def start_service_groups(
    *,
    app: Any,
    app_settings: Any,
    run_pg_rls_auto_ensure: Callable[[Any], Any],
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    worker_inventory: Any | None = None,
) -> StartupServiceGroupHandles:
    """Start the runtime/optional/auxiliary/infra service tail in the legacy order."""
    if worker_inventory is None:
        raise RuntimeError("worker_inventory is required to start service groups")

    runtime_monitor_handles = await _start_runtime_monitors(
        worker_inventory=worker_inventory,
    )
    optional_worker_handles = await _start_optional_workers(
        worker_inventory=worker_inventory,
    )
    await _start_claims_rebuild_worker(
        app_settings,
        worker_inventory=worker_inventory,
    )
    auxiliary_startup_handles = await _start_auxiliary_services(
        app_settings,
        worker_inventory=worker_inventory,
    )
    infra_startup_handles = await _start_infra_services(
        run_pg_rls_auto_ensure=run_pg_rls_auto_ensure,
        worker_inventory=worker_inventory,
    )
    await _start_maintenance_schedulers(
        worker_inventory=worker_inventory,
    )
    connectors_startup_handles = await _start_connectors_startup(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
    )
    return StartupServiceGroupHandles(
        jobs_metrics_stop_event=runtime_monitor_handles.jobs_metrics_stop_event,
        jobs_metrics_task=runtime_monitor_handles.jobs_metrics_task,
        loop_lag_stop_event=runtime_monitor_handles.loop_lag_stop_event,
        loop_lag_task=runtime_monitor_handles.loop_lag_task,
        jobs_metrics_reconcile_stop=optional_worker_handles.jobs_metrics_reconcile_stop,
        jobs_metrics_reconcile_task=optional_worker_handles.jobs_metrics_reconcile_task,
        jobs_crypto_rotate_stop_event=optional_worker_handles.jobs_crypto_rotate_stop_event,
        jobs_crypto_rotate_task=optional_worker_handles.jobs_crypto_rotate_task,
        jobs_webhooks_stop_event=optional_worker_handles.jobs_webhooks_stop_event,
        jobs_webhooks_task=optional_worker_handles.jobs_webhooks_task,
        meetings_webhook_dlq_stop_event=optional_worker_handles.meetings_webhook_dlq_stop_event,
        meetings_webhook_dlq_task=optional_worker_handles.meetings_webhook_dlq_task,
        workflows_dlq_stop_event=optional_worker_handles.workflows_dlq_stop_event,
        workflows_dlq_task=optional_worker_handles.workflows_dlq_task,
        workflows_gc_stop_event=optional_worker_handles.workflows_gc_stop_event,
        workflows_gc_task=optional_worker_handles.workflows_gc_task,
        workflows_maint_stop_event=optional_worker_handles.workflows_maint_stop_event,
        workflows_maint_task=optional_worker_handles.workflows_maint_task,
        jobs_integrity_stop_event=optional_worker_handles.jobs_integrity_stop_event,
        jobs_integrity_task=optional_worker_handles.jobs_integrity_task,
        claims_alerts_task=auxiliary_startup_handles.claims_alerts_task,
        claims_review_metrics_task=auxiliary_startup_handles.claims_review_metrics_task,
        tts_history_cleanup_task=infra_startup_handles.tts_history_cleanup_task,
        tts_history_cleanup_stop_event=infra_startup_handles.tts_history_cleanup_stop_event,
        connectors_jobs_task=connectors_startup_handles.connectors_jobs_task,
        connectors_jobs_stop_event=connectors_startup_handles.connectors_jobs_stop_event,
    )


async def _start_runtime_monitors(**kwargs: Any) -> Any:
    from tldw_Server_API.app.services.startup_runtime_monitors import start_runtime_monitors

    return await start_runtime_monitors(**kwargs)


async def _start_optional_workers(**kwargs: Any) -> Any:
    from tldw_Server_API.app.services.startup_optional_workers import start_optional_workers

    return await start_optional_workers(**kwargs)


async def _start_claims_rebuild_worker(app_settings: Any, **kwargs: Any) -> Any:
    from tldw_Server_API.app.services.startup_claims_rebuild import (
        start_claims_rebuild_worker,
    )

    return await start_claims_rebuild_worker(app_settings, **kwargs)


async def _start_auxiliary_services(app_settings: Any, **kwargs: Any) -> Any:
    from tldw_Server_API.app.services.startup_auxiliary_services import (
        start_auxiliary_services,
    )

    return await start_auxiliary_services(app_settings, **kwargs)


async def _start_infra_services(**kwargs):
    from tldw_Server_API.app.services.startup_infra_services import start_infra_services

    return await start_infra_services(**kwargs)


async def _start_maintenance_schedulers(**kwargs: Any) -> Any:
    """Start maintenance schedulers while preserving no-arg monkeypatch compatibility."""

    from tldw_Server_API.app.services.startup_maintenance_schedulers import (
        start_maintenance_schedulers,
    )

    if "worker_inventory" in kwargs and not _accepts_keyword(
        start_maintenance_schedulers,
        "worker_inventory",
    ):
        kwargs = {key: value for key, value in kwargs.items() if key != "worker_inventory"}
    return await start_maintenance_schedulers(**kwargs)


def _accepts_keyword(func: Callable[..., Any], keyword: str) -> bool:
    """Return whether a callable can accept the given keyword argument."""

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return True

    parameter = signature.parameters.get(keyword)
    if parameter is not None and parameter.kind in {
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }:
        return True

    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


async def _start_connectors_startup(**kwargs):
    from tldw_Server_API.app.services.startup_infra_services import (
        start_connectors_startup,
    )

    return await start_connectors_startup(**kwargs)
