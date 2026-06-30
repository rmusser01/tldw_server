"""Startup worker spec provider catalog."""

from __future__ import annotations

from tldw_Server_API.app.services.lifecycle_worker_catalog import (
    SpecProvider,
    collect_worker_specs,
)
from tldw_Server_API.app.services.lifecycle_worker_specs import (
    WorkerLifecycleContext,
    WorkerSpec,
)


def startup_worker_spec_providers() -> tuple[SpecProvider, ...]:
    """Return providers for every startup-managed worker spec."""

    from tldw_Server_API.app.services.llm_usage_aggregator import (
        provide_llm_usage_aggregator_worker_specs,
    )
    from tldw_Server_API.app.services.startup_auxiliary_services import (
        provide_auxiliary_worker_specs,
    )
    from tldw_Server_API.app.services.startup_claims_rebuild import (
        provide_claims_rebuild_worker_specs,
    )
    from tldw_Server_API.app.services.startup_cleanup_workers import (
        provide_cleanup_worker_specs,
    )
    from tldw_Server_API.app.services.startup_compactor_websub_workers import (
        provide_compactor_websub_worker_specs,
    )
    from tldw_Server_API.app.services.startup_content_jobs_pollers import (
        provide_content_jobs_worker_specs,
    )
    from tldw_Server_API.app.services.startup_infra_services import (
        provide_infra_worker_specs,
    )
    from tldw_Server_API.app.services.startup_maintenance_schedulers import (
        provide_maintenance_scheduler_worker_specs,
    )
    from tldw_Server_API.app.services.startup_notifications_abtest_workers import (
        provide_notifications_abtest_worker_specs,
    )
    from tldw_Server_API.app.services.startup_optional_workers import (
        provide_optional_worker_specs,
    )
    from tldw_Server_API.app.services.startup_primary_jobs_pollers import (
        provide_primary_jobs_worker_specs,
    )
    from tldw_Server_API.app.services.startup_recurring_schedulers import (
        provide_recurring_scheduler_worker_specs,
    )
    from tldw_Server_API.app.services.startup_runtime_monitors import (
        provide_runtime_monitor_worker_specs,
    )
    from tldw_Server_API.app.services.startup_sidecar_owned_jobs_pollers import (
        provide_sidecar_owned_jobs_worker_specs,
    )
    from tldw_Server_API.app.services.startup_study_privilege_jobs_pollers import (
        provide_study_privilege_jobs_worker_specs,
    )
    from tldw_Server_API.app.services.usage_aggregator import (
        provide_usage_aggregator_worker_specs,
    )

    return (
        provide_primary_jobs_worker_specs,
        provide_study_privilege_jobs_worker_specs,
        provide_content_jobs_worker_specs,
        provide_sidecar_owned_jobs_worker_specs,
        provide_notifications_abtest_worker_specs,
        provide_cleanup_worker_specs,
        provide_compactor_websub_worker_specs,
        provide_claims_rebuild_worker_specs,
        provide_usage_aggregator_worker_specs,
        provide_llm_usage_aggregator_worker_specs,
        provide_runtime_monitor_worker_specs,
        provide_optional_worker_specs,
        provide_auxiliary_worker_specs,
        provide_infra_worker_specs,
        provide_maintenance_scheduler_worker_specs,
        provide_recurring_scheduler_worker_specs,
    )


def collect_startup_worker_specs(
    context: WorkerLifecycleContext,
) -> tuple[WorkerSpec, ...]:
    """Collect and validate all startup-managed worker specs."""

    return tuple(collect_worker_specs(context, startup_worker_spec_providers()))
