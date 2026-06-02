import asyncio
from dataclasses import fields
from typing import Any

import pytest
from fastapi import FastAPI

from tldw_Server_API.app.services.lifecycle_worker_specs import (
    ShutdownPhase,
    WorkerLifecycleContext,
    WorkerSpec,
    WorkerSpecValidationError,
)
from tldw_Server_API.app.services.lifespan_worker_runtime_state import (
    LifespanWorkerRuntimeState,
)
from tldw_Server_API.tests.Services.test_worker_lifecycle_ownership_matrix import (
    legacy_worker_names_from_ownership_matrix,
)


def _context() -> WorkerLifecycleContext:
    return WorkerLifecycleContext(
        app=FastAPI(),
        settings={},
        test_mode=True,
        route_enabled=lambda *_args, **_kwargs: True,
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
    )


async def _wait_for_stop(stop_event: asyncio.Event) -> None:
    await stop_event.wait()


def _worker_spec(name: str, **overrides: Any) -> WorkerSpec:
    values: dict[str, Any] = {
        "name": name,
        "task_name": name,
        "category": "test",
        "phase": ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
        "factory": lambda _context, stop_event: _wait_for_stop(stop_event),
    }
    values.update(overrides)
    return WorkerSpec(**values)


def _runtime_managed_worker_names() -> set[str]:
    return {
        field.name
        for field in fields(LifespanWorkerRuntimeState)
        if field.name.endswith("_task")
    }


LEGACY_MANAGED_WORKER_NAMES = (
    legacy_worker_names_from_ownership_matrix() | _runtime_managed_worker_names()
)

TASK5_JOB_POLLER_SPEC_NAMES = {
    "core_jobs_task",
    "files_jobs_task",
    "data_tables_jobs_task",
    "prompt_studio_jobs_task",
    "study_pack_jobs_task",
    "study_suggestions_jobs_task",
    "privilege_snapshot_task",
    "audio_jobs_task",
    "audiobook_jobs_task",
    "presentation_render_jobs_task",
    "media_ingest_jobs_task",
    "media_ingest_heavy_jobs_task",
    "reading_digest_jobs_task",
    "vn_asset_jobs_task",
    "vn_asset_generation_jobs_task",
    "companion_reflection_jobs_task",
}

TASK6_BACKGROUND_SPEC_NAMES = {
    "reminder_jobs_task",
    "admin_backup_jobs_task",
    "admin_byok_validation_jobs_task",
    "admin_maintenance_rotation_jobs_task",
    "recipe_run_jobs_task",
    "jobs_notifications_bridge_task",
    "evals_abtest_jobs_task",
    "ephemeral_cleanup_task",
    "chatbooks_cleanup",
    "storage_cleanup_service",
    "embeddings_compactor_task",
    "websub_renewal_task",
    "claims_rebuild",
    "usage_aggregator",
    "llm_usage_aggregator",
}


@pytest.mark.unit
def test_collect_worker_specs_collects_specs_from_provider_functions() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_catalog import (
        collect_worker_specs,
    )

    provider_calls: list[bool] = []

    def first_provider(context: WorkerLifecycleContext) -> tuple[WorkerSpec, ...]:
        provider_calls.append(context.test_mode)
        return (_worker_spec("alpha_worker"),)

    def second_provider(context: WorkerLifecycleContext) -> list[WorkerSpec]:
        provider_calls.append(context.test_mode)
        return [_worker_spec("beta_worker", depends_on=("alpha_worker",))]

    specs = collect_worker_specs(_context(), [first_provider, second_provider])

    assert [spec.name for spec in specs] == ["alpha_worker", "beta_worker"]
    assert provider_calls == [True, True]


@pytest.mark.unit
def test_collect_worker_specs_rejects_duplicate_provider_names_through_graph_validation() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_catalog import (
        collect_worker_specs,
    )

    def first_provider(_context: WorkerLifecycleContext) -> tuple[WorkerSpec, ...]:
        return (_worker_spec("duplicate_worker"),)

    def second_provider(_context: WorkerLifecycleContext) -> tuple[WorkerSpec, ...]:
        return (_worker_spec("duplicate_worker"),)

    with pytest.raises(WorkerSpecValidationError, match="duplicate.*duplicate_worker"):
        collect_worker_specs(_context(), [first_provider, second_provider])


@pytest.mark.unit
def test_collect_worker_specs_accepts_task5_job_poller_spec_providers() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_catalog import (
        assert_legacy_worker_spec_parity,
        collect_worker_specs,
    )
    from tldw_Server_API.app.services.startup_content_jobs_pollers import (
        provide_content_jobs_worker_specs,
    )
    from tldw_Server_API.app.services.startup_primary_jobs_pollers import (
        provide_primary_jobs_worker_specs,
    )
    from tldw_Server_API.app.services.startup_study_privilege_jobs_pollers import (
        provide_study_privilege_jobs_worker_specs,
    )

    specs = collect_worker_specs(
        _context(),
        [
            provide_primary_jobs_worker_specs,
            provide_study_privilege_jobs_worker_specs,
            provide_content_jobs_worker_specs,
        ],
    )

    assert {spec.name for spec in specs} == TASK5_JOB_POLLER_SPEC_NAMES
    assert_legacy_worker_spec_parity(TASK5_JOB_POLLER_SPEC_NAMES, specs)


@pytest.mark.unit
def test_collect_worker_specs_accepts_task6_background_spec_providers() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_catalog import (
        assert_legacy_worker_spec_parity,
        collect_worker_specs,
    )
    from tldw_Server_API.app.services.llm_usage_aggregator import (
        provide_llm_usage_aggregator_worker_specs,
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
    from tldw_Server_API.app.services.startup_notifications_abtest_workers import (
        provide_notifications_abtest_worker_specs,
    )
    from tldw_Server_API.app.services.startup_sidecar_owned_jobs_pollers import (
        provide_sidecar_owned_jobs_worker_specs,
    )
    from tldw_Server_API.app.services.usage_aggregator import (
        provide_usage_aggregator_worker_specs,
    )

    specs = collect_worker_specs(
        _context(),
        [
            provide_sidecar_owned_jobs_worker_specs,
            provide_notifications_abtest_worker_specs,
            provide_cleanup_worker_specs,
            provide_compactor_websub_worker_specs,
            provide_claims_rebuild_worker_specs,
            provide_usage_aggregator_worker_specs,
            provide_llm_usage_aggregator_worker_specs,
        ],
    )

    assert {spec.name for spec in specs} == TASK6_BACKGROUND_SPEC_NAMES
    assert_legacy_worker_spec_parity(TASK6_BACKGROUND_SPEC_NAMES, specs)


@pytest.mark.unit
def test_legacy_managed_worker_names_from_runtime_fields_and_matrix_have_specs() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_catalog import (
        assert_legacy_worker_spec_parity,
    )

    specs = [_worker_spec(name) for name in sorted(LEGACY_MANAGED_WORKER_NAMES)]

    assert_legacy_worker_spec_parity(LEGACY_MANAGED_WORKER_NAMES, specs)


@pytest.mark.unit
def test_legacy_worker_spec_parity_reports_missing_worker_name() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_catalog import (
        assert_legacy_worker_spec_parity,
    )

    missing_name = "core_jobs_task"
    specs = [
        _worker_spec(name)
        for name in sorted(LEGACY_MANAGED_WORKER_NAMES)
        if name != missing_name
    ]

    with pytest.raises(AssertionError, match=missing_name):
        assert_legacy_worker_spec_parity(LEGACY_MANAGED_WORKER_NAMES, specs)
