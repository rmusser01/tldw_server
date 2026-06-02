from __future__ import annotations

import importlib
import sys

import pytest

from tldw_Server_API.app.services.lifecycle_worker_specs import WorkerLifecycleContext

pytestmark = pytest.mark.unit


def _import_startup_worker_groups():
    sys.modules.pop("tldw_Server_API.app.services.startup_worker_groups", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_worker_groups")


def _context() -> WorkerLifecycleContext:
    return WorkerLifecycleContext(
        app=object(),
        settings={},
        test_mode=True,
        route_enabled=lambda *_args, **_kwargs: True,
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
    )


def test_collect_startup_worker_specs_uses_declarative_provider_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_groups = _import_startup_worker_groups()
    provider_calls: list[str] = []

    def _provider(name: str):
        def _collect(_context: WorkerLifecycleContext):
            provider_calls.append(name)
            return ()

        return _collect

    providers = (
        _provider("primary"),
        _provider("study"),
        _provider("content"),
        _provider("sidecar"),
        _provider("notifications"),
        _provider("cleanup"),
        _provider("compactor"),
        _provider("claims"),
        _provider("usage"),
        _provider("llm-usage"),
        _provider("runtime"),
        _provider("optional"),
        _provider("auxiliary"),
        _provider("infra"),
        _provider("maintenance"),
        _provider("recurring"),
    )
    monkeypatch.setattr(
        startup_groups,
        "startup_worker_spec_providers",
        lambda: providers,
    )

    specs = startup_groups.collect_startup_worker_specs(_context())

    assert specs == ()
    assert provider_calls == [
        "primary",
        "study",
        "content",
        "sidecar",
        "notifications",
        "cleanup",
        "compactor",
        "claims",
        "usage",
        "llm-usage",
        "runtime",
        "optional",
        "auxiliary",
        "infra",
        "maintenance",
        "recurring",
    ]


def test_collect_startup_worker_specs_accepts_real_provider_graph() -> None:
    startup_groups = _import_startup_worker_groups()

    specs = startup_groups.collect_startup_worker_specs(_context())
    spec_names = {spec.name for spec in specs}
    expected_spec_names = {
        "admin_backup_jobs_task",
        "admin_backup_sched_task",
        "admin_byok_validation_jobs_task",
        "admin_maintenance_rotation_jobs_task",
        "audio_jobs_task",
        "audiobook_jobs_task",
        "authnz_scheduler",
        "chatbooks_cleanup",
        "claims_alerts_task",
        "claims_rebuild",
        "claims_review_metrics_task",
        "companion_reflection_jobs_task",
        "companion_reflection_sched_task",
        "connectors_jobs_task",
        "connectors_sync_sched_task",
        "core_jobs_task",
        "data_tables_jobs_task",
        "embeddings_compactor_task",
        "ephemeral_cleanup_task",
        "evals_abtest_jobs_task",
        "files_export_gc_task",
        "files_jobs_task",
        "ingestion_sources_cleanup",
        "jobs_crypto_rotate_task",
        "jobs_integrity_task",
        "jobs_metrics_reconcile_task",
        "jobs_metrics_task",
        "jobs_notifications_bridge_task",
        "jobs_prune_task",
        "jobs_webhooks_task",
        "kanban_activity_cleanup_scheduler",
        "kanban_purge_scheduler",
        "llamacpp_acquisition_jobs_task",
        "llm_usage_aggregator",
        "loop_lag_task",
        "media_ingest_heavy_jobs_task",
        "media_ingest_jobs_task",
        "meetings_webhook_dlq_task",
        "notifications_prune_task",
        "outputs_purge_task",
        "persona_visual_generation_task",
        "persona_visual_portability_task",
        "presentation_render_jobs_task",
        "privilege_snapshot_task",
        "prompt_studio_jobs_task",
        "quality_eval_task",
        "reading_digest_jobs_task",
        "reading_digest_sched_task",
        "recipe_run_jobs_task",
        "reminder_jobs_task",
        "reminders_sched_task",
        "storage_cleanup_service",
        "study_pack_jobs_task",
        "study_suggestions_jobs_task",
        "tts_history_cleanup_task",
        "usage_aggregator",
        "vn_asset_generation_jobs_task",
        "vn_asset_jobs_task",
        "websub_renewal_task",
        "workflows_dlq_task",
        "workflows_gc_task",
        "workflows_maint_task",
        "workflows_sched_task",
    }

    assert len(specs) == len(spec_names)
    assert spec_names == expected_spec_names


def test_startup_worker_groups_no_longer_exposes_legacy_group_start_api() -> None:
    startup_groups = _import_startup_worker_groups()

    assert not hasattr(startup_groups, "StartupWorkerGroupHandles")
    assert not hasattr(startup_groups, "start_worker_groups")
