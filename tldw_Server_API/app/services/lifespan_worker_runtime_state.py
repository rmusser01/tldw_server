"""
Worker/task runtime state used by the application lifespan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LifespanWorkerRuntimeState:
    """Mutable worker/task runtime state shared across startup and shutdown."""

    worker_inventory: Any | None = None
    owned_job_pollers: list[Any] = field(default_factory=list)
    core_jobs_stop_event: Any | None = None
    core_jobs_task: Any | None = None
    files_jobs_stop_event: Any | None = None
    files_jobs_task: Any | None = None
    data_tables_jobs_stop_event: Any | None = None
    data_tables_jobs_task: Any | None = None
    prompt_studio_jobs_stop_event: Any | None = None
    prompt_studio_jobs_task: Any | None = None
    study_pack_jobs_stop_event: Any | None = None
    study_pack_jobs_task: Any | None = None
    study_suggestions_jobs_stop_event: Any | None = None
    study_suggestions_jobs_task: Any | None = None
    privilege_snapshot_stop_event: Any | None = None
    privilege_snapshot_task: Any | None = None
    audio_jobs_stop_event: Any | None = None
    audio_jobs_task: Any | None = None
    audiobook_jobs_stop_event: Any | None = None
    audiobook_jobs_task: Any | None = None
    presentation_render_jobs_stop_event: Any | None = None
    presentation_render_jobs_task: Any | None = None
    media_ingest_jobs_stop_event: Any | None = None
    media_ingest_jobs_task: Any | None = None
    media_ingest_heavy_jobs_stop_event: Any | None = None
    media_ingest_heavy_jobs_task: Any | None = None
    reading_digest_jobs_stop_event: Any | None = None
    reading_digest_jobs_task: Any | None = None
    vn_asset_jobs_stop_event: Any | None = None
    vn_asset_jobs_task: Any | None = None
    vn_asset_generation_jobs_stop_event: Any | None = None
    vn_asset_generation_jobs_task: Any | None = None
    companion_reflection_jobs_stop_event: Any | None = None
    companion_reflection_jobs_task: Any | None = None
    reminder_jobs_stop_event: Any | None = None
    reminder_jobs_task: Any | None = None
    admin_backup_jobs_stop_event: Any | None = None
    admin_backup_jobs_task: Any | None = None
    admin_byok_validation_jobs_stop_event: Any | None = None
    admin_byok_validation_jobs_task: Any | None = None
    admin_maintenance_rotation_jobs_stop_event: Any | None = None
    admin_maintenance_rotation_jobs_task: Any | None = None
    recipe_run_jobs_stop_event: Any | None = None
    recipe_run_jobs_task: Any | None = None
    jobs_notifications_bridge_task: Any | None = None
    evals_abtest_jobs_stop_event: Any | None = None
    evals_abtest_jobs_task: Any | None = None
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

    def apply_startup_worker_bootstrap_handles(self, handles: Any) -> None:
        self.worker_inventory = getattr(handles, "worker_inventory", None)
        self.owned_job_pollers = list(getattr(handles, "owned_job_pollers", []) or [])
        _copy_known_fields(self, getattr(handles, "startup_worker_group_handles", None))
        _copy_known_fields(self, getattr(handles, "startup_service_tail_handles", None))

    def apply_pre_worker_cleanup_handles(self, handles: Any) -> None:
        _copy_known_fields(self, handles)

    def apply_primary_late_stop_worker_handles(self, handles: Any) -> None:
        _copy_known_fields(self, handles)

    def apply_grouped_late_stop_worker_handles(self, handles: Any) -> None:
        _copy_known_fields(self, handles)

    def apply_post_worker_shutdown_handles(self, handles: Any) -> None:
        _copy_known_fields(self, handles)

    def apply_final_cleanup_handles(self, handles: Any) -> None:
        _copy_known_fields(self, handles)


def _copy_known_fields(runtime_state: LifespanWorkerRuntimeState, handles: Any) -> None:
    if handles is None:
        return
    try:
        items = vars(handles).items()
    except TypeError:
        return
    known_fields = runtime_state.__dataclass_fields__
    for name, value in items:
        if name in known_fields:
            setattr(runtime_state, name, value)
