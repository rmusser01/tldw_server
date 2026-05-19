from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class WorkerOwnershipRow:
    issue_name: str
    managed_name: str
    task_name: str | None
    runtime_field: str | None
    stopped_name_key: str
    legacy_helper: str
    category: str
    shutdown_phase: str
    target_state: str


WORKER_OWNERSHIP_MATRIX = (
    WorkerOwnershipRow(
        issue_name="Ephemeral cleanup loop",
        managed_name="ephemeral_cleanup_task",
        task_name="ephemeral_cleanup_task",
        runtime_field=None,
        stopped_name_key="ephemeral_cleanup_task",
        legacy_helper="shutdown_pre_worker_cleanup",
        category="cleanup",
        shutdown_phase="background_worker_shutdown",
        target_state="registry-owned; direct cancel removed; finalizers retained",
    ),
    WorkerOwnershipRow(
        issue_name="Chatbooks cleanup",
        managed_name="chatbooks_cleanup",
        task_name="chatbooks_cleanup_task",
        runtime_field=None,
        stopped_name_key="chatbooks_cleanup",
        legacy_helper="shutdown_pre_worker_cleanup",
        category="cleanup",
        shutdown_phase="background_worker_shutdown",
        target_state="registry-owned; direct stop removed",
    ),
    WorkerOwnershipRow(
        issue_name="Storage cleanup service",
        managed_name="storage_cleanup_service",
        task_name=None,
        runtime_field=None,
        stopped_name_key="storage_cleanup_service",
        legacy_helper="shutdown_pre_worker_cleanup",
        category="cleanup",
        shutdown_phase="background_worker_shutdown",
        target_state="registry-owned stop callback; singleton reset finalizer retained",
    ),
    WorkerOwnershipRow(
        issue_name="Claims rebuild",
        managed_name="claims_rebuild",
        task_name="claims_task",
        runtime_field=None,
        stopped_name_key="claims_rebuild",
        legacy_helper="shutdown_claims_maintenance_tasks",
        category="claims",
        shutdown_phase="background_worker_shutdown",
        target_state="registry-owned; direct cancel removed",
    ),
    WorkerOwnershipRow(
        issue_name="Embeddings compactor",
        managed_name="embeddings_compactor_task",
        task_name="embeddings_compactor_task",
        runtime_field=None,
        stopped_name_key="embeddings_compactor_task",
        legacy_helper="shutdown_notifications_compactor_websub_workers",
        category="embeddings",
        shutdown_phase="background_worker_shutdown",
        target_state="registry-owned; direct stop-event cancellation removed",
    ),
    WorkerOwnershipRow(
        issue_name="WebSub renewal",
        managed_name="websub_renewal_task",
        task_name="websub_renewal_task",
        runtime_field=None,
        stopped_name_key="websub_renewal_task",
        legacy_helper="shutdown_notifications_compactor_websub_workers",
        category="collections-websub",
        shutdown_phase="background_worker_shutdown",
        target_state="registry-owned; direct task cancellation removed",
    ),
    WorkerOwnershipRow(
        issue_name="Usage aggregator",
        managed_name="usage_aggregator",
        task_name="usage_aggregator",
        runtime_field=None,
        stopped_name_key="usage_aggregator",
        legacy_helper="shutdown_usage_aggregators",
        category="usage",
        shutdown_phase="background_worker_shutdown",
        target_state="registry-owned; legacy stop helper removed",
    ),
    WorkerOwnershipRow(
        issue_name="LLM usage aggregator",
        managed_name="llm_usage_aggregator",
        task_name="llm_usage_aggregator",
        runtime_field=None,
        stopped_name_key="llm_usage_aggregator",
        legacy_helper="shutdown_usage_aggregators",
        category="usage",
        shutdown_phase="background_worker_shutdown",
        target_state="registry-owned; legacy stop helper removed",
    ),
    WorkerOwnershipRow(
        issue_name="AuthNZ scheduler",
        managed_name="authnz_scheduler",
        task_name=None,
        runtime_field=None,
        stopped_name_key="authnz_scheduler",
        legacy_helper="shutdown_authnz_scheduler",
        category="recurring-scheduler",
        shutdown_phase="background_worker_shutdown",
        target_state="registry-owned callback; duplicate direct stop removed",
    ),
    WorkerOwnershipRow(
        issue_name="Workflows scheduler",
        managed_name="workflows_sched_task",
        task_name="workflows_recurring_scheduler",
        runtime_field=None,
        stopped_name_key="workflows_sched_task",
        legacy_helper="shutdown_recurring_schedulers",
        category="recurring-scheduler",
        shutdown_phase="background_worker_shutdown",
        target_state="registry-owned callback; duplicate direct stop removed",
    ),
    WorkerOwnershipRow(
        issue_name="Reading digest scheduler",
        managed_name="reading_digest_sched_task",
        task_name="reading_digest_scheduler",
        runtime_field=None,
        stopped_name_key="reading_digest_sched_task",
        legacy_helper="shutdown_recurring_schedulers",
        category="recurring-scheduler",
        shutdown_phase="background_worker_shutdown",
        target_state="registry-owned callback; duplicate direct stop removed",
    ),
    WorkerOwnershipRow(
        issue_name="Admin backup scheduler",
        managed_name="admin_backup_sched_task",
        task_name="admin_backup_scheduler",
        runtime_field=None,
        stopped_name_key="admin_backup_sched_task",
        legacy_helper="shutdown_recurring_schedulers",
        category="recurring-scheduler",
        shutdown_phase="background_worker_shutdown",
        target_state="registry-owned callback; duplicate direct stop removed",
    ),
    WorkerOwnershipRow(
        issue_name="Companion reflection scheduler",
        managed_name="companion_reflection_sched_task",
        task_name="companion_reflection_scheduler",
        runtime_field=None,
        stopped_name_key="companion_reflection_sched_task",
        legacy_helper="shutdown_recurring_schedulers",
        category="recurring-scheduler",
        shutdown_phase="background_worker_shutdown",
        target_state="registry-owned callback; duplicate direct stop removed",
    ),
    WorkerOwnershipRow(
        issue_name="Reminders scheduler",
        managed_name="reminders_sched_task",
        task_name="reminders_scheduler",
        runtime_field=None,
        stopped_name_key="reminders_sched_task",
        legacy_helper="shutdown_recurring_schedulers",
        category="recurring-scheduler",
        shutdown_phase="background_worker_shutdown",
        target_state="registry-owned callback; duplicate direct stop removed",
    ),
    WorkerOwnershipRow(
        issue_name="Connectors sync scheduler",
        managed_name="connectors_sync_sched_task",
        task_name="connectors_sync_scheduler",
        runtime_field=None,
        stopped_name_key="connectors_sync_sched_task",
        legacy_helper="shutdown_recurring_schedulers",
        category="recurring-scheduler",
        shutdown_phase="background_worker_shutdown",
        target_state="registry-owned callback; duplicate direct stop removed",
    ),
    WorkerOwnershipRow(
        issue_name="Quality eval scheduler",
        managed_name="quality_eval_task",
        task_name="rag_quality_eval_scheduler",
        runtime_field=None,
        stopped_name_key="quality_eval_task",
        legacy_helper="none",
        category="maintenance",
        shutdown_phase="background_worker_shutdown",
        target_state="registry-owned; no legacy direct-stop path",
    ),
    WorkerOwnershipRow(
        issue_name="Outputs purge scheduler",
        managed_name="outputs_purge_task",
        task_name="outputs_purge_scheduler",
        runtime_field=None,
        stopped_name_key="outputs_purge_task",
        legacy_helper="none",
        category="maintenance",
        shutdown_phase="background_worker_shutdown",
        target_state="registry-owned; no legacy direct-stop path",
    ),
    WorkerOwnershipRow(
        issue_name="Kanban activity cleanup scheduler",
        managed_name="kanban_activity_cleanup_scheduler",
        task_name="kanban_activity_cleanup_scheduler",
        runtime_field=None,
        stopped_name_key="kanban_activity_cleanup_scheduler",
        legacy_helper="none",
        category="maintenance",
        shutdown_phase="background_worker_shutdown",
        target_state="registry-owned; no legacy direct-stop path",
    ),
    WorkerOwnershipRow(
        issue_name="Ingestion sources cleanup scheduler",
        managed_name="ingestion_sources_cleanup",
        task_name="ingestion_sources_cleanup_task",
        runtime_field=None,
        stopped_name_key="ingestion_sources_cleanup",
        legacy_helper="none",
        category="maintenance",
        shutdown_phase="background_worker_shutdown",
        target_state="registry-owned stop-event worker; no legacy direct-stop path",
    ),
    WorkerOwnershipRow(
        issue_name="Kanban purge scheduler",
        managed_name="kanban_purge_scheduler",
        task_name="kanban_purge_scheduler",
        runtime_field=None,
        stopped_name_key="kanban_purge_scheduler",
        legacy_helper="none",
        category="maintenance",
        shutdown_phase="background_worker_shutdown",
        target_state="registry-owned; no legacy direct-stop path",
    ),
    WorkerOwnershipRow(
        issue_name="File artifacts export GC scheduler",
        managed_name="files_export_gc_task",
        task_name="file_artifacts_export_gc",
        runtime_field=None,
        stopped_name_key="files_export_gc_task",
        legacy_helper="shutdown_claims_maintenance_tasks",
        category="maintenance",
        shutdown_phase="background_worker_shutdown",
        target_state="registry-owned; duplicate direct cancel removed",
    ),
    WorkerOwnershipRow(
        issue_name="Notifications prune scheduler",
        managed_name="notifications_prune_task",
        task_name="notifications_prune_scheduler",
        runtime_field=None,
        stopped_name_key="notifications_prune_task",
        legacy_helper="shutdown_claims_maintenance_tasks",
        category="maintenance",
        shutdown_phase="background_worker_shutdown",
        target_state="registry-owned; duplicate direct cancel removed",
    ),
    WorkerOwnershipRow(
        issue_name="Jobs prune scheduler",
        managed_name="jobs_prune_task",
        task_name="jobs_prune_scheduler",
        runtime_field=None,
        stopped_name_key="jobs_prune_task",
        legacy_helper="shutdown_claims_maintenance_tasks",
        category="maintenance",
        shutdown_phase="background_worker_shutdown",
        target_state="registry-owned; duplicate direct cancel removed",
    ),
)


@pytest.mark.unit
def test_worker_ownership_matrix_covers_phase_2_deletion_candidates() -> None:
    assert {row.managed_name for row in WORKER_OWNERSHIP_MATRIX} == {
        "ephemeral_cleanup_task",
        "chatbooks_cleanup",
        "storage_cleanup_service",
        "claims_rebuild",
        "embeddings_compactor_task",
        "websub_renewal_task",
        "usage_aggregator",
        "llm_usage_aggregator",
        "authnz_scheduler",
        "workflows_sched_task",
        "reading_digest_sched_task",
        "admin_backup_sched_task",
        "companion_reflection_sched_task",
        "reminders_sched_task",
        "connectors_sync_sched_task",
        "quality_eval_task",
        "outputs_purge_task",
        "kanban_activity_cleanup_scheduler",
        "ingestion_sources_cleanup",
        "kanban_purge_scheduler",
        "files_export_gc_task",
        "notifications_prune_task",
        "jobs_prune_task",
    }


@pytest.mark.unit
def test_worker_ownership_matrix_has_unique_managed_names() -> None:
    names = [row.managed_name for row in WORKER_OWNERSHIP_MATRIX]
    assert len(names) == len(set(names))


@pytest.mark.unit
def test_worker_ownership_matrix_rows_have_operational_ownership_fields() -> None:
    for row in WORKER_OWNERSHIP_MATRIX:
        assert row.issue_name
        assert row.managed_name
        assert row.stopped_name_key
        assert row.legacy_helper
        assert row.category
        assert row.shutdown_phase == "background_worker_shutdown"
        assert row.target_state


@pytest.mark.unit
def test_removed_direct_stop_rows_no_longer_advertise_runtime_shutdown_fields() -> None:
    removed_direct_stop_worker_names = {
        "ephemeral_cleanup_task",
        "chatbooks_cleanup",
        "storage_cleanup_service",
        "claims_rebuild",
        "embeddings_compactor_task",
        "websub_renewal_task",
        "usage_aggregator",
        "llm_usage_aggregator",
        "authnz_scheduler",
        "workflows_sched_task",
        "reading_digest_sched_task",
        "admin_backup_sched_task",
        "companion_reflection_sched_task",
        "reminders_sched_task",
        "connectors_sync_sched_task",
        "quality_eval_task",
        "outputs_purge_task",
        "kanban_activity_cleanup_scheduler",
        "ingestion_sources_cleanup",
        "kanban_purge_scheduler",
        "files_export_gc_task",
        "notifications_prune_task",
        "jobs_prune_task",
    }

    assert {
        row.managed_name: row.runtime_field
        for row in WORKER_OWNERSHIP_MATRIX
        if row.managed_name in removed_direct_stop_worker_names
    } == dict.fromkeys(removed_direct_stop_worker_names)
