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
        runtime_field="claims_task",
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
        runtime_field="embeddings_compactor_task",
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
        runtime_field="websub_renewal_task",
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
        runtime_field="usage_task",
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
        runtime_field="llm_usage_task",
        stopped_name_key="llm_usage_aggregator",
        legacy_helper="shutdown_usage_aggregators",
        category="usage",
        shutdown_phase="background_worker_shutdown",
        target_state="registry-owned; legacy stop helper removed",
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
def test_cleanup_rows_no_longer_advertise_runtime_shutdown_fields() -> None:
    cleanup_worker_names = {
        "ephemeral_cleanup_task",
        "chatbooks_cleanup",
        "storage_cleanup_service",
    }

    assert {
        row.managed_name: row.runtime_field
        for row in WORKER_OWNERSHIP_MATRIX
        if row.managed_name in cleanup_worker_names
    } == {
        "ephemeral_cleanup_task": None,
        "chatbooks_cleanup": None,
        "storage_cleanup_service": None,
    }
