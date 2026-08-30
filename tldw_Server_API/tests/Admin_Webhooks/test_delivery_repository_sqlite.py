from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest
import pytest_asyncio

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
    AdminWebhookRepository,
)
from tldw_Server_API.tests.Admin_Webhooks.test_event_expansion import (
    exercise_acknowledgement_second_step_rollback,
    exercise_atomic_disposition_acknowledgement,
    exercise_cancellation_cas_and_processing_preservation,
    exercise_capture_and_history,
    exercise_delivery_state_machine,
    exercise_disposition_scheduling_persistence,
    exercise_enqueue_recovery_contract,
    exercise_enqueue_terminal_orphan_recovery_contract,
    exercise_malformed_persisted_coordinates,
    exercise_persisted_coordinate_matrix,
    exercise_recovery_runtime_and_retention,
    exercise_stale_recovery_and_cancellation,
    exercise_task8_attempt_reservation_and_recovery_contract,
    exercise_task9_test_attempt_contract,
    exercise_task11_health_and_expiry_contract,
)
from tldw_Server_API.tests.Admin_Webhooks.test_redelivery_history_api import (
    exercise_history_loads_only_public_columns,
    exercise_history_projection_is_set_based_and_key_independent,
    exercise_history_reads_one_consistent_snapshot,
    exercise_redelivery_concurrency_and_commit_failure,
    exercise_redelivery_key_family_conflicts_across_sources,
    exercise_redelivery_preconditions_and_audit_rollback,
    exercise_redelivery_replay_ignores_hidden_history_coordinates,
    exercise_redelivery_replay_rows_have_exact_action_shape,
    exercise_redelivery_success_exact_replay_and_malformed_coordinate,
    exercise_single_history_item_loads_only_public_columns,
    exercise_single_history_item_reads_one_consistent_snapshot,
)
from tldw_Server_API.tests.Admin_Webhooks.test_test_delivery import (
    exercise_commit_failure_correlated_audit,
    exercise_concurrent_exact_test_start,
    exercise_post_start_semantic_and_rekey_races,
    exercise_prestart_rejections_are_no_io,
    exercise_processing_replay_and_conflict_precede_current_state,
    exercise_retry_class_terminalization_and_completion_audit_failure,
    exercise_start_races_and_accepted_audit_rollback,
    exercise_test_service_success_and_terminal_replay,
)


@dataclass
class SQLiteDeliveryRepositoryFixture:
    repository: AdminWebhookRepository
    pool: DatabasePool
    integrity_error = sqlite3.IntegrityError

    async def execute(self, query: str, *params: object) -> None:
        await self.pool.execute(query, *params)

    async def fetchval(self, query: str, *params: object) -> object:
        return await self.pool.fetchval(query, *params)

    async def fetchrow(self, query: str, *params: object) -> object:
        return await self.pool.fetchrow(query, *params)


@pytest_asyncio.fixture
async def delivery_repo(tmp_path: Path) -> SQLiteDeliveryRepositoryFixture:
    pool = DatabasePool(
        Settings(
            AUTH_MODE="single_user",
            DATABASE_URL=f"sqlite:///{tmp_path / 'delivery.db'}",
        )
    )
    await pool.initialize()
    fixture = SQLiteDeliveryRepositoryFixture(AdminWebhookRepository(pool), pool)
    try:
        yield fixture
    finally:
        await pool.close()


@pytest.mark.unit
async def test_sqlite_capture_fanout_and_history_contract(
    delivery_repo: SQLiteDeliveryRepositoryFixture,
) -> None:
    await exercise_capture_and_history(delivery_repo)


@pytest.mark.unit
async def test_sqlite_delivery_state_machine_contract(
    delivery_repo: SQLiteDeliveryRepositoryFixture,
) -> None:
    await exercise_delivery_state_machine(delivery_repo)


@pytest.mark.unit
async def test_sqlite_enqueue_recovery_contract(
    delivery_repo: SQLiteDeliveryRepositoryFixture,
) -> None:
    await exercise_enqueue_recovery_contract(delivery_repo)


@pytest.mark.unit
async def test_sqlite_enqueue_terminal_orphan_recovery_contract(
    delivery_repo: SQLiteDeliveryRepositoryFixture,
) -> None:
    await exercise_enqueue_terminal_orphan_recovery_contract(delivery_repo)


@pytest.mark.unit
async def test_sqlite_recovery_runtime_and_retention_contract(
    delivery_repo: SQLiteDeliveryRepositoryFixture,
) -> None:
    await exercise_recovery_runtime_and_retention(delivery_repo)


@pytest.mark.unit
async def test_sqlite_task11_health_and_expiry_contract(
    delivery_repo: SQLiteDeliveryRepositoryFixture,
) -> None:
    await exercise_task11_health_and_expiry_contract(delivery_repo)


@pytest.mark.unit
async def test_sqlite_stale_recovery_and_cancellation_contract(
    delivery_repo: SQLiteDeliveryRepositoryFixture,
) -> None:
    await exercise_stale_recovery_and_cancellation(delivery_repo)


@pytest.mark.unit
async def test_sqlite_cancellation_cas_and_processing_preservation_contract(
    delivery_repo: SQLiteDeliveryRepositoryFixture,
) -> None:
    await exercise_cancellation_cas_and_processing_preservation(delivery_repo)


@pytest.mark.unit
async def test_sqlite_atomic_disposition_acknowledgement_contract(
    delivery_repo: SQLiteDeliveryRepositoryFixture,
) -> None:
    await exercise_atomic_disposition_acknowledgement(delivery_repo)


@pytest.mark.unit
async def test_sqlite_malformed_persisted_coordinates_fail_closed(
    delivery_repo: SQLiteDeliveryRepositoryFixture,
) -> None:
    await exercise_malformed_persisted_coordinates(delivery_repo)


@pytest.mark.unit
async def test_sqlite_disposition_scheduling_persistence_contract(
    delivery_repo: SQLiteDeliveryRepositoryFixture,
) -> None:
    await exercise_disposition_scheduling_persistence(delivery_repo)


@pytest.mark.unit
async def test_sqlite_acknowledgement_second_step_rollback_contract(
    delivery_repo: SQLiteDeliveryRepositoryFixture,
) -> None:
    await exercise_acknowledgement_second_step_rollback(delivery_repo)


@pytest.mark.unit
async def test_sqlite_persisted_coordinate_matrix_contract(
    delivery_repo: SQLiteDeliveryRepositoryFixture,
) -> None:
    await exercise_persisted_coordinate_matrix(delivery_repo)


@pytest.mark.unit
async def test_sqlite_task8_attempt_reservation_and_recovery_contract(
    delivery_repo: SQLiteDeliveryRepositoryFixture,
) -> None:
    await exercise_task8_attempt_reservation_and_recovery_contract(delivery_repo)


@pytest.mark.unit
async def test_sqlite_task9_test_attempt_contract(
    delivery_repo: SQLiteDeliveryRepositoryFixture,
) -> None:
    await exercise_task9_test_attempt_contract(delivery_repo)


@pytest.mark.unit
async def test_sqlite_task9_service_success_and_terminal_replay(
    delivery_repo: SQLiteDeliveryRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await exercise_test_service_success_and_terminal_replay(delivery_repo, monkeypatch)


@pytest.mark.unit
async def test_sqlite_task9_processing_replay_and_conflict_ordering(
    delivery_repo: SQLiteDeliveryRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await exercise_processing_replay_and_conflict_precede_current_state(
        delivery_repo,
        monkeypatch,
    )


@pytest.mark.unit
async def test_sqlite_task9_retry_class_and_audit_failure_contract(
    delivery_repo: SQLiteDeliveryRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await exercise_retry_class_terminalization_and_completion_audit_failure(
        delivery_repo,
        monkeypatch,
    )


@pytest.mark.unit
async def test_sqlite_task9_start_race_and_audit_rollback_contract(
    delivery_repo: SQLiteDeliveryRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await exercise_start_races_and_accepted_audit_rollback(
        delivery_repo,
        monkeypatch,
    )


@pytest.mark.unit
async def test_sqlite_task9_concurrent_exact_start_contract(
    delivery_repo: SQLiteDeliveryRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await exercise_concurrent_exact_test_start(delivery_repo, monkeypatch)


@pytest.mark.unit
async def test_sqlite_task9_prestart_rejections_are_no_io(
    delivery_repo: SQLiteDeliveryRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await exercise_prestart_rejections_are_no_io(delivery_repo, monkeypatch)


@pytest.mark.unit
async def test_sqlite_task9_post_start_semantic_and_rekey_races(
    delivery_repo: SQLiteDeliveryRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await exercise_post_start_semantic_and_rekey_races(delivery_repo, monkeypatch)


@pytest.mark.unit
async def test_sqlite_task9_commit_failure_correlated_audit(
    delivery_repo: SQLiteDeliveryRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await exercise_commit_failure_correlated_audit(delivery_repo, monkeypatch)


@pytest.mark.unit
async def test_sqlite_task10_history_projection_contract(
    delivery_repo: SQLiteDeliveryRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await exercise_history_projection_is_set_based_and_key_independent(
        delivery_repo,
        monkeypatch,
    )


@pytest.mark.unit
async def test_sqlite_task10_history_loads_only_public_columns(
    delivery_repo: SQLiteDeliveryRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await exercise_history_loads_only_public_columns(delivery_repo, monkeypatch)


@pytest.mark.unit
async def test_sqlite_task10_history_uses_one_consistent_snapshot(
    delivery_repo: SQLiteDeliveryRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await exercise_history_reads_one_consistent_snapshot(delivery_repo, monkeypatch)


@pytest.mark.unit
async def test_sqlite_task10_single_history_item_loads_only_public_columns(
    delivery_repo: SQLiteDeliveryRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await exercise_single_history_item_loads_only_public_columns(
        delivery_repo,
        monkeypatch,
    )


@pytest.mark.unit
async def test_sqlite_task10_redelivery_replay_ignores_hidden_history_coordinates(
    delivery_repo: SQLiteDeliveryRepositoryFixture,
) -> None:
    await exercise_redelivery_replay_ignores_hidden_history_coordinates(delivery_repo)


@pytest.mark.unit
async def test_sqlite_task10_single_history_item_uses_one_consistent_snapshot(
    delivery_repo: SQLiteDeliveryRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await exercise_single_history_item_reads_one_consistent_snapshot(
        delivery_repo,
        monkeypatch,
    )


@pytest.mark.unit
async def test_sqlite_task10_redelivery_success_and_replay_contract(
    delivery_repo: SQLiteDeliveryRepositoryFixture,
) -> None:
    await exercise_redelivery_success_exact_replay_and_malformed_coordinate(
        delivery_repo
    )


@pytest.mark.unit
async def test_sqlite_task10_redelivery_key_family_conflict_contract(
    delivery_repo: SQLiteDeliveryRepositoryFixture,
) -> None:
    await exercise_redelivery_key_family_conflicts_across_sources(delivery_repo)


@pytest.mark.unit
async def test_sqlite_task10_redelivery_exact_replay_row_contract(
    delivery_repo: SQLiteDeliveryRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await exercise_redelivery_replay_rows_have_exact_action_shape(
        delivery_repo,
        monkeypatch,
    )


@pytest.mark.unit
async def test_sqlite_task10_redelivery_preconditions_and_rollback_contract(
    delivery_repo: SQLiteDeliveryRepositoryFixture,
) -> None:
    await exercise_redelivery_preconditions_and_audit_rollback(delivery_repo)


@pytest.mark.unit
async def test_sqlite_task10_redelivery_concurrency_and_commit_failure_contract(
    delivery_repo: SQLiteDeliveryRepositoryFixture,
) -> None:
    await exercise_redelivery_concurrency_and_commit_failure(delivery_repo)
