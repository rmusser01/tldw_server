from __future__ import annotations

from dataclasses import dataclass

import asyncpg
import pytest
import pytest_asyncio

from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
    ensure_admin_webhook_canonical_tables_pg,
)
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
    exercise_registration_counts_contract,
    exercise_stale_recovery_and_cancellation,
    exercise_task8_attempt_reservation_and_recovery_contract,
    exercise_task9_test_attempt_contract,
    exercise_task11_future_heartbeat_contract,
    exercise_task11_health_and_expiry_contract,
)
from tldw_Server_API.tests.Admin_Webhooks.test_redelivery_history_api import (
    exercise_history_loads_only_public_columns,
    exercise_history_projection_is_set_based_and_key_independent,
    exercise_history_reads_one_consistent_snapshot,
    exercise_redelivery_accepts_canonical_historical_event_body,
    exercise_redelivery_concurrency_and_commit_failure,
    exercise_redelivery_key_family_conflicts_across_sources,
    exercise_redelivery_preconditions_and_audit_rollback,
    exercise_redelivery_rejects_noncanonical_event_body,
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

pytest_plugins = ("tldw_Server_API.tests.AuthNZ.conftest",)
pytestmark = pytest.mark.postgres


@dataclass
class PostgreSQLDeliveryRepositoryFixture:
    repository: AdminWebhookRepository
    pool: object
    integrity_error = asyncpg.IntegrityConstraintViolationError

    async def execute(self, query: str, *params: object) -> None:
        await self.pool.execute(query, *params)  # type: ignore[attr-defined]

    async def fetchval(self, query: str, *params: object) -> object:
        return await self.pool.fetchval(query, *params)  # type: ignore[attr-defined]

    async def fetchrow(self, query: str, *params: object) -> object:
        return await self.pool.fetchrow(query, *params)  # type: ignore[attr-defined]


@pytest_asyncio.fixture
async def delivery_repo(test_db_pool) -> PostgreSQLDeliveryRepositoryFixture:
    assert await ensure_admin_webhook_canonical_tables_pg(test_db_pool)
    connection = await asyncpg.connect(test_db_pool.settings.DATABASE_URL)
    try:
        await connection.execute(
            """
            TRUNCATE TABLE
                admin_webhook_delivery_attempts,
                admin_webhook_deliveries,
                admin_webhook_events,
                admin_webhook_idempotency,
                admin_webhook_runtime_heartbeats,
                admin_webhook_registrations,
                admin_webhook_sequences,
                admin_webhook_migration_state
            RESTART IDENTITY CASCADE
            """
        )
    finally:
        await connection.close()
    await test_db_pool.execute(
        "INSERT INTO admin_webhook_sequences (name, next_value) VALUES (?, ?)",
        "registration",
        1,
    )
    await test_db_pool.execute(
        """
        INSERT INTO admin_webhook_migration_state (
            singleton_id, schema_version, state_revision, phase
        ) VALUES (?, ?, ?, ?)
        """,
        1,
        1,
        1,
        "migration_pending",
    )
    yield PostgreSQLDeliveryRepositoryFixture(
        AdminWebhookRepository(test_db_pool),
        test_db_pool,
    )


@pytest.mark.integration
async def test_postgres_capture_fanout_and_history_contract(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
) -> None:
    await exercise_capture_and_history(delivery_repo)


@pytest.mark.integration
async def test_postgres_delivery_state_machine_contract(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
) -> None:
    await exercise_delivery_state_machine(delivery_repo)


@pytest.mark.integration
async def test_postgres_enqueue_recovery_contract(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
) -> None:
    await exercise_enqueue_recovery_contract(delivery_repo)


@pytest.mark.integration
async def test_postgres_enqueue_terminal_orphan_recovery_contract(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
) -> None:
    await exercise_enqueue_terminal_orphan_recovery_contract(delivery_repo)


@pytest.mark.integration
async def test_postgres_recovery_runtime_and_retention_contract(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
) -> None:
    await exercise_recovery_runtime_and_retention(delivery_repo)


@pytest.mark.integration
async def test_postgres_task11_health_and_expiry_contract(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
) -> None:
    await exercise_task11_health_and_expiry_contract(delivery_repo)


@pytest.mark.integration
async def test_postgres_registration_counts_contract(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
) -> None:
    await exercise_registration_counts_contract(delivery_repo)


@pytest.mark.integration
async def test_postgres_task11_future_heartbeat_contract(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
) -> None:
    await exercise_task11_future_heartbeat_contract(delivery_repo)


@pytest.mark.integration
async def test_postgres_stale_recovery_and_cancellation_contract(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
) -> None:
    await exercise_stale_recovery_and_cancellation(delivery_repo)


@pytest.mark.integration
async def test_postgres_cancellation_cas_and_processing_preservation_contract(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
) -> None:
    await exercise_cancellation_cas_and_processing_preservation(delivery_repo)


@pytest.mark.integration
async def test_postgres_atomic_disposition_acknowledgement_contract(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
) -> None:
    await exercise_atomic_disposition_acknowledgement(delivery_repo)


@pytest.mark.integration
async def test_postgres_malformed_persisted_coordinates_fail_closed(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
) -> None:
    await exercise_malformed_persisted_coordinates(delivery_repo)


@pytest.mark.integration
async def test_postgres_disposition_scheduling_persistence_contract(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
) -> None:
    await exercise_disposition_scheduling_persistence(delivery_repo)


@pytest.mark.integration
async def test_postgres_acknowledgement_second_step_rollback_contract(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
) -> None:
    await exercise_acknowledgement_second_step_rollback(delivery_repo)


@pytest.mark.integration
async def test_postgres_persisted_coordinate_matrix_contract(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
) -> None:
    await exercise_persisted_coordinate_matrix(delivery_repo)


@pytest.mark.integration
async def test_postgres_task8_attempt_reservation_and_recovery_contract(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
) -> None:
    await exercise_task8_attempt_reservation_and_recovery_contract(delivery_repo)


@pytest.mark.integration
async def test_postgres_task9_test_attempt_contract(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
) -> None:
    await exercise_task9_test_attempt_contract(delivery_repo)


@pytest.mark.integration
async def test_postgres_task9_service_success_and_terminal_replay(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await exercise_test_service_success_and_terminal_replay(delivery_repo, monkeypatch)


@pytest.mark.integration
async def test_postgres_task9_processing_replay_and_conflict_ordering(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await exercise_processing_replay_and_conflict_precede_current_state(
        delivery_repo,
        monkeypatch,
    )


@pytest.mark.integration
async def test_postgres_task9_retry_class_and_audit_failure_contract(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await exercise_retry_class_terminalization_and_completion_audit_failure(
        delivery_repo,
        monkeypatch,
    )


@pytest.mark.integration
async def test_postgres_task9_start_race_and_audit_rollback_contract(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await exercise_start_races_and_accepted_audit_rollback(
        delivery_repo,
        monkeypatch,
    )


@pytest.mark.integration
async def test_postgres_task9_concurrent_exact_start_contract(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await exercise_concurrent_exact_test_start(delivery_repo, monkeypatch)


@pytest.mark.integration
async def test_postgres_task9_prestart_rejections_are_no_io(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await exercise_prestart_rejections_are_no_io(delivery_repo, monkeypatch)


@pytest.mark.integration
async def test_postgres_task9_post_start_semantic_and_rekey_races(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await exercise_post_start_semantic_and_rekey_races(delivery_repo, monkeypatch)


@pytest.mark.integration
async def test_postgres_task9_commit_failure_correlated_audit(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await exercise_commit_failure_correlated_audit(delivery_repo, monkeypatch)


@pytest.mark.integration
async def test_postgres_task10_history_projection_contract(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await exercise_history_projection_is_set_based_and_key_independent(
        delivery_repo,
        monkeypatch,
    )


@pytest.mark.integration
async def test_postgres_task10_history_loads_only_public_columns(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await exercise_history_loads_only_public_columns(delivery_repo, monkeypatch)


@pytest.mark.integration
async def test_postgres_task10_history_uses_one_consistent_snapshot(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await exercise_history_reads_one_consistent_snapshot(delivery_repo, monkeypatch)


@pytest.mark.integration
async def test_postgres_task10_single_history_item_loads_only_public_columns(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await exercise_single_history_item_loads_only_public_columns(
        delivery_repo,
        monkeypatch,
    )


@pytest.mark.integration
async def test_postgres_task10_redelivery_replay_ignores_hidden_history_coordinates(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
) -> None:
    await exercise_redelivery_replay_ignores_hidden_history_coordinates(delivery_repo)


@pytest.mark.integration
async def test_postgres_task10_single_history_item_uses_one_consistent_snapshot(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await exercise_single_history_item_reads_one_consistent_snapshot(
        delivery_repo,
        monkeypatch,
    )


@pytest.mark.integration
async def test_postgres_task10_redelivery_success_and_replay_contract(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
) -> None:
    await exercise_redelivery_success_exact_replay_and_malformed_coordinate(
        delivery_repo
    )


@pytest.mark.integration
async def test_postgres_task10_redelivery_rejects_noncanonical_event_body(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
) -> None:
    await exercise_redelivery_rejects_noncanonical_event_body(delivery_repo)


@pytest.mark.integration
async def test_postgres_task10_redelivery_accepts_canonical_historical_event_body(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
) -> None:
    await exercise_redelivery_accepts_canonical_historical_event_body(delivery_repo)


@pytest.mark.integration
async def test_postgres_task10_redelivery_key_family_conflict_contract(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
) -> None:
    await exercise_redelivery_key_family_conflicts_across_sources(delivery_repo)


@pytest.mark.integration
async def test_postgres_task10_redelivery_exact_replay_row_contract(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await exercise_redelivery_replay_rows_have_exact_action_shape(
        delivery_repo,
        monkeypatch,
    )


@pytest.mark.integration
async def test_postgres_task10_redelivery_preconditions_and_rollback_contract(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
) -> None:
    await exercise_redelivery_preconditions_and_audit_rollback(delivery_repo)


@pytest.mark.integration
async def test_postgres_task10_redelivery_concurrency_and_commit_failure_contract(
    delivery_repo: PostgreSQLDeliveryRepositoryFixture,
) -> None:
    await exercise_redelivery_concurrency_and_commit_failure(delivery_repo)
