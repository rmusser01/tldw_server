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
