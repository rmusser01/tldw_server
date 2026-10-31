"""Live PostgreSQL transaction contracts for dormant Notes task readiness."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import (
    DatabaseBackendFactory,
)
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.models import SyncDatasetCreate
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

pytestmark = pytest.mark.integration

_TASK_CURSOR_1 = "00000000-0000-4000-8000-000000000001"
_TASK_CURSOR_2 = "00000000-0000-4000-8000-000000000002"


def _postgres_store(config: DatabaseConfig) -> SyncV2Store:
    return SyncV2Store(
        SyncDatabase(backend=DatabaseBackendFactory.create_backend(config))
    )


def _close_postgres_store(store: SyncV2Store) -> None:
    store.db.backend.get_pool().close_all()


def _enroll_both_domains(
    store: SyncV2Store,
    dataset_id: str,
    *,
    capture_enabled: bool,
) -> None:
    store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id=dataset_id,
            owner_user_id="user-1",
            domains=[
                "notes.note",
                "chat.conversation",
                "chat.message",
                "attachment.ref",
            ],
        )
    )
    store.transition_notes_task_readiness(
        dataset_id,
        owner_user_id="user-1",
        expected_state="not_enrolled",
        state="enrolling",
        source_dataset_id=dataset_id,
        source_cursor=None,
        source_count=0,
        source_fingerprint=None,
    )
    store.transition_notes_task_activity_readiness(
        dataset_id,
        owner_user_id="user-1",
        expected_state="not_enrolled",
        state="enrolling",
        source_dataset_id=dataset_id,
        source_cursor=None,
        source_count=0,
        source_fingerprint=None,
        task_activity_capture_enabled=True if capture_enabled else None,
    )


def test_postgres_readiness_row_lock_serializes_competing_compare_and_set(
    pg_database_config: DatabaseConfig,
) -> None:
    setup = _postgres_store(pg_database_config)
    contender_a = _postgres_store(pg_database_config)
    contender_b = _postgres_store(pg_database_config)
    dataset_id = "dataset-readiness-cas"
    _enroll_both_domains(setup, dataset_id, capture_enabled=True)
    barrier = threading.Barrier(2)

    def transition(
        store: SyncV2Store,
        cursor: str,
        fingerprint: str,
    ) -> tuple[str, str]:
        barrier.wait(timeout=30)
        try:
            updated = store.transition_notes_task_readiness(
                dataset_id,
                owner_user_id="user-1",
                expected_state="enrolling",
                state="bootstrapping",
                source_dataset_id=dataset_id,
                source_cursor=cursor,
                source_count=1,
                source_fingerprint=fingerprint,
            )
        except SyncStoreError as exc:
            return "error", str(exc)
        return "ok", str(updated.metadata["notes_task_v1"]["source_cursor"])

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(transition, contender_a, _TASK_CURSOR_1, "a" * 64),
                executor.submit(transition, contender_b, _TASK_CURSOR_2, "b" * 64),
            ]
            results = [future.result(timeout=60) for future in futures]

        winners = [detail for status, detail in results if status == "ok"]
        losers = [detail for status, detail in results if status == "error"]
        stored = setup.get_dataset(dataset_id, owner_user_id="user-1")

        assert len(winners) == 1
        assert losers == ["notes_task_readiness_compare_and_set_failed"]
        assert stored is not None
        assert stored.metadata["notes_task_v1"]["source_cursor"] == winners[0]
        assert stored.metadata["notes_task_activity_v1"]["state"] == "enrolling"
        assert stored.metadata["task_activity_capture_enabled"] is True
    finally:
        _close_postgres_store(contender_b)
        _close_postgres_store(contender_a)
        _close_postgres_store(setup)


def test_postgres_readiness_and_capture_update_roll_back_atomically(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutator = _postgres_store(pg_database_config)
    observer = _postgres_store(pg_database_config)
    dataset_id = "dataset-readiness-rollback"
    _enroll_both_domains(mutator, dataset_id, capture_enabled=False)

    def fail_after_update(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("forced PostgreSQL post-update failure")

    try:
        monkeypatch.setattr(mutator.db, "_get_dataset_row", fail_after_update)
        with pytest.raises(RuntimeError, match="forced PostgreSQL post-update failure"):
            mutator.transition_notes_task_activity_readiness(
                dataset_id,
                owner_user_id="user-1",
                expected_state="enrolling",
                state="bootstrapping",
                source_dataset_id=dataset_id,
                source_cursor=None,
                source_count=0,
                source_fingerprint=None,
                task_activity_capture_enabled=True,
            )

        stored = observer.get_dataset(dataset_id, owner_user_id="user-1")
        assert stored is not None
        assert stored.metadata["notes_task_v1"]["state"] == "enrolling"
        assert stored.metadata["notes_task_activity_v1"]["state"] == "enrolling"
        assert stored.metadata.get("task_activity_capture_enabled") is not True
    finally:
        _close_postgres_store(observer)
        _close_postgres_store(mutator)
