"""Actual default-dataset creation must fence prior local review authority."""

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.chacha.note_graph_suggestion_store import NotesGraphDatasetScopeError
from tldw_Server_API.app.core.Sync.v2.materializers.notes import NotesMaterializer
from tldw_Server_API.app.core.Sync.v2.store import SyncStoreError
from tldw_Server_API.tests.Notes_Graph.integration.test_suggestion_legacy_mutations import (
    local_product_db as local_product_db,
)
from tldw_Server_API.tests.Sync.test_sync_v2_profile_bootstrap import _service

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("entry", ["profile", "personal-context", "personal-context-supplied"])
def test_default_creation_reserves_only_real_canonical_authority(local_product_db, tmp_path, entry):
    db = local_product_db
    service, sync_store = _service(tmp_path)
    service.materializers["notes.note"] = NotesMaterializer(db)
    if entry == "profile":
        result = service.bootstrap_profile(
            user_id=db.client_id,
            mode="offline_sync",
            device_id="local-transition-device",
            requested_domains=["notes.note"],
        )
        dataset_id = result.dataset.dataset_id
    else:
        supplied = (
            sync_store.get_or_create_default_personal_dataset(db.client_id)
            if entry == "personal-context-supplied"
            else None
        )
        result = service._profile_manager()._bind_personal_context_dataset(
            user_id=db.client_id,
            manifest=SimpleNamespace(profile_id="local-transition-profile"),
            authority_id="local-transition-authority",
            integrity_key_id="local-transition-key",
            purge_generation=0,
            dataset=supplied,
        )
        dataset_id = result.dataset_id
    with db.transaction() as conn:
        rows = conn.execute(
            "SELECT owner_user_id,dataset_id,task_graph_bound,moodboard_graph_bound,studio_graph_bound "
            "FROM note_task_scope_authority WHERE owner_user_id=?",
            (db.client_id,),
        ).fetchall()
    assert [dict(row) for row in rows] == [
        {
            "owner_user_id": db.client_id,
            "dataset_id": dataset_id,
            "task_graph_bound": False,
            "moodboard_graph_bound": False,
            "studio_graph_bound": False,
        }
    ]


@pytest.mark.parametrize("existing", ["none", "same", "different"])
def test_reservation_preserves_existing_authority_and_caller_rollback(local_product_db, tmp_path, existing):
    db = local_product_db
    service, sync_store = _service(tmp_path)
    service.materializers["notes.note"] = NotesMaterializer(db)
    dataset = sync_store.get_or_create_default_personal_dataset(db.client_id)
    if existing != "none":
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO note_task_scope_authority(owner_user_id,dataset_id,task_graph_bound,moodboard_graph_bound,studio_graph_bound) VALUES (?,?,?,?,?)",
                (db.client_id, dataset.dataset_id if existing == "same" else "another-dataset", True, False, True),
            )
    with db.transaction() as conn:
        before = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM note_task_scope_authority WHERE owner_user_id=?", (db.client_id,)
            ).fetchall()
        ]
    if existing == "different":
        with pytest.raises(NotesGraphDatasetScopeError):
            service.prepare_notes_suggestion_authority(user_id=db.client_id, dataset=dataset)
    else:
        with pytest.raises(RuntimeError, match="caller rollback"):
            with db.transaction():
                service.prepare_notes_suggestion_authority(user_id=db.client_id, dataset=dataset)
                raise RuntimeError("caller rollback")
    with db.transaction() as conn:
        after = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM note_task_scope_authority WHERE owner_user_id=?", (db.client_id,)
            ).fetchall()
        ]
    assert after == before
    if existing == "same":
        service.prepare_notes_suggestion_authority(user_id=db.client_id, dataset=dataset)
        with db.transaction() as conn:
            assert [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM note_task_scope_authority WHERE owner_user_id=?", (db.client_id,)
                ).fetchall()
            ] == before


def test_reservation_rejects_foreign_default_before_product_binding(local_product_db, tmp_path):
    db = local_product_db
    service, sync_store = _service(tmp_path)
    service.materializers["notes.note"] = NotesMaterializer(db)
    foreign = sync_store.get_or_create_default_personal_dataset("foreign-owner")
    with pytest.raises(SyncStoreError):
        service.prepare_notes_suggestion_authority(user_id=db.client_id, dataset=foreign)
    with db.transaction() as conn:
        assert conn.execute("SELECT 1 FROM note_task_scope_authority").fetchone() is None


def test_later_task_binding_reuses_real_reservation_without_rekeying_review(local_product_db, tmp_path):
    db = local_product_db
    service, sync_store = _service(tmp_path)
    service.materializers["notes.note"] = NotesMaterializer(db)
    dataset = sync_store.get_or_create_default_personal_dataset(db.client_id)
    service.prepare_notes_suggestion_authority(user_id=db.client_id, dataset=dataset)
    db.bind_local_task_graph_to_dataset(owner_user_id=db.client_id, target_dataset_id=dataset.dataset_id)
    service.prepare_notes_suggestion_authority(user_id=db.client_id, dataset=dataset)
    with db.transaction() as conn:
        row = conn.execute("SELECT * FROM note_task_scope_authority WHERE owner_user_id=?", (db.client_id,)).fetchone()
    assert row["dataset_id"] == dataset.dataset_id and row["task_graph_bound"]


@pytest.mark.postgres
@pytest.mark.parametrize("first", ["local-product-transaction", "canonical-reservation"])
def test_pg_authority_lock_serializes_local_work_with_enrollment(pg_database_config, tmp_path, first):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from time import monotonic, sleep

    from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
    from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(tmp_path / "concurrent.db", client_id="owner-1", backend=backend)
    store = db.note_graph_suggestion_store
    entered, release, second_ready = Event(), Event(), Event()
    pids = []

    def transaction_one():
        with chacha_operation(independent=True), db.transaction():
            if first == "canonical-reservation":
                store.reserve_canonical_scope(dataset_id="real-canonical")
            else:
                assert store.is_local_scope_available(dataset_id="legacy:owner-1")
            entered.set()
            assert release.wait(10)

    def transaction_two():
        with chacha_operation(independent=True), db.transaction() as conn:
            pids.append(conn.execute("SELECT pg_backend_pid() AS pid").fetchone()["pid"])
            second_ready.set()
            if first == "canonical-reservation":
                return store.is_local_scope_available(dataset_id="legacy:owner-1")
            store.reserve_canonical_scope(dataset_id="real-canonical")
            return True

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            one = executor.submit(transaction_one)
            assert entered.wait(10)
            two = executor.submit(transaction_two)
            assert second_ready.wait(10)
            try:
                deadline = monotonic() + 5
                blocked = False
                while monotonic() < deadline:
                    with chacha_operation(independent=True), db.transaction() as conn:
                        blocked = (
                            conn.execute(
                                "SELECT 1 FROM pg_locks WHERE pid=? AND relation='note_task_scope_authority'::regclass AND NOT granted",
                                (pids[0],),
                            ).fetchone()
                            is not None
                        )
                    if blocked:
                        break
                    sleep(0.01)
                assert blocked and not two.done()
            finally:
                release.set()
            one.result(timeout=10)
            assert two.result(timeout=10) is (first != "canonical-reservation")
        with chacha_operation(independent=True):
            assert not store.is_local_scope_available(dataset_id="legacy:owner-1")
    finally:
        release.set()
        db.close_all_connections()
        backend.get_pool().close_all()
