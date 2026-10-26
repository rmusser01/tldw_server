"""Semantic conversion's two visibility lookups inside one ChaCha operation on PostgreSQL (TASK-13899).

NoteSemanticStore.list_visible_vector_ids opens db.transaction(). Inside an HTTP request's
operation, two lookups at once would hold two managed transactions on one connection state,
which ChaCha refuses (ConcurrentChaChaTransactionError). validate_conversion therefore runs
them one after the other.
"""

import asyncio
import contextlib
import threading

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.tests.Notes_Graph.unit import test_semantic_projector as fakes

pytestmark = pytest.mark.integration


@pytest.fixture
def pg_db(pg_database_config, tmp_path):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(tmp_path / "semantic.db", client_id="1", backend=backend)
    db.close_connection()
    try:
        yield db
    finally:
        db.close_connection()
        backend.get_pool().close_all()


async def test_conversion_lookups_inside_one_operation_do_not_overlap(pg_db):
    source = fakes._projection_chunk(
        generation_id="generation-a", note_id="focus-note", title="Focus", content="alpha beta gamma"
    )
    target = fakes._projection_chunk(
        generation_id="generation-a", note_id="target-note", title="Target", content="alpha beta delta"
    )
    store = fakes._Store(source, target)
    vectors = fakes._VectorStore(source.vector_id, target.vector_id)
    vectors.distance = 0.4
    projector, _service, _factory_calls = fakes._projector(
        store=store, vectors=vectors, ordinary=fakes._ordinary_response()
    )
    fake_lookup, guard = store.list_visible_vector_ids, threading.Lock()
    in_flight, peak, both_inside = [0], [0], threading.Barrier(2)

    def lookup_in_a_postgres_transaction(dataset_id, generation_id, note_id=None):
        with guard:
            in_flight[0] += 1
            peak[0] = max(peak[0], in_flight[0])
        try:
            with pg_db.transaction():  # what NoteSemanticStore.list_visible_vector_ids opens
                pg_db.note_semantic_store.list_visible_vector_ids(dataset_id, generation_id, note_id)
                with contextlib.suppress(threading.BrokenBarrierError):
                    both_inside.wait(timeout=1.0)  # two concurrent lookups are surely inside together
                return fake_lookup(dataset_id, generation_id, note_id)
        finally:
            with guard:
                in_flight[0] -= 1

    store.list_visible_vector_ids = lookup_in_a_postgres_transaction
    outcome = None
    with chacha_operation(independent=True):  # the request's operation
        try:
            await projector.validate_conversion(
                source_note_id="focus-note", target_note_id="target-note", generation_id="generation-a"
            )
        except Exception as exc:  # noqa: BLE001 - the RED records the guard's error by name
            outcome = type(exc).__name__
        for _ in range(60):  # a lookup that gather left behind finishes before the operation closes
            if not in_flight[0]:
                break
            await asyncio.sleep(0.05)

    assert (outcome, peak[0]) == (None, 1)
