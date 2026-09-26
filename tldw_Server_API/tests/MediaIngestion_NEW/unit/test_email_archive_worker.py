"""Real database regressions for archive-local worker ownership."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.scope_context import get_scope, scoped_context
from tldw_Server_API.app.core.Ingestion_Media_Processing import persistence

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def persist_archive(db_path, *, owner=42, payload_suffix=""):
    """Persist three synthetic archive children without chunking or model work."""
    result = {
        "status": "Success",
        "content": "",
        "metadata": {},
        "children": [
            {
                "status": "Success",
                "content": f"Body {i}{payload_suffix}",
                "metadata": {
                    "title": f"Subject {i}{payload_suffix}",
                    "filename": f"child-{i}.eml",
                    "email": {"message_id": f"<child-{i}@example.test>", "subject": f"Subject {i}{payload_suffix}"},
                },
            }
            for i in range(3)
        ],
    }
    with scoped_context(user_id=owner):
        await persistence.persist_doc_item_and_children(
            final_result=result,
            form_data=SimpleNamespace(keywords=[], accept_mbox=True, overwrite_existing=False),
            media_type="email",
            item_input_ref="synthetic.mbox",
            processing_filename="synthetic.mbox",
            chunk_options=None,
            path_kind="upload",
            db_path=str(db_path),
            client_id=f"owner-{owner}",
            loop=asyncio.get_running_loop(),
            claims_context=None,
            email_tenant_id=str(owner),
        )
    return result


async def test_archive_reuses_one_handle_and_retry_preserves_ids(tmp_path, monkeypatch):
    created = []

    def create(client_id, *, db_path):
        db = MediaDatabase(db_path=db_path, client_id=client_id)
        created.append(db)
        return db

    monkeypatch.setattr(persistence, "create_media_database", create)
    monkeypatch.setattr(persistence, "_is_email_native_persist_enabled", lambda: True)
    path = tmp_path / "archive.db"
    result = await persist_archive(path)
    ids = [child["db_id"] for child in result["child_db_results"]]
    assert len(set(ids)) == 3
    assert len(created) == 1
    retry = await persist_archive(path)
    assert [child["db_id"] for child in retry["child_db_results"]] == ids
    assert len(created) == 2
    db = MediaDatabase(db_path=str(path), client_id="verify")
    try:
        assert db.execute_query("SELECT COUNT(*) FROM email_messages").fetchone()[0] == 3
    finally:
        db.close_connection()


async def test_worker_keeps_thread_affinity_and_resets_operation_scope(tmp_path, monkeypatch):
    events = []
    real_create = persistence.create_media_database

    def create(client_id, *, db_path):
        events.append(("create", threading.get_ident(), get_scope().user_id))
        db = real_create(client_id, db_path=db_path)
        real_close = db.close_connection

        def close():
            events.append(("close", threading.get_ident(), get_scope().user_id))
            real_close()

        db.close_connection = close
        return db

    def first(db):
        from tldw_Server_API.app.core.DB_Management.scope_context import set_scope

        events.append(("first", threading.get_ident(), get_scope().user_id))
        set_scope(user_id=999)
        return db.get_media_by_id(1)

    def second(db):
        events.append(("second", threading.get_ident(), get_scope().user_id))
        return db.get_media_by_id(1)

    monkeypatch.setattr(persistence, "create_media_database", create)
    with scoped_context(user_id=42):
        async with persistence._archive_media_db_worker(db_path=str(tmp_path / "scope.db"), client_id="42") as run:
            await run(first)
            await run(second)
        assert get_scope().user_id == 42
    assert [event[0] for event in events] == ["create", "first", "second", "close"]
    assert {event[2] for event in events} == {42}
    assert len({event[1] for event in events}) == 1
    assert events[0][1] != threading.get_ident()


async def test_worker_failure_rolls_back_only_its_transaction(tmp_path):
    def fail(db):
        with db.transaction() as conn:
            conn.execute("UPDATE Media SET title = 'Must roll back', version = version + 1")
            raise ValueError("synthetic write failure")

    def write(db):
        return db.add_media_with_keywords(
            url="email://transaction", title="Committed", media_type="email", content="Body", keywords=[]
        )[0]

    async with persistence._archive_media_db_worker(db_path=str(tmp_path / "rollback.db"), client_id="42") as run:
        media_id = await run(write)
        with pytest.raises(ValueError, match="synthetic write failure"):
            await run(fail)
        assert (await run(lambda db: db.get_media_by_id(media_id)))["title"] == "Committed"
        assert await run(write) == media_id


async def test_cancelled_worker_closes_after_inflight_write(tmp_path, monkeypatch):
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    real_create = persistence.create_media_database

    def create(client_id, *, db_path):
        db = real_create(client_id, db_path=db_path)
        real_close = db.close_connection

        def close():
            real_close()
            closed.set()

        db.close_connection = close
        return db

    def operation(db):
        entered.set()
        if not release.wait(5):
            raise TimeoutError("test did not release worker")
        return db.add_media_with_keywords(
            url="email://cancel", title="Committed", media_type="email", content="Body", keywords=[]
        )[0]

    async def ingest():
        async with persistence._archive_media_db_worker(db_path=str(tmp_path / "cancel.db"), client_id="42") as run:
            await run(operation)

    monkeypatch.setattr(persistence, "create_media_database", create)
    task = asyncio.create_task(ingest())
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not closed.is_set()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
    db = MediaDatabase(db_path=str(tmp_path / "cancel.db"), client_id="verify")
    try:
        assert db.get_media_by_id(1)["title"] == "Committed"
    finally:
        db.close_connection()


async def test_native_failure_preserves_legacy_archive_children(tmp_path, monkeypatch):
    def fail_native(*_args, **_kwargs):
        raise ValueError("synthetic native failure")

    monkeypatch.setattr(persistence, "_is_email_native_persist_enabled", lambda: True)
    monkeypatch.setattr(MediaDatabase, "upsert_email_message_graph", fail_native)
    path = tmp_path / "native-failure.db"
    result = await persist_archive(path)
    assert len(result["child_db_results"]) == 3
    db = MediaDatabase(db_path=str(path), client_id="verify")
    try:
        assert db.execute_query("SELECT COUNT(*) FROM Media").fetchone()[0] == 3
        assert db.execute_query("SELECT COUNT(*) FROM email_messages").fetchone()[0] == 0
    finally:
        db.close_connection()


async def test_concurrent_archives_use_separate_request_scopes(tmp_path, monkeypatch):
    seen = []
    real_create = persistence.create_media_database

    def create(client_id, *, db_path):
        seen.append((db_path, get_scope().user_id))
        return real_create(client_id, db_path=db_path)

    monkeypatch.setattr(persistence, "create_media_database", create)
    first, second = await asyncio.gather(
        persist_archive(tmp_path / "first.db", owner=42),
        persist_archive(tmp_path / "second.db", owner=43),
    )
    assert len(first["child_db_results"]) == len(second["child_db_results"]) == 3
    assert sorted(owner for _, owner in seen) == [42, 43]
    assert len({path for path, _ in seen}) == 2


async def test_second_cancellation_does_not_block_event_loop_during_cleanup(tmp_path, monkeypatch):
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    observed = []
    real_create = persistence.create_media_database

    def create(client_id, *, db_path):
        db = real_create(client_id, db_path=db_path)
        real_close = db.close_connection

        def close():
            real_close()
            closed.set()

        db.close_connection = close
        return db

    def operation(db):
        entered.set()
        release.wait(1)  # Bounds the regression when a broken shutdown blocks the loop.
        return db.get_media_by_id(1)

    async def ingest():
        async with persistence._archive_media_db_worker(
            db_path=str(tmp_path / "cancel-twice.db"), client_id="42"
        ) as run:
            await run(operation)

    def probe_event_loop():
        observed.append(closed.is_set())
        release.set()

    monkeypatch.setattr(persistence, "create_media_database", create)
    task = asyncio.create_task(ingest())
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)  # The task enters the shielded cleanup await.
        task.cancel()
        asyncio.get_running_loop().call_soon(probe_event_loop)
        with pytest.raises(asyncio.CancelledError):
            await task
        assert observed == [False]
        assert closed.is_set()
    finally:
        release.set()


async def test_archive_native_read_and_graph_share_one_connection(tmp_path, monkeypatch):
    read_connections, graph_connections = [], []
    real_read = persistence.read_persisted_email_content
    real_graph = MediaDatabase.upsert_email_message_graph

    def read(db, *args, **kwargs):
        read_connections.append(db._get_txn_conn())
        return real_read(db, *args, **kwargs)

    def graph(db, *args, **kwargs):
        graph_connections.append(db._get_txn_conn())
        return real_graph(db, *args, **kwargs)

    monkeypatch.setattr(persistence, "_is_email_native_persist_enabled", lambda: True)
    monkeypatch.setattr(persistence, "read_persisted_email_content", read)
    monkeypatch.setattr(MediaDatabase, "upsert_email_message_graph", graph)
    result = await persist_archive(tmp_path / "native-connection.db")
    assert len(result["child_db_results"]) == 3
    assert len(read_connections) == len(graph_connections) == 3
    assert all(conn is not None for conn in read_connections)
    assert all(read is graph for read, graph in zip(read_connections, graph_connections))


async def test_late_native_failure_rolls_back_graph_and_preserves_media(tmp_path, monkeypatch):
    real_graph = MediaDatabase.upsert_email_message_graph

    def late_failure(db, *args, **kwargs):
        real_graph(db, *args, **kwargs)
        raise ValueError("synthetic failure after nested graph writes")

    monkeypatch.setattr(persistence, "_is_email_native_persist_enabled", lambda: True)
    monkeypatch.setattr(MediaDatabase, "upsert_email_message_graph", late_failure)
    path = tmp_path / "late-native-failure.db"
    result = await persist_archive(path)
    assert len(result["child_db_results"]) == 3
    db = MediaDatabase(db_path=str(path), client_id="verify")
    try:
        assert db.execute_query("SELECT COUNT(*) FROM Media").fetchone()[0] == 3
        assert db.execute_query("SELECT COUNT(*) FROM email_messages").fetchone()[0] == 0
        assert db.execute_query("SELECT COUNT(*) FROM email_sources").fetchone()[0] == 0
    finally:
        db.close_connection()


async def test_archive_declined_overwrite_retains_saved_native_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(persistence, "_is_email_native_persist_enabled", lambda: True)
    path = tmp_path / "accepted-payload.db"
    original = await persist_archive(path)
    retry = await persist_archive(path, payload_suffix=" Incoming changed payload")
    assert [item["db_id"] for item in retry["child_db_results"]] == [
        item["db_id"] for item in original["child_db_results"]
    ]
    db = MediaDatabase(db_path=str(path), client_id="verify")
    try:
        rows = db.execute_query("SELECT subject, body_text FROM email_messages ORDER BY media_id").fetchall()
        assert [(row["subject"], row["body_text"]) for row in rows] == [(f"Subject {i}", f"Body {i}") for i in range(3)]
    finally:
        db.close_connection()
