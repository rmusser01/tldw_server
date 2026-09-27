"""Native enrichment threads own their PostgreSQL checkouts until completion."""

import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Chat import conversation_enrichment as enrichment
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def enrichment_runtime(request, tmp_path, monkeypatch):
    """Keep actual scheduling, database operations and pool checkout/return."""
    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    db = CharactersRAGDB(tmp_path / "enrichment.db", client_id="3", backend=backend)
    db.close_connection()
    f = SimpleNamespace(
        db=db, backend=backend, threads=[], loans={}, returned=[], failed_returns=[], tags=[], clusters=[]
    )
    original_thread = threading.Thread

    class ObservedThread(original_thread):
        def __init__(self, *, target, daemon):
            super().__init__(target=target, daemon=daemon)
            f.threads.append(self)

    # Force the actual native branch; pytest otherwise intentionally runs inline.
    monkeypatch.setattr(enrichment, "_should_run_inline", lambda: False)
    monkeypatch.setattr(enrichment, "threading", SimpleNamespace(Thread=ObservedThread))
    original_tag = enrichment.auto_tag_conversation
    original_cluster = enrichment.cluster_conversations_for_user

    def observe_tag(*args, **kwargs):
        result = original_tag(*args, **kwargs)
        f.tags.append(result)
        return result

    def observe_cluster(*args, **kwargs):
        result = original_cluster(*args, **kwargs)
        f.clusters.append(result)
        return result

    monkeypatch.setattr(enrichment, "auto_tag_conversation", observe_tag)
    monkeypatch.setattr(enrichment, "cluster_conversations_for_user", observe_cluster)
    if backend is not None:
        pool = backend.get_pool()
        original_get, original_return = pool.get_connection, pool.return_connection
        serial = 0
        lock = threading.Lock()

        def get_connection():
            nonlocal serial
            raw = original_get()
            with lock:
                serial += 1
                f.loans[id(raw)] = (serial, raw, threading.get_ident())
            return raw

        def return_connection(raw):
            with lock:
                lease = f.loans.pop(id(raw), None)
                assert lease is not None, "return must match a live checkout"
            # Retire this generation before the pool makes raw available to a
            # concurrent clustering callback. Never pop that callback's lease.
            try:
                original_return(raw)
            except BaseException:
                with lock:
                    f.failed_returns.append(lease[0])
                raise
            with lock:
                f.returned.append(lease[0])

        monkeypatch.setattr(pool, "get_connection", get_connection)
        monkeypatch.setattr(pool, "return_connection", return_connection)
    try:
        yield f
    finally:
        _join(f)
        db.close_all_connections()
        if backend is not None:
            backend.get_pool().close_all()


def _seed(f, count=1):
    conversation = f.db.add_conversation(
        {"title": "Enrichment fixture", "assistant_kind": "persona", "assistant_id": "fixture"}
    )
    for _ in range(count):
        f.db.add_message({"conversation_id": conversation, "sender": "user", "content": "Citrine invoice question"})
    f.db.close_connection()
    return conversation


def _join(f):
    # Successful tagging may start a real clustering child during its execution.
    index = 0
    while index < len(f.threads):
        thread = f.threads[index]
        thread.join(timeout=10)
        assert not thread.is_alive(), "enrichment callback must finish"
        assert thread.daemon is True
        index += 1


def _assert_returned(f):
    assert not f.loans, "completed callback retained an owned PostgreSQL checkout"
    assert not f.failed_returns, "the actual pool return must succeed"
    assert len(f.returned) == len(set(f.returned)), "each checkout returns at most once"


@pytest.mark.parametrize("missing", [False, True])
def test_tag_early_exit_releases_actual_callback_checkout(enrichment_runtime, missing):
    f = enrichment_runtime
    conversation = "missing" if missing else _seed(f)
    with chacha_operation(independent=True):
        enrichment.schedule_auto_tagging(f.db, conversation, owner_user_id=3)
    _join(f)
    assert [result.reason for result in f.tags] == ["missing_conversation" if missing else "insufficient_new_messages"]
    _assert_returned(f)


def test_empty_clustering_releases_actual_callback_checkout(enrichment_runtime):
    f = enrichment_runtime
    enrichment.schedule_conversation_clustering(f.db, client_id="3")
    _join(f)
    assert f.clusters[0].clusters_written == 0
    _assert_returned(f)


def test_successful_tag_and_spawned_clustering_persist_then_release(enrichment_runtime):
    f = enrichment_runtime
    conversation = _seed(f, count=3)
    enrichment.schedule_auto_tagging(f.db, conversation, owner_user_id=3)
    _join(f)
    assert [result.reason for result in f.tags] == ["updated"]
    assert f.clusters and f.clusters[0].clusters_written == 1
    _assert_returned(f)
    with chacha_operation(independent=True):
        row = f.db.get_conversation_by_id(conversation)
        assert row["topic_label_source"] == "auto"
        assert row["cluster_id"].startswith("topic-")
        assert f.db.get_keywords_for_conversation(conversation)


def test_repeated_callbacks_on_cached_database_release_each_lease(enrichment_runtime):
    f = enrichment_runtime
    conversation = _seed(f)
    for _ in range(3):
        enrichment.schedule_auto_tagging(f.db, conversation, owner_user_id=3)
        _join(f)
        _assert_returned(f)
    assert [result.reason for result in f.tags] == ["insufficient_new_messages"] * 3


@pytest.mark.parametrize("ownership", ["legacy", "operation"])
@pytest.mark.parametrize("commit", [False, True])
def test_callback_does_not_decide_unrelated_caller_transaction(enrichment_runtime, ownership, commit):
    f = enrichment_runtime
    conversation = _seed(f)
    note = f.db.add_note(title="Caller original", content="Unrelated pending write")
    f.db.close_connection()
    with chacha_operation(independent=True) if ownership == "operation" else nullcontext():

        class CallerRollback(Exception):
            pass

        try:
            with f.db.transaction():
                f.db.execute_query("UPDATE notes SET title = ? WHERE id = ?", ("Caller pending", note))
                before = {key: value[0] for key, value in f.loans.items()}
                enrichment.schedule_auto_tagging(f.db, conversation, owner_user_id=3)
                _join(f)
                assert {key: value[0] for key, value in f.loans.items()} == before
                assert f.db.get_note_by_id(note)["title"] == "Caller pending"
                if not commit:
                    raise CallerRollback()
        except CallerRollback:
            pass
        assert f.db.get_note_by_id(note)["title"] == ("Caller pending" if commit else "Caller original")
    f.db.close_connection()
    _assert_returned(f)


def test_inflight_thread_outlives_closed_http_owner_without_early_return(enrichment_runtime, monkeypatch):
    f = enrichment_runtime
    conversation = _seed(f)
    entered, release = threading.Event(), threading.Event()
    count = f.db.count_messages_since

    def gated_count(*args, **kwargs):
        value = count(*args, **kwargs)
        entered.set()
        assert release.wait(timeout=10)
        return value

    monkeypatch.setattr(f.db, "count_messages_since", gated_count)
    try:
        with chacha_operation(independent=True):
            enrichment.schedule_auto_tagging(f.db, conversation, owner_user_id=3)
            assert entered.wait(timeout=10)
        if f.backend is not None:
            assert len(f.loans) == 1
    finally:
        release.set()
        _join(f)
    assert [result.reason for result in f.tags] == ["insufficient_new_messages"]
    _assert_returned(f)


def test_actual_database_failure_releases_callback_checkout(enrichment_runtime, monkeypatch):
    f = enrichment_runtime
    conversation = _seed(f)

    def database_failure(*_args, **_kwargs):
        f.db.execute_query("SELECT missing_enrichment_fixture_column FROM conversations")

    monkeypatch.setattr(f.db, "count_messages_since", database_failure)
    enrichment.schedule_auto_tagging(f.db, conversation, owner_user_id=3)
    _join(f)
    assert f.tags == []
    _assert_returned(f)


@pytest.mark.parametrize("enrichment_runtime", ["postgres"], indirect=True)
def test_successful_callback_does_not_commit_unfinished_write(enrichment_runtime, monkeypatch):
    f = enrichment_runtime
    conversation = _seed(f)
    note = f.db.add_note(title="Committed baseline", content="No automatic success commit")
    f.db.close_connection()
    original_tag = enrichment.auto_tag_conversation

    def leave_pending_write(*args, **kwargs):
        result = original_tag(*args, **kwargs)
        f.db.execute_query("UPDATE notes SET title = ? WHERE id = ?", ("Uncommitted callback", note))
        return result

    monkeypatch.setattr(enrichment, "auto_tag_conversation", leave_pending_write)
    enrichment.schedule_auto_tagging(f.db, conversation, owner_user_id=3)
    _join(f)
    assert [result.reason for result in f.tags] == ["insufficient_new_messages"]
    _assert_returned(f)
    with chacha_operation(independent=True):
        assert f.db.get_note_by_id(note)["title"] == "Committed baseline"


def test_inline_scheduler_keeps_existing_caller_transaction(enrichment_runtime, monkeypatch):
    f = enrichment_runtime
    conversation = _seed(f)
    note = f.db.add_note(title="Original", content="Inline caller")
    f.db.close_connection()
    monkeypatch.setattr(enrichment, "_should_run_inline", lambda: True)

    class CallerRollback(Exception):
        pass

    with pytest.raises(CallerRollback), chacha_operation(independent=True), f.db.transaction():
        f.db.execute_query("UPDATE notes SET title = ? WHERE id = ?", ("Pending", note))
        before = {key: value[0] for key, value in f.loans.items()}
        enrichment.schedule_auto_tagging(f.db, conversation, owner_user_id=3)
        assert not f.threads
        assert {key: value[0] for key, value in f.loans.items()} == before
        assert f.db.get_note_by_id(note)["title"] == "Pending"
        raise CallerRollback()
    with chacha_operation(independent=True):
        assert f.db.get_note_by_id(note)["title"] == "Original"
