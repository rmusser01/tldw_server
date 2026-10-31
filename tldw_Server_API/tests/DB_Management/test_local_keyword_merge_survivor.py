"""Local keyword merge outcomes must survive a new database accessor."""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.organization_sync_store import NotesOrganizationSyncStore
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def keywords(request, tmp_path):
    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    path = tmp_path / "ChaChaNotes.db"
    db = CharactersRAGDB(path, client_id="2", backend=backend)
    instances = [db]

    def reopen(client_id="2"):
        db.close_connection()
        current = CharactersRAGDB(path, client_id=client_id, backend=backend)
        instances.append(current)
        return current

    try:
        yield SimpleNamespace(db=db, backend=backend, reopen=reopen)
    finally:
        for instance in instances:
            instance.close_all_connections()
        if backend is not None:
            backend.get_pool().close_all()


def _keyword(db, name):
    return db.get_keyword_by_id(db.add_keyword(name))


def _merge(db, source, target):
    return db.keyword_store.merge_keywords(
        source_keyword_id=source["id"],
        target_keyword_id=target["id"],
        expected_source_version=source["version"],
        expected_target_version=target["version"],
    )


def _row(db, keyword_id):
    table = db._map_table_for_backend("keywords")
    return dict(db.execute_query(f"SELECT * FROM {table} WHERE id = ?", (keyword_id,), read_only=True).fetchone())  # nosec B608


def test_real_merge_retains_portable_survivor_after_reopen(keywords):
    db = keywords.db
    source = _keyword(db, "Source")
    target = _keyword(db, "Survivor")
    note = db.add_note("Merge note", "Membership must survive")
    db.link_note_to_keyword(note, source["id"])
    result = _merge(db, source, target)
    reopened = keywords.reopen()
    source_after = _row(reopened, source["id"])
    assert result["target_keyword_id"] == target["id"]
    assert source_after["deleted"] and source_after["version"] == 2
    assert [row["id"] for row in reopened.get_keywords_for_note(note)] == [target["id"]]
    assert source_after.get("merged_into_sync_id") == target["sync_id"]


def test_plain_delete_does_not_invent_survivor(keywords):
    db = keywords.db
    source = _keyword(db, "Ordinary deletion")
    db.soft_delete_keyword(source["id"], source["version"])
    assert _row(keywords.reopen(), source["id"]).get("merged_into_sync_id") is None


def test_merge_caller_rollback_preserves_original_links(keywords):
    db = keywords.db
    source, target = _keyword(db, "Rollback source"), _keyword(db, "Rollback target")
    note = db.add_note("Rollback note", "Body")
    db.link_note_to_keyword(note, source["id"])
    with pytest.raises(RuntimeError, match="caller abort"):
        with db.transaction():
            _merge(db, source, target)
            raise RuntimeError("caller abort")
    assert db.get_keyword_by_id(source["id"]) == source
    assert [row["id"] for row in db.get_keywords_for_note(note)] == [source["id"]]


def test_resolver_follows_chain_and_current_live_identity(keywords):
    db = keywords.db
    a, b, c = (_keyword(db, name) for name in ("A", "B", "C"))
    _merge(db, a, b)
    _merge(db, b, c)
    reopened = keywords.reopen()
    resolved = reopened.keyword_store.resolve_merge_survivor(a["sync_id"])
    assert resolved["sync_id"] == c["sync_id"]
    reopened.keyword_store.rename_keyword(c["id"], "Renamed C", c["version"])
    assert reopened.keyword_store.resolve_merge_survivor(a["sync_id"])["keyword"] == "Renamed C"
    reopened.add_keyword("B")
    assert reopened.keyword_store.resolve_merge_survivor(a["sync_id"])["sync_id"] == b["sync_id"]


@pytest.mark.parametrize("restore", ["ordinary", "organization", "flashcard"])
def test_all_restore_writers_clear_redirect_before_later_delete(keywords, restore):
    db = keywords.db
    a, b = _keyword(db, "Restore A"), _keyword(db, "Restore B")
    _merge(db, a, b)
    if restore == "ordinary":
        assert db.add_keyword("Restore A") == a["id"]
    elif restore == "organization":
        NotesOrganizationSyncStore(db).apply_resource(
            domain="notes.keyword", object_id=a["sync_id"], operation="upsert", payload={"keyword": "Restore A"}
        )
    else:
        card = db.add_flashcard({"front": "Restore tag", "back": "Answer"})
        assert db.set_flashcard_tags(card, ["Restore A"])
    restored = db.get_keyword_by_id(a["id"])
    assert restored["sync_id"] == a["sync_id"] and restored["version"] == 3
    assert restored.get("merged_into_sync_id") is None
    db.soft_delete_keyword(a["id"], restored["version"])
    assert db.keyword_store.resolve_merge_survivor(a["sync_id"]) is None


def test_remerge_restored_source_and_idempotent_delete(keywords):
    db = keywords.db
    a, b, c = (_keyword(db, name) for name in ("Remerge A", "Remerge B", "Remerge C"))
    _merge(db, a, b)
    before = _row(db, a["id"])
    assert db.soft_delete_keyword(a["id"], 1)
    assert _row(db, a["id"]) == before
    db.add_keyword("Remerge A")
    _merge(db, db.get_keyword_by_id(a["id"]), c)
    assert db.keyword_store.resolve_merge_survivor(a["sync_id"])["sync_id"] == c["sync_id"]


@pytest.mark.parametrize("invalid", ["missing", "plain-deleted", "malformed", "cycle", "too-long"])
def test_resolver_fails_closed_for_invalid_chain(keywords, invalid):
    db = keywords.db
    source = _keyword(db, "Invalid chain source")
    table = db._map_table_for_backend("keywords")
    if invalid == "missing":
        identity = str(uuid4())
    elif invalid == "plain-deleted":
        db.soft_delete_keyword(source["id"], 1)
        identity = source["sync_id"]
    else:
        target = _keyword(db, "Invalid chain target")
        _merge(db, source, target)
        identity = source["sync_id"]
        with db.transaction() as conn:
            if invalid == "malformed":
                conn.execute(f"UPDATE {table} SET merged_into_sync_id = ? WHERE id = ?", ("invalid", source["id"]))  # nosec B608
            elif invalid == "cycle":
                conn.execute(
                    f"UPDATE {table} SET deleted = ?, merged_into_sync_id = ? WHERE id = ?",  # nosec B608
                    (True, identity, target["id"]),
                )
            else:
                current = target
                for index in range(100):
                    following = _keyword(db, f"Long chain {index}")
                    conn.execute(
                        f"UPDATE {table} SET deleted = ?, merged_into_sync_id = ? WHERE id = ?",  # nosec B608
                        (True, following["sync_id"], current["id"]),
                    )
                    current = following
    assert db.keyword_store.resolve_merge_survivor(identity) is None


@pytest.mark.parametrize("invalid", [None, "", "not-a-uuid", str(uuid4()).upper(), 7])
def test_invalid_input_identity_is_not_guessed(keywords, invalid):
    assert keywords.db.keyword_store.resolve_merge_survivor(invalid) is None


def test_for_update_requires_explicit_transaction_and_preserves_caller_rollback(keywords):
    db = keywords.db
    a, b = _keyword(db, "Locked A"), _keyword(db, "Locked B")
    _merge(db, a, b)
    with pytest.raises(InputError):
        db.keyword_store.resolve_merge_survivor(a["sync_id"], for_update=True)
    with pytest.raises(RuntimeError, match="caller abort"):
        with db.transaction() as conn:
            assert db.keyword_store.resolve_merge_survivor(a["sync_id"], conn=conn, for_update=True)["id"] == b["id"]
            db.add_keyword("Uncommitted unrelated")
            raise RuntimeError("caller abort")
    assert db.get_keyword_by_text("Uncommitted unrelated") is None


def test_selected_owner_and_sqlite_device_labels(keywords):
    db = keywords.db
    a, b = _keyword(db, "Owner A"), _keyword(db, "Owner B")
    _merge(db, a, b)
    other = keywords.reopen(client_id="3")
    if keywords.backend is None:
        assert other.keyword_store.resolve_merge_survivor(a["sync_id"])["id"] == b["id"]
        other.add_keyword("Owner A")
        assert other.get_keyword_by_id(a["id"])["client_id"] == "3"
    else:
        assert other.keyword_store.resolve_merge_survivor(a["sync_id"]) is None
        foreign = _keyword(other, "Owner B")
        table = db._map_table_for_backend("keywords")
        with db.transaction() as conn:
            conn.execute(f"UPDATE {table} SET merged_into_sync_id = ? WHERE id = ?", (foreign["sync_id"], a["id"]))  # nosec B608
        assert db.keyword_store.resolve_merge_survivor(a["sync_id"]) is None


@pytest.mark.parametrize("parent", ["source", "target"])
def test_stale_merge_versions_preserve_original_rows(keywords, parent):
    db = keywords.db
    a, b = _keyword(db, "Version A"), _keyword(db, "Version B")
    arguments = {
        "source_keyword_id": a["id"],
        "target_keyword_id": b["id"],
        "expected_source_version": 1,
        "expected_target_version": 1,
    }
    arguments[f"expected_{parent}_version"] = 0
    with pytest.raises(ConflictError):
        db.keyword_store.merge_keywords(**arguments)
    assert _row(db, a["id"]) == a and _row(db, b["id"]) == b


@pytest.mark.parametrize("keywords", ["postgres"], indirect=True)
def test_opposite_merges_serialize_without_deadlock(keywords, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, current_thread
    from time import monotonic, sleep

    db = keywords.db
    a, b = _keyword(db, "Concurrent A"), _keyword(db, "Concurrent B")
    db.close_connection()
    first_locked, second_started, second_locked, release_first = (Event() for _ in range(4))
    pids = {}
    seen = set()
    original = db._require_selected_owner_row

    def gated(conn, table, item_id, owner, **kwargs):
        thread = current_thread().name
        first_call = thread not in seen and table == "keywords"
        if first_call:
            seen.add(thread)
            pids[thread] = conn.execute("SELECT pg_backend_pid() AS pid").fetchone()["pid"]
            if thread.endswith("_1"):
                second_started.set()
        result = original(conn, table, item_id, owner, **kwargs)
        if first_call:
            if thread.endswith("_0"):
                first_locked.set()
                assert release_first.wait(5)
            else:
                second_locked.set()
        return result

    def run(source, target):
        try:
            return _merge(db, source, target)
        except (ConflictError, CharactersRAGDBError) as error:
            return error
        finally:
            db.close_connection()

    monkeypatch.setattr(db, "_require_selected_owner_row", gated)
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="merge") as pool:
        first = pool.submit(run, a, b)
        assert first_locked.wait(5)
        second = pool.submit(run, b, a)
        try:
            assert second_started.wait(5)
            until = monotonic() + 5
            while not second_locked.is_set():
                waiting = db.backend.execute(
                    "SELECT wait_event_type FROM pg_stat_activity WHERE pid = %s", (pids["merge_1"],)
                ).rows
                if waiting and waiting[0]["wait_event_type"] == "Lock":
                    break
                assert monotonic() < until
                sleep(0.01)
        finally:
            release_first.set()
        results = [first.result(timeout=8), second.result(timeout=8)]
    assert sum(isinstance(result, dict) for result in results) == 1
    assert all(isinstance(result, (dict, ConflictError)) for result in results), [
        type(result).__name__ for result in results
    ]


@pytest.mark.parametrize("keywords", ["postgres"], indirect=True)
@pytest.mark.parametrize("mutation", ["merge", "delete", "restore", "rename"])
def test_locked_resolver_revalidates_chain_changed_before_locks(keywords, monkeypatch, mutation):
    from concurrent.futures import ThreadPoolExecutor

    db = keywords.db
    a, b, c = (_keyword(db, name) for name in ("Race A", "Race B", "Race C"))
    _merge(db, a, b)
    reached = []

    def mutate():
        try:
            with db.transaction() as worker_conn:
                worker_conn.execute("SET LOCAL lock_timeout = '2s'")
                worker_conn.execute("SET LOCAL statement_timeout = '3s'")
                if mutation == "merge":
                    _merge(db, b, c)
                elif mutation == "delete":
                    db.soft_delete_keyword(b["id"], 1)
                elif mutation == "restore":
                    db.add_keyword("Race A")
                else:
                    db.keyword_store.rename_keyword(b["id"], "Race B renamed", 1)
        finally:
            db.close_connection()

    with pytest.raises(ConflictError, match="chain changed"):
        with db.transaction() as conn:
            original = conn.execute

            def gate(query, params=None):
                if "ORDER BY id FOR UPDATE" in query and not reached:
                    reached.append(True)
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        pool.submit(mutate).result(timeout=5)
                return original(query, params)

            monkeypatch.setattr(conn, "execute", gate)
            db.keyword_store.resolve_merge_survivor(a["sync_id"], conn=conn, for_update=True)
    assert reached == [True]


@pytest.mark.parametrize("keywords", ["postgres"], indirect=True)
def test_restricted_runtime_owner_can_merge_and_resolve_only_owned_keywords(keywords):
    db = keywords.db
    a, b = _keyword(db, "Restricted A"), _keyword(db, "Restricted B")
    other = keywords.reopen(client_id="3")
    foreign = _keyword(other, "Restricted B")
    other.close_connection()
    role = db.backend.escape_identifier(f"uat168_{uuid4().hex[:12]}")
    try:
        with db.backend.transaction() as conn:
            db.backend.execute(f"CREATE ROLE {role} NOLOGIN NOSUPERUSER NOBYPASSRLS", connection=conn)
            db.backend.execute(f"GRANT USAGE ON SCHEMA public TO {role}", connection=conn)
            db.backend.execute(
                f"GRANT SELECT,INSERT,UPDATE,DELETE ON ALL TABLES IN SCHEMA public TO {role}", connection=conn
            )
            db.backend.execute(f"GRANT USAGE,SELECT ON ALL SEQUENCES IN SCHEMA public TO {role}", connection=conn)
            db.backend.execute(f"GRANT {role} TO CURRENT_USER", connection=conn)
        with db.transaction() as conn:
            conn.execute(f"SET LOCAL ROLE {role}")
            flags = conn.execute("SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user").fetchone()
            assert not flags["rolsuper"] and not flags["rolbypassrls"]
            _merge(db, a, b)
            assert db.keyword_store.resolve_merge_survivor(a["sync_id"], conn=conn, for_update=True)["id"] == b["id"]
            assert db.keyword_store.resolve_merge_survivor(foreign["sync_id"], conn=conn) is None
    finally:
        with db.backend.transaction() as conn:
            db.backend.execute(f"DROP OWNED BY {role}", connection=conn)
            db.backend.execute(f"DROP ROLE {role}", connection=conn)


@pytest.mark.parametrize("fail_after_links", [False, True])
def test_four_membership_families_and_redirect_are_atomic(keywords, monkeypatch, fail_after_links):
    db = keywords.db
    a, b = _keyword(db, "All links source"), _keyword(db, "All links target")
    note = db.add_note("Four links note", "Body")
    character = db.add_character_card({"name": "Four links character"})
    conversation = db.add_conversation({"character_id": character, "title": "Four links conversation"})
    collection = db.add_keyword_collection("Four links collection")
    card = db.add_flashcard({"front": "Four links card", "back": "Answer"})
    db.link_note_to_keyword(note, a["id"])
    db.keyword_store.link_conversation_to_keyword(conversation, a["id"])
    db.keyword_store.link_collection_to_keyword(collection, a["id"])
    db.set_flashcard_tags(card, [a["keyword"]])
    original = db.keyword_store._merge_keyword_links_for_table
    moved = []

    def failure(conn, **kwargs):
        result = original(conn, **kwargs)
        moved.append(kwargs["link_table"])
        if fail_after_links and len(moved) == 4:
            raise RuntimeError("Controlled failure after all membership moves")
        return result

    monkeypatch.setattr(db.keyword_store, "_merge_keyword_links_for_table", failure)
    if fail_after_links:
        with pytest.raises(RuntimeError, match="after all membership"):
            _merge(db, a, b)
        expected = a["id"]
        assert db.get_keyword_by_id(a["id"])["version"] == 1
    else:
        result = _merge(db, a, b)
        assert all(
            result[key] == 1
            for key in (
                "merged_note_links",
                "merged_conversation_links",
                "merged_collection_links",
                "merged_flashcard_links",
            )
        )
        assert _row(db, a["id"])["merged_into_sync_id"] == b["sync_id"]
        expected = b["id"]
    for table in ("note_keywords", "conversation_keywords", "collection_keywords", "flashcard_keywords"):
        rows = db.execute_query(f"SELECT keyword_id FROM {table}", read_only=True).fetchall()  # nosec B608
        assert [row["keyword_id"] for row in rows] == [expected]
