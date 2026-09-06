"""Guarded logical deletion and recoverable file disposal on both SQL backends."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from tldw_Server_API.app.core.DB_Management.Collections_DB import (
    CollectionsDatabase,
    ReadingArtifactOwnershipConflict,
    ReadingRevisionConflict,
)
from tldw_Server_API.app.services import reading_artifact_cleanup_service as service
from tldw_Server_API.tests.Collections.test_reading_artifact_adoption import adopt
from tldw_Server_API.tests.Collections.test_reading_artifact_cleanup import reserve
from tldw_Server_API.tests.Collections.test_reading_revision_mutations import db as db
from tldw_Server_API.tests.Collections.test_reading_revision_mutations import make_reading

pytestmark = pytest.mark.unit
pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_full_fixtures"]


def captured(db, root):
    item, namespace, reservation = reserve(db, root)
    output = adopt(db, namespace, reservation, root)
    db.create_highlight(item.id, "Body", 0, 4, None, None)
    db.link_note_to_content_item(item_id=item.id, note_id="external-note")
    collection = db.create_media_collection(name="Independent sources", kind="custom")
    entry = db.add_media_collection_item(
        collection_id=collection.id, source_url=item.url, content_item_id=item.id, media_id=42
    )
    return db.get_content_item(item.id), namespace, output, entry


def snapshot(db):
    return {
        table: [dict(row) for row in db.backend.execute(f"SELECT * FROM {table}", ()).rows]
        for table in (
            "content_items",
            "content_item_tags",
            "collection_tags",
            "content_item_note_links",
            "reading_highlights",
            "outputs",
            "reading_output_ownership",
            "reading_artifact_paths",
            "media_collections",
            "media_collection_items",
            "reading_revision_clock",
        )
    }


def test_exact_delete_preserves_containers_and_queues_file_until_cleanup(db, tmp_path):
    parent, namespace, output, entry = captured(db, tmp_path)
    tags = snapshot(db)["collection_tags"]
    assert db.hard_delete_reading_item(parent.id, expected_revision=parent.revision) is True
    with pytest.raises(KeyError):
        db.get_content_item(parent.id)
    with pytest.raises(KeyError):
        db.get_output_artifact(output.id, include_deleted=True)
    for table in ("content_item_tags", "content_item_note_links", "reading_highlights", "reading_output_ownership"):
        assert db.backend.execute(f"SELECT COUNT(*) FROM {table}", ()).scalar == 0
    assert snapshot(db)["collection_tags"] == tags
    assert db.get_media_collection(entry.collection_id).name == "Independent sources"
    remaining = db.get_media_collection_item(entry.id)
    assert remaining.content_item_id is None
    assert remaining.media_id == 42
    assert remaining.source_url == parent.url
    assert db.list_content_items(origin="reading", q="Original")[1] == 0
    path = tmp_path / output.storage_path
    assert path.exists()  # Logical deletion never unlinks inside its transaction.
    pending = db.backend.execute("SELECT * FROM reading_artifact_paths", ()).first
    assert pending["state"] == "pending"
    assert pending["storage_namespace_id"] == namespace
    assert pending["storage_path"] == output.storage_path
    with pytest.raises(ReadingArtifactOwnershipConflict):
        db.create_output_artifact(type_="other", title="Reused", format_="md", storage_path=output.storage_path.upper())
    restarted = CollectionsDatabase.from_backend(user_id=db.user_id, backend=db.backend)
    assert service.drain_reading_artifact_cleanup(restarted, output_root=tmp_path, storage_namespace_id=namespace) == 1
    assert not path.exists()
    assert service.drain_reading_artifact_cleanup(restarted, output_root=tmp_path, storage_namespace_id=namespace) == 0
    with pytest.raises(KeyError):
        db.hard_delete_reading_item(parent.id, expected_revision=parent.revision)


@pytest.mark.parametrize("case", ["stale", "foreign", "missing", "watchlist", "zero", "bool", "string"])
def test_rejected_delete_changes_nothing(db, case):
    item = make_reading(db, origin="watchlist" if case == "watchlist" else "reading")
    writer = CollectionsDatabase.from_backend(user_id="781", backend=db.backend) if case == "foreign" else db
    token = {"zero": 0, "bool": True, "string": str(item.revision)}.get(case, item.revision)
    if case == "stale":
        db.update_content_item(item.id, title="Newer")
    before = snapshot(db)
    error = (
        ReadingRevisionConflict if case == "stale" else (ValueError if case in {"zero", "bool", "string"} else KeyError)
    )
    with pytest.raises(error):
        writer.hard_delete_reading_item(item.id + 99 if case == "missing" else item.id, expected_revision=token)
    assert snapshot(db) == before


@pytest.mark.parametrize("reference", ["manual", "older", "parent", "foreign_parent"])
def test_unproven_legacy_archives_block_without_guessing_ownership(db, reference):
    item = make_reading(db)
    foreign = CollectionsDatabase.from_backend(user_id="781", backend=db.backend)
    output = (foreign if reference == "foreign_parent" else db).create_output_artifact(
        type_="reading_archive",
        title="Legacy",
        format_="md",
        storage_path="legacy.md",
        metadata_json=json.dumps({"item_id": item.id}) if reference in {"manual", "older"} else "{}",
    )
    if reference in {"parent", "foreign_parent"}:
        db.update_content_item(item.id, metadata={"archive_output_id": output.id})
    if reference == "older":
        db.update_content_item(item.id, metadata={"archive_output_id": output.id + 100})
    parent = db.get_content_item(item.id)
    before = snapshot(db)
    with pytest.raises(ReadingArtifactOwnershipConflict):
        db.hard_delete_reading_item(item.id, expected_revision=parent.revision)
    assert snapshot(db) == before


@pytest.mark.parametrize("soft_deleted", [False, True])
def test_surviving_exact_path_output_keeps_shared_file(db, tmp_path, soft_deleted):
    parent, namespace, output, _ = captured(db, tmp_path)
    other = db.create_output_artifact(type_="other", title="Shared", format_="md", storage_path=output.storage_path)
    if soft_deleted:
        db.delete_output_artifact(other.id)
    assert db.hard_delete_reading_item(parent.id, expected_revision=parent.revision) is False
    assert db.get_output_artifact(other.id, include_deleted=True).id == other.id
    assert db.backend.execute("SELECT COUNT(*) FROM reading_artifact_paths", ()).scalar == 0
    assert service.drain_reading_artifact_cleanup(db, output_root=tmp_path, storage_namespace_id=namespace) == 0
    assert (tmp_path / output.storage_path).exists()


@pytest.mark.parametrize("structural", [False, True])
def test_surviving_different_spelling_is_ambiguous_not_discarded(db, tmp_path, structural):
    parent, namespace, output, _ = captured(db, tmp_path)
    alias = db.create_output_artifact(
        type_="reading_archive" if structural else "other",
        title="Alias",
        format_="md",
        storage_path=output.storage_path.upper(),
    )
    if structural:
        other = db.upsert_content_item(
            origin="reading",
            url="https://example.org/other",
            canonical_url="https://example.org/other",
            domain="example.org",
            title="Other",
            summary="Body",
            content_hash="b",
            word_count=1,
            published_at=None,
        )
        db.register_reading_output_ownership(
            other.id, alias.id, expected_revision=other.revision, storage_namespace_id=namespace
        )
    before = snapshot(db)
    with pytest.raises(ReadingArtifactOwnershipConflict):
        db.hard_delete_reading_item(parent.id, expected_revision=parent.revision)
    assert snapshot(db) == before
    assert (tmp_path / output.storage_path).exists()


def test_delete_cancels_staging_without_waiting_for_storage_lock(db, tmp_path):
    item, namespace, reservation = reserve(db, tmp_path)
    with service.reading_storage_lock(tmp_path, storage_namespace_id=namespace):
        assert db.hard_delete_reading_item(item.id, expected_revision=item.revision) is True
    assert db.get_reading_artifact(reservation["token"], namespace)["state"] == "pending"
    with pytest.raises(ReadingArtifactOwnershipConflict):
        service.write_staged_reading_artifact(
            db, reservation["token"], output_root=tmp_path, storage_namespace_id=namespace, body="Too late"
        )
    assert service.drain_reading_artifact_cleanup(db, output_root=tmp_path, storage_namespace_id=namespace) == 1
    assert not (tmp_path / reservation["storage_path"]).exists()


def test_owned_case_variants_each_keep_an_intent_on_case_sensitive_storage(db, tmp_path):
    parent, namespace, output, _ = captured(db, tmp_path)
    alias = db.create_output_artifact(
        type_="reading_archive", title="Second archive", format_="md", storage_path=output.storage_path.upper()
    )
    db.register_reading_output_ownership(
        parent.id, alias.id, expected_revision=parent.revision, storage_namespace_id=namespace
    )
    parent = db.get_content_item(parent.id)
    assert db.hard_delete_reading_item(parent.id, expected_revision=parent.revision)
    paths = {
        row["storage_path"] for row in db.backend.execute("SELECT storage_path FROM reading_artifact_paths", ()).rows
    }
    # Distinct names can be distinct files on Linux. On case-insensitive volumes
    # one unlink plus one validated absence still safely retires both intents.
    assert paths == {output.storage_path, alias.storage_path}
    assert service.drain_reading_artifact_cleanup(db, output_root=tmp_path, storage_namespace_id=namespace) == 2
    assert not (tmp_path / output.storage_path).exists()


def test_exact_parent_predicate_miss_rolls_back_children_and_intents(db, tmp_path, monkeypatch):
    parent, _, _, _ = captured(db, tmp_path)
    before = snapshot(db)
    execute = db.backend.execute

    def miss_parent(query, params=(), **kwargs):
        if query.startswith("DELETE FROM content_items WHERE"):
            assert (
                query
                == "DELETE FROM content_items WHERE id = ? AND user_id = ? AND origin = 'reading' AND revision = ?"
            )
            params = (*params[:-1], parent.revision + 1)
        return execute(query, params, **kwargs)

    monkeypatch.setattr(db.backend, "execute", miss_parent)
    with pytest.raises(ReadingRevisionConflict):
        db.hard_delete_reading_item(parent.id, expected_revision=parent.revision)
    assert snapshot(db) == before
    assert db.list_content_items(origin="reading", q="Original")[1] == 1


def test_known_distinct_volumes_do_not_lose_or_block_cleanup(db, tmp_path):
    first, first_namespace, output, _ = captured(db, tmp_path)
    other_root = tmp_path / "other-volume"
    other_root.mkdir()
    other_namespace = service.provision_reading_storage_namespace(other_root)
    second = db.upsert_content_item(
        origin="reading",
        url="https://example.org/second",
        canonical_url="https://example.org/second",
        domain="example.org",
        title="Second",
        summary="Body",
        content_hash="b",
        word_count=1,
        published_at=None,
    )
    other_output = db.create_output_artifact(
        type_="reading_archive", title="Other volume archive", format_="md", storage_path=output.storage_path
    )
    db.register_reading_output_ownership(
        second.id, other_output.id, expected_revision=second.revision, storage_namespace_id=other_namespace
    )
    assert db.hard_delete_reading_item(first.id, expected_revision=first.revision) is True
    pending = db.backend.execute("SELECT * FROM reading_artifact_paths", ()).first
    assert pending["storage_namespace_id"] == first_namespace
    assert service.drain_reading_artifact_cleanup(db, output_root=other_root, storage_namespace_id=other_namespace) == 0
    assert service.drain_reading_artifact_cleanup(db, output_root=tmp_path, storage_namespace_id=first_namespace) == 1
    assert not (tmp_path / output.storage_path).exists()
    assert db.get_output_artifact(other_output.id).id == other_output.id
    second = db.get_content_item(second.id)
    assert db.hard_delete_reading_item(second.id, expected_revision=second.revision) is True
    assert service.drain_reading_artifact_cleanup(db, output_root=other_root, storage_namespace_id=other_namespace) == 1


def test_deletion_without_files_preserves_clock_against_id_reuse(db):
    parent = make_reading(db)
    clock = db.backend.execute("SELECT value FROM reading_revision_clock WHERE id = 1", ()).scalar
    assert db.hard_delete_reading_item(parent.id, expected_revision=parent.revision) is False
    assert db.backend.execute("SELECT value FROM reading_revision_clock WHERE id = 1", ()).scalar == clock
    replacement = make_reading(db)
    assert replacement.revision > parent.revision
    with pytest.raises(ReadingRevisionConflict):
        db.hard_delete_reading_item(replacement.id, expected_revision=parent.revision)


def test_external_media_and_note_records_survive(db, tmp_path):
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
    from tldw_Server_API.app.core.DB_Management.media_db.media_database import MediaDatabase

    media = MediaDatabase(tmp_path / "media.db", client_id=db.user_id, backend=db.backend)
    media_id, _, _ = media.add_media_with_keywords(
        url="https://example.org/external", title="External", media_type="document", content="Independent body"
    )
    # Notes own a separate database in production; do not merge their unrelated
    # keyword/schema tables into the Collections/Media test backend.
    notes = CharactersRAGDB(tmp_path / "notes.db", client_id=db.user_id)
    note_id = notes.add_note(title="External note", content="Independent note")
    parent = db.upsert_content_item(
        origin="reading",
        url="https://example.org/capture",
        canonical_url="https://example.org/capture",
        domain="example.org",
        title="Capture",
        summary="Body",
        content_hash="c",
        word_count=1,
        published_at=None,
        media_id=media_id,
    )
    db.link_note_to_content_item(item_id=parent.id, note_id=note_id)
    parent = db.get_content_item(parent.id)
    before_media = dict(db.backend.execute("SELECT * FROM Media WHERE id = ?", (media_id,)).first)
    before_note = dict(notes.backend.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).first)
    assert db.hard_delete_reading_item(parent.id, expected_revision=parent.revision) is False
    assert dict(db.backend.execute("SELECT * FROM Media WHERE id = ?", (media_id,)).first) == before_media
    assert dict(notes.backend.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).first) == before_note


@pytest.mark.parametrize(
    "phase", ["intent", "ownership", "output", "tags", "notes", "highlights", "collections", "fts", "parent", "staging"]
)
def test_delete_failure_rolls_back_every_phase(db, tmp_path, monkeypatch, phase):
    parent, namespace, output, _ = captured(db, tmp_path)
    db.reserve_reading_artifact(
        parent.id, expected_revision=parent.revision, storage_namespace_id=namespace, lease_until=2**62
    )
    before = snapshot(db)
    execute = db.backend.execute
    delete_fts = db._delete_content_fts_entry
    prefixes = {
        "intent": "INSERT INTO reading_artifact_paths",
        "ownership": "DELETE FROM reading_output_ownership",
        "output": "DELETE FROM outputs",
        "tags": "DELETE FROM content_item_tags",
        "notes": "DELETE FROM content_item_note_links",
        "highlights": "DELETE FROM reading_highlights",
        "collections": "UPDATE media_collection_items",
        "parent": "DELETE FROM content_items WHERE",
        "staging": "UPDATE reading_artifact_paths SET state",
    }

    def fail_sql(query, *args, **kwargs):
        result = execute(query, *args, **kwargs)
        if phase != "fts" and query.startswith(prefixes[phase]):
            raise RuntimeError("delete rollback")
        return result

    def fail_fts(*args, **kwargs):
        delete_fts(*args, **kwargs)
        raise RuntimeError("delete rollback")

    with monkeypatch.context() as patch:
        patch.setattr(db.backend, "execute", fail_sql)
        if phase == "fts":
            patch.setattr(db, "_delete_content_fts_entry", fail_fts)
        with pytest.raises(RuntimeError, match="delete rollback"):
            db.hard_delete_reading_item(parent.id, expected_revision=parent.revision)
    assert snapshot(db) == before
    assert db.list_content_items(origin="reading", q="Original")[1] == 1
    assert (tmp_path / output.storage_path).exists()


@pytest.mark.parametrize("first", ["mutation", "delete"])
def test_mutation_and_delete_obey_fence_commit_order(db, monkeypatch, first):
    parent = make_reading(db)
    held, entering, release = Event(), Event(), Event()
    lock = db._lock_reading_revision_clock
    connections = []

    def controlled_lock(connection):
        connections.append(connection)
        if len(connections) == 1:
            lock(connection)
            held.set()
            assert release.wait(10)
        else:
            entering.set()
            lock(connection)

    monkeypatch.setattr(db, "_lock_reading_revision_clock", controlled_lock)

    # SQLite waits at BEGIN IMMEDIATE, earlier than the clock hook. Signal entry
    # into the second operation, not acquisition of a lock it cannot yet hold.
    def operation(kind):
        if held.is_set():
            entering.set()
        if kind == "mutation":
            return db.link_note_to_content_item(item_id=parent.id, note_id="racing-note")
        return db.hard_delete_reading_item(parent.id, expected_revision=parent.revision)

    with ThreadPoolExecutor(max_workers=2) as workers:
        leading = workers.submit(operation, first)
        try:
            assert held.wait(10)
            trailing = workers.submit(operation, "delete" if first == "mutation" else "mutation")
            assert entering.wait(10)
        finally:
            release.set()
        leading.result(timeout=15)
        with pytest.raises(ReadingRevisionConflict if first == "mutation" else KeyError):
            trailing.result(timeout=15)
    assert len(connections) == 2
    assert connections[0] is not connections[1]
    if first == "mutation":
        assert db.get_content_item(parent.id).revision > parent.revision
        assert db.list_note_links_for_content_item(parent.id)[0].note_id == "racing-note"
    else:
        assert db.backend.execute("SELECT COUNT(*) FROM content_item_note_links", ()).scalar == 0
        assert db.backend.execute("SELECT COUNT(*) FROM content_items", ()).scalar == 0


def test_sqlite_fts_sql_failure_is_not_suppressed(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError

    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "users"))
    database = CollectionsDatabase.for_user(780)
    parent = make_reading(database)
    assert database._fts_available
    before = snapshot(database)
    execute = database.backend.execute

    def fail_fts_sql(query, *args, **kwargs):
        result = execute(query, *args, **kwargs)
        if "content_items_fts" in query:
            raise DatabaseError("fts rollback")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(database.backend, "execute", fail_fts_sql)
        with pytest.raises(DatabaseError, match="fts rollback"):
            database.hard_delete_reading_item(parent.id, expected_revision=parent.revision)
    assert snapshot(database) == before
    assert database.list_content_items(origin="reading", q="Original")[1] == 1
