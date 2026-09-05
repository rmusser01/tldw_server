"""Persisted revision-clock contracts for guarded Reading mutations."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Barrier

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.media_db.media_database import MediaDatabase

pytestmark = pytest.mark.unit
pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_full_fixtures"]


@pytest.fixture(params=["sqlite", "postgres"])
def db(request, monkeypatch, tmp_path):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "users"))
    if request.param == "postgres":
        # The shared fixture otherwise replaces a named container on connection failure.
        monkeypatch.setenv("TLDW_TEST_NO_DOCKER", "1")
        _, db_name = request.getfixturevalue("isolated_test_environment")
        backend = DatabaseBackendFactory.create_backend(
            DatabaseConfig(
                backend_type=BackendType.POSTGRESQL,
                pg_host=os.getenv("TEST_DB_HOST", "localhost"),
                pg_port=int(os.getenv("TEST_DB_PORT", "5432")),
                pg_database=db_name,
                pg_user=os.getenv("TEST_DB_USER", "tldw_user"),
                pg_password=os.getenv("TEST_DB_PASSWORD", "TestPassword123!"),
                pg_sslmode=os.getenv("TEST_DB_SSLMODE", "prefer"),
            )
        )
        return CollectionsDatabase.from_backend(user_id="780", backend=backend)
    return CollectionsDatabase.for_user(780)


def make_reading(db, *, origin="reading"):
    return db.upsert_content_item(
        origin=origin,
        url="https://example.org/a",
        canonical_url="https://example.org/a",
        domain="example.org",
        title="Original",
        summary="Body",
        content_hash="a",
        word_count=1,
        published_at=None,
        tags=["news"],
    )


@pytest.mark.parametrize("operation", ["edit", "sync", "rollback", "overwrite"])
def test_external_media_writes_do_not_mutate_colliding_reading_highlights(db, tmp_path, operation):
    media = MediaDatabase(tmp_path / "media.db", client_id=db.user_id, backend=db.backend)
    media_id, _, _ = media.add_media_with_keywords(
        url="https://example.org/external", title="External", media_type="document", content="Old external body"
    )
    version = media.create_document_version(media_id=media_id, content="Historical external body")
    media.create_document_version(media_id=media_id, content="Latest external body")
    item = make_reading(db)
    assert media_id == item.id  # Deliberate collision between independent ID domains.
    highlight = db.create_highlight(item.id, "Body", 0, 4, None, None, content_hash_ref="a")
    before = db.get_content_item(item.id)
    if operation == "edit":
        media.apply_media_item_update(media_id=media_id, fields={"content": "Edited external body"})
    elif operation == "sync":
        media.apply_synced_document_content_update(media_id=media_id, content="Synced external body")
    elif operation == "rollback":
        result = media.rollback_to_version(media_id, version["version_number"])
        assert "error" not in result
    else:
        media.add_media_with_keywords(
            url="https://example.org/external",
            title="External",
            media_type="document",
            content="Replaced external body",
            overwrite=True,
        )
    expected = {
        "edit": "Edited external body",
        "sync": "Synced external body",
        "rollback": "Historical external body",
        "overwrite": "Replaced external body",
    }
    assert db.backend.execute("SELECT content FROM Media WHERE id = ?", (media_id,)).scalar == expected[operation]
    assert db.get_highlight(highlight.id) == highlight
    assert db.get_content_item(item.id) == before


def test_item_edits_advance_once_and_equivalent_values_are_noops(db):
    original = make_reading(db)
    changed = db.update_content_item(
        original.id,
        title="Changed",
        tags=[" news ", "new", "new"],
        metadata={"one": 1, "two": 2},
        favorite=True,
    )
    assert changed.revision == original.revision + 1
    same = db.update_content_item(
        original.id,
        title="Changed",
        tags=["new", "news"],
        metadata={"two": 2, "one": 1},
        favorite=True,
    )
    assert same.revision == changed.revision
    assert same.updated_at == changed.updated_at
    assert same.tags == ["new", "news"]
    assert json.loads(same.metadata_json) == {"one": 1, "two": 2}


def test_highlight_crud_advances_revision_and_ignores_equivalent_patch(db):
    item = make_reading(db)
    highlight = db.create_highlight(item.id, "Body", 0, 4, None, None)
    assert db.get_content_item(item.id).revision == item.revision + 1
    changed = db.update_highlight(highlight.id, {"note": "Annotation", "color": "yellow"})
    before = db.get_content_item(item.id)
    assert before.revision == item.revision + 2
    assert db.update_highlight(highlight.id, {"note": "Annotation", "color": "yellow"}) == changed
    assert db.get_content_item(item.id) == before
    assert db.delete_highlight(highlight.id)
    deleted = db.get_content_item(item.id)
    assert deleted.revision == before.revision + 1
    assert not db.delete_highlight(highlight.id)
    assert db.get_content_item(item.id) == deleted


@pytest.mark.parametrize("origin", ["missing", "watchlist", "foreign"])
def test_highlight_create_requires_owned_reading_parent(db, origin):
    item = make_reading(db, origin="watchlist" if origin == "watchlist" else "reading")
    if origin == "missing":
        db.delete_content_item(item.id)
    writer = CollectionsDatabase.from_backend(user_id="781", backend=db.backend) if origin == "foreign" else db
    with pytest.raises(KeyError):
        writer.create_highlight(item.id, "Body", 0, 4, None, None)
    assert db.backend.execute("SELECT COUNT(*) FROM reading_highlights", ()).scalar == 0


@pytest.mark.parametrize("operation", ["create", "update", "delete", "reanchor"])
def test_highlight_failure_rolls_back_child_and_revision(db, monkeypatch, operation):
    item = make_reading(db)
    highlight = db.create_highlight(item.id, "Body", 0, 4, None, None)
    before = db.get_content_item(item.id)
    children = db.list_highlights_by_item(item.id)
    allocate = db._next_reading_revision

    def allocate_then_fail(conn):
        allocate(conn)
        raise RuntimeError("abort highlight mutation")

    monkeypatch.setattr(db, "_next_reading_revision", allocate_then_fail)
    with pytest.raises(RuntimeError, match="abort highlight mutation"):
        if operation == "create":
            db.create_highlight(item.id, "Other", 0, 5, None, None)
        elif operation == "update":
            db.update_highlight(highlight.id, {"note": "Not saved"})
        elif operation == "delete":
            db.delete_highlight(highlight.id)
        else:
            db.reanchor_highlights_for_item(item.id, content_text="Body", content_hash="a")
    assert db.get_content_item(item.id) == before
    assert db.list_highlights_by_item(item.id) == children


def test_reanchor_changes_multiple_highlights_once_and_noops_preserve_revision(db):
    item = make_reading(db)
    first = db.create_highlight(item.id, "Body", 0, 4, None, None)
    second = db.create_highlight(item.id, "Missing", 0, 7, None, None)
    before = db.get_content_item(item.id)
    assert db.reanchor_highlights_for_item(item.id, content_text="Body", content_hash="a") == {
        "updated": 1,
        "stale": 1,
        "skipped": 0,
    }
    after = db.get_content_item(item.id)
    assert after.revision == before.revision + 1
    assert db.get_highlight(first.id).content_hash_ref == "a"
    assert db.get_highlight(second.id).state == "stale"
    db.reanchor_highlights_for_item(item.id, content_text="Body", content_hash="a")
    assert db.get_content_item(item.id) == after


def test_late_reanchor_cannot_overwrite_newer_highlight_edit(db, monkeypatch):
    item = make_reading(db)
    highlight = db.create_highlight(item.id, "Body", 0, 4, None, None)
    from tldw_Server_API.app.core.DB_Management import Collections_DB as module

    find_span = module.find_highlight_span

    def edit_during_matching(*args, **kwargs):
        with ThreadPoolExecutor(max_workers=1) as workers:
            workers.submit(db.update_highlight, highlight.id, {"quote": "New quote"}).result(timeout=15)
        return find_span(*args, **kwargs)

    monkeypatch.setattr(module, "find_highlight_span", edit_during_matching)
    result = db.reanchor_highlights_for_item(item.id, content_text="Body", content_hash="a")
    assert result == {"updated": 0, "stale": 0, "skipped": 1}
    assert db.get_highlight(highlight.id).quote == "New quote"
    assert db.get_highlight(highlight.id).content_hash_ref is None


def test_reanchor_rejects_content_hash_from_an_older_capture(db):
    item = make_reading(db)
    highlight = db.create_highlight(item.id, "Body", 0, 4, None, None)
    before = db.get_content_item(item.id)
    db.reanchor_highlights_for_item(item.id, content_text="Body", content_hash="obsolete")
    assert db.get_highlight(highlight.id) == highlight
    assert db.get_content_item(item.id) == before


def test_upsert_identical_record_is_noop_and_changes_advance(db):
    original = make_reading(db)
    same = make_reading(db)
    assert same.revision == original.revision
    assert same.updated_at == original.updated_at
    db.update_content_item(original.id, title="Changed")
    before = db.get_content_item(original.id)
    restored = make_reading(db)
    assert restored.title == "Original"
    assert restored.revision == before.revision + 1


def test_nonreading_upsert_keeps_existing_refresh_timestamp_semantics(db, monkeypatch):
    original = make_reading(db, origin="watchlist")
    monkeypatch.setattr(
        "tldw_Server_API.app.core.DB_Management.Collections_DB._utcnow_iso",
        lambda: "2099-01-01T00:00:00+00:00",
    )
    refreshed = make_reading(db, origin="watchlist")
    assert refreshed.updated_at == "2099-01-01T00:00:00+00:00"
    assert refreshed.revision == original.revision


def test_nonreading_explicit_update_keeps_timestamp_semantics(db, monkeypatch):
    original = make_reading(db, origin="watchlist")
    monkeypatch.setattr(
        "tldw_Server_API.app.core.DB_Management.Collections_DB._utcnow_iso",
        lambda: "2099-01-01T00:00:00+00:00",
    )
    refreshed = db.update_content_item(original.id, title=original.title)
    assert refreshed.updated_at == "2099-01-01T00:00:00+00:00"
    assert refreshed.revision == original.revision


def test_recreated_reading_item_never_reuses_deleted_revision(db):
    original = make_reading(db)
    db.delete_content_item(original.id)
    recreated = make_reading(db)
    assert recreated.revision > original.revision


def test_failed_tag_write_rolls_back_item_and_revision(db, monkeypatch):
    original = make_reading(db)
    clock = db.backend.execute("SELECT value FROM reading_revision_clock WHERE id = 1", ()).scalar
    original_replace = db._replace_item_tags

    def replace_then_fail(*args, **kwargs):
        original_replace(*args, **kwargs)
        raise RuntimeError("abort tags")

    monkeypatch.setattr(db, "_replace_item_tags", replace_then_fail)
    with pytest.raises(RuntimeError, match="abort tags"):
        db.update_content_item(original.id, title="Do not save", tags=["replacement"])
    remaining = db.get_content_item(original.id)
    assert remaining.title == "Original"
    assert remaining.tags == ["news"]
    assert remaining.revision == original.revision
    assert db.backend.execute("SELECT value FROM reading_revision_clock WHERE id = 1", ()).scalar == clock


def test_failed_insert_rolls_back_item_tags_and_clock(db, monkeypatch):
    clock = db.backend.execute("SELECT value FROM reading_revision_clock WHERE id = 1", ()).scalar
    replace = db._replace_item_tags

    def replace_then_fail(*args, **kwargs):
        replace(*args, **kwargs)
        raise RuntimeError("abort insert")

    monkeypatch.setattr(db, "_replace_item_tags", replace_then_fail)
    with pytest.raises(RuntimeError, match="abort insert"):
        make_reading(db)
    assert db.list_content_items(origin="reading")[1] == 0
    assert db.backend.execute("SELECT COUNT(*) FROM content_item_tags", ()).scalar == 0
    assert db.backend.execute("SELECT value FROM reading_revision_clock WHERE id = 1", ()).scalar == clock


def test_concurrent_identical_upserts_create_one_item_and_revision(db):
    ready = Barrier(2)

    def save():
        ready.wait(timeout=10)
        return make_reading(db)

    with ThreadPoolExecutor(max_workers=2) as workers:
        first, second = [workers.submit(save) for _ in range(2)]
        rows = [first.result(timeout=15), second.result(timeout=15)]
    assert rows[0].id == rows[1].id
    assert rows[0].revision == rows[1].revision
    assert sorted(row.is_new for row in rows) == [False, True]


def test_read_paths_return_persisted_revision(db):
    original = make_reading(db)
    changed = db.update_content_item(original.id, notes="New notes")
    assert db.get_content_item(original.id).revision == changed.revision
    assert db.get_content_item_by_url(original.url).revision == changed.revision
    rows, total = db.list_content_items(origin="reading")
    assert total == 1
    assert rows[0].revision == changed.revision
    db.backend.execute("UPDATE content_items SET media_id = 42 WHERE id = ?", (original.id,))
    assert db.get_content_item_by_media_id(42).revision == changed.revision


def test_fts_failure_rolls_back_item_tags_and_revision(db, monkeypatch):
    original = make_reading(db)
    before = db.get_content_item(original.id)

    def fail_index(*args, **kwargs):
        raise RuntimeError("index unavailable")

    monkeypatch.setattr(db, "_update_content_fts_entry", fail_index)
    with pytest.raises(RuntimeError, match="index unavailable"):
        db.update_content_item(original.id, title="Not committed", tags=["not-committed"])
    assert db.get_content_item(original.id) == before


def test_concurrent_disjoint_item_updates_preserve_both_changes(db):
    original = make_reading(db)
    ready = Barrier(2)

    def update(**kwargs):
        ready.wait(timeout=10)
        return db.update_content_item(original.id, **kwargs).revision

    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(update, title="Edited")
        second = workers.submit(update, notes="Annotated")
        revisions = [first.result(timeout=15), second.result(timeout=15)]
    final = db.get_content_item(original.id)
    assert len(set(revisions)) == 2
    assert final.revision == max(revisions)
    assert final.title == "Edited"
    assert final.notes == "Annotated"


def test_url_read_keeps_tags_and_revision_in_one_snapshot(db, monkeypatch):
    original = make_reading(db)
    fetch = db._fetch_tags_for_item_ids
    armed = True

    def fetch_after_other_writer(item_ids, *, connection=None):
        nonlocal armed
        if armed:
            armed = False
            with ThreadPoolExecutor(max_workers=1) as workers:
                workers.submit(db.update_content_item, original.id, tags=["changed"]).result(timeout=15)
        return fetch(item_ids, connection=connection)

    monkeypatch.setattr(db, "_fetch_tags_for_item_ids", fetch_after_other_writer)
    snapshot = db.get_content_item_by_url(original.url)
    assert snapshot.tags == ["news"]
    assert snapshot.revision == original.revision
    assert db.get_content_item(original.id).tags == ["changed"]


def test_later_adapters_update_existing_item_and_search_index(db):
    original = make_reading(db)
    # First construction may create the file; only subsequent constructions hit the schema memo.
    CollectionsDatabase.from_backend(user_id=db.user_id, backend=db.backend)
    second = CollectionsDatabase.from_backend(user_id=db.user_id, backend=db.backend)
    changed = second.update_content_item(original.id, title="Replacement")
    assert changed.revision > original.revision
    matches, total = second.list_content_items(origin="reading", q="Replacement")
    assert total == 1
    assert matches[0].id == original.id
    assert second.list_content_items(origin="reading", q="Original")[1] == 0


def test_revision_clock_survives_schema_reinitialization(db):
    with db.transaction() as conn:
        first = db._next_reading_revision(conn)
    db.ensure_schema()
    with db.transaction() as conn:
        second = db._next_reading_revision(conn)
    assert first > 0
    assert second > first


def make_archive_output(db):
    return db.create_output_artifact(
        type_="reading_archive",
        title="Capture archive",
        format_="md",
        storage_path="capture.md",
        metadata_json='{"item_id": 99999}',
    )


def mutate_output(db, output_id, operation):
    if operation == "metadata":
        return db.update_output_artifact_metadata(
            output_id, metadata_json='{"changed": true}', chatbook_path="book.zip"
        )
    if operation == "media":
        return db.update_output_media_item_id(output_id, 42)
    if operation == "rename":
        return db.rename_output_artifact(output_id, "Renamed", "renamed.md")
    from tldw_Server_API.app.services.outputs_service import update_output_artifact_db

    return update_output_artifact_db(db, output_id, "Converted", "converted.html", "html", "2030-01-01T00:00:00")


@pytest.mark.parametrize("operation", ["metadata", "media", "rename", "service"])
def test_owned_output_updates_advance_once_and_replays_are_noops(db, operation):
    item = make_reading(db)
    output = make_archive_output(db)
    db.register_reading_output_ownership(
        item.id, output.id, expected_revision=item.revision, storage_namespace_id="test-volume"
    )
    before = db.get_content_item(item.id)
    changed = mutate_output(db, output.id, operation)
    assert changed != output
    updated = db.get_content_item(item.id)
    assert updated.revision == before.revision + 1
    assert mutate_output(db, output.id, operation) == changed
    assert db.get_content_item(item.id) == updated


@pytest.mark.parametrize("operation", ["metadata", "media", "rename", "service"])
def test_owned_output_updates_roll_back_with_parent_and_clock(db, monkeypatch, operation):
    item = make_reading(db)
    output = make_archive_output(db)
    db.register_reading_output_ownership(
        item.id, output.id, expected_revision=item.revision, storage_namespace_id="test-volume"
    )
    before = db.get_content_item(item.id)
    old_row = db.backend.execute("SELECT * FROM outputs WHERE id = ?", (output.id,)).first
    clock = db.backend.execute("SELECT value FROM reading_revision_clock WHERE id = 1", ()).scalar
    advance = db._advance_reading_parent

    def fail(item_id, conn):
        advance(item_id, conn)
        raise RuntimeError("abort output update")

    monkeypatch.setattr(db, "_advance_reading_parent", fail)
    with pytest.raises(RuntimeError, match="abort output update"):
        mutate_output(db, output.id, operation)
    assert db.backend.execute("SELECT * FROM outputs WHERE id = ?", (output.id,)).first == old_row
    assert db.get_content_item(item.id) == before
    assert db.backend.execute("SELECT value FROM reading_revision_clock WHERE id = 1", ()).scalar == clock


@pytest.mark.parametrize("operation", ["metadata", "media", "rename", "service"])
def test_unowned_output_updates_do_not_mutate_reading_items(db, operation):
    item = make_reading(db)
    before = db.get_content_item(item.id)
    output = make_archive_output(db)
    db.update_output_artifact_metadata(output.id, metadata_json=json.dumps({"item_id": item.id}))
    mutate_output(db, output.id, operation)
    assert db.get_content_item(item.id) == before
    assert db.backend.execute("SELECT COUNT(*) FROM reading_output_ownership", ()).scalar == 0


@pytest.mark.parametrize("operation", ["metadata", "media", "rename", "service"])
def test_output_updates_reject_deleted_and_foreign_rows_without_mutation(db, operation):
    output = make_archive_output(db)
    foreign = CollectionsDatabase.from_backend(user_id="781", backend=db.backend)
    with pytest.raises(KeyError):
        mutate_output(foreign, output.id, operation)
    assert db.get_output_artifact(output.id) == output
    db.delete_output_artifact(output.id)
    with pytest.raises(KeyError):
        mutate_output(db, output.id, operation)
    assert db.get_output_artifact(output.id, include_deleted=True) == output


def test_owned_output_metadata_json_normalization_preserves_token(db):
    item = make_reading(db)
    output = make_archive_output(db)
    db.register_reading_output_ownership(
        item.id, output.id, expected_revision=item.revision, storage_namespace_id="test-volume"
    )
    db.update_output_artifact_metadata(output.id, metadata_json='{"a": 1, "b": [true]}')
    before = db.get_content_item(item.id)
    old_row = db.get_output_artifact(output.id)
    db.update_output_artifact_metadata(output.id, metadata_json=' {"b":[true], "a":1} ')
    assert db.get_content_item(item.id) == before
    assert db.get_output_artifact(output.id) == old_row
    db.update_output_artifact_metadata(output.id, metadata_json='{"a": true, "b": [true]}')
    assert db.get_content_item(item.id).revision == before.revision + 1


def test_owned_output_media_link_can_be_cleared(db):
    item = make_reading(db)
    output = make_archive_output(db)
    db.register_reading_output_ownership(
        item.id, output.id, expected_revision=item.revision, storage_namespace_id="test-volume"
    )
    db.update_output_media_item_id(output.id, 42)
    before = db.get_content_item(item.id)
    assert db.update_output_media_item_id(output.id, None).media_item_id is None
    cleared = db.get_content_item(item.id)
    assert cleared.revision == before.revision + 1
    db.update_output_media_item_id(output.id, None)
    assert db.get_content_item(item.id) == cleared


def test_concurrent_identical_owned_output_updates_advance_once(db):
    item = make_reading(db)
    output = make_archive_output(db)
    db.register_reading_output_ownership(
        item.id, output.id, expected_revision=item.revision, storage_namespace_id="test-volume"
    )
    before = db.get_content_item(item.id)
    ready = Barrier(2)

    def update():
        ready.wait(timeout=10)
        return mutate_output(db, output.id, "metadata")

    with ThreadPoolExecutor(max_workers=2) as workers:
        first, second = [workers.submit(update) for _ in range(2)]
        assert first.result(timeout=15) == second.result(timeout=15)
    assert db.get_content_item(item.id).revision == before.revision + 1


def test_owned_output_retention_only_updates_advance_once(db):
    from tldw_Server_API.app.services.outputs_service import update_output_artifact_db

    item = make_reading(db)
    output = make_archive_output(db)
    db.register_reading_output_ownership(
        item.id, output.id, expected_revision=item.revision, storage_namespace_id="test-volume"
    )
    before = db.get_content_item(item.id)
    expiry = "2030-01-01T00:00:00"
    update_output_artifact_db(db, output.id, None, None, None, expiry)
    changed = db.get_content_item(item.id)
    assert changed.revision == before.revision + 1
    assert db.backend.execute("SELECT retention_until FROM outputs WHERE id = ?", (output.id,)).scalar == expiry
    update_output_artifact_db(db, output.id, None, None, None, expiry)
    update_output_artifact_db(db, output.id, None, None, None, None)
    assert db.get_content_item(item.id) == changed


def test_output_update_validates_path_before_one_explicit_connection_fence(db, monkeypatch):
    item = make_reading(db)
    output = make_archive_output(db)
    db.register_reading_output_ownership(
        item.id, output.id, expected_revision=item.revision, storage_namespace_id="test-volume"
    )
    statements = []
    connections = []
    execute = db.backend.execute
    resolve = db.resolve_output_storage_path

    def trace(query, params=None, *, connection=None, **kwargs):
        statements.append(query)
        connections.append(connection)
        return execute(query, params, connection=connection, **kwargs)

    def resolve_before_lock(path):
        assert statements == []
        return resolve(path)

    monkeypatch.setattr(db.backend, "execute", trace)
    monkeypatch.setattr(db, "resolve_output_storage_path", resolve_before_lock)
    mutate_output(db, output.id, "rename")
    assert "UPDATE" in statements[0] and "reading_revision_clock" in statements[0]
    assert connections[0] is not None
    assert all(conn is connections[0] for conn in connections)


def test_explicit_output_ownership_advances_once_and_survives_schema_reinit(db):
    item = make_reading(db)
    output = make_archive_output(db)
    db.register_reading_output_ownership(
        item.id, output.id, expected_revision=item.revision, storage_namespace_id="test-volume"
    )
    linked = db.get_content_item(item.id)
    assert linked.revision == item.revision + 1
    db.ensure_schema()
    db.register_reading_output_ownership(
        item.id, output.id, expected_revision=item.revision, storage_namespace_id="test-volume"
    )
    assert db.get_content_item(item.id) == linked
    row = db.backend.execute("SELECT * FROM reading_output_ownership WHERE output_id = ?", (output.id,)).first
    assert row["item_id"] == item.id
    assert row["user_id"] == db.user_id
    assert row["storage_namespace_id"] == "test-volume"


def test_output_ownership_cannot_be_created_or_reassigned_by_metadata(db):
    item = make_reading(db)
    output = make_archive_output(db)
    db.ensure_schema()
    assert db.backend.execute("SELECT COUNT(*) FROM reading_output_ownership", ()).scalar == 0
    db.register_reading_output_ownership(
        item.id, output.id, expected_revision=item.revision, storage_namespace_id="test-volume"
    )
    db.update_output_artifact_metadata(output.id, metadata_json='{"item_id": 123456}')
    row = db.backend.execute("SELECT item_id FROM reading_output_ownership WHERE output_id = ?", (output.id,)).first
    assert row["item_id"] == item.id


def test_stale_output_ownership_registration_changes_nothing(db):
    from tldw_Server_API.app.core.DB_Management.Collections_DB import ReadingRevisionConflict

    item = make_reading(db)
    output = make_archive_output(db)
    before = db.update_content_item(item.id, title="Newer capture")
    with pytest.raises(ReadingRevisionConflict):
        db.register_reading_output_ownership(
            item.id, output.id, expected_revision=item.revision, storage_namespace_id="test-volume"
        )
    assert db.get_content_item(item.id) == before
    assert db.backend.execute("SELECT COUNT(*) FROM reading_output_ownership", ()).scalar == 0


@pytest.mark.parametrize("conflict", ["parent", "namespace"])
def test_existing_output_ownership_cannot_be_reassigned(db, conflict):
    from tldw_Server_API.app.core.DB_Management.Collections_DB import ReadingArtifactOwnershipConflict

    item = make_reading(db)
    output = make_archive_output(db)
    db.register_reading_output_ownership(
        item.id, output.id, expected_revision=item.revision, storage_namespace_id="test-volume"
    )
    other = db.upsert_content_item(
        origin="reading",
        url="https://example.org/other",
        canonical_url=None,
        domain=None,
        title="Other",
        summary=None,
        content_hash="other",
        word_count=1,
        published_at=None,
    )
    target = other if conflict == "parent" else db.get_content_item(item.id)
    before = db.get_content_item(target.id)
    with pytest.raises(ReadingArtifactOwnershipConflict):
        db.register_reading_output_ownership(
            target.id,
            output.id,
            expected_revision=target.revision,
            storage_namespace_id="other-volume" if conflict == "namespace" else "test-volume",
        )
    assert db.get_content_item(target.id) == before
    row = db.backend.execute("SELECT * FROM reading_output_ownership WHERE output_id = ?", (output.id,)).first
    assert row["item_id"] == item.id
    assert row["storage_namespace_id"] == "test-volume"


def test_output_ownership_registration_rolls_back_on_revision_failure(db, monkeypatch):
    item = make_reading(db)
    output = make_archive_output(db)
    before = db.get_content_item(item.id)
    allocate = db._next_reading_revision

    def fail(conn):
        allocate(conn)
        raise RuntimeError("abort ownership")

    monkeypatch.setattr(db, "_next_reading_revision", fail)
    with pytest.raises(RuntimeError, match="abort ownership"):
        db.register_reading_output_ownership(
            item.id, output.id, expected_revision=item.revision, storage_namespace_id="test-volume"
        )
    assert db.backend.execute("SELECT COUNT(*) FROM reading_output_ownership", ()).scalar == 0
    assert db.get_content_item(item.id) == before


def test_output_ownership_requires_same_user_parent_and_output(db):
    from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError

    item = make_reading(db)
    output = make_archive_output(db)
    foreign = CollectionsDatabase.from_backend(user_id="781", backend=db.backend)
    other = make_reading(foreign)
    with pytest.raises(KeyError):
        foreign.register_reading_output_ownership(
            item.id, output.id, expected_revision=item.revision, storage_namespace_id="test-volume"
        )
    with pytest.raises(KeyError):
        foreign.register_reading_output_ownership(
            other.id, output.id, expected_revision=other.revision, storage_namespace_id="test-volume"
        )
    with pytest.raises(DatabaseError):
        with db.transaction() as conn:
            db.backend.execute(
                "INSERT INTO reading_output_ownership (user_id, item_id, output_id, storage_namespace_id) VALUES (?, ?, ?, ?)",
                (foreign.user_id, other.id, output.id, "test-volume"),
                connection=conn,
            )
    assert db.backend.execute("SELECT COUNT(*) FROM reading_output_ownership", ()).scalar == 0


def test_concurrent_output_ownership_registration_advances_once(db):
    item = make_reading(db)
    output = make_archive_output(db)
    ready = Barrier(2)

    def register():
        ready.wait(timeout=10)
        return db.register_reading_output_ownership(
            item.id, output.id, expected_revision=item.revision, storage_namespace_id="test-volume"
        )

    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(register) for _ in range(2)]
        assert sorted(future.result(timeout=15) for future in futures) == [False, True]
    assert db.get_content_item(item.id).revision == item.revision + 1


@pytest.mark.parametrize("invalid", ["origin", "type", "revision", "namespace"])
def test_output_ownership_rejects_invalid_registration(db, invalid):
    item = make_reading(db, origin="watchlist" if invalid == "origin" else "reading")
    output = db.create_output_artifact(
        type_="summary" if invalid == "type" else "reading_archive",
        title="Output",
        format_="md",
        storage_path="capture.md",
    )
    before = db.get_content_item(item.id)
    with pytest.raises((KeyError, ValueError)):
        db.register_reading_output_ownership(
            item.id,
            output.id,
            expected_revision=0 if invalid == "revision" else item.revision,
            storage_namespace_id="" if invalid == "namespace" else "test-volume",
        )
    assert db.backend.execute("SELECT COUNT(*) FROM reading_output_ownership", ()).scalar == 0
    assert db.get_content_item(item.id) == before


def test_output_ownership_foreign_keys_prevent_dangling_associations(db):
    from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError

    item = make_reading(db)
    output = make_archive_output(db)
    db.register_reading_output_ownership(
        item.id, output.id, expected_revision=item.revision, storage_namespace_id="test-volume"
    )
    for table, row_id in [("content_items", item.id), ("outputs", output.id)]:
        with pytest.raises(DatabaseError):
            with db.transaction() as conn:
                db.backend.execute(f"DELETE FROM {table} WHERE id = ?", (row_id,), connection=conn)
    assert db.get_content_item(item.id).revision > item.revision
    assert db.get_output_artifact(output.id).id == output.id


def test_output_ownership_requires_nonnull_output_identity(db):
    from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError

    item = make_reading(db)
    with pytest.raises(DatabaseError):
        with db.transaction() as conn:
            db.backend.execute(
                "INSERT INTO reading_output_ownership (user_id, item_id, output_id, storage_namespace_id) VALUES (?, ?, NULL, ?)",
                (db.user_id, item.id, "test-volume"),
                connection=conn,
            )
    assert db.backend.execute("SELECT COUNT(*) FROM reading_output_ownership", ()).scalar == 0


def test_note_links_advance_revision_only_when_membership_changes(db):
    item = make_reading(db)
    link = db.link_note_to_content_item(item_id=item.id, note_id="external-note")
    linked = db.get_content_item(item.id)
    assert linked.revision == item.revision + 1
    duplicate = db.link_note_to_content_item(item_id=item.id, note_id="external-note")
    assert duplicate == link
    assert db.get_content_item(item.id) == linked
    assert db.unlink_note_from_content_item(item_id=item.id, note_id="external-note")
    unlinked = db.get_content_item(item.id)
    assert unlinked.revision == linked.revision + 1
    assert not db.unlink_note_from_content_item(item_id=item.id, note_id="external-note")
    assert db.get_content_item(item.id) == unlinked


@pytest.mark.parametrize("unlink", [False, True])
def test_note_link_failure_rolls_back_membership_and_clock(db, monkeypatch, unlink):
    item = make_reading(db)
    if unlink:
        db.link_note_to_content_item(item_id=item.id, note_id="external-note")
    before = db.get_content_item(item.id)
    links = db.list_note_links_for_content_item(item.id)
    clock = db.backend.execute("SELECT value FROM reading_revision_clock WHERE id = 1", ()).scalar
    allocate = db._next_reading_revision

    def allocate_then_fail(conn):
        allocate(conn)
        raise RuntimeError("abort link mutation")

    monkeypatch.setattr(db, "_next_reading_revision", allocate_then_fail)
    mutate = db.unlink_note_from_content_item if unlink else db.link_note_to_content_item
    with pytest.raises(RuntimeError, match="abort link mutation"):
        mutate(item_id=item.id, note_id="external-note")
    assert db.list_note_links_for_content_item(item.id) == links
    assert db.get_content_item(item.id) == before
    assert db.backend.execute("SELECT value FROM reading_revision_clock WHERE id = 1", ()).scalar == clock


def test_concurrent_duplicate_note_links_advance_once(db):
    item = make_reading(db)
    ready = Barrier(2)

    def link():
        ready.wait(timeout=10)
        return db.link_note_to_content_item(item_id=item.id, note_id="shared-note")

    with ThreadPoolExecutor(max_workers=2) as workers:
        first, second = [workers.submit(link) for _ in range(2)]
        assert first.result(timeout=15) == second.result(timeout=15)
    assert db.get_content_item(item.id).revision == item.revision + 1
    assert len(db.list_note_links_for_content_item(item.id)) == 1


def test_concurrent_note_link_and_item_edit_keep_both_revisions(db):
    item = make_reading(db)
    ready = Barrier(2)

    def link():
        ready.wait(timeout=10)
        db.link_note_to_content_item(item_id=item.id, note_id="external-note")

    def edit():
        ready.wait(timeout=10)
        db.update_content_item(item.id, title="Changed")

    with ThreadPoolExecutor(max_workers=2) as workers:
        first, second = workers.submit(link), workers.submit(edit)
        first.result(timeout=15)
        second.result(timeout=15)
    final = db.get_content_item(item.id)
    assert final.revision == item.revision + 2
    assert final.title == "Changed"
    assert [link.note_id for link in db.list_note_links_for_content_item(item.id)] == ["external-note"]


def test_note_links_reject_wrong_owner_and_deleted_parent(db):
    item = make_reading(db)
    db.link_note_to_content_item(item_id=item.id, note_id="external-note")
    before = db.get_content_item(item.id)
    other = CollectionsDatabase.from_backend(user_id="781", backend=db.backend)
    with pytest.raises(KeyError):
        other.link_note_to_content_item(item_id=item.id, note_id="other-note")
    assert not other.unlink_note_from_content_item(item_id=item.id, note_id="external-note")
    assert db.get_content_item(item.id) == before
    assert [link.note_id for link in db.list_note_links_for_content_item(item.id)] == ["external-note"]
    db.delete_content_item(item.id)
    with pytest.raises(KeyError):
        db.link_note_to_content_item(item_id=item.id, note_id="late-note")
    assert not db.unlink_note_from_content_item(item_id=item.id, note_id="external-note")
    assert db.backend.execute("SELECT COUNT(*) FROM content_item_note_links", ()).scalar == 0


def test_nonreading_note_links_preserve_item_revision_and_timestamp(db):
    item = make_reading(db, origin="watchlist")
    before = db.get_content_item(item.id)
    db.link_note_to_content_item(item_id=item.id, note_id="external-note")
    assert db.get_content_item(item.id) == before
    assert db.unlink_note_from_content_item(item_id=item.id, note_id="external-note")
    assert db.get_content_item(item.id) == before


def test_revision_allocation_rolls_back_with_transaction(db):
    with db.transaction() as conn:
        first = db._next_reading_revision(conn)
    with pytest.raises(RuntimeError, match="abort"):
        with db.transaction() as conn:
            db._next_reading_revision(conn)
            raise RuntimeError("abort")
    with db.transaction() as conn:
        next_committed = db._next_reading_revision(conn)
    assert next_committed == first + 1


def test_revision_clock_exhaustion_fails_without_wraparound(db):
    maximum = 2**63 - 1
    db.backend.execute("UPDATE reading_revision_clock SET value = ? WHERE id = 1", (maximum,))
    with pytest.raises(OverflowError, match="revision"):
        with db.transaction() as conn:
            db._next_reading_revision(conn)
    row = db.backend.execute("SELECT value FROM reading_revision_clock WHERE id = 1", ()).first
    assert row["value"] == maximum


def test_existing_items_receive_positive_persisted_revision(db):
    item = make_reading(db)
    row = db.backend.execute("SELECT revision FROM content_items WHERE id = ?", (item.id,)).first
    assert row["revision"] > 0


def test_clock_migration_preserves_existing_item_revision(db):
    item = make_reading(db)
    db.backend.execute("UPDATE content_items SET revision = 57 WHERE id = ?", (item.id,))
    db.ensure_schema()
    row = db.backend.execute("SELECT revision FROM content_items WHERE id = ?", (item.id,)).first
    with db.transaction() as conn:
        allocated = db._next_reading_revision(conn)
    assert row["revision"] == 57
    assert allocated > 57


def test_legacy_schema_upgrade_preserves_item_data(db):
    item = make_reading(db)
    # Reconstruct the pre-feature shape, retaining a real existing capture.
    db.backend.execute("ALTER TABLE content_items DROP COLUMN revision", ())
    db.backend.execute("DROP TABLE reading_revision_clock", ())
    db.ensure_schema()
    db.ensure_schema()
    row = db.backend.execute(
        "SELECT title, revision FROM content_items WHERE id = ?",
        (item.id,),
    ).first
    assert row["title"] == "Original"
    assert row["revision"] > 0


def test_deleted_item_does_not_reset_clock(db):
    item = make_reading(db)
    with db.transaction() as conn:
        issued = db._next_reading_revision(conn)
        db.backend.execute(
            "UPDATE content_items SET revision = ? WHERE id = ?",
            (issued, item.id),
            connection=conn,
        )
    db.delete_content_item(item.id)
    db.ensure_schema()
    with db.transaction() as conn:
        assert db._next_reading_revision(conn) > issued


def test_concurrent_transactions_never_issue_same_revision(db):
    ready = Barrier(2)

    def allocate():
        ready.wait(timeout=10)
        with db.transaction() as conn:
            return db._next_reading_revision(conn)

    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(allocate) for _ in range(2)]
        revisions = [future.result(timeout=15) for future in futures]
    assert len(set(revisions)) == 2
    assert min(revisions) > 0


def test_postgres_search_path_cannot_reset_public_revision_clock(db, monkeypatch):
    if db.backend.backend_type != BackendType.POSTGRESQL:
        pytest.skip("PostgreSQL search-path regression")
    with db.transaction() as conn:
        issued = db._next_reading_revision(conn)
        db.backend.execute("CREATE SCHEMA reading_revision_probe", (), connection=conn)
    original_transaction = db.transaction

    @contextmanager
    def prefixed_transaction():
        with original_transaction() as conn:
            db.backend.execute(
                "SET LOCAL search_path TO reading_revision_probe, public",
                (),
                connection=conn,
            )
            yield conn

    monkeypatch.setattr(db, "transaction", prefixed_transaction)
    db._ensure_reading_revision_schema()
    db._ensure_reading_revision_schema()
    with db.transaction() as conn:
        assert db._next_reading_revision(conn) > issued
