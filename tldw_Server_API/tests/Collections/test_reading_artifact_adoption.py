"""Atomic transition from private staging to a revision-owned Reading archive."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tldw_Server_API.app.core.DB_Management.Collections_DB import (
    ReadingArtifactOwnershipConflict,
    ReadingRevisionConflict,
)
from tldw_Server_API.app.services import reading_artifact_cleanup_service as service
from tldw_Server_API.tests.Collections.test_reading_artifact_cleanup import reserve
from tldw_Server_API.tests.Collections.test_reading_revision_mutations import db as db
from tldw_Server_API.tests.Collections.test_reading_revision_mutations import make_reading

pytestmark = pytest.mark.unit
pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_full_fixtures"]


def adopt(db, namespace, reservation, root):
    return service.write_and_adopt_reading_artifact(
        db,
        reservation["token"],
        output_root=root,
        storage_namespace_id=namespace,
        body="# Captured content",
        title="Capture archive",
        retention_until="2030-01-01T00:00:00",
    )


def test_adoption_commits_output_ownership_reference_and_one_revision(db, tmp_path):
    item, namespace, reservation = reserve(db, tmp_path)
    output = adopt(db, namespace, reservation, tmp_path)
    parent = db.get_content_item(item.id)
    assert parent.revision == item.revision + 1
    assert json.loads(parent.metadata_json)["archive_output_id"] == output.id
    assert output.type == "reading_archive"
    assert output.storage_path == reservation["storage_path"]
    assert (tmp_path / output.storage_path).read_text() == "# Captured content"
    owner = db.backend.execute("SELECT * FROM reading_output_ownership WHERE output_id = ?", (output.id,)).first
    assert owner["item_id"] == item.id
    assert owner["storage_namespace_id"] == namespace
    assert (
        db.backend.execute("SELECT retention_until FROM outputs WHERE id = ?", (output.id,)).scalar
        == "2030-01-01T00:00:00"
    )
    with pytest.raises(KeyError):
        db.get_reading_artifact(reservation["token"], namespace)
    with pytest.raises(KeyError):
        adopt(db, namespace, reservation, tmp_path)
    assert db.get_content_item(item.id) == parent
    assert service.drain_reading_artifact_cleanup(db, output_root=tmp_path, storage_namespace_id=namespace) == 0


@pytest.mark.parametrize("phase", ["output", "ownership", "metadata", "revision", "fts", "staging"])
def test_adoption_failure_rolls_back_and_leaves_file_pending(db, tmp_path, monkeypatch, phase):
    item, namespace, reservation = reserve(db, tmp_path)
    before = db.get_content_item(item.id)
    clock = db.backend.execute("SELECT value FROM reading_revision_clock WHERE id = 1", ()).scalar
    execute = db.backend.execute
    update_fts = db._update_content_fts_entry
    prefixes = {
        "output": "INSERT INTO outputs",
        "ownership": "INSERT INTO reading_output_ownership",
        "metadata": "UPDATE content_items SET metadata_json",
        "revision": "UPDATE content_items SET revision",
        "staging": "DELETE FROM reading_artifact_paths",
    }

    def fail_after_sql(query, *args, **kwargs):
        result = execute(query, *args, **kwargs)
        if phase != "fts" and query.startswith(prefixes[phase]):
            raise RuntimeError("adoption rollback")
        return result

    def fail_fts(*args, **kwargs):
        update_fts(*args, **kwargs)
        raise RuntimeError("adoption rollback")

    with monkeypatch.context() as patch:
        patch.setattr(db.backend, "execute", fail_after_sql)
        if phase == "fts":
            patch.setattr(db, "_update_content_fts_entry", fail_fts)
        with pytest.raises(RuntimeError, match="adoption rollback"):
            adopt(db, namespace, reservation, tmp_path)
    assert db.get_content_item(item.id) == before
    assert db.backend.execute("SELECT value FROM reading_revision_clock WHERE id = 1", ()).scalar == clock
    assert db.backend.execute("SELECT COUNT(*) FROM outputs", ()).scalar == 0
    assert db.backend.execute("SELECT COUNT(*) FROM reading_output_ownership", ()).scalar == 0
    matches, _ = db.list_content_items(origin="reading", q="Original")
    assert any(row.id == item.id for row in matches)
    assert db.get_reading_artifact(reservation["token"], namespace)["state"] == "pending"
    assert service.drain_reading_artifact_cleanup(db, output_root=tmp_path, storage_namespace_id=namespace) == 1
    assert not (tmp_path / reservation["storage_path"]).exists()


@pytest.mark.parametrize("interruption", ["edit", "expire", "cancel", "delete"])
def test_adoption_rechecks_parent_and_reservation_after_write(db, tmp_path, monkeypatch, interruption):
    item, namespace, reservation = reserve(db, tmp_path)
    sync = service._sync_directory

    def interrupt_after_file_sync(directory):
        sync(directory)
        if interruption == "edit":
            with ThreadPoolExecutor(max_workers=1) as workers:
                workers.submit(db.update_content_item, item.id, title="Newer").result(timeout=10)
        elif interruption == "expire":
            db.backend.execute(
                "UPDATE reading_artifact_paths SET lease_until = 0 WHERE token = ?", (reservation["token"],)
            )
        elif interruption == "delete":
            db.delete_content_item(item.id)
        else:
            db.cancel_reading_artifact(reservation["token"], namespace)

    with monkeypatch.context() as patch:
        patch.setattr(service, "_sync_directory", interrupt_after_file_sync)
        with pytest.raises((ReadingRevisionConflict, ReadingArtifactOwnershipConflict, KeyError)):
            adopt(db, namespace, reservation, tmp_path)
    assert db.backend.execute("SELECT COUNT(*) FROM outputs", ()).scalar == 0
    assert db.get_reading_artifact(reservation["token"], namespace)["state"] == "pending"
    assert service.drain_reading_artifact_cleanup(db, output_root=tmp_path, storage_namespace_id=namespace) == 1
    if interruption == "delete":
        with pytest.raises(KeyError):
            db.get_content_item(item.id)


def test_adoption_holds_storage_exclusion_until_transaction_finishes(db, tmp_path, monkeypatch):
    _, namespace, reservation = reserve(db, tmp_path)
    original = db.adopt_reading_artifact

    def check_exclusion(*args, **kwargs):
        with ThreadPoolExecutor(max_workers=1) as workers:
            cleanup = workers.submit(
                service.drain_reading_artifact_cleanup, db, output_root=tmp_path, storage_namespace_id=namespace
            )
            with pytest.raises(service.ReadingStorageBusy):
                cleanup.result(timeout=10)
        return original(*args, **kwargs)

    monkeypatch.setattr(db, "adopt_reading_artifact", check_exclusion)
    assert adopt(db, namespace, reservation, tmp_path).id > 0


def test_adoption_database_primitive_rejects_wrong_user_or_namespace(db, tmp_path):
    from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase

    _, namespace, reservation = reserve(db, tmp_path)
    foreign = CollectionsDatabase.from_backend(user_id="781", backend=db.backend)
    for target, volume in [(foreign, namespace), (db, "wrong-volume")]:
        with pytest.raises(KeyError):
            target.adopt_reading_artifact(reservation["token"], volume, title="wrong", now=int(time.time()))
    assert db.backend.execute("SELECT COUNT(*) FROM outputs", ()).scalar == 0


def test_waiting_writer_cannot_use_a_stale_prelock_timestamp(db, tmp_path):
    _, namespace, reservation = reserve(db, tmp_path)
    db.backend.execute("UPDATE reading_artifact_paths SET lease_until = 1 WHERE token = ?", (reservation["token"],))
    with pytest.raises(ReadingArtifactOwnershipConflict):
        db.validate_reading_artifact_for_write(reservation["token"], namespace, now=0)


def test_adoption_preserves_capture_fields_tags_and_external_associations(db, tmp_path):
    namespace = service.provision_reading_storage_namespace(tmp_path)
    item = make_reading(db)
    db.update_content_item(item.id, metadata={"preserved": {"nested": True}}, notes="Captured notes", tags=["retained"])
    link = db.link_note_to_content_item(item_id=item.id, note_id="external-note")
    db.backend.execute("UPDATE content_items SET media_id = 42 WHERE id = ?", (item.id,))
    before = db.get_content_item(item.id)
    reservation = db.reserve_reading_artifact(
        item.id, expected_revision=before.revision, storage_namespace_id=namespace, lease_until=int(time.time()) + 300
    )
    output = adopt(db, namespace, reservation, tmp_path)
    parent = db.get_content_item(item.id)
    assert parent.title == before.title
    assert parent.notes == before.notes
    assert parent.tags == before.tags
    assert parent.media_id == output.media_item_id == 42
    assert json.loads(parent.metadata_json)["preserved"] == {"nested": True}
    assert db.list_note_links_for_content_item(item.id) == [link]
    assert parent.revision == before.revision + 1
    matches, _ = db.list_content_items(origin="reading", q="Captured notes")
    assert any(row.id == item.id for row in matches)
