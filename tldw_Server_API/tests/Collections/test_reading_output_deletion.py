"""Owned archive deletion and retention share the Reading transaction fence."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase, ReadingRevisionConflict
from tldw_Server_API.app.services import reading_artifact_cleanup_service as cleanup
from tldw_Server_API.app.services.outputs_service import delete_outputs_by_ids
from tldw_Server_API.tests.Collections.test_reading_artifact_adoption import adopt
from tldw_Server_API.tests.Collections.test_reading_artifact_cleanup import reserve
from tldw_Server_API.tests.Collections.test_reading_atomic_delete import snapshot
from tldw_Server_API.tests.Collections.test_reading_revision_mutations import db as db

pytestmark = pytest.mark.unit
pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_full_fixtures"]


def archive(db, root):
    parent, namespace, reservation = reserve(db, root)
    output = adopt(db, namespace, reservation, root)
    return db.get_content_item(parent.id), namespace, output


@pytest.mark.parametrize("hard", [False, True])
def test_owned_deletion_advances_once_clears_reference_and_replays_as_noop(db, tmp_path, hard):
    parent, namespace, output = archive(db, tmp_path)
    assert db.delete_output_artifact(output.id, hard=hard)
    changed = db.get_content_item(parent.id)
    assert changed.revision == parent.revision + 1
    assert "archive_output_id" not in json.loads(changed.metadata_json)
    assert changed.title == parent.title
    assert not db.delete_output_artifact(output.id, hard=hard)
    assert db.get_content_item(parent.id) == changed
    with pytest.raises(ReadingRevisionConflict):
        db.hard_delete_reading_item(parent.id, expected_revision=parent.revision)
    assert (tmp_path / output.storage_path).exists()
    if hard:
        assert db.backend.execute("SELECT COUNT(*) FROM reading_output_ownership", ()).scalar == 0
        assert cleanup.drain_reading_artifact_cleanup(db, output_root=tmp_path, storage_namespace_id=namespace) == 1
        assert not (tmp_path / output.storage_path).exists()
    else:
        assert db.get_output_artifact(output.id, include_deleted=True).id == output.id
        assert db.backend.execute("SELECT COUNT(*) FROM reading_output_ownership", ()).scalar == 1
        assert cleanup.drain_reading_artifact_cleanup(db, output_root=tmp_path, storage_namespace_id=namespace) == 0


def test_hard_delete_after_soft_delete_advances_once_more(db, tmp_path):
    parent, namespace, output = archive(db, tmp_path)
    db.delete_output_artifact(output.id)
    before = db.get_content_item(parent.id)
    assert db.delete_output_artifact(output.id, hard=True)
    assert db.get_content_item(parent.id).revision == before.revision + 1
    assert cleanup.drain_reading_artifact_cleanup(db, output_root=tmp_path, storage_namespace_id=namespace) == 1


@pytest.mark.parametrize("phase", ["intent", "ownership", "output", "reference", "revision", "fts"])
def test_output_delete_rolls_back_every_phase(db, tmp_path, monkeypatch, phase):
    parent, _, output = archive(db, tmp_path)
    before = snapshot(db)
    execute = db.backend.execute
    update_fts = db._update_content_fts_entry
    prefixes = {
        "intent": "INSERT INTO reading_artifact_paths",
        "ownership": "DELETE FROM reading_output_ownership",
        "output": "DELETE FROM outputs",
        "reference": "UPDATE content_items SET metadata_json",
        "revision": "UPDATE content_items SET revision",
    }

    def fail_sql(query, *args, **kwargs):
        result = execute(query, *args, **kwargs)
        if phase != "fts" and query.startswith(prefixes[phase]):
            raise RuntimeError("output rollback")
        return result

    def fail_fts(*args, **kwargs):
        update_fts(*args, **kwargs)
        raise RuntimeError("output rollback")

    with monkeypatch.context() as patch:
        patch.setattr(db.backend, "execute", fail_sql)
        if phase == "fts":
            patch.setattr(db, "_update_content_fts_entry", fail_fts)
        with pytest.raises(RuntimeError, match="output rollback"):
            db.delete_output_artifact(output.id, hard=True)
    assert snapshot(db) == before
    assert (tmp_path / output.storage_path).exists()
    assert db.list_content_items(origin="reading", q="Original")[1] == 1


def test_sibling_output_sharing_keeps_file_until_last_owner(db, tmp_path):
    parent, namespace, first = archive(db, tmp_path)
    second = db.create_output_artifact(
        type_="reading_archive", title="Second", format_="md", storage_path=first.storage_path
    )
    db.register_reading_output_ownership(
        parent.id, second.id, expected_revision=parent.revision, storage_namespace_id=namespace
    )
    db.update_content_item(parent.id, metadata={"archive_output_id": second.id, "keep": "value"})
    before = db.get_content_item(parent.id)
    assert db.delete_output_artifact(first.id, hard=True)
    changed = db.get_content_item(parent.id)
    assert changed.revision == before.revision + 1
    assert json.loads(changed.metadata_json) == {"archive_output_id": second.id, "keep": "value"}
    assert cleanup.drain_reading_artifact_cleanup(db, output_root=tmp_path, storage_namespace_id=namespace) == 0
    assert db.delete_output_artifact(second.id, hard=True)
    assert json.loads(db.get_content_item(parent.id).metadata_json) == {"keep": "value"}
    assert cleanup.drain_reading_artifact_cleanup(db, output_root=tmp_path, storage_namespace_id=namespace) == 1


def test_foreign_output_delete_is_nonmutating(db, tmp_path):
    _, _, output = archive(db, tmp_path)
    before = snapshot(db)
    foreign = CollectionsDatabase.from_backend(user_id="781", backend=db.backend)
    assert not foreign.delete_output_artifact(output.id, hard=True)
    assert snapshot(db) == before


def test_concurrent_duplicate_deletes_advance_only_once(db, tmp_path):
    parent, _, output = archive(db, tmp_path)
    ready = Barrier(2)

    def remove():
        ready.wait(timeout=10)
        return db.delete_output_artifact(output.id, hard=True)

    with ThreadPoolExecutor(max_workers=2) as workers:
        jobs = [workers.submit(remove) for _ in range(2)]
        assert sorted(job.result(timeout=15) for job in jobs) == [False, True]
    assert db.get_content_item(parent.id).revision == parent.revision + 1
    assert db.backend.execute("SELECT COUNT(*) FROM reading_artifact_paths", ()).scalar == 1


def test_service_bulk_delete_uses_owned_boundary_and_counts_actual_rows(db, tmp_path):
    parent, namespace, output = archive(db, tmp_path)
    assert delete_outputs_by_ids(db, int(db.user_id), [output.id, output.id, output.id + 100]) == 1
    assert db.get_content_item(parent.id).revision == parent.revision + 1
    assert cleanup.drain_reading_artifact_cleanup(db, output_root=tmp_path, storage_namespace_id=namespace) == 1
    assert delete_outputs_by_ids(db, int(db.user_id), [output.id]) == 0


def test_service_bulk_delete_rejects_mismatched_user(db, tmp_path):
    _, _, output = archive(db, tmp_path)
    before = snapshot(db)
    with pytest.raises(ValueError, match="output_user_mismatch"):
        delete_outputs_by_ids(db, 781, [output.id])
    assert snapshot(db) == before


@pytest.mark.parametrize("renew", [False, True])
def test_db_retention_purge_rechecks_eligibility_and_uses_owned_cleanup(db, tmp_path, monkeypatch, renew):
    parent, namespace, output = archive(db, tmp_path)
    db.update_output_artifact(output.id, retention_until="2000-01-01T00:00:00+00:00")
    before = db.get_content_item(parent.id)
    remove = db.delete_output_artifact

    def renew_before_fence(output_id, **kwargs):
        if renew:
            db.update_output_artifact(output_id, retention_until="2999-01-01T00:00:00+00:00")
        return remove(output_id, **kwargs)

    monkeypatch.setattr(db, "delete_output_artifact", renew_before_fence)
    assert db.purge_expired_outputs() == (0 if renew else 1)
    assert db.get_content_item(parent.id).revision == before.revision + 1
    if renew:
        assert db.get_output_artifact(output.id).id == output.id
        assert db.backend.execute("SELECT COUNT(*) FROM reading_artifact_paths", ()).scalar == 0
    else:
        assert cleanup.drain_reading_artifact_cleanup(db, output_root=tmp_path, storage_namespace_id=namespace) == 1


@pytest.mark.parametrize("bulk", [False, True])
@pytest.mark.parametrize("file_measurement", [False, True])
def test_existing_audiobook_quota_semantics_match_per_output_delegation(db, bulk, file_measurement):
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

    active = db.create_output_artifact(
        type_="audiobook_mp3",
        title="Active book",
        format_="mp3",
        storage_path="active-book.mp3",
        metadata_json=None if file_measurement else '{"byte_size": 23}',
    )
    old = db.create_output_artifact(
        type_="audiobook_mp3",
        title="Old book",
        format_="mp3",
        storage_path="old-book.mp3",
        metadata_json='{"byte_size": 11}',
    )
    if file_measurement:
        (DatabasePaths.get_user_outputs_dir(int(db.user_id)) / active.storage_path).write_bytes(b"x" * 23)
    db.set_audiobook_output_usage(34)
    assert db.delete_output_artifact(old.id)
    assert db.get_audiobook_output_usage() == 23
    if bulk:
        delete_outputs_by_ids(db, int(db.user_id), [active.id, old.id])
    else:
        assert db.delete_output_artifact(active.id, hard=True)
        assert db.delete_output_artifact(old.id, hard=True)
    assert db.get_audiobook_output_usage() == 0
    assert not db.delete_output_artifact(active.id, hard=True)
    assert db.get_audiobook_output_usage() == 0


@pytest.mark.parametrize("aged", [False, True])
def test_retention_purge_respects_soft_delete_grace(db, tmp_path, aged):
    parent, namespace, output = archive(db, tmp_path)
    db.delete_output_artifact(output.id)
    before = db.get_content_item(parent.id)
    if aged:
        db.backend.execute("UPDATE outputs SET deleted_at = ? WHERE id = ?", ("2000-01-01T00:00:00+00:00", output.id))
    assert db.purge_expired_outputs() == int(aged)
    assert db.get_content_item(parent.id).revision == before.revision + int(aged)
    assert cleanup.drain_reading_artifact_cleanup(db, output_root=tmp_path, storage_namespace_id=namespace) == int(aged)
