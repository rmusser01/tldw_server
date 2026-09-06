"""Real output deletion routes preserve explicit managed-file permission."""

from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from threading import Event
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import outputs
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.services import outputs_purge_scheduler as scheduler
from tldw_Server_API.app.services import outputs_service
from tldw_Server_API.app.services import reading_artifact_cleanup_service as cleanup
from tldw_Server_API.app.services.output_file_operations import OutputFileOperations
from tldw_Server_API.tests.Collections.test_output_file_history_delivery import get_history, history
from tldw_Server_API.tests.Collections.test_output_file_history_delivery import media as media
from tldw_Server_API.tests.Collections.test_output_file_operations_db import insert_binding
from tldw_Server_API.tests.Collections.test_reading_atomic_delete import snapshot
from tldw_Server_API.tests.Collections.test_reading_output_deletion import archive
from tldw_Server_API.tests.Collections.test_reading_revision_mutations import db as db
from tldw_Server_API.tests.Collections.test_reading_revision_mutations import make_reading

pytestmark = pytest.mark.unit
pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_full_fixtures"]


@pytest.fixture
def client(db, tmp_path, monkeypatch):
    monkeypatch.setattr(DatabasePaths, "get_user_outputs_dir", lambda *_args: tmp_path)
    app = FastAPI()
    app.include_router(outputs.router)
    app.dependency_overrides[outputs.get_collections_db_for_user] = lambda: db
    app.dependency_overrides[outputs.get_request_user] = lambda: User(
        id=int(db.user_id), username="reader", email=None, is_active=True
    )
    app.dependency_overrides[outputs.get_media_db_for_user] = lambda: SimpleNamespace(
        mark_tts_history_artifacts_deleted_for_output=lambda **_kwargs: None
    )
    with TestClient(app) as client:
        yield client


def activate(db, root, monkeypatch):
    namespace = cleanup.provision_reading_storage_namespace(root)
    insert_binding(
        db, storage_namespace_id=namespace, operation_bytes=4096, user_pending_bytes=16384, free_space_margin_bytes=1
    )
    monkeypatch.setattr(outputs_service, "_existing_outputs_dir_for_user", lambda user: root)
    return namespace


@pytest.mark.parametrize("shared", [False, True])
def test_activated_delete_commits_quota_and_original_history_without_legacy_effect(
    db, tmp_path, client, monkeypatch, shared, media
):
    namespace = activate(db, tmp_path, monkeypatch)
    output = db.create_output_artifact(
        type_="audiobook_mp3", title="Audio", format_="mp3", storage_path="audio.mp3", metadata_json='{"byte_size": 8}'
    )
    identity = db.backend.execute("SELECT file_incarnation FROM outputs WHERE id = ?", (output.id,)).scalar
    (tmp_path / "audio.mp3").write_bytes(b"original")
    if shared:
        other = db.create_output_artifact(type_="report", title="Other", format_="mp3", storage_path="audio.mp3")
    db.set_audiobook_output_usage(8)
    original_history = history(media, identity, output.id)
    newer_history = history(media, "e" * 32, output.id)
    client.app.dependency_overrides[outputs.get_media_db_for_user] = lambda: media
    response = client.delete(f"/outputs/{output.id}", params={"hard": True, "delete_file": True})
    assert response.status_code == 200, response.text
    assert response.json() == {"success": True, "file_deleted": not shared}
    assert db.get_audiobook_output_usage() == 0
    assert (tmp_path / "audio.mp3").exists() is shared
    if shared:
        assert db.get_output_artifact(other.id) == other
    rows = db.backend.execute("SELECT * FROM output_file_operations").rows
    assert len(rows) == 1 and rows[0]["fs_done"] == 1 and rows[0]["effects_pending"] == 1
    assert json.loads(rows[0]["effects_json"])[0]["incarnation"] == identity
    assert get_history(media, original_history)["output_id"] == output.id
    assert get_history(media, newer_history)["output_id"] == output.id
    assert client.delete(f"/outputs/{output.id}", params={"hard": True, "delete_file": True}).status_code == 404
    assert db.get_audiobook_output_usage() == 0
    writer = OutputFileOperations(db, output_root=tmp_path, storage_namespace_id=namespace)
    assert asyncio.run(writer.deliver_history_due(media))["delivered"] == 1
    assert get_history(media, original_history)["output_id"] is None
    assert get_history(media, newer_history)["output_id"] == output.id
    late = history(media, identity, output.id)
    assert get_history(media, late)["output_id"] is None


def test_activated_delete_defers_failed_unlink_without_losing_claim(db, tmp_path, client, monkeypatch):
    activate(db, tmp_path, monkeypatch)
    output = db.create_output_artifact(type_="report", title="Old", format_="md", storage_path="old.md")
    (tmp_path / "old.md").write_bytes(b"original")
    unlink = os.unlink

    def fail_original(path, *args, **kwargs):
        if path == "old.md":
            raise OSError("private /secret/mount")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", fail_original)
    response = client.delete(f"/outputs/{output.id}", params={"hard": True, "delete_file": True})
    assert response.json() == {"success": True, "file_deleted": False}
    assert (tmp_path / "old.md").read_bytes() == b"original"
    row = db.backend.execute("SELECT * FROM output_file_operations").first
    assert row["phase"] == "committed" and row["fs_done"] == 0 and row["reserved_bytes"] == 8
    assert row["last_error"] == "output_storage_unavailable"


def test_activated_delete_offline_volume_preserves_metadata_and_bytes(db, tmp_path, client, monkeypatch):
    activate(db, tmp_path, monkeypatch)
    output = db.create_output_artifact(type_="report", title="Old", format_="md", storage_path="old.md")
    (tmp_path / "old.md").write_bytes(b"original")
    missing = tmp_path / "offline"
    monkeypatch.setattr(outputs_service, "_existing_outputs_dir_for_user", lambda user: missing)
    absent = client.delete("/outputs/999999", params={"hard": True, "delete_file": True})
    assert absent.status_code == 404
    response = client.delete(f"/outputs/{output.id}", params={"hard": True, "delete_file": True})
    assert response.status_code == 503, response.text
    assert response.json() == {"detail": "output_storage_unavailable"}
    assert db.get_output_artifact(output.id) == output
    assert (tmp_path / "old.md").read_bytes() == b"original" and not missing.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("scheduled", [False, True])
async def test_activated_purge_keeps_unlink_receipt_when_cleanup_ack_fails(
    db, tmp_path, client, monkeypatch, scheduled
):
    activate(db, tmp_path, monkeypatch)
    db.create_output_artifact(
        type_="report", title="Old", format_="md", storage_path="old.md", retention_until="2000-01-01T00:00:00+00:00"
    )
    (tmp_path / "old.md").write_bytes(b"original")

    def fail_ack(*args, **kwargs):
        raise RuntimeError("private /secret/database")

    monkeypatch.setattr(db, "finish_output_file_operation", fail_ack)
    if scheduled:
        monkeypatch.setattr(CollectionsDatabase, "for_user", lambda *_: db)
        assert await scheduler._purge_for_user(int(db.user_id), True, 30) == (1, 1)
    else:
        response = client.post("/outputs/purge", json={"delete_files": True})
        assert response.json() == {"removed": 1, "files_deleted": 1}
    assert not (tmp_path / "old.md").exists()
    row = db.backend.execute("SELECT * FROM output_file_operations").first
    assert row["phase"] == "committed" and row["fs_done"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("committed", [False, True])
async def test_activated_delete_cancellation_drains_commit_and_preserves_outcome(db, tmp_path, monkeypatch, committed):
    activate(db, tmp_path, monkeypatch)
    output = db.create_output_artifact(type_="report", title="Old", format_="md", storage_path="old.md")
    (tmp_path / "old.md").write_bytes(b"original")
    entered, release = Event(), Event()
    apply = db.apply_output_file_operation

    def delayed_commit(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        if committed:
            return apply(*args, **kwargs)
        raise OSError("private /secret/database")

    monkeypatch.setattr(db, "apply_output_file_operation", delayed_commit)
    task = asyncio.create_task(
        outputs_service.delete_output_with_file(
            db,
            int(db.user_id),
            output.id,
            hard=True,
            delete_file=True,
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        if not task.done():
            await asyncio.gather(task, return_exceptions=True)
    row = db.backend.execute("SELECT * FROM output_file_operations").first
    assert row["phase"] == ("committed" if committed else "aborting")
    assert row["effects_pending"] == int(committed)
    assert (tmp_path / "old.md").exists() is not committed
    assert db.backend.execute("SELECT COUNT(*) FROM outputs WHERE id = ?", (output.id,)).scalar == int(not committed)


@pytest.mark.asyncio
@pytest.mark.parametrize("scheduled", [False, True])
@pytest.mark.parametrize("disposal", ["unlink", "shared", "deferred"])
async def test_activated_purges_count_only_completed_unlinks(db, tmp_path, client, monkeypatch, scheduled, disposal):
    activate(db, tmp_path, monkeypatch)
    output = db.create_output_artifact(
        type_="report", title="Old", format_="md", storage_path="old.md", retention_until="2000-01-01T00:00:00+00:00"
    )
    (tmp_path / "old.md").write_bytes(b"original")
    if disposal == "shared":
        db.create_output_artifact(type_="report", title="Keep", format_="md", storage_path="old.md")
    if disposal == "deferred":
        unlink = os.unlink

        def fail_original(path, *args, **kwargs):
            if path == "old.md":
                raise OSError("private /secret/mount")
            return unlink(path, *args, **kwargs)

        monkeypatch.setattr(os, "unlink", fail_original)
    if scheduled:
        monkeypatch.setattr(CollectionsDatabase, "for_user", lambda *_: db)
        removed, files = await scheduler._purge_for_user(int(db.user_id), True, 30)
    else:
        response = client.post("/outputs/purge", json={"delete_files": True})
        assert response.status_code == 200
        removed, files = response.json()["removed"], response.json()["files_deleted"]
    assert (removed, files) == (1, int(disposal == "unlink"))
    assert (tmp_path / "old.md").exists() is (disposal != "unlink")
    with pytest.raises(KeyError):
        db.get_output_artifact(output.id)
    row = db.backend.execute("SELECT * FROM output_file_operations").first
    assert row["effects_pending"] == 1
    assert row["fs_done"] == int(disposal != "deferred")


@pytest.mark.parametrize("hard,delete_file", [(False, False), (False, True), (True, False)])
def test_deletion_of_claimed_output_returns_busy_without_mutation(db, tmp_path, client, monkeypatch, hard, delete_file):
    namespace = activate(db, tmp_path, monkeypatch)
    output = db.create_output_artifact(type_="report", title="Claimed", format_="md", storage_path="old.md")
    (tmp_path / "old.md").write_bytes(b"original")
    writer = OutputFileOperations(db, output_root=tmp_path, storage_namespace_id=namespace)
    operation = asyncio.run(writer.prepare(kind="remove", output_id=output.id, max_output_bytes=0))
    response = client.delete(f"/outputs/{output.id}", params={"hard": hard, "delete_file": delete_file})
    assert response.status_code == 409
    assert response.json() == {"detail": "output_file_busy"}
    assert db.get_output_artifact(output.id) == output
    assert (tmp_path / "old.md").read_bytes() == b"original"
    assert db.get_output_file_operation(operation["token"], namespace)["phase"] == "prepared"


@pytest.mark.parametrize("hard,delete_file", [(False, False), (False, True), (True, False)])
def test_activated_metadata_deletion_needs_no_volume_or_unlink_permission(
    db, tmp_path, client, monkeypatch, hard, delete_file
):
    activate(db, tmp_path, monkeypatch)
    output = db.create_output_artifact(type_="report", title="Keep bytes", format_="md", storage_path="old.md")
    (tmp_path / "old.md").write_bytes(b"original")
    monkeypatch.setattr(outputs_service, "_existing_outputs_dir_for_user", lambda user: tmp_path / "offline")
    response = client.delete(f"/outputs/{output.id}", params={"hard": hard, "delete_file": delete_file})
    assert response.json() == {"success": True, "file_deleted": False}
    assert (tmp_path / "old.md").read_bytes() == b"original"
    assert db.backend.execute("SELECT COUNT(*) FROM output_file_operations").scalar == 0


def test_unbound_owned_store_cannot_use_legacy_unowned_physical_deletion(db, tmp_path, client):
    archive(db, tmp_path)
    output = db.create_output_artifact(type_="report", title="Keep", format_="md", storage_path="old.md")
    (tmp_path / "old.md").write_bytes(b"original")
    response = client.delete(f"/outputs/{output.id}", params={"hard": True, "delete_file": True})
    assert response.status_code == 503
    assert db.get_output_artifact(output.id) == output
    assert (tmp_path / "old.md").read_bytes() == b"original"


@pytest.mark.parametrize("soft_first", [False, True])
def test_activated_audio_deletion_preserves_atomic_quota_on_commit_failure(
    db, tmp_path, client, monkeypatch, soft_first
):
    activate(db, tmp_path, monkeypatch)
    output = db.create_output_artifact(
        type_="audiobook_mp3", title="Audio", format_="mp3", storage_path="audio.mp3", metadata_json='{"byte_size": 8}'
    )
    (tmp_path / "audio.mp3").write_bytes(b"original")
    db.set_audiobook_output_usage(8)
    if soft_first:
        assert client.delete(f"/outputs/{output.id}").status_code == 200
        assert db.get_audiobook_output_usage() == 0
    else:
        update_usage = db.update_audiobook_output_usage

        def lose_accounting(*args, **kwargs):
            update_usage(*args, **kwargs)
            raise RuntimeError("private /secret/accounting")

        monkeypatch.setattr(db, "update_audiobook_output_usage", lose_accounting)
    response = client.delete(f"/outputs/{output.id}", params={"hard": True, "delete_file": True})
    if soft_first:
        assert response.json() == {"success": True, "file_deleted": True}
        assert db.get_audiobook_output_usage() == 0
    else:
        assert response.status_code == 409
        assert db.get_output_artifact(output.id) == output
        assert db.get_audiobook_output_usage() == 8
        assert (tmp_path / "audio.mp3").read_bytes() == b"original"
        row = db.backend.execute("SELECT * FROM output_file_operations").first
        assert row["phase"] == "aborting" and row["effects_pending"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("scheduled", [False, True])
async def test_activated_purge_rechecks_renewal_at_preparation_fence(db, tmp_path, client, monkeypatch, scheduled):
    activate(db, tmp_path, monkeypatch)
    output = db.create_output_artifact(
        type_="report", title="Old", format_="md", storage_path="old.md", retention_until="2000-01-01T00:00:00+00:00"
    )
    (tmp_path / "old.md").write_bytes(b"original")
    prepare = db.prepare_output_file_operation
    renewed = False

    def renew_before_prepare(*args, **kwargs):
        nonlocal renewed
        db.update_output_artifact(output.id, retention_until="2999-01-01T00:00:00+00:00")
        renewed = True
        return prepare(*args, **kwargs)

    monkeypatch.setattr(db, "prepare_output_file_operation", renew_before_prepare)
    if scheduled:
        monkeypatch.setattr(CollectionsDatabase, "for_user", lambda *_: db)
        assert await scheduler._purge_for_user(int(db.user_id), True, 30) == (0, 0)
    else:
        response = client.post("/outputs/purge", json={"delete_files": True})
        assert response.json() == {"removed": 0, "files_deleted": 0}
    assert renewed
    assert (
        db.backend.execute("SELECT retention_until FROM outputs WHERE id = ?", (output.id,)).scalar
        == "2999-01-01T00:00:00+00:00"
    )
    assert (tmp_path / "old.md").read_bytes() == b"original"
    assert db.backend.execute("SELECT COUNT(*) FROM output_file_operations").scalar == 0


@pytest.mark.parametrize("soft_first", [False, True])
def test_http_metadata_only_managed_hard_delete_rejects_without_mutation(db, tmp_path, client, soft_first):
    _, _, output = archive(db, tmp_path)
    if soft_first:
        assert client.delete(f"/outputs/{output.id}").status_code == 200
    before = snapshot(db)
    response = client.delete(f"/outputs/{output.id}", params={"hard": True})
    assert response.status_code == 409
    assert response.json() == {"detail": "reading_file_deletion_required"}
    assert snapshot(db) == before
    assert (tmp_path / output.storage_path).exists()


@pytest.mark.parametrize("soft_first", [False, True])
def test_http_explicit_managed_delete_defers_unlink_until_cleanup(db, tmp_path, client, soft_first):
    parent, namespace, output = archive(db, tmp_path)
    if soft_first:
        assert client.delete(f"/outputs/{output.id}").status_code == 200
    before = db.get_content_item(parent.id)
    response = client.delete(f"/outputs/{output.id}", params={"hard": True, "delete_file": True})
    assert response.status_code == 200
    assert response.json() == {"success": True, "file_deleted": False}
    assert db.get_content_item(parent.id).revision == before.revision + 1
    assert (tmp_path / output.storage_path).exists()
    assert cleanup.drain_reading_artifact_cleanup(db, output_root=tmp_path, storage_namespace_id=namespace) == 1
    assert not (tmp_path / output.storage_path).exists()


@pytest.mark.parametrize("managed", [False, True])
def test_http_delete_rollback_never_unlinks_file(db, tmp_path, client, monkeypatch, managed):
    if managed:
        _, _, output = archive(db, tmp_path)
    else:
        output = db.create_output_artifact(
            type_="newsletter_markdown", title="Generic", format_="md", storage_path="generic.md"
        )
        (tmp_path / output.storage_path).write_text("keep", encoding="utf-8")
    before = snapshot(db)
    execute = db.backend.execute

    def fail_delete(query, *args, **kwargs):
        result = execute(query, *args, **kwargs)
        if query.startswith("DELETE FROM outputs"):
            raise RuntimeError("injected deletion rollback")
        return result

    monkeypatch.setattr(db.backend, "execute", fail_delete)
    with pytest.raises(RuntimeError, match="injected deletion rollback"):
        client.delete(f"/outputs/{output.id}", params={"hard": True, "delete_file": True})
    assert snapshot(db) == before
    assert (tmp_path / output.storage_path).exists()


@pytest.mark.parametrize("delete_file", [False, True])
def test_http_unmanaged_file_options_survive(db, tmp_path, client, delete_file):
    output = db.create_output_artifact(
        type_="newsletter_markdown", title="Generic", format_="md", storage_path="generic.md"
    )
    (tmp_path / output.storage_path).write_text("keep", encoding="utf-8")
    response = client.delete(f"/outputs/{output.id}", params={"hard": True, "delete_file": delete_file})
    assert response.json() == {"success": True, "file_deleted": delete_file}
    assert (tmp_path / output.storage_path).exists() is not delete_file


@pytest.mark.asyncio
@pytest.mark.parametrize("scheduled", [False, True])
@pytest.mark.parametrize("delete_files", [False, True])
async def test_purges_obey_managed_file_permission_and_count_only_deletions(
    db, tmp_path, client, monkeypatch, scheduled, delete_files
):
    parent, namespace, output = archive(db, tmp_path)
    assert activate(db, tmp_path, monkeypatch) == namespace
    db.update_output_artifact(output.id, retention_until="2000-01-01T00:00:00+00:00")
    generic = db.create_output_artifact(
        type_="newsletter_markdown", title="Generic", format_="md", storage_path="generic.md"
    )
    (tmp_path / generic.storage_path).write_text("generic", encoding="utf-8")
    db.update_output_artifact(generic.id, retention_until="2000-01-01T00:00:00+00:00")
    before = db.get_content_item(parent.id)
    history = []

    @contextmanager
    def media_context(*_args, **_kwargs):
        yield SimpleNamespace(
            mark_tts_history_artifacts_deleted_for_output=lambda **kw: history.append(kw["output_id"])
        )

    if scheduled:
        monkeypatch.setattr(CollectionsDatabase, "for_user", lambda *_args: db)
        monkeypatch.setattr(scheduler, "managed_media_database", media_context)
        removed, files = await scheduler._purge_for_user(int(db.user_id), delete_files, 30)
        assert set(history) == ({output.id} if delete_files else {generic.id})
    else:
        response = client.post("/outputs/purge", json={"delete_files": delete_files})
        assert response.status_code == 200
        removed, files = response.json()["removed"], response.json()["files_deleted"]
    assert (removed, files) == (2 if delete_files else 1, int(delete_files))
    assert db.get_content_item(parent.id).revision == before.revision + int(delete_files)
    assert (tmp_path / output.storage_path).exists()
    assert cleanup.drain_reading_artifact_cleanup(db, output_root=tmp_path, storage_namespace_id=namespace) == int(
        delete_files
    )


@pytest.mark.parametrize("include_retention", [False, True])
def test_http_purge_custom_grace_and_retention_selection(db, tmp_path, client, include_retention):
    _, _, output = archive(db, tmp_path)
    db.delete_output_artifact(output.id)
    now = datetime.now(timezone.utc)
    db.backend.execute(
        "UPDATE outputs SET deleted_at = ?, retention_until = ? WHERE id = ?",
        ((now - timedelta(days=10)).isoformat(), (now - timedelta(days=1)).isoformat(), output.id),
    )
    response = client.post(
        "/outputs/purge",
        json={"delete_files": True, "include_retention": include_retention, "soft_deleted_grace_days": 20},
    )
    assert response.json() == {"removed": int(include_retention), "files_deleted": 0}
    if not include_retention:
        response = client.post(
            "/outputs/purge", json={"delete_files": True, "include_retention": False, "soft_deleted_grace_days": 5}
        )
        assert response.json() == {"removed": 1, "files_deleted": 0}


def test_http_purge_renewal_after_scan_keeps_file_and_output(db, tmp_path, client, monkeypatch):
    parent, _, output = archive(db, tmp_path)
    db.update_output_artifact(output.id, retention_until="2000-01-01T00:00:00+00:00")
    find = outputs.find_outputs_to_purge

    def renew_after_scan(**kwargs):
        paths = find(**kwargs)
        db.update_output_artifact(output.id, retention_until="2999-01-01T00:00:00+00:00")
        return paths

    monkeypatch.setattr(outputs, "find_outputs_to_purge", renew_after_scan)
    response = client.post("/outputs/purge", json={"delete_files": True})
    assert response.json() == {"removed": 0, "files_deleted": 0}
    assert db.get_output_artifact(output.id).id == output.id
    assert (tmp_path / output.storage_path).exists()
    assert db.get_content_item(parent.id).revision > parent.revision


def test_ownership_registered_after_initial_read_is_checked_under_fence(db, tmp_path, client, monkeypatch):
    parent = make_reading(db)
    namespace = cleanup.provision_reading_storage_namespace(tmp_path)
    output = db.create_output_artifact(
        type_="reading_archive", title="Archive", format_="md", storage_path="archive.md"
    )
    (tmp_path / output.storage_path).write_text("keep", encoding="utf-8")
    execute = db.backend.execute
    registered = False

    def register_after_read(query, *args, **kwargs):
        nonlocal registered
        result = execute(query, *args, **kwargs)
        if not registered and query.startswith("SELECT id, type, metadata_json, storage_path, deleted FROM outputs"):
            registered = True
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(
                    db.register_reading_output_ownership,
                    parent.id,
                    output.id,
                    expected_revision=parent.revision,
                    storage_namespace_id=namespace,
                ).result(timeout=10)
        return result

    monkeypatch.setattr(db.backend, "execute", register_after_read)
    response = client.delete(f"/outputs/{output.id}", params={"hard": True})
    assert response.status_code == 409
    assert registered
    assert db.get_output_artifact(output.id).id == output.id
    assert (tmp_path / output.storage_path).exists()


@pytest.mark.parametrize("legacy_absolute", [False, True])
def test_unowned_alias_cannot_unlink_a_surviving_managed_archive(db, tmp_path, client, monkeypatch, legacy_absolute):
    _, _, owned = archive(db, tmp_path)
    activate(db, tmp_path, monkeypatch)
    shared = db.create_output_artifact(
        type_="newsletter_markdown", title="Shared", format_="md", storage_path=owned.storage_path
    )
    if legacy_absolute:
        db.backend.execute(
            "UPDATE outputs SET storage_path = ? WHERE id = ?", (str(tmp_path / owned.storage_path), shared.id)
        )
    response = client.delete(f"/outputs/{shared.id}", params={"hard": True, "delete_file": True})
    assert response.status_code == (503 if legacy_absolute else 409)
    assert db.get_output_artifact(shared.id).id == shared.id
    assert (tmp_path / owned.storage_path).exists()
    assert db.get_output_artifact(owned.id).id == owned.id


def test_surviving_shared_output_can_become_owned_after_delete_commit(db, tmp_path, client, monkeypatch):
    parent = make_reading(db)
    namespace = cleanup.provision_reading_storage_namespace(tmp_path)
    first = db.create_output_artifact(
        type_="newsletter_markdown", title="First", format_="md", storage_path="shared.md"
    )
    second = db.create_output_artifact(type_="reading_archive", title="Second", format_="md", storage_path="shared.md")
    (tmp_path / "shared.md").write_text("keep", encoding="utf-8")
    delete = db.delete_output_artifact_record

    def register_after_commit(*args, **kwargs):
        result = delete(*args, **kwargs)
        db.register_reading_output_ownership(
            parent.id, second.id, expected_revision=parent.revision, storage_namespace_id=namespace
        )
        return result

    monkeypatch.setattr(db, "delete_output_artifact_record", register_after_commit)
    response = client.delete(f"/outputs/{first.id}", params={"hard": True, "delete_file": True})
    assert response.json() == {"success": True, "file_deleted": False}
    assert (tmp_path / "shared.md").exists()
    assert db.get_content_item(parent.id).revision > parent.revision


def test_rejected_symlink_disposal_does_not_log_filename(db, tmp_path, client, monkeypatch):
    from tldw_Server_API.app.services import outputs_service

    output = db.create_output_artifact(
        type_="newsletter_markdown", title="Generic", format_="md", storage_path="secret-alias.md"
    )
    (tmp_path / "target.md").write_text("keep", encoding="utf-8")
    (tmp_path / output.storage_path).symlink_to(tmp_path / "target.md")
    messages = []
    monkeypatch.setattr(
        outputs_service,
        "logger",
        SimpleNamespace(
            warning=lambda message, *args: messages.append(message.format(*args)),
            error=lambda message, *args: messages.append(message.format(*args)),
        ),
    )
    response = client.delete(f"/outputs/{output.id}", params={"hard": True, "delete_file": True})
    assert response.json() == {"success": True, "file_deleted": False}
    assert "secret-alias" not in "\n".join(messages)
    assert str(tmp_path) not in "\n".join(messages)
    assert (tmp_path / "target.md").read_text() == "keep"
