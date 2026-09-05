"""Activated generic download lookup/open stays on one verified volume."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import replace
from threading import Event

import pytest

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.services.reading_artifact_cleanup_service import reading_storage_lock
from tldw_Server_API.tests.Collections.test_output_file_operations_storage import db as db
from tldw_Server_API.tests.Collections.test_output_file_operations_storage import prepare, run, service
from tldw_Server_API.tests.Collections.test_output_file_operations_storage import storage as storage
from tldw_Server_API.tests.Collections.test_output_file_recovery import publish_before_interruption
from tldw_Server_API.tests.Collections.test_reading_output_deletion import archive
from tldw_Server_API.tests.Collections.test_reading_output_disposal_routes import client as client

pytestmark = [pytest.mark.unit, pytest.mark.skipif(os.name != "posix", reason="POSIX file protocol")]
pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_full_fixtures"]


@pytest.fixture
def activated(client, storage, monkeypatch):
    root, namespace, row = storage
    monkeypatch.setattr(DatabasePaths, "get_user_outputs_dir", lambda *_: root)
    monkeypatch.setattr(DatabasePaths, "resolve_user_base_directory", lambda *_: root.parent)
    return client, root, namespace, row


def request(client, row, route):
    if route == "title":
        return client.get("/outputs/download/by-name", params={"title": row.title})
    return client.request("HEAD" if route == "head" else "GET", f"/outputs/{row.id}/download")


@pytest.mark.parametrize("route", ["id", "title", "head"])
@pytest.mark.parametrize("problem", ["marker", "missing_root", "version", "busy"])
def test_activated_download_never_falls_back_to_unverified_storage(db, activated, route, problem):
    client, root, namespace, row = activated
    if problem == "marker":
        (root / ".reading-storage-namespace").write_text("0" * 32 + "\n")
    elif problem == "missing_root":
        root.rename(root.with_name("detached"))
    elif problem == "version":
        db.backend.execute("UPDATE output_storage_bindings SET protocol_version = 99 WHERE user_id = ?", (db.user_id,))
    if problem == "busy":
        with reading_storage_lock(root, storage_namespace_id=namespace):
            response = request(client, row, route)
    else:
        response = request(client, row, route)
    assert response.status_code == (409 if problem == "busy" else 503), response.text
    if route != "head":
        assert response.json() == {"detail": "output_file_busy" if problem == "busy" else "output_storage_unavailable"}
    if problem == "missing_root":
        assert not root.exists(), "read recreated a missing mounted output directory"


@pytest.mark.parametrize("route", ["id", "title"])
def test_activated_response_retains_opened_inode_when_path_reused(activated, monkeypatch, route):
    from starlette.responses import FileResponse

    from tldw_Server_API.app.services.output_file_response import OpenedOutputResponse

    client, root, _, row = activated
    touched = []

    def wrap(call):
        async def wrapped(self, scope, receive, send):
            async def intercepted(message):
                if message["type"] == "http.response.start":
                    (root / row.storage_path).unlink()
                    (root / row.storage_path).write_bytes(b"intruder")
                    touched.append(True)
                await send(message)

            return await call(self, scope, receive, intercepted)

        return wrapped

    monkeypatch.setattr(FileResponse, "__call__", wrap(FileResponse.__call__))
    monkeypatch.setattr(OpenedOutputResponse, "__call__", wrap(OpenedOutputResponse.__call__))
    response = request(client, row, route)
    assert touched and response.status_code == 200
    assert response.content == b"original"


@pytest.mark.parametrize("problem", ["wrong_volume", "missing_owner", "inactive"])
@pytest.mark.parametrize("route", ["id", "title", "head"])
def test_owned_archive_requires_unambiguous_matching_namespace(db, activated, problem, route):
    client, root, namespace, _ = activated
    _, _, row = archive(db, root)
    if problem == "wrong_volume":
        db.backend.execute(
            "UPDATE reading_output_ownership SET storage_namespace_id = ? WHERE output_id = ?",
            ("another-volume", row.id),
        )
    elif problem == "missing_owner":
        db.backend.execute("DELETE FROM reading_output_ownership WHERE output_id = ?", (row.id,))
    else:
        db.backend.execute("DELETE FROM output_storage_bindings WHERE user_id = ?", (db.user_id,))
    response = request(client, row, route)
    assert response.status_code == 503
    if route != "head":
        assert response.json() == {"detail": "output_storage_unavailable"}
    assert (root / row.storage_path).exists()


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "directory", "fifo", "absolute", "stage"])
def test_unproved_files_are_not_opened_as_downloads(db, activated, kind):
    client, root, _, row = activated
    path = root / row.storage_path
    if kind in {"symlink", "directory", "fifo"}:
        path.unlink()
        if kind == "symlink":
            (root / "other.md").write_bytes(b"foreign")
            path.symlink_to(root / "other.md")
        elif kind == "directory":
            path.mkdir()
        else:
            os.mkfifo(path)
    elif kind == "hardlink":
        os.link(path, root / "other.md")
    else:
        db.backend.execute(
            "UPDATE outputs SET storage_path = ? WHERE id = ?",
            (str(path) if kind == "absolute" else ".output-stage-" + "1" * 32, row.id),
        )
    response = request(client, row, "id")
    assert response.status_code == 503 and response.json() == {"detail": "output_storage_unavailable"}


@pytest.mark.parametrize(
    "state",
    ["witness", "cleaned_witness", "foreign_witness", "extra_link", "uncommitted", "changed_bytes", "invalid_identity"],
)
def test_reader_accepts_only_recorded_committed_publication(db, activated, storage, state):
    client, root, namespace, original = activated
    writer = service(db, storage)
    operation = prepare(writer, original)
    run(writer.write_chunk, operation["token"], b"replacement", expected_offset=0)
    row = publish_before_interruption(writer, operation["token"])
    stage = root / operation["stage_path"]
    if state in {"cleaned_witness", "foreign_witness"}:
        stage.unlink()
        if state == "foreign_witness":
            stage.write_bytes(b"replacement")
    elif state == "extra_link":
        os.link(stage, root / "extra.md")
    elif state == "uncommitted":
        db.backend.execute(
            "UPDATE output_file_operations SET phase = 'prepared' WHERE token = ?", (operation["token"],)
        )
    elif state == "changed_bytes":
        (root / row.storage_path).write_bytes(b"unacknowledged bytes")
    elif state == "invalid_identity":
        stage.unlink()
        proof = json.loads(db.get_output_file_operation(operation["token"], namespace)["publication_identity_json"])
        proof["nlink"] = 19
        db.backend.execute(
            "UPDATE output_file_operations SET publication_identity_json = ? WHERE token = ?",
            (json.dumps(proof), operation["token"]),
        )
    response = request(client, row, "id")
    if state in {"witness", "cleaned_witness"}:
        assert response.status_code == 200 and response.content == b"replacement"
    else:
        assert response.status_code == 503 and response.json() == {"detail": "output_storage_unavailable"}
    assert db.get_output_file_operation(operation["token"], namespace)["reserved_bytes"]


@pytest.mark.parametrize("change", ["delete", "retarget"])
@pytest.mark.parametrize("route", ["id", "title", "head"])
def test_lookup_is_refreshed_after_prelock_state_change(db, activated, monkeypatch, change, route):
    client, root, _, row = activated
    real_namespace = db.get_output_read_namespace
    changed = []

    def namespace_then_mutate():
        value = real_namespace()
        if change == "delete":
            db.delete_output_artifact(row.id, hard=True)
        else:
            (root / "current.md").write_bytes(b"current bytes")
            db.update_output_artifact(row.id, storage_path="current.md")
        (root / row.storage_path).unlink()
        (root / row.storage_path).write_bytes(b"intruder")
        changed.append(True)
        return value

    monkeypatch.setattr(db, "get_output_read_namespace", namespace_then_mutate)
    response = request(client, row, route)
    assert changed
    assert response.status_code == (404 if change == "delete" else 200)
    if change == "retarget":
        assert response.headers["content-length"] == str(len(b"current bytes"))
        if route != "head":
            assert response.content == b"current bytes"


def test_protected_lookup_and_open_hold_same_directory_after_root_replacement(db, activated, monkeypatch):
    client, root, namespace, row = activated
    real_lookup = db.get_output_file_read_state
    held = []

    def lookup_then_replace(*args, **kwargs):
        with pytest.raises(Exception) as busy:
            with reading_storage_lock(root, storage_namespace_id=namespace):
                pytest.fail("lookup ran outside verified exclusion")
        assert str(busy.value) == "reading_storage_busy"
        result = real_lookup(*args, **kwargs)
        root.rename(root.with_name("original-volume"))
        root.mkdir()
        (root / row.storage_path).write_bytes(b"intruder")
        held.append(True)
        return result

    monkeypatch.setattr(db, "get_output_file_read_state", lookup_then_replace)
    response = request(client, row, "id")
    assert held and response.status_code == 200 and response.content == b"original"
    assert (root / row.storage_path).read_bytes() == b"intruder"


def test_head_closes_opened_descriptor_before_return(activated, monkeypatch):
    client, _, _, row = activated
    real_open, descriptors = os.open, []

    def opened(name, *args, **kwargs):
        fd = real_open(name, *args, **kwargs)
        if name == row.storage_path:
            descriptors.append(fd)
        return fd

    monkeypatch.setattr(os, "open", opened)
    response = request(client, row, "head")
    assert response.status_code == 200 and response.content == b"" and descriptors
    assert "etag" not in response.headers and "content-disposition" not in response.headers
    for fd in descriptors:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_cancelled_protected_open_closes_unreturned_response(db, activated, monkeypatch):
    from tldw_Server_API.app.services.output_file_response import protected_output_response

    _, root, namespace, row = activated
    real_open, descriptors, entered, release = os.open, [], Event(), Event()

    def opened(name, *args, **kwargs):
        fd = real_open(name, *args, **kwargs)
        if name == row.storage_path:
            descriptors.append(fd)
            entered.set()
            assert release.wait(10)
        return fd

    monkeypatch.setattr(os, "open", opened)

    async def exercise():
        task = asyncio.create_task(protected_output_response(db, output_id=row.id))
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done() and os.fstat(descriptors[0]).st_size == 8
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()

    run(exercise)
    with pytest.raises(OSError):
        os.fstat(descriptors[0])
    with reading_storage_lock(root, storage_namespace_id=namespace):
        assert (root / row.storage_path).read_bytes() == b"original"


def test_response_metadata_failure_does_not_leak_file_descriptor(db, activated, monkeypatch):
    client, _, _, row = activated
    real_lookup, real_open = db.get_output_file_read_state, os.open
    descriptors, failures = [], []

    class FailedFormat(str):
        def lower(self):
            failures.append(True)
            raise OSError("private formatter details")

    def lookup(*args, **kwargs):
        current, proof = real_lookup(*args, **kwargs)
        return replace(current, format=FailedFormat(current.format)), proof

    def opened(name, *args, **kwargs):
        fd = real_open(name, *args, **kwargs)
        if name == row.storage_path:
            descriptors.append(fd)
        return fd

    monkeypatch.setattr(db, "get_output_file_read_state", lookup)
    monkeypatch.setattr(os, "open", opened)
    response = request(client, row, "id")
    assert failures and response.status_code == 503
    try:
        for fd in descriptors:
            with pytest.raises(OSError):
                os.fstat(fd)
    finally:
        for fd in descriptors:
            try:
                os.close(fd)
            except OSError:
                pass
