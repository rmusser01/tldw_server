"""Watchlist output consumers use current, volume-verified artifact bytes."""

import asyncio
import json
import os
from threading import Event
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.endpoints import watchlists
from tldw_Server_API.app.services.reading_artifact_cleanup_service import reading_storage_lock
from tldw_Server_API.tests.Collections.test_output_file_operations_storage import run
from tldw_Server_API.tests.Collections.test_output_protected_downloads import activated as activated
from tldw_Server_API.tests.Collections.test_output_protected_downloads import client as client
from tldw_Server_API.tests.Collections.test_output_protected_downloads import db as db
from tldw_Server_API.tests.Collections.test_output_protected_downloads import storage as storage

pytestmark = [pytest.mark.unit, pytest.mark.skipif(os.name != "posix", reason="POSIX file protocol")]
pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_full_fixtures"]


@pytest.fixture
def reader(db, activated):
    client, root, namespace, row = activated
    client.app.include_router(watchlists.router)
    row = db.update_output_artifact_metadata(row.id, metadata_json=json.dumps({"origin": "watchlists"}))
    return client, root, namespace, row


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("fmt,media", [("md", "text/markdown"), ("html", "text/html"), ("mp3", "audio/mpeg")])
def test_watchlist_download_preserves_format_headers_and_range_policy(db, reader, active, fmt, media):
    client, root, _, row = reader
    if not active:
        db.backend.execute("DELETE FROM output_storage_bindings WHERE user_id = ?", (db.user_id,))
    db.update_output_artifact(row.id, format_=fmt, title="My/report")
    (root / row.storage_path).write_bytes(b"0123456789")
    result = client.get(f"/watchlists/outputs/{row.id}/download", headers={"Range": "bytes=2-5"})
    assert result.status_code == (206 if fmt == "mp3" else 200), result.text
    assert result.content == (b"2345" if fmt == "mp3" else b"0123456789")
    assert result.headers["content-type"] == media + ("; charset=utf-8" if fmt != "mp3" else "")
    assert result.headers["content-disposition"] == f'attachment; filename="My_report.{fmt}"'
    assert ("etag" in result.headers) == (fmt == "mp3")


@pytest.mark.parametrize("route", ["download", "detail", "list"])
@pytest.mark.parametrize("problem", ["volume", "busy"])
def test_watchlist_readers_never_fall_back_on_unsafe_storage(reader, route, problem):
    client, root, namespace, row = reader
    url = "/watchlists/outputs" if route == "list" else f"/watchlists/outputs/{row.id}"
    if route == "download":
        url += "/download"
    if problem == "volume":
        (root / ".reading-storage-namespace").write_text("0" * 32 + "\n")
        result = client.get(url)
    else:
        with reading_storage_lock(root, storage_namespace_id=namespace):
            result = client.get(url)
    assert result.status_code == (503 if problem == "volume" else 409), result.text
    assert result.json() == {"detail": "output_storage_unavailable" if problem == "volume" else "output_file_busy"}


@pytest.mark.parametrize("route", ["download", "detail", "list"])
@pytest.mark.parametrize("change", ["delete", "retarget", "origin", "expired"])
def test_watchlist_readers_recheck_current_row_before_open(db, reader, monkeypatch, route, change):
    client, root, _, row = reader
    real_lookup = db.get_output_read_namespace
    changed = []

    def lookup():
        namespace = real_lookup()
        if not changed:
            changed.append(True)
            if change == "delete":
                db.delete_output_artifact(row.id, hard=True)
            elif change == "retarget":
                (root / "current.md").write_bytes(b"current")
                db.update_output_artifact(row.id, title="Current", storage_path="current.md")
            else:
                metadata = {"origin": "watchlists" if change == "expired" else "not-watchlists"}
                if change == "expired":
                    metadata["expires_at"] = "2000-01-01T00:00:00+00:00"
                db.update_output_artifact_metadata(row.id, metadata_json=json.dumps(metadata))
            (root / row.storage_path).write_bytes(b"intruder")
        return namespace

    monkeypatch.setattr(db, "get_output_read_namespace", lookup)
    url = "/watchlists/outputs" if route == "list" else f"/watchlists/outputs/{row.id}"
    if route == "download":
        url += "/download"
    result = client.get(url)
    assert changed, "consumer never entered protected lookup"
    if change != "retarget":
        assert result.status_code == 404, result.text
    elif route == "download":
        assert result.status_code == 200 and result.content == b"current"
        assert result.headers["content-disposition"] == 'attachment; filename="Current.md"'
    else:
        value = result.json()["items"][0] if route == "list" else result.json()
        assert value["title"] == "Current" and value["content"] == "current"


@pytest.mark.parametrize("active", [False, True])
def test_evidence_sidecar_without_registered_provenance_is_only_read_when_inactive(db, reader, active):
    from fastapi import HTTPException

    _, root, _, row = reader
    (root / "evidence.json").write_text('{"schema_version": 1, "readiness": {"level": "ready"}}')
    if not active:
        db.backend.execute("DELETE FROM output_storage_bindings WHERE user_id = ?", (db.user_id,))
    # Exercise the actual evidence loader; HTTP authorization is covered by the API suite.
    args = {"user_id": int(db.user_id), "output_id": row.id, "metadata": {"report_snapshot_path": "evidence.json"}}
    if active:
        with pytest.raises(HTTPException) as error:
            run(watchlists._load_output_report_evidence_payload, **args, collections_db=db)
        assert error.value.status_code == 503
    else:
        payload = run(watchlists._load_output_report_evidence_payload, **args, collections_db=db)
        assert payload["snapshot"]["readiness"] == {"level": "ready"}


@pytest.mark.parametrize("route", ["download", "detail"])
def test_inline_text_limit_rejects_before_reading_and_closes_descriptor(reader, monkeypatch, route):
    client, root, _, row = reader
    with (root / row.storage_path).open("wb") as file:
        file.truncate(8 * 1024 * 1024 + 1)
    opened = []
    real_open = os.open

    def capture(name, *args, **kwargs):
        fd = real_open(name, *args, **kwargs)
        if name == row.storage_path:
            opened.append(fd)
        return fd

    monkeypatch.setattr(os, "open", capture)
    result = client.get(f"/watchlists/outputs/{row.id}" + ("/download" if route == "download" else ""))
    assert result.status_code == 413, result.text[:100]
    assert result.json() == {"detail": "output_content_too_large"}
    assert opened
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)


@pytest.mark.parametrize("active", [False, True])
def test_watchlist_missing_download_preserves_error_category(db, reader, active):
    client, root, _, row = reader
    if not active:
        db.backend.execute("DELETE FROM output_storage_bindings WHERE user_id = ?", (db.user_id,))
    (root / row.storage_path).unlink()
    result = client.get(f"/watchlists/outputs/{row.id}/download")
    assert result.status_code == 404
    assert result.json() == {"detail": "output_file_missing"}


@pytest.mark.parametrize("route,fmt", [("download", "md"), ("download", "html"), ("download", "mp3"), ("detail", "md")])
def test_watchlist_reads_opened_inode_even_when_name_is_recycled(db, reader, monkeypatch, route, fmt):
    client, root, _, row = reader
    db.update_output_artifact(row.id, format_=fmt)
    pread = os.pread
    changed = []

    def replace_before_read(fd, size, offset):
        if not changed:
            changed.append(True)
            (root / row.storage_path).unlink()
            (root / row.storage_path).write_bytes(b"intruder")
        return pread(fd, size, offset)

    monkeypatch.setattr(os, "pread", replace_before_read)
    result = client.get(f"/watchlists/outputs/{row.id}" + ("/download" if route == "download" else ""))
    assert changed and result.status_code == 200
    assert (result.content if route == "download" else result.json()["content"]) == (
        b"original" if route == "download" else "original"
    )


@pytest.mark.parametrize("active", [False, True])
def test_text_download_preserves_utf8_and_universal_newlines(db, reader, active):
    client, root, _, row = reader
    if not active:
        db.backend.execute("DELETE FROM output_storage_bindings WHERE user_id = ?", (db.user_id,))
    (root / row.storage_path).write_bytes("café\r\nline\rend\n".encode())
    result = client.get(f"/watchlists/outputs/{row.id}/download")
    assert result.status_code == 200 and result.content == "café\nline\nend\n".encode()
    assert int(result.headers["content-length"]) == len(result.content)


def test_cancelled_watchlist_inline_read_drains_worker_then_closes(db, reader, monkeypatch):
    _, _, _, row = reader
    entered, release = Event(), Event()
    pread = os.pread
    descriptors = []

    def blocked(fd, size, offset):
        descriptors.append(fd)
        entered.set()
        assert release.wait(10)
        return pread(fd, size, offset)

    monkeypatch.setattr(os, "pread", blocked)

    async def exercise():
        task = asyncio.create_task(watchlists._row_to_output(row, user_id=int(db.user_id), collections_db=db))
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


def test_inactive_evidence_invalid_path_keeps_legacy_missing_error(db, reader):
    from fastapi import HTTPException

    _, _, _, row = reader
    db.backend.execute("DELETE FROM output_storage_bindings WHERE user_id = ?", (db.user_id,))
    with pytest.raises(HTTPException) as error:
        run(
            watchlists._load_output_report_evidence_payload,
            user_id=int(db.user_id),
            output_id=row.id,
            metadata={"report_snapshot_path": "../evidence.json"},
            collections_db=db,
        )
    assert error.value.status_code == 404 and error.value.detail == "report_snapshot_missing"


@pytest.mark.parametrize("removed", [False, True])
def test_retry_delivery_uses_plan_from_same_protected_snapshot_as_content(db, reader, monkeypatch, removed):
    from fastapi import HTTPException

    _, root, _, row = reader
    metadata = {
        "origin": "watchlists",
        "delivery_plan": {"email": {"enabled": True, "recipients": ["old@example.com"]}},
    }
    db.backend.execute("UPDATE outputs SET run_id = ?, job_id = ? WHERE id = ?", (10, 7, row.id))
    db.update_output_artifact_metadata(row.id, metadata_json=json.dumps(metadata))
    lookup = db.get_output_read_namespace
    changed, deliveries = [], []

    def update_before_lock():
        namespace = lookup()
        if not changed:
            changed.append(True)
            current = {"origin": "watchlists", "current_marker": "keep"}
            if not removed:
                current["delivery_plan"] = {"email": {"enabled": True, "recipients": ["current@example.com"]}}
            db.update_output_artifact_metadata(row.id, metadata_json=json.dumps(current))
            (root / row.storage_path).write_bytes(b"current content")
        return namespace

    class Notifications:
        def __init__(self, **kwargs):
            pass

        async def deliver_email(self, **kwargs):
            deliveries.append(kwargs)
            return SimpleNamespace(channel="email", status="sent", details={})

    monkeypatch.setattr(db, "get_output_read_namespace", update_before_lock)
    monkeypatch.setattr(watchlists, "NotificationsService", Notifications)
    args = {
        "run_id": 10,
        "target_user_id": None,
        "current_user": SimpleNamespace(id=int(db.user_id), email="actor@example.com", role="admin"),
        "db": SimpleNamespace(get_run=lambda _: SimpleNamespace(id=10, job_id=7)),
        "collections_db": db,
    }
    if removed:
        with pytest.raises(HTTPException) as error:
            run(watchlists.retry_run_delivery, **args)
        assert error.value.status_code == 400 and not deliveries
    else:
        result = run(watchlists.retry_run_delivery, **args)
        assert result.retried and changed
        assert deliveries[0]["recipients"] == ["current@example.com"]
        assert deliveries[0]["text_body"] == "current content"
        current = json.loads(db.get_output_artifact(row.id).metadata_json)
        assert current["current_marker"] == "keep"
        assert current["delivery_plan"]["email"]["recipients"] == ["current@example.com"]
