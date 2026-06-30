from __future__ import annotations

import io
import json
import zipfile
from contextlib import asynccontextmanager

import aiosqlite
import pytest


pytestmark = pytest.mark.unit

_LEAK = "backend exploded /tmp/secret-token"


class _FakeJobManager:
    def __init__(self) -> None:
        self.completed: dict[str, object] | None = None
        self.failed: dict[str, object] | None = None

    def complete_job(
        self,
        jid: int,
        *,
        result: dict[str, object] | None = None,
        worker_id: str | None = None,
        lease_id: str | None = None,
        completion_token: str | None = None,
    ) -> None:
        self.completed = {
            "jid": jid,
            "result": result or {},
            "worker_id": worker_id,
            "lease_id": lease_id,
            "completion_token": completion_token,
        }

    def fail_job(
        self,
        jid: int,
        *,
        error: str,
        retryable: bool,
        worker_id: str | None = None,
        lease_id: str | None = None,
        completion_token: str | None = None,
        backoff_seconds: int | None = None,
    ) -> None:
        self.failed = {
            "jid": jid,
            "error": error,
            "retryable": retryable,
            "worker_id": worker_id,
            "lease_id": lease_id,
            "completion_token": completion_token,
            "backoff_seconds": backoff_seconds,
        }


class _SQLitePool:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    @asynccontextmanager
    async def transaction(self):
        yield self._db


def _assert_safe_payload(value: object) -> None:
    text = json.dumps(value, sort_keys=True)
    assert "backend exploded" not in text
    assert "/tmp/secret-token" not in text


async def _create_source(db: aiosqlite.Connection, *, source_type: str = "local_directory") -> dict:
    from tldw_Server_API.app.core.Ingestion_Sources.service import (
        create_source,
        ensure_ingestion_sources_schema,
    )

    await ensure_ingestion_sources_schema(db)
    return await create_source(
        db,
        user_id=1,
        payload={
            "source_type": source_type,
            "sink_type": "notes",
            "policy": "canonical",
            "config": {},
        },
    )


@pytest.mark.asyncio
async def test_sink_failure_event_payload_uses_safe_error_label(tmp_path, monkeypatch):
    import tldw_Server_API.app.services.ingestion_sources_worker as worker

    meta_db_path = tmp_path / "ingestion_sources.sqlite3"
    async with aiosqlite.connect(str(meta_db_path)) as db:
        db.row_factory = aiosqlite.Row
        source = await _create_source(db)

        def _fail_sink_change(**kwargs):
            raise RuntimeError(_LEAK)

        monkeypatch.setattr(worker, "_apply_change_to_sink", _fail_sink_change)

        result = await worker._apply_snapshot_changes(
            db=db,
            sink_db=object(),
            sink_type="notes",
            policy="canonical",
            source_id=int(source["id"]),
            jid=101,
            current_items={"alpha.md": {"relative_path": "alpha.md", "content_hash": "hash-1"}},
            extraction_failures={},
        )

        assert result["degraded_items"] == 1
        event_cur = await db.execute(
            "SELECT payload_json FROM ingestion_item_events WHERE event_type = ?",
            ("sink_failed",),
        )
        payload = json.loads((await event_cur.fetchone())["payload_json"])
        assert payload["error"] == "ingestion_source_sink_failed"
        assert payload["error_type"] == "RuntimeError"
        _assert_safe_payload(payload)


@pytest.mark.asyncio
async def test_extraction_failure_event_payload_uses_safe_error_label(tmp_path):
    import tldw_Server_API.app.services.ingestion_sources_worker as worker

    meta_db_path = tmp_path / "ingestion_sources.sqlite3"
    async with aiosqlite.connect(str(meta_db_path)) as db:
        db.row_factory = aiosqlite.Row
        source = await _create_source(db)

        result = await worker._apply_snapshot_changes(
            db=db,
            sink_db=object(),
            sink_type="notes",
            policy="canonical",
            source_id=int(source["id"]),
            jid=102,
            current_items={},
            extraction_failures={"alpha.md": {"error": _LEAK}},
        )

        assert result["degraded_items"] == 1
        event_cur = await db.execute(
            "SELECT payload_json FROM ingestion_item_events WHERE event_type = ?",
            ("ingestion_failed",),
        )
        payload = json.loads((await event_cur.fetchone())["payload_json"])
        assert payload["error"] == "ingestion_source_item_failed"
        _assert_safe_payload(payload)


@pytest.mark.asyncio
async def test_sync_job_failure_status_metadata_uses_safe_error_label(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))

    import tldw_Server_API.app.services.ingestion_sources_worker as worker
    from tldw_Server_API.app.core.Ingestion_Sources.archive_snapshot import (
        persist_archive_artifact,
    )
    from tldw_Server_API.app.core.Ingestion_Sources.service import (
        create_source_snapshot,
        update_source_snapshot,
    )

    meta_db_path = tmp_path / "ingestion_sources.sqlite3"
    async with aiosqlite.connect(str(meta_db_path)) as db:
        db.row_factory = aiosqlite.Row
        source = await _create_source(db, source_type="archive_snapshot")

        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, "w") as archive:
            archive.writestr("export/alpha.md", "# Alpha\n")
        staged_snapshot = await create_source_snapshot(
            db,
            source_id=int(source["id"]),
            snapshot_kind="archive_snapshot",
            status="staged",
            summary={"filename": "notes.zip"},
        )
        artifact = await persist_archive_artifact(
            db,
            user_id=1,
            source_id=int(source["id"]),
            snapshot_id=int(staged_snapshot["id"]),
            filename="notes.zip",
            archive_bytes=archive_buffer.getvalue(),
        )
        await update_source_snapshot(
            db,
            snapshot_id=int(staged_snapshot["id"]),
            summary={"artifact_id": int(artifact["id"])},
        )

        async def _fake_get_db_pool():
            return _SQLitePool(db)

        async def _fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        def _fail_create_sink_db(**kwargs):
            raise RuntimeError(_LEAK)

        monkeypatch.setattr(worker, "get_db_pool", _fake_get_db_pool, raising=False)
        monkeypatch.setattr(worker.asyncio, "to_thread", _fake_to_thread)
        monkeypatch.setattr(worker, "_create_sink_db", _fail_create_sink_db)

        jm = _FakeJobManager()
        await worker._process_sync_job(
            jm,
            jid=103,
            lease_id="lease-103",
            worker_id="worker-1",
            source_id=int(source["id"]),
            user_id=1,
        )

        assert jm.completed is None
        assert jm.failed is not None
        assert jm.failed["error"] == "ingestion_source_sync_failed"
        assert jm.failed["retryable"] is False
        _assert_safe_payload(jm.failed)

        state_cur = await db.execute(
            "SELECT last_sync_status, last_error FROM ingestion_source_state WHERE source_id = ?",
            (int(source["id"]),),
        )
        state = await state_cur.fetchone()
        assert state["last_sync_status"] == "failure"
        assert state["last_error"] == "ingestion_source_sync_failed"

        snapshot_cur = await db.execute(
            "SELECT status, summary_json FROM ingestion_source_snapshots WHERE id = ?",
            (int(staged_snapshot["id"]),),
        )
        snapshot = await snapshot_cur.fetchone()
        snapshot_summary = json.loads(snapshot["summary_json"])
        assert snapshot["status"] == "failed"
        assert snapshot_summary["error"] == "ingestion_source_sync_failed"
        _assert_safe_payload(snapshot_summary)

        artifact_cur = await db.execute(
            "SELECT status, metadata_json FROM ingestion_source_artifacts WHERE id = ?",
            (int(artifact["id"]),),
        )
        artifact_row = await artifact_cur.fetchone()
        artifact_metadata = json.loads(artifact_row["metadata_json"])
        assert artifact_row["status"] == "failed"
        assert artifact_metadata["error"] == "ingestion_source_sync_failed"
        _assert_safe_payload(artifact_metadata)
