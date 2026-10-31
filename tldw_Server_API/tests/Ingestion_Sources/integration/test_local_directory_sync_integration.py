from __future__ import annotations

import json
from contextlib import asynccontextmanager

import aiosqlite
import pytest


class _FakeJobManager:
    def __init__(self) -> None:
        self.completed: dict[str, object] | None = None
        self.failed: dict[str, object] | None = None

    def renew_job_lease(self, *args, **kwargs) -> None:
        return None

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


@pytest.mark.asyncio
@pytest.mark.integration
async def test_local_directory_rescan_applies_create_change_and_delete_for_notes_sink(
    tmp_path,
    monkeypatch,
):
    source_root = tmp_path / "allowed" / "docs"
    source_root.mkdir(parents=True)
    alpha_path = source_root / "alpha.md"
    beta_path = source_root / "beta.md"
    alpha_path.write_text("# Alpha\n\nfirst alpha body\n", encoding="utf-8")
    beta_path.write_text("# Beta\n\nfirst beta body\n", encoding="utf-8")

    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("INGESTION_SOURCE_ALLOWED_ROOTS", str(tmp_path / "allowed"))
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))

    import tldw_Server_API.app.services.ingestion_sources_worker as worker
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.app.core.Ingestion_Sources.service import (
        create_source,
        ensure_ingestion_sources_schema,
    )

    meta_db_path = tmp_path / "ingestion_sources.sqlite3"
    async with aiosqlite.connect(str(meta_db_path)) as db:
        db.row_factory = aiosqlite.Row
        await ensure_ingestion_sources_schema(db)
        source = await create_source(
            db,
            user_id=1,
            payload={
                "source_type": "local_directory",
                "sink_type": "notes",
                "policy": "canonical",
                "config": {"path": str(source_root)},
            },
        )

        async def _fake_get_db_pool():
            return _SQLitePool(db)

        monkeypatch.setattr(worker, "get_db_pool", _fake_get_db_pool, raising=False)

        first_job = _FakeJobManager()
        await worker._process_sync_job(
            first_job,
            jid=201,
            lease_id="lease-201",
            worker_id="worker-1",
            source_id=int(source["id"]),
            user_id=1,
        )
        assert first_job.failed is None

        first_items_cur = await db.execute(
            "SELECT normalized_relative_path, binding_json FROM ingestion_source_items WHERE source_id = ? ORDER BY normalized_relative_path ASC",
            (int(source["id"]),),
        )
        first_items = await first_items_cur.fetchall()
        item_bindings = {
            row["normalized_relative_path"]: json.loads(row["binding_json"])
            for row in first_items
        }
        beta_note_id = str(item_bindings["beta.md"]["note_id"])

        alpha_path.write_text("# Alpha\n\nupdated alpha body\n", encoding="utf-8")
        beta_path.unlink()
        gamma_path = source_root / "gamma.md"
        gamma_path.write_text("# Gamma\n\nnew gamma body\n", encoding="utf-8")

        second_job = _FakeJobManager()
        await worker._process_sync_job(
            second_job,
            jid=202,
            lease_id="lease-202",
            worker_id="worker-1",
            source_id=int(source["id"]),
            user_id=1,
        )

        assert second_job.failed is None
        assert second_job.completed is not None
        assert int(second_job.completed["result"]["processed"]) == 3
        assert int(second_job.completed["result"]["created"]) == 1
        assert int(second_job.completed["result"]["changed"]) == 1
        assert int(second_job.completed["result"]["deleted"]) == 1

        item_cur = await db.execute(
            "SELECT normalized_relative_path, sync_status, binding_json, present_in_source "
            "FROM ingestion_source_items WHERE source_id = ? ORDER BY normalized_relative_path ASC",
            (int(source["id"]),),
        )
        rows = await item_cur.fetchall()
        by_path = {row["normalized_relative_path"]: row for row in rows}

        assert by_path["alpha.md"]["sync_status"] == "sync_managed"
        assert by_path["alpha.md"]["present_in_source"] == 1
        assert by_path["beta.md"]["sync_status"] == "archived_upstream_removed"
        assert by_path["beta.md"]["present_in_source"] == 0
        assert by_path["gamma.md"]["sync_status"] == "sync_managed"
        assert by_path["gamma.md"]["present_in_source"] == 1

        notes_db = CharactersRAGDB(
            db_path=str(DatabasePaths.get_chacha_db_path(1)),
            client_id="1",
        )
        active_notes = {
            note["title"]: note
            for note in notes_db.list_notes(limit=20)
        }
        assert active_notes["Alpha"]["content"] == "# Alpha\n\nupdated alpha body\n"
        assert active_notes["Gamma"]["content"] == "# Gamma\n\nnew gamma body\n"

        deleted_notes = {
            str(note["id"]): note
            for note in notes_db.list_deleted_notes(limit=20)
        }
        assert beta_note_id in deleted_notes
