from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager

import aiosqlite
import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes


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


@pytest.fixture()
def ingestion_sources_client():
    os.environ.setdefault("TEST_MODE", "true")
    os.environ.setdefault("ROUTES_STABLE_ONLY", "false")
    os.environ["ROUTES_ENABLE"] = "ingestion-sources"
    os.environ.setdefault("AUTH_MODE", "single_user")
    os.environ.setdefault("TESTING", "true")

    from tldw_Server_API.app.api.v1.endpoints import ingestion_sources as ingestion_sources_router
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings
    from tldw_Server_API.app.main import app

    paths = {route.path for route in iter_served_routes(app.routes)}
    if "/api/v1/ingestion-sources/{source_id}/sync" not in paths:
        app.include_router(ingestion_sources_router.router, prefix="/api/v1", tags=["ingestion-sources"])

    api_key = get_settings().SINGLE_USER_API_KEY
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    client = TestClient(app)
    return client, headers


@pytest.mark.asyncio
@pytest.mark.integration
async def test_reattach_allows_next_sync_to_apply_pending_upstream_note_change(
    tmp_path,
    monkeypatch,
    ingestion_sources_client,
):
    client, auth_headers = ingestion_sources_client

    source_root = tmp_path / "allowed" / "docs"
    source_root.mkdir(parents=True)
    source_file = source_root / "alpha.md"
    source_file.write_text("# Alpha\n\nfirst sync body\n", encoding="utf-8")

    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("INGESTION_SOURCE_ALLOWED_ROOTS", str(tmp_path / "allowed"))
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))

    import tldw_Server_API.app.api.v1.endpoints.ingestion_sources as endpoint_module
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
        monkeypatch.setattr(endpoint_module, "get_db_pool", _fake_get_db_pool)

        first_job = _FakeJobManager()
        await worker._process_sync_job(
            first_job,
            jid=101,
            lease_id="lease-101",
            worker_id="worker-1",
            source_id=int(source["id"]),
            user_id=1,
        )
        assert first_job.failed is None

        item_cur = await db.execute(
            "SELECT id, binding_json FROM ingestion_source_items WHERE source_id = ? AND normalized_relative_path = ?",
            (int(source["id"]), "alpha.md"),
        )
        item_row = await item_cur.fetchone()
        item_id = int(item_row["id"])
        binding = json.loads(item_row["binding_json"])
        note_id = str(binding["note_id"])
        current_version = int(binding["current_version"])

        notes_db = CharactersRAGDB(
            db_path=str(DatabasePaths.get_chacha_db_path(1)),
            client_id="1",
        )
        notes_db.update_note(
            note_id,
            {"title": "Alpha local edit", "content": "manual local edit"},
            expected_version=current_version,
        )

        source_file.write_text("# Alpha\n\nupstream changed body\n", encoding="utf-8")

        detached_job = _FakeJobManager()
        await worker._process_sync_job(
            detached_job,
            jid=102,
            lease_id="lease-102",
            worker_id="worker-1",
            source_id=int(source["id"]),
            user_id=1,
        )
        assert detached_job.failed is None
        assert detached_job.completed is not None
        assert detached_job.completed["result"]["detached_conflicts"] == 1

        note_after_detach = notes_db.get_note_by_id(note_id=note_id)
        assert note_after_detach is not None
        assert note_after_detach["content"] == "manual local edit"

        response = client.post(
            f"/api/v1/ingestion-sources/{int(source['id'])}/items/{item_id}/reattach",
            headers=auth_headers,
        )
        assert response.status_code == 200, response.text

        reattach_job = _FakeJobManager()
        await worker._process_sync_job(
            reattach_job,
            jid=103,
            lease_id="lease-103",
            worker_id="worker-1",
            source_id=int(source["id"]),
            user_id=1,
        )

        assert reattach_job.failed is None
        assert reattach_job.completed is not None
        assert int(reattach_job.completed["result"]["processed"]) == 1

        final_item_cur = await db.execute(
            "SELECT sync_status, binding_json FROM ingestion_source_items WHERE id = ?",
            (item_id,),
        )
        final_item = await final_item_cur.fetchone()
        final_binding = json.loads(final_item["binding_json"])
        assert final_item["sync_status"] == "sync_managed"
        assert final_binding["sync_status"] == "sync_managed"

        note_after_reattach = notes_db.get_note_by_id(note_id=note_id)
        assert note_after_reattach is not None
        assert note_after_reattach["content"] == "# Alpha\n\nupstream changed body"
