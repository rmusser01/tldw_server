from __future__ import annotations

import inspect
import io
import json
import os
import tarfile
import zipfile

import pytest
from fastapi.testclient import TestClient
from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes


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


@pytest.mark.unit
def test_mutating_ingestion_source_routes_apply_rate_limit_dependency(ingestion_sources_client):
    del ingestion_sources_client

    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit
    from tldw_Server_API.app.main import app

    expected_paths = {
        "/api/v1/ingestion-sources/{source_id}/sync",
        "/api/v1/ingestion-sources/{source_id}/archive",
    }
    route_dependencies = {
        route.path: {dependency.call for dependency in route.dependant.dependencies}
        for route in iter_served_routes(app.routes)
        if getattr(route, "path", None) in expected_paths
    }

    assert set(route_dependencies) == expected_paths
    for dependencies in route_dependencies.values():
        assert check_rate_limit in dependencies


@pytest.mark.unit
def test_ingestion_source_endpoint_functions_have_docstrings():
    import tldw_Server_API.app.api.v1.endpoints.ingestion_sources as ep

    missing = [
        func.__name__
        for func in (
            ep.create_ingestion_source,
            ep.list_ingestion_sources,
            ep.get_ingestion_source,
            ep.patch_ingestion_source,
            ep.list_ingestion_source_items,
            ep.trigger_ingestion_source_sync,
            ep.upload_ingestion_source_archive,
            ep.reattach_ingestion_source_item,
        )
        if not inspect.getdoc(func)
    ]

    assert missing == []


@pytest.mark.integration
def test_manual_sync_endpoint_enqueues_job(ingestion_sources_client, monkeypatch):
    client, auth_headers = ingestion_sources_client

    import tldw_Server_API.app.api.v1.endpoints.ingestion_sources as ep

    queued_jobs: list[dict[str, object]] = []

    class _FakeTx:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _FakePool:
        def transaction(self):
            return _FakeTx()

    async def _fake_get_db_pool():
        return _FakePool()

    async def _fake_ensure_schema(_db):
        return None

    async def _fake_get_source_by_id(_db, *, source_id, user_id=None):
        return {
            "id": source_id,
            "user_id": 1 if user_id is None else user_id,
            "enabled": True,
        }

    def _fake_enqueue_ingestion_source_job(*, user_id, source_id, job_type="sync", idempotency_key=None, payload=None):
        queued_jobs.append(
            {
                "user_id": user_id,
                "source_id": source_id,
                "job_type": job_type,
                "idempotency_key": idempotency_key,
            }
        )
        return {"id": "job-17", "status": "queued"}

    monkeypatch.setattr(ep, "get_db_pool", _fake_get_db_pool)
    monkeypatch.setattr(ep, "ensure_ingestion_sources_schema", _fake_ensure_schema)
    monkeypatch.setattr(ep, "get_source_by_id", _fake_get_source_by_id)
    monkeypatch.setattr(ep, "enqueue_ingestion_source_job", _fake_enqueue_ingestion_source_job)

    response = client.post(
        "/api/v1/ingestion-sources/17/sync",
        headers=auth_headers,
    )

    assert response.status_code == 202, response.text
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["source_id"] == 17
    assert payload["job_id"] == "job-17"
    assert queued_jobs[0]["source_id"] == 17


@pytest.mark.integration
def test_create_source_endpoint_returns_400_for_invalid_local_directory(
    ingestion_sources_client,
    monkeypatch,
):
    client, auth_headers = ingestion_sources_client

    import tldw_Server_API.app.api.v1.endpoints.ingestion_sources as ep

    async def _unexpected_get_db_pool():
        raise AssertionError("create endpoint should reject invalid local directory payloads before opening the DB")

    def _fake_validate_local_directory_source(_config):
        raise ValueError("Path outside allowed roots")

    monkeypatch.setattr(ep, "get_db_pool", _unexpected_get_db_pool)
    monkeypatch.setattr(ep, "validate_local_directory_source", _fake_validate_local_directory_source)

    response = client.post(
        "/api/v1/ingestion-sources/",
        headers=auth_headers,
        json={
            "source_type": "local_directory",
            "sink_type": "notes",
            "policy": "canonical",
            "config": {"path": "/tmp/outside"},
        },
    )

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "Path outside allowed roots"


@pytest.mark.integration
def test_create_source_endpoint_returns_400_for_invalid_git_repository_payload(
    ingestion_sources_client,
    monkeypatch,
):
    client, auth_headers = ingestion_sources_client

    import tldw_Server_API.app.api.v1.endpoints.ingestion_sources as ep

    async def _unexpected_get_db_pool():
        raise AssertionError(
            "create endpoint should reject invalid git repository payloads before opening the DB"
        )

    monkeypatch.setattr(ep, "get_db_pool", _unexpected_get_db_pool)

    response = client.post(
        "/api/v1/ingestion-sources/",
        headers=auth_headers,
        json={
            "source_type": "git_repository",
            "sink_type": "notes",
            "policy": "canonical",
            "config": {
                "mode": "remote_github_repo",
                "repo_url": "https://gitlab.com/example/repo",
            },
        },
    )

    assert response.status_code == 400, response.text
    assert "GitHub" in response.json()["detail"]


@pytest.mark.integration
def test_create_source_endpoint_rejects_git_repository_media_sink(
    ingestion_sources_client,
    monkeypatch,
):
    client, auth_headers = ingestion_sources_client

    import tldw_Server_API.app.api.v1.endpoints.ingestion_sources as ep

    async def _unexpected_get_db_pool():
        raise AssertionError(
            "create endpoint should reject unsupported git_repository sink combinations before opening the DB"
        )

    monkeypatch.setattr(ep, "get_db_pool", _unexpected_get_db_pool)

    response = client.post(
        "/api/v1/ingestion-sources/",
        headers=auth_headers,
        json={
            "source_type": "git_repository",
            "sink_type": "media",
            "policy": "canonical",
            "config": {
                "mode": "remote_github_repo",
                "repo_url": "https://github.com/example/repo",
            },
        },
    )

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "Git repository sources currently support the notes sink only"


@pytest.mark.integration
def test_archive_upload_endpoint_stages_snapshot_and_enqueues_job(tmp_path, ingestion_sources_client, monkeypatch):
    client, auth_headers = ingestion_sources_client
    os.environ["USER_DB_BASE_DIR"] = str(tmp_path / "user_dbs")
    os.environ["TEST_MODE"] = "true"

    import aiosqlite
    import tldw_Server_API.app.api.v1.endpoints.ingestion_sources as ep
    from tldw_Server_API.app.core.Ingestion_Sources.service import (
        create_source,
        ensure_ingestion_sources_schema,
    )

    queued_jobs: list[dict[str, object]] = []

    class _FakePool:
        def __init__(self, db):
            self._db = db

        class _Tx:
            def __init__(self, db):
                self._db = db

            async def __aenter__(self):
                return self._db

            async def __aexit__(self, exc_type, exc, tb):
                return False

        def transaction(self):
            return self._Tx(self._db)

    async def _run_test() -> None:
        meta_db_path = tmp_path / "ingestion_sources.sqlite3"
        async with aiosqlite.connect(str(meta_db_path)) as db:
            db.row_factory = aiosqlite.Row
            await ensure_ingestion_sources_schema(db)
            source = await create_source(
                db,
                user_id=1,
                payload={
                    "source_type": "archive_snapshot",
                    "sink_type": "notes",
                    "policy": "canonical",
                    "config": {},
                },
            )

            async def _fake_get_db_pool():
                return _FakePool(db)

            def _fake_enqueue_ingestion_source_job(*, user_id, source_id, job_type="sync", idempotency_key=None, payload=None):
                queued_jobs.append(
                    {
                        "user_id": user_id,
                        "source_id": source_id,
                        "job_type": job_type,
                        "idempotency_key": idempotency_key,
                    }
                )
                return {"id": "job-29", "status": "queued"}

            monkeypatch.setattr(ep, "get_db_pool", _fake_get_db_pool)
            monkeypatch.setattr(ep, "enqueue_ingestion_source_job", _fake_enqueue_ingestion_source_job)

            archive_buffer = io.BytesIO()
            with zipfile.ZipFile(archive_buffer, "w") as archive:
                archive.writestr("export/alpha.md", "# Alpha\n\nzip body\n")
            archive_bytes = archive_buffer.getvalue()

            response = client.post(
                f"/api/v1/ingestion-sources/{int(source['id'])}/archive",
                headers={"X-API-KEY": auth_headers["X-API-KEY"]},
                files={"archive": ("notes.zip", archive_bytes, "application/zip")},
            )

            assert response.status_code == 202, response.text
            payload = response.json()
            assert payload["status"] == "queued"
            assert payload["source_id"] == int(source["id"])
            assert payload["job_id"] == "job-29"
            assert payload["snapshot_status"] == "staged"

            snapshot_cur = await db.execute(
                "SELECT id, status, snapshot_kind, summary_json FROM ingestion_source_snapshots WHERE source_id = ?",
                (int(source["id"]),),
            )
            snapshot_row = await snapshot_cur.fetchone()
            snapshot_summary = json.loads(snapshot_row["summary_json"])
            assert snapshot_row["status"] == "staged"
            assert snapshot_row["snapshot_kind"] == "archive_snapshot"
            assert snapshot_summary["filename"] == "notes.zip"
            assert snapshot_summary["item_count"] == 1
            assert "artifact_id" in snapshot_summary
            assert "items" not in snapshot_summary

            artifact_cur = await db.execute(
                "SELECT snapshot_id, artifact_kind, status, storage_path, metadata_json "
                "FROM ingestion_source_artifacts WHERE source_id = ?",
                (int(source["id"]),),
            )
            artifact_row = await artifact_cur.fetchone()
            assert artifact_row is not None
            assert artifact_row["snapshot_id"] == int(snapshot_row["id"])
            assert artifact_row["artifact_kind"] == "archive_upload"
            assert artifact_row["status"] == "staged"
            artifact_metadata = json.loads(artifact_row["metadata_json"])
            assert artifact_metadata["filename"] == "notes.zip"
            assert artifact_metadata["byte_size"] == len(archive_bytes)
            artifact_path = artifact_row["storage_path"]
            assert artifact_path is not None
            assert os.path.exists(artifact_path)
            with open(artifact_path, "rb") as stored_handle:
                assert stored_handle.read() == archive_bytes
            assert queued_jobs[0]["source_id"] == int(source["id"])

    import asyncio

    asyncio.run(_run_test())


@pytest.mark.integration
def test_archive_upload_endpoint_rejects_unsupported_archive_type(
    tmp_path,
    ingestion_sources_client,
    monkeypatch,
):
    client, auth_headers = ingestion_sources_client
    os.environ["USER_DB_BASE_DIR"] = str(tmp_path / "user_dbs")
    os.environ["TEST_MODE"] = "true"

    import aiosqlite
    import tldw_Server_API.app.api.v1.endpoints.ingestion_sources as ep
    from tldw_Server_API.app.core.Ingestion_Sources.service import (
        create_source,
        ensure_ingestion_sources_schema,
    )

    queued_jobs: list[dict[str, object]] = []

    class _FakePool:
        def __init__(self, db):
            self._db = db

        class _Tx:
            def __init__(self, db):
                self._db = db

            async def __aenter__(self):
                return self._db

            async def __aexit__(self, exc_type, exc, tb):
                return False

        def transaction(self):
            return self._Tx(self._db)

    async def _run_test() -> None:
        meta_db_path = tmp_path / "ingestion_sources.sqlite3"
        async with aiosqlite.connect(str(meta_db_path)) as db:
            db.row_factory = aiosqlite.Row
            await ensure_ingestion_sources_schema(db)
            source = await create_source(
                db,
                user_id=1,
                payload={
                    "source_type": "archive_snapshot",
                    "sink_type": "notes",
                    "policy": "canonical",
                    "config": {},
                },
            )

            async def _fake_get_db_pool():
                return _FakePool(db)

            def _fake_enqueue_ingestion_source_job(*, user_id, source_id, job_type="sync", idempotency_key=None, payload=None):
                queued_jobs.append(
                    {
                        "user_id": user_id,
                        "source_id": source_id,
                        "job_type": job_type,
                        "idempotency_key": idempotency_key,
                    }
                )
                return {"id": "job-unsupported", "status": "queued"}

            monkeypatch.setattr(ep, "get_db_pool", _fake_get_db_pool)
            monkeypatch.setattr(ep, "enqueue_ingestion_source_job", _fake_enqueue_ingestion_source_job)

            response = client.post(
                f"/api/v1/ingestion-sources/{int(source['id'])}/archive",
                headers={"X-API-KEY": auth_headers["X-API-KEY"]},
                files={"archive": ("notes.rar", b"not-a-rar", "application/octet-stream")},
            )

            assert response.status_code == 400, response.text
            assert "Unsupported archive type" in response.json()["detail"]

            snapshot_cur = await db.execute(
                "SELECT COUNT(*) AS count FROM ingestion_source_snapshots WHERE source_id = ?",
                (int(source["id"]),),
            )
            snapshot_row = await snapshot_cur.fetchone()
            assert snapshot_row["count"] == 0
            assert queued_jobs == []

    import asyncio

    asyncio.run(_run_test())


@pytest.mark.integration
def test_archive_upload_endpoint_rejects_oversized_archive(
    tmp_path,
    ingestion_sources_client,
    monkeypatch,
):
    client, auth_headers = ingestion_sources_client
    os.environ["USER_DB_BASE_DIR"] = str(tmp_path / "user_dbs")
    os.environ["TEST_MODE"] = "true"

    import aiosqlite
    import tldw_Server_API.app.api.v1.endpoints.ingestion_sources as ep
    from tldw_Server_API.app.core.Ingestion_Sources.service import (
        create_source,
        ensure_ingestion_sources_schema,
    )

    queued_jobs: list[dict[str, object]] = []

    class _FakePool:
        def __init__(self, db):
            self._db = db

        class _Tx:
            def __init__(self, db):
                self._db = db

            async def __aenter__(self):
                return self._db

            async def __aexit__(self, exc_type, exc, tb):
                return False

        def transaction(self):
            return self._Tx(self._db)

    async def _run_test() -> None:
        meta_db_path = tmp_path / "ingestion_sources.sqlite3"
        async with aiosqlite.connect(str(meta_db_path)) as db:
            db.row_factory = aiosqlite.Row
            await ensure_ingestion_sources_schema(db)
            source = await create_source(
                db,
                user_id=1,
                payload={
                    "source_type": "archive_snapshot",
                    "sink_type": "notes",
                    "policy": "canonical",
                    "config": {},
                },
            )

            async def _fake_get_db_pool():
                return _FakePool(db)

            def _fake_enqueue_ingestion_source_job(*, user_id, source_id, job_type="sync", idempotency_key=None, payload=None):
                queued_jobs.append(
                    {
                        "user_id": user_id,
                        "source_id": source_id,
                        "job_type": job_type,
                        "idempotency_key": idempotency_key,
                    }
                )
                return {"id": "job-too-large", "status": "queued"}

            monkeypatch.setattr(ep, "get_db_pool", _fake_get_db_pool)
            monkeypatch.setattr(ep, "enqueue_ingestion_source_job", _fake_enqueue_ingestion_source_job)
            monkeypatch.setenv("INGESTION_SOURCES_ARCHIVE_UPLOAD_MAX_BYTES", "16")

            response = client.post(
                f"/api/v1/ingestion-sources/{int(source['id'])}/archive",
                headers={"X-API-KEY": auth_headers["X-API-KEY"]},
                files={"archive": ("notes.zip", b"x" * 17, "application/zip")},
            )

            assert response.status_code == 413, response.text
            assert "exceeds the configured maximum size" in response.json()["detail"]

            snapshot_cur = await db.execute(
                "SELECT COUNT(*) AS count FROM ingestion_source_snapshots WHERE source_id = ?",
                (int(source["id"]),),
            )
            snapshot_row = await snapshot_cur.fetchone()
            assert snapshot_row["count"] == 0
            assert queued_jobs == []

    import asyncio

    asyncio.run(_run_test())


@pytest.mark.integration
def test_archive_upload_endpoint_accepts_tar_gz(tmp_path, ingestion_sources_client, monkeypatch):
    client, auth_headers = ingestion_sources_client
    os.environ["USER_DB_BASE_DIR"] = str(tmp_path / "user_dbs")
    os.environ["TEST_MODE"] = "true"

    import aiosqlite
    import tldw_Server_API.app.api.v1.endpoints.ingestion_sources as ep
    from tldw_Server_API.app.core.Ingestion_Sources.service import (
        create_source,
        ensure_ingestion_sources_schema,
    )

    queued_jobs: list[dict[str, object]] = []

    class _FakePool:
        def __init__(self, db):
            self._db = db

        class _Tx:
            def __init__(self, db):
                self._db = db

            async def __aenter__(self):
                return self._db

            async def __aexit__(self, exc_type, exc, tb):
                return False

        def transaction(self):
            return self._Tx(self._db)

    async def _run_test() -> None:
        meta_db_path = tmp_path / "ingestion_sources.sqlite3"
        async with aiosqlite.connect(str(meta_db_path)) as db:
            db.row_factory = aiosqlite.Row
            await ensure_ingestion_sources_schema(db)
            source = await create_source(
                db,
                user_id=1,
                payload={
                    "source_type": "archive_snapshot",
                    "sink_type": "notes",
                    "policy": "canonical",
                    "config": {},
                },
            )

            async def _fake_get_db_pool():
                return _FakePool(db)

            def _fake_enqueue_ingestion_source_job(*, user_id, source_id, job_type="sync", idempotency_key=None, payload=None):
                queued_jobs.append(
                    {
                        "user_id": user_id,
                        "source_id": source_id,
                        "job_type": job_type,
                        "idempotency_key": idempotency_key,
                    }
                )
                return {"id": "job-41", "status": "queued"}

            monkeypatch.setattr(ep, "get_db_pool", _fake_get_db_pool)
            monkeypatch.setattr(ep, "enqueue_ingestion_source_job", _fake_enqueue_ingestion_source_job)

            archive_buffer = io.BytesIO()
            with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
                payload = b"# Alpha\n\nfrom tar upload\n"
                member = tarfile.TarInfo("export/alpha.md")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            archive_bytes = archive_buffer.getvalue()

            response = client.post(
                f"/api/v1/ingestion-sources/{int(source['id'])}/archive",
                headers={"X-API-KEY": auth_headers["X-API-KEY"]},
                files={"archive": ("notes.tar.gz", archive_bytes, "application/gzip")},
            )

            assert response.status_code == 202, response.text
            payload = response.json()
            assert payload["status"] == "queued"
            assert payload["source_id"] == int(source["id"])
            assert payload["job_id"] == "job-41"
            assert payload["snapshot_status"] == "staged"

            snapshot_cur = await db.execute(
                "SELECT id, status, summary_json FROM ingestion_source_snapshots WHERE source_id = ?",
                (int(source["id"]),),
            )
            snapshot_row = await snapshot_cur.fetchone()
            snapshot_summary = json.loads(snapshot_row["summary_json"])
            assert snapshot_row["status"] == "staged"
            assert snapshot_summary["filename"] == "notes.tar.gz"
            assert snapshot_summary["item_count"] == 1
            assert queued_jobs[0]["source_id"] == int(source["id"])

    import asyncio

    asyncio.run(_run_test())


@pytest.mark.integration
def test_list_source_items_endpoint_returns_tracked_items(tmp_path, ingestion_sources_client, monkeypatch):
    client, auth_headers = ingestion_sources_client

    import aiosqlite
    import tldw_Server_API.app.api.v1.endpoints.ingestion_sources as ep
    from tldw_Server_API.app.core.Ingestion_Sources.service import (
        create_source,
        ensure_ingestion_sources_schema,
        upsert_source_item,
    )

    class _FakePool:
        def __init__(self, db):
            self._db = db

        class _Tx:
            def __init__(self, db):
                self._db = db

            async def __aenter__(self):
                return self._db

            async def __aexit__(self, exc_type, exc, tb):
                return False

        def transaction(self):
            return self._Tx(self._db)

    async def _run_test() -> None:
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
                    "config": {"path": "/tmp/example"},
                },
            )
            await upsert_source_item(
                db,
                source_id=int(source["id"]),
                normalized_relative_path="alpha.md",
                content_hash="hash-1",
                sync_status="sync_managed",
                binding={"note_id": "note-1", "sync_status": "sync_managed", "current_version": 2},
                present_in_source=True,
            )

            async def _fake_get_db_pool():
                return _FakePool(db)

            monkeypatch.setattr(ep, "get_db_pool", _fake_get_db_pool)

            response = client.get(
                f"/api/v1/ingestion-sources/{int(source['id'])}/items",
                headers=auth_headers,
            )

            assert response.status_code == 200, response.text
            payload = response.json()
            assert len(payload) == 1
            assert payload[0]["normalized_relative_path"] == "alpha.md"
            assert payload[0]["sync_status"] == "sync_managed"
            assert payload[0]["binding"]["note_id"] == "note-1"
            assert payload[0]["present_in_source"] is True

    import asyncio

    asyncio.run(_run_test())


@pytest.mark.integration
def test_source_responses_include_last_successful_sync_summary(tmp_path, ingestion_sources_client, monkeypatch):
    client, auth_headers = ingestion_sources_client

    import aiosqlite
    import tldw_Server_API.app.api.v1.endpoints.ingestion_sources as ep
    from tldw_Server_API.app.core.Ingestion_Sources.service import (
        create_source,
        create_source_snapshot,
        ensure_ingestion_sources_schema,
    )

    class _FakePool:
        def __init__(self, db):
            self._db = db

        class _Tx:
            def __init__(self, db):
                self._db = db

            async def __aenter__(self):
                return self._db

            async def __aexit__(self, exc_type, exc, tb):
                return False

        def transaction(self):
            return self._Tx(self._db)

    async def _run_test() -> None:
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
                    "config": {"path": "/tmp/example"},
                },
            )
            snapshot = await create_source_snapshot(
                db,
                source_id=int(source["id"]),
                snapshot_kind="local_directory",
                status="success",
                summary={
                    "processed": 3,
                    "degraded_items": 2,
                    "sink_failed_items": 1,
                    "ingestion_failed_items": 1,
                },
            )
            await db.execute(
                "UPDATE ingestion_source_state "
                "SET last_successful_snapshot_id = ?, last_sync_status = 'success' "
                "WHERE source_id = ?",
                (int(snapshot["id"]), int(source["id"])),
            )

            async def _fake_get_db_pool():
                return _FakePool(db)

            monkeypatch.setattr(ep, "get_db_pool", _fake_get_db_pool)

            detail_response = client.get(
                f"/api/v1/ingestion-sources/{int(source['id'])}",
                headers=auth_headers,
            )

            assert detail_response.status_code == 200, detail_response.text
            detail_payload = detail_response.json()
            assert detail_payload["last_successful_sync_summary"]["processed"] == 3
            assert detail_payload["last_successful_sync_summary"]["degraded_items"] == 2
            assert detail_payload["last_successful_sync_summary"]["sink_failed_items"] == 1

            list_response = client.get(
                "/api/v1/ingestion-sources",
                headers=auth_headers,
            )

            assert list_response.status_code == 200, list_response.text
            list_payload = list_response.json()
            assert len(list_payload) == 1
            assert list_payload[0]["last_successful_sync_summary"]["ingestion_failed_items"] == 1

    import asyncio

    asyncio.run(_run_test())


@pytest.mark.integration
def test_reattach_item_endpoint_clears_detached_status(tmp_path, ingestion_sources_client, monkeypatch):
    client, auth_headers = ingestion_sources_client

    os.environ["USER_DB_BASE_DIR"] = str(tmp_path / "user_dbs")
    os.environ["TEST_MODE"] = "true"

    import aiosqlite
    import tldw_Server_API.app.api.v1.endpoints.ingestion_sources as ep
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.app.core.Ingestion_Sources.service import (
        create_source,
        ensure_ingestion_sources_schema,
        upsert_source_item,
    )

    class _FakePool:
        def __init__(self, db):
            self._db = db

        class _Tx:
            def __init__(self, db):
                self._db = db

            async def __aenter__(self):
                return self._db

            async def __aexit__(self, exc_type, exc, tb):
                return False

        def transaction(self):
            return self._Tx(self._db)

    async def _run_test() -> None:
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
                    "config": {"path": "/tmp/example"},
                },
            )

            notes_db = CharactersRAGDB(
                db_path=str(DatabasePaths.get_chacha_db_path(1)),
                client_id="1",
            )
            note_id = notes_db.add_note(title="Alpha", content="manual body")
            note = notes_db.get_note_by_id(note_id=note_id)
            assert note is not None

            item = await upsert_source_item(
                db,
                source_id=int(source["id"]),
                normalized_relative_path="alpha.md",
                content_hash="hash-1",
                sync_status="conflict_detached",
                binding={
                    "note_id": note_id,
                    "sync_status": "conflict_detached",
                    "current_version": 1,
                },
                present_in_source=True,
            )

            async def _fake_get_db_pool():
                return _FakePool(db)

            monkeypatch.setattr(ep, "get_db_pool", _fake_get_db_pool)

            response = client.post(
                f"/api/v1/ingestion-sources/{int(source['id'])}/items/{int(item['id'])}/reattach",
                headers=auth_headers,
            )

            assert response.status_code == 200, response.text
            payload = response.json()
            assert payload["id"] == int(item["id"])
            assert payload["sync_status"] == "sync_managed"
            assert payload["binding"]["note_id"] == note_id
            assert payload["binding"]["sync_status"] == "sync_managed"
            assert payload["binding"]["current_version"] == int(note["version"])

            item_cur = await db.execute(
                "SELECT sync_status, binding_json FROM ingestion_source_items WHERE id = ?",
                (int(item["id"]),),
            )
            item_row = await item_cur.fetchone()
            assert item_row["sync_status"] == "sync_managed"
            persisted_binding = json.loads(item_row["binding_json"])
            assert persisted_binding["sync_status"] == "sync_managed"
            assert persisted_binding["current_version"] == int(note["version"])

    import asyncio

    asyncio.run(_run_test())


@pytest.mark.integration
def test_patch_source_endpoint_allows_git_repository_identity_updates_before_first_success(
    tmp_path,
    ingestion_sources_client,
    monkeypatch,
):
    client, auth_headers = ingestion_sources_client

    import aiosqlite
    import tldw_Server_API.app.api.v1.endpoints.ingestion_sources as ep
    from tldw_Server_API.app.core.Ingestion_Sources.service import (
        create_source,
        ensure_ingestion_sources_schema,
        get_source_by_id,
    )

    class _FakePool:
        def __init__(self, db):
            self._db = db

        class _Tx:
            def __init__(self, db):
                self._db = db

            async def __aenter__(self):
                return self._db

            async def __aexit__(self, exc_type, exc, tb):
                return False

        def transaction(self):
            return self._Tx(self._db)

    async def _run_test() -> None:
        meta_db_path = tmp_path / "ingestion_sources.sqlite3"
        async with aiosqlite.connect(str(meta_db_path)) as db:
            db.row_factory = aiosqlite.Row
            await ensure_ingestion_sources_schema(db)
            source = await create_source(
                db,
                user_id=1,
                payload={
                    "source_type": "git_repository",
                    "sink_type": "notes",
                    "policy": "canonical",
                    "config": {
                        "mode": "remote_github_repo",
                        "repo_url": "https://github.com/example/repo",
                    },
                },
            )

            async def _fake_get_db_pool():
                return _FakePool(db)

            monkeypatch.setattr(ep, "get_db_pool", _fake_get_db_pool)

            response = client.patch(
                f"/api/v1/ingestion-sources/{int(source['id'])}",
                headers=auth_headers,
                json={
                    "config": {
                        "mode": "remote_github_repo",
                        "repo_url": "https://github.com/example/repo",
                        "ref": "main",
                        "root_subpath": "docs/notes",
                    }
                },
            )

            assert response.status_code == 200, response.text
            payload = response.json()
            assert payload["config"] == {
                "mode": "remote_github_repo",
                "repo_url": "https://github.com/example/repo",
                "repo_owner": "example",
                "repo_name": "repo",
                "ref": "main",
                "root_subpath": "docs/notes",
            }

            persisted = await get_source_by_id(db, source_id=int(source["id"]), user_id=1)
            assert persisted is not None
            assert persisted["config"] == payload["config"]

    import asyncio

    asyncio.run(_run_test())


@pytest.mark.integration
def test_patch_source_endpoint_updates_mutable_fields(tmp_path, ingestion_sources_client, monkeypatch):
    client, auth_headers = ingestion_sources_client

    import aiosqlite
    import tldw_Server_API.app.api.v1.endpoints.ingestion_sources as ep
    from tldw_Server_API.app.core.Ingestion_Sources.service import (
        create_source,
        ensure_ingestion_sources_schema,
        get_source_by_id,
    )

    class _FakePool:
        def __init__(self, db):
            self._db = db

        class _Tx:
            def __init__(self, db):
                self._db = db

            async def __aenter__(self):
                return self._db

            async def __aexit__(self, exc_type, exc, tb):
                return False

        def transaction(self):
            return self._Tx(self._db)

    async def _run_test() -> None:
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
                    "enabled": True,
                    "schedule_enabled": False,
                    "schedule": {},
                    "config": {"path": "/tmp/example"},
                },
            )

            async def _fake_get_db_pool():
                return _FakePool(db)

            monkeypatch.setattr(ep, "get_db_pool", _fake_get_db_pool)

            response = client.patch(
                f"/api/v1/ingestion-sources/{int(source['id'])}",
                headers=auth_headers,
                json={
                    "enabled": False,
                    "schedule_enabled": True,
                    "schedule": {"interval_minutes": 15},
                    "policy": "import_only",
                },
            )

            assert response.status_code == 200, response.text
            payload = response.json()
            assert payload["enabled"] is False
            assert payload["schedule_enabled"] is True
            assert payload["schedule_config"] == {"interval_minutes": 15}
            assert payload["policy"] == "import_only"

            persisted = await get_source_by_id(db, source_id=int(source["id"]), user_id=1)
            assert persisted is not None
            assert persisted["enabled"] is False
            assert persisted["schedule_enabled"] is True
            assert persisted["schedule_config"] == {"interval_minutes": 15}
            assert persisted["policy"] == "import_only"

    import asyncio

    asyncio.run(_run_test())


@pytest.mark.integration
def test_patch_source_endpoint_rejects_sink_change_after_first_success(tmp_path, ingestion_sources_client, monkeypatch):
    client, auth_headers = ingestion_sources_client

    import aiosqlite
    import tldw_Server_API.app.api.v1.endpoints.ingestion_sources as ep
    from tldw_Server_API.app.core.Ingestion_Sources.service import (
        create_source,
        ensure_ingestion_sources_schema,
        get_source_by_id,
    )

    class _FakePool:
        def __init__(self, db):
            self._db = db

        class _Tx:
            def __init__(self, db):
                self._db = db

            async def __aenter__(self):
                return self._db

            async def __aexit__(self, exc_type, exc, tb):
                return False

        def transaction(self):
            return self._Tx(self._db)

    async def _run_test() -> None:
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
                    "config": {"path": "/tmp/example"},
                },
            )
            await db.execute(
                "UPDATE ingestion_source_state SET last_successful_snapshot_id = ? WHERE source_id = ?",
                (9, int(source["id"])),
            )

            async def _fake_get_db_pool():
                return _FakePool(db)

            monkeypatch.setattr(ep, "get_db_pool", _fake_get_db_pool)

            response = client.patch(
                f"/api/v1/ingestion-sources/{int(source['id'])}",
                headers=auth_headers,
                json={"sink_type": "media"},
            )

            assert response.status_code == 409, response.text
            assert response.json()["detail"] == "Source identity is immutable after the first successful sync"

            persisted = await get_source_by_id(db, source_id=int(source["id"]), user_id=1)
            assert persisted is not None
            assert persisted["sink_type"] == "notes"

    import asyncio

    asyncio.run(_run_test())
