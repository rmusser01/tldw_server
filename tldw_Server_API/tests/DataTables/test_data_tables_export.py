import contextlib
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.API_Deps.Collections_DB_Deps import get_collections_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.endpoints.data_tables import router as data_tables_router
from tldw_Server_API.app.api.v1.endpoints.files import router as files_router
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase

pytestmark = pytest.mark.integration


def _principal_override():
    async def _override(request=None) -> AuthPrincipal:
        principal = AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject="test-user",
            token_type="single_user",  # nosec B106 - auth principal stub token type, not a credential
            jti=None,
            roles=["admin"],
            permissions=["media.create", "media.read", "media.update", "media.delete"],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )
        if request is not None:
            request.state.auth = AuthContext(
                principal=principal,
                ip=None,
                user_agent=None,
                request_id=None,
            )
        return principal

    return _override


def _build_app(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))

    base_dir = tmp_path / "user_dbs"
    shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir(parents=True, exist_ok=True)

    prev_base_dir = settings.get("USER_DB_BASE_DIR")
    settings.USER_DB_BASE_DIR = str(base_dir)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))

    media_db_path = DatabasePaths.get_media_db_path(1)
    media_db = MediaDatabase(db_path=str(media_db_path), client_id="test_client")
    collections_db = CollectionsDatabase.for_user(user_id=1)

    app = FastAPI()
    app.include_router(data_tables_router, prefix="/api/v1", tags=["data-tables"])
    app.include_router(files_router, prefix="/api/v1", tags=["files"])

    async def _override_user() -> User:
        return User(id=1, username="tester", email=None, is_active=True, is_admin=True)

    async def _override_media_db():
        yield media_db

    async def _override_collections_db():
        return collections_db

    app.dependency_overrides[get_request_user] = _override_user
    app.dependency_overrides[get_auth_principal] = _principal_override()
    app.dependency_overrides[get_media_db_for_user] = _override_media_db
    app.dependency_overrides[get_collections_db_for_user] = _override_collections_db
    return app, media_db, collections_db, prev_base_dir


def _restore_settings(prev_base_dir):
    if prev_base_dir is not None:
        settings.USER_DB_BASE_DIR = prev_base_dir
    else:
        with contextlib.suppress(AttributeError):
            del settings.USER_DB_BASE_DIR


def test_export_data_table_csv(tmp_path, monkeypatch):
    app, media_db, collections_db, prev_base_dir = _build_app(tmp_path, monkeypatch)
    try:
        table = media_db.create_data_table(
            name="Roster",
            prompt="Export",
            description="Export table",
            status="ready",
            row_count=2,
        )
        table_id = int(table.get("id"))
        media_db.insert_data_table_columns(
            table_id,
            [
                {"name": "Name", "type": "text", "position": 0},
                {"name": "Score", "type": "number", "position": 1},
            ],
        )
        columns = media_db.list_data_table_columns(table_id)
        column_ids = [col.get("column_id") for col in columns]
        media_db.insert_data_table_rows(
            table_id,
            [
                {"row_index": 0, "row_json": {column_ids[0]: "Ada", column_ids[1]: 95}},
                {"row_index": 1, "row_json": {column_ids[0]: "Bob", column_ids[1]: 87}},
            ],
        )
        media_db.update_data_table(table_id, row_count=2)

        with TestClient(app) as client:
            table_uuid = table.get("uuid")
            resp = client.get(
                f"/api/v1/data-tables/{table_uuid}/export?format=csv&async_mode=sync&download=true"
            )
            assert resp.status_code == 200, resp.text
            assert resp.headers.get("content-type", "").startswith("text/csv")
            assert resp.headers.get("content-disposition")
            assert "Name,Score" in resp.text
            assert "Ada,95" in resp.text
    finally:
        app.dependency_overrides.clear()
        media_db.close_connection()
        collections_db.close()
        _restore_settings(prev_base_dir)


def test_export_data_table_async_pending(tmp_path, monkeypatch):
    app, media_db, collections_db, prev_base_dir = _build_app(tmp_path, monkeypatch)
    try:
        table = media_db.create_data_table(
            name="Roster Async",
            prompt="Export",
            description="Export table",
            status="ready",
            row_count=1,
        )
        table_id = int(table.get("id"))
        media_db.insert_data_table_columns(
            table_id,
            [{"name": "Name", "type": "text", "position": 0}],
        )
        columns = media_db.list_data_table_columns(table_id)
        column_ids = [col.get("column_id") for col in columns]
        media_db.insert_data_table_rows(
            table_id,
            [{"row_index": 0, "row_json": {column_ids[0]: "Ada"}}],
        )
        media_db.update_data_table(table_id, row_count=1)

        async def fake_create_artifact(self, file_req, request_id=None):  # noqa: ANN001, ARG001
            return (
                SimpleNamespace(
                    file_id=77,
                    export={"status": "pending", "format": "csv", "job_id": "job-77"},
                ),
                202,
            )

        monkeypatch.setattr(
            "tldw_Server_API.app.api.v1.endpoints.data_tables.FileArtifactsService.create_artifact",
            fake_create_artifact,
        )

        with TestClient(app) as client:
            table_uuid = table.get("uuid")
            resp = client.get(f"/api/v1/data-tables/{table_uuid}/export?format=csv&async_mode=async")
            assert resp.status_code == 202, resp.text
            payload = resp.json()
            export_info = payload["export"]
            assert export_info["status"] == "pending"
            assert export_info["job_id"]
    finally:
        app.dependency_overrides.clear()
        media_db.close_connection()
        collections_db.close()
        _restore_settings(prev_base_dir)


def test_export_maps_table_fetch_database_error(tmp_path, monkeypatch):
    app, media_db, collections_db, prev_base_dir = _build_app(tmp_path, monkeypatch)

    def fail_fetch(self, *args, **kwargs):  # noqa: ANN001, ARG001
        raise DatabaseError("driver failed")

    monkeypatch.setattr(MediaDatabase, "get_data_table_by_uuid", fail_fetch)

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/api/v1/data-tables/table-uuid/export?format=csv&async_mode=sync")
            assert resp.status_code == 500, resp.text
            assert resp.json()["detail"] == "Failed to export data table"
    finally:
        app.dependency_overrides.clear()
        media_db.close_connection()
        collections_db.close()
        _restore_settings(prev_base_dir)


def test_export_maps_columns_database_error(tmp_path, monkeypatch):
    app, media_db, collections_db, prev_base_dir = _build_app(tmp_path, monkeypatch)
    try:
        table = media_db.create_data_table(
            name="Roster",
            prompt="Export",
            description="Export table",
            status="ready",
            row_count=1,
        )

        def fail_columns(self, *args, **kwargs):  # noqa: ANN001, ARG001
            raise DatabaseError("driver failed")

        monkeypatch.setattr(MediaDatabase, "list_data_table_columns", fail_columns)

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get(
                f"/api/v1/data-tables/{table.get('uuid')}/export?format=csv&async_mode=sync"
            )
            assert resp.status_code == 500, resp.text
            assert resp.json()["detail"] == "Failed to export data table"
    finally:
        app.dependency_overrides.clear()
        media_db.close_connection()
        collections_db.close()
        _restore_settings(prev_base_dir)


def test_export_maps_rows_database_error(tmp_path, monkeypatch):
    app, media_db, collections_db, prev_base_dir = _build_app(tmp_path, monkeypatch)
    try:
        table = media_db.create_data_table(
            name="Roster",
            prompt="Export",
            description="Export table",
            status="ready",
            row_count=1,
        )
        table_id = int(table.get("id"))
        media_db.insert_data_table_columns(
            table_id,
            [{"name": "Name", "type": "text", "position": 0}],
        )

        def fail_rows(self, *args, **kwargs):  # noqa: ANN001, ARG001
            raise DatabaseError("driver failed")

        monkeypatch.setattr(MediaDatabase, "list_data_table_rows", fail_rows)

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get(
                f"/api/v1/data-tables/{table.get('uuid')}/export?format=csv&async_mode=sync"
            )
            assert resp.status_code == 500, resp.text
            assert resp.json()["detail"] == "Failed to export data table"
    finally:
        app.dependency_overrides.clear()
        media_db.close_connection()
        collections_db.close()
        _restore_settings(prev_base_dir)
