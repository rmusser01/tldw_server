import asyncio

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.endpoints import data_tables as data_tables_endpoint
from tldw_Server_API.app.api.v1.endpoints.data_tables import (
    _wait_for_job_completion,
    get_job_manager,
)
from tldw_Server_API.app.api.v1.schemas.data_tables_schemas import DATA_TABLES_MAX_ROWS_LIMIT
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError, InputError
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase


pytestmark = pytest.mark.integration


class _LoggerStub:
    def __init__(self):
        self.debugs = []
        self.errors = []
        self.exceptions = []

    def debug(self, *args, **kwargs):
        self.debugs.append((args, kwargs))

    def error(self, *args, **kwargs):
        self.errors.append((args, kwargs))

    def exception(self, *args, **kwargs):
        self.exceptions.append((args, kwargs))


def _assert_sanitized_error_log(
    logger_stub: _LoggerStub,
    expected_message: str,
    raw_marker: str,
) -> None:
    assert logger_stub.exceptions == []
    assert logger_stub.errors
    args, kwargs = logger_stub.errors[-1]
    rendered = " ".join(str(arg) for arg in args)

    assert args == (expected_message,)
    assert raw_marker not in rendered
    assert "/private/" not in rendered
    assert raw_marker not in str(kwargs)
    assert "/private/" not in str(kwargs)


def _assert_sanitized_debug_log(
    logger_stub: _LoggerStub,
    expected_message: str,
    raw_marker: str,
) -> None:
    assert logger_stub.debugs
    args, kwargs = logger_stub.debugs[-1]
    rendered = " ".join(str(arg) for arg in args)

    assert args == (expected_message,)
    assert raw_marker not in rendered
    assert "/private/" not in rendered
    assert raw_marker not in str(kwargs)
    assert "/private/" not in str(kwargs)


def test_generate_and_get_data_table(tmp_path, data_tables_app_factory):
    db_path = tmp_path / "media.db"
    app, _ = data_tables_app_factory(db_path)

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/data-tables/generate",
            json={
                "name": "Test Table",
                "prompt": "Extract data",
                "description": "demo",
                "sources": [{"source_type": "chat", "source_id": "chat_1", "title": "Chat 1"}],
                "column_hints": [{"name": "Name", "type": "text"}],
                "model": "gpt-test",
                "max_rows": 10,
            },
        )
        assert resp.status_code == 202, resp.text
        payload = resp.json()
        table_uuid = payload["table"]["uuid"]
        job_id = payload["job_id"]
        assert table_uuid
        assert job_id

        detail = client.get(f"/api/v1/data-tables/{table_uuid}")
        assert detail.status_code == 200, detail.text
        detail_payload = detail.json()
        assert detail_payload["table"]["uuid"] == table_uuid
        assert detail_payload["sources"]
        assert detail_payload["pagination"] == {
            "mode": "offset",
            "limit": 200,
            "offset": 0,
            "total": 0,
            "has_more": False,
            "next_offset": None,
        }


def test_mark_data_table_generate_failed_sanitizes_update_failure_log(monkeypatch):
    logger_stub = _LoggerStub()

    class _FailingUpdateDb:
        def update_data_table(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise RuntimeError("failed marker update at /private/data-tables.db")

    monkeypatch.setattr(data_tables_endpoint, "logger", logger_stub)

    data_tables_endpoint._mark_data_table_generate_failed(
        _FailingUpdateDb(),
        table_id=12,
        owner_user_id=1,
        exc=RuntimeError("original failure at /private/original.db"),
    )

    _assert_sanitized_debug_log(
        logger_stub,
        "data_tables.generate: failed to mark table as failed",
        "failed marker update",
    )


def test_generate_maps_create_input_error(tmp_path, data_tables_app_factory, monkeypatch):
    db_path = tmp_path / "media.db"
    app, _ = data_tables_app_factory(db_path)

    def fail_create(self, *args, **kwargs):  # noqa: ANN001, ARG001
        raise InputError("generate_validation_failed")

    monkeypatch.setattr(MediaDatabase, "create_data_table", fail_create)

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/data-tables/generate",
            json={
                "name": "Test Table",
                "prompt": "Extract data",
                "sources": [{"source_type": "chat", "source_id": "chat_1"}],
            },
        )

        assert resp.status_code == 400, resp.text
        assert resp.json()["detail"] == "generate_validation_failed"


def test_generate_maps_create_database_error(tmp_path, data_tables_app_factory, monkeypatch):
    db_path = tmp_path / "media.db"
    app, _ = data_tables_app_factory(db_path)
    logger_stub = _LoggerStub()

    def fail_create(self, *args, **kwargs):  # noqa: ANN001, ARG001
        raise DatabaseError("data table driver exploded at /private/data-tables.db")

    monkeypatch.setattr(MediaDatabase, "create_data_table", fail_create)
    monkeypatch.setattr(data_tables_endpoint, "logger", logger_stub)

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post(
            "/api/v1/data-tables/generate",
            json={
                "name": "Test Table",
                "prompt": "Extract data",
                "sources": [{"source_type": "chat", "source_id": "chat_1"}],
            },
        )

        assert resp.status_code == 500, resp.text
        assert resp.json()["detail"] == "Failed to submit data table job"
        _assert_sanitized_error_log(
            logger_stub,
            "data_tables.generate failed",
            "data table driver exploded",
        )


def test_generate_source_insert_database_error_marks_failed(tmp_path, data_tables_app_factory, monkeypatch):
    db_path = tmp_path / "media.db"
    app, _ = data_tables_app_factory(db_path)

    def fail_insert(self, *args, **kwargs):  # noqa: ANN001, ARG001
        raise DatabaseError("driver failed")

    monkeypatch.setattr(MediaDatabase, "insert_data_table_sources", fail_insert)

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post(
            "/api/v1/data-tables/generate",
            json={
                "name": "Test Table",
                "prompt": "Extract data",
                "sources": [{"source_type": "chat", "source_id": "chat_1"}],
            },
        )

        assert resp.status_code == 500, resp.text
        assert resp.json()["detail"] == "Failed to submit data table job"

    verify_db = MediaDatabase(db_path=str(db_path), client_id="verify_client")
    try:
        rows = verify_db.list_data_tables(limit=10, offset=0)
        assert len(rows) == 1
        assert rows[0]["status"] == "failed"
        assert rows[0].get("last_error") == "Data table generation failed"
    finally:
        verify_db.close_connection()


def test_generate_job_create_error_marks_failed(tmp_path, data_tables_app_factory, monkeypatch):
    db_path = tmp_path / "media.db"
    app, _ = data_tables_app_factory(db_path)
    logger_stub = _LoggerStub()

    class _StubJobManager:
        def create_job(self, **kwargs):  # noqa: ANN003
            raise RuntimeError("job manager exploded at /private/jobs.db")

    app.dependency_overrides[get_job_manager] = lambda: _StubJobManager()
    monkeypatch.setattr(data_tables_endpoint, "logger", logger_stub)

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/api/v1/data-tables/generate",
                json={
                    "name": "Test Table",
                    "prompt": "Extract data",
                    "sources": [{"source_type": "chat", "source_id": "chat_1"}],
                },
            )

            assert resp.status_code == 500, resp.text
            assert resp.json()["detail"] == "Failed to submit data table job"
            _assert_sanitized_error_log(
                logger_stub,
                "data_tables.generate failed",
                "job manager exploded",
            )
    finally:
        app.dependency_overrides.pop(get_job_manager, None)

    verify_db = MediaDatabase(db_path=str(db_path), client_id="verify_client")
    try:
        rows = verify_db.list_data_tables(limit=10, offset=0)
        assert len(rows) == 1
        assert rows[0]["status"] == "failed"
        assert rows[0].get("last_error") == "Data table generation failed"
    finally:
        verify_db.close_connection()


def test_generate_wait_maps_fetch_database_error(tmp_path, data_tables_app_factory, monkeypatch):
    db_path = tmp_path / "media.db"
    app, _ = data_tables_app_factory(db_path)

    class _StubJobManager:
        def create_job(self, **kwargs):  # noqa: ANN003
            return {"id": 106, "uuid": "job-106", "status": "queued"}

    app.dependency_overrides[get_job_manager] = lambda: _StubJobManager()

    async def completed_job(*args, **kwargs):  # noqa: ANN001, ARG001
        return {"status": "completed"}

    def fail_fetch(self, *args, **kwargs):  # noqa: ANN001, ARG001
        raise DatabaseError("driver failed")

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.data_tables._wait_for_job_completion",
        completed_job,
    )
    monkeypatch.setattr(MediaDatabase, "get_data_table_by_uuid", fail_fetch)

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/api/v1/data-tables/generate?wait_for_completion=true",
                json={
                    "name": "Test Table",
                    "prompt": "Extract data",
                    "sources": [{"source_type": "chat", "source_id": "chat_1"}],
                },
            )

            assert resp.status_code == 500, resp.text
            assert resp.json()["detail"] == "Failed to submit data table job"
    finally:
        app.dependency_overrides.pop(get_job_manager, None)


def test_generate_wait_maps_detail_database_error(tmp_path, data_tables_app_factory, monkeypatch):
    db_path = tmp_path / "media.db"
    app, _ = data_tables_app_factory(db_path)

    class _StubJobManager:
        def create_job(self, **kwargs):  # noqa: ANN003
            return {"id": 107, "uuid": "job-107", "status": "queued"}

    app.dependency_overrides[get_job_manager] = lambda: _StubJobManager()

    async def completed_job(*args, **kwargs):  # noqa: ANN001, ARG001
        return {"status": "completed"}

    def fail_detail(self, *args, **kwargs):  # noqa: ANN001, ARG001
        raise DatabaseError("driver failed")

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.data_tables._wait_for_job_completion",
        completed_job,
    )
    monkeypatch.setattr(MediaDatabase, "list_data_table_columns", fail_detail)

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/api/v1/data-tables/generate?wait_for_completion=true",
                json={
                    "name": "Test Table",
                    "prompt": "Extract data",
                    "sources": [{"source_type": "chat", "source_id": "chat_1"}],
                },
            )

            assert resp.status_code == 500, resp.text
            assert resp.json()["detail"] == "Failed to submit data table job"
    finally:
        app.dependency_overrides.pop(get_job_manager, None)


def test_generate_wait_returns_job_failure_conflict(tmp_path, data_tables_app_factory, monkeypatch):
    db_path = tmp_path / "media.db"
    app, _ = data_tables_app_factory(db_path)

    class _StubJobManager:
        def create_job(self, **kwargs):  # noqa: ANN003
            return {"id": 108, "uuid": "job-108", "status": "queued"}

    app.dependency_overrides[get_job_manager] = lambda: _StubJobManager()

    async def failed_job(*args, **kwargs):  # noqa: ANN001, ARG001
        return {"status": "failed", "error_message": "generation failed"}

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.data_tables._wait_for_job_completion",
        failed_job,
    )

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/api/v1/data-tables/generate?wait_for_completion=true",
                json={
                    "name": "Test Table",
                    "prompt": "Extract data",
                    "sources": [{"source_type": "chat", "source_id": "chat_1"}],
                },
            )

            assert resp.status_code == 409, resp.text
            assert resp.json()["detail"] == "generation failed"
    finally:
        app.dependency_overrides.pop(get_job_manager, None)


def test_generate_wait_returns_not_found_when_table_missing(tmp_path, data_tables_app_factory, monkeypatch):
    db_path = tmp_path / "media.db"
    app, _ = data_tables_app_factory(db_path)

    class _StubJobManager:
        def create_job(self, **kwargs):  # noqa: ANN003
            return {"id": 109, "uuid": "job-109", "status": "queued"}

    app.dependency_overrides[get_job_manager] = lambda: _StubJobManager()

    async def completed_job(*args, **kwargs):  # noqa: ANN001, ARG001
        return {"status": "completed"}

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.data_tables._wait_for_job_completion",
        completed_job,
    )
    monkeypatch.setattr(MediaDatabase, "get_data_table_by_uuid", lambda *args, **kwargs: None)

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/api/v1/data-tables/generate?wait_for_completion=true",
                json={
                    "name": "Test Table",
                    "prompt": "Extract data",
                    "sources": [{"source_type": "chat", "source_id": "chat_1"}],
                },
            )

            assert resp.status_code == 404, resp.text
            assert resp.json()["detail"] == "data_table_not_found"
    finally:
        app.dependency_overrides.pop(get_job_manager, None)


def test_list_update_delete_data_table(tmp_path, data_tables_app_factory):
    db_path = tmp_path / "media.db"
    seed_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
    table = seed_db.create_data_table(
        name="Seed Table",
        prompt="Seed prompt",
        description="Seed",
        status="ready",
        row_count=1,
    )
    table_id = int(table.get("id"))
    seed_db.insert_data_table_columns(
        table_id,
        [
            {"name": "Name", "type": "text", "position": 0},
            {"name": "Value", "type": "number", "position": 1},
        ],
    )
    columns = seed_db.list_data_table_columns(table_id)
    column_ids = [col.get("column_id") for col in columns]
    seed_db.insert_data_table_rows(
        table_id,
        [{"row_index": 0, "row_json": {column_ids[0]: "Alpha", column_ids[1]: 10}}],
    )
    seed_db.close_connection()

    app, _ = data_tables_app_factory(db_path)
    with TestClient(app) as client:
        resp = client.get("/api/v1/data-tables")
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["count"] >= 1
        assert payload["has_more"] is False
        assert payload["next_offset"] is None
        assert payload["pagination"] == {
            "mode": "offset",
            "limit": 50,
            "offset": 0,
            "total": payload["total"],
            "has_more": False,
            "next_offset": None,
        }

        table_uuid = table.get("uuid")
        detail = client.get(f"/api/v1/data-tables/{table_uuid}?rows_limit=1&rows_offset=0")
        assert detail.status_code == 200, detail.text
        detail_payload = detail.json()
        assert detail_payload["rows_limit"] == 1
        assert detail_payload["rows_offset"] == 0
        assert detail_payload["has_more"] is False
        assert detail_payload["next_offset"] is None
        assert detail_payload["pagination"] == {
            "mode": "offset",
            "limit": 1,
            "offset": 0,
            "total": 1,
            "has_more": False,
            "next_offset": None,
        }

        patch = client.patch(
            f"/api/v1/data-tables/{table_uuid}",
            json={"name": "Renamed Table"},
        )
        assert patch.status_code == 200, patch.text
        assert patch.json()["name"] == "Renamed Table"

        delete = client.delete(f"/api/v1/data-tables/{table_uuid}")
        assert delete.status_code == 200, delete.text
        assert delete.json()["success"] is True


def test_get_data_table_without_rows_preserves_pagination_window(tmp_path, data_tables_app_factory) -> None:
    db_path = tmp_path / "media.db"
    seed_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
    table = seed_db.create_data_table(
        name="Seed Table",
        prompt="Seed prompt",
        description="Seed",
        status="ready",
        row_count=2,
    )
    table_id = int(table.get("id"))
    seed_db.insert_data_table_columns(
        table_id,
        [{"name": "Name", "type": "text", "position": 0}],
    )
    columns = seed_db.list_data_table_columns(table_id)
    column_id = columns[0].get("column_id")
    seed_db.insert_data_table_rows(
        table_id,
        [
            {"row_index": 0, "row_json": {column_id: "Alpha"}},
            {"row_index": 1, "row_json": {column_id: "Beta"}},
        ],
    )
    seed_db.close_connection()

    app, _ = data_tables_app_factory(db_path)
    with TestClient(app) as client:
        table_uuid = table.get("uuid")
        detail = client.get(
            f"/api/v1/data-tables/{table_uuid}"
            "?include_rows=false&include_sources=false&rows_limit=1&rows_offset=0"
        )
        assert detail.status_code == 200, detail.text
        detail_payload = detail.json()
        assert detail_payload["rows"] == []
        assert detail_payload["pagination"] == {
            "mode": "offset",
            "limit": 1,
            "offset": 0,
            "total": 2,
            "has_more": True,
            "next_offset": 1,
        }
        assert detail_payload["has_more"] is True
        assert detail_payload["next_offset"] == 1


def test_list_data_tables_maps_database_error(tmp_path, data_tables_app_factory, monkeypatch):
    db_path = tmp_path / "media.db"
    app, _ = data_tables_app_factory(db_path)

    def fail_list(self, *args, **kwargs):  # noqa: ANN001, ARG001
        raise DatabaseError("driver failed")

    monkeypatch.setattr(MediaDatabase, "list_data_tables", fail_list)

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/api/v1/data-tables")

        assert resp.status_code == 500, resp.text
        assert resp.json()["detail"] == "Failed to list data tables"


def test_get_data_table_maps_database_error(tmp_path, data_tables_app_factory, monkeypatch):
    db_path = tmp_path / "media.db"
    app, _ = data_tables_app_factory(db_path)

    def fail_get(self, *args, **kwargs):  # noqa: ANN001, ARG001
        raise DatabaseError("driver failed")

    monkeypatch.setattr(MediaDatabase, "get_data_table_by_uuid", fail_get)

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/api/v1/data-tables/table-uuid")

        assert resp.status_code == 500, resp.text
        assert resp.json()["detail"] == "Failed to fetch data table"


def test_update_data_table_maps_database_error(tmp_path, data_tables_app_factory, monkeypatch):
    db_path = tmp_path / "media.db"
    seed_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
    table = seed_db.create_data_table(
        name="Seed Table",
        prompt="Seed prompt",
        description="Seed",
        status="ready",
    )
    seed_db.close_connection()

    def fail_update(self, *args, **kwargs):  # noqa: ANN001, ARG001
        raise DatabaseError("driver failed")

    monkeypatch.setattr(MediaDatabase, "update_data_table", fail_update)

    app, _ = data_tables_app_factory(db_path)
    with TestClient(app, raise_server_exceptions=False) as client:
        patch = client.patch(
            f"/api/v1/data-tables/{table.get('uuid')}",
            json={"name": "Renamed Table"},
        )

        assert patch.status_code == 500, patch.text
        assert patch.json()["detail"] == "Failed to update data table"


def test_delete_data_table_maps_database_error(tmp_path, data_tables_app_factory, monkeypatch):
    db_path = tmp_path / "media.db"
    seed_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
    table = seed_db.create_data_table(
        name="Seed Table",
        prompt="Seed prompt",
        description="Seed",
        status="ready",
    )
    seed_db.close_connection()

    def fail_delete(self, *args, **kwargs):  # noqa: ANN001, ARG001
        raise DatabaseError("driver failed")

    monkeypatch.setattr(MediaDatabase, "soft_delete_data_table", fail_delete)

    app, _ = data_tables_app_factory(db_path)
    with TestClient(app, raise_server_exceptions=False) as client:
        delete = client.delete(f"/api/v1/data-tables/{table.get('uuid')}")

        assert delete.status_code == 500, delete.text
        assert delete.json()["detail"] == "Failed to delete data table"


def test_job_status_and_cancel(tmp_path, data_tables_app_factory):
    db_path = tmp_path / "media.db"
    app, _ = data_tables_app_factory(db_path)

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/data-tables/generate",
            json={
                "name": "Job Table",
                "prompt": "Extract data",
                "sources": [{"source_type": "chat", "source_id": "chat_job"}],
            },
        )
        assert resp.status_code == 202, resp.text
        job_id = resp.json()["job_id"]

        status_resp = client.get(f"/api/v1/data-tables/jobs/{job_id}")
        assert status_resp.status_code == 200, status_resp.text
        assert status_resp.json()["id"] == job_id

        cancel_resp = client.delete(f"/api/v1/data-tables/jobs/{job_id}")
        assert cancel_resp.status_code == 200, cancel_resp.text
        assert cancel_resp.json()["success"] is True


def test_data_table_job_status_rejects_boolean_admin_without_claims(tmp_path, data_tables_app_factory):
    db_path = tmp_path / "media.db"
    app, _ = data_tables_app_factory(db_path)

    class _StubJobManager:
        def get_job(self, _job_id: int):
            return {
                "id": 9,
                "domain": "data_tables",
                "owner_user_id": "2",
                "status": "queued",
                "job_type": "data_table_generate",
            }

    async def _principal_override():
        return AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject=None,
            token_type="access",  # nosec B106 - auth principal stub token type, not a credential
            jti=None,
            roles=["user"],
            permissions=["media.read"],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )

    app.dependency_overrides[get_job_manager] = lambda: _StubJobManager()
    app.dependency_overrides[get_auth_principal] = _principal_override
    try:
        with TestClient(app) as client:
            resp = client.get("/api/v1/data-tables/jobs/9")
            assert resp.status_code == 403
    finally:
        app.dependency_overrides.pop(get_job_manager, None)
        app.dependency_overrides.pop(get_auth_principal, None)


def test_generate_uses_configured_data_tables_queue(tmp_path, data_tables_app_factory, monkeypatch):
    db_path = tmp_path / "media.db"
    monkeypatch.setenv("DATA_TABLES_JOBS_QUEUE", "data-tables-custom")
    app, _ = data_tables_app_factory(db_path)

    captured: dict[str, object] = {}

    class _StubJobManager:
        def create_job(self, **kwargs):
            captured.update(kwargs)
            return {"id": 101, "uuid": "job-101", "status": "queued"}

    app.dependency_overrides[get_job_manager] = lambda: _StubJobManager()
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/api/v1/data-tables/generate",
                json={
                    "name": "Queue Table",
                    "prompt": "Extract data",
                    "sources": [{"source_type": "chat", "source_id": "chat_queue"}],
                },
            )
            assert resp.status_code == 202, resp.text
            assert captured.get("queue") == "data-tables-custom"
    finally:
        app.dependency_overrides.pop(get_job_manager, None)


def test_regenerate_uses_configured_data_tables_queue(tmp_path, data_tables_app_factory, monkeypatch):
    db_path = tmp_path / "media.db"
    monkeypatch.setenv("DATA_TABLES_JOBS_QUEUE", "data-tables-custom")

    seed_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
    table = seed_db.create_data_table(
        name="Regen Seed",
        prompt="Regenerate me",
        status="ready",
    )
    table_id = int(table.get("id"))
    seed_db.insert_data_table_sources(
        table_id,
        [{"source_type": "chat", "source_id": "chat_source"}],
    )
    seed_db.close_connection()

    app, _ = data_tables_app_factory(db_path)
    captured: dict[str, object] = {}

    class _StubJobManager:
        def create_job(self, **kwargs):
            captured.update(kwargs)
            return {"id": 102, "uuid": "job-102", "status": "queued"}

    app.dependency_overrides[get_job_manager] = lambda: _StubJobManager()
    try:
        with TestClient(app) as client:
            resp = client.post(f"/api/v1/data-tables/{table.get('uuid')}/regenerate", json={})
            assert resp.status_code == 202, resp.text
            assert captured.get("queue") == "data-tables-custom"
    finally:
        app.dependency_overrides.pop(get_job_manager, None)


def test_regenerate_after_admin_patch_preserves_table_owner(tmp_path, data_tables_app_factory):
    db_path = tmp_path / "media.db"
    seed_db = MediaDatabase(db_path=str(db_path), client_id="seed_client")
    table = seed_db.create_data_table(
        name="Owner-Segregated Table",
        prompt="Regenerate me",
        status="ready",
        owner_user_id=77,
    )
    table_id = int(table.get("id"))
    seed_db.insert_data_table_sources(
        table_id,
        [{"source_type": "chat", "source_id": "chat_owner_77"}],
        owner_user_id=77,
    )
    seed_db.close_connection()

    app, _ = data_tables_app_factory(db_path)
    captured: dict[str, object] = {}

    class _StubJobManager:
        def create_job(self, **kwargs):
            captured.update(kwargs)
            return {"id": 103, "uuid": "job-103", "status": "queued"}

    app.dependency_overrides[get_job_manager] = lambda: _StubJobManager()
    try:
        with TestClient(app) as client:
            patch = client.patch(
                f"/api/v1/data-tables/{table.get('uuid')}",
                json={"name": "Renamed by admin"},
            )
            assert patch.status_code == 200, patch.text

            regen = client.post(f"/api/v1/data-tables/{table.get('uuid')}/regenerate", json={})
            assert regen.status_code == 202, regen.text
            payload = captured.get("payload")
            assert isinstance(payload, dict)
            assert payload.get("user_id") == "77"
    finally:
        app.dependency_overrides.pop(get_job_manager, None)

    verify_db = MediaDatabase(db_path=str(db_path), client_id="verify_client")
    try:
        assert verify_db.get_data_table(table_id, owner_user_id=77) is not None
        assert verify_db.get_data_table(table_id, owner_user_id="verify_client") is None
    finally:
        verify_db.close_connection()


def test_regenerate_maps_table_fetch_database_error(tmp_path, data_tables_app_factory, monkeypatch):
    db_path = tmp_path / "media.db"
    app, _ = data_tables_app_factory(db_path)

    def fail_fetch(self, *args, **kwargs):  # noqa: ANN001, ARG001
        raise DatabaseError("driver failed")

    monkeypatch.setattr(MediaDatabase, "get_data_table_by_uuid", fail_fetch)

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post("/api/v1/data-tables/table-uuid/regenerate", json={})

        assert resp.status_code == 500, resp.text
        assert resp.json()["detail"] == "Failed to regenerate data table"


def test_regenerate_maps_sources_database_error(tmp_path, data_tables_app_factory, monkeypatch):
    db_path = tmp_path / "media.db"
    seed_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
    table = seed_db.create_data_table(
        name="Seed Table",
        prompt="Seed prompt",
        description="Seed",
        status="ready",
    )
    seed_db.close_connection()

    def fail_sources(self, *args, **kwargs):  # noqa: ANN001, ARG001
        raise DatabaseError("driver failed")

    monkeypatch.setattr(MediaDatabase, "list_data_table_sources", fail_sources)

    app, _ = data_tables_app_factory(db_path)
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post(f"/api/v1/data-tables/{table.get('uuid')}/regenerate", json={})

        assert resp.status_code == 500, resp.text
        assert resp.json()["detail"] == "Failed to regenerate data table"


def test_regenerate_maps_counts_database_error(tmp_path, data_tables_app_factory, monkeypatch):
    db_path = tmp_path / "media.db"
    seed_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
    table = seed_db.create_data_table(
        name="Seed Table",
        prompt="Seed prompt",
        description="Seed",
        status="ready",
    )
    table_id = int(table.get("id"))
    seed_db.insert_data_table_sources(
        table_id,
        [{"source_type": "chat", "source_id": "chat_source"}],
    )
    seed_db.close_connection()

    def fail_counts(self, *args, **kwargs):  # noqa: ANN001, ARG001
        raise DatabaseError("driver failed")

    monkeypatch.setattr(MediaDatabase, "get_data_table_counts", fail_counts)

    app, _ = data_tables_app_factory(db_path)
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post(f"/api/v1/data-tables/{table.get('uuid')}/regenerate", json={})

        assert resp.status_code == 500, resp.text
        assert resp.json()["detail"] == "Failed to regenerate data table"


def test_regenerate_maps_update_database_error(tmp_path, data_tables_app_factory, monkeypatch):
    db_path = tmp_path / "media.db"
    seed_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
    table = seed_db.create_data_table(
        name="Seed Table",
        prompt="Seed prompt",
        description="Seed",
        status="ready",
    )
    table_id = int(table.get("id"))
    seed_db.insert_data_table_sources(
        table_id,
        [{"source_type": "chat", "source_id": "chat_source"}],
    )
    seed_db.close_connection()

    def fail_update(self, *args, **kwargs):  # noqa: ANN001, ARG001
        raise DatabaseError("driver failed")

    monkeypatch.setattr(MediaDatabase, "update_data_table", fail_update)

    app, _ = data_tables_app_factory(db_path)
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post(f"/api/v1/data-tables/{table.get('uuid')}/regenerate", json={})

        assert resp.status_code == 500, resp.text
        assert resp.json()["detail"] == "Failed to regenerate data table"


def test_regenerate_maps_response_table_database_error(tmp_path, data_tables_app_factory, monkeypatch):
    db_path = tmp_path / "media.db"
    seed_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
    table = seed_db.create_data_table(
        name="Seed Table",
        prompt="Seed prompt",
        description="Seed",
        status="ready",
    )
    table_id = int(table.get("id"))
    seed_db.insert_data_table_sources(
        table_id,
        [{"source_type": "chat", "source_id": "chat_source"}],
    )
    seed_db.close_connection()

    def fail_table(self, *args, **kwargs):  # noqa: ANN001, ARG001
        raise DatabaseError("driver failed")

    def noop_update(self, *args, **kwargs):  # noqa: ANN001, ARG001
        return None

    monkeypatch.setattr(MediaDatabase, "update_data_table", noop_update)
    monkeypatch.setattr(MediaDatabase, "get_data_table", fail_table)

    app, _ = data_tables_app_factory(db_path)
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post(f"/api/v1/data-tables/{table.get('uuid')}/regenerate", json={})

        assert resp.status_code == 500, resp.text
        assert resp.json()["detail"] == "Failed to regenerate data table"


def test_regenerate_wait_maps_fetch_database_error(tmp_path, data_tables_app_factory, monkeypatch):
    db_path = tmp_path / "media.db"
    seed_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
    table = seed_db.create_data_table(
        name="Seed Table",
        prompt="Seed prompt",
        description="Seed",
        status="ready",
    )
    table_id = int(table.get("id"))
    seed_db.insert_data_table_sources(
        table_id,
        [{"source_type": "chat", "source_id": "chat_source"}],
    )
    seed_db.close_connection()

    app, _ = data_tables_app_factory(db_path)

    class _StubJobManager:
        def create_job(self, **kwargs):  # noqa: ANN003
            return {"id": 104, "uuid": "job-104", "status": "queued"}

    app.dependency_overrides[get_job_manager] = lambda: _StubJobManager()

    async def completed_job(*args, **kwargs):  # noqa: ANN001, ARG001
        return {"status": "completed"}

    original_get_by_uuid = MediaDatabase.get_data_table_by_uuid
    call_count = {"value": 0}

    def fail_second_fetch(self, *args, **kwargs):  # noqa: ANN001
        call_count["value"] += 1
        if call_count["value"] == 1:
            return original_get_by_uuid(self, *args, **kwargs)
        raise DatabaseError("driver failed")

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.data_tables._wait_for_job_completion",
        completed_job,
    )
    monkeypatch.setattr(MediaDatabase, "get_data_table_by_uuid", fail_second_fetch)

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                f"/api/v1/data-tables/{table.get('uuid')}/regenerate?wait_for_completion=true",
                json={},
            )

            assert resp.status_code == 500, resp.text
            assert resp.json()["detail"] == "Failed to regenerate data table"
    finally:
        app.dependency_overrides.pop(get_job_manager, None)


def test_regenerate_wait_maps_detail_database_error(tmp_path, data_tables_app_factory, monkeypatch):
    db_path = tmp_path / "media.db"
    seed_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
    table = seed_db.create_data_table(
        name="Seed Table",
        prompt="Seed prompt",
        description="Seed",
        status="ready",
    )
    table_id = int(table.get("id"))
    seed_db.insert_data_table_sources(
        table_id,
        [{"source_type": "chat", "source_id": "chat_source"}],
    )
    seed_db.close_connection()

    app, _ = data_tables_app_factory(db_path)

    class _StubJobManager:
        def create_job(self, **kwargs):  # noqa: ANN003
            return {"id": 105, "uuid": "job-105", "status": "queued"}

    app.dependency_overrides[get_job_manager] = lambda: _StubJobManager()

    async def completed_job(*args, **kwargs):  # noqa: ANN001, ARG001
        return {"status": "completed"}

    def fail_detail(self, *args, **kwargs):  # noqa: ANN001, ARG001
        raise DatabaseError("driver failed")

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.data_tables._wait_for_job_completion",
        completed_job,
    )
    monkeypatch.setattr(MediaDatabase, "list_data_table_columns", fail_detail)

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                f"/api/v1/data-tables/{table.get('uuid')}/regenerate?wait_for_completion=true",
                json={},
            )

            assert resp.status_code == 500, resp.text
            assert resp.json()["detail"] == "Failed to regenerate data table"
    finally:
        app.dependency_overrides.pop(get_job_manager, None)


def test_update_content_rejects_duplicate_row_indexes(tmp_path, data_tables_app_factory):
    db_path = tmp_path / "media.db"
    seed_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
    table = seed_db.create_data_table(name="Edit Table", prompt="p", status="ready")
    seed_db.close_connection()

    app, _ = data_tables_app_factory(db_path)
    with TestClient(app) as client:
        resp = client.put(
            f"/api/v1/data-tables/{table.get('uuid')}/content",
            json={
                "columns": [
                    {"name": "Name", "type": "text"},
                    {"name": "Score", "type": "number"},
                ],
                "rows": [
                    {"row_index": 0, "data": {"Name": "Alice", "Score": 95}},
                    {"row_index": 0, "data": {"Name": "Bob", "Score": 88}},
                ],
            },
        )
        assert resp.status_code == 400, resp.text
        assert resp.json()["detail"] == "duplicate_row_index"


def test_update_content_persists_rows_and_columns(tmp_path, data_tables_app_factory):
    db_path = tmp_path / "media.db"
    seed_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
    table = seed_db.create_data_table(name="Edit Table", prompt="p", status="ready")
    seed_db.close_connection()

    app, _ = data_tables_app_factory(db_path)
    with TestClient(app) as client:
        resp = client.put(
            f"/api/v1/data-tables/{table.get('uuid')}/content",
            json={
                "columns": [
                    {"name": "Name", "type": "text"},
                    {"name": "Score", "type": "number"},
                ],
                "rows": [
                    {"row_index": 0, "data": {"Name": "Alice", "Score": 95}},
                    {"row_index": 1, "data": {"Name": "Bob", "Score": 88}},
                ],
            },
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["table"]["uuid"] == table.get("uuid")
        assert body["table"]["status"] == "ready"
        assert body["table"]["row_count"] == 2
        assert [column["name"] for column in body["columns"]] == ["Name", "Score"]
        assert len(body["rows"]) == 2
        assert {row["row_index"] for row in body["rows"]} == {0, 1}
        assert all(len(row["data"]) == 2 for row in body["rows"])


def test_update_content_maps_persistence_input_error(tmp_path, data_tables_app_factory, monkeypatch):
    db_path = tmp_path / "media.db"
    seed_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
    table = seed_db.create_data_table(name="Edit Table", prompt="p", status="ready")
    seed_db.close_connection()

    def fail_persist(self, *args, **kwargs):  # noqa: ANN001, ARG001
        raise InputError("persist_validation_failed")

    monkeypatch.setattr(MediaDatabase, "persist_data_table_generation", fail_persist)

    app, _ = data_tables_app_factory(db_path)
    with TestClient(app) as client:
        resp = client.put(
            f"/api/v1/data-tables/{table.get('uuid')}/content",
            json={
                "columns": [{"name": "Name", "type": "text"}],
                "rows": [{"row_index": 0, "data": {"Name": "Alice"}}],
            },
        )

        assert resp.status_code == 400, resp.text
        assert resp.json()["detail"] == "persist_validation_failed"


def test_update_content_maps_persistence_database_error(tmp_path, data_tables_app_factory, monkeypatch):
    db_path = tmp_path / "media.db"
    seed_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
    table = seed_db.create_data_table(name="Edit Table", prompt="p", status="ready")
    seed_db.close_connection()

    def fail_persist(self, *args, **kwargs):  # noqa: ANN001, ARG001
        raise DatabaseError("driver failed")

    monkeypatch.setattr(MediaDatabase, "persist_data_table_generation", fail_persist)

    app, _ = data_tables_app_factory(db_path)
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.put(
            f"/api/v1/data-tables/{table.get('uuid')}/content",
            json={
                "columns": [{"name": "Name", "type": "text"}],
                "rows": [{"row_index": 0, "data": {"Name": "Alice"}}],
            },
        )

        assert resp.status_code == 500, resp.text
        assert resp.json()["detail"] == "Failed to update data table content"


def test_update_content_maps_readback_database_error(tmp_path, data_tables_app_factory, monkeypatch):
    db_path = tmp_path / "media.db"
    seed_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
    table = seed_db.create_data_table(name="Edit Table", prompt="p", status="ready")
    seed_db.close_connection()

    def fail_readback(self, *args, **kwargs):  # noqa: ANN001, ARG001
        raise DatabaseError("driver failed")

    monkeypatch.setattr(MediaDatabase, "get_data_table", fail_readback)

    app, _ = data_tables_app_factory(db_path)
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.put(
            f"/api/v1/data-tables/{table.get('uuid')}/content",
            json={
                "columns": [{"name": "Name", "type": "text"}],
                "rows": [{"row_index": 0, "data": {"Name": "Alice"}}],
            },
        )

        assert resp.status_code == 500, resp.text
        assert resp.json()["detail"] == "Failed to update data table content"


def test_update_content_maps_detail_database_error(tmp_path, data_tables_app_factory, monkeypatch):
    db_path = tmp_path / "media.db"
    seed_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
    table = seed_db.create_data_table(name="Edit Table", prompt="p", status="ready")
    seed_db.close_connection()

    def fail_detail(self, *args, **kwargs):  # noqa: ANN001, ARG001
        raise DatabaseError("driver failed")

    monkeypatch.setattr(MediaDatabase, "list_data_table_columns", fail_detail)

    app, _ = data_tables_app_factory(db_path)
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.put(
            f"/api/v1/data-tables/{table.get('uuid')}/content",
            json={
                "columns": [{"name": "Name", "type": "text"}],
                "rows": [{"row_index": 0, "data": {"Name": "Alice"}}],
            },
        )

        assert resp.status_code == 500, resp.text
        assert resp.json()["detail"] == "Failed to update data table content"


def test_generate_rejects_max_rows_above_limit(tmp_path, data_tables_app_factory):
    db_path = tmp_path / "media.db"
    app, _ = data_tables_app_factory(db_path)

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/data-tables/generate",
            json={
                "name": "Too Many Rows",
                "prompt": "Extract data",
                "sources": [{"source_type": "chat", "source_id": "chat_1"}],
                "max_rows": DATA_TABLES_MAX_ROWS_LIMIT + 1,
            },
        )
        assert resp.status_code == 422, resp.text


def test_wait_for_completion_treats_quarantined_as_terminal():
    class _StubJobManager:
        def get_job(self, _job_id: int):
            return {"id": 99, "status": "quarantined"}

    job = asyncio.run(
        _wait_for_job_completion(
            _StubJobManager(),
            99,
            timeout_seconds=1,
            poll_interval=0.0,
        )
    )
    assert job["status"] == "quarantined"
