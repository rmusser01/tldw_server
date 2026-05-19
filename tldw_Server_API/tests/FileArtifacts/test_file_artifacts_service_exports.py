import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tldw_Server_API.app.core.File_Artifacts import file_artifacts_service as service_mod
from tldw_Server_API.app.api.v1.schemas.file_artifacts_schemas import (
    FileCreateOptions,
    FileCreateRequest,
    FileExportRequest,
)
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.File_Artifacts.adapters.base import ExportResult
from tldw_Server_API.app.core.File_Artifacts.file_artifacts_service import FileArtifactsService
from tldw_Server_API.app.core.exceptions import FileArtifactsError, FileArtifactsJobError


pytestmark = pytest.mark.unit


@pytest.fixture()
def collections_db(monkeypatch: pytest.MonkeyPatch) -> CollectionsDatabase:
    base_dir = Path.cwd() / "Databases" / "test_user_dbs_file_artifacts_service"
    shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    prev_base_dir = settings.get("USER_DB_BASE_DIR")
    settings.USER_DB_BASE_DIR = str(base_dir)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))

    try:
        yield CollectionsDatabase.for_user(user_id=777)
    finally:
        if prev_base_dir is not None:
            settings.USER_DB_BASE_DIR = prev_base_dir
        else:
            try:
                del settings.USER_DB_BASE_DIR
            except AttributeError:
                pass


def _count_file_artifacts(cdb: CollectionsDatabase) -> int:
    res = cdb.backend.execute(
        "SELECT COUNT(*) AS count FROM file_artifacts WHERE user_id = ?",
        (cdb.user_id,),
    )
    return int(res.rows[0]["count"]) if res.rows else 0


@pytest.mark.asyncio
async def test_export_failure_rolls_back_artifact(collections_db: CollectionsDatabase, monkeypatch: pytest.MonkeyPatch) -> None:
    service = FileArtifactsService(collections_db, user_id=collections_db.user_id)
    request = FileCreateRequest(
        file_type="data_table",
        payload={"columns": ["Name"], "rows": [["Ada"]]},
        export=FileExportRequest(format="csv", mode="url", async_mode="sync"),
        options=FileCreateOptions(persist=True),
    )

    async def _boom(*_args, **_kwargs) -> None:
        """Force export to fail for rollback coverage."""
        raise FileArtifactsError("export_failed")

    monkeypatch.setattr(service, "_export_sync", _boom)

    with pytest.raises(FileArtifactsError, match="export_failed"):
        await service.create_artifact(request)

    assert _count_file_artifacts(collections_db) == 0


@pytest.mark.asyncio
async def test_inline_export_skips_generated_file_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    cdb = MagicMock()
    cdb.update_file_artifact_export = MagicMock()
    service = FileArtifactsService(cdb, user_id=1)
    register_mock = AsyncMock()
    monkeypatch.setattr(service, "_register_generated_file_export", register_mock)

    export_req = FileExportRequest(format="csv", mode="inline", async_mode="sync")
    export_result = ExportResult(status="ready", content_type="text/csv", content=b"hello")
    options = FileCreateOptions(persist=True)

    await service._finalize_export(
        file_id=1,
        export_req=export_req,
        export_result=export_result,
        options=options,
        file_type="data_table",
    )

    assert register_mock.await_count == 0


def test_delete_temp_export_file_failure_log_omits_raw_path_and_error(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_mock = MagicMock()
    monkeypatch.setattr(service_mod, "logger", logger_mock)
    cdb = MagicMock()
    cdb.resolve_temp_output_storage_path.side_effect = RuntimeError("delete leaked /private/delete-token")
    service = FileArtifactsService(cdb, user_id=1)

    service._delete_temp_export_file("tenant/path/raw-delete-token.csv")

    logger_mock.warning.assert_called_once_with(
        "file_artifacts: failed to delete export file error_type={}",
        "RuntimeError",
    )
    rendered_log_call = str(logger_mock.warning.call_args)
    assert "raw-delete-token" not in rendered_log_call
    assert "/private/delete-token" not in rendered_log_call


def test_rollback_failure_log_omits_raw_file_id_and_error(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_mock = MagicMock()
    monkeypatch.setattr(service_mod, "logger", logger_mock)
    cdb = MagicMock()
    cdb.get_file_artifact.return_value = SimpleNamespace(export_storage_path="existing-export.csv")
    cdb.delete_file_artifact.side_effect = RuntimeError("rollback leaked /private/rollback-token")
    service = FileArtifactsService(cdb, user_id=1)
    delete_mock = MagicMock()
    monkeypatch.setattr(service, "_delete_temp_export_file", delete_mock)

    service._rollback_artifact(987654321)

    delete_mock.assert_called_once_with("existing-export.csv")
    cdb.delete_file_artifact.assert_called_once_with(987654321, hard=True)
    logger_mock.warning.assert_called_once_with(
        "file_artifacts: failed to rollback file artifact error_type={}",
        "RuntimeError",
    )
    rendered_log_call = str(logger_mock.warning.call_args)
    assert "987654321" not in rendered_log_call
    assert "/private/rollback-token" not in rendered_log_call


@pytest.mark.asyncio
async def test_enqueue_export_job_failure_log_omits_raw_ids_and_error(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_mock = MagicMock()
    monkeypatch.setattr(service_mod, "logger", logger_mock)
    cdb = MagicMock()
    job_manager = MagicMock()
    job_manager.create_job.side_effect = RuntimeError("enqueue leaked /private/raw-enqueue-token")
    service = FileArtifactsService(cdb, user_id=1, job_manager=job_manager)
    adapter = SimpleNamespace(file_type="data_table", export_formats={"csv"})

    with pytest.raises(FileArtifactsError, match="export_job_enqueue_failed"):
        await service._handle_export(
            adapter=adapter,
            structured={"columns": ["Name"], "rows": [["Ada"]]},
            file_id=24681357,
            export_req=FileExportRequest(format="csv", mode="url", async_mode="async"),
            options=FileCreateOptions(persist=True),
            request_id="request-raw-token",
        )

    logger_mock.error.assert_called_once_with(
        "file_artifacts: failed to enqueue export job error_type={}",
        "RuntimeError",
    )
    rendered_log_call = str(logger_mock.error.call_args)
    assert "24681357" not in rendered_log_call
    assert "request-raw-token" not in rendered_log_call
    assert "/private/raw-enqueue-token" not in rendered_log_call


@pytest.mark.asyncio
async def test_reset_export_state_failure_log_omits_raw_file_id_and_error(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_mock = MagicMock()
    monkeypatch.setattr(service_mod, "logger", logger_mock)
    cdb = MagicMock()
    cdb.update_file_artifact_export.side_effect = [
        None,
        RuntimeError("reset leaked /private/raw-reset-token"),
    ]
    service = FileArtifactsService(cdb, user_id=1)
    monkeypatch.setattr(
        service,
        "_write_export_file",
        AsyncMock(return_value=("tenant/path/raw-storage-token.csv", 5)),
    )
    monkeypatch.setattr(service, "_delete_temp_export_file", MagicMock())
    monkeypatch.setattr(
        service,
        "_register_generated_file_export",
        AsyncMock(side_effect=RuntimeError("register leaked /private/raw-register-token")),
    )

    with pytest.raises(RuntimeError, match="raw-register-token"):
        await service._finalize_export(
            file_id=135792468,
            export_req=FileExportRequest(format="csv", mode="url", async_mode="sync"),
            export_result=ExportResult(status="ready", content_type="text/csv", content=b"hello"),
            options=FileCreateOptions(persist=True),
            file_type="data_table",
        )

    logger_mock.warning.assert_called_once_with(
        "file_artifacts: failed to reset export state error_type={}",
        "RuntimeError",
    )
    rendered_log_call = str(logger_mock.warning.call_args)
    assert "135792468" not in rendered_log_call
    assert "/private/raw-reset-token" not in rendered_log_call


def test_validation_failure_log_omits_raw_request_id_and_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_mock = MagicMock()
    monkeypatch.setattr(service_mod, "logger", logger_mock)

    FileArtifactsService._log_validation_failure(
        "validation-request-token",
        "data_table",
        {"path": "/private/raw-validation-token"},
    )

    logger_mock.warning.assert_called_once_with(
        "file_artifacts.create validation failed file_type={} detail_type={}",
        "data_table",
        "dict",
    )
    rendered_log_call = str(logger_mock.warning.call_args)
    assert "validation-request-token" not in rendered_log_call
    assert "/private/raw-validation-token" not in rendered_log_call


def test_export_failure_log_omits_raw_request_id_and_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_mock = MagicMock()
    monkeypatch.setattr(service_mod, "logger", logger_mock)

    FileArtifactsService._log_export_failure(
        "export-request-token",
        "data_table",
        "csv",
        RuntimeError("export leaked /private/raw-export-token"),
    )

    logger_mock.warning.assert_called_once_with(
        "file_artifacts.export failed file_type={} format={} detail_type={}",
        "data_table",
        "csv",
        "RuntimeError",
    )
    rendered_log_call = str(logger_mock.warning.call_args)
    assert "export-request-token" not in rendered_log_call
    assert "/private/raw-export-token" not in rendered_log_call


@pytest.mark.asyncio
async def test_jobs_worker_failure_logs_omit_raw_file_id_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.File_Artifacts import jobs_worker as worker_mod

    logger_mock = MagicMock()
    monkeypatch.setattr(worker_mod, "logger", logger_mock)
    row = SimpleNamespace(
        export_status="pending",
        export_storage_path=None,
        export_format="csv",
        export_bytes=None,
        export_content_type=None,
        export_job_id="job-raw-token",
        file_type="data_table",
        structured_json=json.dumps({"columns": ["Name"], "rows": [["Ada"]]}),
    )
    cdb = MagicMock()
    cdb.get_file_artifact.return_value = row
    cdb.update_file_artifact_export.side_effect = RuntimeError("worker reset leaked /private/raw-worker-reset-token")

    class _CdbContext:
        def __enter__(self):
            return cdb

        def __exit__(self, *_args):
            return False

    class _Service:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_adapter(self, file_type: str):
            return SimpleNamespace(file_type=file_type, export_formats={"csv"})

        async def export_artifact_for_job(self, **_kwargs):
            raise RuntimeError("worker export leaked /private/raw-worker-export-token")

    monkeypatch.setattr(worker_mod.CollectionsDatabase, "for_user", lambda **_kwargs: _CdbContext())
    monkeypatch.setattr(worker_mod, "FileArtifactsService", _Service)

    with pytest.raises(FileArtifactsJobError):
        await worker_mod._handle_export_job(
            {
                "job_type": "file_artifact_export",
                "owner_user_id": "1",
                "payload": {
                    "file_id": 975318642,
                    "export_format": "csv",
                    "user_id": "1",
                },
            }
        )

    logger_mock.error.assert_called_once_with(
        "file_artifacts worker: export failed error_type={}",
        "RuntimeError",
    )
    logger_mock.warning.assert_called_once_with(
        "file_artifacts worker: failed to reset export status error_type={}",
        "RuntimeError",
    )
    rendered_log_calls = str(logger_mock.error.call_args_list) + str(logger_mock.warning.call_args_list)
    assert "975318642" not in rendered_log_calls
    assert "/private/raw-worker-export-token" not in rendered_log_calls
    assert "/private/raw-worker-reset-token" not in rendered_log_calls
