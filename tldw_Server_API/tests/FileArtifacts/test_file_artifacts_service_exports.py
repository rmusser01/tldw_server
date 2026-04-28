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
from tldw_Server_API.app.core.exceptions import FileArtifactsError


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
