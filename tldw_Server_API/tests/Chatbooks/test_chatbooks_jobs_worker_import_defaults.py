from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Chatbooks.chatbook_models import ImportStatus
from tldw_Server_API.app.core.Chatbooks.services import jobs_worker


@pytest.mark.asyncio
async def test_handle_import_defaults_import_media_to_false(tmp_path):
    archive_path = tmp_path / "input.chatbook"
    archive_path.write_text("dummy", encoding="utf-8")

    class FakeService:
        def __init__(self):
            self.import_job = SimpleNamespace(
                job_id="job-1",
                status=ImportStatus.PENDING,
                completed_at=None,
                error_message=None,
            )
            self.called_args = None

        def _claim_import_job(self, _job_id: str) -> bool:
            return True

        def _get_import_job(self, _job_id: str):
            return self.import_job

        def _save_import_job(self, ij):
            self.import_job = ij

        def _resolve_import_archive_path(self, _file_ref: str) -> Path:
            return archive_path

        def _import_chatbook_sync(
            self,
            file_path,
            selections,
            conflict_resolution,
            prefix_imported,
            import_media,
            import_embeddings,
        ):
            self.called_args = {
                "file_path": file_path,
                "selections": selections,
                "conflict_resolution": conflict_resolution,
                "prefix_imported": prefix_imported,
                "import_media": import_media,
                "import_embeddings": import_embeddings,
            }
            return True, "ok", {"imported_items": {"note": 1}, "warnings": []}

    service = FakeService()

    result = await jobs_worker._handle_import(
        service,
        payload={"file_token": str(archive_path)},
        job_id="job-1",
    )

    assert result == {"imported_items": {"note": 1}, "warnings": []}
    assert service.called_args is not None
    assert service.called_args["import_media"] is False
    assert service.import_job.status == ImportStatus.COMPLETED
    assert isinstance(service.import_job.completed_at, datetime)
    assert not archive_path.exists()


@pytest.mark.asyncio
async def test_handle_import_dispatches_openwebui_json_without_archive_import(tmp_path):
    json_path = tmp_path / "openwebui.json"
    json_path.write_text("[]", encoding="utf-8")

    class FakeService:
        def __init__(self):
            self.import_job = SimpleNamespace(
                job_id="job-openwebui",
                status=ImportStatus.PENDING,
                completed_at=None,
                error_message=None,
            )
            self.openwebui_called = False
            self.archive_called = False

        def _claim_import_job(self, _job_id: str) -> bool:
            return True

        def _get_import_job(self, _job_id: str):
            return self.import_job

        def _save_import_job(self, ij):
            self.import_job = ij

        def _resolve_import_upload_path(self, _file_ref: str) -> Path:
            return json_path

        def _resolve_import_archive_path(self, _file_ref: str) -> Path:
            raise AssertionError("OpenWebUI JSON imports must not use archive path resolution")

        def _import_chatbook_sync(self, *args, **kwargs):
            self.archive_called = True
            raise AssertionError("OpenWebUI JSON imports must not call the archive importer")

        def import_openwebui_json(
            self,
            file_path,
            conflict_resolution,
            prefix_imported,
        ):
            self.openwebui_called = True
            assert file_path == str(json_path)
            assert conflict_resolution.value == "skip"
            assert prefix_imported is False
            return True, "ok", {
                "imported_chats": 1,
                "skipped_chats": 0,
                "failed_chats": 0,
                "imported_messages": 2,
                "skipped_messages": 0,
                "duplicate_chats": 0,
                "warnings": [],
            }

    service = FakeService()

    result = await jobs_worker._handle_import(
        service,
        payload={"file_token": str(json_path), "source_format": "openwebui_json"},
        job_id="job-openwebui",
    )

    assert result == {
        "openwebui_result": {
            "imported_chats": 1,
            "skipped_chats": 0,
            "failed_chats": 0,
            "imported_messages": 2,
            "skipped_messages": 0,
            "duplicate_chats": 0,
            "warnings": [],
        }
    }
    assert service.openwebui_called is True
    assert service.archive_called is False
    assert service.import_job.status == ImportStatus.COMPLETED
    assert not json_path.exists()


@pytest.mark.asyncio
async def test_handle_import_dispatches_openwebui_db_with_selected_user_without_archive_import(tmp_path):
    db_path = tmp_path / "webui.db"
    db_path.write_bytes(b"SQLite format 3\x00")

    class FakeService:
        def __init__(self):
            self.import_job = SimpleNamespace(
                job_id="job-openwebui-db",
                status=ImportStatus.PENDING,
                completed_at=None,
                error_message=None,
            )
            self.openwebui_db_called = False
            self.archive_called = False

        def _claim_import_job(self, _job_id: str) -> bool:
            return True

        def _get_import_job(self, _job_id: str):
            return self.import_job

        def _save_import_job(self, ij):
            self.import_job = ij

        def _resolve_import_upload_path(self, _file_ref: str) -> Path:
            return db_path

        def _resolve_import_archive_path(self, _file_ref: str) -> Path:
            raise AssertionError("OpenWebUI DB imports must not use archive path resolution")

        def _import_chatbook_sync(self, *args, **kwargs):
            self.archive_called = True
            raise AssertionError("OpenWebUI DB imports must not call the archive importer")

        def import_openwebui_db(
            self,
            file_path,
            *,
            selected_user_id,
            conflict_resolution,
            prefix_imported,
        ):
            self.openwebui_db_called = True
            assert file_path == str(db_path)
            assert selected_user_id == "user-a"
            assert conflict_resolution.value == "skip"
            assert prefix_imported is True
            return True, "ok", {
                "selected_user_id": "user-a",
                "selected_user_label": "Alice",
                "imported_chats": 1,
                "skipped_chats": 0,
                "failed_chats": 0,
                "imported_messages": 3,
                "skipped_messages": 0,
                "duplicate_chats": 0,
                "mirrored_folders": 1,
                "folder_links": 1,
                "warnings": [],
            }

    service = FakeService()

    result = await jobs_worker._handle_import(
        service,
        payload={
            "file_token": str(db_path),
            "source_format": "openwebui_db",
            "selected_openwebui_user_id": "user-a",
            "prefix_imported": True,
        },
        job_id="job-openwebui-db",
    )

    assert result == {
        "openwebui_db_result": {
            "selected_user_id": "user-a",
            "selected_user_label": "Alice",
            "imported_chats": 1,
            "skipped_chats": 0,
            "failed_chats": 0,
            "imported_messages": 3,
            "skipped_messages": 0,
            "duplicate_chats": 0,
            "mirrored_folders": 1,
            "folder_links": 1,
            "warnings": [],
        }
    }
    assert service.openwebui_db_called is True
    assert service.archive_called is False
    assert service.import_job.status == ImportStatus.COMPLETED
    assert not db_path.exists()


@pytest.mark.asyncio
async def test_handle_import_requires_openwebui_db_selected_user(tmp_path):
    db_path = tmp_path / "webui.db"
    db_path.write_bytes(b"SQLite format 3\x00")

    class FakeService:
        def __init__(self):
            self.import_job = SimpleNamespace(
                job_id="job-openwebui-db-missing-user",
                status=ImportStatus.PENDING,
                completed_at=None,
                error_message=None,
            )

        def _claim_import_job(self, _job_id: str) -> bool:
            return True

        def _get_import_job(self, _job_id: str):
            return self.import_job

        def _save_import_job(self, ij):
            self.import_job = ij

    service = FakeService()

    with pytest.raises(jobs_worker.ChatbooksJobError, match="selected_openwebui_user_id is required"):
        await jobs_worker._handle_import(
            service,
            payload={"file_token": str(db_path), "source_format": "openwebui_db"},
            job_id="job-openwebui-db-missing-user",
        )

    assert service.import_job.status == ImportStatus.FAILED
    assert service.import_job.error_message == "selected_openwebui_user_id is required for OpenWebUI DB imports"


@pytest.mark.asyncio
async def test_handle_import_cleans_openwebui_db_file_when_import_fails(tmp_path):
    db_path = tmp_path / "webui.db"
    db_path.write_bytes(b"SQLite format 3\x00")

    class FakeService:
        def __init__(self):
            self.import_job = SimpleNamespace(
                job_id="job-openwebui-db-fails",
                status=ImportStatus.PENDING,
                completed_at=None,
                error_message=None,
            )

        def _claim_import_job(self, _job_id: str) -> bool:
            return True

        def _get_import_job(self, _job_id: str):
            return self.import_job

        def _save_import_job(self, ij):
            self.import_job = ij

        def _resolve_import_upload_path(self, _file_ref: str) -> Path:
            return db_path

        def _resolve_import_archive_path(self, _file_ref: str) -> Path:
            raise AssertionError("OpenWebUI DB imports must not use archive path resolution")

        def import_openwebui_db(
            self,
            file_path,
            *,
            selected_user_id,
            conflict_resolution,
            prefix_imported,
        ):
            assert file_path == str(db_path)
            assert selected_user_id == "user-a"
            return False, "Invalid OpenWebUI SQLite database", None

    service = FakeService()

    with pytest.raises(jobs_worker.ChatbooksJobError, match="Invalid OpenWebUI SQLite database"):
        await jobs_worker._handle_import(
            service,
            payload={
                "file_token": str(db_path),
                "source_format": "openwebui_db",
                "selected_openwebui_user_id": "user-a",
            },
            job_id="job-openwebui-db-fails",
        )

    assert service.import_job.status == ImportStatus.FAILED
    assert service.import_job.error_message == "Invalid OpenWebUI SQLite database"
    assert not db_path.exists()


@pytest.mark.asyncio
async def test_handle_import_marks_claimed_validation_failure_as_failed():
    class FakeService:
        def __init__(self):
            self.import_job = SimpleNamespace(
                job_id="job-missing-file",
                status=ImportStatus.PENDING,
                completed_at=None,
                error_message=None,
            )

        def _claim_import_job(self, _job_id: str) -> bool:
            self.import_job.status = ImportStatus.IN_PROGRESS
            return True

        def _get_import_job(self, _job_id: str):
            return self.import_job

        def _save_import_job(self, ij):
            self.import_job = ij

    service = FakeService()

    with pytest.raises(jobs_worker.ChatbooksJobError, match="Missing file reference"):
        await jobs_worker._handle_import(service, payload={}, job_id="job-missing-file")

    assert service.import_job.status == ImportStatus.FAILED
    assert service.import_job.error_message == "Missing file reference for import job"
    assert isinstance(service.import_job.completed_at, datetime)
