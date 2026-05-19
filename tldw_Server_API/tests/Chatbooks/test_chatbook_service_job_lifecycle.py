import json
from datetime import datetime
from unittest.mock import patch
from uuid import uuid4

from tldw_Server_API.app.core.Chatbooks.chatbook_models import (
    ExportJob,
    ExportStatus,
    ImportJob,
    ImportStatus,
)
from tldw_Server_API.tests.Chatbooks.test_chatbook_service import mock_db, service  # noqa: F401


def test_get_export_job_parses_varied_timestamps(service, mock_db, tmp_path):
    """Ensure timestamp parser handles common DB formats."""
    mock_row = {
        "job_id": "job-plain",
        "user_id": service.user_id,
        "status": "completed",
        "chatbook_name": "Test",
        "output_path": str(tmp_path / "test.zip"),
        "created_at": "2024-01-01 00:00:00",
        "started_at": "2024-01-01 00:00:00+00:00",
        "completed_at": "2024-01-01 00:00:00.000001",
        "error_message": None,
        "progress_percentage": 100,
        "total_items": 0,
        "processed_items": 0,
        "file_size_bytes": 0,
        "download_url": None,
        "expires_at": "2024-01-02 00:00:00",
        "metadata": {},
    }
    mock_db.execute_query.return_value = [mock_row]

    job = service._get_export_job("job-plain")

    assert job is not None
    assert job.created_at is not None
    assert job.started_at is not None
    assert job.completed_at is not None

    mock_row["created_at"] = "2024-01-03T00:00:00Z"
    mock_db.execute_query.return_value = [mock_row]
    job = service._get_export_job("job-zulu")
    assert job is not None
    assert job.created_at is not None


def test_parse_timestamp_accepts_numeric_epoch(service):
    """Numeric epoch values should be parsed as UTC datetimes."""
    epoch = 1_700_000_000
    parsed = service._parse_timestamp(epoch)
    assert parsed == datetime.utcfromtimestamp(epoch)


def test_parse_timestamp_normalizes_timezone_offsets(service):
    """Timestamps with explicit offsets should normalize to naive UTC."""
    parsed = service._parse_timestamp("2024-01-01T05:30:00+05:30")
    assert parsed == datetime(2024, 1, 1, 0, 0)


def test_delete_export_job_removes_file_and_record(service, mock_db, tmp_path):
    """Completed export jobs should be removable and delete their archive."""
    export_dir = tmp_path / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    service.export_dir = export_dir

    file_path = export_dir / "export.zip"
    file_path.write_text("payload", encoding="utf-8")

    job = ExportJob(
        job_id="job-1",
        user_id=service.user_id,
        status=ExportStatus.COMPLETED,
        chatbook_name="Test",
        output_path=str(file_path),
    )

    with patch.object(service, "_get_export_job", return_value=job):
        ok = service.delete_export_job(job.job_id)

    assert ok is True
    assert not file_path.exists()
    mock_db.execute_query.assert_any_call(
        "DELETE FROM export_jobs WHERE job_id = ? AND user_id = ?",
        (job.job_id, service.user_id),
        commit=True,
    )


def test_delete_import_job_removes_record(service, mock_db):
    """Cancelled import jobs should be removable."""
    job = ImportJob(
        job_id="job-2",
        user_id=service.user_id,
        status=ImportStatus.CANCELLED,
        chatbook_path="dummy",
    )

    with patch.object(service, "_get_import_job", return_value=job):
        ok = service.delete_import_job(job.job_id)

    assert ok is True
    mock_db.execute_query.assert_any_call(
        "DELETE FROM import_jobs WHERE job_id = ? AND user_id = ?",
        (job.job_id, service.user_id),
        commit=True,
    )


def test_delete_export_job_rejects_non_terminal_status(service):
    """Non-terminal export jobs should not be removable."""
    job = ExportJob(
        job_id="job-3",
        user_id=service.user_id,
        status=ExportStatus.PENDING,
        chatbook_name="Test",
    )

    with patch.object(service, "_get_export_job", return_value=job):
        ok = service.delete_export_job(job.job_id)

    assert ok is False


def test_get_export_job_status(service, mock_db, tmp_path):
    """Test retrieving export job status."""
    export_path = tmp_path / "export.chatbook"
    metadata = json.dumps(
        {
            "conversation_count": 5,
            "note_count": 3,
            "character_count": 2,
        }
    )
    mock_db.execute_query.return_value = [
        (
            "job123",
            "test_user",
            "completed",
            "Test Export",
            str(export_path),
            "2024-01-01T00:00:00",
            "2024-01-01T00:01:00",
            "2024-01-01T00:05:00",
            None,
            100,
            100,
            100,
            1024,
            metadata,
            None,
        )
    ]

    result = service.get_export_job_status("job123")

    assert result["job_id"] == "job123"
    assert result["status"] == "completed"
    assert result["file_path"] == str(export_path)
    assert result["content_summary"]["conversations"] == 5


def test_cancel_export_job(service, mock_db):
    """Test cancelling an export job."""
    mock_db.execute_query.return_value = [
        {
            "job_id": "job123",
            "user_id": "test_user",
            "status": "pending",
            "chatbook_name": "Test",
            "output_path": None,
            "created_at": "2024-01-01T00:00:00",
            "started_at": None,
            "completed_at": None,
            "error_message": None,
            "progress_percentage": 0,
            "total_items": 0,
            "processed_items": 0,
            "file_size_bytes": 0,
            "download_url": None,
            "expires_at": None,
        }
    ]

    result = service.cancel_export_job("job123")

    assert result is True


def test_create_import_job(service, mock_db, tmp_path):
    """Test creating an import job."""
    test_uuid = uuid4()
    job_id = str(test_uuid)
    test_file_path = tmp_path / "test.chatbook"

    with patch("tldw_Server_API.app.core.Chatbooks.chatbook_service.uuid4", return_value=test_uuid):
        mock_db.execute_query.return_value = None

        result = service.create_import_job(
            file_path=str(test_file_path),
            conflict_strategy="skip",
        )

    assert result["job_id"] == job_id
    assert result["status"] == "pending"


def test_get_import_job_status(service, mock_db, tmp_path):
    """Test retrieving import job status."""
    import_path = tmp_path / "import.chatbook"
    mock_db.execute_query.return_value = [
        (
            "job456",
            "test_user",
            "completed",
            str(import_path),
            "2024-01-01T00:00:00",
            "2024-01-01T00:01:00",
            "2024-01-01T00:10:00",
            None,
            100,
            10,
            10,
            10,
            0,
            2,
            "[]",
            "[]",
        )
    ]

    result = service.get_import_job_status("job456")

    assert result["job_id"] == "job456"
    assert result["status"] == "completed"
    assert result["successful_items"] == 10
    assert result["conflicts_found"] == 2
    assert result["conflicts_resolved"]["skipped"] == 2


def test_list_export_jobs(service, mock_db):
    """Test listing export jobs."""
    mock_db.execute_query.return_value = [
        (
            "job1",
            "test_user",
            "completed",
            "Export 1",
            None,
            "2024-01-01T00:00:00",
            None,
            None,
            None,
            100,
            0,
            0,
            0,
            None,
            None,
        ),
        (
            "job2",
            "test_user",
            "pending",
            "Export 2",
            None,
            "2024-01-01T00:00:00",
            None,
            None,
            None,
            50,
            0,
            0,
            0,
            None,
            None,
        ),
    ]

    results = service.list_export_jobs()

    assert len(results) == 2
    assert results[0]["chatbook_name"] == "Export 1"
    assert results[1]["status"] == "pending"


def test_list_import_jobs(service, mock_db, tmp_path):
    """Test listing import jobs."""
    first_import_path = tmp_path / "import.chatbook"
    second_import_path = tmp_path / "import2.chatbook"

    mock_db.execute_query.return_value = [
        (
            "job3",
            "test_user",
            "completed",
            str(first_import_path),
            "2024-01-01T00:00:00",
            None,
            None,
            None,
            100,
            5,
            5,
            5,
            0,
            0,
            "[]",
            "[]",
        ),
        (
            "job4",
            "test_user",
            "failed",
            str(second_import_path),
            "2024-01-01T00:00:00",
            None,
            None,
            "File not found",
            0,
            0,
            0,
            0,
            0,
            0,
            "[]",
            "[]",
        ),
    ]

    results = service.list_import_jobs()

    assert len(results) == 2
    assert results[0]["successful_items"] == 5
    assert results[1]["error_message"] == "File not found"
