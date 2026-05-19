from types import SimpleNamespace

from tldw_Server_API.app.core.Chatbooks.chatbook_models import ExportStatus, ImportStatus
from tldw_Server_API.app.core.Chatbooks.jobs_adapter import ChatbooksJobsAdapter


def test_apply_export_status_keeps_cancelled_when_jobs_row_lags():
    adapter = ChatbooksJobsAdapter(owner_user_id=None)
    job = SimpleNamespace(status=ExportStatus.CANCELLED)

    adapter.apply_export_status(job, {"status": "processing"})

    assert job.status is ExportStatus.CANCELLED


def test_apply_import_status_keeps_cancelled_when_jobs_row_lags():
    adapter = ChatbooksJobsAdapter(owner_user_id=None)
    job = SimpleNamespace(status=ImportStatus.CANCELLED)

    adapter.apply_import_status(job, {"status": "queued"})

    assert job.status is ImportStatus.CANCELLED
