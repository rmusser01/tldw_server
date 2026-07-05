from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Chatbooks import jobs_adapter as chatbooks_jobs_adapter
from tldw_Server_API.app.core.Chatbooks.chatbook_models import ExportStatus, ImportStatus
from tldw_Server_API.app.core.Chatbooks.jobs_adapter import ChatbooksJobsAdapter


@pytest.fixture(autouse=True)
def _no_real_jobs_manager(monkeypatch):
    monkeypatch.setattr(chatbooks_jobs_adapter, "_jobs_manager", lambda: SimpleNamespace())


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


@pytest.mark.parametrize(
    ("core_status", "expected"),
    [
        ("queued", ExportStatus.PENDING),
        ("processing", ExportStatus.IN_PROGRESS),
        ("completed", ExportStatus.COMPLETED),
        ("failed", ExportStatus.FAILED),
        ("cancelled", ExportStatus.CANCELLED),
        ("quarantined", ExportStatus.FAILED),
    ],
)
def test_apply_export_status_mapping_contract(core_status, expected):
    adapter = ChatbooksJobsAdapter(owner_user_id=None)
    job = SimpleNamespace(status=ExportStatus.PENDING)

    adapter.apply_export_status(job, {"status": core_status})

    assert job.status is expected


@pytest.mark.parametrize("initial_status", [ExportStatus.COMPLETED, ExportStatus.FAILED])
def test_apply_export_status_preserves_terminal_statuses(initial_status):
    adapter = ChatbooksJobsAdapter(owner_user_id=None)
    job = SimpleNamespace(status=initial_status)

    adapter.apply_export_status(job, {"status": "queued"})

    assert job.status is initial_status


@pytest.mark.parametrize(
    ("core_status", "expected"),
    [
        ("queued", ImportStatus.PENDING),
        ("processing", ImportStatus.IN_PROGRESS),
        ("completed", ImportStatus.COMPLETED),
        ("failed", ImportStatus.FAILED),
        ("cancelled", ImportStatus.CANCELLED),
        ("quarantined", ImportStatus.FAILED),
    ],
)
def test_apply_import_status_mapping_contract(core_status, expected):
    adapter = ChatbooksJobsAdapter(owner_user_id=None)
    job = SimpleNamespace(status=ImportStatus.PENDING)

    adapter.apply_import_status(job, {"status": core_status})

    assert job.status is expected


@pytest.mark.parametrize("initial_status", [ImportStatus.COMPLETED, ImportStatus.FAILED])
def test_apply_import_status_preserves_terminal_statuses(initial_status):
    adapter = ChatbooksJobsAdapter(owner_user_id=None)
    job = SimpleNamespace(status=initial_status)

    adapter.apply_import_status(job, {"status": "queued"})

    assert job.status is initial_status


def test_map_jobs_prefers_payload_chatbooks_job_id(monkeypatch):
    class FakeJobManager:
        def list_jobs(self, **kwargs):
            assert kwargs == {
                "domain": "chatbooks",
                "queue": None,
                "status": None,
                "owner_user_id": "user-1",
                "job_type": "export",
                "limit": 1,
            }
            return [
                {
                    "id": 42,
                    "uuid": "jobs-uuid",
                    "domain": "chatbooks",
                    "job_type": "export",
                    "owner_user_id": "user-1",
                    "payload": {"chatbooks_job_id": "legacy-export-id"},
                    "status": "queued",
                }
            ]

    monkeypatch.setattr(chatbooks_jobs_adapter, "_jobs_manager", lambda: FakeJobManager())
    adapter = ChatbooksJobsAdapter(owner_user_id="user-1")

    mapped = adapter.map_jobs(job_ids=["legacy-export-id"], job_type="export", limit=1)

    assert set(mapped) == {"legacy-export-id"}
    assert mapped["legacy-export-id"]["uuid"] == "jobs-uuid"
