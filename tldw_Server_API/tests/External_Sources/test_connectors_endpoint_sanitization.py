import sys
import types
from unittest.mock import MagicMock

from tldw_Server_API.app.api.v1.endpoints import connectors


def test_extract_request_base_debug_log_is_sanitized(monkeypatch):
    class _BadRequest:
        @property
        def base_url(self):
            raise ValueError("request base leaked at /private/connectors-callback")

    fake_logger = MagicMock()
    monkeypatch.setattr(connectors, "logger", fake_logger)

    result = connectors._extract_request_base(_BadRequest())

    assert result == ""
    fake_logger.debug.assert_called_once_with("Failed to resolve base_url from request")


def test_load_active_job_warning_log_is_sanitized(monkeypatch):
    class _FailingJobManager:
        def get_job(self, job_id: int):
            assert job_id == 42
            raise RuntimeError("connectors job backend exploded at /private/connectors-jobs.db")

    fake_jobs_module = types.ModuleType("jobs_manager")
    fake_jobs_module.JobManager = _FailingJobManager
    fake_logger = MagicMock()

    monkeypatch.setitem(sys.modules, "tldw_Server_API.app.core.Jobs.manager", fake_jobs_module)
    monkeypatch.setattr(connectors, "logger", fake_logger)

    result = connectors._load_active_job({"active_job_id": "42"})

    assert result is None
    fake_logger.warning.assert_called_once_with("Failed to load active connectors job")
