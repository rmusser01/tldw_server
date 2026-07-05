import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.tests.Jobs.parity.scenarios import run_complete_idempotency_scenario


@pytest.mark.unit
def test_completion_idempotent_sqlite(tmp_path, monkeypatch):
    db_path = tmp_path / "jobs.db"
    monkeypatch.setenv("JOBS_DB_PATH", str(db_path))
    run_complete_idempotency_scenario(lambda: JobManager())
