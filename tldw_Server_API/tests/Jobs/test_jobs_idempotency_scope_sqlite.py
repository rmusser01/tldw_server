import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables
from tldw_Server_API.tests.Jobs.parity.scenarios import (
    run_idempotent_create_preserves_original_request_ids_scenario,
    run_idempotent_create_scope_scenario,
)


@pytest.fixture()
def jobs_db(tmp_path):
    db_path = tmp_path / "jobs.db"
    ensure_jobs_tables(db_path)
    return db_path


def test_idempotency_scoped_to_domain_queue_type_sqlite(jobs_db):
    run_idempotent_create_scope_scenario(lambda: JobManager(jobs_db))


def test_idempotent_create_preserves_original_request_id_sqlite(jobs_db):
    run_idempotent_create_preserves_original_request_ids_scenario(lambda: JobManager(jobs_db))
