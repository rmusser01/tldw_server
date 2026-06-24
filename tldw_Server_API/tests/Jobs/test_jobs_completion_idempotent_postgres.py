import pytest

psycopg = pytest.importorskip("psycopg")
pytestmark = pytest.mark.pg_jobs

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.tests.Jobs.parity.scenarios import run_complete_idempotency_scenario


def test_completion_idempotent_postgres(jobs_pg_dsn):
    run_complete_idempotency_scenario(lambda: JobManager(None, backend="postgres", db_url=jobs_pg_dsn))
