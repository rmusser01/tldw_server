import pytest

psycopg = pytest.importorskip("psycopg")
pytestmark = pytest.mark.pg_jobs

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.tests.Jobs.parity.scenarios import (
    run_idempotent_create_preserves_original_request_ids_scenario,
    run_idempotent_create_scope_scenario,
)


def test_idempotency_scoped_to_domain_queue_type_postgres(jobs_pg_dsn):
    run_idempotent_create_scope_scenario(lambda: JobManager(None, backend="postgres", db_url=jobs_pg_dsn))


def test_idempotent_create_preserves_original_request_id_postgres(jobs_pg_dsn):
    run_idempotent_create_preserves_original_request_ids_scenario(
        lambda: JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    )
