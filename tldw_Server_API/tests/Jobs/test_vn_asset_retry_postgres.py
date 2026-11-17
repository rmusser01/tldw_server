"""Reuse the real V1 SDK lifecycle against the official PostgreSQL Jobs fixture."""

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.tests.VN_Assets.test_slot_generation_state import (
    chacha_db as chacha_db,
)
from tldw_Server_API.tests.VN_Assets.test_slot_generation_state import (
    character_id as character_id,
)
from tldw_Server_API.tests.VN_Assets.test_slot_generation_state import (
    fake_jobs as fake_jobs,
)
from tldw_Server_API.tests.VN_Assets.test_slot_generation_state import (
    pack_with_slots as pack_with_slots,
)
from tldw_Server_API.tests.VN_Assets.test_slot_generation_state import (
    service as service,
)
from tldw_Server_API.tests.VN_Assets.test_slot_generation_state import (
    test_v1_outcome_follows_real_sdk_retry_budget as test_v1_outcome_follows_real_sdk_retry_budget,
)

pytestmark = [pytest.mark.pg_jobs, pytest.mark.integration]


@pytest.fixture
def v1_retry_jobs(jobs_pg_dsn, monkeypatch):
    """An explicit cutoff of one still permits zero-backoff PostgreSQL test retries."""
    monkeypatch.setenv("JOBS_QUARANTINE_THRESHOLD", "1")
    return JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
