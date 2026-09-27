"""Owner scoping for JobManager.get_job.

The jobs table is global and shared by every user. Request handlers must be
able to read a job without being handed somebody else's row, so get_job takes
an optional owner filter. Workers keep calling it unscoped on purpose.
"""

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables


@pytest.fixture()
def jobs_db(tmp_path):
    db_path = tmp_path / "jobs.db"
    ensure_jobs_tables(db_path)
    yield db_path


def _make_job(jm: JobManager, owner: str) -> int:
    job = jm.create_job(
        domain="connectors",
        queue="default",
        job_type="sync",
        payload={"action": "sync"},
        owner_user_id=owner,
    )
    return int(job["id"])


@pytest.mark.unit
def test_get_job_scoped_to_owner_returns_the_row(jobs_db):
    jm = JobManager(jobs_db)
    job_id = _make_job(jm, "alice")

    found = jm.get_job(job_id, owner_user_id="alice")

    assert found is not None
    assert int(found["id"]) == job_id


@pytest.mark.unit
def test_get_job_scoped_to_another_owner_returns_none(jobs_db):
    """The regression: a foreign job must read as absent, not as data."""
    jm = JobManager(jobs_db)
    job_id = _make_job(jm, "alice")

    assert jm.get_job(job_id, owner_user_id="bob") is None


@pytest.mark.unit
def test_get_job_unscoped_still_returns_any_owner(jobs_db):
    """Workers process other users' jobs by design; that path is unchanged."""
    jm = JobManager(jobs_db)
    job_id = _make_job(jm, "alice")

    found = jm.get_job(job_id)

    assert found is not None
    assert int(found["id"]) == job_id


@pytest.mark.unit
@pytest.mark.parametrize("blank", ["", "   "])
def test_get_job_rejects_a_blank_owner_instead_of_widening(jobs_db, blank):
    """A caller that cannot name an owner must fail, not read every row."""
    jm = JobManager(jobs_db)
    _make_job(jm, "alice")

    with pytest.raises(ValueError, match="owner_user_id"):
        jm.get_job(1, owner_user_id=blank)
