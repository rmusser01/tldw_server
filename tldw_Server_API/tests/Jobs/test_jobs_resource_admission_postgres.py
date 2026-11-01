"""Run the resource protocol against the official Jobs PostgreSQL fixture."""

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.tests.Jobs.test_jobs_resource_admission_sqlite import (
    CONFLICT_REFERENCE_FIELDS,
    ResourceAdmissionCases,
    make_command,
    make_request,
    snapshot,
    sql,
)

pytestmark = pytest.mark.pg_jobs


@pytest.fixture
def manager(jobs_pg_dsn, monkeypatch):
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_CLAIMS", "default,other")
    return JobManager(backend="postgres", db_url=jobs_pg_dsn)


class TestPostgres(ResourceAdmissionCases):
    pass


@pytest.mark.parametrize("key", [None, "request-1"])
def test_restricted_conflict_reference_is_owner_scoped(jobs_pg_dsn, monkeypatch, key):
    from dataclasses import replace

    import psycopg

    from tldw_Server_API.app.core.Jobs.pg_migrations import ensure_jobs_rls_policies_pg
    from tldw_Server_API.tests.Jobs.test_jobs_rls_postgres import _dsn_or_skip

    seed = JobManager(backend="postgres", db_url=jobs_pg_dsn)
    foreign = seed.admit_resource_job(make_command(), request=make_request())
    _, restricted = _dsn_or_skip(monkeypatch)
    ensure_jobs_rls_policies_pg(jobs_pg_dsn)
    manager = JobManager(backend="postgres", db_url=restricted)
    own_command = make_command(owner_user_id="2", payload={**make_command().payload, "owner_user_id": "2"})
    with JobManager.scoped_rls_context(is_admin=False, domain_allowlist="claims", owner_user_id="2"):
        own = manager.admit_resource_job(own_command, request=make_request(own_command))
        changed = replace(own_command, payload={**own_command.payload, "min_size": 3})
        before = snapshot(manager)
        conflict = manager.admit_resource_job(changed, request=make_request(changed, key=key))
        assert conflict.job == {field: own.job[field] for field in CONFLICT_REFERENCE_FIELDS}
        assert foreign.job["uuid"] not in str(conflict)
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            manager.admit_resource_job(make_command(), request=make_request(key=key))
        assert snapshot(manager) == before


def test_restricted_owner_scope_admits_and_replays_from_unset_context(jobs_pg_dsn, monkeypatch):
    from dataclasses import replace

    import psycopg

    from tldw_Server_API.app.core.Jobs.pg_migrations import ensure_jobs_rls_policies_pg
    from tldw_Server_API.tests.Jobs.test_jobs_rls_postgres import _dsn_or_skip

    _, restricted = _dsn_or_skip(monkeypatch)
    ensure_jobs_rls_policies_pg(jobs_pg_dsn)
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    with psycopg.connect(restricted) as conn:
        assert conn.execute("SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user").fetchone() == (
            False,
            False,
        )
    manager = JobManager(backend="postgres", db_url=restricted)
    JobManager.clear_rls_context()
    try:
        assert sql(manager, "SELECT * FROM jobs", rows=True) == []
        with JobManager.scoped_rls_context(is_admin=False, domain_allowlist="claims", owner_user_id="1"):
            first = manager.admit_resource_job(make_command(), request=make_request())
            replay = manager.admit_resource_job(make_command(), request=make_request())
            assert (first.kind, replay.kind) == ("created", "request_replayed")
            assert first.job["uuid"] == replay.job["uuid"]
        assert JobManager._RLS_OWNER_USER_ID.get() is None
        other = make_command(owner_user_id="2", payload={**make_command().payload, "owner_user_id": "2"})
        with JobManager.scoped_rls_context(is_admin=False, domain_allowlist="claims", owner_user_id="2"):
            second = manager.admit_resource_job(other, request=make_request(other))
            assert second.kind == "created"
            assert second.job["uuid"] != first.job["uuid"]
            assert sql(manager, "SELECT uuid FROM jobs", rows=True) == [{"uuid": second.job["uuid"]}]
            # A foreign owner request cannot resolve or reveal the first Job.
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                manager.admit_resource_job(make_command(), request=make_request())
            with pytest.raises(ValueError):
                manager.admit_resource_job(other, request=replace(make_request(other), scope=make_request().scope))
        assert JobManager._RLS_OWNER_USER_ID.get() is None
        assert sql(manager, "SELECT * FROM job_resource_heads", rows=True) == []
        assert sql(manager, "SELECT * FROM job_resource_request_receipts", rows=True) == []
    finally:
        JobManager.clear_rls_context()


def test_restricted_fair_share_counts_other_domains_without_widening_admission(jobs_pg_dsn, monkeypatch):
    import psycopg

    from tldw_Server_API.app.core.exceptions import BadRequestError
    from tldw_Server_API.app.core.Jobs.pg_migrations import ensure_jobs_rls_policies_pg
    from tldw_Server_API.tests.Jobs.test_jobs_rls_postgres import _dsn_or_skip

    # Fixture-owned setup; the actual admission below uses the restricted role.
    seed = JobManager(backend="postgres", db_url=jobs_pg_dsn)
    for owner in ("1", "2"):
        seed.create_job(domain="other-domain", queue="default", job_type="ordinary", payload={}, owner_user_id=owner)
    _, restricted = _dsn_or_skip(monkeypatch)
    ensure_jobs_rls_policies_pg(jobs_pg_dsn)
    monkeypatch.setenv("JOBS_MAX_PER_USER", "1")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    manager = JobManager(backend="postgres", db_url=restricted)
    JobManager.clear_rls_context()
    try:
        with JobManager.scoped_rls_context(is_admin=False, domain_allowlist="claims", owner_user_id="1"):
            before = snapshot(manager)
            assert sql(manager, "SELECT * FROM jobs", rows=True) == []
            with pytest.raises(BadRequestError, match="maximum concurrent job limit"):
                manager.admit_resource_job(make_command(), request=make_request())
            assert snapshot(manager) == before
            assert manager._count_active_jobs_for_user("1") == 1
            assert manager._count_active_jobs_for_user("2") == 0
            assert (
                JobManager._RLS_IS_ADMIN.get(),
                JobManager._RLS_DOMAIN_ALLOWLIST.get(),
                JobManager._RLS_OWNER_USER_ID.get(),
            ) == (False, "claims", "1")

            # At two slots only this owner's other-domain work counts, not owner 2.
            monkeypatch.setenv("JOBS_MAX_PER_USER", "2")
            first = manager.admit_resource_job(make_command(), request=make_request())
            assert first.kind == "created"
            monkeypatch.setenv("JOBS_MAX_PER_USER", "1")
            admitted = snapshot(manager)
            replay = manager.admit_resource_job(make_command(), request=make_request())
            assert replay.kind == "request_replayed"
            assert replay.job["uuid"] == first.job["uuid"]
            assert manager.admit_resource_job(make_command(), request=make_request(key=None)).kind == "active_reused"
            assert snapshot(manager) == admitted
            assert sql(manager, "SELECT domain FROM jobs", rows=True) == [{"domain": "claims"}]

            # The scalar aggregate scope must not widen the actual INSERT context.
            monkeypatch.setenv("JOBS_MAX_PER_USER", "100")
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                manager.create_job(
                    domain="other-domain", queue="default", job_type="ordinary", payload={}, owner_user_id="1"
                )
            assert JobManager._RLS_DOMAIN_ALLOWLIST.get() == "claims"
        assert JobManager._RLS_OWNER_USER_ID.get() is None
    finally:
        JobManager.clear_rls_context()
