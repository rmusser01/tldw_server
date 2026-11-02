"""Owner status is a Jobs projection, never an administration capability."""

import json
import threading
from contextlib import contextmanager
from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.tests.Jobs.test_jobs_resource_admission_sqlite import (
    archive,
    make_command,
    make_request,
    sql,
)


def principal(owner=1, *, roles=None, permissions=None):
    return AuthPrincipal(
        kind="user",
        user_id=owner,
        roles=roles or ["user"],
        permissions=permissions or ["claims.admin"],
        is_admin=False,
    )


def context_values():
    return (
        JobManager._RLS_IS_ADMIN.get(),
        JobManager._RLS_DOMAIN_ALLOWLIST.get(),
        JobManager._RLS_OWNER_USER_ID.get(),
    )


def admit(manager, owner="1"):
    command = make_command(owner_user_id=owner)
    command.payload["owner_user_id"] = owner
    result = manager.admit_resource_job(command, request=make_request(command))
    assert result.kind == "created"
    return result.job


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.delenv("JOBS_DB_URL", raising=False)
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    return JobManager()


def build_app(user, *, fail_auth=False):
    from tldw_Server_API.app.api.v1.endpoints import jobs_owned

    app = FastAPI()
    app.include_router(jobs_owned.router, prefix="/api/v1")

    async def authenticate():
        if fail_auth:
            raise HTTPException(401, "Authentication required")
        return user

    app.dependency_overrides[get_auth_principal] = authenticate
    return app


def test_owned_read_allowlists_identity_progress_result_and_failure(manager):
    job = admit(manager)
    sql(
        manager,
        "UPDATE jobs SET status='failed', progress_percent=25, "
        "progress_message=?, last_error=?, error_code=?, result=? WHERE id=?",
        (
            "secret provider message",
            "secret raw error",
            "claims_cluster_foreign_reference",
            json.dumps(
                {
                    "clusters_created": 2,
                    "claims_assigned": 4,
                    "claims_skipped": 0,
                    "method": "exact",
                    "outcome": "ok",
                    "claim_text": "secret",
                    "path": "/secret",
                    "provider": "secret",
                }
            ),
            job["id"],
        ),
    )
    with TestClient(build_app(principal())) as client:
        response = client.get(f"/api/v1/jobs/owned/{job['uuid']}")
    assert response.status_code == 200, response.text
    assert response.json() == {
        "id": job["id"],
        "uuid": job["uuid"],
        "domain": "claims",
        "job_type": "claims_rebuild_clusters",
        "status": "failed",
        "progress_percent": 25.0,
        "failure_code": "claims_cluster_foreign_reference",
        "result": {
            "clusters_created": 2,
            "claims_assigned": 4,
            "claims_skipped": 0,
            "method": "exact",
            "outcome": "ok",
        },
    }
    assert "secret" not in response.text


@pytest.mark.parametrize("case", ["foreign", "missing", "type", "domain"])
def test_owned_read_hides_foreign_missing_and_disallowed_jobs(manager, case):
    if case == "missing":
        job_uuid = str(uuid4())
    elif case == "foreign":
        job_uuid = admit(manager, "2")["uuid"]
    else:
        job = manager.create_job(
            domain="other" if case == "domain" else "claims",
            queue="default",
            job_type="claims_rebuild_clusters" if case == "domain" else "private",
            payload={"secret": True},
            owner_user_id="1",
        )
        job_uuid = job["uuid"]
    with TestClient(build_app(principal())) as client:
        response = client.get(f"/api/v1/jobs/owned/{job_uuid}")
    assert response.status_code == 404
    assert response.json() == {"detail": "Job not found"}


@pytest.mark.parametrize("owner", [None, 0])
def test_owned_route_rejects_unroutable_principal(manager, owner):
    job = admit(manager)
    with TestClient(build_app(principal(owner))) as client:
        assert client.get(f"/api/v1/jobs/owned/{job['uuid']}").status_code == 404


def test_owned_route_requires_authentication(manager):
    job = admit(manager)
    with TestClient(build_app(principal(), fail_auth=True)) as client:
        assert client.get(f"/api/v1/jobs/owned/{job['uuid']}").status_code == 401


def test_owned_projection_drops_untrusted_values(manager):
    from tldw_Server_API.app.core.Jobs.owned_status import get_owned_job_status

    job = admit(manager)
    sql(
        manager,
        "UPDATE jobs SET progress_percent=NULL, error_code=?, result=? WHERE id=?",
        (
            "secret exception",
            json.dumps(
                {
                    "clusters_created": True,
                    "claims_assigned": -1,
                    "claims_skipped": 2**63,
                    "method": "secret",
                    "outcome": "secret",
                    "reason": "secret",
                    "similarity_threshold": 2,
                    "unknown": "secret",
                }
            ),
            job["id"],
        ),
    )
    status = get_owned_job_status(manager, job_uuid=job["uuid"], owner_user_id="1")
    assert status.progress_percent is None
    assert status.failure_code is None
    assert status.result == {}


@pytest.mark.parametrize(
    "code",
    [
        "claims_cluster_corruption",
        "claims_cluster_resource_limit",
        "claims_cluster_configuration",
        "claims_cluster_attempt_inactive",
        "claims_cluster_storage_unavailable",
        "claims_cluster_failed",
        "claims_cluster_vector_access",
        "claims_cluster_vector_busy",
        "claims_cluster_vector_closed",
        "claims_cluster_vector_configuration",
        "claims_cluster_vector_corruption",
        "claims_cluster_vector_forked_handle",
        "claims_cluster_vector_invalid_request",
        "claims_cluster_vector_lock_unsupported",
        "claims_cluster_vector_log_coverage",
        "claims_cluster_vector_mixed_access",
        "claims_cluster_vector_path_alias",
        "claims_cluster_vector_read_only",
        "claims_cluster_vector_shutdown",
        "claims_cluster_vector_source_changed",
        "claims_cluster_vector_unsupported_sdk",
        "claims_cluster_arbitrary_secret",
    ],
)
def test_owned_projection_actual_cluster_failure_vocabulary_is_closed(manager, code):
    from tldw_Server_API.app.core.Jobs.owned_status import get_owned_job_status

    job = admit(manager)
    sql(manager, "UPDATE jobs SET error_code=?, last_error='secret' WHERE id=?", (code, job["id"]))
    status = get_owned_job_status(manager, job_uuid=job["uuid"], owner_user_id="1")
    assert status.failure_code == (None if code == "claims_cluster_arbitrary_secret" else code)
    assert "secret" not in status.model_dump_json()


def test_owned_projection_superseded_reason_is_explicit(manager):
    from tldw_Server_API.app.core.Jobs.owned_status import get_owned_job_status

    job = admit(manager)
    sql(
        manager,
        "UPDATE jobs SET result=? WHERE id=?",
        (json.dumps({"outcome": "skipped", "reason": "superseded", "secret": "secret"}), job["id"]),
    )
    assert get_owned_job_status(manager, job_uuid=job["uuid"], owner_user_id="1").result == {
        "outcome": "skipped",
        "reason": "superseded",
    }


def test_owned_status_preserves_canonical_quarantined_state(manager):
    job = admit(manager)
    sql(manager, "UPDATE jobs SET status='quarantined' WHERE id=?", (job["id"],))
    with TestClient(build_app(principal())) as client:
        response = client.get(f"/api/v1/jobs/owned/{job['uuid']}")
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "quarantined"


def test_owned_uuid_survives_reused_id_and_expires_to_404(manager):
    old = admit(manager)
    archive(manager, old)
    replacement = manager.create_job(domain="other", queue="default", job_type="private", payload={}, owner_user_id="2")
    assert old["id"] == replacement["id"] == 1
    with TestClient(build_app(principal())) as client:
        response = client.get(f"/api/v1/jobs/owned/{old['uuid']}")
        assert response.status_code == 200, response.text
        assert response.json()["uuid"] == old["uuid"] != replacement["uuid"]
        sql(manager, "DELETE FROM jobs_archive WHERE uuid=?", (old["uuid"],))
        assert client.get(f"/api/v1/jobs/owned/{old['uuid']}").status_code == 404


def test_owned_helper_restores_nested_and_error_context(manager, monkeypatch):
    from tldw_Server_API.app.core.Jobs.owned_status import get_owned_job_status

    job = admit(manager)
    original = manager.get_job_or_archived_by_uuid

    def nested_lookup(*args, **kwargs):
        assert context_values() == (False, "claims", "1")
        with JobManager.scoped_rls_context(is_admin=False, domain_allowlist="notes", owner_user_id="2"):
            assert context_values() == (False, "notes", "2")
        return original(*args, **kwargs)

    with JobManager.scoped_rls_context(is_admin=True, domain_allowlist="prior", owner_user_id="prior"):
        monkeypatch.setattr(manager, "get_job_or_archived_by_uuid", nested_lookup)
        assert get_owned_job_status(manager, job_uuid=job["uuid"], owner_user_id="1") is not None
        assert context_values() == (True, "prior", "prior")

        def failing_lookup(*args, **kwargs):
            assert context_values() == (False, "claims", "1")
            raise RuntimeError("lookup failed")

        monkeypatch.setattr(manager, "get_job_or_archived_by_uuid", failing_lookup)
        with pytest.raises(RuntimeError, match="lookup failed"):
            get_owned_job_status(manager, job_uuid=job["uuid"], owner_user_id="1")
        assert context_values() == (True, "prior", "prior")


@pytest.mark.pg_jobs
@pytest.mark.integration
def test_owned_status_establishes_and_restores_jobs_rls_context(jobs_pg_dsn, tmp_path, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import jobs_owned
    from tldw_Server_API.app.core.Jobs.owned_status import get_owned_job_status
    from tldw_Server_API.app.core.Jobs.pg_migrations import ensure_jobs_rls_policies_pg, ensure_jobs_tables_pg
    from tldw_Server_API.tests.Jobs.test_jobs_rls_postgres import _dsn_or_skip

    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "observer.db"))
    admin = JobManager(backend="postgres", db_url=jobs_pg_dsn)
    own = admit(admin)
    archive(admin, own)
    active = admit(admin, "2")
    _, restricted_dsn = _dsn_or_skip(monkeypatch)
    ensure_jobs_tables_pg(jobs_pg_dsn)
    ensure_jobs_rls_policies_pg(jobs_pg_dsn)
    restricted = JobManager(backend="postgres", db_url=restricted_dsn)
    with JobManager.scoped_rls_context(is_admin=False, domain_allowlist=None, owner_user_id=None):
        assert restricted.get_job_or_archived_by_uuid(own["uuid"]) is None
        for owner, job in [("1", own), ("2", active), ("1", own)]:
            assert get_owned_job_status(restricted, job_uuid=job["uuid"], owner_user_id=owner).uuid == job["uuid"]
            other = active if owner == "1" else own
            assert get_owned_job_status(restricted, job_uuid=other["uuid"], owner_user_id=owner) is None
            assert context_values() == (False, None, None)

    # Middleware runs on the event loop; the synchronous route must scope and
    # restore tokens in the actual worker, including inherited privileged context.
    app = build_app(principal())
    observed = []
    event_threads = []
    database_scopes = []
    expected_owner = "1"
    real_read = jobs_owned.get_owned_job_status
    real_cursor = JobManager._pg_cursor

    @contextmanager
    def checked_cursor(self, connection):
        with real_cursor(self, connection) as cursor:
            cursor.execute(
                "SELECT current_user AS role, "
                "current_setting('app.is_admin', true) AS admin, "
                "current_setting('app.domain_allowlist', true) AS domain, "
                "current_setting('app.owner_user_id', true) AS owner"
            )
            scope = cursor.fetchone()
            assert dict(scope) == {"role": "jobs_rls", "admin": "false", "domain": "claims", "owner": expected_owner}
            database_scopes.append(dict(scope))
            yield cursor

    monkeypatch.setattr(JobManager, "_pg_cursor", checked_cursor)

    def observed_read(*args, **kwargs):
        before = context_values()
        try:
            return real_read(*args, **kwargs)
        finally:
            observed.append((threading.get_ident(), before, context_values()))

    monkeypatch.setattr(jobs_owned, "get_owned_job_status", observed_read)

    @app.middleware("http")
    async def inherited_context(request, call_next):
        event_threads.append(threading.get_ident())
        with JobManager.scoped_rls_context(is_admin=True, domain_allowlist="prior", owner_user_id="prior"):
            response = await call_next(request)
            assert context_values() == (True, "prior", "prior")
            return response

    with TestClient(app) as client:
        assert client.get(f"/api/v1/jobs/owned/{own['uuid']}").status_code == 200
        assert client.get(f"/api/v1/jobs/owned/{active['uuid']}").status_code == 404
        expected_owner = "2"
        app.dependency_overrides[get_auth_principal] = lambda: principal(2)
        assert client.get(f"/api/v1/jobs/owned/{active['uuid']}").status_code == 200

        def failing_lookup(*args, **kwargs):
            assert context_values() == (False, "claims", "2")
            raise RuntimeError("lookup failed")

        monkeypatch.setattr(JobManager, "get_job_or_archived_by_uuid", failing_lookup)
        with pytest.raises(RuntimeError, match="lookup failed"):
            client.get(f"/api/v1/jobs/owned/{active['uuid']}")
    assert len(database_scopes) == 3
    assert len(observed) == 4
    assert all(
        worker != event_threads[0] and before == after == (True, "prior", "prior") for worker, before, after in observed
    )


def test_owned_openapi_uuid_and_auth_dependency(manager):
    from tldw_Server_API.app.api.v1.endpoints import jobs_owned

    app = build_app(principal())
    operation = app.openapi()["paths"]["/api/v1/jobs/owned/{job_uuid}"]["get"]
    assert operation["parameters"][0]["schema"]["format"] == "uuid"
    route = next(r for r in jobs_owned.router.routes if r.path == "/jobs/owned/{job_uuid}")
    assert [d.call for d in route.dependant.dependencies] == [get_auth_principal]
