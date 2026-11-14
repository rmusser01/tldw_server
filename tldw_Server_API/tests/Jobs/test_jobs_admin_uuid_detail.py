"""Admin links bind the logical UUID without weakening existing guards."""

from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.tests.Jobs.test_jobs_owned_status_api import (
    admit,
    context_values,
    principal,
)
from tldw_Server_API.tests.Jobs.test_jobs_resource_admission_sqlite import archive, sql


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.delenv("JOBS_DB_URL", raising=False)
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    return JobManager()


def admin_app(user):
    from tldw_Server_API.app.api.v1.endpoints import jobs_admin

    app = FastAPI()
    app.dependency_overrides[get_auth_principal] = lambda: user
    app.include_router(jobs_admin.router, prefix="/api/v1")
    return app


def test_admin_uuid_link_cannot_follow_reused_id(manager):
    old = admit(manager)
    archive(manager, old)
    replacement = manager.create_job(
        domain="claims", queue="default", job_type="private", payload={}, owner_user_id="2"
    )
    assert old["id"] == replacement["id"] == 1
    with TestClient(admin_app(principal(roles=["admin"]))) as client:
        response = client.get(f"/api/v1/jobs/{old['id']}", params={"domain": "claims", "job_uuid": old["uuid"]})
        assert response.status_code == 200, response.text
        assert response.json()["uuid"] == old["uuid"] != replacement["uuid"]
        assert response.json()["archived"] is True
        assert client.get(f"/api/v1/jobs/{old['id']}").json()["uuid"] == replacement["uuid"]
        sql(manager, "DELETE FROM jobs_archive WHERE uuid=?", (old["uuid"],))
        assert client.get(f"/api/v1/jobs/{old['id']}", params={"job_uuid": old["uuid"]}).status_code == 404


def test_admin_uuid_live_precondition_and_openapi(manager):
    job = admit(manager)
    app = admin_app(principal(roles=["admin"]))
    with TestClient(app) as client:
        assert client.get(f"/api/v1/jobs/{job['id']}", params={"job_uuid": str(uuid4())}).status_code == 404
        assert client.get(f"/api/v1/jobs/{job['id']}", params={"job_uuid": job["uuid"].upper()}).status_code == 200
        assert client.get(f"/api/v1/jobs/{job['id']}", params={"job_uuid": "bad"}).status_code == 422
    parameters = app.openapi()["paths"]["/api/v1/jobs/{job_id}"]["get"]["parameters"]
    param = next(p for p in parameters if p["name"] == "job_uuid")
    assert param["required"] is False
    assert {"type": "string", "format": "uuid"} in param["schema"]["anyOf"]


def test_claims_permission_only_does_not_grant_jobs_administration(manager):
    from tldw_Server_API.app.api.v1.endpoints import jobs_owned

    job = admit(manager)
    app = admin_app(principal())
    app.include_router(jobs_owned.router, prefix="/api/v1")
    with TestClient(app) as client:
        assert client.get(f"/api/v1/jobs/owned/{job['uuid']}").status_code == 200
        assert client.get(f"/api/v1/jobs/{job['id']}", params={"job_uuid": job["uuid"]}).status_code == 403
        assert (
            client.get("/api/v1/jobs/queue/status", params={"domain": "claims", "queue": "default"}).status_code == 403
        )


@pytest.mark.asyncio
async def test_admin_preflight_awaits_existing_claim_bypass_without_setting_rls(monkeypatch):
    from tldw_Server_API.app.api.v1.API_Deps.jobs_admin_deps import authorize_jobs_admin_domain_read

    with JobManager.scoped_rls_context(is_admin=False, domain_allowlist="before", owner_user_id="2"):
        # Platform admin permission bypass is accepted even without an admin role.
        await authorize_jobs_admin_domain_read(principal(permissions=["*"]), domain="claims")
        assert context_values() == (False, "before", "2")
        with pytest.raises(HTTPException) as denied:
            await authorize_jobs_admin_domain_read(principal(), domain="claims")
        assert denied.value.status_code == 403
        monkeypatch.setenv("JOBS_DOMAIN_SCOPED_RBAC", "1")
        monkeypatch.setenv("JOBS_DOMAIN_ALLOWLIST_1", "other")
        with pytest.raises(HTTPException) as denied:
            await authorize_jobs_admin_domain_read(principal(roles=["admin"]), domain="claims")
        assert denied.value.status_code == 403
        assert context_values() == (False, "before", "2")
