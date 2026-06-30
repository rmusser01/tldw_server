import os
from pathlib import Path
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.api.v1.endpoints import workflows as wf_mod
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.permissions import WORKFLOWS_RUNS_READ
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user


@pytest.fixture()
def client_and_db(tmp_path, auth_headers):
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))

    async def override_user():
        # Match the run's owner and tenant for strict owner checks
        return User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True, tenant_id="default")

    async def override_principal():
        return AuthPrincipal(
            kind="user",
            user_id=1,
            username="tester",
            email="t@e.com",
            roles=["admin"],
            permissions=[WORKFLOWS_RUNS_READ],
        )

    def override_db():

        return db

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_auth_principal] = override_principal
    app.dependency_overrides[wf_mod._get_db] = override_db

    with TestClient(app, headers=auth_headers) as client:
        yield client, db

    app.dependency_overrides.clear()


def _bootstrap_run_with_artifact(db: WorkflowsDatabase, tmpdir: Path):
    run_id = f"run-range-{uuid4()}"
    tenant = "default"
    # Use user_id that matches override_user().id
    user_id = "1"
    db.create_run(run_id=run_id, tenant_id=tenant, user_id=user_id, inputs={})
    # Create a small file
    afile = tmpdir / "sample.bin"
    data = b"0123456789abcdefghijklmnopqrstuvwxyz"
    afile.write_bytes(data)
    db.add_artifact(
        artifact_id=f"art-range-{uuid4()}",
        tenant_id=tenant,
        run_id=run_id,
        step_run_id=None,
        type="blob",
        uri=f"file://{afile}",
        size_bytes=len(data),
        mime_type="application/octet-stream",
        checksum_sha256=None,
        metadata={"workdir": str(tmpdir)},
    )
    return run_id


def _bootstrap_run_with_outside_artifact(db: WorkflowsDatabase, tmpdir: Path):
    run_id = f"run-outside-{uuid4()}"
    tenant = "default"
    user_id = "1"
    db.create_run(run_id=run_id, tenant_id=tenant, user_id=user_id, inputs={})
    afile = tmpdir / "outside.bin"
    afile.write_bytes(b"outside-scope")
    db.add_artifact(
        artifact_id=f"art-outside-{uuid4()}",
        tenant_id=tenant,
        run_id=run_id,
        step_run_id=None,
        type="blob",
        uri=f"file://{afile}",
        size_bytes=afile.stat().st_size,
        mime_type="application/octet-stream",
        checksum_sha256=None,
        metadata={},
    )
    return run_id


def test_artifact_download_with_range(monkeypatch, tmp_path, client_and_db):


     # Enable permissive MIME
    monkeypatch.setenv("WORKFLOWS_ARTIFACT_ALLOWED_MIME", "application/octet-stream")
    client, db = client_and_db
    run_id = _bootstrap_run_with_artifact(db, tmp_path)
    # Sanity: run should be present in the overridden DB instance
    assert db.get_run(run_id) is not None
    # auth: tests use single-user mode; dependency injects admin-like claims
    # Fetch artifact list to get id
    r = client.get(f"/api/v1/workflows/runs/{run_id}/artifacts")
    assert r.status_code == 200
    art_id = r.json()[0]["artifact_id"]
    # Request a range
    headers = {"Range": "bytes=0-9"}
    r2 = client.get(f"/api/v1/workflows/artifacts/{art_id}/download", headers=headers)
    assert r2.status_code == 206
    assert r2.headers.get("Content-Range", "").startswith("bytes 0-9/")
    assert r2.headers.get("Accept-Ranges") == "bytes"
    assert r2.content == b"0123456789"


def test_artifact_download_blocks_outside_scope(monkeypatch, tmp_path, client_and_db):
    monkeypatch.setenv("WORKFLOWS_ARTIFACT_ALLOWED_MIME", "application/octet-stream")
    monkeypatch.setenv("WORKFLOWS_ARTIFACT_VALIDATE_STRICT", "true")
    monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    client, db = client_and_db
    run_id = _bootstrap_run_with_outside_artifact(db, tmp_path)
    r = client.get(f"/api/v1/workflows/runs/{run_id}/artifacts")
    assert r.status_code == 200
    art_id = r.json()[0]["artifact_id"]
    resp = client.get(f"/api/v1/workflows/artifacts/{art_id}/download")
    assert resp.status_code == 400
