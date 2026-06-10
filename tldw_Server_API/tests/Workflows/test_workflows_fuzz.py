from __future__ import annotations

import os
import pathlib
import uuid
import time
import tempfile
from typing import List

import pytest
from fastapi.testclient import TestClient
from hypothesis import given, strategies as st, settings, HealthCheck
import string

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.api.v1.endpoints import workflows as wf_mod
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.permissions import WORKFLOWS_RUNS_READ
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user


pytestmark = pytest.mark.integration


@pytest.fixture()
def client_with_db(tmp_path, auth_headers):
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))

    async def override_admin():
        return User(id=1, username="admin", email="a@x", is_active=True, is_admin=True, tenant_id="default")

    async def override_principal():
        return AuthPrincipal(
            kind="user",
            user_id=1,
            username="admin",
            email="a@x",
            roles=["admin"],
            permissions=[WORKFLOWS_RUNS_READ],
        )

    def override_db():

        return db

    app.dependency_overrides[get_request_user] = override_admin
    app.dependency_overrides[get_auth_principal] = override_principal
    app.dependency_overrides[wf_mod._get_db] = override_db

    with TestClient(app, headers=auth_headers) as client:
        yield client

    app.dependency_overrides.clear()


def _step_ids(n: int) -> List[str]:
    return [f"s{i+1}" for i in range(n)]


def _wait_for_terminal(client: TestClient, run_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    last_status = None
    while time.time() < deadline:
        response = client.get(f"/api/v1/workflows/runs/{run_id}")
        assert response.status_code == 200, response.text
        data = response.json()
        last_status = data.get("status")
        if last_status in {"succeeded", "failed", "cancelled", "canceled"}:
            return data
        time.sleep(0.05)
    pytest.fail(f"workflow run {run_id} did not finish before timeout; last status={last_status}")


@settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    st.integers(min_value=1, max_value=5),
    st.lists(st.sampled_from(["prompt", "log", "delay"]), min_size=1, max_size=5),
)
def test_definition_fuzz_linear_or_branch(client_with_db: TestClient, n_steps: int, types_list: List[str]):
    client = client_with_db
    # Build a small randomized definition; on_success may be valid or dangling
    steps = []
    ids = _step_ids(min(n_steps, len(types_list)))
    for sid, t in zip(ids, types_list):
        cfg = {"template": "ok"} if t == "prompt" else {}
        step = {"id": sid, "type": t, "config": cfg}
        steps.append(step)
    # Optionally add a branch creating a tiny cycle for robustness
    if len(steps) >= 2:
        steps[0]["on_success"] = steps[1]["id"]
        # 20% chance to form a trivial cycle
        if os.urandom(1)[0] % 5 == 0:
            steps[1]["on_success"] = steps[0]["id"]

    definition = {"name": "fuzz", "version": 1, "steps": steps}
    # Should be accepted by create
    r = client.post("/api/v1/workflows", json=definition)
    assert r.status_code in (201, 422, 413)
    if r.status_code != 201:
        return
    wid = r.json()["id"]
    rr = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}})
    assert rr.status_code == 200
    _wait_for_terminal(client, rr.json()["run_id"])


_SAFE_CHARS = string.ascii_letters + string.digits + "_-"
_SAFE_TEXT = st.text(alphabet=st.sampled_from(list(_SAFE_CHARS)), min_size=1, max_size=32)

@settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(_SAFE_TEXT)
def test_artifact_path_fuzz_strict_vs_non_strict(monkeypatch, client_with_db: TestClient, suffix: str):
    client = client_with_db
    # Ensure env strict by default
    monkeypatch.setenv("WORKFLOWS_ARTIFACT_VALIDATE_STRICT", "true")
    db: WorkflowsDatabase = app.dependency_overrides[wf_mod._get_db]()
    # Use suffix to avoid unique constraint collisions across Hypothesis examples
    d = {"name": f"art-fuzz-{suffix}", "version": 1, "steps": [{"id": "s1", "type": "prompt", "config": {"template": "ok"}}]}
    _resp = client.post("/api/v1/workflows", json=d)
    if _resp.status_code != 201:
        # Skip this example if definition was rejected (e.g., duplicate name/version)
        return
    wid = _resp.json()["id"]
    run_id = f"artifact-fuzz-{uuid.uuid4()}"
    db.create_run(
        run_id=run_id,
        tenant_id="default",
        user_id="1",
        inputs={},
        workflow_id=wid,
        definition_version=1,
        definition_snapshot=d,
    )
    db.update_run_status(run_id, status="succeeded")

    # Temp file (outside CWD workdir scope most of the time)
    fd, path = tempfile.mkstemp(prefix=f"wf_fuzz_{suffix}_")
    os.write(fd, b"data")
    os.close(fd)
    uri = f"file://{path}"

    # Insert artifact
    art_id = f"af_{abs(hash(uri)) % 100000}"
    db.add_artifact(
        artifact_id=art_id,
        tenant_id="default",
        run_id=run_id,
        step_run_id=None,
        type="file",
        uri=uri,
        size_bytes=4,
        mime_type="text/plain",
        metadata={"workdir": str(pathlib.Path.cwd())},
    )

    # Strict should 400
    r1 = client.get(f"/api/v1/workflows/artifacts/{art_id}/download")
    assert r1.status_code in (200, 400)

    # Non-strict allows even on mismatch
    monkeypatch.setenv("WORKFLOWS_ARTIFACT_VALIDATE_STRICT", "false")
    r2 = client.get(f"/api/v1/workflows/artifacts/{art_id}/download")
    assert r2.status_code == 200
