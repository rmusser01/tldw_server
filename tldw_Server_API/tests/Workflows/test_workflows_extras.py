import importlib.machinery
import json
import sys
import time
import types
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

# Stub heavyweight audio deps before app import for local test stability.
if "torch" not in sys.modules:
    _fake_torch = types.ModuleType("torch")
    _fake_torch.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    _fake_torch.Tensor = object
    _fake_torch.nn = types.SimpleNamespace(Module=object)
    sys.modules["torch"] = _fake_torch

if "faster_whisper" not in sys.modules:
    _fake_fw = types.ModuleType("faster_whisper")
    _fake_fw.__spec__ = importlib.machinery.ModuleSpec("faster_whisper", loader=None)

    class _StubWhisperModel:
        def __init__(self, *args, **kwargs):
            pass

    _fake_fw.WhisperModel = _StubWhisperModel
    _fake_fw.BatchedInferencePipeline = _StubWhisperModel
    sys.modules["faster_whisper"] = _fake_fw

if "transformers" not in sys.modules:
    _fake_tf = types.ModuleType("transformers")
    _fake_tf.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)

    class _StubProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class _StubModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    _fake_tf.AutoProcessor = _StubProcessor
    _fake_tf.Qwen2AudioForConditionalGeneration = _StubModel
    sys.modules["transformers"] = _fake_tf

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.api.v1.endpoints import workflows as wf_mod
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.permissions import (
    WORKFLOWS_RUNS_CONTROL,
    WORKFLOWS_RUNS_READ,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user


pytestmark = pytest.mark.integration


@pytest.fixture()
def client_with_wf(tmp_path, auth_headers):
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    permissions = [WORKFLOWS_RUNS_READ, WORKFLOWS_RUNS_CONTROL]

    async def override_user():
        return User(
            id=1,
            username="tester",
            email="t@e.com",
            is_active=True,
            is_admin=True,
            tenant_id="default",
            roles=["admin"],
            permissions=permissions,
        )

    async def override_principal():
        return AuthPrincipal(
            kind="user",
            user_id=1,
            username="tester",
            email="t@e.com",
            roles=["admin"],
            permissions=permissions,
        )

    def override_db():

        return db

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_auth_principal] = override_principal
    app.dependency_overrides[wf_mod._get_db] = override_db

    with TestClient(app, headers=auth_headers) as client:
        yield client

    app.dependency_overrides.clear()


def test_rag_search_workflow(client_with_wf: TestClient):
    client = client_with_wf
    definition = {
        "name": "rag",
        "version": 1,
        "steps": [
            {"id": "s1", "type": "rag_search", "config": {"query": "hello world", "top_k": 1}},
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    # wait for completion
    for _ in range(50):
        r = client.get(f"/api/v1/workflows/runs/{run_id}")
        data = r.json()
        if data["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.05)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert isinstance(out.get("documents", []), list)


def test_ws_events_stream(client_with_wf: TestClient):
    client = client_with_wf
    definition = {
        "name": "ws-events",
        "version": 1,
        "steps": [
            {"id": "s1", "type": "prompt", "config": {"template": "A"}},
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]

    # Create token via MCP JWT manager
    from tldw_Server_API.app.core.MCP_unified.auth.jwt_manager import get_jwt_manager
    jwtm = get_jwt_manager()
    token = jwtm.create_access_token(subject="1", username="tester", roles=["user"], permissions=[])

    with client.websocket_connect(f"/api/v1/workflows/ws?run_id={run_id}&token={token}") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "snapshot"
        # Read next few messages until run_completed
        got_completed = False
        deadline = time.time() + 2
        while time.time() < deadline and not got_completed:
            ev = ws.receive_json()
            if ev.get("event_type") == "run_completed":
                got_completed = True
        assert got_completed


def test_mcp_tool_workflow(client_with_wf: TestClient):
    client = client_with_wf
    definition = {
        "name": "mcp-echo",
        "version": 1,
        "steps": [
            {"id": "s1", "type": "mcp_tool", "config": {"tool_name": "echo", "arguments": {"message": "ping"}}},
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    for _ in range(50):
        data = client.get(f"/api/v1/workflows/runs/{run_id}").json()
        if data["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.05)
    assert data["status"] == "succeeded"
    assert (data.get("outputs") or {}).get("result") == "ping"


def test_webhook_step_noop(client_with_wf: TestClient, monkeypatch):
    client = client_with_wf
    # Ensure webhook manager will not attempt outbound calls
    monkeypatch.setenv("TEST_MODE", "1")
    definition = {
        "name": "webhook-noop",
        "version": 1,
        "steps": [
            {"id": "s1", "type": "webhook", "config": {"event": "evaluation.progress", "data": {"x": 1}}},
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {"user_id": "1"}}).json()["run_id"]
    for _ in range(50):
        data = client.get(f"/api/v1/workflows/runs/{run_id}").json()
        if data["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.05)
    assert data["status"] == "succeeded"


def _wait_for_status(client: TestClient, run_id: str, allowed: set[str], timeout_seconds: float = 8.0) -> dict:
    deadline = time.time() + timeout_seconds
    last = {}
    while time.time() < deadline:
        last = client.get(f"/api/v1/workflows/runs/{run_id}").json()
        if last.get("status") in allowed:
            return last
        time.sleep(0.05)
    return last


def _patch_acp_runner(monkeypatch):
    class _StubRunner:
        async def create_session(self, **kwargs):
            return f"acp-session-{uuid4().hex}"

        async def verify_session_access(self, session_id, user_id):
            return True

        async def prompt(self, session_id, prompt):
            return {
                "stopReason": "end_turn",
                "content": [{"type": "output_text", "text": "ok"}],
                "usage": {"input_tokens": 10, "output_tokens": 20},
            }

    runner = _StubRunner()

    async def _stub_get_runner_client():
        return runner

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.integration.acp.get_runner_client",
        _stub_get_runner_client,
    )


def test_l1_acp_pipeline_template_run_succeeds(client_with_wf: TestClient, monkeypatch):
    _patch_acp_runner(monkeypatch)
    client = client_with_wf

    template = client.get("/api/v1/workflows/templates/pipeline_l1_acp").json()
    create_resp = client.post("/api/v1/workflows", json=template)
    assert create_resp.status_code in (200, 201), create_resp.text
    workflow_id = create_resp.json()["id"]

    run_resp = client.post(
        f"/api/v1/workflows/{workflow_id}/run",
        json={"inputs": {"task": "domain-only task", "workspace_id": "ws-1", "workspace_group_id": "wg-1"}},
    )
    assert run_resp.status_code == 200, run_resp.text
    run_id = run_resp.json()["run_id"]

    data = _wait_for_status(client, run_id, {"succeeded", "failed"})
    assert data.get("status") == "succeeded"


def test_l2_acp_pipeline_template_approval_flow(client_with_wf: TestClient, monkeypatch):
    _patch_acp_runner(monkeypatch)
    client = client_with_wf

    template = client.get("/api/v1/workflows/templates/pipeline_l2_acp").json()
    create_resp = client.post("/api/v1/workflows", json=template)
    assert create_resp.status_code in (200, 201), create_resp.text
    workflow_id = create_resp.json()["id"]

    run_resp = client.post(
        f"/api/v1/workflows/{workflow_id}/run",
        json={"inputs": {"task": "domain-only task", "reviewer_user_id": "1"}},
    )
    assert run_resp.status_code == 200, run_resp.text
    run_id = run_resp.json()["run_id"]

    waiting = _wait_for_status(client, run_id, {"waiting_approval", "failed"})
    assert waiting.get("status") == "waiting_approval"

    approve = client.post(
        f"/api/v1/workflows/runs/{run_id}/steps/impl_review/approve",
        json={"comment": "looks good"},
    )
    assert approve.status_code == 200, approve.text

    terminal = _wait_for_status(client, run_id, {"succeeded", "failed"})
    assert terminal.get("status") == "succeeded"


def test_l2_acp_pipeline_reject_uses_on_failure(client_with_wf: TestClient, monkeypatch):
    _patch_acp_runner(monkeypatch)
    client = client_with_wf

    template = client.get("/api/v1/workflows/templates/pipeline_l2_acp").json()
    create_resp = client.post("/api/v1/workflows", json=template)
    assert create_resp.status_code in (200, 201), create_resp.text
    workflow_id = create_resp.json()["id"]

    run_resp = client.post(
        f"/api/v1/workflows/{workflow_id}/run",
        json={"inputs": {"task": "domain-only task", "reviewer_user_id": "1"}},
    )
    assert run_resp.status_code == 200, run_resp.text
    run_id = run_resp.json()["run_id"]

    waiting = _wait_for_status(client, run_id, {"waiting_approval", "failed"})
    assert waiting.get("status") == "waiting_approval"

    reject = client.post(
        f"/api/v1/workflows/runs/{run_id}/steps/impl_review/reject",
        json={"comment": "needs changes"},
    )
    assert reject.status_code == 200, reject.text

    resumed = _wait_for_status(client, run_id, {"waiting_approval", "succeeded", "failed"})
    assert resumed.get("status") in {"waiting_approval", "succeeded"}


def test_l1_acp_pipeline_persists_schema_version_on_acp_steps(client_with_wf: TestClient, monkeypatch):
    _patch_acp_runner(monkeypatch)
    client = client_with_wf

    template = client.get("/api/v1/workflows/templates/pipeline_l1_acp").json()
    workflow_id = client.post("/api/v1/workflows", json=template).json()["id"]
    run_id = client.post(
        f"/api/v1/workflows/{workflow_id}/run",
        json={"inputs": {"task": "domain-only task", "workspace_id": "ws-1", "workspace_group_id": "wg-1"}},
    ).json()["run_id"]

    terminal = _wait_for_status(client, run_id, {"succeeded", "failed"})
    assert terminal.get("status") == "succeeded"

    db: WorkflowsDatabase = app.dependency_overrides[wf_mod._get_db]()
    impl_step_run = db.get_latest_step_run(run_id=run_id, step_id="impl")
    assert impl_step_run is not None
    impl_outputs = json.loads(impl_step_run.get("outputs_json") or "{}")
    assert impl_outputs.get("acp_output_schema_version") == "1.0"
