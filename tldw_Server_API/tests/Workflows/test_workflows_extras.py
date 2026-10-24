import asyncio
import importlib.machinery
import json
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocket, WebSocketDisconnect

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

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.endpoints import workflows as wf_mod
from tldw_Server_API.app.core.AuthNZ.permissions import (
    WORKFLOWS_RUNS_CONTROL,
    WORKFLOWS_RUNS_READ,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.main import app

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
        "metadata": {"mcp": {"allowlist": ["echo"], "scopes": ["*"]}},
        "steps": [
            {"id": "s1", "type": "mcp_tool", "config": {"tool_name": "echo", "arguments": {"message": "ping"}}},
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(
        f"/api/v1/workflows/{wid}/run",
        json={"inputs": {}},
        params={"mode": "sync"},
    ).json()["run_id"]
    data = _wait_for_status(client, run_id, {"succeeded", "failed"})
    assert data["status"] == "succeeded", data
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
    run_id = client.post(
        f"/api/v1/workflows/{wid}/run",
        json={"inputs": {"user_id": "1"}},
        params={"mode": "sync"},
    ).json()["run_id"]
    data = _wait_for_status(client, run_id, {"succeeded", "failed"})
    assert data["status"] == "succeeded", data


def _wait_for_status(client: TestClient, run_id: str, allowed: set[str], timeout_seconds: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last = {}
    while time.monotonic() < deadline:
        last = client.get(f"/api/v1/workflows/runs/{run_id}").json()
        if last.get("status") in allowed:
            return last
        time.sleep(0.05)
    pytest.fail(
        f"workflow run {run_id} did not reach {sorted(allowed)} "
        f"within {timeout_seconds}s; last={last}"
    )


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
        params={"mode": "sync"},
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
        params={"mode": "sync"},
    ).json()["run_id"]

    terminal = _wait_for_status(client, run_id, {"succeeded", "failed"})
    assert terminal.get("status") == "succeeded"

    db: WorkflowsDatabase = app.dependency_overrides[wf_mod._get_db]()
    impl_step_run = db.get_latest_step_run(run_id=run_id, step_id="impl")
    assert impl_step_run is not None
    impl_outputs = json.loads(impl_step_run.get("outputs_json") or "{}")
    assert impl_outputs.get("acp_output_schema_version") == "1.0"


@pytest.mark.parametrize("target, close_code", [("owned", None), ("foreign", 4403), ("missing", 4404)])
def test_ws_authorization_during_concurrent_run_reads(client_with_wf, target, close_code):
    client = client_with_wf
    db = app.dependency_overrides[wf_mod._get_db]()
    for index in range(16):
        db.create_run(
            run_id=f"ws-run-{index}",
            tenant_id="default",
            user_id="1" if index == 0 else f"other-{index}",
            inputs={},
            workflow_id=None,
            definition_version=1,
            definition_snapshot={"name": "concurrent-ws", "version": 1, "steps": []},
        )
    token = wf_mod.get_jwt_manager().create_access_token(subject="1", username="tester", roles=["user"], permissions=[])
    run_id = {"owned": "ws-run-0", "foreign": "ws-run-1", "missing": "missing-run"}[target]
    stop = Event()
    ready = Barrier(9, timeout=10)

    def read_runs(index):
        ready.wait()
        while not stop.is_set():
            db.get_run(f"ws-run-{index}")

    with ThreadPoolExecutor(max_workers=8) as executor:
        readers = [executor.submit(read_runs, index) for index in range(8)]
        try:
            ready.wait()
            for _ in range(20):
                if close_code is None:
                    with client.websocket_connect(f"/api/v1/workflows/ws?run_id={run_id}&token={token}") as ws:
                        assert ws.receive_json()["run"]["run_id"] == "ws-run-0"
                else:
                    with pytest.raises(WebSocketDisconnect) as rejection:
                        with client.websocket_connect(f"/api/v1/workflows/ws?run_id={run_id}&token={token}"):
                            pytest.fail("Unauthorized WebSocket was accepted")
                    assert rejection.value.code == close_code
        finally:
            stop.set()
            for reader in readers:
                reader.result(timeout=10)


@pytest.mark.parametrize(
    "scenario, expected_code, expected_reason",
    [
        ("missing_credentials", 4401, "authentication_required"),
        ("invalid_token", 4401, "invalid_token"),
        ("missing_run", 4404, "run_not_found"),
        ("foreign_run", 4403, "not_run_owner"),
        ("cookie_rejected", 4403, "cookie_auth_rejected"),
    ],
)
def test_ws_rejections_have_safe_diagnostics(client_with_wf, monkeypatch, scenario, expected_code, expected_reason):
    db = app.dependency_overrides[wf_mod._get_db]()
    run_id = "private-run-do-not-log"
    db.create_run(
        run_id=run_id,
        tenant_id="default",
        user_id="1",
        inputs={"secret": "private-input-do-not-log"},  # nosec B105: synthetic leak-detection sentinel.
        workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": "diagnostics", "version": 1, "steps": []},
    )
    token = wf_mod.get_jwt_manager().create_access_token(
        subject="2" if scenario == "foreign_run" else "1", username="tester", roles=["user"], permissions=[]
    )
    if scenario == "invalid_token":
        token = "private-token-do-not-log"  # nosec B105: deliberately invalid test credential.
    if scenario == "cookie_rejected":

        async def reject_cookie(websocket):
            websocket.state.single_user_cookie_websocket_close_code = 4403
            return None

        monkeypatch.setattr(wf_mod, "resolve_single_user_cookie_websocket", reject_cookie)
    if scenario == "missing_run":
        run_id = "missing-private-run-do-not-log"
    query = f"run_id={run_id}"
    if scenario not in {"missing_credentials", "cookie_rejected"}:
        query += f"&token={token}"
    records = []
    sink = wf_mod.logger.add(
        lambda message: records.append(message.record),
        filter=lambda record: record["extra"].get("endpoint") == "workflows_ws",
    )
    try:
        with pytest.raises(WebSocketDisconnect) as rejection:
            with client_with_wf.websocket_connect(f"/api/v1/workflows/ws?{query}"):
                pytest.fail("Rejected WebSocket was accepted")
    finally:
        wf_mod.logger.remove(sink)
    assert rejection.value.code == expected_code
    assert rejection.value.reason == expected_reason
    assert len(records) == 1
    record = records[0]
    assert record["level"].name == "WARNING"
    assert record["extra"]["close_code"] == expected_code
    assert record["extra"]["reason"] == expected_reason
    assert f"code={expected_code}" in record["message"]
    assert expected_reason in record["message"]
    diagnostic = record["message"] + str(record["extra"])
    assert all(secret not in diagnostic for secret in (token, run_id, "private-input-do-not-log"))
    assert record["exception"] is None


def test_ws_stream_failure_diagnostics_exclude_exception_contents(client_with_wf, monkeypatch):
    db = app.dependency_overrides[wf_mod._get_db]()
    db.create_run(
        run_id="diagnostic-stream",
        tenant_id="default",
        user_id="1",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": "diagnostics", "version": 1, "steps": []},
    )

    def fail_events(*args, **kwargs):
        raise ValueError("private-exception-do-not-log")

    monkeypatch.setattr(db, "get_events", fail_events)
    token = wf_mod.get_jwt_manager().create_access_token(subject="1", username="tester", roles=["user"], permissions=[])
    records = []
    sink = wf_mod.logger.add(
        lambda message: records.append(message.record),
        filter=lambda record: record["extra"].get("endpoint") == "workflows_ws",
    )
    try:
        with client_with_wf.websocket_connect(f"/api/v1/workflows/ws?run_id=diagnostic-stream&token={token}") as ws:
            assert ws.receive_json()["type"] == "snapshot"
            with pytest.raises(WebSocketDisconnect) as failure:
                ws.receive_json()
    finally:
        wf_mod.logger.remove(sink)
    assert failure.value.code == 1011
    assert failure.value.reason == "stream_error"
    record = next(record for record in records if record["extra"].get("reason") == "stream_error")
    assert record["extra"]["error_type"] == "ValueError"
    assert "private-exception-do-not-log" not in record["message"]
    assert token not in record["message"]
    assert record["exception"] is None


def test_ws_lookup_failure_has_safe_diagnostics(client_with_wf, monkeypatch):
    db = app.dependency_overrides[wf_mod._get_db]()

    def fail_lookup(*args, **kwargs):
        raise ValueError("private-lookup-do-not-log")

    monkeypatch.setattr(db, "get_run", fail_lookup)
    token = wf_mod.get_jwt_manager().create_access_token(subject="1", username="tester", roles=["user"], permissions=[])
    records = []
    sink = wf_mod.logger.add(
        lambda message: records.append(message.record),
        filter=lambda record: record["extra"].get("endpoint") == "workflows_ws",
    )
    try:
        with pytest.raises(WebSocketDisconnect) as failure:
            with client_with_wf.websocket_connect(f"/api/v1/workflows/ws?run_id=diagnostic-lookup&token={token}"):
                pytest.fail("Failed lookup was accepted")
    finally:
        wf_mod.logger.remove(sink)
    assert failure.value.code == 1011
    assert failure.value.reason == "run_lookup_error"
    assert len(records) == 1
    assert records[0]["extra"]["error_type"] == "ValueError"
    assert "private-lookup-do-not-log" not in records[0]["message"]
    assert records[0]["exception"] is None


@pytest.mark.parametrize("read_method", ["get_run", "get_events"])
def test_ws_pending_database_read_keeps_http_responsive(client_with_wf, monkeypatch, read_method):
    """A pending disk read must not hold the ASGI loop needed by unrelated requests."""
    client = client_with_wf
    db = app.dependency_overrides[wf_mod._get_db]()
    run_id = uuid4().hex
    db.create_run(
        run_id=run_id,
        tenant_id="default",
        user_id="1",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": "pending-read", "version": 1, "steps": []},
    )
    token = wf_mod.get_jwt_manager().create_access_token(subject="1")
    entered, http_completed = Event(), Event()
    released_by_http = []
    original = getattr(db, read_method)

    def pending_read(*args, **kwargs):
        entered.set()
        released_by_http.append(http_completed.wait(timeout=5))
        return original(*args, **kwargs)

    monkeypatch.setattr(db, read_method, pending_read)

    def receive_snapshot():
        with client.websocket_connect(f"/api/v1/workflows/ws?run_id={run_id}&token={token}") as ws:
            return ws.receive_json()

    with ThreadPoolExecutor(max_workers=1) as executor:
        connected = executor.submit(receive_snapshot)
        try:
            assert entered.wait(timeout=10), "WebSocket did not reach the database read"
            response = client.get("/health")
        finally:
            http_completed.set()
        snapshot = connected.result(timeout=10)
    assert response.status_code == 200
    assert snapshot["run"]["run_id"] == run_id
    assert released_by_http and all(released_by_http), "Database read blocked the ASGI event loop"


@pytest.mark.asyncio
@pytest.mark.parametrize("close_code", [1000, 1001, 1006])
async def test_ws_peer_disconnect_finishes_without_asgi_exception(client_with_wf, monkeypatch, close_code):
    """A closed transport is normal endpoint completion, not an unhandled server error."""
    db = app.dependency_overrides[wf_mod._get_db]()
    run_id = uuid4().hex
    db.create_run(
        run_id=run_id,
        tenant_id="default",
        user_id="1",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": "peer-disconnect", "version": 1, "steps": []},
    )
    token = wf_mod.get_jwt_manager().create_access_token(subject="1")

    async def receive():
        return {"type": "websocket.connect"}

    sent = []

    async def send(message):
        sent.append(message)

    async def disconnected_transport(self, payload):
        raise WebSocketDisconnect(close_code)

    monkeypatch.setattr(wf_mod.WebSocketStream, "send_json", disconnected_transport)
    records = []
    sink = wf_mod.logger.add(
        lambda message: records.append(message.record),
        filter=lambda record: record["extra"].get("endpoint") == "workflows_ws",
    )
    try:
        await wf_mod.workflows_ws(
            WebSocket({"type": "websocket", "headers": []}, receive, send),
            run_id,
            token=token,
            types=None,
            db=db,
        )
    finally:
        wf_mod.logger.remove(sink)
    assert sent == [{"type": "websocket.accept", "subprotocol": None, "headers": []}]
    assert len(records) == 1
    assert records[0]["level"].name == "INFO"
    assert records[0]["extra"]["reason"] == "peer_disconnected"
    assert records[0]["extra"]["close_code"] == close_code
    assert records[0]["exception"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("clock_resolution", [0.0001, 0.015625])
async def test_ws_history_burst_allows_other_loop_work(client_with_wf, monkeypatch, clock_resolution):
    """A writable transport must not let a whole history page monopolize the loop."""
    db = app.dependency_overrides[wf_mod._get_db]()
    run_id = uuid4().hex
    db.create_run(
        run_id=run_id,
        tenant_id="default",
        user_id="1",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": "history-fairness", "version": 1, "steps": []},
    )
    for index in range(500):
        db.append_event("default", run_id, "history", {"index": index})
    token = wf_mod.get_jwt_manager().create_access_token(subject="1")
    event_sequences, probe_counts = [], []
    loop = asyncio.get_running_loop()
    clock = types.SimpleNamespace(now=0.0)
    budget = wf_mod._workflow_history_budget()
    # Control only the work clock; keep real event-loop scheduling and the shared budget.
    monkeypatch.setattr(
        budget,
        "_loop",
        types.SimpleNamespace(
            time=lambda: int(clock.now / clock_resolution) * clock_resolution,
            call_later=loop.call_later,
        ),
    )

    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        # Real writable sockets can complete sends without suspending the task.
        if message["type"] == "websocket.send":
            frame = json.loads(message["text"])
            if "event_seq" in frame:
                event_sequences.append(frame["event_seq"])
                clock.now += 0.0004
                if len(event_sequences) == 1:
                    asyncio.get_running_loop().call_soon(lambda: probe_counts.append(len(event_sequences)))
                if len(event_sequences) == 500:
                    raise WebSocketDisconnect(1000)

    await wf_mod.workflows_ws(
        WebSocket({"type": "websocket", "headers": []}, receive, send),
        run_id,
        token=token,
        types=None,
        db=db,
    )
    await asyncio.sleep(0)
    assert event_sequences == list(range(1, 501))
    assert probe_counts and probe_counts[0] < 500, "History send monopolized the event loop"
