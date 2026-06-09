import importlib.machinery
import builtins
import json
import sys
import time
import types
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
from tldw_Server_API.app.core.MCP_unified.auth.jwt_manager import get_jwt_manager
from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService
from tldw_Server_API.app.core.AuthNZ.settings import get_settings as get_auth_settings
from tldw_Server_API.app.api.v1.endpoints import workflows as wf_mod
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user


pytestmark = pytest.mark.integration


@pytest.fixture()
def client_with_workflows_db(tmp_path, auth_headers):
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))

    async def override_user():
        return User(
            id=1,
            username="tester",
            email="t@e.com",
            is_active=True,
            is_admin=True,
            roles=["admin"],
        )

    def override_db():

        return db

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[wf_mod._get_db] = override_db

    with TestClient(app, headers=auth_headers) as client:
        yield client

    app.dependency_overrides.clear()


def _wait_for_run_row(db: WorkflowsDatabase, run_id: str, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = db.get_run(run_id)
        if run is not None:
            return run
        time.sleep(0.02)
    pytest.fail(f"workflow run row was not created for {run_id}")


def test_create_and_run_saved_workflow(client_with_workflows_db: TestClient):
    client = client_with_workflows_db

    definition = {
        "name": "hello",
        "version": 1,
        "steps": [
            {"id": "s1", "type": "prompt", "config": {"template": "Hello {{ inputs.name }}"}},
        ],
    }

    # Create
    resp = client.post("/api/v1/workflows", json=definition)
    assert resp.status_code == 201, resp.text
    wid = resp.json()["id"]

    # Run
    run_resp = client.post(f"/api/v1/workflows/{wid}/run?mode=async", json={"inputs": {"name": "Alice"}})
    assert run_resp.status_code == 200, run_resp.text
    run_id = run_resp.json()["run_id"]

    # Poll until complete
    for _ in range(50):
        r = client.get(f"/api/v1/workflows/runs/{run_id}")
        assert r.status_code == 200
        data = r.json()
        if data["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.05)
    assert data["status"] == "succeeded"
    assert (data.get("outputs") or {}).get("text") == "Hello Alice"

    # Events include run_completed
    ev = client.get(f"/api/v1/workflows/runs/{run_id}/events")
    assert ev.status_code == 200
    types = [e["event_type"] for e in ev.json()]
    assert "run_completed" in types


def test_run_adhoc_requires_workflows_scope_like_saved_run(tmp_path):
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    workflow_id = db.create_definition(
        tenant_id="default",
        name="scope-check",
        version=1,
        owner_id="1",
        visibility="private",
        description=None,
        tags=[],
        definition={
            "name": "scope-check",
            "version": 1,
            "steps": [{"id": "s1", "type": "prompt", "config": {"template": "hi"}}],
        },
    )

    async def override_user():
        return User(
            id=1,
            username="tester",
            email="t@e.com",
            is_active=True,
            is_admin=False,
            roles=["user"],
            tenant_id="default",
        )

    def override_db():
        return db

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[wf_mod._get_db] = override_db

    jwt_service = JWTService(get_auth_settings())
    token = jwt_service.create_virtual_access_token(
        user_id=1,
        username="tester",
        role="user",
        scope="not-workflows",
        ttl_minutes=5,
    )

    headers = {"Authorization": f"Bearer {token}"}
    client = TestClient(app, headers=headers)
    try:
        saved_resp = client.post(f"/api/v1/workflows/{workflow_id}/run", json={"inputs": {}})
        adhoc_resp = client.post(
            "/api/v1/workflows/run",
            json={
                "definition": {
                    "name": "adhoc",
                    "version": 1,
                    "steps": [{"id": "s1", "type": "prompt", "config": {"template": "hi"}}],
                },
                "inputs": {},
            },
        )
    finally:
        client.close()
        app.dependency_overrides.clear()

    assert saved_resp.status_code == 403
    assert adhoc_resp.status_code == 403


def test_run_adhoc_rejects_virtual_key_excluded_by_endpoint_allowlist(tmp_path):
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))

    async def override_user():
        return User(
            id=1,
            username="tester",
            email="t@e.com",
            is_active=True,
            is_admin=False,
            roles=["user"],
            tenant_id="default",
        )

    def override_db():
        return db

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[wf_mod._get_db] = override_db

    jwt_service = JWTService(get_auth_settings())
    token = jwt_service.create_virtual_access_token(
        user_id=1,
        username="tester",
        role="user",
        scope="workflows",
        ttl_minutes=5,
        additional_claims={"allowed_endpoints": ["workflows.run_saved"]},
    )

    headers = {"Authorization": f"Bearer {token}"}
    client = TestClient(app, headers=headers)
    try:
        adhoc_resp = client.post(
            "/api/v1/workflows/run",
            json={
                "definition": {
                    "name": "adhoc",
                    "version": 1,
                    "steps": [{"id": "s1", "type": "prompt", "config": {"template": "hi"}}],
                },
                "inputs": {},
            },
        )
    finally:
        client.close()
        app.dependency_overrides.clear()

    assert adhoc_resp.status_code == 403


def test_adhoc_limits_and_validation(client_with_workflows_db: TestClient):
    client = client_with_workflows_db

    # Unknown step type
    bad_def = {
        "definition": {
            "name": "x",
            "version": 1,
            "steps": [{"id": "s", "type": "unknown", "config": {}}],
        },
        "inputs": {},
    }
    r = client.post("/api/v1/workflows/run", json=bad_def)
    assert r.status_code == 422

    # Too large definition (>256KB)
    large_payload = "x" * (260 * 1024)
    big_def = {
        "definition": {
            "name": "big",
            "version": 1,
            "steps": [{"id": "s", "type": "prompt", "config": {"template": large_payload}}],
        }
    }
    r2 = client.post("/api/v1/workflows/run", json=big_def)
    assert r2.status_code == 413


def test_websocket_auth_and_events(client_with_workflows_db: TestClient):
    client = client_with_workflows_db

    # Create def and run
    definition = {
        "name": "hello-ws",
        "version": 1,
        "steps": [{"id": "s1", "type": "prompt", "config": {"template": "Hi {{ inputs.name }}"}}],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {"name": "Bob"}}).json()["run_id"]

    # Token for user id 1
    jwtm = get_jwt_manager()
    token = jwtm.create_access_token(subject="1", username="tester", roles=["user"], permissions=[])

    # Connect and receive snapshot
    with client.websocket_connect(f"/api/v1/workflows/ws?run_id={run_id}&token={token}") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "snapshot"

    # Unauthorized with wrong user
    bad_token = jwtm.create_access_token(subject="999", username="intruder", roles=["user"], permissions=[])
    with pytest.raises(Exception):
        with client.websocket_connect(f"/api/v1/workflows/ws?run_id={run_id}&token={bad_token}"):
            pass


def test_step_types_and_runs_listing(client_with_workflows_db: TestClient):
    client = client_with_workflows_db

    # Step types discovery
    st = client.get("/api/v1/workflows/step-types")
    assert st.status_code == 200
    items = st.json()
    names = [i.get("name") for i in items]
    assert "prompt" in names and "rag_search" in names
    assert "deep_research" in names

    # Create two definitions: one succeeds, one fails
    ok_def = {
        "name": "list-ok",
        "version": 1,
        "steps": [{"id": "s1", "type": "prompt", "config": {"template": "Hello"}}],
    }
    fail_def = {
        "name": "list-fail",
        "version": 1,
        "steps": [{"id": "s1", "type": "prompt", "config": {"template": "bad", "force_error": True}}],
    }
    ok_id = client.post("/api/v1/workflows", json=ok_def).json()["id"]
    fail_id = client.post("/api/v1/workflows", json=fail_def).json()["id"]

    ok_run = client.post(f"/api/v1/workflows/{ok_id}/run", json={"inputs": {}}).json()["run_id"]
    fail_run = client.post(f"/api/v1/workflows/{fail_id}/run", json={"inputs": {}}).json()["run_id"]

    import time
    # Wait for terminal statuses
    deadline = time.time() + 5
    statuses = {}
    while time.time() < deadline and (len(statuses) < 2):
        for rid in (ok_run, fail_run):
            if rid in statuses:
                continue
            data = client.get(f"/api/v1/workflows/runs/{rid}").json()
            if data["status"] in ("succeeded", "failed", "cancelled"):
                statuses[rid] = data["status"]
        time.sleep(0.05)
    assert statuses.get(ok_run) == "succeeded"
    assert statuses.get(fail_run) == "failed"

    # List succeeded
    lst_ok = client.get("/api/v1/workflows/runs", params={"status": ["succeeded"], "limit": 10, "offset": 0}).json()
    assert any(r["run_id"] == ok_run for r in lst_ok.get("runs", []))
    # owner exposed for admin review (present even for single-user)
    assert "user_id" in lst_ok.get("runs", [])[0]
    # List failed
    lst_fail = client.get("/api/v1/workflows/runs", params=[("status", "failed")]).json()
    assert any(r["run_id"] == fail_run for r in lst_fail.get("runs", []))
    # Ordering and pagination
    lst_page = client.get("/api/v1/workflows/runs", params={"limit": 1, "offset": 0, "order_by": "created_at", "order": "desc"}).json()
    assert isinstance(lst_page.get("runs"), list) and len(lst_page["runs"]) == 1
    if lst_page.get("next_offset") is not None:
        lst_next = client.get("/api/v1/workflows/runs", params={"limit": 1, "offset": lst_page["next_offset"]}).json()
        assert isinstance(lst_next.get("runs"), list)


def test_runs_list_created_after_before(client_with_workflows_db: TestClient):
    client = client_with_workflows_db
    # Create two runs with a small time gap
    d = {"name": "time-filter", "version": 1, "steps": [{"id": "s1", "type": "prompt", "config": {"template": "ok"}}]}
    wid = client.post("/api/v1/workflows", json=d).json()["id"]
    r1 = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    import time
    time.sleep(0.05)
    r2 = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]

    # Extract created_at via DB
    from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
    from tldw_Server_API.app.api.v1.endpoints import workflows as wf_mod
    db: WorkflowsDatabase = client.app.dependency_overrides[wf_mod._get_db]()
    _wait_for_run_row(db, r1)
    _wait_for_run_row(db, r2)

    c1 = "2026-01-01T00:00:00+00:00"
    c2 = "2026-01-01T00:00:01+00:00"
    db._conn.execute("UPDATE workflow_runs SET created_at = ? WHERE run_id = ?", (c1, r1))
    db._conn.execute("UPDATE workflow_runs SET created_at = ? WHERE run_id = ?", (c2, r2))
    db._conn.commit()

    # created_after just after c1 should include r2 only
    from datetime import datetime, timedelta
    try:
        dt1 = datetime.fromisoformat(c1)
    except Exception:
        dt1 = datetime.strptime(c1.split('.') [0], "%Y-%m-%dT%H:%M:%S")
    ca = (dt1 + timedelta(milliseconds=1)).isoformat()
    lst_after = client.get("/api/v1/workflows/runs", params={"created_after": ca}).json()
    after_ids = [r["run_id"] for r in lst_after.get("runs", [])]
    assert r2 in after_ids
    # created_before just before c2 should include r1 only (if ordering tight)
    try:
        dt2 = datetime.fromisoformat(c2)
    except Exception:
        dt2 = datetime.strptime(c2.split('.') [0], "%Y-%m-%dT%H:%M:%S")
    cb = (dt2 - timedelta(milliseconds=1)).isoformat()
    lst_before = client.get("/api/v1/workflows/runs", params={"created_before": cb}).json()
    before_ids = [r["run_id"] for r in lst_before.get("runs", [])]
    assert r1 in before_ids


def test_runs_multi_status_and_ordering(client_with_workflows_db: TestClient):
    client = client_with_workflows_db
    # Prepare: one success, one failure
    ok_def = {"name": "ms-ok", "version": 1, "steps": [{"id": "s1", "type": "prompt", "config": {"template": "ok"}}]}
    bad_def = {"name": "ms-bad", "version": 1, "steps": [{"id": "s1", "type": "prompt", "config": {"template": "bad", "force_error": True}}]}
    wid_ok = client.post("/api/v1/workflows", json=ok_def).json()["id"]
    wid_bad = client.post("/api/v1/workflows", json=bad_def).json()["id"]
    rid_ok = client.post(f"/api/v1/workflows/{wid_ok}/run", json={"inputs": {}}).json()["run_id"]
    import time as _t
    _t.sleep(0.02)
    rid_fail = client.post(f"/api/v1/workflows/{wid_bad}/run", json={"inputs": {}}).json()["run_id"]

    # Wait for terminal
    deadline = _t.time() + 5
    while _t.time() < deadline:
        st_ok = client.get(f"/api/v1/workflows/runs/{rid_ok}").json()["status"]
        st_fail = client.get(f"/api/v1/workflows/runs/{rid_fail}").json()["status"]
        if st_ok in ("succeeded","failed","cancelled") and st_fail in ("succeeded","failed","cancelled"):
            break
        _t.sleep(0.02)

    # Filter for both statuses together
    both = client.get("/api/v1/workflows/runs", params=[("status","succeeded"),("status","failed"), ("order","asc"), ("order_by","created_at"), ("limit", 50)]).json()
    ids = [r["run_id"] for r in both.get("runs", [])]
    assert rid_ok in ids and rid_fail in ids
    # Asc ordering should have the earlier run first
    idx_ok = ids.index(rid_ok)
    idx_fail = ids.index(rid_fail)
    assert idx_ok < idx_fail


def test_artifact_download_scope_non_strict(monkeypatch, client_with_workflows_db: TestClient):
    client = client_with_workflows_db
    # Relax validation to non-strict
    monkeypatch.setenv("WORKFLOWS_ARTIFACT_VALIDATE_STRICT", "false")

    # Create definition and run to own the artifact
    d = {"name": "artifacts", "version": 1, "steps": [{"id": "s1", "type": "prompt", "config": {"template": "ok"}}]}
    wid = client.post("/api/v1/workflows", json=d).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]

    # Prepare a temp file outside of the recorded workdir
    import os, tempfile, pathlib
    fd, path = tempfile.mkstemp(prefix="wf_non_strict_")
    os.write(fd, b"hello")
    os.close(fd)
    uri = f"file://{path}"

    # Add artifact with a different workdir in metadata to trigger scope mismatch
    from tldw_Server_API.app.api.v1.endpoints import workflows as wf_mod
    db = client.app.dependency_overrides[wf_mod._get_db]()
    art_id = "a_non_strict"
    db.add_artifact(
        artifact_id=art_id,
        tenant_id="default",
        run_id=run_id,
        step_run_id=None,
        type="file",
        uri=uri,
        size_bytes=5,
        mime_type="text/plain",
        metadata={"workdir": str(pathlib.Path.cwd())},
    )
    # Download should succeed due to non-strict
    r = client.get(f"/api/v1/workflows/artifacts/{art_id}/download")
    assert r.status_code == 200
    # Now strict mode should block
    monkeypatch.setenv("WORKFLOWS_ARTIFACT_VALIDATE_STRICT", "true")
    r2 = client.get(f"/api/v1/workflows/artifacts/{art_id}/download")
    assert r2.status_code == 400


def test_artifact_download_per_run_non_block(monkeypatch, client_with_workflows_db: TestClient):
    """Per-run validation_mode='non-block' should allow download even when env is strict."""
    client = client_with_workflows_db
    # Ensure env strict mode
    monkeypatch.setenv("WORKFLOWS_ARTIFACT_VALIDATE_STRICT", "true")

    # Create definition and run
    d = {"name": "artifacts2", "version": 1, "steps": [{"id": "s1", "type": "prompt", "config": {"template": "ok"}}]}
    wid = client.post("/api/v1/workflows", json=d).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]

    # Prepare a temp file outside of the recorded workdir
    import os, tempfile, pathlib
    fd, path = tempfile.mkstemp(prefix="wf_non_block_")
    os.write(fd, b"hello2")
    os.close(fd)
    uri = f"file://{path}"

    # Add artifact with different workdir to trigger scope mismatch
    from tldw_Server_API.app.api.v1.endpoints import workflows as wf_mod
    db = client.app.dependency_overrides[wf_mod._get_db]()
    art_id = "a_per_run"
    db.add_artifact(
        artifact_id=art_id,
        tenant_id="default",
        run_id=run_id,
        step_run_id=None,
        type="file",
        uri=uri,
        size_bytes=6,
        mime_type="text/plain",
        metadata={"workdir": str(pathlib.Path.cwd())},
    )

    # First, strict mode should block
    r_block = client.get(f"/api/v1/workflows/artifacts/{art_id}/download")
    assert r_block.status_code == 400

    # Patch get_run to return validation_mode='non-block' for this run
    orig_get_run = db.get_run
    def _patched_get_run(rid: str):
        run = orig_get_run(rid)
        if run and run.run_id == run_id:
            try:
                setattr(run, 'validation_mode', 'non-block')
            except Exception:
                _ = None
        return run
    monkeypatch.setattr(db, 'get_run', _patched_get_run, raising=False)

    # Now should succeed even with strict env due to per-run override
    r_nonblock = client.get(f"/api/v1/workflows/artifacts/{art_id}/download")
    assert r_nonblock.status_code == 200


def test_runs_admin_owner_filter(client_with_workflows_db: TestClient):
    """Admins can filter by owner; non-admins cannot override owner param."""
    client = client_with_workflows_db

    # Create one run as admin (user id=1)
    def_admin = {"name": "admin-def", "version": 1, "steps": [{"id": "s1", "type": "prompt", "config": {"template": "hi"}}]}
    wid_admin = client.post("/api/v1/workflows", json=def_admin).json()["id"]
    r_admin = client.post(f"/api/v1/workflows/{wid_admin}/run", json={"inputs": {}}).json()["run_id"]

    # Swap dependency to simulate a different, non-admin owner (user id=2)
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User as _User, get_request_user as _gru
    from tldw_Server_API.app.api.v1.endpoints import workflows as _wf_mod
    prev_dep = client.app.dependency_overrides[_gru]

    async def override_user2():
        return _User(id=2, username="owner2", email="o2@x", is_active=True, is_admin=False)

    client.app.dependency_overrides[_gru] = override_user2
    def_u2 = {"name": "u2-def", "version": 1, "steps": [{"id": "s1", "type": "prompt", "config": {"template": "yo"}}]}
    wid_u2 = client.post("/api/v1/workflows", json=def_u2).json()["id"]
    r_u2 = client.post(f"/api/v1/workflows/{wid_u2}/run", json={"inputs": {}}).json()["run_id"]

    # Restore admin for listing operations
    client.app.dependency_overrides[_gru] = prev_dep


def test_step_types_includes_branch_map(client_with_workflows_db: TestClient):
    client = client_with_workflows_db
    r = client.get("/api/v1/workflows/step-types")
    assert r.status_code == 200
    names = [s.get("name") for s in r.json()]
    assert "branch" in names
    assert "map" in names
    # Schemas include example and min_engine_version
    for st in r.json():
        assert "schema" in st
        assert "min_engine_version" in st


def test_step_types_includes_acp_stage(client_with_workflows_db: TestClient):
    client = client_with_workflows_db
    r = client.get("/api/v1/workflows/step-types")
    assert r.status_code == 200
    acp = next((item for item in r.json() if item.get("name") == "acp_stage"), None)
    assert acp is not None
    schema = acp.get("schema") or {}
    properties = schema.get("properties") or {}
    assert "stage" in properties
    assert "workspace_id" in properties
    assert "workspace_group_id" in properties


def test_step_types_includes_deep_research_select_bundle_fields(client_with_workflows_db: TestClient):
    client = client_with_workflows_db
    response = client.get("/api/v1/workflows/step-types")
    assert response.status_code == 200
    selector = next(
        (
            item
            for item in response.json()
            if item.get("name") == "deep_research_select_bundle_fields"
        ),
        None,
    )
    assert selector is not None
    schema = selector.get("schema") or {}
    properties = schema.get("properties") or {}
    assert "fields" in properties
    assert properties["fields"]["type"] == "array"


def test_create_workflow_accepts_acp_stage_definition(client_with_workflows_db: TestClient):
    client = client_with_workflows_db
    definition = {
        "name": "acp-stage-definition",
        "version": 1,
        "steps": [
            {
                "id": "a1",
                "type": "acp_stage",
                "config": {
                    "stage": "impl",
                    "prompt_template": "Implement {{ inputs.task }}",
                },
            }
        ],
    }
    resp = client.post("/api/v1/workflows", json=definition)
    assert resp.status_code in (200, 201), resp.text


def test_create_workflow_rejects_invalid_deep_research_definition(client_with_workflows_db: TestClient):
    client = client_with_workflows_db
    definition = {
        "name": "invalid-deep-research-definition",
        "version": 1,
        "steps": [
            {
                "id": "r1",
                "type": "deep_research",
                "config": {
                    "query": "launch research",
                    "source_policy": "unsupported_policy",
                },
            }
        ],
    }
    resp = client.post("/api/v1/workflows", json=definition)
    assert resp.status_code == 422


def test_create_workflow_rejects_invalid_deep_research_wait_definition(client_with_workflows_db: TestClient):
    client = client_with_workflows_db
    definition = {
        "name": "invalid-deep-research-wait-definition",
        "version": 1,
        "steps": [
            {
                "id": "rw1",
                "type": "deep_research_wait",
                "config": {
                    "poll_interval_seconds": 0,
                    "include_bundle": True,
                },
            }
        ],
    }
    resp = client.post("/api/v1/workflows", json=definition)
    assert resp.status_code == 422


def test_create_workflow_rejects_invalid_deep_research_wait_definition_without_jsonschema(
    monkeypatch,
    client_with_workflows_db: TestClient,
):
    client = client_with_workflows_db
    original_import = builtins.__import__

    def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "jsonschema":
            raise ImportError("simulated missing jsonschema")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _patched_import)

    definition = {
        "name": "invalid-deep-research-wait-definition-no-jsonschema",
        "version": 1,
        "steps": [
            {
                "id": "rw1",
                "type": "deep_research_wait",
                "config": {
                    "run": {"console_url": "/research?run=missing"},
                    "poll_interval_seconds": 0,
                },
            }
        ],
    }

    resp = client.post("/api/v1/workflows", json=definition)
    assert resp.status_code == 422


def test_create_workflow_rejects_invalid_deep_research_load_bundle_definition(
    client_with_workflows_db: TestClient,
):
    client = client_with_workflows_db
    definition = {
        "name": "invalid-deep-research-load-bundle-definition",
        "version": 1,
        "steps": [
            {
                "id": "rl1",
                "type": "deep_research_load_bundle",
                "config": {
                    "run": {"bundle_url": "/api/v1/research/runs/missing/bundle"},
                },
            }
        ],
    }
    resp = client.post("/api/v1/workflows", json=definition)
    assert resp.status_code == 422


def test_create_workflow_rejects_invalid_deep_research_load_bundle_definition_without_jsonschema(
    monkeypatch,
    client_with_workflows_db: TestClient,
):
    client = client_with_workflows_db
    original_import = builtins.__import__

    def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "jsonschema":
            raise ImportError("simulated missing jsonschema")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _patched_import)

    definition = {
        "name": "invalid-deep-research-load-bundle-definition-no-jsonschema",
        "version": 1,
        "steps": [
            {
                "id": "rl1",
                "type": "deep_research_load_bundle",
                "config": {
                    "run": {"bundle_url": "/api/v1/research/runs/missing/bundle"},
                },
            }
        ],
    }

    resp = client.post("/api/v1/workflows", json=definition)
    assert resp.status_code == 422


def test_create_workflow_rejects_invalid_deep_research_select_bundle_fields_definition(
    client_with_workflows_db: TestClient,
):
    client = client_with_workflows_db
    definition = {
        "name": "invalid-deep-research-select-bundle-fields-definition",
        "version": 1,
        "steps": [
            {
                "id": "rs1",
                "type": "deep_research_select_bundle_fields",
                "config": {
                    "run_id": "{{ deep_research_wait.run_id }}",
                    "fields": ["question", "not_allowed"],
                },
            }
        ],
    }
    resp = client.post("/api/v1/workflows", json=definition)
    assert resp.status_code == 422


def test_create_workflow_rejects_unknown_config_key_for_deep_research_select_bundle_fields(
    client_with_workflows_db: TestClient,
):
    client = client_with_workflows_db
    definition = {
        "name": "invalid-deep-research-select-bundle-fields-extra-key",
        "version": 1,
        "steps": [
            {
                "id": "rs1",
                "type": "deep_research_select_bundle_fields",
                "config": {
                    "run_id": "{{ deep_research_wait.run_id }}",
                    "fields": ["question"],
                    "unknown_flag": True,
                },
            }
        ],
    }
    resp = client.post("/api/v1/workflows", json=definition)
    assert resp.status_code == 422


def test_create_workflow_rejects_invalid_deep_research_select_bundle_fields_definition_without_jsonschema(
    monkeypatch,
    client_with_workflows_db: TestClient,
):
    client = client_with_workflows_db
    original_import = builtins.__import__

    def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "jsonschema":
            raise ImportError("simulated missing jsonschema")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _patched_import)

    definition = {
        "name": "invalid-deep-research-select-bundle-fields-no-jsonschema",
        "version": 1,
        "steps": [
            {
                "id": "rs1",
                "type": "deep_research_select_bundle_fields",
                "config": {
                    "run": {"run_id": "research-session-8"},
                    "fields": ["question"],
                    "unknown_flag": True,
                },
            }
        ],
    }

    resp = client.post("/api/v1/workflows", json=definition)
    assert resp.status_code == 422


def test_run_workflow_launches_deep_research_session(monkeypatch, client_with_workflows_db: TestClient):
    client = client_with_workflows_db
    captured: dict[str, object] = {}

    class _FakeSession:
        id = "research-session-1"
        status = "queued"
        phase = "drafting_plan"
        control_state = "running"

    class _FakeResearchService:
        def create_session(self, **kwargs):
            captured["create_session_kwargs"] = kwargs
            return _FakeSession()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.research.launch._build_research_service",
        lambda: _FakeResearchService(),
    )

    definition = {
        "name": "launch-deep-research",
        "version": 1,
        "steps": [
            {
                "id": "r1",
                "type": "deep_research",
                "config": {
                    "query": "{{ inputs.topic }}",
                    "source_policy": "balanced",
                    "autonomy_mode": "checkpointed",
                },
            }
        ],
    }

    create = client.post("/api/v1/workflows", json=definition)
    assert create.status_code == 201, create.text
    wid = create.json()["id"]

    run_id = client.post(
        f"/api/v1/workflows/{wid}/run",
        json={"inputs": {"topic": "evidence-backed forecasting"}},
    ).json()["run_id"]

    deadline = time.time() + 5
    data = {}
    while time.time() < deadline:
        data = client.get(f"/api/v1/workflows/runs/{run_id}").json()
        if data["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.05)

    assert data["status"] == "succeeded"
    assert (data.get("outputs") or {}) == {
        "run_id": "research-session-1",
        "status": "queued",
        "phase": "drafting_plan",
        "control_state": "running",
        "console_url": "/research?run=research-session-1",
        "bundle_url": "/api/v1/research/runs/research-session-1/bundle",
        "query": "evidence-backed forecasting",
        "source_policy": "balanced",
        "autonomy_mode": "checkpointed",
    }
    assert captured["create_session_kwargs"] == {
        "owner_user_id": "1",
        "query": "evidence-backed forecasting",
        "source_policy": "balanced",
        "autonomy_mode": "checkpointed",
        "limits_json": None,
        "provider_overrides": None,
    }


def test_run_workflow_waits_for_deep_research_completion(monkeypatch, client_with_workflows_db: TestClient):
    client = client_with_workflows_db
    captured: dict[str, object] = {}

    class _LaunchSession:
        id = "research-session-2"
        status = "queued"
        phase = "drafting_plan"
        control_state = "running"

    class _CompletedSession:
        id = "research-session-2"
        status = "completed"
        phase = "completed"
        control_state = "running"
        completed_at = "2026-03-07T13:30:00+00:00"

    class _LaunchResearchService:
        def create_session(self, **kwargs):
            captured["create_session_kwargs"] = kwargs
            return _LaunchSession()

    class _WaitResearchService:
        def get_session(self, **kwargs):
            captured["get_session_kwargs"] = kwargs
            return _CompletedSession()

        def get_bundle(self, **kwargs):
            captured["get_bundle_kwargs"] = kwargs
            return {"concise_answer": "Bundle ready"}

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.research.launch._build_research_service",
        lambda: _LaunchResearchService(),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.research.wait._build_research_service",
        lambda: _WaitResearchService(),
    )

    definition = {
        "name": "launch-and-wait-deep-research",
        "version": 1,
        "steps": [
            {
                "id": "launch",
                "type": "deep_research",
                "config": {
                    "query": "{{ inputs.topic }}",
                },
            },
            {
                "id": "wait",
                "type": "deep_research_wait",
                "config": {
                    "run_id": "{{ launch.run_id }}",
                    "include_bundle": True,
                },
            },
        ],
    }

    create = client.post("/api/v1/workflows", json=definition)
    assert create.status_code == 201, create.text
    wid = create.json()["id"]

    run_id = client.post(
        f"/api/v1/workflows/{wid}/run",
        json={"inputs": {"topic": "evidence-backed forecasting"}},
    ).json()["run_id"]

    deadline = time.time() + 5
    data = {}
    while time.time() < deadline:
        data = client.get(f"/api/v1/workflows/runs/{run_id}").json()
        if data["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.05)

    assert data["status"] == "succeeded"
    assert (data.get("outputs") or {}) == {
        "run_id": "research-session-2",
        "status": "completed",
        "phase": "completed",
        "control_state": "running",
        "completed_at": "2026-03-07T13:30:00+00:00",
        "bundle_url": "/api/v1/research/runs/research-session-2/bundle",
        "bundle": {"concise_answer": "Bundle ready"},
    }
    assert captured["create_session_kwargs"] == {
        "owner_user_id": "1",
        "query": "evidence-backed forecasting",
        "source_policy": "balanced",
        "autonomy_mode": "checkpointed",
        "limits_json": None,
        "provider_overrides": None,
    }
    assert captured["get_session_kwargs"] == {
        "owner_user_id": "1",
        "session_id": "research-session-2",
    }
    assert captured["get_bundle_kwargs"] == {
        "owner_user_id": "1",
        "session_id": "research-session-2",
    }


def test_run_workflow_pauses_for_deep_research_checkpoint(monkeypatch, client_with_workflows_db: TestClient):
    client = client_with_workflows_db

    class _LaunchSession:
        id = "research-session-12"
        status = "queued"
        phase = "drafting_plan"
        control_state = "running"

    class _CheckpointSession:
        id = "research-session-12"
        status = "waiting_human"
        phase = "awaiting_source_review"
        control_state = "running"
        completed_at = None
        latest_checkpoint_id = "checkpoint-6"

    class _CheckpointSnapshot:
        checkpoint = {
            "checkpoint_id": "checkpoint-6",
            "checkpoint_type": "sources_review",
        }

    class _LaunchResearchService:
        def create_session(self, **kwargs):
            return _LaunchSession()

    class _WaitResearchService:
        def get_session(self, **kwargs):
            return _CheckpointSession()

        def get_stream_snapshot(self, **kwargs):
            return _CheckpointSnapshot()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.research.launch._build_research_service",
        lambda: _LaunchResearchService(),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.research.wait._build_research_service",
        lambda: _WaitResearchService(),
    )

    definition = {
        "name": "launch-and-pause-on-deep-research-checkpoint",
        "version": 1,
        "steps": [
            {
                "id": "launch",
                "type": "deep_research",
                "config": {
                    "query": "{{ inputs.topic }}",
                },
            },
            {
                "id": "wait",
                "type": "deep_research_wait",
                "config": {
                    "run_id": "{{ launch.run_id }}",
                    "include_bundle": False,
                    "poll_interval_seconds": 0.1,
                    "timeout_seconds": 1,
                },
            },
        ],
    }

    create = client.post("/api/v1/workflows", json=definition)
    assert create.status_code == 201, create.text
    wid = create.json()["id"]

    run_id = client.post(
        f"/api/v1/workflows/{wid}/run",
        json={"inputs": {"topic": "evidence-backed forecasting"}},
    ).json()["run_id"]

    deadline = time.time() + 3
    data = {}
    while time.time() < deadline:
        data = client.get(f"/api/v1/workflows/runs/{run_id}").json()
        if data["status"] in ("waiting_human", "waiting_approval", "failed", "cancelled", "succeeded"):
            break
        time.sleep(0.05)

    assert data["status"] == "waiting_human"
    assert (data.get("outputs") or {}) == {
        "__status__": "waiting_human",
        "reason": "research_checkpoint",
        "run_id": "research-session-12",
        "research_phase": "awaiting_source_review",
        "research_control_state": "running",
        "research_checkpoint_id": "checkpoint-6",
        "research_checkpoint_type": "sources_review",
        "research_console_url": "/research?run=research-session-12",
        "active_poll_seconds": pytest.approx(0.0, rel=0.5),
    }


@pytest.mark.asyncio
async def test_resume_workflows_waiting_on_research_checkpoint_resumes_only_matching_links(
    tmp_path,
    monkeypatch,
):
    from tldw_Server_API.app.core.Workflows import research_wait_bridge

    db = WorkflowsDatabase(str(tmp_path / "workflow-research-waits.db"))

    definition = {
        "name": "resume-bridge",
        "version": 1,
        "steps": [
            {
                "id": "wait",
                "type": "deep_research_wait",
                "config": {"run_id": "{{ inputs.run_id }}"},
            }
        ],
    }

    def _seed_waiting_run(run_id: str, research_run_id: str, checkpoint_id: str) -> None:
        db.create_run(
            run_id=run_id,
            tenant_id="default",
            user_id="1",
            inputs={"run_id": research_run_id},
            workflow_id=None,
            definition_version=1,
            definition_snapshot=definition,
        )
        db.update_run_status(run_id, status="waiting_human", status_reason="awaiting_review")
        step_run_id = f"{run_id}:wait:1"
        db.create_step_run(
            step_run_id=step_run_id,
            tenant_id="default",
            run_id=run_id,
            step_id="wait",
            name="wait",
            step_type="deep_research_wait",
        )
        wait_payload = {
            "__status__": "waiting_human",
            "reason": "research_checkpoint",
            "run_id": research_run_id,
            "research_checkpoint_id": checkpoint_id,
            "research_checkpoint_type": "sources_review",
            "active_poll_seconds": 1.25,
        }
        db.complete_step_run(
            step_run_id=step_run_id,
            status="waiting_human",
            outputs=wait_payload,
        )
        db.update_run_status(
            run_id,
            status="waiting_human",
            status_reason="awaiting_review",
            outputs=wait_payload,
        )
        db.upsert_research_wait_link(
            wait_id=f"{run_id}:wait",
            tenant_id="default",
            workflow_run_id=run_id,
            step_id="wait",
            research_run_id=research_run_id,
            checkpoint_id=checkpoint_id,
            checkpoint_type="sources_review",
            wait_status="waiting",
            wait_payload=wait_payload,
            active_poll_seconds=1.25,
        )

    _seed_waiting_run("wf-match", "research-session-21", "checkpoint-21")
    _seed_waiting_run("wf-other", "research-session-22", "checkpoint-22")

    scheduled: list[dict[str, object]] = []

    monkeypatch.setattr(research_wait_bridge, "_build_workflows_db", lambda: db)
    monkeypatch.setattr(
        research_wait_bridge,
        "_schedule_resume",
        lambda **kwargs: scheduled.append(kwargs),
    )

    resumed = await research_wait_bridge.resume_workflows_waiting_on_research_checkpoint(
        research_run_id="research-session-21",
        checkpoint_id="checkpoint-21",
    )

    assert resumed == 1
    assert len(scheduled) == 1
    assert scheduled[0]["workflow_run_id"] == "wf-match"
    assert scheduled[0]["step_id"] == "wait"
    assert scheduled[0]["wait_payload"]["research_checkpoint_id"] == "checkpoint-21"

    matched_link = db.get_research_wait_link(workflow_run_id="wf-match", step_id="wait")
    other_link = db.get_research_wait_link(workflow_run_id="wf-other", step_id="wait")
    assert matched_link is not None
    assert matched_link["wait_status"] == "resumed"
    assert other_link is not None
    assert other_link["wait_status"] == "waiting"


@pytest.mark.asyncio
async def test_resume_workflows_waiting_on_research_checkpoint_keeps_failed_schedule_retryable(
    tmp_path,
    monkeypatch,
):
    from tldw_Server_API.app.core.Workflows import research_wait_bridge

    db = WorkflowsDatabase(str(tmp_path / "workflow-research-waits-retry.db"))

    definition = {
        "name": "resume-bridge-retry",
        "version": 1,
        "steps": [
            {
                "id": "wait",
                "type": "deep_research_wait",
                "config": {"run_id": "{{ inputs.run_id }}"},
            }
        ],
    }

    db.create_run(
        run_id="wf-retry",
        tenant_id="default",
        user_id="1",
        inputs={"run_id": "research-session-31"},
        workflow_id=None,
        definition_version=1,
        definition_snapshot=definition,
    )
    db.update_run_status("wf-retry", status="waiting_human", status_reason="awaiting_review")
    step_run_id = "wf-retry:wait:1"
    db.create_step_run(
        step_run_id=step_run_id,
        tenant_id="default",
        run_id="wf-retry",
        step_id="wait",
        name="wait",
        step_type="deep_research_wait",
    )
    wait_payload = {
        "__status__": "waiting_human",
        "reason": "research_checkpoint",
        "run_id": "research-session-31",
        "research_checkpoint_id": "checkpoint-31",
        "research_checkpoint_type": "sources_review",
        "active_poll_seconds": 1.25,
    }
    db.complete_step_run(
        step_run_id=step_run_id,
        status="waiting_human",
        outputs=wait_payload,
    )
    db.upsert_research_wait_link(
        wait_id="wf-retry:wait",
        tenant_id="default",
        workflow_run_id="wf-retry",
        step_id="wait",
        research_run_id="research-session-31",
        checkpoint_id="checkpoint-31",
        checkpoint_type="sources_review",
        wait_status="waiting",
        wait_payload=wait_payload,
        active_poll_seconds=1.25,
    )

    monkeypatch.setattr(research_wait_bridge, "_build_workflows_db", lambda: db)

    def _boom(**_kwargs):
        raise RuntimeError("scheduler unavailable")

    monkeypatch.setattr(research_wait_bridge, "_schedule_resume", _boom)

    resumed = await research_wait_bridge.resume_workflows_waiting_on_research_checkpoint(
        research_run_id="research-session-31",
        checkpoint_id="checkpoint-31",
    )

    assert resumed == 0
    link = db.get_research_wait_link(workflow_run_id="wf-retry", step_id="wait")
    assert link is not None
    assert link["wait_status"] == "waiting"
    claimed_again = db.claim_research_waits_for_resume(
        research_run_id="research-session-31",
        checkpoint_id="checkpoint-31",
    )
    assert [row["wait_id"] for row in claimed_again] == ["wf-retry:wait"]


def test_research_checkpoint_approval_auto_resumes_waiting_workflow(
    monkeypatch,
    client_with_workflows_db: TestClient,
):
    client = client_with_workflows_db
    from tldw_Server_API.app.api.v1.endpoints import research_runs
    from tldw_Server_API.app.core.Workflows import research_wait_bridge

    db = client.app.dependency_overrides[wf_mod._get_db]()
    state = {"approved": False}

    class _LaunchSession:
        id = "research-session-31"
        status = "queued"
        phase = "drafting_plan"
        control_state = "running"

    class _CheckpointSession:
        id = "research-session-31"
        status = "waiting_human"
        phase = "awaiting_source_review"
        control_state = "running"
        completed_at = None
        latest_checkpoint_id = "checkpoint-31"

    class _CompletedSession:
        id = "research-session-31"
        status = "completed"
        phase = "completed"
        control_state = "running"
        completed_at = "2026-03-07T16:00:00+00:00"
        latest_checkpoint_id = "checkpoint-31"

    class _CheckpointSnapshot:
        checkpoint = {
            "checkpoint_id": "checkpoint-31",
            "checkpoint_type": "sources_review",
        }

    class _LaunchResearchService:
        def create_session(self, **kwargs):
            return _LaunchSession()

    class _WaitResearchService:
        def get_session(self, **kwargs):
            if state["approved"]:
                return _CompletedSession()
            return _CheckpointSession()

        def get_stream_snapshot(self, **kwargs):
            return _CheckpointSnapshot()

    class _ApproveResearchService:
        def approve_checkpoint(self, **kwargs):
            state["approved"] = True
            return {
                "id": kwargs["session_id"],
                "status": "queued",
                "phase": "collecting",
                "control_state": "running",
                "progress_percent": 45.0,
                "progress_message": "collecting sources",
                "active_job_id": "job-31",
                "latest_checkpoint_id": kwargs["checkpoint_id"],
            }

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.research.launch._build_research_service",
        lambda: _LaunchResearchService(),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.research.wait._build_research_service",
        lambda: _WaitResearchService(),
    )
    monkeypatch.setattr(research_wait_bridge, "_build_workflows_db", lambda: db)
    client.app.dependency_overrides[research_runs.get_research_service] = (
        lambda: _ApproveResearchService()
    )

    definition = {
        "name": "launch-pause-resume-deep-research",
        "version": 1,
        "steps": [
            {
                "id": "launch",
                "type": "deep_research",
                "config": {
                    "query": "{{ inputs.topic }}",
                },
            },
            {
                "id": "wait",
                "type": "deep_research_wait",
                "config": {
                    "run_id": "{{ launch.run_id }}",
                    "include_bundle": False,
                    "poll_interval_seconds": 0.1,
                    "timeout_seconds": 2,
                },
            },
            {
                "id": "prompt",
                "type": "prompt",
                "config": {
                    "template": "checkpoint cleared",
                },
            },
        ],
    }

    create = client.post("/api/v1/workflows", json=definition)
    assert create.status_code == 201, create.text
    workflow_id = create.json()["id"]

    run_id = client.post(
        f"/api/v1/workflows/{workflow_id}/run",
        json={"inputs": {"topic": "checkpoint-aware waiting"}},
    ).json()["run_id"]

    deadline = time.time() + 3
    data = {}
    while time.time() < deadline:
        data = client.get(f"/api/v1/workflows/runs/{run_id}").json()
        if data["status"] in ("waiting_human", "failed", "cancelled", "succeeded"):
            break
        time.sleep(0.05)

    assert data["status"] == "waiting_human"

    approve_resp = client.post(
        "/api/v1/research/runs/research-session-31/checkpoints/checkpoint-31/patch-and-approve",
        json={},
    )
    assert approve_resp.status_code == 200, approve_resp.text

    deadline = time.time() + 5
    resumed = {}
    while time.time() < deadline:
        resumed = client.get(f"/api/v1/workflows/runs/{run_id}").json()
        if resumed["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.05)

    assert resumed["status"] == "succeeded"
    assert (resumed.get("outputs") or {}) == {"text": "checkpoint cleared"}

    wait_link = db.get_research_wait_link(workflow_run_id=run_id, step_id="wait")
    assert wait_link is not None
    assert wait_link["wait_status"] == "resumed"
    wait_step = db.get_latest_step_run(run_id=run_id, step_id="wait")
    assert wait_step is not None
    assert wait_step["status"] == "succeeded"
    wait_outputs = json.loads(wait_step["outputs_json"] or "{}")
    assert wait_outputs["run_id"] == "research-session-31"
    assert wait_outputs["status"] == "completed"


def test_run_workflow_loads_bundle_refs_after_wait(monkeypatch, client_with_workflows_db: TestClient):
    client = client_with_workflows_db
    captured: dict[str, object] = {}

    class _LaunchSession:
        id = "research-session-8"
        status = "queued"
        phase = "drafting_plan"
        control_state = "running"

    class _CompletedSession:
        id = "research-session-8"
        status = "completed"
        phase = "completed"
        control_state = "running"
        completed_at = "2026-03-07T14:30:00+00:00"

    class _LaunchResearchService:
        def create_session(self, **kwargs):
            captured["create_session_kwargs"] = kwargs
            return _LaunchSession()

    class _WaitResearchService:
        def get_session(self, **kwargs):
            captured["wait_get_session_kwargs"] = kwargs
            return _CompletedSession()

    class _LoadBundleSnapshot:
        artifacts = [
            {
                "artifact_name": "bundle.json",
                "artifact_version": 1,
                "content_type": "application/json",
                "phase": "packaging",
                "job_id": "job-99",
            }
        ]

    class _LoadBundleResearchService:
        def get_session(self, **kwargs):
            captured["load_get_session_kwargs"] = kwargs
            return _CompletedSession()

        def get_bundle(self, **kwargs):
            captured["load_get_bundle_kwargs"] = kwargs
            return {
                "question": "Investigate evidence-backed forecasting",
                "outline": {"sections": [{"title": "Overview"}, {"title": "Findings"}]},
                "claims": [
                    {"text": "Claim A", "citations": [{"source_id": "src_1"}]},
                    {"text": "Claim B", "citations": [{"source_id": "src_2"}]},
                ],
                "source_inventory": [
                    {"source_id": "src_1", "title": "Source 1"},
                    {"source_id": "src_2", "title": "Source 2"},
                ],
                "unresolved_questions": ["Need more contradictory evidence"],
            }

        def get_stream_snapshot(self, **kwargs):
            captured["load_get_stream_snapshot_kwargs"] = kwargs
            return _LoadBundleSnapshot()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.research.launch._build_research_service",
        lambda: _LaunchResearchService(),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.research.wait._build_research_service",
        lambda: _WaitResearchService(),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.research.load_bundle._build_research_service",
        lambda: _LoadBundleResearchService(),
    )

    definition = {
        "name": "launch-wait-load-deep-research",
        "version": 1,
        "steps": [
            {
                "id": "launch",
                "type": "deep_research",
                "config": {
                    "query": "{{ inputs.topic }}",
                },
            },
            {
                "id": "wait",
                "type": "deep_research_wait",
                "config": {
                    "run_id": "{{ launch.run_id }}",
                    "include_bundle": False,
                },
            },
            {
                "id": "load",
                "type": "deep_research_load_bundle",
                "config": {
                    "run_id": "{{ wait.run_id }}",
                },
            },
        ],
    }

    create = client.post("/api/v1/workflows", json=definition)
    assert create.status_code == 201, create.text
    wid = create.json()["id"]

    run_id = client.post(
        f"/api/v1/workflows/{wid}/run",
        json={"inputs": {"topic": "evidence-backed forecasting"}},
    ).json()["run_id"]

    deadline = time.time() + 5
    data = {}
    while time.time() < deadline:
        data = client.get(f"/api/v1/workflows/runs/{run_id}").json()
        if data["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.05)

    assert data["status"] == "succeeded"
    assert (data.get("outputs") or {}) == {
        "run_id": "research-session-8",
        "status": "completed",
        "phase": "completed",
        "control_state": "running",
        "completed_at": "2026-03-07T14:30:00+00:00",
        "bundle_url": "/api/v1/research/runs/research-session-8/bundle",
        "bundle_summary": {
            "question": "Investigate evidence-backed forecasting",
            "outline_titles": ["Overview", "Findings"],
            "claim_count": 2,
            "source_count": 2,
            "unresolved_question_count": 1,
        },
        "artifacts": [
            {
                "artifact_name": "bundle.json",
                "artifact_version": 1,
                "content_type": "application/json",
                "phase": "packaging",
                "job_id": "job-99",
            }
        ],
    }
    assert "bundle" not in (data.get("outputs") or {})
    assert captured["load_get_session_kwargs"] == {
        "owner_user_id": "1",
        "session_id": "research-session-8",
    }
    assert captured["load_get_bundle_kwargs"] == {
        "owner_user_id": "1",
        "session_id": "research-session-8",
    }
    assert captured["load_get_stream_snapshot_kwargs"] == {
        "owner_user_id": "1",
        "session_id": "research-session-8",
    }


def test_run_workflow_launches_waits_selects_research_bundle_fields_and_uses_them_downstream(
    monkeypatch,
    client_with_workflows_db: TestClient,
):
    client = client_with_workflows_db

    class _LaunchSession:
        id = "research-session-9"
        status = "queued"
        phase = "drafting_plan"
        control_state = "running"

    class _CompletedSession:
        id = "research-session-9"
        status = "completed"
        phase = "completed"
        control_state = "running"
        completed_at = "2026-03-08T09:00:00+00:00"

    class _LaunchResearchService:
        def create_session(self, **kwargs):
            return _LaunchSession()

    class _WaitResearchService:
        def get_session(self, **kwargs):
            return _CompletedSession()

    class _SelectResearchService:
        def get_session(self, **kwargs):
            return _CompletedSession()

        def get_bundle(self, **kwargs):
            return {
                "question": "Investigate evidence-backed forecasting",
                "verification_summary": {"supported_claim_count": 2},
                "unsupported_claims": [{"text": "Claim X"}],
            }

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.research.launch._build_research_service",
        lambda: _LaunchResearchService(),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.research.wait._build_research_service",
        lambda: _WaitResearchService(),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.research.select_bundle_fields._build_research_service",
        lambda: _SelectResearchService(),
    )

    definition = {
        "name": "launch-wait-select-deep-research",
        "version": 1,
        "steps": [
            {
                "id": "launch",
                "type": "deep_research",
                "config": {
                    "query": "{{ inputs.topic }}",
                },
            },
            {
                "id": "wait",
                "type": "deep_research_wait",
                "config": {
                    "run_id": "{{ launch.run_id }}",
                    "include_bundle": False,
                },
            },
            {
                "id": "select",
                "type": "deep_research_select_bundle_fields",
                "config": {
                    "run_id": "{{ wait.run_id }}",
                    "fields": ["question", "verification_summary", "unsupported_claims"],
                },
            },
            {
                "id": "prompt",
                "type": "prompt",
                "config": {
                    "template": "{{ select.selected_fields.question }} :: {{ select.selected_fields.verification_summary.supported_claim_count }}",
                },
            },
        ],
    }

    create = client.post("/api/v1/workflows", json=definition)
    assert create.status_code == 201, create.text
    wid = create.json()["id"]

    run_id = client.post(
        f"/api/v1/workflows/{wid}/run",
        json={"inputs": {"topic": "evidence-backed forecasting"}},
    ).json()["run_id"]

    deadline = time.time() + 5
    data = {}
    while time.time() < deadline:
        data = client.get(f"/api/v1/workflows/runs/{run_id}").json()
        if data["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.05)

    assert data["status"] == "succeeded"
    assert (data.get("outputs") or {}) == {
        "text": "Investigate evidence-backed forecasting :: 2"
    }


def test_artifact_manifest_verify_mismatch(monkeypatch, client_with_workflows_db: TestClient):
    client = client_with_workflows_db
    # Create definition and run
    d = {"name": "manifest", "version": 1, "steps": [{"id": "s1", "type": "prompt", "config": {"template": "ok"}}]}
    wid = client.post("/api/v1/workflows", json=d).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]

    # Create temp artifact with wrong checksum
    import os, tempfile
    fd, path = tempfile.mkstemp(prefix="wf_manifest_")
    os.write(fd, b"data")
    os.close(fd)
    uri = f"file://{path}"
    from tldw_Server_API.app.api.v1.endpoints import workflows as wf_mod
    db = client.app.dependency_overrides[wf_mod._get_db]()
    db.add_artifact(
        artifact_id="a_manifest",
        tenant_id="default",
        run_id=run_id,
        step_run_id=None,
        type="file",
        uri=uri,
        size_bytes=4,
        mime_type="text/plain",
        checksum_sha256="deadbeef",
        metadata={}
    )
    r = client.get(f"/api/v1/workflows/runs/{run_id}/artifacts/manifest", params={"verify": True})
    assert r.status_code == 200
    body = r.json()
    assert body.get("integrity_summary", {}).get("mismatch_count", 0) >= 1


def test_completion_webhook_tenant_denylist_blocked(monkeypatch, client_with_workflows_db: TestClient):
    client = client_with_workflows_db
    # Deny host globally
    monkeypatch.setenv("WORKFLOWS_WEBHOOK_DENYLIST", "deny.test")
    definition = {
        "name": "webhook-deny",
        "version": 1,
        "on_completion_webhook": {"url": "https://deny.test/hook", "include_outputs": False},
        "steps": [
            {"id": "s1", "type": "prompt", "config": {"template": "ok"}},
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    # Wait for completion
    import time
    deadline = time.time() + 3
    while time.time() < deadline:
        st = client.get(f"/api/v1/workflows/runs/{run_id}").json()["status"]
        if st in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.05)
    ev = client.get(f"/api/v1/workflows/runs/{run_id}/events").json()
    statuses = [e.get("payload", {}).get("status") for e in ev if e.get("event_type") == "webhook_delivery"]
    # blocked event should be present
    assert "blocked" in statuses

    # Admin can filter by specific owner (user 2) and should see that run
    # Ensure there is at least one run owned by user id=2
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User as _User, get_request_user as _gru
    prev_dep = client.app.dependency_overrides[_gru]
    async def override_user2():
        return _User(id=2, username="owner2", email="o2@x", is_active=True, is_admin=False)
    client.app.dependency_overrides[_gru] = override_user2
    def_u2 = {"name": "u2-def", "version": 1, "steps": [{"id": "s1", "type": "prompt", "config": {"template": "yo"}}]}
    wid_u2 = client.post("/api/v1/workflows", json=def_u2).json()["id"]
    r_u2 = client.post(f"/api/v1/workflows/{wid_u2}/run", json={"inputs": {}}).json()["run_id"]
    client.app.dependency_overrides[_gru] = prev_dep

    lst_owner2 = client.get("/api/v1/workflows/runs", params={"owner": "2", "limit": 50}).json()
    ids2 = [r.get("run_id") for r in lst_owner2.get("runs", [])]
    assert r_u2 in ids2

    # Non-admin cannot override owner: they should only see their own runs (user id=2)
    client.app.dependency_overrides[_gru] = override_user2
    lst_nonadmin = client.get("/api/v1/workflows/runs", params={"owner": "1", "limit": 50}).json()
    runs = lst_nonadmin.get("runs", [])
    assert runs, "expected some runs for non-admin user"
    assert all(str(r.get("user_id")) == "2" for r in runs)
    # Restore admin override after test
    client.app.dependency_overrides[_gru] = prev_dep
