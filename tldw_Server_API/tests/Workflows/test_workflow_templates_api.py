import importlib.machinery
import sys
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
from tldw_Server_API.app.api.v1.endpoints import workflows as wf_mod
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.permissions import (
    WORKFLOWS_RUNS_CONTROL,
    WORKFLOWS_RUNS_READ,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user


@pytest.fixture()
def client(tmp_path, auth_headers) -> TestClient:
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

    with TestClient(app, headers=auth_headers) as c:
        yield c

    app.dependency_overrides.clear()


def test_templates_list_and_get(client: TestClient):
    # List templates
    r = client.get("/api/v1/workflows/templates")
    assert r.status_code == 200, r.text
    items = r.json()
    assert isinstance(items, list)
    assert any(it.get("name") == "paper_roundup" for it in items), items
    # Ensure human-friendly title present
    one = next((x for x in items if x.get("name") == "paper_roundup"), None)
    assert one is not None and isinstance(one.get("title"), str) and len(one.get("title")) > 0

    # Retrieve one by name
    r2 = client.get("/api/v1/workflows/templates/paper_roundup")
    assert r2.status_code == 200, r2.text
    data = r2.json()
    assert isinstance(data, dict)
    assert data.get("name") == "paper_roundup"
    assert isinstance(data.get("steps"), list)


def test_templates_invalid_and_missing(client: TestClient):
    # Invalid name rejected
    # Non-encoded traversal may be normalized by the router; accept 400 or 404
    bad = client.get("/api/v1/workflows/templates/../../etc/passwd")
    assert bad.status_code in (400, 404)

    # URL-encoded traversal should be rejected with 400 (guarded before resolution)
    bad_enc = client.get("/api/v1/workflows/templates/%2E%2E/%2E%2E/etc/passwd")
    assert bad_enc.status_code == 400

    # Missing name 404
    miss = client.get("/api/v1/workflows/templates/not_a_template")
    assert miss.status_code == 404


def test_template_create_and_run_flow(client: TestClient):
    # Fetch a template and create+run
    tpl = client.get("/api/v1/workflows/templates/paper_roundup").json()
    create = client.post("/api/v1/workflows", json=tpl)
    assert create.status_code in (200, 201), create.text
    wid = create.json().get("id")
    assert wid, create.text

    run = client.post(f"/api/v1/workflows/{wid}/run?mode=async", json={"inputs": {}})
    assert run.status_code == 200, run.text
    run_id = run.json().get("run_id")
    assert run_id, run.text

    # Fetch status at least once and verify shape
    st = client.get(f"/api/v1/workflows/runs/{run_id}")
    assert st.status_code == 200, st.text
    sj = st.json()
    assert isinstance(sj, dict) and sj.get("id") == run_id and sj.get("status")

    # Events should be retrievable
    ev = client.get(f"/api/v1/workflows/runs/{run_id}/events")
    assert ev.status_code == 200, ev.text
    assert isinstance(ev.json(), list)


def test_templates_tags_endpoint(client: TestClient):
    r = client.get("/api/v1/workflows/templates/tags")
    assert r.status_code == 200, r.text
    tags = r.json()
    assert isinstance(tags, list)
    # Should include a few known tags from bundled templates
    expected = {"tts", "pdf", "research", "rag", "policy"}
    assert expected.issubset(set(tags)), f"Missing tags: {expected - set(tags)} in {tags}"


def test_templates_include_acp_pipeline_variants(client: TestClient):
    r = client.get("/api/v1/workflows/templates")
    assert r.status_code == 200, r.text
    names = {it.get("name") for it in r.json()}
    assert {"pipeline_l1_acp", "pipeline_l2_acp", "pipeline_l3_acp"}.issubset(names)


def test_get_pipeline_l1_acp_template_shape(client: TestClient):
    r = client.get("/api/v1/workflows/templates/pipeline_l1_acp")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("name") == "pipeline_l1_acp"
    assert isinstance(data.get("steps"), list)
    assert any((step or {}).get("type") == "acp_stage" for step in (data.get("steps") or []))
    tags = set(data.get("tags") or [])
    assert {"acp", "pipeline", "domain", "l1"}.issubset(tags)


def test_templates_tags_include_acp_pipeline(client: TestClient):
    r = client.get("/api/v1/workflows/templates/tags")
    assert r.status_code == 200, r.text
    tags = set(r.json())
    assert {"acp", "pipeline"}.issubset(tags)
