import importlib.machinery
import sys
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
from tldw_Server_API.app.api.v1.endpoints import workflows as wf_mod
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.AuthNZ.permissions import WORKFLOWS_RUNS_READ
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase


pytestmark = pytest.mark.integration


@pytest.fixture()
def client_with_investigation_db(tmp_path, auth_headers):
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    state = {
        "user": User(
            id=1,
            username="tester",
            email="tester@example.com",
            is_active=True,
            is_admin=True,
            tenant_id="default",
            roles=["admin"],
        )
    }

    async def override_user():
        return state["user"]

    def override_db():
        return db

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[wf_mod._get_db] = override_db

    client = TestClient(app, headers=auth_headers)
    try:
        yield client, db, state
    finally:
        client.close()
        app.dependency_overrides.clear()


def _seed_failed_run(db: WorkflowsDatabase) -> str:
    run_id = f"run-investigation-{uuid4().hex[:8]}"
    step_run_id = f"{run_id}:s1:1"
    db.create_run(
        run_id=run_id,
        tenant_id="default",
        user_id="1",
        inputs={"target": "hook"},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={
            "name": "investigation-failure",
            "version": 1,
            "steps": [
                {"id": "s1", "name": "Call hook", "type": "webhook", "retry": 1, "config": {"url": "https://example.invalid/hook"}},
            ],
        },
    )
    db.update_run_status(
        run_id,
        status="failed",
        status_reason="transient_network_error",
        started_at="2026-03-11T10:00:00",
        ended_at="2026-03-11T10:00:04",
        error="transient_network_error: upstream reset",
    )
    db.create_step_run(
        step_run_id=step_run_id,
        tenant_id="default",
        run_id=run_id,
        step_id="s1",
        name="Call hook",
        step_type="webhook",
        status="running",
        inputs={"config": {"url": "https://example.invalid/hook"}},
    )
    db.update_step_attempt(step_run_id=step_run_id, attempt=7)
    attempt1 = db.create_step_attempt(
        tenant_id="default",
        run_id=run_id,
        step_run_id=step_run_id,
        step_id="s1",
        attempt_number=1,
        metadata={
            "step_type": "webhook",
            "category": "runtime",
            "blame_scope": "external_dependency",
            "retry_recommendation": "conditional",
        },
    )
    db.complete_step_attempt(
        attempt_id=attempt1,
        status="failed",
        reason_code_core="transient_network_error",
        reason_code_detail="RuntimeError",
        retryable=True,
        error_summary="upstream reset",
        metadata={
            "step_type": "webhook",
            "category": "runtime",
            "blame_scope": "external_dependency",
            "retry_recommendation": "conditional",
        },
    )
    attempt2 = db.create_step_attempt(
        tenant_id="default",
        run_id=run_id,
        step_run_id=step_run_id,
        step_id="s1",
        attempt_number=2,
        metadata={
            "step_type": "webhook",
            "category": "runtime",
            "blame_scope": "external_dependency",
            "retry_recommendation": "conditional",
        },
    )
    db.complete_step_attempt(
        attempt_id=attempt2,
        status="failed",
        reason_code_core="transient_network_error",
        reason_code_detail="RuntimeError",
        retryable=True,
        error_summary="upstream reset",
        metadata={
            "step_type": "webhook",
            "category": "runtime",
            "blame_scope": "external_dependency",
            "retry_recommendation": "conditional",
            "failure_envelope": {"reason_code_core": "transient_network_error"},
        },
    )
    db.complete_step_run(
        step_run_id=step_run_id,
        status="failed",
        outputs={},
        error="transient_network_error: upstream reset",
    )
    db.append_event(
        "default",
        run_id,
        "step_failed",
        {"step_id": "s1", "reason_code": "transient_network_error"},
        step_run_id=step_run_id,
    )
    db.append_event(
        "default",
        run_id,
        "webhook_delivery",
        {"step_id": "s1", "status": "failed", "response_status": 502},
        step_run_id=step_run_id,
    )
    db.append_event(
        "default",
        run_id,
        "run_failed",
        {"reason_code": "transient_network_error"},
        step_run_id=step_run_id,
    )
    db.add_artifact(
        artifact_id=f"artifact-{uuid4().hex[:8]}",
        tenant_id="default",
        run_id=run_id,
        step_run_id=step_run_id,
        type="log",
        uri="file:///tmp/workflow.log",
        metadata={"kind": "stderr_excerpt"},
    )
    return run_id


def _seed_succeeded_run(db: WorkflowsDatabase) -> str:
    run_id = f"run-investigation-success-{uuid4().hex[:8]}"
    step_run_id = f"{run_id}:s1:1"
    db.create_run(
        run_id=run_id,
        tenant_id="default",
        user_id="1",
        inputs={"target": "hook"},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={
            "name": "investigation-success",
            "version": 1,
            "steps": [
                {"id": "s1", "name": "Render output", "type": "prompt", "config": {"template": "ok"}},
            ],
        },
    )
    db.update_run_status(
        run_id,
        status="succeeded",
        started_at="2026-03-11T10:00:00",
        ended_at="2026-03-11T10:00:02",
        outputs={"text": "ok"},
    )
    db.create_step_run(
        step_run_id=step_run_id,
        tenant_id="default",
        run_id=run_id,
        step_id="s1",
        name="Render output",
        step_type="prompt",
        status="running",
        inputs={"config": {"template": "ok"}},
    )
    attempt_id = db.create_step_attempt(
        tenant_id="default",
        run_id=run_id,
        step_run_id=step_run_id,
        step_id="s1",
        attempt_number=1,
        metadata={"step_type": "prompt"},
    )
    db.complete_step_attempt(
        attempt_id=attempt_id,
        status="succeeded",
        metadata={"step_type": "prompt"},
    )
    db.complete_step_run(
        step_run_id=step_run_id,
        status="succeeded",
        outputs={"text": "ok"},
    )
    db.append_event(
        "default",
        run_id,
        "step_completed",
        {"step_id": "s1"},
        step_run_id=step_run_id,
    )
    db.append_event(
        "default",
        run_id,
        "run_completed",
        {"success": True},
        step_run_id=step_run_id,
    )
    return run_id


def _seed_cancelled_run(db: WorkflowsDatabase) -> str:
    run_id = f"run-investigation-cancelled-{uuid4().hex[:8]}"
    step_run_id = f"{run_id}:s1:1"
    db.create_run(
        run_id=run_id,
        tenant_id="default",
        user_id="1",
        inputs={"target": "hook"},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={
            "name": "investigation-cancelled",
            "version": 1,
            "steps": [
                {"id": "s1", "name": "Render output", "type": "prompt", "config": {"template": "ok"}},
            ],
        },
    )
    db.update_run_status(
        run_id,
        status="cancelled",
        status_reason="cancelled_by_user",
        started_at="2026-03-11T10:00:00",
        ended_at="2026-03-11T10:00:02",
    )
    db.create_step_run(
        step_run_id=step_run_id,
        tenant_id="default",
        run_id=run_id,
        step_id="s1",
        name="Render output",
        step_type="prompt",
        status="running",
        inputs={"config": {"template": "ok"}},
    )
    attempt_id = db.create_step_attempt(
        tenant_id="default",
        run_id=run_id,
        step_run_id=step_run_id,
        step_id="s1",
        attempt_number=1,
        metadata={"step_type": "prompt"},
    )
    db.complete_step_attempt(
        attempt_id=attempt_id,
        status="succeeded",
        metadata={"step_type": "prompt"},
    )
    db.complete_step_run(
        step_run_id=step_run_id,
        status="succeeded",
        outputs={"text": "ok"},
    )
    db.append_event(
        "default",
        run_id,
        "step_completed",
        {"step_id": "s1"},
        step_run_id=step_run_id,
    )
    db.append_event(
        "default",
        run_id,
        "run_cancelled",
        {"reason": "cancelled_by_user"},
        step_run_id=step_run_id,
    )
    return run_id


def test_investigation_endpoint_returns_primary_failure(client_with_investigation_db: tuple[TestClient, WorkflowsDatabase, dict]):
    client, db, _state = client_with_investigation_db
    run_id = _seed_failed_run(db)

    resp = client.get(f"/api/v1/workflows/runs/{run_id}/investigation")

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["primary_failure"]["reason_code_core"] == "transient_network_error"
    assert data["failed_step"]["attempt_count"] == 7
    assert data["failed_step"]["step_id"] == "s1"
    assert data["recommended_actions"]
    assert data["primary_failure"]["internal_detail"]["event_count"] >= 3


def test_investigation_endpoint_omits_failure_for_successful_run(client_with_investigation_db: tuple[TestClient, WorkflowsDatabase, dict]):
    client, db, _state = client_with_investigation_db
    run_id = _seed_succeeded_run(db)

    resp = client.get(f"/api/v1/workflows/runs/{run_id}/investigation")

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "succeeded"
    assert data["failed_step"] is None
    assert data["primary_failure"] is None
    assert data["attempts"] == []
    assert data["recommended_actions"] == []


def test_investigation_endpoint_omits_failure_for_cancelled_run(client_with_investigation_db: tuple[TestClient, WorkflowsDatabase, dict]):
    client, db, _state = client_with_investigation_db
    run_id = _seed_cancelled_run(db)

    resp = client.get(f"/api/v1/workflows/runs/{run_id}/investigation")

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "cancelled"
    assert data["failed_step"] is None
    assert data["primary_failure"] is None
    assert data["attempts"] == []
    assert data["recommended_actions"] == []


def test_steps_endpoint_returns_step_history(client_with_investigation_db: tuple[TestClient, WorkflowsDatabase, dict]):
    client, db, _state = client_with_investigation_db
    run_id = _seed_failed_run(db)

    resp = client.get(f"/api/v1/workflows/runs/{run_id}/steps")

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["steps"][0]["step_id"] == "s1"
    assert data["steps"][0]["attempt_count"] == 7
    assert data["steps"][0]["latest_failure"]["reason_code_core"] == "transient_network_error"


def test_steps_endpoint_counts_sparse_attempt_numbers(client_with_investigation_db: tuple[TestClient, WorkflowsDatabase, dict]):
    client, db, _state = client_with_investigation_db
    run_id = f"run-sparse-attempts-{uuid4().hex[:8]}"
    step_run_id = f"{run_id}:s1:1"
    db.create_run(
        run_id=run_id,
        tenant_id="default",
        user_id="1",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": "sparse-attempts", "version": 1, "steps": []},
    )
    db.create_step_run(
        step_run_id=step_run_id,
        tenant_id="default",
        run_id=run_id,
        step_id="s1",
        name="Sparse attempts",
        step_type="prompt",
        status="failed",
        inputs={},
    )
    for attempt_number in (1, 3):
        attempt_id = db.create_step_attempt(
            tenant_id="default",
            run_id=run_id,
            step_run_id=step_run_id,
            step_id="s1",
            attempt_number=attempt_number,
            status="running",
            metadata={"step_type": "prompt"},
        )
        db.complete_step_attempt(
            attempt_id=attempt_id,
            status="failed",
            reason_code_core="runtime_error",
            metadata={"step_type": "prompt"},
        )

    resp = client.get(f"/api/v1/workflows/runs/{run_id}/steps")

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["steps"][0]["attempt_count"] == 3


def test_step_attempts_endpoint_returns_attempt_timeline(client_with_investigation_db: tuple[TestClient, WorkflowsDatabase, dict]):
    client, db, _state = client_with_investigation_db
    run_id = _seed_failed_run(db)

    resp = client.get(f"/api/v1/workflows/runs/{run_id}/steps/s1/attempts")

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data["attempts"]) == 2
    assert data["attempts"][0]["attempt_number"] == 1
    assert data["attempts"][0]["reason_code_core"] == "transient_network_error"
    assert data["attempts"][0]["metadata"]["retry_recommendation"] == "conditional"


def test_investigation_redacts_operator_detail_for_non_admin(client_with_investigation_db: tuple[TestClient, WorkflowsDatabase, dict]):
    client, db, state = client_with_investigation_db
    run_id = _seed_failed_run(db)
    state["user"] = User(
        id=1,
        username="tester",
        email="tester@example.com",
        is_active=True,
        is_admin=False,
        tenant_id="default",
        roles=["user"],
        permissions=[WORKFLOWS_RUNS_READ],
    )

    resp = client.get(f"/api/v1/workflows/runs/{run_id}/investigation")

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["primary_failure"]["reason_code_core"] == "transient_network_error"
    assert data["primary_failure"]["internal_detail"] is None
