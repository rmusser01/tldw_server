from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import setup as setup_endpoint
from tldw_Server_API.app.core.Setup.first_run_state import (
    REQUIRED_FIRST_RUN_STEPS,
    FirstRunStateStore,
)
from tldw_Server_API.app.main import app


def _setup_needs_setup(monkeypatch):
    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "get_status_snapshot",
        lambda: {
            "enabled": True,
            "setup_completed": False,
            "needs_setup": True,
            "auth_mode": "single_user",
            "allow_remote_setup_access": False,
            "remote_access_env_override": False,
            "remote_access_active": False,
        },
    )


def test_first_run_state_endpoint_returns_backend_state(monkeypatch, tmp_path):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)

    client = TestClient(app)
    response = client.get("/api/v1/setup/first-run/state")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "not_started"
    assert body["first_chat"]["completed"] is False


def test_first_run_skip_endpoint_records_skipped(monkeypatch, tmp_path):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)

    client = TestClient(app)
    response = client.post("/api/v1/setup/first-run/skip", json={"reason": "user_skip"})

    assert response.status_code == 200
    assert response.json()["status"] == "skipped"


def test_first_run_metadata_returns_auth_and_setup_path_guidance(monkeypatch, tmp_path):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)

    client = TestClient(app)
    response = client.get("/api/v1/setup/first-run/metadata")

    assert response.status_code == 200
    body = response.json()
    assert "auth_mode" in body
    assert "manual_auth_required" in body
    assert "bundled_single_user_auth_available" in body
    assert "frontend_origin" in body["connection"]
    assert "api_origin" in body["connection"]
    assert body["connection"]["browser_access"] in {"local", "lan", "remote", "unknown"}
    assert {path["key"] for path in body["setup_paths"]} >= {
        "docker_single_user",
        "local_single_user",
        "multi_user",
    }
    assert body["multi_user_exit"]["guide_path"]


def test_completed_setup_rejects_first_run_writes(monkeypatch, tmp_path):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)
    store = FirstRunStateStore(state_path)
    for step in REQUIRED_FIRST_RUN_STEPS:
        store.update_step(step, {"acknowledged": True})
    store.record_first_chat_success(
        provider="openai",
        model="gpt-4.1-mini",
        response_id="chatcmpl-test",
    )
    store.mark_completed()

    client = TestClient(app)
    response = client.post(
        "/api/v1/setup/first-run/state",
        json={"step": "providers", "data": {}},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "setup_already_completed"


def test_legacy_completed_setup_rejects_first_run_writes_without_state_file(monkeypatch, tmp_path):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "get_status_snapshot",
        lambda: {
            "enabled": True,
            "setup_completed": True,
            "completed": True,
            "needs_setup": False,
            "auth_mode": "single_user",
            "allow_remote_setup_access": False,
            "remote_access_env_override": False,
            "remote_access_active": False,
        },
    )

    client = TestClient(app)
    response = client.post(
        "/api/v1/setup/first-run/state",
        json={"step": "providers", "data": {}},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "setup_already_completed"
