import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app import main as app_main
from tldw_Server_API.app.api.v1.endpoints import setup as setup_endpoint
from tldw_Server_API.app.core.Setup.first_run_state import (
    REQUIRED_FIRST_RUN_STEPS,
    FirstRunStateStore,
    FirstRunStatus,
)
from tldw_Server_API.app.core.Setup.provider_catalog import REQUIRED_SETUP_PROVIDER_KEYS

app = app_main.app
app_main._shared_is_explicit_pytest_runtime = lambda: True


@pytest.fixture
def setup_client():
    # These setup API tests exercise handlers only. Entering TestClient lifespan
    # starts background services, and context-manager shutdown has timed out here.
    client = TestClient(app)
    try:
        yield client
    finally:
        client.close()


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


def _fail_if_state_update_reaches_endpoint(*_args, **_kwargs):
    pytest.fail("terminal first-run state should be rejected by the shared write guard")


def _make_setup_metadata_request(
    *,
    client_host: str,
    host: str,
    forwarded_for: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> Request:
    headers = [(b"host", host.encode("ascii"))]
    if forwarded_for is not None:
        headers.append((b"x-forwarded-for", forwarded_for.encode("ascii")))
    for key, value in (extra_headers or {}).items():
        headers.append((key.encode("ascii"), value.encode("ascii")))
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/setup/first-run/metadata",
            "headers": headers,
            "client": (client_host, 4444),
            "scheme": "http",
            "server": (host, 80),
        }
    )


def test_first_run_state_endpoint_returns_backend_state(monkeypatch, tmp_path, setup_client):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)

    response = setup_client.get("/api/v1/setup/first-run/state")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "not_started"
    assert body["first_chat"]["completed"] is False


def test_first_run_skip_endpoint_records_skipped(monkeypatch, tmp_path, setup_client):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)

    response = setup_client.post("/api/v1/setup/first-run/skip", json={"reason": "user_skip"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "skipped"
    assert body["skip_reason"] == "user_skip"


def test_first_run_skip_endpoint_filters_secret_like_reason(monkeypatch, tmp_path, setup_client):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)

    response = setup_client.post(
        "/api/v1/setup/first-run/skip",
        json={"reason": "hf_abcdef1234567890"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "skipped"
    assert body["skip_reason"] is None
    assert "hf_abcdef1234567890" not in str(body)


def test_first_run_metadata_returns_auth_and_setup_path_guidance(monkeypatch, tmp_path, setup_client):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)

    response = setup_client.get("/api/v1/setup/first-run/metadata")

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


def test_first_run_provider_catalog_returns_required_provider_keys(setup_client):
    response = setup_client.get("/api/v1/setup/first-run/providers/catalog")

    assert response.status_code == 200
    body = response.json()
    keys = {provider["provider_key"] for provider in body["providers"]}
    assert set(REQUIRED_SETUP_PROVIDER_KEYS) <= keys


def test_first_run_provider_save_masks_key_and_writes_config(monkeypatch, tmp_path, setup_client):
    state_path = tmp_path / "first_run_state.json"
    config_path = tmp_path / "config.txt"
    config_path.write_text("[API]\ndefault_api = openai\n", encoding="utf-8")
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    monkeypatch.setattr(setup_endpoint.setup_manager, "get_config_file_path", lambda: config_path)
    _setup_needs_setup(monkeypatch)

    raw_key = "sk-abcdefghijklmnopqrstuvwxyz"
    response = setup_client.post(
        "/api/v1/setup/first-run/providers",
        json={"provider_key": "openai", "api_key": raw_key, "make_default": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["provider_key"] == "openai"
    assert body["status"] == "saved"
    assert body["masked_api_key"] == "sk-...wxyz"
    assert body["make_default"] is True
    assert raw_key not in str(body)
    config_text = config_path.read_text(encoding="utf-8")
    assert "openai_api_key = sk-abcdefghijklmnopqrstuvwxyz" in config_text
    assert "default_api = openai" in config_text


def test_first_run_provider_save_rejects_blank_hosted_api_key_without_config_write(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    config_path = tmp_path / "config.txt"
    config_path.write_text("[API]\ndefault_api = openai\n", encoding="utf-8")
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    monkeypatch.setattr(setup_endpoint.setup_manager, "get_config_file_path", lambda: config_path)
    _setup_needs_setup(monkeypatch)

    response = setup_client.post(
        "/api/v1/setup/first-run/providers",
        json={"provider_key": "openai", "api_key": "   "},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "provider_api_key_required"
    config_text = config_path.read_text(encoding="utf-8")
    assert "openai_api_key" not in config_text


def test_completed_setup_rejects_provider_save_through_first_run_write_guard(
    monkeypatch,
    tmp_path,
    setup_client,
):
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

    response = setup_client.post(
        "/api/v1/setup/first-run/providers",
        json={"provider_key": "openai", "api_key": "sk-abcdefghijklmnopqrstuvwxyz"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "setup_already_completed"


def test_first_run_provider_validate_returns_typed_response_without_token_echo(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)
    raw_token = "secret-local-token"

    async def _fake_validate(payload):
        assert payload.api_key == raw_token
        return setup_endpoint.SetupProviderValidationResponse(
            provider_key=payload.provider_key,
            status="ready",
            models=["local-model"],
        )

    monkeypatch.setattr(setup_endpoint, "validate_local_openai_endpoint", _fake_validate)

    response = setup_client.post(
        "/api/v1/setup/first-run/providers/validate",
        json={
            "provider_key": "ollama",
            "base_url": "http://127.0.0.1:11434/v1",
            "model": "local-model",
            "api_key": raw_token,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "provider_key": "ollama",
        "status": "ready",
        "failure_category": None,
        "message": None,
        "models": ["local-model"],
    }
    assert raw_token not in str(body)


def test_first_run_hosted_provider_validate_rejects_blank_key_with_typed_response(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)

    response = setup_client.post(
        "/api/v1/setup/first-run/providers/validate",
        json={"provider_key": "openai", "api_key": "   "},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["provider_key"] == "openai"
    assert body["status"] == "failed"
    assert body["failure_category"] == "provider_api_key_required"


def test_first_run_hosted_provider_validate_accepts_plausible_key_without_echo(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)
    raw_key = "sk-abcdefghijklmnopqrstuvwxyz"

    response = setup_client.post(
        "/api/v1/setup/first-run/providers/validate",
        json={"provider_key": "openai", "api_key": raw_key},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["provider_key"] == "openai"
    assert body["status"] == "accepted"
    assert body["failure_category"] is None
    assert raw_key not in str(body)


def test_first_run_metadata_classifies_forwarded_remote_browser_as_remote(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)
    original_overrides = dict(app.dependency_overrides)
    app.dependency_overrides[setup_endpoint.require_local_setup_access] = lambda: None
    try:
        response = setup_client.get(
            "/api/v1/setup/first-run/metadata",
            headers={
                "host": "localhost",
                "x-forwarded-for": "203.0.113.10",
            },
        )
    finally:
        app.dependency_overrides = original_overrides

    assert response.status_code == 200
    body = response.json()
    assert body["connection"]["browser_access"] == "remote"
    assert body["bundled_single_user_auth_available"] is False
    assert body["manual_auth_required"] is True


def test_first_run_metadata_ignores_spoofed_local_forwarded_for_from_remote_client(monkeypatch):
    _setup_needs_setup(monkeypatch)
    request = _make_setup_metadata_request(
        client_host="203.0.113.10",
        host="example.test",
        forwarded_for="127.0.0.1",
    )

    metadata = setup_endpoint.build_first_run_metadata(request)

    assert metadata.connection.browser_access == "remote"
    assert metadata.bundled_single_user_auth_available is False
    assert metadata.manual_auth_required is True


def test_first_run_metadata_rejects_mixed_forwarded_for_chain_from_trusted_local_proxy(monkeypatch):
    _setup_needs_setup(monkeypatch)
    request = _make_setup_metadata_request(
        client_host="127.0.0.1",
        host="localhost",
        forwarded_for="127.0.0.1, 203.0.113.10",
    )

    metadata = setup_endpoint.build_first_run_metadata(request)

    assert metadata.connection.browser_access == "remote"
    assert metadata.bundled_single_user_auth_available is False
    assert metadata.manual_auth_required is True


def test_first_run_metadata_rejects_conflicting_forwarded_client_headers(monkeypatch):
    _setup_needs_setup(monkeypatch)
    request = _make_setup_metadata_request(
        client_host="127.0.0.1",
        host="localhost",
        forwarded_for="127.0.0.1",
        extra_headers={"forwarded": "for=203.0.113.10"},
    )

    metadata = setup_endpoint.build_first_run_metadata(request)

    assert metadata.connection.browser_access == "remote"
    assert metadata.bundled_single_user_auth_available is False
    assert metadata.manual_auth_required is True


def test_first_run_metadata_classifies_x_real_ip_from_trusted_local_proxy_as_remote(monkeypatch):
    _setup_needs_setup(monkeypatch)
    request = _make_setup_metadata_request(
        client_host="127.0.0.1",
        host="localhost",
        extra_headers={"x-real-ip": "203.0.113.10"},
    )

    metadata = setup_endpoint.build_first_run_metadata(request)

    assert metadata.connection.browser_access == "remote"
    assert metadata.bundled_single_user_auth_available is False
    assert metadata.manual_auth_required is True


def test_first_run_metadata_classifies_forwarded_for_from_trusted_local_proxy_as_remote(monkeypatch):
    _setup_needs_setup(monkeypatch)
    request = _make_setup_metadata_request(
        client_host="127.0.0.1",
        host="localhost",
        extra_headers={"forwarded": "for=203.0.113.10;proto=https"},
    )

    metadata = setup_endpoint.build_first_run_metadata(request)

    assert metadata.connection.browser_access == "remote"
    assert metadata.bundled_single_user_auth_available is False
    assert metadata.manual_auth_required is True


def test_first_run_metadata_rejects_mixed_forwarded_header_chain_from_trusted_local_proxy(monkeypatch):
    _setup_needs_setup(monkeypatch)
    request = _make_setup_metadata_request(
        client_host="127.0.0.1",
        host="localhost",
        extra_headers={"forwarded": "for=127.0.0.1, for=203.0.113.10"},
    )

    metadata = setup_endpoint.build_first_run_metadata(request)

    assert metadata.connection.browser_access == "remote"
    assert metadata.bundled_single_user_auth_available is False
    assert metadata.manual_auth_required is True


def test_first_run_metadata_treats_malformed_forwarded_header_chain_as_unknown(monkeypatch):
    _setup_needs_setup(monkeypatch)
    request = _make_setup_metadata_request(
        client_host="127.0.0.1",
        host="localhost",
        extra_headers={"forwarded": "for=127.0.0.1, for=not-an-ip"},
    )

    metadata = setup_endpoint.build_first_run_metadata(request)

    assert metadata.connection.browser_access == "unknown"
    assert metadata.bundled_single_user_auth_available is False
    assert metadata.manual_auth_required is True


def test_first_run_metadata_proxy_evidence_without_client_ip_is_not_local(monkeypatch):
    _setup_needs_setup(monkeypatch)
    request = _make_setup_metadata_request(
        client_host="127.0.0.1",
        host="localhost",
        extra_headers={"x-forwarded-proto": "https"},
    )

    metadata = setup_endpoint.build_first_run_metadata(request)

    assert metadata.connection.browser_access == "unknown"
    assert metadata.bundled_single_user_auth_available is False
    assert metadata.manual_auth_required is True


def test_completed_setup_rejects_first_run_writes(monkeypatch, tmp_path, setup_client):
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

    response = setup_client.post(
        "/api/v1/setup/first-run/state",
        json={"step": "providers", "data": {}},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "setup_already_completed"


def test_legacy_completed_setup_rejects_first_run_writes_without_state_file(
    monkeypatch,
    tmp_path,
    setup_client,
):
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

    response = setup_client.post(
        "/api/v1/setup/first-run/state",
        json={"step": "providers", "data": {}},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "setup_already_completed"


def test_disabled_legacy_completed_setup_rejects_first_run_writes_without_state_file(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "get_status_snapshot",
        lambda: {
            "enabled": False,
            "setup_completed": True,
            "completed": True,
            "needs_setup": False,
            "auth_mode": "single_user",
            "allow_remote_setup_access": False,
            "remote_access_env_override": False,
            "remote_access_active": False,
        },
    )

    response = setup_client.post(
        "/api/v1/setup/first-run/state",
        json={"step": "providers", "data": {}},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "setup_already_completed"


def test_disabled_incomplete_setup_rejects_first_run_writes_without_state_file(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "get_status_snapshot",
        lambda: {
            "enabled": False,
            "setup_completed": False,
            "completed": False,
            "needs_setup": False,
            "auth_mode": "single_user",
            "allow_remote_setup_access": False,
            "remote_access_env_override": False,
            "remote_access_active": False,
        },
    )

    response = setup_client.post(
        "/api/v1/setup/first-run/state",
        json={"step": "providers", "data": {}},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "setup_disabled"


def test_enabled_inconsistent_not_needed_setup_rejects_first_run_writes_without_state_file(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "get_status_snapshot",
        lambda: {
            "enabled": True,
            "setup_completed": False,
            "completed": False,
            "needs_setup": False,
            "auth_mode": "single_user",
            "allow_remote_setup_access": False,
            "remote_access_env_override": False,
            "remote_access_active": False,
        },
    )

    response = setup_client.post(
        "/api/v1/setup/first-run/state",
        json={"step": "providers", "data": {}},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "setup_already_completed"


def test_first_run_state_rejects_unknown_step_with_empty_data(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)

    response = setup_client.post(
        "/api/v1/setup/first-run/state",
        json={"step": "not_real", "data": {}},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "unsupported_first_run_step_data"
    state = FirstRunStateStore(state_path).load()
    assert state.current_step != "not_real"
    assert "not_real" not in state.completed_steps
    assert "not_real" not in state.step_data


def test_first_run_state_rejects_unsupported_public_step_data(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)

    response = setup_client.post(
        "/api/v1/setup/first-run/state",
        json={
            "step": "providers",
            "data": {
                "value": "sk-raw",
                "endpoint_config": {"url": "http://localhost:11434"},
                "acknowledged": True,
            },
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "unsupported_first_run_step_data"
    state_response = setup_client.get("/api/v1/setup/first-run/state")
    assert state_response.status_code == 200
    assert "sk-raw" not in str(state_response.json())


def test_first_run_state_rejects_secret_like_allowed_public_step_value(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)

    response = setup_client.post(
        "/api/v1/setup/first-run/state",
        json={
            "step": "providers",
            "data": {
                "acknowledged": True,
                "default_provider": "sk-raw",
            },
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "unsupported_first_run_step_data"
    state_response = setup_client.get("/api/v1/setup/first-run/state")
    assert state_response.status_code == 200
    assert "sk-raw" not in str(state_response.json())


def test_first_run_state_rejects_huggingface_token_like_allowed_public_step_value(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)

    response = setup_client.post(
        "/api/v1/setup/first-run/state",
        json={
            "step": "providers",
            "data": {
                "acknowledged": True,
                "default_provider": "hf_abcdef1234567890",
            },
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "unsupported_first_run_step_data"
    state_response = setup_client.get("/api/v1/setup/first-run/state")
    assert state_response.status_code == 200
    assert "hf_abcdef1234567890" not in str(state_response.json())


def test_first_run_state_persists_allowed_public_step_data(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)

    response = setup_client.post(
        "/api/v1/setup/first-run/state",
        json={
            "step": "providers",
            "data": {
                "acknowledged": True,
                "default_provider": "openai",
            },
        },
    )

    assert response.status_code == 200
    providers_data = response.json()["step_data"]["providers"]
    assert providers_data["acknowledged"] is True
    assert providers_data["default_provider"] == "openai"


def test_first_run_state_get_projects_only_public_step_data(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    state = FirstRunStateStore(state_path).update_step(
        "providers",
        {
            "acknowledged": True,
            "default_provider": "openai",
        },
    )
    payload = setup_endpoint.json.loads(state.model_dump_json())
    payload["step_data"]["providers"].update(
        {
            "api_key": "sk-raw",
            "nested": {"token": "secret-token"},
        }
    )
    state_path.write_text(setup_endpoint.json.dumps(payload), encoding="utf-8")

    response = setup_client.get("/api/v1/setup/first-run/state")

    assert response.status_code == 200
    body = response.json()
    providers_data = body["step_data"]["providers"]
    assert providers_data == {
        "acknowledged": True,
        "default_provider": "openai",
    }
    rendered_body = str(body)
    assert "sk-raw" not in rendered_body
    assert "secret-token" not in rendered_body


def test_first_run_state_get_filters_secret_like_allowed_public_step_values(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    state = FirstRunStateStore(state_path).update_step(
        "setup_path",
        {
            "acknowledged": True,
            "selected_options": {
                "adapter": "sk-raw",
                "mode": "local",
            },
        },
    )
    payload = setup_endpoint.json.loads(state.model_dump_json())
    state_path.write_text(setup_endpoint.json.dumps(payload), encoding="utf-8")

    response = setup_client.get("/api/v1/setup/first-run/state")

    assert response.status_code == 200
    body = response.json()
    assert body["step_data"]["setup_path"] == {"acknowledged": True}
    assert "sk-raw" not in str(body)


def test_first_run_state_get_filters_huggingface_token_like_allowed_public_step_values(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    state = FirstRunStateStore(state_path).update_step(
        "providers",
        {
            "acknowledged": True,
            "default_provider": "hf_abcdef1234567890",
        },
    )
    payload = setup_endpoint.json.loads(state.model_dump_json())
    state_path.write_text(setup_endpoint.json.dumps(payload), encoding="utf-8")

    response = setup_client.get("/api/v1/setup/first-run/state")

    assert response.status_code == 200
    body = response.json()
    assert body["step_data"]["providers"] == {"acknowledged": True}
    assert "hf_abcdef1234567890" not in str(body)


def test_first_run_state_get_filters_non_public_step_fields(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    state = FirstRunStateStore(state_path).update_step(
        "providers",
        {
            "acknowledged": True,
            "default_provider": "openai",
        },
    )
    payload = setup_endpoint.json.loads(state.model_dump_json())
    payload.update(
        {
            "current_step": "hf_abcdef1234567890",
            "completed_steps": ["providers", "not_real", "hf_abcdef1234567890"],
            "acknowledged_steps": ["providers", "hf_abcdef1234567890"],
            "skipped_steps": ["audio_defaults", "hf_abcdef1234567890"],
        }
    )
    state_path.write_text(setup_endpoint.json.dumps(payload), encoding="utf-8")

    response = setup_client.get("/api/v1/setup/first-run/state")

    assert response.status_code == 200
    body = response.json()
    assert body["current_step"] is None
    assert body["completed_steps"] == ["providers"]
    assert body["acknowledged_steps"] == ["providers"]
    assert body["skipped_steps"] == ["audio_defaults"]
    assert "hf_abcdef1234567890" not in str(body)
    assert "not_real" not in str(body)


def test_skipped_first_run_state_rejected_by_shared_write_guard(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)
    FirstRunStateStore(state_path).mark_skipped(reason="user_skip")
    monkeypatch.setattr(FirstRunStateStore, "update_step", _fail_if_state_update_reaches_endpoint)

    response = setup_client.post(
        "/api/v1/setup/first-run/state",
        json={"step": "providers", "data": {}},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "state_skipped"


def test_blocked_first_run_state_rejected_by_shared_write_guard(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)
    state_path.write_text("{", encoding="utf-8")
    recovered = FirstRunStateStore(state_path).load()
    assert recovered.status == FirstRunStatus.BLOCKED
    monkeypatch.setattr(FirstRunStateStore, "update_step", _fail_if_state_update_reaches_endpoint)

    response = setup_client.post(
        "/api/v1/setup/first-run/state",
        json={"step": "providers", "data": {}},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "state_blocked"
