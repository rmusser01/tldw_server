from configparser import ConfigParser

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


def _persist_required_first_run_step_data(
    store: FirstRunStateStore,
    *,
    skip_steps: set[str] | None = None,
) -> None:
    step_payloads = {
        "setup_path": {"acknowledged": True, "setup_path_key": "local"},
        "privacy_security": {"acknowledged": True, "local_only": True},
        "providers": {
            "acknowledged": True,
            "default_provider": "openai",
            "default_model": "gpt-4.1-mini",
        },
        "ingest_defaults": {
            "acknowledged": True,
            "allow_local_file_ingest": False,
            "chunking_profile": "balanced",
            "metadata_mode": "automatic",
        },
        "audio_defaults": {"acknowledged": True, "mode": "skip"},
        "optional_advanced": {
            "acknowledged": True,
            "rag": "defer",
            "storage_paths": "defer",
        },
    }
    skipped = skip_steps or set()
    for step, payload in step_payloads.items():
        if step not in skipped:
            store.update_step(step, payload)


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


@pytest.mark.parametrize("reason", ["hf_abcdef1234567890", "/Users/me/.env"])
def test_first_run_skip_endpoint_filters_unsafe_reason_before_persistence(
    monkeypatch,
    tmp_path,
    setup_client,
    reason,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)

    response = setup_client.post(
        "/api/v1/setup/first-run/skip",
        json={"reason": reason},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "skipped"
    assert body["skip_reason"] is None
    assert reason not in str(body)
    assert FirstRunStateStore(state_path).load().skip_reason is None
    assert reason not in state_path.read_text(encoding="utf-8")


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


def test_first_run_provider_save_places_new_key_in_api_section(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    config_path = tmp_path / "config.txt"
    config_path.write_text(
        "[API]\n" "default_api = openai\n" "\n" "[Local-API]\n" "ollama_api_IP = http://127.0.0.1:11434/v1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    monkeypatch.setattr(setup_endpoint.setup_manager, "get_config_file_path", lambda: config_path)
    _setup_needs_setup(monkeypatch)

    response = setup_client.post(
        "/api/v1/setup/first-run/providers",
        json={"provider_key": "openai", "api_key": "sk-abcdefghijklmnopqrstuvwxyz"},
    )

    parser = ConfigParser()
    parser.optionxform = str
    parser.read(config_path, encoding="utf-8")
    assert response.status_code == 200
    assert response.json()["status"] == "saved"
    assert parser.get("API", "openai_api_key") == "sk-abcdefghijklmnopqrstuvwxyz"
    assert not parser.has_option("Local-API", "openai_api_key")


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


def test_first_run_provider_save_writes_kobold_runtime_endpoint_key(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    config_path = tmp_path / "config.txt"
    config_path.write_text("[API]\ndefault_api = openai\n\n[Local-API]\n", encoding="utf-8")
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    monkeypatch.setattr(setup_endpoint.setup_manager, "get_config_file_path", lambda: config_path)
    _setup_needs_setup(monkeypatch)

    response = setup_client.post(
        "/api/v1/setup/first-run/providers",
        json={
            "provider_key": "koboldcpp",
            "base_url": "http://127.0.0.1:5001/api/v1/generate",
            "make_default": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "saved"
    assert body["make_default"] is True
    config_text = config_path.read_text(encoding="utf-8")
    assert "default_api = kobold" in config_text
    assert "kobold_api_IP = http://127.0.0.1:5001/api/v1/generate" in config_text
    assert "kobold_openai_api_IP" not in config_text


def test_completed_setup_rejects_provider_save_through_first_run_write_guard(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)
    store = FirstRunStateStore(state_path)
    _persist_required_first_run_step_data(store)
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


def test_first_run_provider_save_refreshes_runtime_config_cache(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)
    writes: list[dict[str, dict[str, str]]] = []
    refresh_calls: list[bool] = []

    def _capture_update_config(updates, *, create_backup=True):  # noqa: ARG001
        writes.append(updates)
        return None

    monkeypatch.setattr(setup_endpoint.setup_manager, "update_config", _capture_update_config)
    monkeypatch.setattr(
        setup_endpoint,
        "clear_config_cache",
        lambda: refresh_calls.append(True),
        raising=False,
    )

    response = setup_client.post(
        "/api/v1/setup/first-run/providers",
        json={"provider_key": "openai", "api_key": "sk-abcdefghijklmnopqrstuvwxyz"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "saved"
    assert body["requires_restart"] is False
    assert writes == [{"API": {"openai_api_key": "sk-abcdefghijklmnopqrstuvwxyz"}}]
    assert refresh_calls == [True]


def test_first_run_provider_save_requires_restart_when_cache_refresh_fails(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)

    monkeypatch.setattr(setup_endpoint.setup_manager, "update_config", lambda updates: None)

    def _raise_refresh_failure():
        raise RuntimeError("raw refresh failure")

    monkeypatch.setattr(setup_endpoint, "clear_config_cache", _raise_refresh_failure, raising=False)

    response = setup_client.post(
        "/api/v1/setup/first-run/providers",
        json={"provider_key": "openai", "api_key": "sk-abcdefghijklmnopqrstuvwxyz"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "saved"
    assert body["requires_restart"] is True
    assert body["failure_category"] == "config_cache_refresh_failed"
    assert "raw refresh failure" not in str(body)


def test_first_run_first_chat_endpoint_records_success_on_ready_response(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)

    async def _fake_verify_first_chat(*, provider, model, prompt):
        assert provider == "openai"
        assert model == "gpt-4.1-mini"
        assert prompt == "Say hello."
        return {
            "status": "ready",
            "provider": provider,
            "model": model,
            "response_id": "chatcmpl-first-run",
            "response_text": "Hello from setup.",
            "failure_category": None,
            "message": None,
        }

    monkeypatch.setattr(setup_endpoint, "verify_first_chat", _fake_verify_first_chat, raising=False)

    response = setup_client.post(
        "/api/v1/setup/first-run/first-chat",
        json={"provider": "openai", "model": "gpt-4.1-mini", "prompt": "Say hello."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["response_text"] == "Hello from setup."
    state = FirstRunStateStore(state_path).load()
    assert state.first_chat.completed is True
    assert state.first_chat.provider == "openai"
    assert state.first_chat.model == "gpt-4.1-mini"
    assert state.first_chat.response_id == "chatcmpl-first-run"


def test_first_run_first_chat_endpoint_failure_does_not_record_or_echo_raw_details(
    monkeypatch,
    tmp_path,
    setup_client,
):
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAuthenticationError
    from tldw_Server_API.app.core.Setup import first_chat_verifier

    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)
    raw_detail = "bad sk-secret-token at /Users/local/private/config.txt"

    async def _fake_call_chat_completion(**_kwargs):
        raise ChatAuthenticationError(raw_detail, provider="openai")

    monkeypatch.setattr(first_chat_verifier, "_call_chat_completion", _fake_call_chat_completion)

    response = setup_client.post(
        "/api/v1/setup/first-run/first-chat",
        json={"provider": "openai", "model": "gpt-4.1-mini"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert body["failure_category"] == "auth_failed"
    assert body["response_id"] is None
    assert body["response_text"] is None
    rendered_body = str(body)
    assert "sk-secret-token" not in rendered_body
    assert "/Users/local/private" not in rendered_body
    state = FirstRunStateStore(state_path).load()
    assert state.first_chat.completed is False


@pytest.mark.parametrize(
    "payload",
    [
        {"provider": "sk-secret-token", "model": "gpt-4.1-mini"},
        {"provider": "openai", "model": "/Users/me/.env"},
    ],
)
def test_first_run_first_chat_endpoint_rejects_unsafe_provider_or_model_before_verify(
    monkeypatch,
    tmp_path,
    setup_client,
    payload,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)

    async def _fake_verify_first_chat(**_kwargs):
        pytest.fail("unsafe first-chat metadata should be rejected before verification")

    monkeypatch.setattr(setup_endpoint, "verify_first_chat", _fake_verify_first_chat, raising=False)

    response = setup_client.post("/api/v1/setup/first-run/first-chat", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"] == setup_endpoint.UNSUPPORTED_FIRST_RUN_STEP_DATA_DETAIL
    assert not state_path.exists()


def test_first_run_complete_rejects_without_first_chat(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)
    completion_calls: list[bool] = []
    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "mark_setup_completed",
        lambda completed: completion_calls.append(completed),
    )

    response = setup_client.post(
        "/api/v1/setup/first-run/complete",
        json={"acknowledged_steps": list(REQUIRED_FIRST_RUN_STEPS)},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "first_chat_required"
    assert completion_calls == []


def test_first_run_complete_rejects_acknowledgements_without_persisted_required_step_data(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)
    FirstRunStateStore(state_path).record_first_chat_success(
        provider="openai",
        model="gpt-4.1-mini",
        response_id="chatcmpl-test",
    )
    completion_calls: list[bool] = []
    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "mark_setup_completed",
        lambda completed: completion_calls.append(completed),
    )

    response = setup_client.post(
        "/api/v1/setup/first-run/complete",
        json={"acknowledged_steps": list(REQUIRED_FIRST_RUN_STEPS)},
    )

    assert response.status_code == 409
    assert response.json()["detail"].startswith("required_steps_missing:")
    assert completion_calls == []
    assert FirstRunStateStore(state_path).load().status == FirstRunStatus.FIRST_CHAT_COMPLETE


def test_first_run_complete_rejects_missing_required_acknowledgements(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)
    store = FirstRunStateStore(state_path)
    _persist_required_first_run_step_data(store, skip_steps={"audio_defaults"})
    store.record_first_chat_success(
        provider="openai",
        model="gpt-4.1-mini",
        response_id="chatcmpl-test",
    )
    completion_calls: list[bool] = []
    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "mark_setup_completed",
        lambda completed: completion_calls.append(completed),
    )

    acknowledged_steps = [step for step in REQUIRED_FIRST_RUN_STEPS if step != "audio_defaults"]
    response = setup_client.post(
        "/api/v1/setup/first-run/complete",
        json={"acknowledged_steps": acknowledged_steps},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "required_steps_missing:audio_defaults"
    assert completion_calls == []
    assert FirstRunStateStore(state_path).load().status == FirstRunStatus.FIRST_CHAT_COMPLETE


def test_first_run_complete_succeeds_after_first_chat_and_required_acknowledgements(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)
    store = FirstRunStateStore(state_path)
    _persist_required_first_run_step_data(store)
    store.record_first_chat_success(
        provider="openai",
        model="gpt-4.1-mini",
        response_id="chatcmpl-test",
    )
    completion_calls: list[bool] = []

    def _mark_legacy_complete(completed):
        state = FirstRunStateStore(state_path).load()
        assert state.status == FirstRunStatus.COMPLETED
        assert set(state.acknowledged_steps) == set(REQUIRED_FIRST_RUN_STEPS)
        completion_calls.append(completed)

    monkeypatch.setattr(setup_endpoint.setup_manager, "mark_setup_completed", _mark_legacy_complete)

    response = setup_client.post(
        "/api/v1/setup/first-run/complete",
        json={"acknowledged_steps": list(REQUIRED_FIRST_RUN_STEPS)},
    )

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert completion_calls == [True]
    assert FirstRunStateStore(state_path).load().status == FirstRunStatus.COMPLETED


def test_first_run_defaults_endpoints_persist_state_and_allow_skip_or_defer(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)

    ingest_response = setup_client.post(
        "/api/v1/setup/first-run/ingest-defaults",
        json={
            "allow_local_file_ingest": True,
            "chunking_profile": "balanced",
            "metadata_mode": "basic",
            "allowed_local_roots": [str(tmp_path / "ingest")],
        },
    )
    audio_response = setup_client.post(
        "/api/v1/setup/first-run/audio-defaults",
        json={
            "mode": "skip",
            "stt_provider": None,
            "tts_provider": None,
            "tts_voice": None,
        },
    )
    advanced_response = setup_client.post(
        "/api/v1/setup/first-run/optional-advanced",
        json={
            "rag": "defer",
            "storage_paths": "skip",
            "values": {"notes": "state-only"},
        },
    )

    assert ingest_response.json() == {
        "status": "saved",
        "step": "ingest_defaults",
        "requires_restart": False,
    }
    assert audio_response.json() == {
        "status": "saved",
        "step": "audio_defaults",
        "requires_restart": False,
    }
    assert advanced_response.json() == {
        "status": "saved",
        "step": "optional_advanced",
        "requires_restart": False,
    }
    assert ingest_response.status_code == 200
    assert audio_response.status_code == 200
    assert advanced_response.status_code == 200
    state = FirstRunStateStore(state_path).load()
    assert state.step_data["ingest_defaults"]["acknowledged"] is True
    assert state.step_data["ingest_defaults"]["chunking_profile"] == "balanced"
    assert "allowed_local_roots" not in state.step_data["ingest_defaults"]
    assert str(tmp_path / "ingest") not in state_path.read_text(encoding="utf-8")
    assert state.step_data["audio_defaults"]["mode"] == "skip"
    assert state.step_data["optional_advanced"]["rag"] == "defer"
    assert state.step_data["optional_advanced"]["storage_paths"] == "skip"


def test_first_run_optional_advanced_allows_common_public_value_keys(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)

    response = setup_client.post(
        "/api/v1/setup/first-run/optional-advanced",
        json={
            "rag": "defer",
            "storage_paths": "defer",
            "values": {
                "default_value": "balanced",
                "display_input": "compact",
                "token_budget": "small",
            },
        },
    )

    assert response.status_code == 200
    state = FirstRunStateStore(state_path).load()
    assert state.step_data["optional_advanced"]["values"] == {
        "default_value": "balanced",
        "display_input": "compact",
        "token_budget": "small",
    }


def test_first_run_optional_advanced_allows_storage_path_values(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)
    storage_root = str(tmp_path / "storage" / "media")

    response = setup_client.post(
        "/api/v1/setup/first-run/optional-advanced",
        json={
            "rag": "defer",
            "storage_paths": "configure",
            "values": {
                "storage_paths": [storage_root],
                "storage_locations": {"media": storage_root},
            },
        },
    )

    assert response.status_code == 200
    state = FirstRunStateStore(state_path).load()
    assert state.step_data["optional_advanced"]["values"]["storage_paths"] == [storage_root]
    assert state.step_data["optional_advanced"]["values"]["storage_locations"] == {
        "media": storage_root,
    }


def test_first_run_ingest_defaults_rejects_relative_escape_roots(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)

    response = setup_client.post(
        "/api/v1/setup/first-run/ingest-defaults",
        json={
            "allow_local_file_ingest": True,
            "chunking_profile": "balanced",
            "metadata_mode": "automatic",
            "allowed_local_roots": ["../outside"],
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid_allowed_local_roots"


def test_first_run_optional_advanced_rejects_secret_like_values(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)

    response = setup_client.post(
        "/api/v1/setup/first-run/optional-advanced",
        json={
            "rag": "defer",
            "storage_paths": "defer",
            "values": {"api_key": "sk-secret-token"},
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == setup_endpoint.UNSUPPORTED_FIRST_RUN_STEP_DATA_DETAIL
    assert not state_path.exists()


@pytest.mark.parametrize(
    "secret_text",
    [
        "debug sk-secret-token",
        "cached hf_abcdef1234567890",
        "token github_pat_abcdef1234567890",
        "authorization Bearer secret-token-value",
    ],
)
def test_first_run_optional_advanced_rejects_embedded_secret_like_values(
    monkeypatch,
    tmp_path,
    setup_client,
    secret_text,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)

    response = setup_client.post(
        "/api/v1/setup/first-run/optional-advanced",
        json={
            "rag": "defer",
            "storage_paths": "defer",
            "values": {"notes": secret_text},
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == setup_endpoint.UNSUPPORTED_FIRST_RUN_STEP_DATA_DETAIL
    assert not state_path.exists()


@pytest.mark.parametrize(
    "secret_key",
    [
        "hf_abcdef1234567890",
        "/Users/me/.env",
    ],
)
def test_first_run_optional_advanced_rejects_secret_or_path_like_nested_keys(
    monkeypatch,
    tmp_path,
    setup_client,
    secret_key,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)

    response = setup_client.post(
        "/api/v1/setup/first-run/optional-advanced",
        json={
            "rag": "defer",
            "storage_paths": "defer",
            "values": {"notes": {secret_key: "state-only"}},
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == setup_endpoint.UNSUPPORTED_FIRST_RUN_STEP_DATA_DETAIL
    assert not state_path.exists()


def test_first_run_optional_advanced_rejects_nested_path_like_values(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)

    response = setup_client.post(
        "/api/v1/setup/first-run/optional-advanced",
        json={
            "rag": "defer",
            "storage_paths": "defer",
            "values": {"notes": {"debug_path": "/Users/me/.env"}},
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == setup_endpoint.UNSUPPORTED_FIRST_RUN_STEP_DATA_DETAIL
    assert not state_path.exists()


def test_first_run_optional_advanced_rejects_embedded_path_like_values(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)

    response = setup_client.post(
        "/api/v1/setup/first-run/optional-advanced",
        json={
            "rag": "defer",
            "storage_paths": "defer",
            "values": {"notes": "debug file is /Users/me/.env"},
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == setup_endpoint.UNSUPPORTED_FIRST_RUN_STEP_DATA_DETAIL
    assert not state_path.exists()


@pytest.mark.parametrize(
    "path_text",
    [
        "debug file is /root/.env",
        "config lives at /opt/app/config.txt",
        "volume path /Volumes/Data/.env",
        "mount path /mnt/data/config.txt",
    ],
)
def test_first_run_optional_advanced_rejects_broader_embedded_path_like_values(
    monkeypatch,
    tmp_path,
    setup_client,
    path_text,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)

    response = setup_client.post(
        "/api/v1/setup/first-run/optional-advanced",
        json={
            "rag": "defer",
            "storage_paths": "defer",
            "values": {"notes": path_text},
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == setup_endpoint.UNSUPPORTED_FIRST_RUN_STEP_DATA_DETAIL
    assert not state_path.exists()


def test_generic_first_run_state_rejects_invalid_ingest_roots(
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
            "step": "ingest_defaults",
            "data": {
                "acknowledged": True,
                "allow_local_file_ingest": True,
                "allowed_local_roots": ["../outside"],
            },
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid_allowed_local_roots"
    assert not state_path.exists()


def test_generic_first_run_state_validates_but_does_not_persist_ingest_roots(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)
    raw_root = str(tmp_path / "ingest")

    response = setup_client.post(
        "/api/v1/setup/first-run/state",
        json={
            "step": "ingest_defaults",
            "data": {
                "acknowledged": True,
                "allow_local_file_ingest": True,
                "chunking_profile": "balanced",
                "metadata_mode": "automatic",
                "allowed_local_roots": [raw_root],
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert "allowed_local_roots" not in body["step_data"]["ingest_defaults"]
    state = FirstRunStateStore(state_path).load()
    assert "allowed_local_roots" not in state.step_data["ingest_defaults"]
    assert raw_root not in state_path.read_text(encoding="utf-8")


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


def test_first_run_kobold_provider_validate_uses_native_validator(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)
    raw_token = "secret-local-token"

    async def _fail_openai_validate(payload):  # noqa: ARG001
        pytest.fail("koboldcpp validation should not use OpenAI-compatible /models")

    async def _fake_kobold_validate(payload):
        assert payload.provider_key == "koboldcpp"
        assert payload.base_url == "http://127.0.0.1:5001/api/v1/generate"
        assert payload.api_key == raw_token
        return setup_endpoint.SetupProviderValidationResponse(
            provider_key=payload.provider_key,
            status="ready",
        )

    monkeypatch.setattr(setup_endpoint, "validate_local_openai_endpoint", _fail_openai_validate)
    monkeypatch.setattr(
        setup_endpoint,
        "validate_native_kobold_endpoint",
        _fake_kobold_validate,
        raising=False,
    )

    response = setup_client.post(
        "/api/v1/setup/first-run/providers/validate",
        json={
            "provider_key": "koboldcpp",
            "base_url": "http://127.0.0.1:5001/api/v1/generate",
            "api_key": raw_token,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["provider_key"] == "koboldcpp"
    assert body["status"] == "ready"
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
    _persist_required_first_run_step_data(store)
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


@pytest.mark.parametrize(
    "state_factory",
    [
        "completed",
        "skipped",
        "blocked",
    ],
)
def test_setup_config_allows_terminal_first_run_state_for_recovery_surface(
    monkeypatch,
    tmp_path,
    setup_client,
    state_factory,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)
    store = FirstRunStateStore(state_path)
    if state_factory == "completed":
        _persist_required_first_run_step_data(store)
        store.record_first_chat_success(
            provider="openai",
            model="gpt-4.1-mini",
            response_id="chatcmpl-test",
        )
        store.mark_completed()
    elif state_factory == "skipped":
        store.mark_skipped(reason="user_skip")
    else:
        state_path.write_text("{", encoding="utf-8")
        assert store.load().status == FirstRunStatus.BLOCKED
    writes: list[dict[str, dict[str, str]]] = []

    def _capture_update_config(updates, *, create_backup=True):  # noqa: ARG001
        writes.append(updates)
        return tmp_path / "config.bak"

    monkeypatch.setattr(setup_endpoint.setup_manager, "update_config", _capture_update_config)

    response = setup_client.post(
        "/api/v1/setup/config",
        json={"updates": {"API": {"default_api": "openai"}}},
    )

    assert response.status_code == 200
    assert writes == [{"API": {"default_api": "openai"}}]


def test_setup_complete_route_uses_recovery_write_guard_without_terminal_state_block(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)
    FirstRunStateStore(state_path).mark_skipped(reason="user_skip")
    complete_calls: list[setup_endpoint.SetupCompleteRequest] = []

    async def _fake_complete_setup_flow(payload, background_tasks):  # noqa: ARG001
        complete_calls.append(payload)
        return {
            "success": True,
            "message": "completed",
            "requires_restart": True,
            "install_plan_submitted": False,
        }

    monkeypatch.setattr(setup_endpoint, "_complete_setup_flow", _fake_complete_setup_flow)

    response = setup_client.post("/api/v1/setup/complete", json={})

    assert response.status_code == 200
    assert len(complete_calls) == 1


def test_setup_config_refreshes_runtime_config_cache_after_write(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)
    refresh_calls: list[bool] = []

    monkeypatch.setattr(setup_endpoint.setup_manager, "update_config", lambda updates: tmp_path / "config.bak")
    monkeypatch.setattr(
        setup_endpoint,
        "clear_config_cache",
        lambda: refresh_calls.append(True),
        raising=False,
    )

    response = setup_client.post(
        "/api/v1/setup/config",
        json={"updates": {"API": {"default_api": "openai"}}},
    )

    assert response.status_code == 200
    assert response.json()["requires_restart"] is True
    assert refresh_calls == [True]


@pytest.mark.parametrize(
    "updates",
    [
        {"Setup": {"setup_completed": True}},
        {"Setup": {"enable_first_time_setup": False}},
        {"setup": {"SETUP_COMPLETED": "true"}},
    ],
)
def test_setup_config_rejects_lifecycle_flag_updates(
    monkeypatch,
    tmp_path,
    setup_client,
    updates,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)
    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "update_config",
        lambda *_args, **_kwargs: pytest.fail("lifecycle flags must not hit update_config"),
    )

    response = setup_client.post("/api/v1/setup/config", json={"updates": updates})

    assert response.status_code == 400
    assert response.json()["detail"] == "setup_lifecycle_flags_not_allowed"
    assert FirstRunStateStore(state_path).load().status == FirstRunStatus.NOT_STARTED


def test_setup_complete_rejects_without_first_chat_and_does_not_mark_legacy_complete(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)
    completion_calls: list[bool] = []
    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "mark_setup_completed",
        lambda completed: completion_calls.append(completed),
    )

    response = setup_client.post("/api/v1/setup/complete", json={})

    assert response.status_code == 409
    assert response.json()["detail"] == "first_chat_required"
    assert completion_calls == []


def test_setup_complete_marks_legacy_complete_only_after_first_run_completion(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)
    store = FirstRunStateStore(state_path)
    _persist_required_first_run_step_data(store)
    store.record_first_chat_success(
        provider="openai",
        model="gpt-4.1-mini",
        response_id="chatcmpl-test",
    )
    completion_calls: list[bool] = []
    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "mark_setup_completed",
        lambda completed: completion_calls.append(completed),
    )

    response = setup_client.post("/api/v1/setup/complete", json={})

    assert response.status_code == 200
    assert FirstRunStateStore(state_path).load().status == FirstRunStatus.COMPLETED
    assert completion_calls == [True]


def test_setup_complete_does_not_persist_first_run_completion_when_legacy_write_fails(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)
    store = FirstRunStateStore(state_path)
    _persist_required_first_run_step_data(store)
    store.record_first_chat_success(
        provider="openai",
        model="gpt-4.1-mini",
        response_id="chatcmpl-test",
    )

    def _raise_legacy_completion_failure(completed):  # noqa: ARG001
        raise RuntimeError("raw legacy config path /tmp/secret-config")

    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "mark_setup_completed",
        _raise_legacy_completion_failure,
    )

    response = setup_client.post("/api/v1/setup/complete", json={})

    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to persist setup completion."}
    state = FirstRunStateStore(state_path).load()
    assert state.status == FirstRunStatus.FIRST_CHAT_COMPLETE
    assert state.completed_at is None


def test_setup_complete_does_not_mark_completed_when_disable_prompt_write_fails(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)
    store = FirstRunStateStore(state_path)
    _persist_required_first_run_step_data(store)
    store.record_first_chat_success(
        provider="openai",
        model="gpt-4.1-mini",
        response_id="chatcmpl-test",
    )
    completion_calls: list[bool] = []

    def _raise_disable_prompt_write_failure(updates, *, create_backup=True):  # noqa: ARG001
        if updates == {setup_endpoint.setup_manager.SETUP_SECTION: {"enable_first_time_setup": False}}:
            raise RuntimeError("raw disable prompt path /tmp/config.txt")
        return tmp_path / "config.bak"

    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "mark_setup_completed",
        lambda completed: completion_calls.append(completed),
    )
    monkeypatch.setattr(setup_endpoint.setup_manager, "update_config", _raise_disable_prompt_write_failure)

    response = setup_client.post(
        "/api/v1/setup/complete",
        json={"disable_first_time_setup": True},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to persist setup completion."}
    assert completion_calls == []
    state = FirstRunStateStore(state_path).load()
    assert state.status == FirstRunStatus.FIRST_CHAT_COMPLETE
    assert state.completed_at is None


def test_setup_complete_does_not_mark_legacy_when_first_run_completion_write_fails(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)
    store = FirstRunStateStore(state_path)
    _persist_required_first_run_step_data(store)
    store.record_first_chat_success(
        provider="openai",
        model="gpt-4.1-mini",
        response_id="chatcmpl-test",
    )
    completion_calls: list[bool] = []
    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "mark_setup_completed",
        lambda completed: completion_calls.append(completed),
    )
    original_save_unlocked = FirstRunStateStore._save_unlocked

    def _fail_completed_state_save(self, state):
        if state.status == FirstRunStatus.COMPLETED:
            raise OSError("raw state path /tmp/first-run-state")
        return original_save_unlocked(self, state)

    monkeypatch.setattr(FirstRunStateStore, "_save_unlocked", _fail_completed_state_save)

    response = setup_client.post("/api/v1/setup/complete", json={})

    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to persist setup completion."}
    assert completion_calls == []
    state = FirstRunStateStore(state_path).load()
    assert state.status == FirstRunStatus.FIRST_CHAT_COMPLETE
    assert state.completed_at is None


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
                "default_model": "gpt-4.1-mini",
            },
        },
    )

    assert response.status_code == 200
    providers_data = response.json()["step_data"]["providers"]
    assert providers_data["acknowledged"] is True
    assert providers_data["default_provider"] == "openai"
    assert providers_data["default_model"] == "gpt-4.1-mini"


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
            "default_model": "gpt-4.1-mini",
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
        "default_model": "gpt-4.1-mini",
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


def test_first_run_state_get_filters_unsafe_first_chat_metadata(
    monkeypatch,
    tmp_path,
    setup_client,
):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    state = FirstRunStateStore(state_path).record_first_chat_success(
        provider="openai",
        model="gpt-4.1-mini",
        response_id="chatcmpl-test",
    )
    payload = setup_endpoint.json.loads(state.model_dump_json())
    payload["first_chat"].update(
        {
            "provider": "sk-secret-token",
            "model": "/Users/me/.env",
            "response_id": "hf_abcdef1234567890",
        }
    )
    state_path.write_text(setup_endpoint.json.dumps(payload), encoding="utf-8")

    response = setup_client.get("/api/v1/setup/first-run/state")

    assert response.status_code == 200
    body = response.json()
    assert body["first_chat"]["completed"] is True
    assert body["first_chat"]["provider"] is None
    assert body["first_chat"]["model"] is None
    assert body["first_chat"]["response_id"] is None
    rendered_body = str(body)
    assert "sk-secret-token" not in rendered_body
    assert "/Users/me/.env" not in rendered_body
    assert "hf_abcdef1234567890" not in rendered_body


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
