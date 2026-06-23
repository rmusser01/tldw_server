from __future__ import annotations

import stat
import subprocess
import sys
from configparser import ConfigParser
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import llamacpp as lp
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal


def _admin_principal() -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        api_key_id=None,
        subject=None,
        token_type="access",
        jti=None,
        roles=["admin"],
        permissions=[],
        is_admin=True,
        org_ids=[],
        team_ids=[],
    )


class _Logger:
    def error(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        return


class _ManagerWithoutHandler:
    logger = _Logger()
    llamacpp = None


class _HandlerWithConfig:
    def __init__(self, executable_path: Path, **config_overrides: Any) -> None:
        config_values = {
            "enabled": True,
            "executable_path": executable_path,
            "models_dir": Path("models/gguf_models"),
            "default_host": "127.0.0.1",
            "default_port": 8080,
            "default_threads": None,
            "default_n_gpu_layers": 0,
            "default_ctx_size": 2048,
            "allow_unvalidated_args": False,
            "allow_cli_secrets": False,
            "port_autoselect": True,
            "port_probe_max": 10,
            "allowed_paths": [],
            "log_output_file": None,
        }
        config_values.update(config_overrides)
        self.config = type(
            "Config",
            (),
            config_values,
        )()
        self._active_server_model = None
        self._active_server_host = None
        self._active_server_port = None
        self._active_server_process = None


class _ManagerWithHandler:
    logger = _Logger()

    def __init__(self, executable_path: Path, **config_overrides: Any) -> None:
        self.llamacpp = _HandlerWithConfig(executable_path, **config_overrides)


class _ExistingManagementManager:
    logger = _Logger()

    def __init__(self) -> None:
        self.llamacpp = self

    async def start_server(self, *, backend: str, model_name: str, server_args=None, **kwargs):  # noqa: ANN001
        _ = (backend, server_args, kwargs)
        return {"model": model_name}

    async def stop_server(self, *, backend: str, **kwargs):  # noqa: ANN001
        _ = (backend, kwargs)
        return "stopped"

    async def get_server_status(self, backend: str):
        _ = backend
        return {"status": "running"}

    async def list_models(self):
        return ["existing.gguf"]


def _make_app_with_manager(manager) -> FastAPI:  # noqa: ANN001
    app = FastAPI()
    app.include_router(lp.router, prefix="/api/v1")
    app.state.llm_manager = manager

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
        principal = _admin_principal()
        ip = request.client.host if getattr(request, "client", None) else None
        ua = request.headers.get("User-Agent") if getattr(request, "headers", None) else None
        request_id = request.headers.get("X-Request-ID") if getattr(request, "headers", None) else None
        request.state.auth = AuthContext(
            principal=principal,
            ip=ip,
            user_agent=ua,
            request_id=request_id,
        )
        return principal

    async def _fake_check_rate_limit() -> None:
        return

    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal
    app.dependency_overrides[auth_deps.check_rate_limit] = _fake_check_rate_limit
    app.dependency_overrides[lp.check_rate_limit] = _fake_check_rate_limit
    return app


def _llamacpp_parser(**overrides: str) -> ConfigParser:
    parser = ConfigParser()
    parser.add_section("LlamaCpp")
    values = {
        "enabled": "true",
        "executable_path": "vendor/llama.cpp/server",
        "models_dir": "models/gguf_models",
        "default_host": "127.0.0.1",
        "default_port": "8080",
        "default_threads": "",
        "default_n_gpu_layers": "0",
        "default_ctx_size": "2048",
        "allow_unvalidated_args": "false",
        "allow_cli_secrets": "false",
        "port_autoselect": "true",
        "port_probe_max": "10",
        "allowed_paths": "models/gguf_models, /srv/models",
        "log_output_file": "",
    }
    values.update(overrides)
    parser["LlamaCpp"] = values
    return parser


@pytest.mark.unit
def test_llamacpp_config_reports_saved_active_and_restart_required(monkeypatch):
    app = _make_app_with_manager(_ManagerWithoutHandler())

    monkeypatch.setattr(
        lp.llamacpp_config_service,
        "get_config_state",
        lambda llm_manager: {
            "saved_config": {
                "enabled": True,
                "executable_path": "vendor/llama.cpp/server",
                "models_dir": "models/gguf_models",
                "default_host": "127.0.0.1",
                "default_port": 8080,
                "allowed_paths": ["models/gguf_models"],
            },
            "active_config": {"handler_configured": False},
            "restart_required": True,
            "restart_reasons": ["handler_not_configured"],
            "env_overrides": {"models_dir": False},
            "warnings": ["handler is not configured"],
        },
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/llamacpp/config")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["saved_config"]["enabled"] is True
    assert body["active_config"]["handler_configured"] is False
    assert body["restart_required"] is True
    assert body["restart_reasons"] == ["handler_not_configured"]
    assert body["env_overrides"]["models_dir"] is False
    assert body["warnings"] == ["handler is not configured"]


@pytest.mark.unit
def test_llamacpp_config_service_reports_saved_vs_active_missing_handler(monkeypatch):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_config_service

    monkeypatch.setattr(llamacpp_config_service, "load_comprehensive_config", _llamacpp_parser)
    monkeypatch.setattr(
        llamacpp_config_service,
        "get_env_overrides",
        lambda: {field: False for field in llamacpp_config_service.LLAMACPP_ENV_OVERRIDES},
    )

    state = llamacpp_config_service.get_config_state(_ManagerWithoutHandler())

    assert state["saved_config"]["enabled"] is True
    assert state["saved_config"]["allowed_paths"] == ["models/gguf_models", "/srv/models"]
    assert state["active_config"] == {"handler_configured": False}
    assert state["restart_required"] is True
    assert state["restart_reasons"] == ["handler_not_configured"]


@pytest.mark.unit
def test_llamacpp_config_service_reports_default_port_change_restart_required(monkeypatch):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_config_service

    monkeypatch.setattr(
        llamacpp_config_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(default_port="9090"),
    )
    monkeypatch.setattr(
        llamacpp_config_service,
        "get_env_overrides",
        lambda: {field: False for field in llamacpp_config_service.LLAMACPP_ENV_OVERRIDES},
    )

    state = llamacpp_config_service.get_config_state(
        _ManagerWithHandler(Path("vendor/llama.cpp/server"), default_port=8080)
    )

    assert state["active_config"]["default_port"] == 8080
    assert state["saved_config"]["default_port"] == 9090
    assert state["restart_required"] is True
    assert "default_port_changed" in state["restart_reasons"]


@pytest.mark.unit
def test_llamacpp_config_service_reports_security_flag_change_restart_required(monkeypatch):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_config_service

    monkeypatch.setattr(
        llamacpp_config_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(allow_cli_secrets="true"),
    )
    monkeypatch.setattr(
        llamacpp_config_service,
        "get_env_overrides",
        lambda: {field: False for field in llamacpp_config_service.LLAMACPP_ENV_OVERRIDES},
    )

    state = llamacpp_config_service.get_config_state(
        _ManagerWithHandler(Path("vendor/llama.cpp/server"), allow_cli_secrets=False)
    )

    assert state["active_config"]["allow_cli_secrets"] is False
    assert state["saved_config"]["allow_cli_secrets"] is True
    assert state["restart_required"] is True
    assert "allow_cli_secrets_changed" in state["restart_reasons"]


@pytest.mark.unit
def test_llamacpp_config_service_reports_malformed_saved_config_warnings(monkeypatch):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_config_service

    monkeypatch.setattr(
        llamacpp_config_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(enabled="not-a-bool", default_port="not-a-port"),
    )

    state = llamacpp_config_service.get_config_state(_ManagerWithoutHandler())

    assert state["saved_config"]["enabled"] is False
    assert state["saved_config"]["default_port"] is None
    assert state["restart_required"] is False
    assert "Invalid boolean value for LlamaCpp.enabled." in state["warnings"]
    assert "Invalid integer value for LlamaCpp.default_port." in state["warnings"]


@pytest.mark.unit
def test_llamacpp_config_put_rejects_environment_overridden_fields(monkeypatch):
    app = _make_app_with_manager(_ManagerWithoutHandler())

    def _reject_locked(payload, llm_manager):  # noqa: ANN001
        _ = (payload, llm_manager)
        raise lp.HTTPException(
            status_code=409,
            detail={
                "message": "Some llama.cpp config fields are controlled by environment variables.",
                "locked_fields": ["models_dir"],
            },
        )

    monkeypatch.setattr(lp.llamacpp_config_service, "update_config_state", _reject_locked)

    with TestClient(app) as client:
        response = client.put("/api/v1/llamacpp/config", json={"models_dir": "models/new"})

    assert response.status_code == 409
    assert response.json()["detail"]["locked_fields"] == ["models_dir"]


@pytest.mark.unit
def test_llamacpp_config_service_rejects_environment_overridden_fields(monkeypatch):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_config_service

    monkeypatch.setattr(
        llamacpp_config_service,
        "get_env_overrides",
        lambda: {**{field: False for field in llamacpp_config_service.LLAMACPP_ENV_OVERRIDES}, "models_dir": True},
    )

    with pytest.raises(lp.HTTPException) as exc_info:
        llamacpp_config_service.update_config_state({"models_dir": "models/new"}, _ManagerWithoutHandler())

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["locked_fields"] == ["models_dir"]


@pytest.mark.unit
def test_llamacpp_registered_model_paths_is_not_advertised_as_env_override():
    from tldw_Server_API.app.core.Local_LLM import llamacpp_config_service

    assert "registered_model_paths" not in llamacpp_config_service.LLAMACPP_ENV_OVERRIDES
    assert "LLAMACPP_REGISTERED_MODEL_PATHS" not in llamacpp_config_service.LLAMACPP_ENV_OVERRIDES.values()


@pytest.mark.unit
def test_llamacpp_config_write_lock_path_is_config_sibling(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_config_lock

    config_path = tmp_path / "config.txt"
    monkeypatch.setattr(llamacpp_config_lock.setup_manager, "get_config_file_path", lambda: config_path)

    assert llamacpp_config_lock.llamacpp_config_lock_path() == tmp_path / ".llamacpp.lock"


@pytest.mark.unit
def test_llamacpp_config_put_updates_with_setup_manager_and_refreshes_cache(monkeypatch):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_config_service

    calls: list[tuple[str, Any]] = []

    monkeypatch.setattr(
        llamacpp_config_service,
        "get_env_overrides",
        lambda: {field: False for field in llamacpp_config_service.LLAMACPP_ENV_OVERRIDES},
    )
    monkeypatch.setattr(
        llamacpp_config_service.setup_manager,
        "update_config",
        lambda updates: calls.append(("update_config", updates)),
    )
    monkeypatch.setattr(
        llamacpp_config_service,
        "refresh_config_cache",
        lambda: calls.append(("refresh_config_cache", None)),
    )
    monkeypatch.setattr(
        llamacpp_config_service,
        "get_config_state",
        lambda llm_manager: {
            "saved_config": {
                "enabled": True,
                "models_dir": "models/new",
                "allowed_paths": ["models/new"],
            },
            "active_config": {"handler_configured": False},
            "restart_required": True,
            "restart_reasons": ["handler_not_configured"],
            "env_overrides": {"models_dir": False, "allowed_paths": False},
            "warnings": [],
        },
    )

    class FakeLock:
        def __enter__(self) -> "FakeLock":
            calls.append(("lock_enter", None))
            return self

        def __exit__(self, *exc: Any) -> None:
            calls.append(("lock_exit", None))

    monkeypatch.setattr(llamacpp_config_service, "llamacpp_config_write_lock", lambda: FakeLock(), raising=False)

    result = llamacpp_config_service.update_config_state(
        {"models_dir": "models/new", "allowed_paths": ["models/new"], "default_port": None},
        _ManagerWithoutHandler(),
    )

    assert calls == [
        ("lock_enter", None),
        ("update_config", {"LlamaCpp": {"models_dir": "models/new", "allowed_paths": "models/new"}}),
        ("refresh_config_cache", None),
        ("lock_exit", None),
    ]
    assert result["saved_config"]["models_dir"] == "models/new"


@pytest.mark.unit
def test_llamacpp_config_put_explicit_null_clears_nullable_scalar(monkeypatch):
    from tldw_Server_API.app.api.v1.schemas.llamacpp_admin_schemas import LlamaCppConfigUpdateRequest
    from tldw_Server_API.app.core.Local_LLM import llamacpp_config_service

    calls: list[tuple[str, Any]] = []

    monkeypatch.setattr(
        llamacpp_config_service,
        "get_env_overrides",
        lambda: {field: False for field in llamacpp_config_service.LLAMACPP_ENV_OVERRIDES},
    )
    monkeypatch.setattr(
        llamacpp_config_service.setup_manager,
        "update_config",
        lambda updates: calls.append(("update_config", updates)),
    )
    monkeypatch.setattr(
        llamacpp_config_service,
        "refresh_config_cache",
        lambda: calls.append(("refresh_config_cache", None)),
    )
    monkeypatch.setattr(
        llamacpp_config_service,
        "get_config_state",
        lambda llm_manager: {
            "saved_config": {"enabled": False, "default_host": None, "allowed_paths": []},
            "active_config": {"handler_configured": False},
            "restart_required": False,
            "restart_reasons": [],
            "env_overrides": {},
            "warnings": [],
        },
    )

    llamacpp_config_service.update_config_state(
        LlamaCppConfigUpdateRequest(default_host=None),
        _ManagerWithoutHandler(),
    )

    assert calls == [
        ("update_config", {"LlamaCpp": {"default_host": ""}}),
        ("refresh_config_cache", None),
    ]


@pytest.mark.unit
def test_llamacpp_config_put_explicit_null_clears_nullable_integer_through_setup_validation(monkeypatch):
    from tldw_Server_API.app.api.v1.schemas.llamacpp_admin_schemas import LlamaCppConfigUpdateRequest
    from tldw_Server_API.app.core.Local_LLM import llamacpp_config_service
    from tldw_Server_API.app.core.Setup import setup_manager

    calls: list[tuple[str, Any]] = []
    payload = LlamaCppConfigUpdateRequest(default_port=None)
    updates = llamacpp_config_service._payload_to_updates(payload)
    parser = _llamacpp_parser(default_port="8080")

    assert updates == {"default_port": ""}
    setup_manager._validate_updates(parser, {"LlamaCpp": updates})

    monkeypatch.setattr(
        llamacpp_config_service,
        "get_env_overrides",
        lambda: {field: False for field in llamacpp_config_service.LLAMACPP_ENV_OVERRIDES},
    )

    def _capture_update(update_payload):  # noqa: ANN001
        calls.append(("update_config", update_payload))

    monkeypatch.setattr(llamacpp_config_service.setup_manager, "update_config", _capture_update)
    monkeypatch.setattr(
        llamacpp_config_service,
        "refresh_config_cache",
        lambda: calls.append(("refresh_config_cache", None)),
    )
    monkeypatch.setattr(
        llamacpp_config_service,
        "get_config_state",
        lambda llm_manager: {
            "saved_config": {"enabled": False, "default_port": None, "allowed_paths": []},
            "active_config": {"handler_configured": False},
            "restart_required": False,
            "restart_reasons": [],
            "env_overrides": {},
            "warnings": [],
        },
    )

    llamacpp_config_service.update_config_state(
        payload,
        _ManagerWithoutHandler(),
    )

    assert calls == [
        ("update_config", {"LlamaCpp": {"default_port": ""}}),
        ("refresh_config_cache", None),
    ]


@pytest.mark.unit
def test_llamacpp_config_schema_rejects_boolean_null_clears():
    from pydantic import ValidationError
    from tldw_Server_API.app.api.v1.schemas.llamacpp_admin_schemas import LlamaCppConfigUpdateRequest

    # Boolean fields are not semantically clearable: they are concrete feature
    # flags in [LlamaCpp], so clients must set true/false instead of null.
    with pytest.raises(ValidationError):
        LlamaCppConfigUpdateRequest(enabled=None)


@pytest.mark.unit
def test_llamacpp_config_put_returns_safe_error_when_config_write_fails(monkeypatch):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_config_service

    app = _make_app_with_manager(_ManagerWithoutHandler())
    monkeypatch.setattr(
        llamacpp_config_service,
        "get_env_overrides",
        lambda: {field: False for field in llamacpp_config_service.LLAMACPP_ENV_OVERRIDES},
    )

    def _fail_update(updates):  # noqa: ANN001
        _ = updates
        raise RuntimeError("failed writing /private/sensitive/config.txt")

    monkeypatch.setattr(llamacpp_config_service.setup_manager, "update_config", _fail_update)

    with TestClient(app) as client:
        response = client.put("/api/v1/llamacpp/config", json={"models_dir": "models/new"})

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to update llama.cpp configuration."
    assert "/private/sensitive" not in response.text


@pytest.mark.unit
def test_llamacpp_config_rejects_multiline_values_before_writing(monkeypatch):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_config_service

    updates: list[dict[str, dict[str, str]]] = []
    monkeypatch.setattr(
        llamacpp_config_service,
        "get_env_overrides",
        lambda: {field: False for field in llamacpp_config_service.LLAMACPP_ENV_OVERRIDES},
    )
    monkeypatch.setattr(llamacpp_config_service.setup_manager, "update_config", updates.append)

    with pytest.raises(lp.HTTPException) as exc_info:
        llamacpp_config_service.update_config_state(
            {"default_host": "127.0.0.1\nbad = true"},
            _ManagerWithoutHandler(),
        )

    assert exc_info.value.status_code == 400
    assert "line breaks" in str(exc_info.value.detail)
    assert updates == []


@pytest.mark.unit
def test_llamacpp_config_rejects_delimited_allowed_paths_before_writing(monkeypatch):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_config_service

    updates: list[dict[str, dict[str, str]]] = []
    monkeypatch.setattr(
        llamacpp_config_service,
        "get_env_overrides",
        lambda: {field: False for field in llamacpp_config_service.LLAMACPP_ENV_OVERRIDES},
    )
    monkeypatch.setattr(llamacpp_config_service.setup_manager, "update_config", updates.append)

    with pytest.raises(lp.HTTPException) as exc_info:
        llamacpp_config_service.update_config_state(
            {"allowed_paths": ["/srv/models,shadow"]},
            _ManagerWithoutHandler(),
        )

    assert exc_info.value.status_code == 400
    assert "cannot contain comma" in str(exc_info.value.detail)
    assert updates == []


@pytest.mark.unit
def test_setup_manager_rejects_multiline_values_as_writer_defense():
    from tldw_Server_API.app.core.Setup import setup_manager

    parser = _llamacpp_parser(default_host="127.0.0.1")

    with pytest.raises(ValueError) as exc_info:
        setup_manager._validate_updates(parser, {"LlamaCpp": {"default_host": "127.0.0.1\nbad = true"}})

    assert "line breaks" in str(exc_info.value)


@pytest.mark.unit
def test_llamacpp_validate_default_is_stat_only_and_does_not_execute(tmp_path: Path):
    script = tmp_path / "llama-server"
    marker = tmp_path / "probe-ran"
    script.write_text(f"#!/bin/sh\ntouch {marker}\nprintf 'llama-server version test\\n'\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    app = _make_app_with_manager(_ManagerWithoutHandler())

    with TestClient(app) as client:
        response = client.post("/api/v1/llamacpp/validate", json={"binary_path": str(script)})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["valid"] is True
    assert body["exists"] is True
    assert body["executable"] is True
    assert body["version_output"] is None
    assert body["help_output"] is None
    assert not marker.exists()


@pytest.mark.unit
def test_llamacpp_validate_endpoint_offloads_probe_to_threadpool(monkeypatch, tmp_path: Path):
    calls: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []
    probe_path = str(tmp_path / "llama-server")

    def fake_validate(binary_path: str, timeout_seconds: float, **kwargs: Any) -> dict[str, Any]:
        return {
            "valid": True,
            "exists": True,
            "executable": True,
            "resolved_path": binary_path,
            "version_output": None,
            "help_output": None,
            "warnings": [f"timeout={timeout_seconds}", f"probe={kwargs['run_probe']}"],
        }

    async def fake_run_in_threadpool(func: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append((func, args, kwargs))
        return func(*args, **kwargs)

    monkeypatch.setattr(lp, "run_in_threadpool", fake_run_in_threadpool)
    monkeypatch.setattr(lp.llamacpp_config_service, "validate_binary", fake_validate)
    app = _make_app_with_manager(_ManagerWithoutHandler())

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/llamacpp/validate",
            json={"binary_path": probe_path, "timeout_seconds": 2.0, "run_probe": True},
        )

    assert response.status_code == 200, response.text
    assert len(calls) == 1
    func, args, kwargs = calls[0]
    assert func is fake_validate
    assert args == (probe_path, 2.0)
    assert isinstance(kwargs["llm_manager"], _ManagerWithoutHandler)
    assert kwargs["run_probe"] is True


@pytest.mark.unit
def test_llamacpp_validate_run_probe_requires_saved_or_active_binary_path(tmp_path: Path):
    script = tmp_path / "llama-server"
    marker = tmp_path / "probe-ran"
    script.write_text(f"#!/bin/sh\ntouch {marker}\nprintf 'llama-server version test\\n'\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    app = _make_app_with_manager(_ManagerWithoutHandler())

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/llamacpp/validate",
            json={"binary_path": str(script), "run_probe": True},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["valid"] is False
    assert body["exists"] is True
    assert body["executable"] is True
    assert body["version_output"] is None
    assert "Binary probe requires the path to be saved first." in body["warnings"]
    assert not marker.exists()


@pytest.mark.unit
def test_llamacpp_validate_run_probe_executes_saved_binary(monkeypatch):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_config_service

    monkeypatch.setattr(
        llamacpp_config_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(executable_path=sys.executable),
    )
    app = _make_app_with_manager(_ManagerWithoutHandler())

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/llamacpp/validate",
            json={"binary_path": sys.executable, "run_probe": True},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["valid"] is True
    assert body["exists"] is True
    assert body["executable"] is True
    assert "Python" in body["version_output"]


@pytest.mark.unit
def test_llamacpp_validate_probe_empty_success_output_is_valid(monkeypatch):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_config_service

    monkeypatch.setattr(
        llamacpp_config_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(executable_path=sys.executable),
    )

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(llamacpp_config_service.subprocess, "run", fake_run)

    result = llamacpp_config_service.validate_binary(sys.executable, run_probe=True)

    assert result["valid"] is True
    assert result["version_output"] == ""
    assert not any("did not return" in warning for warning in result["warnings"])


@pytest.mark.unit
def test_llamacpp_validate_run_probe_executes_active_handler_binary():
    app = _make_app_with_manager(_ManagerWithHandler(Path(sys.executable)))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/llamacpp/validate",
            json={"binary_path": sys.executable, "run_probe": True},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["valid"] is True
    assert body["exists"] is True
    assert body["executable"] is True
    assert "Python" in body["version_output"]


@pytest.mark.unit
def test_llamacpp_validate_reports_missing_binary_without_leaking_path(tmp_path: Path):
    missing = tmp_path / "private" / "missing-llama-server"
    app = _make_app_with_manager(_ManagerWithoutHandler())

    with TestClient(app) as client:
        response = client.post("/api/v1/llamacpp/validate", json={"binary_path": str(missing)})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["valid"] is False
    assert body["exists"] is False
    assert body["executable"] is False
    assert body["resolved_path"] is None
    assert str(tmp_path) not in " ".join(body["warnings"])
    assert "missing-llama-server" in " ".join(body["warnings"])


@pytest.mark.unit
def test_llamacpp_existing_management_api_compatibility_after_admin_facade(monkeypatch):
    app = _make_app_with_manager(_ExistingManagementManager())
    monkeypatch.setattr(
        lp.llamacpp_config_service,
        "get_config_state",
        lambda llm_manager: {
            "saved_config": {"enabled": False, "allowed_paths": []},
            "active_config": {"handler_configured": True},
            "restart_required": False,
            "restart_reasons": [],
            "env_overrides": {},
            "warnings": [],
        },
    )

    with TestClient(app) as client:
        config_response = client.get("/api/v1/llamacpp/config")
        start_response = client.post("/api/v1/llamacpp/start_server", json={"model_filename": "existing.gguf"})
        stop_response = client.post("/api/v1/llamacpp/stop_server", json={})
        status_response = client.get("/api/v1/llamacpp/status")
        models_response = client.get("/api/v1/llamacpp/models")

    assert config_response.status_code == 200, config_response.text
    assert start_response.status_code == 200, start_response.text
    assert start_response.json()["status"] == "started"
    assert stop_response.status_code == 200, stop_response.text
    assert status_response.status_code == 200, status_response.text
    assert models_response.status_code == 200, models_response.text
    assert models_response.json()["available_models"] == ["existing.gguf"]
