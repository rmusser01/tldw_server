from __future__ import annotations

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


class _Process:
    pid = 1234
    returncode = None


class _Config:
    def __init__(self, log_output_file: str | None = None) -> None:
        self.default_host = "0.0.0.0"
        self.default_port = 8080
        self.log_output_file = log_output_file


class _Handler:
    def __init__(
        self,
        *,
        status: str = "running",
        host: str = "0.0.0.0",
        port: int = 8080,
        log_output_file: str | None = None,
        status_log_file: str | None = None,
        active_log: bool = False,
    ) -> None:
        self.config = _Config(log_output_file)
        self._status = status
        self._active_server_process = _Process() if status == "running" else None
        self._active_server_host = host
        self._active_server_port = port
        self._active_server_model = "toy.gguf"
        self._status_log_file = status_log_file
        self._active_server_log_handle = object() if active_log else None

    async def get_server_status(self) -> dict[str, Any]:
        return {
            "status": self._status,
            "host": self._active_server_host,
            "port": self._active_server_port,
            "model": self._active_server_model,
            "pid": 1234 if self._status == "running" else None,
            "log_file": self._status_log_file,
        }

    async def list_models(self) -> list[str]:
        return ["toy.gguf"]

    def get_metrics(self) -> dict[str, int]:
        return {"starts": 1}


class _Manager:
    logger = _Logger()

    def __init__(self, handler: _Handler) -> None:
        self.llamacpp = handler

    async def start_server(self, *, backend: str, model_name: str | None = None, server_args: dict | None = None):
        return {"status": "started", "backend": backend, "model": model_name, "server_args": server_args or {}}

    async def stop_server(self, *, backend: str, pid: int | None = None, port: int | None = None) -> str:
        _ = (pid, port)
        return f"{backend} stopped"

    async def get_server_status(self, *, backend: str) -> dict[str, str]:
        return {"status": "running", "backend": backend, "model": "toy.gguf"}


def _make_app(manager: _Manager) -> FastAPI:
    app = FastAPI()
    app.include_router(lp.router, prefix="/api/v1")
    app.state.llm_manager = manager

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
        principal = _admin_principal()
        ip = request.client.host if getattr(request, "client", None) else None
        ua = request.headers.get("User-Agent") if getattr(request, "headers", None) else None
        request_id = request.headers.get("X-Request-ID") if getattr(request, "headers", None) else None
        request.state.auth = AuthContext(principal=principal, ip=ip, user_agent=ua, request_id=request_id)
        return principal

    async def _fake_check_rate_limit() -> None:
        return

    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal
    app.dependency_overrides[auth_deps.check_rate_limit] = _fake_check_rate_limit
    app.dependency_overrides[lp.check_rate_limit] = _fake_check_rate_limit
    return app


@pytest.mark.unit
def test_use_in_chat_writes_only_llama_endpoint_and_refreshes(monkeypatch):
    calls: list[tuple[str, Any]] = []

    monkeypatch.setattr(
        lp.llamacpp_provider_service.setup_manager,
        "update_config",
        lambda updates: calls.append(("update_config", updates)),
    )
    monkeypatch.setattr(
        lp.llamacpp_provider_service,
        "refresh_config_cache",
        lambda: calls.append(("refresh_config_cache", None)),
    )

    class FakeLock:
        def __enter__(self) -> "FakeLock":
            calls.append(("lock_enter", None))
            return self

        def __exit__(self, *exc: Any) -> None:
            calls.append(("lock_exit", None))

    monkeypatch.setattr(lp.llamacpp_provider_service, "llamacpp_config_write_lock", lambda: FakeLock())
    app = _make_app(_Manager(_Handler(host="0.0.0.0", port=8181)))

    with TestClient(app) as client:
        response = client.post("/api/v1/llamacpp/use-in-chat")

    assert response.status_code == 200, response.text
    assert calls == [
        ("lock_enter", None),
        ("update_config", {"Local-API": {"llama_api_IP": "http://127.0.0.1:8181"}}),
        ("refresh_config_cache", None),
        ("lock_exit", None),
    ]
    body = response.json()
    assert body["provider"] == "llama"
    assert body["endpoint"] == "http://127.0.0.1:8181"
    assert body["updated"] is True
    assert body["effective"] is True
    assert body["warnings"] == []


@pytest.mark.unit
def test_use_in_chat_requires_running_managed_server():
    app = _make_app(_Manager(_Handler(status="stopped")))

    with TestClient(app) as client:
        response = client.post("/api/v1/llamacpp/use-in-chat")

    assert response.status_code == 409
    assert "running" in response.json()["detail"].lower()


@pytest.mark.unit
def test_use_in_chat_reports_ineffective_when_provider_endpoint_env_override_exists(monkeypatch):
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        lp.llamacpp_provider_service,
        "get_provider_endpoint_env_override",
        lambda: "TLDW_TEST_LLAMA_ENDPOINT",
    )
    monkeypatch.setattr(
        lp.llamacpp_provider_service.setup_manager,
        "update_config",
        lambda updates: calls.append(updates),
    )
    monkeypatch.setattr(lp.llamacpp_provider_service, "refresh_config_cache", lambda: None)
    app = _make_app(_Manager(_Handler(host="127.0.0.1", port=8181)))

    with TestClient(app) as client:
        response = client.post("/api/v1/llamacpp/use-in-chat")

    assert response.status_code == 200, response.text
    assert calls == [{"Local-API": {"llama_api_IP": "http://127.0.0.1:8181"}}]
    body = response.json()
    assert body["updated"] is True
    assert body["effective"] is False
    assert "TLDW_TEST_LLAMA_ENDPOINT" in body["warnings"][0]


@pytest.mark.unit
def test_log_tail_reads_only_configured_log_file_and_redacts(tmp_path: Path):
    configured_log = tmp_path / "configured.log"
    arbitrary_log = tmp_path / "arbitrary.log"
    configured_log.write_text(
        "\n".join(
            [
                "first line",
                "api_key=sk-secret token=tok-secret hf_token=hf-secret",
                "last line",
            ]
        ),
        encoding="utf-8",
    )
    arbitrary_log.write_text("do-not-read", encoding="utf-8")
    app = _make_app(
        _Manager(
            _Handler(
                log_output_file=str(configured_log),
                status_log_file=str(configured_log),
                active_log=True,
            )
        )
    )

    with TestClient(app) as client:
        response = client.get(f"/api/v1/llamacpp/logs/tail?lines=2&path={arbitrary_log}")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["lines"] == ["api_key=[REDACTED] token=[REDACTED] hf_token=[REDACTED]", "last line"]
    assert body["truncated"] is True
    assert "do-not-read" not in response.text


@pytest.mark.unit
def test_log_tail_configured_path_without_active_log_evidence_returns_warning(tmp_path: Path):
    configured_log = tmp_path / "configured.log"
    configured_log.write_text("must-not-read", encoding="utf-8")
    app = _make_app(_Manager(_Handler(log_output_file=str(configured_log))))

    with TestClient(app) as client:
        response = client.get("/api/v1/llamacpp/logs/tail?lines=10")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["lines"] == []
    assert body["truncated"] is False
    assert body["warnings"]
    assert "must-not-read" not in response.text


@pytest.mark.unit
def test_log_tail_status_log_file_must_match_configured_canonical_path(tmp_path: Path):
    configured_log = tmp_path / "configured.log"
    other_log = tmp_path / "other.log"
    status_symlink = tmp_path / "status.log"
    configured_log.write_text("configured-secret", encoding="utf-8")
    other_log.write_text("other-secret", encoding="utf-8")
    status_symlink.symlink_to(other_log)
    app = _make_app(
        _Manager(
            _Handler(
                log_output_file=str(configured_log),
                status_log_file=str(status_symlink),
                active_log=True,
            )
        )
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/llamacpp/logs/tail?lines=10")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["lines"] == []
    assert body["truncated"] is False
    assert body["warnings"]
    assert "configured-secret" not in response.text
    assert "other-secret" not in response.text


@pytest.mark.unit
def test_log_tail_redacts_api_exposed_secret_syntaxes(tmp_path: Path):
    configured_log = tmp_path / "configured.log"
    configured_log.write_text(
        "\n".join(
            [
                "Authorization: Bearer sk-live-abcdefghijklmnopqrstuvwxyz123456",
                "api_key: sk-colon-abcdefghijklmnopqrstuvwxyz123456",
                '{"api_key": "sk-json-abcdefghijklmnopqrstuvwxyz123456"}',
                "--api-key sk-cli-abcdefghijklmnopqrstuvwxyz123456 --hf-token hf_cli_secret --token plain_cli_secret",
                "--api-key=sk-cli-equals-abcdefghijklmnopqrstuvwxyz123456",
                'api_key="quoted-secret-value"',
                "'token' = 'quoted-token-secret'",
                "token = spaced-secret-value",
                "standalone sk-standalone-abcdefghijklmnopqrstuvwxyz123456",
            ]
        ),
        encoding="utf-8",
    )
    app = _make_app(
        _Manager(
            _Handler(
                log_output_file=str(configured_log),
                status_log_file=str(configured_log),
                active_log=True,
            )
        )
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/llamacpp/logs/tail?lines=20")

    assert response.status_code == 200, response.text
    text = response.text
    assert "Bearer sk-live" not in text
    assert "sk-colon" not in text
    assert "sk-json" not in text
    assert "sk-cli" not in text
    assert "sk-cli-equals" not in text
    assert "hf_cli_secret" not in text
    assert "plain_cli_secret" not in text
    assert "quoted-secret-value" not in text
    assert "quoted-token-secret" not in text
    assert "spaced-secret-value" not in text
    assert "sk-standalone" not in text
    assert text.count("[REDACTED") >= 11


@pytest.mark.unit
def test_log_tail_missing_or_unconfigured_log_returns_empty_warning():
    app = _make_app(_Manager(_Handler(log_output_file=None)))

    with TestClient(app) as client:
        response = client.get("/api/v1/llamacpp/logs/tail?lines=200")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["lines"] == []
    assert body["truncated"] is False
    assert body["warnings"]


@pytest.mark.unit
def test_hardware_snapshot_reports_cpu_ram_and_optional_gpu_warning(monkeypatch):
    class _VirtualMemory:
        total = 16
        available = 8

    class _Psutil:
        @staticmethod
        def virtual_memory() -> _VirtualMemory:
            return _VirtualMemory()

        @staticmethod
        def cpu_count(logical: bool = True) -> int:
            _ = logical
            return 12

    monkeypatch.setattr(lp.llamacpp_hardware_service, "psutil", _Psutil)
    monkeypatch.setattr(lp.llamacpp_hardware_service, "load_nvml_snapshot", lambda: ([], ["nvml_unavailable"]))
    app = _make_app(_Manager(_Handler()))

    with TestClient(app) as client:
        response = client.get("/api/v1/llamacpp/hardware")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ram_total_bytes"] == 16
    assert body["ram_available_bytes"] == 8
    assert body["cpu_count"] == 12
    assert body["gpus"] == []
    assert body["warnings"] == ["nvml_unavailable"]
