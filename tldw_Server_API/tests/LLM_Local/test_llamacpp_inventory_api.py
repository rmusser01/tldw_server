from __future__ import annotations

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
    def error(self, *args: Any, **kwargs: Any) -> None:
        return


class _Handler:
    def __init__(self) -> None:
        self.started: dict[str, Any] | None = None

    async def start_server_by_path(
        self,
        model_path: Path,
        *,
        model_label: str | None = None,
        server_args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.started = {
            "model_path": model_path,
            "model_label": model_label,
            "server_args": server_args or {},
        }
        return {"status": "started", "model": model_label, "path": str(model_path)}


class _Manager:
    logger = _Logger()

    def __init__(self) -> None:
        self.llamacpp = _Handler()


def _make_app_with_manager(manager: _Manager) -> FastAPI:
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


def _llamacpp_parser(models_dir: Path, **overrides: str) -> ConfigParser:
    parser = ConfigParser()
    parser.add_section("LlamaCpp")
    values = {
        "enabled": "true",
        "models_dir": str(models_dir),
        "allowed_paths": "",
        "registered_model_paths": "",
    }
    values.update(overrides)
    parser["LlamaCpp"] = values
    return parser


def _config_state(models_dir: Path, **overrides: str) -> dict[str, Any]:
    parser = _llamacpp_parser(models_dir, **overrides)
    section = parser["LlamaCpp"]
    return {
        "saved_config": {
            "enabled": True,
            "models_dir": section.get("models_dir"),
            "allowed_paths": [part.strip() for part in section.get("allowed_paths", "").split(",") if part.strip()],
            "registered_model_paths": [
                part.strip() for part in section.get("registered_model_paths", "").split(",") if part.strip()
            ],
        },
        "active_config": {"handler_configured": True},
        "restart_required": False,
        "restart_reasons": [],
        "env_overrides": {},
        "warnings": [],
    }


@pytest.mark.unit
def test_inventory_recursively_scans_gguf_and_skips_mmproj(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service

    models_dir = tmp_path / "models"
    nested = models_dir / "nested"
    nested.mkdir(parents=True)
    model = nested / "Llama-3-8B-Q4_K_M.gguf"
    model.write_text("fake model")
    (nested / "mmproj-model-f16.gguf").write_text("projector")

    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir),
    )
    monkeypatch.setattr(lp.llamacpp_config_service, "get_config_state", lambda llm_manager: _config_state(models_dir))
    app = _make_app_with_manager(_Manager())

    with TestClient(app) as client:
        response = client.get("/api/v1/llamacpp/inventory")

    assert response.status_code == 200, response.text
    body = response.json()
    assert [item["basename"] for item in body["models"]] == ["Llama-3-8B-Q4_K_M.gguf"]
    item = body["models"][0]
    assert item["model_id"].startswith("gguf:")
    assert item["source"] == "models_dir"
    assert item["metadata"]["quantization"] == "Q4_K_M"
    assert item["metadata"]["parameter_hint"] == "8B"
    assert body["scan_limited"] is False


@pytest.mark.unit
def test_inventory_reports_registered_path_warnings_without_failing(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_model = outside_dir / "external.Q5_K_M.gguf"
    outside_model.write_text("fake model")
    missing_model = outside_dir / "missing.gguf"
    text_file = outside_dir / "notes.txt"
    text_file.write_text("not gguf")

    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(
            models_dir,
            registered_model_paths=f"{outside_model}, {missing_model}, {text_file}",
        ),
    )
    monkeypatch.setattr(
        lp.llamacpp_config_service,
        "get_config_state",
        lambda llm_manager: _config_state(
            models_dir,
            registered_model_paths=f"{outside_model}, {missing_model}, {text_file}",
        ),
    )

    app = _make_app_with_manager(_Manager())
    with TestClient(app) as client:
        response = client.get("/api/v1/llamacpp/inventory")

    assert response.status_code == 200, response.text
    body = response.json()
    by_basename = {item["basename"]: item for item in body["models"]}
    assert set(by_basename) == {"external.Q5_K_M.gguf", "missing.gguf", "notes.txt"}
    assert any("outside allowed" in warning.lower() for warning in by_basename["external.Q5_K_M.gguf"]["warnings"])
    assert any("missing" in warning.lower() for warning in by_basename["missing.gguf"]["warnings"])
    assert any("gguf" in warning.lower() for warning in by_basename["notes.txt"]["warnings"])


@pytest.mark.unit
def test_inventory_model_ids_are_stable_for_canonical_path(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model = models_dir / "stable.gguf"
    model.write_text("fake model")
    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir),
    )

    first = llamacpp_inventory_service.scan_inventory(limit=500)
    second = llamacpp_inventory_service.scan_inventory(limit=500)

    assert first.models[0].model_id == second.models[0].model_id
    assert first.models[0].model_id == llamacpp_inventory_service.model_id_for_path(model)


@pytest.mark.unit
def test_register_path_persists_and_returns_inventory_item(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    external = tmp_path / "external" / "registered.gguf"
    external.parent.mkdir()
    external.write_text("fake model")
    updates: list[dict[str, dict[str, str]]] = []
    refreshed = {"called": False}

    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir, registered_model_paths=""),
    )
    monkeypatch.setattr(llamacpp_inventory_service.setup_manager, "update_config", updates.append)
    monkeypatch.setattr(llamacpp_inventory_service, "refresh_config_cache", lambda: refreshed.__setitem__("called", True))

    app = _make_app_with_manager(_Manager())
    with TestClient(app) as client:
        response = client.post("/api/v1/llamacpp/models/register-path", json={"path": str(external)})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["basename"] == "registered.gguf"
    assert body["source"] == "registered_path"
    assert updates == [{"LlamaCpp": {"registered_model_paths": str(external.resolve())}}]
    assert refreshed["called"] is True


@pytest.mark.unit
def test_start_by_model_resolves_model_id_and_uses_handler_path_start(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model = models_dir / "launch.gguf"
    model.write_text("fake model")
    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir),
    )
    model_id = llamacpp_inventory_service.model_id_for_path(model)
    manager = _Manager()
    app = _make_app_with_manager(manager)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/llamacpp/start-by-model",
            json={"model_id": model_id, "server_args": {"port": 8123}},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "started"
    assert body["backend"] == "llamacpp"
    assert body["model_id"] == model_id
    assert manager.llamacpp.started == {
        "model_path": model.resolve(),
        "model_label": "launch.gguf",
        "server_args": {"port": 8123},
    }


@pytest.mark.unit
def test_start_by_model_rejects_outside_registered_path_before_fake_handler(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_model = outside_dir / "external.gguf"
    outside_model.write_text("fake model")
    registered_value = str(outside_model)

    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir, registered_model_paths=registered_value),
    )
    monkeypatch.setattr(
        lp.llamacpp_config_service,
        "get_config_state",
        lambda llm_manager: _config_state(models_dir, registered_model_paths=registered_value),
    )
    model_id = llamacpp_inventory_service.model_id_for_path(outside_model)
    manager = _Manager()
    app = _make_app_with_manager(manager)

    with TestClient(app) as client:
        inventory_response = client.get("/api/v1/llamacpp/inventory")
        start_response = client.post(
            "/api/v1/llamacpp/start-by-model",
            json={"model_id": model_id, "server_args": {"port": 8123}},
        )

    assert inventory_response.status_code == 200, inventory_response.text
    item = inventory_response.json()["models"][0]
    assert item["basename"] == "external.gguf"
    assert any("outside allowed" in warning.lower() for warning in item["warnings"])
    assert start_response.status_code == 400, start_response.text
    assert "outside allowed" in start_response.json()["detail"].lower()
    assert str(outside_model) not in start_response.text
    assert manager.llamacpp.started is None
