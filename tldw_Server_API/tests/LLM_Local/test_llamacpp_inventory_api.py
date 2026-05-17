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
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import ServerError


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


class _FailingStartHandler:
    started = None

    async def start_server_by_path(
        self,
        model_path: Path,
        *,
        model_label: str | None = None,
        server_args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = (model_path, model_label, server_args)
        raise ServerError("llama-server stderr: failed loading /private/sensitive/model.gguf")


class _Manager:
    logger = _Logger()

    def __init__(self) -> None:
        self.llamacpp = _Handler()


class _ManagerWithFailingStart:
    logger = _Logger()

    def __init__(self) -> None:
        self.llamacpp = _FailingStartHandler()


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
    (nested / "projector-Llama-3-f16.gguf").write_text("projector")

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
def test_assets_endpoint_lists_gguf_and_mmproj(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "chat.Q4_K_M.gguf").write_text("base")
    (models_dir / "mmproj-chat-f16.gguf").write_text("projector")
    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir),
    )
    monkeypatch.setattr(lp.llamacpp_config_service, "get_config_state", lambda llm_manager: _config_state(models_dir))
    app = _make_app_with_manager(_Manager())

    with TestClient(app) as client:
        response = client.get("/api/v1/llamacpp/assets")

    assert response.status_code == 200, response.text
    kinds = {asset["kind"] for asset in response.json()["assets"]}
    assert {"gguf", "mmproj"} <= kinds


@pytest.mark.unit
def test_import_folder_persists_allowlisted_folder_and_returns_folder_asset(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service

    models_dir = tmp_path / "models"
    imported = tmp_path / "imported"
    models_dir.mkdir()
    imported.mkdir()
    updates: list[dict[str, dict[str, str]]] = []
    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir, allowed_paths=str(imported), imported_asset_folders=""),
    )
    monkeypatch.setattr(llamacpp_inventory_service.setup_manager, "update_config", updates.append)
    monkeypatch.setattr(llamacpp_inventory_service, "refresh_config_cache", lambda: None)
    app = _make_app_with_manager(_Manager())

    with TestClient(app) as client:
        response = client.post("/api/v1/llamacpp/assets/import-folder", json={"path": str(imported)})

    assert response.status_code == 200, response.text
    assert response.json()["kind"] == "folder"
    assert updates[-1]["LlamaCpp"]["imported_asset_folders"] == str(imported)


@pytest.mark.unit
def test_import_folder_preview_returns_summary_without_persisting(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service

    models_dir = tmp_path / "models"
    imported = tmp_path / "imported"
    models_dir.mkdir()
    imported.mkdir()
    (imported / "chat.Q4_K_M.gguf").write_text("base")
    (imported / "mmproj-chat-f16.gguf").write_text("projector")
    updates: list[dict[str, dict[str, str]]] = []
    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir, allowed_paths=str(imported), imported_asset_folders=""),
    )
    monkeypatch.setattr(llamacpp_inventory_service.setup_manager, "update_config", updates.append)
    monkeypatch.setattr(llamacpp_inventory_service, "refresh_config_cache", lambda: None)
    app = _make_app_with_manager(_Manager())

    with TestClient(app) as client:
        response = client.post("/api/v1/llamacpp/assets/import-folder/preview", json={"path": str(imported)})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["folder"]["kind"] == "folder"
    assert body["asset_counts"] == {"gguf": 1, "mmproj": 1}
    assert {asset["kind"] for asset in body["assets"]} == {"gguf", "mmproj"}
    assert body["will_persist"] is False
    assert updates == []


@pytest.mark.unit
def test_asset_endpoints_offload_blocking_inventory_work_to_threadpool(monkeypatch):
    calls: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []
    config_state = {
        "saved_config": {
            "models_dir": None,
            "allowed_paths": [],
            "registered_model_paths": [],
            "imported_asset_folders": [],
        },
        "active_config": {"handler_configured": True},
        "restart_required": False,
        "restart_reasons": [],
        "env_overrides": {},
        "warnings": [],
    }
    asset_payload = {
        "asset_id": "folder:fake",
        "kind": "folder",
        "identity_basis": "resolved_path",
        "path": "/models",
        "resolved_path": "/models",
        "display_name": "models",
        "source": "imported_folder",
        "size_bytes": None,
        "modified_at": None,
        "metadata": {},
        "capabilities": ["asset_folder"],
        "mmproj_asset_ids": [],
        "base_model_asset_ids": [],
        "warnings": [],
    }

    def fake_get_config_state(llm_manager: Any) -> dict[str, Any]:
        assert isinstance(llm_manager, _Manager)
        return config_state

    def fake_scan_assets(received_config_state: dict[str, Any]) -> dict[str, Any]:
        assert received_config_state is config_state
        return {"assets": [], "warnings": [], "scan_limited": False}

    def fake_register_asset_path(path: Path) -> dict[str, Any]:
        assert path == Path("/models/base.gguf")
        return asset_payload

    def fake_import_asset_folder(path: Path) -> dict[str, Any]:
        assert path == Path("/models")
        return asset_payload

    def fake_preview_import_asset_folder(path: Path) -> dict[str, Any]:
        assert path == Path("/models")
        return {
            "folder": asset_payload,
            "assets": [],
            "asset_counts": {},
            "warnings": [],
            "scan_limited": False,
            "will_persist": False,
        }

    async def fake_run_in_threadpool(func: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append((func, args, kwargs))
        return func(*args, **kwargs)

    monkeypatch.setattr(lp, "run_in_threadpool", fake_run_in_threadpool)
    monkeypatch.setattr(lp.llamacpp_config_service, "get_config_state", fake_get_config_state)
    monkeypatch.setattr(lp.llamacpp_inventory_service, "scan_assets", fake_scan_assets)
    monkeypatch.setattr(lp.llamacpp_inventory_service, "register_asset_path", fake_register_asset_path)
    monkeypatch.setattr(lp.llamacpp_inventory_service, "import_asset_folder", fake_import_asset_folder)
    monkeypatch.setattr(
        lp.llamacpp_inventory_service,
        "preview_import_asset_folder",
        fake_preview_import_asset_folder,
    )
    app = _make_app_with_manager(_Manager())

    with TestClient(app) as client:
        list_response = client.get("/api/v1/llamacpp/assets")
        register_response = client.post("/api/v1/llamacpp/assets/register-path", json={"path": "/models/base.gguf"})
        preview_response = client.post("/api/v1/llamacpp/assets/import-folder/preview", json={"path": "/models"})
        import_response = client.post("/api/v1/llamacpp/assets/import-folder", json={"path": "/models"})

    assert list_response.status_code == 200, list_response.text
    assert register_response.status_code == 200, register_response.text
    assert preview_response.status_code == 200, preview_response.text
    assert import_response.status_code == 200, import_response.text
    assert [call[0] for call in calls] == [
        fake_get_config_state,
        fake_scan_assets,
        fake_register_asset_path,
        fake_preview_import_asset_folder,
        fake_import_asset_folder,
    ]


@pytest.mark.unit
def test_legacy_inventory_excludes_mmproj_assets_after_asset_v2(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "chat.gguf").write_text("base")
    (models_dir / "mmproj-chat.gguf").write_text("projector")
    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir),
    )

    inventory = llamacpp_inventory_service.scan_inventory(limit=500)

    assert [item.basename for item in inventory.models] == ["chat.gguf"]


@pytest.mark.unit
def test_legacy_inventory_excludes_registered_mmproj_and_projector_paths(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service
    from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import ModelNotFoundError

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    base = models_dir / "chat.gguf"
    mmproj = models_dir / "mmproj-chat.gguf"
    projector = models_dir / "projector-chat.gguf"
    base.write_text("base")
    mmproj.write_text("projector")
    projector.write_text("projector")
    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir, registered_model_paths=f"{base}, {mmproj}, {projector}"),
    )

    inventory = llamacpp_inventory_service.scan_inventory(limit=500)

    assert [item.basename for item in inventory.models] == ["chat.gguf"]
    with pytest.raises(ModelNotFoundError):
        llamacpp_inventory_service.resolve_model_id(llamacpp_inventory_service.model_id_for_path(mmproj))
    with pytest.raises(ModelNotFoundError):
        llamacpp_inventory_service.resolve_model_id(llamacpp_inventory_service.model_id_for_path(projector))


@pytest.mark.unit
def test_resolve_model_id_rejects_mmproj_asset_id(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service
    from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import ModelNotFoundError

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    projector = models_dir / "mmproj-chat.gguf"
    projector.write_text("projector")
    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir),
    )

    with pytest.raises(ModelNotFoundError):
        llamacpp_inventory_service.resolve_model_id(llamacpp_inventory_service.asset_id_for_path(projector, "mmproj"))


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
        lambda: _llamacpp_parser(models_dir, allowed_paths=str(external.parent), registered_model_paths=""),
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
def test_register_path_rejects_outside_allowed_paths_without_persisting(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    outside = tmp_path / "outside" / "registered.gguf"
    outside.parent.mkdir()
    outside.write_text("fake model")
    updates: list[dict[str, dict[str, str]]] = []

    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir, registered_model_paths=""),
    )
    monkeypatch.setattr(llamacpp_inventory_service.setup_manager, "update_config", updates.append)

    app = _make_app_with_manager(_Manager())
    with TestClient(app) as client:
        response = client.post("/api/v1/llamacpp/models/register-path", json={"path": str(outside)})

    assert response.status_code == 400, response.text
    assert "outside allowed" in response.json()["detail"].lower()
    assert str(outside) not in response.text
    assert updates == []


@pytest.mark.unit
def test_register_path_rejects_delimiter_path_without_persisting(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model = models_dir / "bad,name.gguf"
    model.write_text("fake model")
    updates: list[dict[str, dict[str, str]]] = []

    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir, registered_model_paths=""),
    )
    monkeypatch.setattr(llamacpp_inventory_service.setup_manager, "update_config", updates.append)

    app = _make_app_with_manager(_Manager())
    with TestClient(app) as client:
        response = client.post("/api/v1/llamacpp/models/register-path", json={"path": str(model)})

    assert response.status_code == 400, response.text
    assert "delimiter" in response.json()["detail"].lower()
    assert str(model) not in response.text
    assert updates == []


@pytest.mark.unit
def test_register_path_preserves_existing_paths_under_lock(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    existing = tmp_path / "external" / "existing.gguf"
    new_model = tmp_path / "external" / "new.gguf"
    existing.parent.mkdir()
    existing.write_text("fake model")
    new_model.write_text("fake model")
    events: list[str] = []
    updates: list[dict[str, dict[str, str]]] = []

    class FakeLock:
        def __enter__(self) -> "FakeLock":
            events.append("lock_enter")
            return self

        def __exit__(self, *exc: Any) -> None:
            events.append("lock_exit")

    def fake_load_config() -> ConfigParser:
        events.append("read_config")
        return _llamacpp_parser(
            models_dir,
            allowed_paths=str(existing.parent),
            registered_model_paths=str(existing),
        )

    def fake_update_config(payload: dict[str, dict[str, str]]) -> None:
        events.append("write_config")
        updates.append(payload)

    monkeypatch.setattr(llamacpp_inventory_service, "llamacpp_config_write_lock", lambda: FakeLock(), raising=False)
    monkeypatch.setattr(llamacpp_inventory_service, "load_comprehensive_config", fake_load_config)
    monkeypatch.setattr(llamacpp_inventory_service.setup_manager, "update_config", fake_update_config)
    monkeypatch.setattr(llamacpp_inventory_service, "refresh_config_cache", lambda: events.append("refresh"))

    item = llamacpp_inventory_service.register_model_path(new_model)

    persisted = updates[0]["LlamaCpp"]["registered_model_paths"]
    assert item.basename == "new.gguf"
    assert str(existing.resolve()) in persisted
    assert str(new_model.resolve()) in persisted
    assert events.index("lock_enter") < events.index("read_config") < events.index("write_config") < events.index("lock_exit")
    assert events.index("write_config") < events.index("refresh") < events.index("lock_exit")


@pytest.mark.unit
def test_register_path_rejects_unresolvable_path_without_persisting(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service
    from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import ServerError

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    bad_path = tmp_path / "loop.gguf"
    updates: list[dict[str, dict[str, str]]] = []
    original_resolve = Path.resolve

    def fake_resolve(self: Path, *args: Any, **kwargs: Any) -> Path:
        if self == bad_path:
            raise RuntimeError("symlink loop under /private/sensitive")
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir),
    )
    monkeypatch.setattr(llamacpp_inventory_service.setup_manager, "update_config", updates.append)
    monkeypatch.setattr(Path, "resolve", fake_resolve)

    with pytest.raises(ServerError) as exc_info:
        llamacpp_inventory_service.register_model_path(bad_path)

    assert "could not be resolved" in str(exc_info.value)
    assert "sensitive" not in str(exc_info.value)
    assert updates == []


@pytest.mark.unit
def test_inventory_skips_unresolvable_scan_entry_with_warning(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    bad = models_dir / "bad.gguf"
    good = models_dir / "good.gguf"
    bad.write_text("bad")
    good.write_text("good")
    original_resolve = Path.resolve

    def fake_resolve(self: Path, *args: Any, **kwargs: Any) -> Path:
        if self == bad:
            raise RuntimeError("symlink loop under /private/sensitive")
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir),
    )
    monkeypatch.setattr(Path, "resolve", fake_resolve)

    inventory = llamacpp_inventory_service.scan_inventory(limit=500)

    assert [item.basename for item in inventory.models] == ["good.gguf"]
    assert any("could not inspect" in warning.lower() for warning in inventory.warnings)
    assert not any("sensitive" in warning for warning in inventory.warnings)


@pytest.mark.unit
def test_registered_paths_are_not_hidden_by_models_dir_scan_limit(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    for index in range(5):
        (models_dir / f"model-{index}.gguf").write_text("fake model")
    registered_dir = tmp_path / "registered"
    registered_dir.mkdir()
    registered_model = registered_dir / "explicit.gguf"
    registered_model.write_text("fake model")

    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(
            models_dir,
            allowed_paths=str(registered_dir),
            registered_model_paths=str(registered_model),
        ),
    )

    inventory = llamacpp_inventory_service.scan_inventory(limit=1)
    resolved = llamacpp_inventory_service.resolve_model_id(llamacpp_inventory_service.model_id_for_path(registered_model))

    assert [item.basename for item in inventory.models] == ["explicit.gguf"]
    assert inventory.scan_limited is True
    assert resolved == registered_model.resolve()


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


@pytest.mark.unit
def test_start_by_model_sanitizes_handler_startup_server_error(monkeypatch, tmp_path: Path):
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
    app = _make_app_with_manager(_ManagerWithFailingStart())

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/llamacpp/start-by-model",
            json={"model_id": model_id, "server_args": {"port": 8123}},
        )

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "Failed to start llama.cpp server for the selected model."
    assert "private" not in response.text
    assert "stderr" not in response.text
