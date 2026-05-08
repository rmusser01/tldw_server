from __future__ import annotations

import builtins
import os
from pathlib import Path
import subprocess
import sys

import pytest
from fastapi import APIRouter, FastAPI

from tldw_Server_API.tests.helpers.app_main_state import (
    clear_app_main,
    import_app_main,
    restore_app_main,
    snapshot_app_main,
)


@pytest.mark.unit
def test_router_contract_includes_key_paths() -> None:
    from tldw_Server_API.app.main import app

    paths = {route.path for route in app.routes}
    expected_paths = {
        "/health",
        "/openapi.json",
        "/api/v1/chat/completions",
        "/api/v1/rag/search",
    }
    missing_paths = sorted(expected_paths - paths)
    assert not missing_paths, f"Missing expected paths: {missing_paths}"
    assert any(path.startswith("/api/v1/media/process") for path in paths), (
        "Expected at least one media process route under /api/v1/media/process*"
    )


@pytest.mark.integration
def test_openapi_contains_core_tags(client_user_only) -> None:
    response = client_user_only.get("/openapi.json")
    assert response.status_code == 200

    payload = response.json()
    tags = {tag["name"] for tag in payload.get("tags", [])}
    expected_tags = {"chat", "audio", "media", "rag-unified"}
    missing_tags = sorted(expected_tags - tags)
    assert not missing_tags, f"Missing expected tags: {missing_tags}"


@pytest.mark.unit
def test_router_registry_idempotent_registration() -> None:
    from tldw_Server_API.app.api.v1.router_registry import include_router_idempotent

    app = FastAPI()
    test_router = APIRouter()

    @test_router.get("/health")
    def _health() -> dict[str, str]:
        return {"status": "ok"}

    include_router_idempotent(app, test_router, prefix="/api/v1", tags=["health"])
    include_router_idempotent(app, test_router, prefix="/api/v1", tags=["health"])

    route_signatures = {
        (route.path, tuple(sorted(getattr(route, "methods", set()))))
        for route in app.routes
        if getattr(route, "path", "").startswith("/api/v1")
    }
    assert route_signatures == {("/api/v1/health", ("GET",))}


@pytest.mark.integration
def test_minimal_app_uses_router_registry(client_user_only) -> None:
    assert hasattr(client_user_only.app.state, "_tldw_router_registry")


@pytest.mark.unit
def test_minimal_app_import_does_not_probe_setup_router_directly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("ULTRA_MINIMAL_APP", "0")

    original_import = builtins.__import__
    existing_main = snapshot_app_main()
    clear_app_main()

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        setup_module = "tldw_Server_API.app.api.v1.endpoints.setup"
        endpoints_module = "tldw_Server_API.app.api.v1.endpoints"
        requested_attrs = {fromlist} if isinstance(fromlist, str) else set(fromlist or ())
        if (
            name == setup_module
            or name.startswith(f"{setup_module}.")
            or (name == endpoints_module and "setup" in requested_attrs)
        ):
            raise AssertionError("setup router should be registered through router groups")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)

    try:
        imported_main = import_app_main()
        assert imported_main.app is not None
        assert any(route.path == "/health" for route in imported_main.app.routes)
    finally:
        restore_app_main(existing_main)


@pytest.mark.integration
def test_app_import_outside_explicit_pytest_runtime_has_no_duplicate_routes(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env.update(
        {
            "AUTH_MODE": "single_user",
            "SINGLE_USER_API_KEY": "test-api-key-12345",
            "TEST_MODE": "true",
            "MINIMAL_TEST_APP": "0",
            "ULTRA_MINIMAL_APP": "0",
            "USER_DB_BASE_DIR": str(tmp_path / "user_databases"),
        }
    )
    env.pop("PYTEST_CURRENT_TEST", None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import tldw_Server_API.app.main as main; "
                "main._fail_on_duplicate_route_method_pairs("
                "main.app, context='subprocess-import'"
                "); "
                "vlm_routes = ["
                "route for route in main.app.routes "
                "if getattr(route, 'path', None) == '/api/v1/vlm/backends'"
                "]; "
                "assert len(vlm_routes) == 1, "
                "f'expected one VLM route, found {len(vlm_routes)}'; "
                "print('NO_DUPLICATES')"
            ),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        "Expected app import to succeed outside explicit pytest runtime without duplicate routes.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "NO_DUPLICATES" in result.stdout
