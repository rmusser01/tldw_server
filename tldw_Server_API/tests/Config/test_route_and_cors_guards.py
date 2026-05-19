import asyncio
import builtins
import importlib
import os
from pathlib import Path
import re

from fastapi import FastAPI
import pytest
from starlette.requests import Request

# Keep this module importable even when local/dev env sets ALLOWED_ORIGINS='*'.
os.environ["ALLOWED_ORIGINS"] = "http://localhost:3000"

from tldw_Server_API.app.core import config as config_mod

importlib.reload(config_mod)

from tldw_Server_API.app import main as app_main

app_main = importlib.reload(app_main)


def test_main_app_route_guard_passes_for_current_routes() -> None:
    app_main._fail_on_duplicate_route_method_pairs(app_main.app, context="unit-test")


def test_config_loads_dotenv_before_module_level_allowed_origins_read() -> None:
    source = Path(config_mod.__file__).read_text(encoding="utf-8")
    module_level_load_idx = source.find("\n_load_env_files_early()\n")
    allowed_origins_read_idx = source.find('_ENV_ALLOWED = os.getenv("ALLOWED_ORIGINS")')

    if module_level_load_idx == -1:
        pytest.fail("Expected module-level _load_env_files_early() call in config.py")
    if allowed_origins_read_idx == -1:
        pytest.fail("Expected ALLOWED_ORIGINS env read in config.py")
    if module_level_load_idx >= allowed_origins_read_idx:
        pytest.fail(
            "Expected .env bootstrap before module-level ALLOWED_ORIGINS read in config.py"
        )


def test_duplicate_route_guard_raises_for_duplicate_path_method_pairs() -> None:
    app = FastAPI()

    @app.get("/dup")
    async def dup_one() -> dict[str, str]:
        return {"ok": "1"}

    @app.get("/dup")
    async def dup_two() -> dict[str, str]:
        return {"ok": "2"}

    with pytest.raises(RuntimeError, match="Duplicate route registrations detected"):
        app_main._fail_on_duplicate_route_method_pairs(app, context="unit-test")


def test_duplicate_route_guard_allows_same_path_different_methods() -> None:
    app = FastAPI()

    @app.get("/item")
    async def get_item() -> dict[str, str]:
        return {"method": "get"}

    @app.post("/item")
    async def post_item() -> dict[str, str]:
        return {"method": "post"}

    app_main._fail_on_duplicate_route_method_pairs(app, context="unit-test")


def test_resolve_cors_origins_rejects_empty_values() -> None:
    with pytest.raises(RuntimeError, match="ALLOWED_ORIGINS is empty"):
        app_main._resolve_cors_origins_or_raise([])

    with pytest.raises(RuntimeError, match="ALLOWED_ORIGINS is empty"):
        app_main._resolve_cors_origins_or_raise(["", "   "])


def test_resolve_cors_origins_normalizes_and_accepts_non_empty_values() -> None:
    assert app_main._resolve_cors_origins_or_raise([" http://localhost:3000 ", "https://example.com"]) == [
        "http://localhost:3000",
        "https://example.com",
    ]


def test_validate_cors_configuration_rejects_wildcard_with_credentials() -> None:
    with pytest.raises(RuntimeError, match="cannot include '\\*'"):
        app_main._validate_cors_configuration_or_raise(["*", "http://localhost:3000"], allow_credentials=True)


def test_validate_cors_configuration_allows_wildcard_without_credentials() -> None:
    app_main._validate_cors_configuration_or_raise(["*"], allow_credentials=False)


def test_default_allowed_origins_include_ipv6_loopback_dev_hosts() -> None:
    origins = config_mod.get_default_allowed_origins()

    assert "http://[::1]" in origins
    assert "http://[::1]:3000" in origins
    assert "http://[::1]:3001" in origins
    assert "http://[::1]:8000" in origins


def test_compute_dev_cors_origin_regex_allows_private_lan_origins_in_non_production() -> None:
    pattern = app_main._compute_dev_cors_origin_regex(
        ["http://localhost:3000"],
        enforce_explicit_origins=False,
    )

    if not pattern:
        pytest.fail("Expected development private-LAN regex when explicit origins are not enforced")

    should_match = (
        "http://192.168.5.184:3000",
        "https://10.0.0.42:5173",
        "http://172.16.0.8:8080",
        "http://127.0.0.1:3000",
        "https://localhost:3000",
    )
    for candidate in should_match:
        if re.match(pattern, candidate) is None:
            pytest.fail(f"Expected origin to match development private-LAN regex: {candidate}")

    if re.match(pattern, "https://example.com") is not None:
        pytest.fail("Did not expect public internet origin to match development private-LAN regex")


def test_compute_dev_cors_origin_regex_disabled_for_wildcard_or_production() -> None:
    if app_main._compute_dev_cors_origin_regex(
        ["*"],
        enforce_explicit_origins=False,
    ) is not None:
        pytest.fail("Expected no dev private-LAN regex when wildcard origins are configured")

    if app_main._compute_dev_cors_origin_regex(
        ["http://localhost:3000"],
        enforce_explicit_origins=True,
    ) is not None:
        pytest.fail("Expected no dev private-LAN regex when explicit origins are enforced")


def test_should_allow_cors_credentials_env_override_false() -> None:
    prior_allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS")
    try:
        os.environ["CORS_ALLOW_CREDENTIALS"] = "false"
        reloaded_config = importlib.reload(config_mod)
        assert reloaded_config.should_allow_cors_credentials() is False
    finally:
        _restore_env("CORS_ALLOW_CREDENTIALS", prior_allow_credentials)
        importlib.reload(config_mod)


def test_should_allow_cors_credentials_env_override_true() -> None:
    prior_allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS")
    try:
        os.environ["CORS_ALLOW_CREDENTIALS"] = "true"
        reloaded_config = importlib.reload(config_mod)
        assert reloaded_config.should_allow_cors_credentials() is True
    finally:
        _restore_env("CORS_ALLOW_CREDENTIALS", prior_allow_credentials)
        importlib.reload(config_mod)


def test_get_cors_runtime_diagnostics_reports_env_sources() -> None:
    prior_disable_cors = os.getenv("DISABLE_CORS")
    prior_allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS")
    prior_allowed_origins = os.getenv("ALLOWED_ORIGINS")
    try:
        os.environ["DISABLE_CORS"] = "false"
        os.environ["CORS_ALLOW_CREDENTIALS"] = "true"
        os.environ["ALLOWED_ORIGINS"] = "http://127.0.0.1:8080,http://localhost:3000"
        reloaded_config = importlib.reload(config_mod)

        diagnostics = reloaded_config.get_cors_runtime_diagnostics()
        assert diagnostics["disable_cors"] is False
        assert diagnostics["disable_cors_source"] == "env"
        assert diagnostics["allow_credentials"] is True
        assert diagnostics["allow_credentials_source"] == "env"
        assert diagnostics["allowed_origins_source"] == "env"
        assert diagnostics["allowed_origins_count"] == 2
        assert diagnostics["allowed_origins"] == ["http://127.0.0.1:8080", "http://localhost:3000"]
        assert diagnostics["config_path"]
    finally:
        _restore_env("DISABLE_CORS", prior_disable_cors)
        _restore_env("CORS_ALLOW_CREDENTIALS", prior_allow_credentials)
        _restore_env("ALLOWED_ORIGINS", prior_allowed_origins)
        importlib.reload(config_mod)


def test_get_cors_runtime_diagnostics_marks_empty_origins_env_as_default() -> None:
    prior_allowed_origins = os.getenv("ALLOWED_ORIGINS")
    try:
        os.environ["ALLOWED_ORIGINS"] = ""
        reloaded_config = importlib.reload(config_mod)
        diagnostics = reloaded_config.get_cors_runtime_diagnostics()
        assert diagnostics["allowed_origins_source"] == "default(empty-env)"
        assert diagnostics["allowed_origins_count"] >= 1
    finally:
        _restore_env("ALLOWED_ORIGINS", prior_allowed_origins)
        importlib.reload(config_mod)


def test_get_cors_runtime_diagnostics_uses_local_fallback_for_explicit_empty_list_outside_production() -> None:
    prior_allowed_origins = os.getenv("ALLOWED_ORIGINS")
    prior_env = os.getenv("ENV")
    try:
        os.environ["ENV"] = "development"
        os.environ["ALLOWED_ORIGINS"] = "[]"
        reloaded_config = importlib.reload(config_mod)
        diagnostics = reloaded_config.get_cors_runtime_diagnostics()
        assert diagnostics["allowed_origins_source"] == "env(local-fallback)"
        assert diagnostics["allowed_origins_fallback"] is True
        assert "http://localhost:3000" in diagnostics["allowed_origins"]
    finally:
        _restore_env("ENV", prior_env)
        _restore_env("ALLOWED_ORIGINS", prior_allowed_origins)
        importlib.reload(config_mod)


def test_compute_openapi_cors_allow_origin_echoes_allowed_explicit_origin() -> None:
    allow_origin = app_main._compute_openapi_cors_allow_origin(
        "http://localhost:3000",
        allow_all_origins=False,
        allow_credentials=True,
        allowed_openapi_origins={"http://localhost:3000"},
    )
    assert allow_origin == "http://localhost:3000"


def test_global_unhandled_exception_handler_keeps_cors_for_allowed_origin() -> None:
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/boom",
        "raw_path": b"/boom",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"origin", b"http://localhost:3000")],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "app": app_main.app,
    }
    request = Request(scope)

    response = asyncio.run(
        app_main._global_unhandled_exception_handler(request, RuntimeError("boom"))
    )

    assert response.status_code == 500
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_run_startup_config_validation_logs_warning_on_import_error(monkeypatch) -> None:
    warnings: list[str] = []

    class _StubLogger:
        def warning(self, message: str) -> None:
            warnings.append(message)

    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001, D401
        if name == "tldw_Server_API.app.core.config":
            raise ModuleNotFoundError("config import failed for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(app_main, "logger", _StubLogger(), raising=True)
    monkeypatch.setattr(builtins, "__import__", _fake_import)

    app_main._run_startup_config_validation()

    assert warnings == ["Config validation could not run: config import failed for test"]


def test_compute_openapi_cors_allow_origin_rejects_disallowed_origin() -> None:
    allow_origin = app_main._compute_openapi_cors_allow_origin(
        "http://evil.local",
        allow_all_origins=False,
        allow_credentials=True,
        allowed_openapi_origins={"http://localhost:3000"},
    )
    assert allow_origin is None


def test_compute_openapi_cors_allow_origin_supports_wildcard_when_credentials_disabled() -> None:
    allow_origin = app_main._compute_openapi_cors_allow_origin(
        "http://any.local",
        allow_all_origins=True,
        allow_credentials=False,
        allowed_openapi_origins=set(),
    )
    assert allow_origin == "*"


def test_main_import_fails_for_wildcard_origins_with_credentials() -> None:
    prior_allowed_origins = os.getenv("ALLOWED_ORIGINS")
    prior_allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS")
    prior_disable_cors = os.getenv("DISABLE_CORS")
    try:
        os.environ["DISABLE_CORS"] = "false"
        os.environ["ALLOWED_ORIGINS"] = "*"
        os.environ["CORS_ALLOW_CREDENTIALS"] = "true"
        importlib.reload(config_mod)
        with pytest.raises(RuntimeError, match="cannot include '\\*'"):
            importlib.reload(app_main)
    finally:
        _restore_env("DISABLE_CORS", prior_disable_cors)
        _restore_env("ALLOWED_ORIGINS", prior_allowed_origins)
        _restore_env("CORS_ALLOW_CREDENTIALS", prior_allow_credentials)
        importlib.reload(config_mod)
        importlib.reload(app_main)


def test_main_import_allows_wildcard_origins_when_credentials_disabled() -> None:
    prior_allowed_origins = os.getenv("ALLOWED_ORIGINS")
    prior_allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS")
    prior_disable_cors = os.getenv("DISABLE_CORS")
    try:
        os.environ["DISABLE_CORS"] = "false"
        os.environ["ALLOWED_ORIGINS"] = "*"
        os.environ["CORS_ALLOW_CREDENTIALS"] = "false"
        reloaded_config = importlib.reload(config_mod)
        reloaded_main = importlib.reload(app_main)

        assert reloaded_config.ALLOWED_ORIGINS == ["*"]
        assert reloaded_main._resolve_cors_origins_or_raise(reloaded_config.ALLOWED_ORIGINS) == ["*"]
    finally:
        _restore_env("DISABLE_CORS", prior_disable_cors)
        _restore_env("ALLOWED_ORIGINS", prior_allowed_origins)
        _restore_env("CORS_ALLOW_CREDENTIALS", prior_allow_credentials)
        importlib.reload(config_mod)
        importlib.reload(app_main)


def test_empty_string_allowed_origins_env_falls_back_to_defaults() -> None:
    prior_allowed_origins = os.getenv("ALLOWED_ORIGINS")
    try:
        os.environ["ALLOWED_ORIGINS"] = ""
        reloaded_config = importlib.reload(config_mod)
        reloaded_main = importlib.reload(app_main)

        assert "http://localhost:3000" in reloaded_config.ALLOWED_ORIGINS
        assert reloaded_main._resolve_cors_origins_or_raise(reloaded_config.ALLOWED_ORIGINS)
    finally:
        _restore_env("ALLOWED_ORIGINS", prior_allowed_origins)
        importlib.reload(config_mod)
        importlib.reload(app_main)


def test_main_import_falls_back_for_explicit_empty_allowed_origins_list_outside_production() -> None:
    prior_allowed_origins = os.getenv("ALLOWED_ORIGINS")
    prior_disable_cors = os.getenv("DISABLE_CORS")
    prior_env = os.getenv("ENV")
    try:
        os.environ["ENV"] = "development"
        os.environ["DISABLE_CORS"] = "false"
        os.environ["ALLOWED_ORIGINS"] = "[]"
        reloaded_config = importlib.reload(config_mod)
        effective_origins, origin_source, fallback_used = reloaded_config.resolve_runtime_allowed_origins(
            reloaded_config.ALLOWED_ORIGINS
        )
        importlib.reload(app_main)

        assert reloaded_config.ALLOWED_ORIGINS == []
        assert origin_source == "env(local-fallback)"
        assert fallback_used is True
        assert "http://localhost:3000" in effective_origins
    finally:
        _restore_env("ENV", prior_env)
        _restore_env("DISABLE_CORS", prior_disable_cors)
        _restore_env("ALLOWED_ORIGINS", prior_allowed_origins)
        importlib.reload(config_mod)
        importlib.reload(app_main)


def test_main_import_fails_for_explicit_empty_allowed_origins_list_in_production() -> None:
    prior_allowed_origins = os.getenv("ALLOWED_ORIGINS")
    prior_disable_cors = os.getenv("DISABLE_CORS")
    prior_env = os.getenv("ENV")
    try:
        os.environ["ENV"] = "production"
        os.environ["DISABLE_CORS"] = "false"
        os.environ["ALLOWED_ORIGINS"] = "[]"
        reloaded_config = importlib.reload(config_mod)
        effective_origins, origin_source, fallback_used = reloaded_config.resolve_runtime_allowed_origins(
            reloaded_config.ALLOWED_ORIGINS
        )

        assert reloaded_config.ALLOWED_ORIGINS == []
        assert effective_origins == []
        assert origin_source == "env"
        assert fallback_used is False
        with pytest.raises(RuntimeError, match="ALLOWED_ORIGINS is empty"):
            importlib.reload(app_main)
    finally:
        _restore_env("ENV", prior_env)
        _restore_env("DISABLE_CORS", prior_disable_cors)
        _restore_env("ALLOWED_ORIGINS", prior_allowed_origins)
        importlib.reload(config_mod)
        importlib.reload(app_main)


def _route_method_count(app: FastAPI, path: str, method: str) -> int:
    method_upper = method.upper()
    count = 0
    for route in app.routes:
        route_path = getattr(route, "path", None)
        route_methods = getattr(route, "methods", set()) or set()
        if route_path == path and method_upper in route_methods:
            count += 1
    return count


def _restore_env(key: str, prior_value: str | None) -> None:
    if prior_value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = prior_value


def test_ultra_minimal_uses_control_plane_health_routes_only() -> None:
    prior_ultra = os.getenv("ULTRA_MINIMAL_APP")
    prior_minimal = os.getenv("MINIMAL_TEST_APP")
    prior_test_mode = os.getenv("TEST_MODE")

    try:
        os.environ["TEST_MODE"] = "1"
        os.environ["ULTRA_MINIMAL_APP"] = "1"
        os.environ["MINIMAL_TEST_APP"] = "0"
        reloaded_main = importlib.reload(app_main)

        assert _route_method_count(reloaded_main.app, "/health", "GET") == 1
        assert _route_method_count(reloaded_main.app, "/ready", "GET") == 1
        assert _route_method_count(reloaded_main.app, "/health/ready", "GET") == 1
        assert _route_method_count(reloaded_main.app, "/api/v1/health", "GET") == 0
    finally:
        _restore_env("ULTRA_MINIMAL_APP", prior_ultra)
        _restore_env("MINIMAL_TEST_APP", prior_minimal)
        _restore_env("TEST_MODE", prior_test_mode)
        importlib.reload(app_main)
