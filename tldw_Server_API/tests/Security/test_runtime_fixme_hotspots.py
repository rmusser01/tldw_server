import importlib
import os
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, status

from tldw_Server_API.app import main as app_main
from tldw_Server_API.app.api.v1.endpoints import sync as sync_endpoints
from tldw_Server_API.app.core import config as config_mod


pytestmark = pytest.mark.unit


class _DummyUser:
    def __init__(self, username: str):
        self.username = username


def _restore_env(key: str, prior_value: str | None) -> None:
    if prior_value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = prior_value


def build_app_for_test():
    importlib.reload(config_mod)
    reloaded_main = importlib.reload(app_main)
    return reloaded_main.app


def build_app_with_origins(origins: list[str]):
    os.environ["ALLOWED_ORIGINS"] = ",".join(origins)
    return build_app_for_test()


def get_cors_allow_origins(app) -> list[str]:
    for middleware in getattr(app, "user_middleware", []):
        middleware_cls = getattr(middleware, "cls", None)
        if getattr(middleware_cls, "__name__", "") != "CORSMiddleware":
            continue
        kwargs = getattr(middleware, "kwargs", {}) or {}
        origins = kwargs.get("allow_origins", [])
        return [str(origin) for origin in origins]
    return []


def test_cors_policy_not_wildcard_in_production() -> None:
    prior_env = os.getenv("ENV")
    prior_disable_cors = os.getenv("DISABLE_CORS")
    prior_allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS")
    prior_allowed_origins = os.getenv("ALLOWED_ORIGINS")
    try:
        os.environ["ENV"] = "production"
        os.environ["DISABLE_CORS"] = "false"
        os.environ["CORS_ALLOW_CREDENTIALS"] = "false"
        os.environ["ALLOWED_ORIGINS"] = "http://localhost:3000"
        app = build_app_for_test()
        assert "*" not in get_cors_allow_origins(app)  # nosec B101
    finally:
        _restore_env("ENV", prior_env)
        _restore_env("DISABLE_CORS", prior_disable_cors)
        _restore_env("CORS_ALLOW_CREDENTIALS", prior_allow_credentials)
        _restore_env("ALLOWED_ORIGINS", prior_allowed_origins)
        importlib.reload(config_mod)
        importlib.reload(app_main)


def test_production_cors_requires_explicit_origin_list() -> None:
    prior_env = os.getenv("ENV")
    prior_disable_cors = os.getenv("DISABLE_CORS")
    prior_allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS")
    prior_allowed_origins = os.getenv("ALLOWED_ORIGINS")
    try:
        os.environ["ENV"] = "production"
        os.environ["DISABLE_CORS"] = "false"
        os.environ["CORS_ALLOW_CREDENTIALS"] = "false"
        with pytest.raises(RuntimeError, match="cannot include '\\*' in production"):
            build_app_with_origins(["*"])
    finally:
        _restore_env("ENV", prior_env)
        _restore_env("DISABLE_CORS", prior_disable_cors)
        _restore_env("CORS_ALLOW_CREDENTIALS", prior_allow_credentials)
        _restore_env("ALLOWED_ORIGINS", prior_allowed_origins)
        importlib.reload(config_mod)
        importlib.reload(app_main)


def test_non_production_empty_cors_origin_list_falls_back_to_local_defaults() -> None:
    prior_env = os.getenv("ENV")
    prior_disable_cors = os.getenv("DISABLE_CORS")
    prior_allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS")
    prior_allowed_origins = os.getenv("ALLOWED_ORIGINS")
    try:
        os.environ["ENV"] = "development"
        os.environ["DISABLE_CORS"] = "false"
        os.environ["CORS_ALLOW_CREDENTIALS"] = "false"
        os.environ["ALLOWED_ORIGINS"] = "[]"
        app = build_app_for_test()
        origins = get_cors_allow_origins(app)
        assert "http://localhost:3000" in origins  # nosec B101
        assert "http://127.0.0.1:8080" in origins  # nosec B101
    finally:
        _restore_env("ENV", prior_env)
        _restore_env("DISABLE_CORS", prior_disable_cors)
        _restore_env("CORS_ALLOW_CREDENTIALS", prior_allow_credentials)
        _restore_env("ALLOWED_ORIGINS", prior_allowed_origins)
        importlib.reload(config_mod)
        importlib.reload(app_main)


@pytest.mark.asyncio
async def test_legacy_sync_send_endpoint_is_replaced() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await sync_endpoints.receive_changes_from_client(
            request=SimpleNamespace(),
            user_id=_DummyUser("sync-user"),
        )

    assert exc_info.value.status_code == status.HTTP_410_GONE  # nosec B101
    assert exc_info.value.detail["replacement"] == "/api/v1/sync/push"  # nosec B101
    assert "fts disabled" not in str(exc_info.value.detail).lower()  # nosec B101


@pytest.mark.asyncio
async def test_legacy_sync_get_endpoint_is_replaced() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await sync_endpoints.send_changes_to_client(
            request=SimpleNamespace(),
            user_id=_DummyUser("sync-user"),
        )

    assert exc_info.value.status_code == status.HTTP_410_GONE  # nosec B101
    assert exc_info.value.detail["replacement"] == "/api/v1/sync/pull"  # nosec B101
    assert "fts disabled" not in str(exc_info.value.detail).lower()  # nosec B101
