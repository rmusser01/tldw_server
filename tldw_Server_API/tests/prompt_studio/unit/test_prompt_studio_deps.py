import types
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, status
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import prompt_studio_deps as deps
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import DatabaseError
from tldw_Server_API.app.core.Prompt_Management.prompt_studio import quota_config


@pytest.fixture(autouse=True)
def reset_cache():
    """Ensure dependency cache is isolated per test."""
    with deps._db_lock:
        deps._db_instances_cache.clear()
    try:
        yield
    finally:
        with deps._db_lock:
            deps._db_instances_cache.clear()


def _make_backend(connection_string: str):
    backend = MagicMock()
    backend.backend_type = BackendType.POSTGRESQL
    backend.config = types.SimpleNamespace(
        connection_string=connection_string,
        sqlite_path=None,
        pg_database=None,
    )
    return backend


def _make_request(headers: dict[str, str] | None = None) -> Request:
    raw_headers = [
        (name.lower().encode("latin-1"), value.encode("latin-1"))
        for name, value in (headers or {}).items()
    ]
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/prompt-studio/projects/",
            "headers": raw_headers,
            "query_string": b"",
            "server": ("testserver", 80),
            "scheme": "http",
            "client": ("testclient", 50000),
        }
    )


def _logged_text(mock_method) -> str:
    return " ".join(str(call) for call in mock_method.call_args_list)


def test_get_or_create_prompt_studio_db_passes_backend(monkeypatch, tmp_path):


    db_path = tmp_path / "u-123" / "prompt_studio.db"

    def fake_path(user_id: str):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return db_path

    monkeypatch.setattr(deps, "_get_prompt_studio_db_path_for_user", fake_path)

    backend = _make_backend("postgres://primary")
    monkeypatch.setattr(deps, "get_content_backend_instance", lambda: backend)

    mock_instance = object()
    create_mock = MagicMock(return_value=mock_instance)
    monkeypatch.setattr(deps, "create_prompt_studio_database", create_mock)

    first = deps._get_or_create_prompt_studio_db("user-123", "client-xyz")
    second = deps._get_or_create_prompt_studio_db("user-123", "client-xyz")

    assert first is mock_instance
    assert second is mock_instance
    assert create_mock.call_count == 1

    kwargs = create_mock.call_args.kwargs
    assert kwargs["backend"] is backend
    assert kwargs["db_path"] == db_path


def test_backend_signature_in_cache_includes_connection(monkeypatch, tmp_path):


    db_path = tmp_path / "u-456" / "prompt_studio.db"

    def fake_path(user_id: str):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return db_path

    monkeypatch.setattr(deps, "_get_prompt_studio_db_path_for_user", fake_path)

    instance_a = object()
    instance_b = object()
    create_mock = MagicMock(side_effect=[instance_a, instance_b])
    monkeypatch.setattr(deps, "create_prompt_studio_database", create_mock)

    backend_a = _make_backend("postgres://primary")
    backend_b = _make_backend("postgres://replica")

    monkeypatch.setattr(deps, "get_content_backend_instance", lambda: backend_a)
    first = deps._get_or_create_prompt_studio_db("user-123", "client-xyz")
    assert first is instance_a

    monkeypatch.setattr(deps, "get_content_backend_instance", lambda: backend_b)
    second = deps._get_or_create_prompt_studio_db("user-123", "client-xyz")

    assert create_mock.call_count == 2
    assert second is instance_b

    # Switching back to backend_a should reuse cached instance without creating a third database
    monkeypatch.setattr(deps, "get_content_backend_instance", lambda: backend_a)
    third = deps._get_or_create_prompt_studio_db("user-123", "client-xyz")
    assert create_mock.call_count == 2
    assert third is instance_a


def test_get_or_create_prompt_studio_db_logs_safe_creation_failure(monkeypatch, tmp_path):
    sensitive_text = "database password leaked: sk-secret-value"
    db_path = tmp_path / "u-789" / "prompt_studio.db"

    def fake_path(user_id: str):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return db_path

    def fail_create(*_args, **_kwargs):
        raise RuntimeError(sensitive_text)

    logger_mock = MagicMock()
    monkeypatch.setattr(deps, "_get_prompt_studio_db_path_for_user", fake_path)
    monkeypatch.setattr(deps, "get_content_backend_instance", lambda: None)
    monkeypatch.setattr(deps, "create_prompt_studio_database", fail_create)
    monkeypatch.setattr(deps, "logger", logger_mock)

    with pytest.raises(HTTPException) as exc_info:
        deps._get_or_create_prompt_studio_db("user-789", "client-xyz")

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc_info.value.detail == "Failed to initialize database"
    logged_text = " ".join(str(call) for call in logger_mock.error.call_args_list)
    assert sensitive_text not in logged_text
    assert "Failed to create PromptStudioDatabase for user" in logged_text


def test_shutdown_prompt_studio_deps_logs_safe_close_failure(monkeypatch):
    sensitive_text = "close failed for token sk-close-secret"

    class BrokenCloseDb:
        def close(self):
            raise RuntimeError(sensitive_text)

    logger_mock = MagicMock()
    monkeypatch.setattr(deps, "logger", logger_mock)

    with deps._db_lock:
        deps._db_instances_cache["broken"] = BrokenCloseDb()

    deps.shutdown_prompt_studio_deps()

    logged_text = " ".join(str(call) for call in logger_mock.error.call_args_list)
    assert sensitive_text not in logged_text
    assert "Error closing database instance" in logged_text
    with deps._db_lock:
        assert not deps._db_instances_cache


@pytest.mark.asyncio
async def test_require_project_access_maps_database_error():
    class BrokenProjectDb:
        def get_project(self, project_id: int):
            raise DatabaseError(f"project {project_id} lookup failed")

    with pytest.raises(HTTPException) as exc_info:
        await deps.require_project_access(
            project_id=42,
            user_context={"user_id": "user-1", "is_admin": False},
            db=BrokenProjectDb(),
        )

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc_info.value.detail == "Database error"


@pytest.mark.asyncio
async def test_require_project_access_logs_safe_database_error(monkeypatch):
    sensitive_text = "lookup failed with password=super-secret-token"

    class BrokenProjectDb:
        def get_project(self, project_id: int):
            raise DatabaseError(sensitive_text)

    logger_mock = MagicMock()
    monkeypatch.setattr(deps, "logger", logger_mock)

    with pytest.raises(HTTPException) as exc_info:
        await deps.require_project_access(
            project_id=42,
            user_context={"user_id": "user-1", "is_admin": False},
            db=BrokenProjectDb(),
        )

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc_info.value.detail == "Database error"
    logged_text = " ".join(str(call) for call in logger_mock.error.call_args_list)
    assert sensitive_text not in logged_text
    assert "Database error checking project access" in logged_text


@pytest.mark.asyncio
async def test_get_prompt_studio_user_patched_hook_sanitizes_quota_lookup_failure_log(monkeypatch):
    sensitive_text = "/tmp/prompt-studio/private-policy.json?token=sk-quota-secret"

    async def fail_quota_policy(_user_id: str):
        raise RuntimeError(sensitive_text)

    logger_mock = MagicMock()
    monkeypatch.setenv("TEST_MODE", "false")
    monkeypatch.setenv("TLDW_TEST_MODE", "false")
    monkeypatch.setattr(deps, "get_current_active_user", lambda: {"id": "123"}, raising=True)
    monkeypatch.setattr(quota_config, "apply_prompt_studio_quota_policy", fail_quota_policy)
    monkeypatch.setattr(deps, "logger", logger_mock)

    user_context = await deps.get_prompt_studio_user(_make_request())

    assert user_context["user_id"] == "123"
    logged_text = _logged_text(logger_mock.debug)
    assert sensitive_text not in logged_text
    assert "Prompt Studio quota policy lookup failed" in logged_text
    assert "RuntimeError" in logged_text


@pytest.mark.asyncio
async def test_get_prompt_studio_user_request_user_sanitizes_quota_lookup_failure_log(monkeypatch):
    sensitive_text = "/var/db/prompt-studio/user-profile.sqlite password=secret"

    async def fail_quota_policy(_user_id: str):
        raise RuntimeError(sensitive_text)

    async def fake_get_request_user(*_args, **_kwargs):
        return types.SimpleNamespace(id="456", roles=[], permissions=[])

    logger_mock = MagicMock()
    monkeypatch.setenv("TEST_MODE", "false")
    monkeypatch.setenv("TLDW_TEST_MODE", "false")
    monkeypatch.setattr(deps, "get_current_active_user", lambda: None, raising=True)
    monkeypatch.setattr(deps, "get_request_user", fake_get_request_user, raising=True)
    monkeypatch.setattr(quota_config, "apply_prompt_studio_quota_policy", fail_quota_policy)
    monkeypatch.setattr(deps, "logger", logger_mock)

    user_context = await deps.get_prompt_studio_user(_make_request(headers={"X-API-KEY": "abc"}))

    assert user_context["user_id"] == "456"
    logged_text = _logged_text(logger_mock.debug)
    assert sensitive_text not in logged_text
    assert "Prompt Studio quota policy lookup failed" in logged_text
    assert "RuntimeError" in logged_text


@pytest.mark.asyncio
async def test_check_rate_limit_sanitizes_rg_policy_lookup_failure_log(monkeypatch):
    sensitive_text = "/srv/private/rg-policy.yaml api_key=sk-rg-secret"

    class BrokenPolicyContext:
        def get(self, key, default=None):
            if key == "rg_policy_id":
                raise RuntimeError(sensitive_text)
            if key == "user_id":
                return "user-123"
            return default

    async def allow_rate_limit(*_args, **_kwargs):
        return True, {}

    logger_mock = MagicMock()
    monkeypatch.setenv("TEST_MODE", "false")
    monkeypatch.setenv("TLDW_TEST_MODE", "false")
    monkeypatch.setattr(deps, "_authnz_check_rate_limit", allow_rate_limit, raising=True)
    monkeypatch.setattr(deps, "logger", logger_mock)

    allowed = await deps.check_rate_limit(
        operation="default",
        user_context=BrokenPolicyContext(),
        security_config=deps.SecurityConfig(enable_rate_limiting=True),
    )

    assert allowed is True
    logged_text = _logged_text(logger_mock.debug)
    assert sensitive_text not in logged_text
    assert "Prompt Studio rate-limit bypass: failed to read rg_policy_id" in logged_text
    assert "RuntimeError" in logged_text


@pytest.mark.asyncio
async def test_check_rate_limit_sanitizes_shared_limiter_unavailable_warning_log(monkeypatch):
    sensitive_text = "redis://:sk-redis-secret@localhost:6379/0"

    async def raise_shared_limiter(*_args, **_kwargs):
        raise RuntimeError(sensitive_text)

    logger_mock = MagicMock()
    monkeypatch.setenv("TEST_MODE", "false")
    monkeypatch.setenv("TLDW_TEST_MODE", "false")
    monkeypatch.setattr(deps, "_authnz_check_rate_limit", raise_shared_limiter, raising=True)
    monkeypatch.setattr(deps, "_PROMPT_STUDIO_RATE_LIMIT_SHIM_LOGGED", False, raising=True)
    monkeypatch.setattr(deps, "logger", logger_mock)

    with pytest.raises(HTTPException) as exc_info:
        await deps.check_rate_limit(
            operation="optimize",
            user_context={"user_id": "user-123"},
            security_config=deps.SecurityConfig(enable_rate_limiting=True),
        )

    assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert exc_info.value.detail == "Prompt Studio rate limiter is temporarily unavailable"
    logged_text = _logged_text(logger_mock.warning)
    assert sensitive_text not in logged_text
    assert "Prompt Studio shared rate limiter unavailable" in logged_text
    assert "RuntimeError" in logged_text
