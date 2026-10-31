import threading
import types
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest
from cachetools import LRUCache
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


class _CloseRecordingPromptStudioDb:
    """Small Prompt Studio DB stand-in with observable connection ownership."""

    def __init__(self, client_id: str, tenant_user_id: str) -> None:
        self.client_id = client_id
        self.tenant_user_id = tenant_user_id
        self.user_id: str | None = None
        self.close_count = 0

    def close_connection(self) -> None:
        self.close_count += 1

    def close(self) -> None:
        self.close_connection()


def _install_close_recording_cache(
    monkeypatch,
    tmp_path,
    *,
    maxsize: int,
) -> tuple[LRUCache, list[_CloseRecordingPromptStudioDb]]:
    """Install a bounded cache and a close-recording DB factory."""
    cache = LRUCache(maxsize=maxsize)
    created: list[_CloseRecordingPromptStudioDb] = []

    def create_db(client_id, *, tenant_user_id, **_kwargs):
        db = _CloseRecordingPromptStudioDb(client_id, tenant_user_id)
        created.append(db)
        return db

    monkeypatch.setattr(deps, "_db_instances_cache", cache)
    monkeypatch.setattr(
        deps,
        "_get_prompt_studio_db_path_for_user",
        lambda _user_id: tmp_path / "prompt-studio.db",
    )
    monkeypatch.setattr(deps, "get_content_backend_instance", lambda: None)
    monkeypatch.setattr(deps, "create_prompt_studio_database", create_db)
    return cache, created


def test_get_or_create_prompt_studio_db_passes_backend(monkeypatch, tmp_path):


    db_path = tmp_path / "u-123" / "prompt_studio.db"

    def fake_path(user_id: str):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return db_path

    monkeypatch.setattr(deps, "_get_prompt_studio_db_path_for_user", fake_path)

    backend = _make_backend("postgres://primary")
    monkeypatch.setattr(deps, "get_content_backend_instance", lambda: backend)

    mock_instance = types.SimpleNamespace()
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


def test_get_or_create_prompt_studio_db_separates_tenant_from_audit_client(
    monkeypatch,
    tmp_path,
):
    db_path = tmp_path / "tenant-42" / "prompt_studio.db"
    backend = _make_backend("postgres://primary")
    instance = types.SimpleNamespace()
    create_mock = MagicMock(return_value=instance)

    monkeypatch.setattr(
        deps,
        "_get_prompt_studio_db_path_for_user",
        lambda _user_id: db_path,
    )
    monkeypatch.setattr(deps, "get_content_backend_instance", lambda: backend)
    monkeypatch.setattr(deps, "create_prompt_studio_database", create_mock)

    result = deps._get_or_create_prompt_studio_db("tenant-42", "audit-client-9")

    assert result is instance
    assert create_mock.call_args.args == ("audit-client-9",)
    assert create_mock.call_args.kwargs["tenant_user_id"] == "tenant-42"
    assert create_mock.call_args.kwargs["backend"] is backend
    assert instance.user_id == "tenant-42"


def test_prompt_studio_db_cache_isolates_request_audit_clients(
    monkeypatch,
    tmp_path,
):
    db_path = tmp_path / "tenant-42" / "prompt_studio.db"
    backend = _make_backend("postgres://primary")
    instances = [
        types.SimpleNamespace(client_id="audit-client-a"),
        types.SimpleNamespace(client_id="audit-client-b"),
    ]
    create_mock = MagicMock(side_effect=instances)

    monkeypatch.setattr(
        deps,
        "_get_prompt_studio_db_path_for_user",
        lambda _user_id: db_path,
    )
    monkeypatch.setattr(deps, "get_content_backend_instance", lambda: backend)
    monkeypatch.setattr(deps, "create_prompt_studio_database", create_mock)

    client_a = deps._get_or_create_prompt_studio_db(
        "tenant-42",
        "audit-client-a",
    )
    client_b = deps._get_or_create_prompt_studio_db(
        "tenant-42",
        "audit-client-b",
    )
    client_a_again = deps._get_or_create_prompt_studio_db(
        "tenant-42",
        "audit-client-a",
    )

    assert client_a is instances[0]
    assert client_b is instances[1]
    assert client_a_again is instances[0]
    assert client_a is not client_b
    assert create_mock.call_count == 2


def test_prompt_studio_db_cache_isolates_concurrent_audit_clients(
    monkeypatch,
    tmp_path,
):
    db_path = tmp_path / "tenant-42" / "prompt_studio.db"
    backend = _make_backend("postgres://primary")
    request_barrier = threading.Barrier(2)

    class RecordingFactory:
        def __init__(self):
            self._lock = threading.Lock()
            self.calls = []

        def __call__(self, client_id, **kwargs):
            instance = types.SimpleNamespace(
                client_id=client_id,
                tenant_user_id=kwargs["tenant_user_id"],
            )
            with self._lock:
                self.calls.append((client_id, kwargs, instance))
            return instance

        def snapshot(self):
            with self._lock:
                return list(self.calls)

    factory = RecordingFactory()
    monkeypatch.setattr(
        deps,
        "_get_prompt_studio_db_path_for_user",
        lambda _user_id: db_path,
    )
    monkeypatch.setattr(deps, "get_content_backend_instance", lambda: backend)
    monkeypatch.setattr(deps, "create_prompt_studio_database", factory)

    def lookup_twice(client_id):
        request_barrier.wait(timeout=5)
        first = deps._get_or_create_prompt_studio_db("tenant-42", client_id)
        request_barrier.wait(timeout=5)
        return first, deps._get_or_create_prompt_studio_db("tenant-42", client_id)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(lookup_twice, "audit-client-a")
        future_b = executor.submit(lookup_twice, "audit-client-b")
        client_a, client_a_again = future_a.result(timeout=10)
        client_b, client_b_again = future_b.result(timeout=10)

    assert client_a is client_a_again
    assert client_b is client_b_again
    assert client_a is not client_b
    assert client_a.client_id == "audit-client-a"
    assert client_b.client_id == "audit-client-b"
    assert client_a.tenant_user_id == "tenant-42"
    assert client_b.tenant_user_id == "tenant-42"

    calls = factory.snapshot()
    assert sorted(call[0] for call in calls) == ["audit-client-a", "audit-client-b"]
    assert all(call[1]["backend"] is backend for call in calls)


def test_prompt_studio_db_cache_closes_idle_capacity_eviction(
    monkeypatch,
    tmp_path,
):
    cache, created = _install_close_recording_cache(
        monkeypatch,
        tmp_path,
        maxsize=1,
    )

    first = deps._get_or_create_prompt_studio_db("tenant-42", "audit-client-a")
    second = deps._get_or_create_prompt_studio_db("tenant-42", "audit-client-b")

    assert first is created[0]
    assert second is created[1]
    assert first.close_count == 1
    assert second.close_count == 0
    assert list(cache.values()) == [second]


def test_prompt_studio_db_cache_defers_active_eviction_until_all_callers_release(
    monkeypatch,
    tmp_path,
):
    cache, created = _install_close_recording_cache(
        monkeypatch,
        tmp_path,
        maxsize=1,
    )
    managed_scope = getattr(deps, "managed_prompt_studio_db", None)
    assert callable(managed_scope), (
        "Prompt Studio DB callers need a managed scope so capacity eviction can "
        "defer closing in-use instances"
    )

    acquired = threading.Barrier(3)
    release_first = threading.Event()
    release_second = threading.Event()
    observed: list[_CloseRecordingPromptStudioDb] = []
    observed_lock = threading.Lock()

    def hold_db(release: threading.Event):
        with managed_scope(
            {"user_id": "tenant-42", "client_id": "audit-client-a"}
        ) as db:
            with observed_lock:
                observed.append(db)
            acquired.wait(timeout=5)
            assert release.wait(timeout=5)
            return db

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_caller = executor.submit(hold_db, release_first)
        second_caller = executor.submit(hold_db, release_second)
        try:
            acquired.wait(timeout=5)
            active_db = observed[0]
            assert observed == [active_db, active_db]

            replacement = deps._get_or_create_prompt_studio_db(
                "tenant-42",
                "audit-client-b",
            )
            assert list(cache.values()) == [replacement]
            assert active_db.close_count == 0

            release_first.set()
            assert first_caller.result(timeout=5) is active_db
            assert active_db.close_count == 0

            release_second.set()
            assert second_caller.result(timeout=5) is active_db
            assert active_db.close_count == 1
        finally:
            release_first.set()
            release_second.set()

    assert created == [active_db, replacement]


def test_prompt_studio_db_cache_client_churn_stays_bounded_and_closes_evictions(
    monkeypatch,
    tmp_path,
):
    cache, created = _install_close_recording_cache(
        monkeypatch,
        tmp_path,
        maxsize=2,
    )

    for index in range(25):
        deps._get_or_create_prompt_studio_db(
            "tenant-42",
            f"caller-controlled-audit-client-{index}",
        )

    cached_ids = {id(db) for db in cache.values()}
    unclosed = [db for db in created if db.close_count == 0]
    evicted = [db for db in created if db.close_count == 1]

    assert len(cache) == 2
    assert {id(db) for db in unclosed} == cached_ids
    assert len(unclosed) == 2
    assert len(evicted) == 23
    assert all(db.close_count <= 1 for db in created)


def test_backend_signature_in_cache_includes_connection(monkeypatch, tmp_path):


    db_path = tmp_path / "u-456" / "prompt_studio.db"

    def fake_path(user_id: str):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return db_path

    monkeypatch.setattr(deps, "_get_prompt_studio_db_path_for_user", fake_path)

    instance_a = types.SimpleNamespace()
    instance_b = types.SimpleNamespace()
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
