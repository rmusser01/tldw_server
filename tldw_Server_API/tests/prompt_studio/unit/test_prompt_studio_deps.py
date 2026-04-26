import types
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, status

from tldw_Server_API.app.api.v1.API_Deps import prompt_studio_deps as deps
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import DatabaseError


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
