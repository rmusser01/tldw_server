import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.API_Deps import kanban_deps


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clear_kanban_dependency_state():
    with kanban_deps._kanban_db_lock:
        kanban_deps._kanban_db_instances.clear()
        kanban_deps._kanban_db_health_checks.clear()
    yield
    kanban_deps.shutdown_kanban_executor(wait=True)
    with kanban_deps._kanban_db_lock:
        kanban_deps._kanban_db_instances.clear()
        kanban_deps._kanban_db_health_checks.clear()


@pytest.mark.asyncio
async def test_kanban_init_sanitizes_runtime_errors(monkeypatch):
    def _raise_init_error(_user_id):
        raise RuntimeError("kanban backend exploded at /private/db/path")

    monkeypatch.setattr(kanban_deps, "_create_kanban_db", _raise_init_error)

    with pytest.raises(HTTPException) as exc_info:
        await kanban_deps._get_or_init_db_instance(123)

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Could not initialize Kanban database for user"
