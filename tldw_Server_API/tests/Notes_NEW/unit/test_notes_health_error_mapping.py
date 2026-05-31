import pytest

from tldw_Server_API.app.api.v1.endpoints import notes as notes_endpoint


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_notes_health_sanitizes_storage_failure(monkeypatch):
    def _raise_storage_failure():
        raise RuntimeError("notes storage exploded at /private/chachanotes")

    monkeypatch.setattr(notes_endpoint, "resolve_chacha_user_base_dir", _raise_storage_failure)

    response = await notes_endpoint.notes_health()

    assert response["status"] == "unhealthy"
    assert response["error"] == "Notes health check failed"
