import pytest
from fastapi import HTTPException, status

from tldw_Server_API.app.core.DB_Management.chacha.runtime import (
    ChaChaRuntimeManager,
    ChaChaRuntimeUnavailableError,
)


def test_runtime_manager_exposes_explicit_resettable_surface():
    runtime = ChaChaRuntimeManager()

    assert hasattr(runtime, "get_or_create")
    assert hasattr(runtime, "shutdown")
    assert hasattr(runtime, "snapshot")
    runtime.reset_for_tests()


@pytest.mark.asyncio
async def test_dependency_maps_runtime_unavailable_to_503(monkeypatch):
    import tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps as deps

    class _Runtime:
        async def get_or_create(self, *_args, **_kwargs):
            raise ChaChaRuntimeUnavailableError("ChaChaNotes shutdown in progress")

    monkeypatch.setattr(deps, "_CHACHA_RUNTIME", _Runtime())

    with pytest.raises(HTTPException) as exc:
        await deps.get_chacha_db_for_user_id(1, "1")

    assert exc.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "shutdown" in exc.value.detail.lower()
