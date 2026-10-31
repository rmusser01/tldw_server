import pytest
from fastapi import HTTPException, status

from tldw_Server_API.app.core.DB_Management.chacha.runtime import (
    ChaChaRuntimeManager,
)
from tldw_Server_API.app.core.Utils.app_services import app_services


def test_runtime_manager_shutdown_and_resume_toggle_admission():
    runtime = ChaChaRuntimeManager()

    runtime.shutdown()
    assert runtime.shutting_down is True
    runtime.resume()
    assert runtime.shutting_down is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call",
    [
        lambda deps: deps.get_chacha_db_for_user_id(4242),
        lambda deps: deps.get_chacha_db_for_owner(-1),
        lambda deps: deps.get_chacha_db_for_owner(0),
    ],
    ids=["user-4242", "owner-minus-1", "owner-0"],
)
async def test_closed_runtime_refuses_with_503_before_touching_the_filesystem(monkeypatch, tmp_path, call):
    import tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps as deps

    runtime = ChaChaRuntimeManager()
    runtime.shutdown()
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))  # where a user directory would appear
    monkeypatch.setattr(app_services(), "chacha_runtime", runtime)

    with pytest.raises(HTTPException) as exc:
        await call(deps)

    assert exc.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert exc.value.detail == "ChaChaNotes shutdown in progress"
    assert list(tmp_path.iterdir()) == []
