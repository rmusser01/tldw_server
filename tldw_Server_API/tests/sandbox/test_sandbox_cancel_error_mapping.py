from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import sandbox as sandbox_endpoint


@pytest.mark.asyncio
async def test_cancel_run_sanitizes_internal_failure(monkeypatch):
    monkeypatch.setattr(sandbox_endpoint, "_require_run_owner", lambda *_args, **_kwargs: "run-1")

    def _raise_cancel_failure(_run_id):
        raise RuntimeError("sandbox cancel backend exploded at /private/sandbox.sock")

    monkeypatch.setattr(sandbox_endpoint._service, "cancel_run", _raise_cancel_failure)

    with pytest.raises(HTTPException) as exc_info:
        await sandbox_endpoint.cancel_run(
            run_id="run-1",
            current_user=SimpleNamespace(id=1),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == {
        "error": {
            "code": "cancel_failed",
            "message": "Failed to cancel sandbox run",
            "details": {"run_id": "run-1"},
        }
    }
